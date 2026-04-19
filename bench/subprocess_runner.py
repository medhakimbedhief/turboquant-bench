"""
Isolated subprocess benchmark runner.

WHY SUBPROCESS ISOLATION?
--------------------------
vLLM allocates its paged KV cache at engine startup and does not release it
while the engine is alive. If you reuse one engine across context lengths:
  - The KV pool is sized for the first context length you used
  - VRAM readings for later runs are polluted by the first engine's allocation
  - You cannot compare "32K baseline" vs "32K TQ" because the baseline engine
    already ate up all the KV memory budget for 32K tokens

The solution: each measurement (one context length, one mode, one repetition)
runs in a FRESH Python subprocess that:
  1. Creates a new vLLM engine (KV cache sized exactly for this context)
  2. Runs perplexity + TTFT + decode throughput
  3. Optionally frees the KV cache (TQ mode) and records bytes freed
  4. Writes results as a JSON line to stdout
  5. Exits — CUDA context destroyed, VRAM fully released

The notebook cell calls run_single_benchmark(), which returns the results dict.
If the subprocess crashes (OOM, Triton error), an error dict is returned so
the notebook can print a warning and continue to the next measurement.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap


def run_single_benchmark(
    config: dict,
    context_length: int,
    use_tq: bool = False,
) -> dict:
    """
    Run one complete benchmark in an isolated subprocess.

    Parameters
    ----------
    config : dict
        Experiment CONFIG dict from the notebook.
    context_length : int
        Number of tokens in the WikiText-2 prompt.
    use_tq : bool
        If True, install TurboQuant hooks before measuring.

    Returns
    -------
    dict with keys:
        context_length, use_tq,
        perplexity (float),
        peak_vram_mb (int, from nvidia-smi),
        ttft_s (float),
        decode_tps (float),
        num_hooked_layers (int, TQ only, else 0),
        tq_bytes_freed (int, TQ only, else 0),
        error (str, only present on failure)
    """
    script = _generate_benchmark_script(config, context_length, use_tq)

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    env = os.environ.copy()
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,  # 30 min max per run
        )

        # Parse the last valid JSON line from stdout
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                return data
            except json.JSONDecodeError:
                continue

        # No JSON found → subprocess failed without structured output
        stderr_tail = result.stderr[-3000:] if result.stderr else "(no stderr)"
        return {
            "error": f"No JSON output. Exit={result.returncode}. stderr:\n{stderr_tail}",
            "context_length": context_length,
            "use_tq": use_tq,
        }

    except subprocess.TimeoutExpired:
        return {
            "error": "timeout (>30 min)",
            "context_length": context_length,
            "use_tq": use_tq,
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _generate_benchmark_script(
    config: dict, context_length: int, use_tq: bool
) -> str:
    """
    Generate a self-contained Python script for the benchmark subprocess.

    The script is an f-string that embeds config and context_length as
    Python literals so the subprocess has no external dependencies beyond
    the packages installed by setup.sh and the bench/ modules.
    """
    config_repr = json.dumps(config)
    use_tq_repr = "True" if use_tq else "False"
    num_tokens = config.get("num_generate_tokens", 256)

    return textwrap.dedent(f"""\
        import os, sys, json, math, time, subprocess
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Make bench.* importable (script lives in a temp dir, repo root is 2 levels up
        # from bench/, but we embed the actual repo root path at generation time)
        _repo_root = {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r}
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)

        import torch
        from vllm import LLM, SamplingParams
        from bench.memory import get_nvidia_smi_mb, reset_memory_tracking
        from bench.perplexity import compute_perplexity
        from bench.latency import measure_ttft, measure_decode_throughput

        CONFIG = json.loads({config_repr!r})
        CONTEXT_LENGTH = {context_length}
        USE_TQ = {use_tq_repr}
        NUM_GENERATE_TOKENS = {num_tokens}

        def get_prompt():
            \"\"\"Load WikiText-2 and return the first CONTEXT_LENGTH tokens decoded back to text.\"\"\"
            from datasets import load_dataset
            from transformers import AutoTokenizer

            ds = load_dataset(
                CONFIG["perplexity_dataset"],
                CONFIG["perplexity_subset"],
                split="test",
            )
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"])
            full_text = " ".join(row["text"] for row in ds if row["text"].strip())
            token_ids = tokenizer.encode(full_text, add_special_tokens=False)
            # Take first CONTEXT_LENGTH tokens, decode back to a clean string
            token_ids = token_ids[:CONTEXT_LENGTH]
            return tokenizer.decode(token_ids, skip_special_tokens=True)

        def main():
            result = {{
                "context_length": CONTEXT_LENGTH,
                "use_tq": USE_TQ,
                "perplexity": None,
                "peak_vram_mb": None,
                "ttft_s": None,
                "decode_tps": None,
                "num_hooked_layers": 0,
                "tq_bytes_freed": 0,
            }}

            try:
                prompt = get_prompt()

                # --- Engine creation ---
                reset_memory_tracking()
                vram_before = get_nvidia_smi_mb()

                llm = LLM(
                    model=CONFIG["model_id"],
                    dtype=CONFIG["dtype"],
                    gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
                    tensor_parallel_size=CONFIG["tensor_parallel_size"],
                    max_model_len=CONTEXT_LENGTH,
                    max_num_seqs=1,
                    trust_remote_code=True,
                )

                if USE_TQ:
                    engine = llm.llm_engine
                    core = getattr(engine, "engine_core", engine)
                    inner = getattr(core, "engine_core", core)
                    executor = inner.model_executor

                    key_bits = CONFIG["tq_key_bits"]
                    value_bits = CONFIG["tq_value_bits"]

                    def _install(worker):
                        from turboquant.vllm_attn_backend import (
                            install_turboquant_hooks, MODE_ACTIVE,
                        )
                        states = install_turboquant_hooks(
                            worker.model_runner,
                            key_bits=key_bits,
                            value_bits=value_bits,
                            buffer_size=128,
                            mode=MODE_ACTIVE,
                        )
                        return len(states)

                    hooks = executor.collective_rpc(_install)
                    result["num_hooked_layers"] = hooks[0]

                    if hooks[0] == 0:
                        def _list_attn(worker):
                            return [
                                n for n, _ in worker.model_runner.model.named_modules()
                                if "attn" in n.lower() or "attention" in n.lower()
                            ][:20]
                        names = executor.collective_rpc(_list_attn)
                        result["error"] = (
                            "TurboQuant hooked 0 layers. "
                            "Attention module names: " + str(names[0])
                        )
                        print(json.dumps(result))
                        sys.exit(0)

                # --- Perplexity ---
                result["perplexity"] = compute_perplexity(llm, prompt)

                if USE_TQ:
                    # Reset TQ capture state before next generation
                    def _reset(worker):
                        tq_states = getattr(worker.model_runner, "_tq_states", {{}})
                        for s in tq_states.values():
                            s.reset()
                        return len(tq_states)
                    executor.collective_rpc(_reset)

                # --- TTFT ---
                result["ttft_s"] = round(measure_ttft(llm, prompt), 4)

                if USE_TQ:
                    def _reset2(worker):
                        tq_states = getattr(worker.model_runner, "_tq_states", {{}})
                        for s in tq_states.values():
                            s.reset()
                    executor.collective_rpc(_reset2)

                # --- Decode throughput ---
                result["decode_tps"] = round(
                    measure_decode_throughput(llm, prompt, NUM_GENERATE_TOKENS), 2
                )

                # --- VRAM (nvidia-smi, post-generation) ---
                result["peak_vram_mb"] = get_nvidia_smi_mb()

                # --- Free KV cache (TQ only) ---
                if USE_TQ:
                    def _reset3(worker):
                        tq_states = getattr(worker.model_runner, "_tq_states", {{}})
                        for s in tq_states.values():
                            s.reset()
                    executor.collective_rpc(_reset3)

                    def _free(worker):
                        from turboquant.vllm_attn_backend import free_kv_cache
                        return free_kv_cache(worker.model_runner)
                    freed = executor.collective_rpc(_free)
                    result["tq_bytes_freed"] = freed[0]

            except Exception as exc:
                result["error"] = str(exc)

            print(json.dumps(result))

        if __name__ == "__main__":
            main()
    """)
