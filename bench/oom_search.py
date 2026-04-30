"""
Binary search for the maximum context length that fits in GPU VRAM.

Each candidate context length is tested in a fresh subprocess so that an
OOM error kills the child process, not the notebook kernel. The parent
simply checks the subprocess exit code.

Search range: [4096, 131072] tokens
Granularity: stops when upper - lower < 1024 tokens
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap


def find_max_context(config: dict, use_tq: bool = False) -> int:
    """
    Binary-search for the maximum context length before OOM.

    Parameters
    ----------
    config : dict
        Experiment CONFIG dict. gpu_memory_utilization must match
        what is used in the benchmark runs for a fair comparison.
    use_tq : bool
        If True, install TurboQuant hooks before the test generation.

    Returns
    -------
    int  Maximum context length that succeeded (tokens).
    """
    lo, hi = 4096, 131072
    best = lo

    print(f"OOM search ({'TurboQuant' if use_tq else 'Baseline'}):")
    print(f"  Range: [{lo:,} – {hi:,}] tokens, granularity 1,024")

    while hi - lo >= 1024:
        mid = ((lo + hi) // 2 // 1024) * 1024  # round to nearest 1024

        success = _try_context_length(config, mid, use_tq)
        status = "OK" if success else "OOM"
        print(f"  ctx={mid:>7,}: {status}")

        if success:
            best = mid
            lo = mid + 1024
        else:
            hi = mid - 1024

    print(f"  => Max context: {best:,} tokens")
    return best


def _try_context_length(config: dict, context_length: int, use_tq: bool) -> bool:
    """
    Run a minimal vLLM engine creation + 1 generation in a subprocess.

    Returns True if the subprocess exits with code 0 (success).
    Any non-zero exit (OOM, RuntimeError, timeout) returns False.
    """
    script = _generate_oom_probe_script(config, context_length, use_tq)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        script_path = f.name

    env = os.environ.copy()
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TURBOQUANT_REPO_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _generate_oom_probe_script(config: dict, context_length: int, use_tq: bool) -> str:
    """Generate Python code for the OOM probe subprocess."""
    config_repr = json.dumps(config)
    use_tq_repr = "True" if use_tq else "False"

    return textwrap.dedent(f"""\
        import os, sys
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        _repo_root = os.environ.get("TURBOQUANT_REPO_ROOT") or {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r}
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)

        import json
        from vllm import LLM, SamplingParams

        config = json.loads({config_repr!r})
        context_length = {context_length}
        use_tq = {use_tq_repr}

        llm = LLM(
            model=config["model_id"],
            dtype=config["dtype"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            tensor_parallel_size=config["tensor_parallel_size"],
            max_model_len=context_length,
            max_num_seqs=1,
            trust_remote_code=True,
        )

        if use_tq:
            engine = llm.llm_engine
            core = getattr(engine, "engine_core", engine)
            inner = getattr(core, "engine_core", core)
            executor = inner.model_executor

            key_bits = config["tq_key_bits"]
            value_bits = config["tq_value_bits"]

            def _install(worker):
                from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE
                return len(install_turboquant_hooks(
                    worker.model_runner,
                    key_bits=key_bits, value_bits=value_bits,
                    buffer_size=128, mode=MODE_ACTIVE,
                ))
            executor.collective_rpc(_install)

        # Short prompt to avoid tokenization issues at edge lengths
        prompt = "Hello" * 10
        params = SamplingParams(max_tokens=1, temperature=0)
        llm.generate([prompt], params)

        print("OK")
        sys.exit(0)
    """)
