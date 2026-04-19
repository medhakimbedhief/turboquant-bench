"""
vLLM engine creation for baseline and TurboQuant modes.

Both functions are designed to be called from INSIDE a subprocess
(via subprocess_runner.py). They must not be called in the notebook
kernel directly, since each subprocess gets a clean GPU state.

TQ hook installation follows the pattern from 0xSero/turboquant/benchmark.py:
  executor.collective_rpc() dispatches to each worker's model_runner.
The vLLM 0.18 executor access chain is:
  llm.llm_engine -> engine_core -> engine_core -> model_executor
(each getattr uses a fallback so the code degrades gracefully if the
 internal structure changes across minor vLLM releases).
"""

import os


def _set_env():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get_executor(llm):
    """Navigate vLLM 0.18's nested engine structure to reach the executor."""
    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    return inner.model_executor


def create_baseline_engine(config: dict, context_length: int):
    """
    Create a standard vLLM LLM instance (BF16, no TurboQuant).

    Parameters
    ----------
    config : dict
        The experiment CONFIG dict from the notebook.
    context_length : int
        max_model_len to pass to vLLM (determines KV cache pool size).

    Returns
    -------
    vllm.LLM
    """
    _set_env()
    from vllm import LLM

    return LLM(
        model=config["model_id"],
        dtype=config["dtype"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        tensor_parallel_size=config["tensor_parallel_size"],
        max_model_len=context_length,
        max_num_seqs=1,
        trust_remote_code=True,
    )


def create_tq_engine(config: dict, context_length: int):
    """
    Create a vLLM engine with TurboQuant KV-cache compression hooks installed.

    Steps:
    1. Create the engine identically to baseline.
    2. Install TQ hooks via executor.collective_rpc().
    3. Verify ≥1 layer was hooked; raise RuntimeError with diagnostic info if not.

    Parameters
    ----------
    config : dict
        The experiment CONFIG dict.
    context_length : int
        max_model_len for vLLM.

    Returns
    -------
    tuple[vllm.LLM, int]
        (llm, num_hooked_layers)
    """
    _set_env()
    llm = create_baseline_engine(config, context_length)
    executor = _get_executor(llm)

    key_bits = config["tq_key_bits"]
    value_bits = config["tq_value_bits"]

    def _install(worker):
        from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE
        states = install_turboquant_hooks(
            worker.model_runner,
            key_bits=key_bits,
            value_bits=value_bits,
            buffer_size=128,
            mode=MODE_ACTIVE,
        )
        return len(states)

    hooks_per_gpu = executor.collective_rpc(_install)
    num_hooked = hooks_per_gpu[0]

    if num_hooked == 0:
        # Diagnostic: list the model's attention-related module names so the
        # caller can figure out what hook registration needs to match.
        def _list_attn_names(worker):
            return [
                name
                for name, _ in worker.model_runner.model.named_modules()
                if "attn" in name.lower() or "attention" in name.lower()
            ][:30]

        attn_names = executor.collective_rpc(_list_attn_names)
        raise RuntimeError(
            "TurboQuant attached to 0 attention layers.\n"
            "Attention module names found in the model:\n"
            + "\n".join(f"  {n}" for n in attn_names[0])
        )

    print(f"TurboQuant attached to {num_hooked}/32 attention layers")
    return llm, num_hooked


def reset_tq_states(llm) -> int:
    """
    Reset TurboQuant capture state on all workers.

    Must be called between requests when reusing a TQ engine, otherwise
    the compressed store from the previous request bleeds into the next.

    Returns the number of TQ states that were reset.
    """
    executor = _get_executor(llm)

    def _reset(worker):
        tq_states = getattr(worker.model_runner, "_tq_states", {})
        for state in tq_states.values():
            state.reset()
        return len(tq_states)

    results = executor.collective_rpc(_reset)
    return results[0]


def free_kv_cache(llm) -> int:
    """
    Free vLLM's paged KV cache for TQ-hooked layers.

    After prefill, TurboQuant has compressed all KV pairs into its own store.
    Calling this replaces the large paged tensors with minimal placeholders,
    then flushes the CUDA allocator cache, reclaiming VRAM.

    Returns bytes freed (from GPU 0).
    """
    executor = _get_executor(llm)

    def _free(worker):
        from turboquant.vllm_attn_backend import free_kv_cache as tq_free
        return tq_free(worker.model_runner)

    freed = executor.collective_rpc(_free)
    return freed[0]
