"""
Latency measurement: TTFT and decode throughput.

TTFT approximation
------------------
vLLM's offline LLM.generate() interface does not expose streaming events,
so true TTFT cannot be measured without the server API. The standard
approximation is:

    TTFT ≈ time to generate exactly 1 token  (= mostly prefill)

This is a slight overestimate (includes 1 decode step) but is consistent
and reproducible.

Decode throughput
-----------------
    decode_tps = (total_tokens - 1) / (total_time - ttft)

where total_time covers all decode steps for num_tokens tokens and ttft
is the prefill estimate above.

Both functions do ONE warmup generation before measuring to ensure CUDA
kernels are compiled and cached (first call is always slower).
"""

import time

from vllm import SamplingParams


def measure_ttft(llm, prompt_text: str) -> float:
    """
    Approximate time-to-first-token (TTFT) in seconds.

    Runs one warmup generation, then times a single 1-token generation.

    Parameters
    ----------
    llm : vllm.LLM
    prompt_text : str

    Returns
    -------
    float  seconds
    """
    params = SamplingParams(max_tokens=1, temperature=0)
    # Warmup: compile kernels, fill CUDA caches
    llm.generate([prompt_text], params)
    # Measure
    t0 = time.perf_counter()
    llm.generate([prompt_text], params)
    return time.perf_counter() - t0


def measure_decode_throughput(llm, prompt_text: str, num_tokens: int) -> float:
    """
    Measure decode throughput in tokens per second.

    Parameters
    ----------
    llm : vllm.LLM
    prompt_text : str
    num_tokens : int
        Number of tokens to generate (config["num_generate_tokens"]).

    Returns
    -------
    float  tokens per second
    """
    params = SamplingParams(max_tokens=num_tokens, temperature=0)
    # Warmup
    llm.generate([prompt_text], params)

    ttft = measure_ttft(llm, prompt_text)

    t0 = time.perf_counter()
    output = llm.generate([prompt_text], params)
    total_time = time.perf_counter() - t0

    actual_tokens = len(output[0].outputs[0].token_ids)
    decode_time = max(total_time - ttft, 1e-3)
    # Subtract 1 because the first token is accounted for in ttft
    return max(actual_tokens - 1, 1) / decode_time
