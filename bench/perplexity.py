"""
WikiText-2 perplexity computation.

Perplexity is measured on the INPUT text (prefill), not on generated text.
We use vLLM's prompt_logprobs feature to get P(token_i | token_0..i-1)
for every position in the prompt, then compute:

    PPL = exp(-1/N * sum(log P(token_i | context)))

Setting prompt_logprobs=1 guarantees that the actual prompt token's logprob
is always present in the returned dict (vLLM includes the real token even
when it isn't in the top-k).
"""

import math


def compute_perplexity(llm, prompt_text: str) -> float:
    """
    Compute the perplexity of ``prompt_text`` under ``llm``.

    Parameters
    ----------
    llm : vllm.LLM
        A fully initialised LLM instance (baseline or TQ engine).
    prompt_text : str
        The text to score. Should be natural text (e.g., WikiText-2).

    Returns
    -------
    float
        Perplexity. Returns inf if no valid logprobs were found.
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    # Encode without special tokens so we count exactly the content tokens
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    params = SamplingParams(
        prompt_logprobs=1,   # top-1 + actual token always present
        max_tokens=1,        # we only need prefill logprobs
        temperature=0,
    )

    outputs = llm.generate([prompt_text], params)
    output = outputs[0]

    # prompt_logprobs is a list[Optional[dict[int, Logprob]]]
    # position 0 → None (no context for the first token)
    # positions 1..N → dict mapping token_id -> Logprob namedtuple
    logprobs_list = output.prompt_logprobs

    total_nll = 0.0
    count = 0

    for i, lp_dict in enumerate(logprobs_list):
        if lp_dict is None:
            continue
        if i >= len(token_ids):
            break
        actual_token = token_ids[i]
        if actual_token in lp_dict:
            total_nll -= lp_dict[actual_token].logprob
            count += 1

    if count == 0:
        return float("inf")

    return math.exp(total_nll / count)
