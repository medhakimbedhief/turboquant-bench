# TurboQuant KV Cache Benchmark

Reproducible benchmarking experiment for [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV cache compression on a single RTX 3090 (24 GB).

**Model**: `VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct` (same architecture as Llama 3.1 8B, ungated — no HF token required)

## Quick Start (RunPod RTX 3090)

```bash
# 1. Clone this repo
git clone <this-repo> turboquant-bench && cd turboquant-bench

# 2. Run setup (installs all deps, downloads model, pre-caches dataset)
bash setup.sh

# 3. Launch Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser

# 4. Open notebook/turboquant_experiment.ipynb and Run All
```

## What the experiment measures

| Metric | Why |
|--------|-----|
| Peak VRAM (MB) | How much memory does the KV cache consume? |
| Perplexity (WikiText-2) | Does compression hurt language model quality? |
| TTFT / Decode tok/s | What is the latency cost of TQ? |
| Max context before OOM | How much more context fits on the same GPU? |

Tested at context lengths: **2K, 8K, 32K tokens** with 3 repetitions each (median reported).

## Project structure

```
turboquant-bench/
├── README.md
├── requirements.txt
├── setup.sh                         # One-shot setup for RunPod
│
├── notebook/
│   └── turboquant_experiment.ipynb  # Primary deliverable — run this
│
├── bench/                           # Supporting modules (imported by notebook)
│   ├── engine.py                    # vLLM engine creation (baseline & TQ)
│   ├── perplexity.py                # WikiText-2 perplexity computation
│   ├── memory.py                    # VRAM tracking (nvidia-smi)
│   ├── latency.py                   # TTFT and decode tok/s
│   ├── oom_search.py                # Binary search for max context
│   └── subprocess_runner.py         # Isolated subprocess benchmark runner
│
└── results/                         # JSON data + PNG plots written here
```

## Key design decisions

- **Subprocess isolation**: Each measurement runs in a fresh subprocess. vLLM allocates its KV cache at engine startup and never releases it while the engine is alive. Reusing an engine would make VRAM readings meaningless. See `bench/subprocess_runner.py`.
- **Fair comparison**: `gpu_memory_utilization`, `dtype`, and input text are identical between baseline and TQ runs. The only variable is TurboQuant.
- **nvidia-smi for VRAM**: More accurate than `torch.cuda.max_memory_allocated()` for vLLM workloads since the paged KV allocator operates outside PyTorch's tracked heap.

## References

- TurboQuant paper: https://arxiv.org/abs/2504.19874
- TurboQuant implementation: https://github.com/0xSero/turboquant
- Model: https://huggingface.co/VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct
