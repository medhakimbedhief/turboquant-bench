#!/bin/bash
set -e

echo "=== TurboQuant Benchmark Setup ==="
echo "Model: VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct (ungated, no HF_TOKEN needed)"
echo ""

# 1. Install Python dependencies
echo "[1/5] Installing Python dependencies..."
pip install "vllm>=0.18.0" torch transformers datasets matplotlib pandas jupyter ipywidgets
echo "Done."

# 2. Install TurboQuant
echo "[2/5] Installing TurboQuant from 0xSero/turboquant..."
if [ -d "/opt/turboquant" ]; then
    echo "  /opt/turboquant already exists, pulling latest..."
    cd /opt/turboquant && git pull && pip install -e . && cd -
else
    git clone https://github.com/0xSero/turboquant.git /opt/turboquant
    cd /opt/turboquant && pip install -e . && cd -
fi
echo "Done."

# 3. Download model (ungated — no HF_TOKEN required)
echo "[3/5] Downloading VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct..."
huggingface-cli download VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct
echo "Done."

# 4. Pre-cache WikiText-2 dataset
echo "[4/5] Pre-caching WikiText-2 dataset..."
python -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
print(f'  train: {len(ds[\"train\"])} rows')
print(f'  test:  {len(ds[\"test\"])} rows')
"
echo "Done."

# 5. Smoke test
echo "[5/5] Running smoke test..."
python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.version.cuda}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  GPU:      {props.name}')
    print(f'  VRAM:     {props.total_memory / 1024**3:.1f} GB')
else:
    print('  GPU:      NOT AVAILABLE')

import vllm
print(f'  vLLM:     {vllm.__version__}')

import turboquant
print(f'  TQ:       OK (v{turboquant.__version__})')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Launch the notebook:"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
echo ""
echo "Then open: notebook/turboquant_experiment.ipynb"
