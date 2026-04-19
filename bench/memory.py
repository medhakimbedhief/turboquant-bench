"""
VRAM tracking utilities.

nvidia-smi is the authoritative measurement because vLLM's paged KV blocks
are allocated in bulk at engine startup via a custom allocator — torch's
peak memory stats track PyTorch allocations correctly, but nvidia-smi gives
the ground-truth view from the driver.
"""

import subprocess
import torch


def reset_memory_tracking() -> None:
    """Reset PyTorch peak memory stats and flush the allocator cache."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def get_peak_vram_mb() -> float:
    """Peak PyTorch-tracked VRAM since last reset_memory_tracking(), in MB."""
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def get_current_vram_mb() -> float:
    """Currently allocated PyTorch VRAM, in MB."""
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_nvidia_smi_mb(gpu_index: int = 0) -> int:
    """
    Query nvidia-smi for the amount of GPU memory currently in use.

    Returns used MB for the specified gpu_index, or -1 if nvidia-smi fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 2:
                idx = int(parts[0].strip())
                used_mb = int(parts[1].strip())
                if idx == gpu_index:
                    return used_mb
        return -1
    except Exception:
        return -1
