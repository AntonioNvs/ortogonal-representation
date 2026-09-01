"""CUDA device selection shared across experiment scripts."""

from __future__ import annotations

import torch

import config as cfg


def get_device(gpu_id: int | None = None) -> torch.device:
    """Return ``cuda:{gpu_id}``, defaulting to ``cfg.DEFAULT_GPU_ID``."""
    if gpu_id is None:
        gpu_id = cfg.DEFAULT_GPU_ID
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested GPU {gpu_id} is unavailable. "
                f"Visible devices: {torch.cuda.device_count()}"
            )
        device = torch.device(f"cuda:{gpu_id}")
        print(f"-> Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        return device
    print("-> CUDA unavailable, using CPU")
    return torch.device("cpu")
