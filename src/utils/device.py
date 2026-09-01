"""CUDA device selection shared across experiment scripts."""

from __future__ import annotations

import torch

import config as cfg


def _scatter_cuda_works(device: torch.device) -> bool:
    """Return True when torch_scatter scatter ops run on ``device``."""
    if device.type != "cuda":
        return True
    try:
        import torch_scatter

        x = torch.tensor([1.0, 2.0], device=device)
        idx = torch.tensor([0, 0], device=device)
        torch_scatter.scatter_max(x, idx)
        return True
    except Exception:
        return False


def get_device(gpu_id: int | None = None) -> torch.device:
    """Return ``cuda:{gpu_id}``, defaulting to ``cfg.DEFAULT_GPU_ID``.

    Falls back to CPU when CUDA is visible but PyG scatter extensions were
    built without GPU support (common torch/torch-scatter wheel mismatch).
    """
    if gpu_id is None:
        gpu_id = cfg.DEFAULT_GPU_ID
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested GPU {gpu_id} is unavailable. "
                f"Visible devices: {torch.cuda.device_count()}"
            )
        device = torch.device(f"cuda:{gpu_id}")
        if _scatter_cuda_works(device):
            print(f"-> Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return device
        print(
            "-> CUDA is available but torch_scatter lacks GPU support; "
            "falling back to CPU. Reinstall with:\n"
            "   pip install --force-reinstall torch-scatter "
            "-f https://data.pyg.org/whl/torch-<torch-version>+cu<cuda>.html"
        )
    else:
        print("-> CUDA unavailable, using CPU")
    return torch.device("cpu")
