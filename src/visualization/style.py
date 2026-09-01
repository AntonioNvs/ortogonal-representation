"""Shared seaborn styling and figure export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt

_PLOT_EXTENSIONS = (".png", ".svg", ".pdf", ".jpeg", ".jpg", ".webp")


def apply_plot_style() -> None:
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    except ImportError:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def resolve_plot_output_dir(output_path: str) -> Tuple[Path, str]:
    """Map a CLI path to ``(folder, basename)`` for multi-format export.

    Examples:
        ``output/plots/foo.png`` → ``output/plots/foo/``, basename ``foo``
        ``output/plots/foo``     → ``output/plots/foo/``, basename ``foo``
    """
    path = Path(output_path)
    if path.suffix.lower() in _PLOT_EXTENSIONS:
        title = path.stem
        out_dir = path.parent / title
    else:
        title = path.name or "figure"
        out_dir = path
    return out_dir, title


def save_figure(
    fig: plt.Figure,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save PNG, SVG, and PDF inside a dedicated folder named after the plot title."""
    out_dir, title = resolve_plot_output_dir(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{title}.png"
    svg_path = out_dir / f"{title}.svg"
    pdf_path = out_dir / f"{title}.pdf"

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if metadata is not None:
        meta_path = out_dir / f"{title}.meta.json"
        payload = {**metadata, "title": title, "formats": ["png", "svg", "pdf"]}
        with open(meta_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    return out_dir
