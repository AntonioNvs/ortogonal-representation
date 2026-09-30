"""Shared seaborn styling and figure export.

Style follows the project's plot style reference: whitegrid theme, DejaVu Sans,
despined axes, ``dimgrey`` tick/annotation text, and dual PNG/SVG/PDF export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

_PLOT_EXTENSIONS = (".png", ".svg", ".pdf", ".jpeg", ".jpg", ".webp")


def apply_plot_style() -> None:
    """Set the shared figure theme (whitegrid + DejaVu Sans)."""
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({"figure.facecolor": "white"})


def finalize_axes(ax) -> None:
    """Strip tick marks and soften tick labels for a single axes."""
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")


def despine_axes(*, top: bool = False, right: bool = False) -> None:
    """Remove spines from every axes in the current figure."""
    sns.despine(left=True, bottom=True, top=top, right=right)


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
