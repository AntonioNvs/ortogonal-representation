#!/usr/bin/env python3
"""SSAC27 abstract figures from frozen application JSONs (separate panels).

  A — Shapley attribution
  B — equal-car GOAT
  C — 2026 mobility screen
  D — retrospective undervalued leads 2014–today

    python src/experiments/plots/plot_ssac27_abstract_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _panel_a(ax, shares: dict) -> None:
    labels = ["Driver", "Constructor", "Context"]
    vals = [shares["driver"], shares["constructor"], shares["context"]]
    colors = ["#1a5f7a", "#d71920", "#8a8a8a"]
    left = 0.0
    for v, c, lab in zip(vals, colors, labels):
        ax.barh(0, v, left=left, height=0.55, color=c, alpha=0.9)
        mid = left + v / 2
        ax.text(
            mid,
            0,
            f"{lab}\n{100 * v:.0f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            linespacing=1.15,
        )
        left += v
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_xlabel("Share of modeled systematic race utility", fontsize=10)
    ax.set_title("What does the model attribute?", fontsize=12, loc="left", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _panel_b(ax, ranking: list[dict], top_n: int = 5) -> None:
    rows = sorted(ranking, key=lambda r: r["expected_points"], reverse=True)[:top_n]
    rows = list(reversed(rows))
    y = np.arange(len(rows))
    pts = [r["expected_points"] for r in rows]
    labels = [f"{r['driver_name']} ({int(r['peak_season'])})" for r in rows]
    p_title = [r["p_title"] for r in rows]

    ax.barh(y, pts, color="#d71920", alpha=0.85, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Expected points (equal car, 24 races)", fontsize=10)
    ax.set_title("Who wins in equal cars?", fontsize=12, loc="left", pad=10)
    x_hi = max(pts) * 1.22
    ax.set_xlim(0, x_hi)
    for yi, pt, p in zip(y, pts, p_title):
        ax.text(pt + x_hi * 0.015, yi, f"{100 * p:.0f}% title", va="center", fontsize=9, color="#333")
    ax.spines[["top", "right"]].set_visible(False)


def _panel_c(ax, undervalued: dict) -> None:
    forecast = undervalued["watchlists"]["forecast"]["watchlist"]
    as_of = undervalued["watchlists"]["forecast"]["as_of_season"]
    max_age = undervalued.get("watchlists", {}).get("eligibility", {}).get("max_age")

    rows = list(reversed(forecast[:5]))
    y = np.arange(len(rows))
    uvs = [r["uv"] for r in rows]
    labels = []
    for r in rows:
        age = r.get("age")
        age_s = f", {age:.0f}y" if age is not None and np.isfinite(age) else ""
        labels.append(f"{r['driver_name']} ({as_of}{age_s})")
    colors = [
        "#1a5f7a" if r["driver_name"] == "Arvid Lindblad" else "#6b8a9a"
        for r in rows
    ]

    ax.barh(y, uvs, color=colors, alpha=0.9, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Undervaluation index  z(skill) − z(car)", fontsize=10)
    title = f"Mobility screen — {as_of}"
    if max_age is not None:
        title = f"Mobility screen — {as_of} (age ≤ {int(max_age)})"
    ax.set_title(title, fontsize=12, loc="left", pad=10)
    x_hi = max(uvs) * 1.18 if uvs else 1.0
    x_lo = min(0.0, min(uvs) * 1.15) if uvs else 0.0
    ax.set_xlim(x_lo, x_hi)
    ax.axvline(0.0, color="#bbbbbb", lw=0.8, zorder=0)
    for yi, uv, row in zip(y, uvs, rows):
        tag = " ← lead" if row["driver_name"] == "Arvid Lindblad" else ""
        offset = x_hi * 0.02 if uv >= 0 else -x_hi * 0.02
        ha = "left" if uv >= 0 else "right"
        ax.text(uv + offset, yi, f"{uv:.2f}{tag}", va="center", ha=ha, fontsize=9, color="#222")
    ax.spines[["top", "right"]].set_visible(False)


def _panel_d(ax, undervalued: dict) -> None:
    """Season leads: most undervalued eligible driver each year, 2014–today."""
    screens = undervalued.get("seasonal_screens") or []
    max_age = undervalued.get("watchlists", {}).get("eligibility", {}).get("max_age")
    if not screens:
        ax.text(0.5, 0.5, "No seasonal_screens in undervalued.json\n(re-run run_undervalued_backtest.py)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666")
        ax.set_axis_off()
        return

    rows = list(screens)
    y = np.arange(len(rows))
    uvs = [r["lead"]["uv"] for r in rows]
    labels = []
    for r in rows:
        age = r["lead"].get("age")
        age_s = f", {age:.0f}y" if age is not None and np.isfinite(age) else ""
        labels.append(f"{r['lead']['driver_name']} ({int(r['season'])}{age_s})")
    highlight = {
        "Max Verstappen", "Carlos Sainz", "Lando Norris", "Charles Leclerc",
        "George Russell", "Oscar Piastri", "Arvid Lindblad",
    }
    colors = [
        "#1a5f7a" if r["lead"]["driver_name"] in highlight else "#6b8a9a"
        for r in rows
    ]

    ax.barh(y, uvs, color=colors, alpha=0.9, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Undervaluation index  z(skill) − z(car)", fontsize=10)
    title = "Most undervalued by season (2014–today)"
    if max_age is not None:
        title = f"Most undervalued by season (2014–today, age ≤ {int(max_age)})"
    ax.set_title(title, fontsize=12, loc="left", pad=10)
    x_hi = max(uvs) * 1.15 if uvs else 1.0
    ax.set_xlim(0, x_hi)
    for yi, uv in zip(y, uvs):
        ax.text(uv + x_hi * 0.015, yi, f"{uv:.2f}", va="center", fontsize=8, color="#222")
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def render_separate(
    *,
    goat_path: Path,
    undervalued_path: Path,
    benchmark_path: Path,
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    goat = _load_json(goat_path)
    undervalued = _load_json(undervalued_path)
    bench = _load_json(benchmark_path)
    shares = bench["sources"]["orthogonal_shapley"]["shapley_season_mean"]

    out_dir.mkdir(parents=True, exist_ok=True)

    fig_a, ax_a = plt.subplots(figsize=(7.2, 2.6), dpi=220)
    _panel_a(ax_a, shares)
    fig_a.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.30)
    _save(fig_a, out_dir / "panel_a_attribution.png")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(7.6, 4.2), dpi=220)
    _panel_b(ax_b, goat["ranking"], top_n=5)
    fig_b.subplots_adjust(left=0.32, right=0.96, top=0.88, bottom=0.14)
    _save(fig_b, out_dir / "panel_b_goat_equal_car.png")
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(7.8, 4.2), dpi=220)
    _panel_c(ax_c, undervalued)
    fig_c.subplots_adjust(left=0.32, right=0.96, top=0.88, bottom=0.14)
    _save(fig_c, out_dir / "panel_c_mobility_screen_2026.png")
    plt.close(fig_c)

    n_seasons = len(undervalued.get("seasonal_screens") or [])
    fig_h = max(4.5, 0.38 * max(n_seasons, 8) + 1.2)
    fig_d, ax_d = plt.subplots(figsize=(7.8, fig_h), dpi=220)
    _panel_d(ax_d, undervalued)
    fig_d.subplots_adjust(left=0.34, right=0.96, top=0.92, bottom=0.10)
    _save(fig_d, out_dir / "panel_d_retrospective_undervalued_2014.png")
    plt.close(fig_d)


def main() -> None:
    p = argparse.ArgumentParser(description="SSAC27 abstract figures (separate panels)")
    p.add_argument("--goat", default="output/applications/goat_equal_car/goat_equal_car.json")
    p.add_argument("--undervalued", default="output/applications/undervalued/undervalued.json")
    p.add_argument("--benchmark", default="output/validation_benchmark/benchmark.json")
    p.add_argument(
        "--output-dir",
        default="output/applications/ssac27_abstract_figure",
    )
    args = p.parse_args()
    render_separate(
        goat_path=Path(args.goat),
        undervalued_path=Path(args.undervalued),
        benchmark_path=Path(args.benchmark),
        out_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
