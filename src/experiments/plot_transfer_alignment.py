"""Plots for the transfer tier-alignment experiment.

Reads the CSVs written by ``experiments.transfer_tier_alignment`` (so tweaking
a parameter only re-renders -- it never re-runs inference) and produces four
figures:

  * scatter -- per transfer, X = tier move (jittered) vs Y = predicted car-effect
    Delta = pred_new - pred_old. Negative slope => plausible.
  * sign    -- mean Delta per direction bucket (promotion / lateral / demotion)
    with +/-1 SEM error bars. Expect promotion < 0, demotion > 0, lateral ~ 0.
  * drivers -- per-race deep-dive for chosen drivers: actual vs pred_new vs
    pred_old across the transfer season (finishing position, 1 at top).
  * models  -- one bar per model of Spearman rho (or sign-agreement), making the
    "orthogonality sharpening" claim (high-lambda aligns best).

Usage
-----
    python -m src.experiments.plot_transfer_alignment --plots scatter,sign,drivers,models
    python -m src.experiments.plot_transfer_alignment --plots drivers \\
        --drivers hamilton,alonso,sainz,gasly --min-transfer-year 2023
    python -m src.experiments.plot_transfer_alignment --plots scatter \\
        --models high --annotate-top-k 8
    python -m src.experiments.plot_transfer_alignment --plots sign --models high
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _p in (ROOT_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg

DEFAULT_DRIVERS = ["hamilton", "alonso", "sainz", "gasly"]
MODEL_LAMBDA_LABEL = {"model_no_orthogonal": "zero (λ=0)",
                      "model_ablation_l01": "low (λ=0.1)",
                      "model_orthogonal": "high (λ=1.0)"}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_transfers(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "tier_alignment_transfers.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} — run experiments.transfer_tier_alignment first."
        )
    df = pd.read_csv(path)
    for col in ("tier_dir", "delta_mean", "pred_new_mean", "pred_old_mean", "actual_mean"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_races(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "tier_alignment_races.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} — run experiments.transfer_tier_alignment first."
        )
    df = pd.read_csv(path)
    for col in ("actual", "pred_new", "pred_old", "delta", "tier_dir", "round"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_summary(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "tier_alignment_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} — run experiments.transfer_tier_alignment first."
        )
    return pd.read_csv(path)


def _filter_transfers(df: pd.DataFrame, min_year, max_year, models):
    df = df[(df["new_season"] >= min_year) & (df["new_season"] <= max_year)]
    if models:
        df = df[df["model"].isin(models)]
    return df


def _model_label(model_name: str) -> str:
    return MODEL_LAMBDA_LABEL.get(model_name, model_name)


def _savefig(fig, output_dir: str, name: str):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 1: scatter — market direction vs predicted car-effect
# ---------------------------------------------------------------------------

def plot_scatter(transfers: pd.DataFrame, output_dir: str, annotate_top_k: int,
                 drivers: list[str], seed: int):
    models = sorted(transfers["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(seed)

    for ax, model in zip(axes, models):
        g = transfers[transfers["model"] == model].dropna(subset=["tier_dir", "delta_mean"])
        jitter = rng.uniform(-0.18, 0.18, size=len(g))
        x = g["tier_dir"].to_numpy() + jitter
        y = g["delta_mean"].to_numpy()

        ax.scatter(x, y, s=28, alpha=0.75, color="#333366", zorder=3)
        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
        ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=1)

        # Annotate top-k by |Delta| plus any explicitly requested drivers.
        annotate = set(drivers)
        top = g.assign(_abs=np.abs(g["delta_mean"])).nlargest(annotate_top_k, "_abs")
        for _, row in top.iterrows():
            annotate.add(str(row["driver_ref"]))

        for _, row in g.iterrows():
            if str(row["driver_ref"]) in annotate:
                ax.annotate(str(row["driver_ref"]),
                            (row["tier_dir"], row["delta_mean"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)

        ax.set_title(_model_label(model))
        ax.set_xlabel("tier move (score(new@T) - score(old@T))")
        ax.set_xticks([-2, -1, 0, 1, 2])
        if ax is axes[0]:
            ax.set_ylabel("Delta = pred_new - pred_old (position)")

    fig.suptitle("Market direction vs predicted car-effect\n(negative slope = plausible: "
                 "up-tier => better finish)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "tier_alignment_scatter.png")


# ---------------------------------------------------------------------------
# Plot 2: sign — mean Delta per direction bucket
# ---------------------------------------------------------------------------

def plot_sign(transfers: pd.DataFrame, output_dir: str):
    models = sorted(transfers["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]

    order = ["promotion", "lateral", "demotion"]

    def bucket(td):
        if pd.isna(td):
            return None
        if td > 0:
            return "promotion"
        if td < 0:
            return "demotion"
        return "lateral"

    for ax, model in zip(axes, models):
        g = transfers[transfers["model"] == model].copy()
        g["_bucket"] = g["tier_dir"].apply(bucket)
        g = g.dropna(subset=["_bucket", "delta_mean"])

        means, sems, labels = [], [], []
        for b in order:
            sub = g[g["_bucket"] == b]["delta_mean"]
            if len(sub) == 0:
                means.append(0.0)
                sems.append(0.0)
            else:
                means.append(sub.mean())
                sems.append(sub.std(ddof=1) / np.sqrt(len(sub)))
            labels.append(f"{b}\n(n={len(sub)})")

        x = np.arange(len(order))
        colors = ["#2a7a3a", "#888888", "#a32c2c"]
        ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(_model_label(model))
        if ax is axes[0]:
            ax.set_ylabel("mean Delta (position; <0 = better)")

    fig.suptitle("Does the sign of the car-effect agree with the tier move?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir, "tier_alignment_sign.png")


# ---------------------------------------------------------------------------
# Plot 3: drivers — per-race deep-dive
# ---------------------------------------------------------------------------

def plot_drivers(races: pd.DataFrame, output_dir: str, drivers: list[str], models: list[str]):
    if not drivers:
        # fall back to bundled defaults that are actually present in the data
        drivers = [d for d in DEFAULT_DRIVERS if d in set(races["driver_ref"].astype(str))]
        if not drivers:
            print("  no drivers found in data — skipping drivers plot")
            return

    if models:
        races = races[races["model"].isin(models)]

    # One panel per (driver, transfer-season), most recent transfer per driver.
    panels = []
    for d in drivers:
        sub = races[races["driver_ref"].astype(str) == d]
        if sub.empty:
            print(f"  driver '{d}' not found — skipping")
            continue
        newest_season = sub["new_season"].max()
        sub = sub[sub["new_season"] == newest_season]
        panels.append((d, sub))

    if not panels:
        return

    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (driver, sub) in zip(axes, panels):
        for model in sorted(sub["model"].unique()):
            m = sub[sub["model"] == model].sort_values("round")
            ax.plot(m["round"], m["actual"], marker="o", ls="-", label=f"{_model_label(model)} actual")
            ax.plot(m["round"], m["pred_new"], marker="s", ls="--", label=f"{_model_label(model)} pred_new")
            ax.plot(m["round"], m["pred_old"], marker="^", ls=":", label=f"{_model_label(model)} pred_old")
        ax.invert_yaxis()
        ax.set_xlabel("round")
        ax.set_ylabel("finishing position")
        ax.set_title(f"{driver}: {sub['old_constructor_ref'].iloc[0]} -> "
                     f"{sub['new_constructor_ref'].iloc[0]} ({sub['new_season'].iloc[0]})")
        ax.legend(fontsize=6, ncol=1, loc="best")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Counterfactual vs actual (position, 1 at top)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "tier_alignment_drivers.png")


# ---------------------------------------------------------------------------
# Plot 4: models — orthogonality sharpening
# ---------------------------------------------------------------------------

def plot_models(summary: pd.DataFrame, output_dir: str, metric: str):
    metric = metric or "rho"
    col = "spearman_rho" if metric == "rho" else "overall_agree_frac"
    if col not in summary.columns or summary[col].isna().all():
        print(f"  summary has no usable '{col}' column — skipping models plot")
        return

    df = summary.dropna(subset=[col]).sort_values("lambda_orthogonal")
    labels = [_model_label(m) for m in df["model"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(df)), df[col], color="#333366", alpha=0.85)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_ylabel("Spearman rho(tier_dir, -Delta)" if metric == "rho" else "sign-agreement rate")
    ax.set_title("Orthogonality sharpening: higher lambda should align best")
    for b, v in zip(bars, df[col]):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.03),
                f"{v:+.3f}" if metric == "rho" else f"{v:.0%}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_dir, "tier_alignment_models.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot the transfer tier-alignment experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--plots", type=str, default="scatter,sign,drivers,models",
                        help="Comma list of plots: scatter,sign,drivers,models.")
    parser.add_argument("--min-transfer-year", type=int, default=2000,
                        help="Earliest transfer season to show (default: 2000).")
    parser.add_argument("--max-transfer-year", type=int, default=2100,
                        help="Latest transfer season to show (default: 2100).")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma list of model names to restrict to (default: all).")
    parser.add_argument("--drivers", type=str, default=None,
                        help="Comma list of driver refs for the drivers plot.")
    parser.add_argument("--annotate-top-k", type=int, default=8,
                        help="Annotate top-k transfers by |Delta| in the scatter.")
    parser.add_argument("--metric", type=str, default="rho", choices=["rho", "agree"],
                        help="Metric for the models plot: rho or agree.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory with the CSVs + where PNGs are saved.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the scatter jitter.")
    args = parser.parse_args()

    output_dir = args.output_dir or cfg.TRANSFER_TIER_ALIGNMENT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    plots = [p.strip() for p in args.plots.split(",") if p.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else []
    drivers = [d.strip() for d in args.drivers.split(",") if d.strip()] if args.drivers else []

    transfers = load_transfers(output_dir)
    transfers = _filter_transfers(transfers, args.min_transfer_year, args.max_transfer_year, models)

    if "scatter" in plots:
        print("Plotting scatter...")
        plot_scatter(transfers, output_dir, args.annotate_top_k, drivers, args.seed)
    if "sign" in plots:
        print("Plotting sign...")
        plot_sign(transfers, output_dir)
    if "drivers" in plots:
        print("Plotting drivers...")
        races = load_races(output_dir)
        races = races[(races["new_season"] >= args.min_transfer_year)
                      & (races["new_season"] <= args.max_transfer_year)]
        plot_drivers(races, output_dir, drivers, models)
    if "models" in plots:
        print("Plotting models...")
        plot_models(load_summary(output_dir), output_dir, args.metric)


if __name__ == "__main__":
    main()
