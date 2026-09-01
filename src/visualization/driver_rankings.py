"""Plot driver season rankings with uncertainty."""

from __future__ import annotations

import logging
import warnings
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Common shorthand -> Ergast driverRef aliases.
DRIVER_ALIASES: dict[str, str] = {
    "verstappen": "max_verstappen",
    "max": "max_verstappen",
    "magnussen": "kevin_magnussen",
    "kvyat": "daniil_kvyat",
    "perez": "sergio_perez",
    "alonso": "fernando_alonso",
    "hamilton": "lewis_hamilton",
    "leclerc": "charles_leclerc",
    "sainz": "carlos_sainz",
    "russell": "george_russell",
    "piastri": "oscar_piastri",
    "stroll": "lance_stroll",
    "gasly": "pierre_gasly",
    "ocon": "esteban_ocon",
    "bottas": "valtteri_bottas",
    "hulkenberg": "nico_hulkenberg",
    "albon": "alexander_albon",
    "tsunoda": "yuki_tsunoda",
    "ricciardo": "daniel_ricciardo",
}

BOOTSTRAP_CAPTION = (
    "Rank uncertainty: resample each driver's finished races within the season 200×, "
    "recompute cumulative skill, re-rank the grid; shaded band = 2.5th–97.5th percentile "
    "of resulting rank."
)


def _format_driver_name(ref: str) -> str:
    """Turn driverRef into a display name (e.g. max_verstappen -> Max Verstappen)."""
    parts = str(ref).split("_")
    if len(parts) > 1 and parts[0] in {"max", "lewis", "charles", "carlos", "george", "fernando"}:
        return " ".join(p.capitalize() for p in parts)
    return parts[-1].replace("_", " ").title()


def _build_ref_index(df: pd.DataFrame) -> dict[str, int]:
    pairs = df[["driverId", "driverRef"]].dropna().drop_duplicates(subset=["driverRef"])
    return {
        str(ref).lower(): int(did)
        for did, ref in zip(pairs["driverId"], pairs["driverRef"])
    }


def _resolve_single_driver(query: str, refs: dict[str, int]) -> Tuple[Optional[int], str]:
    """Resolve one driver query to driverId. Returns (id, canonical_ref)."""
    key = str(query).lower().strip()
    if key.isdigit():
        return int(key), key

    if key in refs:
        return refs[key], key

    alias = DRIVER_ALIASES.get(key)
    if alias and alias in refs:
        return refs[alias], alias

    suffix_matches = [r for r in refs if r.endswith(f"_{key}") or r == key]
    if len(suffix_matches) == 1:
        canonical = suffix_matches[0]
        return refs[canonical], canonical

    if len(suffix_matches) > 1:
        warnings.warn(f"Ambiguous driver '{query}': matches {suffix_matches}; using first.")
        canonical = sorted(suffix_matches)[0]
        return refs[canonical], canonical

    substring = [r for r in refs if key in r]
    if len(substring) == 1:
        return refs[substring[0]], substring[0]

    return None, key


def _resolve_drivers(
    df: pd.DataFrame,
    drivers: Iterable[str],
    season: Optional[int] = None,
) -> List[int]:
    """Map driverRef strings (case-insensitive, with aliases) to driverId."""
    scope = df[df["season"] == season] if season is not None and "season" in df.columns else df
    refs = _build_ref_index(scope if not scope.empty else df)

    ids: List[int] = []
    missing: List[str] = []
    for d in drivers:
        did, _ = _resolve_single_driver(d, refs)
        if did is not None:
            ids.append(did)
        else:
            missing.append(str(d))

    if missing:
        available = sorted(refs.keys())
        season_note = f" in season {season}" if season is not None else ""
        raise ValueError(
            f"Could not resolve driver(s){season_note}: {missing}. "
            f"Available refs: {', '.join(available[:20])}"
            + (" ..." if len(available) > 20 else "")
        )
    return ids


def resolve_driver_labels(
    rankings: pd.DataFrame,
    drivers: List[str],
    season: int,
) -> List[dict]:
    """Return resolved driver metadata for CLI logging."""
    scope = rankings[rankings["season"] == season]
    refs = _build_ref_index(scope if not scope.empty else rankings)
    out = []
    for d in drivers:
        did, canonical = _resolve_single_driver(d, refs)
        if did is None:
            out.append({"query": d, "driverId": None, "driverRef": None})
            continue
        row = scope[scope["driverId"] == did].iloc[0]
        out.append(
            {
                "query": d,
                "driverId": did,
                "driverRef": row.get("driverRef", canonical),
                "constructorRef": row.get("constructorRef", ""),
            }
        )
    return out


def _apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
        }
    )


def plot_driver_rankings(
    rankings: pd.DataFrame,
    season: int,
    drivers: List[str],
    output_path: Optional[str] = None,
    mode: str = "rank",
    title: Optional[str] = None,
) -> plt.Figure:
    """Dual-panel figure: cumulative rank (inverted y) + per-race skill."""
    _apply_plot_style()
    driver_ids = _resolve_drivers(rankings, drivers, season=season)
    sub = rankings[(rankings["season"] == season) & (rankings["driverId"].isin(driver_ids))].copy()
    if sub.empty:
        raise ValueError(f"No data for season={season} drivers={drivers}")

    meta = (
        sub.groupby("driverId")
        .agg(
            driverRef=("driverRef", "first"),
            constructorRef=("constructorRef", "first"),
        )
        .to_dict("index")
    )

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(driver_ids), 1)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.28)

    ax_rank, ax_skill = axes
    band_labeled = False

    for i, did in enumerate(driver_ids):
        g = sub[sub["driverId"] == did].sort_values("round")
        if g.empty:
            continue

        info = meta.get(did, {})
        ref = info.get("driverRef", str(did))
        team = info.get("constructorRef", "")
        name = _format_driver_name(ref)
        label = f"{name} ({team})" if team else name
        color = colors[i]

        rounds = g["round"].to_numpy()
        ranks = g["rank"].to_numpy()

        ax_rank.plot(rounds, ranks, marker="o", label=label, linewidth=2.2, color=color, markersize=5)
        if "rank_lo" in g.columns and g["rank_lo"].notna().any():
            band_label = "95% bootstrap rank interval" if not band_labeled else None
            ax_rank.fill_between(
                rounds,
                g["rank_lo"].to_numpy(),
                g["rank_hi"].to_numpy(),
                alpha=0.22,
                color=color,
                label=band_label,
            )
            band_labeled = True

        ax_skill.plot(
            rounds,
            g["race_skill"].to_numpy(),
            marker="o",
            linestyle="--",
            linewidth=1.8,
            markersize=4,
            alpha=0.9,
            color=color,
            label=label,
        )
        ax_skill.plot(
            rounds,
            g["season_skill"].to_numpy(),
            linestyle="-",
            linewidth=1.2,
            alpha=0.45,
            color=color,
        )

    ax_rank.invert_yaxis()
    ax_rank.set_ylabel("Season rank (1 = best)")
    ax_rank.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_rank.set_title("Cumulative season rank (as-of-round)", loc="left", fontsize=12, pad=8)
    ax_rank.legend(loc="upper right", framealpha=0.92)
    ax_rank.grid(True, alpha=0.35)

    ax_skill.set_xlabel("Round")
    ax_skill.set_ylabel("Skill score")
    ax_skill.set_title("Per-race skill (dashed) and cumulative mean (solid, faint)", loc="left", fontsize=12, pad=8)
    ax_skill.legend(loc="upper right", framealpha=0.92)
    ax_skill.grid(True, alpha=0.35)

    main_title = title or f"{season} cumulative driver ranking"
    fig.suptitle(main_title, fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.955,
        "Car-adjusted performance f(D,T,R) · rank uses only rounds 1…r (causal as-of-round)",
        ha="center",
        fontsize=10,
        color="0.35",
    )
    fig.text(0.5, 0.02, BOOTSTRAP_CAPTION, ha="center", fontsize=9, color="0.45", wrap=True)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        svg_path = output_path.rsplit(".", 1)[0] + ".svg"
        fig.savefig(svg_path, bbox_inches="tight")
    return fig
