"""
Driver feature extraction for the MLP driver encoder.

Computes 8 tabular features per driver **per race**, using a rolling window
of the last N races (shifted by 1 to avoid data leakage).  Each row in the
output represents a driver's feature state *entering* a specific race —
i.e., computed exclusively from historical data prior to that event.

Features (all computed as rolling aggregates of the previous N races)
--------
1. avg_qualifying_pos   — Mean qualifying position
2. teammate_delta       — Mean signed grid delta vs teammate (+ = ahead)
3. fastest_lap_rate     — Rate of races with the fastest lap (rank == 1)
4. position_delta       — Mean positions gained (grid - positionOrder)
5. podium_rate          — Rate of races finishing in Top 3
6. crash_rate           — Rate of races with driver-error DNF
7. points_per_finish    — Mean points in races where the car survived
8. experience           — Cumulative race count up to (but not including)
                          the current race
"""
import logging
import os

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


def _load_tables(team_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load results, qualifying, and races CSVs with proper types.

    Returns
    -------
    results_df, qualifying_df, races_df
    """
    results_df = pd.read_csv(os.path.join(team_dir, "results.csv"))
    qualifying_df = pd.read_csv(os.path.join(team_dir, "qualifying.csv"))
    races_df = pd.read_csv(os.path.join(team_dir, "races.csv"))

    int_cols_results = ["raceId", "driverId", "constructorId", "statusId"]
    for col in int_cols_results:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df = results_df.dropna(subset=int_cols_results)
    for col in int_cols_results:
        results_df[col] = results_df[col].astype(int)

    int_cols_qual = ["raceId", "driverId", "constructorId"]
    for col in int_cols_qual:
        qualifying_df[col] = pd.to_numeric(qualifying_df[col], errors="coerce")
    qualifying_df = qualifying_df.dropna(subset=int_cols_qual)
    for col in int_cols_qual:
        qualifying_df[col] = qualifying_df[col].astype(int)

    results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")
    qualifying_df["date"] = pd.to_datetime(qualifying_df["date"], errors="coerce")
    races_df["date"] = pd.to_datetime(races_df["date"], errors="coerce")

    logger.info(f"  results:    {len(results_df)} rows")
    logger.info(f"  qualifying: {len(qualifying_df)} rows")
    logger.info(f"  races:      {len(races_df)} rows")

    return results_df, qualifying_df, races_df


def _build_per_race_metrics(
    results_df: pd.DataFrame,
    qualifying_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a single DataFrame with one row per (raceId, driverId) containing
    the raw per-race metrics that will be aggregated via rolling window.

    Columns added:
        qual_pos        — qualifying position (from qualifying table)
        teammate_delta  — signed grid delta vs teammate for this race
        has_fastest_lap — 1 if driver had fastest lap (rank == 1), else 0
        pos_delta       — positions gained (grid - positionOrder)
        is_podium       — 1 if positionOrder <= 3, else 0
        is_driver_error — 1 if DNF was driver's fault, else 0
        points_if_finish — points if car finished (NaN otherwise)
        date            — race date (for chronological sorting)
    """
    qual_pos = qualifying_df.groupby(["raceId", "driverId"])["position"].first()
    qual_pos = pd.to_numeric(qual_pos, errors="coerce").rename("qual_pos")

    df = results_df[
        ["raceId", "driverId", "constructorId", "grid", "positionOrder",
         "points", "rank", "statusId", "date"]
    ].copy()

    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    df = df.merge(
        qual_pos.reset_index(),
        on=["raceId", "driverId"],
        how="left",
    )

    df = _add_teammate_delta(df)

    df["has_fastest_lap"] = (df["rank"] == 1).astype(int)

    df["pos_delta"] = df["grid"] - df["positionOrder"]

    df["is_podium"] = (df["positionOrder"] <= 3).astype(int)

    df["status_text"] = df["statusId"].map(cfg.STATUS_ID_MAP).fillna("Unknown")
    df["is_driver_error"] = df["status_text"].isin(cfg.DRIVER_ERROR_STATUSES).astype(int)

    df["points_if_finish"] = np.where(
        df["statusId"].isin(cfg.FINISHED_STATUS_IDS),
        df["points"],
        np.nan,
    )

    error_count = df["is_driver_error"].sum()
    logger.info(
        f"  Per-race metrics: {len(df)} rows, "
        f"{error_count} driver errors ({100 * error_count / len(df):.1f}%)"
    )
    return df


def _add_teammate_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``teammate_delta`` column: signed grid-position gap vs teammate.

    For each (raceId, constructorId) group with exactly 2 drivers:
    - Driver who started ahead gets positive delta
    - Driver who started behind gets negative delta

    Uses ``qual_pos`` (qualifying position) with ``grid`` as fallback.
    """
    df = df.copy()
    df["start_pos"] = df["qual_pos"].fillna(df["grid"])
    df["teammate_delta"] = np.nan

    for (_, _), group in df.groupby(["raceId", "constructorId"]):
        if len(group) != 2:
            continue

        idxs = group.index.tolist()
        pos_a = group.loc[idxs[0], "start_pos"]
        pos_b = group.loc[idxs[1], "start_pos"]

        if pd.isna(pos_a) or pd.isna(pos_b):
            continue

        gap = abs(pos_b - pos_a)

        if pos_a <= pos_b:
            df.loc[idxs[0], "teammate_delta"] = gap
            df.loc[idxs[1], "teammate_delta"] = -gap
        else:
            df.loc[idxs[0], "teammate_delta"] = -gap
            df.loc[idxs[1], "teammate_delta"] = gap

    return df


def _apply_rolling_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    For each driver (sorted chronologically), compute rolling-window
    aggregates of their past races, **shifted by 1** to exclude the
    current race and prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Per-race metrics (one row per raceId + driverId).
    window : int
        Number of past races for the rolling window.

    Returns
    -------
    pd.DataFrame
        Same rows, with 8 rolling feature columns added.
    """
    df = df.sort_values(["driverId", "date"]).copy()

    rolling_cols: dict[str, list] = {
        "avg_qualifying_pos": [],
        "teammate_delta": [],
        "fastest_lap_rate": [],
        "position_delta": [],
        "podium_rate": [],
        "crash_rate": [],
        "points_per_finish": [],
        "experience": [],
    }

    for driver_id, group in df.groupby("driverId", sort=False):
        g = group.sort_values("date")

        rolling_cols["avg_qualifying_pos"].extend(
            g["qual_pos"].shift(1).rolling(window, min_periods=1).mean().tolist()
        )
        rolling_cols["teammate_delta"].extend(
            g["teammate_delta"].shift(1).rolling(window, min_periods=1).mean().tolist()
        )
        rolling_cols["fastest_lap_rate"].extend(
            g["has_fastest_lap"].shift(1).rolling(window, min_periods=1).mean().tolist()
        )
        rolling_cols["position_delta"].extend(
            g["pos_delta"].shift(1).rolling(window, min_periods=1).mean().tolist()
        )
        rolling_cols["podium_rate"].extend(
            g["is_podium"].shift(1).rolling(window, min_periods=1).mean().tolist()
        )
        rolling_cols["crash_rate"].extend(
            g["is_driver_error"].shift(1).rolling(window, min_periods=1).mean().tolist()
        )

        ppf_shifted = g["points_if_finish"].shift(1)
        ppf_rolling = ppf_shifted.rolling(window, min_periods=1).mean()
        rolling_cols["points_per_finish"].extend(ppf_rolling.tolist())

        experience = pd.Series(range(len(g)), index=g.index, dtype=float)
        rolling_cols["experience"].extend(experience.tolist())

    for col_name, values in rolling_cols.items():
        df[col_name] = values

    before_drop = len(df)
    df = df.dropna(subset=["avg_qualifying_pos"])
    after_drop = len(df)
    logger.info(
        f"  Rolling features applied (window={window}). "
        f"Dropped {before_drop - after_drop} first-race rows, "
        f"{after_drop} instances remain."
    )

    return df


def extract_driver_features(output_dir: str) -> pd.DataFrame:
    """
    Run the per-race rolling-window driver feature extraction pipeline.

    Produces one row per (raceId, driverId) with 8 rolling features
    computed from the driver's previous N races (no data leakage).

    Parameters
    ----------
    output_dir : str
        Root output directory (e.g., ``./output``).

    Returns
    -------
    pd.DataFrame
        Per-race driver features.
    """
    team_dir = os.path.join(output_dir, "team")
    driver_dir = os.path.join(output_dir, "driver")
    os.makedirs(driver_dir, exist_ok=True)

    logger.info("Phase 1: Loading and type-casting tables...")
    results_df, qualifying_df, races_df = _load_tables(team_dir)

    logger.info("Phase 2: Computing per-race raw metrics...")
    per_race_df = _build_per_race_metrics(results_df, qualifying_df)

    logger.info(f"Phase 3: Applying rolling window (W={cfg.ROLLING_WINDOW}) with shift(1)...")
    features_df = _apply_rolling_features(per_race_df, cfg.ROLLING_WINDOW)

    output_cols = [
        "raceId", "driverId", "constructorId", "date",
        "avg_qualifying_pos", "teammate_delta", "fastest_lap_rate",
        "position_delta", "podium_rate", "crash_rate",
        "points_per_finish", "experience",
        "positionOrder", "statusId",
    ]
    output_df = features_df[output_cols].copy()

    output_df["teammate_delta"] = output_df["teammate_delta"].fillna(0.0)
    output_df["points_per_finish"] = output_df["points_per_finish"].fillna(0.0)

    csv_path = os.path.join(driver_dir, "driver_features_per_race.csv")
    output_df.to_csv(csv_path, index=False)
    logger.info(f"Driver features exported -> {csv_path}")
    logger.info(
        f"  Shape: {output_df.shape[0]} instances x {output_df.shape[1]} columns"
    )

    return output_df
