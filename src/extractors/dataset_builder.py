"""
Unified dataset builder for the multimodal F1 prediction model.

Joins the three orthogonal feature vectors (Driver, Track, Team) into a
single tabular dataset where each row is a race instance:

    Instance = (raceId, driverId) -> [Driver Vec | Track Vec | Team key] -> top3

The ``top3`` binary target indicates whether the driver finished on the
podium (positionOrder <= 3) in that specific race.

This module consumes the outputs of the driver and track extractors and
produces the final ``instances.csv`` that will feed the PyTorch dataloader.
"""
import logging
import os

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


def build_dataset(data_dir: str, dataset_dir: str) -> pd.DataFrame:
    """
    Build the unified per-race instance dataset.

    Joins:
    - Driver rolling features (``driver/driver_features_per_race.csv``)
    - Track features (``track/track_features_*.csv``)
    - Races table (``team/races.csv``) for raceId <-> circuitId join

    Parameters
    ----------
    data_dir : str
        Directory containing the generated data (driver, track, team).
    dataset_dir : str
        Directory to save the final dataset.

    Returns
    -------
    pd.DataFrame
        The unified dataset with one row per (raceId, driverId).
    """
    driver_path = os.path.join(data_dir, "driver", "driver_features_per_race.csv")
    if not os.path.exists(driver_path):
        raise FileNotFoundError(
            f"Driver features not found at {driver_path}. "
            f"Run the 'driver' module first."
        )
    driver_df = pd.read_csv(driver_path)
    driver_df["date"] = pd.to_datetime(driver_df["date"], errors="coerce")
    logger.info(f"  Driver features loaded: {len(driver_df)} instances")

    races_path = os.path.join(data_dir, "team", "races.csv")
    races_df = pd.read_csv(races_path)
    races_df["raceId"] = pd.to_numeric(races_df["raceId"], errors="coerce").astype(int)

    race_info = races_df[["raceId", "year", "round", "circuitId"]].copy()
    race_info["circuitId"] = pd.to_numeric(race_info["circuitId"], errors="coerce").astype(int)

    logger.info(f"  Races table loaded: {len(races_df)} races")

    track_dir = os.path.join(data_dir, "track")
    track_dfs = []

    if os.path.exists(track_dir):
        for fname in sorted(os.listdir(track_dir)):
            if fname.startswith("track_features_") and fname.endswith(".csv"):
                tf = pd.read_csv(os.path.join(track_dir, fname))
                track_dfs.append(tf)
                logger.info(f"  Track features loaded: {fname} ({len(tf)} circuits)")

    if track_dfs:
        track_df = pd.concat(track_dfs, ignore_index=True)
    else:
        logger.warning("  No track feature files found. Track columns will be empty.")
        track_df = pd.DataFrame()

    track_feature_cols = [
        "altitude_m", "length_m", "corners_count", "rotation",
        "avg_track_temp", "avg_air_temp", "avg_humidity",
    ]

    df = driver_df.merge(race_info, on="raceId", how="left")

    logger.info(f"  After race info join: {len(df)} instances")

    if not track_df.empty and "circuit_id" in track_df.columns:
        track_join = track_df[["year", "circuit_id"] + track_feature_cols].copy()
        track_join = track_join.rename(columns={"circuit_id": "circuitId"})
        track_join["circuitId"] = pd.to_numeric(track_join["circuitId"], errors="coerce")
        track_join = track_join.dropna(subset=["circuitId"])
        track_join["circuitId"] = track_join["circuitId"].astype(int)

        track_join = track_join.drop_duplicates(subset=["year", "circuitId"])

        df = df.merge(
            track_join,
            on=["year", "circuitId"],
            how="left",
        )
        logger.info(f"  After track join: {len(df)} instances")
    else:
        for col in track_feature_cols:
            df[col] = float("nan")
        logger.warning("  Track features could not be joined.")

    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["top3"] = (df["positionOrder"] <= 3).astype(int)

    target_positive = df["top3"].sum()
    logger.info(
        f"  Target: {target_positive} podiums / {len(df)} instances "
        f"({100 * target_positive / len(df):.1f}% positive rate)"
    )

    df["split"] = "unused"
    df.loc[df["year"].isin(cfg.TRAIN_YEARS), "split"] = "train"
    df.loc[df["year"].isin(cfg.VAL_YEARS), "split"] = "val"
    df.loc[df["year"].isin(cfg.TEST_YEARS), "split"] = "test"

    split_counts = df["split"].value_counts()
    logger.info(f"  Split distribution: {split_counts.to_dict()}")

    driver_feature_cols = [
        "avg_qualifying_pos", "teammate_delta", "fastest_lap_rate",
        "position_delta", "podium_rate", "crash_rate",
        "points_per_finish", "experience",
    ]

    # Missingness flags BEFORE imputation. 1 = original value was missing.
    for col in driver_feature_cols + track_feature_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # Driver features residualized against (constructorId, year) baseline.
    # Subtracts the team-year average from each driver feature so that the
    # remaining signal is, by construction, less contaminated by team strength.
    # Baseline computed only on the train split to avoid temporal leakage.
    train_mask_for_baseline = df["split"] == "train"
    residual_cols: list[str] = []
    for col in driver_feature_cols:
        if col not in df.columns:
            continue
        baseline = (
            df.loc[train_mask_for_baseline]
            .groupby(["constructorId", "year"])[col]
            .mean()
            .rename("baseline")
            .reset_index()
        )
        df = df.merge(baseline, on=["constructorId", "year"], how="left")
        global_mean = df.loc[train_mask_for_baseline, col].mean()
        df["baseline"] = df["baseline"].fillna(global_mean)
        df[f"{col}_resid"] = df[col] - df["baseline"]
        df = df.drop(columns=["baseline"])
        residual_cols.append(f"{col}_resid")

    # Train-only medians used for imputation. Avoids leakage from val/test.
    train_mask = df["split"] == "train"
    medians = df.loc[train_mask, driver_feature_cols + track_feature_cols].median(numeric_only=True)
    for col in driver_feature_cols + track_feature_cols:
        if col in df.columns:
            fill_val = medians.get(col, 0.0)
            if pd.isna(fill_val):
                fill_val = 0.0
            df[col] = df[col].fillna(fill_val)

    missing_cols = [f"{c}_missing" for c in driver_feature_cols + track_feature_cols if c in df.columns]

    key_cols = ["raceId", "driverId", "constructorId", "year", "round", "date"]
    target_cols = ["top3"]
    split_cols = ["split"]

    final_cols = (
        key_cols + target_cols + driver_feature_cols + residual_cols
        + track_feature_cols + missing_cols + split_cols
    )

    final_cols = [c for c in final_cols if c in df.columns]
    output_df = df[final_cols].copy()

    csv_path = os.path.join(dataset_dir, "instances.csv")
    output_df.to_csv(csv_path, index=False)

    logger.info(f"  Dataset exported -> {csv_path}")
    logger.info(f"  Final shape: {output_df.shape[0]} rows x {output_df.shape[1]} columns")

    for col in driver_feature_cols + track_feature_cols:
        if col in output_df.columns:
            non_null = output_df[col].notna().sum()
            logger.info(f"    {col:25s} non-null: {non_null}/{len(output_df)}")

    return output_df
