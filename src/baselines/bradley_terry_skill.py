"""Walk-forward Bradley-Terry skill scorer using skill_dataset pairs."""

from __future__ import annotations

import pandas as pd
import torch

import config as cfg
from baselines.bradley_terry import BradleyTerry
from data.enriched_dataset import EnrichedF1Dataset
from data.mobility import build_race_pairs_for_bt, compute_support_scores
from data.skill_dataset import SkillDatasetConfig, build_skill_dataset


def _constructor_season_index(df: pd.DataFrame) -> dict[tuple[int, int], int]:
    pairs = df[["constructorId", "year"]].drop_duplicates().sort_values(["constructorId", "year"])
    return {
        (int(r.constructorId), int(r.year)): i
        for i, r in enumerate(pairs.itertuples(index=False))
    }


def _driver_index(df: pd.DataFrame) -> dict[int, int]:
    ids = sorted(df["driverId"].astype(int).unique())
    return {int(d): i for i, d in enumerate(ids)}


def load_bradley_terry_skill(
    device=None,
    lr: float = 0.1,
    epochs_per_step: int = 30,
    max_year: int = 2025,
) -> pd.DataFrame:
    """Walk-forward BT skill: skill(driver, T) uses only races with year <= T."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    db = EnrichedF1Dataset().get_db(upto_test_timestamp=False)
    skill_df = build_skill_dataset(db, SkillDatasetConfig(max_year=max_year))
    pairs = build_race_pairs_for_bt(skill_df)

    drv_idx = _driver_index(skill_df)
    cs_idx = _constructor_season_index(skill_df)
    num_drivers = len(drv_idx)
    num_cs = len(cs_idx)

    pairs["cs_A"] = pairs.apply(
        lambda r: cs_idx[(int(r["constructorA"]), int(r["year"]))], axis=1
    )
    pairs["cs_B"] = pairs.apply(
        lambda r: cs_idx[(int(r["constructorB"]), int(r["year"]))], axis=1
    )
    pairs["idx_A"] = pairs["driverA"].map(drv_idx)
    pairs["idx_B"] = pairs["driverB"].map(drv_idx)

    driver_A = torch.tensor(pairs["idx_A"].to_numpy(), dtype=torch.long, device=device)
    driver_B = torch.tensor(pairs["idx_B"].to_numpy(), dtype=torch.long, device=device)
    cs_A = torch.tensor(pairs["cs_A"].to_numpy(), dtype=torch.long, device=device)
    cs_B = torch.tensor(pairs["cs_B"].to_numpy(), dtype=torch.long, device=device)
    labels = torch.ones(len(pairs), device=device)
    years = pairs["year"].to_numpy()

    active_by_year: dict[int, set[int]] = {}
    for year, grp in skill_df.groupby("year"):
        active_by_year[int(year)] = set(grp["driverId"].astype(int))

    model = BradleyTerry(num_drivers, num_cs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rows = []
    for T in sorted(active_by_year):
        mask = years <= T
        for _ in range(epochs_per_step):
            logits = model.logits(driver_A[mask], driver_B[mask], cs_A[mask], cs_B[mask])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        theta = model.driver_skill().detach().cpu().numpy()
        for driver_id in sorted(active_by_year[T]):
            idx = drv_idx[driver_id]
            rows.append(
                {"driverId": driver_id, "season": T, "skill_score": float(theta[idx])}
            )

    skill = pd.DataFrame(rows).sort_values(["driverId", "season"]).reset_index(drop=True)
    support = compute_support_scores(skill_df)
    merged = skill.merge(
        support[["driverId", "season", "support_score", "support_bucket"]],
        on=["driverId", "season"],
        how="left",
    )
    return merged
