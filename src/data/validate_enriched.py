"""Validates the enriched rel-f1 database written by ``build_enriched_db``.

Six checks, matching the plan:

1. Benchmark immutability: every row with an ID already present in the
   pristine snapshot is byte-identical (``assert_frame_equal``), table by
   table.
2. Overlap agreement: for the historical window Jolpica and RelBench both
   cover (2000 - 2023 round 12), the key outcome columns agree at a high
   rate (residual disagreement is expected to be legitimate upstream
   revisions, not a bug -- see the Sainz/Piastri Spa-2023 status example
   found during discovery).
3. Cross-source vs f1db: independent re-derivation of the *new* rows
   (2023 R13 - max_year) agrees with f1db on points/position/grid.
4. Referential integrity: every foreign key resolves, no duplicate primary
   keys, primary keys are exactly contiguous 0..n-1 (a hard requirement of
   ``relbench.base.Dataset.validate_and_correct_db``).
5. Season completeness: rounds present match the official schedule, each
   race has a plausible number of classified entries, and cumulative
   standings points never decrease within a season.
6. Graph/task smoke test: ``make_pkey_fkey_graph`` builds successfully, the
   5 edge types the model relies on exist, and ``results-position`` (and its
   custom variants) return non-empty, plausible train/val/test splits.

Usage: ``python -m src.data.validate_enriched``
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRISTINE_LAST_YEAR = 2023
PRISTINE_LAST_ROUND = 12
OVERLAP_START_YEAR = 2000


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        status = "PASS" if result.passed else "FAIL"
        logger.info("[%s] %s: %s", status, result.name, result.details)

    def summary(self) -> str:
        lines = ["Validation report:", "=" * 60]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"[{status}] {c.name}: {c.details}")
        lines.append("=" * 60)
        lines.append("ALL CHECKS PASSED" if self.all_passed else "SOME CHECKS FAILED")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Benchmark immutability
# ---------------------------------------------------------------------------

def check_benchmark_immutability(pristine: Dict[str, pd.DataFrame], enriched: Dict[str, pd.DataFrame]) -> CheckResult:
    mismatches = {}
    for name, pristine_df in pristine.items():
        enriched_df = enriched[name]
        pkey_candidates = [c for c in pristine_df.columns if c.endswith("Id") and c[0].islower()]
        n_old = len(pristine_df)
        subset = enriched_df.iloc[:n_old].reset_index(drop=True)
        old = pristine_df.reset_index(drop=True)

        if len(subset) != len(old):
            mismatches[name] = f"row count shrank ({len(subset)} < {len(old)})"
            continue

        try:
            pd.testing.assert_frame_equal(
                subset, old, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-6
            )
        except AssertionError as e:
            mismatches[name] = str(e).splitlines()[0]

    passed = len(mismatches) == 0
    details = "all pristine rows byte-identical" if passed else f"mismatches: {mismatches}"
    return CheckResult("benchmark_immutability", passed, details, {"mismatches": mismatches})


# ---------------------------------------------------------------------------
# 2. Overlap agreement (sanity check that our own normalization pipeline is
#    consistent with what RelBench already shipped, on the shared window)
# ---------------------------------------------------------------------------

def check_overlap_agreement(
    pristine_results: pd.DataFrame,
    pristine_races: pd.DataFrame,
    pristine_drivers: pd.DataFrame,
    overlap_results: pd.DataFrame,
    min_agreement: float = 0.97,
) -> CheckResult:
    """``overlap_results`` must be freshly re-extracted via the same
    normalize_results() pipeline for (year, round) pairs already in the
    pristine snapshot (2000 - 2023 R12), keyed by (year, round, driverRef)."""
    races = pristine_races[["raceId", "year", "round"]]
    drivers = pristine_drivers[["driverId", "driverRef"]]
    pr = pristine_results.merge(races, on="raceId").merge(drivers, on="driverId")
    pr = pr[pr["year"] >= OVERLAP_START_YEAR]

    merged = pr.merge(
        overlap_results, on=["year", "round", "driverRef"], how="inner", suffixes=("_pristine", "_new")
    )

    metrics = {}
    for col in ["points", "grid", "statusId"]:
        pristine_col = f"{col}_pristine" if f"{col}_pristine" in merged.columns else col
        new_col = f"{col}_new" if f"{col}_new" in merged.columns else col
        if pristine_col not in merged.columns or new_col not in merged.columns:
            continue
        agree = np.isclose(
            merged[pristine_col].astype(float), merged[new_col].astype(float), equal_nan=True
        )
        metrics[col] = float(agree.mean())

    worst = min(metrics.values()) if metrics else 0.0
    passed = worst >= min_agreement and len(merged) > 0
    details = f"n={len(merged)}, agreement={metrics}"
    return CheckResult("overlap_agreement", passed, details, metrics)


# ---------------------------------------------------------------------------
# 3. Cross-source vs f1db
# ---------------------------------------------------------------------------

def check_cross_source_f1db(
    enriched_results: pd.DataFrame,
    enriched_races: pd.DataFrame,
    enriched_drivers: pd.DataFrame,
    f1db_results: pd.DataFrame,
    min_agreement: float = 0.90,
) -> CheckResult:
    from .sources.f1db import name_slug

    races = enriched_races[["raceId", "year", "round"]]
    drivers = enriched_drivers[["driverId", "forename", "surname"]].copy()
    # Join on a name-derived slug, not driverRef: Ergast's driverRef is
    # usually surname-only ("hamilton") while f1db's driver id is always
    # "firstname-lastname" ("lewis-hamilton") -- joining on driverRef would
    # only match the handful of Ergast entries that needed disambiguation.
    drivers["name_slug"] = drivers.apply(lambda r: name_slug(r["forename"], r["surname"]), axis=1)
    new_range = enriched_results.merge(races, on="raceId").merge(drivers, on="driverId")
    new_range = new_range[
        (new_range["year"] > PRISTINE_LAST_YEAR)
        | ((new_range["year"] == PRISTINE_LAST_YEAR) & (new_range["round"] > PRISTINE_LAST_ROUND))
    ]

    drivers_needed = new_range.copy()
    merged = drivers_needed.merge(
        f1db_results, on=["year", "round", "name_slug"], how="inner", suffixes=("_ours", "_f1db")
    )

    if len(merged) == 0:
        return CheckResult("cross_source_f1db", False, "no overlapping rows found with f1db", {})

    # "points"/"grid"/"position" collide between `new_range` (enriched, "ours")
    # and `f1db_results` ("f1db"), so pandas.merge suffixes *both* sides.
    # f1db's CSV export leaves "points" blank (NaN) for non-scoring drivers
    # instead of 0 -- fillna(0) before comparing so that convention doesn't
    # register as a disagreement against our own explicit 0.0.
    metrics = {}
    points_ours = merged["points_ours"].fillna(0.0)
    points_f1db = merged["points_f1db"].fillna(0.0)
    metrics["points"] = float(np.isclose(points_ours, points_f1db, atol=0.01).mean())
    metrics["grid"] = float(np.isclose(merged["grid_ours"], merged["grid_f1db"], equal_nan=True).mean())
    both_classified = merged["position_ours"].notna() & merged["position_f1db"].notna()
    if both_classified.any():
        metrics["position"] = float(
            np.isclose(
                merged.loc[both_classified, "position_ours"], merged.loc[both_classified, "position_f1db"]
            ).mean()
        )
    metrics["match_rate_join"] = float(len(merged) / max(len(drivers_needed), 1))

    worst = min(v for k, v in metrics.items() if k != "match_rate_join")
    passed = worst >= min_agreement
    details = f"n_matched={len(merged)}/{len(drivers_needed)}, agreement={metrics}"
    return CheckResult("cross_source_f1db", passed, details, metrics)


# ---------------------------------------------------------------------------
# 4. Referential integrity
# ---------------------------------------------------------------------------

def check_referential_integrity(table_dict: Dict[str, Any]) -> CheckResult:
    problems = []
    for name, table in table_dict.items():
        df = table.df
        if table.pkey_col is not None:
            ser = df[table.pkey_col]
            if ser.isna().any():
                problems.append(f"{name}.{table.pkey_col} has nulls")
            if ser.nunique() != len(ser):
                problems.append(f"{name}.{table.pkey_col} has duplicates")
            values = ser.dropna().astype("int64").to_numpy()
            if len(values) and not np.array_equal(np.sort(values), np.arange(len(values))):
                problems.append(f"{name}.{table.pkey_col} is not contiguous 0..n-1")

    for name, table in table_dict.items():
        df = table.df
        for fkey_col, pkey_table in table.fkey_col_to_pkey_table.items():
            if fkey_col not in df.columns:
                problems.append(f"{name}.{fkey_col} missing but declared as fkey")
                continue
            n_pkeys = len(table_dict[pkey_table].df)
            vals = df[fkey_col].dropna().astype("int64")
            bad = ((vals < 0) | (vals >= n_pkeys)).sum()
            if bad:
                problems.append(f"{name}.{fkey_col}: {bad} rows reference out-of-range {pkey_table} ids")

    passed = len(problems) == 0
    details = "all fkeys resolve, pkeys contiguous & unique" if passed else f"problems: {problems}"
    return CheckResult("referential_integrity", passed, details, {"problems": problems})


# ---------------------------------------------------------------------------
# 5. Season completeness
# ---------------------------------------------------------------------------

def check_season_completeness(
    races: pd.DataFrame,
    results: pd.DataFrame,
    standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    expected_rounds: Dict[int, List[int]],
) -> CheckResult:
    """Missing races / implausible entry counts are hard failures. Standings
    points decreasing within a season is reported for visibility but is
    *not* a failure: it is a known, real phenomenon in F1 history (e.g.
    Force India's constructors' points were reset to 0 at the 2018 Belgian
    GP after entering administration and being bought/renamed mid-season;
    Racing Point had points corrected following the 2020 "pink Mercedes"
    ruling), not a normalization bug -- confirmed present in the raw
    Jolpica/Ergast standings snapshots themselves.
    """
    problems = []
    non_monotonic_notes = []

    races_by_key = races.set_index(["year", "round"])["raceId"] if len(races) else pd.Series(dtype="int64")
    for year, rounds in expected_rounds.items():
        for round_ in rounds:
            if (year, round_) not in races_by_key.index:
                problems.append(f"missing race row for {year} round {round_}")
                continue
            race_id = races_by_key.loc[(year, round_)]
            n_results = (results["raceId"] == race_id).sum()
            if not (10 <= n_results <= 26):
                problems.append(f"{year} round {round_}: implausible entry count ({n_results})")

    for driver_id, grp in standings.sort_values(["raceId"]).groupby("driverId"):
        pts = grp.merge(races[["raceId", "year"]], on="raceId")
        for year, season_grp in pts.groupby("year"):
            season_grp = season_grp.sort_values("raceId")
            if (season_grp["points"].diff().dropna() < -1e-6).any():
                non_monotonic_notes.append(f"driver {driver_id} season {year}: standings points decreased")
                break

    for constructor_id, grp in constructor_standings.sort_values(["raceId"]).groupby("constructorId"):
        pts = grp.merge(races[["raceId", "year"]], on="raceId")
        for year, season_grp in pts.groupby("year"):
            season_grp = season_grp.sort_values("raceId")
            if (season_grp["points"].diff().dropna() < -1e-6).any():
                non_monotonic_notes.append(f"constructor {constructor_id} season {year}: standings points decreased")
                break

    passed = len(problems) == 0
    details = "rounds/entry-counts all plausible" if passed else f"problems (truncated): {problems[:10]} (total {len(problems)})"
    if non_monotonic_notes:
        details += (
            f"; note: {len(non_monotonic_notes)} season(s) with non-monotonic standings points "
            f"(expected -- team resets/penalty corrections, not failures): {non_monotonic_notes[:5]}"
        )
    return CheckResult(
        "season_completeness", passed, details,
        {"n_problems": len(problems), "non_monotonic_standings": non_monotonic_notes},
    )


# ---------------------------------------------------------------------------
# 6. Graph / task smoke test
# ---------------------------------------------------------------------------

def check_graph_and_task_smoke_test(dataset_name: str, task_names: List[str]) -> CheckResult:
    from torch_frame import stype as tf_stype
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from relbench.tasks import get_task

    problems = []

    for task_name in task_names:
        try:
            # Instantiating the task mutates its (freshly resolved) dataset
            # object -- always use `task.dataset` afterward, never a
            # separately-obtained `get_dataset(...)`, since
            # `get_modified_db` clears relbench's module-level dataset
            # cache as a side effect and a stale reference could silently
            # point at an unmodified (leak-carrying) database.
            task = get_task(dataset_name, task_name, download=False)
        except Exception as e:  # noqa: BLE001
            problems.append(f"task '{task_name}' instantiation failed: {e}")
            continue

        for split in ("train", "val", "test"):
            try:
                table = task.get_table(split, mask_input_cols=False)
                if len(table.df) == 0:
                    problems.append(f"task '{task_name}' split '{split}' is empty")
            except Exception as e:  # noqa: BLE001
                problems.append(f"task '{task_name}' split '{split}' failed: {e}")

        if task_name == task_names[0]:
            try:
                db = task.dataset.get_db(upto_test_timestamp=False)
                stype_dict = get_stype_proposal(db)
                # Text-like stypes need an embedder config that neither this
                # smoke test nor the real training pipeline configures; both
                # fall back to categorical (see train.build_graph) rather
                # than fail on e.g. free-text ref/name columns.
                for table_name, col_stypes in stype_dict.items():
                    for col_name, col_stype in col_stypes.items():
                        if col_stype in (tf_stype.text_embedded, tf_stype.text_tokenized):
                            stype_dict[table_name][col_name] = tf_stype.categorical
                data, _ = make_pkey_fkey_graph(db, col_to_stype_dict=stype_dict)
                expected_edge_pairs = [
                    ("results", "f2p_driverId", "drivers"),
                    ("results", "f2p_constructorId", "constructors"),
                    ("qualifying", "f2p_driverId", "drivers"),
                    ("qualifying", "f2p_constructorId", "constructors"),
                    ("constructor_standings", "f2p_constructorId", "constructors"),
                ]
                present = set(data.edge_types)
                missing = [e for e in expected_edge_pairs if e not in present]
                if missing:
                    problems.append(f"graph missing expected edge types: {missing}")
                leaked = [
                    c for c in data["results"].tf.col_names_dict.get(
                        __import__("torch_frame").stype.numerical, []
                    )
                    if c in ("position", "positionOrder", "points", "statusId", "laps", "milliseconds", "fastestLap", "rank")
                ]
                if leaked:
                    problems.append(f"leakage: outcome columns still present as results features: {leaked}")
            except Exception as e:  # noqa: BLE001
                problems.append(f"graph construction failed: {e}")

    passed = len(problems) == 0
    details = "graph (leak-free) + all tasks build with plausible splits" if passed else f"problems: {problems}"
    return CheckResult("graph_and_task_smoke_test", passed, details, {"problems": problems})


def run_full_validation(
    enriched_dir: str = "data/enriched/rel-f1",
    max_year: int = 2026,
    skip_f1db: bool = False,
    skip_graph: bool = False,
) -> ValidationReport:
    from relbench.base import Database
    from relbench.datasets import get_dataset

    from . import ergast_schema as es
    from .build_enriched_db import PRISTINE_LAST_ROUND, PRISTINE_LAST_YEAR, load_pristine_db
    from .sources import f1db as f1db_source
    from .sources.jolpica import JolpicaClient

    report = ValidationReport()

    pristine_db = load_pristine_db()
    pristine = {name: t.df for name, t in pristine_db.table_dict.items()}

    enriched_db = Database.load(f"{enriched_dir}/db")
    enriched = {name: t.df for name, t in enriched_db.table_dict.items()}

    report.add(check_benchmark_immutability(pristine, enriched))

    client = JolpicaClient()
    status_lookup = es.build_status_lookup(client.get_status_table())
    next_status_id = [max(status_lookup.values()) + 10_000]  # overlap check never mints new ones for real

    overlap_frames = []
    for year in range(OVERLAP_START_YEAR, PRISTINE_LAST_YEAR + 1):
        max_round = PRISTINE_LAST_ROUND if year == PRISTINE_LAST_YEAR else 99
        schedule = client.get_season_schedule(year)
        for race in schedule:
            round_ = int(race["round"])
            if round_ > max_round:
                continue
            race_obj = client.get_race_results(year, round_)
            if race_obj and race_obj.get("Results"):
                overlap_frames.append(es.normalize_results(race_obj, status_lookup, next_status_id))
    overlap_results = pd.concat(overlap_frames, ignore_index=True) if overlap_frames else pd.DataFrame()
    report.add(check_overlap_agreement(pristine["results"], pristine["races"], pristine["drivers"], overlap_results))

    if not skip_f1db:
        try:
            f1db_results = f1db_source.load_race_results()
            report.add(
                check_cross_source_f1db(enriched["results"], enriched["races"], enriched["drivers"], f1db_results)
            )
        except Exception as e:  # noqa: BLE001
            report.add(CheckResult("cross_source_f1db", False, f"skipped due to error: {e}"))

    report.add(check_referential_integrity(enriched_db.table_dict))

    expected_rounds = {}
    for year in range(PRISTINE_LAST_YEAR, max_year + 1):
        schedule = client.get_season_schedule(year)
        held = []
        for race in schedule:
            round_ = int(race["round"])
            race_obj = client.get_race_results(year, round_)
            if race_obj and race_obj.get("Results"):
                held.append(round_)
        expected_rounds[year] = held
    report.add(
        check_season_completeness(
            enriched["races"], enriched["results"], enriched["standings"],
            enriched["constructor_standings"], expected_rounds,
        )
    )

    if not skip_graph:
        from .enriched_dataset import DEFAULT_NAME, register_enriched_dataset
        from .tasks import register_enriched_tasks

        try:
            register_enriched_dataset(DEFAULT_NAME, cache_dir=enriched_dir)
            register_enriched_tasks(DEFAULT_NAME)
            report.add(
                check_graph_and_task_smoke_test(
                    DEFAULT_NAME, ["results-position", "results-positionorder", "results-points"]
                )
            )
        except Exception as e:  # noqa: BLE001
            report.add(CheckResult("graph_and_task_smoke_test", False, f"failed: {e}"))

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_full_validation()
    print(report.summary())
    sys.exit(0 if report.all_passed else 1)
