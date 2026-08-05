"""Builds the enriched rel-f1 database.

Pristine RelBench rows are copied byte-for-byte (never re-derived, never
re-sorted, never re-indexed) and newly extracted rows for 2023 round 13
through the most recently completed round of ``max_year`` are appended with
freshly minted, strictly-increasing integer keys.

This deliberately bypasses ``relbench.base.Database.reindex_pkeys_and_fkeys``
(which the base ``Dataset.get_db()`` would otherwise call on every from-scratch
build): that function re-sorts each table by its time column with pandas'
default (non-stable) quicksort, which could silently reassign different
integer IDs to *existing* rows whenever several rows share an exact
timestamp (true for every race: all of a race's results/qualifying/standings
rows share the same date). Since old rows are already correctly time-sorted
sorted by construction (that's how the pristine snapshot was built in the
first place) and every new row is chronologically after all of them, a plain
append preserves the "sorted by time_col, pkey == row position" invariant
that RelBench's ``Table.upto()``/``validate_and_correct_db`` rely on, without
ever touching an existing ID.

The result is written with ``relbench.base.Database.save()`` directly into
``<output_dir>/db/``, i.e. the exact on-disk format ``Dataset.get_db()``
expects to find in its cache directory -- so ``EnrichedF1Dataset`` never
needs to call ``make_db()`` (and therefore never triggers a reindex) as long
as this script has run at least once.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from relbench.base import Database, Table
from relbench.datasets import get_dataset

from . import ergast_schema as es
from .sources.jolpica import JolpicaClient

logger = logging.getLogger(__name__)

# Last (year, round) already fully populated (results/qualifying/standings)
# in the frozen rel-f1 snapshot. Confirmed empirically: races exist through
# 2023 round 22, but all child tables stop after round 12 (2023-07-30).
PRISTINE_LAST_YEAR = 2023
PRISTINE_LAST_ROUND = 12

# (pkey_col, fkey_col_to_pkey_table, time_col) -- mirrors relbench.datasets.f1.F1Dataset
TABLE_SCHEMAS: Dict[str, Tuple[str, Dict[str, str], Optional[str]]] = {
    "circuits": ("circuitId", {}, None),
    "drivers": ("driverId", {}, None),
    "constructors": ("constructorId", {}, None),
    "races": ("raceId", {"circuitId": "circuits"}, "date"),
    "results": (
        "resultId",
        {"raceId": "races", "driverId": "drivers", "constructorId": "constructors"},
        "date",
    ),
    "standings": ("driverStandingsId", {"raceId": "races", "driverId": "drivers"}, "date"),
    "constructor_results": (
        "constructorResultsId",
        {"raceId": "races", "constructorId": "constructors"},
        "date",
    ),
    "constructor_standings": (
        "constructorStandingsId",
        {"raceId": "races", "constructorId": "constructors"},
        "date",
    ),
    "qualifying": (
        "qualifyId",
        {"raceId": "races", "driverId": "drivers", "constructorId": "constructors"},
        "date",
    ),
}

# Exact pristine dtypes, so concatenation never silently upcasts/downcasts a
# column relative to the frozen snapshot.
COLUMN_DTYPES: Dict[str, Dict[str, str]] = {
    "circuits": {
        "circuitId": "Int64", "circuitRef": "object", "name": "object",
        "location": "object", "country": "object", "lat": "float64",
        "lng": "float64", "alt": "float64",
    },
    "drivers": {
        "driverId": "Int64", "driverRef": "object", "code": "object",
        "forename": "object", "surname": "object", "dob": "datetime64[ns]",
        "nationality": "object",
    },
    "constructors": {
        "constructorId": "Int64", "constructorRef": "object",
        "name": "object", "nationality": "object",
    },
    "races": {
        "raceId": "Int64", "year": "int64", "round": "int64",
        "circuitId": "Int64", "name": "object", "date": "datetime64[ns]",
        "time": "object",
    },
    "results": {
        "resultId": "Int64", "raceId": "Int64", "driverId": "Int64",
        "constructorId": "Int64", "number": "float64", "grid": "int64",
        "position": "float64", "positionOrder": "int64", "points": "float64",
        "laps": "int64", "milliseconds": "float64", "fastestLap": "float64",
        "rank": "float64", "statusId": "int64", "date": "datetime64[ns]",
    },
    "standings": {
        "driverStandingsId": "Int64", "raceId": "Int64", "driverId": "Int64",
        "points": "float64", "position": "int64", "wins": "int64",
        "date": "datetime64[ns]",
    },
    "constructor_results": {
        "constructorResultsId": "Int64", "raceId": "Int64",
        "constructorId": "Int64", "points": "float64", "date": "datetime64[ns]",
    },
    "constructor_standings": {
        "constructorStandingsId": "Int64", "raceId": "Int64",
        "constructorId": "Int64", "points": "float64", "position": "int64",
        "wins": "int64", "date": "datetime64[ns]",
    },
    "qualifying": {
        "qualifyId": "Int64", "raceId": "Int64", "driverId": "Int64",
        "constructorId": "Int64", "number": "int64", "position": "int64",
        "date": "datetime64[ns]",
    },
}


def load_pristine_db() -> Database:
    """Read-only access to the frozen rel-f1 snapshot. Never mutated."""
    dataset = get_dataset("rel-f1", download=False)
    return dataset.get_db(upto_test_timestamp=False)


class IdAllocator:
    """Reuses existing integer IDs for known ref strings; mints new ones
    (max_existing + 1, +2, ...) for unseen ones in first-seen order."""

    def __init__(self, existing_map: Dict[str, int]):
        self.map: Dict[str, int] = dict(existing_map)
        self._next = (max(self.map.values()) + 1) if self.map else 0
        self.new_refs: List[str] = []

    def resolve(self, ref: str) -> int:
        if ref not in self.map:
            self.map[ref] = self._next
            self._next += 1
            self.new_refs.append(ref)
        return self.map[ref]


def _ref_to_id_map(df: pd.DataFrame, id_col: str, ref_col: str) -> Dict[str, int]:
    return {ref: int(i) for ref, i in zip(df[ref_col], df[id_col])}


def _determine_new_rounds(
    client: JolpicaClient,
    pristine_races: pd.DataFrame,
    max_year: int,
    refresh_last_n_rounds: int,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, Any]]]:
    """Returns the sorted list of (year, round) pairs that still need to be
    fetched, plus the already-fetched race-results payload for each (so the
    "has this race happened yet" check isn't wasted)."""
    covered = {
        (int(y), int(r))
        for y, r in zip(pristine_races["year"], pristine_races["round"])
        if y < PRISTINE_LAST_YEAR or (y == PRISTINE_LAST_YEAR and r <= PRISTINE_LAST_ROUND)
    }

    candidates: List[Tuple[int, int]] = []
    for year in range(PRISTINE_LAST_YEAR, max_year + 1):
        for race in client.get_season_schedule(year):
            round_ = int(race["round"])
            if (year, round_) not in covered:
                candidates.append((year, round_))
    candidates.sort()

    confirmed: List[Tuple[int, int]] = []
    payloads: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for idx, (year, round_) in enumerate(candidates):
        force = idx >= len(candidates) - refresh_last_n_rounds if refresh_last_n_rounds else False
        race_obj = client.get_race_results(year, round_, force_refresh=force)
        if race_obj and race_obj.get("Results"):
            confirmed.append((year, round_))
            payloads[(year, round_)] = race_obj
        else:
            logger.info("Skipping %s round %s: not yet held / no results published.", year, round_)

    return confirmed, payloads


def _resolve_ref_column(df: pd.DataFrame, ref_col: str, id_col: str, allocator: IdAllocator) -> pd.DataFrame:
    if len(df) == 0:
        df = df.copy()
        df[id_col] = pd.Series(dtype="int64")
        return df.drop(columns=[ref_col]) if ref_col in df.columns else df
    df = df.copy()
    df[id_col] = df[ref_col].map(allocator.resolve).astype("int64")
    return df.drop(columns=[ref_col])


def _cast_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    dtypes = COLUMN_DTYPES[name]
    df = df.copy()
    for col, dtype in dtypes.items():
        if col not in df.columns:
            df[col] = pd.Series([np.nan] * len(df))
        if dtype == "Int64":
            df[col] = pd.array(df[col].astype("float64").round().values, dtype="Int64")
        else:
            df[col] = df[col].astype(dtype)
    return df[list(dtypes.keys())]


def build_enriched_db(
    output_dir: str = "data/enriched/rel-f1",
    max_year: int = 2026,
    refresh_last_n_rounds: int = 0,
    client: Optional[JolpicaClient] = None,
) -> Dict[str, Any]:
    """Runs the full extraction + reconciliation + write pipeline.

    Returns the manifest dict that is also written to
    ``<output_dir>/manifest.json``.
    """
    client = client or JolpicaClient()
    pristine_db = load_pristine_db()
    pristine = {name: table.df.copy() for name, table in pristine_db.table_dict.items()}

    # --- status lookup: extend defensively past both the live enum's max id
    # and the highest id already used by the pristine snapshot.
    raw_status = client.get_status_table()
    status_lookup = es.build_status_lookup(raw_status)
    next_status_id = [
        max(list(status_lookup.values()) + [int(pristine["results"]["statusId"].max())]) + 1
    ]

    # --- entity ID allocators, seeded from the pristine tables
    driver_alloc = IdAllocator(_ref_to_id_map(pristine["drivers"], "driverId", "driverRef"))
    constructor_alloc = IdAllocator(_ref_to_id_map(pristine["constructors"], "constructorId", "constructorRef"))
    circuit_alloc = IdAllocator(_ref_to_id_map(pristine["circuits"], "circuitId", "circuitRef"))

    # --- metadata caches used later to build rows for newly-minted entities
    driver_meta: Dict[str, Any] = {}
    constructor_meta: Dict[str, Any] = {}
    for year in range(PRISTINE_LAST_YEAR, max_year + 1):
        for d in client.get_season_drivers(year):
            driver_meta[d["driverId"]] = d
            driver_alloc.resolve(d["driverId"])
        for c in client.get_season_constructors(year):
            constructor_meta[c["constructorId"]] = c
            constructor_alloc.resolve(c["constructorId"])

    circuit_meta = {c["circuitId"]: c for c in client.get_circuits()}

    # --- determine which rounds actually need enrichment
    rounds, race_payloads = _determine_new_rounds(client, pristine["races"], max_year, refresh_last_n_rounds)
    logger.info("Enriching %d new round(s): %s", len(rounds), rounds)

    existing_race_key_to_id = {
        (int(y), int(r)): int(rid)
        for rid, y, r in zip(pristine["races"]["raceId"], pristine["races"]["year"], pristine["races"]["round"])
    }
    next_race_id = int(pristine["races"]["raceId"].max()) + 1

    new_race_rows: List[Dict[str, Any]] = []
    results_frames, quali_frames, sprint_frames = [], [], []
    dstanding_frames, cstanding_frames = [], []

    for idx, (year, round_) in enumerate(rounds):
        force = idx >= len(rounds) - refresh_last_n_rounds if refresh_last_n_rounds else False
        race_result_obj = race_payloads[(year, round_)]
        quali_obj = client.get_qualifying(year, round_, force_refresh=force)
        sprint_obj = client.get_sprint(year, round_, force_refresh=force)
        dstand_obj = client.get_driver_standings(year, round_, force_refresh=force)
        cstand_obj = client.get_constructor_standings(year, round_, force_refresh=force)

        circuit_ref = race_result_obj.get("Circuit", {}).get("circuitId")
        if circuit_ref:
            circuit_alloc.resolve(circuit_ref)

        key = (year, round_)
        if key not in existing_race_key_to_id:
            new_race_rows.append({
                "raceId": next_race_id,
                "year": year,
                "round": round_,
                "circuitRef": circuit_ref,
                "name": race_result_obj.get("raceName"),
                "date": es._race_datetime(race_result_obj["date"], race_result_obj.get("time")),
                "time": es._strip_tz(race_result_obj.get("time")),
            })
            existing_race_key_to_id[key] = next_race_id
            next_race_id += 1

        race_id = existing_race_key_to_id[key]
        race_date = es._race_datetime(race_result_obj["date"], race_result_obj.get("time"))

        res_df = es.normalize_results(race_result_obj, status_lookup, next_status_id)
        res_df["raceId"] = race_id
        results_frames.append(res_df)

        q_df = es.normalize_qualifying(quali_obj)
        if len(q_df):
            q_df["raceId"] = race_id
        quali_frames.append(q_df)

        sp_df = es.normalize_sprint(sprint_obj)
        if len(sp_df):
            sp_df["raceId"] = race_id
        sprint_frames.append(sp_df)

        ds_df = es.normalize_driver_standings(dstand_obj, race_date)
        if len(ds_df):
            ds_df["raceId"] = race_id
        dstanding_frames.append(ds_df)

        cs_df = es.normalize_constructor_standings(cstand_obj, race_date)
        if len(cs_df):
            cs_df["raceId"] = race_id
        cstanding_frames.append(cs_df)

    def _concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
        non_empty = [f for f in frames if len(f)]
        return pd.concat(non_empty, ignore_index=True) if non_empty else (frames[0] if frames else pd.DataFrame())

    results_new_raw = _concat(results_frames)
    quali_new_raw = _concat(quali_frames)
    sprint_new_raw = _concat(sprint_frames)
    dstanding_new_raw = _concat(dstanding_frames)
    cstanding_new_raw = _concat(cstanding_frames)

    # constructor_results has no direct endpoint: derive it (main + sprint
    # points) while constructorRef/date are still available, before the ID
    # resolution pass below drops the ref columns.
    cresults_new_raw = es.compute_constructor_results(
        results_new_raw if len(results_new_raw) else pd.DataFrame(columns=["year", "round", "constructorRef", "points", "date"]),
        sprint_new_raw if len(sprint_new_raw) else pd.DataFrame(columns=["year", "round", "constructorRef", "points", "date"]),
    )
    race_key_df = pd.DataFrame(
        [{"year": y, "round": r, "raceId": rid} for (y, r), rid in existing_race_key_to_id.items()]
    )
    cresults_new_raw = cresults_new_raw.merge(race_key_df, on=["year", "round"], how="left")

    # --- resolve ref strings -> final integer IDs
    results_new = _resolve_ref_column(results_new_raw, "driverRef", "driverId", driver_alloc)
    results_new = _resolve_ref_column(results_new, "constructorRef", "constructorId", constructor_alloc)

    quali_new = _resolve_ref_column(quali_new_raw, "driverRef", "driverId", driver_alloc)
    quali_new = _resolve_ref_column(quali_new, "constructorRef", "constructorId", constructor_alloc)

    sprint_new = _resolve_ref_column(sprint_new_raw, "driverRef", "driverId", driver_alloc)
    sprint_new = _resolve_ref_column(sprint_new, "constructorRef", "constructorId", constructor_alloc)

    dstanding_new = _resolve_ref_column(dstanding_new_raw, "driverRef", "driverId", driver_alloc)
    cstanding_new = _resolve_ref_column(cstanding_new_raw, "constructorRef", "constructorId", constructor_alloc)
    cresults_new = _resolve_ref_column(cresults_new_raw, "constructorRef", "constructorId", constructor_alloc)

    races_new = pd.DataFrame(new_race_rows)
    if len(races_new):
        races_new = _resolve_ref_column(races_new, "circuitRef", "circuitId", circuit_alloc)

    drivers_new = (
        es.normalize_drivers([driver_meta[ref] for ref in driver_alloc.new_refs if ref in driver_meta])
        if driver_alloc.new_refs else pd.DataFrame()
    )
    if len(drivers_new):
        drivers_new["driverId"] = drivers_new["driverRef"].map(driver_alloc.map)

    constructors_new = (
        es.normalize_constructors([constructor_meta[ref] for ref in constructor_alloc.new_refs if ref in constructor_meta])
        if constructor_alloc.new_refs else pd.DataFrame()
    )
    if len(constructors_new):
        constructors_new["constructorId"] = constructors_new["constructorRef"].map(constructor_alloc.map)

    circuits_new = (
        es.normalize_circuits([circuit_meta[ref] for ref in circuit_alloc.new_refs if ref in circuit_meta])
        if circuit_alloc.new_refs else pd.DataFrame()
    )
    if len(circuits_new):
        circuits_new["circuitId"] = circuits_new["circuitRef"].map(circuit_alloc.map)

    # --- assign synthetic pkeys for every new row, in the already-established
    # chronological append order (see module docstring for why this must
    # never re-sort existing rows).
    def _assign_pkey(pristine_df: pd.DataFrame, new_df: pd.DataFrame, pkey: str) -> pd.DataFrame:
        if len(new_df) == 0:
            return new_df
        start = int(pristine_df[pkey].max()) + 1
        new_df = new_df.copy()
        new_df[pkey] = np.arange(start, start + len(new_df))
        return new_df

    races_new = _assign_pkey(pristine["races"], races_new, "raceId") if len(races_new) else races_new
    results_new = _assign_pkey(pristine["results"], results_new, "resultId")
    quali_new = _assign_pkey(pristine["qualifying"], quali_new, "qualifyId")
    dstanding_new = _assign_pkey(pristine["standings"], dstanding_new, "driverStandingsId")
    cstanding_new = _assign_pkey(pristine["constructor_standings"], cstanding_new, "constructorStandingsId")
    cresults_new = _assign_pkey(pristine["constructor_results"], cresults_new, "constructorResultsId")

    # --- final concatenation + exact-dtype cast, table by table
    final_tables: Dict[str, pd.DataFrame] = {}
    concat_plan = {
        "circuits": (pristine["circuits"], circuits_new),
        "drivers": (pristine["drivers"], drivers_new),
        "constructors": (pristine["constructors"], constructors_new),
        "races": (pristine["races"], races_new),
        "results": (pristine["results"], results_new),
        "standings": (pristine["standings"], dstanding_new),
        "constructor_results": (pristine["constructor_results"], cresults_new),
        "constructor_standings": (pristine["constructor_standings"], cstanding_new),
        "qualifying": (pristine["qualifying"], quali_new),
    }
    for name, (old_df, new_df) in concat_plan.items():
        combined = pd.concat([old_df, new_df], ignore_index=True) if len(new_df) else old_df.copy()
        final_tables[name] = _cast_table(name, combined)

    # sprint_results is kept OUTSIDE db/ so relbench's Database.load() (which
    # globs every *.parquet file in the directory) never picks it up as a
    # graph node type -- it is not part of the RelBench schema.
    sprint_final = pd.concat(
        [sprint_new_raw if len(sprint_new_raw) else pd.DataFrame()], ignore_index=True
    )
    if len(sprint_final):
        sprint_final = _resolve_ref_column(sprint_final, "driverRef", "driverId", driver_alloc)
        sprint_final = _resolve_ref_column(sprint_final, "constructorRef", "constructorId", constructor_alloc)

    # --- write output
    out_path = Path(output_dir)
    db_path = out_path / "db"
    db_path.mkdir(parents=True, exist_ok=True)

    table_dict = {}
    for name, df in final_tables.items():
        pkey_col, fkeys, time_col = TABLE_SCHEMAS[name]
        table_dict[name] = Table(df=df, fkey_col_to_pkey_table=fkeys, pkey_col=pkey_col, time_col=time_col)

    db = Database(table_dict)
    db.save(str(db_path))

    if len(sprint_final):
        sprint_path = out_path / "sprint_results.parquet"
        sprint_final.to_parquet(sprint_path, index=False)

    manifest = _build_manifest(
        out_path, pristine, final_tables, rounds, driver_alloc, constructor_alloc,
        circuit_alloc, status_lookup, max_year,
    )
    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Enriched rel-f1 database written to %s", db_path)
    return manifest


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(
    out_path: Path,
    pristine: Dict[str, pd.DataFrame],
    final_tables: Dict[str, pd.DataFrame],
    rounds: List[Tuple[int, int]],
    driver_alloc: IdAllocator,
    constructor_alloc: IdAllocator,
    circuit_alloc: IdAllocator,
    status_lookup: Dict[str, int],
    max_year: int,
) -> Dict[str, Any]:
    db_path = out_path / "db"
    table_hashes = {
        name: _sha256_of_file(db_path / f"{name}.parquet")
        for name in final_tables
        if (db_path / f"{name}.parquet").exists()
    }
    rows_added = {
        name: int(len(final_tables[name]) - len(pristine[name]))
        for name in final_tables
    }
    seasons_covered = sorted({y for y, _ in rounds})
    rounds_by_season = {
        str(y): sorted(r for yy, r in rounds if yy == y) for y in seasons_covered
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Jolpica F1 API (https://api.jolpi.ca/ergast/), Ergast-schema compatible",
        "pristine_cutoff": {"year": PRISTINE_LAST_YEAR, "round": PRISTINE_LAST_ROUND},
        "max_year_requested": max_year,
        "new_rounds_added": len(rounds),
        "rounds_by_season": rounds_by_season,
        "rows_added_per_table": rows_added,
        "final_row_counts": {name: int(len(df)) for name, df in final_tables.items()},
        "table_sha256": table_hashes,
        "new_entities": {
            "drivers": driver_alloc.new_refs,
            "constructors": constructor_alloc.new_refs,
            "circuits": circuit_alloc.new_refs,
        },
        "status_lookup_size": len(status_lookup),
        "id_allocation": {
            "drivers": driver_alloc.map,
            "constructors": constructor_alloc.map,
            "circuits": circuit_alloc.map,
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_enriched_db()
