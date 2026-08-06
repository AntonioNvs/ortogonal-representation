"""Normalizes raw Jolpica (Ergast-schema) JSON payloads into DataFrames that
mirror the exact column layout RelBench's ``rel-f1`` uses (see
``relbench.datasets.f1.F1Dataset.make_db``).

Every function here is keyed by *reference strings* (``driverRef``,
``constructorRef``, ``circuitRef``) and ``(year, round)`` pairs rather than
final integer IDs. Reconciling those refs against RelBench's existing integer
ID space (reusing IDs for known entities, minting new ones for new entities)
is the job of ``build_enriched_db.py`` — this module only knows how to read
Ergast-shaped JSON.

Semantics validated empirically against the pristine ``rel-f1`` cache
(see the plan discussion for details):

- ``positionOrder`` is simply the 1-based rank of a driver's entry in the
  ``Results`` array (Jolpica already returns results in classification
  order); this was confirmed to equal Ergast's own numeric ``position``
  field even for retirees.
- ``position`` (the *task target*) must be NaN whenever ``positionText`` is
  not a plain integer (i.e. the driver is not classified: "R", "D", "E",
  "W", "F", "N"). RelBench's own ``position`` column has this exact
  semantic (21.9% NaN over 2000-2023), whereas Ergast's raw numeric
  ``position`` field is always populated (it equals ``positionOrder``).
- ``milliseconds`` comes from ``Time.millis`` and is only present for
  classified, same-lap finishers.
- ``rank``/``fastestLap`` come from ``FastestLap.rank``/``FastestLap.lap``.
- ``constructor_results`` has no direct REST endpoint; RelBench's version
  includes sprint points, confirmed by reconciling
  ``constructor_results.points`` against ``sum(results.points)`` on the
  historical overlap: every mismatch (38/4706 rows since 2000) falls
  exactly on a sprint weekend. We reproduce that by summing main-race
  points plus sprint points per (race, constructor).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _to_float(value: Optional[str]) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_int(value: Optional[str]) -> float:
    """Returns a float so callers can hold NaN before a final cast."""
    v = _to_float(value)
    return v


def _safe_int(value: Optional[str], default: int = 0) -> int:
    v = _to_float(value)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return int(v)


def _strip_tz(time_str: Optional[str]) -> str:
    if not time_str:
        return "00:00:00"
    return re.sub(r"[Zz]$", "", time_str.strip())


def _race_datetime(date_str: str, time_str: Optional[str]) -> pd.Timestamp:
    return pd.to_datetime(f"{date_str} {_strip_tz(time_str)}")


# ---------------------------------------------------------------------------
# Static / near-static entity tables
# ---------------------------------------------------------------------------

def normalize_circuits(raw_circuits: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for c in raw_circuits:
        loc = c.get("Location", {})
        rows.append({
            "circuitRef": c["circuitId"],
            "name": c.get("circuitName"),
            "location": loc.get("locality"),
            "country": loc.get("country"),
            "lat": _to_float(loc.get("lat")),
            "lng": _to_float(loc.get("long")),
            # Ergast's REST API does not expose altitude at all; RelBench's
            # pristine table sources it from a different raw dump and
            # already tolerates missing values ("\\N" -> NaN) for 3 rows.
            "alt": np.nan,
        })
    return pd.DataFrame(rows)


def normalize_drivers(raw_drivers: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for d in raw_drivers:
        rows.append({
            "driverRef": d["driverId"],
            "code": d.get("code"),
            "forename": d.get("givenName"),
            "surname": d.get("familyName"),
            "dob": pd.to_datetime(d.get("dateOfBirth")) if d.get("dateOfBirth") else pd.NaT,
            "nationality": d.get("nationality"),
        })
    return pd.DataFrame(rows)


def normalize_constructors(raw_constructors: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for c in raw_constructors:
        rows.append({
            "constructorRef": c["constructorId"],
            "name": c.get("name"),
            "nationality": c.get("nationality"),
        })
    return pd.DataFrame(rows)


def build_status_lookup(raw_status: List[Dict[str, Any]]) -> Dict[str, int]:
    """Text -> canonical Ergast/Jolpica statusId, straight from /status.json."""
    return {s["status"]: int(s["statusId"]) for s in raw_status}


# ---------------------------------------------------------------------------
# Per-race (schedule) table
# ---------------------------------------------------------------------------

def normalize_races(raw_schedule: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in raw_schedule:
        date_str = r["date"]
        time_str = r.get("time")
        rows.append({
            "year": int(r["season"]),
            "round": int(r["round"]),
            "circuitRef": r["Circuit"]["circuitId"],
            "name": r.get("raceName"),
            "date": _race_datetime(date_str, time_str),
            "time": _strip_tz(time_str),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-race child tables (results, qualifying, sprint, standings)
# ---------------------------------------------------------------------------

def normalize_results(
    race_obj: Dict[str, Any],
    status_lookup: Dict[str, int],
    next_status_id: List[int],
) -> pd.DataFrame:
    """``next_status_id`` is a 1-element mutable list used as an out-param
    counter so callers can allocate fresh IDs for any status text not found
    in the canonical Ergast lookup (defensive; not expected to trigger)."""
    year = int(race_obj["season"])
    round_ = int(race_obj["round"])
    race_date = _race_datetime(race_obj["date"], race_obj.get("time"))

    rows = []
    for idx, res in enumerate(race_obj.get("Results", []), start=1):
        status_text = res.get("status", "Unknown")
        if status_text not in status_lookup:
            status_lookup[status_text] = next_status_id[0]
            next_status_id[0] += 1

        position_text = res.get("positionText", "")
        position = _to_float(res.get("position")) if position_text.isdigit() else np.nan

        fastest_lap = res.get("FastestLap", {})
        time_info = res.get("Time", {})

        rows.append({
            "year": year,
            "round": round_,
            "driverRef": res["Driver"]["driverId"],
            "constructorRef": res["Constructor"]["constructorId"],
            "number": _to_float(res.get("number")),
            "grid": _safe_int(res.get("grid")),
            "position": position,
            "positionOrder": idx,
            "points": _to_float(res.get("points")),
            "laps": _safe_int(res.get("laps")),
            "milliseconds": _to_float(time_info.get("millis")),
            "fastestLap": _to_float(fastest_lap.get("lap")),
            "rank": _to_float(fastest_lap.get("rank")),
            "status": status_text,
            "statusId": status_lookup[status_text],
            "date": race_date,
        })
    return pd.DataFrame(rows)


def normalize_sprint(race_obj: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Kept for future use (not wired into the graph or the target). Schema
    mirrors ``normalize_results`` minus the removed columns."""
    columns = [
        "year", "round", "driverRef", "constructorRef", "number", "grid",
        "position", "positionOrder", "points", "laps", "milliseconds",
        "status", "statusId", "date",
    ]
    if race_obj is None:
        return pd.DataFrame(columns=columns)

    year = int(race_obj["season"])
    round_ = int(race_obj["round"])
    race_date = _race_datetime(race_obj["date"], race_obj.get("time"))

    rows = []
    for idx, res in enumerate(race_obj.get("SprintResults", []), start=1):
        position_text = res.get("positionText", "")
        position = _to_float(res.get("position")) if position_text.isdigit() else np.nan
        time_info = res.get("Time", {})
        rows.append({
            "year": year,
            "round": round_,
            "driverRef": res["Driver"]["driverId"],
            "constructorRef": res["Constructor"]["constructorId"],
            "number": _to_float(res.get("number")),
            "grid": _safe_int(res.get("grid")),
            "position": position,
            "positionOrder": idx,
            "points": _to_float(res.get("points")),
            "laps": _safe_int(res.get("laps")),
            "milliseconds": _to_float(time_info.get("millis")),
            "status": res.get("status", "Unknown"),
            "statusId": np.nan,
            "date": race_date,
        })
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def normalize_qualifying(race_obj: Optional[Dict[str, Any]]) -> pd.DataFrame:
    columns = ["year", "round", "driverRef", "constructorRef", "number", "position", "date"]
    if race_obj is None:
        return pd.DataFrame(columns=columns)

    year = int(race_obj["season"])
    round_ = int(race_obj["round"])
    # Qualifying happens the day before the race, matching f1.py's convention.
    quali_date = _race_datetime(race_obj["date"], race_obj.get("time")) - pd.Timedelta(days=1)

    rows = []
    for res in race_obj.get("QualifyingResults", []):
        rows.append({
            "year": year,
            "round": round_,
            "driverRef": res["Driver"]["driverId"],
            "constructorRef": res["Constructor"]["constructorId"],
            "number": _safe_int(res.get("number")),
            "position": _safe_int(res.get("position")),
            "date": quali_date,
        })
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def normalize_driver_standings(standings_obj: Optional[Dict[str, Any]], race_date: pd.Timestamp) -> pd.DataFrame:
    columns = ["year", "round", "driverRef", "points", "position", "wins", "date"]
    if standings_obj is None:
        return pd.DataFrame(columns=columns)

    year = int(standings_obj["season"])
    round_ = int(standings_obj["round"])

    rows = []
    for s in standings_obj.get("DriverStandings", []):
        rows.append({
            "year": year,
            "round": round_,
            "driverRef": s["Driver"]["driverId"],
            "points": _to_float(s.get("points")),
            "position": _safe_int(s.get("position")),
            "wins": _safe_int(s.get("wins")),
            "date": race_date,
        })
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def normalize_constructor_standings(standings_obj: Optional[Dict[str, Any]], race_date: pd.Timestamp) -> pd.DataFrame:
    columns = ["year", "round", "constructorRef", "points", "position", "wins", "date"]
    if standings_obj is None:
        return pd.DataFrame(columns=columns)

    year = int(standings_obj["season"])
    round_ = int(standings_obj["round"])

    rows = []
    for s in standings_obj.get("ConstructorStandings", []):
        rows.append({
            "year": year,
            "round": round_,
            "constructorRef": s["Constructor"]["constructorId"],
            "points": _to_float(s.get("points")),
            "position": _safe_int(s.get("position")),
            "wins": _safe_int(s.get("wins")),
            "date": race_date,
        })
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def compute_constructor_results(results_df: pd.DataFrame, sprint_df: pd.DataFrame) -> pd.DataFrame:
    """RelBench's constructor_results.points = main race points + sprint
    points, summed per (year, round, constructor). There is no direct
    Ergast/Jolpica endpoint for this, so we derive it from the two result
    tables we already fetch."""
    parts = [results_df[["year", "round", "constructorRef", "points", "date"]]]
    if len(sprint_df) > 0:
        parts.append(sprint_df[["year", "round", "constructorRef", "points", "date"]])
    combined = pd.concat(parts, ignore_index=True)

    grouped = (
        combined.groupby(["year", "round", "constructorRef"], as_index=False)
        .agg(points=("points", "sum"), date=("date", "first"))
    )
    return grouped
