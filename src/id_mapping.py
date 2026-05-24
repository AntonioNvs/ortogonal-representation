"""
Unified ID mapping between FastF1 and RelBench (rel-f1) datasets.

Creates consistent integer indices for drivers and circuits across both
data sources, enabling cross-referencing between telemetry, team data,
and track features.

Important: The RelBench dataset must be loaded with
``get_db(upto_test_timestamp=False)`` to access the full database
(1950–2023).  The default call (without the flag) applies a timestamp
filter that truncates the data to the training split (~up to 2009).
"""
import json
import logging
import os

import fastf1
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


def _get_db(dataset):
    """
    Return the full RelBench database, bypassing the training-split timestamp.

    ``get_db(upto_test_timestamp=False)`` gives access to all races from
    1950 through 2023.  The default ``get_db()`` cuts the data off at the
    training split boundary (~2009), yielding empty results for modern seasons.
    """
    if hasattr(dataset, "get_db"):
        try:
            return dataset.get_db(upto_test_timestamp=False)
        except TypeError:
            return dataset.get_db()
    return dataset.db


def get_table_df(dataset, table_name: str) -> pd.DataFrame:
    """
    Safely extract a DataFrame from a RelBench dataset table.

    Always uses the full database (upto_test_timestamp=False) so data
    from 2014–2023 is accessible.
    """
    db = _get_db(dataset)

    table = None
    if hasattr(db, "table_dict") and table_name in db.table_dict:
        table = db.table_dict[table_name]
    elif hasattr(db, "__getitem__"):
        try:
            table = db[table_name]
        except (KeyError, TypeError):
            pass

    if table is None:
        raise KeyError(f"Table '{table_name}' not found in RelBench database")

    return table.df.copy() if hasattr(table, "df") else pd.DataFrame(table)


def list_tables(dataset) -> list[str]:
    """Return all table names available in the full RelBench database."""
    db = _get_db(dataset)
    if hasattr(db, "table_dict"):
        return list(db.table_dict.keys())
    return []


def build_circuit_mapping(
    dataset, schedule: pd.DataFrame, year: int
) -> dict:
    """
    Build circuit mapping: FastF1 EventName → unified circuit index.

    Strategy: match by (year, round) — shared between both sources.
    With upto_test_timestamp=False, rel-f1 has data up to 2023.

    Returns
    -------
    dict
        {event_name: {circuit_id, circuit_idx, round, race_name, ...}}
    """
    races_df = get_table_df(dataset, "races")
    circuits_df = get_table_df(dataset, "circuits")

    year_races = races_df[races_df["year"] == year]
    race_events = schedule[schedule["RoundNumber"] > 0]

    circuit_map: dict = {}

    for _, event in race_events.iterrows():
        round_num = int(event["RoundNumber"])
        event_name = str(event["EventName"])

        race_match = year_races[year_races["round"] == round_num]
        if race_match.empty:
            logger.warning(
                f"No rel-f1 race for {event_name} (round {round_num}, {year})"
            )
            continue

        race = race_match.iloc[0]
        circuit_id = race["circuitId"]

        info: dict = {
            "circuit_id": int(circuit_id),
            "race_id": int(race["raceId"]),
            "round": round_num,
            "race_name": str(race.get("name", event_name)),
        }

        circuit = circuits_df[circuits_df["circuitId"] == circuit_id]
        if not circuit.empty:
            c = circuit.iloc[0]
            info.update(
                {
                    "circuit_name": str(c.get("name", "")),
                    "circuit_ref": str(c.get("circuitRef", "")),
                    "location": str(c.get("location", "")),
                    "country": str(c.get("country", "")),
                    "lat": float(c["lat"]) if pd.notna(c.get("lat")) else None,
                    "lng": float(c["lng"]) if pd.notna(c.get("lng")) else None,
                    "altitude_m": float(c["alt"]) if pd.notna(c.get("alt")) else None,
                }
            )

        circuit_map[event_name] = info

    sorted_events = sorted(circuit_map, key=lambda e: circuit_map[e]["round"])
    for idx, ev in enumerate(sorted_events):
        circuit_map[ev]["circuit_idx"] = idx

    logger.info(f"Circuit mapping: {len(circuit_map)} circuits for {year}")
    return circuit_map


def build_driver_mapping(
    dataset, schedule: pd.DataFrame, year: int, cache_dir: str | None = None
) -> dict:
    """
    Build driver mapping: FastF1 Abbreviation → unified driver index.

    Strategy:
    1. Collect rel-f1 driverIds that raced in *year* via results table.
    2. Match rel-f1 driver 'code' column with FastF1 abbreviation.
    3. Fallback: surname[:3] match for drivers whose 3-letter code differs.

    Returns
    -------
    dict
        {abbreviation: {driver_id, driver_idx, name, nationality, ...}}
    """
    drivers_df = get_table_df(dataset, "drivers")
    results_df = get_table_df(dataset, "results")
    races_df = get_table_df(dataset, "races")

    year_race_ids = races_df[races_df["year"] == year]["raceId"].tolist()
    year_driver_ids = results_df[results_df["raceId"].isin(year_race_ids)][
        "driverId"
    ].unique()
    year_drivers = drivers_df[drivers_df["driverId"].isin(year_driver_ids)].copy()

    fastf1_abbrs: set[str] = set()
    fastf1_details: dict[str, dict] = {}
    race_events = schedule[schedule["RoundNumber"] > 0]

    if cache_dir:
        fastf1.Cache.enable_cache(str(cache_dir))

    for _, event in race_events.iterrows():
        try:
            session = fastf1.get_session(
                year, int(event["RoundNumber"]), cfg.SESSION_TYPE
            )
            session.load(telemetry=False, laps=False, weather=False)
            if session.results is not None and not session.results.empty:
                res = session.results
                fastf1_abbrs = set(res["Abbreviation"].dropna().unique())
                for _, row in res.iterrows():
                    abbr = row.get("Abbreviation", "")
                    if pd.isna(abbr) or not abbr:
                        continue
                    fastf1_details[str(abbr)] = {
                        "forename": str(row.get("FirstName", "")),
                        "surname": str(row.get("LastName", "")),
                        "nationality": str(row.get("CountryCode", "")),
                    }
                break
        except Exception as e:
            logger.debug(f"Session load for driver list failed: {e}")
            continue

    by_code: dict[str, pd.Series] = {}
    by_surname3: dict[str, pd.Series] = {}
    if "code" in year_drivers.columns:
        for _, drv in year_drivers.iterrows():
            code = drv.get("code", "")
            if not pd.isna(code) and str(code).strip() not in ("", "\\N"):
                by_code[str(code).strip()] = drv
            surname = str(drv.get("surname", "")).upper()
            if len(surname) >= 3:
                by_surname3.setdefault(surname[:3], drv)

    driver_map: dict = {}

    for code, drv in by_code.items():
        ff1 = fastf1_details.get(code, {})
        driver_map[code] = {
            "driver_id": int(drv["driverId"]),
            "driver_ref": str(drv.get("driverRef", "")),
            "forename": ff1.get("forename") or str(drv.get("forename", "")),
            "surname": ff1.get("surname") or str(drv.get("surname", "")),
            "name": (
                f"{ff1.get('forename') or drv.get('forename', '')} "
                f"{ff1.get('surname') or drv.get('surname', '')}"
            ).strip(),
            "nationality": ff1.get("nationality") or str(drv.get("nationality", "")),
        }

    for abbr in fastf1_abbrs:
        if abbr in driver_map:
            continue
        if abbr in by_surname3:
            drv = by_surname3[abbr]
            ff1 = fastf1_details.get(abbr, {})
            driver_map[abbr] = {
                "driver_id": int(drv["driverId"]),
                "driver_ref": str(drv.get("driverRef", "")),
                "forename": ff1.get("forename") or str(drv.get("forename", "")),
                "surname": ff1.get("surname") or str(drv.get("surname", "")),
                "name": (
                    f"{ff1.get('forename') or drv.get('forename', '')} "
                    f"{ff1.get('surname') or drv.get('surname', '')}"
                ).strip(),
                "nationality": ff1.get("nationality") or str(drv.get("nationality", "")),
            }
        else:
            ff1 = fastf1_details.get(abbr, {})
            driver_map[abbr] = {
                "driver_id": None,
                "driver_ref": "",
                "forename": ff1.get("forename", ""),
                "surname": ff1.get("surname", ""),
                "name": f"{ff1.get('forename', '')} {ff1.get('surname', '')}".strip(),
                "nationality": ff1.get("nationality", ""),
            }

    for idx, abbr in enumerate(sorted(driver_map)):
        driver_map[abbr]["driver_idx"] = idx

    logger.info(f"Driver mapping: {len(driver_map)} drivers for {year}")
    return driver_map


def save_mappings(
    driver_map: dict, circuit_map: dict, output_dir: str, year: int
) -> None:
    """Save mappings to CSV and JSON for human inspection."""
    map_dir = os.path.join(output_dir, "mappings")
    os.makedirs(map_dir, exist_ok=True)

    driver_rows = [
        {"abbreviation": abbr, **info}
        for abbr, info in sorted(
            driver_map.items(), key=lambda x: x[1]["driver_idx"]
        )
    ]
    pd.DataFrame(driver_rows).to_csv(
        os.path.join(map_dir, f"driver_mapping_{year}.csv"), index=False
    )

    circuit_rows = [
        {"event_name": ev, **info}
        for ev, info in sorted(
            circuit_map.items(), key=lambda x: x[1]["circuit_idx"]
        )
    ]
    pd.DataFrame(circuit_rows).to_csv(
        os.path.join(map_dir, f"circuit_mapping_{year}.csv"), index=False
    )

    combined = {"year": year, "drivers": driver_map, "circuits": circuit_map}
    json_path = os.path.join(map_dir, f"mappings_{year}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Mappings saved to {map_dir}")


def build_all_mappings(dataset, year: int, cache_dir: str | None = None):
    """
    Build driver and circuit mappings for a given year.

    Returns
    -------
    tuple
        (driver_map, circuit_map, fastf1_schedule)
    """
    if cache_dir:
        fastf1.Cache.enable_cache(str(cache_dir))

    schedule = fastf1.get_event_schedule(year)

    driver_map = build_driver_mapping(dataset, schedule, year, cache_dir)
    circuit_map = build_circuit_mapping(dataset, schedule, year)

    return driver_map, circuit_map, schedule
