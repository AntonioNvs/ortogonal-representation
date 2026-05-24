"""
Track feature extraction for the MLP pista encoder.

Combines static data from the RelBench ``circuits`` table (altitude via ``alt``
column) with dynamic/geometric data from FastF1 (track rotation, corner count,
estimated track length, and weather conditions from the qualifying session).

The rel-f1 database is accessed with ``upto_test_timestamp=False`` (via the
shared ``get_table_df`` helper), so data from 2014–2023 is available and the
``races`` table can be joined by ``year``/``round`` for any modern season.

Note: the rel-f1 ``circuits`` table has ``alt`` (altitude in metres) but
no ``length`` or ``turns`` columns.  Track length is estimated from FastF1
circuit corner distances (requires ``telemetry=True``), with an official FIA
fallback table for when telemetry is unavailable.

Output
------
A single CSV per season at ``output/track/track_features_{year}.csv``
with one row per Grand Prix circuit.
"""
import logging
import os

import fastf1
import pandas as pd
from tqdm import tqdm

import config as cfg
from id_mapping import get_table_df

logger = logging.getLogger(__name__)


_FALLBACK_LENGTH_M: dict[str, float] = {
    "albert_park": 5278.0,
    "bahrain": 5412.0,
    "jeddah": 6174.0,
    "catalunya": 4675.0,
    "monaco": 3337.0,
    "villeneuve": 4361.0,
    "red_bull_ring": 4318.0,
    "silverstone": 5891.0,
    "hungaroring": 4381.0,
    "spa": 7004.0,
    "zandvoort": 4259.0,
    "monza": 5793.0,
    "marina_bay": 5063.0,
    "suzuka": 5807.0,
    "americas": 5513.0,
    "rodriguez": 4304.0,
    "interlagos": 4309.0,
    "miami": 5412.0,
    "losail": 5380.0,
    "vegas": 6201.0,
    "yas_marina": 5281.0,
    "baku": 6003.0,
    "sochi": 5848.0,
    "shanghai": 5451.0,
    "portimao": 4653.0,
    "mugello": 5245.0,
    "imola": 4909.0,
}


def extract_single_track(
    dataset,
    year: int,
    round_num: int,
    event_name: str,
    cache_dir: str,
) -> dict | None:
    """
    Extract the feature vector for a single circuit.

    Features
    --------
    From rel-f1 (static, joined by year/round → circuitId):
        altitude_m      — circuit altitude in metres (from ``circuits.alt``)
        circuit_ref     — rel-f1 circuitRef slug

    From FastF1 (geometric / environmental):
        rotation        — circuit rotation angle (degrees)
        corners_count   — number of official corners
        length_m        — track length in metres (estimated from corner distances
                          with telemetry=True, or from FIA fallback table)
        avg_track_temp  — mean track surface temperature (°C)
        avg_air_temp    — mean ambient temperature (°C)
        avg_humidity    — mean relative humidity (%)
    """
    altitude_m = 0.0
    circuit_ref = None
    circuit_id = None

    try:
        races_df = get_table_df(dataset, "races")
        circuits_df = get_table_df(dataset, "circuits")

        race = races_df[
            (races_df["year"] == year) & (races_df["round"] == round_num)
        ]

        if not race.empty:
            circuit_id = race.iloc[0]["circuitId"]
            circuit = circuits_df[circuits_df["circuitId"] == circuit_id]

            if not circuit.empty:
                c = circuit.iloc[0]
                circuit_ref = str(c.get("circuitRef", ""))
                alt_val = c.get("alt", None)
                if alt_val is not None and pd.notna(alt_val):
                    altitude_m = float(alt_val)
        else:
            logger.warning(
                f"No rel-f1 race entry for {event_name} "
                f"(year={year}, round={round_num})"
            )
    except Exception as e:
        logger.warning(f"Static track data error for {event_name}: {e}")

    rotation = 0.0
    corners_count = 0
    length_m = _FALLBACK_LENGTH_M.get(circuit_ref or "", 5000.0)
    avg_track_temp = float("nan")
    avg_air_temp = float("nan")
    avg_humidity = float("nan")

    try:
        session = fastf1.get_session(year, round_num, "Q")
        session.load(telemetry=True, laps=True, weather=True)

        circuit_info = session.get_circuit_info()
        rotation = float(circuit_info.rotation)
        corners = circuit_info.corners
        corners_count = len(corners)

        if "Distance" in corners.columns and corners["Distance"].notna().any():
            max_dist = corners["Distance"].max()
            if pd.notna(max_dist) and max_dist > 0:
                length_m = round(float(max_dist) * 1.05, 0)

        weather = session.weather_data
        if weather is not None and not weather.empty:
            if "TrackTemp" in weather.columns:
                avg_track_temp = round(float(weather["TrackTemp"].mean()), 1)
            if "AirTemp" in weather.columns:
                avg_air_temp = round(float(weather["AirTemp"].mean()), 1)
            if "Humidity" in weather.columns:
                avg_humidity = round(float(weather["Humidity"].mean()), 1)

    except Exception as e:
        logger.warning(f"FastF1 data unavailable for {event_name}: {e}")

    return {
        "year": year,
        "round": round_num,
        "event_name": event_name,
        "circuit_id": circuit_id,
        "circuit_ref": circuit_ref,
        "altitude_m": altitude_m,
        "length_m": length_m,
        "corners_count": corners_count,
        "rotation": rotation,
        "avg_track_temp": avg_track_temp,
        "avg_air_temp": avg_air_temp,
        "avg_humidity": avg_humidity,
    }


def extract_season_tracks(
    dataset,
    year: int,
    output_dir: str,
    cache_dir: str,
) -> pd.DataFrame:
    """
    Extract track features for every circuit on the season calendar.

    Returns the DataFrame and saves it as CSV.
    """
    fastf1.Cache.enable_cache(str(cache_dir))

    schedule = fastf1.get_event_schedule(year)
    race_events = schedule[schedule["RoundNumber"] > 0]

    tracks: list[dict] = []

    for _, event in tqdm(
        race_events.iterrows(),
        total=len(race_events),
        desc=f"Track features {year}",
    ):
        result = extract_single_track(
            dataset,
            year,
            int(event["RoundNumber"]),
            str(event["EventName"]),
            cache_dir,
        )
        if result:
            tracks.append(result)

    df = pd.DataFrame(tracks)

    track_dir = os.path.join(output_dir, "track")
    os.makedirs(track_dir, exist_ok=True)
    csv_path = os.path.join(track_dir, f"track_features_{year}.csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"Track features saved: {len(df)} circuits → {csv_path}")
    return df
