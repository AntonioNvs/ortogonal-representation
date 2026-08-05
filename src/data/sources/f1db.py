"""Minimal reader for the f1db (https://github.com/f1db/f1db) CSV release,
used only as an independent secondary source to cross-validate the rows the
enrichment pipeline derives from Jolpica.

f1db uses its own schema (hyphenated string IDs, different column names,
many more tables) -- we don't attempt full schema compatibility, just enough
to join on natural keys (year, round, driver ref) and compare a handful of
outcome columns.
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

RELEASES_API = "https://api.github.com/repos/f1db/f1db/releases/latest"


def _to_ref(f1db_slug: str) -> str:
    """f1db uses hyphenated slugs ("max-verstappen", "red-bull"); Ergast/
    RelBench use underscores ("max_verstappen", "red_bull"). Used only for
    constructors, where Ergast's ``constructorRef`` is consistently a
    "firstname_lastname"-style slug matching f1db's own id (unlike drivers,
    see ``name_slug`` below).
    """
    return f1db_slug.replace("-", "_")


def name_slug(first_name: str, last_name: str) -> str:
    """Normalizes a (first, last) name pair into a comparable ascii slug.

    Ergast's ``driverRef`` is usually just the surname ("hamilton"), only
    falling back to "firstname_lastname" for disambiguation (e.g.
    "max_verstappen", since another Verstappen raced decades earlier) --  so
    joining on ``driverRef`` against f1db's "firstname-lastname" driver id
    matches only a small minority of drivers. Instead, both sides are
    reduced to the same "firstname_lastname" slug (accents stripped,
    non-alphanumerics collapsed to underscores) derived straight from the
    driver's name, which is a stable, source-independent key.
    """
    full = f"{first_name} {last_name}"
    ascii_full = unicodedata.normalize("NFKD", full).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_full.lower()).strip("_")


def download_f1db_csv(cache_dir: str = "data/raw/f1db", force: bool = False) -> Path:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    zip_path = cache_path / "f1db-csv.zip"
    extracted_dir = cache_path / "extracted"

    if not force and extracted_dir.exists() and any(extracted_dir.iterdir()):
        return extracted_dir

    resp = requests.get(RELEASES_API, timeout=20)
    resp.raise_for_status()
    release = resp.json()
    asset = next(a for a in release["assets"] if a["name"] == "f1db-csv.zip")

    with open(cache_path / "release_meta.json", "w") as f:
        json.dump({"tag_name": release["tag_name"], "asset": asset["name"]}, f, indent=2)

    data = requests.get(asset["browser_download_url"], timeout=120).content
    zip_path.write_bytes(data)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(extracted_dir)

    return extracted_dir


def load_race_results(cache_dir: str = "data/raw/f1db") -> pd.DataFrame:
    """Returns columns: year, round, driverRef (f1db's own hyphenated id,
    kept for debugging), name_slug (join key -- see ``name_slug``), points,
    position, grid."""
    extracted_dir = download_f1db_csv(cache_dir)
    df = pd.read_csv(extracted_dir / "f1db-races-race-results.csv")
    drivers = pd.read_csv(extracted_dir / "f1db-drivers.csv")[["id", "firstName", "lastName"]]
    drivers["name_slug"] = drivers.apply(lambda r: name_slug(r["firstName"], r["lastName"]), axis=1)
    driver_slug = drivers.set_index("id")["name_slug"]

    out = pd.DataFrame({
        "year": df["year"],
        "round": df["round"],
        "driverRef": df["driverId"].map(_to_ref),
        "name_slug": df["driverId"].map(driver_slug),
        "points": df["points"],
        "position": df["positionNumber"],
        "grid": df["gridPositionNumber"],
    })
    return out


def load_constructor_results(cache_dir: str = "data/raw/f1db") -> pd.DataFrame:
    """Returns per-(race, constructor) summed points from f1db's race results,
    matching how RelBench's constructor_results.points is defined (main +
    sprint points)."""
    extracted_dir = download_f1db_csv(cache_dir)
    main = pd.read_csv(extracted_dir / "f1db-races-race-results.csv")
    sprint_path = extracted_dir / "f1db-races-sprint-race-results.csv"
    frames = [main[["year", "round", "constructorId", "points"]]]
    if sprint_path.exists():
        sprint = pd.read_csv(sprint_path)
        frames.append(sprint[["year", "round", "constructorId", "points"]])
    combined = pd.concat(frames, ignore_index=True)
    combined["points"] = pd.to_numeric(combined["points"], errors="coerce").fillna(0.0)
    grouped = combined.groupby(["year", "round", "constructorId"], as_index=False)["points"].sum()
    grouped["constructorRef"] = grouped["constructorId"].map(_to_ref)
    return grouped[["year", "round", "constructorRef", "points"]]
