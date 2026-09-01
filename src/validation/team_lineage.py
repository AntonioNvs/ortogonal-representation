"""Constructor lineage mapping for rebrand continuity."""

from __future__ import annotations

import pandas as pd

TEAM_LINEAGES: dict[str, tuple[str, ...]] = {
    "red_bull": ("stewart", "jaguar", "red_bull"),
    "mercedes": ("tyrrell", "bar", "honda", "brawn", "mercedes"),
    "aston_martin": ("jordan", "midland", "spyker", "force_india", "racing_point", "aston_martin"),
    "rb": ("minardi", "toro_rosso", "alphatauri", "rb"),
    "alpine": ("benetton", "renault", "lotus_f1", "alpine"),
    "caterham": ("lotus_racing", "team_lotus", "caterham"),
    "audi": ("sauber", "bmw_sauber", "alfa", "kick_sauber", "stake", "audi"),
    "manor": ("virgin", "marussia", "manor"),
    "williams": ("williams"),
    "ferrari": ("ferrari"),
    "mclaren": ("mclaren"),
    "haas": ("haas"),
}

REF_TO_LINEAGE: dict[str, str] = {}
for _lid, _refs in TEAM_LINEAGES.items():
    for _r in _refs:
        REF_TO_LINEAGE[_r.lower()] = _lid


def ref_to_lineage(ref: str) -> str:
    r = str(ref).lower().strip()
    return REF_TO_LINEAGE.get(r, r)


def build_lineage_map(constructors_df: pd.DataFrame) -> pd.DataFrame:
    df = constructors_df[["constructorId", "constructorRef", "name"]].copy()
    df["lineage_id"] = df["constructorRef"].astype(str).map(ref_to_lineage)
    return df


def lineage_id_by_constructor(constructors_df: pd.DataFrame) -> dict:
    lm = build_lineage_map(constructors_df)
    return dict(zip(lm["constructorId"], lm["lineage_id"]))


def lineage_row_label(season_names: pd.DataFrame) -> str:
    """Row label for heatmaps; ``Old/New`` when the name changed within the window."""
    if season_names.empty:
        return "Unknown"
    ordered = season_names.sort_values("season")
    first = str(ordered.iloc[0]["display_name"])
    last = str(ordered.iloc[-1]["display_name"])
    if first == last and ordered["display_name"].nunique() == 1:
        return first
    if first == last:
        distinct = ordered.drop_duplicates("display_name", keep="first")
        first = str(distinct.iloc[0]["display_name"])
        last = str(distinct.iloc[-1]["display_name"])
    if first == last:
        return first
    return f"{first}/{last}"
