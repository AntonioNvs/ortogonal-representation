"""DB-backed display names for drivers and constructors."""

from __future__ import annotations

import pandas as pd

DRIVER_ALIASES: dict[str, str] = {
    "verstappen": "max_verstappen",
    "max": "max_verstappen",
    "norris": "lando_norris",
    "lando": "lando_norris",
    "hamilton": "lewis_hamilton",
    "lewis": "lewis_hamilton",
    "leclerc": "charles_leclerc",
    "charles": "charles_leclerc",
    "sainz": "carlos_sainz",
    "carlos": "carlos_sainz",
    "russell": "george_russell",
    "george": "george_russell",
    "alonso": "fernando_alonso",
    "fernando": "fernando_alonso",
    "piastri": "oscar_piastri",
    "oscar": "oscar_piastri",
    "perez": "sergio_perez",
    "sergio": "sergio_perez",
    "bottas": "valtteri_bottas",
    "gasly": "pierre_gasly",
    "ocon": "esteban_ocon",
    "stroll": "lance_stroll",
    "albon": "alexander_albon",
    "tsunoda": "yuki_tsunoda",
    "hulkenberg": "nico_hulkenberg",
    "ricciardo": "daniel_ricciardo",
    "magnussen": "kevin_magnussen",
}


def format_driver_name(forename: str | None, surname: str | None, driver_ref: str | None = None) -> str:
    fn = str(forename or "").strip()
    sn = str(surname or "").strip()
    if fn and sn:
        return f"{fn} {sn}"
    if sn:
        return sn.title()
    if driver_ref:
        parts = str(driver_ref).split("_")
        return " ".join(p.capitalize() for p in parts)
    return "Unknown Driver"


def format_constructor_name(name: str | None, constructor_ref: str | None = None) -> str:
    if name and str(name).strip():
        return str(name).strip()
    if constructor_ref:
        return str(constructor_ref).replace("_", " ").title()
    return "Unknown Team"


def build_driver_name_map(drivers_df: pd.DataFrame) -> dict[int, str]:
    out = {}
    for row in drivers_df.itertuples(index=False):
        out[int(row.driverId)] = format_driver_name(
            getattr(row, "forename", None),
            getattr(row, "surname", None),
            getattr(row, "driverRef", None),
        )
    return out


def build_constructor_name_map(constructors_df: pd.DataFrame) -> dict[int, str]:
    out = {}
    for row in constructors_df.itertuples(index=False):
        out[int(row.constructorId)] = format_constructor_name(
            getattr(row, "name", None),
            getattr(row, "constructorRef", None),
        )
    return out


def resolve_driver_ref(query: str, refs: dict[str, int]) -> tuple[int | None, str]:
    key = str(query).lower().strip()
    if key.isdigit():
        return int(key), key
    if key in refs:
        return refs[key], key
    alias = DRIVER_ALIASES.get(key)
    if alias and alias in refs:
        return refs[alias], alias
    suffix = [r for r in refs if r.endswith(f"_{key}") or r == key]
    if len(suffix) == 1:
        return refs[suffix[0]], suffix[0]
    if len(suffix) > 1:
        canonical = sorted(suffix)[0]
        return refs[canonical], canonical
    sub = [r for r in refs if key in r]
    if len(sub) == 1:
        return refs[sub[0]], sub[0]
    return None, key
