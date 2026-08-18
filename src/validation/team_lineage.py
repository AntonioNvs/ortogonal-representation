"""Deterministic constructor "lineage" mapping (acquisitions & renames).

A Formula 1 team often changes its constructor entry between seasons without
actually ceasing to exist: it gets bought (Jaguar -> Red Bull), sold (Jordan ->
Midland -> Spyker -> Force India -> Racing Point -> Aston Martin), or rebranded
(Sauber -> BMW Sauber -> Alfa Romeo -> Audi). In the RelBench/Ergast data these
are *different* ``constructorId`` values, so a naive per-constructor analysis
treats each as a brand-new team: their smoothed points share (and hence tier /
rank) resets at the boundary even though the organisation is the same.

This module maps those entries back into a single *lineage* so that (a) a team's
rank/score carries across the rebrand and (b) plots can show the continuity as
``X/Y`` (previous name / new name).

Keying is by ``constructorRef`` (Ergast slug), which is stable within an entry
and unambiguous (unlike ``name``, which carries sponsor prefixes and collides on
the "Lotus" name dispute). Any ref not listed below is treated as its own
standalone lineage, so an unknown/typo'd slug degrades to "no merge" rather than
a wrong merge.
"""

from __future__ import annotations

import pandas as pd

# lineage_id -> chronologically ordered constructorRef chain. ``lineage_id`` is
# the current/latest slug for readability; the chain is the full organisation
# history. Entries before MIN_YEAR=2000 are harmless — they simply never appear.
TEAM_LINEAGES: dict[str, tuple[str, ...]] = {
    # Stewart (1997-99) -> Jaguar (2000-04) -> Red Bull (2005+)
    "red_bull": ("stewart", "jaguar", "red_bull"),
    # Tyrrell (pre-98) -> BAR (1999-05) -> Honda (2006-08) -> Brawn (2009) -> Mercedes (2010+)
    "mercedes": ("tyrrell", "bar", "honda", "brawn", "mercedes"),
    # Jordan (1991-05) -> Midland (2006) -> Spyker (2007) -> Force India (2008-18)
    # -> Racing Point (2019-20) -> Aston Martin (2021+)
    "aston_martin": ("jordan", "midland", "spyker", "force_india", "racing_point", "aston_martin"),
    # Minardi (1985-05) -> Toro Rosso (2006-19) -> AlphaTauri (2020-23) -> RB (2024+)
    "rb": ("minardi", "toro_rosso", "alphatauri", "rb"),
    # Benetton (1986-01) -> Renault (2002-11) -> Lotus F1 (2012-15) -> Renault (2016-20)
    # -> Alpine (2021+). ``renault`` is a single ref covering both Renault eras.
    "alpine": ("benetton", "renault", "lotus_f1", "alpine"),
    # Team Lotus / Lotus Racing (2010-11) -> Caterham (2012-14). NOT the Enstone
    # "Lotus F1" above — the two Lotus entries are unrelated teams.
    "caterham": ("lotus_racing", "team_lotus", "caterham"),
    # Sauber (1993-05, 2010-18) -> BMW Sauber (2006-09) -> Alfa Romeo (2019-23) -> Audi (2026+).
    # ``sauber`` is a single ref covering both the pre-BMW and post-BMW eras.
    "audi": ("sauber", "bmw_sauber", "alfa", "audi"),
    # Virgin (2010-11) -> Marussia (2012-15) -> Manor (2016)
    "manor": ("virgin", "marussia", "manor"),
}

# Precomputed ref -> lineage_id lookup.
REF_TO_LINEAGE: dict[str, str] = {}
for _lid, _refs in TEAM_LINEAGES.items():
    for _r in _refs:
        REF_TO_LINEAGE[_r.lower()] = _lid


def ref_to_lineage(ref: str) -> str:
    """Map a ``constructorRef`` to its lineage id (standalone refs map to themselves)."""
    r = str(ref).lower().strip()
    return REF_TO_LINEAGE.get(r, r)


def build_lineage_map(constructors_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate the constructors table with ``lineage_id``.

    Returns ``[constructorId, constructorRef, name, lineage_id]``.
    """
    df = constructors_df[["constructorId", "constructorRef", "name"]].copy()
    df["lineage_id"] = df["constructorRef"].astype(str).map(ref_to_lineage)
    return df


def lineage_id_by_constructor(constructors_df: pd.DataFrame) -> dict:
    """``constructorId -> lineage_id`` mapping (convenience for ``compute_team_tiers``)."""
    lm = build_lineage_map(constructors_df)
    return dict(zip(lm["constructorId"], lm["lineage_id"]))


def lineage_label(
    rows: pd.DataFrame,
    name_col: str = "name",
    season_col: str = "season",
    sep: str = "/",
) -> str:
    """Build the ``X/Y`` display label for a lineage from its observed names.

    ``rows`` are the (name, season) pairs for one lineage within the plotted
    window. Distinct names are joined with ``sep`` in chronological order of
    first appearance; a team that never rebranded yields just its own name.
    """
    if rows.empty:
        return ""
    names = (
        rows.sort_values(season_col)
        .drop_duplicates(name_col)[name_col]
        .astype(str)
        .tolist()
    )
    deduped: list[str] = []
    for n in names:
        if not deduped or deduped[-1] != n:
            deduped.append(n)
    return sep.join(deduped)
