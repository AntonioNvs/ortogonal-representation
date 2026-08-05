"""Shared, dependency-free helpers for manipulating a RelBench ``Database``
in place. Split out of ``train.py`` so both the training script and
``EnrichedF1Dataset`` can use the exact same year-windowing logic without a
circular import.
"""

from __future__ import annotations

import numpy as np
from relbench.base import Database


def filter_db_by_years(db: Database, min_year: int, max_year: int) -> Database:
    """Filter a RelBench database to races in [min_year, max_year] (in place),
    and remap primary/foreign keys to maintain contiguous integer ranges
    starting at 0.

    Row order is preserved (only rows are dropped), which is what keeps the
    "sorted by time_col, pkey == row position" invariant that
    ``Dataset.validate_and_correct_db``/``Table.upto`` rely on: since the
    input is already time-sorted and we only ever remove a prefix and/or a
    suffix of years, what remains is still a time-sorted contiguous block.
    """
    races_df = db.table_dict["races"].df.copy()
    races_df = races_df[(races_df["year"] >= min_year) & (races_df["year"] <= max_year)]
    valid_race_ids = set(races_df["raceId"].unique())

    for name, table in list(db.table_dict.items()):
        if name == "races":
            table.df = races_df
        elif "raceId" in table.df.columns:
            table.df = table.df[table.df["raceId"].isin(valid_race_ids)].copy()

    mappings = {}
    for name, table in db.table_dict.items():
        pkey = table.pkey_col
        if pkey is not None:
            old_keys = table.df[pkey].values
            mapping = {old_key: i for i, old_key in enumerate(old_keys)}
            mappings[name] = mapping
            table.df[pkey] = np.arange(len(table.df))

    for name, table in db.table_dict.items():
        for fkey_col, pkey_table in table.fkey_col_to_pkey_table.items():
            if pkey_table in mappings:
                mapping = mappings[pkey_table]
                table.df[fkey_col] = table.df[fkey_col].map(mapping)
                if table.df[fkey_col].isnull().any():
                    table.df = table.df[table.df[fkey_col].notnull()].copy()
                table.df[fkey_col] = table.df[fkey_col].astype(np.int64)

    return db
