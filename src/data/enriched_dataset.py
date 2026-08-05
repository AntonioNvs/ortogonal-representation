"""RelBench-compatible ``Dataset`` subclass serving the enriched rel-f1
database built by ``build_enriched_db.py``.

Design notes (see the plan for the full discussion):

- ``get_db`` is overridden (not just ``make_db``) so that the mandatory
  ``Database.reindex_pkeys_and_fkeys()`` call the base class performs on a
  from-scratch build never runs against our data -- see
  ``build_enriched_db``'s module docstring for why that matters. In normal
  operation the on-disk cache at ``<cache_dir>/db`` is populated ahead of
  time by the pipeline, so ``make_db`` is never actually invoked; it exists
  only to fail loudly and helpfully if someone forgets to run the pipeline
  first.
- Year windowing (``min_year``/``max_year``) happens *inside* ``get_db``,
  before ``AutoCompleteTask`` ever touches the database. This sidesteps a
  real conflict: RelBench's ``AutoCompleteTask`` stashes removed target
  columns keyed by the entity table's primary key, and any pkey remapping
  done *after* that point (like the legacy ``filter_db_by_years`` used to
  do) would silently desync the two. Doing the windowing first means the
  task always sees an already-final, self-consistent id space.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from relbench.base import Database, Dataset
from relbench.datasets import register_dataset

from .db_utils import filter_db_by_years

DEFAULT_CACHE_DIR = "data/enriched/rel-f1"
DEFAULT_NAME = "rel-f1-enriched"

# Extended-mode defaults: wide enough to cover essentially the whole modern
# aero/hybrid era while leaving a meaningful test tail in the newly-added
# 2024-2026 data. Overridable via the dataset constructor / cfg.
DEFAULT_MIN_YEAR = 2000
DEFAULT_MAX_YEAR = 2026
DEFAULT_VAL_TIMESTAMP = pd.Timestamp("2022-01-01")
DEFAULT_TEST_TIMESTAMP = pd.Timestamp("2024-01-01")


class EnrichedF1Dataset(Dataset):
    """rel-f1, enriched with 2023 R13 through the latest completed round of
    2026 (see ``manifest.json`` next to the db for exact provenance)."""

    def __init__(
        self,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        min_year: Optional[int] = DEFAULT_MIN_YEAR,
        max_year: Optional[int] = DEFAULT_MAX_YEAR,
        val_timestamp: pd.Timestamp = DEFAULT_VAL_TIMESTAMP,
        test_timestamp: pd.Timestamp = DEFAULT_TEST_TIMESTAMP,
    ):
        super().__init__(cache_dir=cache_dir)
        self.min_year = min_year
        self.max_year = max_year
        self.val_timestamp = pd.Timestamp(val_timestamp)
        self.test_timestamp = pd.Timestamp(test_timestamp)

    def make_db(self) -> Database:
        raise RuntimeError(
            f"No enriched rel-f1 database found at '{self.cache_dir}/db'. "
            "Run `python -m src.data.pipeline build` (see src/data/pipeline.py) "
            "before instantiating EnrichedF1Dataset."
        )

    @lru_cache(maxsize=None)
    def get_db(self, upto_test_timestamp: bool = True) -> Database:
        db_path = f"{self.cache_dir}/db"
        if self.cache_dir and Path(db_path).exists() and any(Path(db_path).iterdir()):
            db = Database.load(db_path)
        else:
            # Only ever hit for a genuinely from-scratch build; make_db()
            # always raises (see above), matching the fail-fast contract.
            db = self.make_db()

        if self.min_year is not None or self.max_year is not None:
            db = filter_db_by_years(
                db,
                self.min_year if self.min_year is not None else 1950,
                self.max_year if self.max_year is not None else 2100,
            )

        if upto_test_timestamp:
            db = db.upto(self.test_timestamp)

        self.validate_and_correct_db(db)

        if self.target_col:
            db = self.get_modified_db(db)

        return db


def register_enriched_dataset(name: str = DEFAULT_NAME, cache_dir: str = DEFAULT_CACHE_DIR, **kwargs) -> None:
    """Registers EnrichedF1Dataset with relbench's dataset registry under
    ``name``, so the standard ``get_dataset(name)`` / ``get_task(name, ...)``
    machinery works transparently. Safe to call multiple times.

    ``register_dataset`` would otherwise default ``cache_dir`` to somewhere
    under ``~/.cache/relbench/`` -- we override it to point at the repo's
    own ``data/enriched/rel-f1`` so there is a single, obvious location for
    the enriched database (the same one ``build_enriched_db.py`` writes to).
    """
    register_dataset(name, EnrichedF1Dataset, cache_dir=cache_dir, **kwargs)
