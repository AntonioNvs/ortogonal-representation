"""Registers RelBench tasks for the enriched rel-f1 dataset.

``results-position`` mirrors the official ``rel-f1`` task byte-for-byte in
semantics (same entity table, target column and ``remove_columns``), so
metrics computed on the ``benchmark`` split mode (pristine data + official
timestamps) are directly comparable to the RelBench leaderboard. Two
additional variants exist for the application layer (see the plan for the
zero-inflation / DNF-dropping trade-offs that motivated them):

- ``results-positionorder``: target is ``positionOrder`` (never null, keeps
  DNFs in the training signal) instead of ``position`` (~22% null, dropped
  by ``AutoCompleteTask.make_table``'s ``dropna``).
- ``results-points``: target is ``points`` (the user's original request),
  kept available for empirical comparison despite its zero-inflation.

All three are registered as :class:`EfficientAutoCompleteTask`, a drop-in
subclass of ``relbench.base.AutoCompleteTask`` that avoids materializing a
``pd.date_range`` with one entry per second across the entire split window
(see its docstring) -- necessary once splits can span decades, which is the
whole point of the enriched, multi-year dataset.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from relbench.base import Database, Table, TaskType
from relbench.base.task_autocomplete import AutoCompleteTask
from relbench.tasks import register_task

from .enriched_dataset import DEFAULT_NAME

RESULTS_OUTCOME_COLUMNS = [
    "statusId", "position", "positionOrder", "points", "laps",
    "milliseconds", "fastestLap", "rank",
]

# Qualifying's only outcome column. ``number`` (car number) and ``date`` are
# legitimate pre-race features and are kept; ``position`` is the target.
QUALIFYING_OUTCOME_COLUMNS = ["position"]


class EfficientAutoCompleteTask(AutoCompleteTask):
    """Same semantics as ``relbench.base.AutoCompleteTask``, but computes the
    split's time window bounds directly instead of building the full
    ``pd.date_range(freq='-1s')`` the base implementation uses only to read
    off ``.min()``/``.max()``. For a multi-decade window that intermediate
    array can have billions of entries; for the enriched dataset (whose
    entire point is to span more years than the frozen benchmark) that is a
    real memory/latency problem, not a hypothetical one.
    """

    def _get_table(self, split: str) -> Table:
        db = self.dataset.get_db(upto_test_timestamp=split != "test")

        if split == "train":
            start = self.dataset.val_timestamp - self.timedelta
            end = db.min_timestamp
        elif split == "val":
            if self.dataset.val_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "val timestamp + timedelta is larger than max timestamp! "
                    "This would cause val labels to be generated with "
                    "insufficient aggregation time."
                )
            start = self.dataset.test_timestamp - self.timedelta
            end = self.dataset.val_timestamp
        elif split == "test":
            if self.dataset.test_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )
            start = db.max_timestamp
            end = self.dataset.test_timestamp
        else:
            raise ValueError(f"Unknown split: {split}")

        n_seconds = abs((start - end) / self.timedelta) + 1
        if split == "train" and n_seconds < 3:
            raise RuntimeError(f"The number of training time frames is too few. ({n_seconds} given)")

        min_timestamp, max_timestamp = (start, end) if start <= end else (end, start)
        table = self._make_table_range(db, min_timestamp, max_timestamp)
        table = self.filter_dangling_entities(table)
        return table

    def _make_table_range(self, db: Database, min_timestamp: pd.Timestamp, max_timestamp: pd.Timestamp) -> Table:
        entity_table = db.table_dict[self.entity_table].df  # noqa: F841 (used by duckdb.sql via local scope)
        entity_table_removed_cols = db.table_dict[self.entity_table].removed_cols  # noqa: F841
        entity_col = db.table_dict[self.entity_table].pkey_col

        df = duckdb.sql(f"""
            SELECT
                entity_table.{self.time_col},
                entity_table.{entity_col},
                entity_table_removed_cols.{self.target_col}
            FROM entity_table
            LEFT JOIN entity_table_removed_cols
                ON entity_table.{entity_col} = entity_table_removed_cols.{entity_col}
            WHERE entity_table.{self.time_col} > '{min_timestamp}'
              AND entity_table.{self.time_col} <= '{max_timestamp}'
        """).df()

        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            df[self.target_col] = self.transform_target(df[self.target_col])

        df = df.dropna(subset=[self.target_col])

        return Table(
            df=df,
            fkey_col_to_pkey_table={entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


def register_enriched_tasks(dataset_name: str = DEFAULT_NAME, include_official_position: bool = True) -> None:
    """Registers ``results-position`` (official semantics) plus the
    ``results-positionorder`` and ``results-points`` variants against
    ``dataset_name``. ``cache_dir=None`` disables RelBench's own on-disk task
    table cache (which otherwise lives under ``~/.cache/relbench/`` and would
    silently go stale whenever the enrichment pipeline is re-run with more
    rounds); the in-process ``lru_cache`` on ``BaseTask.get_table`` is kept.

    ``include_official_position=False`` is used for the pristine "rel-f1"
    dataset in "benchmark" split mode: RelBench already registers
    "results-position" there as the plain (unmodified) ``AutoCompleteTask``,
    which we deliberately leave untouched to make the "benchmark" split mode
    a zero-risk, byte-for-byte match with the RelBench leaderboard. Only the
    two custom variants (not part of RelBench's own registry) are added.
    """
    if include_official_position:
        register_task(
            dataset_name,
            "results-position",
            EfficientAutoCompleteTask,
            task_type=TaskType.REGRESSION,
            entity_table="results",
            target_col="position",
            remove_columns=[("results", c) for c in RESULTS_OUTCOME_COLUMNS if c != "position"],
            cache_dir=None,
        )
    register_task(
        dataset_name,
        "results-positionorder",
        EfficientAutoCompleteTask,
        task_type=TaskType.REGRESSION,
        entity_table="results",
        target_col="positionOrder",
        remove_columns=[("results", c) for c in RESULTS_OUTCOME_COLUMNS if c != "positionOrder"],
        cache_dir=None,
    )
    register_task(
        dataset_name,
        "results-points",
        EfficientAutoCompleteTask,
        task_type=TaskType.REGRESSION,
        entity_table="results",
        target_col="points",
        remove_columns=[("results", c) for c in RESULTS_OUTCOME_COLUMNS if c != "points"],
        cache_dir=None,
    )
    register_task(
        dataset_name,
        "qualifying-position",
        EfficientAutoCompleteTask,
        task_type=TaskType.REGRESSION,
        entity_table="qualifying",
        target_col="position",
        remove_columns=[("qualifying", c) for c in QUALIFYING_OUTCOME_COLUMNS if c != "position"],
        cache_dir=None,
    )


def register_all(
    enriched_db_dir: str = None,
    min_year: int = None,
    max_year: int = None,
    val_timestamp=None,
    test_timestamp=None,
) -> None:
    """Idempotent one-shot registration for both split modes: the enriched
    dataset/tasks (used by "extended" mode) and the two custom task variants
    layered on top of the pristine "rel-f1" dataset (used by "benchmark"
    mode). Safe to call multiple times (e.g. once per training run).

    ``min_year``/``max_year``/``val_timestamp``/``test_timestamp`` are
    forwarded to ``EnrichedF1Dataset`` so ``src/config.py`` stays the single
    source of truth for the "extended" split window; when omitted, the
    dataset class's own defaults are used.
    """
    from .enriched_dataset import DEFAULT_CACHE_DIR, register_enriched_dataset

    extra_kwargs = {}
    if min_year is not None:
        extra_kwargs["min_year"] = min_year
    if max_year is not None:
        extra_kwargs["max_year"] = max_year
    if val_timestamp is not None:
        extra_kwargs["val_timestamp"] = val_timestamp
    if test_timestamp is not None:
        extra_kwargs["test_timestamp"] = test_timestamp

    register_enriched_dataset(cache_dir=enriched_db_dir or DEFAULT_CACHE_DIR, **extra_kwargs)
    register_enriched_tasks(DEFAULT_NAME, include_official_position=True)
    register_enriched_tasks("rel-f1", include_official_position=False)
