"""
RelBench rel-f1 data extraction for the HeteroGraphSAGE team encoder.

Exports all relational tables from the rel-f1 dataset as CSV files,
filtered to the hybrid era (2014+).  Also exports the train/val/test
label splits for the ``driver-top3`` task.

Note: ``get_db(upto_test_timestamp=False)`` is used via the shared
``get_table_df`` helper so all data up to 2023 is accessible.
"""
import logging
import os

import pandas as pd

import config as cfg
from id_mapping import get_table_df, list_tables

logger = logging.getLogger(__name__)


def load_relbench_dataset():
    """
    Load the rel-f1 dataset from RelBench.

    The first call downloads the dataset (~5 MB); subsequent calls use
    a local cache managed by RelBench internally.
    """
    try:
        from relbench.datasets import get_dataset
    except ImportError:
        from relbench.base import get_dataset

    logger.info(f"Loading RelBench dataset '{cfg.RELBENCH_DATASET}'...")
    dataset = get_dataset(cfg.RELBENCH_DATASET)
    logger.info("RelBench dataset loaded.")
    return dataset


def extract_all_tables(dataset, output_dir: str, year: int | None = None) -> dict:
    """
    Export every rel-f1 table to CSV, optionally filtered to the hybrid era.

    Filtering strategy (applied when *year* is provided):
    - Tables with a ``year`` column: keep rows with year >= HYBRID_ERA_START.
    - Tables with a ``date`` column: parse and filter by year.
    - Tables with a ``raceId`` column (and no year/date): join through the
      ``races`` table to filter by year.
    - Tables with none of the above: exported in full (reference tables
      like ``circuits`` or ``constructors``).

    Uses ``get_db(upto_test_timestamp=False)`` internally (via ``get_table_df``)
    so data from 2014–2023 is accessible.

    Returns
    -------
    dict
        {table_name: number_of_rows_exported}
    """
    table_dir = os.path.join(output_dir, "team")
    os.makedirs(table_dir, exist_ok=True)

    available = list_tables(dataset)
    logger.info(f"Available rel-f1 tables: {available}")

    races_df = None
    hybrid_race_ids: set = set()
    if year:
        try:
            races_df = get_table_df(dataset, "races")
            hybrid_race_ids = set(
                races_df[races_df["year"] >= cfg.HYBRID_ERA_START]["raceId"]
            )
        except Exception:
            logger.warning("Cannot load 'races' table for temporal filtering.")

    exported: dict = {}

    for table_name in available:
        try:
            df = get_table_df(dataset, table_name)
        except Exception as e:
            logger.warning(f"Cannot read table '{table_name}': {e}")
            continue

        filtered = df

        if year:
            if "year" in df.columns:
                filtered = df[df["year"] >= cfg.HYBRID_ERA_START]
            elif "date" in df.columns:
                try:
                    dates = pd.to_datetime(df["date"], errors="coerce")
                    filtered = df[dates.dt.year >= cfg.HYBRID_ERA_START]
                except Exception:
                    pass
            elif "raceId" in df.columns and table_name != "races" and hybrid_race_ids:
                filtered = df[df["raceId"].isin(hybrid_race_ids)]

        csv_path = os.path.join(table_dir, f"{table_name}.csv")
        filtered.to_csv(csv_path, index=False)
        exported[table_name] = len(filtered)

        logger.info(f"  {table_name}: {len(filtered)} rows → {csv_path}")

    return exported


def extract_task_labels(dataset, output_dir: str) -> None:
    """
    Export train / val / test label splits for the configured task.

    These DataFrames contain (entity_id, seed_time, target_label) and are
    used to define the prediction target during model training.
    """
    task_dir = os.path.join(output_dir, "task")
    os.makedirs(task_dir, exist_ok=True)

    logger.info(f"Loading task '{cfg.TASK_NAME}'...")
    try:
        from relbench.tasks import get_task
    except ImportError:
        from relbench.base import get_task

    task = get_task(cfg.RELBENCH_DATASET, cfg.TASK_NAME, download=True)

    for split_label in ["train", "val", "test"]:
        try:
            table = task.get_table(split_label)
        except Exception as e:
            logger.warning(f"Task split '{split_label}' not found: {e}")
            continue

        df = table.df if hasattr(table, "df") else pd.DataFrame(table)

        csv_path = os.path.join(task_dir, f"{cfg.TASK_NAME}_{split_label}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"  {split_label}: {len(df)} rows → {csv_path}")

    logger.info(f"Task labels exported to {task_dir}")
