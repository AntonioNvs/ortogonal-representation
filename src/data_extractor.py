"""
High-level data extraction orchestrator.

Coordinates driver features (tabular), team (RelBench), and track (hybrid)
extraction modules, builds unified ID mappings, and produces a final
JSON index that ties everything together.
"""
import json
import logging
import os
import sys

import config as cfg
from id_mapping import build_all_mappings, save_mappings

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with timestamped console output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_extraction(
    year: int,
    modules: list[str],
    output_dir: str | None = None,
    cache_dir: str | None = None,
) -> None:
    """
    Run the data extraction pipeline for a given season.

    Parameters
    ----------
    year : int
        Season year to extract (e.g., 2023).
    modules : list[str]
        Which extraction modules to run.
        Valid values: ``'all'``, ``'driver'``, ``'team'``, ``'track'``, ``'dataset'``.
    output_dir : str, optional
        Root directory for generated files (default: ``./output``).
    cache_dir : str, optional
        FastF1 cache directory (default: ``./cache``).
    """
    base_output_dir = output_dir or str(cfg.OUTPUT_DIR)
    data_dir = os.path.join(base_output_dir, "data")
    dataset_dir = os.path.join(base_output_dir, "dataset")
    cache_dir = cache_dir or str(cfg.CACHE_DIR)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    if "all" in modules:
        modules = ["team", "track", "driver", "dataset"]

    logger.info("=" * 64)
    logger.info(f"  F1 DATA EXTRACTION — SEASON {year}")
    logger.info(f"  Modules: {modules}")
    logger.info(f"  Data Output: {data_dir}")
    logger.info(f"  Dataset Output: {dataset_dir}")
    logger.info(f"  Cache:   {cache_dir}")
    logger.info("=" * 64)

    dataset = None
    if any(m in modules for m in ("team", "track")):
        from extractors.team import load_relbench_dataset

        dataset = load_relbench_dataset()

    driver_map: dict = {}
    circuit_map: dict = {}

    if dataset is not None:
        driver_map, circuit_map, _schedule = build_all_mappings(
            dataset, year, cache_dir
        )
        save_mappings(driver_map, circuit_map, data_dir, year)

    if "team" in modules:
        logger.info("")
        logger.info("=" * 64)
        logger.info(f"  MODULE: TEAM DATA (RelBench) — {year}")
        logger.info("=" * 64)

        from extractors.team import extract_all_tables, extract_task_labels

        extract_all_tables(dataset, data_dir, year)
        extract_task_labels(dataset, data_dir)

    if "track" in modules:
        logger.info("")
        logger.info("=" * 64)
        logger.info(f"  MODULE: TRACK FEATURES — {year}")
        logger.info("=" * 64)

        from extractors.track import extract_season_tracks

        extract_season_tracks(dataset, year, data_dir, cache_dir)

    if "driver" in modules:
        logger.info("")
        logger.info("=" * 64)
        logger.info(f"  MODULE: DRIVER FEATURES — {year}")
        logger.info("=" * 64)

        from extractors.driver import extract_driver_features

        extract_driver_features(data_dir)

    if "dataset" in modules:
        logger.info("")
        logger.info("=" * 64)
        logger.info(f"  MODULE: DATASET BUILDER")
        logger.info("=" * 64)

        from extractors.dataset_builder import build_dataset

        build_dataset(data_dir, dataset_dir)

    build_unified_index(year, data_dir, dataset_dir, driver_map, circuit_map)

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"  EXTRACTION COMPLETE — {year}")
    logger.info(f"  Data Output directory: {data_dir}")
    logger.info("=" * 64)


def build_unified_index(
    year: int,
    data_dir: str,
    dataset_dir: str,
    driver_map: dict | None = None,
    circuit_map: dict | None = None,
) -> None:
    """
    Build and persist the unified JSON index.

    The index ties together driver/circuit mappings, pipeline configuration,
    and the file layout so the downstream PyTorch pipeline knows exactly
    where to find everything.
    """
    unified_dir = os.path.join(data_dir, "unified")
    os.makedirs(unified_dir, exist_ok=True)

    driver_path = os.path.join(data_dir, "driver", "driver_features_per_race.csv")
    instances_path = os.path.join(dataset_dir, "instances.csv")

    index = {
        "year": year,
        "config": {
            "task": cfg.TASK_NAME,
            "rolling_window": cfg.ROLLING_WINDOW,
        },
        "drivers": driver_map or {},
        "circuits": circuit_map or {},
        "driver_features_file": "driver/driver_features_per_race.csv" if os.path.exists(driver_path) else None,
        "instances_file": "dataset/instances.csv" if os.path.exists(instances_path) else None,
    }

    json_path = os.path.join(unified_dir, f"index_{year}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Unified index -> {json_path}")
