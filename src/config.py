"""
Central configuration for the F1 orthogonal representation pipeline.

Single source of truth for temporal splits, dataset names, and task config.
All removed: FastF1-specific constants, status maps, rolling window settings.
"""

# Single source of truth for train/val/test temporal split.
# Data range: 2000-2023 (from RelBench rel-f1).
TRAIN_YEARS = list(range(2000, 2022))  # 2000..2021
VAL_YEARS = [2022]
TEST_YEARS = [2023]

RELBENCH_DATASET = "rel-f1"
TASK_NAME = "driver-top3"
