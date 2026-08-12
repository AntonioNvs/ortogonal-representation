"""
Central configuration for the F1 orthogonal representation pipeline.

Single source of truth for the data source, task, temporal splits and
dataset names. All removed: FastF1-specific constants, status maps, rolling
window settings.

Target task
-----------
The model predicts a driver's outcome for a race via a RelBench
``AutoCompleteTask`` (see ``src/data/tasks.py``), which removes the target
(and every other outcome column that would leak it) from the entity table
before the graph is built. Three targets are registered and selectable via
``TASK_NAME``:

- "results-position" (default): official rel-f1 semantics, target =
  ``position`` (NaN for non-classified/DNF entries, ~22% of rows). Directly
  comparable to the RelBench leaderboard in "benchmark" split mode.
- "results-positionorder": target = ``positionOrder`` (never null, keeps
  DNFs -- useful for the driver-vs-constructor ranking application, where a
  reliability signal matters).
- "results-points": target = ``points`` (the originally requested target;
  kept for comparison despite being zero-inflated, see the plan).

Split modes
-----------
- "benchmark": pristine RelBench ``rel-f1`` (frozen at mid-2023) with the
  official val/test timestamps (2005 / 2010) -- produces numbers comparable
  to the RelBench leaderboard.
- "extended" (default): the enriched database (2000-2026, see
  ``src/data/build_enriched_db.py``) with its own, more recent val/test
  timestamps, to actually make use of the 2023-2026 data added by the
  enrichment pipeline.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Data source / task selection
# ---------------------------------------------------------------------------

DATA_SOURCE = "enriched"  # "enriched" | "pristine"
SPLIT_MODE = "extended"  # "extended" | "benchmark"
TASK_NAME = "results-position"  # "results-position" | "results-positionorder" | "results-points"

TARGET_COL_BY_TASK = {
    "results-position": "position",
    "results-positionorder": "positionOrder",
    "results-points": "points",
}
TARGET_COL = TARGET_COL_BY_TASK[TASK_NAME]

RELBENCH_DATASET = "rel-f1"
ENRICHED_DATASET_NAME = "rel-f1-enriched"
ENRICHED_DB_DIR = "data/enriched/rel-f1"

# ---------------------------------------------------------------------------
# Temporal windows
# ---------------------------------------------------------------------------

MIN_YEAR = 2000
MAX_YEAR = 2026

# Official rel-f1 benchmark timestamps (relbench.datasets.f1.F1Dataset).
BENCHMARK_MIN_YEAR = 1950
BENCHMARK_MAX_YEAR = 2023
BENCHMARK_VAL_TIMESTAMP = pd.Timestamp("2005-01-01")
BENCHMARK_TEST_TIMESTAMP = pd.Timestamp("2010-01-01")

# Extended-mode timestamps: leaves a meaningful validation window in
# 2022-2023 and a test window entirely inside the newly-enriched
# 2024-2026 data.
EXTENDED_VAL_TIMESTAMP = pd.Timestamp("2022-01-01")
EXTENDED_TEST_TIMESTAMP = pd.Timestamp("2024-01-01")


def active_dataset_name(split_mode: str = None) -> str:
    split_mode = split_mode or SPLIT_MODE
    return ENRICHED_DATASET_NAME if split_mode == "extended" else RELBENCH_DATASET


def active_window(split_mode: str = None):
    """Returns (min_year, max_year, val_timestamp, test_timestamp) for the
    given split mode (defaults to cfg.SPLIT_MODE)."""
    split_mode = split_mode or SPLIT_MODE
    if split_mode == "extended":
        return MIN_YEAR, MAX_YEAR, EXTENDED_VAL_TIMESTAMP, EXTENDED_TEST_TIMESTAMP
    return BENCHMARK_MIN_YEAR, BENCHMARK_MAX_YEAR, BENCHMARK_VAL_TIMESTAMP, BENCHMARK_TEST_TIMESTAMP


def _years_from_timestamps(min_year, max_year, val_ts, test_ts):
    train_years = list(range(min_year, val_ts.year))
    val_years = list(range(val_ts.year, test_ts.year))
    test_years = list(range(test_ts.year, max_year + 1))
    return train_years, val_years, test_years


_min_year, _max_year, _val_ts, _test_ts = active_window()

# Derived from the timestamps above (single source of truth), but kept as
# plain year lists because add_edge_year_masks() and temporal_curve_runner.py
# key their leakage-prevention masks off race year, not exact timestamps.
TRAIN_YEARS, VAL_YEARS, TEST_YEARS = _years_from_timestamps(_min_year, _max_year, _val_ts, _test_ts)

# Walk-forward temporal curve experiment defaults.
TEMPORAL_CURVE_TARGET_YEARS = list(TEST_YEARS)
TEMPORAL_CURVE_MODEL = "high"  # lambda_orthogonal = 1.0

# Default GPU for all training and experiment scripts.
DEFAULT_GPU_ID = 7

# ---------------------------------------------------------------------------
# Kalman-GNN configuration
# ---------------------------------------------------------------------------

# Sliding window
KALMAN_WINDOW_SIZE = 20       # K: number of recent races in GNN sliding window
KALMAN_WARMUP_RACES = 20      # minimum races before computing losses

# Model dimensions
KALMAN_STATE_DIM = 32         # dimension of driver/constructor state embedding v
KALMAN_MSG_DIM = 8            # dimension of GNN output h_r

# Training
KALMAN_LR = 0.001             # learning rate
KALMAN_WEIGHT_DECAY = 1e-5    # AdamW weight decay
KALMAN_EPOCHS = 5             # epochs (fewer needed since sequence is ~500 races)
KALMAN_ACCUMULATION_STEPS = 4 # gradient accumulation across N races
KALMAN_GRAD_CLIP = 1.0        # gradient norm clipping

# Loss weights
KALMAN_LAMBDA_PRED = 1.0      # beat-teammate BCE
KALMAN_LAMBDA_SMOOTH = 0.1    # temporal smoothness ||v_r - v_{r-1}||²
KALMAN_LAMBDA_CONTRAST = 0.05 # InfoNCE temporal contrast
KALMAN_LAMBDA_SKILL = 0.05    # skill forward consistency

# Contrastive loss
KALMAN_CONTRAST_GAP_MIN = 1   # minimum race gap for positive pairs
KALMAN_CONTRAST_GAP_MAX = 5   # maximum race gap for positive pairs
KALMAN_CONTRAST_TEMP = 0.1    # InfoNCE temperature
KALMAN_CONTRAST_EVERY = 5     # compute contrastive loss every N races

# Smoke test overrides
KALMAN_SMOKE_WINDOW_SIZE = 5
KALMAN_SMOKE_STATE_DIM = 8
KALMAN_SMOKE_MSG_DIM = 4
KALMAN_SMOKE_EPOCHS = 1
KALMAN_SMOKE_WARMUP = 5

# ---------------------------------------------------------------------------
# Career validation framework (deterministic team tiers + forward outcome)
# ---------------------------------------------------------------------------

# Window (in seasons) of the trailing moving average used to smooth a team's
# points share before tier assignment. Trailing (data <= T) keeps it leak-free.
TIER_WINDOW = 3

# Fixed per-season tier proportions (share of the grid). The remainder of the
# integer split goes to tier B. Kept here so the runner can pass them through.
TIER_S_FRAC = 0.30
TIER_A_FRAC = 0.35

# Forward horizon (in seasons): a driver's career outcome at time T is the mean
# tier of the teams they drive for in seasons T+1 .. T+TIER_HORIZON.
TIER_HORIZON = 3

# Mapping from tier label to a scalar score for ranking/correlation.
TIER_TO_SCORE = {"S": 3, "A": 2, "B": 1}

# Earliest season considered by the career-validation framework. The upper
# bound is derived from the data (currently ~2026).
CAREER_VALIDATION_MIN_YEAR = 2000

# Default output directory for career-validation artifacts.
CAREER_VALIDATION_OUTPUT_DIR = "output/career_validation"
