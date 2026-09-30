"""Anchored logistic calibration to [0, 10] using training-only statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CalibrationParams:
    """Parameters for skill_0_10 = 10 * sigmoid(alpha * z)."""

    mu: float
    sigma: float
    alpha: float = 1.0

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "alpha": self.alpha}


def fit_calibration(
    raw_scores: Iterable[float],
    *,
    alpha: float = 1.0,
    min_sigma: float = 1e-6,
) -> CalibrationParams:
    """Fit location/scale on training raw scores only."""
    arr = np.asarray(list(raw_scores), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return CalibrationParams(mu=0.0, sigma=1.0, alpha=alpha)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else min_sigma
    if sigma < min_sigma:
        sigma = min_sigma
    return CalibrationParams(mu=mu, sigma=sigma, alpha=alpha)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_to_0_10(
    raw_scores: np.ndarray | Iterable[float],
    params: CalibrationParams,
) -> np.ndarray:
    """Map raw higher-is-better scores to [0, 10] via anchored logistic."""
    z = (np.asarray(raw_scores, dtype=float) - params.mu) / params.sigma
    return 10.0 * _sigmoid(params.alpha * z)


def calibrate_interval(
    lo: float,
    hi: float,
    params: CalibrationParams,
) -> tuple[float, float]:
    """Propagate uncertainty through monotone calibration (order-preserving)."""
    a, b = calibrate_to_0_10(np.array([lo, hi]), params)
    return float(min(a, b)), float(max(a, b))


def distribution_diagnostics(scores_0_10: np.ndarray) -> dict:
    """Score spread / concentration diagnostics for validation reports."""
    x = np.asarray(scores_0_10, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "iqr": float("nan"),
            "central_mass_4_6": float("nan"),
            "saturation_low": float("nan"),
            "saturation_high": float("nan"),
        }
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "iqr": float(q75 - q25),
        "central_mass_4_6": float(np.mean((x >= 4.0) & (x <= 6.0))),
        "saturation_low": float(np.mean(x <= 0.5)),
        "saturation_high": float(np.mean(x >= 9.5)),
    }
