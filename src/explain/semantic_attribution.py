"""Semantic attribution via Shapley variance decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from skill.decomposition import shapley_variance_shares


@dataclass
class SemanticAttribution:
    driver: float
    constructor: float
    context: float
    grid: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "driver": self.driver,
            "constructor": self.constructor,
            "context": self.context,
            "grid": self.grid,
            "total": self.total,
        }


def semantic_attribution_from_race_df(race_df: pd.DataFrame) -> SemanticAttribution:
    """Mean Shapley variance shares from race-level contributions."""
    shares = shapley_variance_shares(
        race_df["contrib_driver"].to_numpy(),
        race_df["contrib_constructor"].to_numpy(),
        race_df["contrib_context"].to_numpy(),
    )
    return SemanticAttribution(
        driver=shares["driver"],
        constructor=shares["constructor"],
        context=shares["context"],
        grid=shares.get("context", 0.0),
        total=1.0,
    )


def semantic_attribution(*args, **kwargs):
    """Legacy entrypoint — pass a race DataFrame as first argument."""
    if args and isinstance(args[0], pd.DataFrame):
        return semantic_attribution_from_race_df(args[0])
    raise TypeError("semantic_attribution now requires a race-level DataFrame export")
