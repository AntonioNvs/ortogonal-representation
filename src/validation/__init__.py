"""Deterministic, model-agnostic validation framework.

Teams are tiered (S/A/B) per season from their points share; drivers are traced
through their careers; and a driver's skill score (from any architecture, via a
"skill scorer" adapter) is correlated against their forward career outcome.
"""

from .team_tiers import TIER_TO_SCORE  # noqa: F401
