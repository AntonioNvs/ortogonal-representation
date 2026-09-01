"""Legacy alias for bayesian_ssm (replaces deterministic IPF)."""

from baselines.bayesian_ssm import export_bayesian_ssm, load_bayesian_comparator_skill as _load

__all__ = ["export_bayesian_ssm", "load_bayesian_comparator_skill"]


def load_bayesian_comparator_skill(db, max_year: int = 2025):
    return _load(db, max_year=max_year)
