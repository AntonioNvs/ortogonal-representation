"""Skill scoring package — model-agnostic export contract."""

from skill.calibration import CalibrationParams, calibrate_to_0_10, distribution_diagnostics, fit_calibration
from skill.contract import InferenceMode, SkillExport, SkillExportMetadata
from skill.decomposition import aggregate_season_shapley, bootstrap_shapley_ci, shapley_variance_shares
from skill.export import build_skill_export, race_to_season_summary

__all__ = [
    "CalibrationParams",
    "InferenceMode",
    "SkillExport",
    "SkillExportMetadata",
    "aggregate_season_shapley",
    "bootstrap_shapley_ci",
    "build_skill_export",
    "calibrate_to_0_10",
    "distribution_diagnostics",
    "fit_calibration",
    "race_to_season_summary",
    "shapley_variance_shares",
]
