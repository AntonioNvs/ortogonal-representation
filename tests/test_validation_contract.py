"""Tests for skill contract, calibration, decomposition, and lineage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skill.calibration import calibrate_to_0_10, distribution_diagnostics, fit_calibration
from skill.contract import InferenceMode, SkillExport, SkillExportMetadata
from skill.decomposition import shapley_variance_shares
from skill.export import build_skill_export
from utils.naming import format_driver_name, format_constructor_name
from validation.team_lineage import ref_to_lineage, TEAM_LINEAGES


def test_calibration_anchors():
    from skill.calibration import CalibrationParams

    params = CalibrationParams(mu=0.0, sigma=1.0, alpha=1.0)
    out = calibrate_to_0_10(np.array([-2.0, 0.0, 2.0]), params)
    assert out.min() >= 0.0 and out.max() <= 10.0
    assert out[1] == pytest.approx(5.0, abs=0.01)
    assert out[0] == pytest.approx(1.19, abs=0.1)
    assert out[2] == pytest.approx(8.81, abs=0.1)


def test_shapley_shares_sum_to_one():
    n = 200
    d = np.random.randn(n)
    c = np.random.randn(n) * 0.5
    x = np.random.randn(n) * 0.2
    shares = shapley_variance_shares(d, c, x)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0 for v in shares.values())


def test_skill_export_validate():
    race_df = pd.DataFrame(
        {
            "driverId": [1, 1],
            "season": [2024, 2024],
            "round": [1, 2],
            "raceId": [10, 11],
            "constructorId": [5, 5],
            "lineage_id": ["red_bull", "red_bull"],
            "driver_name": ["Max Verstappen", "Max Verstappen"],
            "constructor_name": ["Red Bull", "Red Bull"],
            "raw_skill": [1.0, 1.2],
            "contrib_driver": [1.0, 1.2],
            "contrib_constructor": [0.5, 0.5],
            "contrib_context": [0.0, 0.0],
            "contrib_residual": [0.0, 0.0],
            "as_of_round": [1, 2],
            "support_bucket": ["high", "high"],
        }
    )
    export = build_skill_export(
        race_df,
        skill_source="test",
        inference_mode=InferenceMode.FILTERED,
        train_years=[2024],
    )
    export.validate()
    assert (export.race["skill_0_10"] >= 0).all()
    assert (export.race["skill_0_10"] <= 10).all()


def test_lineage_audi_sauber():
    assert ref_to_lineage("sauber") == "audi"
    assert ref_to_lineage("alfa") == "audi"
    assert ref_to_lineage("audi") == "audi"
    assert "kick_sauber" in TEAM_LINEAGES["audi"] or "sauber" in TEAM_LINEAGES["audi"]


def test_lineage_row_label():
    from validation.team_lineage import lineage_row_label

    single = pd.DataFrame({"season": [2020, 2021], "display_name": ["Ferrari", "Ferrari"]})
    assert lineage_row_label(single) == "Ferrari"

    transition = pd.DataFrame(
        {
            "season": [2019, 2020, 2024],
            "display_name": ["Sauber", "Alfa Romeo", "Audi"],
        }
    )
    assert lineage_row_label(transition) == "Sauber/Audi"


def test_format_names():
    assert format_driver_name("Max", "Verstappen") == "Max Verstappen"
    assert format_constructor_name("Red Bull Racing", "red_bull") == "Red Bull Racing"


def test_distribution_diagnostics():
    diag = distribution_diagnostics(np.linspace(1, 9, 100))
    assert diag["n"] == 100
    assert diag["central_mass_4_6"] < 0.5


def test_resolve_plot_output_dir():
    from visualization.style import resolve_plot_output_dir

    out_dir, title = resolve_plot_output_dir("output/plots/tier_heatmap_2014_2025.png")
    assert title == "tier_heatmap_2014_2025"
    assert str(out_dir).endswith("output/plots/tier_heatmap_2014_2025")

    out_dir2, title2 = resolve_plot_output_dir("output/plots/tier_heatmap_2014_2025")
    assert title2 == "tier_heatmap_2014_2025"
    assert out_dir == out_dir2


def test_qualifying_z_from_position():
    from data.race_panel import _standardize_qualifying_lap

    qualifying = pd.DataFrame(
        {
            "qualifyId": [0, 1, 2, 3],
            "raceId": [1, 1, 1, 2],
            "driverId": [10, 11, 12, 10],
            "constructorId": [1, 2, 3, 1],
            "number": [1, 2, 3, 1],
            "position": [1, 2, 3, 1],
            "date": pd.to_datetime(["2020-01-01"] * 4),
        }
    )
    races = pd.DataFrame(
        {
            "raceId": [1, 2],
            "year": [2020, 2020],
            "round": [1, 2],
            "circuitId": [1, 1],
            "name": ["GP1", "GP2"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-08"]),
        }
    )

    class _MiniDb:
        table_dict = {
            "qualifying": type("T", (), {"df": qualifying})(),
            "races": type("T", (), {"df": races})(),
        }

    z = _standardize_qualifying_lap(pd.DataFrame(), _MiniDb())
    assert len(z) == 4
    assert z.loc[(10, 1)] < z.loc[(11, 1)] < z.loc[(12, 1)]
