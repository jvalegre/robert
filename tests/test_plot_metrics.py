#!/usr/bin/env python

"""Tests for VERIFY metrics plotting."""

import pytest

from robert.utils import plot_metrics


@pytest.fixture
def verify_plot_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    verify_dir = tmp_path / "VERIFY"
    verify_dir.mkdir()
    (verify_dir / "RF_No_PFI").touch()
    return tmp_path


def test_plot_metrics_equal_values(verify_plot_env):
    """Degenerate axis limits (all metrics equal) must not break plotting."""
    model_data = {"model": "RF_db.csv"}
    verify_metrics = {
        "metrics": [0.5, 0.5, 0.5, 0.5],
        "test_names": ["Model", "y-mean", "y-shuffle", "one-hot"],
        "colors": ["#808080", "#1f77b4", "#1f77b4", "#1f77b4"],
        "higher_thres": 0.3,
        "unclear_higher_thres": 0.2,
        "lower_thres": 0.3,
        "unclear_lower_thres": 0.2,
    }
    verify_results = {"error_type": "r2"}

    msg = plot_metrics(model_data, "No_PFI", verify_metrics, verify_results)
    png = verify_plot_env / "VERIFY" / "VERIFY_tests_RF_No_PFI.png"
    assert png.is_file()
    assert "VERIFY plot saved" in msg
