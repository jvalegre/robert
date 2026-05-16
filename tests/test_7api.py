#!/usr/bin/env python

######################################################.
# 	          Testing API with pytest 	             #
######################################################.

"""Tests for RobertModel, YAML config, plot verbosity, and custom PREDICT paths."""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robert import RobertModel
from robert.api import _resolve_prediction_id_column
from robert.argument_parser import set_options
from robert.predict import predict
from robert.utils import (
    load_from_yaml,
    plot_verbosity_level,
    should_plot_curate_pearson,
    should_plot_generate_heatmap,
    should_plot_predict_deep_diagnostics,
    should_plot_predict_results,
    should_plot_verify_metrics,
)

_REPO = Path(__file__).resolve().parent.parent
_REG_CSV = _REPO / "tests" / "Robert_example.csv"
_CLAS_CSV = _REPO / "tests" / "Robert_example_clas.csv"
_FIXTURE_MODEL = _REPO / "tests" / "fixtures" / "custom_predict_model"


@pytest.fixture
def custom_model_dir(tmp_path):
    """Minimal GENERATE-style folder (params CSV + _db.csv)."""
    dest = tmp_path / "custom_model"
    dest.mkdir()
    shutil.copy(_FIXTURE_MODEL / "RF.csv", dest / "RF.csv")
    shutil.copy(_FIXTURE_MODEL / "RF_db.csv", dest / "RF_db.csv")
    return dest


def _holdout_for_predict(X: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    """Holdout where ``Name`` is unique (CSV repeats one tail name)."""
    tail = X.iloc[n_fit:]
    return tail.drop_duplicates(subset=["Name"], keep="first")


# --- YAML ---


def test_yaml_unknown_key_warns_and_known_key_applies(capsys):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write("not_a_robert_option: 1\nseed: 99\n")
        path = f.name
    try:
        opts = set_options({})
        opts.varfile = path
        load_from_yaml(opts)
        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert "not_a_robert_option" in text
        assert opts.seed == 99
    finally:
        os.unlink(path)


def test_yaml_missing_file_message():
    opts = set_options({})
    opts.varfile = os.path.join(tempfile.gettempdir(), "robert_nonexistent_params_xyz.yaml")
    _, msg = load_from_yaml(opts)
    assert "not found" in msg.lower()


# --- PREDICT / plot verbosity ---


def test_predict_custom_params_dir(
    custom_model_dir, fast_robert_kwargs, tmp_path, monkeypatch
):
    """Custom params_dir must not raise NameError on suffixes/suffix_titles."""
    monkeypatch.chdir(tmp_path)
    predict(
        params_dir=str(custom_model_dir),
        predict_diagnostics=False,
        command_line=False,
        **fast_robert_kwargs,
    )
    assert (tmp_path / "PREDICT" / "RF_custom.csv").is_file()


def test_plot_verbosity_level_defaults_and_bounds():
    assert plot_verbosity_level(SimpleNamespace()) == 2
    assert plot_verbosity_level(SimpleNamespace(plot_verbosity="bogus")) == 2
    assert plot_verbosity_level(SimpleNamespace(plot_verbosity=-5)) == 0
    assert plot_verbosity_level(SimpleNamespace(plot_verbosity=99)) == 2


def test_should_plot_predict_tiers():
    off = SimpleNamespace(predict_diagnostics=False, plot_verbosity=2)
    assert not should_plot_predict_results(off)
    assert not should_plot_predict_deep_diagnostics(off)

    mid = SimpleNamespace(predict_diagnostics=True, plot_verbosity=1)
    assert should_plot_predict_results(mid)
    assert not should_plot_predict_deep_diagnostics(mid)

    full = SimpleNamespace(predict_diagnostics=True, plot_verbosity=2)
    assert should_plot_predict_results(full)
    assert should_plot_predict_deep_diagnostics(full)


def test_stage_flags_match_levels():
    low = SimpleNamespace(plot_verbosity=0)
    mid = SimpleNamespace(plot_verbosity=1)
    assert not should_plot_curate_pearson(low)
    assert should_plot_curate_pearson(mid)
    assert not should_plot_generate_heatmap(low)
    assert should_plot_generate_heatmap(mid)
    assert not should_plot_verify_metrics(low)
    assert should_plot_verify_metrics(mid)


def test_predict_plot_verbosity_zero_skips_pngs(
    custom_model_dir, tmp_path, monkeypatch, fast_robert_kwargs
):
    monkeypatch.chdir(tmp_path)
    predict(
        params_dir=str(custom_model_dir),
        predict_diagnostics=True,
        plot_verbosity=0,
        command_line=False,
        **fast_robert_kwargs,
    )
    predict_root = tmp_path / "PREDICT"
    pngs = list(predict_root.rglob("*.png"))
    assert pngs == []
    assert (predict_root / "RF_custom.csv").is_file()


def test_predict_plot_verbosity_one_skips_shap(
    custom_model_dir, tmp_path, monkeypatch, fast_robert_kwargs
):
    monkeypatch.chdir(tmp_path)
    predict(
        params_dir=str(custom_model_dir),
        predict_diagnostics=True,
        plot_verbosity=1,
        command_line=False,
        **fast_robert_kwargs,
    )
    predict_root = tmp_path / "PREDICT"
    shap_pngs = list(predict_root.rglob("SHAP*.png"))
    assert not shap_pngs
    assert list(predict_root.rglob("*.png"))


# --- RobertModel API ---


def test_fit_predict_regression(tmp_path, fast_robert_kwargs):
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    n_fit = 25
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **fast_robert_kwargs,
    )
    model.fit(X.iloc[:n_fit], y.iloc[:n_fit])
    assert model.is_fitted_
    info = model.best_model_info()
    assert "model" in info and "descriptors" in info
    X_hold = _holdout_for_predict(X, n_fit)
    preds = model.predict(X_hold)
    assert preds.shape == (len(X_hold),)
    preds2, sd = model.predict(X_hold, return_std=True)
    assert preds2.shape == sd.shape
    assert np.allclose(preds, preds2)
    y_hat, hw = model.predict(X_hold, return_uncertainty="conformal")
    assert y_hat.shape == hw.shape
    assert np.isfinite(hw).all() and (hw >= 0).all()
    y_b, sd_b, hw_b = model.predict(X_hold, return_uncertainty="both")
    assert np.allclose(y_b, preds)
    assert sd_b.shape == hw_b.shape


def test_fit_predict_classification(tmp_path, fast_robert_kwargs):
    df = pd.read_csv(_CLAS_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    n_fit = 22
    model = RobertModel(
        problem_type="clas",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **fast_robert_kwargs,
    )
    model.fit(X.iloc[:n_fit], y.iloc[:n_fit])
    X_hold = _holdout_for_predict(X, n_fit)
    preds = model.predict(X_hold)
    assert preds.shape == (len(X_hold),)
    acc = model.score(X_hold, y.loc[X_hold.index])
    assert 0.0 <= acc <= 1.0
    with pytest.raises(ValueError, match="conformal"):
        model.predict(X_hold, return_uncertainty="conformal")


def test_deprecated_type_filter_kwargs(tmp_path, fast_robert_kwargs):
    with pytest.warns(DeprecationWarning, match="problem_type"):
        RobertModel(
            type="reg",
            workdir=tmp_path,
            filter_mode="no_pfi",
            **fast_robert_kwargs,
        )
    with pytest.warns(DeprecationWarning, match="filter_mode"):
        RobertModel(
            problem_type="reg",
            workdir=tmp_path,
            filter="no_pfi",
            **fast_robert_kwargs,
        )


def test_names_col_matches_model_data_after_fit(tmp_path, fast_robert_kwargs):
    """``names_col_`` must match GENERATE params so PREDICT and API agree on the id column."""
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **fast_robert_kwargs,
    )
    model.fit(X.iloc[:20], y.iloc[:20])
    assert model.names_col_ == str(model.model_data_["names"])


def test_names_col_default_matches_model_after_fit(tmp_path, fast_robert_kwargs):
    """Auto-inserted ``__robert_name__`` must match what CURATE stores in params."""
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names=None,
        **fast_robert_kwargs,
    )
    model.fit(X.iloc[:15], y.iloc[:15])
    assert model.names_col_ == str(model.model_data_["names"])


def test_resolve_prediction_id_column_prefers_exact_then_model_then_casefold():
    # Exact match on API names column (model_names irrelevant when names_key present).
    df = pd.DataFrame({"Name": ["a", "b"], "Target_values_pred": [1.0, 2.0]})
    assert _resolve_prediction_id_column(df, "Name", "id") == "Name"
    # Column from GENERATE params when API key differs only by case.
    df2 = pd.DataFrame({"name": ["x"], "Target_values_pred": [3.0]})
    assert _resolve_prediction_id_column(df2, "Name", "name") == "name"
    # No column matching names_key, model_names, or a unique casefold hit.
    df3 = pd.DataFrame({"other": [1]})
    with pytest.raises(RuntimeError, match="Could not find row id column"):
        _resolve_prediction_id_column(df3, "Name", "missing")


def test_predict_row_order_matches_input_order(tmp_path, fast_robert_kwargs):
    """``predict`` matches ``X`` row order vs PREDICT CSV row order."""
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    n_fit = 25
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **fast_robert_kwargs,
    )
    model.fit(X.iloc[:n_fit], y.iloc[:n_fit])
    X_hold = _holdout_for_predict(X, n_fit)
    pred_natural = model.predict(X_hold)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(X_hold))
    X_perm = X_hold.iloc[perm].reset_index(drop=True)
    pred_perm = model.predict(X_perm)
    name_to_pred = dict(zip(X_perm["Name"].astype(str), pred_perm))
    pred_realigned = np.array([name_to_pred[str(n)] for n in X_hold["Name"]])
    assert pred_realigned.shape == pred_natural.shape
    assert np.allclose(pred_natural, pred_realigned)


def test_fit_accepts_unused_fit_params(tmp_path, fast_robert_kwargs):
    """Sklearn Pipeline may pass extra fit kwargs; they should not raise."""
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **fast_robert_kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model.fit(X.iloc[:10], y.iloc[:10], sample_weight=None)


def test_noninteractive_mpl_forces_agg_in_subprocess():
    """Force Agg inside _noninteractive_mpl when backend was not Agg."""
    repo = _REPO
    script = f"""
import sys
sys.path.insert(0, {str(repo)!r})
import matplotlib
matplotlib.use("svg", force=True)
import robert.curate  # loads utils / pyplot before robert.api
from robert.api import _noninteractive_mpl
assert "agg" not in matplotlib.get_backend().lower()
with _noninteractive_mpl():
    assert "agg" in matplotlib.get_backend().lower()
assert "svg" in matplotlib.get_backend().lower()
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
