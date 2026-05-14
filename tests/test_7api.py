#!/usr/bin/env python

"""Tests for robert.api.RobertModel."""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robert import RobertModel
from robert.api import _resolve_prediction_id_column

_REPO = Path(__file__).resolve().parent.parent
_REG_CSV = _REPO / "tests" / "Robert_example.csv"
_CLAS_CSV = _REPO / "tests" / "Robert_example_clas.csv"

_FAST = {
    "model": ["RF"],
    "n_iter": 2,
    "init_points": 2,
    "repeat_kfolds": 2,
    "kfold": 3,
    "pfi_epochs": 1,
    "seed": 42,
}


def _holdout_for_predict(X: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    """Holdout where ``Name`` is unique (CSV repeats one tail name)."""
    tail = X.iloc[n_fit:]
    return tail.drop_duplicates(subset=["Name"], keep="first")


def test_fit_predict_regression(tmp_path):
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    n_fit = 25
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **_FAST,
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


def test_fit_predict_classification(tmp_path):
    df = pd.read_csv(_CLAS_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    n_fit = 22
    model = RobertModel(
        problem_type="clas",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **_FAST,
    )
    model.fit(X.iloc[:n_fit], y.iloc[:n_fit])
    X_hold = _holdout_for_predict(X, n_fit)
    preds = model.predict(X_hold)
    assert preds.shape == (len(X_hold),)
    acc = model.score(X_hold, y.loc[X_hold.index])
    assert 0.0 <= acc <= 1.0
    with pytest.raises(ValueError, match="conformal"):
        model.predict(X_hold, return_uncertainty="conformal")


def test_deprecated_type_filter_kwargs(tmp_path):
    with pytest.warns(DeprecationWarning, match="problem_type"):
        RobertModel(
            type="reg",
            workdir=tmp_path,
            filter_mode="no_pfi",
            **_FAST,
        )
    with pytest.warns(DeprecationWarning, match="filter_mode"):
        RobertModel(
            problem_type="reg",
            workdir=tmp_path,
            filter="no_pfi",
            **_FAST,
        )


def test_names_col_matches_model_data_after_fit(tmp_path):
    """``names_col_`` must match GENERATE params so PREDICT and API agree on the id column."""
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **_FAST,
    )
    model.fit(X.iloc[:20], y.iloc[:20])
    assert model.names_col_ == str(model.model_data_["names"])


def test_names_col_default_matches_model_after_fit(tmp_path):
    """Auto-inserted ``__robert_name__`` must match what CURATE stores in params."""
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names=None,
        **_FAST,
    )
    model.fit(X.iloc[:15], y.iloc[:15])
    assert model.names_col_ == str(model.model_data_["names"])


def test_resolve_prediction_id_column_prefers_exact_then_model_then_casefold():
    df = pd.DataFrame({"Name": ["a", "b"], "Target_values_pred": [1.0, 2.0]})
    assert _resolve_prediction_id_column(df, "Name", "id") == "Name"
    df2 = pd.DataFrame({"name": ["x"], "Target_values_pred": [3.0]})
    assert _resolve_prediction_id_column(df2, "Name", "name") == "name"
    df3 = pd.DataFrame({"other": [1]})
    with pytest.raises(RuntimeError, match="Could not find row id column"):
        _resolve_prediction_id_column(df3, "Name", "missing")


def test_predict_row_order_matches_input_order(tmp_path):
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
        **_FAST,
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


def test_fit_accepts_unused_fit_params(tmp_path):
    """Sklearn Pipeline may pass extra fit kwargs; they should not raise."""
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **_FAST,
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
