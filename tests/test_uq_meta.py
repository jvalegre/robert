#!/usr/bin/env python

"""Tests for top-k meta-model uncertainty helpers and API."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robert import RobertModel
from robert.utils import (
    aggregate_meta_uq_decomposition,
    discover_top_k_model_candidates,
)

_REG_CSV = Path(__file__).resolve().parent / "Robert_example.csv"

_FAST = {
    "model": ["RF", "GB"],
    "n_iter": 2,
    "init_points": 2,
    "repeat_kfolds": 2,
    "kfold": 3,
    "pfi_epochs": 1,
    "seed": 42,
    "uq_enable_meta": True,
    "uq_top_k_models": 2,
    "uq_model_weighting": "uniform",
}


def test_aggregate_meta_uq_regression_decomposition():
    """Total variance equals within + between for regression."""
    preds = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
    sds = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.3]])
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    y, uq_m, uq_meta, uq_tot = aggregate_meta_uq_decomposition(
        preds, sds, weights, "reg"
    )
    assert y.shape == (2,)
    assert np.all(uq_tot >= uq_m - 1e-9)
    assert np.all(uq_tot >= uq_meta - 1e-9)
    assert np.all(uq_m >= 0) and np.all(uq_meta >= 0)


def test_discover_top_k_empty_dir(tmp_path):
    assert discover_top_k_model_candidates(tmp_path, 3) == []


def test_fit_predict_meta_uncertainty_modes(tmp_path):
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
    X_hold = X.iloc[n_fit:].drop_duplicates(subset=["Name"], keep="first")
    y_hat, uq_meta = model.predict(X_hold, return_uncertainty="meta")
    assert y_hat.shape == uq_meta.shape
    assert np.isfinite(uq_meta).all() and (uq_meta >= 0).all()
    _, uq_total = model.predict(X_hold, return_uncertainty="total")
    y_d, uq_m, uq_meta2, uq_tot = model.predict(
        X_hold, return_uncertainty="decomposed"
    )
    assert y_d.shape == uq_m.shape == uq_meta2.shape == uq_tot.shape
    assert np.all(uq_tot >= uq_m - 1e-9)
    assert np.allclose(uq_tot, uq_total, rtol=1e-5, atol=1e-5)


def test_meta_uncertainty_requires_enable_flag(tmp_path):
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    fast = {k: v for k, v in _FAST.items() if k != "uq_enable_meta"}
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **fast,
    )
    model.fit(X.iloc[:20], y.iloc[:20])
    X_hold = X.iloc[20:].drop_duplicates(subset=["Name"], keep="first")
    with pytest.raises(ValueError, match="uq_enable_meta"):
        model.predict(X_hold, return_uncertainty="total")
