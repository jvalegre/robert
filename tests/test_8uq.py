#!/usr/bin/env python

######################################################.
# 	          Testing UQ with pytest   	             #
######################################################.

"""Tests for meta-model and automatic uncertainty quantification."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robert import RobertModel
from robert.uq_auto import (
    CANDIDATE_CV_SD,
    apply_uncertainty_scaler,
    evaluate_uq_candidates,
    fit_uncertainty_scaler,
    score_uncertainty_candidate,
)
from robert.utils import (
    aggregate_meta_uq_decomposition,
    discover_top_k_model_candidates,
)

_REPO = Path(__file__).resolve().parent.parent
_REG_CSV = _REPO / "tests" / "Robert_example.csv"
_CLAS_CSV = _REPO / "tests" / "Robert_example_clas.csv"


class _Args:
    uq_auto_candidates = ["cv_sd", "conformal"]
    uq_auto_scaler = "global_multiplicative"
    uq_auto_metric_weights = None
    uq_auto_min_samples = 5
    uq_auto_random_state = 0
    conformal_coverage = 0.9
    seed = 0


# --- meta-model UQ ---


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


def test_fit_predict_meta_uncertainty_modes(tmp_path, fast_robert_kwargs):
    df = pd.read_csv(_REG_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    n_fit = 25
    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        uq_enable_meta=True,
        uq_top_k_models=2,
        uq_model_weighting="uniform",
        **{**fast_robert_kwargs, "model": ["RF", "GB"]},
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


def test_meta_uncertainty_requires_enable_flag(tmp_path, fast_robert_kwargs):
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
    X_hold = X.iloc[20:].drop_duplicates(subset=["Name"], keep="first")
    with pytest.raises(ValueError, match="uq_enable_meta"):
        model.predict(X_hold, return_uncertainty="total")


# --- automatic UQ ---


def test_global_multiplicative_scaler_monotone():
    u = np.array([0.5, 1.0, 2.0])
    r = np.array([0.6, 1.2, 2.4])
    params = fit_uncertainty_scaler("global_multiplicative", u, r)
    scaled = apply_uncertainty_scaler("global_multiplicative", u, params)
    assert np.all(scaled >= 0)
    assert scaled[0] < scaled[1] < scaled[2]


def test_none_scaler_identity():
    u = np.array([1.0, 2.0, 3.0])
    params = fit_uncertainty_scaler("none", u, u)
    out = apply_uncertainty_scaler("none", u, params)
    assert np.allclose(out, u)


def test_score_prefers_calibrated_scale():
    abs_res = np.array([1.0, 1.0, 1.0, 1.0])
    bad = np.full(4, 10.0)
    good = np.full(4, 1.0)
    assert score_uncertainty_candidate(good, abs_res, 0.9) < score_uncertainty_candidate(
        bad, abs_res, 0.9
    )


def test_evaluate_uq_candidates_deterministic():
    rng = np.random.default_rng(0)
    n = 30
    y = rng.normal(size=n)
    oof = y + rng.normal(scale=0.2, size=n)
    Xy = {
        "y_train": y.tolist(),
        "y_pred_train_all": [[v] for v in oof],
        "y_pred_train_sd": (0.3 + 0.1 * rng.random(n)).tolist(),
        "conformal_half_width": 0.5,
    }
    sel1 = evaluate_uq_candidates(Xy, _Args(), "reg")
    sel2 = evaluate_uq_candidates(Xy, _Args(), "reg")
    assert sel1["selected"] == sel2["selected"]
    assert sel1["selected"] in (CANDIDATE_CV_SD, "conformal")


def test_fit_predict_auto_uncertainty(tmp_path, fast_robert_kwargs):
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
        uq_auto_enable=True,
    )
    model.fit(X.iloc[:n_fit], y.iloc[:n_fit])
    X_hold = X.iloc[n_fit:].drop_duplicates(subset=["Name"], keep="first")
    y_hat, u_auto = model.predict(X_hold, return_uncertainty="auto")
    assert y_hat.shape == u_auto.shape
    assert np.isfinite(u_auto).all() and (u_auto >= 0).all()
    y_d, u2, meta = model.predict(X_hold, return_uncertainty="auto_decomposed")
    assert y_d.shape == u2.shape
    assert isinstance(meta, dict)
    assert meta.get("selected") in ("cv_sd", "conformal", "meta_total")
    meta_path = tmp_path / "PREDICT" / "uq_auto_metadata.json"
    assert meta_path.is_file()


def test_auto_classification_raises(tmp_path, fast_robert_kwargs):
    df = pd.read_csv(_CLAS_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    model = RobertModel(
        problem_type="clas",
        filter_mode="no_pfi",
        workdir=tmp_path,
        names="Name",
        **fast_robert_kwargs,
    )
    model.fit(X.iloc[:20], y.iloc[:20])
    X_hold = X.iloc[20:].drop_duplicates(subset=["Name"], keep="first")
    with pytest.raises(ValueError, match="auto"):
        model.predict(X_hold, return_uncertainty="auto")
