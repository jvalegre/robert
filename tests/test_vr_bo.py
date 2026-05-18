#!/usr/bin/env python

"""Tests for Voting Regressor/Classifier Bayesian optimization."""

from types import SimpleNamespace

import numpy as np

from robert.argument_parser import options_add
from robert.utils import load_minimal_model, load_model, model_adjust_params


def _vr_adapter(problem_type="reg"):
    args = options_add()
    args.seed = 42
    args.type = problem_type
    return SimpleNamespace(args=args)


def test_vr_member_hyperparameters_affect_predictions():
    adapter = _vr_adapter("reg")
    rng = np.random.RandomState(0)
    X = rng.rand(24, 6)
    y = X.sum(axis=1) + rng.randn(24) * 0.05

    params_low = model_adjust_params(adapter, "VR", dict(load_minimal_model("VR")))
    params_high = model_adjust_params(
        adapter,
        "VR",
        {**load_minimal_model("VR"), "rf_n_estimators": 90, "gb_n_estimators": 90},
    )

    model_low = load_model(adapter, "VR", **params_low)
    model_high = load_model(adapter, "VR", **params_high)
    model_low.fit(X, y)
    model_high.fit(X, y)

    assert not np.allclose(model_low.predict(X), model_high.predict(X))


def test_vr_ensemble_weights_affect_predictions():
    adapter = _vr_adapter("reg")
    rng = np.random.RandomState(1)
    X = rng.rand(24, 6)
    y = X.sum(axis=1) + rng.randn(24) * 0.05

    base = load_minimal_model("VR")
    params_a = model_adjust_params(
        adapter, "VR", {**base, "w_rf": 5.0, "w_gb": 0.2, "w_nn": 0.2}
    )
    params_b = model_adjust_params(
        adapter, "VR", {**base, "w_rf": 0.2, "w_gb": 5.0, "w_nn": 0.2}
    )

    model_a = load_model(adapter, "VR", **params_a)
    model_b = load_model(adapter, "VR", **params_b)
    model_a.fit(X, y)
    model_b.fit(X, y)

    assert not np.allclose(model_a.predict(X), model_b.predict(X))


def test_vr_bo_bounds_include_member_models():
    from robert.utils import BO_hyperparams

    bounds = BO_hyperparams("VR")
    assert "w_rf" in bounds
    assert "rf_n_estimators" in bounds
    assert "gb_learning_rate" in bounds
    assert "nn_hidden_layer_1" in bounds


def test_vr_classification_loads():
    adapter = _vr_adapter("clas")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X = np.random.RandomState(2).rand(len(y), 4)
    params = model_adjust_params(adapter, "VR", dict(load_minimal_model("VR")))
    model = load_model(adapter, "VR", **params)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
