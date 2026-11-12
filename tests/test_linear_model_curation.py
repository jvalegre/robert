"""Regression checks for linear models in CURATE's descriptor selection."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robert.utils import correlation_filter, load_minimal_model


class _Log:
    def write(self, message):
        pass


@pytest.mark.parametrize(
    ("model", "pred_type", "error_type"),
    [("RIDGE", "reg", "rmse"), ("LOGISTIC", "clas", "mcc")],
)
def test_curate_selects_descriptors_for_linear_models(model, pred_type, error_type):
    rng = np.random.default_rng(4)
    rows = 18
    frame = pd.DataFrame({f"descriptor_{index}": rng.normal(size=rows)
                          for index in range(7)})
    frame["target"] = (
        np.arange(rows) % 3 if pred_type == "clas"
        else 2 * frame["descriptor_0"] - frame["descriptor_1"] + rng.normal(size=rows)
    )
    frame["name"] = [f"point_{index}" for index in range(rows)]
    args = SimpleNamespace(
        ignore=["name"], y="target", thres_x=1.0, thres_y=0.0,
        corr_filter_x=False, corr_filter_y=False, rfecv_filter=True,
        repeat_kfolds=1, kfold=3, seed=0, pfi_epochs=1,
        model=[model], type=pred_type, error_type=error_type, log=_Log(),
    )

    filtered, per_model = correlation_filter(SimpleNamespace(args=args), frame)

    assert model in per_model
    assert "target" in per_model[model]
    assert "name" in per_model[model]
    assert 2 <= len(per_model[model].columns) - 2 <= 6
    assert len(filtered) == rows


@pytest.mark.parametrize("model", ["RIDGE", "LOGISTIC"])
def test_linear_models_have_minimal_parameters(model):
    assert isinstance(load_minimal_model(model), dict)
