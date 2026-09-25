#!/usr/bin/env python

######################################################.
# 	    Testing test_select() with pytest 	         #
######################################################.

"""Direct checks on which points EVEN/STRATIFIED actually select as the test set - the
GENERATE pytests only check downstream metrics and the recorded split *name*, not whether
the selected points themselves have the properties EVEN/STRATIFIED are supposed to guarantee
(even coverage of the y-range for regression, preserved class balance for classification)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from robert.utils import test_select as _test_select


def test_split_even_regression_spreads_test_points_across_y_range():
    """EVEN should pick one point per quantile bin of the sorted y-range, not points bunched
    in a single region of it."""
    n = 60
    rng = np.random.default_rng(0)
    y = pd.Series(rng.uniform(0, 100, n))
    X = pd.DataFrame({"x1": rng.uniform(0, 1, n)})

    args = SimpleNamespace(test_set=0.2, type="reg", split="EVEN", seed=42)
    runner = SimpleNamespace(args=args)

    test_points = _test_select(runner, X, y)

    expected_size = max(round(0.2 * n), 4)
    assert len(test_points) == expected_size

    # split the full sorted range into as many contiguous chunks as there are test points -
    # every chunk should have contributed at least one selected point, otherwise some region
    # of the y-range was skipped entirely
    sorted_idx = y.sort_values().index.tolist()
    chunks = np.array_split(sorted_idx, expected_size)
    for chunk in chunks:
        assert any(idx in test_points for idx in chunk), (
            "EVEN split left a region of the y-range with no test point"
        )


def test_split_stratified_classification_preserves_class_balance():
    """STRATIFIED should keep the test set's class proportions close to the full dataset's,
    and never leave a class entirely out of the test set."""
    n_class0, n_class1 = 70, 30
    y = pd.Series([0] * n_class0 + [1] * n_class1)
    X = pd.DataFrame({"x1": np.arange(n_class0 + n_class1, dtype=float)})

    args = SimpleNamespace(test_set=0.2, type="clas", split="STRATIFIED", seed=42)
    runner = SimpleNamespace(args=args)

    test_points = _test_select(runner, X, y)

    expected_size = max(round(0.2 * len(y)), 4)
    assert len(test_points) == expected_size

    test_labels = y.loc[test_points]
    full_frac_class1 = (y == 1).mean()
    test_frac_class1 = (test_labels == 1).mean()
    assert abs(test_frac_class1 - full_frac_class1) <= 0.10

    # both classes must actually appear in the test set - a real regression here would be
    # e.g. RND accidentally being used instead of STRATIFIED, which could by chance leave a
    # minority class entirely out of a small test set
    assert set(test_labels.unique()) == {0, 1}

    train_labels = y.drop(test_points)
    assert set(train_labels.unique()) == {0, 1}
