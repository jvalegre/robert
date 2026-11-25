"""Full-data CV uses the existing CV protocol without changing other predictions."""

import copy
import math
from html import unescape
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from robert.predict_utils import full_data_cv, print_predict
from robert.report import report
from robert.report_utils import cv_error_difference, get_predict_scores
from robert.utils import (
    _cv_within_partition, applicability_domain_plot, boundary_plot,
    get_prediction_results, load_n_predict,
)


def _dataset(test_points=(1, 6, 11, 16)):
    rng = np.random.default_rng(43)
    X = pd.DataFrame(rng.normal(size=(20, 2)), columns=['x1', 'x2'])
    y = pd.Series(3*X.x1 - 2*X.x2 + rng.normal(size=20))
    indices = list(test_points)
    data = {
        'X_train': X.drop(indices), 'X_test': X.loc[indices],
        'X_train_scaled': X.drop(indices), 'X_test_scaled': X.loc[indices],
        'y_train': y.drop(indices), 'y_test': y.loc[indices],
        'X_external': X.iloc[:3]*100, 'X_external_scaled': X.iloc[:3]*100,
        'y_external': pd.Series([1000., 2000., 3000.]),
    }
    return X, y, data


@pytest.mark.parametrize('repeats,folds', [(10, 5), (3, 4), (1, 5)])
@pytest.mark.parametrize('test_points', [(1, 6, 11, 16), ()])
def test_full_cv_matches_manual_pooled_oof_and_preserves_predictions(repeats, folds, test_points):
    X, y, data = _dataset(test_points)
    model = dict(model='MVL', type='reg', params={}, repeat_kfolds=repeats, kfold=folds)
    runner = SimpleNamespace(args=SimpleNamespace(type='reg'))
    data = load_n_predict(runner, model, data)
    original = copy.deepcopy(data)

    # Independently assign folds on the canonical full database, fit on each
    # training fold and pool ALL held-out predictions, rather than their means.
    predictions = np.empty((len(y), repeats))
    for repeat in range(repeats):
        for train, valid in KFold(folds, shuffle=True, random_state=repeat).split(X):
            fitted = LinearRegression().fit(X.iloc[train], y.iloc[train])
            predictions[valid, repeat] = fitted.predict(X.iloc[valid])
    expected = get_prediction_results(model, np.repeat(y, repeats), predictions.reshape(-1))

    assert full_data_cv(runner, data, model) is data
    assert [data[f'{metric}_full'] for metric in ('r2', 'mae', 'rmse')] == pytest.approx(expected)
    if repeats > 1:
        averaged_rmse = np.sqrt(np.mean((y - predictions.mean(axis=1))**2))
        assert data['rmse_full'] > averaged_rmse
    for key, value in original.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(data[key], value)
        elif isinstance(value, pd.Series):
            pd.testing.assert_series_equal(data[key], value)
        else:
            np.testing.assert_equal(data[key], value)


@pytest.mark.parametrize('repeats,folds', [(10, 5), (3, 4), (1, 5)])
def test_partition_cv_pools_oof_errors_without_changing_predictions_or_refitting_extra_models(monkeypatch, repeats, folds):
    _, _, data = _dataset()
    X, y = data['X_train_scaled'].to_numpy(), data['y_train'].to_numpy()
    X_test = data['X_test_scaled'].to_numpy()
    model = dict(model='MVL', type='reg', params={}, repeat_kfolds=repeats, kfold=folds)
    runner = SimpleNamespace(args=SimpleNamespace(type='reg'))
    oof = np.empty((len(y), repeats))
    held_out = []
    for repeat in range(repeats):
        for train, valid in KFold(folds, shuffle=True, random_state=repeat).split(X):
            fitted = LinearRegression().fit(X[train], y[train])
            oof[valid, repeat] = fitted.predict(X[valid])
            held_out.append(fitted.predict(X_test))
    fits = []

    class RecordingLinearRegression(LinearRegression):
        def fit(self, X, y):
            fits.append(len(y))
            return super().fit(X, y)

    monkeypatch.setattr('robert.utils.load_model', lambda *args, **kwargs: RecordingLinearRegression())
    partition = (runner, model, X, y, X_test, data['y_test'].to_numpy())
    train_pred, test_pred, test_sd, rmse_cv = _cv_within_partition(*partition, return_cv_rmse=True)
    assert len(fits) == repeats*folds
    assert rmse_cv == pytest.approx(np.sqrt(np.mean((y[:, None]-oof)**2)))
    np.testing.assert_allclose(train_pred, oof.mean(axis=1))
    np.testing.assert_allclose(test_pred, np.mean(held_out, axis=0))
    np.testing.assert_allclose(test_sd, np.round(np.std(held_out, axis=0), 2))
    if repeats > 1:
        assert rmse_cv > np.sqrt(np.mean((y-train_pred)**2))
    # Existing callers still get their original three prediction arrays.
    legacy = _cv_within_partition(*partition)
    assert len(legacy) == 3
    for actual, expected in zip(legacy, (train_pred, test_pred, test_sd)):
        np.testing.assert_allclose(actual, expected)


def test_boundary_dat_and_report_use_cv_training_errors_and_keep_test_scores(monkeypatch):
    _, _, data = _dataset()
    model = dict(model='MVL', type='reg', params={}, repeat_kfolds=3, kfold=4, names='Name')
    messages = []
    runner = SimpleNamespace(args=SimpleNamespace(type='reg', log=SimpleNamespace(write=messages.append)))
    data = load_n_predict(runner, model, data)
    full_data_cv(runner, data, model)
    for split in ('train', 'test'):
        data[f'names_{split}'] = pd.Series([f'point-{i}' for i in data[f'y_{split}'].index])
    data.update(pred_min=data['y_train'].min(), pred_max=data['y_train'].max(),
                pred_range=data['y_train'].max()-data['y_train'].min())
    partitions = []

    def capture_partition(*args, **kwargs):
        result = _cv_within_partition(*args, **kwargs)
        partitions.append((args, result))
        return result

    monkeypatch.setattr('robert.utils._cv_within_partition', capture_partition)
    monkeypatch.setattr('robert.utils._boundary_scatter', lambda *args, **kwargs: None)
    monkeypatch.setattr('robert.utils._williams_plot', lambda *args, **kwargs: None)
    print_predict(runner, data, model, 'No_PFI')
    boundary_plot(runner, model, data, 'unused_plot_path')
    applicability_domain_plot(runner, model, data, 'unused_plot_path')
    lines = ['------- (No PFI) -------\n'] + '\n'.join(messages).splitlines()
    scores = get_predict_scores(lines, 'No PFI', 'reg', {})
    assert len(partitions) == 3
    assert 'Test RMSE Low = ' in '\n'.join(lines)
    assert 'Test RMSE High = ' in '\n'.join(lines)
    assert 'CV RMSE Low = ' in '\n'.join(lines)
    assert 'CV RMSE High = ' in '\n'.join(lines)
    assert 'CV RMSE of averaged OOF predictions' in '\n'.join(lines)
    for side, (args, result) in zip(('high', 'low', 'ad'), partitions):
        y_train, y_test = np.asarray(args[3]), np.asarray(args[5])
        assert len(y_train) == 16 and len(y_test) == 4
        if side == 'high':
            assert y_train.max() < y_test.min()
        elif side == 'low':
            assert y_train.min() > y_test.max()
        _, _, _, pooled_cv_rmse = result
        test_rmse = np.sqrt(np.mean((y_test-result[1])**2))
        logged_cv_rmse = float(f'{pooled_cv_rmse:.2g}')
        assert scores[f'rmse_{side}_cv_No PFI'] == logged_cv_rmse
        assert scores[f'rmse_{side}_No PFI'] == float(f'{test_rmse:.2g}')
        assert scores[f'full_cv_diff_{side}_No PFI'] == pytest.approx(
            abs(logged_cv_rmse/scores['rmse_full_No PFI'] - 1)*100
        )


def _prediction_summary(suffix='No_PFI', baseline=1.0, boundary_cv=True):
    lines = [
        f"------- {'(No PFI)' if suffix == 'No_PFI' else 'with PFI filter'} -------\n",
        f'   o  Summary of results MVL_{suffix}:\n',
        '      -  Points CV (train+valid.):Test = 16:4\n',
        '      -  Proportion CV (train+valid.):Test = 80:20\n',
        '      -  Number of descriptors = 2\n',
        '      -  Proportion (train+valid.) points:descriptors = 16:2\n',
        '      -  10x 5-fold CV : R2 = 0.8, MAE = 0.9, RMSE = 1.15\n',
        '      -  Test : R2 = 0.8, MAE = 0.9, RMSE = 1.2\n',
        '      -  Average SD in test set = 0.1\n',
        '      -  y range of dataset (train+valid.) = 0 to 10, total 10\n',
        '      -  Train fit (in-fold) : R2 = 0.9, MAE = 0.8, RMSE = 1.0\n',
        '      -  Average SD in train+validation (out-of-fold) = 0.1\n',
    ]
    if baseline is not None:
        lines.append(f'      -  Full data 10x 5-fold CV : R2 = 0.9, MAE = 0.8, RMSE = {baseline}\n')
    # Deliberately move the boundary results beyond every former lookahead limit.
    lines.extend(['      -  Optional diagnostic\n']*45)
    lines.extend([
        '      -  Sorted CV extremes (repeated) : Test RMSE Low = 0.7, Test RMSE High = 1.4\n',
        '      -  Real range extension (Low/High) : Low crossing = 0.8, High crossing = 0.9\n',
        '      -  Degradation baseline (remaining 80%) : vs Low (top 80%) RMSE = 0.9, vs High (bottom 80%) RMSE = 0.8\n',
        '      -  y SD at low extreme : 0.9\n',
        '      -  y SD at high extreme : 0.9\n',
        '      -  y range at low extreme : 0.9\n',
        '      -  y range at high extreme : 0.9\n',
        '      -  Spearman rank (Low/High) : Low (internal) = 0.8, High (internal) = 0.9, global = 0.95\n',
    ])
    if boundary_cv:
        lines.extend([
            '      -  Sorted CV training pools (repeated) : CV RMSE Low = 1.4, CV RMSE High = 1.3\n',
            '      -  Applicability domain (leverage) : Test RMSE = 1.1, CV RMSE = 0.9, h* = 0.5\n',
        ])
    else:
        lines = [line.replace('Test RMSE Low', 'RMSE Low').replace('Test RMSE High', 'RMSE High') for line in lines]
        lines.append('      -  Applicability domain (leverage) : RMSE = 1.1, h* = 0.5\n')
    return lines


@pytest.mark.parametrize('legacy_cv_label', [False, True])
def test_dat_parser_separates_pfi_baselines_and_retains_boundary_scores(legacy_cv_label):
    lines = _prediction_summary() + _prediction_summary('PFI', baseline=2.)
    if legacy_cv_label:
        lines = [line.replace('CV RMSE Low', 'CV RMSE vs Low').replace('CV RMSE High', 'CV RMSE vs High')
                 for line in lines]
    scores = get_predict_scores(lines, 'No PFI', 'reg', {})
    scores = get_predict_scores(lines, 'PFI', 'reg', scores)
    assert scores['rmse_full_No PFI'] == 1.
    assert scores['rmse_full_PFI'] == 2.
    assert scores['scaled_rmse_full_No PFI'] == 10.
    assert scores['scaled_rmse_high_No PFI'] == 14.
    assert scores['crossing_high_No PFI'] == 0.9
    assert scores['spearman_score_No PFI'] > 0
    assert scores['applicability_domain_score_No PFI'] > 0
    assert scores['rmse_high_cv_No PFI'] == 1.3
    assert scores['rmse_low_cv_No PFI'] == 1.4
    assert scores['rmse_ad_cv_No PFI'] == 0.9
    assert scores['rmse_ad_No PFI'] == 1.1  # held-out error remains the scoring metric
    for metric, expected in [('cv', 15.), ('high', 30.), ('low', 40.), ('ad', 10.)]:
        assert scores[f'full_cv_diff_{metric}_No PFI'] == pytest.approx(expected)
    assert scores['full_cv_diff_cv_PFI'] == pytest.approx(42.5)
    callout = unescape(report.section_a_callouts(scores, 'No PFI'))
    assert 'Pass</strong> · Interpolation: 15.0%' in callout
    assert 'Unclear</strong> · Extrapol. high y values: 30.0%' in callout
    assert 'Warning</strong> · Extrapol. low y values: 40.0%' in callout
    assert 'Pass</strong> · Extrapol. descriptor-space: 10.0%' in callout
    assert 'CV RMSE difference vs. full-data CV' in callout


def test_old_dat_has_unavailable_comparisons_without_borrowing_pfi_metrics():
    lines = _prediction_summary(baseline=None) + _prediction_summary('PFI')
    scores = get_predict_scores(lines, 'No PFI', 'reg', {})
    assert 'rmse_full_No PFI' not in scores
    assert all(scores[f'full_cv_diff_{metric}_No PFI'] is None for metric in ('cv', 'high', 'low', 'ad'))
    assert report.section_a_callouts(scores, 'No PFI').count('N/A</strong>') == 4
    assert 'Rerun PREDICT' in report.section_a_callouts(scores, 'No PFI')


def test_new_baseline_does_not_change_existing_scores():
    old = get_predict_scores(_prediction_summary(baseline=None), 'No PFI', 'reg', {})
    new = get_predict_scores(_prediction_summary(), 'No PFI', 'reg', {})
    for key in old:
        if not key.startswith('full_cv_diff_'):
            assert new[key] == old[key]


def test_legacy_boundary_logs_keep_scores_but_not_test_based_split_comparisons():
    legacy = get_predict_scores(_prediction_summary(boundary_cv=False), 'No PFI', 'reg', {})
    current = get_predict_scores(_prediction_summary(), 'No PFI', 'reg', {})
    assert legacy['full_cv_diff_cv_No PFI'] == current['full_cv_diff_cv_No PFI']
    for side in ('high', 'low', 'ad'):
        assert legacy[f'full_cv_diff_{side}_No PFI'] is None
        assert legacy[f'rmse_{side}_No PFI'] == current[f'rmse_{side}_No PFI']
    for key, value in legacy.items():
        if not key.startswith('full_cv_diff_'):
            assert current[key] == value
    html = report.section_a_callouts(legacy, 'No PFI')
    assert html.count('N/A</strong>') == 3
    assert 'Rerun PREDICT' in html


def test_split_comparisons_do_not_use_test_errors_or_averaged_cv_degradation_errors():
    lines = _prediction_summary()
    scores = get_predict_scores(lines, 'No PFI', 'reg', {})
    changed = [line.replace('Test RMSE Low = 0.7', 'Test RMSE Low = 7.0')
               .replace('Test RMSE High = 1.4', 'Test RMSE High = 14.0')
               .replace('Test RMSE = 1.1', 'Test RMSE = 11.0')
               .replace('vs Low (top 80%) RMSE = 0.9', 'vs Low (top 80%) RMSE = 9.0')
               .replace('vs High (bottom 80%) RMSE = 0.8', 'vs High (bottom 80%) RMSE = 8.0')
               for line in lines]
    changed_scores = get_predict_scores(changed, 'No PFI', 'reg', {})
    for side in ('cv', 'high', 'low', 'ad'):
        assert changed_scores[f'full_cv_diff_{side}_No PFI'] == scores[f'full_cv_diff_{side}_No PFI']


def test_print_predict_logs_actual_full_data_cv_settings():
    _, y, data = _dataset()
    model = dict(model='MVL', type='reg', params={}, repeat_kfolds=3, kfold=4)
    messages = []
    runner = SimpleNamespace(args=SimpleNamespace(type='reg', log=SimpleNamespace(write=messages.append)))
    data = load_n_predict(runner, model, data)
    full_data_cv(runner, data, model)
    data.update(pred_min=y.min(), pred_max=y.max(), pred_range=y.max()-y.min())
    print_predict(runner, data, model, 'No_PFI')
    lines = messages[0].splitlines()
    sd_line = next(i for i, line in enumerate(lines) if 'Average SD in train+validation' in line)
    assert lines[sd_line+1].startswith('      -  Full data 3x 4-fold CV : R2 = ')
    assert 'MAE = ' in lines[sd_line+1] and 'RMSE = ' in lines[sd_line+1]
    parsed = get_predict_scores(['------- (No PFI) -------\n'] + lines, 'No PFI', 'reg', {})
    assert parsed['rmse_full_No PFI'] == float(f"{data['rmse_full']:.2}")
    assert parsed['cv_type_No PFI'] == '3x 4-fold CV'


@pytest.mark.parametrize('rmse,baseline,expected', [
    (0., 0., 0.), (1., 0., math.inf), (0., 1., 100.),
    (1.3e-12, 1e-12, 30.), (None, 1., None), (1., None, None),
    (math.nan, 1., None), (1., math.inf, None),
])
def test_error_difference_handles_zero_tiny_and_unavailable_values(rmse, baseline, expected):
    actual = cv_error_difference(rmse, baseline)
    assert actual == pytest.approx(expected) if expected is not None else actual is None


def _classification_dataset(n_classes=2, with_test=True):
    rng = np.random.default_rng(43)
    y = pd.Series(np.arange(30) % n_classes)
    X = pd.DataFrame(rng.normal(size=(30, 2)), columns=['x1', 'x2'])
    X['x1'] += y
    indices = [2, 7, 12, 17, 22, 27] if with_test else []
    data = {
        'X_train': X.drop(indices), 'X_test': X.loc[indices],
        'X_train_scaled': X.drop(indices), 'X_test_scaled': X.loc[indices],
        'y_train': y.drop(indices), 'y_test': y.loc[indices],
        'X_external': X.iloc[:3]*100, 'X_external_scaled': X.iloc[:3]*100,
        'y_external': y.iloc[:3],
    }
    return X, y, data


@pytest.mark.parametrize('repeats,folds', [(10, 5), (3, 4), (1, 5)])
@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.parametrize('with_test', [True, False])
def test_full_classification_cv_matches_manual_stratified_oof(repeats, folds, n_classes, with_test):
    X, y, data = _classification_dataset(n_classes, with_test)
    params = dict(max_depth=2, random_state=17)
    model = dict(model='DecisionTreeClassifier', type='clas', params=params,
                 repeat_kfolds=repeats, kfold=folds)
    runner = SimpleNamespace(args=SimpleNamespace(type='clas'))
    data = load_n_predict(runner, model, data)
    original = copy.deepcopy(data)
    predictions = np.empty((len(y), repeats))
    for repeat in range(repeats):
        cv = StratifiedKFold(folds, shuffle=True, random_state=repeat)
        for train, valid in cv.split(X, y):
            fitted = DecisionTreeClassifier(**params).fit(X.iloc[train], y.iloc[train])
            predictions[valid, repeat] = fitted.predict(X.iloc[valid])
    expected = get_prediction_results(model, np.repeat(y, repeats), predictions.reshape(-1))

    assert full_data_cv(runner, data, model) is data
    assert [data[f'{metric}_full'] for metric in ('acc', 'f1', 'mcc')] == pytest.approx(expected)
    assert not any(f'{metric}_full' in data for metric in ('r2', 'mae', 'rmse'))
    for key, value in original.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(data[key], value)
        elif isinstance(value, pd.Series):
            pd.testing.assert_series_equal(data[key], value)
        else:
            np.testing.assert_equal(data[key], value)


def _classification_summary(suffix='No_PFI', baseline=0.8, mcc=0.92):
    lines = [
        f"------- {'(No PFI)' if suffix == 'No_PFI' else 'with PFI filter'} -------\n",
        f'   o  Summary of results DecisionTreeClassifier_{suffix}:\n',
        '      -  Points CV (train+valid.):Test = 24:6\n',
        '      -  Proportion CV (train+valid.):Test = 80:20\n',
        '      -  Number of descriptors = 2\n',
        '      -  Proportion (train+valid.) points:descriptors = 24:2\n',
        f'      -  10x 5-fold CV : Accur. = 0.9, F1 score = 0.9, MCC = {mcc}\n',
        '      -  Test : Accur. = 0.9, F1 score = 0.9, MCC = 0.8\n',
        '      -  Train fit (in-fold) : Accur. = 0.9, F1 score = 0.9, MCC = 0.95\n',
    ]
    if baseline is not None:
        lines.append(f'      -  Full data 10x 5-fold CV : Accur. = 0.9, F1 score = 0.9, MCC = {baseline}\n')
    lines.extend([
        '      -  Average agreement in test set (10 repeats) = 97.0\n',
        '      -  Average agreement in train+validation (out-of-fold) (10 repeats) = 94.0\n',
        '      -  MCC coefficient of variation (10 repeats) = 10.0\n',
    ])
    return lines


@pytest.mark.parametrize('mcc,baseline,expected,status', [
    (0.92, 0.8, 15., 'Pass'), (0.96, 0.8, 20., 'Unclear'),
    (0.56, 0.8, 30., 'Unclear'), (0.5, 0.8, 37.5, 'Warning'),
    (-0.4, -0.8, 50., 'Warning'), (0.4, -0.8, 150., 'Warning'),
    (-0.8, -0.8, 0., 'Pass'), (0., 0., 0., 'Pass'),
    (0.4, 0., math.inf, 'Warning'), (-0.4, 0., math.inf, 'Warning'),
])
def test_classification_parser_and_callout_use_mcc(mcc, baseline, expected, status):
    scores = get_predict_scores(_classification_summary(baseline=baseline, mcc=mcc), 'No PFI', 'clas', {})
    assert scores['mcc_full_No PFI'] == baseline
    assert scores['full_cv_diff_cv_No PFI'] == pytest.approx(expected)
    html = unescape(report.section_a_callouts(scores, 'No PFI', 'clas'))
    assert 'MCC difference vs. full-data CV' in html
    assert f'{status}</strong> · Interpolation:' in html
    assert ('unbounded' if math.isinf(expected) else f'{expected:.1f}%') in html
    assert 'scaled RMSE' not in html
    assert 'High y values' not in html and 'Applicab. domain' not in html


def test_classification_baselines_are_separate_and_existing_scores_unchanged():
    lines = _classification_summary() + _classification_summary('PFI', baseline=0.4)
    scores = get_predict_scores(lines, 'No PFI', 'clas', {})
    scores = get_predict_scores(lines, 'PFI', 'clas', scores)
    assert scores['full_cv_diff_cv_No PFI'] == pytest.approx(15.)
    assert scores['full_cv_diff_cv_PFI'] == pytest.approx(130.)
    old_lines = _classification_summary(baseline=None) + _classification_summary('PFI')
    old_scores = get_predict_scores(old_lines, 'No PFI', 'clas', {})
    assert 'mcc_full_No PFI' not in old_scores
    assert old_scores['full_cv_diff_cv_No PFI'] is None
    for key, value in old_scores.items():
        if not key.startswith('full_cv_diff_'):
            assert scores[key] == value
    html = report.section_a_callouts(old_scores, 'No PFI', 'clas')
    assert html.count('N/A</strong>') == 1
    assert 'Rerun PREDICT' in html


def test_print_predict_logs_classification_baseline_and_actual_cv_settings():
    _, _, data = _classification_dataset()
    model = dict(model='DecisionTreeClassifier', type='clas', params=dict(max_depth=2, random_state=17),
                 repeat_kfolds=3, kfold=4)
    messages = []
    runner = SimpleNamespace(args=SimpleNamespace(type='clas', log=SimpleNamespace(write=messages.append)))
    data = load_n_predict(runner, model, data)
    full_data_cv(runner, data, model)
    print_predict(runner, data, model, 'No_PFI')
    lines = messages[0].splitlines()
    full_line = next(line for line in lines if '-  Full data ' in line)
    assert full_line.startswith('      -  Full data 3x 4-fold CV : Accur. = ')
    assert 'F1 score = ' in full_line and 'MCC = ' in full_line
    scores = get_predict_scores(['------- (No PFI) -------\n'] + lines, 'No PFI', 'clas', {})
    assert scores['mcc_full_No PFI'] == float(f"{data['mcc_full']:.2}")
    assert scores['cv_type_No PFI'] == '3x 4-fold CV'
