"""Boundary warning references and their placement in the results summary."""

from html import unescape
from pathlib import Path
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from robert.report import report


@pytest.fixture
def warning_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def create(suffix='No PFI', model='MVL'):
        renderer = report.__new__(report)
        renderer.args = SimpleNamespace(csv_name='input.csv', error_type='rmse')
        renderer.report_model_name = model
        renderer.model_suffix = ''
        scores = {f'{key}_{suffix}': value for key, value in {
            'flawed_mod_score': 0, 'failed_tests': 0,
            'train_val_gap_score': 2, 'diff_scaled_rmse_score': 2, 'diff_mcc_score': 2,
            'cv_sd_score': 2, 'cv_score_combined': 2, 'test_score_combined': 2,
            'sorted_cv_high_score': 2, 'sorted_cv_low_score': 2,
            'applicability_domain_score': 2, 'degradation_score': 2,
            'spearman_high': 0.9, 'spearman_low': 0.9,
            'interp_score': 8, 'extrap_score': 8,
        }.items()}
        warnings = {f'{key}_{suffix}': value for key, value in {
            'severe_warnings': [], 'moderate_warnings': [],
            'y_dist_info': '', 'pearson_info': '', 'max_sd': 0,
        }.items()}
        return renderer, scores, warnings

    return create


def _summary_columns(renderer, scores, warnings, suffix, pred_type='reg'):
    params = pd.DataFrame([{
        'model': renderer.report_model_name, 'y': 'target', 'names': 'Name',
        'X_descriptors': '[]', 'params': '{}',
    }])
    html = unescape(renderer.print_summary_page(pred_type, scores, warnings, suffix, params))
    interpolation = html.split('Interpolation warnings</div>', 1)[1].split('</td>', 1)[0]
    boundary = html.split('Boundary robustness warnings</div>', 1)[1].split('</td>', 1)[0]
    return interpolation, boundary


def _warning_reference_tables():
    document = (Path(__file__).resolve().parents[1] / 'docs/Report/score.rst').read_text(encoding='utf-8')
    reference = document.split('   .. tab-item:: List of warnings\n', 1)[1].split('   .. tab-item:: ', 1)[0]
    return reference.split('      .. list-table::')[1:]


@pytest.mark.parametrize('table_index', [0, 1])
def test_documentation_warnings_follow_severity_and_section_order(table_index):
    section_order = ['', 'B.1', 'B.2', 'B.3', 'B.4', 'B.5', 'B.6', 'C', 'D', 'E', 'F', 'G', 'H']
    table = _warning_reference_tables()[table_index]
    rows = re.split(r'(?m)^         \* - ', table)[2:]
    priorities = []
    for row in rows:
        cells = row.split('\n           - ')
        section = re.search(r'\(Section ([A-Z](?:\.\d+)?)\)', cells[0])
        priorities.append((not cells[2].lstrip().startswith('**Severe.**'),
                           section_order.index(section[1] if section else '')))
    assert priorities == sorted(priorities)


@pytest.mark.parametrize('suffix', ['No PFI', 'PFI'])
@pytest.mark.parametrize('pred_type', ['reg', 'clas'])
@pytest.mark.parametrize('failed_tests', [0, 1])
def test_summary_warning_order_matches_documentation(warning_context, suffix, pred_type, failed_tests):
    renderer, scores, warnings = warning_context(suffix, model='RF')
    scores.update({f'{key}_{suffix}': value for key, value in {
        'flawed_mod_score': -1, 'failed_tests': failed_tests,
        'train_val_gap_score': 0, 'diff_scaled_rmse_score': 0, 'diff_mcc_score': 0,
        'cv_sd_score': 0, 'cv_score_combined': 0, 'test_score_combined': 0,
        'sorted_cv_high_score': 0, 'sorted_cv_low_score': 0, 'applicability_domain_score': 0,
    }.items()})
    warnings.update({f'{key}_{suffix}': value for key, value in {
        'low_relevance_descp': True, 'bimodal_y': True, 'mvl_baseline': 'similar',
        'y_dist_info': 'WARNING! Your data is not uniform',
        'pearson_info': 'WARNING! High correlations', 'max_sd': 7, 'outlier_pct': 6,
    }.items()})
    warnings['ambiguous_params'] = ['max_features', 'max_samples']
    original_scores = scores.copy()
    renderer.analyze_warnings(scores, suffix, warnings, pred_type)
    columns = _summary_columns(renderer, scores, warnings, suffix, pred_type)

    messages = warnings[f'severe_warnings_{suffix}'] + warnings[f'moderate_warnings_{suffix}']
    for title, column, table in zip(('Interpolation', 'Boundary robustness'), columns, _warning_reference_tables()):
        order = [line.removeprefix('         * - ') for line in table.splitlines()
                 if line.startswith('         * - ')][1:]
        shown = sorted((message for message in messages if message in column), key=column.index)
        if pred_type == 'clas' and title == 'Boundary robustness':
            assert not shown
            continue
        assert shown
        normalized = [re.sub(r'^\w+(?= = 1 \(integer\))', '{parameter}', message) for message in shown]
        ranks = [order.index(message) for message in normalized]
        assert ranks == sorted(ranks), (title, shown)
    assert scores == original_scores


@pytest.mark.parametrize('suffix', ['No PFI', 'PFI'])
@pytest.mark.parametrize('model,changes,expected,severity', [
    ('RF', {}, 'Tree-based algorithm, not designed to extrapolate outside the training range', 'severe'),
    ('KNeighborsRegressor', {}, 'Nearest-neighbour algorithm, not designed to extrapolate outside the training range', 'severe'),
    ('MVL', {'applicability_domain_score': 0}, 'Poor extrapolation to descriptors very different from training (Section B.3)', 'moderate'),
    ('MVL', {'sorted_cv_high_score': 0}, 'Difficult to extrapolate new predictions in the upper zone (Section B.1)', 'moderate'),
    ('MVL', {'sorted_cv_low_score': 0}, 'Difficult to extrapolate new predictions in the lower zone (Section B.2)', 'moderate'),
    ('MVL', {'sorted_cv_high_score': 0, 'sorted_cv_low_score': 0}, [
        'Difficult to extrapolate new predictions in the upper zone (Section B.1)',
        'Difficult to extrapolate new predictions in the lower zone (Section B.2)',
    ], 'moderate'),
    # These standalone warnings were removed from the report; keep them disabled.
    ('MVL', {'degradation_score': 0}, None, None),
    ('MVL', {'spearman_high': 0.4}, None, None),
    ('MVL', {'spearman_low': 0.4}, None, None),
    ('MVL', {'spearman_high': 0.4, 'spearman_low': 0.4}, None, None),
])
def test_boundary_warning_severity_and_placement(warning_context, suffix, model, changes, expected, severity):
    renderer, scores, warnings = warning_context(suffix, model)
    scores.update({f'{key}_{suffix}': value for key, value in changes.items()})
    original_scores = scores.copy()
    renderer.analyze_warnings(scores, suffix, warnings, 'reg')
    expected_warnings = expected if isinstance(expected, list) else [expected]
    assert warnings[f'moderate_warnings_{suffix}'] == (expected_warnings if severity == 'moderate' else [])
    assert warnings[f'severe_warnings_{suffix}'] == (expected_warnings if severity == 'severe' else [])
    assert scores == original_scores
    if expected is None:
        return
    interpolation, boundary = _summary_columns(renderer, scores, warnings, suffix)
    for message in expected_warnings:
        assert '(Boundary robustness)' not in message
        assert message in boundary
        assert message not in interpolation
    assert f'{severity.upper()} WARNING' in boundary


@pytest.mark.parametrize('suffix', ['No PFI', 'PFI'])
@pytest.mark.parametrize('score,expected', [(0, True), (0.5, True), (0.99, True),
                                         (1.0, False), (1.5, False), (2.0, False), (None, False)])
def test_descriptor_warning_uses_strict_one_point_threshold(warning_context, suffix, score, expected):
    renderer, scores, warnings = warning_context(suffix)
    scores[f'applicability_domain_score_{suffix}'] = score
    renderer.analyze_warnings(scores, suffix, warnings, 'reg')
    message = 'Poor extrapolation to descriptors very different from training (Section B.3)'
    assert (message in warnings[f'moderate_warnings_{suffix}']) == expected
    assert warnings[f'severe_warnings_{suffix}'] == []


@pytest.mark.parametrize('suffix', ['No PFI', 'PFI'])
@pytest.mark.parametrize('side,zone,section', [('high', 'upper', 'B.1'), ('low', 'lower', 'B.2')])
@pytest.mark.parametrize('components,expected', [
    ((0, 0, 0), True), ((0.5, 0.5, 0.5), True), ((1.5, 0.3, 0.1), True),
    ((1, 0.5, 0.5), False), ((0, 1, 1), False), ((1.5, 1, 1), False), ((2, 1, 1), False),
])
def test_y_zone_warning_uses_strict_two_point_total(warning_context, suffix, side, zone, section, components, expected):
    renderer, scores, warnings = warning_context(suffix)
    for key, value in zip((f'sorted_cv_{side}_score', f'spearman_score_{side}', f'degradation_score_{side}'), components):
        scores[f'{key}_{suffix}'] = value
    renderer.analyze_warnings(scores, suffix, warnings, 'reg')
    message = f'Difficult to extrapolate new predictions in the {zone} zone (Section {section})'
    assert warnings[f'moderate_warnings_{suffix}'] == ([message] if expected else [])
    assert warnings[f'severe_warnings_{suffix}'] == []


def test_missing_boundary_scores_do_not_create_warnings(warning_context):
    renderer, scores, warnings = warning_context()
    for key in ('applicability_domain_score', 'sorted_cv_high_score', 'sorted_cv_low_score'):
        scores.pop(f'{key}_No PFI')
    renderer.analyze_warnings(scores, 'No PFI', warnings, 'reg')
    assert warnings['moderate_warnings_No PFI'] == []
    assert warnings['severe_warnings_No PFI'] == []


def test_interpolation_test_warning_is_not_confused_with_boundary_section_three(warning_context):
    suffix = 'No PFI'
    renderer, scores, warnings = warning_context()
    scores[f'test_score_combined_{suffix}'] = 0
    scores[f'applicability_domain_score_{suffix}'] = 0
    scores[f'spearman_high_{suffix}'] = 0.4
    renderer.analyze_warnings(scores, suffix, warnings, 'reg')
    interpolation, boundary = _summary_columns(renderer, scores, warnings, suffix)
    assert 'Weak test-set predictions (Section B.3)' in interpolation
    assert 'Weak test-set predictions' not in boundary
    assert 'Poor extrapolation to descriptors very different from training (Section B.3)' in boundary
    assert 'Difficult to prioritize' not in boundary
    assert 'Difficult to prioritize' not in interpolation


@pytest.mark.parametrize('current_scores,scope', [
    ((6, 4), 'Boundary robustness'),
    ((4, 4), 'Interpolation and Boundary robustness'),
])
def test_shared_model_screening_warning_keeps_its_existing_section_f_reference(warning_context, current_scores, scope):
    suffix = 'No PFI'
    renderer, scores, warnings = warning_context(model='RF')
    renderer.model_suffix = '_RF'
    renderer.model_scores = {suffix: {'MVL': (5, 5), 'RF': current_scores}}
    renderer.analyze_warnings(scores, suffix, warnings, 'reg')
    expected = f'A simpler MVL model scores as high or higher in {scope} (Section F)'
    assert expected in warnings[f'moderate_warnings_{suffix}']
    interpolation, boundary = _summary_columns(renderer, scores, warnings, suffix)
    assert expected in boundary
    assert (expected in interpolation) == ('Interpolation' in scope)


@pytest.mark.parametrize('pred_type', ['reg', 'clas'])
def test_no_new_warnings_for_healthy_metrics(warning_context, pred_type):
    renderer, scores, warnings = warning_context()
    scores['spearman_high_No PFI'] = scores['spearman_low_No PFI'] = 0.5
    renderer.analyze_warnings(scores, 'No PFI', warnings, pred_type)
    assert warnings['moderate_warnings_No PFI'] == []
    assert warnings['severe_warnings_No PFI'] == []


def test_classification_does_not_generate_boundary_warnings(warning_context):
    renderer, scores, warnings = warning_context(model='RF')
    for key in ('sorted_cv_high_score', 'sorted_cv_low_score', 'applicability_domain_score',
                'degradation_score', 'spearman_high', 'spearman_low'):
        scores[f'{key}_No PFI'] = 0
    renderer.analyze_warnings(scores, 'No PFI', warnings, 'clas')
    assert warnings['moderate_warnings_No PFI'] == []
    assert warnings['severe_warnings_No PFI'] == []
