"""Representative bot GUI regressions chosen by measured line coverage.

Each source group has its own namespace so its helpers cannot shadow those
from another group. The exported checks remain ordinary pytest tests.
"""

import sys
from types import ModuleType


def _load_case_group(label, source, checks, fixtures):
    module = ModuleType(f"{__name__}.{label}")
    module.__file__ = __file__
    sys.modules[module.__name__] = module
    exec(compile(source, __file__, "exec"), module.__dict__)
    for name in fixtures:
        globals()[name] = module.__dict__[name]
    for name in checks:
        if name.startswith("test_"):
            public_name = f"test_{label}_{name[5:]}"
        else:
            public_name = f"Test{label.title()}{name[4:]}"
        globals()[public_name] = module.__dict__[name]

# Selected checks from test_bot_all_models_results.py
_load_case_group('all_models_results', """\"\"\"Model-aware workflow evidence and GUI state for the ROBERT bot.\"\"\"
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from gui_easyrob.bot.bot_context import GuiSnapshot, build_gui_snapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.bot_rag import KnowledgeBase
from gui_easyrob.bot.heuristics import diagnose_snapshot
from gui_easyrob.bot.workflow_results import WorkflowResultStore, render_workflow_answer, render_workflow_evidence, scope_workflow_result_snapshot
from gui_easyrob.tabs.result_catalog import ResultCatalog

def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding='utf-8')

def _all_models_run(root: Path) -> Path:
    source = root / 'input.csv'
    _write(source, 'name,target\\na,1\\n')
    for model, values in (('GB', (0.81, 0.82)), ('RF', (0.31, 0.32))):
        _write(root / 'PREDICT' / f'PREDICT_{model}_data.dat', f'Starting model with all variables (No PFI)\\n- Test : R2 = {values[0]}\\nStarting model with PFI\\n- Test : R2 = {values[1]}\\nWARNING! {model} PFI issue\\nTime PREDICT: 1 seconds\\n')
        _write(root / 'VERIFY' / f'VERIFY_{model}_data.dat', 'Time VERIFY: 1 seconds\\n')
        for variant in ('No_PFI', 'PFI'):
            _write(root / 'REPORT_models' / f'ROBERT_report_{model}_{variant}.pdf', '%PDF-1.4')
            _write(root / 'PREDICT' / f'{model}_{variant}.csv', 'target,target_pred\\n1,1\\n')
            _write(root / 'PREDICT' / f'Results_boundary_williams_{model}_{variant}.csv', 'leverage\\n0.2\\n')
    _write(root / 'ROBERT_report_GB_No_PFI.pdf', '%PDF-1.4')
    _write(root / 'ROBERT_report_RF_PFI.pdf', '%PDF-1.4')
    _write(root / 'GENERATE' / 'Best_model' / 'No_PFI' / 'GB.csv', 'model\\nGB\\n')
    _write(root / 'GENERATE' / 'Best_model' / 'PFI' / 'RF_PFI.csv', 'model\\nRF\\n')
    return source

def test_bot_discovers_per_model_logs_and_archived_reports(tmp_path):
    source = _all_models_run(tmp_path)
    result = WorkflowResultStore().refresh(str(source), '')
    assert {item[3] for item in result.metrics} == {'PREDICT/PREDICT_GB_data.dat', 'PREDICT/PREDICT_RF_data.dat'}
    assert len(result.metrics) == 4
    assert {item.path for item in result.artifacts if item.kind == 'pdf'} >= {f'REPORT_models/ROBERT_report_{model}_{variant}.pdf' for model in ('GB', 'RF') for variant in ('No_PFI', 'PFI')}
    assert {stage.model for stage in result.stages if stage.name == 'PREDICT'} == {'GB', 'RF'}

def test_bot_scopes_best_models_and_explicit_model_questions(tmp_path):
    source = _all_models_run(tmp_path)
    result = WorkflowResultStore().refresh(str(source), '')
    best = (('No_PFI', 'GB'), ('PFI', 'RF'))
    selected = scope_workflow_result_snapshot(result, 'Summarize the report', selected_variants=best, available_models=('GB', 'RF'))
    answer = render_workflow_answer(selected, 'Summarize the report', full=True)
    assert 'REPORT_models/ROBERT_report_GB_No_PFI.pdf' in answer
    assert 'REPORT_models/ROBERT_report_RF_PFI.pdf' in answer
    assert 'REPORT_models/ROBERT_report_GB_PFI.pdf' not in answer
    assert 'REPORT_models/ROBERT_report_RF_No_PFI.pdf' not in answer
    assert '0.81' in answer and '0.32' in answer
    assert '0.82' not in answer and '0.31' not in answer
    assert 'RF PFI issue' in answer
    assert 'GB PFI issue' not in answer
    bounded = render_workflow_evidence(selected, 'Summarize the report', full=True, max_chars=1600)
    assert len(bounded) <= 1600
    assert '0.81' in bounded and '0.32' in bounded
    assert 'ROBERT_report_GB_No_PFI.pdf' in bounded
    assert 'ROBERT_report_RF_PFI.pdf' in bounded
    explicit = scope_workflow_result_snapshot(result, 'What is RF No PFI test R2?', selected_variants=best, available_models=('GB', 'RF'))
    answer = render_workflow_answer(explicit, 'What is RF No PFI test R2?')
    assert not explicit.warnings
    assert '0.31' in answer
    assert '0.81' not in answer and '0.32' not in answer
    assert 'RF PFI issue' not in answer

def test_gui_snapshot_captures_results_buttons_and_selected_variants(tmp_path):
    source = _all_models_run(tmp_path)
    catalog = ResultCatalog.discover(tmp_path)
    enabled = {'Report': True, 'Predictions': False, 'Images': True, 'Interactive plots': True}
    workspace = SimpleNamespace(buttons={name: SimpleNamespace(isEnabled=lambda value=value: value, isChecked=lambda name=name: name == 'Images') for name, value in enabled.items()}, current_model=lambda: None, model_selector=SimpleNamespace(count=lambda: 3, itemText=lambda index: ('Best models', 'GB', 'RF')[index]))
    window = SimpleNamespace(tab_widget=SimpleNamespace(currentIndex=lambda: 0, tabText=lambda index: 'Results'), results_workspace=workspace, _result_catalog=catalog, _result_view_source_path=str(source), file_path=str(source))
    snapshot = build_gui_snapshot(window)
    assert snapshot.result_model == 'Best models'
    assert snapshot.result_models == ('GB', 'RF')
    assert snapshot.result_selected_variants == (('No_PFI', 'GB'), ('PFI', 'RF'))
    assert snapshot.result_active_view == 'Images'
    assert snapshot.result_enabled_views == ('Report', 'Images', 'Interactive plots')
    assert snapshot.result_disabled_views == ('Predictions',)
    assert 'available for Best models' in diagnose_snapshot(snapshot, 'Is the Interactive plots view available?').summary
    assert 'unavailable' in diagnose_snapshot(snapshot, 'Why is the Predictions view disabled?').summary

def test_all_models_summary_keeps_each_variant_in_local_prompt(tmp_path):
    source = _all_models_run(tmp_path)
    snapshot = WorkflowResultStore().refresh(str(source), '')
    extra = tuple((replace(item, model='NN') for item in snapshot.report_assessments))
    snapshot = replace(snapshot, report_assessments=snapshot.report_assessments + extra)
    evidence = render_workflow_evidence(snapshot, 'Summarize all models', full=True, max_chars=1600)
    assert len(evidence) <= 1600
    assert all((f'{model} {variant}:' in evidence for model in ('GB', 'RF', 'NN') for variant in ('No_PFI', 'PFI')))
    answer = render_workflow_answer(snapshot, 'Summarize all models', full=True)
    assert all((f"**{model} {variant.replace('_', ' ')}:**" in answer for model in ('GB', 'RF', 'NN') for variant in ('No_PFI', 'PFI')))

def test_summary_button_question_uses_gui_model_selection(tmp_path):
    source = _all_models_run(tmp_path)
    result = WorkflowResultStore().refresh(str(source), '')
    gui = GuiSnapshot(active_tab='Results', workflow='Full Workflow', main_csv_path=str(source), test_csv_path='', target_column='target', prediction_type='Regression', process_running=False, run_enabled=True, stop_enabled=False, recent_console='', console_terms=[], workflow_results=result, result_model='Best models', result_models=('GB', 'RF'), result_selected_variants=(('No_PFI', 'GB'), ('PFI', 'RF')))
    engine = BotEngine(KnowledgeBase.from_directory(strict=False))
    answer = engine.answer('Summarize the ROBERT report', gui, 'Heuristic', 'OpenAI', '')
    assert 'ROBERT_report_GB_No_PFI.pdf' in answer
    assert 'ROBERT_report_RF_PFI.pdf' in answer
    assert 'ROBERT_report_GB_PFI.pdf' not in answer
    assert 'ROBERT_report_RF_No_PFI.pdf' not in answer""", ('test_bot_discovers_per_model_logs_and_archived_reports', 'test_bot_scopes_best_models_and_explicit_model_questions', 'test_gui_snapshot_captures_results_buttons_and_selected_variants', 'test_all_models_summary_keeps_each_variant_in_local_prompt', 'test_summary_button_question_uses_gui_model_selection'), ())

# Selected checks from test_bot_main_window.py
_load_case_group('main_window', """import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
GUI_ROOT = Path(__file__).resolve().parents[1] / 'robert' / 'gui_easyrob'
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QToolButton
from PySide6.QtWidgets import QMessageBox
from gui_easyrob.bot.bot_window import BotWindow
from bot.local_llm import LocalLLMError
from bot.token_usage import TokenUsage
from bot.answer_metadata import AnswerMetadata, EvidenceOrigin, SourceCitation
from bot.workflow_results import WorkflowResultSnapshot
from main.window import EasyROB
import main.window as main_window_module

class TestMainWindowBotWindow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.bot_window = BotWindow()
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.holder = SimpleNamespace(bot_window=self.bot_window, bot_panel=self.bot_window.panel, bot_toggle_btn=self.toggle)
        self.holder._sync_bot_toggle = lambda visible: EasyROB._sync_bot_toggle(self.holder, visible)
        self.holder._refresh_local_model_status = lambda: None

    class FakeSettings:

        def __init__(self):
            self.values = {}

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

        def remove(self, key):
            self.values.pop(key, None)

        def sync(self):
            pass

    def tearDown(self):
        self.bot_window.hide()
        self.app.processEvents()
        self.bot_window.deleteLater()
        self.toggle.deleteLater()
        self.app.processEvents()

    def test_toggle_opens_and_hides_floating_bot_window(self):
        EasyROB.toggle_bot_panel(self.holder, True)
        self.app.processEvents()
        self.assertTrue(self.bot_window.isVisible())
        EasyROB.toggle_bot_panel(self.holder, False)
        self.app.processEvents()
        self.assertFalse(self.bot_window.isVisible())

    def test_refresh_workflow_results_updates_snapshot_and_summary_action(self):
        result = WorkflowResultSnapshot(kind='ROBERT', status='completed_with_warnings', root_name='run', summary_available=True)
        store = Mock()
        store.refresh.return_value = result
        self.holder.workflow_result_store = store
        self.holder.file_path = 'train.csv'
        self.holder.csv_test_path = 'test.csv'
        self.holder.worker = None
        self.holder.bot_panel.set_workflow_summary_available = Mock()
        EasyROB._refresh_workflow_result_snapshot(self.holder)
        store.refresh.assert_called_once_with('train.csv', 'test.csv', process_running=False)
        self.assertIs(self.holder.workflow_result_snapshot, result)
        self.holder.bot_panel.set_workflow_summary_available.assert_called_once_with(True, status='completed_with_warnings', workflow_kind='ROBERT')

    def test_load_and_save_web_search_preference(self):
        self.holder.bot_settings = self.FakeSettings()
        self.holder.bot_settings.values = {EasyROB.BOT_WEB_SEARCH_SETTING: False}
        EasyROB._load_bot_preferences(self.holder)
        self.assertFalse(self.bot_window.panel.web_search_checkbox.isChecked())
        self.bot_window.panel.web_search_checkbox.setChecked(True)
        self.bot_window.panel.api_key_edit.setText('saved-key')
        original_information = main_window_module.QMessageBox.information
        main_window_module.QMessageBox.information = Mock()
        try:
            EasyROB.save_bot_api_preferences(self.holder)
        finally:
            main_window_module.QMessageBox.information = original_information
        self.assertIs(self.holder.bot_settings.values[EasyROB.BOT_WEB_SEARCH_SETTING], True)

    def test_ready_handler_preserves_cross_package_source_metadata(self):
        self.holder.bot_history = []
        self.holder._bot_active_request_id = 4
        self.holder._bot_conversation_generation = 2
        source = SourceCitation.create('Official', 'https://example.org/source')
        EasyROB._on_bot_answer_ready(self.holder, 4, 2, 'Answer', False, True, TokenUsage(10, 5, 15, search_requests=1, provider='OpenAI'), AnswerMetadata(evidence_origin=EvidenceOrigin.WEB_AND_DOCUMENTATION, sources=(source,)))
        self.app.processEvents()
        self.assertIn('Web + documentation', self.bot_window.panel.last_usage_label.text())
        self.assertEqual(len(self.bot_window.panel.source_links), 1)

    def test_ready_handler_auto_reset_keeps_only_latest_user_question(self):
        self.bot_window.show()
        self.app.processEvents()
        self.holder.bot_history = [{'role': 'user', 'content': 'Old question'}, {'role': 'assistant', 'content': 'Old answer'}, {'role': 'user', 'content': 'Current question'}]
        self.holder.bot_worker = object()
        self.holder._bot_active_request_id = 4
        self.holder._bot_conversation_generation = 2
        self.holder.bot_panel.set_busy(True)
        self.holder.bot_panel.clear_history = self.bot_window.panel.clear_history
        self.holder.bot_panel.append_message = self.bot_window.panel.append_message
        self.holder.bot_panel.input_edit = self.bot_window.panel.input_edit
        EasyROB._on_bot_answer_ready(self.holder, 4, 2, 'New answer', True, True)
        self.app.processEvents()
        self.assertEqual(self.holder.bot_history, [{'role': 'user', 'content': 'Current question', 'memory_eligible': True}, {'role': 'assistant', 'content': 'New answer', 'memory_eligible': True}])
        rendered = self.bot_window.panel.history_view.toPlainText()
        self.assertIn('Current question', rendered)
        self.assertIn('New answer', rendered)
        self.assertNotIn('Old question', rendered)

    def test_start_bot_reply_captures_history_before_current_question(self):

        class FakeSignal:

            def __init__(self):
                self.callbacks = []

            def connect(self, callback):
                self.callbacks.append(callback)

        class FakeWorker:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.succeeded = FakeSignal()
                self.failed = FakeSignal()
                self.finished = FakeSignal()
                FakeWorker.instances.append(self)

            def start(self):
                pass

            def deleteLater(self):
                pass
        original_worker = main_window_module.BotReplyWorker
        original_snapshot = main_window_module.build_gui_snapshot
        main_window_module.BotReplyWorker = FakeWorker
        main_window_module.build_gui_snapshot = lambda window: SimpleNamespace(active_tab='ROBERT')
        try:
            self.holder.bot_history = [{'role': 'user', 'content': 'Old popup question'}, {'role': 'assistant', 'content': 'Old popup answer'}]
            self.holder.bot_worker = None
            self.holder.bot_engine = object()
            self.holder.bot_panel.set_busy = Mock()
            self.holder.bot_panel.append_message = Mock()
            self.holder.bot_panel.mode_combo.setCurrentText('Heuristic')
            self.holder._bot_next_request_id = 0
            self.holder._bot_active_request_id = None
            self.holder._bot_conversation_generation = 0
            EasyROB._start_bot_reply(self.holder, 'what is this popup?')
        finally:
            main_window_module.BotReplyWorker = original_worker
            main_window_module.build_gui_snapshot = original_snapshot
        self.assertEqual(FakeWorker.instances[0].kwargs['conversation_history'], ({'role': 'user', 'content': 'Old popup question'}, {'role': 'assistant', 'content': 'Old popup answer'}))
        self.assertEqual(self.holder.bot_history, [{'role': 'user', 'content': 'Old popup question'}, {'role': 'assistant', 'content': 'Old popup answer'}, {'role': 'user', 'content': 'what is this popup?', 'memory_eligible': True}])

    def test_popup_tracking_helpers_store_single_ephemeral_popup_context(self):
        holder = SimpleNamespace(bot_active_popup=None, bot_popup_context=None, bot_popup_context_closed_at=None)
        holder._remember_popup = lambda **kwargs: EasyROB._remember_popup(holder, **kwargs)
        popup_state = EasyROB._track_open_popup(holder, title='WARNING!', text='Expected output CSV was not found.', buttons=['OK'], kind='warning', source='aqme_output')
        self.assertIs(holder.bot_active_popup, popup_state)
        self.assertIs(holder.bot_popup_context, popup_state)
        self.assertEqual(holder.bot_popup_context['source'], 'aqme_output')
        self.assertNotIn('closed_at', holder.bot_popup_context)
        age = EasyROB.bot_popup_context_age_seconds.__get__(holder, EasyROB)
        self.assertIsInstance(age, float)
        self.assertGreaterEqual(age, 0.0)
        time.sleep(0.01)
        EasyROB._clear_active_popup(holder, popup_state)
        self.assertIsNone(holder.bot_active_popup)
        self.assertIs(holder.bot_popup_context, popup_state)
        self.assertIn('closed_at', holder.bot_popup_context)

    def test_local_ai_decline_falls_back_to_heuristic_for_same_question(self):
        local_manager = Mock()
        local_manager.is_model_ready.return_value = False
        local_manager.model_exists.return_value = False
        self.holder.bot_history = []
        self.holder.bot_worker = None
        self.holder.local_model_worker = None
        self.holder.bot_engine = SimpleNamespace(_local_manager=local_manager)
        self.holder._pending_local_bot_question = None
        self.holder._ask_to_prepare_local_model = Mock(return_value=False)
        self.holder._prepare_local_model_for_bot = Mock()
        self.holder._start_bot_reply = Mock()
        self.bot_window.panel.mode_combo.setCurrentText('Local AI')
        EasyROB.handle_bot_question(self.holder, 'Why is ROBERT blocked?')
        self.holder._ask_to_prepare_local_model.assert_called_once()
        self.holder._prepare_local_model_for_bot.assert_not_called()
        self.assertEqual(self.holder._pending_local_bot_question, None)
        self.holder._start_bot_reply.assert_called_once_with('Why is ROBERT blocked?', mode_override='Heuristic')

    def test_local_ai_prepare_failure_shows_message_and_falls_back_to_heuristic(self):
        self.holder.bot_history = []
        self.holder.bot_worker = None
        self.holder.local_model_worker = Mock()
        self.holder._pending_local_bot_question = 'Why is ROBERT blocked?'
        self.holder._start_bot_reply = Mock()
        self.holder.bot_panel.append_message = Mock()
        self.holder.bot_panel.set_busy = Mock()
        self.holder.bot_panel.set_local_progress_active = Mock()
        self.holder.bot_panel.set_local_status = Mock()
        EasyROB._on_local_model_prepare_failed(self.holder, 'disk full')
        self.assertEqual(self.holder._pending_local_bot_question, None)
        self.holder.bot_panel.append_message.assert_called_once()
        message = self.holder.bot_panel.append_message.call_args.args[1]
        self.assertIn('could not be prepared', message)
        self.assertIn('disk full', message)
        self.holder._start_bot_reply.assert_called_once_with('Why is ROBERT blocked?', mode_override='Heuristic')
if __name__ == '__main__':
    unittest.main()""", ('TestMainWindowBotWindow',), ())

# Selected checks from test_bot_results_workspace.py
_load_case_group('results_workspace', """\"\"\"Navigation and availability checks for the consolidated Results view.\"\"\"
from PySide6.QtWidgets import QApplication, QStackedWidget, QWidget
from gui_easyrob.tabs.results_workspace import ResultsWorkspace
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.heuristics import diagnose_snapshot

def make_workspace():
    app = QApplication.instance() or QApplication([])
    report, predictions, images, plots = (QWidget(), QWidget(), QWidget(), QWidget())
    workspace = ResultsWorkspace(report, predictions, images, plots)
    return (workspace, report, predictions, images, plots)

def test_main_window_has_one_results_tab_with_existing_views(tmp_path):
    app = QApplication.instance() or QApplication([])
    from gui_easyrob.main.window import EasyROB
    window = EasyROB()
    try:
        names = [window.tab_widget.tabText(index) for index in range(window.tab_widget.count())]
        assert 'Results' in names
        assert 'Check model' in names
        assert 'Evaluate' not in names
        assert names.index('Check model') + 1 == names.index('Results')
        assert window.tab_widget.widget(names.index('Check model')) is window.evaluate_tab
        assert 'Reports' not in names
        assert 'Predictions' not in names
        assert 'Images' not in names
        assert window.results_workspace.content.widget(0) is window.results_tab
        assert window.results_workspace.content.widget(1) is window.predictions_tab
        assert window.results_workspace.content.widget(2) is window.images_tab
        assert window.results_workspace.content.widget(3) is window.interactive_plots
        results_index = window.tab_widget.indexOf(window.results_workspace)
        assert not window.tab_widget.isTabEnabled(results_index)
        window.predictions_tab.availabilityChanged.emit(True)
        assert window.results_workspace.buttons['Predictions'].isEnabled()
        assert window.tab_widget.isTabEnabled(results_index)
        window.predictions_tab.availabilityChanged.emit(False)
        assert not window.tab_widget.isTabEnabled(results_index)
        source = tmp_path / 'input.csv'
        source.write_text('target\\n1\\n')
        report = tmp_path / 'ROBERT_report_No_PFI.pdf'
        report.write_bytes(b'placeholder')
        predict_dir = tmp_path / 'PREDICT'
        predict_dir.mkdir()
        (predict_dir / 'figure.png').write_bytes(b'placeholder')
        window.file_path = str(source)
        window.check_for_pdfs(str(source))
        window.check_for_images(str(source))
        assert window.results_workspace.buttons['Report'].isEnabled()
        assert window.results_workspace.buttons['Images'].isEnabled()
        window.show_result_view('Report')
        assert window.tab_widget.currentWidget() is window.results_workspace
        assert window.results_workspace.content.currentWidget() is window.results_tab
        from gui_easyrob.bot.bot_context import build_gui_snapshot
        window._result_view_source_path = str(source)
        snapshot = build_gui_snapshot(window)
        assert snapshot.active_tab == 'Results'
        assert snapshot.main_csv_path == str(source)
        report.unlink()
        window.check_for_pdfs(str(source))
        assert window.results_workspace.content.currentWidget() is window.images_tab
        (predict_dir / 'figure.png').unlink()
        predict_dir.rmdir()
        window.check_for_images(str(source))
        assert not window.tab_widget.isTabEnabled(results_index)
    finally:
        window.close()

def test_changing_run_keeps_results_open_when_another_view_is_available(tmp_path):
    app = QApplication.instance() or QApplication([])
    from gui_easyrob.main.window import EasyROB
    report_run = tmp_path / 'report_run'
    image_run = tmp_path / 'image_run'
    report_run.mkdir()
    image_run.mkdir()
    report_source = report_run / 'input.csv'
    image_source = image_run / 'input.csv'
    report_source.write_text('target\\n1\\n')
    image_source.write_text('target\\n1\\n')
    (report_run / 'ROBERT_report_No_PFI.pdf').write_bytes(b'placeholder')
    (image_run / 'PREDICT').mkdir()
    (image_run / 'PREDICT' / 'figure.png').write_bytes(b'placeholder')
    window = EasyROB()
    try:
        window._pending_refresh_path = str(report_source)
        window._execute_refresh_tabs()
        window.tab_widget.setCurrentWidget(window.results_workspace)
        assert window.results_workspace.content.currentWidget() is window.results_tab
        window._pending_refresh_path = str(image_source)
        window._execute_refresh_tabs()
        assert window.tab_widget.currentWidget() is window.results_workspace
        assert window.results_workspace.content.currentWidget() is window.images_tab
    finally:
        window.close()

def test_prediction_csv_enables_interactive_plots_without_external_predictions(tmp_path):
    app = QApplication.instance() or QApplication([])
    from gui_easyrob.main.window import EasyROB
    selected = tmp_path / 'input.csv'
    selected.write_text('target\\n1\\n')
    predict = tmp_path / 'PREDICT'
    predict.mkdir()
    (predict / 'GB_No_PFI.csv').write_text('target,target_pred\\n1,2\\n')
    window = EasyROB()
    try:
        window._pending_refresh_path = str(selected)
        window._execute_refresh_tabs()
        assert window.results_workspace.buttons['Interactive plots'].isEnabled()
        assert not window.results_workspace.buttons['Predictions'].isEnabled()
        assert not window.results_workspace.buttons['Images'].isEnabled()
        assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_workspace))
        assert window.results_workspace.show_view('Interactive plots')
        assert window.results_workspace.content.currentWidget() is window.interactive_plots
        (predict / 'GB_No_PFI.csv').unlink()
        window._execute_refresh_tabs()
        assert not window.results_workspace.buttons['Interactive plots'].isEnabled()
        assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_workspace))
    finally:
        window.close()""", ('test_main_window_has_one_results_tab_with_existing_views', 'test_changing_run_keeps_results_open_when_another_view_is_available', 'test_prediction_csv_enables_interactive_plots_without_external_predictions'), ())
