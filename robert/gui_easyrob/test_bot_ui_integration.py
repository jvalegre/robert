"""Integration checks for bot context across the desktop workspaces."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from gui_easyrob.bot.bot_context import GuiSnapshot, build_gui_snapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.answer_metadata import EvidenceOrigin, WebSearchMode
from gui_easyrob.bot.llm_providers import ProviderAnswer, build_llm_prompts
from gui_easyrob.bot.token_usage import TokenUsage
from gui_easyrob.bot.workflow_results import (
    WorkflowResultSnapshot,
    WorkflowStageResult,
    render_workflow_answer,
    render_workflow_evidence,
)
from gui_easyrob.main.window import EasyROB


class RecordingProvider:
    spec = SimpleNamespace(supports_web_search=True)

    def __init__(self):
        self.calls = []

    def generate_answer_with_usage(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderAnswer("The recorded test RMSE is 4.2.", TokenUsage())


@pytest.mark.parametrize(
    ("classic_tab", "stale_web_page", "expected_workflow", "expected_csv"),
    (
        ("ROBERT", "CheckML", "Full Workflow", "C:/runs/robert.csv"),
        ("CheckML", "Workflow", "CHECKML", "C:/runs/checkml.csv"),
    ),
)
def test_bot_uses_visible_classic_tab_instead_of_stale_web_page(
    classic_tab, stale_web_page, expected_workflow, expected_csv,
):
    app = QApplication.instance() or QApplication([])
    window = EasyROB()
    classic = object()
    web = object()
    store_calls = []
    expected_value = "4.2" if classic_tab == "ROBERT" else "8.7"

    class RecordingStore:
        def refresh(self, path, test_path, process_running=False):
            store_calls.append(path)
            source = "PREDICT/PREDICT_data.dat"
            return WorkflowResultSnapshot(
                kind="ROBERT", status="completed", root_name="run", summary_available=True,
                stages=(WorkflowStageResult(name="PREDICT", status="completed", source=source),),
                metrics=(("RMSE", expected_value, "Test", source),),
            )

    try:
        window.file_path = "C:/runs/robert.csv"
        window.checkml_tab.csv_path = "C:/runs/checkml.csv"
        window.tab_widget.setCurrentWidget(
            window.checkml_tab if classic_tab == "CheckML" else window.tab_widget.widget(0)
        )
        window.web_active_page = stale_web_page
        window._ui_stack = SimpleNamespace(currentWidget=lambda: classic)
        window._web_shell = SimpleNamespace(view=web)
        window.workflow_result_store = RecordingStore()

        window._refresh_workflow_result_snapshot()
        snapshot = build_gui_snapshot(window)
        answer = BotEngine().answer(
            "What was the test RMSE in my run?", snapshot, "Heuristic", "Test", "",
        )

        assert snapshot.active_tab == classic_tab
        assert snapshot.workflow == expected_workflow
        assert snapshot.main_csv_path == expected_csv
        assert store_calls == [expected_csv]
        assert expected_value in answer
    finally:
        del window._ui_stack
        del window._web_shell
        window.close()


def test_bot_uses_web_page_when_web_surface_is_visible():
    app = QApplication.instance() or QApplication([])
    window = EasyROB()
    classic = object()
    web = object()
    try:
        window.file_path = "C:/runs/robert.csv"
        window.checkml_tab.csv_path = "C:/runs/checkml.csv"
        window.web_active_page = "CheckML"
        window._ui_stack = SimpleNamespace(currentWidget=lambda: web)
        window._web_shell = SimpleNamespace(view=web)

        snapshot = build_gui_snapshot(window)

        assert snapshot.active_tab == "CheckML"
        assert snapshot.workflow == "CHECKML"
        assert snapshot.main_csv_path == "C:/runs/checkml.csv"
    finally:
        del window._ui_stack
        del window._web_shell
        window.close()


def test_bot_uses_checkml_run_when_results_view_points_to_checkml():
    app = QApplication.instance() or QApplication([])
    window = EasyROB()
    try:
        window.tab_widget.setTabEnabled(window.tab_widget.indexOf(window.results_workspace), True)
        window.tab_widget.setCurrentWidget(window.results_workspace)
        window.file_path = "C:/runs/robert.csv"
        window.checkml_tab.csv_path = "C:/runs/checkml.csv"
        window._result_view_source_path = "C:/runs/checkml.csv"
        window._result_view_source_kind = "CHECKML"

        snapshot = build_gui_snapshot(window)

        assert snapshot.active_tab == "Results"
        assert snapshot.workflow == "CHECKML"
        assert snapshot.main_csv_path == "C:/runs/checkml.csv"
    finally:
        window.close()


@pytest.mark.parametrize(
    ("workflow", "active_tab", "question", "expected_tutorial"),
    (
        ("Full Workflow", "ROBERT", "How do I run this workflow?", "Overview tutorial summary"),
        ("CHECKML", "CheckML", "How do I run this workflow?", "Check ML tutorial summary"),
        ("Full Workflow", "ROBERT", "¿Cómo ejecuto este workflow?", "Overview tutorial summary"),
        ("CHECKML", "CheckML", "¿Cómo ejecuto este workflow?", "Check ML tutorial summary"),
    ),
)
def test_contextual_workflow_guidance_follows_selected_workspace(
    workflow, active_tab, question, expected_tutorial,
):
    snapshot = GuiSnapshot(
        active_tab, workflow, "", "", "", "regression",
        False, True, False, "", [],
    )

    answer = BotEngine().answer(
        question, snapshot, "Heuristic", "Test", "",
    )

    assert expected_tutorial in answer
    assert "Enable AQME Workflow" not in answer


def test_verify_explanation_in_checkml_does_not_invent_model_search():
    snapshot = GuiSnapshot(
        "CheckML", "CHECKML", "", "", "", "regression",
        False, True, False, "", [],
    )

    answer = BotEngine().answer(
        "What does VERIFY do here?", snapshot, "Heuristic", "Test", "",
    )
    system_prompt, _ = build_llm_prompts(
        "What does VERIFY do here?", snapshot, (), question_intent="module-explanation",
    )

    assert "CheckML" in answer
    assert "selected model" in answer
    assert "after GENERATE" not in answer
    assert "When the current GUI workflow is CHECKML" in system_prompt


def test_spanish_verify_explanation_in_checkml_uses_checkml_context():
    snapshot = GuiSnapshot(
        "CheckML", "CHECKML", "", "", "", "regression",
        False, True, False, "", [],
    )

    answer = BotEngine().answer(
        "¿Qué hace VERIFY aquí?", snapshot, "Heuristic", "Test", "",
    )

    assert "En CheckML" in answer
    assert "modelo seleccionado" in answer
    assert "after GENERATE" not in answer


@pytest.mark.parametrize("workflow", ("Full Workflow", "CHECKML"))
def test_bot_keeps_generic_explanations_separate_from_recorded_results(workflow):
    source = "PREDICT/PREDICT_data.dat"
    results = WorkflowResultSnapshot(
        kind="ROBERT", status="completed", root_name="sample", summary_available=True,
        stages=(WorkflowStageResult(name="PREDICT", status="completed", source=source),),
        metrics=(("RMSE", "4.2", "Test", source),),
    )
    snapshot = GuiSnapshot(
        "Results", workflow, "sample.csv", "", "target", "regression",
        False, True, False, "", [], workflow_results=results,
    )
    engine = BotEngine()

    generic = engine.answer("What is RMSE?", snapshot, "Heuristic", "Test", "")
    specific = engine.answer("What was the test RMSE in my run?", snapshot, "Heuristic", "Test", "")

    assert "4.2" not in generic
    assert "4.2" in specific
    assert source in specific


def test_result_metric_is_available_even_without_a_stage_summary():
    source = "PREDICT/PREDICT_data.dat"
    results = WorkflowResultSnapshot(
        kind="ROBERT", status="completed", root_name="sample", summary_available=True,
        metrics=(("RMSE", "4.2", "Test", source),),
    )

    answer = render_workflow_answer(results, "What was the test RMSE in my run?")
    evidence = render_workflow_evidence(results, "What was the test RMSE in my run?")

    assert "Test RMSE: **4.2**" in answer
    assert "Metric Test RMSE=4.2" in evidence
    assert "requested stage is not present" not in answer


@pytest.mark.parametrize(
    ("workflow", "active_tab", "tutorial_label"),
    (
        ("Full Workflow", "ROBERT", "Overview"),
        ("CHECKML", "CheckML", "Check ML"),
    ),
)
def test_cloud_bot_receives_the_selected_workflow_tutorial(
    workflow, active_tab, tutorial_label,
):
    provider = RecordingProvider()
    engine = BotEngine(provider_registry={"Test": provider})
    snapshot = GuiSnapshot(
        active_tab, workflow, "", "", "", "regression",
        False, True, False, "", [],
    )

    engine.answer_with_metadata(
        "How do I run this workflow?", snapshot, "Cloud AI", "Test", "test-key",
    )

    assert len(provider.calls) == 1
    assert provider.calls[0]["gui_context"].workflow == workflow
    assert provider.calls[0]["tutorial_label"] == tutorial_label
    assert provider.calls[0]["web_search_mode"] is WebSearchMode.OFF


def test_cloud_bot_uses_local_result_evidence_for_a_metric_question():
    provider = RecordingProvider()
    engine = BotEngine(provider_registry={"Test": provider})
    source = "PREDICT/PREDICT_data.dat"
    results = WorkflowResultSnapshot(
        kind="ROBERT", status="completed", root_name="sample", summary_available=True,
        stages=(WorkflowStageResult(name="PREDICT", status="completed", source=source),),
        metrics=(("RMSE", "4.2", "Test", source),),
    )
    snapshot = GuiSnapshot(
        "Results", "Full Workflow", "sample.csv", "", "target", "regression",
        False, True, False, "", [], workflow_results=results,
    )

    answer = engine.answer_with_metadata(
        "What was the test RMSE in my run?", snapshot, "Cloud AI", "Test", "test-key",
    )

    assert len(provider.calls) == 1
    assert provider.calls[0]["gui_context"].workflow_results.metrics == results.metrics
    assert provider.calls[0]["web_search_mode"] is WebSearchMode.OFF
    assert answer.metadata.evidence_origin is EvidenceOrigin.LOCAL_SYSTEM
