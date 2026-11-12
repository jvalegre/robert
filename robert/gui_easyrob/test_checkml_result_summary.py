"""Regression checks for the CheckML result summary action."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui_easyrob.main.window import EasyROB
from gui_easyrob.bot.workflow_results import (
    WorkflowReportAssessment,
    WorkflowResultSnapshot,
    WorkflowStageResult,
    render_workflow_answer,
)
from gui_easyrob.web_shell import WorkflowBridge


def test_checkml_result_snapshot_refreshes_without_enabling_bot(tmp_path):
    app = QApplication.instance() or QApplication([])
    csv_path = tmp_path / "model_data.csv"
    csv_path.write_text("code_name,target,descriptor\nfirst,1.0,2.0\n", encoding="utf-8")
    verify_dir = tmp_path / "VERIFY"
    verify_dir.mkdir()
    (verify_dir / "VERIFY_data.dat").write_text("VERIFY completed\n", encoding="utf-8")

    window = EasyROB(mode="checkml")
    try:
        window.checkml_tab.set_csv_path(str(csv_path))
        assert not window.bot_panel.workflow_summary_button.isEnabled()
        assert window.workflow_result_snapshot is not None
        assert window.workflow_result_snapshot.summary_available
        bridge = WorkflowBridge(window)
        try:
            assert not bridge._state()["bot"]["canSummarize"]
        finally:
            bridge._timer.stop()
    finally:
        window.close()


def test_checkml_only_summary_cannot_dispatch_bot(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    csv_path = tmp_path / "model_data.csv"
    csv_path.write_text("code_name,target,descriptor\nfirst,1.0,2.0\n", encoding="utf-8")
    verify_dir = tmp_path / "VERIFY"
    verify_dir.mkdir()
    (verify_dir / "VERIFY_data.dat").write_text("VERIFY completed\n", encoding="utf-8")

    window = EasyROB(mode="checkml")
    try:
        window.checkml_tab.set_csv_path(str(csv_path))
        window.bot_panel.mode_combo.setCurrentText("Cloud AI")
        window.bot_panel.api_key_edit.clear()
        calls = []
        monkeypatch.setattr(window, "_start_bot_reply", lambda question, mode_override=None: calls.append(mode_override))
        monkeypatch.setattr(window, "handle_bot_question", lambda question: calls.append("cloud"))
        window._handle_workflow_summary_requested()
        assert calls == []
        window.bot_panel.api_key_edit.setText("configured-key")
        window._handle_workflow_summary_requested()
        assert calls == []
    finally:
        window.close()


def test_checkml_result_snapshot_refreshes_after_run_without_bot(tmp_path):
    app = QApplication.instance() or QApplication([])
    csv_path = tmp_path / "model_data.csv"
    csv_path.write_text("code_name,target,descriptor\nfirst,1.0,2.0\n", encoding="utf-8")

    window = EasyROB(mode="checkml")
    try:
        window.checkml_tab.set_csv_path(str(csv_path))
        assert not window.bot_panel.workflow_summary_button.isEnabled()
        verify_dir = tmp_path / "VERIFY"
        verify_dir.mkdir()
        (verify_dir / "VERIFY_data.dat").write_text("VERIFY completed\n", encoding="utf-8")
        window.checkml_tab._refresh_sibling_tabs()
        assert window.workflow_result_snapshot is not None
        assert window.workflow_result_snapshot.summary_available
        assert not window.bot_panel.workflow_summary_button.isEnabled()
    finally:
        window.close()


def test_full_summary_includes_workflow_when_prompt_mentions_prediction_quality():
    snapshot = WorkflowResultSnapshot(
        kind="ROBERT",
        status="completed",
        root_name="example",
        summary_available=True,
        stages=(WorkflowStageResult(name="VERIFY", status="completed", source="VERIFY_data.dat"),),
        report_assessments=(
            WorkflowReportAssessment(
                variant="No PFI (MVL)",
                model="MVL",
                prediction_type="regression",
                interpolation_score=0,
                boundary_score=6,
            ),
        ),
    )

    answer = render_workflow_answer(
        snapshot,
        "Summarize the ROBERT report and do not use universal thresholds to call the model good or bad.",
        full=True,
    )

    assert "**Workflow result:**" in answer
    assert "**Stage results**" in answer
    assert "**ROBERT report assessment**" in answer
