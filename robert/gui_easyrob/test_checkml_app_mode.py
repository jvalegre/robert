"""Checks for the dedicated CheckML desktop entry point."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QLabel, QTabWidget

from gui_easyrob.main.window import EasyROB
from gui_easyrob.main.window import SOFTWARE_VERSIONS
from gui_easyrob.bot.bot_context import build_gui_snapshot
from gui_easyrob.web_shell import WorkflowBridge
from gui_easyrob import checkml_launcher


def test_checkml_mode_limits_navigation_and_tutorial():
    app = QApplication.instance() or QApplication([])
    window = EasyROB(mode="checkml")
    try:
        assert window.windowTitle() == "CheckML"
        assert [window.tab_widget.tabText(index) for index in range(window.tab_widget.count())] == [
            "CheckML", "Results",
        ]
        assert window.tab_widget.currentWidget() is window.checkml_tab
        results_index = window.tab_widget.indexOf(window.results_workspace)
        assert not window.tab_widget.isTabEnabled(results_index)
        window.results_workspace.set_available("Report", True)
        assert window.tab_widget.isTabEnabled(results_index)
        assert window.tutorial_topics() == (("check_model", "Check ML"),)
        assert window.bot_panel.suggested_combo.count() == 2
        assert "CheckML" in window.bot_panel.subtitle_label.text()

        window.web_active_page = "Results"
        assert build_gui_snapshot(window).workflow == "CHECKML"
        window.web_active_page = "CheckML"

        window.show_tutorial_dialog()
        app.processEvents()
        tutorial_tabs = window.tutorial_dialog.findChildren(QTabWidget)
        assert len(tutorial_tabs) == 1
        assert tutorial_tabs[0].count() == 1
        assert tutorial_tabs[0].tabText(0) == "Check ML"

        bridge = WorkflowBridge(window)
        try:
            assert bridge.getTutorials().count('"id": "check_model"') == 1
            assert '"id": "overview"' not in bridge.getTutorials()
        finally:
            bridge._timer.stop()
    finally:
        window.close()


def test_easyrob_mode_keeps_existing_navigation():
    topics = EasyROB.tutorial_topics(SimpleNamespace(app_mode="easyrob"))
    assert ("check_model", "Check ML") in topics
    assert len(topics) > 1


def test_checkml_entry_selects_dedicated_mode(monkeypatch):
    modes = []
    monkeypatch.setattr(checkml_launcher, "launch_gui", lambda mode: modes.append(mode))
    checkml_launcher.main()
    assert modes == ["checkml"]


def test_checkml_only_mode_does_not_expose_or_dispatch_bot(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = EasyROB(mode="checkml")
    calls = []
    monkeypatch.setattr(window, "_start_bot_reply", lambda question, **kwargs: calls.append(question))
    monkeypatch.setattr(window, "_ensure_bot_welcome_message", lambda: calls.append("welcome"))
    bridge = WorkflowBridge(window)
    try:
        assert window.tab_widget.cornerWidget().isHidden()
        assert not window.bot_toggle_btn.isEnabled()
        window.handle_bot_question("How do I run this workflow?")
        bridge.performAction("botOpen", "")
        bridge.performAction("openBot", "")
        bridge.setBotDraft("How do I run this workflow?")
        assert not window.bot_toggle_btn.isChecked()
        assert not window.bot_panel.input_edit.text()
        assert calls == []
    finally:
        bridge._timer.stop()
        window.close()


def test_embedded_checkml_keeps_the_robert_bot():
    app = QApplication.instance() or QApplication([])
    window = EasyROB()
    try:
        window.tab_widget.setCurrentWidget(window.checkml_tab)
        assert window.bot_toggle_btn.isEnabled()
        assert not window.tab_widget.cornerWidget().isHidden()
    finally:
        window.close()


def test_checkml_version_dialog_shows_its_identity_and_relevant_dependencies(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = EasyROB(mode="checkml")
    captured = {}

    def inspect_dialog(dialog):
        captured["title"] = dialog.windowTitle()
        captured["labels"] = [label.text() for label in dialog.findChildren(QLabel)]
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", inspect_dialog)
    try:
        window.show_version_dialog()
        assert captured["title"] == "About CheckML"
        assert any("CheckML" in text for text in captured["labels"])
        assert f"Version {SOFTWARE_VERSIONS['Dependencies']['ROBERT']}" in captured["labels"]
        dependencies = next(text for text in captured["labels"] if "<table" in text)
        assert "Python" in dependencies
        assert "ROBERT" in dependencies
        assert "AQME" not in dependencies
        assert "xtb" not in dependencies
        assert "OpenBabel" not in dependencies
    finally:
        window.close()
