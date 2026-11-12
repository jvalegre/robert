"""Focused behavior check for the modern dashboard's Python bridge."""

import json
import sys

from PySide6.QtCore import QCoreApplication, QEvent, Signal
from PySide6.QtWidgets import QApplication, QWidget

from gui_easyrob.main.window import EasyROB
from gui_easyrob.bot.bot_context import build_gui_snapshot
from gui_easyrob.web_shell import WorkflowBridge
import gui_easyrob.main.window as window_module


class _Signal:
    def connect(self, callback):
        pass


class _Profile:
    downloadRequested = _Signal()


class _Page:
    def profile(self):
        return _Profile()


class _InertWebView(QWidget):
    urlChanged = Signal(object)

    def setUrl(self, url):
        self.url = url

    def page(self):
        return _Page()


def test_workflow_bridge_updates_existing_controls(monkeypatch):
    app = QApplication.instance() or QApplication([])
    molssi_module = sys.modules[window_module.MolSSIDatabasesTab.__module__]
    monkeypatch.setattr(molssi_module, "QWebEngineView", _InertWebView)
    window = EasyROB()
    try:
        bridge = WorkflowBridge(window)
        initial = json.loads(bridge.getState())
        assert initial["groups"]["workflow"]
        assert initial["groups"]["general"]
        assert not initial["actions"]["aqme"]
        updates = []
        bridge.logChanged.connect(lambda payload: updates.append(json.loads(payload)))
        window.console_output.setPlainText("A" * 13001)
        window.checkml_tab.console_output.setPlainText("B" * 8001)
        bridge.push_state()
        assert len(json.loads(bridge.getState())["log"]) == 13001
        assert len(json.loads(bridge.getState())["checkmlLog"]) == 8001
        assert {update["kind"] for update in updates} == {"workflow", "checkml"}
        window.console_output.append("New line")
        bridge.push_state()
        assert updates[-1]["kind"] == "workflow"
        assert updates[-1]["offset"] == 13001
        assert updates[-1]["text"] == "\nNew line"
        window.console_output.clear()
        bridge.push_state()
        assert updates[-1]["replace"]
        assert updates[-1]["text"] == ""
        bridge.setField("workflow.type_dropdown", "Classification")
        bridge.setField("general.seed", "17")
        bridge.setField("workflow.all_models_toggle", "false")
        assert window.type_dropdown.currentText() == "Classification"
        assert window.options_tab.seed.text() == "17"
        assert not window.all_models_toggle.isChecked()
        monkeypatch.setattr(window_module.QMessageBox, "information", lambda *args: None)
        bridge.setField("workflow.aqme_workflow", "true")
        assert json.loads(bridge.getState())["actions"]["aqme"]
        bridge.setField("workflow.aqme_workflow", "false")
        assert not json.loads(bridge.getState())["actions"]["aqme"]

        window.available_list.addItem("exclude_me")
        bridge.performAction("ignore", "exclude_me")
        assert window.ignore_list.item(0).text() == "exclude_me"
        bridge.performAction("restore", "exclude_me")
        assert window.available_list.item(0).text() == "exclude_me"

        window.checkml_tab.csv_path = "C:/runs/checkml.csv"
        bridge.performAction("activePage", "checkml")
        snapshot = build_gui_snapshot(window)
        assert snapshot.active_tab == "CheckML"
        assert snapshot.main_csv_path == "C:/runs/checkml.csv"
        bridge.performAction("activePage", "results")
        bridge.performAction("activeResultView", "Interactive plots")
        snapshot = build_gui_snapshot(window)
        assert snapshot.active_tab == "Results"
        assert snapshot.result_active_view == "Interactive plots"
        bridge.performAction("activePage", "workflow")
        assert build_gui_snapshot(window).active_tab == "Workflow"
        assert window.web_active_result_view == ""
    finally:
        window.results_tab.shared_pool.waitForDone()
        window.predictions_tab._thread_pool.waitForDone()
        app.removeEventFilter(window)
        window.hide()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
