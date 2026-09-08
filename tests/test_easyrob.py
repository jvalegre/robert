#!/usr/bin/env python
"""
End-to-end and widget-level tests for the easyROB main GUI.

This module:

- Configures an offscreen Qt backend for headless testing.
- Provides fixtures for the main EasyROB window and shared output folder.
- Contains fast sanity tests for widgets, tabs, and basic behaviour.
- Contains heavier end-to-end workflows that launch the real ROBERT
  subprocess from the GUI and inspect the resulting folders/PDFs.
"""

import os
import sys
import time
import shutil
from pathlib import Path

# ----------------------------------------------------------------------
# Qt backend – MUST be set before importing PySide6
# ----------------------------------------------------------------------
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ----------------------------------------------------------------------
# Make project importable (GUI_easyROB.easyrob, etc.)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party imports
import pandas as pd
import pytest
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtWidgets import (
    QListWidgetItem,
    QMessageBox,
    QDialog,
    QFileDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)
from rdkit import Chem

# Local project imports
from robert.gui_easyrob.main.window import EasyROB
import robert.gui_easyrob.easyrob as easyrob_module
import robert.gui_easyrob.main.window as window_module
import robert.gui_easyrob.tabs.aqme as aqme_module
import robert.gui_easyrob.tabs.predictions as predictions_module
import robert.gui_easyrob.tabs.results as results_module

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
TEST_OUTPUT_DIR_NAME = "test-easyrob-output"

WORKFLOW_MAX_WAIT_S = 240.0
WORKFLOW_POLL_INTERVAL_S = 0.5
STOP_MAX_WAIT_S = 60.0

SCENARIO_CONFIG = {
    "regression": {
        "analysis_type": "Regression",
        "workflow": "Full Workflow",
        "seed": "42",
        "kfold": "2",
        "description": "Baseline regression full workflow",
    },
    "existing_dirs_stop": {
        "analysis_type": "Regression",
        "workflow": "Full Workflow",
        "seed": "42",
        "kfold": "2",
        "description": "Re-run with existing folders and manual stop",
    },
    "aqme_regression": {
        "analysis_type": "Regression",
        "workflow": "Full Workflow",
        "seed": "42",
        "kfold": "2",
        "description": "Regression full workflow with AQME mapping",
    },
}


def install_message_box_stubs(monkeypatch, question_handler=None):
    """Patch QMessageBox methods to keep tests non-blocking and record calls."""
    calls = {"info": [], "question": []}

    def info_stub(*args, **kwargs):
        title = args[1] if len(args) > 1 else ""
        text = args[2] if len(args) > 2 else ""
        print(f"[QMessageBox.information/warning] {title}: {text}")
        calls["info"].append((title, text))
        return QMessageBox.Ok

    def question_stub(
        parent,
        title,
        text,
        buttons=QMessageBox.Yes | QMessageBox.No,
        default=QMessageBox.No,
    ):
        print(f"[QMessageBox.question] {title}: {text}")
        calls["question"].append((title, text))
        if question_handler is not None:
            return question_handler(title, text, buttons, default)
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "information", info_stub)
    monkeypatch.setattr(QMessageBox, "warning", info_stub)
    monkeypatch.setattr(QMessageBox, "question", question_stub)
    return calls


def print_scenario_banner(title, scenario_id=None):
    """Print a consistent test scenario header."""
    print("\n" + "=" * 80)
    print(f"TEST SCENARIO: {title}")
    if scenario_id is not None:
        print(f"Scenario ID: {scenario_id}")
    print("=" * 80)


def copy_example_csv(filename, destination_dir):
    """Copy a test CSV into the shared output directory and return its new path."""
    source_path = Path(__file__).with_name(filename)
    assert source_path.exists(), f"Example CSV not found: {source_path}"
    copied_path = destination_dir / source_path.name
    shutil.copyfile(source_path, copied_path)
    return copied_path


def clean_output_paths(base_dir, directory_names=(), file_patterns=()):
    """Remove selected directories and glob-matched files from a test output folder."""
    for name in directory_names:
        path = base_dir / name
        if path.is_dir():
            shutil.rmtree(path)

    for pattern in file_patterns:
        for path in base_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def dump_console_output(title, text):
    """Print console output with a clear, reusable wrapper."""
    print(f"\n{title}")
    print("----- BEGIN CONSOLE OUTPUT -----")
    print(text.encode("utf-8", errors="replace").decode())
    print("----- END CONSOLE OUTPUT -----")


def process_events_until(predicate, timeout_s, poll_interval_s=WORKFLOW_POLL_INTERVAL_S):
    """Process Qt events until a condition becomes true or the timeout expires."""
    elapsed = 0.0
    while elapsed < timeout_s:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(poll_interval_s)
        elapsed += poll_interval_s
    return False


def wait_for_workflow_start(window, baseline_text, timeout_s=60.0):
    """Wait until the ROBERT subprocess is running and the console starts updating."""

    def workflow_started():
        current_console = window.console_output.toPlainText()
        process = getattr(window.worker, "process", None)
        return (
            process is not None
            and process.poll() is None
            and len(current_console) > len(baseline_text)
        )

    started = process_events_until(workflow_started, timeout_s)
    if started:
        print("[OK] Workflow started; console output detected and process is running")
    return started


def wait_for_workflow_completion(
    window,
    output_dir,
    expected_dirs,
    report_pdf,
    initial_console_text,
    timeout_s=WORKFLOW_MAX_WAIT_S,
):
    """Wait until ROBERT finishes and all expected outputs are materialized."""
    workflow_started = False
    elapsed = 0.0
    last_console = initial_console_text
    last_process = None

    while elapsed < timeout_s:
        QCoreApplication.processEvents()
        current_console = window.console_output.toPlainText()
        process = getattr(window.worker, "process", None)
        last_console = current_console
        last_process = process

        if not workflow_started and len(current_console) > len(initial_console_text):
            workflow_started = True
            print("[OK] Workflow started (console output detected)")

        all_dirs_exist = all((output_dir / name).is_dir() for name in expected_dirs)
        pdf_exists = report_pdf.is_file()
        if workflow_started and all_dirs_exist and pdf_exists:
            print("[OK] Workflow completed (all output folders AND report PDF detected)")
            return True, workflow_started, last_console, last_process

        if process is not None and process.poll() is not None and not (all_dirs_exist and pdf_exists):
            print(
                f"[WARN] Process exited with code {process.returncode} "
                "but not all outputs (folders + PDF) are present yet."
            )
            return False, workflow_started, last_console, last_process

        time.sleep(WORKFLOW_POLL_INTERVAL_S)
        elapsed += WORKFLOW_POLL_INTERVAL_S

    return False, workflow_started, last_console, last_process


def run_full_workflow_and_wait(window, qtbot, output_dir, expected_dirs, report_pdf):
    """Launch ROBERT from the GUI and wait until the full workflow completes."""
    baseline_text = window.console_output.toPlainText()
    qtbot.mouseClick(window.run_button, Qt.LeftButton)
    assert not window.run_button.isEnabled()

    completed, started, last_console, last_process = wait_for_workflow_completion(
        window=window,
        output_dir=output_dir,
        expected_dirs=expected_dirs,
        report_pdf=report_pdf,
        initial_console_text=baseline_text,
    )
    if not started:
        dump_console_output("[DEBUG] Console at timeout (no start detected):", last_console)
        pytest.fail("Workflow did not start within timeout")
    if not completed:
        print("\n[DEBUG] Console at timeout (no completion detected):")
        print("Existing entries in output_dir:", [path.name for path in output_dir.iterdir()])
        print("Report PDF exists:", report_pdf.is_file())
        if last_process is not None:
            print("Process return code:", last_process.returncode)
        dump_console_output("[DEBUG] Timed out console snapshot:", last_console)
        pytest.fail("Workflow did not complete within timeout")

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_output_dir():
    """
    Shared output directory for end-to-end tests.

    Behaviour:
    - Always start the pytest session with a clean folder.
    - Optionally keep the folder at the end if EASYROB_KEEP_TEST_OUTPUT=1.
    """
    base_dir = Path(__file__).resolve().parent  # tests/ directory
    out_dir = base_dir / TEST_OUTPUT_DIR_NAME

    # Read debug flag from environment
    keep_after = os.getenv("EASYROB_KEEP_TEST_OUTPUT", "0") == "1"

    # Always start from a clean state
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Provide the directory to tests
    yield out_dir

    # Clean up at the end, unless we are in debug mode
    if keep_after:
        print(f"[DEBUG] Keeping test output dir: {out_dir}")
        return

    try:
        if out_dir.exists():
            shutil.rmtree(out_dir)
    except Exception as exc:
        print(f"[WARN] Could not remove test output dir {out_dir}: {exc}")


@pytest.fixture
def easyrob_window(qtbot, monkeypatch):
    """
    Create an EasyROB main window for GUI tests.

    Heavy background checks that are not relevant for tests are patched out.
    """
    window = EasyROB()
    qtbot.addWidget(window)

    # Avoid slow or environment-dependent checks during tests
    monkeypatch.setattr(window, "check_for_pdfs", lambda *args, **kwargs: None)
    monkeypatch.setattr(window, "check_for_images", lambda *args, **kwargs: None)
    monkeypatch.setattr(window, "check_aqme_workflow", lambda *args, **kwargs: None)

    return window


# =====================================================
# Basic Initialization Tests
# =====================================================


def test_easyrob_window_starts(easyrob_window):
    """Main window initializes without crashing and key widgets exist."""
    window = easyrob_window

    assert "easyROB" in window.windowTitle()
    assert window.available_list is not None
    assert window.ignore_list is not None
    assert window.run_button is not None
    assert window.csv_test_label is not None
    assert window.file_label is not None
    assert window.tab_widget is not None


def test_easyrob_factory_returns_main_window_class():
    """The lightweight factory module returns the real main window class."""
    assert easyrob_module.get_main_window_class() is EasyROB

def test_all_tabs_created(easyrob_window):
    """All expected top-level tabs are present."""
    window = easyrob_window

    tab_names = [window.tab_widget.tabText(i) for i in range(window.tab_widget.count())]

    assert "ROBERT" in tab_names
    assert "AQME" in tab_names
    assert "Advanced Options" in tab_names
    assert "MolSSI Databases" in tab_names
    assert "Reports" in tab_names
    assert "Images" in tab_names
    assert "Predictions" in tab_names


def test_dropdowns_populated(easyrob_window):
    """Analysis type dropdown is created and properly initialized."""
    window = easyrob_window

    assert window.type_dropdown.count() == 2
    items = [window.type_dropdown.itemText(i) for i in range(window.type_dropdown.count())]
    assert "Regression" in items
    assert "Classification" in items


# =====================================================
# Column Selection Tests
# =====================================================


def test_move_to_selected_and_back(easyrob_window):
    """Items move correctly between available and ignored lists."""
    window = easyrob_window

    # Seed available_list
    for name in ["col1", "col2", "col3"]:
        window.available_list.addItem(QListWidgetItem(name))

    assert window.available_list.count() == 3
    assert window.ignore_list.count() == 0

    # Select and move two items
    for i in range(2):
        item = window.available_list.item(i)
        item.setSelected(True)

    window.move_to_selected()

    # Verify movement
    assert window.available_list.count() == 1
    assert window.ignore_list.count() == 2

    available_items = [
        window.available_list.item(i).text()
        for i in range(window.available_list.count())
    ]
    ignore_items = [
        window.ignore_list.item(i).text()
        for i in range(window.ignore_list.count())
    ]

    assert "col3" in available_items
    assert set(ignore_items) == {"col1", "col2"}

    # Move one back
    window.ignore_list.item(0).setSelected(True)
    window.move_to_available()

    assert window.available_list.count() == 2
    assert window.ignore_list.count() == 1


# =====================================================
# CSV Columns Loading Tests
# =====================================================


def test_load_csv_columns(easyrob_window, tmp_path):
    """CSV columns are loaded correctly into dropdowns and lists."""
    window = easyrob_window

    # Create a CSV with specific columns
    csv_path = tmp_path / "test_columns.csv"
    df = pd.DataFrame(
        {
            "ID": [1, 2],
            "Name": ["a", "b"],
            "Target": [10.5, 20.3],
            "Feature1": [1, 2],
        }
    )
    df.to_csv(csv_path, index=False)

    window.file_path = str(csv_path)
    window.load_csv_columns()

    # Check dropdowns
    y_items = [window.y_dropdown.itemText(i) for i in range(window.y_dropdown.count())]
    names_items = [
        window.names_dropdown.itemText(i) for i in range(window.names_dropdown.count())
    ]

    assert set(y_items) == {"ID", "Name", "Target", "Feature1"}
    assert set(names_items) == {"ID", "Name", "Target", "Feature1"}

    # Check available list
    available_items = [
        window.available_list.item(i).text()
        for i in range(window.available_list.count())
    ]
    assert set(available_items) == {"ID", "Name", "Target", "Feature1"}


def test_load_csv_columns_auto_ignores_smiles_and_prefers_code_name(easyrob_window, tmp_path):
    """SMILES is auto-ignored and code_name is auto-selected when present."""
    window = easyrob_window

    csv_path = tmp_path / "smiles_columns.csv"
    pd.DataFrame(
        {
            "SMILES": ["C", "CC"],
            "code_name": ["mol1", "mol2"],
            "target": [1.0, 2.0],
            "Feature1": [10, 20],
        }
    ).to_csv(csv_path, index=False)

    window.file_path = str(csv_path)
    window.load_csv_columns()

    available_items = {
        window.available_list.item(i).text()
        for i in range(window.available_list.count())
    }
    ignored_items = {
        window.ignore_list.item(i).text()
        for i in range(window.ignore_list.count())
    }

    assert "SMILES" not in available_items
    assert ignored_items == {"SMILES"}
    assert window.names_dropdown.currentText() == "code_name"


def test_set_file_path_updates_ui_and_skips_redundant_reload(easyrob_window, tmp_path, monkeypatch):
    """set_file_path updates labels and avoids reloading unchanged files unless forced."""
    window = easyrob_window
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv_path, index=False)

    calls = {
        "load_csv_columns": 0,
        "refresh_tabs": 0,
        "check_molssi_descriptors": 0,
        "update_smiles": 0,
        "check_aqme_workflow": 0,
    }

    monkeypatch.setattr(window, "load_csv_columns", lambda: calls.__setitem__("load_csv_columns", calls["load_csv_columns"] + 1))
    monkeypatch.setattr(window, "refresh_tabs", lambda file_path: calls.__setitem__("refresh_tabs", calls["refresh_tabs"] + 1))
    monkeypatch.setattr(window, "_is_molssi_csv", lambda file_path: False)
    monkeypatch.setattr(window, "check_molssi_descriptors", lambda: calls.__setitem__("check_molssi_descriptors", calls["check_molssi_descriptors"] + 1))
    monkeypatch.setattr(window, "_update_unified_smiles_context", lambda: calls.__setitem__("update_smiles", calls["update_smiles"] + 1))
    monkeypatch.setattr(window, "check_aqme_workflow", lambda: calls.__setitem__("check_aqme_workflow", calls["check_aqme_workflow"] + 1))
    window.tab_widget_aqme.df_mapped_smiles = object()

    window.set_file_path(str(csv_path))

    assert window.file_path == str(csv_path)
    assert window.file_label.label.text() == f"Selected: {csv_path.name}"
    assert window.file_label.toolTip() == str(csv_path)
    assert window.tab_widget_aqme.df_mapped_smiles is None
    assert calls == {
        "load_csv_columns": 1,
        "refresh_tabs": 1,
        "check_molssi_descriptors": 1,
        "update_smiles": 1,
        "check_aqme_workflow": 1,
    }

    window.set_file_path(str(csv_path))
    assert calls["load_csv_columns"] == 1

    window.set_file_path(str(csv_path), force=True)
    assert calls["load_csv_columns"] == 2


def test_set_and_clear_csv_test_path_updates_ui(easyrob_window, tmp_path, monkeypatch):
    """Selecting and clearing the optional test CSV keeps labels and dependent hooks in sync."""
    window = easyrob_window
    csv_path = tmp_path / "external_test.csv"
    pd.DataFrame({"SMILES": ["C"], "target": [1.0]}).to_csv(csv_path, index=False)

    calls = {"update_smiles": 0, "check_aqme_workflow": 0, "refresh_tabs": 0}
    monkeypatch.setattr(window, "_update_unified_smiles_context", lambda: calls.__setitem__("update_smiles", calls["update_smiles"] + 1))
    monkeypatch.setattr(window, "check_aqme_workflow", lambda: calls.__setitem__("check_aqme_workflow", calls["check_aqme_workflow"] + 1))
    monkeypatch.setattr(window, "refresh_tabs", lambda file_path: calls.__setitem__("refresh_tabs", calls["refresh_tabs"] + 1))

    window.set_csv_test_path(str(csv_path))

    assert window.csv_test_path == str(csv_path)
    assert window.csv_test_label.label.text() == f"Selected: {csv_path.name}"
    assert window.csv_test_label.toolTip() == str(csv_path)
    assert not window.clear_test_button.isHidden()
    assert calls == {"update_smiles": 1, "check_aqme_workflow": 1, "refresh_tabs": 1}

    window.clear_test_file()

    assert window.csv_test_path is None
    assert window.csv_test_label.label.text() == "Drag & Drop a CSV external test file here (optional)"
    assert window.csv_test_label.toolTip() == ""
    assert window.clear_test_button.isHidden()
    assert calls["update_smiles"] == 2
    assert calls["check_aqme_workflow"] == 2


def test_reset_ui_after_process_restores_buttons(easyrob_window):
    """The post-process reset restores the expected idle GUI state."""
    window = easyrob_window

    window.run_button.setDisabled(True)
    window.run_aqme_button.setDisabled(True)
    window.stop_button.setDisabled(False)
    window.progress.setRange(0, 0)

    window._reset_ui_after_process()

    assert window.run_button.isEnabled()
    assert window.run_aqme_button.isEnabled()
    assert not window.stop_button.isEnabled()
    assert window.progress.minimum() == 0
    assert window.progress.maximum() == 100


def test_open_external_url_uses_browser(easyrob_window, monkeypatch):
    """open_external_url delegates to the browser helper."""
    opened = {}
    monkeypatch.setattr(window_module.webbrowser, "open", lambda url, new=0: opened.update({"url": url, "new": new}))

    easyrob_window.open_external_url("https://example.com")

    assert opened == {"url": "https://example.com", "new": 2}


def test_close_event_ignores_when_running_worker_is_not_stopped(easyrob_window, monkeypatch):
    """Closing is aborted if ROBERT is still running and the user declines stopping it."""
    window = easyrob_window

    class DummyEvent:
        def __init__(self):
            self.ignored = 0

        def ignore(self):
            self.ignored += 1

    class DummyWorker:
        def isRunning(self):
            return True

    monkeypatch.setattr(window_module.QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    event = DummyEvent()
    window.worker = DummyWorker()
    window.closeEvent(event)

    assert event.ignored == 1
    assert not window.molssi_is_closing


def test_close_event_stops_worker_and_shuts_down(easyrob_window, monkeypatch):
    """Closing with a running worker stops the process and starts async shutdown."""
    window = easyrob_window

    class DummyEvent:
        def __init__(self):
            self.ignored = 0

        def ignore(self):
            self.ignored += 1

    class DummySignal:
        def __init__(self):
            self.connected = []

        def connect(self, fn):
            self.connected.append(fn)

    class DummyWorker:
        def __init__(self):
            self.process_finished = DummySignal()
            self.stop_calls = 0

        def isRunning(self):
            return True

        def stop(self):
            self.stop_calls += 1

    class DummyLoop:
        def __init__(self):
            self.exec_calls = 0
            self.quit_calls = 0

        def exec(self):
            self.exec_calls += 1

        def quit(self):
            self.quit_calls += 1

    worker = DummyWorker()
    loop = DummyLoop()
    shutdown_calls = {"n": 0}
    timer_calls = {"n": 0}

    monkeypatch.setattr(window_module.QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(window_module, "QEventLoop", lambda: loop)
    monkeypatch.setattr(window_module.QTimer, "singleShot", lambda ms, fn: timer_calls.__setitem__("n", timer_calls["n"] + 1))
    monkeypatch.setattr(window, "_shutdown_molssi_async", lambda: shutdown_calls.__setitem__("n", shutdown_calls["n"] + 1))

    event = DummyEvent()
    window.worker = worker
    window.closeEvent(event)

    assert window.molssi_is_closing
    assert worker.stop_calls == 1
    assert len(worker.process_finished.connected) == 1
    assert loop.exec_calls == 1
    assert timer_calls["n"] == 1
    assert shutdown_calls["n"] == 1
    assert event.ignored == 1


def test_close_event_without_running_worker_starts_async_shutdown(easyrob_window, monkeypatch):
    """Closing without an active ROBERT worker still routes through async cleanup."""
    window = easyrob_window

    class DummyEvent:
        def __init__(self):
            self.ignored = 0

        def ignore(self):
            self.ignored += 1

    shutdown_calls = {"n": 0}
    monkeypatch.setattr(window, "_shutdown_molssi_async", lambda: shutdown_calls.__setitem__("n", shutdown_calls["n"] + 1))

    event = DummyEvent()
    window.worker = None
    window.closeEvent(event)

    assert window.molssi_is_closing
    assert shutdown_calls["n"] == 1
    assert event.ignored == 1


def test_show_contact_dialog_executes_modal(easyrob_window, monkeypatch):
    """Contact dialog is created and executed."""
    exec_calls = {"n": 0}
    monkeypatch.setattr(window_module.QDialog, "exec", lambda self: exec_calls.__setitem__("n", exec_calls["n"] + 1))

    easyrob_window.show_contact_dialog()

    assert exec_calls["n"] == 1


def test_show_version_dialog_executes_modal(easyrob_window, monkeypatch):
    """Version dialog is created and executed."""
    exec_calls = {"n": 0}
    monkeypatch.setattr(window_module.QDialog, "exec", lambda self: exec_calls.__setitem__("n", exec_calls["n"] + 1))

    easyrob_window.show_version_dialog()

    assert exec_calls["n"] == 1


def test_show_tutorial_dialog_reuses_visible_dialog(easyrob_window):
    """Reopening the tutorial dialog reuses the visible instance."""
    class DummyDialog:
        def __init__(self):
            self.raise_calls = 0
            self.activate_calls = 0

        def isVisible(self):
            return True

        def raise_(self):
            self.raise_calls += 1

        def activateWindow(self):
            self.activate_calls += 1

    dialog = DummyDialog()
    easyrob_window.tutorial_dialog = dialog

    easyrob_window.show_tutorial_dialog()

    assert dialog.raise_calls == 1
    assert dialog.activate_calls == 1


def test_check_for_pdfs_and_images_runs_with_existing_outputs(easyrob_window, tmp_path):
    """Results/images detection runs without error when expected outputs exist."""
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch_glob_path = str(tmp_path / "ROBERT_report.pdf")
    original_image_folders = easyrob_window.image_folders
    easyrob_window.image_folders = ["PREDICT"]
    (tmp_path / "PREDICT").mkdir()

    # Make the file-system dependent checks deterministic.
    original_glob = window_module.glob.glob
    original_exists = window_module.os.path.exists
    window_module.glob.glob = lambda pattern: [monkeypatch_glob_path]
    window_module.os.path.exists = lambda path: path.endswith("PREDICT")

    try:
        easyrob_window.check_for_pdfs(str(csv_path))
        easyrob_window.check_for_images(str(csv_path))
    finally:
        window_module.glob.glob = original_glob
        window_module.os.path.exists = original_exists
        easyrob_window.image_folders = original_image_folders


def test_refresh_tabs_updates_children_and_schedules_once(easyrob_window, monkeypatch):
    """refresh_tabs stores the latest path and coalesces duplicate scheduling."""
    timer_calls = {"n": 0}
    monkeypatch.setattr(window_module.QTimer, "singleShot", lambda ms, fn: timer_calls.__setitem__("n", timer_calls["n"] + 1))

    easyrob_window._refresh_scheduled = False
    easyrob_window.refresh_tabs("one.csv")
    easyrob_window.refresh_tabs("two.csv")

    assert easyrob_window._pending_refresh_path == "two.csv"
    assert timer_calls["n"] == 1


def test_execute_refresh_tabs_calls_all_child_refreshes(easyrob_window, monkeypatch):
    """_execute_refresh_tabs fans out the refresh to child tabs and file-based checks."""
    calls = {"results": 0, "images": 0, "predictions": 0, "pdfs": 0, "imgs": 0}

    monkeypatch.setattr(easyrob_window.results_tab, "refresh_with_new_path", lambda p: calls.__setitem__("results", calls["results"] + 1))
    monkeypatch.setattr(easyrob_window.images_tab, "refresh_with_new_path", lambda p: calls.__setitem__("images", calls["images"] + 1))
    monkeypatch.setattr(easyrob_window.predictions_tab, "refresh_with_new_path", lambda p: calls.__setitem__("predictions", calls["predictions"] + 1))
    monkeypatch.setattr(easyrob_window, "check_for_pdfs", lambda p: calls.__setitem__("pdfs", calls["pdfs"] + 1))
    monkeypatch.setattr(easyrob_window, "check_for_images", lambda p: calls.__setitem__("imgs", calls["imgs"] + 1))

    easyrob_window._pending_refresh_path = "demo.csv"
    easyrob_window._refresh_scheduled = True
    easyrob_window._execute_refresh_tabs()

    assert not easyrob_window._refresh_scheduled
    assert calls == {"results": 1, "images": 1, "predictions": 1, "pdfs": 1, "imgs": 1}


def test_stop_process_confirms_and_stops_worker(easyrob_window, monkeypatch):
    """stop_process marks manual stop and schedules worker stop after confirmation."""
    class DummySignal:
        def connect(self, fn):
            self.fn = fn

    class DummyWorker:
        def __init__(self):
            self.stop_calls = 0
            self.process_finished = DummySignal()

        def isRunning(self):
            return True

        def stop(self):
            self.stop_calls += 1

    worker = DummyWorker()
    timer_calls = {"n": 0}
    monkeypatch.setattr(window_module.QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(window_module.QTimer, "singleShot", lambda ms, fn: (timer_calls.__setitem__("n", timer_calls["n"] + 1), fn())[1])

    easyrob_window.worker = worker
    easyrob_window.stop_button.setDisabled(False)
    easyrob_window.stop_process()

    assert easyrob_window.manual_stop
    assert worker.stop_calls == 1
    assert timer_calls["n"] == 1
    assert not easyrob_window.stop_button.isEnabled()
    easyrob_window.worker = None


def test_stop_process_returns_when_user_declines(easyrob_window, monkeypatch):
    """stop_process does nothing if the user rejects the confirmation dialog."""
    monkeypatch.setattr(window_module.QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    easyrob_window.manual_stop = False
    easyrob_window.stop_process()

    assert not easyrob_window.manual_stop


# =====================================================
# Button Connection Tests
# =====================================================


def test_run_button_calls_run_robert(easyrob_window, qtbot, monkeypatch):
    """Clicking the Run button calls run_robert."""
    window = easyrob_window

    call_count = {"n": 0}

    def fake_run_robert():
        call_count["n"] += 1

    monkeypatch.setattr(window, "run_robert", fake_run_robert)

    qtbot.mouseClick(window.run_button, Qt.LeftButton)

    assert call_count["n"] == 1


def test_stop_button_calls_stop_robert(easyrob_window, qtbot, monkeypatch):
    """Clicking the Stop button calls the current stop handler."""
    window = easyrob_window

    call_count = {"n": 0}

    def fake_stop_process():
        call_count["n"] += 1

    monkeypatch.setattr(window, "stop_process", fake_stop_process)
    window.stop_button.clicked.disconnect()
    window.stop_button.clicked.connect(window.stop_process)

    # Enable stop button
    window.stop_button.setDisabled(False)
    qtbot.mouseClick(window.stop_button, Qt.LeftButton)

    assert call_count["n"] == 1


# =====================================================
# Advanced Options Tab Tests
# =====================================================


def test_advanced_options_general_section(easyrob_window):
    """General section of Advanced Options is initialized."""
    window = easyrob_window
    options_tab = window.options_tab

    assert options_tab.auto_type is not None
    assert options_tab.auto_type.isChecked()
    assert options_tab.seed is not None
    assert options_tab.kfold is not None
    assert options_tab.repeat_kfolds is not None
    assert options_tab.split is not None
    assert options_tab.split.count() > 0


def test_advanced_options_curate_section(easyrob_window):
    """CURATE section of Advanced Options is initialized."""
    window = easyrob_window
    options_tab = window.options_tab

    assert options_tab.categoricalstr is not None
    assert options_tab.corr_filter_xbool is not None
    assert options_tab.corr_filter_ybool is not None
    assert options_tab.desc_thresfloat is not None
    assert options_tab.thres_xfloat is not None
    assert options_tab.thres_yfloat is not None


def test_advanced_options_generate_section(easyrob_window):
    """GENERATE section of Advanced Options is initialized."""
    window = easyrob_window
    options_tab = window.options_tab

    assert options_tab.modellist is not None
    assert len(options_tab.modellist) > 0
    assert options_tab.error_type is not None
    assert options_tab.init_points is not None
    assert options_tab.n_iter is not None
    assert options_tab.pfi_filter is not None
    assert options_tab.auto_test is not None
    assert options_tab.test_set is not None


def test_advanced_options_predict_section(easyrob_window):
    """PREDICT section of Advanced Options is initialized."""
    window = easyrob_window
    options_tab = window.options_tab

    assert options_tab.t_value is not None
    assert options_tab.shap_show is not None
    assert options_tab.pfi_show is not None


def test_model_selection_changes_with_type(easyrob_window):
    """Selected models differ between regression and classification."""
    window = easyrob_window
    options_tab = window.options_tab

    # Regression
    window.type_dropdown.setCurrentText("Regression")
    regression_checked = {
        model for model, cb in options_tab.modellist.items() if cb.isChecked()
    }

    # Classification
    window.type_dropdown.setCurrentText("Classification")
    classification_checked = {
        model for model, cb in options_tab.modellist.items() if cb.isChecked()
    }

    assert regression_checked != classification_checked


# =====================================================
# AQME Tab Tests
# =====================================================


def test_aqme_tab_exists(easyrob_window):
    """AQME tab is created and exposes basic attributes."""
    window = easyrob_window

    assert window.tab_widget_aqme is not None
    assert hasattr(window.tab_widget_aqme, "atoms")
    assert hasattr(window.tab_widget_aqme, "descriptor_level")
    assert hasattr(window.tab_widget_aqme, "solvent")


# =====================================================
# Results and Images Tab Tests
# =====================================================


def test_results_tab_exists(easyrob_window):
    """Results tab is created and configured."""
    window = easyrob_window

    assert window.results_tab is not None
    assert hasattr(window.results_tab, "pdf_tab_widget")


def test_images_tab_exists(easyrob_window):
    """Images tab is created and configured."""
    window = easyrob_window

    assert window.images_tab is not None
    assert hasattr(window.images_tab, "folder_tabs")


def test_predictions_tab_placeholder_when_no_csvs(easyrob_window, monkeypatch):
    """Predictions tab stays in placeholder mode when no prediction CSVs are found."""
    availability = []
    window = easyrob_window

    window.predictions_tab.availabilityChanged.connect(availability.append)
    monkeypatch.setattr(predictions_module, "find_prediction_csvs", lambda path: {})

    window.predictions_tab.refresh_with_new_path("demo.csv")

    assert not window.predictions_tab.placeholder.isHidden()
    assert window.predictions_tab.subtabs.isHidden()
    assert availability[-1] is False


def test_predictions_filter_dataframe_orders_core_columns(easyrob_window, monkeypatch):
    """Predictions dataframe is reordered into the GUI-specific display layout."""
    df = pd.DataFrame(
        {
            "foo": [1],
            "SMILES": ["C"],
            "sample_id": ["mol1"],
            "target_pred": [1.2],
            "target_pred_sd": [0.3],
        }
    )
    monkeypatch.setattr(easyrob_window.predictions_tab, "_extract_names_column_from_predict", lambda: "sample_id")

    filtered = easyrob_window.predictions_tab._filter_prediction_dataframe(df)

    assert list(filtered.columns) == ["Image", "sample_id", "SMILES", "target_pred", "target_pred_sd"]


def test_predictions_extract_names_column_from_predict(tmp_path):
    """The names field is extracted from the stored PREDICT command line."""
    predict_dir = tmp_path / "PREDICT"
    predict_dir.mkdir()
    dat_path = predict_dir / "PREDICT_data.dat"
    dat_path.write_text('--names "code_name"\n', encoding="utf-8")
    (tmp_path / "input.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    tab = predictions_module.PredictionsTab()
    tab._base_path = str(tmp_path / "input.csv")

    assert tab._extract_names_column_from_predict() == "code_name"


def test_results_tab_detects_and_refreshes_pdf_tabs(tmp_path, monkeypatch):
    """Results tab discovers PDFs and refreshes when a new path is provided."""
    monkeypatch.setattr(results_module.QTimer, "singleShot", lambda ms, fn: None)
    monkeypatch.setattr(results_module, "PDFViewer", lambda pdf_path, thread_pool: results_module.QWidget())

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first_pdf = run_dir / "ROBERT_report.pdf"
    first_pdf.write_text("pdf", encoding="utf-8")

    tab = results_module.ResultsTab(None, str(run_dir / "input.csv"))

    assert first_pdf.name in tab.title_to_path
    assert tab.pdf_tab_widget.count() == 1

    second_dir = tmp_path / "run2"
    second_dir.mkdir()
    second_pdf = second_dir / "ROBERT_report_2.pdf"
    second_pdf.write_text("pdf", encoding="utf-8")

    tab.refresh_with_new_path(str(second_dir / "other.csv"))

    assert tab.pdf_tab_widget.count() == 1
    assert second_pdf.name in tab.title_to_path


def test_predictions_show_header_menu_sorts_and_histogram(monkeypatch):
    """Header context menu routes to sort and histogram actions."""
    tab = predictions_module.PredictionsTab()
    df = pd.DataFrame({"num": [2, 1], "txt": ["b", "a"]})

    class DummyHeader:
        def __init__(self):
            self._parent = None

        def logicalIndexAt(self, pos):
            return 0

        def parent(self):
            return self._parent

        def mapToGlobal(self, pos):
            return pos

    class DummyTable:
        def __init__(self):
            self.sort_calls = []

        def sortByColumn(self, column, order):
            self.sort_calls.append((column, order))

    class DummyMenu:
        chosen_index = 0

        def __init__(self, parent):
            self.actions = []

        def addAction(self, text):
            action = object()
            self.actions.append((text, action))
            return action

        def addSeparator(self):
            return None

        def exec(self, pos):
            return self.actions[DummyMenu.chosen_index][1]

    monkeypatch.setattr(predictions_module, "QMenu", DummyMenu)
    histogram_calls = {"n": 0}
    monkeypatch.setattr(tab, "_show_histogram", lambda series, name: histogram_calls.__setitem__("n", histogram_calls["n"] + 1))

    header = DummyHeader()
    table = DummyTable()
    header._parent = table

    DummyMenu.chosen_index = 0
    tab._show_header_menu(None, df, header)
    assert table.sort_calls[-1] == (0, Qt.AscendingOrder)

    DummyMenu.chosen_index = 2
    tab._show_header_menu(None, df, header)
    assert histogram_calls["n"] == 1


def test_predictions_show_histogram_menu_non_numeric_shows_message(monkeypatch):
    """Non-numeric columns show an informational popup instead of plotting."""
    tab = predictions_module.PredictionsTab()
    df = pd.DataFrame({"txt": ["a", "b"]})
    info_calls = {"n": 0}

    class DummyHeader:
        def logicalIndexAt(self, pos):
            return 0

    monkeypatch.setattr(predictions_module.QMessageBox, "information", lambda *args, **kwargs: info_calls.__setitem__("n", info_calls["n"] + 1))

    tab._show_histogram_menu_header(None, df, DummyHeader())

    assert info_calls["n"] == 1


def test_predictions_show_histogram_uses_matplotlib(monkeypatch):
    """Histogram plotting delegates to matplotlib without blocking."""
    tab = predictions_module.PredictionsTab()
    series = pd.Series([1, 2, 3])
    calls = {"figure": 0, "title": 0, "xlabel": 0, "ylabel": 0, "grid": 0, "show": 0, "hist": 0}

    monkeypatch.setattr(predictions_module.plt, "figure", lambda: calls.__setitem__("figure", calls["figure"] + 1))
    monkeypatch.setattr(predictions_module.plt, "title", lambda name: calls.__setitem__("title", calls["title"] + 1))
    monkeypatch.setattr(predictions_module.plt, "xlabel", lambda name: calls.__setitem__("xlabel", calls["xlabel"] + 1))
    monkeypatch.setattr(predictions_module.plt, "ylabel", lambda name: calls.__setitem__("ylabel", calls["ylabel"] + 1))
    monkeypatch.setattr(predictions_module.plt, "grid", lambda enabled: calls.__setitem__("grid", calls["grid"] + 1))
    monkeypatch.setattr(predictions_module.plt, "show", lambda block=False: calls.__setitem__("show", calls["show"] + 1))
    monkeypatch.setattr(pd.Series, "hist", lambda self, bins=30: calls.__setitem__("hist", calls["hist"] + 1))

    tab._show_histogram(series, "value")

    assert calls == {"figure": 1, "title": 1, "xlabel": 1, "ylabel": 1, "grid": 1, "show": 1, "hist": 1}


def test_predictions_add_loaded_df_replaces_loading_tab(monkeypatch):
    """Loaded prediction data replaces the placeholder tab widget."""
    tab = predictions_module.PredictionsTab()
    tab._base_path = "demo.csv"
    tab.subtabs.addTab(predictions_module.QLabel("Loading"), "No PFI")

    df = pd.DataFrame({"SMILES": ["C"], "target_pred": [1.0]})
    monkeypatch.setattr(tab, "_filter_prediction_dataframe", lambda frame: frame)
    monkeypatch.setattr(predictions_module, "evaluate_predictions_for_model", lambda base, frame, key: {"pdf_path": "report.pdf", "model": key, "scenario": "demo"})
    monkeypatch.setattr(predictions_module, "get_robert_report_path", lambda base: "report.pdf")
    monkeypatch.setattr(predictions_module, "extract_robert_fragment_image", lambda path, key: None)
    monkeypatch.setattr(predictions_module, "extract_boundary_scores", lambda path: {"No_PFI": None})
    monkeypatch.setattr(predictions_module, "extract_boundary_fragment", lambda path, key: None)
    monkeypatch.setattr(predictions_module, "find_external_test_pixmaps", lambda base: {})
    widget = predictions_module.QWidget()
    monkeypatch.setattr(tab, "_create_table_with_stats", lambda frame, info, pdf_image: widget)

    tab._add_loaded_df("No_PFI", df)

    assert tab.subtabs.widget(0) is widget
    assert tab.subtabs.tabText(0) == "No PFI"


def test_predictions_refresh_with_new_path_loads_csvs_synchronously(tmp_path, monkeypatch):
    """refresh_with_new_path discovers CSVs and materializes tabs when tasks run synchronously."""
    csv_test_dir = tmp_path / "PREDICT" / "csv_test"
    csv_test_dir.mkdir(parents=True)

    no_pfi_path = csv_test_dir / "demo_No_PFI.csv"
    pfi_path = csv_test_dir / "demo_PFI.csv"
    pd.DataFrame({"SMILES": ["C"], "target_pred": [1.0]}).to_csv(no_pfi_path, index=False)
    pd.DataFrame({"SMILES": ["CC"], "target_pred": [2.0]}).to_csv(pfi_path, index=False)

    tab = predictions_module.PredictionsTab()
    created = []

    class FakePlaceholder:
        def __init__(self):
            self.hidden = False

        def hide(self):
            self.hidden = True

        def show(self):
            self.hidden = False

        def isHidden(self):
            return self.hidden

    class FakeSubtabs:
        def __init__(self):
            self.hidden = True
            self.tabs = []

        def clear(self):
            self.tabs.clear()

        def hide(self):
            self.hidden = True

        def show(self):
            self.hidden = False

        def isHidden(self):
            return self.hidden

        def addTab(self, widget, name):
            self.tabs.append([widget, name])

        def count(self):
            return len(self.tabs)

        def tabText(self, index):
            return self.tabs[index][1]

        def removeTab(self, index):
            self.tabs.pop(index)

        def insertTab(self, index, widget, name):
            self.tabs.insert(index, [widget, name])

        def widget(self, index):
            return self.tabs[index][0]

    monkeypatch.setattr(tab, "placeholder", FakePlaceholder())
    monkeypatch.setattr(tab, "subtabs", FakeSubtabs())

    class DummyDoneSignal:
        def __init__(self):
            self._callback = None

        def connect(self, callback):
            self._callback = callback

        def emit(self, key, df):
            assert self._callback is not None
            self._callback(key, df)

    class DummySignals:
        def __init__(self):
            self.done = DummyDoneSignal()

    class FakeLoadCsvTask:
        def __init__(self, key, path):
            self.key = key
            self.path = path
            self.signals = DummySignals()

        def run(self):
            self.signals.done.emit(self.key, pd.read_csv(self.path))

    monkeypatch.setattr(
        tab._thread_pool,
        "start",
        lambda task: task.run(),
    )
    monkeypatch.setattr(predictions_module, "LoadCsvTask", FakeLoadCsvTask)
    monkeypatch.setattr(
        predictions_module,
        "evaluate_predictions_for_model",
        lambda base, frame, key: {"pdf_path": "report.pdf", "model": key, "scenario": "demo"},
    )
    monkeypatch.setattr(predictions_module, "get_robert_report_path", lambda base: "report.pdf")
    monkeypatch.setattr(predictions_module, "extract_robert_fragment_image", lambda path, key: None)
    monkeypatch.setattr(predictions_module, "extract_boundary_scores", lambda path: {})
    monkeypatch.setattr(predictions_module, "extract_boundary_fragment", lambda path, key: None)
    monkeypatch.setattr(predictions_module, "find_external_test_pixmaps", lambda base: {})
    monkeypatch.setattr(
        tab,
        "_create_table_with_stats",
        lambda frame, info, pdf_image: created.append((info["model"], frame.copy())) or object(),
    )

    tab.refresh_with_new_path(str(tmp_path / "input.csv"))

    assert tab.placeholder.isHidden()
    assert not tab.subtabs.isHidden()
    assert tab.subtabs.count() == 2
    assert {tab.subtabs.tabText(i) for i in range(tab.subtabs.count())} == {"No PFI", "PFI"}
    assert {model for model, _ in created} == {"No_PFI", "PFI"}


def test_results_clear_pdf_tabs_removes_placeholders(tmp_path, monkeypatch):
    """clear_pdf_tabs removes tracked tabs and resets internal maps."""
    monkeypatch.setattr(results_module.QTimer, "singleShot", lambda ms, fn: None)
    monkeypatch.setattr(results_module, "PDFViewer", lambda pdf_path, thread_pool: results_module.QWidget())

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pdf_path = run_dir / "ROBERT_report.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")

    tab = results_module.ResultsTab(None, str(run_dir / "input.csv"))
    assert tab.pdf_tab_widget.count() == 1

    tab.clear_pdf_tabs()

    assert tab.pdf_tab_widget.count() == 0
    assert tab.pdf_tabs == {}
    assert tab.title_to_path == {}


def test_results_maybe_materialize_tab_builds_viewer(monkeypatch, tmp_path):
    """Selecting a placeholder PDF tab materializes a real viewer."""
    monkeypatch.setattr(results_module.QTimer, "singleShot", lambda ms, fn: None)
    monkeypatch.setattr(results_module, "PDFViewer", lambda pdf_path, thread_pool: results_module.QWidget())

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pdf_path = run_dir / "ROBERT_report.pdf"
    tab = results_module.ResultsTab(None, str(run_dir / "input.csv"))
    pdf_path.write_text("pdf", encoding="utf-8")
    tab.clear_pdf_tabs()
    tab.pdf_tabs[str(pdf_path)] = None
    tab.title_to_path[pdf_path.name] = str(pdf_path)
    tab.pdf_tab_widget.blockSignals(True)
    tab.pdf_tab_widget.addTab(results_module.QWidget(), pdf_path.name)
    tab.pdf_tab_widget.blockSignals(False)

    viewer = results_module.QWidget()
    monkeypatch.setattr(tab, "_materialize_pdf_viewer", lambda index, path: tab.pdf_tabs.__setitem__(path, viewer))

    tab._maybe_materialize_tab(0)

    assert tab.pdf_tabs[str(pdf_path)] is viewer


def test_results_index_of_title_returns_expected_index(tmp_path, monkeypatch):
    """Tab titles can be resolved back to their index."""
    monkeypatch.setattr(results_module.QTimer, "singleShot", lambda ms, fn: None)
    monkeypatch.setattr(results_module, "PDFViewer", lambda pdf_path, thread_pool: results_module.QWidget())

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pdf_path = run_dir / "ROBERT_report.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")

    tab = results_module.ResultsTab(None, str(run_dir / "input.csv"))

    assert tab._index_of_title(pdf_path.name) == 0
    assert tab._index_of_title("missing.pdf") == -1


# =====================================================
# Workflow Selector Tests
# =====================================================


def test_workflow_selector_options(easyrob_window):
    """Workflow selector has all required entries and default."""
    window = easyrob_window

    expected_workflows = ["Full Workflow", "CURATE", "GENERATE", "PREDICT", "VERIFY", "REPORT"]

    workflow_items = [
        window.workflow_selector.itemText(i)
        for i in range(window.workflow_selector.count())
    ]

    assert set(workflow_items) == set(expected_workflows)
    assert window.workflow_selector.currentText() == "Full Workflow"


# =====================================================
# Console Output / Progress Tests
# =====================================================


def test_console_output_widget_exists(easyrob_window):
    """Console output widget exists and is read-only."""
    window = easyrob_window

    assert window.console_output is not None
    assert window.console_output.isReadOnly()


def test_progress_bar_exists(easyrob_window):
    """Progress bar exists and has 0–100 range."""
    window = easyrob_window

    assert window.progress is not None
    assert window.progress.minimum() == 0
    assert window.progress.maximum() == 100


# =====================================================
# REAL End-to-End User Workflow Test (GUI)
# =====================================================

@pytest.mark.parametrize(
    "test_scenario",
    [
        "regression",
        "existing_dirs_stop",
        "aqme_regression",
    ],
)
def test_full_user_workflow_end_to_end(
    easyrob_window, test_output_dir, qtbot, monkeypatch, test_scenario
):
    """
    End-to-end tests of realistic user workflows in the easyROB GUI.

    Scenarios
    ---------
    - regression:
        * First full workflow run (baseline).
        * Uses easyrob_example_1.csv in a persistent output directory.
        * Waits for completion and dumps console output.

    - existing_dirs_stop:
        * Assumes a previous full regression run has already created
          CURATE/GENERATE/PREDICT/VERIFY and ROBERT_report.pdf in test_output_dir.
        * Re-runs ROBERT with the same CSV in the same folder.
        * Auto-answers "Yes" to the "existing folders from a previous run" popup.
        * Waits until the workflow starts.
        * Clicks Stop and auto-answers "Yes" to the "stop ROBERT" popup.
        * Verifies that the workflow stops cleanly and the GUI returns to idle.

        NOTE: this test is intentionally dependent on the 'regression'
        scenario having run earlier in the same pytest session.

    - aqme_regression:
        * Full workflow using easyrob_example_train.csv and easyrob_example_test.csv (carboxylic acids).
        * AQME workflow checkbox enabled.
        * Existing ROBERT folders (if any) are removed before starting.
        * AQME generates a mapped CSV and ROBERT runs with it.
        * Check predictions tab 
    """
    window = easyrob_window
    config = SCENARIO_CONFIG[test_scenario]

    message_box_calls = install_message_box_stubs(monkeypatch)
    print_scenario_banner(config["description"], test_scenario)

    source_name = (
        "easyrob_example_train.csv"
        if test_scenario == "aqme_regression"
        else "easyrob_example_1.csv"
    )
    csv_path = copy_example_csv(source_name, test_output_dir)

    csv_test_path = None
    if test_scenario == "aqme_regression":
        csv_test_path = copy_example_csv("easyrob_example_test.csv", test_output_dir)

    print("\n[SETUP] Using example CSV:", csv_path)
    print("[SETUP] Outputs will be written under:", test_output_dir)
    if csv_test_path is not None:
        print("[SETUP] Using example external test CSV:", csv_test_path)

    # ------------------------------------------------------------------
    # 3. Load CSV in the GUI
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading CSV in GUI...")
    window.set_file_path(str(csv_path))
    assert window.file_path == str(csv_path)
    print(f"[OK] CSV loaded: {Path(window.file_path).name}")

    if csv_test_path is not None:
        window.set_csv_test_path(str(csv_test_path))
        assert window.csv_test_path == str(csv_test_path)
        print(f"[OK] External test CSV loaded: {Path(window.csv_test_path).name}")

    # ------------------------------------------------------------------
    # 4. Verify and select Y and Names
    # ------------------------------------------------------------------
    print("\n[STEP 2] Verifying columns loaded...")
    y_items = [window.y_dropdown.itemText(i) for i in range(window.y_dropdown.count())]
    names_items = [
        window.names_dropdown.itemText(i) for i in range(window.names_dropdown.count())
    ]
    print("  • y_dropdown items:", y_items)
    print("  • names_dropdown items:", names_items)

    if test_scenario == "aqme_regression":
        assert "target" in y_items
        assert "code_name" in names_items
        window.y_dropdown.setCurrentText("target")
        window.names_dropdown.setCurrentText("code_name")
    else:
        assert "solubility" in y_items
        assert "code_name" in names_items
        window.y_dropdown.setCurrentText("solubility")
        window.names_dropdown.setCurrentText("code_name")

    print(f"[OK] Target column (y): {window.y_dropdown.currentText()}")
    print(f"[OK] Names column: {window.names_dropdown.currentText()}")

    # ------------------------------------------------------------------
    # 5. Configure ignored columns / features
    # ------------------------------------------------------------------
    print("\n[STEP 3] Configuring ignored columns...")

    if test_scenario == "aqme_regression":
        cols_to_ignore = []
    else:
        cols_to_ignore = ["SMILES"]

    # Select manually requested columns
    for col_to_ignore in cols_to_ignore:
        for i in range(window.available_list.count()):
            item = window.available_list.item(i)
            if item.text() == col_to_ignore:
                item.setSelected(True)
                break

    window.move_to_selected()

    # Collect ignored items from GUI
    ignore_items = [window.ignore_list.item(i).text() for i in range(window.ignore_list.count())]
    print(f"  • Ignored columns: {ignore_items}")

    actual = set(ignore_items)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    # 1. SMILES must ALWAYS be ignored (auto rule)
    assert "SMILES" in actual, "SMILES should always be auto-ignored"

    # 2. Manually selected columns must be present
    for col in cols_to_ignore:
        assert col in actual, f"Expected column '{col}' to be ignored"

    # 3. Optional strict check (no unexpected extras except SMILES)
    expected = set(cols_to_ignore) | {"SMILES"}

    missing = expected - actual
    extra = actual - expected

    assert not missing and not extra, (
        f"\nMissing items: {missing}"
        f"\nExtra items: {extra}"
    )

    available_items = [
        window.available_list.item(i).text()
        for i in range(window.available_list.count())
    ]
    print(f"  • Features used: {available_items}")

    # ------------------------------------------------------------------
    # 6. Analysis type and Advanced Options
    # ------------------------------------------------------------------
    print("\n[STEP 4] Setting analysis type and Advanced Options...")
    window.type_dropdown.setCurrentText(config["analysis_type"])
    assert window.type_dropdown.currentText() == config["analysis_type"]

    options_tab = window.options_tab
    options_tab.seed.setText(config["seed"])
    options_tab.kfold.setText(config["kfold"])
    options_tab.repeat_kfolds.setText("1")
    options_tab.split.setCurrentText("even")

    print(f"  • seed: {options_tab.seed.text()}")
    print(f"  • kfold: {options_tab.kfold.text()}")
    print(f"  • repeat_kfolds: {options_tab.repeat_kfolds.text()}")
    print(f"  • split: {options_tab.split.currentText()}")

    checked_models = {
        model for model, cb in options_tab.modellist.items() if cb.isChecked()
    }
    print(f"  • Selected models: {checked_models}")
    assert len(checked_models) > 0

    # ------------------------------------------------------------------
    # 7. AQME setup for aqme_regression
    # ------------------------------------------------------------------
    if test_scenario == "aqme_regression":
        print("\n[STEP 4bis] Preparing AQME workflow...")
        aqme = window.tab_widget_aqme

        monkeypatch.setattr(
            predictions_module,
            "extract_robert_fragment_image",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            predictions_module,
            "extract_boundary_fragment",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            predictions_module,
            "extract_boundary_scores",
            lambda *args, **kwargs: {},
        )
        monkeypatch.setattr(
            predictions_module,
            "find_external_test_pixmaps",
            lambda *args, **kwargs: {},
        )
        monkeypatch.setattr(
            window.predictions_tab,
            "_create_table_with_stats",
            lambda df, info, pdf_image: predictions_module.QWidget(),
        )

        if not window.aqme_workflow.isChecked():
            window.aqme_workflow.setChecked(True)

        aqme.file_path = window.file_path
        aqme.csv_df = pd.read_csv(window.file_path)
        aqme.smiles_column = "SMILES"

        aqme.smarts_targets = ["C(=O)O"]
        aqme.selected_atoms = [0]

        aqme.generate_mapped_smiles(
            aqme.smarts_targets[0],
            aqme.selected_atoms,
            aqme.csv_df[aqme.smiles_column].dropna(),
        )

        assert hasattr(aqme, "df_mapped_smiles")
        assert aqme.df_mapped_smiles is not None
        print("  • AQME df_mapped_smiles is available")

    # ------------------------------------------------------------------
    # 8. Workflow selector
    # ------------------------------------------------------------------
    print("\n[STEP 5] Selecting workflow...")
    window.workflow_selector.setCurrentText(config["workflow"])
    assert window.workflow_selector.currentText() == config["workflow"]
    print(f"[OK] Workflow: {window.workflow_selector.currentText()}")

    print("\n[STEP 6] Pre-execution checks...")
    assert window.file_path is not None
    assert window.run_button.isEnabled()
    print("[OK] All parameters configured, run_button is enabled.")

    output_dir = Path(window.file_path).parent
    expected_dirs = ["PREDICT", "VERIFY", "CURATE", "GENERATE"]
    report_pdf = output_dir / "ROBERT_report_No_PFI.pdf"

    if test_scenario == "aqme_regression":
        print("[SETUP] Cleaning existing ROBERT folders for aqme_regression...")
        clean_output_paths(output_dir, directory_names=expected_dirs)
        if report_pdf.is_file():
            report_pdf.unlink()

    # ------------------------------------------------------------------
    # SPECIAL CASE: existing_dirs_stop → only re-run + stop
    # ------------------------------------------------------------------
    if test_scenario == "existing_dirs_stop":
        # Bootstrap the expected output state if it is not already present.
        if not all((output_dir / d).is_dir() for d in expected_dirs) or not report_pdf.is_file():
            print("[SETUP] Existing output folders missing; running baseline workflow first...")
            run_full_workflow_and_wait(window, qtbot, output_dir, expected_dirs, report_pdf)
            QCoreApplication.processEvents()
            assert all((output_dir / d).is_dir() for d in expected_dirs)
            assert report_pdf.is_file()

        print("\n[STEP 7] Re-run with existing folders + manual stop...")
        baseline_text = window.console_output.toPlainText()

        qtbot.mouseClick(window.run_button, Qt.LeftButton)
        assert not window.run_button.isEnabled()

        started = wait_for_workflow_start(window, baseline_text)
        if not started:
            pytest.fail("Re-run did not start within timeout after existing-folders popup")

        print("\n[STEP 8] Clicking Stop ROBERT button...")
        qtbot.mouseClick(window.stop_button, Qt.LeftButton)

        print("[STEP 9] Waiting for workflow to stop and GUI to return to idle state...")
        stopped = process_events_until(
            lambda: (
                getattr(window, "worker", None) is None
                and window.run_button.isEnabled()
                and not window.stop_button.isEnabled()
            ),
            STOP_MAX_WAIT_S,
        )
        if not stopped:
            pytest.fail("Re-run did not stop within timeout after pressing Stop")
        print("[OK] Re-run stopped and GUI buttons reset to idle state")

        # Verify popups
        assert any(
            "detected folders from a previous run" in text
            for (_, text) in message_box_calls["question"]
        ), "Expected an 'existing folders from a previous run' QMessageBox.question"

        assert any(
            "Are you sure you want to stop the process?" in text
            for (_, text) in message_box_calls["question"]
        ), "Expected a stop-process QMessageBox.question"

        final_console_text = window.console_output.toPlainText()
        dump_console_output("[STEP 10] Final console output (existing_dirs_stop):", final_console_text)

        assert window.file_path == str(csv_path)

        print("\n" + "=" * 80)
        print(f"[OK] TEST SCENARIO '{test_scenario}' PASSED")
        print("=" * 80)
        return

    # ------------------------------------------------------------------
    # 10. Normal run to completion (regression, aqme_regression)
    # ------------------------------------------------------------------
    print("\n[STEP 7] EXECUTING run_robert() from GUI (real subprocess)...")
    initial_console_text = window.console_output.toPlainText()
    initial_progress = window.progress.value()

    print("\n[STEP 8] Waiting for workflow to complete (with timeout)...")
    run_full_workflow_and_wait(window, qtbot, output_dir, expected_dirs, report_pdf)

    # ------------------------------------------------------------------
    # 11. Final assertions for regression / aqme_regression
    # ------------------------------------------------------------------
    print("\n[STEP 9] Final console output:")
    final_console_text = window.console_output.toPlainText()
    console_len = len(final_console_text)

    print(
        f"Console length: {console_len} chars "
        f"(+{console_len - len(initial_console_text)} vs initial)"
    )
    dump_console_output("----- BEGIN FULL CONSOLE OUTPUT -----", final_console_text)

    print("\n[STEP 10] Progress bar state...")
    final_progress = window.progress.value()
    print(f"Progress: {initial_progress} → {final_progress}")

    print("\n[STEP 11] Final assertions...")
    assert window.file_path == str(csv_path)
    assert options_tab.seed.text() == config["seed"]
    assert options_tab.kfold.text() == config["kfold"]
    assert window.type_dropdown.currentText() == config["analysis_type"]
    assert len(final_console_text) >= len(initial_console_text)

    if test_scenario == "aqme_regression":
        base, ext = os.path.splitext(str(csv_path))
        mapped_csv_path = base + "_mapped.csv"
        assert os.path.isfile(mapped_csv_path), "AQME mapped CSV file was not created"
        print(f"[OK] AQME mapped CSV exists: {mapped_csv_path}")

        predict_csv_test_dir = output_dir / "PREDICT" / "csv_test"
        assert predict_csv_test_dir.is_dir(), "PREDICT/csv_test directory was not created"

        prediction_csvs = predictions_module.find_prediction_csvs(str(csv_path))
        assert prediction_csvs, "Predictions CSVs were not generated for the external test set"
        print(f"[OK] Predictions CSVs detected: {sorted(prediction_csvs)}")

    print("\n" + "=" * 80)
    print(f"[OK] TEST SCENARIO '{test_scenario}' PASSED")
    print("=" * 80)


def test_run_aqme_only_end_to_end(easyrob_window, test_output_dir, qtbot, monkeypatch):
    """
    End-to-end AQME-only workflow using the dedicated example CSV.

    Goal:
    - load a CSV in the GUI
    - launch AQME from the real GUI button
    - wait for the subprocess to finish
    - verify AQME outputs were generated
    """
    window = easyrob_window
    finished = {"exit_code": None}

    def _on_process_finished_stub(exit_code):
        finished["exit_code"] = exit_code
        window.manual_stop = False
        window._reset_ui_after_process()
        window.worker = None

    install_message_box_stubs(monkeypatch)
    monkeypatch.setattr(window, "on_process_finished", _on_process_finished_stub)

    csv_path = copy_example_csv("easyrob_example_aqme.csv", test_output_dir)

    print_scenario_banner("AQME-only descriptor generation")
    print("[SETUP] Using AQME CSV:", csv_path)

    output_dir = csv_path.parent
    clean_output_paths(
        output_dir,
        directory_names=["AQME", "CSEARCH", "QDESCP", "AQME_RUNS"],
        file_patterns=["AQME-ROBERT_*_easyrob_example_aqme.csv"],
    )

    window.set_file_path(str(csv_path))
    assert window.file_path == str(csv_path)
    print(f"[OK] CSV loaded: {Path(window.file_path).name}")

    window.tab_widget_aqme.atoms.clear()
    window.tab_widget_aqme.selected_atoms = []
    assert window.run_aqme_button.isEnabled()

    print("\n[STEP 1] Launching AQME from GUI...")
    qtbot.mouseClick(window.run_aqme_button, Qt.LeftButton)

    process_events_until(
        lambda: finished["exit_code"] is not None,
        WORKFLOW_MAX_WAIT_S,
    )

    assert finished["exit_code"] == 0, "AQME did not finish successfully"

    final_console_text = window.console_output.toPlainText()
    dump_console_output("[STEP 2] Final AQME console output:", final_console_text)

    aqme_runs_dir = output_dir / "AQME_RUNS"
    generated_csvs = sorted(output_dir.glob("AQME-ROBERT_*_easyrob_example_aqme.csv"))

    assert "Running AQME" in final_console_text
    assert "Time QDESCP:" in final_console_text
    assert aqme_runs_dir.is_dir(), "AQME_RUNS directory was not created"
    assert generated_csvs, "AQME descriptor CSV outputs were not generated"
    print(f"[OK] AQME generated CSVs: {[p.name for p in generated_csvs]}")

    print("\n" + "=" * 80)
    print("[OK] TEST SCENARIO 'aqme_only' PASSED")
    print("=" * 80)


# =====================================================
# ChemDraw → popup → table → CSV → main window test
# =====================================================

def test_open_chemdraw_popup_end_to_end_cdxml(
    easyrob_window, qtbot, monkeypatch, test_output_dir
):
    """
    End-to-end test for the ChemDraw → popup → table → CSV → main window flow
    using a real CDXML file and the GUI methods:

        open_chemdraw_popup -> ChemDrawFileDialog -> load_chemdraw_file ->
        show_molecule_table_dialog -> save_to_csv -> set_file_path -> load_csv_columns
    """
    window = easyrob_window
    aqme_tab = window.tab_widget_aqme

    # --------------------------------------------------------------
    # 1. Locate the test CDXML file
    # --------------------------------------------------------------
    cdxml_path = Path(__file__).with_name("chemdraw_example.cdxml")
    assert cdxml_path.exists(), f"Test CDXML file not found: {cdxml_path}"

    # --------------------------------------------------------------
    # 2. Stub the pre-dialog QMessageBox.information (non-blocking)
    # --------------------------------------------------------------
    def _msgbox_info_stub(*args, **kwargs):
        # Simulate clicking "OK"
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "information", _msgbox_info_stub)

    # --------------------------------------------------------------
    # 3. Fake ChemDrawFileDialog so that it returns our CDXML path
    # --------------------------------------------------------------
    class FakeChemDrawFileDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.main_chemdraw_path = str(cdxml_path)

        def exec(self):
            # Simulate user clicking "OK" in the ChemDraw file dialog
            return QDialog.Accepted

    # Patch the symbol that open_chemdraw_popup uses
    monkeypatch.setattr(
        aqme_module, "ChemDrawFileDialog", FakeChemDrawFileDialog
    )

    # --------------------------------------------------------------
    # 4. Stub QFileDialog.getSaveFileName so CSV is written to tmp_path
    # --------------------------------------------------------------
    csv_path = test_output_dir / "chemdraw_table_output.csv"


    def _fake_get_save_file_name(*args, **kwargs):
        return (str(csv_path), "CSV Files (*.csv)")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", _fake_get_save_file_name)

    # --------------------------------------------------------------
    # 5. Stub QDialog.exec for the *table* dialog:
    #    - detect the "ChemDraw Molecules" dialog,
    #    - fill code_name + target,
    #    - click "Save as CSV",
    #    - return Accepted.
    #    Any other QDialog.exec uses the original implementation.
    # --------------------------------------------------------------
    original_exec = QDialog.exec

    def _fake_dialog_exec(self: QDialog):
        # Let non-ChemDraw-table dialogs behave normally
        if self.windowTitle() != "ChemDraw Molecules":
            return original_exec(self)

        table = self.findChild(QTableWidget)
        assert table is not None, "ChemDraw table dialog should contain a QTableWidget."

        headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
        assert "SMILES" in headers
        assert "code_name" in headers
        assert "target" in headers

        smiles_idx = headers.index("SMILES")
        code_name_idx = headers.index("code_name")
        target_idx = headers.index("target")

        # Fill code_name and target for each row
        for row in range(table.rowCount()):
            smi_item = table.item(row, smiles_idx)
            assert smi_item is not None
            assert smi_item.text().strip() != ""

            code_item = table.item(row, code_name_idx)
            if code_item is None:
                code_item = QTableWidgetItem(f"mol_{row + 1}")
                table.setItem(row, code_name_idx, code_item)
            else:
                code_item.setText(f"mol_{row + 1}")

            target_item = table.item(row, target_idx)
            if target_item is None:
                target_item = QTableWidgetItem(str(1.0 + row))
                table.setItem(row, target_idx, target_item)
            else:
                target_item.setText(str(1.0 + row))

        # Locate "Save as CSV" button
        save_button = None
        for btn in self.findChildren(QPushButton):
            if "Save as CSV" in btn.text():
                save_button = btn
                break
        assert save_button is not None, "Could not find 'Save as CSV' button in ChemDraw dialog."

        # Click it → this will call save_to_csv() and then dialog.accept()
        save_button.click()

        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", _fake_dialog_exec)

    # --------------------------------------------------------------
    # 6. Trigger the whole workflow from the AQME tab
    # --------------------------------------------------------------
    # This should:
    #   - show the pre-dialog QMessageBox (stubbed),
    #   - construct FakeChemDrawFileDialog (returns our CDXML path),
    #   - call load_chemdraw_file(main_path),
    #   - call show_molecule_table_dialog(mols),
    #   - run _fake_dialog_exec on the table dialog,
    #   - write CSV,
    #   - call main_window.set_file_path(path) and load_csv_columns().
    aqme_tab.open_chemdraw_popup()

    # Process pending events
    QCoreApplication.processEvents()

    # --------------------------------------------------------------
    # 7. Assertions: CSV on disk + main window updated
    # --------------------------------------------------------------
    assert csv_path.exists(), "CSV file was not created from ChemDraw CDXML flow."

    # Main window should now point to this CSV
    assert window.file_path == str(csv_path)

    # Columns should be loaded into dropdowns
    y_items = [window.y_dropdown.itemText(i) for i in range(window.y_dropdown.count())]
    names_items = [window.names_dropdown.itemText(i) for i in range(window.names_dropdown.count())]

    assert "SMILES" in y_items
    assert "code_name" in y_items
    assert "target" in y_items

    available_items = [
        window.available_list.item(i).text()
        for i in range(window.available_list.count())
    ]
    ignored_items = [
        window.ignore_list.item(i).text()
        for i in range(window.ignore_list.count())
    ]
    assert "SMILES" in ignored_items
    assert "code_name" in available_items

    # Basic CSV sanity check
    df = pd.read_csv(csv_path)
    assert {"SMILES", "code_name", "target"}.issubset(df.columns)
    assert len(df) > 0
    # All SMILES should be non-empty
    assert df["SMILES"].astype(str).str.strip().ne("").all()
    # All code_name should be non-empty
    assert df["code_name"].astype(str).str.strip().ne("").all()


# =====================================================
# AQME SMARTS / atom-selection / mapped SMILES test
# =====================================================


def test_aqme_atom_selection_generates_mapped_smiles(
    easyrob_window, tmp_path, qtbot, monkeypatch
):
    """
    Simulate AQME pattern detection + atom selection:

    - Create a simple CSV with a SMILES column.
    - Configure AQMETab with this CSV and a known SMARTS pattern.
    - Call display_molecule() to set up the viewer and atom coords.
    - Simulate selecting an atom via handle_atom_selection().
    - Verify df_mapped_smiles is created and SMILES column is updated.
    """
    window = easyrob_window
    aqme = window.tab_widget_aqme

    # ------------------------------------------------------------------
    # 1. Create a small CSV with a SMILES column
    # ------------------------------------------------------------------
    csv_path = tmp_path / "aqme_smiles.csv"
    df = pd.DataFrame(
        {
            "SMILES": ["Cl", "ClC"],  # pattern [Cl] appears exactly once in each
            "code_name": ["mol1", "mol2"],
            "target": [1.0, 2.0],
        }
    )
    df.to_csv(csv_path, index=False)

    aqme.file_path = str(csv_path)
    aqme.csv_df = pd.read_csv(csv_path)
    aqme.smiles_column = "SMILES"
    aqme.unified_smiles = aqme.build_unified_smiles_context(str(csv_path))

    # ------------------------------------------------------------------
    # 2. Inject a known SMARTS and display the pattern
    # ------------------------------------------------------------------
    aqme.smarts_targets = ["[Cl]"]

    # Initialize attributes that display_molecule expects
    aqme.selected_atoms = []
    aqme.mol_viewer_container.resize(500, 500)

    # Build mol, atom_coords, etc.
    aqme.display_molecule()

    # Sanity check: we should have a SMARTS pattern and atom coords
    assert aqme.smarts_targets
    assert hasattr(aqme, "atom_coords")
    assert aqme.atom_coords is not None
    assert len(aqme.atom_coords) > 0

    # ------------------------------------------------------------------
    # 3. Simulate user selecting atom 0 in the SMARTS pattern
    # ------------------------------------------------------------------
    aqme.handle_atom_selection(0)

    # After selection, df_mapped_smiles should exist
    assert hasattr(aqme, "df_mapped_smiles")
    assert aqme.df_mapped_smiles is not None

    mapped_df = aqme.df_mapped_smiles
    assert "SMILES" in mapped_df.columns
    assert len(mapped_df) == len(df)

    # There should be mapping numbers in the SMILES (e.g. [Cl:1])
    for s in mapped_df["SMILES"]:
        assert s is None or ":" in s, "Expected atom mapping numbers in mapped SMILES"
