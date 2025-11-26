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
from GUI_easyROB.easyrob import EasyROB

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
TEST_OUTPUT_DIR_NAME = "test-easyrob-output"

WORKFLOW_MAX_WAIT_S = 120.0
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
    monkeypatch.setattr(window, "check_for_pdfs", lambda: None)
    monkeypatch.setattr(window, "check_for_images", lambda: None)
    monkeypatch.setattr(window, "check_aqme_workflow", lambda: None)

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


def test_all_tabs_created(easyrob_window):
    """All expected top-level tabs are present."""
    window = easyrob_window

    tab_names = [window.tab_widget.tabText(i) for i in range(window.tab_widget.count())]

    assert "ROBERT" in tab_names
    assert "AQME" in tab_names
    assert "Advanced Options" in tab_names
    assert "Help" in tab_names
    assert "Results" in tab_names
    assert "Images" in tab_names


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
    """Clicking the Stop button calls stop_robert."""
    window = easyrob_window

    call_count = {"n": 0}

    def fake_stop_robert():
        call_count["n"] += 1

    monkeypatch.setattr(window, "stop_robert", fake_stop_robert)

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
        * Full workflow using easyrob_example_2.csv (carboxylic acids).
        * AQME workflow checkbox enabled.
        * Existing ROBERT folders (if any) are removed before starting.
        * AQME generates a mapped CSV and ROBERT runs with it.
    """
    window = easyrob_window
    config = SCENARIO_CONFIG[test_scenario]

    # ------------------------------------------------------------------
    # 0. Make QMessageBox non-blocking during tests
    # ------------------------------------------------------------------
    msgbox_info_calls = []
    msgbox_question_calls = []

    def _msgbox_info_stub(*args, **kwargs):
        title = args[1] if len(args) > 1 else ""
        text = args[2] if len(args) > 2 else ""
        print(f"[QMessageBox.information/warning] {title}: {text}")
        msgbox_info_calls.append((title, text))
        return QMessageBox.Ok

    def _msgbox_question_stub(
        parent,
        title,
        text,
        buttons=QMessageBox.Yes | QMessageBox.No,
        default=QMessageBox.No,
    ):
        """
        Non-blocking replacement for QMessageBox.question.

        It auto-answers "Yes" but also inspects the text to distinguish:
        - Existing folders popup
        - Stop ROBERT popup
        """
        print(f"[QMessageBox.question] {title}: {text}")
        msgbox_question_calls.append((title, text))

        if "detected folders from a previous run" in text:
            return QMessageBox.Yes
        if "Are you sure you want to stop ROBERT?" in text:
            return QMessageBox.Yes
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "information", _msgbox_info_stub)
    monkeypatch.setattr(QMessageBox, "warning", _msgbox_info_stub)
    monkeypatch.setattr(QMessageBox, "question", _msgbox_question_stub)

    # ------------------------------------------------------------------
    # 1. Scenario header
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TEST SCENARIO: {config['description']}")
    print(f"Scenario ID: {test_scenario}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 2. Load example CSV into persistent test folder
    # ------------------------------------------------------------------
    if test_scenario == "aqme_regression":
        source_csv = Path(__file__).with_name("easyrob_example_2.csv")
    else:
        source_csv = Path(__file__).with_name("easyrob_example_1.csv")

    assert source_csv.exists(), f"Example CSV not found: {source_csv}"

    csv_path = test_output_dir / source_csv.name
    shutil.copyfile(source_csv, csv_path)

    print("\n[SETUP] Using example CSV:", csv_path)
    print("[SETUP] Outputs will be written under:", test_output_dir)

    # ------------------------------------------------------------------
    # 3. Load CSV in the GUI
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading CSV in GUI...")
    window.set_file_path(str(csv_path))
    assert window.file_path == str(csv_path)
    print(f"✓ CSV loaded: {Path(window.file_path).name}")

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

    print(f"✓ Target column (y): {window.y_dropdown.currentText()}")
    print(f"✓ Names column: {window.names_dropdown.currentText()}")

    # ------------------------------------------------------------------
    # 5. Configure ignored columns / features
    # ------------------------------------------------------------------
    print("\n[STEP 3] Configuring ignored columns...")
    if test_scenario == "aqme_regression":
        cols_to_ignore = []
    else:
        cols_to_ignore = ["SMILES"]

    for col_to_ignore in cols_to_ignore:
        for i in range(window.available_list.count()):
            item = window.available_list.item(i)
            if item.text() == col_to_ignore:
                item.setSelected(True)
                break
    window.move_to_selected()

    ignore_items = [window.ignore_list.item(i).text() for i in range(window.ignore_list.count())]
    print(f"  • Ignored columns: {ignore_items}")
    assert set(ignore_items) == set(cols_to_ignore)

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
    print(f"✓ Workflow: {window.workflow_selector.currentText()}")

    # ------------------------------------------------------------------
    # 9. Pre-execution checks
    # ------------------------------------------------------------------
    print("\n[STEP 6] Pre-execution checks...")
    assert window.file_path is not None
    assert window.run_button.isEnabled()
    print("✓ All parameters configured, run_button is enabled.")

    output_dir = Path(window.file_path).parent
    expected_dirs = ["PREDICT", "VERIFY", "CURATE", "GENERATE"]
    report_pdf = output_dir / "ROBERT_report.pdf"

    # AQME: start from a clean folder (no existing folders popup)
    if test_scenario == "aqme_regression":
        print("[SETUP] Cleaning existing ROBERT folders for aqme_regression...")
        for dname in expected_dirs:
            dpath = output_dir / dname
            if dpath.is_dir():
                shutil.rmtree(dpath)
        if report_pdf.is_file():
            report_pdf.unlink()

    # Helper: wait for workflow to start
    def wait_for_workflow_start(baseline_text, timeout_s=60.0):
        elapsed = 0.0
        while elapsed < timeout_s:
            QCoreApplication.processEvents()
            current_console = window.console_output.toPlainText()
            process_local = getattr(window.worker, "process", None)
            if process_local is not None and process_local.poll() is None:
                if len(current_console) > len(baseline_text):
                    print(
                        "✓ Workflow started; console output detected "
                        "and process is running"
                    )
                    return True
            time.sleep(WORKFLOW_POLL_INTERVAL_S)
            elapsed += WORKFLOW_POLL_INTERVAL_S
        return False

    # ------------------------------------------------------------------
    # SPECIAL CASE: existing_dirs_stop → only re-run + stop
    # ------------------------------------------------------------------
    if test_scenario == "existing_dirs_stop":
        # Make sure folders already exist thanks to the 'regression' scenario
        if not all((output_dir / d).is_dir() for d in expected_dirs) or not report_pdf.is_file():
            pytest.fail(
                "ROBERT output folders and/or ROBERT_report.pdf are missing. "
                "This scenario assumes that the 'regression' end-to-end scenario "
                "has already created them in test_output_dir."
            )

        print("\n[STEP 7] Re-run with existing folders + manual stop...")
        baseline_text = window.console_output.toPlainText()

        qtbot.mouseClick(window.run_button, Qt.LeftButton)
        assert not window.run_button.isEnabled()

        started = wait_for_workflow_start(baseline_text)
        if not started:
            pytest.fail("Re-run did not start within timeout after existing-folders popup")

        print("\n[STEP 8] Clicking Stop ROBERT button...")
        qtbot.mouseClick(window.stop_button, Qt.LeftButton)

        print("[STEP 9] Waiting for workflow to stop and GUI to return to idle state...")
        elapsed = 0.0

        while elapsed < STOP_MAX_WAIT_S:
            QCoreApplication.processEvents()

            worker_alive = getattr(window, "worker", None) is not None
            run_enabled = window.run_button.isEnabled()
            stop_enabled = window.stop_button.isEnabled()

            if (not worker_alive) and run_enabled and (not stop_enabled):
                print("✓ Re-run stopped and GUI buttons reset to idle state")
                break

            time.sleep(WORKFLOW_POLL_INTERVAL_S)
            elapsed += WORKFLOW_POLL_INTERVAL_S
        else:
            pytest.fail("Re-run did not stop within timeout after pressing Stop")

        # Verify popups
        assert any(
            "detected folders from a previous run" in text
            for (_, text) in msgbox_question_calls
        ), "Expected an 'existing folders from a previous run' QMessageBox.question"

        assert any(
            "Are you sure you want to stop ROBERT?" in text
            for (_, text) in msgbox_question_calls
        ), "Expected a 'stop ROBERT' QMessageBox.question"

        # Final console
        final_console_text = window.console_output.toPlainText()
        print("\n[STEP 10] Final console output (existing_dirs_stop):")
        print("----- BEGIN CONSOLE OUTPUT -----")
        print(final_console_text.encode("utf-8", errors="replace").decode())
        print("----- END CONSOLE OUTPUT -----")

        assert window.file_path == str(csv_path)

        print("\n" + "=" * 80)
        print(f"✓ TEST SCENARIO '{test_scenario}' PASSED")
        print("=" * 80)
        return

    # ------------------------------------------------------------------
    # 10. Normal run to completion (regression, aqme_regression)
    # ------------------------------------------------------------------
    print("\n[STEP 7] EXECUTING run_robert() from GUI (real subprocess)...")
    initial_console_text = window.console_output.toPlainText()
    initial_progress = window.progress.value()

    qtbot.mouseClick(window.run_button, Qt.LeftButton)
    assert not window.run_button.isEnabled()

    print("\n[STEP 8] Waiting for workflow to complete (with timeout)...")
    workflow_started = False
    workflow_completed = False

    elapsed = 0.0
    while elapsed < WORKFLOW_MAX_WAIT_S:
        QCoreApplication.processEvents()
        current_console = window.console_output.toPlainText()
        process = getattr(window.worker, "process", None)

        if not workflow_started and len(current_console) > len(initial_console_text):
            workflow_started = True
            print("✓ Workflow started (console output detected)")

        all_dirs_exist = all((output_dir / d).is_dir() for d in expected_dirs)
        pdf_exists = report_pdf.is_file()

        if workflow_started and all_dirs_exist and pdf_exists:
            workflow_completed = True
            print("✓ Workflow completed (all output folders AND report PDF detected)")
            break

        if process is not None and process.poll() is not None and not (
            all_dirs_exist and pdf_exists
        ):
            print(
                f"[WARN] Process exited with code {process.returncode} "
                "but not all outputs (folders + PDF) are present yet."
            )
            break

        if window.run_button.isEnabled() and not workflow_completed:
            print("[INFO] run_button re-enabled (popup probably closed).")

        time.sleep(WORKFLOW_POLL_INTERVAL_S)
        elapsed += WORKFLOW_POLL_INTERVAL_S

    if not workflow_started:
        print("\n[DEBUG] Console at timeout (no start detected):")
        print(current_console)
        pytest.fail("Workflow did not start within timeout")

    if not workflow_completed:
        print("\n[DEBUG] Console at timeout (no completion detected):")
        print("Existing entries in output_dir:", [p.name for p in output_dir.iterdir()])
        print("Report PDF exists:", report_pdf.is_file())
        print(current_console)
        pytest.fail("Workflow did not complete within timeout")

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
    print("\n----- BEGIN FULL CONSOLE OUTPUT -----")
    print(final_console_text.encode("utf-8", errors="replace").decode())
    print("----- END FULL CONSOLE OUTPUT -----")

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
        print(f"✓ AQME mapped CSV exists: {mapped_csv_path}")

    print("\n" + "=" * 80)
    print(f"✓ TEST SCENARIO '{test_scenario}' PASSED")
    print("=" * 80)


# =====================================================
# ChemDraw → popup → table → CSV → main window test
# =====================================================

import GUI_easyROB.easyrob as easyrob_module

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
        easyrob_module, "ChemDrawFileDialog", FakeChemDrawFileDialog
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
    assert "SMILES" in available_items
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