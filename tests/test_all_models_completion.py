"""Completion messages for standard and all-model ROBERT reports."""

import pytest

from gui_easyrob.main import window as window_module


ALL_MODELS_OUTPUT = """\
o ROBERT_report_GB_PFI.pdf was created successfully in the working directory!
o ROBERT_report_MVL_No_PFI.pdf was created successfully in the working directory!
o ROBERT_report_RF_PFI.pdf was created successfully in the working directory!
o All 8 model PDFs were moved to REPORT_models/
o ROBERT_report_RF_No_PFI.pdf (best for Interpolation) and ROBERT_report_GB_No_PFI.pdf (best for Boundary robustness) were kept in the working directory
"""


@pytest.mark.parametrize(
    ("output", "exit_code", "expected_success"),
    [
        (ALL_MODELS_OUTPUT, 0, True),
        ("o ROBERT_report_No_PFI.pdf was created successfully in the working directory!", 0, True),
        (ALL_MODELS_OUTPUT, 1, False),
        ("o ROBERT_report_GB_PFI.pdf was created successfully in the working directory!", 0, False),
    ],
)
def test_report_completion_dialog_requires_finished_report_run(
    disposed_easyrob_window, monkeypatch, output, exit_code, expected_success,
):
    window = disposed_easyrob_window
    dialogs = []
    warnings = []

    class CompletionDialog:
        Information = 1
        ActionRole = 2
        AcceptRole = 3

        def __init__(self, parent):
            self.title = ""
            self.message = ""

        def setIcon(self, icon):
            pass

        def setWindowTitle(self, title):
            self.title = title

        def setText(self, message):
            self.message = message

        def addButton(self, button, role):
            pass

        def exec(self):
            dialogs.append((self.title, self.message))

    try:
        monkeypatch.setattr(window_module, "QMessageBox", CompletionDialog)
        monkeypatch.setattr(window, "refresh_tabs", lambda path: None)
        monkeypatch.setattr(window, "_reset_ui_after_process", lambda: None)
        monkeypatch.setattr(
            window, "_show_tracked_process_warning",
            lambda *args: warnings.append(args),
        )
        window.current_process = "ROBERT"
        window.workflow_selector.setCurrentText("Full Workflow")
        window.console_output.setPlainText(output)

        window.on_process_finished(exit_code)

        assert bool(dialogs) is expected_success
        assert bool(warnings) is not expected_success
        if expected_success:
            assert dialogs == [("Success!", "ROBERT has completed successfully.")]
    finally:
        window.hide()
