"""Checks that failed CheckML runs report their outcome to the user."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui_easyrob.tabs import checkml


def test_checkml_nonzero_exit_shows_error_notification(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    tab = checkml.CheckMLTab()
    tab.csv_path = str(tmp_path / "input.csv")
    notices = []
    monkeypatch.setattr(checkml.QMessageBox, "warning", lambda _parent, title, message: notices.append((title, message)))
    try:
        tab._on_process_finished(2)
        assert notices == [("CHECKML", "CHECKML finished with errors (exit code 2). Check the log above.")]
        assert tab.run_button.isEnabled()
    finally:
        tab.close()


def test_checkml_missing_report_shows_error_notification(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    tab = checkml.CheckMLTab()
    tab.csv_path = str(tmp_path / "input.csv")
    notices = []
    monkeypatch.setattr(checkml.QMessageBox, "warning", lambda _parent, title, message: notices.append((title, message)))
    try:
        tab._on_process_finished(0)
        assert notices == [("CHECKML", "CHECKML did not create a new report. Check the log above.")]
        assert tab.run_button.isEnabled()
    finally:
        tab.close()


def test_checkml_unchanged_report_from_previous_run_is_not_success(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    tab = checkml.CheckMLTab()
    tab.csv_path = str(tmp_path / "input.csv")
    report = tmp_path / "ROBERT_report_No_PFI.pdf"
    report.write_bytes(b"previous report")
    report_stat = report.stat()
    tab._report_before_run = (report_stat.st_mtime_ns, report_stat.st_size)
    warnings = []
    successes = []
    monkeypatch.setattr(checkml.QMessageBox, "warning", lambda _parent, title, message: warnings.append((title, message)))
    monkeypatch.setattr(checkml.QMessageBox, "information", lambda _parent, title, message: successes.append((title, message)))
    try:
        tab._on_process_finished(0)
        assert warnings == [("CHECKML", "CHECKML did not create a new report. Check the log above.")]
        assert not successes
    finally:
        tab.close()
