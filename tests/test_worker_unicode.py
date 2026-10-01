"""Regression tests for Unicode output from the GUI subprocess worker."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui_easyrob.utils.utils_gui import RobertWorker


def test_worker_streams_unicode_from_python_subprocess(tmp_path, monkeypatch, qapp):
    script = tmp_path / "emit_unicode.py"
    script.write_text(
        'import sys\nprint("start \\u2192 done")\nprint("warning \\u2192 done", file=sys.stderr)\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")

    command = f'"{sys.executable.replace(chr(92), "/")}" "{script.as_posix()}"'
    worker = RobertWorker(command)
    stdout_lines = []
    stderr_lines = []
    exit_codes = []
    worker.output_received.connect(stdout_lines.append)
    worker.error_received.connect(stderr_lines.append)
    worker.process_finished.connect(exit_codes.append)

    worker.run()
    qapp.processEvents()

    assert exit_codes == [0]
    assert stdout_lines == ['<span style="color:white;">start → done</span>']
    assert stderr_lines == ['<span style="color:red;">warning → done</span>']
