"""Pytest configuration for the ROBERT test suite."""

import os
import shutil
import sys
import tempfile


def pytest_configure(config):
    """
    Normalize the test process environment for ROBERT and its subprocesses.

    * Linux/macOS: prepend the active env's ``lib`` to ``LD_LIBRARY_PATH`` so
      Pango/HarfBuzz load correctly for WeasyPrint (see CircleCI / conda).
    * All platforms: headless Qt so ``python -m robert`` subprocesses do not
      require a display or full XKB stack (PySide6 is imported by the package).
    """
    if sys.platform != "win32":
        lib = os.path.join(sys.prefix, "lib")
        if os.path.isdir(lib):
            prev = os.environ.get("LD_LIBRARY_PATH", "")
            if not (prev.startswith(lib + os.pathsep) or prev == lib):
                os.environ["LD_LIBRARY_PATH"] = lib + (
                    os.pathsep + prev if prev else ""
                )

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")

import base64
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "robert"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))


WORKFLOW_TEST_MODULES = {
    "test_1curate.py",
    "test_2generate.py",
    "test_3verify.py",
    "test_4predict.py",
    "test_6evaluate.py",
}


@pytest.fixture(scope="session")
def workflow_workspace():
    """Share isolated output across the sequential scientific workflow tests."""
    with tempfile.TemporaryDirectory(prefix="robert-workflow-") as directory:
        workspace = Path(directory)
        input_dir = workspace / "tests"
        input_dir.mkdir()
        for source in (REPOSITORY_ROOT / "tests").iterdir():
            if source.is_file() and source.suffix in {".csv", ".yaml", ".txt", ".cdxml"}:
                shutil.copy2(source, input_dir / source.name)
        yield workspace


@pytest.fixture(autouse=True)
def isolate_workflow_outputs(request, monkeypatch):
    """Keep pipeline outputs away from the repository and OneDrive sync state."""
    if request.node.path.name not in WORKFLOW_TEST_MODULES:
        return

    workspace = request.getfixturevalue("workflow_workspace")
    module = request.module
    monkeypatch.chdir(workspace)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")])),
    )
    for attribute, folder in (
        ("path_main", None),
        ("path_curate", "CURATE"),
        ("path_generate", "GENERATE"),
        ("path_verify", "VERIFY"),
        ("path_predict", "PREDICT"),
    ):
        if hasattr(module, attribute):
            monkeypatch.setattr(module, attribute, str(workspace / folder) if folder else str(workspace))


@pytest.fixture(scope="session")
def small_test_gif(tmp_path_factory):
    """Provide a tiny animation for bot GUI tests."""
    path = tmp_path_factory.mktemp("bot_gui") / "frame.gif"
    path.write_bytes(base64.b64decode("R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="))
    return path


@pytest.fixture
def disposed_easyrob_window(qapp):
    """Create an EasyROB window and stop its background PDF work during teardown."""
    from PySide6.QtCore import QCoreApplication, QEvent
    import shiboken6

    from gui_easyrob.main.window import EasyROB

    window = EasyROB()
    yield window

    if shiboken6.isValid(window):
        window.results_tab.shared_pool.waitForDone()
        qapp.removeEventFilter(window)
        window.hide()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.fixture(autouse=True)
def use_small_gif_in_bot_gui_tests(request, small_test_gif, monkeypatch):
    """Keep bot GUI tests independent of the large tutorial animation."""
    if Path(request.node.path).name not in {
        "test_bot_compact_gui.py",
    }:
        return

    for module_name in (
        "gui_easyrob.bot.bot_panel",
        "bot.bot_panel",
        "robert.gui_easyrob.bot.bot_panel",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(module, "API_SETUP_GIF_PATH", small_test_gif)
