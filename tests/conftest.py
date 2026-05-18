"""Pytest configuration for the ROBERT test suite."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Module output folders legacy integration tests swap under the repo root.
ROBERT_MODULE_DIR_NAMES = (
    "CURATE",
    "GENERATE",
    "GENERATE_reg",
    "GENERATE_clas",
    "PREDICT",
    "VERIFY",
    "AQME",
    "EVALUATE",
)


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


def robert_module_dirs(root: Path) -> set[str]:
    """Names of ROBERT module directories present under ``root``."""
    return {name for name in ROBERT_MODULE_DIR_NAMES if (root / name).is_dir()}


def aqme_installed() -> bool:
    """Return True when the optional AQME package is importable."""
    try:
        import aqme  # noqa: F401

        return True
    except ImportError:
        return False


def restore_regression_generate_layout(root: Path) -> None:
    """
    Undo a half-finished clas/reg GENERATE rename left by a failed test.

    After a successful clas test: ``GENERATE`` (reg) + ``GENERATE_clas``.
    Mid-clas failure may leave: ``GENERATE_reg`` + ``GENERATE`` (clas).
    """
    reg_backup = root / "GENERATE_reg"
    generate = root / "GENERATE"
    clas = root / "GENERATE_clas"
    if reg_backup.is_dir() and generate.is_dir():
        generate.rename(clas)
        reg_backup.rename(generate)
    elif reg_backup.is_dir() and not generate.is_dir():
        reg_backup.rename(generate)


@contextmanager
def clas_generate_layout(root: Path):
    """
    Temporarily point ``GENERATE`` at the classification screening outputs.

    Requires ``GENERATE`` (regression) and ``GENERATE_clas`` from
    ``test_2generate`` (e.g. ``reduced_clas``).
    """
    restore_regression_generate_layout(root)
    generate = root / "GENERATE"
    clas = root / "GENERATE_clas"
    if not clas.is_dir():
        pytest.skip(
            "GENERATE_clas missing under repo root; run "
            "tests/test_2generate.py::test_GENERATE[reduced_clas] first."
        )
    if not generate.is_dir():
        pytest.skip(
            "GENERATE missing under repo root; run GENERATE integration tests first."
        )
    generate.rename(root / "GENERATE_reg")
    clas.rename(generate)
    try:
        yield
    finally:
        generate.rename(clas)
        (root / "GENERATE_reg").rename(generate)


@pytest.fixture
def repo_root() -> Path:
    """Repository root (stable even if the process cwd changes during a test)."""
    return REPO_ROOT


@pytest.fixture(autouse=True)
def restore_process_cwd():
    """Restore the process working directory after each test."""
    cwd_before = os.getcwd()
    yield
    try:
        os.chdir(cwd_before)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def drain_qt_thread_pool_after_test():
    """Avoid 'QThread destroyed while still running' abort on pytest exit."""
    yield
    try:
        from PySide6.QtCore import QCoreApplication, QThreadPool
    except ImportError:
        return
    app = QCoreApplication.instance()
    if app is None:
        return
    pool = QThreadPool.globalInstance()
    if pool is not None:
        pool.waitForDone(10_000)
    app.processEvents()


@pytest.fixture
def fast_robert_kwargs():
    """Reduced CV/BO settings for faster integration tests."""
    return {
        "model": ["RF"],
        "n_iter": 2,
        "init_points": 2,
        "repeat_kfolds": 2,
        "kfold": 3,
        "pfi_epochs": 1,
        "seed": 42,
    }
