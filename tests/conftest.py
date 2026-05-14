"""Pytest configuration for the ROBERT test suite."""

import os
import sys


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
