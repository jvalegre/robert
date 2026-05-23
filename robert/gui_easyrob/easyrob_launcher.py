"""
Application entry point (bootstrap).

This module is responsible for initializing the Qt application,
configuring the runtime environment, and launching the main window.

Key responsibilities:
- Configure environment variables for stable rendering
- Initialize QApplication
- Instantiate and display the main application UI

Notes:
- Rendering is forced to software mode to avoid GPU/OpenGL issues
  in certain environments (VMs, remote desktops, unstable drivers).
"""

import os
import sys


def configure_environment():
    """Configure Qt environment for stable rendering."""
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    os.environ.setdefault("QT_OPENGL", "software")
    # os.environ["QTWEBENGINE_DISABLE_GPU"] = "1"

    existing_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
    extra_flags = "--log-level=3 --disable-logging"

    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        f"{existing_flags} {extra_flags}".strip() if existing_flags else extra_flags
    )

    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"


configure_environment()

from PySide6.QtCore import Qt, QCoreApplication  # noqa: E402
from PySide6.QtGui import QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def main():
    """Main entry point for the EasyROB application."""
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

    app = QApplication(sys.argv)

    # Import AFTER QApplication to avoid lunching issues
    try:
        from easyrob import get_main_window_class
    except ImportError:
        from robert.gui_easyrob.easyrob import get_main_window_class

    EasyROB = get_main_window_class()

    palette = app.palette()
    bg_color = palette.color(QPalette.Window)
    text_color = "white" if bg_color.lightness() < 128 else "black"

    app.setStyleSheet(f"QLabel {{ color: {text_color}; }}")

    window = EasyROB()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
