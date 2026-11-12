"""
Application entry point (bootstrap).

This module is responsible for initializing the Qt application,
configuring the runtime environment, and launching the main window.

Key responsibilities:
- Configure environment variables for stable rendering
- Initialize QApplication
- Instantiate and display the main application UI

Notes:
- Offscreen tests use software rendering. Desktop sessions use the available
  graphics driver unless the environment explicitly requests software mode.
"""

import os
import sys

def configure_environment():
    """Configure Qt environment for stable rendering."""
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        os.environ.setdefault("QT_QUICK_BACKEND", "software")
        os.environ.setdefault("QT_OPENGL", "software")
    elif sys.platform == "linux":
        os.environ.setdefault("QT_QUICK_BACKEND", "software")

    existing_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
    extra_flags = "--log-level=3 --disable-logging"
    if sys.platform == "linux":
        extra_flags += " --disable-gpu"

    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        f"{existing_flags} {extra_flags}".strip() if existing_flags else extra_flags
    )

    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"


configure_environment()

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon, QPalette

APP_NAME = "EasyRob"
APP_DISPLAY_NAME = "EasyRob"
APP_ORGANIZATION = "The Alegre Group"
APP_DESKTOP_FILE_NAME = "easyrob"
APP_USER_MODEL_ID = "com.thealegregroup.easyrob"
CHECKML_APP_USER_MODEL_ID = "com.thealegregroup.checkml"


def configure_process_identity(mode="easyrob"):
    """Configure process-level identity used by desktop shells."""
    if sys.platform != "win32":
        return

    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            CHECKML_APP_USER_MODEL_ID if mode == "checkml" else APP_USER_MODEL_ID
        )
    except Exception:
        pass

def main(mode="easyrob"):
    """Launch the EasyROB or CheckML desktop application."""
    if mode not in {"easyrob", "checkml"}:
        raise ValueError(f"Unsupported GUI mode: {mode}")
    configure_process_identity(mode)
    if os.environ.get("QT_OPENGL", "").lower() == "software":
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

    app = QApplication(sys.argv)
    app_name = "CheckML" if mode == "checkml" else APP_NAME
    app.setApplicationName(app_name)
    app.setApplicationDisplayName(app_name)
    app.setOrganizationName(APP_ORGANIZATION)
    app.setDesktopFileName("checkml" if mode == "checkml" else APP_DESKTOP_FILE_NAME)

    # Import AFTER QApplication to avoid lunching issues
    try:
        from easyrob import get_main_window_class
        from utils.utils_gui import AssetLibrary
    except ImportError:
        from robert.gui_easyrob.easyrob import get_main_window_class
        from robert.gui_easyrob.utils.utils_gui import AssetLibrary

    icon_asset = AssetLibrary.CheckML_icon if mode == "checkml" else AssetLibrary.Robert_icon
    with icon_asset.get_path() as path_icon:
        icon = QIcon(str(path_icon))
        if not icon.isNull():
            app.setWindowIcon(icon)

    EasyROB = get_main_window_class()

    palette = app.palette()
    bg_color = palette.color(QPalette.Window)
    text_color = "white" if bg_color.lightness() < 128 else "black"

    app.setStyleSheet(f"QLabel {{ color: {text_color}; }}")

    window = EasyROB(mode=mode) if mode == "checkml" else EasyROB()
    if not app.windowIcon().isNull():
        window.setWindowIcon(app.windowIcon())
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
