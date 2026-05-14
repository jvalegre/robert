"""
Core infrastructure and shared utilities for easyROB.

This module provides the foundational layer used across the entire GUI,
including Qt abstractions, process management, shared widgets, and
cross-cutting utilities.

Responsibilities:
- Centralize common Qt classes and third-party imports used throughout the GUI
- Provide reusable widgets (e.g., drag-and-drop inputs, combo boxes)
- Manage long-running subprocesses with real-time output streaming
- Offer utility helpers for file handling, CSV parsing, and asset resolution
- Handle platform-specific behavior (e.g., process control, frozen builds)

Architecture:
- Acts as a shared dependency layer for all GUI modules
- Exposes commonly used classes to reduce repetitive imports
- Integrates threading (QThread) and subprocess control for async workflows
- Provides asset resolution compatible with both development and frozen apps

Notes:
- This module intentionally groups cross-project infrastructure
- Domain-specific logic is delegated to specialized modules
  (e.g., predictions_utils, molssi_utils, aqme_utils)
- Should remain focused on generic, reusable functionality

"""

# ------------------------------------------------------------
# Standard library
# ------------------------------------------------------------
import csv
import glob
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
from functools import partial
from io import BytesIO
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import as_file, files

# ------------------------------------------------------------
# Third-party libraries
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import fitz

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolfiles import MolsFromCDXMLFile
from rdkit.Chem.rdmolops import GetMolFrags

from ansi2html import Ansi2HTMLConverter

# ------------------------------------------------------------
# Qt (PySide6)
# ------------------------------------------------------------
from PySide6.QtCore import (
    QByteArray,
    QEventLoop,
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QRunnable,
    QRect,
    QSize,
    QSortFilterProxyModel,
    QThread,
    QThreadPool,
    QTimer,
    Qt,
    Signal,
    Slot,
    QUrl,
)

from PySide6.QtGui import (
    QDesktopServices,
    QFontMetrics,
    QIcon,
    QImage,
    QMouseEvent,
    QPalette,
    QPixmap,
    QWheelEvent,
)

from PySide6.QtWebEngineCore import QWebEngineDownloadRequest
from PySide6.QtWebEngineWidgets import QWebEngineView

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QStyleOptionHeader,
    QTabWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

class DropLabel(QFrame):
    """Frame-based drop target with an optional file dialog button."""

    def __init__(self, text, parent=None, file_filter="CSV Files (*.csv)", extensions=(".csv",)):
        super().__init__(parent)
        self.file_filter = file_filter
        self.valid_extensions = extensions
        self.setAcceptDrops(True)
        self.callback = None
        self.full_file_path = None

        self.setStyleSheet("font-size: 14px; border: none;")
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.label = QLabel(text, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            """
            font-size: 11px;
            font-style: italic;
            color: gray;
            font-weight: bold;
            border: 2px dashed gray;
            padding: 5px;
            border-radius: 5px;
            """
        )
        self.layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.open_file_dialog)
        self.browse_button.setFixedSize(120, 30)
        self.browse_button.setStyleSheet(
            "padding: 6px 12px; font-size: 14px; border-radius: 5px; "
            "background-color: #555; color: white; border: 1px solid #777;"
        )
        self.layout.addWidget(self.browse_button, alignment=Qt.AlignCenter)

    def set_callback(self, callback):
        """Set the callback function to be called when a file is selected."""
        self.callback = callback

    def set_file_type(self, file_filter, extensions):
        """Set the file type filter and valid extensions."""
        self.file_filter = file_filter
        self.valid_extensions = extensions

    def open_file_dialog(self):
        """Open a file dialog to select a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self.file_filter)
        if file_path and self.callback:
            self.set_file_path(file_path)

    def dragEnterEvent(self, event):
        """Allow dropping of files."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle dropped files."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(self.valid_extensions):
                self.set_file_path(file_path)
            else:
                self.label.setText("⚠ Invalid file format.")

    def set_file_path(self, file_path):
        """Set the file path and update the label."""
        self.full_file_path = file_path
        file_name = Path(file_path).name
        self.label.setText(f"Selected: {file_name}")
        self.label.setToolTip(file_path)
        if self.callback:
            self.callback(file_path)

    def setText(self, text):
        """Set the text of the label."""
        self.label.setText(text)

class RobertWorker(QThread):
    """QThread that runs a subprocess asynchronously and streams real-time output."""

    output_received = Signal(str)
    error_received = Signal(str)
    process_finished = Signal(int)
    request_stop = Signal()

    def __init__(self, command, working_dir=None):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.process = None
        self._stop_requested = False
        self.ansi_converter = Ansi2HTMLConverter(dark_bg=True)
        self.is_windows = platform.system() == "Windows"
        self.request_stop.connect(self._handle_stop)

    def run(self):
        """Run the subprocess and stream output in real-time."""
        try:
            if self.is_windows:
                self.process = subprocess.Popen(
                    shlex.split(self.command),
                    cwd=self.working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                )
            else:
                self.process = subprocess.Popen(
                    shlex.split(self.command),
                    cwd=self.working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    preexec_fn=os.setsid,
                )

            def read_stdout():
                """Read stdout in real-time and emit formatted output."""
                try:
                    for line in self.process.stdout:
                        if self._stop_requested:
                            break
                        formatted_line = self.ansi_converter.convert(line.strip(), full=False)
                        self.output_received.emit(formatted_line)
                except Exception as exc:
                    self.error_received.emit(f"Error reading stdout: {exc}")

            def read_stderr():
                """Read stderr in real-time and emit formatted output."""
                try:
                    for line in self.process.stderr:
                        if self._stop_requested:
                            break
                        formatted_line = f'<span style="color:red;">{line.strip()}</span>'
                        self.error_received.emit(formatted_line)

                    reset_line = self.ansi_converter.convert("\033[0m", full=False)
                    self.output_received.emit(reset_line)
                except Exception as exc:
                    self.error_received.emit(f"Error reading stderr: {exc}")

            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            exit_code = self.process.wait() if self.process else -1
            stdout_thread.join()
            stderr_thread.join()

            reset_html = self.ansi_converter.convert("\033[0m", full=False)
            self.output_received.emit(reset_html)

            self.process = None
            self.process_finished.emit(-1 if self._stop_requested else exit_code)
        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            self.error_received.emit(f"Error in run(): {exc}\n{tb}")

    def stop(self):
        """Stop the subprocess and emit a stop signal."""
        self.request_stop.emit()

    def _handle_stop(self):
        """Handle the stop signal and terminate the subprocess."""
        self._stop_requested = True
        if not self.process:
            return

        try:
            parent = psutil.Process(self.process.pid)
            procs = parent.children(recursive=True)
            procs.append(parent)

            for proc in procs:
                try:
                    proc.terminate()
                except Exception:
                    pass

            _, alive = psutil.wait_procs(procs, timeout=2)
            for proc in alive:
                try:
                    proc.kill()
                except Exception:
                    pass
        except Exception as exc:
            self.error_received.emit(f"Error stopping process: {exc}")

def smart_read_csv(filepath):
    """Read a CSV file with automatic delimiter detection."""
    try:
        with open(filepath, "r", encoding="utf-8") as file_obj:
            first_line = file_obj.readline()
            delimiter = ";" if first_line.count(";") > first_line.count(",") else ","

        return pd.read_csv(filepath, encoding="utf-8", delimiter=delimiter)
    except (FileNotFoundError, OSError):
        return None

class NoScrollComboBox(QComboBox):
    """Combo box that ignores wheel events while the popup is closed."""

    def wheelEvent(self, event: QWheelEvent):
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()

class AssetPath:
    """Resolve asset paths both in development and in frozen distributions."""

    def __init__(self, filename):
        self._filename = filename

    def get_path(self):
        if getattr(sys, "frozen", False):
            return (
                Path.cwd()
                / "_internal"
                / "robert_env"
                / "Lib"
                / "site-packages"
                / "robert"
                / "icons"
                / self._filename
            )
        return as_file(files("robert") / "icons" / self._filename)

class AssetLibrary:
    """Central registry of asset files used by the GUI."""

    Info_icon = AssetPath("info_icon.png")
    Robert_logo_transparent = AssetPath("Robert_logo_transparent.png")
    Robert_icon = AssetPath("Robert_icon.png")
    Play_icon = AssetPath("play_icon.png")
    Stop_icon = AssetPath("stop_icon.png")
    Youtube_icon = AssetPath("youtube_icon.ico")
    Documentation_icon = AssetPath("documentation_icon.png")
    Pdf_icon = AssetPath("pdf_icon.png")
