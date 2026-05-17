"""
AQME utility helpers for easyROB.

This module provides domain-specific utilities used by the AQME tab, including
ChemDraw file handling and SMARTS pattern detection.

Responsibilities:
- Provide UI components for ChemDraw file selection
- Perform maximum common substructure (MCS) detection using RDKit
- Execute computationally expensive tasks in separate processes
- Bridge multiprocessing results back into the Qt event loop via signals

Architecture:
- Uses multiprocessing (Process + Queue) to avoid blocking the GUI
- Wraps background execution with Qt-compatible worker (MCSProcessWorker)
- Implements timeout handling to prevent long-running or stuck processes

Notes:
- Designed to isolate chemistry-heavy logic from GUI modules
- Ensures responsiveness when processing large molecular datasets

"""

# ------------------------------------------------------------
# Standard library
# ------------------------------------------------------------
import os
from multiprocessing import Process, Queue

# ------------------------------------------------------------
# Third-party libraries
# ------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import rdFMCS
from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

# ------------------------------------------------------------
# Local imports
# ------------------------------------------------------------
from .utils_gui import DropLabel

class ChemDrawFileDialog(QDialog):
    """Dialog that collects the main ChemDraw/SDF input file."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select ChemDraw Files")
        self.setMinimumWidth(500)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        self.setSizeGripEnabled(True)

        self.main_chemdraw_path = None
        self.optional_chemdraw_path = None

        layout = QVBoxLayout(self)

        title = QLabel("Please select your ChemDraw files")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.main_label = DropLabel(
            "Drag & Drop a main .sdf, .cdxml, or .mol",
            self,
            file_filter="ChemDraw Files (*.sdf *.cdxml *.mol *.cdx)",
            extensions=(".sdf", ".cdxml", ".mol", ".cdx"),
        )
        self.main_label.set_callback(self.set_main_file)
        layout.addWidget(self.main_label)

        self.continue_button = QPushButton("Continue")
        self.continue_button.setStyleSheet("padding: 8px; font-weight: bold;")
        self.continue_button.clicked.connect(self.continue_clicked)
        layout.addWidget(self.continue_button, alignment=Qt.AlignRight)

    def set_main_file(self, path):
        """Set the main ChemDraw file path and update the label."""
        self.main_chemdraw_path = path
        self.main_label.setText(f"Selected: {os.path.basename(path)}")

    def continue_clicked(self):
        """Check if a main ChemDraw file has been selected and accept the dialog."""
        if not self.main_chemdraw_path:
            QMessageBox.warning(self, "Missing File", "Please select a main ChemDraw file.")
            return
        self.accept()

def mcs_process(smiles_list, result_queue):
    """Find the maximum common substructure for a list of SMILES."""
    try:
        mol_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            mol_with_hs = Chem.AddHs(mol)
            if mol_with_hs:
                mol_list.append(mol_with_hs)

        if not mol_list:
            result_queue.put(("error", "No valid molecules."))
            return

        mcs_result = rdFMCS.FindMCS(mol_list)
        if mcs_result and mcs_result.smartsString:
            result_queue.put(("success", mcs_result.smartsString))
        else:
            result_queue.put(("error", "⚠️ No common SMARTS pattern found."))
    except Exception as exc:
        result_queue.put(("error", f"❌ MCS failed: {str(exc)}"))


class MCSProcessWorker(QObject):
    """Async wrapper around a subprocess-based MCS calculation."""

    finished = Signal(str)
    error = Signal(str)
    timeout = Signal()

    def __init__(self, smiles_list, timeout_ms=10000):
        super().__init__()
        self.smiles_list = smiles_list
        self.queue = Queue()
        self.process = None
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._on_timeout)
        self.timeout_ms = timeout_ms

    def start(self):
        """Start the MCS process and the timeout timer."""
        self.process = Process(target=mcs_process, args=(self.smiles_list, self.queue))
        self.process.start()
        self.timer.start(self.timeout_ms)
        QTimer.singleShot(100, self.check_result)

    def check_result(self):
        """Check if the MCS process has produced a result or if it is still running."""
        if not self.queue.empty():
            status, msg = self.queue.get()
            self.timer.stop()
            self.process.join()
            if status == "success":
                self.finished.emit(msg)
            else:
                self.error.emit(msg)
        elif self.process.is_alive():
            QTimer.singleShot(100, self.check_result)
        else:
            self.timer.stop()
            self.process.join()

    def _on_timeout(self):
        """Terminate the MCS process and emit a timeout signal."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            self.timeout.emit()