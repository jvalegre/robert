import sys
import os
import pandas as pd
from pathlib import Path
import subprocess
import shlex
import glob
import fitz  
from importlib.resources import files, as_file
from ansi2html import Ansi2HTMLConverter
from rdkit.Chem.rdmolfiles import MolsFromCDXMLFile
from rdkit.Chem.rdmolops import GetMolFrags
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import csv
from rdkit.Chem import rdFMCS
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
import rdkit
import shutil
import tempfile
import platform
import psutil
from functools import partial
from multiprocessing import Process, Queue
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget,
    QComboBox, QListWidget, QProgressBar, QMessageBox, QHBoxLayout, QFrame, QTabWidget, 
    QLineEdit, QTextEdit, QSizePolicy, QFormLayout, QGridLayout, QGroupBox, QCheckBox, 
    QScrollArea, QFileDialog, QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,QInputDialog,
    QSlider,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QPixmap, QPalette, QIcon, QImage, QMouseEvent, QWheelEvent
from PySide6.QtCore import (Qt, Slot, QThread, Signal, QObject, QUrl, QTimer, QEventLoop)

os.environ["QT_QUICK_BACKEND"] = "software"

class AQMETab(QWidget):
    def __init__(self, tab_parent=None, main_window=None, help_tab=None, web_view=None):

        super().__init__(tab_parent)  # tab_parent = QTabWidget
        self.main_tab_widget = tab_parent # Reference to the main QTabWidget
        self.main_window = main_window  # Reference to the main window, accessible to csv_df, csv_path, etc... 
        self.help_tab = help_tab
        self.web_view = web_view
        self.selected_atoms = []
        self.box_features = "QGroupBox { font-weight: bold; }"

        # === Main vertical layout ===
        main_layout = QVBoxLayout(self)

       # --- ChemDraw Button (modern purple style + top spacing) ---
        self.chemdraw_button = QPushButton("Generate CSV from ChemDraw Files or SDF file")
        self.chemdraw_button.setCursor(Qt.PointingHandCursor)
        self.chemdraw_button.setFixedSize(400, 42)

        self.chemdraw_button.setStyleSheet("""
            QPushButton {
                background-color: #7E57C2;
                color: white;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #6A42B8;
            }
            QPushButton:pressed {
                background-color: #5E35B1;
            }
        """)

        self.chemdraw_button.clicked.connect(self.open_chemdraw_popup)

        # Center button horizontally
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.chemdraw_button)
        button_layout.addStretch()

        # Add vertical spacing above the button
        button_container = QVBoxLayout()
        button_container.addSpacing(50)
        button_container.addLayout(button_layout)

        main_layout.addLayout(button_container)


        # === Viewer container with label + viewer stacked ===
        self.mol_viewer_container = QWidget()
        self.mol_viewer_container.setFixedSize(500, 500) 
        self.mol_viewer_container.setStyleSheet("background: transparent;")

        # Layout with relative positioning
        mol_layout = QGridLayout(self.mol_viewer_container)
        mol_layout.setContentsMargins(0, 0, 0, 0)
        mol_layout.setSpacing(0)

        # === mol_viewer (molecule display) ===
        self.mol_viewer = QLabel(self.mol_viewer_container)
        self.mol_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mol_viewer.setWordWrap(True)

        # Allow text selection
        self.mol_viewer.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard 
        )
        self.set_mol_viewer_message("📄 Select a CSV with a SMILES column to display a common SMARTS pattern.")
        self.mol_viewer.setFixedSize(500, 500)


        # === mol_info_label ===
        self.mol_info_label = QLabel("🔬 Info here", self.mol_viewer_container)
        self.mol_info_label.setStyleSheet("""
            color: #222;
            background-color: rgba(240, 240, 240, 220);
            font-size: 11px;
            font-style: italic;
            padding: 4px 8px;
            margin: 6px;
            border-radius: 6px;
            border: 1px solid #aaa;
        """)

        self.mol_info_label.setWordWrap(True)  
        self.mol_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.mol_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.mol_info_label.setMaximumWidth(600)  
        self.mol_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # === Set up the molecule viewer ===
        mol_layout.addWidget(self.mol_viewer, 0, 0)
        mol_layout.addWidget(self.mol_info_label, 0, 0, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        mol_wrapper_layout = QHBoxLayout()
        mol_wrapper_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mol_wrapper_layout.addWidget(self.mol_viewer_container)
        main_layout.addLayout(mol_wrapper_layout)

        # === AQME Box at the bottom ===
        aqme_box = QGroupBox("AQME")
        aqme_box.setMaximumHeight(115)  
        aqme_box.setStyleSheet(self.box_features)
        aqme_layout = QFormLayout()

        self.atoms = QLineEdit(placeholderText="e.g., Au or C=O")
        self.descriptor_level = QComboBox()
        self.descriptor_level.addItems(["interpret", "denovo", "full"])

        aqme_layout.addRow(QLabel("QDESCP Atoms:"), self.atoms)
        aqme_layout.addRow(QLabel("Descriptor Level:"), self.descriptor_level)

        # Help button
        help_button = QPushButton("Help AQME parameters")
        with AssetLibrary.Info_icon.get_path() as icon_path:
            help_button.setIcon(QIcon(str(icon_path)))

        help_button.setCursor(Qt.PointingHandCursor)
        help_button.setStyleSheet("padding: 4px; font-weight: bold;")
        help_button.clicked.connect(lambda: self.go_to_help_section("AQME"))

        aqme_layout.addRow("", help_button)
        aqme_layout.setAlignment(help_button, Qt.AlignRight)

        aqme_box.setLayout(aqme_layout)
        main_layout.addWidget(aqme_box)

    def go_to_help_section(self, anchor):
            """Navigates to the help section in the web view."""
            base_url = "https://robert.readthedocs.io/en/latest/Technical/defaults.html"
            full_url = f"{base_url}#{anchor.lower()}"
            self.main_tab_widget.setCurrentWidget(self.help_tab)
            self.web_view.setUrl(QUrl(full_url))

    def set_mol_viewer_message(self, message, tooltip=None):
        """Display a styled message in the molecule viewer, with optional tooltip."""
        self.mol_viewer.setText(message)
        self.mol_viewer.setToolTip(tooltip if tooltip else "")
        self.mol_viewer.setStyleSheet("""
            color: #222;
            background-color: rgba(255, 255, 255, 230);            
            font-size: 11px;
            font-style: italic;
            padding: 4px 8px;
            margin: 6px;
            border-radius: 6px;
            border: 1px solid #aaa;
        """)
    def detect_patterns_and_display(self):
        """Detects patterns in the loaded CSV and displays the first molecule."""

        try:
            self.csv_df = smart_read_csv(self.file_path) # Store the DataFrame for later use
            self.smiles_column = next((col for col in self.csv_df.columns if col.lower() == "smiles"), None)

            self.set_mol_viewer_message("🔬 Detecting common SMARTS pattern...")

            # === Auto SMARTS detection ===
            self.auto_pattern()

        except Exception as e:
            self.set_mol_viewer_message("❌ Failed to load or process the CSV.")
            self.mol_info_label.setText("🔬 Info here")

    def _on_mcs_success(self, smarts):
        """Handle successful MCS detection."""
        self.smarts_targets.append(smarts)
        self.mol_info_label.setText("🔬 Info here")
        self.display_molecule()

    def _on_mcs_error(self, message):
        """Handle MCS detection error."""
        self.set_mol_viewer_message(
            message,
            tooltip="SMARTS pattern detection failed."
        )
        self.mol_info_label.setText("🔬 Info here")

    def _on_mcs_timeout(self):
        """Handle MCS detection timeout."""
        self.set_mol_viewer_message(
            "⏱️ Timeout: MCS (Maximum Common Substructure) took too long and was aborted.",
            tooltip="SMARTS pattern detection failed."
        )
        self.mol_info_label.setText("🔬 Info here")


    def auto_pattern(self):
        """
        Auto-detect common SMARTS pattern in molecules from CSV 'SMILES' column.
        Uses a separate process to avoid blocking the GUI.
        """

        self.mol_info_label.setText("🔬 Info here")
        self.smarts_targets = []

        if self.smiles_column is None:
            return

        smiles_list = self.csv_df[self.smiles_column].dropna().tolist()

        # Instantiate a QThread for MCS detection
        self.mcs_worker = MCSProcessWorker(smiles_list, timeout_ms=30000) # 30 seconds timeout

        # Connect signals to handle success, error, and timeout
        self.mcs_worker.finished.connect(self._on_mcs_success)
        self.mcs_worker.error.connect(self._on_mcs_error)
        self.mcs_worker.timeout.connect(self._on_mcs_timeout)

        # Launch the MCS detection process
        self.mcs_worker.start()

    def display_molecule(self):
        """Display a SMARTS molecule and highlight atoms based on user selection."""
        rdkit.rdBase.DisableLog('rdApp.*')
        rdDepictor.SetPreferCoordGen(True)

        self.metal_atomic_numbers = {
            3, 11, 19, 37, 55, 87,
            4, 12, 20, 38, 56, 88,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            72, 73, 74, 75, 76, 77, 78, 79, 80,
            13, 49, 50, 81, 82, 83
        }

        try:
            self.metal_found = False
            self.metal_atoms_to_highlight = set()
            metal_found_in_this_mol = False

            if not self.smarts_targets:
                self.set_mol_viewer_message("⚠️ No SMARTS patterns available.")
                self.mol_info_label.setText("🔬 Info here")
                return

            pattern_mol = Chem.MolFromSmarts(self.smarts_targets[0])
            if pattern_mol is None:
                self.set_mol_viewer_message("⚠️ Invalid SMARTS pattern.")
                self.mol_info_label.setText("🔬 Info here")
                return

            self.multiple_matches_detected = False

            for smiles in self.csv_df[self.smiles_column]:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if mol is None:
                    continue

                matches = mol.GetSubstructMatches(pattern_mol)

                for match in matches:
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetAtomicNum() in self.metal_atomic_numbers:
                            metal_found_in_this_mol = True
                            self.metal_found = True
                            self.metal_atoms_to_highlight.add(idx)
                            break
                    if metal_found_in_this_mol:
                        break

                if len(matches) > 1:
                    self.multiple_matches_detected = True
                    if not metal_found_in_this_mol:
                        self.set_mol_viewer_message(
                            f"⚠️ <b>Multiple matches detected<b>: the common substructure '{self.smarts_targets[0]}' appears more than once in the molecule '{smiles}'. "
                            "Atomic descriptor selection has been disabled to avoid ambiguity."
                        )
                        self.mol_info_label.setText("🔬 Info here")
                        return

            self.mol = pattern_mol
            self.molecule_image_width = self.mol_viewer_container.width()
            self.molecule_image_height = self.mol_viewer_container.height()

            if self.metal_found and self.multiple_matches_detected:
                highlight_atoms = set(self.metal_atoms_to_highlight)
            else:
                highlight_atoms = set(self.selected_atoms)

            highlight_colors = {idx: (0.698, 0.4, 1.0) for idx in highlight_atoms} if highlight_atoms else {}

            drawer = rdMolDraw2D.MolDraw2DCairo(self.molecule_image_width, self.molecule_image_height)
            drawer.drawOptions().bondLineWidth = 1.5
            drawer.DrawMolecule(
                self.mol,
                highlightAtoms=list(highlight_atoms),
                highlightAtomColors=highlight_colors
            )
            drawer.FinishDrawing()

            png_bytes = drawer.GetDrawingText()
            pixmap = QPixmap()
            pixmap.loadFromData(png_bytes)
            self.atom_coords = [drawer.GetDrawCoords(i) for i in range(self.mol.GetNumAtoms())]

            if self.mol_viewer:
                if pixmap.isNull():
                    self.set_mol_viewer_message("⚠️ Could not render molecule image.")
                    self.mol_info_label.setText("🔬 Info here")
                else:
                    self.mol_viewer.setPixmap(pixmap)

                    if self.metal_found and self.multiple_matches_detected:
                        self.mol_info_label.setText(
                            '🧪 <b>SMARTS pattern loaded. Metal atom(s) automatically selected.</b><br>'
                            '<span style="color:red;">⚠️ Multiple matches were found. '
                            'Atomic descriptors will be generated for the detected metal atom(s). '
                            'Manual atom selection has been disabled to avoid ambiguity.</span>'
                        )
                    elif self.metal_found and not self.selected_atoms:
                        self.mol_info_label.setText(
                            '🧪 <b>SMARTS pattern loaded. Click to select atoms.</b><br>'
                            '<span style="color:red;">⚠️ No atoms selected. Descriptors will only be generated for the detected metal.</span>'
                        )
                    else:
                        if highlight_atoms:
                            self.mol_info_label.setText(f"🔬 {len(highlight_atoms)} atom(s) selected.")
                        else:
                            self.mol_info_label.setText(
                                '🧪 <b>SMARTS pattern loaded. Click to select atoms.</b><br>'
                                '<span style="color:red;">⚠️ WARNING! No atoms selected. Atomic descriptors will not be generated.</span>'
                            )

        except Exception as e:
            self.set_mol_viewer_message("❌ Error displaying molecule.", tooltip=str(e))
            self.mol_info_label.setText("🔬 Info here")

    def handle_atom_selection(self, atom_idx):
        """Handle the selection of an atom in the pattern."""

        if not hasattr(self, 'selected_atoms'):
            self.selected_atoms = []
        
        if getattr(self, 'metal_found', False) and getattr(self, 'multiple_matches_detected', False):
            # Prevent manual selection when metal match has been auto-selected due to ambiguity
            return

        # If the atom is already selected, deselect it
        if atom_idx in self.selected_atoms:
            self.selected_atoms.remove(atom_idx)
        else:
            # Otherwise, add the atom to the selection list
            self.selected_atoms.append(atom_idx)

        self.display_molecule()  # Update the visualization

        # Update the mapping regardless of selection or deselection
        self.generate_mapped_smiles(
            self.smarts_targets[0],
            self.selected_atoms,
            self.csv_df[self.smiles_column].dropna()
        )


    def generate_mapped_smiles(self, smarts_pattern, selected_pattern_indices, smiles_list):
        """
        Generate mapped SMILES using a SMARTS pattern and selected atom indices.
        Updates self.df_mapped_smiles with a copy of the original CSV where 'SMILES' is replaced.

        """

        # Parse the SMARTS pattern to a molecule object
        pattern_mol = Chem.MolFromSmarts(smarts_pattern)
        if pattern_mol is None:
            raise ValueError("Invalid SMARTS pattern")

        mapped_smiles = []

        for smiles in smiles_list:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None:
                mapped_smiles.append(None)
                continue

            # Get *all* substructure matches, continue processing if only one match is found
            matches = mol.GetSubstructMatches(pattern_mol)
            if len(matches) > 1:
                return  # Multiple matches found, return without processing
            elif not matches:
                mapped_smiles.append(None)
                continue

            # One match only → proceed
            match = matches[0]

            # Clear existing atom map numbers
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

            # Assign map numbers to selected atoms
            for i, pattern_idx in enumerate(selected_pattern_indices):
                if pattern_idx < len(match):
                    mol.GetAtomWithIdx(match[pattern_idx]).SetAtomMapNum(i + 1)

            mapped_smiles.append(Chem.MolToSmiles(mol))

        # Replace SMILES column in CSV
        df = smart_read_csv(self.file_path)
        df_mapped = df.copy()
        df_mapped[self.smiles_column] = mapped_smiles
        self.df_mapped_smiles = df_mapped


    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events to select atoms and crate pattern.
        The logic is to check if the mouse press event is within the molecule_viewer area."""

        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if self.mol_viewer_container and self.mol_viewer_container.geometry().contains(pos.toPoint()):
                relative_pos = self.mol_viewer_container.mapFrom(self, pos.toPoint())
                x = relative_pos.x()
                y = relative_pos.y()
                selected_atom = self.get_atom_at_position(x, y)
                if selected_atom is not None:
                    self.handle_atom_selection(selected_atom)
                    self.display_molecule()  

    def get_atom_at_position(self, x, y):
        """Get the atom index at the given position by 
        checking the distance from the atom coordinates. 
        The atom coordinates are found using RDKit.
        The logic is to check if the distance between the mouse click
        and the atom coordinates is less than a threshold."""

        if not hasattr(self, 'atom_coords'):
            return None
        elif self.atom_coords is not None:
            for idx, coord in enumerate(self.atom_coords):
                if len(self.smarts_targets[0]) <= 30: # small molecule = bigger click area
                    if (coord.x - x) ** 2 + (coord.y - y) ** 2 < 300: 
                        return idx 
                if len(self.smarts_targets[0]) <= 50 and len(self.smarts_targets[0]) > 30: # medium molecule = medium click area
                    if (coord.x - x) ** 2 + (coord.y - y) ** 2 < 200: 
                        return idx 
                elif len(self.smarts_targets[0]) > 50 : # big molecule = smaller click area 
                    if (coord.x - x) ** 2 + (coord.y - y) ** 2 < 100: 
                        return idx 
            return None

    def open_chemdraw_popup(self):
        """Open the ChemDraw file dialog and process selected file."""
        dialog = ChemDrawFileDialog(self)
        if dialog.exec():
            main_path = dialog.main_chemdraw_path
            self.load_chemdraw_file(main_path)

    def load_chemdraw_file(self, main_path):
        """Opens a ChemDraw file and displays the molecules in a table."""
        def load_mols_from_path(path):
            """Load molecules from a ChemDraw or SDF file."""
            if path.endswith('.cdxml'):
                try:
                    mols = MolsFromCDXMLFile(path, sanitize=False, removeHs=False)
                    total_count = len(mols)
                    valid_mols = []

                    for mol in mols:
                        if mol is not None:
                            fragments = GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                            valid_mols.extend(fragments)

                    valid_count = len(valid_mols)

                    if valid_count == 0:
                        QMessageBox.warning(self, "CDXML Warning", f"No valid molecules found in the file:\n{path}")
                        return []

                    elif valid_count < total_count:
                        failed_count = total_count - valid_count
                        QMessageBox.warning(
                            self,
                            "CDXML Partial Load",
                            f"File loaded with partial success.\n{failed_count} out of {total_count} molecules failed sanitization and were skipped."
                        )

                    return valid_mols

                except Exception as e:
                    QMessageBox.critical(self, "CDXML Read Error", f"Failed to read {path}:\n{str(e)}")
                    return []

            elif path.endswith('.sdf'):
                return [mol for mol in Chem.SDMolSupplier(path) if mol is not None]
            
            elif path.endswith(".cdx"): 
                QMessageBox.critical(
                    self,
                    "Unsupported CDX Format",
                    (
                        "The selected file is in CDX format, which is currently not supported for automatic import.To proceed, please use the ChemDraw application to export your structures in CDXML format.\n\n"
                        "➤ If you already have a ChemDraw file in CDX format, open it in ChemDraw and re-save or export it as a .cdxml file.\n\n"
                        "➤ If re-exporting the entire file is not feasible, you can also manually copy each individual molecule:\n\n"
                        "   1. Open the CDX file in ChemDraw.\n"
                        "   2. Select a molecule.\n"
                        "   3. Press Ctrl+C (or Cmd+C on Mac).\n"
                        "   4. Paste it (Ctrl+V or Cmd+V) into a new ChemDraw document.\n"
                        "   5. Save the new document as a CDXML file.\n\n"
                        "This ensures proper structure recognition and compatibility with the application.\n\n"
                    )
                )
                return []

            else:
                mol = Chem.MolFromMolFile(path)
                return [mol] if mol else []

        mols_main = load_mols_from_path(main_path)

        if not mols_main:
            QMessageBox.warning(self, "Error", "No valid molecules found in the file.")
            return

        self.show_molecule_table_dialog(mols_main)

    def show_molecule_table_dialog(self, mols):
        """
        Create and display a dialog showing a table of molecules,
        with the ability to add/remove columns, edit 'target' column name,
        and save the table as CSV. Includes various field validations.
        """
        # --- Dialog Setup ---
        dialog = QDialog(self)
        dialog.setWindowTitle("ChemDraw Molecules")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint)
        dialog.setSizeGripEnabled(True)
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        # --- Table Columns ---
        base_headers = ["Image", "SMILES", "code_name", "target"]
        extra_columns = ["charge", "mult", "complex_type", "sample", "geom"]
        complex_type_options = ["", "squareplanar", "squarepyramidal", "linear", "trigonalplanar"]

        # Table widget setup
        table = QTableWidget(len(mols), len(base_headers))
        table.setHorizontalHeaderLabels(base_headers)

        # Save indexes of required columns
        self.smiles_col_index = base_headers.index("SMILES")
        self.code_name_col_index = base_headers.index("code_name")
        self.target_col_index = base_headers.index("target")

        # --- Header Double Click Handler (for renaming 'target') ---
        def on_header_double_clicked(index):
            """
            Allow renaming ONLY for the 'target' column when double-clicked.
            """
            target_index = self.target_col_index
            if index != target_index:
                return  # Only allow renaming for the 'target' column
            current_text = table.horizontalHeaderItem(index).text()
            new_text, ok = QInputDialog.getText(
                dialog, "Edit Column Name",
                f"Rename column '{current_text}':", text=current_text
            )
            if ok and new_text.strip():
                table.setHorizontalHeaderItem(index, QTableWidgetItem(new_text.strip()))

        # Connect header double click signal
        table.horizontalHeader().sectionDoubleClicked.connect(on_header_double_clicked)

        # --- Populate Table with Molecule Data ---
        for row, mol in enumerate(mols):
            # Create image for molecule and put in cell (column 0)
            img = Draw.MolToImage(mol, size=(100, 100))
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            qimg = QImage.fromData(buffer.getvalue())
            label = QLabel()
            label.setPixmap(QPixmap.fromImage(qimg).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            widget = QWidget()
            hbox = QHBoxLayout()
            hbox.addWidget(label)
            hbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
            widget.setLayout(hbox)
            table.setRowHeight(row, 110)
            table.setCellWidget(row, 0, widget)

            # Set SMILES (column 1)
            smi = Chem.MolToSmiles(mol, canonical=False)
            table.setItem(row, 1, QTableWidgetItem(smi))
            # code_name (column 2), target (column 3) initialized empty
            table.setItem(row, 2, QTableWidgetItem(""))
            table.setItem(row, 3, QTableWidgetItem(""))

        # --- Set Default Table Column Widths ---
        default_width = 150
        for col in range(table.columnCount()):
            table.setColumnWidth(col, default_width)
        layout.addWidget(table)

        # --- Checkbox Controls for Optional Columns ---
        checkbox_layout = QHBoxLayout()
        checkboxes = {}

        def toggle_column(col_name, state):
            """
            Add or remove an extra column based on the corresponding checkbox.
            Handles special widget for 'complex_type' column.
            """
            def set_all_column_widths(width):
                for col in range(table.columnCount()):
                    table.setColumnWidth(col, width)

            current_headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
            if state:  # Checkbox checked: add column if not present
                if col_name not in current_headers:
                    idx = table.columnCount()
                    table.insertColumn(idx)
                    header_item = QTableWidgetItem(col_name)
                    header_item.setFlags(header_item.flags() & ~Qt.ItemIsEditable)
                    table.setHorizontalHeaderItem(idx, header_item)
                    for row in range(table.rowCount()):
                        if col_name == "complex_type":
                            combo = QComboBox()
                            combo.addItems(complex_type_options)
                            combo.setCurrentIndex(0)
                            table.setCellWidget(row, idx, combo)
                        else:
                            table.setItem(row, idx, QTableWidgetItem(""))
            else:  # Checkbox unchecked: remove column if present
                if col_name in current_headers:
                    idx = current_headers.index(col_name)
                    table.removeColumn(idx)
            set_all_column_widths(default_width)

        # Create checkboxes for each optional column
        for col_name in extra_columns:
            cb = QCheckBox(col_name)
            cb.stateChanged.connect(partial(toggle_column, col_name))
            checkbox_layout.addWidget(cb)
            checkboxes[col_name] = cb

        layout.addLayout(checkbox_layout)

        # --- Save as CSV Button ---
        save_button = QPushButton("💾 Save as CSV")
        save_button.setStyleSheet("padding: 6px; font-weight: bold;")
        layout.addWidget(save_button, alignment=Qt.AlignmentFlag.AlignRight)

        def save_to_csv():
            """
            Collect all table data and save to a CSV file.
            Includes validation for required fields, uniqueness, types, and empty checks.
            """
            headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]

            # --- Mandatory column presence check ---
            try:
                smiles_idx = headers.index("SMILES")
                code_name_idx = headers.index("code_name")
            except ValueError as e:
                QMessageBox.warning(dialog, "WARNING!", f"Column missing: {str(e)}")
                return

            # --- Data validation ---
            code_names = []
            for row in range(table.rowCount()):
                # Check 'SMILES' not empty
                item = table.item(row, smiles_idx)
                if not item or not item.text().strip():
                    QMessageBox.warning(dialog, "WARNING!", f"Please fill in all 'SMILES' fields before saving.")
                    return

                # Check 'code_name' not empty
                item = table.item(row, code_name_idx)
                if not item or not item.text().strip():
                    QMessageBox.warning(dialog, "WARNING!", f"Please fill in all 'code_name' fields before saving.")
                    return

                code_names.append(table.item(row, code_name_idx).text().strip())

                # Validate 'charge' column if present (must be int, not empty)
                if "charge" in headers:
                    charge_idx = headers.index("charge")
                    item = table.item(row, charge_idx)
                    val = item.text().strip() if item else ""
                    if val == "":
                        QMessageBox.warning(dialog, "WARNING!", f"Column 'charge' cannot be empty.")
                        return
                    if not (val.lstrip('-').isdigit() and '.' not in val):
                        QMessageBox.warning(dialog, "WARNING!", f"Column 'charge' must be an integer.")
                        return

                # Validate 'mult' column if present (must be int, not empty)
                if "mult" in headers:
                    mult_idx = headers.index("mult")
                    item = table.item(row, mult_idx)
                    val = item.text().strip() if item else ""
                    if val == "":
                        QMessageBox.warning(dialog, "WARNING!", f"Column 'mult' cannot be empty.")
                        return
                    if not (val.lstrip('-').isdigit() and '.' not in val):
                        QMessageBox.warning(dialog, "WARNING!", f"Column 'mult' must be an integer.")
                        return

                # Validate 'complex_type' if present (must be selected)
                if "complex_type" in headers:
                    complex_type_idx = headers.index("complex_type")
                    combo = table.cellWidget(row, complex_type_idx)
                    if combo is not None and combo.currentText().strip() == "":
                        QMessageBox.warning(
                            dialog, "WARNING!",
                            f"Column 'complex_type' cannot be empty. Please select a value."
                        )
                        return
                    
                # Validate 'sample' column if present (must be int, not empty)
                if "sample" in headers:
                    sample_idx = headers.index("sample")
                    for row in range(table.rowCount()):
                        item = table.item(row, sample_idx)
                        val = item.text().strip() if item else ""
                        if val == "":
                            QMessageBox.warning(dialog, "WARNING!", f"Column 'sample' cannot be empty.")
                            return
                        if not (val.lstrip('-').isdigit() and '.' not in val):
                            QMessageBox.warning(dialog, "WARNING!", f"Column 'sample' must be an integer.")
                            return

                # Validate 'GEOM' column if present (must not be empty)
                if "geom" in headers:
                    geom_idx = headers.index("geom")
                    for row in range(table.rowCount()):
                        item = table.item(row, geom_idx)
                        val = item.text().strip() if item else ""
                        if val == "":
                            QMessageBox.warning(dialog, "WARNING!", f"Column 'geom' cannot be empty.")
                            return


            # --- Uniqueness check for 'code_name' ---
            duplicates = [name for name in set(code_names) if code_names.count(name) > 1]
            if duplicates:
                QMessageBox.warning(
                    dialog, "WARNING!",
                    f"The following 'code_name' values are duplicated:\n\n{', '.join(duplicates)}\n\nPlease make them unique before saving."
                )
                return

            # --- Numeric check for 'target' column ---
            for row in range(table.rowCount()):
                item = table.item(row, self.target_col_index)
                val = item.text().strip() if item else ""
                if not val:
                    QMessageBox.warning(dialog, "WARNING!", f"Target column is empty.")
                    return
                try:
                    float(val)
                except ValueError:
                    QMessageBox.warning(dialog, "WARNING!", f"Target column must be numeric.")
                    return

            # --- File dialog to select save path ---
            path, _ = QFileDialog.getSaveFileName(dialog, "Save CSV", "", "CSV Files (*.csv)")
            if not path:
                return

            # --- Prepare headers and write CSV ---
            save_headers = [h for h in headers if h != "Image"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(save_headers)
                for row in range(table.rowCount()):
                    row_values = []
                    for col in range(table.columnCount()):
                        header = headers[col]
                        if header == "Image":
                            continue
                        if header == "complex_type":
                            combo = table.cellWidget(row, col)
                            row_values.append(combo.currentText() if combo else "")
                        else:
                            item = table.item(row, col)
                            row_values.append(item.text() if item else "")
                    writer.writerow(row_values)

            # --- Optional: Update main window file path and show message ---
            if hasattr(self, "main_window") and self.main_window:
                self.main_window.set_file_path(path)
            dialog.accept()
            QMessageBox.information(dialog, "Success", "CSV file saved and loaded successfully!")

        save_button.clicked.connect(save_to_csv)

        dialog.setLayout(layout)
        dialog.exec()

class AdvancedOptionsTab(QWidget):
    """Tab for advanced options in the easyROB application."""
    def __init__(self, type_dropdown, tab_widget, help_tab, web_view):
        super().__init__()
        self.type = type_dropdown
        self.tab_widget = tab_widget  # Reference to the main QTabWidget
        self.help_tab = help_tab
        self.web_view = web_view
        main_layout = QVBoxLayout(self)
        grid_layout = QGridLayout()  
        self.box_features = "QGroupBox { font-weight: bold; }"

        # Create section boxes
        general_box = self.create_general_section()
        curate_box = self.create_curate_section()
        generate_box = self.create_generate_section()
        predict_box = self.create_predict_section()

        # GENERAL (Top Row)
        grid_layout.addWidget(general_box, 0, 0, 1, 2)

        # CURATE / GENERATE (Middle Row)
        grid_layout.addWidget(curate_box, 1, 0, 1, 1)
        grid_layout.addWidget(generate_box, 1, 1, 1, 1)

        # PREDICT (Bottom Row, Full Width)
        grid_layout.addWidget(predict_box, 2, 0, 1, 2)


        # Add the grid layout to the main layout
        main_layout.addLayout(grid_layout)
        self.setLayout(main_layout)

    def go_to_help_section(self, anchor):
        """Navigates to the help section in the web view."""
        base_url = "https://robert.readthedocs.io/en/latest/Technical/defaults.html"
        
        if anchor.upper() == "GENERAL":
            full_url = base_url
        else:
            full_url = f"{base_url}#{anchor.lower()}"
        
        self.tab_widget.setCurrentWidget(self.help_tab)
        self.web_view.setUrl(QUrl(full_url))

    def create_help_button(self, topic: str) -> QPushButton:
        """Returns a styled Help button with icon and click behavior."""
        button = QPushButton(f"Help {topic.upper()} parameters")

        # Load icon using importlib.resources
        with AssetLibrary.Info_icon.get_path() as icon_path:
            button.setIcon(QIcon(str(icon_path)))

        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet("padding: 4px; font-weight: bold;")
        button.clicked.connect(lambda: self.go_to_help_section(topic))
        return button

    def create_general_section(self):
        """Creates the GENERAL section box."""
        box = QGroupBox("GENERAL")
        box.setStyleSheet(self.box_features)

        layout = QFormLayout()

        self.auto_type = QCheckBox()
        self.auto_type.setChecked(True)
        layout.addRow(QLabel("auto_type:"), self.auto_type)

        self.seed = QLineEdit()
        self.seed.setPlaceholderText("0")
        layout.addRow(QLabel("seed:"), self.seed)
        
        self.kfold = QLineEdit()
        self.kfold.setPlaceholderText("5")
        layout.addRow(QLabel("kfold:"), self.kfold)

        self.repeat_kfolds = QLineEdit()
        self.repeat_kfolds.setPlaceholderText("10")
        layout.addRow(QLabel("repeat_kfolds:"), self.repeat_kfolds)

        # --- Help button at the bottom ---
        help_button = self.create_help_button("GENERAL")

        # Add with right alignment
        layout.addRow("", help_button)  # Adds it as a new row with empty label
        layout.setAlignment(help_button, Qt.AlignRight)

        box.setLayout(layout)
        return box

    def create_curate_section(self):
        """Creates the CURATE section with a box and input fields."""
        box = QGroupBox("CURATE")
        box.setStyleSheet(self.box_features)  
        layout = QFormLayout()

        # Add new input fields for additional options
        self.categoricalstr = QComboBox()
        self.categoricalstr.addItems(["onehot", "numbers"])
        layout.addRow(QLabel("categorical:"), self.categoricalstr)

        self.corr_filter_xbool = QCheckBox()
        self.corr_filter_xbool.setChecked(True)
        layout.addRow(QLabel("corr_filter_x:"), self.corr_filter_xbool)

        self.corr_filter_ybool = QCheckBox()
        self.corr_filter_ybool.setChecked(False)
        layout.addRow(QLabel("corr_filter_y:"), self.corr_filter_ybool)

        self.desc_thresfloat = QLineEdit()
        self.desc_thresfloat.setPlaceholderText("25")
        layout.addRow(QLabel("desc_thres:"), self.desc_thresfloat)

        self.thres_xfloat = QLineEdit()
        self.thres_xfloat.setPlaceholderText("0.7")
        layout.addRow(QLabel("thres_x:"), self.thres_xfloat)

        self.thres_yfloat = QLineEdit()
        self.thres_yfloat.setPlaceholderText("0.001")
        layout.addRow(QLabel("thres_y:"), self.thres_yfloat)

        # --- Help button at the bottom ---
        help_button = self.create_help_button("CURATE")

        # Add with right alignment
        layout.addRow("", help_button)  # Adds it as a new row with empty label
        layout.setAlignment(help_button, Qt.AlignRight)

        box.setLayout(layout)
        return box

    def create_generate_section(self):
        """Creates the GENERATE section with a box and input fields."""
        box = QGroupBox("GENERATE")
        box.setStyleSheet(self.box_features)  
        layout = QFormLayout()

        self.model_group = QGroupBox("Models")
        self.model_layout = QGridLayout()  # Grid layout for better spacing
        self.modellist = {}
        self.model_group.setLayout(self.model_layout)
        layout.addRow(self.model_group)

        def update_model_options():
            """Updates the model options based on the selected type."""

            # Determine which models should be checked by default
            if self.type.currentText() == "Regression":
                default_checked_models = ["RF", "GB", "NN", "MVL"]  # Regression defaults
            else:
                default_checked_models = ["RF", "GB", "NN", "AdaB"]  # Classification defaults

            # Update check states instead of recreating widgets
            for model, checkbox in self.modellist.items():
                checkbox.setChecked(model in default_checked_models)

        # Create checkboxes (only once)
        all_models = ["RF", "MVL", "GB", "NN", "GP", "AdaB"]
        row, col = 0, 0
        for model in all_models:
            checkbox = QCheckBox(model)
            self.modellist[model] = checkbox
            self.model_layout.addWidget(checkbox, row, col)

            col += 1  # Move to the next column
            if col > 1:  # Two columns max
                col = 0
                row += 1

        # Connect signal to update check states when type changes
        self.type.currentIndexChanged.connect(update_model_options)
        update_model_options()  # Initialize with correct models

        # Error type selection that changes dynamically but is also user-selectable
        self.error_type = QComboBox()
        layout.addRow(QLabel("error_type:"), self.error_type)
        
        def update_error_type():
            self.error_type.clear()
            if self.type.currentText() == "Regression":
                self.error_type.addItems(["rmse", "mae", "r2"])
            else:
                self.error_type.addItems(["mcc", "f1", "acc"])
        
        self.type.currentIndexChanged.connect(update_error_type)
        update_error_type()  # Initialize with the correct default values

        self.init_points = QLineEdit()
        self.init_points.setPlaceholderText("10")
        layout.addRow(QLabel("init_points:"), self.init_points)

        self.n_iter = QLineEdit()
        self.n_iter.setPlaceholderText("10")
        layout.addRow(QLabel("n_iter:"), self.n_iter)

        self.expect_improv = QLineEdit()
        self.expect_improv.setPlaceholderText("0.05")
        layout.addRow(QLabel("expect_improv:"), self.expect_improv)

        self.pfi_filter = QCheckBox()
        self.pfi_filter.setChecked(True)
        layout.addRow(QLabel("pfi_filter:"), self.pfi_filter)

        self.pfi_epochs = QLineEdit()
        self.pfi_epochs.setPlaceholderText("5")
        layout.addRow(QLabel("pfi_epochs:"), self.pfi_epochs)

        self.pfi_threshold = QLineEdit()
        self.pfi_threshold.setPlaceholderText("0.2")
        layout.addRow(QLabel("pfi_threshold:"), self.pfi_threshold)

        self.pfi_max = QLineEdit()
        self.pfi_max.setPlaceholderText("0")
        layout.addRow(QLabel("pfi_max:"), self.pfi_max)

        self.auto_test = QCheckBox()
        self.auto_test.setChecked(True)
        layout.addRow(QLabel("auto_test:"), self.auto_test)

        self.test_set = QLineEdit()
        self.test_set.setPlaceholderText("0.1")
        layout.addRow(QLabel("test_set:"), self.test_set)

        # --- Help button at the bottom ---
        help_button = self.create_help_button("GENERATE")

        # Add with right alignment
        layout.addRow("", help_button)  # Adds it as a new row with empty label
        layout.setAlignment(help_button, Qt.AlignRight)

        box.setLayout(layout)
        return box

    def create_predict_section(self):
        """Creates the PREDICT section with a box and input fields."""
        box = QGroupBox("PREDICT")
        box.setStyleSheet(self.box_features)  
        layout = QFormLayout()
        
        self.t_value = QLineEdit()
        self.t_value.setPlaceholderText("2")
        layout.addRow(QLabel("t_value:"), self.t_value)
        
        self.shap_show = QLineEdit()
        self.shap_show.setPlaceholderText("10")
        layout.addRow(QLabel("shap_show:"), self.shap_show)
        
        self.pfi_show = QLineEdit()
        self.pfi_show.setPlaceholderText("10")
        layout.addRow(QLabel("pfi_show:"), self.pfi_show)

        # --- Help button at the bottom ---
        help_button = self.create_help_button("PREDICT")

        # Add with right alignment
        layout.addRow("", help_button)  # Adds it as a new row with empty label
        layout.setAlignment(help_button, Qt.AlignRight)

        box.setLayout(layout)
        return box

class ResultsTab(QWidget):
    def __init__(self, main_tab_widget, file_path):
        super().__init__()

        self.main_tab_widget = main_tab_widget
        self.base_path = os.path.dirname(file_path)
        self.pdf_tabs = {}

        # Internal tab widget for PDF files
        self.pdf_tab_widget = QTabWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.pdf_tab_widget)
        self.setLayout(layout)

        # Check for existing PDFs 
        self.check_for_pdfs()

    def refresh_with_new_path(self, file_path):
        """Updates the path and reloads PDF tabs."""
        self.base_path = os.path.dirname(file_path)
        self.clear_pdf_tabs()
        self.check_for_pdfs()

    def clear_pdf_tabs(self):
        """Remove all existing PDF tabs."""
        for pdf_viewer in self.pdf_tabs.values():
            index = self.pdf_tab_widget.indexOf(pdf_viewer)
            if index != -1:
                self.pdf_tab_widget.removeTab(index)
        self.pdf_tabs.clear()

    def check_for_pdfs(self):
        """Checks for new PDFs and updates the UI dynamically."""
        pdf_pattern = os.path.join(self.base_path, "ROBERT_report*.pdf")
        pdf_files = sorted(glob.glob(pdf_pattern))

        # Remove missing PDFs
        for pdf in list(self.pdf_tabs.keys()):
            if pdf not in pdf_files:
                index = self.pdf_tab_widget.indexOf(self.pdf_tabs[pdf])
                if index != -1:
                    self.pdf_tab_widget.removeTab(index)
                del self.pdf_tabs[pdf]

        # Add new PDFs as tabs
        for pdf in pdf_files:
            if pdf not in self.pdf_tabs:
                self.add_pdf_tab(pdf)

    def add_pdf_tab(self, pdf_path):
        """Creates a new internal tab displaying the PDF."""
        pdf_viewer = PDFViewer(pdf_path)
        index = self.pdf_tab_widget.addTab(pdf_viewer, os.path.basename(pdf_path))
        self.pdf_tabs[pdf_path] = pdf_viewer
        self.pdf_tab_widget.setCurrentIndex(index)

class PDFViewer(QWidget):
    """Widget to display a PDF inside a scrollable area with zoom control and threading."""
    def __init__(self, pdf_path):
        super().__init__()
        self.pdf_path = pdf_path
        self.current_zoom = 2.0  # Default zoom
        self.image_cache = {}  # {(page_num, zoom): QPixmap}
        self.threads = []  # Keep references to threads to avoid premature garbage collection

        layout = QVBoxLayout(self)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 60)
        self.zoom_slider.setValue(int(self.current_zoom * 10))
        self.zoom_slider.valueChanged.connect(self.on_zoom_change)
        layout.addWidget(self.zoom_slider)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        self.container = QWidget()
        self.scroll_area.setWidget(self.container)
        self.vbox = QVBoxLayout(self.container)

        self.render_pdf()

    def on_zoom_change(self, value):
        new_zoom = value / 10.0
        if new_zoom != self.current_zoom:
            self.current_zoom = new_zoom
            self.render_pdf()

    def render_pdf(self):
        # Clear UI
        while self.vbox.count():
            child = self.vbox.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clear threads (let them finish in background if running)
        self.threads.clear()

        doc = fitz.open(self.pdf_path)
        self.page_count = len(doc)
        doc.close()

        for page_num in range(self.page_count):
            placeholder = QLabel()
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.vbox.addWidget(placeholder)

            key = (page_num, self.current_zoom)
            if key in self.image_cache:
                placeholder.setPixmap(self.image_cache[key])
            else:
                thread = PageRenderThread(self.pdf_path, page_num, self.current_zoom)
                thread.finished.connect(self.on_page_rendered)
                thread.start()
                self.threads.append(thread)  # keep reference to avoid garbage collection

    def on_page_rendered(self, page_num, zoom, pixmap):
        if zoom != self.current_zoom:
            return  # ignore outdated render

        self.image_cache[(page_num, zoom)] = pixmap

        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Replace placeholder
        self.vbox.itemAt(page_num).widget().deleteLater()
        self.vbox.insertWidget(page_num, label)

class ImagesTab(QWidget):
    """Images tab for displaying images from multiple folders as results of Robert workflow."""

    def __init__(self, main_tab_widget, image_folders, file_path):
        super().__init__()

        self.main_tab_widget = main_tab_widget
        self.image_folders = image_folders
        self.base_path = os.path.dirname(file_path)
        self.folder_widgets = {}

        self.folder_tabs = QTabWidget()
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.folder_tabs)
        self.setLayout(self.layout)

        self.check_for_images()

    def refresh_with_new_path(self, file_path):
        """Update image base path and refresh image tabs."""
        self.base_path = os.path.dirname(file_path)
        self.clear_image_tabs()
        self.check_for_images()

    def clear_image_tabs(self):
        """Clear all image folders and their widgets."""
        for i in reversed(range(self.folder_tabs.count())):
            widget = self.folder_tabs.widget(i)
            if widget:
                widget.deleteLater()
            self.folder_tabs.removeTab(i)
        self.folder_widgets.clear()

    def check_for_images(self):
        """Scan folders and update tabs with new images."""
        folder_names = {
            "CURATE": "CURATE",
            "GENERATE/Raw_data": "GENERATE",
            "PREDICT": "PREDICT",
            "VERIFY": "VERIFY",
        }

        folder_order = ["CURATE", "GENERATE/Raw_data", "PREDICT", "VERIFY"]

        for folder in folder_order:
            full_folder_path = os.path.join(self.base_path, folder)
            if not os.path.exists(full_folder_path):
                continue

            image_files = sorted(glob.glob(os.path.join(full_folder_path, "*.[pjg][np][g]")))

            if folder not in self.folder_widgets:
                folder_widget = QWidget()
                folder_layout = QVBoxLayout(folder_widget)
                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)

                image_grid = QGridLayout()
                container = QWidget()
                container.setLayout(image_grid)
                scroll_area.setWidget(container)
                folder_layout.addWidget(scroll_area)
                folder_widget.setLayout(folder_layout)

                tab_name = folder_names.get(folder, os.path.basename(folder))
                self.folder_tabs.addTab(folder_widget, tab_name)
                self.folder_widgets[folder] = image_grid

            image_grid = self.folder_widgets[folder]

            while image_grid.count():
                item = image_grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            row, col = 0, 0
            max_columns = 3
            for img_path in image_files:
                image_label = ImageLabel(img_path, size=300)
                image_grid.addWidget(image_label, row, col)
                col += 1
                if col >= max_columns:
                    col = 0
                    row += 1

class ImageLabel(QLabel):
    """Custom QLabel for displaying an image with right-click copy functionality in the "Images" tab."""

    def __init__(self, image_path, size=400):  
        super().__init__()

        self.image_path = image_path
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Load image
        pixmap = QPixmap(self.image_path)
        if pixmap.isNull():
            self.setText("Failed to load image.")
        else:
            self.setPixmap(pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # Enable right-click copying
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, position):
        """Creates a right-click menu for copying the image."""
        menu = QMessageBox()
        menu.setWindowTitle("Image Options")
        menu.setText(f"Copy image: {self.image_path}?")
        menu.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if menu.exec() == QMessageBox.StandardButton.Yes:
            clipboard = QApplication.clipboard()
            clipboard.setPixmap(QPixmap(self.image_path))

class EasyROB(QMainWindow):
    """Main window for the easyROB application."""
    def __init__(self):
        super().__init__()
        self.file_path = ""
        self.csv_test_path = ""
        self.process = None  
        self.available_list = None
        self.ignore_list = None
        self.manual_stop = False
        self.worker = None
        self._last_loaded_file_path = None
        self.initUI()
        self.clear_test_button.setVisible(False) # Hide the button initially

    def closeEvent(self, event):
        """Intercept window close to clean up running threads safely."""
        worker = getattr(self, 'worker', None)
        if worker is not None and worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Exit Confirmation",
                "ROBERT is still running. Do you want to stop the process and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                loop = QEventLoop()
                worker.process_finished.connect(loop.quit)

                worker.stop()  # safely stop the process tree

                # Safety timeout: stop waiting after 5 seconds
                QTimer.singleShot(5000, loop.quit)
                loop.exec()

                if worker.isRunning():
                    print("[DEBUG] Worker did not stop in time, forcing exit anyway.")
                else:
                    print("[DEBUG] Worker stopped successfully.")

                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def move_to_selected(self):
        """Move selected items from available_list to selected_list."""
        selected_items = self.available_list.selectedItems()
        for item in selected_items:
            self.ignore_list.addItem(item.text())  #  Add to right list
            row = self.available_list.row(item)  #  Get correct row index
            self.available_list.takeItem(row)  #  Remove from left list

    def move_to_available(self):
        """Move selected items from selected_list back to available_list."""
        selected_items = self.ignore_list.selectedItems()
        for item in selected_items:
            self.available_list.addItem(item.text())  #  Add back to left list
            row = self.ignore_list.row(item)  #  Get correct row index
            self.ignore_list.takeItem(row)  #  Remove from right list

    def clear_test_file(self):
        """Clear the selected test file."""
        self.csv_test_path = None
        self.csv_test_label.setText("Drag & Drop a CSV test file here (optional)")
        self.clear_test_button.setVisible(False)

    def initUI(self):
        """Initializes the main user interface."""
        # Parameters for the GUI
        font_size = '14px' # Font size for the titles
        box_features = "QComboBox { border: 1px solid gray; }" # Styling for the combo 
        box_features_ignore = "QListWidget { border: 1px solid gray; }"  # Styling for the list widget

        self.setWindowTitle("easyROB")
        self.setGeometry(100, 100, 800, 400)  
        
        # Create a QTabWidget to hold the tabs.
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # ===============================
        # "Main" Tab (Original Interface)
        # ===============================
        # Create scrollable area for the "Robert" tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # main_tab is the scrollable container
        main_tab = QWidget()
        scroll_area.setWidget(main_tab)

        # Use this layout to build your GUI
        main_layout = QVBoxLayout(main_tab)

        # Add to the QTabWidget
        self.tab_widget.addTab(scroll_area, "ROBERT")

        # --- Add logo with frame ---
        with AssetLibrary.Robert_logo_transparent.get_path() as path_logo:
            pixmap = QPixmap(str(path_logo))
            scaled_pixmap = pixmap.scaled(400, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            logo_label = QLabel(self)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)

            logo_frame = QFrame()
            logo_layout = QVBoxLayout()
            logo_layout.addWidget(logo_label, alignment=Qt.AlignCenter)
            logo_frame.setLayout(logo_layout)

            main_layout.addWidget(logo_frame, alignment=Qt.AlignCenter)

        # --- Set window icon ---
        with AssetLibrary.Robert_icon.get_path() as path_icon:
            self.setWindowIcon(QIcon(str(path_icon)))

        # --- Input CSV File (Required) ---
        input_layout = QVBoxLayout()
        self.file_title = QLabel("Select Input CSV File", self)
        self.file_title.setAlignment(Qt.AlignCenter)
        self.file_title.setStyleSheet(f"font-weight: bold; font-size: {font_size};")
        self.file_label = DropLabel(
            "Drag & Drop a CSV file here",
            self,
            file_filter="CSV Files (*.csv)",
            extensions=(".csv",)
        )
        self.file_label.set_callback(self.set_file_path)        
        input_layout.addWidget(self.file_title)
        input_layout.addWidget(self.file_label)

        # --- Test CSV File (Optional) ---
        test_layout = QVBoxLayout()

        self.csv_test_title = QLabel("Select Test CSV File (optional)", self)
        self.csv_test_title.setAlignment(Qt.AlignCenter)
        self.csv_test_title.setStyleSheet(f"font-weight: bold; font-size: {font_size};")

        self.csv_test_label = DropLabel(
            "Drag & Drop a CSV test file here (optional)",
            self,
            file_filter="CSV Files (*.csv)",
            extensions=(".csv",)
        )
        self.csv_test_label.set_callback(self.set_csv_test_path)

        self.clear_test_button = QPushButton("✖")
        self.clear_test_button.setFixedSize(30, 30)
        self.clear_test_button.setStyleSheet(
            "background-color: #900; color: white; font-weight: bold; border-radius: 5px;"
        )
        self.clear_test_button.setToolTip("Clear selected test CSV file")
        self.clear_test_button.clicked.connect(self.clear_test_file)

        test_label_container = QWidget()
        test_label_inner_layout = QHBoxLayout(test_label_container)
        test_label_inner_layout.setContentsMargins(0, 0, 0, 0)
        test_label_inner_layout.addWidget(self.csv_test_label)
        test_label_inner_layout.addWidget(self.clear_test_button)

        test_layout.addWidget(self.csv_test_title)
        test_layout.addWidget(test_label_container)


        # --- CSV Section with Button in the Middle ---
        csv_layout = QHBoxLayout()
        csv_layout.addLayout(input_layout)
        csv_layout.addLayout(test_layout)

        # --- Add All to Main Layout ---
        main_layout.addLayout(csv_layout)
   
        # --- Select column for --y ---
        self.y_label = QLabel("Select Target Column (y)")
        self.y_label.setStyleSheet(f"font-weight: bold; font-size: {font_size};")
        main_layout.addWidget(self.y_label)
        self.y_dropdown = NoScrollComboBox()        
        main_layout.addWidget(self.y_dropdown)
        self.y_dropdown.setStyleSheet(box_features)
        
        # --- Select prediction type ---
        self.type_label = QLabel("Prediction Type")
        self.type_label.setStyleSheet(f"font-weight: bold; font-size: {font_size};")
        main_layout.addWidget(self.type_label)
        self.type_dropdown = NoScrollComboBox()
        self.type_dropdown.addItems(["Regression", "Classification"])
        main_layout.addWidget(self.type_dropdown)
        self.type_dropdown.setStyleSheet(box_features)
        
        # --- Select column for --names ---
        self.names_label = QLabel("Select name column")
        self.names_label.setStyleSheet(f"font-weight: bold; font-size: {font_size};")
        main_layout.addWidget(self.names_label)
        self.names_dropdown = NoScrollComboBox()
        main_layout.addWidget(self.names_dropdown) 
        self.names_dropdown.setStyleSheet(box_features)
     
        # Main horizontal layout for column selection
        column_layout = QHBoxLayout()

        # Left side (Available Columns)
        left_layout = QVBoxLayout()
        self.available_label = QLabel("Available Columns")
        self.available_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.MultiSelection)
        self.available_list.setStyleSheet(box_features_ignore)
        left_layout.addWidget(self.available_label)
        left_layout.addWidget(self.available_list)

        # Button layout (Centered between lists)
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignVCenter)  # Ensure vertical centering

        self.add_button = QPushButton(">>")
        self.add_button.setFixedSize(40, 30)
        self.add_button.clicked.connect(self.move_to_selected)   # Moves selected items to "Ignored Columns"
        self.remove_button = QPushButton("<<")
        self.remove_button.setFixedSize(40, 30)
        self.remove_button.clicked.connect(self.move_to_available)  # Moves selected items back to "Available Columns"

        # Add buttons to the button layout
        button_layout.addStretch()  
        button_layout.addWidget(self.add_button, alignment=Qt.AlignCenter)
        button_layout.addWidget(self.remove_button, alignment=Qt.AlignCenter)
        button_layout.addStretch()  

        # Right side (Ignored Columns)
        right_layout = QVBoxLayout()
        self.ignored_label = QLabel("Ignored Columns")
        self.ignored_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.ignore_list = QListWidget()
        self.ignore_list.setSelectionMode(QListWidget.MultiSelection)
        self.ignore_list.setStyleSheet(box_features_ignore)
        right_layout.addWidget(self.ignored_label)
        right_layout.addWidget(self.ignore_list)

        # Add sub-layouts to the main horizontal layout
        column_layout.addLayout(left_layout)
        column_layout.addLayout(button_layout)
        column_layout.addLayout(right_layout)

        # Create a container for the column layout and resize it
        column_container = QWidget()
        column_container.setLayout(column_layout)
        column_container.setFixedHeight(150) 

        # Insert the column container into the main layout
        main_layout.addWidget(column_container)
        main_layout.addSpacing(10)

        # AQME Workflow Checkbox
        self.aqme_workflow = QCheckBox("Enable AQME Workflow") 
        self.aqme_workflow.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.aqme_workflow.stateChanged.connect(self.check_aqme_workflow)
        main_layout.addWidget(self.aqme_workflow)
        main_layout.addSpacing(10)  

        # Workflow selection dropdown
        self.workflow_selector = NoScrollComboBox()
        self.workflow_selector.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Add options
        self.workflow_selector.addItems([
            "Full Workflow",
            "CURATE",
            "GENERATE",
            "PREDICT",
            "VERIFY",
            "REPORT"
        ])

        # Set default selection
        self.workflow_selector.setCurrentText("Full Workflow")

        # Add to layout
        main_layout.addWidget(self.workflow_selector)
        main_layout.addSpacing(10)

        # --- Run button ---
        self.run_button = QPushButton(" Run ROBERT")
        self.run_button.setFixedSize(200, 40)  # Adjust button size

        with AssetLibrary.Play_icon.get_path() as icon_play_path:
            self.run_button.setIcon(QIcon(str(icon_play_path)))

        # Apply button styling
        self.run_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
                border-radius: 10px;
                background-color: #444;
                color: white;
                border: 2px solid #666;
            }
            QPushButton:hover {
                background-color: #666;
                border: 2px solid #888;
            }
            QPushButton:pressed {
                background-color: #222;
                border: 2px solid #444;
            }
        """)

        self.run_button.clicked.connect(self.run_robert)

        # --- Stop Button ---
        self.stop_button = QPushButton("Stop ROBERT")
        self.stop_button.setFixedSize(200, 40)

        with AssetLibrary.Stop_icon.get_path() as icon_stop_path:
            self.stop_button.setIcon(QIcon(str(icon_stop_path)))

        self.stop_button.setDisabled(True)  # Initially disabled
        self.stop_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
                border-radius: 10px;
                background-color: #b00000;
                color: white;
                border: 2px solid #900000;
            }
            QPushButton:hover {
                background-color: #d00000;
                border: 2px solid #a00000;
            }
            QPushButton:pressed {
                background-color: #800000;
                border: 2px solid #600000;
            }
        """)
        self.stop_button.clicked.connect(self.stop_robert)

        # Add button layout to the main layout
        button_container = QHBoxLayout()
        button_container.addWidget(self.run_button)
        button_container.addWidget(self.stop_button)
        main_layout.addLayout(button_container)

        # --- Console Output Setup ---
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: black;
                color: white;
                padding: 5px;
                font-family: monospace;
            }
            QScrollBar:vertical {
                background: #2e2e2e;
                width: 12px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #5a5a5a;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #787878;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                background: none;
                height: 0px;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        # Set minimum height for the console output and make it expandable
        self.console_output.setMinimumHeight(275)  
        self.console_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create ANSI converter to display colors in the console and special characters
        self.ansi_converter = Ansi2HTMLConverter(dark_bg=True)  # Preserves colors
        main_layout.addWidget(QLabel("Console Output"))
        main_layout.addWidget(self.console_output, stretch=1)

        # --- Progress bar ---
        main_layout.addStretch()
        self.progress = QProgressBar()
        self.progress.setFixedHeight(10)  # Adjust height for a sleeker look
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid gray;
                border-radius: 10px;
                background: #f0f0f0;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 5px;
                border-radius: 10px;
            }
        """)
        main_layout.addWidget(self.progress)

        # ============
        # Create Tabs
        # ============

        # Results tab (must be created early so others can reference it)
        self.results_tab = QTabWidget()
        self.results_tab = ResultsTab(self.tab_widget, self.file_path)

        # Help tab
        self.help_tab = QWidget()
        help_layout = QVBoxLayout(self.help_tab)
        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl("https://robert.readthedocs.io/en/latest/index.html#"))
        help_layout.addWidget(self.web_view)

        # AQME tab (depends on ResultsTab via main_window)
        self.tab_widget_aqme = AQMETab(
            tab_parent=self.tab_widget,
            main_window=self,
            help_tab=self.help_tab,
            web_view=self.web_view
        )

        # Options tab
        self.options_tab = AdvancedOptionsTab(
            self.type_dropdown,
            self.tab_widget,
            self.help_tab,
            self.web_view
        )

        # Images tab
        self.image_folders = ["PREDICT", "GENERATE/Raw_data", "VERIFY", "CURATE"]
        self.images_tab = ImagesTab(self.tab_widget, self.image_folders, self.file_path)

        # ===============================
        # Add Tabs to Tab Widget (Display order)
        # ===============================

        self.tab_widget.addTab(self.tab_widget_aqme, "AQME")
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_widget_aqme), False)

        self.tab_widget.addTab(self.options_tab, "Advanced Options")
        self.tab_widget.addTab(self.help_tab, "Help")

        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.results_tab), False)

        self.tab_widget.addTab(self.images_tab, "Images")
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.images_tab), False)

    def check_for_images(self):
        """Enable or disable the 'Images' tab based on folder existence."""
        if not hasattr(self, 'file_path'):
            return  # Exit early if file_path is not set

        run_dir = os.path.dirname(self.file_path)

        has_folders = any(
            os.path.exists(os.path.join(run_dir, folder)) for folder in self.image_folders
        )

        tab_index = self.tab_widget.indexOf(self.images_tab)
        if tab_index != -1:
            self.tab_widget.setTabEnabled(tab_index, has_folders)

    def check_for_pdfs(self):
        """Enable or disable the 'Results' tab based on PDF presence."""
        if not hasattr(self, 'file_path'):
            return  # Exit early if file_path is not set

        run_dir = os.path.dirname(self.file_path)
        pdf_pattern = os.path.join(run_dir, "ROBERT_report*.pdf")

        has_pdfs = bool(glob.glob(pdf_pattern))

        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.results_tab), has_pdfs)

    def check_aqme_workflow(self):
        """Enable or disable the AQME tab and refresh its content based on checkbox state and file path."""
        is_checked = self.aqme_workflow.isChecked()
        tab_index = self.tab_widget.indexOf(self.tab_widget_aqme)

        if tab_index == -1:
            return

        if is_checked:
            # Enable the AQME tab if not already enabled
            if not self.tab_widget.isTabEnabled(tab_index):
                self.tab_widget.setTabEnabled(tab_index, True)
                QMessageBox.information(self, "AQME Tab Enabled", "AQME tab unlocked to specify AQME parameters.")

            # Always refresh AQME tab content if file path is available
            if hasattr(self, 'file_path') and self.file_path:
                self.tab_widget_aqme.selected_atoms = []
                self.tab_widget_aqme.file_path = self.file_path
                self.tab_widget_aqme.detect_patterns_and_display()
                self._last_loaded_file_path = self.file_path

        else:
            # Disable the AQME tab if the checkbox is unchecked
            if self.tab_widget.isTabEnabled(tab_index):
                self.tab_widget.setTabEnabled(tab_index, False)

    def refresh_tabs(self):
        """Remove and recreate the Images and Results tabs using the current file_path."""

        # Remove and delete the existing Images tab if it exists
        if hasattr(self, "images_tab"):
            index = self.tab_widget.indexOf(self.images_tab)
            if index != -1:
                self.tab_widget.removeTab(index)
            self.images_tab.deleteLater()
            del self.images_tab

        # Remove and delete the existing Results tab if it exists
        if hasattr(self, "results_tab"):
            index = self.tab_widget.indexOf(self.results_tab)
            if index != -1:
                self.tab_widget.removeTab(index)
            self.results_tab.deleteLater()
            del self.results_tab

        # Recreate the tabs using the current file_path
        self.images_tab = ImagesTab(self.tab_widget, self.image_folders, self.file_path)
        self.results_tab = ResultsTab(self.tab_widget, self.file_path)

        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.addTab(self.images_tab, "Images")
            
    def select_file(self):
        """Opens file dialog to select a CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.set_file_path(file_path)

    def select_csv_test_file(self):
        """Opens file dialog to select a test CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Test CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.set_csv_test_path(file_path)

    def set_file_path(self, file_path):
        """Sets the path for the input CSV file and updates the interface."""
        if getattr(self, 'file_path', None) != file_path:
            self.file_path = file_path
            file_name = Path(file_path).name
            self.file_label.setText(f"Selected: {file_name}")
            self.file_label.setToolTip(file_path)
            self._last_loaded_file_path = None  # reset 

            # Clear previous content 
            if hasattr(self.tab_widget_aqme, "df_mapped_smiles"):
                self.tab_widget_aqme.df_mapped_smiles = None

            self.load_csv_columns()

            # Update tabs with new information
            self.refresh_tabs()

            # Check for PDFs and images and unlock tabs, also check AQME workflow pattern detection
            self.check_aqme_workflow()
            self.check_for_pdfs()
            self.check_for_images()

            # informatio for "Results" tab and "Images" tab for update the display
            if hasattr(self, "images_tab") and hasattr(self.images_tab, "refresh_with_new_path"):
                self.images_tab.refresh_with_new_path(file_path)

            if hasattr(self, "results_tab") and hasattr(self.results_tab, "refresh_with_new_path"):
                self.results_tab.refresh_with_new_path(file_path)

    def set_csv_test_path(self, file_path):
        """Sets the path for the test CSV file and updates the label."""
        self.csv_test_path = file_path
        file_name = Path(file_path).name
        self.csv_test_label.setText(f"Selected: {file_name}")
        self.csv_test_label.setToolTip(file_path)
        self.clear_test_button.setVisible(True)

    def set_main_chemdraw_path(self, file_path):
        self.main_chemdraw_path = file_path
        file_name = Path(file_path).name
        self.main_chemdraw_label.setText(f"Selected: {file_name}")
        self.main_chemdraw_label.setToolTip(file_path)

    def load_csv_columns(self):
        """Loads column names from the selected CSV file and updates all selection fields."""
        if self.file_path:
            df = smart_read_csv(self.file_path)
            columns = list(df.columns)

            # --- Update Dropdowns ---
            self.y_dropdown.clear()
            self.y_dropdown.addItems(columns)
            self.names_dropdown.clear()
            self.names_dropdown.addItems(columns)

            # --- Update Dual-List Selection ---
            self.available_list.clear()  
            if self.ignore_list:  
                self.ignore_list.clear()
            self.available_list.addItems(columns)  

    def rename_existing_pdf(self, base_filename, directory):
        """Renames an existing PDF file in the given directory by adding an incremental number."""
        base_path = os.path.join(directory, base_filename)

        if not os.path.exists(base_path):
            return  # No existing file, so nothing to rename

        # Find the next available numbered filename
        index = 1
        while os.path.exists(os.path.join(directory, f"ROBERT_report_{index}.pdf")):
            index += 1

        new_path = os.path.join(directory, f"ROBERT_report_{index}.pdf")
        os.rename(base_path, new_path)

    def run_robert(self):
        """Runs the ROBERT workflow with the selected parameters."""
        if not self.file_path or not self.y_dropdown.currentText() or not self.names_dropdown.currentText():
            QMessageBox.warning(self, "WARNING!", "Please select a CSV file, a column for target value, and a name column.")
            return
        
        # Disable the Play button while the process is running
        self.run_button.setDisabled(True)
        self.stop_button.setDisabled(False)  # Enable Stop button

        # Clear previous content and reset with blank baseline
        self.console_output.clear()
        self.console_output.setHtml("<pre style='color:white; background-color:black; font-family:monospace;'></pre>")

        # Work directory
        run_dir = os.path.dirname(self.file_path)

        # Check and rename existing "ROBERT_report.pdf" files
        self.rename_existing_pdf("ROBERT_report.pdf", run_dir)

        # Check for leftover workflow folders
        folders_to_check = ["CURATE", "GENERATE", "PREDICT", "VERIFY", "AQME", "CSEARCH", "QDESCP"]
        existing_folders = [f for f in folders_to_check if os.path.exists(os.path.join(run_dir, f))]

        if existing_folders and self.workflow_selector.currentText() == "Full Workflow":
            message = (
                "ROBERT detected folders from a previous run.\n\n"
                "These folders may cause problems if the previous run was interrupted,\n"
                "or will be overwritten if the previous run completed successfully.\n\n"
                "Are you sure you want to continue and delete them?"
            )

            confirmation = QMessageBox.question(
                self,
                "WARNING!",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if confirmation == QMessageBox.No:
                self.run_button.setDisabled(False) # Re-enable Play button
                self.stop_button.setDisabled(True)  # Enable Stop button
                return
            
            # Try deleting the folders
            for folder in existing_folders:
                folder_path = os.path.join(run_dir, folder)
                try:
                    shutil.rmtree(folder_path)
                    self.console_output.append(f"[INFO] Deleted folder: {folder_path}")
                except Exception as e:
                    self.console_output.append(f"[ERROR] Could not delete folder '{folder_path}': {e}, try to delete it manually.")
                    self.run_button.setDisabled(False)
                    self.stop_button.setDisabled(True)
                    return
                
        # Save mapped CSV only if available from AQME workflow
        if hasattr(self.tab_widget_aqme, "df_mapped_smiles") and \
            self.tab_widget_aqme.df_mapped_smiles is not None and \
            len(self.tab_widget_aqme.selected_atoms) > 0:

            base, ext = os.path.splitext(self.tab_widget_aqme.file_path)
            mapped_csv_path = base + "_mapped.csv"

            # Check if the mapped CSV already exists and pop up a confirmation dialog if it does
            if os.path.exists(mapped_csv_path):
                confirmation = QMessageBox.question(
                    self,
                    "WARNING!",
                    f"The mapped file '{mapped_csv_path}' already exists.\n"
                    "Do you want to overwrite it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if confirmation == QMessageBox.No:
                    self.run_button.setDisabled(False) # Re-enable Play button
                    self.stop_button.setDisabled(True)  # Enable Stop button
                    return

            self.tab_widget_aqme.df_mapped_smiles.to_csv(mapped_csv_path, index=False)
            self.mapped_csv_path = mapped_csv_path  

        # Check if the mapped CSV exists and is valid for use it in AQME-ROBERT workflow
        if hasattr(self, "mapped_csv_path") and \
            os.path.isfile(self.mapped_csv_path) and \
            len(self.tab_widget_aqme.selected_atoms) > 0:

            # Use the mapped CSV path for the workflow
            selected_file_path = self.mapped_csv_path
        else:
            selected_file_path = self.file_path

        python_pointer = "python"

        if getattr(sys,"frozen", False):
            embeded_env = Path.cwd()/"_internal"/"robert_env"
            match sys.platform:
                case("win32"):
                    python_pointer = embeded_env/"python.exe"
                case("linux"): 
                    python_pointer = embeded_env/"bin"/"python"
                case("darwin"):
                    python_pointer = embeded_env/"bin"/"python3"
        
        # Build the base command.
        command = (
            f'"{python_pointer}" -u -m robert --csv_name "{os.path.basename(selected_file_path)}" '
            f'--y "{self.y_dropdown.currentText()}" '
            f'--names "{self.names_dropdown.currentText()}"'
        )
        
        if self.csv_test_path:
            command += f' --csv_test "{os.path.basename(self.csv_test_path)}"'
        
        if self.type_dropdown.currentText() == "Classification":
            command += ' --type "clas"'
        
        selected_columns = [
            self.ignore_list.item(i).text() 
            for i in range(self.ignore_list.count())  
        ]

        if selected_columns:
            formatted_columns = [f"'{col}'" for col in selected_columns]  # Wrap each item in single quotes
            command += f' --ignore "[{", ".join(formatted_columns)}]"'  # Wrap the entire list in double quotes

        if self.workflow_selector.currentText() == "CURATE":
            command += ' --curate'
        elif self.workflow_selector.currentText() == "GENERATE":
            command += ' --generate'
        elif self.workflow_selector.currentText() == "PREDICT":
            command += ' --predict'        
        elif self.workflow_selector.currentText() == "VERIFY":
            command += ' --verify'
        elif self.workflow_selector.currentText() == "REPORT":
            command += ' --report'

        # Add the additional parameters from the Advanced Options tab
        # GENERAL Section
        self.seed_value = self.options_tab.seed.text().strip()
        self.kfold_value = self.options_tab.kfold.text().strip()
        self.repeat_kfolds_value = self.options_tab.repeat_kfolds.text().strip()
        self.auto_type_value = self.options_tab.auto_type.isChecked()


        # AQME Section
        self.descriptor_level_selected = self.tab_widget_aqme.descriptor_level.currentText()
        self.atoms_selected = self.tab_widget_aqme.atoms.text().strip()

        # CURATE Section 
        self.categorical_value = self.options_tab.categoricalstr.currentText().strip()
        self.corr_filter_x_value = self.options_tab.corr_filter_xbool.isChecked()
        self.corr_filter_y_value = self.options_tab.corr_filter_ybool.isChecked()
        self.desc_thres_value = self.options_tab.desc_thresfloat.text().strip()
        self.thres_x_value = self.options_tab.thres_xfloat.text().strip()
        self.thres_y_value = self.options_tab.thres_yfloat.text().strip()

        # GENERATE Section
        # Detect selected type
        type_mode = self.type_dropdown.currentText()

        # Default values based on type
        default_models = {"RF", "GB", "NN", "MVL"} if type_mode == "Regression" else {"RF", "GB", "NN", "AdaB"}
        default_error_type = "rmse" if type_mode == "Regression" else "mcc"

        # Collect current values from GUI

        selected_models = {
            model for model, checkbox in self.options_tab.modellist.items()
            if checkbox.isChecked()
        }

        self.error_type_value = self.options_tab.error_type.currentText().strip()
        self.init_points_value = self.options_tab.init_points.text().strip()
        self.n_iter_value = self.options_tab.n_iter.text().strip()
        self.expect_improv_value = self.options_tab.expect_improv.text().strip()
        self.pfi_filter_value = self.options_tab.pfi_filter.isChecked()
        self.pfi_epochs_value = self.options_tab.pfi_epochs.text().strip()
        self.pfi_threshold_value = self.options_tab.pfi_threshold.text().strip()
        self.pfi_max_value = self.options_tab.pfi_max.text().strip()
        self.auto_test_value = self.options_tab.auto_test.isChecked()
        self.test_set_value = self.options_tab.test_set.text().strip()

        # PREDICT Section 

        self.t_value = self.options_tab.t_value.text().strip()
        self.shap_show = self.options_tab.shap_show.text().strip()
        self.pfi_show = self.options_tab.pfi_show.text().strip()

        # GENERAL Section command
        # --auto_type (default is True)
        if not self.auto_type_value:
            command += " --auto_type False"

        if self.seed_value:
            command += f' --seed {self.seed_value}'
        
        if self.kfold_value:
            command += f' --kfold {self.kfold_value}'
        
        if self.repeat_kfolds_value:
            command += f' --repeat_kfolds {self.repeat_kfolds_value}'

        # AQME Section command
        if self.aqme_workflow.isChecked():
            command += ' --aqme'
            command += f' --descp_lvl {self.tab_widget_aqme.descriptor_level.currentText()}'

            atoms_entries = []

            # Add 1-based indices from SMARTS selection
            selected_pattern_indices = self.tab_widget_aqme.selected_atoms
            if selected_pattern_indices:
                mapped_numbers = list(range(1, len(selected_pattern_indices) + 1))
                atoms_entries.extend(mapped_numbers)

            # Add manual SMARTS/text fragments
            atoms_text = self.tab_widget_aqme.atoms.text().strip()
            if atoms_text:
                fragments = [frag.strip() for frag in atoms_text.split(",") if frag.strip()]
                atoms_entries.extend(fragments)

            if atoms_entries:
                atoms_str = "[" + ",".join(str(e) for e in atoms_entries) + "]"
                command += f' --qdescp_keywords "--qdescp_atoms {atoms_str}"'

        # CURATE Section command
        if self.categorical_value != "onehot": # Default is "onehot"
            command += f' --categorical {self.categorical_value}'

        if not self.corr_filter_x_value:  # Default es True
            command += ' --corr_filter_x False'

        if self.corr_filter_y_value:  # Default es False
            command += ' --corr_filter_y True'

        if self.desc_thres_value:
            command += f' --desc_thres {self.desc_thres_value}'

        if self.thres_x_value:
            command += f' --thres_x {self.thres_x_value}'

        if self.thres_y_value:
            command += f' --thres_y {self.thres_y_value}'

        # GENERATE Section command 

        # --model (only if selection is different from default)
        if selected_models != default_models:
            model_list_str = "[" + ",".join(f"'{m}'" for m in sorted(selected_models)) + "]"
            command += f' --model "{model_list_str}"'

        # --error_type (only if different from default)
        if self.error_type_value != default_error_type:
            command += f' --error_type {self.error_type_value}'

        # --init_points (default: 10)
        if self.init_points_value:
            command += f' --init_points {self.init_points_value}'

        # --n_iter (default: 10)
        if self.n_iter_value:
            command += f' --n_iter {self.n_iter_value}'

        # --expect_improv (default: 0.05)
        if self.expect_improv_value:
            command += f' --expect_improv {self.expect_improv_value}'

        # --pfi_filter (default: True)
        if not self.pfi_filter_value:
            command += " --pfi_filter False"

        # --pfi_epochs (default: 5)
        if self.pfi_epochs_value:
            command += f' --pfi_epochs {self.pfi_epochs_value}'

        # --pfi_threshold (default: 0.2)
        if self.pfi_threshold_value:
            command += f' --pfi_threshold {self.pfi_threshold_value}'

        # --pfi_max (default: 0)
        if self.pfi_max_value:
            command += f' --pfi_max {self.pfi_max_value}'

        # --auto_test (default: True)
        if not self.auto_test_value:
            command += " --auto_test False"

        # --test_set (default: 0.1)
        if self.test_set_value:
            command += f' --test_set {self.test_set_value}'
        
        # PREDICT Section command

        # --t_value (default: 2)
        if self.t_value:
            command += f" --t_value {self.t_value}"

        # --shap_show (default: 10)
        if self.shap_show:
            command += f" --shap_show {self.shap_show}"

        # --pfi_show (default: 10)
        if self.pfi_show:
            command += f" --pfi_show {self.pfi_show}"

        def check_variables(self):
            """Validates the values extracted from the Advanced Options tab."""
            errors = []

            # Check that name column and target column are not the same
            if self.names_dropdown.currentText() == self.y_dropdown.currentText():
                QMessageBox.warning(
                    self,
                    "Invalid Selection",
                    "The name column and the target value column cannot be the same. Please select different columns."
                )
                return False

            # GENERAL
            if self.seed_value and not self.seed_value.isdigit():
                errors.append("Seed must be an integer.")

            if self.kfold_value and not self.kfold_value.isdigit():
                errors.append("kfold must be an integer.")

            if self.repeat_kfolds_value and not self.repeat_kfolds_value.isdigit():
                errors.append("repeat_kfolds must be an integer.")

            # AQME
            if self.aqme_workflow.isChecked():
                available_columns = [self.available_list.item(i).text() for i in range(self.available_list.count())]
                lowercase_columns = [col.lower() for col in available_columns]

                if "smiles" not in lowercase_columns:
                    errors.append("The column 'SMILES' must be present in the CSV file to use the AQME Workflow.")

            # CURATE
            if self.desc_thres_value:
                try:
                    float(self.desc_thres_value)
                except ValueError:
                    errors.append("desc_thres must be a number.")

            if self.thres_x_value:
                try:
                    float(self.thres_x_value)
                except ValueError:
                    errors.append("thres_x must be a number.")

            if self.thres_y_value:
                try:
                    float(self.thres_y_value)
                except ValueError:
                    errors.append("thres_y must be a number.")

            # GENERATE
            if self.init_points_value and not self.init_points_value.isdigit():
                errors.append("init_points must be an integer.")

            if self.n_iter_value and not self.n_iter_value.isdigit():
                errors.append("n_iter must be an integer.")

            if self.expect_improv_value:
                try:
                    float(self.expect_improv_value)
                except ValueError:
                    errors.append("expect_improv must be a number.")

            if self.pfi_epochs_value and not self.pfi_epochs_value.isdigit():
                errors.append("pfi_epochs must be an integer.")

            if self.pfi_threshold_value:
                try:
                    float(self.pfi_threshold_value)
                except ValueError:
                    errors.append("pfi_threshold must be a number.")

            if self.pfi_max_value and not self.pfi_max_value.isdigit():
                errors.append("pfi_max must be an integer.")

            if self.test_set_value:
                try:
                    value = float(self.test_set_value)
                    if not (0 <= value <= 1):
                        errors.append("test_set must be between 0 and 1.")
                except ValueError:
                    errors.append("test_set must be a number between 0 and 1.")

            # PREDICT
            if self.t_value and not self.t_value.isdigit():
                errors.append("t_value must be an integer.")

            if self.shap_show and not self.shap_show.isdigit():
                errors.append("shap_show must be an integer.")

            if self.pfi_show and not self.pfi_show.isdigit():
                errors.append("pfi_show must be an integer.")

            if errors:
                QMessageBox.warning(self, "Invalid Parameters", "\n".join(errors))
                return False

            return True

        if check_variables(self):  # Check if the parameters are valid
            self.console_output.append("<b><span style='color:orangered;'>Running ROBERT...</span></b><br>")
            self.progress.setRange(0, 0)  # Indeterminate progress
            self.worker = RobertWorker(command, os.path.dirname(selected_file_path))
            self.worker.output_received.connect(self.console_output.append)
            self.worker.error_received.connect(self.console_output.append)
            self.worker.process_finished.connect(self.on_process_finished)
            self.worker.start()
        else:
            # Reset the buttons and progress bar
            self.run_button.setDisabled(False)
            self.stop_button.setDisabled(True)
            self.progress.setRange(0, 100)
            self.console_output.append("WARNING! Invalid parameters. Please fix them before running.")

    def stop_robert(self):
        """Stops the ROBERT process safely after user confirmation, non-blocking."""

        confirmation = QMessageBox.question(
            self, 
            "WARNING!", 
            "Are you sure you want to stop ROBERT?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )

        if confirmation == QMessageBox.No:
            return  

        self.manual_stop = True

        if self.worker and self.worker.isRunning():
            self.console_output.append("<br><b><span style='color:orangered;'>Stopping ROBERT...</span></b>")
            self.progress.setRange(0, 100)
            self.stop_button.setDisabled(True)
            QTimer.singleShot(0, self.worker.stop) 
            
    @Slot(int)
    def on_process_finished(self, exit_code):
        """Handles the cleanup after the ROBERT process finishes."""

        # Reset buttons and progress bar
        self.run_button.setDisabled(False)
        self.stop_button.setDisabled(True)
        self.progress.setRange(0, 100)

        # Clean up the worker process
        if self.worker:
            if self.worker.process and self.worker.process.poll() is None:
                self.worker.stop()
            self.worker = None

        # Handle manual stop
        if exit_code == -1:
            self.console_output.clear()
            QMessageBox.information(self, "WARNING!", "ROBERT has been successfully stopped.")
            self.manual_stop = False
            return  # Exit early, don't evaluate further

        # Workflow complete messages
        output_text = self.console_output.toPlainText()
        workflow = self.workflow_selector.currentText()

        # Refresh tabs and check state BEFORE showing the message box
        self.check_for_pdfs()
        self.check_for_images()
        self.refresh_tabs()

        if hasattr(self, "images_tab") and hasattr(self.images_tab, "refresh_with_new_path"):
            self.images_tab.refresh_with_new_path(self.file_path)

        if hasattr(self, "results_tab") and hasattr(self.results_tab, "refresh_with_new_path"):
            self.results_tab.refresh_with_new_path(self.file_path)

        # Check for successful completion
        if not self.manual_stop and (workflow == "Full Workflow" or workflow == "REPORT"):
            if exit_code == 0 and "ROBERT_report.pdf was created successfully" in output_text:
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setWindowTitle("Success!")
                msg_box.setText("ROBERT has completed successfully.")

                view_report_button = QPushButton("View Report PDF")
                with AssetLibrary.Pdf_icon.get_path() as icon_path:
                    view_report_button.setIcon(QIcon(str(icon_path)))
                msg_box.addButton(view_report_button, QMessageBox.ActionRole)
                msg_box.addButton("OK", QMessageBox.AcceptRole)
                view_report_button.clicked.connect(lambda: self.tab_widget.setCurrentWidget(self.results_tab))
                msg_box.exec()
            else:
                QMessageBox.warning(self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs.")

        elif workflow == "CURATE":
            if exit_code == 0 and "Time CURATE: " in output_text:
                QMessageBox.information(self, "Success", "ROBERT has successfully completed the CURATE step.")
            else:
                QMessageBox.warning(self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs.")

        elif workflow == "GENERATE":
            if exit_code == 0 and "Time GENERATE: " in output_text:
                QMessageBox.information(self, "Success", "ROBERT has successfully completed the GENERATE step.")
            else:
                QMessageBox.warning(self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs.")

        elif workflow == "PREDICT":
            if exit_code == 0 and "Time PREDICT: " in output_text:
                QMessageBox.information(self, "Success", "ROBERT has successfully completed the PREDICT step.")
            else:
                QMessageBox.warning(self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs.")

        elif workflow == "VERIFY":
            if exit_code == 0 and "Time VERIFY: " in output_text:
                QMessageBox.information(self, "Success", "ROBERT has successfully completed the VERIFY step.")
            else:
                QMessageBox.warning(self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs.")

        # Final cleanup
        self.manual_stop = False

class DropLabel(QFrame):
    """A custom QLabel that accepts file drops and opens a file dialog."""
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
        self.label.setStyleSheet("""
            font-size: 11px; 
            font-style: italic; 
            color: gray; 
            font-weight: bold;
            border: 2px dashed gray; 
            padding: 5px;
            border-radius: 5px;
        """)
        self.layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.open_file_dialog)
        self.browse_button.setFixedSize(120, 30)
        self.browse_button.setStyleSheet(
            "padding: 6px 12px; font-size: 14px; border-radius: 5px; background-color: #555; color: white; border: 1px solid #777;"
        )
        self.layout.addWidget(self.browse_button, alignment=Qt.AlignCenter)

    def set_callback(self, callback):
        """Set the callback function to be called with the file path."""
        self.callback = callback

    def set_file_type(self, file_filter, extensions):
        """Set the file type for the file dialog."""
        self.file_filter = file_filter
        self.valid_extensions = extensions

    def open_file_dialog(self):
        """Open a file dialog to select a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self.file_filter)
        if file_path and self.callback:
            self.set_file_path(file_path)

    def dragEnterEvent(self, event):
        """Allow dropping files."""
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
        """Sets the file path and updates the label."""
        self.full_file_path = file_path
        file_name = Path(file_path).name
        self.label.setText(f"Selected: {file_name}")
        self.label.setToolTip(file_path) 
        if self.callback:
            self.callback(file_path)

    def setText(self, text):
        """Sets the label text."""
        self.label.setText(text)

class RobertWorker(QThread):
    """
    QThread that runs a subprocess asynchronously and streams real-time output.
    Supports safe termination of the entire process tree on all major OSes.
    """

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
        """
        Runs the subprocess and streams output in real time.
        Terminates cleanly if stop is requested.
        """
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
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
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
                    preexec_fn=os.setsid
                )

            def read_stdout():
                try:
                    for line in self.process.stdout:
                        if self._stop_requested:
                            break
                        formatted_line = self.ansi_converter.convert(line.strip(), full=False)
                        self.output_received.emit(formatted_line)
                except Exception as e:
                    self.error_received.emit(f"Error reading stdout: {e}")

            def read_stderr():
                try:
                    for line in self.process.stderr:
                        if self._stop_requested:
                            break
                        formatted_line = f'<span style="color:red;">{line.strip()}</span>'
                        self.error_received.emit(formatted_line)

                    # Reset line to avoid any leftover ANSI red text
                    reset_line = self.ansi_converter.convert("\033[0m", full=False)
                    self.output_received.emit(reset_line)

                except Exception as e:
                    self.error_received.emit(f"Error reading stderr: {e}")

            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)

            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to finish
            if self.process:
                exit_code = self.process.wait()
            else:
                exit_code = -1

            # Wait for both reader threads to finish
            stdout_thread.join()
            stderr_thread.join()

            self.process = None
            if self._stop_requested:
                self.process_finished.emit(-1)
            else:
                self.process_finished.emit(exit_code)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error_received.emit(f"Error in run(): {e}\n{tb}")

    def stop(self):
        """
        Request to stop the subprocess and its children.
        Call this from the UI thread.
        """
        self.request_stop.emit()

    def _handle_stop(self):
        """
        Terminates the subprocess and all its child processes (cross-platform).
        Runs in the worker thread.
        """
        self._stop_requested = True
        if not self.process:
            return
        try:
            parent = psutil.Process(self.process.pid)
            procs = parent.children(recursive=True)
            procs.append(parent)
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass
            gone, alive = psutil.wait_procs(procs, timeout=2)
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    pass
        except Exception as e:
            self.error_received.emit(f"Error stopping process: {e}")

class ChemDrawFileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select ChemDraw Files")
        self.setMinimumWidth(500)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        self.setSizeGripEnabled(True)

        self.main_chemdraw_path = None
        self.optional_chemdraw_path = None

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Please select your ChemDraw files")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Required file
        self.main_label = DropLabel(
            "Drag & Drop a main .sdf, .cdxml, .mol or .cdx file here",
            self,
            file_filter="ChemDraw Files (*.sdf *.cdxml *.mol *.cdx)",
            extensions=(".sdf", ".cdxml", ".mol", ".cdx")
        )
        self.main_label.set_callback(self.set_main_file)
        layout.addWidget(self.main_label)

        # Static warning message for user awareness
        self.warning_label = QLabel("⚠️ The file should only contain molecular structures without associated names.")
        self.warning_label.setStyleSheet("color: red; font-weight: bold;")
        self.warning_label.setWordWrap(True)
        self.warning_label.setAlignment(Qt.AlignCenter)  # Center the text
        layout.addWidget(self.warning_label)

        # Continue button
        self.continue_button = QPushButton("Continue")
        self.continue_button.setStyleSheet("padding: 8px; font-weight: bold;")
        self.continue_button.clicked.connect(self.continue_clicked)
        layout.addWidget(self.continue_button, alignment=Qt.AlignRight)

    def set_main_file(self, path):
        self.main_chemdraw_path = path
        self.main_label.setText(f"Selected: {os.path.basename(path)}")

    def set_optional_file(self, path):
        self.optional_chemdraw_path = path
        self.optional_label.setText(f"Selected: {os.path.basename(path)}")

    def continue_clicked(self):
        if not self.main_chemdraw_path:
            QMessageBox.warning(self, "Missing File", "Please select a main ChemDraw file.")
            return
        self.accept()  # close the dialog with success

def smart_read_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        delimiter = ';' if first_line.count(';') > first_line.count(',') else ','
    df = pd.read_csv(filepath, encoding='utf-8', delimiter=delimiter)
    return df

class NoScrollComboBox(QComboBox):
    """
    A custom QComboBox that prevents scrolling when the dropdown is closed.
    """     
    def wheelEvent(self, event: QWheelEvent):
        # Only allow scrolling when dropdown is open
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()

class AssetPath:
    """
    A class to manage and retrieve the file path of an asset.
    Attributes:
        _filename (str): The name of the file for which the path is managed.
    Methods:
        get_path():
            Retrieves the full path to the asset file. If the application is
            running in a frozen state (e.g., packaged with PyInstaller), it
            constructs the path relative to the current working directory.
            Otherwise, it retrieves the path using the importlib.resources API.
    """
    def __init__(self,filename):
        self._filename = filename
    
    def get_path(self):
        """
        Retrieves the file path to the icon file associated with the current instance.

        If the application is running in a frozen state (e.g., packaged with PyInstaller),
        the method constructs the path relative to the current working directory. Otherwise,
        it uses the `importlib.resources` module to locate the file within the package.

        Returns:
            Path: The resolved file path to the icon file.
        """
        if getattr(sys,"frozen",False):
            return Path.cwd()/"_internal"/"robert_env"/"Lib"/"site-packages"/"robert"/"icons"/ self._filename
        else:
            return as_file(files("robert") / "icons" / self._filename)


class AssetLibrary:
    """
    AssetLibrary is a class that provides a centralized collection of asset paths 
    used in the application. Each attribute represents a specific asset, such as 
    icons or logos, and is initialized using the AssetPath function.

    Attributes:
        Info_icon (AssetPath): Path to the "info_icon.png" asset.
        Robert_logo_transparent (AssetPath): Path to the "Robert_logo_transparent.png" asset.
        Robert_icon (AssetPath): Path to the "Robert_icon.png" asset.
        Play_icon (AssetPath): Path to the "play_icon.png" asset.
        Stop_icon (AssetPath): Path to the "stop_icon.png" asset.
        Pdf_icon (AssetPath): Path to the "pdf_icon.png" asset.
    """
    Info_icon = AssetPath("info_icon.png")
    Robert_logo_transparent = AssetPath("Robert_logo_transparent.png")
    Robert_icon = AssetPath("Robert_icon.png")
    Play_icon = AssetPath("play_icon.png")
    Stop_icon = AssetPath("stop_icon.png")
    Pdf_icon = AssetPath("pdf_icon.png")

def mcs_process(smiles_list, result_queue):
    """Find the Maximum Common Substructure (MCS) in a list of SMILES strings."""
    try:
        mol_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mol_with_Hs = Chem.AddHs(mol)
            if mol_with_Hs:
                mol_list.append(mol_with_Hs)
        if not mol_list:
            result_queue.put(('error', 'No valid molecules.'))
            return
        mcs_result = rdFMCS.FindMCS(mol_list)
        if mcs_result and mcs_result.smartsString:
            result_queue.put(('success', mcs_result.smartsString))
        else:
            result_queue.put(('error', '⚠️ No common SMARTS pattern found.'))
    except Exception as e:
        result_queue.put(('error', f'❌ MCS failed: {str(e)}'))

class MCSProcessWorker(QObject):
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
        self.process = Process(target=mcs_process, args=(self.smiles_list, self.queue))
        self.process.start()
        self.timer.start(self.timeout_ms)
        QTimer.singleShot(100, self.check_result) 

    def check_result(self):
        if not self.queue.empty():
            status, msg = self.queue.get()
            self.timer.stop()
            self.process.join()
            if status == 'success':
                self.finished.emit(msg)
            else:
                self.error.emit(msg)
        elif self.process.is_alive():
            QTimer.singleShot(100, self.check_result)
        else:
            self.timer.stop()
            self.process.join()

    def _on_timeout(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            self.timeout.emit()

class PageRenderThread(QThread):
    finished = Signal(int, float, QPixmap)  # page_num, zoom, pixmap

    def __init__(self, pdf_path, page_num, zoom):
        super().__init__()
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.zoom = zoom

    def run(self):
        doc = fitz.open(self.pdf_path)
        page = doc.load_page(self.page_num)
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=matrix)
        img = QPixmap()
        img.loadFromData(pix.tobytes("ppm"))
        doc.close()

        self.finished.emit(self.page_num, self.zoom, img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # --- Global Text Color Based on System Mode ---
    # Check the background color lightness to decide whether to use white or black text.
    palette = app.palette()
    bg_color = palette.color(QPalette.Window)
    if bg_color.lightness() < 128:  # Dark mode detected.
        text_color = "white"
    else:
        text_color = "black"
        
    # Apply a global stylesheet for QLabel elements.
    app.setStyleSheet(f"QLabel {{ color: {text_color}; }}")
    
    # --- Main Application Window ---
    window = EasyROB()
    window.show()
    sys.exit(app.exec())