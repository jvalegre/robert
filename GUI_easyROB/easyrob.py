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
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import csv
from rdkit.Chem import rdFMCS
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
import rdkit
import traceback
import shutil
import tempfile
import platform
import signal

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget,
    QComboBox, QListWidget, QProgressBar, QMessageBox, QHBoxLayout, QFrame, QTabWidget, 
    QLineEdit, QTextEdit, QSizePolicy, QFormLayout, QGridLayout, QGroupBox, QCheckBox, 
    QScrollArea, QFileDialog, QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,QInputDialog,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QPixmap, QPalette, QIcon, QImage, QMouseEvent
from PySide6.QtCore import (Qt, Slot, QThread, Signal, QTimer, QUrl)

os.environ["QT_QUICK_BACKEND"] = "software"

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

class AQMETab(QWidget):
    def __init__(self, tab_parent=None, main_window=None, help_tab=None, web_view=None):

        super().__init__(tab_parent)  # tab_parent = QTabWidget
        self.main_tab_widget = tab_parent # Reference to the main QTabWidget
        self.main_window = main_window  # Reference to the main window, accessible to csv_df, csv_path, etc... 
        self.help_tab = help_tab
        self.web_view = web_view
        self.selected_atoms = []
        self.check_multiple_matches = False
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
            self.csv_df = pd.read_csv(self.file_path) # Store the DataFrame for later use

            # === Auto SMARTS detection ===
            self.auto_pattern()

            # === Display first molecule ===
            self.display_molecule()

        except Exception as e:
            self.set_mol_viewer_message("❌ Failed to load or process the CSV.")
            self.mol_info_label.setText("🔬 Info here")

    def auto_pattern(self):
        """
        Auto-detect common SMARTS pattern in molecules from CSV 'SMILES' column.

        This function will use RDKit's MCS algorithm to find the Maximum Common Substructure
        (MCS) in the molecules read from the CSV file. If a pattern is found, it will be added
        to the 'smarts_targets' attribute.

        If the function fails to detect a pattern (e.g., due to invalid molecules), it will not
        raise an error, but will print an error message.

        """
        self.smarts_targets = []

        try:
            self.csv_df = pd.read_csv(self.file_path)
            if 'SMILES' not in self.csv_df.columns:
                raise ValueError("CSV must have a SMILES column")

            mol_list = []
            for smi in self.csv_df['SMILES'].dropna():

                mol = Chem.MolFromSmiles(smi)
                mol_list_with_Hs = Chem.AddHs(mol) 
                if mol_list_with_Hs:
                    mol_list.append(mol_list_with_Hs)

            if not mol_list:
                raise ValueError("No valid molecules.")

            # Get MCS, just one
            mcs_result = rdFMCS.FindMCS(mol_list)
            if mcs_result and mcs_result.smartsString:
                smarts = mcs_result.smartsString
                self.smarts_targets.append(smarts)
            else:
                # No common substructure found
                self.set_mol_viewer_message(
                "⚠️ No common SMARTS pattern was found among the molecules.",
                tooltip="No shared substructure could be detected with the current SMILES list."
            )
            self.mol_info_label.setText("🔬 Info here")

        except Exception as e:
            self.set_mol_viewer_message(
                "❌ Failed to detect SMARTS pattern. Check your CSV.",
                tooltip=f"auto_pattern error: {str(e)}"
            )
            self.mol_info_label.setText("🔬 Info here")

    def display_molecule(self):
        """Display a SMARTS molecule and highlight atoms based on user selection."""
        rdkit.rdBase.DisableLog('rdApp.*')
        rdDepictor.SetPreferCoordGen(True)

        try:
            if not self.smarts_targets:
                self.set_mol_viewer_message("⚠️ No SMARTS patterns available.")
                self.mol_info_label.setText("🔬 Info here")
                return

            # Convert SMARTS to RDKit Mol
            pattern_mol = Chem.MolFromSmarts(self.smarts_targets[0])
            if pattern_mol is None:
                self.set_mol_viewer_message("⚠️ Invalid SMARTS pattern.")
                self.mol_info_label.setText("🔬 Info here")
                return

            # Check for multiple matches in the dataset
            for smiles in self.csv_df["SMILES"]:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if mol is None:
                    continue
                matches = mol.GetSubstructMatches(pattern_mol)
                if len(matches) > 1:
                    self.check_multiple_matches = True
                    self.set_mol_viewer_message(
                    f"⚠️ Multiple matches detected: the common substructure appears more than once in the molecule '{smiles}'. "
                    "Atomic descriptor selection has been disabled to avoid ambiguity."
                    )   
                    self.mol_info_label.setText("🔬 Info here")
                    return

            self.check_multiple_matches = False  # All clear

            # Prepare for drawing the SMARTS pattern molecule
            self.mol = pattern_mol
            self.molecule_image_width = self.mol_viewer_container.width()
            self.molecule_image_height = self.mol_viewer_container.height()

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

            # Display image
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
                    if highlight_atoms:
                        self.mol_info_label.setText(f"🔬 {len(highlight_atoms)} atom(s) selected.")
                    else:
                        self.mol_info_label.setText(
                        '🧪 SMARTS pattern loaded. Click to select atoms.<br>'
                        '<span style="color:red;">⚠️ WARNING! No atoms selected. Atomic descriptors will not be generated.</span>'
                    )

        except Exception as e:
            self.set_mol_viewer_message("❌ Error displaying molecule.", tooltip=str(e))
            self.mol_info_label.setText("🔬 Info here")

    def handle_atom_selection(self, atom_idx):
        """Handle the selection of an atom in the pattern."""

        if not hasattr(self, 'selected_atoms'):
            self.selected_atoms = []

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
            self.csv_df['SMILES'].dropna()
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
        df = pd.read_csv(self.file_path)
        df_mapped = df.copy()
        df_mapped["SMILES"] = mapped_smiles
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

            if path.endswith('.cdxml'):
                try:
                    mols = MolsFromCDXMLFile(path, sanitize=True, removeHs=True)
                    return [mol for mol in mols if mol is not None]
                except Exception as e:
                    QMessageBox.critical(self, "CDXML Read Error", f"Failed to read {path}:\n{str(e)}")
                    return []
                
            elif path.endswith('.sdf'):
                return [mol for mol in Chem.SDMolSupplier(path) if mol is not None]
            
            elif path.endswith(".cdx"): 
                with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_sdf:
                    temp_sdf_path = temp_sdf.name

                try:
                    subprocess.run(["obabel", path, "-O", temp_sdf_path], check=True)
                except subprocess.CalledProcessError as e:
                    QMessageBox.critical(self, "CDX Conversion Error", f"Open Babel failed:\n{str(e)}")
                    return []

                mols = [mol for mol in Chem.SDMolSupplier(temp_sdf_path) if mol is not None]

                if os.path.exists(temp_sdf_path):
                    os.remove(temp_sdf_path)

                return mols
            else:
                mol = Chem.MolFromMolFile(path)
                return [mol] if mol else []

        mols_main = load_mols_from_path(main_path)

        if not mols_main:
            QMessageBox.warning(self, "Error", "No valid molecules found in the file.")
            return

        self.show_molecule_table_dialog(mols_main)

    def show_molecule_table_dialog(self, mols):
        """Create and display the molecule table dialog with save CSV functionality."""
        dialog = QDialog(self)
        dialog.setWindowTitle("ChemDraw Molecules")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint)
        dialog.setSizeGripEnabled(True)
        dialog.resize(800, 600)
        layout = QVBoxLayout(dialog)

        headers = ["Image", "SMILES", "code_name", "target"]
        table = QTableWidget(len(mols), len(headers))
        table.setHorizontalHeaderLabels(headers)

        # Make column headers editable on double-click
        def on_header_double_clicked(index):
            current_text = table.horizontalHeaderItem(index).text()
            new_text, ok = QInputDialog.getText(dialog, "Edit Column Name",
                                                f"Rename column '{current_text}':",
                                                text=current_text)
            if ok and new_text.strip():
                table.setHorizontalHeaderItem(index, QTableWidgetItem(new_text.strip()))

        table.horizontalHeader().sectionDoubleClicked.connect(on_header_double_clicked)

        for row, mol in enumerate(mols):
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

            smi = Chem.MolToSmiles(mol) if mol else ""
            table.setItem(row, 1, QTableWidgetItem(smi))
            table.setItem(row, 2, QTableWidgetItem(""))  # code_name
            table.setItem(row, 3, QTableWidgetItem(""))  # target

        layout.addWidget(table)

        # Save Button
        save_button = QPushButton("💾 Save as CSV")
        save_button.setStyleSheet("padding: 6px; font-weight: bold;")
        layout.addWidget(save_button, alignment=Qt.AlignmentFlag.AlignRight)

        def save_to_csv():
            """Save the table data to a CSV file."""
            # Validate that all code_name cells are filled
            for row in range(table.rowCount()):
                code_item = table.item(row, 2)
                target_item = table.item(row, 3)

                # Check if code_name and target are filled
                if (not code_item or not code_item.text().strip() or
                        not target_item or not target_item.text().strip()):
                        QMessageBox.warning(dialog, "WARNING!", "Please fill in all 'code_name' and 'target' fields before saving.")
                        return  # Cancel saving
                
            # Check for duplicate 'code_name' entries
            code_names = [table.item(row, 2).text().strip() for row in range(table.rowCount())]
            duplicates = [name for name in set(code_names) if code_names.count(name) > 1]

            if duplicates:
                QMessageBox.warning(
                    dialog,
                    "WARNING!",
                    f"The following 'code_name' values are duplicated:\n\n{', '.join(duplicates)}\n\nPlease make them unique before saving."
                )
                return  # Cancel saving
    
            # Ask user where to save
            path, _ = QFileDialog.getSaveFileName(dialog, "Save CSV", "", "CSV Files (*.csv)")
            if not path:
                return

            # Proceed to write CSV
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["SMILES", "code_name", "target"])

                for row in range(table.rowCount()):
                    smi = table.item(row, 1).text() if table.item(row, 1) else ""
                    code = table.item(row, 2).text() if table.item(row, 2) else ""
                    target = table.item(row, 3).text() if table.item(row, 3) else ""
                    writer.writerow([smi, code, target])

            # Automatically load CSV into the main GUI 
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
    """Tab for displaying results dynamically as PDFs are generated."""
    def __init__(self, main_tab_widget):
        super().__init__()
        
        self.main_tab_widget = main_tab_widget  # Reference to the main QTabWidget
        self.pdf_tabs = {}  # Store open PDF tabs

        # Timer to check for new PDFs every 2 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_for_pdfs)
        self.timer.start(2000)

        self.check_for_pdfs()  # Initial check

    def check_for_pdfs(self):
        """Checks for new PDFs and updates the UI dynamically."""
        pdf_files = sorted(glob.glob("ROBERT_report*.pdf"))  # Detect PDFs

        # Remove missing PDFs
        for pdf in list(self.pdf_tabs.keys()):
            if pdf not in pdf_files:
                index = self.main_tab_widget.indexOf(self.pdf_tabs[pdf])
                if index != -1:
                    self.main_tab_widget.removeTab(index)
                del self.pdf_tabs[pdf]

        # Add new PDFs as tabs
        for pdf in pdf_files:
            if pdf not in self.pdf_tabs:
                self.add_pdf_tab(pdf)

    def add_pdf_tab(self, pdf_path):
        """Creates a new tab displaying the PDF."""
        pdf_viewer = PDFViewer(pdf_path)
        index = self.main_tab_widget.addTab(pdf_viewer, pdf_path.split("/")[-1])
        self.pdf_tabs[pdf_path] = pdf_viewer
        self.main_tab_widget.setCurrentIndex(index)

class PDFViewer(QWidget):
    """Widget to display a PDF inside a scrollable area."""

    def __init__(self, pdf_path):
        super().__init__()

        layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # Container widget for PDF pages
        self.container = QWidget()
        self.scroll_area.setWidget(self.container)
        self.vbox = QVBoxLayout(self.container)

        self.load_pdf(pdf_path)

    def load_pdf(self, pdf_path):
        """Loads and renders the PDF pages and centers them."""
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            zoom = 1.2  # size of the PDF pages
            pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(zoom, zoom))

            img = QPixmap()
            img.loadFromData(pix.tobytes("ppm"))

            # Create a container with a horizontal layout to center the QLabel
            page_container = QWidget()
            hbox = QHBoxLayout(page_container)
            hbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

            label = QLabel()
            label.setPixmap(img)
            hbox.addWidget(label)

            self.vbox.addWidget(page_container)

        doc.close()

class ImagesTab(QWidget):
    """Images tab for displaying images from multiple folders as results of Robert workflow."""
    
    def __init__(self, main_tab_widget, image_folders):
        super().__init__()

        self.main_tab_widget = main_tab_widget  # Reference to the main QTabWidget
        self.image_folders = image_folders  # List of folders to monitor
        self.folder_widgets = {}  # Dictionary to store folder tabs

        # Create the QTabWidget for sub-tabs inside "Images"
        self.folder_tabs = QTabWidget()
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.folder_tabs)
        self.setLayout(self.layout)

        # Timer to check for new images every 2 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_for_images)
        self.timer.start(2000)

        self.check_for_images()  # Initial check

    def check_for_images(self):
        """Dynamically updates sub-tabs based on image folders with custom names and order."""

        # Define custom names and order for folders
        folder_names = {
            "CURATE": "CURATE",
            "GENERATE/Raw_data": "GENERATE",
            "PREDICT": "PREDICT",
            "VERIFY": "VERIFY",
        }

        # Define the exact order of the tabs
        folder_order = [ "CURATE", "GENERATE/Raw_data","PREDICT", "VERIFY",]

        # Loop through folders in the defined order
        for folder in folder_order:
            if not os.path.exists(folder):
                continue  # Skip if the folder doesn't exist

            image_files = sorted(glob.glob(os.path.join(folder, "*.[pjg][np][g]")))  # Detect images

            # If the folder does not have a tab, create one
            if folder not in self.folder_widgets:
                folder_widget = QWidget()
                folder_layout = QVBoxLayout(folder_widget)
                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)

                # Grid layout for images
                image_grid = QGridLayout()
                container = QWidget()
                container.setLayout(image_grid)
                scroll_area.setWidget(container)
                folder_layout.addWidget(scroll_area)

                folder_widget.setLayout(folder_layout)

                # Get the custom name for the tab
                tab_name = folder_names.get(folder, os.path.basename(folder))  # Default to folder name if not found

                # Add tab with custom name
                self.folder_tabs.addTab(folder_widget, tab_name)

                # Store reference for later updates
                self.folder_widgets[folder] = image_grid

            # Retrieve the image grid for the corresponding folder tab
            image_grid = self.folder_widgets[folder]

            # Remove previous images before updating
            while image_grid.count():
                item = image_grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            # Add new images in the grid layout
            row, col = 0, 0
            max_columns = 3  # Number of images per row
            for img_path in image_files:
                image_label = ImageLabel(img_path, size=300)  # Larger image size
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
        """Intercept window close to clean up running threads."""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Exit Confirmation",
                "ROBERT is still running. Do you want to stop the process and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait()
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
        self.y_dropdown = QComboBox()
        main_layout.addWidget(self.y_dropdown)
        self.y_dropdown.setStyleSheet(box_features)
        
        # --- Select prediction type ---
        self.type_label = QLabel("Prediction Type")
        self.type_label.setStyleSheet(f"font-weight: bold; font-size: {font_size};")
        main_layout.addWidget(self.type_label)
        self.type_dropdown = QComboBox()
        self.type_dropdown.addItems(["Regression", "Classification"])
        main_layout.addWidget(self.type_dropdown)
        self.type_dropdown.setStyleSheet(box_features)
        
        # --- Select column for --names ---
        self.names_label = QLabel("Select name column")
        self.names_label.setStyleSheet(f"font-weight: bold; font-size: {font_size};")
        main_layout.addWidget(self.names_label)
        self.names_dropdown = QComboBox()
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
        main_layout.addWidget(self.aqme_workflow)
        main_layout.addSpacing(10)  

        # Workflow selection dropdown
        self.workflow_selector = QComboBox()
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

        # # Add button layout to the main layout
        button_container = QHBoxLayout()
        button_container.addWidget(self.run_button)
        button_container.addWidget(self.stop_button)
        main_layout.addLayout(button_container)

        # --- Console Output Setup ---
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("background-color: black; color: white; padding: 5px; font-family: monospace;")
        self.console_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.console_output.setFixedHeight(275)

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
        self.tab_widget_results = QTabWidget()
        self.results_tab = ResultsTab(self.tab_widget_results)

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
        self.images_tab = ImagesTab(self.tab_widget, self.image_folders)

        # ===============================
        # Add Tabs to Tab Widget (Display order)
        # ===============================

        self.tab_widget.addTab(self.tab_widget_aqme, "AQME")
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_widget_aqme), False)

        self.tab_widget.addTab(self.options_tab, "Advanced Options")
        self.tab_widget.addTab(self.help_tab, "Help")
        self.tab_widget.addTab(self.tab_widget_results, "Results")
        self.tab_widget.addTab(self.images_tab, "Images")

        # =========================================
        # Create Independent Timers for Each Check
        # =========================================

        # PDF Checking Timer
        self.timer_pdfs = QTimer(self)
        self.timer_pdfs.timeout.connect(self.check_for_pdfs)
        self.timer_pdfs.start(2000)

        # Image Checking Timer
        self.timer_images = QTimer(self)
        self.timer_images.timeout.connect(self.check_for_images)
        self.timer_images.start(2000)

        # AQME Checkbox Monitoring Timer
        self.timer_aqme = QTimer(self)
        self.timer_aqme.timeout.connect(self.check_aqme_workflow)
        self.timer_aqme.start(1000)

        # Run initial checks
        self.check_for_pdfs()
        self.check_for_images()

    def check_for_images(self):
        """Enable or disable the 'Images' tab based on folder existence."""
        has_folders = any(os.path.exists(folder) for folder in self.image_folders)

        # Get the index of the "Images" tab
        tab_index = self.tab_widget.indexOf(self.images_tab)

        # Enable or disable the tab based on folder presence
        if tab_index != -1:
            self.tab_widget.setTabEnabled(tab_index, has_folders)

    def check_for_pdfs(self):
        """Enable or disable the 'Results' tab based on PDF presence."""
        has_pdfs = bool(glob.glob("ROBERT_report*.pdf"))
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_widget_results), has_pdfs)

    def check_aqme_workflow(self):
        """Enable or disable the AQME tab based on the checkbox state (polled every 2 seconds)."""
        is_checked = self.aqme_workflow.isChecked()
        tab_index = self.tab_widget.indexOf(self.tab_widget_aqme)

        if tab_index != -1:
            tab_enabled = self.tab_widget.isTabEnabled(tab_index)

            if is_checked and not tab_enabled:
                self.tab_widget.setTabEnabled(tab_index, True)
                self.tab_widget.setCurrentWidget(self.tab_widget_aqme)
                QMessageBox.information(self, "AQME Tab Enabled", "AQME tab unlocked to specify AQME parameters.")

            # Reset selected atoms if the file path has changed and display the new pattern if there is any
            if (
                is_checked and tab_enabled and
                hasattr(self, 'file_path') and self.file_path and
                self.file_path != getattr(self, '_last_loaded_file_path', None)
            ):
                self.tab_widget_aqme.selected_atoms = []
                self.tab_widget_aqme.file_path = self.file_path
                self.tab_widget_aqme.detect_patterns_and_display()
                self._last_loaded_file_path = self.file_path

            elif not is_checked and tab_enabled:
                self.tab_widget.setTabEnabled(tab_index, False)

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
        """Sets the path for the input CSV file and updates the label."""
        self.file_path = file_path  
        file_name = Path(file_path).name
        self.file_label.setText(f"Selected: {file_name}")
        self.file_label.setToolTip(file_path)  
        self.load_csv_columns()

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
            df = pd.read_csv(self.file_path, encoding='utf-8')
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

    def rename_existing_pdf(self, base_filename):
        """Renames an existing PDF file by adding an incremental number."""
        if not os.path.exists(base_filename):
            return  # No existing file, so nothing to rename

        # Find the next available numbered filename
        index = 1
        while os.path.exists(f"ROBERT_report_{index}.pdf"):
            index += 1
        
        new_filename = f"ROBERT_report_{index}.pdf"
        os.rename(base_filename, new_filename)

    def run_robert(self):
        """Runs the ROBERT workflow with the selected parameters."""
        if not self.file_path or not self.y_dropdown.currentText() or not self.names_dropdown.currentText():
            QMessageBox.warning(self, "WARNING!", "Please select a CSV file, a column for target value, and a name column.")
            return
        
        # Check and rename existing "ROBERT_report.pdf" files
        self.rename_existing_pdf("ROBERT_report.pdf")

        # Disable the Play button while the process is running
        self.run_button.setDisabled(True)
        self.stop_button.setDisabled(False)  # Enable Stop button
        
        # Check for leftover workflow folders
        folders_to_check = ["CURATE", "GENERATE", "PREDICT", "VERIFY", "AQME", "CSEARCH", "QDESCP"]
        existing_folders = [f for f in folders_to_check if os.path.exists(f)]

        if existing_folders:
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
                try:
                    shutil.rmtree(folder)
                    self.console_output.append(f"[INFO] Deleted folder: {folder}")
                except Exception as e:
                    self.console_output.append(f"[ERROR] Could not delete folder '{folder}': {e}, try to delete it manually.")
                    self.run_button.setDisabled(False) # Re-enable Play button
                    self.stop_button.setDisabled(True)  # Enable Stop button
                    return  # Prevent running ROBERT if cleanup fails
                
        # Save mapped CSV only if available from AQME workflow
        if hasattr(self.tab_widget_aqme, "df_mapped_smiles"):
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
        if hasattr(self, "mapped_csv_path") and os.path.isfile(self.mapped_csv_path):
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
                if "SMILES" not in available_columns:
                    errors.append("The column 'SMILES' must be present in CSV file to use AQME Workflow.")

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
            self.console_output.clear()
            self.progress.setRange(0, 0)  # Indeterminate progress
            self.worker = RobertWorker(command, os.getcwd())
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
        """Stops the ROBERT process safely after user confirmation."""
        
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
            self.worker.stop()  
            self.worker.wait()
            self.worker = None
            self.on_process_finished(-1)

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
                view_report_button.clicked.connect(lambda: self.tab_widget.setCurrentWidget(self.tab_widget_results))
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
    """A QThread that runs a subprocess asynchronously and streams real-time output."""

    output_received = Signal(str)
    error_received = Signal(str)
    process_finished = Signal(int)

    def __init__(self, command, working_dir=None):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.process = None
        self.ansi_converter = Ansi2HTMLConverter(dark_bg=True)  # Ensures dark mode support
        self.is_windows = platform.system() == "Windows"


    def run(self):
        """Runs the subprocess and streams output line by line in real-time with ANSI support."""
        if self.is_windows:
            self.process = subprocess.Popen(
                shlex.split(self.command),
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
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

        # Read output in real-time and convert ANSI to HTML for read correctly console
        with self.process.stdout:
            for line in iter(self.process.stdout.readline, ''):
                formatted_line = self.ansi_converter.convert(line.strip(), full=False)
                self.output_received.emit(formatted_line)

        with self.process.stderr:
            for line in iter(self.process.stderr.readline, ''):
                formatted_line = f'<span style="color:red;">{line.strip()}</span>'
                self.error_received.emit(formatted_line)  # Display errors in red

        self.process.wait()
        self.process_finished.emit(self.process.returncode)

    def stop(self):
        """Stops the subprocess and its children."""
        if not self.process:
            return

        try:
            if self.is_windows:
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
                if self.process.poll() is None:
                    self.process.kill()
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except Exception as e:
            print(f"Error stopping process: {e}")
        finally:
            self.process = None

#TODO por
def robert_target(queue_out, queue_err, sys_args):
    """Target function for the ROBERT process in a separate process."""
    class StreamToQueue:
        def __init__(self, queue):
            self.queue = queue
        def write(self, msg):
            if msg.strip():
                self.queue.put(msg.strip())
        def flush(self): pass

    sys.stdout = StreamToQueue(queue_out)
    sys.stderr = StreamToQueue(queue_err)

    try:
        main("exe", sys_args)
        queue_out.put("__FINISHED_OK__")
    except Exception:
        queue_err.put(traceback.format_exc())
        queue_out.put("__FINISHED_ERROR__")

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