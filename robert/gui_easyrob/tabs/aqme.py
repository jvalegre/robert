"""
AQME integration tab for easyROB.

This module implements the AQME advanced options part of the GUI, handling
molecular preprocessing, SMARTS pattern detection, and ChemDraw integration.

Responsibilities:
- Load and validate molecular datasets (SMILES, CDXML, SDF)
- Detect common substructures (MCS/SMARTS)
- Enable interactive atom selection for descriptor generation
- Generate mapped SMILES and curated CSV files
- Provide ChemDraw-to-CSV conversion tools

Import strategy:
- Supports dual execution modes:
  1. Local (portable execution)
  2. Installed package (environment / entry point)

- Uses try/except fallback to resolve imports accordingly.

Notes:
- Combines UI logic with domain-specific chemistry operations (RDKit).
- Acts as a bridge between user input and AQME descriptor workflows.

"""

# ------------------------------------------------------------
# Import resolution (local vs installed package)
# ------------------------------------------------------------
try:
    from utils.utils_gui import (
        AssetLibrary,
        BytesIO,
        Chem,
        Draw,
        QCheckBox,
        QComboBox,
        QDesktopServices,
        QDialog,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QIcon,
        QImage,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMessageBox,
        QMouseEvent,
        QPixmap,
        QPushButton,
        QSizePolicy,
        QTableWidget,
        QTableWidgetItem,
        QUrl,
        QVBoxLayout,
        QWidget,
        Qt,
        smart_read_csv,
        GetMolFrags,
        MolsFromCDXMLFile,
        rdDepictor,
        rdMolDraw2D,
        rdkit,
    )

    from utils.aqme_utils import ChemDrawFileDialog, MCSProcessWorker

except ImportError as e:
    from robert.gui_easyrob.utils.utils_gui import (
        AssetLibrary,
        BytesIO,
        Chem,
        Draw,
        QCheckBox,
        QComboBox,
        QDesktopServices,
        QDialog,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QIcon,
        QImage,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMessageBox,
        QMouseEvent,
        QPixmap,
        QPushButton,
        QSizePolicy,
        QTableWidget,
        QTableWidgetItem,
        QUrl,
        QVBoxLayout,
        QWidget,
        Qt,
        smart_read_csv,
        GetMolFrags,
        MolsFromCDXMLFile,
        rdDepictor,
        rdMolDraw2D,
        rdkit,
    )

    from robert.gui_easyrob.utils.aqme_utils import ChemDrawFileDialog, MCSProcessWorker

# ---- Standard library (keep explicit) ----
import os
import csv
from functools import partial

class AQMETab(QWidget):
    """Tab responsible for AQME-oriented chemistry preparation workflows."""
    def __init__(self, tab_parent=None, main_window=None):

        super().__init__(tab_parent)  # tab_parent = QTabWidget
        self.main_tab_widget = tab_parent # Reference to the main QTabWidget
        self.main_window = main_window  # Reference to the main window, accessible to csv_df, csv_path, etc... 
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
        self.mol_viewer_container.setFixedSize(400, 400)
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
        self.mol_viewer.setFixedSize(400, 400)

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
        aqme_box.setMaximumHeight(200)  
        aqme_box.setStyleSheet(self.box_features)
        aqme_layout = QFormLayout()

        self.atoms = QLineEdit(placeholderText="e.g., Au or C=O")
        self.descriptor_level = QComboBox()
        self.descriptor_level.addItems(["interpret", "denovo", "full"])
        self.solvent = QComboBox()
        self.solvent.addItems([
            "None",
            # "Acetone",
            # "Acetonitrile",
            # "Aniline",
            # "Benzaldehyde",
            # "Benzene",
            # "CH2Cl2",
            # "CHCl3",
            # "CS2",
            # "Dioxane",
            # "DMF",
            # "DMSO",
            # "Ether",
            # "Ethylacetate",
            # "Furane",
            # "Hexadecane",
            # "Hexane",
            # "Methanol",
            # "Nitromethane",
            # "Octanol",
            # "Octanol (wet)",
            # "Phenol",
            # "Toluene",
            # "THF",
            # "Water"
        ])

        aqme_layout.addRow(QLabel("QDESCP Atoms:"), self.atoms)
        aqme_layout.addRow(QLabel("Descriptor Level:"), self.descriptor_level)
        aqme_layout.addRow(QLabel("Solvent:"), self.solvent)

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
        """Open the corresponding documentation section in the browser."""

        base_url = "https://robert.readthedocs.io/en/latest/Technical/defaults.html"

        if anchor:
            full_url = f"{base_url}#{anchor.lower()}"
        else:
            full_url = base_url

        QDesktopServices.openUrl(QUrl(full_url))

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

    def build_unified_smiles_context(self, train_csv_path, test_csv_path=None):
        """
        Returns a list of SMILES for train/test CSVs (if test exists). This is used ONLY for chemical validation.
        (FMCS, ambiguity checks, metal detection).
        """
        train_df = smart_read_csv(train_csv_path)
        smiles_col = next(
            (c for c in train_df.columns if c.lower() == "smiles"),
            None
        )
        if smiles_col is None:
            raise ValueError("TRAIN CSV has no SMILES column")

        unified_smiles = (
            train_df[smiles_col]
            .dropna()
            .astype(str)
            .tolist()
        )

        if test_csv_path:
            test_df = smart_read_csv(test_csv_path)
            test_smiles_col = next(
                (c for c in test_df.columns if c.lower() == "smiles"),
                None
            )
            if test_smiles_col is None:
                raise ValueError("TEST CSV has no SMILES column")

            unified_smiles.extend(
                test_df[test_smiles_col]
                .dropna()
                .astype(str)
                .tolist()
            )

        return unified_smiles
    
    def generate_mapped_csv_from_smiles(
        self,
        csv_path,
        smarts,
        selected_atoms,
        suffix="_mapped"
    ):
        """Generate a new CSV file with mapped SMILES based on the provided SMARTS pattern"""

        df = smart_read_csv(csv_path)
        smiles_col = next(
            (c for c in df.columns if c.lower() == "smiles"),
            None
        )

        if smiles_col is None:
            raise ValueError("CSV has no SMILES column")

        pattern_mol = Chem.MolFromSmarts(smarts)
        mapped_smiles = []

        for smiles in df[smiles_col]:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None:
                mapped_smiles.append(None)
                continue

            matches = mol.GetSubstructMatches(pattern_mol)
            if len(matches) != 1:
                mapped_smiles.append(None)
                continue

            match = matches[0]
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

            for i, idx in enumerate(selected_atoms):
                mol.GetAtomWithIdx(match[idx]).SetAtomMapNum(i + 1)

            mapped_smiles.append(Chem.MolToSmiles(mol))

        df_out = df.copy()
        df_out[smiles_col] = mapped_smiles

        base, _ = os.path.splitext(csv_path)
        out_csv = f"{base}{suffix}.csv"
        df_out.to_csv(out_csv, index=False)

        return out_csv

    def auto_pattern(self):
        """
        Auto-detect common SMARTS pattern.
        - TRAIN only → exploratory mode
        - TRAIN + TEST → mapping mode (contract must hold for all)
        """

        self.mol_info_label.setText("🔬 Info here")
        self.smarts_targets = []

        if self.smiles_column is None:
            return

        # -------------------------------
        # Decide FMCS context (ROBUST)
        # -------------------------------
        unified_smiles = getattr(self, "unified_smiles", None)

        if unified_smiles:
            # TRAIN + TEST → strong contract
            smiles_list = unified_smiles
        else:
            # TRAIN only → exploratory
            smiles_list = (
                self.csv_df[self.smiles_column]
                .dropna()
                .astype(str)
                .tolist()
            )

        if not smiles_list:
            self.set_mol_viewer_message(
                "⚠️ No molecules available for SMARTS detection."
            )
            return

        # -------------------------------
        # Launch MCS worker
        # -------------------------------
        self.mcs_worker = MCSProcessWorker(
            smiles_list,
            timeout_ms=60000
        )

        self.mcs_worker.finished.connect(self._on_mcs_success)
        self.mcs_worker.error.connect(self._on_mcs_error)
        self.mcs_worker.timeout.connect(self._on_mcs_timeout)

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

            unified_smiles = getattr(self, "unified_smiles", None)
            if not unified_smiles:
                self.set_mol_viewer_message(
                    "⚠️ No molecules available for pattern matching."
                )
                self.mol_info_label.setText("🔬 Info here")
                return

            for smiles in unified_smiles:
                metal_found_in_this_mol = False

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
                            f"⚠️ <b>Multiple matches detected<b>: the common substructure "
                            f"'{self.smarts_targets[0]}' appears more than once in the molecule "
                            f"'{smiles}'. Atomic descriptor selection has been disabled to avoid ambiguity."
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

            highlight_colors = (
                {idx: (0.698, 0.4, 1.0) for idx in highlight_atoms}
                if highlight_atoms else {}
            )

            drawer = rdMolDraw2D.MolDraw2DCairo(
                self.molecule_image_width,
                self.molecule_image_height
            )
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
            self.atom_coords = [
                drawer.GetDrawCoords(i)
                for i in range(self.mol.GetNumAtoms())
            ]

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
                            '<span style="color:red;">⚠️ No atoms selected. '
                            'Descriptors will only be generated for the detected metal.</span>'
                        )
                    else:
                        if highlight_atoms:
                            self.mol_info_label.setText(
                                f"🔬 {len(highlight_atoms)} atom(s) selected."
                            )
                        else:
                            self.mol_info_label.setText(
                                '🧪 <b>SMARTS pattern loaded. Click to select atoms.</b><br>'
                                '<span style="color:red;">⚠️ WARNING! No atoms selected. '
                                'Atomic descriptors will not be generated.</span>'
                            )

        except Exception as e:
            self.set_mol_viewer_message(
                "❌ Error displaying molecule.",
                tooltip=str(e)
            )
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
        # Pre-dialog notice about file quality and format
        QMessageBox.information(
            self,
            "Before Selecting Your File",
            (
                "<b>Before continuing:</b><br><br>"
                "Please ensure your ChemDraw file is saved in <b>CDXML</b> format. Verify that the molecular structures are valid and free from editing errors such as:<br><br>"
                "• Red highlights in ChemDraw (invalid valences or atoms)<br>"
                "• Incorrect or broken bonds<br>"
                "• Unconnected fragments or misdrawn connections<br><br>"
                "<i>When everything looks correct, click OK to select your file.</i>"
            )
        )

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
                QMessageBox.warning(
                    self,
                    " Unsupported File Format",
                    (
                        "<b>CDX format is not supported! No issue — it only requires exporting to CDXML format.</b><br><br>"
                        "<b>How to convert your file:</b><br>"
                        "1. Open your CDX file in ChemDraw.<br>"
                        "2. Go to <b>File → Save As</b> or <b>Export</b>.<br>"
                        "3. Choose <b>CDXML (*.cdxml)</b> as the format.<br><br>"
                        "<b>If this method does not work, you can manually convert individual molecules:</b><br>"
                        "1. Open the CDX file in ChemDraw.<br>"
                        "2. Select a molecule.<br>"
                        "3. Press <b>Ctrl+C</b> (or <b>Cmd+C</b> on Mac).<br>"
                        "4. Paste it into a new ChemDraw document.<br>"
                        "5. Save it as <b>CDXML</b>.<br><br>"
                        "This ensures proper structure recognition and full compatibility with easyROB."
                    )
                )
                return None

            else:
                mol = Chem.MolFromMolFile(path)
                return [mol] if mol else []

        mols_main = load_mols_from_path(main_path)

        # If the function returned None, it means we already handled a special case (like .cdx)
        if mols_main is None:
            return
        
        # If the function returned an empty list, it means there were no valid molecules
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