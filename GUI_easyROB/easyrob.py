import sys
import os
import pandas as pd
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget,
    QComboBox, QListWidget, QProgressBar, QMessageBox, QHBoxLayout, QFrame, QTabWidget, 
    QLineEdit, QTextEdit, QSizePolicy, QFormLayout, QGridLayout, QGroupBox, QCheckBox, 
    QScrollArea, QFileDialog, QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QPixmap, QPalette, QIcon, QImage
from PySide6.QtCore import (Qt, QProcess, Slot, QThread, Signal, QTimer, QUrl)
import subprocess
import shlex
import glob
import fitz  
from ansi2html import Ansi2HTMLConverter
from rdkit.Chem.rdmolfiles import MolsFromCDXMLFile


from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import csv


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
        aqme_box = self.create_aqme_section()
        curate_box = self.create_curate_section()
        generate_box = self.create_generate_section()
        predict_box = self.create_predict_section()

        # Apply Background Colors
        general_box.setStyleSheet("QGroupBox { background-color: #696969;}")  
        aqme_box.setStyleSheet("QGroupBox { background-color: #696969;}")  
        curate_box.setStyleSheet("QGroupBox { background-color: #696969;}")  
        generate_box.setStyleSheet("QGroupBox { background-color: #696969;}")  
        predict_box.setStyleSheet("QGroupBox { background-color: #696969;}")  

        # GENERAL / AQME (Top Row)
        grid_layout.addWidget(general_box, 0, 0, 1, 1)  # General left
        grid_layout.addWidget(aqme_box, 0, 1, 1, 1)  # AQME right

        # CURATE / GENERATE (Middle Row)
        grid_layout.addWidget(curate_box, 1, 0, 1, 1)  # Curate left
        grid_layout.addWidget(generate_box, 1, 1, 1, 1)  # Generate right

        # PREDICT (Bottom, Full Width)
        grid_layout.addWidget(predict_box, 2, 0, 1, 2)  # Predict spans both columns

        # Add the grid layout to the main layout
        main_layout.addLayout(grid_layout)
        self.setLayout(main_layout)

    def go_to_help_section(self, anchor):
        """Navigates to the help section in the web view."""
        base_url = "https://robert.readthedocs.io/en/latest/Technical/defaults.html"
        full_url = f"{base_url}#{anchor}"
        self.tab_widget.setCurrentWidget(self.help_tab)
        self.web_view.setUrl(QUrl(full_url))

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
        help_button = QPushButton("Help GENERAL parameters")
        help_button.setIcon(QIcon("icons/info_icon.png"))  # Use your actual icon path
        help_button.setCursor(Qt.PointingHandCursor)
        help_button.setStyleSheet("padding: 4px; font-weight: bold;")
        help_button.clicked.connect(lambda: self.go_to_help_section("general"))

        # Add with right alignment
        layout.addRow("", help_button)  # Adds it as a new row with empty label
        layout.setAlignment(help_button, Qt.AlignRight)

        box.setLayout(layout)
        return box

    def create_aqme_section(self):
        """Creates the AQME section with a box and input fields."""
        
        # Create section container
        box = QGroupBox("AQME")
        box.setStyleSheet(self.box_features)
        layout = QFormLayout()

        # Create input fields and store widget references
        self.atoms = QLineEdit(placeholderText="e.g., Au or C=O")
        self.descriptor_level = QComboBox()
        self.descriptor_level.addItems(["interpret", "denovo", "full"])

        # Add fields to the layout
        layout.addRow(QLabel("QDESCP Atoms:"), self.atoms)
        layout.addRow(QLabel("Descriptor Level:"), self.descriptor_level)

        # --- Help button at the bottom ---
        help_button = QPushButton("Help AQME parameters")
        help_button.setIcon(QIcon("icons/info_icon.png"))  # Use your actual icon path
        help_button.setCursor(Qt.PointingHandCursor)
        help_button.setStyleSheet("padding: 4px; font-weight: bold;")
        help_button.clicked.connect(lambda: self.go_to_help_section("aqme"))

        # Add with right alignment
        layout.addRow("", help_button)  # Adds it as a new row with empty label
        layout.setAlignment(help_button, Qt.AlignRight)

        # Set layout
        box.setLayout(layout)

        # Return just the box
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
        help_button = QPushButton("Help CURATE parameters")
        help_button.setIcon(QIcon("icons/info_icon.png"))  # Use your actual icon path
        help_button.setCursor(Qt.PointingHandCursor)
        help_button.setStyleSheet("padding: 4px; font-weight: bold;")
        help_button.clicked.connect(lambda: self.go_to_help_section("curate"))

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
        help_button = QPushButton("Help GENERATE parameters")
        help_button.setIcon(QIcon("icons/info_icon.png"))  # Use your actual icon path
        help_button.setCursor(Qt.PointingHandCursor)
        help_button.setStyleSheet("padding: 4px; font-weight: bold;")
        help_button.clicked.connect(lambda: self.go_to_help_section("generate"))

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
        
        self.alpha = QLineEdit()
        self.alpha.setPlaceholderText("0.05")
        layout.addRow(QLabel("alpha:"), self.alpha)
        
        self.shap_show = QLineEdit()
        self.shap_show.setPlaceholderText("10")
        layout.addRow(QLabel("shap_show:"), self.shap_show)
        
        self.pfi_show = QLineEdit()
        self.pfi_show.setPlaceholderText("10")
        layout.addRow(QLabel("pfi_show:"), self.pfi_show)

        # --- Help button at the bottom ---
        help_button = QPushButton("Help PREDICT parameters")
        help_button.setIcon(QIcon("icons/info_icon.png"))  # Use your actual icon path
        help_button.setCursor(Qt.PointingHandCursor)
        help_button.setStyleSheet("padding: 4px; font-weight: bold;")
        help_button.clicked.connect(lambda: self.go_to_help_section("predict"))

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
        self.initUI()

    def load_chemdraw_file(self, main_path, optional_path=None):
        """
        Opens one or two ChemDraw files (.sdf, .cdxml) and displays the molecules in a table.
        - If one file is given → table has Image, SMILES, code_name, target
        - If two files are given → table has Image, SMILES_1, SMILES_2, code_name, target
        """

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
            else:
                mol = Chem.MolFromMolFile(path)
                return [mol] if mol else []

        # Load molecules
        mols_main = load_mols_from_path(main_path)
        mols_optional = load_mols_from_path(optional_path) if optional_path else []

        if not mols_main:
            QMessageBox.warning(self, "Error", "No valid molecules found in the main file.")
            return

        max_len = max(len(mols_main), len(mols_optional))
        mols_main += [None] * (max_len - len(mols_main))
        mols_optional += [None] * (max_len - len(mols_optional)) if mols_optional else []

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("ChemDraw Molecules")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint)
        dialog.setSizeGripEnabled(True)
        dialog.resize(1100, 600)
        layout = QVBoxLayout(dialog)

        # Define headers based on file count
        if mols_optional:
            headers = ["Image", "SMILES_1", "SMILES_2", "code_name", "target"]
        else:
            headers = ["Image", "SMILES", "code_name", "target"]

        table = QTableWidget(max_len, len(headers))
        table.setHorizontalHeaderLabels(headers)

        for row in range(max_len):
            
            # Set molecule(s) for the row
            mol1 = mols_main[row]
            mol2 = mols_optional[row] if mols_optional else None

            # --- Display image(s) ---
            hbox = QHBoxLayout()
            hbox.setSpacing(5)
            widget = QWidget()

            for mol in [mol1, mol2] if mol2 else [mol1]:
                if mol:
                    img = Draw.MolToImage(mol, size=(100, 100))
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    qimg = QImage.fromData(buffer.getvalue())
                    label = QLabel()
                    label.setPixmap(QPixmap.fromImage(qimg).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    hbox.addWidget(label)

            widget.setLayout(hbox)
            table.setRowHeight(row, 110)
            table.setCellWidget(row, 0, widget)

            # --- Fill SMILES and placeholders ---
            if mol2:
                smi1 = Chem.MolToSmiles(mol1) if mol1 else ""
                smi2 = Chem.MolToSmiles(mol2) if mol2 else ""
                table.setItem(row, 1, QTableWidgetItem(smi1))
                table.setItem(row, 2, QTableWidgetItem(smi2))
                table.setItem(row, 3, QTableWidgetItem(""))  # code_name
                table.setItem(row, 4, QTableWidgetItem(""))  # target
            else:
                smi = Chem.MolToSmiles(mol1) if mol1 else ""
                table.setItem(row, 1, QTableWidgetItem(smi))
                table.setItem(row, 2, QTableWidgetItem(""))  # code_name
                table.setItem(row, 3, QTableWidgetItem(""))  # target


        layout.addWidget(table)

        # Save Button
        save_button = QPushButton("💾 Save as CSV")
        save_button.setStyleSheet("padding: 6px; font-weight: bold;")
        layout.addWidget(save_button, alignment=Qt.AlignRight)

        def save_to_csv():
            path, _ = QFileDialog.getSaveFileName(dialog, "Save CSV", "", "CSV Files (*.csv)")
            if not path:
                return

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if mols_optional:
                    writer.writerow(["SMILES_1", "SMILES_2", "code_name", "target"])
                else:
                    writer.writerow(["SMILES", "code_name", "target"])

                for row in range(table.rowCount()):
                    if mols_optional:
                        smi1 = table.item(row, 1).text() if table.item(row, 1) else ""
                        smi2 = table.item(row, 2).text() if table.item(row, 2) else ""
                        code = table.item(row, 3).text() if table.item(row, 3) else ""
                        target = table.item(row, 4).text() if table.item(row, 4) else ""
                        writer.writerow([smi1, smi2, code, target])
                    else:
                        smi = table.item(row, 1).text() if table.item(row, 1) else ""
                        code = table.item(row, 2).text() if table.item(row, 2) else ""
                        target = table.item(row, 3).text() if table.item(row, 3) else ""
                        writer.writerow([smi, code, target])

        save_button.clicked.connect(save_to_csv)
        dialog.setLayout(layout)
        dialog.exec()

    def open_chemdraw_popup(self):
        dialog = ChemDrawFileDialog(self)
        if dialog.exec():
            # If accepted, retrieve paths and pass them
            main_path = dialog.main_chemdraw_path
            optional_path = dialog.optional_chemdraw_path if dialog.optional_chemdraw_path else None
            self.load_chemdraw_file(main_path, optional_path)

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
        self.tab_widget.addTab(scroll_area, "Robert")


        # --- Add logo with frame ---
        path_logo = Path(__file__).parent / "icons" / "Robert_logo_transparent.png"       
        if path_logo.exists():
            pixmap = QPixmap(str(path_logo))

            # Resize the image while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(400, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label = QLabel(self)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
          
            # Add the logo to a frame and center it
            logo_frame = QFrame()
            logo_layout = QVBoxLayout()
            logo_layout.addWidget(logo_label, alignment=Qt.AlignCenter)
            logo_frame.setLayout(logo_layout)
            main_layout.addWidget(logo_frame, alignment=Qt.AlignCenter)

        # --- Set window icon ---
        path_icon = Path(__file__).parent / "icons" / "Robert_icon_n.png"
        try:
            self.setWindowIcon(QPixmap(str(path_icon)))
        except Exception:
            pass 

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

        test_layout.addWidget(self.csv_test_title)
        test_layout.addWidget(self.csv_test_label)
        
        # --- ChemDraw Button Centered ---
        self.chemdraw_button = QPushButton("Generate CSV from ChemDraw Files or SDF file")
        self.chemdraw_button.setStyleSheet("padding: 10px; font-weight: bold;")
        self.chemdraw_button.clicked.connect(self.open_chemdraw_popup)

        button_layout = QVBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.chemdraw_button, alignment=Qt.AlignCenter)
        button_layout.addStretch()

        # --- CSV Section with Button in the Middle ---
        csv_layout = QHBoxLayout()
        csv_layout.addLayout(input_layout)
        csv_layout.addLayout(button_layout)
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

        # Add layouts to the main horizontal layout
        column_layout.addLayout(left_layout)
        column_layout.addLayout(button_layout)
        column_layout.addLayout(right_layout)

        # Inserting into the main layout
        main_layout.addLayout(column_layout)
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
        # Get the absolute path of the icon
        icon_play_path = os.path.join(os.path.dirname(__file__), "icons", "play_icon.png")

        self.run_button = QPushButton(" Run ROBERT")
        self.run_button.setFixedSize(200, 40)  # Adjust button size

        # Load Play icon from the "icons" directory
        self.run_button.setIcon(QIcon(icon_play_path))

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
        stop_icon_path = os.path.join(os.path.dirname(__file__), "icons", "stop_icon.png")
        self.stop_button = QPushButton("Stop ROBERT")
        self.stop_button.setFixedSize(200, 40)
        self.stop_button.setIcon(QIcon(stop_icon_path))
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
        self.console_output.setMinimumHeight(200) # Set minimum height

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

        # --- Launch the process and connect output signals ---
        self.process = QProcess(self)
        self.process.setWorkingDirectory(os.path.dirname(self.file_path))
        self.process.finished.connect(self.on_process_finished)

        # Connect signals to capture standard output and error
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        
        # ================
        # Create Help Tab 
        # ================
        self.help_tab = QWidget()
        help_layout = QVBoxLayout(self.help_tab)

        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl("https://robert.readthedocs.io/en/latest/index.html#"))
        help_layout.addWidget(self.web_view)

        # ===============================
        # "Options" Tab (Additional Parameters)
        # ===============================
        self.options_tab = AdvancedOptionsTab(self.type_dropdown, self.tab_widget, self.help_tab, self.web_view)

        # ===============================
        # Add tabs to the main tab widget in desired order
        # ===============================
        self.tab_widget.addTab(self.options_tab, "Advanced Options")
        self.tab_widget.addTab(self.help_tab, "Help")

        # ===============================
        # "Help" Tab (Documentation)
        # ===============================
        self.help_tab = QWidget()
        help_layout = QVBoxLayout(self.help_tab)

        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl("https://robert.readthedocs.io/en/latest/index.html#"))
        help_layout.addWidget(self.web_view)


        # ===============================
        # "Results" Tab (PDF report)
        # ===============================

        # This tab will be dynamically enabled/disabled based on PDF presence
        # The order of the tabs is important for avoid the error of relaunch the rest of tabs
        self.tab_widget_results = QTabWidget()
        self.results_tab = ResultsTab(self.tab_widget_results)

        # Add the "Results" tab to the main tab widget
        self.tab_widget.addTab(self.tab_widget_results, "Results")

        # ===============================
        # "Images" Tab (Multiple Folders)
        # ===============================

        # Define paths to the folders you want to monitor
        self.image_folders = ["PREDICT", "GENERATE/Raw_data", "VERIFY", "CURATE"]

        # Create the "Images" tab
        self.images_tab = ImagesTab(self.tab_widget, self.image_folders)
        self.tab_widget.addTab(self.images_tab, "Images")

        # ===============================
        # Create Independent Timers for Each Check
        # ===============================

        # PDF Checking Timer
        self.timer_pdfs = QTimer(self)
        self.timer_pdfs.timeout.connect(self.check_for_pdfs)
        self.timer_pdfs.start(2000)

        # Image Checking Timer
        self.timer_images = QTimer(self)
        self.timer_images.timeout.connect(self.check_for_images)
        self.timer_images.start(2000)

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
 
    def select_file(self):
        """Opens file dialog to select a CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.set_file_path(file_path)

    def set_file_path(self, file_path):
        """Updates only the DropLabel text, keeping the title unchanged."""
        self.file_path = file_path
        self.file_label.setText(f"Selected: {file_path}")
        self.load_csv_columns()

    def select_csv_test_file(self):
        """Opens file dialog to select a test CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Test CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.set_csv_test_path(file_path)

    def set_csv_test_path(self, file_path):
        """Updates only the DropLabel text, keeping the title unchanged."""
        self.csv_test_path = file_path
        self.csv_test_label.setText(f"Selected: {file_path}")

    def set_main_chemdraw_path(self, file_path):
        self.main_chemdraw_path = file_path
        self.main_chemdraw_label.setText(f"Selected: {file_path}")

    def set_optional_chemdraw_path(self, file_path):
        self.optional_chemdraw_path = file_path
        self.optional_chemdraw_label.setText(f"Selected: {file_path}")

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

        # Build the base command.
        command = (
            f'python -u -m robert --csv_name "{os.path.basename(self.file_path)}" '
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
        self.descriptor_level_selected = self.options_tab.descriptor_level.currentText()
        self.atoms_selected = self.options_tab.atoms.text().strip()

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
        self.alpha = self.options_tab.alpha.text().strip()
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
            command += ' --aqme ' 

            # Always include descp_lvl
            command +=  f'--descp_lvl {self.descriptor_level_selected}'

            # Add qdescp_atoms only if it's not empty
            if self.atoms_selected:
            
                # Split the string by commas and strip whitespace
                atoms = [atom.strip() for atom in self.atoms_selected.split(',') if atom.strip()]
                atoms_str = '[' + ','.join(atoms) + ']'

                # Append to the command inside the --qdescp_keywords argument
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

        # --alpha (default: 0.05)
        if self.alpha:
            command += f" --alpha {self.alpha}"

        # --shap_show (default: 10)
        if self.shap_show:
            command += f" --shap_show {self.shap_show}"

        # --pfi_show (default: 10)
        if self.pfi_show:
            command += f" --pfi_show {self.pfi_show}"

        def check_variables(self):
            """Validates the values extracted from the Advanced Options tab."""
            errors = []

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

            if self.alpha:
                try:
                    value = float(self.alpha)
                    if not (0 <= value <= 1):
                        errors.append("alpha must be between 0 and 1.")
                except ValueError:
                    errors.append("alpha must be a number between 0 and 1.")

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
        
        # Confirmation dialog
        confirmation = QMessageBox.question(
            self, 
            "WARNING!", 
            "Are you sure you want to stop ROBERT?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )

        # If user selects "No", do nothing and return
        if confirmation == QMessageBox.No:
            return  
        
        # Set the flag for manual stop
        self.manual_stop = True

        # --- Proceed to stop the process ---
        if self.worker and self.worker.isRunning():
            self.worker.stop()  # Calls stop() in RobertWorker
            self.worker.wait()  # Ensure thread cleanup before setting to None
            self.worker = None  # Cleanup after 
            self.on_process_finished(-1)  # Call cleanup function

    def handle_stdout(self):
        """ Handles the output of the ROBERT process. """
        output = self.process.readAllStandardOutput().data().decode("utf-8")
        self.console_output.append(output)

    def handle_stderr(self):
        """ Handles the error output of the ROBERT process. """
        error_output = self.process.readAllStandardError().data().decode("utf-8")
        self.console_output.append(error_output)
        
    @Slot(int, QProcess.ExitStatus)
    def on_process_finished(self, exit_code):
        """Handles the cleanup after the ROBERT process finishes."""

        # Reset the buttons and progress bar
        self.run_button.setDisabled(False)
        self.stop_button.setDisabled(True)
        self.progress.setRange(0, 100)

        if self.worker:  # Ensure the subprocess is properly closed
            self.worker.quit()  # Stops the QThread
            self.worker.wait()  # Ensures cleanup
            self.worker = None  # Reset the worker

        #Check console output for success message
        if not self.manual_stop and self.workflow_selector.currentText() == "Full Workflow" or self.workflow_selector.currentText() == "REPORT":
            if exit_code == 0 and "o  ROBERT_report.pdf was created successfully in the working directory!" in self.console_output.toPlainText():
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setWindowTitle("Success!")
                msg_box.setText("ROBERT has completed successfully.")

                # Add custom "View Report" button
                view_report_button = QPushButton("View Report PDF")
                view_report_button.setIcon(QIcon("icons/pdf_icon.png"))
                msg_box.addButton(view_report_button, QMessageBox.ActionRole)

                # Add "OK" button 
                msg_box.addButton("OK", QMessageBox.AcceptRole)

                # Connect view_report_button to show results tab
                view_report_button.clicked.connect(lambda: self.tab_widget.setCurrentWidget(self.tab_widget_results))

                msg_box.exec()

            else:
                QMessageBox.warning(self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs.")

        # If the process was launched for a specific workflow, update the dropdown to the next step
        if self.workflow_selector.currentText() == "CURATE":
                if exit_code == 0 and "Time CURATE: " in self.console_output.toPlainText():
                    QMessageBox.information(self, "Success", "ROBERT has successfully completed the CURATE step.")
                else:
                    QMessageBox.warning(self, "WARNING!","ROBERT encountered an issue while finishing. Please check the logs.")

        if self.workflow_selector.currentText() == "GENERATE":
            if exit_code == 0 and "Time GENERATE: " in self.console_output.toPlainText():
                QMessageBox.information(self, "Success", "ROBERT has successfully completed the GENERATE step.")
            else:
                QMessageBox.warning(self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs.")

        if self.workflow_selector.currentText() == "PREDICT":
            if exit_code == 0 and "Time PREDICT: " in self.console_output.toPlainText():
                QMessageBox.information(self, "Success", "ROBERT has successfully completed the PREDICT step.")
            else:
                QMessageBox.warning(self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs.")

        if self.workflow_selector.currentText() == "VERIFY":
            if exit_code == 0 and "Time VERIFY: " in self.console_output.toPlainText():
                QMessageBox.information(self, "Success", "ROBERT has successfully completed the VERIFY step.")
            else:
                QMessageBox.warning(self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs.")

        # Show message box for stopping   
        if exit_code == -1:
            self.console_output.clear()  # Clear the console output
            QMessageBox.information(self, "WARNING!", "ROBERT has been successfully stopped.")
             
        # Reset the manual stop flag after the process is finished
        self.manual_stop = False

class DropLabel(QFrame):
    """A widget for file selection via drag-and-drop or browsing. Can be configured for different file types."""

    def __init__(self, text, parent=None, file_filter="CSV Files (*.csv)", extensions=(".csv",)):
        super().__init__(parent)
        
        self.file_filter = file_filter
        self.valid_extensions = extensions
        self.setAcceptDrops(True)
        self.callback = None  

        self.setStyleSheet("font-size: 14px; border: none;")
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # --- Instruction Label with Border ---
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

        # --- Browse Button ---
        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.open_file_dialog)
        self.browse_button.setFixedSize(120, 30)  
        self.browse_button.setStyleSheet(
            "padding: 6px 12px; font-size: 14px; border-radius: 5px; background-color: #555; color: white; border: 1px solid #777;"
        )
        self.layout.addWidget(self.browse_button, alignment=Qt.AlignCenter)

    def set_callback(self, callback):
        """Sets the callback function to be called when a file is selected."""
        self.callback = callback

    def set_file_type(self, file_filter, extensions):
        """Allows changing file type after creation (e.g. for .csv or .sdf)."""
        self.file_filter = file_filter
        self.valid_extensions = extensions

    def open_file_dialog(self):
        """Opens file dialog to select a file based on current filter."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self.file_filter)
        if file_path and self.callback:
            self.set_file_path(file_path)

    def dragEnterEvent(self, event):
        """Handles drag event to check if file is valid."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handles dropped file selection."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(self.valid_extensions):
                self.set_file_path(file_path)
            else:
                self.label.setText("⚠ Invalid file format.")

    def set_file_path(self, file_path):
        """Updates only the DropLabel text, keeping the title unchanged."""
        self.label.setText(f"Selected: {os.path.basename(file_path)}")
        if self.callback:
            self.callback(file_path)

    def setText(self, text):
        """Updates the label text inside DropLabel."""
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

    def run(self):
        """Runs the subprocess and streams output line by line in real-time with ANSI support."""
        self.process = subprocess.Popen(
            shlex.split(self.command),
            cwd=self.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
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
        """Stops the subprocess."""
        if self.process:
            self.process.kill()
            self.process = None

class ChemDrawFileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select ChemDraw Files")
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
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
            "Drag & Drop a main .sdf or .cdxml file here",
            self,
            file_filter="ChemDraw Files (*.sdf *.cdxml *.mol)",
            extensions=(".sdf", ".cdxml", ".mol")
        )
        self.main_label.set_callback(self.set_main_file)
        layout.addWidget(self.main_label)

        # Optional file
        self.optional_label = DropLabel(
            "Optional: Drag & Drop an additional .sdf or .cdxml file here",
            self,
            file_filter="ChemDraw Files (*.sdf *.cdxml *.mol)",
            extensions=(".sdf", ".cdxml", ".mol")
        )
        self.optional_label.set_callback(self.set_optional_file)
        layout.addWidget(self.optional_label)

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