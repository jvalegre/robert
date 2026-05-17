"""
Main application window and workflow for easyROB.

This module implements the central GUI controller of the application. It is
responsible not only for building the interface, but also for coordinating
the full workflow across different components.

Responsibilities:
- Construct and manage the full GUI (tabs, widgets, layouts)
- Coordinate data flow between tabs and UI components
- Launch and manage background processes (ROBERT, AQME, MolSSI)
- Handle user interactions, validation, and workflow decisions
- Synchronize UI state with filesystem outputs (CSV, reports, images)

Import strategy:
- Supports dual execution modes:
  1. Portable/local execution (running from source folder)
  2. Installed package execution (environment / entry point)

- Imports use a try/except fallback:
  local modules are attempted first, with package imports as fallback.
  This ensures compatibility without requiring strict packaging.

Notes:
- This module intentionally centralizes orchestration logic.
- Its size reflects its role as the integration point of the application.

"""

# ------------------------------------------------------------
# Import resolution (local vs installed package)
# ------------------------------------------------------------
# Try local imports first (portable mode). If they fail,
# fall back to the installed package structure.

import webbrowser

try:

    from version import SOFTWARE_VERSIONS
    from utils import utils_gui, molssi_utils
    from tabs import predictions, aqme, advanced_options, molssi, results, images

except ImportError as e:

    from robert.gui_easyrob.version import SOFTWARE_VERSIONS
    from robert.gui_easyrob.utils import utils_gui, molssi_utils
    from robert.gui_easyrob.tabs import predictions, aqme, advanced_options, molssi, results, images


# ------------------------------------------------------------
# Extract commonly used symbols
# ------------------------------------------------------------
# We alias frequently used classes/functions locally to:
# - avoid long module paths
# - improve readability across this large file
# - keep usage consistent regardless of import mode

Ansi2HTMLConverter = utils_gui.Ansi2HTMLConverter
AssetLibrary = utils_gui.AssetLibrary
Chem = utils_gui.Chem
DropLabel = utils_gui.DropLabel
NoScrollComboBox = utils_gui.NoScrollComboBox
Path = utils_gui.Path
QApplication = utils_gui.QApplication
QCheckBox = utils_gui.QCheckBox
QDesktopServices = utils_gui.QDesktopServices
QDialog = utils_gui.QDialog
QEventLoop = utils_gui.QEventLoop
QFileDialog = utils_gui.QFileDialog
QFrame = utils_gui.QFrame
QHBoxLayout = utils_gui.QHBoxLayout
QIcon = utils_gui.QIcon
QLabel = utils_gui.QLabel
QListWidget = utils_gui.QListWidget
QMainWindow = utils_gui.QMainWindow
QMessageBox = utils_gui.QMessageBox
QPixmap = utils_gui.QPixmap
QProgressBar = utils_gui.QProgressBar
QPushButton = utils_gui.QPushButton
QScrollArea = utils_gui.QScrollArea
QSize = utils_gui.QSize
QSizePolicy = utils_gui.QSizePolicy
QStackedWidget = utils_gui.QStackedWidget
QStatusBar = utils_gui.QStatusBar
QStyle = utils_gui.QStyle
QTabWidget = utils_gui.QTabWidget
QTextEdit = utils_gui.QTextEdit
QTimer = utils_gui.QTimer
QToolButton = utils_gui.QToolButton
QUrl = utils_gui.QUrl
QVBoxLayout = utils_gui.QVBoxLayout
QWidget = utils_gui.QWidget
Qt = utils_gui.Qt
Slot = utils_gui.Slot
glob = utils_gui.glob
os = utils_gui.os
pd = utils_gui.pd
re = utils_gui.re
shutil = utils_gui.shutil
smart_read_csv = utils_gui.smart_read_csv
sys = utils_gui.sys

# Workers (background processes)
MolSSIDownloadWorker = molssi_utils.MolSSIDownloadWorker
MolSSIWorker = molssi_utils.MolSSIWorker
RobertWorker = utils_gui.RobertWorker

# Tabs (UI modules)
PredictionsTab = predictions.PredictionsTab
AQMETab = aqme.AQMETab
AdvancedOptionsTab = advanced_options.AdvancedOptionsTab
MolSSIDatabasesTab = molssi.MolSSIDatabasesTab
ResultsTab = results.ResultsTab
ImagesTab = images.ImagesTab

# ------------------------------------------------------------
# Base directory (used for assets, tutorials, etc.)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

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
        self._molssi_workers = set() # Keep track of MolSSI workers
        self.molssi_is_closing = False
        self.initUI()
        self.clear_test_button.setVisible(False) # Hide the button initially
        self.molssi_tab.load_test_requested.connect(self.set_csv_test_path) # Connect signal with molssi tab donwload test requested

    def closeEvent(self, event):
        """Handle the window close event, ensuring proper shutdown of workers."""
        worker = getattr(self, 'worker', None)

        if worker is not None and worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Exit Confirmation",
                "ROBERT is still running. Do you want to stop the process and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

            # User said YES → now we are really closing
            self.molssi_is_closing = True
            loop = QEventLoop()
            worker.process_finished.connect(loop.quit)
            worker.stop()

            QTimer.singleShot(5000, loop.quit)
            loop.exec()

            self._shutdown_molssi_async()
            event.ignore()
            return

        # No ROBERT → real close
        self.molssi_is_closing = True
        self._shutdown_molssi_async()
        event.ignore()

    def _reset_ui_after_process(self):
        """Reset UI elements after the ROBERT process has finished."""
        self.run_button.setDisabled(False)
        self.run_aqme_button.setDisabled(False)
        self.stop_button.setDisabled(True)
        self.progress.setRange(0, 100)

    def _shutdown_molssi_async(self):
        """Shut down MolSSI workers asynchronously."""
        self.hide()

        for w in self._molssi_workers:
            if w.isRunning():
                w.quit()

        QTimer.singleShot(100, self._poll_molssi_exit)

    def _poll_molssi_exit(self):
        """Poll MolSSI workers until all have exited, then quit application."""
        if any(w.isRunning() for w in self._molssi_workers):
            QTimer.singleShot(100, self._poll_molssi_exit)
            return

        QApplication.quit()

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
            
    def open_external_url(self,url: str):
        """Open URL using the system default browser."""
        try:
            webbrowser.open(url, new=2)  # new=2 → new tab if possible
        except Exception as e:
            print(f"Failed to open URL: {url}\nError: {e}")

    def initUI(self):
        """Initializes the main user interface."""

        # Initial window size
        self.resize(750, 775)

        # Parameters for the GUI
        box_features = """
        QComboBox {
            border: 1px solid palette(mid);
            border-radius: 6px;
            padding: 4px;
        }
        """

        box_features_ignore = """
        QListWidget {
            border: 1px solid palette(mid);
            border-radius: 6px;
        }
        """
        self.setWindowTitle("easyROB")
        
        # Create main tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # ---------------------------------
        # Bottom status bar (clean version)
        # ---------------------------------

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_bar.setSizeGripEnabled(False)

        self.status_bar.setStyleSheet("""
        QStatusBar::item {
            border: none;
        }
        """)

        tool_style = """
        QToolButton {
            border: none;
            padding: 4px;
        }
        QToolButton:hover {
            text-decoration: underline;
        }
        """

        # Tutorial
        tutorial_btn = QToolButton()
        tutorial_btn.setText("Tutorial")
        tutorial_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        tutorial_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        tutorial_btn.setIconSize(QSize(14, 14))
        tutorial_btn.setCursor(Qt.PointingHandCursor)
        tutorial_btn.setStyleSheet(tool_style)
        tutorial_btn.clicked.connect(self.show_tutorial_dialog)

        # YouTube
        youtube_btn = QToolButton()
        youtube_btn.setText("YouTube")
        youtube_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        with AssetLibrary.Youtube_icon.get_path() as icon_youtube_path:
            youtube_btn.setIcon(QIcon(str(icon_youtube_path)))
        youtube_btn.setIconSize(QSize(18, 18))
        youtube_btn.setCursor(Qt.PointingHandCursor)
        youtube_btn.setStyleSheet(tool_style)
        youtube_btn.clicked.connect(
            lambda: self.open_external_url("https://www.youtube.com/@thealegregroup4964/videos")
        )

        # Documentation
        docs_btn = QToolButton()
        docs_btn.setText("Documentation")
        docs_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        with AssetLibrary.Documentation_icon.get_path() as icon_documentation_path:
            docs_btn.setIcon(QIcon(str(icon_documentation_path)))
        docs_btn.setIconSize(QSize(18, 18))
        docs_btn.setCursor(Qt.PointingHandCursor)
        docs_btn.setStyleSheet(tool_style)
        docs_btn.clicked.connect(
            lambda: self.open_external_url("https://robert.readthedocs.io/en/latest/")
        )

        # Contact
        contact_btn = QToolButton()
        contact_btn.setText("Contact")
        contact_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        contact_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        contact_btn.setIconSize(QSize(14, 14))
        contact_btn.setCursor(Qt.PointingHandCursor)
        contact_btn.setStyleSheet(tool_style)
        contact_btn.clicked.connect(self.show_contact_dialog)

        # Version
        version_btn = QToolButton()
        version_btn.setText("Version")
        version_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        version_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView))
        version_btn.setIconSize(QSize(14, 14))
        version_btn.setCursor(Qt.PointingHandCursor)
        version_btn.setStyleSheet(tool_style)
        version_btn.clicked.connect(self.show_version_dialog)

        self.status_bar.addWidget(tutorial_btn)
        self.status_bar.addWidget(docs_btn)
        self.status_bar.addWidget(youtube_btn)
        self.status_bar.addPermanentWidget(contact_btn)
        self.status_bar.addPermanentWidget(version_btn)

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
            scaled_pixmap = pixmap.scaled(300, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)

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
        self.file_title.setStyleSheet("font-weight:600; font-size:16px;")
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

        self.csv_test_title = QLabel("Select External Test CSV File (optional)", self)
        self.csv_test_title.setAlignment(Qt.AlignCenter)
        self.csv_test_title.setStyleSheet("font-weight:600; font-size:16px;")

        self.csv_test_label = DropLabel(
            "Drag & Drop a external CSV test file here (optional)",
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
        self.y_label.setStyleSheet("font-size:13px;")
        main_layout.addWidget(self.y_label)
        self.y_dropdown = NoScrollComboBox()        
        main_layout.addWidget(self.y_dropdown)
        self.y_dropdown.setStyleSheet(box_features)
        
        # --- Select prediction type ---
        self.type_label = QLabel("Prediction Type")
        self.type_label.setStyleSheet("font-size:13px;")
        main_layout.addWidget(self.type_label)
        self.type_dropdown = NoScrollComboBox()
        self.type_dropdown.addItems(["Regression", "Classification"])
        main_layout.addWidget(self.type_dropdown)
        self.type_dropdown.setStyleSheet(box_features)
        
        # --- Select column for --names ---
        self.names_label = QLabel("Select name column")
        self.names_label.setStyleSheet("font-size:13px;")
        main_layout.addWidget(self.names_label)
        self.names_dropdown = NoScrollComboBox()
        main_layout.addWidget(self.names_dropdown) 
        self.names_dropdown.setStyleSheet(box_features)
     
        # Main horizontal layout for column selection
        column_layout = QHBoxLayout()

        # Left side (Available Columns)
        left_layout = QVBoxLayout()
        self.available_label = QLabel("Available Columns")
        self.available_label.setStyleSheet("font-size:13px; font-weight:500;")
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.MultiSelection)
        self.available_list.setStyleSheet(box_features_ignore)
        left_layout.addWidget(self.available_label)
        left_layout.addWidget(self.available_list)

        # Button layout (Centered between lists)
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignVCenter)  # Ensure vertical centering

        self.add_button = QPushButton(">>")
        self.add_button.setFixedSize(30, 24)
        self.add_button.clicked.connect(self.move_to_selected)   # Moves selected items to "Ignored Columns"
        self.remove_button = QPushButton("<<")
        self.remove_button.setFixedSize(30, 24)
        self.remove_button.clicked.connect(self.move_to_available) # Moves selected items back to "Available Columns"
        button_style = """
            QPushButton {
                border: 1px solid palette(mid);
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton:hover {
                background: palette(light);
            }
            """

        self.add_button.setStyleSheet(button_style)
        self.remove_button.setStyleSheet(button_style)  

        # Add buttons to the button layout
        button_layout.addStretch()  
        button_layout.addWidget(self.add_button, alignment=Qt.AlignCenter)
        button_layout.addWidget(self.remove_button, alignment=Qt.AlignCenter)
        button_layout.addStretch()  

        # Right side (Ignored Columns)
        right_layout = QVBoxLayout()
        self.ignored_label = QLabel("Ignored Columns")
        self.ignored_label.setStyleSheet("font-size:13px; font-weight:500;")
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
        column_container.setFixedHeight(120) 

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

        self.run_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
                border-radius: 10px;
                background-color: #6A0DAD;      /* Purple */
                color: white;
                border: 2px solid #7B2CBF;
            }
            QPushButton:hover {
                background-color: #7B2CBF;      /* Lighter purple */
                border: 2px solid #9D4EDD;
            }
            QPushButton:pressed {
                background-color: #4B0082;      /* Darker purple */
                border: 2px solid #5A189A;
            }
            QPushButton:disabled {
                background-color: #3A2A4D;
                border: 2px solid #3A2A4D;
                color: #AAA;
            }
        """)

        self.run_button.clicked.connect(self.run_robert)

        # --- Run AQME button ---
        self.run_aqme_button = QPushButton(" Run AQME (descriptor generation)")
        self.run_aqme_button.setFixedSize(270, 40)  # Slightly wider for text
        self.run_aqme_button.setCursor(Qt.PointingHandCursor)

        with AssetLibrary.Play_icon.get_path() as icon_play_path:
            self.run_aqme_button.setIcon(QIcon(str(icon_play_path)))

        # Apply custom styles
        self.run_aqme_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
                border-radius: 10px;
                background-color: #1E88E5;
                color: white;
                border: 2px solid #1976D2;
            }
            QPushButton:hover {
                background-color: #1976D2;
                border: 2px solid #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
                border: 2px solid #08306B;
            }
            QPushButton:disabled {
                background-color: #1E88E5;
                border: 2px solid #1976D2;
                color: rgba(255, 255, 255, 120);
            }
        """)


        self.run_aqme_button.clicked.connect(self.run_aqme)

        # --- Stop Button ---
        self.stop_button = QPushButton("Stop")
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
                background-color: #C62828;
                color: white;
                border: 2px solid #B71C1C;
            }
            QPushButton:hover {
                background-color: #D32F2F;
                border: 2px solid #C62828;
            }
            QPushButton:pressed {
                background-color: #8E0000;
                border: 2px solid #5F0000;
            }
            QPushButton:disabled {
                background-color: #C62828;
                border: 2px solid #B71C1C;
                color: rgba(255, 255, 255, 120);
            }
        """)

        self.stop_button.clicked.connect(self.stop_process)

        # Add button layout to the main layout
        button_container = QHBoxLayout()
        button_container.addWidget(self.run_button)
        button_container.addWidget(self.run_aqme_button)
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
        self.console_output.setMinimumHeight(250)  
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

        # Add Help tab to the main tab widget
        self.molssi_tab = MolSSIDatabasesTab(self.tab_widget)

        # AQME tab (depends on ResultsTab via main_window)
        self.tab_widget_aqme = AQMETab(
            tab_parent=self.tab_widget,
            main_window=self,
        )

        # Options tab
        self.options_tab = AdvancedOptionsTab(
            self.type_dropdown,
            self.tab_widget,
        )

        # The content should have a normal "Preferred" policy so its sizeHint can be larger than the viewport.
        self.options_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.options_tab.setMinimumSize(0, 0)

        # Wrap options tab in a scroll area to handle large content
        options_scroll = QScrollArea()
        options_scroll.setWidgetResizable(True)
        options_scroll.setMinimumSize(0, 0)  # Prevent the scroll area from imposing a big minimum on the window
        options_scroll.setWidget(self.options_tab)

        # Images tab
        self.image_folders = ["PREDICT", "GENERATE/Raw_data", "VERIFY", "CURATE"]
        self.images_tab = ImagesTab(self.tab_widget, self.image_folders, self.file_path)

        # Results tab (must be created early so others can reference it)
        self.results_tab = ResultsTab(self.tab_widget, self.file_path)

        # Predictions tab
        self.predictions_tab = PredictionsTab(self.tab_widget)

        # ===============================
        # Add Tabs to Tab Widget (Display order)
        # ===============================

        self.tab_widget.addTab(self.tab_widget_aqme, "AQME")
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_widget_aqme), False)

        self.tab_widget.addTab(options_scroll, "Advanced Options")
        
        self.tab_widget.addTab(self.molssi_tab, "MolSSI Databases")

        self.tab_widget.addTab(self.results_tab, "Reports")
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.results_tab), False)

        self.tab_widget.addTab(self.images_tab, "Images")
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.images_tab), False)

        self.tab_widget.addTab(self.predictions_tab, "Predictions")

        # Start disabled
        self.tab_widget.setTabEnabled(
            self.tab_widget.indexOf(self.predictions_tab),
            False
        )

        # React to availability decided by the tab itself
        self.predictions_tab.availabilityChanged.connect(
            lambda ok: self.tab_widget.setTabEnabled(
                self.tab_widget.indexOf(self.predictions_tab),
                ok
            )
        )

    def show_contact_dialog(self):
        """Display contact dialog with clickable and copyable emails."""

        dialog = QDialog(self)
        dialog.setWindowTitle("Contact & Support")
        dialog.setMinimumWidth(350)

        layout = QVBoxLayout(dialog)

        label = QLabel()
        label.setTextFormat(Qt.RichText)
        label.setTextInteractionFlags(
            Qt.TextSelectableByMouse |
            Qt.LinksAccessibleByMouse
        )
        label.setOpenExternalLinks(True)

        label.setText(
            """
            <b>For questions, feedback or collaboration:</b><br><br>

            📧 <a href="mailto:miguel.martinez@csic.es">miguel.martinez@csic.es</a><br>
            📧 <a href="mailto:ddalmau@unizar.es">ddalmau@unizar.es</a><br>
            📧 <a href="mailto:jv.alegre@csic.es">jv.alegre@csic.es</a><br><br>
            
            We are happy to hear from you.
            """
        )

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)

        layout.addWidget(label)
        layout.addWidget(close_button, alignment=Qt.AlignRight)

        dialog.exec()

    def load_tutorial(self, name):
        """Load tutorial blocks from markdown file."""

        tutorial_dir = BASE_DIR / "tutorials"
        file_path = tutorial_dir / f"{name}.md"

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        return [
            block.strip().replace("\n", " ")
            for block in text.split("---")
            if block.strip()
    ]

    def show_tutorial_dialog(self):
        """Display workflow tutorial dialog."""

        if hasattr(self, "tutorial_dialog") and self.tutorial_dialog.isVisible():
            self.tutorial_dialog.raise_()
            self.tutorial_dialog.activateWindow()
            return

        self.tutorial_dialog = QDialog(self)
        dialog = self.tutorial_dialog

        dialog.setWindowTitle("Workflow Tutorial")
        dialog.setFixedSize(900, 700)

        self.tutorial_layout = QVBoxLayout(dialog)

        header = QLabel(
            "<b style='font-size:20px;'>easyROB Workflow Guide</b><br>"
            "<span style='color:gray;'>Interface overview and practical tutorials</span>"
        )
        header.setAlignment(Qt.AlignCenter)

        self.tutorial_layout.addWidget(header)

        dialog.show()

        QTimer.singleShot(0, self._build_tutorial_tabs)

    def _build_tutorial_tabs(self):
        """Build the heavy UI after the dialog is visible."""

        tabs = QTabWidget()

        tutorials = [
            ("overview", "Overview"),
            ("csv", "From CSV"),
            ("chemdraw", "From ChemDraw"),
            ("predictions", "New Predictions"),
            ("descriptors", "Generate Descriptors"),
        ]

        for folder, title in tutorials:
            texts = self.load_tutorial(folder)
            tabs.addTab(
                self.create_tutorial_tab(folder, texts),
                title
            )

        self.tutorial_layout.addWidget(tabs)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.tutorial_dialog.close)

        self.tutorial_layout.addWidget(close_button, alignment=Qt.AlignRight)

    def create_tutorial_tab(self, folder_name, texts):
        """Create a tutorial tab with image left and text right."""

        base = BASE_DIR / "tutorials" / "tutorial_images" / folder_name
        images = sorted(
            base.glob(f"{folder_name}_*.png"),
            key=lambda p: [int(s) if s.isdigit() else s for s in p.stem.split("_")]
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)

        stacked = QStackedWidget()

        for i, image_path in enumerate(images):

            page = QWidget()
            page_layout = QHBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(20)

            # -------- Image container (fixed) --------
            image_container = QWidget()
            image_container.setFixedSize(540, 520)

            image_layout = QVBoxLayout(image_container)
            image_layout.setContentsMargins(0, 0, 0, 0)

            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)

            pixmap = QPixmap(str(image_path))

            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    520,
                    520,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled)
            else:
                image_label.setText(f"Image not found:\n{image_path}")

            image_layout.addWidget(image_label, alignment=Qt.AlignCenter)

            # -------- Text (scrollable) --------
            text = texts[i] if i < len(texts) else ""
            text = f"<div style='text-align:justify'>{text}</div>"

            text_label = QLabel(text)
            text_label.setWordWrap(True)
            text_label.setAlignment(Qt.AlignTop)
            text_label.setTextFormat(Qt.RichText)

            text_label.setStyleSheet("""
                font-size:14px;
                padding:12px;
            """)

            text_label.setFixedWidth(320)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(text_label)
            scroll.setFixedWidth(340)
            scroll.setFrameShape(QFrame.NoFrame)

            page_layout.addWidget(image_container)
            page_layout.addWidget(scroll)

            stacked.addWidget(page)

        layout.addWidget(stacked)

        # -------- Navigation --------
        nav_layout = QHBoxLayout()

        prev_btn = QPushButton("Previous")
        next_btn = QPushButton("Next")

        step_label = QLabel()
        step_label.setAlignment(Qt.AlignCenter)

        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(step_label)
        nav_layout.addWidget(next_btn)

        layout.addLayout(nav_layout)

        def update_step():
            step_label.setText(f"Step {stacked.currentIndex()+1} / {stacked.count()}")

        def next_step():
            i = (stacked.currentIndex() + 1) % stacked.count()
            stacked.setCurrentIndex(i)
            update_step()

        def prev_step():
            i = (stacked.currentIndex() - 1) % stacked.count()
            stacked.setCurrentIndex(i)
            update_step()

        next_btn.clicked.connect(next_step)
        prev_btn.clicked.connect(prev_step)

        update_step()

        return tab
        
    def show_version_dialog(self):
        """Display styled version dialog."""

        dialog = QDialog(self)
        dialog.setWindowTitle("About easyROB")
        dialog.setMinimumWidth(420)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)

        # ---- App title ----
        title = QLabel("<b style='font-size:20px;'>easyROB</b>")
        title.setAlignment(Qt.AlignCenter)

        version = QLabel(f"Version {SOFTWARE_VERSIONS['easyROB']}")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: gray;")

        layout.addWidget(title)
        layout.addWidget(version)

        # ---- Divider ----
        divider = QLabel("<hr>")
        layout.addWidget(divider)

        # ---- Dependencies title ----
        deps_title = QLabel("<b>Dependencies</b>")
        layout.addWidget(deps_title)

        # ---- Dependencies list ----
        deps_html = "<table style='margin-left:15px;'>"

        for name, ver in SOFTWARE_VERSIONS["Dependencies"].items():
            deps_html += f"""
            <tr>
                <td style='padding:4px 20px 4px 0;'>{name}</td>
                <td style='padding:4px 0; color: gray;'>{ver}</td>
            </tr>
            """

        deps_html += "</table>"

        deps_label = QLabel()
        deps_label.setTextFormat(Qt.RichText)
        deps_label.setText(deps_html)
        deps_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout.addWidget(deps_label)

        # ---- Close button ----
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignRight)

        dialog.exec()

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

    def check_for_images(self, base_path: str):
        """Enable or disable the 'Images' tab based on image folder presence."""
        if not base_path:
            return

        run_dir = os.path.dirname(base_path)

        has_folders = any(
            os.path.exists(os.path.join(run_dir, folder))
            for folder in self.image_folders
        )

        tab_index = self.tab_widget.indexOf(self.images_tab)
        if tab_index != -1:
            self.tab_widget.setTabEnabled(tab_index, has_folders)

    def check_for_pdfs(self, base_path: str):
        """Enable or disable the 'Results' tab based on PDF presence."""
        if not base_path:
            return

        run_dir = os.path.dirname(base_path)
        pdf_pattern = os.path.join(run_dir, "ROBERT_report*.pdf")
        has_pdfs = bool(glob.glob(pdf_pattern))

        tab_index = self.tab_widget.indexOf(self.results_tab)
        if tab_index != -1:
            self.tab_widget.setTabEnabled(tab_index, has_pdfs)

    def refresh_tabs(self, file_path):
        """Refresh the Results, Images, and Predictions tabs"""

        if not file_path:
            return

        # Save latest requested path
        self._pending_refresh_path = file_path

        # If a refresh is already scheduled, don't schedule another
        if getattr(self, "_refresh_scheduled", False):
            return

        self._refresh_scheduled = True

        # Schedule refresh after a short delay for avoid freeze popups 
        QTimer.singleShot(50, self._execute_refresh_tabs)

    def _execute_refresh_tabs(self):
        self._refresh_scheduled = False

        file_path = getattr(self, "_pending_refresh_path", None)
        if not file_path:
            return

        if hasattr(self, "results_tab"):
            self.results_tab.refresh_with_new_path(file_path)

        if hasattr(self, "images_tab"):
            self.images_tab.refresh_with_new_path(file_path)

        if hasattr(self, "predictions_tab"):
            self.predictions_tab.refresh_with_new_path(file_path)

        self.check_for_pdfs(file_path)
        self.check_for_images(file_path)

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

    def set_file_path(self, file_path: str, force: bool = False):
        """
        Sets the path for the input CSV file and updates the interface.
        Reloads if the file path changed OR the file was modified (mtime) OR force=True.
        """
        p = Path(file_path)
        current_path = getattr(self, 'file_path', None)
        current_mtime = getattr(self, '_file_mtime', None)

        # Compute new file's modification time (None if missing)
        try:
            new_mtime = p.stat().st_mtime if p.exists() else None
        except OSError:
            new_mtime = None

        same_path = (current_path == file_path)
        same_mtime = (current_mtime == new_mtime)

        # If nothing changed and not forced, bail early
        if same_path and same_mtime and not force:
            return

        # Update internal state
        self.file_path = file_path
        self._file_mtime = new_mtime

        file_name = p.name
        self.file_label.setText(f"Selected: {file_name}")
        self.file_label.setToolTip(file_path)
        self._last_loaded_file_path = None  # reset

        # Clear previous content
        if hasattr(self.tab_widget_aqme, "df_mapped_smiles"):
            self.tab_widget_aqme.df_mapped_smiles = None

        # Reload downstream state
        self.load_csv_columns()
        self.refresh_tabs(file_path)

        # Check for MolSSI descriptors 
        if not self._is_molssi_csv(file_path):
            self.check_molssi_descriptors()

        # Update unified SMILES context (TRAIN + optional TEST)
        self._update_unified_smiles_context()

        # Check for AQME workflow
        self.check_aqme_workflow()

    def set_csv_test_path(self, file_path):

        """Sets the path for the test CSV file and updates the label."""
        self.csv_test_path = file_path
        file_name = Path(file_path).name
        self.csv_test_label.setText(f"Selected: {file_name}")
        self.csv_test_label.setToolTip(file_path)
        self.clear_test_button.setVisible(True)

        # Update unified SMILES context (TRAIN + TEST)
        self._update_unified_smiles_context()
        self.check_aqme_workflow()

        # Refresh tabs
        self.refresh_tabs(file_path)

    def clear_test_file(self):
        """Clear the selected test file and resync dependent state."""
        self.csv_test_path = None

        # Reset UI
        self.csv_test_label.setText(
            "Drag & Drop a CSV external test file here (optional)"
        )
        self.csv_test_label.setToolTip("")
        self.clear_test_button.setVisible(False)

        # Rebuild chemical context (TRAIN-only)
        self._update_unified_smiles_context()

        # Refresh AQME tab / FMCS if enabled
        self.check_aqme_workflow()

    def _get_unmapped_csv(self, csv_path: str) -> str:
        """
        If csv_path ends with '_mapped.csv', return the original CSV path.
        Otherwise return csv_path unchanged.
        """
        if csv_path and csv_path.endswith("_mapped.csv"):
            return csv_path.replace("_mapped.csv", ".csv")
        return csv_path

    def start_molssi_test_download(self, library_slug):
        """
        Automatically download a MolSSI Excel library and trigger
        the existing post-download pipeline.
        """
        filename = f"{library_slug}_library.xlsx"

        download_dir = Path(self.file_path).parent
        download_dir.mkdir(parents=True, exist_ok=True)

        target_path = download_dir / filename

        urls = [
            f"https://descriptor-libraries.molssi.org/{library_slug}/content/{filename}",
            f"https://descriptor-libraries.molssi.org/{library_slug}/",
        ]

        # --------------------------------------------------
        # Popup: downloading external dataset (BLOCKING UX)
        # --------------------------------------------------
        popup = QMessageBox(self)
        popup.setWindowTitle("Downloading MolSSI dataset")
        popup.setText(
            "Downloading MolSSI database for use as an external test set.\n\n"
            "Please wait…"
        )
        popup.setStandardButtons(QMessageBox.NoButton)
        popup.setModal(True)
        popup.show()

        # --------------------------------------------------
        # Worker
        # --------------------------------------------------
        self._molssi_download_worker = MolSSIDownloadWorker(urls, target_path, self)

        def _finished(path):
            popup.close()
            popup.deleteLater()

            self.molssi_tab.handle_external_download(
                path,
                context="molssi_test"
            )

        def _error(msg):
            popup.close()
            popup.deleteLater()

            QMessageBox.warning(
                self,
                "MolSSI download failed",
                msg
            )

        self._molssi_download_worker.finished.connect(_finished)
        self._molssi_download_worker.error.connect(_error)
        self._molssi_download_worker.start()

    def _is_molssi_csv(self, file_path: str) -> bool:
        """
        Return True if the file path corresponds to a MolSSI-generated CSV.
        """
        return "_molssi_" in Path(file_path).stem
    
    def check_molssi_descriptors(self):
        """Check for MolSSI descriptors in the current dataset."""
        worker = MolSSIWorker(
            self.df,
            self.file_path,
            should_abort=lambda: self.molssi_is_closing,
            debug=True
        )

        self._molssi_workers.add(worker)

        worker.finished.connect(self.on_molssi_finished)
        worker.finished.connect(lambda _: self._molssi_workers.discard(worker))
        worker.finished.connect(worker.deleteLater)

        worker.start()
                
    def on_molssi_finished(self, result):
        """Called when MolSSI worker finishes."""

        # The sender should always be the worker emitting the signal
        worker = self.sender()
        if worker is None:
            # Defensive guard: signal emitted without a valid sender
            return

        # Ignore signals not coming from MolSSIWorker instances
        if not isinstance(worker, MolSSIWorker):
            return

        # Discard results from outdated workers (e.g. user loaded a new file)
        if worker.file_path != self.file_path:
            return

        # from here on, logic is SAFE
        self.current_download_context = "active"

        if not result["available"]:
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("MolSSI descriptors available")
        msg.setIcon(QMessageBox.Question)

        msg.setText(
            "Curated molecular descriptors are available for all molecules via the "
            "<a href='https://descriptor-libraries.molssi.org/'>"
            "MolSSI Descriptor Libraries</a>.<br><br>"
            "If selected, a new CSV dataset will be generated and loaded, preserving "
            "all original columns and adding the MolSSI descriptors.<br><br>"
            "Alternatively, descriptors can be generated locally using AQME.<br><br>"
            "More information is available in the MolSSI Databases tab."
        )

        msg.setTextFormat(Qt.RichText)
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)

        reply = msg.exec()

        if reply != QMessageBox.Yes:
            return

        input_path = Path(self.file_path)
        output_path = input_path.with_name(
            f"{input_path.stem}_molssi_{result['library']}_{result['data_type']}.csv"
        )

        df_out = result["export_df"]
        df_out.to_csv(output_path, index=False)

        self.set_file_path(str(output_path), force=True)

        # =====================================================
        # Gate for FULL MolSSI TEST DATASET
        # =====================================================
        can_export_test = result.get("export_available", False)

        if not can_export_test:
            return

        # =====================================================
        # Popup 2 — Offer full MolSSI dataset as TEST set
        # (UI only, no logic yet)
        # =====================================================
        msg = QMessageBox(self)
        msg.setWindowTitle("External MolSSI test dataset available")
        msg.setIcon(QMessageBox.Question)

        msg.setText(
            "Your molecules are fully covered by the MolSSI "
            f"{result['library']} ({result['data_type']}) database.\n\n"

            "In addition to generating descriptors for your current dataset, "
            "you can also load the complete MolSSI database as an external test set.\n\n"

            "This external dataset contains all molecules available in the MolSSI "
            "database with the same type of descriptors and can be used to:\n\n"
            "• Evaluate model performance\n"
            "• Validate predictions\n"
            "• Explore new candidate molecules\n\n"

            "The test dataset will be loaded separately and will NOT modify your "
            "current dataset.\n\n"

            "Do you want to load the full MolSSI dataset as a test set?"
        )

        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        reply = msg.exec()

        if reply == QMessageBox.Yes:
            # Mark context
            self.current_download_context = "molssi_test"

            # Start automatic MolSSI download
            self.start_molssi_test_download(
                library_slug=result["library"]
            )

    def _update_unified_smiles_context(self):
        """
        Build unified SMILES context (TRAIN + optional TEST)
        and store it in AQME tab for mol_viewer.
        """
        # TRAIN must exist
        if not hasattr(self, "file_path") or not self.file_path:
            return

        # AQME tab must exist
        if not hasattr(self, "tab_widget_aqme"):
            return

        if self.csv_test_path:
            unified_smiles = self.tab_widget_aqme.build_unified_smiles_context(
                self.file_path,
                self.csv_test_path
            )
        else:
            unified_smiles = self.tab_widget_aqme.build_unified_smiles_context(
                self.file_path
            )
        
        self.tab_widget_aqme.unified_smiles = unified_smiles

    def set_main_chemdraw_path(self, file_path):
        self.main_chemdraw_path = file_path
        file_name = Path(file_path).name
        self.main_chemdraw_label.setText(f"Selected: {file_name}")
        self.main_chemdraw_label.setToolTip(file_path)

    def load_csv_columns(self):
        """Loads column names from the selected CSV file and updates all selection fields."""
        if self.file_path:
            self.df = smart_read_csv(self.file_path)
            df = self.df
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

            # ---------------------------------------------------------
            # Auto-select ignore and names from CSV columns
            # ---------------------------------------------------------

            # Normalize headers for case-insensitive matching
            lower_map = {col.lower(): col for col in columns}

            # Auto-ignore "SMILES" or "smiles"
            if "smiles" in lower_map:
                smiles_col = lower_map["smiles"]

                # Move SMILES from available_list → ignore_list
                items = self.available_list.findItems(smiles_col, Qt.MatchExactly)
                for item in items:
                    row = self.available_list.row(item)
                    self.available_list.takeItem(row)
                    self.ignore_list.addItem(smiles_col)

            # Auto-select "code_name" as the names field
            if "code_name" in columns:
                self.names_dropdown.setCurrentText("code_name")

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

    def check_atomic_descriptors(self, context: str) -> bool:
        """
        Warn the user if atomic descriptors will not be generated.

        context:
            "AQME"   -> warn if no atoms are selected
            "ROBERT" -> warn only if AQME workflow is enabled AND no atoms are selected
        """

        # -----------------------------------------
        # Check if any atoms are selected
        # -----------------------------------------
        has_graphical_atoms = bool(self.tab_widget_aqme.selected_atoms)

        atoms_text = self.tab_widget_aqme.atoms.text().strip()
        has_manual_atoms = bool(atoms_text)

        no_atoms_selected = not (has_graphical_atoms or has_manual_atoms)

        # -----------------------------------------
        # Context-specific logic
        # -----------------------------------------
        if context == "ROBERT" and not self.aqme_workflow.isChecked():
            return True  # AQME not enabled → no warning needed

        if not no_atoms_selected:
            return True  # Atomic descriptors will be generated

        # -----------------------------------------
        # Build message
        # -----------------------------------------
        if context == "AQME":
            title = "Atomic descriptors not selected"
            message = (
                "AQME will run using molecular (global) descriptors.\n\n"
                "If specific atoms are selected in the AQME tab, AQME can also include "
                "atomic (local) descriptors, which may add chemically meaningful "
                "information in some cases.\n\n"
                "No atoms are currently selected. This is not an error.\n"
                "Atom selection may not always be available depending on the structures.\n\n"
                "To access the AQME tab and enable atom selection, check "
                "\"Enable AQME Workflow\".\n\n"
                "You can safely continue with the current setup, or go back and select "
                "atoms if that option is available and relevant.\n\n"
                "Do you want to continue?"
            )

        else:  # ROBERT
            title = "Atomic descriptors not selected"
            message = (
                "ROBERT will run using molecular (global) descriptors generated by AQME.\n\n"
                "If specific atoms are selected in the AQME tab, AQME can also include "
                "atomic (local) descriptors, which may add chemically meaningful "
                "information in some cases.\n\n"
                "No atoms are currently selected. This is not an error.\n"
                "Atom selection may not always be available depending on the structures.\n\n"
                "You can safely continue with the current setup, or go back and select "
                "atoms if that option is available and relevant.\n\n"
                "Do you want to continue?"
            )

        reply = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        return reply == QMessageBox.Yes
    
    def handle_predict_with_test(self, run_dir):
        """Preflight checks for PREDICT with test CSV."""

        # ---------------------------------------------
        # 1. Ask user how to handle descriptors
        # ---------------------------------------------
        choice = self._ask_descriptor_strategy()

        if choice == "cancel":
            return "abort"

        if choice == "existing":
            # Do nothing special, continue normal flow
            return "continue"

        if choice == "aqme":
            # Need AQME before ROBERT
            return self._run_aqme_for_test_descriptors(run_dir)

        return "abort"

    def _ask_descriptor_strategy(self):
        """Asks the user how to handle descriptors."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Descriptor handling")
        msg.setText(
            "Prediction on a test CSV requires using the same descriptors "
            "that were used to train the model.\n\n"
            "Select AQME if the model was trained using AQME-generated descriptors. "
            "This will generate the same descriptors for the test set.\n\n"
            "Select the second option only if your test CSV already contains "
            "the required descriptors generated without AQME."
        )
        aqme_btn = msg.addButton(
            "Generate descriptors with AQME",
            QMessageBox.AcceptRole
        )
        existing_btn = msg.addButton(
            "Descriptors already present",
            QMessageBox.AcceptRole
        )
        cancel_btn = msg.addButton(
            "Cancel",
            QMessageBox.RejectRole
        )

        msg.exec()

        clicked = msg.clickedButton()

        if clicked == aqme_btn:
            return "aqme"

        if clicked == existing_btn:
            return "existing"

        return "cancel"
    
    def _run_aqme_for_test_descriptors(self, run_dir):
        """
        Automatically detects how AQME descriptors were generated
        and prints the AQME command that would be launched for the test CSV.
        """

        aqme_dat = os.path.join(run_dir, "AQME", "AQME_data.dat")
        qdescp_dat = os.path.join(run_dir, "QDESCP_data.dat")

        # ==================================================
        # Scenario 1: AQME-ROBERT (integrated workflow)
        # ==================================================
        if os.path.exists(aqme_dat):

            robert_cmd = self._read_command_from_dat(
                aqme_dat,
                expected_prefix="Command line used in ROBERT"
            )

            # Extract --qdescp_keywords "..." if present
            qdescp_keywords = None
            match = re.search(r'--qdescp_keywords\s+"([^"]+)"', robert_cmd)
            if match:
                qdescp_keywords = match.group(1)

            aqme_cmd = self._build_test_aqme_command(
                qdescp_keywords=qdescp_keywords
            )

            return aqme_cmd

        # ==================================================
        # Scenario 2: AQME -> ROBERT (separate runs)
        # ==================================================
        if os.path.exists(qdescp_dat):

            aqme_cmd_original = self._read_command_from_dat(
                qdescp_dat,
                expected_prefix="Command line used in AQME"
            )

            aqme_cmd = self._build_test_aqme_command(
                original_command=aqme_cmd_original
            )

            return aqme_cmd

        # ==================================================
        # No valid AQME metadata found
        # ==================================================
        QMessageBox.warning(
            self,
            "AQME information not found",
            "Could not automatically determine how descriptors were generated.\n\n"
            "No AQME metadata files were found for this model."
        )

        return None
    
    def _read_command_from_dat(self, dat_path, expected_prefix):
        """
        Reads a .dat file and extracts the command line
        matching the expected prefix.
        """

        with open(dat_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(expected_prefix):
                    return line.split(":", 1)[1].strip()

        raise RuntimeError(
            f"Expected command line not found in {dat_path}"
    )

    def _build_test_aqme_command(self, original_command=None, qdescp_keywords=None):
        """Builds an AQME command for generating descriptors for the test CSV."""

        python_pointer = "python"
        if getattr(sys, "frozen", False):
            env = Path.cwd() / "_internal" / "robert_env"
            if sys.platform == "win32":
                python_pointer = env / "python.exe"
            elif sys.platform == "darwin":
                python_pointer = env / "bin" / "python3"
            else:
                python_pointer = env / "bin" / "python"

        test_csv = os.path.basename(self.csv_test_path)

        # ==================================================
        # AQME -> ROBERT: reuse original AQME command
        # ==================================================
        if original_command:
            # 1. Remove any leading python executable
            #    Keep everything from "-m aqme" onwards
            match = re.search(r'(-m\s+aqme.*)', original_command)
            if not match:
                raise RuntimeError("Could not locate '-m aqme' in AQME command")

            aqme_args = match.group(1)

            # 2. Replace CSV references
            aqme_args = re.sub(
                r'--input\s+"[^"]+"',
                f'--input "{test_csv}"',
                aqme_args
            )
            aqme_args = re.sub(
                r'--csv_name\s+"[^"]+"',
                f'--csv_name "{test_csv}"',
                aqme_args
            )

            # 3. Rebuild command with correct python
            cmd = f'"{python_pointer}" -u {aqme_args}'
            return cmd

        # ==================================================
        # AQME-ROBERT: build fresh AQME command
        # ==================================================
        cmd = (
            f'"{python_pointer}" -u -m aqme --qdescp '
            f'--input "{test_csv}" '
            f'--program xtb '
            f'--csv_name "{test_csv}" '
            f'--robert'
        )

        if qdescp_keywords:
            cmd += f' {qdescp_keywords}'

        return cmd
    
    def run_test_aqme(self, aqme_command, run_dir):
        """Launches an AQME worker for generating descriptors for the test CSV."""
        
        self.console_output.append(
            "<b><span style='color:cyan;'>Running AQME...</span></b><br>"
        )
        self.progress.setRange(0, 0)

        self.current_process = "AQME"
        self.aqme_role = "test" 

        self.worker = RobertWorker(aqme_command, run_dir)
        self.worker.output_received.connect(self.console_output.append)
        self.worker.error_received.connect(self.console_output.append)
        self.worker.process_finished.connect(self.on_process_finished)
        self.worker.start()

    def _detect_aqme_output_csv(self):
        """
        Detects the AQME-generated CSV for ROBERT prediction.
        Expected format:
            AQME-ROBERT_full_<original_csv_name>.csv
        """

        # Directory where AQME was run
        run_dir = os.path.dirname(self.csv_test_path)

        # Original CSV name without extension)
        original_name = os.path.splitext(
            os.path.basename(self.csv_test_path)
        )[0]

        # Expected output CSV
        expected_csv = f"AQME-ROBERT_full_{original_name}.csv"
        expected_path = os.path.join(run_dir, expected_csv)

        if os.path.exists(expected_path):
            return expected_path

        return None
    
    def _write_atom_mapping_dat(
        self,
        smarts: str,
        selected_atoms: list,
        run_dir: str,
        filename: str = "AtomMapping_data.dat"
    ):
        """
        Write an atomic mapping contract to disk.

        """

        if not smarts or not selected_atoms:
            return

        os.makedirs(run_dir, exist_ok=True)
        dat_path = os.path.join(run_dir, filename)

        try:
            pattern_mol = Chem.MolFromSmarts(smarts)
            if pattern_mol is None:
                raise ValueError("Invalid SMARTS pattern")

            pattern_atoms = pattern_mol.GetNumAtoms()

            with open(dat_path, "w", encoding="utf-8") as f:
                # --------------------------------------------------
                # Header
                # --------------------------------------------------
                f.write("EasyROB — Standalone GUI\n\n")

                # --------------------------------------------------
                # SMARTS pattern
                # --------------------------------------------------
                f.write("-" * 60 + "\n")
                f.write("SMARTS pattern\n")
                f.write("-" * 60 + "\n")
                f.write(f"{smarts}\n\n")
                f.write(f"Pattern atoms: {pattern_atoms}\n\n")

                # --------------------------------------------------
                # Atom mapping
                # --------------------------------------------------
                f.write("-" * 60 + "\n")
                f.write("Atom mapping (pattern index → atomMap number)\n")
                f.write("-" * 60 + "\n\n")

                for i, pattern_idx in enumerate(selected_atoms):
                    atom = pattern_mol.GetAtomWithIdx(pattern_idx)
                    element = atom.GetSymbol()
                    atom_map_number = i + 1

                    f.write(
                        f"   o  Pattern atom {pattern_idx:<3} → "
                        f"atomMap {atom_map_number:<2}  (Element: {element})\n"
                    )

                f.write("\n")

                # --------------------------------------------------
                # Notes
                # --------------------------------------------------
                f.write("-" * 60 + "\n")
                f.write("Notes\n")
                f.write("-" * 60 + "\n")
                f.write(
                    "- Pattern indices refer to atom positions in the SMARTS pattern.\n"
                    "- atomMap numbers are fixed and define descriptor ordering.\n"
                    "- This contract is only applied when a unique SMARTS match is found.\n"
                    "- If multiple matches or ambiguities are detected, mapping is ignored.\n\n"
                )

        except Exception as e:
            self.console_output.append(
                f"<span style='color:orange;'>WARNING: Failed to write atom mapping dat: {e}</span>"
            )
    def _check_generate_folder(self, run_dir):
        """Checks if a GENERATE folder exists in the run directory."""
        
        generate_dir = os.path.join(run_dir, "GENERATE")

        if not os.path.exists(generate_dir):
            QMessageBox.warning(
                self,
                "Trained model not found",
                "No trained model was found in this folder.\n\n"
                "Prediction with a test CSV requires an existing model "
                "generated in a previous run (GENERATE step).\n"
            )
            return False

        return True
    
    def _validate_robert_workflow(self):
        """
        Validates workflow state and resolves mismatches.
        Returns True if execution can continue, False otherwise.
        """

        workflow = self.workflow_selector.currentText()

        # ---------------------------------------------------
        # Detect mismatch: test CSV loaded but not PREDICT
        # ---------------------------------------------------
        if self.csv_test_path and not self.file_path and workflow not in ["PREDICT", "REPORT"]:
            reply = QMessageBox.question(
                self,
                "Possible workflow mismatch",
                "You loaded a test CSV but selected 'Full Workflow'.\n\n"
                "Did you mean to generate predictions instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                self.workflow_selector.setCurrentText("PREDICT")
                workflow = "PREDICT"
            else:
                QMessageBox.information(
                    self,
                    "Execution stopped",
                    "To run 'Full Workflow' in ROBERT, please load the training CSV "
                    "and select the appropriate target and name columns."
                )
                return False

        # ---------------------------------------------------
        # Full Workflow validation
        # ---------------------------------------------------
        if workflow == "Full Workflow":
            if not self.file_path:
                QMessageBox.warning(
                    self,
                    "WARNING!",
                    "Please load a training CSV file before running the workflow."
                )
                return False

        # ---------------------------------------------------
        # PREDICT validation
        # ---------------------------------------------------
        if workflow == "PREDICT":

            if not self.csv_test_path:
                QMessageBox.warning(
                    self,
                    "WARNING!",
                    "Please select a test CSV file for prediction."
                )
                return False

            run_dir = os.path.dirname(self.csv_test_path)

            if not self._check_generate_folder(run_dir):
                return False

        # ---------------------------------------------------
        # REPORT validation
        # ---------------------------------------------------
        if workflow == "REPORT":

            if not self.file_path and not self.csv_test_path:
                QMessageBox.warning(
                    self,
                    "WARNING!",
                    "Please load a CSV file to determine the report directory."
                )
                return False

        return True

    def run_robert(self):
        """Runs the ROBERT workflow with the selected parameters."""

        # --------------------------------------------------
        # Validate workflow
        # --------------------------------------------------
        if not self._validate_robert_workflow():
            return
        
        # --------------------------------------------------
        # Init process
        # --------------------------------------------------
        self.current_process = "ROBERT"

        self.run_button.setDisabled(True)
        self.run_aqme_button.setDisabled(True)
        self.stop_button.setDisabled(False)

        self.console_output.clear()
        self.console_output.setHtml(
            "<pre style='color:white; background-color:black; font-family:monospace;'></pre>"
        )

        # Path to run directory 
        if self.file_path:
            run_dir = os.path.dirname(self.file_path)
        elif self.csv_test_path:
            run_dir = os.path.dirname(self.csv_test_path)

        # --------------------------------------------------
        # Cleanup folders (ROBERT / AQME aware)
        # --------------------------------------------------
        folders_to_check = ["CURATE", "GENERATE", "PREDICT", "VERIFY", "AQME"]

        if self.aqme_workflow.isChecked():
            folders_to_check.extend(["CSEARCH", "QDESCP"])

        existing_folders = [
            f for f in folders_to_check
            if os.path.exists(os.path.join(run_dir, f))
        ]

        if existing_folders and self.workflow_selector.currentText() == "Full Workflow":
            confirmation = QMessageBox.question(
                self,
                "WARNING!",
                "ROBERT detected folders from a previous run.\n\n"
                "These folders may cause problems if the previous run was interrupted,\n"
                "or will be overwritten if the previous run completed successfully.\n\n"
                "Are you sure you want to continue and delete them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if confirmation == QMessageBox.No:
                self._reset_ui_after_process()
                return

            for folder in existing_folders:
                try:
                    shutil.rmtree(os.path.join(run_dir, folder))
                except Exception as e:
                    self.console_output.append(
                        f"[ERROR] Could not delete folder '{folder}': {e}"
                    )
                    self._reset_ui_after_process()
                    return   
      
        # --------------------------------------------------
        # Collect GUI values
        # --------------------------------------------------
        self._collect_robert_gui_values()

        if not self.check_variables_robert():
            self._reset_ui_after_process()
            self.console_output.append(
                "WARNING! Invalid parameters. Please fix them before running."
            )
            return
        
        # Rename pdf if full workflow or report selected
        wf_predict = self.workflow_selector.currentText()
        if wf_predict == "Full Workflow" or wf_predict == "REPORT":
            self.rename_existing_pdf("ROBERT_report.pdf", run_dir)

        # ==================================================
        # Cache mapped CSVs (TRAIN + TEST)
        # ==================================================
        self.mapped_train_csv = None
        self.mapped_test_csv = None
        is_robert_mapped = False

        selected_atoms_for_robert = []

        if (
            hasattr(self.tab_widget_aqme, "selected_atoms")
            and self.tab_widget_aqme.selected_atoms
        ):
            selected_atoms_for_robert = list(self.tab_widget_aqme.selected_atoms)
            smarts = self.tab_widget_aqme.smarts_targets[0]

            train_source_csv = self._get_unmapped_csv(self.file_path)

            self.mapped_train_csv = self.tab_widget_aqme.generate_mapped_csv_from_smiles(
                train_source_csv,
                smarts,
                selected_atoms_for_robert
            )

            is_robert_mapped = True

            if getattr(self, "csv_test_path", None):
                test_source_csv = self._get_unmapped_csv(self.csv_test_path)

                self.mapped_test_csv = self.tab_widget_aqme.generate_mapped_csv_from_smiles(
                    test_source_csv,
                    smarts,
                    selected_atoms_for_robert
                )

            #  Save atomic mapping contract in .dat
            run_dir = os.path.dirname(self.file_path)
            self._write_atom_mapping_dat(
                smarts=smarts,
                selected_atoms=selected_atoms_for_robert,
                run_dir=run_dir
            )

        # --------------------------------------------------
        # Decide REAL input CSVs for ROBERT
        # --------------------------------------------------
        selected_file_path = (
            self.mapped_train_csv
            if is_robert_mapped and self.mapped_train_csv
            else self.file_path
        )

        self.csv_test_path = (
            self.mapped_test_csv
            if is_robert_mapped and self.mapped_test_csv
            else getattr(self, "csv_test_path", None)
        ) 

        # --------------------------------------------------------------------------
        # AQME-origin CSV check, disable AQME workflow if detected previously runned
        # --------------------------------------------------------------------------
        csv_name = os.path.basename(selected_file_path)
        comes_from_aqme = csv_name.startswith("AQME-ROBERT_")

        if comes_from_aqme and self.aqme_workflow.isChecked():
            self.aqme_workflow.setChecked(False)

        # ----------------------------------------------------------
        # AQME check, show warning if no atomic descriptors selected
        # ----------------------------------------------------------
        if not comes_from_aqme and wf_predict != "PREDICT" and wf_predict != "ROBERT":
            if not self.check_atomic_descriptors("ROBERT"):
                self._reset_ui_after_process()
                return
            
        # --------------------------------------------------
        # PREDICT + csv_test preflight
        # --------------------------------------------------
        if (
            self.workflow_selector.currentText() == "PREDICT"
            and self.csv_test_path
        ):
            run_dir = os.path.dirname(self.csv_test_path)

            # -----------------------------------------------
            # Step 1: Detect atomic mapping contract (.dat)
            # -----------------------------------------------
            dat_path = os.path.join(run_dir, "AtomMapping_data.dat")

            if os.path.isfile(dat_path):

                self.console_output.append(
                    f"[INFO] Atomic mapping contract detected: {dat_path}"
                )

                try:
                    # Read contract
                    contract = self._read_atom_mapping_dat(dat_path)

                    # Validate + Apply in one step
                    new_mapped_csv = self._apply_mapping_smarts(
                        self.csv_test_path,
                        contract
                    )

                    self.console_output.append(
                        f"[INFO] Generated mapped test CSV: {new_mapped_csv}"
                    )

                    # Save original test CSV only 
                    if not getattr(self, "_original_test_csv_path", None):
                        self._original_test_csv_path = self.csv_test_path

                    # Replace test CSV with mapped version
                    self.csv_test_path = new_mapped_csv

                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Atomic mapping mismatch",
                        f"{e}\n\n"
                        "Prediction cannot continue because atomic descriptors "
                        "would not be consistent with the trained model."
                    )
                    self._reset_ui_after_process()
                    return

            # --------------------------------------------------
            # Preflight checks for PREDICT with test CSV
            # including potential AQME descriptor generation
            # --------------------------------------------------
            result = self.handle_predict_with_test(run_dir)

            if result == "abort":
                self._reset_ui_after_process()
                return

            if result not in ("continue", "abort"):
                # AQME command → launch AQME first
                # Then run ROBERT in "on_process_finished"
                self.run_test_aqme(result, run_dir)
                return

        # --------------------------------------------------
        # Build and launch ROBERT
        # --------------------------------------------------
        command = self.build_robert_command(selected_file_path)

        self.console_output.append(
            "<b><span style='color:purple;'>Running ROBERT...</span></b><br>"
        )
        self.progress.setRange(0, 0)

        # Determine run directory train or test CSV
        if selected_file_path:
            run_dir = os.path.dirname(selected_file_path)
        elif self.csv_test_path:
            run_dir = os.path.dirname(self.csv_test_path)
     
        self.worker = RobertWorker(command, run_dir)
        self.worker.output_received.connect(self.console_output.append)
        self.worker.error_received.connect(self.console_output.append)
        self.worker.process_finished.connect(self.on_process_finished)
        self.worker.start()
    
    def _read_atom_mapping_dat(self, dat_path):
        """
        Read atomic mapping contract from .dat file.

        Returns:
            dict with keys:
                - smarts (str)
                - pattern_atoms (int)
                - mapping (list of dicts)
        Raises:
            ValueError if parsing fails.
        """

        if not os.path.isfile(dat_path):
            raise FileNotFoundError("AtomMapping_data.dat not found")

        with open(dat_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        smarts = None
        pattern_atoms = None
        mapping = []

        for line in lines:

            # SMARTS line
            if line.startswith("SMARTS pattern"):
                continue  # skip header line

            if smarts is None and not line.startswith("-") and "Pattern atoms" not in line and "Pattern atom" not in line:
                # First non-header SMARTS candidate 
                if "[" in line or "#" in line:
                    smarts = line

            # Pattern atoms
            if line.startswith("Pattern atoms:"):
                pattern_atoms = int(line.split(":")[1].strip())

            # Mapping lines
            if line.startswith("o  Pattern atom") or "Pattern atom" in line:
                # Extract using regex
                match = re.search(
                    r"Pattern atom\s+(\d+)\s+→\s+atomMap\s+(\d+)\s+\(Element:\s+(\w+)\)",
                    line
                )
                if match:
                    pattern_idx = int(match.group(1))
                    map_num = int(match.group(2))
                    element = match.group(3)

                    mapping.append({
                        "pattern_idx": pattern_idx,
                        "map_num": map_num,
                        "element": element
                    })

        if smarts is None or pattern_atoms is None or not mapping:
            raise ValueError("Invalid atom_mapping.dat format")
        
        return {
            "smarts": smarts,
            "pattern_atoms": pattern_atoms,
            "mapping": mapping
        }
    
    def _apply_mapping_smarts(self, csv_path, contract):
        """
        Validate and apply atomic mapping contract to CSV.
        If validation fails, raises ValueError.
        If already correctly mapped, returns original path.
        Otherwise generates *_mapped.csv and returns its path.
        """

        df = smart_read_csv(csv_path)

        smiles_col = next(
            (c for c in df.columns if c.lower() == "smiles"),
            None
        )

        if smiles_col is None:
            raise ValueError("CSV has no SMILES column")

        smarts = contract["smarts"]
        mapping = contract["mapping"]
        expected_pattern_atoms = contract["pattern_atoms"]

        pattern_mol = Chem.MolFromSmarts(smarts)

        if pattern_mol is None:
            raise ValueError("Invalid SMARTS in contract")

        if pattern_mol.GetNumAtoms() != expected_pattern_atoms:
            raise ValueError("SMARTS atom count mismatch with contract")
        
        # --------------------------------------------------
        # Step 1: Detect if already mapped correctly
        # --------------------------------------------------
        if csv_path.endswith("_mapped.csv"):
            return csv_path

        # --------------------------------------------------
        # Step 2: Apply mapping
        # --------------------------------------------------
        mapped_smiles = []

        for smiles in df[smiles_col].dropna().astype(str):

            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            matches = mol.GetSubstructMatches(pattern_mol)

            if len(matches) != 1:
                raise ValueError(
                    f"SMARTS does not produce unique match for molecule: {smiles}"
                )

            match = matches[0]

            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

            for entry in mapping:
                pattern_idx = entry["pattern_idx"]
                map_num = entry["map_num"]

                mol.GetAtomWithIdx(match[pattern_idx]).SetAtomMapNum(map_num)

            mapped_smiles.append(Chem.MolToSmiles(mol))

        df_out = df.copy()
        df_out[smiles_col] = mapped_smiles

        base, _ = os.path.splitext(csv_path)
        out_csv = f"{base}_mapped.csv"

        df_out.to_csv(out_csv, index=False)

        return out_csv

    def build_robert_command(self, selected_file_path):
        """Builds the ROBERT command based on GUI selections."""
        python_pointer = "python"

        # --------------------------------------------------
        # Detect embedded Python (frozen app)
        # --------------------------------------------------
        if getattr(sys, "frozen", False):
            embedded_env = Path.cwd() / "_internal" / "robert_env"
            if sys.platform == "win32":
                python_pointer = embedded_env / "python.exe"
            elif sys.platform == "darwin":
                python_pointer = embedded_env / "bin" / "python3"
            else:
                python_pointer = embedded_env / "bin" / "python"

        wf = self.workflow_selector.currentText()

        # ==================================================
        # REPORT (standalone, no CSV, no params)
        # ==================================================
        if wf == "REPORT":
            return f'"{python_pointer}" -u -m robert --report'

        # ==================================================
        # PREDICT (uses existing model, no training params)
        # ==================================================
        if wf == "PREDICT":
            command = f'"{python_pointer}" -u -m robert --predict'

            # Add csv_name ONLY if it resolves to a non-empty basename
            if selected_file_path:
                csv_name = os.path.basename(selected_file_path)
                if csv_name:
                    command += f' --csv_name "{csv_name}"'

            # Add csv_test ONLY if it resolves to a non-empty basename
            if self.csv_test_path:
                csv_test = os.path.basename(self.csv_test_path)
                if csv_test:
                    command += f' --csv_test "{csv_test}"'

            return command
        
        # ==================================================
        # NORMAL WORKFLOW (CURATE / GENERATE / VERIFY)
        # ==================================================
        command = (
            f'"{python_pointer}" -u -m robert '
            f'--csv_name "{os.path.basename(selected_file_path)}" '
            f'--y "{self.y_dropdown.currentText()}" '
            f'--names "{self.names_dropdown.currentText()}"'
        )

        # ---------- TEST CSV ----------
        if self.csv_test_path:
            command += f' --csv_test "{os.path.basename(self.csv_test_path)}"'

        # ---------- TYPE ----------
        if self.type_dropdown.currentText() == "Classification":
            command += ' --type "clas"'

        # ---------- IGNORE COLUMNS ----------
        selected_columns = [
            self.ignore_list.item(i).text()
            for i in range(self.ignore_list.count())
        ]
        if selected_columns:
            formatted_columns = [f"'{col}'" for col in selected_columns]
            command += f' --ignore "[{", ".join(formatted_columns)}]"'

        # ---------- WORKFLOW FLAG ----------
        workflow_map = {
            "CURATE": "--curate",
            "GENERATE": "--generate",
            "VERIFY": "--verify",
        }
        if wf in workflow_map:
            command += f" {workflow_map[wf]}"

        # ---------- GENERAL ----------
        if not self.auto_type_value:
            command += " --auto_type False"

        if self.seed_value:
            command += f' --seed {self.seed_value}'

        if self.kfold_value:
            command += f' --kfold {self.kfold_value}'

        if self.repeat_kfolds_value:
            command += f' --repeat_kfolds {self.repeat_kfolds_value}'

        if self.split_value != "even":
            command += f' --split {self.split_value.lower()}'

        # ---------- AQME ----------
        if self.aqme_workflow.isChecked():
            command += ' --aqme'
            command += f' --descp_lvl {self.descriptor_level_selected}'

            atoms_entries = []

            if self.tab_widget_aqme.selected_atoms:
                atoms_entries.extend(
                    range(1, len(self.tab_widget_aqme.selected_atoms) + 1)
                )

            atoms_text = self.atoms_selected
            if atoms_text:
                atoms_entries.extend(
                    [a.strip() for a in atoms_text.split(",") if a.strip()]
                )

            if atoms_entries:
                atoms_str = "[" + ",".join(str(e) for e in atoms_entries) + "]"
                command += f' --qdescp_keywords "--qdescp_atoms {atoms_str}"'

            if self.solvent_selected != "None":
                command += (
                    f' --qdescp_keywords "--qdescp_solvent {self.solvent_selected}"'
                )

        # ---------- CURATE ----------
        if self.categorical_value != "onehot":
            command += f' --categorical {self.categorical_value}'

        if not self.corr_filter_x_value:
            command += ' --corr_filter_x False'

        if self.corr_filter_y_value:
            command += ' --corr_filter_y True'

        if self.desc_thres_value:
            command += f' --desc_thres {self.desc_thres_value}'

        if self.thres_x_value:
            command += f' --thres_x {self.thres_x_value}'

        if self.thres_y_value:
            command += f' --thres_y {self.thres_y_value}'

        # ---------- GENERATE ----------
        if self.selected_models != self.default_models:
            model_list = "[" + ",".join(
                f"'{m}'" for m in sorted(self.selected_models)
            ) + "]"
            command += f' --model "{model_list}"'

        if self.error_type_value != self.default_error_type:
            command += f' --error_type {self.error_type_value}'

        if self.init_points_value:
            command += f' --init_points {self.init_points_value}'

        if self.n_iter_value:
            command += f' --n_iter {self.n_iter_value}'

        if not self.pfi_filter_value:
            command += " --pfi_filter False"

        if self.pfi_epochs_value:
            command += f' --pfi_epochs {self.pfi_epochs_value}'

        if self.pfi_threshold_value:
            command += f' --pfi_threshold {self.pfi_threshold_value}'

        if self.pfi_max_value:
            command += f' --pfi_max {self.pfi_max_value}'

        if not self.auto_test_value:
            command += " --auto_test False"

        if self.test_set_value:
            command += f' --test_set {self.test_set_value}'

        # ---------- PREDICT OPTIONS (shared flags) ----------
        if self.t_value:
            command += f' --t_value {self.t_value}'

        if self.shap_show:
            command += f' --shap_show {self.shap_show}'

        if self.pfi_show:
            command += f' --pfi_show {self.pfi_show}'

        return command

    def _collect_robert_gui_values(self):
        """Collects and stores all GUI values needed to run ROBERT."""

        # ---------- GENERAL ----------
        self.seed_value = self.options_tab.seed.text().strip()
        self.kfold_value = self.options_tab.kfold.text().strip()
        self.repeat_kfolds_value = self.options_tab.repeat_kfolds.text().strip()
        self.auto_type_value = self.options_tab.auto_type.isChecked()
        self.split_value = self.options_tab.split.currentText().strip()

        # ---------- AQME ----------
        self.descriptor_level_selected = self.tab_widget_aqme.descriptor_level.currentText()
        self.atoms_selected = self.tab_widget_aqme.atoms.text().strip()
        self.solvent_selected = self.tab_widget_aqme.solvent.currentText()

        # ---------- CURATE ----------
        self.categorical_value = self.options_tab.categoricalstr.currentText().strip()
        self.corr_filter_x_value = self.options_tab.corr_filter_xbool.isChecked()
        self.corr_filter_y_value = self.options_tab.corr_filter_ybool.isChecked()
        self.desc_thres_value = self.options_tab.desc_thresfloat.text().strip()
        self.thres_x_value = self.options_tab.thres_xfloat.text().strip()
        self.thres_y_value = self.options_tab.thres_yfloat.text().strip()

        # ---------- GENERATE ----------
        type_mode = self.type_dropdown.currentText()

        self.default_models = (
            {"RF", "GB", "NN", "MVL"} if type_mode == "Regression"
            else {"RF", "GB", "NN", "AdaB"}
        )

        self.default_error_type = (
            "rmse" if type_mode == "Regression" else "mcc"
        )
        self.selected_models = {
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

        # ---------- PREDICT ----------
        self.t_value = self.options_tab.t_value.text().strip()
        self.shap_show = self.options_tab.shap_show.text().strip()
        self.pfi_show = self.options_tab.pfi_show.text().strip()

    def _ensure_test_name_column(self):
        """
        Ensures that the selected name column exists in the test CSV.
        If missing, tries to map from 'code_name'.
        Returns True if OK, False if blocking.
        """

        if not self.csv_test_path:
            # No test loaded → nothing to validate
            return True

        name_col = self.names_dropdown.currentText()

        try:
            df_test = pd.read_csv(self.csv_test_path)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Test dataset error",
                f"Could not read test CSV file:\n\n{e}"
            )
            return False

        # -----------------------------
        # Case 1: column already exists
        # -----------------------------
        if name_col in df_test.columns:
            return True

        # -----------------------------
        # Case 2: try code_name fallback
        # -----------------------------
        if "code_name" in df_test.columns:
            df_test = df_test.rename(columns={"code_name": name_col})

            try:
                df_test.to_csv(self.csv_test_path, index=False)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Test dataset error",
                    f"Could not update test CSV file:\n\n{e}"
                )
                return False

            # Reload test CSV into GUI / state
            self.set_csv_test_path(self.csv_test_path)

            return True

        # -----------------------------
        # Case 3: hard fail
        # -----------------------------
        QMessageBox.warning(
            self,
            "Incompatible test dataset",
            f"The selected name column '{name_col}' is not present in the test dataset.\n\n"
            "The test CSV does not contain this column, nor a fallback 'code_name' column.\n\n"
            "Please select a compatible test dataset or change the name column."
        )

        return False

    def check_variables_robert(self):
        """Validates the values extracted from the Advanced Options tab."""
        errors = []

        # predict workflow selection
        workflow = self.workflow_selector.currentText()
        is_predict = workflow == "PREDICT"
        is_report = workflow == "REPORT"

        # ------------------------
        # PREDICT: minimal required input
        # ------------------------
        if is_predict and not self.file_path and not self.csv_test_path:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Predict requires at least one CSV file."
            )
            return False

        # -----------------------------------------
        # Basic checks (skip in PREDICT and REPORT)
        # -----------------------------------------
        if not is_predict and not is_report:
            if self.names_dropdown.currentText() == self.y_dropdown.currentText():
                QMessageBox.warning(
                    self,
                    "Invalid Selection",
                    "The name column and the target value column cannot be the same. Please select different columns."
                )
                return False

        # --------------------------------------------------------
        # Test name column in TEST CSV (skip in PREDICT and REPORT)
        # -------------------------------------------------------
        if not is_predict and not is_report:
            if not self._ensure_test_name_column():
                return False

        # ------------------------
        # GENERAL
        # ------------------------
        if self.seed_value and not self.seed_value.isdigit():
            errors.append("Seed must be an integer.")

        if self.kfold_value and not self.kfold_value.isdigit():
            errors.append("kfold must be an integer.")

        if self.repeat_kfolds_value and not self.repeat_kfolds_value.isdigit():
            errors.append("repeat_kfolds must be an integer.")

        # ------------------------
        # AQME (skip in PREDICT)
        # ------------------------
        if self.aqme_workflow.isChecked() and not is_predict:

            total_columns = []
            total_columns += [self.available_list.item(i).text() for i in range(self.available_list.count())]
            total_columns += [self.ignore_list.item(i).text() for i in range(self.ignore_list.count())]
            lowercase_columns = [col.lower() for col in total_columns]

            if not any(col.startswith("smiles") for col in lowercase_columns):
                errors.append(
                    "The CSV file does not contain a column with the name 'smiles' or a column starting with "
                    "'smiles_'. Please make sure the column exists."
                )

        # ------------------------
        # CURATE (skip in PREDICT)
        # ------------------------
        if not is_predict:
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

        # ------------------------
        # GENERATE (skip in PREDICT)
        # ------------------------
        if not is_predict:
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

        # ------------------------
        # PREDICT
        # ------------------------
        if self.t_value and not self.t_value.isdigit():
            errors.append("t_value must be an integer.")

        if self.shap_show and not self.shap_show.isdigit():
            errors.append("shap_show must be an integer.")

        if self.pfi_show and not self.pfi_show.isdigit():
            errors.append("pfi_show must be an integer.")

        # ------------------------
        # Final error report
        # ------------------------
        if errors:
            QMessageBox.warning(self, "Invalid Parameters", "\n".join(errors))
            return False

        return True
    
    def build_aqme_command(self, selected_file_path, selected_atoms_override=None):
        """Builds the AQME command based on the GUI selections."""

        python_pointer = "python"

        if getattr(sys, "frozen", False):
            embeded_env = Path.cwd() / "_internal" / "robert_env"
            if sys.platform == "win32":
                python_pointer = embeded_env / "python.exe"
            elif sys.platform == "darwin":
                python_pointer = embeded_env / "bin" / "python3"
            else:
                python_pointer = embeded_env / "bin" / "python"

        csv_name = os.path.basename(selected_file_path)

        command = (
            f'"{python_pointer}" -u -m aqme --qdescp '
            f'--input "{csv_name}" '
            f'--program xtb '
            f'--csv_name "{csv_name}" '
            f'--robert'
        )

        # ------------------------
        # qdescp_atoms
        # ------------------------
        atoms_entries = []

        atoms_source = (
            selected_atoms_override
            if selected_atoms_override is not None
            else getattr(self.tab_widget_aqme, "selected_atoms", [])
        )

        if atoms_source:
            atoms_entries.extend(range(1, len(atoms_source) + 1))

        atoms_text = self.tab_widget_aqme.atoms.text().strip()
        if atoms_text:
            atoms_entries.extend(
                [a.strip() for a in atoms_text.split(",") if a.strip()]
            )

        if atoms_entries:
            atoms_str = "[" + ",".join(str(a) for a in atoms_entries) + "]"
            command += f' --qdescp_atoms "{atoms_str}"'

        solvent = self.tab_widget_aqme.solvent.currentText()
        if solvent != "None":
            command += f' --qdescp_solvent "{solvent}"'

        return command, os.path.dirname(selected_file_path)

    def run_aqme(self):
        """Runs the AQME descriptor generation workflow."""

        # ----------------
        # Init AQME state
        # ----------------
        self.aqme_runs = []
        self.aqme_command_queue = []
        self.aqme_is_running = True
        self.is_aqme_mapped = False
        self.mapped_train_csv = None
        self.mapped_test_csv = None

        if not self.file_path:
            QMessageBox.warning(self, "WARNING!", "Please select a CSV file.")
            return

        if not self.check_atomic_descriptors("AQME"):
            return

        self.current_process = "AQME"

        self.run_button.setDisabled(True)
        self.run_aqme_button.setDisabled(True)
        self.stop_button.setDisabled(False)

        self.console_output.clear()
        self.console_output.setHtml(
            "<pre style='color:white; background-color:black; font-family:monospace;'></pre>"
        )

        run_dir = os.path.dirname(self.file_path)

        # ---------------------------
        # Warn if AQME folders exist
        # ---------------------------
        folders_to_check = ["AQME", "CSEARCH", "QDESCP", "AQME_RUNS"]
        existing_folders = [
            f for f in folders_to_check
            if os.path.exists(os.path.join(run_dir, f))
        ]

        if existing_folders:
            confirmation = QMessageBox.question(
                self,
                "AQME folders detected",
                "Folders from a previous AQME run were found:\n\n"
                f"{', '.join(existing_folders)}\n\n"
                "They may be reused or overwritten.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if confirmation == QMessageBox.No:
                self._reset_ui_after_process()
                return

        # ------------------------------------------------
        # Cache selected atoms for atomic descriptors
        # ------------------------------------------------
        selected_atoms_for_aqme = []

        if (
            hasattr(self.tab_widget_aqme, "selected_atoms")
            and self.tab_widget_aqme.selected_atoms
        ):
            selected_atoms_for_aqme = list(self.tab_widget_aqme.selected_atoms)

            smarts = self.tab_widget_aqme.smarts_targets[0]

            self.mapped_train_csv = self.tab_widget_aqme.generate_mapped_csv_from_smiles(
                self.file_path,
                smarts,
                selected_atoms_for_aqme
            )

            self.is_aqme_mapped = True

            if self.csv_test_path:
                self.mapped_test_csv = self.tab_widget_aqme.generate_mapped_csv_from_smiles(
                    self.csv_test_path,
                    smarts,
                    selected_atoms_for_aqme
                )

            #  Save atomic mapping contract in .dat
            run_dir = os.path.dirname(self.file_path)
            self._write_atom_mapping_dat(
                smarts=smarts,
                selected_atoms=selected_atoms_for_aqme,
                run_dir=run_dir
            )

        # ------------------------------------------------
        # Decide REAL input CSVs for AQME
        # ------------------------------------------------
        train_input_csv = (
            self.mapped_train_csv
            if self.is_aqme_mapped
            else self.file_path
        )

        test_input_csv = (
            self.mapped_test_csv
            if self.is_aqme_mapped and self.mapped_test_csv
            else self.csv_test_path
        )

        # -------------------------
        # Build AQME command queue
        # -------------------------
        main_cmd, self.aqme_run_dir = self.build_aqme_command(
            train_input_csv,
            selected_atoms_override=selected_atoms_for_aqme
        )

        self.aqme_command_queue.append({
            "command": main_cmd,
            "csv": train_input_csv,
            "role": "train"
        })

        if test_input_csv:
            test_cmd, _ = self.build_aqme_command(
                test_input_csv,
                selected_atoms_override=selected_atoms_for_aqme
            )
            self.aqme_command_queue.append({
                "command": test_cmd,
                "csv": test_input_csv,
                "role": "test"
            })

        # ------------
        # Launch AQME
        # ------------
        self.console_output.append(
            "<b><span style='color:deepskyblue;'>Running AQME (descriptor generation)...</span></b><br>"
        )
        self.progress.setRange(0, 0)
        self._run_next_aqme()

    def _run_next_aqme(self):
        """ Runs the next AQME command in the queue."""
        if not self.aqme_command_queue:
            return

        run = self.aqme_command_queue.pop(0)
        self.current_aqme_run = run  # keep reference

        self.aqme_input_csv = run["csv"]

        self.worker = RobertWorker(run["command"], self.aqme_run_dir)
        self.worker.output_received.connect(self.console_output.append)
        self.worker.error_received.connect(self.console_output.append)
        self.worker.process_finished.connect(self._on_aqme_step_finished)
        self.worker.start()

    def stop_process(self):
        """Stops the ROBERT and AQME process safely after user confirmation, non-blocking."""

        confirmation = QMessageBox.question(
            self, 
            "WARNING!", 
            "Are you sure you want to stop the process?",
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

    def _on_aqme_step_finished(self, exit_code):
        """Handles the completion of an AQME step and manages the queue."""

        if exit_code != 0:
            self.aqme_command_queue.clear()
            self.on_process_finished(exit_code)
            return

        # --------------------------------------------------
        # ARCHIVE AQME OUTPUT FOLDERS FOR THIS RUN (INLINE)
        # --------------------------------------------------
        try:
            run_dir = Path(self.aqme_run_dir)
            csv_path = Path(self.current_aqme_run["csv"])
            base_name = csv_path.stem  # e.g. main / test

            archive_root = run_dir / "AQME_RUNS" / base_name
            archive_root.mkdir(parents=True, exist_ok=True)

            for folder_name in ["CSEARCH", "QDESCP"]:
                src = run_dir / folder_name
                if src.exists() and src.is_dir():
                    dst = archive_root / folder_name

                    # If destination already exists, remove it
                    if dst.exists():
                        shutil.rmtree(dst)

                    shutil.move(str(src), str(dst))

        except Exception as e:
            self.console_output.append(f"WARNING! Failed to archive AQME folders: {e}")

        # --------------------------------------------------
        # Store finished run
        # --------------------------------------------------
        self.aqme_runs.append(self.current_aqme_run)

        # --------------------------------------------------
        # Launch next AQME run if pending
        # --------------------------------------------------
        if self.aqme_command_queue:
            self.console_output.append(
                "<span style='color:cyan;'>Running AQME on next CSV...</span>"
            )
            self._run_next_aqme()
            return

        # --------------------------------------------------
        # All AQME runs finished
        # --------------------------------------------------
        self.aqme_is_running = False
        self.on_process_finished(exit_code)

    @Slot(int)
    def on_process_finished(self, exit_code):
        """Handles the cleanup after the process (ROBERT or AQME) finishes."""

        # --------------------------------------------------
        # Clean up the worker process
        # --------------------------------------------------
        if self.worker:
            if self.worker.process and self.worker.process.poll() is None:
                self.worker.stop()
            self.worker = None

        # --------------------------------------------------
        # Handle manual stop (common to ROBERT & AQME)
        # --------------------------------------------------
        if exit_code == -1:
            self.console_output.clear()
            QMessageBox.information(
                self,
                "WARNING!",
                f"{self.current_process} has been successfully stopped."
            )
            self.manual_stop = False
            self._reset_ui_after_process()
            return

        output_text = self.console_output.toPlainText()

        # ==================================================
        # AQME COMPLETION LOGIC
        # ==================================================
        if self.current_process == "AQME":

            # ==================================================
            # AQME SUCCESS
            # ==================================================
            if exit_code == 0 and "Time QDESCP:" in output_text:

                # =============================================
                # AQME TEST -> chain directly to ROBERT PREDICT
                # =============================================
                if getattr(self, "aqme_role", None) == "test":

                    # Reset AQME state to avoid conflicts with future runs
                    self.aqme_role = None

                    new_test_csv = self._detect_aqme_output_csv()

                    if not new_test_csv:
                        QMessageBox.warning(
                            self,
                            "AQME error",
                            "AQME finished successfully, but the expected output CSV was not found.\n\n"
                            "Descriptor generation for the test set completed, but the generated "
                            "CSV file could not be detected, so prediction cannot continue."
                        )
                        self.manual_stop = False
                        self._reset_ui_after_process()
                        return

                    self.aqme_test_csv_path = new_test_csv

                    self.console_output.append(
                        f"[AQME] Using generated test CSV: {self.aqme_test_csv_path}"
                    )

                    # --------------------------------------------------
                    # Launch ROBERT prediction (force AQME output as test CSV)
                    # --------------------------------------------------
                
                    # Save original test CSV only 
                    if not getattr(self, "_original_test_csv_path", None):
                        self._original_test_csv_path = self.csv_test_path

                    # Override test CSV with AQME output
                    self.csv_test_path = self.aqme_test_csv_path

                    command = self.build_robert_command(self.file_path)
                    run_dir = os.path.dirname(self.aqme_test_csv_path)

                    self.current_process = "ROBERT"

                    self.worker = RobertWorker(command, run_dir)
                    self.worker.output_received.connect(self.console_output.append)
                    self.worker.error_received.connect(self.console_output.append)
                    self.worker.process_finished.connect(self.on_process_finished)
                    self.worker.start()
                    return 

                # =============================================
                # AQME TRAIN -> original popup logic (unchanged)
                # =============================================
                train_run = next(
                    (r for r in self.aqme_runs if r["role"] == "train"),
                    None
                )

                if not train_run:
                    QMessageBox.warning(
                        self,
                        "WARNING!",
                        "No AQME train output found."
                    )
                    self.manual_stop = False
                    self._reset_ui_after_process()
                    return

                aqme_base = train_run["csv"]
                base_dir = os.path.dirname(aqme_base)
                base_name = os.path.splitext(os.path.basename(aqme_base))[0]

                aqme_csvs = {
                    "denovo": os.path.join(base_dir, f"AQME-ROBERT_denovo_{base_name}.csv"),
                    "interpret": os.path.join(base_dir, f"AQME-ROBERT_interpret_{base_name}.csv"),
                    "full": os.path.join(base_dir, f"AQME-ROBERT_full_{base_name}.csv"),
                }

                # ---------------------------------------------
                # Compute number of generated descriptors
                # ---------------------------------------------
                descriptor_counts = {}

                try:
                    original_df = pd.read_csv(train_run["csv"])
                    n_original_cols = len(original_df.columns)

                    for level, path in aqme_csvs.items():
                        if os.path.isfile(path):
                            aqme_df = pd.read_csv(path)
                            descriptor_counts[level] = (
                                len(aqme_df.columns) - n_original_cols
                            )
                except Exception:
                    descriptor_counts = {}

                # ---------------------------------------------
                # Create popup
                # ---------------------------------------------
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("AQME completed successfully")
                msg.setTextFormat(Qt.RichText)
                msg.setText(
                    f"""
                    <b>AQME descriptor generation completed successfully.</b><br>

                    <b>What was generated?</b><br>
                    Descriptor CSV files combining electronic, steric, and structural information
                    from <b>GFN2-xTB</b>, <b>Morfeus</b>, and <b>RDKit</b>.<br><br>

                    <b>Available descriptor levels:</b><br>
                    • <b>DeNovo</b> - Simple, interpretable descriptors for fast chemical insight
                    ({descriptor_counts.get("denovo", "N/A")} descriptors).<br>
                    • <b>Interpret</b> - Balanced descriptor set
                    <i>(recommended for ML with ROBERT)</i>
                    ({descriptor_counts.get("interpret", "N/A")} descriptors).<br>
                    • <b>Full</b> - Complete descriptor space for advanced analyses
                    ({descriptor_counts.get("full", "N/A")} descriptors).<br><br>

                    <b>Next step</b><br>
                    To run <b>machine learning workflows</b> in ROBERT, the selected CSV file
                    <b>must include a target (y) column</b> (the property you want to predict).<br><br>

                    <b>If your CSV already contains a target column</b>, you can directly proceed
                    to ROBERT and select it as the target variable.
                    If needed, you can always add or modify the target column later and reload the CSV into ROBERT
                    to perform full machine learning workflows.<br><br>

                    <b>Select a file to load into ROBERT, or cancel to continue.</b>
                    """
                )

                btn_interpret = msg.addButton(
                    "Interpret descriptors (recommended)",
                    QMessageBox.ActionRole
                )
                btn_denovo = msg.addButton(
                    "DeNovo descriptors",
                    QMessageBox.ActionRole
                )
                btn_full = msg.addButton(
                    "Full descriptors",
                    QMessageBox.ActionRole
                )
                msg.addButton("Cancel", QMessageBox.RejectRole)

                msg.exec()
                clicked = msg.clickedButton()

                # ---------------------------------------------
                # Determine selected descriptor level
                # ---------------------------------------------
                user_cancelled = False

                if clicked == btn_interpret:
                    selected_level = "interpret"
                elif clicked == btn_denovo:
                    selected_level = "denovo"
                elif clicked == btn_full:
                    selected_level = "full"
                else:
                    user_cancelled = True
                    selected_level = None

                # ---------------------------------------------
                # Load TRAIN and TEST
                # ---------------------------------------------
                if not user_cancelled:
                    for run in self.aqme_runs:
                        base_dir = self.aqme_run_dir
                        base_name = os.path.splitext(os.path.basename(run["csv"]))[0]

                        output_csv = os.path.join(
                            base_dir,
                            f"AQME-ROBERT_{selected_level}_{base_name}.csv"
                        )

                        if not os.path.isfile(output_csv):
                            continue

                        if run["role"] == "train":
                            self.set_file_path(output_csv, force=True)
                        elif run["role"] == "test":
                            self.set_csv_test_path(output_csv)

                # ---------------------------------------------
                # Organize AQME outputs and mapped CSVs
                # ---------------------------------------------
                try:
                    runs_root = Path(self.aqme_run_dir) / "AQME_RUNS"
                    runs_root.mkdir(parents=True, exist_ok=True)

                    for run in self.aqme_runs:
                        base_name = Path(run["csv"]).stem
                        base_dir = runs_root / base_name
                        csv_dir = base_dir / "CSVs"
                        csv_dir.mkdir(parents=True, exist_ok=True)

                        # --------------------------------------------------
                        # 1) Move NON-selected AQME descriptor CSVs
                        # --------------------------------------------------
                        for level in ["denovo", "interpret", "full"]:
                            if not user_cancelled and level == selected_level:
                                continue # keep active CSV where it is

                            csv_name = f"AQME-ROBERT_{level}_{base_name}.csv"
                            csv_path = Path(self.aqme_run_dir) / csv_name

                            if not csv_path.exists():
                                continue

                            target_path = csv_dir / csv_name
                            if target_path.exists():
                                target_path.unlink()

                            shutil.move(str(csv_path), str(target_path))
                            
                        # --------------------------------------------------
                        # 2) Move mapped CSVs (PER RUN, TRAIN + TEST)
                        # --------------------------------------------------
                        if run["role"] == "train":
                            mapped_csv = getattr(self, "mapped_train_csv", None)
                        elif run["role"] == "test":
                            mapped_csv = getattr(self, "mapped_test_csv", None)
                        else:
                            mapped_csv = None

                        if mapped_csv:
                            mapped_path = Path(mapped_csv)
                            if mapped_path.exists():
                                target_mapped = csv_dir / mapped_path.name
                                if target_mapped.exists():
                                    target_mapped.unlink()
                                shutil.move(str(mapped_path), str(target_mapped))

                except Exception as e:
                    self.console_output.append(
                        f"<span style='color:orange;'>WARNING: Failed to organize AQME CSVs: {e}</span>"
                    )

            # ==================================================
            # AQME FAILURE
            # ==================================================
            else:
                QMessageBox.warning(
                    self,
                    "WARNING!",
                    "AQME encountered an issue while finishing. Please check the logs."
                )
            # End of AQME workflow
            self.manual_stop = False
            self._reset_ui_after_process()

            return

        # ==================================================
        # ROBERT COMPLETION LOGIC (UNCHANGED)
        # ==================================================
        workflow = self.workflow_selector.currentText()

        # --------------------------------------------------
        # Refresh tabs using correct path (TRAIN > TEST)
        # --------------------------------------------------
        if hasattr(self, "file_path") and self.file_path:
            self.refresh_tabs(self.file_path)
        elif hasattr(self, "csv_test_path") and self.csv_test_path:
            self.refresh_tabs(self.csv_test_path)

        # ------------------------
        # Full workflow / REPORT
        # ------------------------
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

                view_report_button.clicked.connect(
                    lambda: self.tab_widget.setCurrentWidget(self.results_tab)
                )

                msg_box.exec()
            else:
                QMessageBox.warning(
                    self,
                    "WARNING!",
                    "ROBERT encountered an issue while finishing. Please check the logs."
                )

        # ------------------------
        # Individual ROBERT steps
        # ------------------------
        elif workflow == "CURATE":
            if exit_code == 0 and "Time CURATE:" in output_text:
                QMessageBox.information(
                    self, "Success", "ROBERT has successfully completed the CURATE step."
                )
            else:
                QMessageBox.warning(
                    self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs."
                )

        elif workflow == "GENERATE":
            if exit_code == 0 and "Time GENERATE:" in output_text:
                QMessageBox.information(
                    self, "Success", "ROBERT has successfully completed the GENERATE step."
                )
            else:
                QMessageBox.warning(
                    self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs."
                )

        elif workflow == "PREDICT":
            if exit_code == 0 and "Time PREDICT:" in output_text:
                QMessageBox.information(
                    self, "Success", "ROBERT has successfully completed the PREDICT step."
                )
            else:
                QMessageBox.warning(
                    self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs."
                )

        elif workflow == "VERIFY":
            if exit_code == 0 and "Time VERIFY:" in output_text:
                QMessageBox.information(
                    self, "Success", "ROBERT has successfully completed the VERIFY step."
                )
            else:
                QMessageBox.warning(
                    self, "WARNING!", "ROBERT encountered an issue while finishing. Please check the logs."
                )

        # Restore previous test CSV if overridden for test workflow aqme generation
        if getattr(self, "_original_test_csv_path", None):
            self.csv_test_path = self._original_test_csv_path
            self._original_test_csv_path = None

        # --------------------------------------------------
        # Final cleanup
        # --------------------------------------------------
        self._reset_ui_after_process()
        self.manual_stop = False
