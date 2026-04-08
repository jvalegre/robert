"""
Advanced options tab for easyROB.

This module implements a form-based interface that allows users to configure
advanced parameters for different stages of the ROBERT workflow.

Each section (GENERAL, CURATE, GENERATE, PREDICT) maps directly to a stage
in the pipeline, enabling structured configuration without overloading the
main application window.

Import strategy:
- Supports dual execution modes:
  1. Local (portable execution from source folder)
  2. Installed package (environment / entry point)

- Imports use a try/except fallback:
  local modules are attempted first, with package imports as fallback.

Notes:
- This module focuses purely on UI configuration.
- No execution logic or background processing is handled here.

"""

# ------------------------------------------------------------
# Import resolution (local vs installed package)
# ------------------------------------------------------------
# Attempt local imports first (portable mode). If they fail,
# fall back to installed package imports.
try:

    from utils.utils_gui import (
        AssetLibrary,
        QCheckBox,
        QComboBox,
        QDesktopServices,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QIcon,
        QLabel,
        QLineEdit,
        QPushButton,
        QUrl,
        QVBoxLayout,
        QWidget,
        Qt,
    )

except ImportError as e:

    from robert.gui_easyrob.utils.utils_gui import (
        AssetLibrary,
        QCheckBox,
        QComboBox,
        QDesktopServices,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QIcon,
        QLabel,
        QLineEdit,
        QPushButton,
        QUrl,
        QVBoxLayout,
        QWidget,
        Qt,
    )

class AdvancedOptionsTab(QWidget):
    """Tab for advanced options in the easyROB application."""
    def __init__(self, type_dropdown, tab_widget):
        super().__init__()
        self.type = type_dropdown
        self.tab_widget = tab_widget  # Reference to the main QTabWidget
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
        """Open a documentation section in the browser."""
        
        base_url = "https://robert.readthedocs.io/en/latest/Technical/defaults.html"

        if anchor.upper() == "GENERAL":
            full_url = base_url
        else:
            full_url = f"{base_url}#{anchor.lower()}"

        QDesktopServices.openUrl(QUrl(full_url))

    def create_help_button(self, topic: str) -> QPushButton:
        """Return a styled Help button linking to documentation."""

        button = QPushButton(f"Help {topic.upper()} parameters")

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

        self.split = QComboBox()
        self.split.addItems([ "even", "RND", "stratified", "KN", "extra_q1", "extra_q5" ])
        layout.addRow(QLabel("split:"), self.split)

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