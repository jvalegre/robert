"""
EVALUATE tab for easyROB.

Lets a user evaluate their own scikit-learn model (regression or classification) under
ROBERT's standard rigor - repeated CV, VERIFY's flawed-model tests, boundary robustness (for
regression), outliers, SHAP, and a scored PDF report - without going through GENERATE's
Bayesian hyperparameter search.

Responsibilities:
- Collect a CSV plus the y/names columns and prediction type - the train/internal-test split
  always uses ROBERT's own default (no manual split-point control exposed here)
- Optionally collect a separate external test CSV (--csv_test), for a model already evaluated
  against a held-out set the user manages themselves
- Collect the model to evaluate: ROBERT's own default (MVL, no tuning), a CSV naming a
  scikit-learn class and its hyperparameters, or a pre-fitted model saved with joblib/pickle
- Score the model as-is on the descriptors already present in the input CSV - ROBERT never
  re-curates them, so the model is evaluated on exactly what the user provided
- Build the equivalent "python -m robert --evaluate ..." command and run it with the same
  RobertWorker used by the main ROBERT tab, streaming output live
- Refresh the Reports/Predictions/Images tabs once the run finishes successfully

Import strategy mirrors the rest of gui_easyrob/tabs: a local (portable) import path is tried
first, falling back to the installed-package path.
"""

try:
    from utils.utils_gui import (
        AssetLibrary,
        DropLabel,
        NoScrollComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QIcon,
        QLabel,
        QLineEdit,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        Qt,
        RobertWorker,
        smart_read_csv,
    )
except ImportError:
    from robert.gui_easyrob.utils.utils_gui import (
        AssetLibrary,
        DropLabel,
        NoScrollComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QIcon,
        QLabel,
        QLineEdit,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        Qt,
        RobertWorker,
        smart_read_csv,
    )

import csv
import inspect
import os
import sys
import tempfile
from pathlib import Path

# shared style for section group boxes, matching the "Advanced Options" tab's convention
BOX_STYLE = "QGroupBox { font-weight: bold; }"

MODEL_PARAMS_EXAMPLE = """\
<p>A CSV with two columns: <b>param</b> and <b>value</b>.</p>
<p>One row must have <code>param = model</code>, naming any scikit-learn regressor/classifier \
class exactly (e.g. <code>RandomForestRegressor</code>). Every other row is one \
hyperparameter for that class.</p>
<p><b>Example:</b></p>
<table cellpadding="3" border="1" style="border-collapse:collapse;">
<tr><th>param</th><th>value</th></tr>
<tr><td>model</td><td>RandomForestRegressor</td></tr>
<tr><td>n_estimators</td><td>200</td></tr>
<tr><td>max_depth</td><td>5</td></tr>
</table>
"""


def _make_info_icon(label, rich_text):
    """A Help button matching the "Advanced Options" tab's own help-button style (icon + text
    on a normal bordered button), with a tooltip and a click-to-open popup with details."""

    icon = QPushButton(label)
    with AssetLibrary.Info_icon.get_path() as icon_path:
        icon.setIcon(QIcon(str(icon_path)))
    icon.setCursor(Qt.PointingHandCursor)
    icon.setStyleSheet("padding: 4px; font-weight: bold;")
    icon.setToolTip(rich_text)

    def _show_popup():
        box = QMessageBox()
        box.setWindowTitle("Hyperparameters CSV format")
        box.setTextFormat(Qt.RichText)
        box.setText(rich_text)
        box.exec()

    icon.clicked.connect(_show_popup)
    return icon


# Estimators excluded from the picker - verified empirically (instantiate + fit + predict on a
# small single-target dataset shaped like ROBERT's typical input) rather than guessed. Two
# failure patterns account for all of them:
# 1. Meta-estimators whose one required constructor argument IS another model (estimator/
#    estimators) - there's no way to express "another scikit-learn model" as plain CSV text, so
#    these fail immediately at instantiation (missing required positional argument)
# 2. Multi-task/multi-component models that reject a single-column y or need more components
#    than ROBERT ever has features for single-target regression - these fail at fit() instead
_INCOMPATIBLE_REGRESSORS = {
    'CCA', 'IsotonicRegression', 'MultiOutputRegressor', 'MultiTaskElasticNet',
    'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV', 'PLSCanonical',
    'RegressorChain', 'StackingRegressor', 'VotingRegressor',
}
_INCOMPATIBLE_CLASSIFIERS = {
    'ClassifierChain', 'FixedThresholdClassifier', 'MultiOutputClassifier',
    'OneVsOneClassifier', 'OneVsRestClassifier', 'OutputCodeClassifier',
    'SelfTrainingClassifier', 'StackingClassifier', 'TunedThresholdClassifierCV',
    'VotingClassifier',
}


def _sklearn_estimators(pred_type):
    """(name, class) pairs for every scikit-learn regressor/classifier registered via
    sklearn.utils.all_estimators() - imported lazily so sklearn's own import cost is only paid
    once the user actually opens this model-source option. Excludes the models in
    _INCOMPATIBLE_REGRESSORS/_INCOMPATIBLE_CLASSIFIERS above, which can't work with EVALUATE's
    plain param=value interface regardless of what the user fills in."""

    from sklearn.utils import all_estimators

    key = 'regressor' if pred_type == 'reg' else 'classifier'
    excluded = _INCOMPATIBLE_REGRESSORS if pred_type == 'reg' else _INCOMPATIBLE_CLASSIFIERS
    try:
        pairs = all_estimators(type_filter=key)
        return sorted((p for p in pairs if p[0] not in excluded), key=lambda pair: pair[0])
    except Exception:
        return []


def _constructor_params(model_cls):
    """(param_name, default_repr) pairs for a class's __init__, skipping self/*args/**kwargs.
    default_repr is only a display hint (placeholder text) - never parsed back, so any
    representation is fine."""

    try:
        signature = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return []

    params = []
    for name, parameter in signature.parameters.items():
        if name == 'self':
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        default_repr = '' if parameter.default is inspect.Parameter.empty else repr(parameter.default)
        params.append((name, default_repr))
    return params


class EvaluateTab(QWidget):
    """Tab dedicated to running EVALUATE."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.worker = None
        self.csv_path = None
        self.csv_test_path = None
        self.model_params_path = None
        self.model_file_path = None
        self.manual_stop = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(18)

        title = QLabel("Evaluate your own model with ROBERT's standard rigor")
        title.setStyleSheet("font-weight:600; font-size:16px;")
        outer.addWidget(title)

        subtitle = QLabel(
            "Runs repeated cross-validation, VERIFY's flawed-model tests, boundary robustness "
            "(regression), outliers, SHAP, and a scored PDF - for a model you already chose, "
            "instead of ROBERT's own Bayesian hyperparameter search."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray; font-size: 12px;")
        outer.addWidget(subtitle)

        # --- Input data group ---
        input_box = QGroupBox("INPUT DATA")
        input_box.setStyleSheet(BOX_STYLE)
        input_layout = QVBoxLayout(input_box)
        input_layout.setSpacing(10)
        input_layout.setContentsMargins(12, 16, 12, 12)

        # main CSV (left) and optional external test CSV (right), side by side - same layout
        # as the main "ROBERT" tab's own CSV pickers, to save vertical space
        csv_row = QHBoxLayout()
        csv_row.setSpacing(16)

        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        left_title = QLabel("Input CSV")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setStyleSheet("font-weight:600; font-size:12px;")
        left_col.addWidget(left_title)
        self.csv_label = DropLabel(
            "Drag & Drop a CSV file here",
            self,
            file_filter="CSV Files (*.csv)",
            extensions=(".csv",),
        )
        self.csv_label.set_callback(self.set_csv_path)
        left_col.addWidget(self.csv_label)
        csv_row.addLayout(left_col)

        right_col = QVBoxLayout()
        right_col.setSpacing(4)
        right_title = QLabel("Already had an external test set for this model? (optional)")
        right_title.setAlignment(Qt.AlignCenter)
        right_title.setWordWrap(True)
        right_title.setStyleSheet("font-weight:600; font-size:12px;")
        right_col.addWidget(right_title)
        self.csv_test_label = DropLabel(
            "Drag & Drop an external test CSV here (optional)",
            self,
            file_filter="CSV Files (*.csv)",
            extensions=(".csv",),
        )
        self.csv_test_label.set_callback(self.set_csv_test_path)
        right_col.addWidget(self.csv_test_label)
        csv_row.addLayout(right_col)

        input_layout.addLayout(csv_row)

        split_info = QLabel(
            "The train / internal-test split always uses ROBERT's own default (~20% of the "
            "data, evenly spread across the y range)."
        )
        split_info.setWordWrap(True)
        split_info.setStyleSheet("color: gray; font-size: 11px;")
        input_layout.addWidget(split_info)

        row1 = QHBoxLayout()
        row1.setSpacing(16)

        y_col = QVBoxLayout()
        y_col.setSpacing(4)
        y_col.addWidget(QLabel("y (target column)"))
        self.y_dropdown = NoScrollComboBox()
        y_col.addWidget(self.y_dropdown)
        row1.addLayout(y_col)

        names_col = QVBoxLayout()
        names_col.setSpacing(4)
        names_col.addWidget(QLabel("names (ID column)"))
        self.names_dropdown = NoScrollComboBox()
        names_col.addWidget(self.names_dropdown)
        row1.addLayout(names_col)

        type_col = QVBoxLayout()
        type_col.setSpacing(4)
        type_col.addWidget(QLabel("Prediction type"))
        self.type_dropdown = NoScrollComboBox()
        self.type_dropdown.addItems(["Regression", "Classification"])
        type_col.addWidget(self.type_dropdown)
        row1.addLayout(type_col)

        input_layout.addLayout(row1)
        outer.addWidget(input_box)

        # --- Model source group ---
        model_box = QGroupBox("MODEL TO EVALUATE")
        model_box.setStyleSheet(BOX_STYLE)
        model_layout = QVBoxLayout(model_box)
        model_layout.setSpacing(10)
        model_layout.setContentsMargins(12, 16, 12, 12)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        self.model_source_dropdown = NoScrollComboBox()
        self.model_source_dropdown.addItems([
            "Default (MVL, no hyperparameters to tune)",
            "Pick a scikit-learn model (fill in hyperparameters)",
            "From a hyperparameters CSV (param,value)",
            "From a saved model file (joblib/pickle)",
        ])
        self.model_source_dropdown.currentIndexChanged.connect(self._on_model_source_changed)
        source_row.addWidget(self.model_source_dropdown, stretch=1)

        self.model_params_info = _make_info_icon("Help CSV format", MODEL_PARAMS_EXAMPLE)
        self.model_params_info.hide()
        source_row.addWidget(self.model_params_info)

        model_layout.addLayout(source_row)

        # --- "Pick a scikit-learn model" picker: a model dropdown (filtered by Prediction
        # type) plus a scrollable form of that class's own constructor hyperparameters, each
        # pre-filled with its scikit-learn default as a placeholder. Blank fields simply aren't
        # sent, so scikit-learn's own default applies - only touched fields override it
        self.sklearn_picker_container = QWidget()
        sklearn_picker_layout = QVBoxLayout(self.sklearn_picker_container)
        sklearn_picker_layout.setContentsMargins(0, 0, 0, 0)
        sklearn_picker_layout.setSpacing(8)

        self.sklearn_model_dropdown = NoScrollComboBox()
        self.sklearn_model_dropdown.currentIndexChanged.connect(self._rebuild_sklearn_param_form)
        sklearn_picker_layout.addWidget(self.sklearn_model_dropdown)

        self.sklearn_param_fields = {}
        sklearn_params_widget = QWidget()
        self.sklearn_params_form = QFormLayout(sklearn_params_widget)
        sklearn_params_scroll = QScrollArea()
        sklearn_params_scroll.setWidgetResizable(True)
        sklearn_params_scroll.setMaximumHeight(240)
        sklearn_params_scroll.setWidget(sklearn_params_widget)
        sklearn_picker_layout.addWidget(sklearn_params_scroll)

        self.sklearn_picker_container.hide()
        model_layout.addWidget(self.sklearn_picker_container)

        self.model_params_label = DropLabel(
            "Drag & Drop a CSV with 'param,value' rows (one row must be param='model', "
            "e.g. value='RandomForestRegressor') - click the info icon above for an example",
            self,
            file_filter="CSV Files (*.csv)",
            extensions=(".csv",),
        )
        self.model_params_label.set_callback(self.set_model_params_path)
        self.model_params_label.hide()
        model_layout.addWidget(self.model_params_label)

        self.model_file_label = DropLabel(
            "Drag & Drop a pre-fitted scikit-learn model saved with joblib/pickle",
            self,
            file_filter="Model files (*.joblib *.pkl *.pickle);;All files (*)",
            extensions=(".joblib", ".pkl", ".pickle"),
        )
        self.model_file_label.set_callback(self.set_model_file_path)
        self.model_file_label.hide()
        model_layout.addWidget(self.model_file_label)

        outer.addWidget(model_box)

        # refreshes the sklearn model list when Prediction type changes (only matters while the
        # "Pick a scikit-learn model" source is active - regressors vs classifiers differ)
        self.type_dropdown.currentIndexChanged.connect(self._on_type_changed)

        # Note: k-fold/CV repetitions/seed aren't exposed here - this tab always uses ROBERT's
        # own defaults for those.

        # --- Run / Stop (centered) ---
        run_row = QHBoxLayout()
        run_row.addStretch(1)

        self.run_button = QPushButton(" Run EVALUATE")
        self.run_button.setFixedSize(200, 40)
        self.run_button.setCursor(Qt.PointingHandCursor)
        with AssetLibrary.Play_icon.get_path() as icon_play_path:
            self.run_button.setIcon(QIcon(str(icon_play_path)))
        self.run_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
                border-radius: 10px;
                background-color: #6A0DAD;
                color: white;
                border: 2px solid #7B2CBF;
            }
            QPushButton:hover {
                background-color: #7B2CBF;
                border: 2px solid #9D4EDD;
            }
            QPushButton:pressed {
                background-color: #4B0082;
                border: 2px solid #5A189A;
            }
            QPushButton:disabled {
                background-color: #3A2A4D;
                border: 2px solid #3A2A4D;
                color: #AAA;
            }
        """)
        self.run_button.clicked.connect(self.run_evaluate)
        run_row.addWidget(self.run_button)

        run_row.addSpacing(16)

        self.stop_button = QPushButton(" Stop")
        self.stop_button.setFixedSize(200, 40)
        self.stop_button.setCursor(Qt.PointingHandCursor)
        with AssetLibrary.Stop_icon.get_path() as icon_stop_path:
            self.stop_button.setIcon(QIcon(str(icon_stop_path)))
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
        self.stop_button.setDisabled(True)
        self.stop_button.clicked.connect(self.stop_run)
        run_row.addWidget(self.stop_button)

        run_row.addStretch(1)
        outer.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()
        outer.addWidget(self.progress)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet(
            "background-color: black; color: white; font-family: monospace; border-radius: 4px;"
        )
        self.console_output.setMinimumHeight(220)
        outer.addWidget(self.console_output)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def set_csv_path(self, file_path):
        self.csv_path = file_path

        df = smart_read_csv(file_path)
        if df is None:
            QMessageBox.warning(self, "EVALUATE", f"Could not read CSV file:\n{file_path}")
            return

        columns = list(df.columns)
        self.y_dropdown.clear()
        self.y_dropdown.addItems(columns)
        self.names_dropdown.clear()
        self.names_dropdown.addItems(columns)

        lower_map = {col.lower(): col for col in columns}
        if "code_name" in lower_map:
            self.names_dropdown.setCurrentText(lower_map["code_name"])

    def set_csv_test_path(self, file_path):
        self.csv_test_path = file_path

    def set_model_params_path(self, file_path):
        self.model_params_path = file_path

    def set_model_file_path(self, file_path):
        self.model_file_path = file_path

    def _on_model_source_changed(self, index):
        self.sklearn_picker_container.hide()
        self.model_params_label.hide()
        self.model_params_info.hide()
        self.model_file_label.hide()
        if index == 1:
            self.sklearn_picker_container.show()
            self._refresh_sklearn_model_list()
        elif index == 2:
            self.model_params_label.show()
            self.model_params_info.show()
        elif index == 3:
            self.model_file_label.show()

    def _on_type_changed(self):
        if self.model_source_dropdown.currentIndex() == 1:
            self._refresh_sklearn_model_list()

    def _current_pred_type(self):
        return 'clas' if self.type_dropdown.currentText() == "Classification" else 'reg'

    def _refresh_sklearn_model_list(self):
        current_name = self.sklearn_model_dropdown.currentText()
        names = [name for name, _ in _sklearn_estimators(self._current_pred_type())]

        self.sklearn_model_dropdown.blockSignals(True)
        self.sklearn_model_dropdown.clear()
        self.sklearn_model_dropdown.addItems(names)
        if current_name in names:
            self.sklearn_model_dropdown.setCurrentText(current_name)
        self.sklearn_model_dropdown.blockSignals(False)

        self._rebuild_sklearn_param_form()

    def _rebuild_sklearn_param_form(self):
        while self.sklearn_params_form.rowCount() > 0:
            self.sklearn_params_form.removeRow(0)
        self.sklearn_param_fields = {}

        model_name = self.sklearn_model_dropdown.currentText()
        if not model_name:
            return

        estimators = dict(_sklearn_estimators(self._current_pred_type()))
        model_cls = estimators.get(model_name)
        if model_cls is None:
            return

        for param_name, default_repr in _constructor_params(model_cls):
            field = QLineEdit()
            field.setPlaceholderText(default_repr if default_repr else "(required)")
            self.sklearn_param_fields[param_name] = field
            self.sklearn_params_form.addRow(QLabel(param_name), field)

    def _write_sklearn_params_csv(self):
        """Writes the picked model + any non-blank hyperparameter fields to a temp CSV, reusing
        the exact same --model_params mechanism as the "upload a CSV" model source."""

        rows = [("param", "value"), ("model", self.sklearn_model_dropdown.currentText())]
        for param_name, field in self.sklearn_param_fields.items():
            text = field.text().strip()
            if text:
                rows.append((param_name, text))

        fd, path = tempfile.mkstemp(prefix="robert_gui_model_params_", suffix=".csv")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerows(rows)
        return path

    # ------------------------------------------------------------------
    # Validation + command building
    # ------------------------------------------------------------------
    def _validate(self):
        if not self.csv_path:
            QMessageBox.warning(self, "EVALUATE", "Please select an input CSV file.")
            return False

        if not self.y_dropdown.currentText():
            QMessageBox.warning(self, "EVALUATE", "Please select the y (target) column.")
            return False

        if not self.names_dropdown.currentText():
            QMessageBox.warning(self, "EVALUATE", "Please select the names (ID) column.")
            return False

        source_index = self.model_source_dropdown.currentIndex()
        if source_index == 1 and not self.sklearn_model_dropdown.currentText():
            QMessageBox.warning(
                self, "EVALUATE", "Please pick a scikit-learn model, or choose a "
                "different model source."
            )
            return False
        if source_index == 2 and not self.model_params_path:
            QMessageBox.warning(
                self, "EVALUATE", "Please select a hyperparameters CSV, or choose a "
                "different model source."
            )
            return False
        if source_index == 3 and not self.model_file_path:
            QMessageBox.warning(
                self, "EVALUATE", "Please select a saved model file, or choose a "
                "different model source."
            )
            return False

        return True

    def _resolve_python_executable(self):
        python_pointer = sys.executable or "python"
        if getattr(sys, "frozen", False):
            embedded_env = Path.cwd() / "_internal" / "robert_env"
            if sys.platform == "win32":
                python_pointer = embedded_env / "python.exe"
            elif sys.platform == "darwin":
                python_pointer = embedded_env / "bin" / "python3"
            else:
                python_pointer = embedded_env / "bin" / "python"
        return str(python_pointer)

    def _build_command(self):
        python_pointer = self._resolve_python_executable()

        command = (
            f'"{python_pointer}" -u -m robert --evaluate '
            f'--csv_name "{os.path.basename(self.csv_path)}" '
            f'--y "{self.y_dropdown.currentText()}" '
            f'--names "{self.names_dropdown.currentText()}"'
        )

        if self.type_dropdown.currentText() == "Classification":
            command += ' --type "clas"'

        if self.csv_test_path:
            command += f' --csv_test "{self.csv_test_path}"'

        source_index = self.model_source_dropdown.currentIndex()
        if source_index == 1:
            temp_params_csv = self._write_sklearn_params_csv()
            command += f' --model_params "{temp_params_csv}"'
        elif source_index == 2:
            command += f' --model_params "{self.model_params_path}"'
        elif source_index == 3:
            command += f' --model_file "{self.model_file_path}"'

        return command

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------
    def run_evaluate(self):
        if not self._validate():
            return

        command = self._build_command()
        run_dir = os.path.dirname(self.csv_path)

        self.run_button.setDisabled(True)
        self.stop_button.setDisabled(False)
        self.manual_stop = False

        self.console_output.clear()
        self.console_output.append(
            "<b><span style='color:purple;'>Running EVALUATE...</span></b><br>"
        )
        self.progress.show()
        self.progress.setRange(0, 0)

        self.worker = RobertWorker(command, run_dir)
        self.worker.output_received.connect(self.console_output.append)
        self.worker.error_received.connect(self.console_output.append)
        self.worker.process_finished.connect(self._on_process_finished)
        self.worker.start()

    def stop_run(self):
        if self.worker:
            self.manual_stop = True
            self.worker.stop()

    def _reset_ui_after_process(self):
        self.run_button.setDisabled(False)
        self.stop_button.setDisabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()

    def _on_process_finished(self, exit_code):
        if self.worker:
            if self.worker.process and self.worker.process.poll() is None:
                self.worker.stop()
            self.worker = None

        if self.manual_stop or exit_code == -1:
            self.manual_stop = False
            self._reset_ui_after_process()
            QMessageBox.information(self, "EVALUATE", "EVALUATE has been stopped.")
            return

        self._reset_ui_after_process()

        if exit_code == 0:
            self.console_output.append(
                "<b><span style='color:green;'>EVALUATE finished successfully.</span></b>"
            )
            self._refresh_sibling_tabs()
            QMessageBox.information(
                self, "EVALUATE",
                "EVALUATE finished successfully.\n\n"
                "Check the 'Reports' tab for the PDF and the 'Predictions' tab for the "
                "prediction dashboard."
            )
        else:
            self.console_output.append(
                "<b><span style='color:red;'>EVALUATE finished with errors "
                f"(exit code {exit_code}). Check the log above.</span></b>"
            )

    def _refresh_sibling_tabs(self):
        """Refreshes the Reports/Predictions/Images tabs, mirroring the main ROBERT tab."""
        main_window = self.window()
        run_dir = os.path.dirname(self.csv_path) if self.csv_path else None
        if not run_dir:
            return

        for attr in ("results_tab", "images_tab", "predictions_tab"):
            tab = getattr(main_window, attr, None)
            if tab is not None and hasattr(tab, "refresh_with_new_path"):
                try:
                    tab.refresh_with_new_path(run_dir)
                except Exception:
                    pass
