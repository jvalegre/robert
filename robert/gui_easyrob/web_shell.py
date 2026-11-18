"""Web presentation for easyROB controls backed by the existing Qt workflow."""

import base64
import csv
from io import BytesIO
import json
import html
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from PySide6.QtCore import QBuffer, QIODevice, QObject, QTimer, QUrl, QUrlQuery, Qt, Signal, Slot
from PySide6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QLineEdit, QWidget

try:
    from gui_easyrob.web_results import WebResultsService
    from gui_easyrob.bot.bot_panel import _render_markdown
except ImportError:
    from robert.gui_easyrob.web_results import WebResultsService
    from robert.gui_easyrob.bot.bot_panel import _render_markdown


FIELD_GROUPS = {
    "workflow": (
        ("y_dropdown", "Target column"),
        ("type_dropdown", "Prediction type"),
        ("names_dropdown", "Name column"),
        ("workflow_selector", "Workflow"),
        ("aqme_workflow", "Calculate descriptors with AQME"),
        ("all_models_toggle", "Report every selected model"),
    ),
    "general": (
        ("auto_type", "Automatic prediction type"),
        ("seed", "Random seed"),
        ("kfold", "Cross-validation folds"),
        ("repeat_kfolds", "Cross-validation repeats"),
        ("split", "Split method"),
    ),
    "curate": (
        ("categoricalstr", "Categorical encoding"),
        ("corr_filter_xbool", "Descriptor correlation filter"),
        ("corr_filter_ybool", "Target correlation filter"),
        ("rfecv_filterbool", "RFECV filter"),
        ("desc_thresfloat", "Descriptor threshold"),
        ("thres_xfloat", "Descriptor correlation threshold"),
        ("thres_yfloat", "Target correlation threshold"),
    ),
    "generate": (
        ("error_type", "Optimization metric"),
        ("init_points", "Initial points"),
        ("n_iter", "Optimization iterations"),
        ("expect_improv", "Expected improvement"),
        ("pfi_filter", "PFI filter"),
        ("pfi_epochs", "PFI epochs"),
        ("pfi_threshold", "PFI threshold"),
        ("pfi_max", "Maximum PFI features"),
        ("auto_test", "Automatic test split"),
        ("test_set", "Test set size"),
    ),
    "predict": (
        ("t_value", "t value"),
        ("shap_show", "SHAP plots"),
        ("pfi_show", "PFI plots"),
    ),
    "aqme": (
        ("atoms", "QDESCP atoms"),
        ("descriptor_level", "Descriptor level"),
        ("solvent", "Solvent"),
    ),
    "checkml": (
        ("y_dropdown", "Target column"),
        ("names_dropdown", "Name column"),
        ("type_dropdown", "Prediction type"),
        ("model_source_dropdown", "Model source"),
        ("sklearn_model_dropdown", "Scikit-learn model"),
    ),
}


class WorkflowBridge(QObject):
    """Translate a small allowlisted web API into existing Qt widget actions."""

    stateChanged = Signal(str)
    logChanged = Signal(str)
    chemdrawReady = Signal(str)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._results = WebResultsService(window)
        self._last_state = ""
        self._log_snapshots = {"workflow": "", "checkml": ""}
        self._log_revisions = {"workflow": None, "checkml": None}
        self._molecule_cache_key = None
        self._molecule_image = ""
        self._chemdraw_rows = []
        self._timer = QTimer(self)
        self._timer.setInterval(1200)
        self._timer.timeout.connect(self.push_state)
        self._timer.start()

    def _owner(self, group):
        if group == "workflow":
            return self.window
        if group in {"general", "curate", "generate", "predict"}:
            return self.window.options_tab
        if group == "aqme":
            return self.window.tab_widget_aqme
        if group == "checkml":
            return self.window.checkml_tab
        return None

    def _widget(self, key):
        group, separator, name = key.partition(".")
        if not separator or name not in {item[0] for item in FIELD_GROUPS.get(group, ())}:
            return None
        return getattr(self._owner(group), name, None)

    @staticmethod
    def _field(key, label, widget):
        field = {"key": key, "label": label, "enabled": widget.isEnabled()}
        if isinstance(widget, QComboBox):
            field.update({
                "type": "select",
                "value": widget.currentText(),
                "options": [widget.itemText(index) for index in range(widget.count())],
            })
        elif isinstance(widget, QLineEdit):
            field.update({"type": "text", "value": widget.text(), "placeholder": widget.placeholderText()})
        else:
            field.update({"type": "toggle", "value": widget.isChecked()})
        return field

    def _molecule_state(self):
        tab = self.window.tab_widget_aqme
        pixmap = tab.mol_viewer.pixmap()
        cache_key = pixmap.cacheKey() if pixmap is not None and not pixmap.isNull() else None
        if cache_key != self._molecule_cache_key:
            self._molecule_cache_key = cache_key
            self._molecule_image = ""
            if cache_key is not None:
                buffer = QBuffer()
                buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                if pixmap.save(buffer, "PNG"):
                    self._molecule_image = "data:image/png;base64," + bytes(buffer.data().toBase64()).decode("ascii")
        def plain_text(value):
            return html.unescape(re.sub(r"<[^>]*>", "", value or ""))
        return {
            "image": self._molecule_image,
            "message": plain_text(tab.mol_viewer.text()) if not self._molecule_image else "",
            "info": plain_text(tab.mol_info_label.text()),
            "selectedAtoms": len(tab.selected_atoms),
            "canSelect": bool(self._molecule_image and tab.smarts_targets)
            and not (getattr(tab, "metal_found", False) and getattr(tab, "multiple_matches_detected", False)),
        }

    def _bot_state(self):
        panel = self.window.bot_panel
        return {
            "mode": panel.mode_combo.currentText(),
            "modes": [panel.mode_combo.itemText(index) for index in range(panel.mode_combo.count())],
            "provider": panel.provider_combo.currentText(),
            "providers": [panel.provider_combo.itemText(index) for index in range(panel.provider_combo.count())],
            "webSearch": panel.web_search_checkbox.isChecked(),
            "apiSaved": bool(panel.delete_api_button.property("has_saved_api")),
            "draft": panel.input_edit.text(),
            "canSend": panel.send_button.isEnabled(),
            "busy": bool(panel._busy),
            "messages": [
                {"role": role, "content": content, "html": _render_markdown(content)}
                for role, content in panel._message_records[-80:]
            ],
            "lastUsage": panel.last_usage_label.text(),
            "sessionUsage": panel.session_usage_label.text(),
            "requestStage": panel.request_stage_label.text(),
            "suggestions": [
                {"label": panel.suggested_combo.itemText(index), "index": index}
                for index in range(1, panel.suggested_combo.count())
            ],
            "canSummarize": panel.workflow_summary_button.isEnabled(),
        }

    def _state(self):
        window = self.window
        groups = {}
        for group, entries in FIELD_GROUPS.items():
            owner = self._owner(group)
            groups[group] = [
                self._field(f"{group}.{name}", label, getattr(owner, name))
                for name, label in entries
                if getattr(owner, name, None) is not None
                and (group != "checkml" or name != "sklearn_model_dropdown"
                     or window.checkml_tab.model_source_dropdown.currentIndex() == 1)
            ]
        groups["generate"].extend(
            self._field(f"model.{name}", name, widget)
            for name, widget in window.options_tab.modellist.items()
        )
        checkml = window.checkml_tab
        if checkml.model_source_dropdown.currentIndex() == 1:
            groups["checkml"].extend(
                self._field(f"checkml_param.{name}", name, widget)
                for name, widget in checkml.sklearn_param_fields.items()
            )
        progress = getattr(window, "workflow_progress", None)
        return {
            "groups": groups,
            "files": {
                "train": window.file_path or "",
                "test": window.csv_test_path or "",
                "checkmlTrain": checkml.csv_path or "",
                "checkmlTest": checkml.csv_test_path or "",
                "checkmlParams": checkml.model_params_path or "",
                "checkmlModel": checkml.model_file_path or "",
            },
            "columns": {
                "available": [window.available_list.item(index).text() for index in range(window.available_list.count())],
                "ignored": [window.ignore_list.item(index).text() for index in range(window.ignore_list.count())],
            },
            "checkmlColumns": [
                {"name": checkml.ignore_list.item(index).text(),
                 "ignored": checkml.ignore_list.item(index).checkState() == Qt.Checked}
                for index in range(checkml.ignore_list.count())
            ],
            "checkmlSource": checkml.model_source_dropdown.currentIndex(),
            "actions": {
                "run": window.run_button.isEnabled(),
                "runAqme": window.run_aqme_button.isEnabled(),
                "stop": window.stop_button.isEnabled(),
                "checkmlRun": checkml.run_button.isEnabled(),
                "checkmlStop": checkml.stop_button.isEnabled(),
                "results": window.results_workspace.has_available_views(),
                "aqme": window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.tab_widget_aqme)),
            },
            "results": {
                "revision": getattr(window, "_results_revision", 0),
                "available": dict(window.results_workspace._available),
                "models": [
                    {"label": window.results_workspace.model_selector.itemText(index),
                     "value": window.results_workspace.model_selector.itemData(index) or ""}
                    for index in range(window.results_workspace.model_selector.count())
                ],
                "selectedModel": window.results_workspace.current_model() or "",
                "predictionFiles": [
                    {"key": key, "label": key.replace("_", " ")}
                    for key in window.predictions_tab.csv_paths
                ],
                "files": {
                    "Report": [Path(path).name for path in window.results_tab.pdf_tabs],
                    "Predictions": [Path(path).name for path in window.predictions_tab.csv_paths.values()],
                    "Interactive plots": [Path(path).name for path in window.interactive_plots.paths],
                },
                "gallery": window.images_tab.gallery_state,
            },
            "molecule": self._molecule_state(),
            "bot": self._bot_state(),
            "progress": [label.text() for label in window.workflow_stage_status_labels],
            "progressStates": list(progress.statuses()) if progress else ["idle"] * 5,
            "aqmeProgress": progress.aqme_status() if progress else (
                "idle" if window.aqme_workflow.isChecked() else "inactive"
            ),
            "activity": "" if window.workflow_activity_label.isHidden() else window.workflow_activity_label.text(),
        }

    @Slot(result=str)
    def getState(self):
        state = self._state()
        state["log"] = self.window.console_output.toPlainText()
        state["checkmlLog"] = self.window.checkml_tab.console_output.toPlainText()
        return json.dumps(state, ensure_ascii=False)

    @Slot(result=str)
    def getLogs(self):
        return json.dumps({
            "workflow": self.window.console_output.toPlainText(),
            "checkml": self.window.checkml_tab.console_output.toPlainText(),
        }, ensure_ascii=False)

    def _push_logs(self):
        editors = {
            "workflow": self.window.console_output,
            "checkml": self.window.checkml_tab.console_output,
        }
        for kind, editor in editors.items():
            revision = editor.document().revision()
            if revision == self._log_revisions[kind]:
                continue
            self._log_revisions[kind] = revision
            current = editor.toPlainText()
            previous = self._log_snapshots[kind]
            if current == previous:
                continue
            replace = not current.startswith(previous)
            payload = {
                "kind": kind,
                "offset": 0 if replace else len(previous),
                "text": current if replace else current[len(previous):],
                "replace": replace,
            }
            self._log_snapshots[kind] = current
            self.logChanged.emit(json.dumps(payload, ensure_ascii=False))

    @Slot(result=str)
    def getTutorials(self):
        root = Path(__file__).resolve().parent / "tutorials"
        topics = self.window.tutorial_topics()
        catalog = []
        for slug, title in topics:
            blocks = self.window.load_tutorial(slug)
            images = sorted(
                (root / "tutorial_images" / slug).glob(f"{slug}_*.png"),
                key=lambda path: tuple(
                    int(segment) if segment.isdigit() else segment
                    for segment in path.stem.split("_")
                ),
            )
            catalog.append({
                "id": slug, "title": title,
                "steps": [
                    {"image": QUrl.fromLocalFile(str(path)).toString(),
                     "html": blocks[index] if index < len(blocks) else ""}
                    for index, path in enumerate(images)
                ],
            })
        return json.dumps(catalog, ensure_ascii=False)

    @Slot(int, int, float, result=str)
    def getReportPage(self, report_index, page_index, zoom):
        return json.dumps(self._results.report_page(report_index, page_index, zoom))

    @Slot(str, str, str, bool, int, result=str)
    def getPredictionTable(self, key, query, sort_column, descending, page_index):
        try:
            result = self._results.prediction_table(key, query, sort_column, descending, page_index)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            result = {"error": f"Unable to load predictions: {error}"}
        return json.dumps(result, ensure_ascii=False)

    @Slot(str, str, result=str)
    def getPredictionHistogram(self, key, column):
        try:
            result = self._results.prediction_histogram(key, column)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            result = {"error": f"Unable to load histogram: {error}"}
        return json.dumps(result, ensure_ascii=False)

    @Slot(str, result=str)
    def getPredictionDiagnostics(self, key):
        try:
            result = self._results.prediction_diagnostics(key)
        except (OSError, ValueError, KeyError) as error:
            result = {"error": f"Unable to load diagnostics: {error}"}
        return json.dumps(result, ensure_ascii=False)

    @Slot(int, result=str)
    def getPlotData(self, index):
        try:
            result = self._results.plot_data(index)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            result = {"error": f"Unable to load plot: {error}"}
        return json.dumps(result, ensure_ascii=False)

    @Slot(int, int, bool, result=str)
    def getPlotMolecule(self, plot_index, row_index, williams):
        return self._results.plot_molecule(plot_index, row_index, williams)

    def _show_chemdraw_editor(self, molecules):
        rows = []
        for molecule in molecules:
            smiles = Chem.MolToSmiles(molecule, canonical=False)
            image = ""
            try:
                buffer = BytesIO()
                Draw.MolToImage(molecule, size=(100, 100)).save(buffer, format="PNG")
                image = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            except (RuntimeError, ValueError):
                pass
            rows.append({"image": image, "SMILES": smiles, "code_name": "", "target": ""})
        self._chemdraw_rows = rows
        self.chemdrawReady.emit(json.dumps({"rows": rows}, ensure_ascii=False))

    @Slot(str, result=str)
    def saveChemdrawCsv(self, payload):
        try:
            document = json.loads(payload)
            rows = document["rows"]
            optional = document["columns"]
            target_column = str(document["targetColumn"]).strip()
        except (ValueError, KeyError, TypeError):
            return json.dumps({"error": "Molecule table is invalid"})
        allowed = ("charge", "mult", "complex_type", "sample", "geom")
        if not isinstance(rows, list) or not rows or len(rows) != len(self._chemdraw_rows):
            return json.dumps({"error": "Molecule rows do not match the imported file"})
        if (not isinstance(optional, list) or any(not isinstance(name, str) for name in optional)
                or len(set(optional)) != len(optional) or any(name not in allowed for name in optional)):
            return json.dumps({"error": "Optional columns are invalid"})
        if not target_column or target_column in {"SMILES", "code_name", *optional}:
            return json.dumps({"error": "Choose a unique target column name"})
        identifiers = []
        normalized = []
        complex_types = {"squareplanar", "squarepyramidal", "linear", "trigonalplanar"}
        for row in rows:
            if not isinstance(row, dict):
                return json.dumps({"error": "Molecule row is invalid"})
            smiles = str(row.get("SMILES", "")).strip()
            identifier = str(row.get("code_name", "")).strip()
            target = str(row.get("target", "")).strip()
            if not smiles:
                return json.dumps({"error": "Please fill in all SMILES fields"})
            if not identifier:
                return json.dumps({"error": "Please fill in all code_name fields"})
            if not target:
                return json.dumps({"error": "Target column is empty"})
            try:
                float(target)
            except ValueError:
                return json.dumps({"error": "Target column must be numeric"})
            values = [smiles, identifier, target]
            for name in optional:
                value = str(row.get(name, "")).strip()
                if name in {"charge", "mult", "sample"} and (not value or not value.lstrip("-").isdigit()):
                    return json.dumps({"error": f"Column {name} must be an integer"})
                if name == "complex_type" and value not in complex_types:
                    return json.dumps({"error": "Select a complex_type for every molecule"})
                if name == "geom" and not value:
                    return json.dumps({"error": "Column geom cannot be empty"})
                values.append(value)
            identifiers.append(identifier)
            normalized.append(values)
        if len(set(identifiers)) != len(identifiers):
            return json.dumps({"error": "code_name values must be unique"})
        path, _ = QFileDialog.getSaveFileName(self.window, "Save CSV", "", "CSV Files (*.csv)")
        if not path:
            return json.dumps({"cancelled": True})
        if " " in Path(path).name:
            return json.dumps({"error": "Choose a CSV filename without spaces so ROBERT can load it."})
        try:
            with open(path, "w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(["SMILES", "code_name", target_column, *optional])
                writer.writerows(normalized)
        except OSError as error:
            return json.dumps({"error": f"Unable to save CSV: {error}"})
        if self.window.set_file_path(path) is False:
            return json.dumps({"error": f"CSV was saved but could not be loaded: {path}"}, ensure_ascii=False)
        return json.dumps({"saved": path}, ensure_ascii=False)

    def push_state(self):
        worker = getattr(self.window, "worker", None)
        checkml_worker = getattr(self.window.checkml_tab, "worker", None)
        running = any(
            item is not None and item.isRunning()
            for item in (worker, checkml_worker)
        ) or bool(getattr(self.window, "aqme_is_running", False))
        interval = 400 if running else 1200
        if self._timer.interval() != interval:
            self._timer.setInterval(interval)
        state = json.dumps(self._state(), ensure_ascii=False)
        if state != self._last_state:
            self._last_state = state
            self.stateChanged.emit(state)
        self._push_logs()

    @Slot(str, str)
    def setField(self, key, value):
        if key.startswith("model."):
            widget = self.window.options_tab.modellist.get(key[6:])
        elif key.startswith("checkml_param."):
            widget = self.window.checkml_tab.sklearn_param_fields.get(key[14:])
        else:
            widget = self._widget(key)
        if widget is None or not widget.isEnabled():
            return
        if isinstance(widget, QComboBox):
            index = widget.findText(value)
            if index >= 0:
                widget.setCurrentIndex(index)
        elif isinstance(widget, QLineEdit):
            widget.setText(value)
        elif isinstance(widget, QCheckBox) or hasattr(widget, "setChecked"):
            widget.setChecked(value == "true")
        self.push_state()

    @Slot(str)
    def setBotDraft(self, value):
        if self.window.app_mode == "checkml":
            return
        panel = self.window.bot_panel
        if panel._busy:
            return
        if panel.input_edit.text() != value:
            panel._selected_tutorial_id = ""
            panel.input_edit.setText(value)
        self.push_state()

    @Slot(str)
    def setBotApiKey(self, value):
        if self.window.app_mode == "checkml":
            return
        panel = self.window.bot_panel
        if panel.api_key_edit.isEnabled():
            panel.api_key_edit.setText(value)

    def _choose_file(self, title, file_filter, setter):
        path, _ = QFileDialog.getOpenFileName(self.window, title, "", file_filter)
        if path:
            setter(path)

    @Slot(str, str)
    def performAction(self, action, value):
        window = self.window
        if window.app_mode == "checkml" and action in {
            "botOpen", "botMode", "botProvider", "botWebSearch", "botSuggestion",
            "botSend", "botClear", "botSaveApi", "botDeleteApi", "botResetUsage",
            "botSummarize", "openBot", "openBotLink",
        }:
            return
        checkml = window.checkml_tab
        panel = window.bot_panel
        if action == "activePage" and value in {
            "workflow", "advanced", "aqme", "checkml", "results"
        } and (window.app_mode != "checkml" or value in {"checkml", "results"}):
            window.web_active_page = {
                "workflow": "Workflow",
                "advanced": "Advanced options",
                "aqme": "AQME descriptors",
                "checkml": "CheckML",
                "results": "Results",
            }[value]
            if value != "results":
                window.web_active_result_view = ""
        elif action == "activeResultView" and value in {
            "", "Report", "Predictions", "Images", "Interactive plots"
        }:
            window.web_active_result_view = value
        elif action == "chooseTrain":
            window.select_file()
        elif action == "chooseTest":
            window.select_csv_test_file()
        elif action == "clearTest":
            window.clear_test_file()
        elif action == "chooseCheckmlTrain":
            self._choose_file("Select CHECKML CSV", "CSV Files (*.csv)", checkml.set_csv_path)
        elif action == "chooseCheckmlTest":
            self._choose_file("Select external test CSV", "CSV Files (*.csv)", checkml.set_csv_test_path)
        elif action == "clearCheckmlTest":
            checkml.clear_csv_test_path()
        elif action == "chooseCheckmlParams":
            self._choose_file("Select model parameters", "CSV Files (*.csv)", checkml.set_model_params_path)
        elif action == "chooseCheckmlModel":
            self._choose_file("Select saved model", "Model files (*.joblib *.pkl *.pickle)", checkml.set_model_file_path)
        elif action == "run" and window.run_button.isEnabled():
            window.run_button.click()
        elif action == "runAqme" and window.run_aqme_button.isEnabled():
            window.run_aqme_button.click()
        elif action == "stop" and window.stop_button.isEnabled():
            window.stop_button.click()
        elif action == "checkmlRun" and checkml.run_button.isEnabled():
            checkml.run_button.click()
        elif action == "checkmlStop" and checkml.stop_button.isEnabled():
            checkml.stop_button.click()
        elif action in {"ignore", "restore"}:
            source = window.available_list if action == "ignore" else window.ignore_list
            source.clearSelection()
            for index in range(source.count()):
                item = source.item(index)
                if item.text() == value:
                    item.setSelected(True)
                    break
            if action == "ignore":
                window.move_to_selected()
            else:
                window.move_to_available()
        elif action == "toggleCheckmlIgnore":
            for index in range(checkml.ignore_list.count()):
                item = checkml.ignore_list.item(index)
                if item.text() == value:
                    item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
                    break
        elif action == "setResultModel":
            selector = window.results_workspace.model_selector
            index = selector.findData(value or None)
            if index >= 0:
                selector.setCurrentIndex(index)
        elif action == "selectAtom":
            try:
                x, y = (int(float(position)) for position in value.split(",", 1))
            except (ValueError, TypeError):
                return
            molecule = self._molecule_state()
            if molecule["canSelect"] and 0 <= x < 400 and 0 <= y < 400:
                atom = window.tab_widget_aqme.get_atom_at_position(x, y)
                if atom is not None:
                    window.tab_widget_aqme.handle_atom_selection(atom)
        elif action == "openChemdraw":
            self._choose_file(
                "Select ChemDraw or SDF file",
                "ChemDraw Files (*.sdf *.cdxml *.mol *.cdx)",
                lambda path: window.tab_widget_aqme.load_chemdraw_file(path, self._show_chemdraw_editor),
            )
        elif action == "imageAction":
            image_id, separator, image_action = value.partition("|")
            if separator and image_action in {"open", "copy", "save", "folder"}:
                window.images_tab._bridge.performAction(image_id, image_action)
        elif action == "botOpen":
            window._ensure_bot_welcome_message()
        elif action == "botMode":
            if panel.mode_combo.isEnabled() and value in {panel.mode_combo.itemText(index) for index in range(panel.mode_combo.count())}:
                panel.mode_combo.setCurrentText(value)
        elif action == "botProvider":
            if panel.provider_combo.isEnabled() and panel.provider_combo.findText(value) >= 0:
                panel.provider_combo.setCurrentText(value)
        elif action == "botWebSearch":
            if panel.web_search_checkbox.isEnabled():
                panel.web_search_checkbox.setChecked(value == "true")
        elif action == "botSuggestion":
            try:
                index = int(value)
            except ValueError:
                return
            if 0 < index < panel.suggested_combo.count():
                panel.suggested_combo.setCurrentIndex(index)
        elif action == "botSend" and panel.send_button.isEnabled():
            panel.send_button.click()
        elif action == "botClear":
            window.clear_bot_history()
        elif action == "botSaveApi" and panel.save_api_button.isEnabled():
            panel.save_api_button.click()
        elif action == "botDeleteApi" and panel.delete_api_button.isEnabled():
            panel.delete_api_button.click()
        elif action == "botResetUsage":
            panel.reset_usage_button.click()
        elif action == "botSummarize" and panel.workflow_summary_button.isEnabled():
            panel.workflow_summary_button.click()
        elif action == "openBotLink":
            url = QUrl(value)
            if url.isValid() and url.scheme().lower() in {"http", "https"} and url.host():
                window.open_external_url(value)
        elif action == "openMolssi":
            window.show_molssi_workspace()
        elif action == "molssiHome":
            window.molssi_tab.web_view.setUrl(window.molssi_tab.databases_home_url)
        elif action == "openHelp" and value in {"general", "curate", "generate", "predict"}:
            window.options_tab.go_to_help_section(value)
        elif action == "openHelp" and value == "aqme":
            window.tab_widget_aqme.go_to_help_section("AQME")
        elif action == "checkmlParamHelp" and checkml.model_source_dropdown.currentIndex() == 2:
            checkml.model_params_info.click()
        elif action == "openContact":
            window.show_contact_dialog()
        elif action == "openVersion":
            window.show_version_dialog()
        elif action == "openYouTube":
            window.open_external_url("https://www.youtube.com/@thealegregroup4964/videos")
        elif action == "openAlmos":
            window.open_external_url("https://github.com/MiguelMartzFdez/almos")
        elif action == "openClassic":
            window.show_classic_panel(value)
        elif action == "openResult" and value in {"Report", "Predictions", "Images", "Interactive plots"}:
            if window.results_workspace._available.get(value):
                window.show_result_view(value)
                window.show_classic_panel("Results")
        elif action == "openBot":
            window.bot_toggle_btn.setChecked(True)
        elif action == "openTutorial":
            window.show_tutorial_dialog()
        elif action == "openDocs":
            window.open_external_url("https://robert.readthedocs.io/en/latest/")
        elif action == "openGroqKeys":
            window.open_external_url("https://console.groq.com/keys")
        self.push_state()


class WebShell:
    """Attach the local web dashboard to an existing EasyROB window."""

    def __init__(self, window):
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineWidgets import QWebEngineView

        window.web_active_page = "CheckML" if window.app_mode == "checkml" else "Workflow"
        window.web_active_result_view = ""
        self.bridge = WorkflowBridge(window)
        self.view = QWebEngineView(window)
        self.channel = QWebChannel(self.view.page())
        self.channel.registerObject("workflow", self.bridge)
        self.view.page().setWebChannel(self.channel)
        url = QUrl.fromLocalFile(str(Path(__file__).resolve().parent / "web" / "app.html"))
        if window.app_mode == "checkml":
            query = QUrlQuery()
            query.addQueryItem("mode", "checkml")
            url.setQuery(query)
        self.view.setUrl(url)
