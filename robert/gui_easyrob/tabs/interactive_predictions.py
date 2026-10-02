"""Interactive prediction diagnostics for the Results view."""

from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.image import imread
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)
from rdkit import Chem
from rdkit.Chem import Draw


def discover_prediction_csvs(run_dir):
    """Find model outputs and optional external prediction outputs."""
    predict_dir = Path(run_dir) / "PREDICT"
    paths = list(predict_dir.glob("*.csv"))
    paths.extend((predict_dir / "csv_test").glob("*.csv"))
    return sorted(
        (path for path in paths
         if re.search(r"_(?:No_PFI|PFI)\.csv$", path.name)
         and not path.name.startswith("Results_boundary_williams_")),
        key=lambda path: (path.parent.name == "csv_test", path.name),
    )


def _identifier_column(frame, sources):
    reserved = {"Set", "SMILES"}
    for source in sources:
        for column in frame.columns:
            if (column in source.columns and column not in reserved
                    and not column.endswith(("_pred", "_pred_sd"))
                    and column.lower() not in {"smiles", "target"}
                    and source[column].is_unique):
                if column.lower() in {"code_name", "name", "id", "identifier"}:
                    return column
    return None


def _molecule_smiles(frame, sources):
    smiles_column = next((col for col in frame if col.lower() == "smiles"), None)
    if smiles_column:
        return [str(value) if pd.notna(value) and str(value).strip() else None
                for value in frame[smiles_column]]

    identifier = _identifier_column(frame, sources)
    if not identifier:
        return [None] * len(frame)
    result = [None] * len(frame)
    for source in sources:
        smiles_column = next((col for col in source if col.lower() == "smiles"), None)
        if not smiles_column or identifier not in source or not source[identifier].is_unique:
            continue
        lookup = dict(zip(source[identifier].astype(str), source[smiles_column]))
        for index, value in enumerate(frame[identifier].astype(str)):
            candidate = lookup.get(value)
            if result[index] is None and pd.notna(candidate) and str(candidate).strip():
                result[index] = str(candidate)
    return result


def prepare_prediction_data(frame, sources, reference=None):
    """Extract plot coordinates, using CV data as the diagnostic reference."""
    frame = frame.reset_index(drop=True)
    prediction_columns = [col for col in frame if col.endswith("_pred")]
    if not prediction_columns:
        raise ValueError("Prediction CSV has no predicted target column")
    target = prediction_columns[0][:-5]
    observed = pd.to_numeric(frame[target], errors="coerce") if target in frame else pd.Series(np.nan, index=frame.index)
    predicted = pd.to_numeric(frame[prediction_columns[0]], errors="coerce")
    residual = predicted - observed
    reference = frame if reference is None else reference
    reference_observed = pd.to_numeric(reference[target], errors="coerce") if target in reference else pd.Series(np.nan, index=reference.index)
    reference_predicted = pd.to_numeric(reference[prediction_columns[0]], errors="coerce")
    if "Set" in reference:
        cv_mask = reference["Set"].astype(str).str.casefold() == "cv"
        cv_errors = (reference_predicted[cv_mask] - reference_observed[cv_mask]).abs().dropna()
    else:
        cv_errors = (reference_predicted - reference_observed).abs().dropna()
    error_mean = float(cv_errors.mean()) if len(cv_errors) else 0.0
    error_sd = float(cv_errors.std(ddof=0)) if len(cv_errors) else 0.0
    if not np.isfinite(error_sd) or error_sd == 0:
        error_sd = 1.0
    outlier_score = ((predicted - observed).abs() - error_mean) / error_sd

    return {
        "frame": frame, "target": target, "observed": observed.to_numpy(dtype=float),
        "predicted": predicted.to_numpy(dtype=float), "residual": residual.to_numpy(dtype=float),
        "outlier_score": outlier_score.to_numpy(dtype=float),
        "smiles": _molecule_smiles(frame, sources),
    }


def _prediction_log(run_dir, model=None):
    """Find the log for a selected all-model run or a standard PREDICT run."""
    predict_dir = Path(run_dir) / "PREDICT"
    model_log = predict_dir / f"PREDICT_{model}_data.dat" if model else None
    return model_log if model_log is not None and model_log.is_file() else predict_dir / "PREDICT_data.dat"


def _source_frames(run_dir, selected_file, external=False, model=None):
    """Prefer the selected input; use the external input named in PREDICT_data.dat."""
    candidates = [Path(selected_file)]
    log_path = _prediction_log(run_dir, model)
    if log_path.exists():
        command = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'--csv_test\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', command)
        if match:
            candidates.append(Path(run_dir) / next(group for group in match.groups() if group))
    if external:
        candidates.reverse()
    sources = []
    for path in candidates:
        if path.is_file() and path.suffix.lower() == ".csv":
            try:
                source = pd.read_csv(path)
                if any(column.lower() == "smiles" for column in source):
                    sources.append(source)
            except (OSError, pd.errors.ParserError, UnicodeError):
                continue
    return sources


class InteractivePredictions(QWidget):
    """Linked prediction diagnostics with a molecular structure panel."""

    PLOTS = ("Observed vs predicted", "Outlier analysis", "Williams plot")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paths = []
        self.data = None
        self.williams_data = None
        self.williams_image = None
        self.williams_smiles = []
        self._artists = {}
        self.seed = 0
        self.cv_label = "CV"
        self.outlier_threshold = 2.0
        layout = QVBoxLayout(self)
        subtitle = QLabel("Hover over a point to inspect its molecule and prediction.")
        layout.addWidget(subtitle)
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self._load_selected)
        layout.addWidget(self.selector)
        content = QHBoxLayout()
        chart_column = QVBoxLayout()
        plot_selector = QHBoxLayout()
        self.charts = QStackedWidget()
        self.canvases = []
        self.plot_buttons = []
        for index, title in enumerate(self.PLOTS):
            button = QPushButton(title)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, plot=index: self._select_plot(plot))
            plot_selector.addWidget(button)
            self.plot_buttons.append(button)
            canvas = FigureCanvasQTAgg(Figure(figsize=(7, 5), constrained_layout=True))
            canvas.mpl_connect("motion_notify_event", lambda event, chart=canvas: self._hover(event, chart))
            canvas.mpl_connect("button_press_event", lambda event, chart=canvas: self._hover(event, chart))
            self.charts.addWidget(canvas)
            self.canvases.append(canvas)
        plot_selector.addStretch()
        chart_column.addLayout(plot_selector)
        chart_column.addWidget(self.charts, 1)
        content.addLayout(chart_column, 3)
        self._select_plot(0)
        detail = QVBoxLayout()
        detail.addWidget(QLabel("Datapoint under cursor"))
        self.structure = QLabel("Hover over a datapoint")
        self.structure.setAlignment(Qt.AlignCenter)
        self.structure.setFixedSize(260, 260)
        self.structure.setStyleSheet("background: white; color: #555; border: 1px solid #bbb;")
        detail.addWidget(self.structure)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        detail.addWidget(self.details)
        detail.addStretch()
        content.addLayout(detail, 1)
        layout.addLayout(content, 1)

    def _select_plot(self, index):
        """Switch plot views without adding another level of tabs."""
        self.charts.setCurrentIndex(index)
        for position, button in enumerate(self.plot_buttons):
            button.setChecked(position == index)

    def refresh(self, run_dir, selected_file, paths=None, catalog=None):
        self.run_dir = Path(run_dir)
        self.selected_file = selected_file
        self.catalog = catalog
        self.paths = list(paths) if paths is not None else discover_prediction_csvs(run_dir)
        log_path = _prediction_log(run_dir)
        self.outlier_threshold = 2.0
        if log_path.is_file():
            command = log_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"--t_value(?:\s+|=)([0-9.]+)", command)
            if match:
                self.outlier_threshold = float(match.group(1))
        self.selector.blockSignals(True)
        self.selector.clear()
        for path in self.paths:
            group = "External" if path.parent.name == "csv_test" else "PREDICT"
            self.selector.addItem(f"{group} / {path.stem}")
        self.selector.blockSignals(False)
        if self.paths:
            self._load_selected(0)
        else:
            self.data = None
            self.williams_data = None
            self.williams_image = None
            self._artists.clear()
            self.structure.setPixmap(QPixmap())
            self.structure.setText("No prediction data available")
            self.details.clear()
            for canvas in self.canvases:
                canvas.figure.clear()
                canvas.draw_idle()

    def _load_selected(self, index):
        if index < 0 or index >= len(self.paths):
            return
        path = self.paths[index]
        try:
            frame = pd.read_csv(path)
            external = path.parent.name == "csv_test"
            reference = None
            if external:
                candidates = [candidate for candidate in self.paths
                              if candidate.parent.name == "PREDICT"
                              and path.stem.endswith(candidate.stem)]
                if candidates:
                    reference = pd.read_csv(max(candidates, key=lambda item: len(item.stem)))
            model_path = max(candidates, key=lambda item: len(item.stem)) if external and candidates else path
            suffix = "No_PFI" if model_path.stem.endswith("_No_PFI") else "PFI"
            model = model_path.stem.removesuffix(f"_{suffix}")
            model_log = _prediction_log(self.run_dir, model)
            if model_log.is_file():
                command = model_log.read_text(encoding="utf-8", errors="replace")
                match = re.search(r"--t_value(?:\s+|=)([0-9.]+)", command)
                if match:
                    self.outlier_threshold = float(match.group(1))
            metadata_name = model if suffix == "No_PFI" else f"{model}_PFI"
            metadata = self.run_dir / "GENERATE" / "Best_model" / suffix / f"{metadata_name}.csv"
            if not metadata.is_file():
                metadata = (
                    self.run_dir / "GENERATE" / "All_models" / suffix
                    / model / f"{metadata_name}.csv"
                )
            if metadata.is_file():
                model_info = pd.read_csv(metadata).iloc[0]
                self.seed = int(model_info["seed"])
                self.cv_label = f"{int(model_info['repeat_kfolds'])}x {int(model_info['kfold'])}-fold CV"
            else:
                self.seed = 0
                self.cv_label = "CV"
            self.data = prepare_prediction_data(
                frame,
                _source_frames(self.run_dir, self.selected_file, external, model),
                reference,
            )
            self._load_williams(model_path)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            self.details.setText(f"Unable to load prediction data: {error}")
            return
        self.structure.setPixmap(QPixmap())
        self.structure.setText("Hover over a datapoint")
        self.details.setText("")
        self._draw_charts()

    def _load_williams(self, model_path):
        """Read the exact coordinates saved with the selected model's Williams PNG."""
        stem = f"Results_boundary_williams_{model_path.stem}"
        sidecar = self.run_dir / "PREDICT" / f"{stem}.csv"
        image = self.run_dir / "PREDICT" / f"{stem}.png"
        self.williams_data = None
        self.williams_image = image if image.is_file() else None
        self.williams_smiles = []
        if not sidecar.is_file():
            return
        table = pd.read_csv(sidecar, dtype={"point_id": str})
        required = {"point_id", "name_column", "group", "leverage",
                    "standardized_residual", "h_star", "observed", "predicted"}
        if not required.issubset(table.columns):
            raise ValueError(f"Williams data is missing columns: {', '.join(sorted(required - set(table.columns)))}")
        self.williams_data = table.reset_index(drop=True)
        name_column = str(table["name_column"].iloc[0]) if len(table) else ""
        point_ids = table["point_id"].astype(str)
        smiles = [None] * len(table)
        for source in _source_frames(self.run_dir, self.selected_file):
            if name_column not in source or not source[name_column].is_unique:
                continue
            smiles_column = next((column for column in source if column.lower() == "smiles"), None)
            if smiles_column is None:
                continue
            lookup = dict(zip(source[name_column].astype(str), source[smiles_column]))
            for index, point_id in enumerate(point_ids):
                candidate = lookup.get(point_id)
                if smiles[index] is None and pd.notna(candidate) and str(candidate).strip():
                    smiles[index] = str(candidate)
        self.williams_smiles = smiles

    def _draw_charts(self):
        data = self.data
        self._artists.clear()
        for plot_index, canvas in enumerate(self.canvases):
            figure = canvas.figure
            figure.clear()
            axis = figure.add_subplot(111)
            axis.grid(linestyle="--", linewidth=0.8, alpha=0.5)
            axis.set_axisbelow(True)
            if plot_index == 2:
                self._draw_williams(axis)
                canvas.draw_idle()
                continue
            if plot_index == 0:
                x, y = data["observed"], data["predicted"]
                axis.set(xlabel=data["target"], ylabel=f"Predicted {data['target']}",
                         title=f"Predictions CV and test set of {self.paths[self.selector.currentIndex()].stem}")
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.any():
                    external = self.paths[self.selector.currentIndex()].parent.name == "csv_test"
                    cv_mask = data["frame"]["Set"].eq("CV").to_numpy() if "Set" in data["frame"] else valid
                    fit_mask = valid if external else valid & cv_mask
                    if fit_mask.sum() >= 10:
                        sns.regplot(x=x[fit_mask], y=y[fit_mask], scatter=False,
                                    color=".1", truncate=True, seed=self.seed, ax=axis)
                    padding = 0.1 * np.ptp(x[valid])
                    lower = min(np.min(x[valid]), np.min(y[valid])) - padding
                    upper = max(np.max(x[valid]), np.max(y[valid])) + padding
                    axis.set_xlim(lower, upper)
                    axis.set_ylim(lower, upper)
                    if external:
                        sd_column = f"{data['target']}_pred_sd"
                        if sd_column in data["frame"]:
                            errors = pd.to_numeric(data["frame"][sd_column], errors="coerce").to_numpy(dtype=float)
                            error_mask = valid & np.isfinite(errors)
                            axis.errorbar(x[error_mask], y[error_mask], yerr=errors[error_mask],
                                          fmt="none", ecolor="gray", capsize=3, zorder=1)
            elif plot_index == 1:
                x = y = data["outlier_score"]
                axis.set(xlabel="SD of the errors", ylabel="SD of the errors",
                         title=f"Outlier analysis of {self.paths[self.selector.currentIndex()].stem}")
                valid = np.isfinite(x)
                limit = max(2.5, float(np.max(np.abs(x[valid])) + 0.5)) if valid.any() else 2.5
                threshold = self.outlier_threshold
                if limit > threshold:
                    width = limit - threshold
                    axis.add_patch(Rectangle((threshold, threshold), width, width,
                                             facecolor="grey", alpha=0.3))
                    axis.add_patch(Rectangle((-limit, -limit), width, width,
                                             facecolor="grey", alpha=0.3))
                axis.set_xlim(-limit, limit)
                axis.set_ylim(-limit, limit)
            sets = data["frame"]["Set"].fillna("Data").astype(str) if "Set" in data["frame"] else pd.Series(["External" if self.paths[self.selector.currentIndex()].parent.name == "csv_test" else "Data"] * len(x))
            for label, color in (("CV", "b"), ("Test", "r"), ("External", "r"), ("Data", "b")):
                indices = np.flatnonzero((sets.to_numpy() == label) & np.isfinite(x) & np.isfinite(y))
                if len(indices):
                    legend_label = self.cv_label if label == "CV" else label.lower()
                    artist = axis.scatter(x[indices], y[indices], label=legend_label, color=color,
                                          edgecolors="black", linewidths=0.8, s=50, picker=6, zorder=2)
                    self._artists[artist] = indices
            if not (np.isfinite(x) & np.isfinite(y)).any():
                axis.text(0.5, 0.5, "Observed values are unavailable for this plot",
                          ha="center", va="center", transform=axis.transAxes)
            if plot_index != 1 and axis.get_legend_handles_labels()[0]:
                axis.legend(loc="best", fontsize=8)
            canvas.draw_idle()

    def _draw_williams(self, axis):
        table = self.williams_data
        if table is None:
            axis.set_title("Williams plot: interactive data unavailable")
            axis.set_axis_off()
            if self.williams_image is not None:
                try:
                    axis.imshow(imread(self.williams_image))
                except (OSError, ValueError):
                    axis.text(0.5, 0.5, "Williams image could not be read",
                              ha="center", va="center", transform=axis.transAxes)
            else:
                axis.text(0.5, 0.5, "No Williams plot is available for this model",
                          ha="center", va="center", transform=axis.transAxes)
            return

        axis.set(xlabel="Leverage", ylabel="Standardized residual",
                 title=f"Applicability domain (Williams plot) of {self.paths[self.selector.currentIndex()].stem}")
        leverage = pd.to_numeric(table["leverage"], errors="coerce").to_numpy(dtype=float)
        residual = pd.to_numeric(table["standardized_residual"], errors="coerce").to_numpy(dtype=float)
        threshold = pd.to_numeric(table["h_star"], errors="coerce").to_numpy(dtype=float)
        h_star = threshold[0] if len(threshold) else np.nan
        for group, color in (("Typical 80%", "b"), ("High leverage 20%", "r")):
            indices = np.flatnonzero((table["group"].to_numpy() == group)
                                     & np.isfinite(leverage) & np.isfinite(residual))
            if len(indices):
                artist = axis.scatter(leverage[indices], residual[indices], label=group,
                                      color=color, edgecolors="black", linewidths=0.8,
                                      s=50, picker=6, zorder=2)
                self._artists[artist] = indices
        for bound in (-3, 3):
            axis.axhline(bound, color="dimgray", linestyle="--", linewidth=1)
        if np.isfinite(h_star):
            axis.axvline(h_star, color="dimgray", linestyle="--", linewidth=1)
        valid = np.isfinite(leverage) & np.isfinite(residual)
        x_max = max(float(np.max(leverage[valid]) * 1.1) if valid.any() else 0, h_star * 1.2)
        y_max = max(float(np.max(np.abs(residual[valid])) * 1.1) if valid.any() else 0, 3.5)
        axis.set_xlim(0, x_max)
        axis.set_ylim(-y_max, y_max)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="best", fontsize=8)

    def _hover(self, event, canvas):
        if event.inaxes is None or self.data is None:
            return
        for artist, indices in self._artists.items():
            if artist.axes.figure is not canvas.figure:
                continue
            contains, info = artist.contains(event)
            if contains and len(info.get("ind", [])):
                index = int(indices[info["ind"][0]])
                if canvas is self.canvases[2]:
                    self._show_williams_point(index)
                else:
                    self._show_point(index)
                return

    def _show_williams_point(self, index):
        row = self.williams_data.iloc[index]
        self.details.setText("\n".join([
            str(row["point_id"]), f"Group: {row['group']}",
            f"Leverage: {row['leverage']:.4g}",
            f"Standardized residual: {row['standardized_residual']:.4g}",
            f"Observed: {row['observed']:.4g}", f"Predicted: {row['predicted']:.4g}",
        ]))
        smiles = self.williams_smiles[index]
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            self.structure.setPixmap(QPixmap())
            self.structure.setText("Molecule unavailable")
            return
        buffer = BytesIO()
        Draw.MolToImage(molecule, size=(320, 320)).save(buffer, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())
        self.structure.setPixmap(pixmap.scaled(240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_point(self, index):
        data = self.data
        row = data["frame"].iloc[index]
        identifier = next((row[col] for col in ("code_name", "Name", "name", "ID") if col in row), index + 1)
        lines = [str(identifier)]
        if "Set" in row:
            lines.append(f"Set: {row['Set']}")
        if np.isfinite(data["observed"][index]):
            lines.append(f"Observed: {data['observed'][index]:.4g}")
        lines.append(f"Predicted: {data['predicted'][index]:.4g}")
        for label, key in (("Residual", "residual"), ("Outlier score", "outlier_score")):
            value = data[key][index]
            if np.isfinite(value):
                lines.append(f"{label}: {value:.3g}")
        self.details.setText("\n".join(lines))
        smiles = data["smiles"][index]
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            self.structure.setPixmap(QPixmap())
            self.structure.setText("Molecule unavailable")
            return
        buffer = BytesIO()
        Draw.MolToImage(molecule, size=(320, 320)).save(buffer, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())
        self.structure.setPixmap(pixmap.scaled(240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
