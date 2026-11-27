"""Read-only adapters for presenting existing result data in the web workspace."""

import base64
from functools import lru_cache
from io import BytesIO
import math
from pathlib import Path

import pymupdf as fitz
import pandas as pd
import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QUrl
from rdkit import Chem
from rdkit.Chem import Draw

try:
    from utils.predictions_utils import (
        evaluate_predictions_for_model, extract_boundary_fragment,
        extract_boundary_scores, extract_robert_fragment_image,
        find_external_test_pixmaps, get_robert_report_path,
    )
except ImportError:
    from robert.gui_easyrob.utils.predictions_utils import (
        evaluate_predictions_for_model, extract_boundary_fragment,
        extract_boundary_scores, extract_robert_fragment_image,
        find_external_test_pixmaps, get_robert_report_path,
    )


def _number(value):
    """Return finite JSON numbers and map missing measurements to null."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _cell(value):
    if pd.isna(value):
        return ""
    return str(value)


def _prediction_name_column(frame, configured_name):
    """Use the configured identifier or a recognizable column in the displayed CSV."""
    columns = {str(column).casefold(): column for column in frame.columns}
    for candidate in (configured_name, "code_name", "name", "sample_id", "compound_id", "id"):
        if candidate and str(candidate).casefold() in columns:
            return columns[str(candidate).casefold()]
    return ""


@lru_cache(maxsize=512)
def _molecule_image(smiles, size=96):
    if not smiles or not isinstance(smiles, str):
        return ""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    stream = BytesIO()
    Draw.MolToImage(molecule, size=(size, size)).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _pixmap_image(pixmap):
    if pixmap is None or pixmap.isNull():
        return ""
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not pixmap.save(buffer, "PNG"):
        return ""
    return "data:image/png;base64," + base64.b64encode(bytes(buffer.data())).decode("ascii")


class WebResultsService:
    """Expose only files and rows already selected by the native result workspace."""

    def __init__(self, window):
        self.window = window
        self._table_cache = {}
        self._diagnostic_cache = {}

    def report_page(self, report_index, page_index, zoom):
        paths = list(self.window.results_tab.pdf_tabs)
        if report_index < 0 or report_index >= len(paths):
            return {"error": "Report is unavailable"}
        path = Path(paths[report_index])
        if not path.is_file():
            return {"error": "Report file is unavailable"}
        zoom = min(2.0, max(0.75, zoom))
        try:
            with fitz.open(path) as document:
                count = len(document)
                if count < 1:
                    return {"error": "Report has no pages"}
                page_index = min(count - 1, max(0, page_index))
                pixmap = document.load_page(page_index).get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom), alpha=False,
                )
                image = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
        except (OSError, ValueError, RuntimeError) as error:
            return {"error": f"Unable to render report: {error}"}
        return {"name": path.name, "page": page_index, "pages": count,
                "zoom": zoom, "image": "data:image/png;base64," + image}

    def prediction_table(self, key, query, sort_column, descending, page_index):
        tab = self.window.predictions_tab
        path = tab.csv_paths.get(key)
        if path is None or not Path(path).is_file():
            return {"error": "Prediction table is unavailable"}
        path = Path(path)
        report = tab._report_paths.get(key) or get_robert_report_path(tab._base_path, key)
        report_path = Path(report) if report else None
        cache_key = (str(path), path.stat().st_mtime_ns, str(report or ""),
                     report_path.stat().st_mtime_ns if report_path and report_path.is_file() else 0)
        if cache_key not in self._table_cache:
            frame = tab._filter_prediction_dataframe(pd.read_csv(path))
            info = evaluate_predictions_for_model(tab._base_path, frame, key, report_path=report)
            numeric = {}
            for column in frame.select_dtypes(include="number").columns:
                series = pd.to_numeric(frame[column], errors="coerce").dropna()
                numeric[column] = {"mean": _number(series.mean()), "min": _number(series.min()),
                                   "median": _number(series.median()),
                                   "max": _number(series.max())} if len(series) else {}
            self._table_cache = {cache_key: (frame, info, numeric)}
        frame, info, numeric = self._table_cache[cache_key]
        name_column = _prediction_name_column(frame, tab._extract_names_column_from_predict())
        filtered = frame
        if query and name_column:
            filtered = filtered[filtered[name_column].astype(str).str.contains(query, case=False, regex=False, na=False)]
        if sort_column in filtered:
            filtered = filtered.sort_values(sort_column, ascending=not descending,
                                            na_position="last", kind="mergesort")
        page_size = 30
        pages = max(1, math.ceil(len(filtered) / page_size))
        page_index = min(pages - 1, max(0, page_index))
        visible = filtered.iloc[page_index * page_size:(page_index + 1) * page_size]
        columns = list(frame.columns)
        rows = []
        for _, row in visible.iterrows():
            values = [_cell(row[column]) for column in columns]
            if "Image" in columns:
                values[columns.index("Image")] = _molecule_image(row["Image"])
            rows.append(values)
        scenario = info.get("scenario") or {}
        return {
            "key": key, "columns": columns, "rows": rows, "total": len(filtered),
            "allRows": len(frame), "page": page_index, "pages": pages,
            "nameColumn": name_column, "numeric": numeric,
            "scenario": {"state": scenario.get("state", "UNKNOWN"),
                         "messages": scenario.get("messages", []),
                         "recommendations": scenario.get("recommendations", [])},
        }

    def prediction_histogram(self, key, column):
        path = self.window.predictions_tab.csv_paths.get(key)
        if path is None or not Path(path).is_file():
            return {"error": "Prediction table is unavailable"}
        frame = self.window.predictions_tab._filter_prediction_dataframe(pd.read_csv(path))
        if column not in frame or not pd.api.types.is_numeric_dtype(frame[column]):
            return {"error": "Histogram requires a numeric column"}
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return {"error": "No numeric values are available"}
        counts, edges = np.histogram(values, bins=min(18, max(5, int(math.sqrt(len(values))))))
        return {"column": column, "counts": counts.tolist(), "edges": edges.tolist()}

    def prediction_diagnostics(self, key):
        tab = self.window.predictions_tab
        path = tab.csv_paths.get(key)
        if path is None or not Path(path).is_file():
            return {"error": "Prediction diagnostics are unavailable"}
        path = Path(path)
        report = tab._report_paths.get(key) or get_robert_report_path(tab._base_path, key)
        report = Path(report)
        cache_key = (str(path), path.stat().st_mtime_ns, str(report),
                     report.stat().st_mtime_ns if report.is_file() else 0)
        if cache_key not in self._diagnostic_cache:
            catalog = getattr(tab, "_catalog", None)
            selected_model = getattr(tab, "_selected_model", None)
            external = find_external_test_pixmaps(
                tab._base_path,
                (lambda image_path: catalog.include_image(image_path, selected_model))
                if catalog is not None and catalog.is_all_models else None,
            ).get(key)
            result = {
                "scoreImage": _pixmap_image(extract_robert_fragment_image(report)),
                "boundaryScore": extract_boundary_scores(report),
                "boundaryImage": _pixmap_image(extract_boundary_fragment(report)),
                "externalImage": _pixmap_image(external),
            }
            self._diagnostic_cache = {cache_key: result}
        return self._diagnostic_cache[cache_key]

    def plot_data(self, index):
        viewer = self.window.interactive_plots
        if index < 0 or index >= len(viewer.paths):
            return {"error": "Interactive plot is unavailable"}
        if viewer.selector.currentIndex() != index or viewer.data is None:
            viewer.selector.setCurrentIndex(index)
            if viewer.data is None:
                viewer._load_selected(index)
        data = viewer.data
        if data is None:
            return {"error": viewer.details.text() or "Prediction data is unavailable"}
        frame = data["frame"]
        identifier = data.get("name_column")
        if identifier not in frame:
            identifier = next((column for column in ("code_name", "Name", "name", "ID") if column in frame), None)
        external = viewer.paths[index].parent.name == "csv_test"
        points = []
        for row_index in range(len(frame)):
            row = frame.iloc[row_index]
            group = _cell(row["Set"]) if "Set" in frame else ("External" if external else "Data")
            points.append({
                "index": row_index, "label": _cell(row[identifier]) if identifier else f"Row {row_index + 1}",
                "group": group, "observed": _number(data["observed"][row_index]),
                "predicted": _number(data["predicted"][row_index]),
                "residual": _number(data["residual"][row_index]),
                "outlier": _number(data["outlier_score"][row_index]),
                "sd": _number(row[f"{data['target']}_pred_sd"]) if f"{data['target']}_pred_sd" in frame else None,
            })
        williams = []
        if viewer.williams_data is not None:
            for row_index, row in viewer.williams_data.iterrows():
                williams.append({
                    "index": int(row_index), "label": _cell(row["point_id"]),
                    "group": _cell(row["group"]), "x": _number(row["leverage"]),
                    "y": _number(row["standardized_residual"]),
                    "hStar": _number(row["h_star"]),
                    "observed": _number(row["observed"]),
                    "predicted": _number(row["predicted"]),
                })
        return {
            "name": viewer.paths[index].name, "target": data["target"],
            "cvLabel": viewer.cv_label, "threshold": viewer.outlier_threshold,
            "points": points, "williams": williams,
            "williamsImage": QUrl.fromLocalFile(str(viewer.williams_image)).toString()
            if viewer.williams_image is not None else "",
        }

    def plot_molecule(self, plot_index, row_index, williams):
        viewer = self.window.interactive_plots
        if viewer.selector.currentIndex() != plot_index or viewer.data is None:
            self.plot_data(plot_index)
        if viewer.data is None:
            return ""
        source = viewer.williams_smiles if williams else viewer.data["smiles"]
        if row_index < 0 or row_index >= len(source):
            return ""
        return _molecule_image(source[row_index], 240)
