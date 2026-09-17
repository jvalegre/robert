"""
Prediction utilities and UI components for the easyROB Predictions tab.

This module provides all data-processing logic and supporting Qt components
required to visualize and evaluate ROBERT prediction results.

Responsibilities:
- Discover and load prediction CSV files from workflow outputs
- Extract relevant information from ROBERT PDF reports (scores, fragments)
- Evaluate model performance and generate diagnostic summaries
- Provide Qt models and widgets for interactive data visualization
- Render molecular structures (SMILES → images) using RDKit

Architecture:
- Combines pure utility functions (file discovery, parsing, evaluation)
  with Qt-based components (table models, dashboard widgets, async tasks)
- Uses QRunnable and QThreadPool for asynchronous CSV loading
- Implements caching strategies to optimize table sorting and rendering

Notes:
- Designed as a bridge between backend data (CSV, PDF, chemistry) and UI
- Handles partial failures gracefully (e.g., missing PDFs or invalid data)
- Centralizes prediction-related logic to keep the main GUI clean

"""

# ------------------------------------------------------------
# Standard library
# ------------------------------------------------------------
from io import BytesIO
from pathlib import Path
import re

# ------------------------------------------------------------
# Third-party libraries
# ------------------------------------------------------------
import numpy as np
import pandas as pd

import fitz
import pdfplumber

from rdkit import Chem
from rdkit.Chem import Draw

from PySide6.QtCore import (
    QByteArray,
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QRunnable,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QFontMetrics,
    QIcon,
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOptionHeader,
    QVBoxLayout,
    QWidget,
)

def get_predict_dir(selected_file_path: str) -> Path:
    """Given the path to a selected file, return the corresponding PREDICT/csv_test directory."""
    return Path(selected_file_path).parent / "PREDICT" / "csv_test"

def find_prediction_csvs(selected_file_path: str) -> dict[str, Path]:
    """Search for prediction CSV files in the PREDICT/csv_test directory related to the selected file."""
    predict_dir = get_predict_dir(selected_file_path)
    if not predict_dir.exists():
        return {}

    results: dict[str, Path] = {}
    for path in predict_dir.glob("*.csv"):
        name = path.name
        if name.endswith("_No_PFI.csv"):
            results["No_PFI"] = path
        elif name.endswith("_PFI.csv"):
            results["PFI"] = path
    return results

def get_robert_report_path(selected_file_path: str | Path, model_key: str) -> Path:
    """Given the path to a selected file and model key ("No_PFI" or "PFI"), return the
    corresponding ROBERT report PDF file - v2.2 generates one PDF per suffix instead of a
    single combined report, so the path must depend on which tab/model is being displayed."""
    return Path(selected_file_path).parent / f"ROBERT_report_{model_key}.pdf"

def find_external_test_pixmaps(base_path: str | Path) -> dict[str, QPixmap]:
    """Search for external test images in the PREDICT/csv_test directory related to the selected file."""
    base_path = Path(base_path)
    if base_path.is_file():
        base_path = base_path.parent

    csv_test_dir = base_path / "PREDICT" / "csv_test"
    if not csv_test_dir.exists():
        return {}

    results: dict[str, QPixmap] = {}
    for path in csv_test_dir.glob("*.png"):
        name = path.name
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            continue

        if name.endswith("_No_PFI_external.png"):
            results["No_PFI"] = pixmap
        elif name.endswith("_PFI_external.png"):
            results["PFI"] = pixmap

    return results

def extract_scores_from_robert_report(pdf_path: Path) -> dict | None:
    """Extract the Interpolation score details (the report's headline score) from the ROBERT
    report PDF file. The file itself is already specific to one model/suffix - see
    get_robert_report_path() - so there's a single score to extract, not one per model."""
    return _extract_robert_score_details(pdf_path)

def extract_boundary_fragment(pdf_path: Path) -> QPixmap | None:
    """Render the boundary robustness block from parsed ROBERT report data (regression only -
    classification has no Boundary robustness score, see _extract_boundary_details())."""
    details = _extract_boundary_details(pdf_path)
    if not details:
        return None
    return _render_robert_score_pixmap(details)

def extract_robert_fragment_image(pdf_path: Path) -> QPixmap | None:
    """Render the ROBERT score block from parsed report data."""
    details = _extract_robert_score_details(pdf_path)
    if not details:
        return None
    return _render_robert_score_pixmap(details)

# v2.2 generates one full-width PDF per model/suffix (see get_robert_report_path()) instead
# of a single combined report with a No_PFI/PFI side-by-side layout. Section A's own layout
# now uses that same left/right split for a DIFFERENT pair of columns: Interpolation (the
# report's headline score) on the left, Boundary robustness on the right (regression only -
# blank for classification, see docs/Report/score.rst). Model/suffix selection is handled
# entirely by which PDF gets opened now, so these bboxes are fixed, not model-dependent.
def _score_bbox(page):
    """Left half of page 0: the Interpolation score block."""
    return (0, 0, 300, page.height)


def _boundary_bbox(page):
    """Right half of page 0: the Boundary robustness score block."""
    return (300, 0, page.width, page.height)


def _normalize_boundary_lines(text: str) -> list[str]:
    """Collapse noisy PDF whitespace while preserving the content of each line."""
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def _parse_boundary_block(text: str) -> dict | None:
    """Parse the Boundary robustness summary block into structured data. Boundary robustness
    is scored 0-10, the same scale and "Title . Score N" heading as Interpolation (see
    _parse_robert_score_block()), with two headline metrics instead of the model/points lines
    - shaped to match that function's output so both scores can share
    _render_robert_score_pixmap()."""
    if not text:
        return None

    lines = _normalize_boundary_lines(text)
    title_line = next(
        (line for line in lines if "Boundary robustness" in line and re.search(r"\bScore\s+\d+\b", line, re.IGNORECASE)),
        None,
    )
    if not title_line:
        return None

    score_match = re.search(r"\bScore\s+(\d+)\b", title_line, re.IGNORECASE)
    if not score_match:
        return None

    low_line = next((line for line in lines if "Scaled RMSE (Low" in line), "")
    high_line = next((line for line in lines if "Scaled RMSE (High" in line), "")

    return {
        "title": "Boundary robustness",
        "score": int(score_match.group(1)),
        "model_line": low_line,
        "points_line": high_line,
    }


def _extract_boundary_details(pdf_path: Path) -> dict | None:
    """Extract structured boundary robustness information from the ROBERT report. Returns
    None for classification reports, whose Section A right column has no Boundary robustness
    block at all (not scored for classification - see docs/Report/score.rst)."""
    if not pdf_path.exists():
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return None
            page = pdf.pages[0]
            text = page.within_bbox(_boundary_bbox(page)).extract_text()
            return _parse_boundary_block(text or "")
    except Exception:
        return None


def _parse_robert_score_block(text: str) -> dict | None:
    """Parse the ROBERT score text block into structured data."""
    if not text:
        return None

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    title_line = next((line for line in lines if re.search(r"\bScore\s+\d+\b", line, re.IGNORECASE)), None)
    if not title_line:
        joined_text = " ".join(lines)
        score_match = re.search(r"\bScore\s+(\d+)\b", joined_text, re.IGNORECASE)
        if not score_match:
            return None
        title_match = re.search(r"(.+?)\s*[.\-·]?\s*Score\s+\d+\b", joined_text, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "ROBERT Score"
        return {
            "title": title,
            "score": int(score_match.group(1)),
            "model_line": _extract_robert_model_line(lines, title),
            "points_line": _extract_robert_points_line(lines, title),
        }

    score_match = re.search(r"\bScore\s+(\d+)\b", title_line, re.IGNORECASE)
    if not score_match:
        return None

    title = re.sub(r"\s*[·\.-]?\s*Score\s+\d+\b.*$", "", title_line, flags=re.IGNORECASE).strip()
    return {
        "title": title,
        "score": int(score_match.group(1)),
        "model_line": _extract_robert_model_line(lines, title),
        "points_line": _extract_robert_points_line(lines, title),
    }


def _extract_robert_model_line(lines: list[str], title: str) -> str:
    """Find the line describing model, CV split and test split."""
    for line in lines:
        if line == title or "Score" in line:
            continue
        if "Model" in line or "CV" in line or "Test" in line:
            if "ROBERT Report" in line:
                continue
            if re.search(r"\b\d{4}/\d{2}/\d{2}\b", line):
                continue
            return line
    return ""


def _extract_robert_points_line(lines: list[str], title: str) -> str:
    """Find the line describing training points and descriptors."""
    for line in lines:
        if line == title or "Score" in line:
            continue
        if "Points" in line:
            return line
    return ""


def _extract_robert_score_details(pdf_path: Path) -> dict | None:
    """Extract structured Interpolation score information (the report's headline score) from
    the ROBERT report."""
    if not pdf_path.exists():
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return None
            page = pdf.pages[0]
            text = page.within_bbox(_score_bbox(page)).extract_text()
            return _parse_robert_score_block(text or "")
    except Exception:
        return None


def _robert_score_style(score: int | None) -> dict:
    """Return the visual style associated with a ROBERT score."""
    if score is None:
        return {"label": "UNKNOWN", "segments": 0, "fill": (0.92, 0.90, 0.96), "text": (0.35, 0.35, 0.35)}
    if score <= 0:
        return {"label": "VERY WEAK", "segments": 0, "fill": (1.0, 0.42, 0.42), "text": (1.0, 0.42, 0.42)}
    if score <= 3:
        return {"label": "VERY WEAK", "segments": score, "fill": (1.0, 0.42, 0.42), "text": (1.0, 0.42, 0.42)}
    if score <= 6:
        return {"label": "WEAK", "segments": score, "fill": (1.0, 0.79, 0.38), "text": (1.0, 0.79, 0.38)}
    if score <= 8:
        return {"label": "MODERATE", "segments": score, "fill": (0.60, 0.78, 0.95), "text": (0.60, 0.78, 0.95)}
    return {"label": "STRONG", "segments": min(score, 10), "fill": (0.38, 0.60, 0.80), "text": (0.38, 0.60, 0.80)}


def _render_robert_score_pixmap(details: dict) -> QPixmap | None:
    """Render a synthetic ROBERT score card for the GUI using parsed PDF data."""
    width = 620
    height = 255
    margin = 24
    style = _robert_score_style(details.get("score"))

    doc = fitz.open()
    try:
        page = doc.new_page(width=width, height=height)
        page.draw_rect(
            fitz.Rect(6, 6, width - 6, height - 6),
            color=(0.72, 0.72, 0.72),
            fill=(1.0, 1.0, 1.0),
            width=1.2,
        )
        page.draw_rect(
            fitz.Rect(14, 14, width - 14, height - 14),
            color=(0.86, 0.86, 0.86),
            fill=None,
            width=0.8,
        )

        page.insert_text(
            fitz.Point(margin, 42),
            f"{details['title']}  .  Score {details['score']}",
            fontsize=24,
            fontname="hebo",
            color=(0.05, 0.05, 0.05),
        )
        if details.get("model_line"):
            page.insert_text(
                fitz.Point(margin, 78),
                details["model_line"],
                fontsize=20,
                fontname="helv",
                color=(0.05, 0.05, 0.05),
            )
        if details.get("points_line"):
            page.insert_text(
                fitz.Point(margin, 110),
                details["points_line"],
                fontsize=20,
                fontname="helv",
                color=(0.05, 0.05, 0.05),
            )

        bar_left = margin
        bar_top = 138
        bar_width = width - (margin * 2)
        segments = 10
        gap = 0
        segment_width = bar_width / segments
        segment_height = 20
        empty_fill = (0.92, 0.90, 0.96)
        filled_segments = max(0, min(style["segments"], segments))

        for idx in range(segments):
            x0 = bar_left + idx * (segment_width + gap)
            rect = fitz.Rect(x0, bar_top, x0 + segment_width, bar_top + segment_height)
            fill = style["fill"] if idx < filled_segments else empty_fill
            page.draw_rect(rect, color=(0.05, 0.05, 0.05), fill=fill, width=1.0)

        label = style["label"]
        label_width_map = {
            "VERY WEAK": 210,
            "WEAK": 110,
            "MODERATE": 180,
            "STRONG": 150,
            "UNKNOWN": 160,
        }
        label_x = width - margin - label_width_map.get(label, 160)
        page.insert_text(
            fitz.Point(label_x, 220),
            label,
            fontsize=20,
            fontname="hebo",
            color=style["text"],
        )

        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        qimg = QImage.fromData(QByteArray(pix.tobytes("png")))
        return QPixmap.fromImage(qimg)
    except Exception:
        return None
    finally:
        doc.close()


def extract_boundary_scores(pdf_path: Path) -> int | None:
    """Extract the Boundary robustness score (0-10) from the ROBERT report PDF file.
    Returns None for classification reports, which have no Boundary robustness score."""
    details = _extract_boundary_details(pdf_path)
    return details.get("score") if details else None

def extract_prediction_info(df: pd.DataFrame) -> dict:
    """Extract prediction information from a DataFrame."""
    pred_cols = [col for col in df.columns if col.endswith("_pred")]
    result = {
        "has_pred_column": False,
        "pred_column": None,
        "predictions_identical": None,
        "n_unique": None,
        "n_rows": len(df),
    }

    if len(pred_cols) != 1:
        return result

    col = pred_cols[0]
    series = df[col].dropna()
    result["has_pred_column"] = True
    result["pred_column"] = col
    result["n_unique"] = series.nunique()
    result["predictions_identical"] = None if result["n_unique"] == 0 else result["n_unique"] == 1
    return result

def evaluate_model_scenario(score: int | None, predictions_identical: bool | None) -> dict:
    """Evaluate the model scenario based on ROBERT score and predictions."""
    almos_link = "https://github.com/MiguelMartzFdez/almos"
    almos_html = f'<a href="{almos_link}">ALMOS</a>'
    result = {"state": "UNKNOWN", "messages": [], "recommendations": []}

    if score is None:
        result["messages"].append("No valid ROBERT score was detected. Model reliability cannot be evaluated.")
        result["recommendations"].append("You may verify that ROBERT_report_No_PFI.pdf was generated correctly.")
        return result

    if predictions_identical is True:
        result["state"] = "FAILED"
        result["messages"].append(
            f"ROBERT score is {score}, but all predictions are identical. "
            "The model cannot discriminate between candidates."
        )
        result["recommendations"].append(
            "You may discard these predictions and introduce structural diversity "
            f"with Clustering module in {almos_html}."
        )
        return result

    if 0 <= score <= 3:
        result["state"] = "FAILED"
        result["messages"].append(f"ROBERT score is {score}. Model performance is critically low.")
        result["recommendations"].append(
            "You may avoid using these predictions and rebuild the dataset "
            f"using Clustering module in {almos_html}."
        )
        return result

    if 4 <= score <= 6:
        result["state"] = "WEAK"
        result["messages"].append(f"ROBERT score is {score}. The model works, but reliability is limited.")
        result["recommendations"].append(
            "You may use predictions cautiously and improve robustness "
            f"through Active Learning module with {almos_html}."
        )
        return result

    if 7 <= score <= 8:
        result["state"] = "DECENT"
        result["messages"].append(f"ROBERT score is {score}. The model is solid but can still improve.")
        result["recommendations"].append(
            "You may use these predictions while considering further optimization "
            f"through Active Learning module with {almos_html}."
        )
        return result

    if score > 8:
        result["state"] = "STRONG"
        result["messages"].append(f"ROBERT score is {score}. The model shows strong predictive performance.")
        result["recommendations"].append(
            "You may confidently use these predictions for candidate prioritization."
        )
        return result

    return result

def evaluate_predictions_for_model(selected_file_path: str | Path, df: pd.DataFrame, model_key: str) -> dict:
    """Evaluate predictions for a specific model."""
    pdf_path = get_robert_report_path(selected_file_path, model_key)
    details = extract_scores_from_robert_report(pdf_path)
    score = details.get("score") if details else None
    prediction_info = extract_prediction_info(df)
    scenario = evaluate_model_scenario(
        score=score,
        predictions_identical=prediction_info["predictions_identical"],
    )
    return {
        "model": model_key,
        "pdf_path": pdf_path,
        "score": score,
        "prediction_info": prediction_info,
        "scenario": scenario,
    }

class PredictionDashboardPanel(QWidget):
    """A collapsible dashboard panel to display ROBERT prediction evaluation results and diagnostics."""
    def __init__(self, scenario: dict, pdf_image=None, boundary_score=None, boundary_image=None, external_plot=None, parent=None):
        super().__init__(parent)
        self._pdf_image = pdf_image
        self._boundary_score = boundary_score
        self._boundary_image = boundary_image
        self._external_plot = external_plot
        self.setObjectName("PredictionDashboard")
        self.expanded_width = 500
        self.collapsed_width = 40
        self._expanded = True

        state_colors = {
            "FAILED": "#b00020",
            "WEAK": "#8a6d00",
            "DECENT": "#276dd6",
            "STRONG": "#1b5e20",
            "UNKNOWN": "#666666",
        }
        self.main_color = state_colors.get(scenario.get("state", "UNKNOWN"), "#666666")
        self._build_ui(scenario)
        self._apply_initial_state()

    def _apply_initial_state(self):
        """Apply the initial state based on the expansion status."""
        if self._expanded:
            self.main_layout.setContentsMargins(10, 8, 10, 8)
            self.setMinimumWidth(self.expanded_width)
            self.setMaximumWidth(self.expanded_width)
            self.content.setVisible(True)
            self.toggle_btn.setText("Hide Info ❯")
            self.toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.toggle_btn.setFixedHeight(32)
            self.toggle_btn.setStyleSheet(
                "QPushButton { background: transparent; border-radius: 4px; font-size: 12px; padding: 4px 8px; }"
                "QPushButton:hover { background: rgba(0,0,0,0.05); }"
            )
        else:
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.setMinimumWidth(self.collapsed_width)
            self.setMaximumWidth(self.collapsed_width)
            self.content.setVisible(False)
            self.toggle_btn.setText("❮\nMore\nInfo")
            self.toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.toggle_btn.setMinimumHeight(100)
            self.toggle_btn.setStyleSheet(
                "QPushButton { background: transparent; font-size: 11px; padding: 2px; }"
                "QPushButton:hover { background: rgba(0,0,0,0.05); }"
            )

    def _build_ui(self, scenario):
        """Build the user interface."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 8, 10, 8)
        self.main_layout.setSpacing(12)

        self.toggle_btn = QPushButton()
        self.toggle_btn.clicked.connect(self.toggle)
        self.main_layout.addWidget(self.toggle_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.content = QWidget()
        content_layout = QVBoxLayout(self.content)
        content_layout.setSpacing(8)

        self._build_status_block(content_layout, scenario)
        self._build_pdf_snapshot_block(content_layout)
        self._build_boundary_block(content_layout, self._boundary_score, self._boundary_image)
        self._build_external_validation_block(content_layout, self._external_plot)
        content_layout.addStretch()

        scroll.setWidget(self.content)
        self.main_layout.addWidget(scroll)

    def _build_status_block(self, layout, scenario):
        """Build the status block with messages and recommendations."""

        msg_label = QLabel(" ".join(scenario.get("messages", [])))
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet(f"color: {self.main_color}; font-weight: bold;")
        layout.addWidget(msg_label)

        recommendations = scenario.get("recommendations", [])
        if recommendations:
            rec_label = QLabel(" ".join(recommendations))
            rec_label.setWordWrap(True)
            rec_label.setTextFormat(Qt.RichText)
            rec_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
            rec_label.setOpenExternalLinks(True)
            layout.addWidget(rec_label)

    def _build_pdf_snapshot_block(self, layout):
        """Build the PDF snapshot block with a PDF image."""
        if not self._pdf_image:
            return

        layout.addSpacing(15)
        container = QWidget()
        container.setObjectName("dashboardBlock")
        container.setStyleSheet("QWidget#dashboardBlock { border: 1px solid palette(mid); border-radius: 8px; }")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(14, 14, 14, 14)
        container_layout.setSpacing(10)

        title = QLabel("Model Performance Overview")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        container_layout.addWidget(title)

        image_frame = QWidget()
        image_frame.setObjectName("imageFrame")
        image_frame.setStyleSheet("QWidget#imageFrame { border: 1px solid palette(mid); border-radius: 6px; }")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(6, 6, 6, 6)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(self._pdf_image.scaledToWidth(self.expanded_width - 120, Qt.SmoothTransformation))
        image_layout.addWidget(image_label)
        container_layout.addWidget(image_frame)
        layout.addWidget(container)

    def _build_boundary_block(self, layout, score, pixmap):
        """Build the boundary robustness block with a score and image."""
        if score is None and not pixmap:
            return

        layout.addSpacing(15)
        container = QWidget()
        container.setObjectName("dashboardBlock")
        container.setStyleSheet("QWidget#dashboardBlock { border: 1px solid palette(mid); border-radius: 8px; }")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(14, 14, 14, 14)
        container_layout.setSpacing(10)

        title = QLabel("Boundary Robustness")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        container_layout.addWidget(title)

        subtitle = QLabel("Assessment of how well the model holds up at the edges of the training data range and beyond.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 11px;")
        container_layout.addWidget(subtitle)

        if pixmap:
            image_frame = QWidget()
            image_frame.setObjectName("imageFrame")
            image_frame.setStyleSheet("QWidget#imageFrame { border: 1px solid palette(mid); border-radius: 6px; }")
            image_layout = QVBoxLayout(image_frame)
            image_layout.setContentsMargins(6, 6, 6, 6)
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setPixmap(pixmap.scaledToWidth(self.expanded_width - 120, Qt.SmoothTransformation))
            image_layout.addWidget(image_label)
            container_layout.addWidget(image_frame)

        # Boundary robustness is scored 0-10 (regression only), the same scale used for the
        # main ROBERT score - see _robert_score_style() for the VERY WEAK/WEAK/MODERATE/STRONG
        # bands it's built from
        if score is not None:
            label = _robert_score_style(score)["label"]
            summary_texts = {
                "VERY WEAK": (
                    "No boundary robustness" if score <= 0 else "Very weak boundary robustness",
                    "#b00020",
                    "The model is unreliable at the edges of the training range and beyond. Predictions outside the original data range are unreliable.",
                ),
                "WEAK": (
                    "Weak boundary robustness",
                    "#8a6d00",
                    "The model may tolerate slight deviations beyond the training range, but predictions near extremes can become unstable.",
                ),
                "MODERATE": (
                    "Moderate boundary robustness",
                    "#276dd6",
                    "The model holds up reasonably well at the edges of the training range, although uncertainty increases further from the original data distribution.",
                ),
                "STRONG": (
                    "Strong boundary robustness",
                    "#1b5e20",
                    "The model remains reliable at the edges of the training range and beyond, showing robust predictions even for extreme values.",
                ),
            }
            summary, color, explanation = summary_texts[label]

            summary_label = QLabel(summary)
            summary_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            container_layout.addWidget(summary_label)

            explanation_label = QLabel(explanation)
            explanation_label.setWordWrap(True)
            container_layout.addWidget(explanation_label)

        layout.addWidget(container)

    def _build_external_validation_block(self, layout, pixmap):
        """Build the external validation block with an image."""
        if not pixmap:
            return

        layout.addSpacing(15)
        container = QWidget()
        container.setObjectName("dashboardBlock")
        container.setStyleSheet("QWidget#dashboardBlock { border: 1px solid palette(mid); border-radius: 8px; }")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(14, 14, 14, 14)
        container_layout.setSpacing(10)

        title = QLabel("External Test Set Validation")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        container_layout.addWidget(title)

        subtitle = QLabel("Predicted vs experimental values for molecules with known target data.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 11px;")
        container_layout.addWidget(subtitle)

        image_frame = QWidget()
        image_frame.setObjectName("imageFrame")
        image_frame.setStyleSheet("QWidget#imageFrame { border: 1px solid palette(mid); border-radius: 6px; }")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(6, 6, 6, 6)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(pixmap.scaledToWidth(self.expanded_width - 120, Qt.SmoothTransformation))
        image_layout.addWidget(image_label)
        container_layout.addWidget(image_frame)

        guidance = QLabel(
            "Interpretation:\n"
            "• Alignment with the diagonal indicates agreement between predicted and experimental values.\n"
            "• Larger deviations reflect increased prediction error.\n"
            "• Systematic offsets may suggest model bias or calibration issues."
        )
        guidance.setWordWrap(True)
        guidance.setStyleSheet("font-size: 11px;")
        container_layout.addWidget(guidance)
        layout.addWidget(container)

    def toggle(self):
        self._expanded = not self._expanded
        self._apply_initial_state()


class PandasTableModel(QAbstractTableModel):
    """A Qt table model that wraps a pandas DataFrame, with special handling for SMILES rendering and sorting optimization."""
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        self._sort_cache: dict[int, np.ndarray] = {}

    def rowCount(self, parent=QModelIndex()):
        """Return the number of rows in the DataFrame."""
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        """Return the number of columns in the DataFrame."""
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        """Return the data for a given index and role, with special handling for SMILES rendering."""
        if not index.isValid():
            return None

        value = self._df.iat[index.row(), index.column()]
        column_name = self._df.columns[index.column()]

        if column_name == "Image":
            if role in (Qt.DecorationRole, Qt.DisplayRole):
                smiles = "" if pd.isna(value) else str(value)
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(120, 120))
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    qimg = QImage.fromData(buffer.getvalue())
                    return QPixmap.fromImage(qimg)
            return None

        if role == Qt.DisplayRole:
            return "" if pd.isna(value) else str(value)

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Return the header data for a given section and orientation and role."""
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._df.columns[section]
        return section + 1

    def sort_key(self, column: int) -> np.ndarray:
        """Return the sorting key for a given column, with special handling for numeric columns."""
        if column not in self._sort_cache:
            col = self._df.iloc[:, column]
            if pd.api.types.is_numeric_dtype(col):
                self._sort_cache[column] = col.to_numpy(dtype=float, copy=False)
            else:
                self._sort_cache[column] = col.astype(str).to_numpy()
        return self._sort_cache[column]

class StatsHeader(QHeaderView):
    """A custom header view that displays column names and basic statistics for numeric columns."""
    def __init__(self, df: pd.DataFrame, orientation, parent=None):
        super().__init__(orientation, parent)
        self._df = df
        self.setDefaultAlignment(Qt.AlignCenter)
        self.setSectionResizeMode(QHeaderView.ResizeToContents)
        self._stats_cache = {}

        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                self._stats_cache[col] = {
                    "min": series.min(),
                    "median": series.median(),
                    "mean": series.mean(),
                    "max": series.max(),
                }

        self.setMinimumHeight(100)

    def sizeHint(self):
        """Return the size hint for the header view."""
        size = super().sizeHint()
        size.setHeight(100)
        return size

    def paintSection(self, painter, rect, logical_index):
        """Paint the header section with column name and statistics."""
        painter.save()

        option = QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.text = ""
        option.icon = QIcon()
        option.sortIndicator = QStyleOptionHeader.None_
        self.style().drawControl(QStyle.CE_HeaderSection, option, painter, self)

        col_name = self._df.columns[logical_index]
        stats = self._stats_cache.get(col_name)
        margin = 6
        r = rect.adjusted(margin, margin, -margin, -margin)

        bold_font = painter.font()
        bold_font.setBold(True)
        painter.setFont(bold_font)
        fm = QFontMetrics(bold_font)
        name_height = fm.boundingRect(0, 0, r.width(), 1000, Qt.AlignHCenter | Qt.TextWordWrap, col_name).height()

        name_rect = rect.adjusted(margin, margin, -margin, -margin)
        name_rect.setHeight(name_height)
        painter.drawText(name_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, col_name)

        if stats is not None:
            normal_font = painter.font()
            normal_font.setBold(False)
            painter.setFont(normal_font)
            metrics_text = (
                f"min: {stats['min']:.4g}\n"
                f"median: {stats['median']:.4g}\n"
                f"mean: {stats['mean']:.4g}\n"
                f"max: {stats['max']:.4g}"
            )
            metrics_rect = r
            metrics_rect.setTop(name_rect.bottom() + 6)
            painter.drawText(metrics_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, metrics_text)

        painter.restore()

class ColumnStatsWidget(QWidget):
    """A widget that displays basic statistics for numeric columns in a DataFrame."""
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            layout.addWidget(QLabel("No numeric columns"))
            layout.addStretch()
            return

        stats = numeric_df.agg(["min", "max", "mean"])
        for col in numeric_df.columns:
            label = QLabel(
                f"<b>{col}</b><br>"
                f"min: {stats[col]['min']:.4g}<br>"
                f"max: {stats[col]['max']:.4g}<br>"
                f"mean: {stats[col]['mean']:.4g}"
            )
            label.setStyleSheet("QLabel { padding: 4px; }")
            layout.addWidget(label)

        layout.addStretch()


class LoadCsvSignals(QObject):
    """Signals for the LoadCsvTask."""
    done = Signal(str, pd.DataFrame)

class LoadCsvTask(QRunnable):
    """A task for loading a CSV file."""
    def __init__(self, key: str, path: Path):
        super().__init__()
        self.key = key
        self.path = path
        self.signals = LoadCsvSignals()

    def run(self):
        try:
            df = pd.read_csv(self.path)
            self.signals.done.emit(self.key, df)
        except Exception as exc:
            print(f"Failed to load CSV {self.path}: {exc}")

class NumericSortProxy(QSortFilterProxyModel):
    """A proxy model that optimizes sorting for numeric columns by caching sort keys."""
    def lessThan(self, left, right):
        model = self.sourceModel()
        keys = model.sort_key(left.column())
        return bool(keys[left.row()] < keys[right.row()])
