"""
Image browser tab for easyROB workflow outputs.

This module provides a lightweight interface to explore, preview, and export
images generated during the ROBERT workflow.

Responsibilities:
- Scan workflow output folders (CURATE, GENERATE, PREDICT, VERIFY)
- Display images grouped by workflow stage
- Provide interaction options (preview, open, copy, save, open folder)

Import strategy:
- Supports dual execution modes:
  1. Local (portable execution)
  2. Installed package (environment / entry point)

- Imports use a try/except fallback to ensure compatibility.

Notes:
- Focuses on visualization and file interaction only.
- No workflow logic or processing is handled here.

"""

try:
    from utils.utils_gui import (
        QApplication,
        QComboBox,
        QDesktopServices,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMenu,
        QMessageBox,
        QPixmap,
        QPushButton,
        QScrollArea,
        QStackedWidget,
        QUrl,
        QVBoxLayout,
        QWidget,
        Qt,
    )

except ImportError as e:
    from robert.gui_easyrob.utils.utils_gui import (
        QApplication,
        QComboBox,
        QDesktopServices,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMenu,
        QMessageBox,
        QPixmap,
        QPushButton,
        QScrollArea,
        QStackedWidget,
        QUrl,
        QVBoxLayout,
        QWidget,
        Qt,
    )

# ---- Standard library ----
import os
import json
import hashlib
from pathlib import Path

from PySide6.QtCore import QObject, Slot


IMAGE_FOLDERS = (
    ("CURATE", "CURATE"),
    ("GENERATE/Raw_data", "GENERATE"),
    ("PREDICT", "PREDICT"),
    ("PREDICT/csv_test", "PREDICT / external test"),
    ("VERIFY", "VERIFY"),
)


def find_image_files(folder_path):
    """Return supported image files directly inside a workflow folder."""
    if not os.path.isdir(folder_path):
        return []
    return sorted(
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
        and os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg"}
    )


def build_gallery_state(base_path, catalog=None, model=None):
    """Build browser data and an ID-to-path map from workflow images."""
    stages = []
    paths = {}
    for folder, label in IMAGE_FOLDERS:
        directory = Path(base_path) / folder
        items = []
        for filename in find_image_files(str(directory)):
            path = Path(filename)
            if catalog is not None and not catalog.include_image(path, model):
                continue
            image_id = "image-" + hashlib.blake2s(str(path).encode("utf-8"), digest_size=8).hexdigest()
            paths[image_id] = path
            items.append({
                "id": image_id,
                "name": path.name,
                "url": QUrl.fromLocalFile(str(path)).toString(),
            })
        if items:
            stages.append({"id": folder, "label": label, "images": items})
    return {"stages": stages, "total": len(paths)}, paths


class ImageActionBridge(QObject):
    """Expose only listed image actions to the local browser page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths = {}

    def set_paths(self, paths):
        self._paths = dict(paths)

    @Slot(str, str)
    def performAction(self, image_id, action):
        path = self._paths.get(image_id)
        if path is None or not path.is_file():
            return
        if action == "open":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        elif action == "folder":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
        elif action == "copy":
            QApplication.clipboard().setPixmap(QPixmap(str(path)))
        elif action == "save":
            target_path, _ = QFileDialog.getSaveFileName(
                self.parent(), "Save Image As", path.name, "Images (*.png *.jpg *.jpeg)"
            )
            if target_path:
                QPixmap(str(path)).save(target_path)


class ImagesTab(QWidget):
    """Images tab for displaying images from multiple folders as workflow results."""

    def __init__(self, main_tab_widget, image_folders, file_path):
        super().__init__()

        # Store references and base workflow path
        self.main_tab_widget = main_tab_widget
        self.image_folders = image_folders
        self.base_path = os.path.dirname(file_path)
        self.catalog = None
        self.model = None
        self.folder_widgets = {}

        # Retain the widget view for fast headless tests and browser failures.
        self._views = QStackedWidget(self)
        self._legacy_content = QWidget(self)
        self.layout = QVBoxLayout(self._legacy_content)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self._views)
        self._views.addWidget(self._legacy_content)
        self._web_view = None
        self._web_ready = False
        self._bridge = ImageActionBridge(self)
        self.gallery_state = {"stages": [], "total": 0}

        # Top section: title + subtitle (+ inline help)
        # Title
        title_label = QLabel("Workflow Images")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Subtitle row (subtitle + help button inline)
        subtitle_row = QHBoxLayout()
        subtitle_row.setSpacing(6)

        subtitle_label = QLabel("Browse, open and export generated images")
        subtitle_label.setStyleSheet("color: gray; font-size: 12px;")

        # Help button inline with subtitle
        help_button = QPushButton("?")
        help_button.setFixedSize(18, 18)
        help_button.setStyleSheet("font-size: 11px;")
        help_button.setToolTip(
            "Double-click: Open image\n"
            "Right-click: Copy, Save, or Open folder"
        )
        help_button.clicked.connect(self.show_help_dialog)

        # Build subtitle row
        subtitle_row.addWidget(subtitle_label)
        subtitle_row.addWidget(help_button)
        subtitle_row.addStretch()  # Push everything left

        # Add to main layout
        self.layout.addWidget(title_label)
        self.layout.addLayout(subtitle_row)

        # A single view selector keeps Images free of nested tab bars.
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View"))
        self.view_selector = QComboBox()
        self.view_selector.setMaximumWidth(280)
        view_row.addWidget(self.view_selector)
        view_row.addStretch()
        self.layout.addLayout(view_row)
        self.folder_tabs = QStackedWidget()
        self.view_selector.currentIndexChanged.connect(self.folder_tabs.setCurrentIndex)
        self.layout.addWidget(self.folder_tabs)

        self.check_for_images()

    def showEvent(self, event):
        super().showEvent(event)
        self.activate_web()

    def activate_web(self):
        """Start Chromium only when this result view is opened interactively."""
        if QApplication.platformName() == "offscreen" and os.environ.get("EASYROB_WEBENGINE_TEST") != "1":
            return
        if self._web_view is not None:
            return
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineWidgets import QWebEngineView

        self._web_view = QWebEngineView(self)
        self._web_channel = QWebChannel(self._web_view.page())
        self._web_channel.registerObject("imageActions", self._bridge)
        self._web_view.page().setWebChannel(self._web_channel)
        self._web_view.loadFinished.connect(self._on_web_loaded)
        self._views.addWidget(self._web_view)
        page_path = Path(__file__).resolve().parent.parent / "web" / "images.html"
        self._web_view.setUrl(QUrl.fromLocalFile(str(page_path)))

    def _on_web_loaded(self, success):
        if not success:
            self._views.setCurrentWidget(self._legacy_content)
            return
        self._web_ready = True
        self._views.setCurrentWidget(self._web_view)
        self._sync_web()

    def _sync_web(self):
        state, paths = build_gallery_state(self.base_path, self.catalog, self.model)
        self.gallery_state = state
        self._bridge.set_paths(paths)
        if self._web_ready:
            payload = json.dumps(state, ensure_ascii=False)
            self._web_view.page().runJavaScript(f"window.easyrobGallery.setData({payload})")

    def show_help_dialog(self):
        """Show usage instructions."""
        msg = QMessageBox(self)
        msg.setWindowTitle("How to use")
        msg.setText(
            "Image interaction:\n\n"
            "• Double-click → Open image\n"
            "• Right-click → Copy, Save, or Open folder\n\n"
            "Images are grouped by workflow stage."
        )
        msg.exec()

    def refresh_with_new_path(self, file_path, catalog=None, model=None):
        """Update image base path and refresh image tabs."""
        selected_view = self.view_selector.currentText()
        self.base_path = os.path.dirname(file_path)
        self.catalog = catalog
        self.model = model
        self.clear_image_tabs()
        self.check_for_images()
        index = self.view_selector.findText(selected_view)
        if index >= 0:
            self.view_selector.setCurrentIndex(index)

    def clear_image_tabs(self):
        """Clear all image folders and their widgets."""
        self.view_selector.blockSignals(True)
        self.view_selector.clear()
        self.view_selector.blockSignals(False)
        while self.folder_tabs.count():
            widget = self.folder_tabs.widget(0)
            if widget:
                self.folder_tabs.removeWidget(widget)
                widget.deleteLater()
        self.folder_widgets.clear()

    def check_for_images(self):
        """Scan folders and update tabs with new images."""
        folder_names = {
            "CURATE": "CURATE",
            "GENERATE/Raw_data": "GENERATE",
            "PREDICT": "PREDICT",
            "PREDICT/csv_test": "PREDICT / external test",
            "VERIFY": "VERIFY",
        }

        folder_order = [
            "CURATE", "GENERATE/Raw_data", "PREDICT", "PREDICT/csv_test", "VERIFY",
        ]

        for folder in folder_order:
            full_folder_path = os.path.join(self.base_path, folder)
            if not os.path.exists(full_folder_path):
                continue

            image_files = find_image_files(full_folder_path)
            if self.catalog is not None:
                image_files = [
                    path for path in image_files
                    if self.catalog.include_image(Path(path), self.model)
                ]
            if not image_files:
                continue

            if folder not in self.folder_widgets:
                folder_widget = QWidget()
                folder_layout = QVBoxLayout(folder_widget)

                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)

                image_grid = QGridLayout()
                container = QWidget()
                container.setLayout(image_grid)
                scroll_area.setWidget(container)

                folder_layout.addWidget(scroll_area)
                folder_widget.setLayout(folder_layout)

                tab_name = folder_names.get(folder, os.path.basename(folder))
                self.folder_tabs.addWidget(folder_widget)
                self.view_selector.addItem(tab_name)
                self.folder_widgets[folder] = image_grid

            image_grid = self.folder_widgets[folder]

            # Clear old images
            while image_grid.count():
                item = image_grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            row, col = 0, 0
            max_columns = 3

            for img_path in image_files:
                image_label = ImageLabel(img_path, size=300)
                image_grid.addWidget(image_label, row, col)

                col += 1
                if col >= max_columns:
                    col = 0
                    row += 1

        self._sync_web()


class ImageLabel(QLabel):
    """Custom QLabel for displaying an image with context menu and double-click support."""

    def __init__(self, image_path, size=400):
        super().__init__()

        self.image_path = image_path
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip("Right-click for options\nDouble-click to open")

        pixmap = QPixmap(self.image_path)

        if pixmap.isNull():
            self.setText("Failed to load image.")
        else:
            self.setPixmap(
                pixmap.scaled(
                    size,
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def mouseDoubleClickEvent(self, event):
        """Open image on double-click."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.image_path))

    def show_context_menu(self, position):
        """Create context menu."""
        menu = QMenu(self)

        copy_action = menu.addAction("Copy image")
        open_action = menu.addAction("Open image")
        open_folder_action = menu.addAction("Open containing folder")
        save_as_action = menu.addAction("Save as...")

        action = menu.exec(self.mapToGlobal(position))

        if action == copy_action:
            clipboard = QApplication.clipboard()
            clipboard.setPixmap(QPixmap(self.image_path))

        elif action == open_action:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.image_path))

        elif action == open_folder_action:
            folder = os.path.dirname(self.image_path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

        elif action == save_as_action:
            target_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Image As",
                os.path.basename(self.image_path),
                "Images (*.png *.jpg *.jpeg)",
            )
            if target_path:
                QPixmap(self.image_path).save(target_path)
