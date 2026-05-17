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
        QTabWidget,
        QUrl,
        QVBoxLayout,
        QWidget,
        Qt,
    )

except ImportError as e:
    from robert.gui_easyrob.utils.utils_gui import (
        QApplication,
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
        QTabWidget,
        QUrl,
        QVBoxLayout,
        QWidget,
        Qt,
    )

# ---- Standard library ----
import os
import glob

class ImagesTab(QWidget):
    """Images tab for displaying images from multiple folders as workflow results."""

    def __init__(self, main_tab_widget, image_folders, file_path):
        super().__init__()

        # Store references and base workflow path
        self.main_tab_widget = main_tab_widget
        self.image_folders = image_folders
        self.base_path = os.path.dirname(file_path)
        self.folder_widgets = {}

        # Main layout
        self.layout = QVBoxLayout(self)

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

        # Tabs area for image folders
        self.folder_tabs = QTabWidget()
        self.layout.addWidget(self.folder_tabs)

        self.setLayout(self.layout)

        self.check_for_images()

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

    def refresh_with_new_path(self, file_path):
        """Update image base path and refresh image tabs."""
        self.base_path = os.path.dirname(file_path)
        self.clear_image_tabs()
        self.check_for_images()

    def clear_image_tabs(self):
        """Clear all image folders and their widgets."""
        for i in reversed(range(self.folder_tabs.count())):
            widget = self.folder_tabs.widget(i)
            if widget:
                widget.deleteLater()
            self.folder_tabs.removeTab(i)
        self.folder_widgets.clear()

    def check_for_images(self):
        """Scan folders and update tabs with new images."""
        folder_names = {
            "CURATE": "CURATE",
            "GENERATE/Raw_data": "GENERATE",
            "PREDICT": "PREDICT",
            "VERIFY": "VERIFY",
        }

        folder_order = ["CURATE", "GENERATE/Raw_data", "PREDICT", "VERIFY"]

        for folder in folder_order:
            full_folder_path = os.path.join(self.base_path, folder)
            if not os.path.exists(full_folder_path):
                continue

            image_files = sorted(
                glob.glob(os.path.join(full_folder_path, "*.[pjg][np][g]"))
            )

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
                self.folder_tabs.addTab(folder_widget, tab_name)
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