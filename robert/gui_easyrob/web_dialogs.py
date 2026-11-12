"""Apply the web workspace palette to native modal dialogs."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QMessageBox


_DIALOG_STYLE = """
QDialog, QMessageBox, QInputDialog {
    background-color: #f8f5fc;
    color: #30213f;
    font-size: 12px;
}
QDialog QLabel, QMessageBox QLabel, QInputDialog QLabel {
    color: #3e2a52;
    font-size: 12px;
}
QDialog QPushButton, QMessageBox QPushButton, QInputDialog QPushButton {
    min-width: 74px;
    min-height: 27px;
    padding: 6px 13px;
    border: 1px solid #d2bde6;
    border-radius: 8px;
    background-color: #eee5f7;
    color: #523477;
    font-weight: 600;
}
QDialog QPushButton:hover, QMessageBox QPushButton:hover, QInputDialog QPushButton:hover {
    background-color: #e2d1f1;
}
QDialog QPushButton:default, QMessageBox QPushButton:default, QInputDialog QPushButton:default {
    background-color: #7340aa;
    border-color: #7340aa;
    color: white;
}
QDialog QLineEdit, QDialog QComboBox, QDialog QTextEdit, QDialog QPlainTextEdit {
    border: 1px solid #d8c9e6;
    border-radius: 8px;
    padding: 6px;
    background-color: white;
    color: #342548;
    selection-background-color: #c9a7e8;
}
"""


def style_dialog(dialog):
    """Style a modal once, preserving its existing buttons and return values."""
    if not isinstance(dialog, QDialog) or isinstance(dialog, QFileDialog):
        return
    if dialog.property("easyrobModernDialog"):
        return
    dialog.setProperty("easyrobModernDialog", True)
    dialog.setStyleSheet(dialog.styleSheet() + "\n" + _DIALOG_STYLE)
    if isinstance(dialog, QMessageBox):
        button_box = dialog.findChild(QDialogButtonBox)
        if button_box is not None:
            buttons = button_box.buttons()
            for button in buttons:
                text_width = button.fontMetrics().horizontalAdvance(button.text())
                button.setMinimumWidth(text_width + 36)
                button.setMinimumHeight(40)
            if len(buttons) >= 3:
                button_box.setOrientation(Qt.Vertical)
            dialog.adjustSize()
