"""Layout regression checks for styled multi-action message boxes."""

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

from gui_easyrob.web_dialogs import style_dialog


def test_descriptor_choices_keep_full_button_labels():
    app = QApplication.instance() or QApplication([])
    dialog = QMessageBox()
    dialog.setWindowTitle("AQME completed successfully")
    dialog.setText("Select a file to load into ROBERT, or cancel to continue.")
    labels = (
        "Interpret descriptors (recommended)",
        "DeNovo descriptors",
        "Full descriptors",
        "Cancel",
    )
    for label in labels:
        dialog.addButton(label, QMessageBox.ActionRole)
    style_dialog(dialog)
    dialog.show()
    app.processEvents()
    try:
        button_box = dialog.findChild(QDialogButtonBox)
        assert button_box is not None
        for button in button_box.buttons():
            required = button.fontMetrics().horizontalAdvance(button.text()) + 32
            assert button.width() >= required, (button.text(), button.width(), required)
            assert dialog.rect().contains(button.mapTo(dialog, button.rect().topLeft()))
            assert dialog.rect().contains(button.mapTo(dialog, button.rect().bottomRight()))
    finally:
        dialog.close()
