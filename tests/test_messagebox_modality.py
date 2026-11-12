import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget

from gui_easyrob.utils import utils_gui


class TestMessageBoxModality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()
        self.parent.show()
        self.app.processEvents()

    def tearDown(self):
        self.parent.close()
        self.parent.deleteLater()
        self.app.processEvents()

    def test_constructed_message_box_defaults_to_window_modal(self):
        box = utils_gui.QMessageBox(self.parent)
        self.assertEqual(box.windowModality(), Qt.WindowModal)
        box.deleteLater()

    def test_static_warning_uses_window_modal(self):
        seen = {}

        def fake_exec(box):
            seen["modality"] = box.windowModality()
            return utils_gui.QMessageBox.Ok

        with patch.object(utils_gui.QMessageBox, "exec", new=fake_exec):
            result = utils_gui.QMessageBox.warning(self.parent, "Title", "Body")

        self.assertEqual(result, utils_gui.QMessageBox.Ok)
        self.assertEqual(seen["modality"], Qt.WindowModal)

    def test_static_question_uses_window_modal(self):
        seen = {}

        def fake_exec(box):
            seen["modality"] = box.windowModality()
            return utils_gui.QMessageBox.Yes

        with patch.object(utils_gui.QMessageBox, "exec", new=fake_exec):
            result = utils_gui.QMessageBox.question(
                self.parent,
                "Title",
                "Proceed?",
                utils_gui.QMessageBox.Yes | utils_gui.QMessageBox.No,
                utils_gui.QMessageBox.Yes,
            )

        self.assertEqual(result, utils_gui.QMessageBox.Yes)
        self.assertEqual(seen["modality"], Qt.WindowModal)


if __name__ == "__main__":
    unittest.main()
