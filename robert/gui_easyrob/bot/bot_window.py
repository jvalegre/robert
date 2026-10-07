"""Floating top-level window that hosts the robBOT panel."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QHideEvent, QShowEvent
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .bot_panel import BotPanel

__all__ = ["BotWindow"]


class BotWindow(QWidget):
    visibility_changed = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("robBOT")
        self.resize(700, 820)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.panel = BotPanel(self)
        self.panel.close_requested.connect(self.hide)
        layout.addWidget(self.panel)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        super().closeEvent(event)
