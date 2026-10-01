"""Navigation container for report, prediction, image, and interactive results."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)
from gui_easyrob.result_navigation import choose_available_result_view


class ResultsWorkspace(QWidget):
    """Show one result view at a time while keeping each view alive."""

    availabilityChanged = Signal(bool)
    modelChanged = Signal(object)

    def __init__(self, report, predictions, images, interactive_plots, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.model_bar = QWidget()
        model_layout = QHBoxLayout(self.model_bar)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(QLabel("Model"))
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(190)
        self.model_selector.currentIndexChanged.connect(
            lambda index: self.modelChanged.emit(self.current_model())
        )
        model_layout.addWidget(self.model_selector)
        model_layout.addStretch()
        self.model_bar.hide()
        layout.addWidget(self.model_bar)
        selector = QHBoxLayout()
        selector.setSpacing(8)
        self.content = QStackedWidget()
        self.content.setFrameShape(QFrame.StyledPanel)
        self.buttons = {}
        self._available = {}
        for name, widget in (
            ("Report", report),
            ("Predictions", predictions),
            ("Images", images),
            ("Interactive plots", interactive_plots),
        ):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setEnabled(False)
            button.setMinimumWidth(100)
            button.setStyleSheet(
                "QPushButton:checked { background-color: #4559d8; color: white; "
                "font-weight: 600; }"
            )
            button.clicked.connect(lambda checked=False, view=name: self.show_view(view))
            self.buttons[name] = button
            self._available[name] = False
            selector.addWidget(button)
            self.content.addWidget(widget)
        selector.addStretch()
        layout.addLayout(selector)
        layout.addWidget(self.content, 1)

    def current_model(self):
        """Return a model name, or None for ROBERT's best models."""
        return self.model_selector.currentData()

    def set_models(self, models, preferred=None):
        """Offer one shared model selector only when additional models exist."""
        models = tuple(models)
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        if models:
            self.model_selector.addItem("Best models", None)
            for model in models:
                self.model_selector.addItem(model, model)
            if preferred in models:
                self.model_selector.setCurrentIndex(models.index(preferred) + 1)
        self.model_selector.blockSignals(False)
        self.model_bar.setVisible(bool(models))

    def has_available_views(self):
        """Whether any result source can be opened."""
        return any(self._available.values())

    def set_available(self, view, available):
        """Update a view's availability and select a valid fallback if needed."""
        button = self.buttons[view]
        available = bool(available)
        if self._available[view] == available:
            return
        self._available[view] = available
        button.setEnabled(available)
        selected_name = tuple(self.buttons)[self.content.currentIndex()]
        next_view = choose_available_result_view(
            self._available,
            selected_name,
            any(candidate.isChecked() for candidate in self.buttons.values()),
        )
        if next_view is not None:
            self.show_view(next_view)
        else:
            for candidate in self.buttons.values():
                candidate.setChecked(False)
        self.availabilityChanged.emit(self.has_available_views())

    def show_view(self, view):
        """Select an available view; return whether the switch happened."""
        if not self._available[view]:
            return False
        index = list(self.buttons).index(view)
        self.content.setCurrentIndex(index)
        for name, button in self.buttons.items():
            button.setChecked(name == view)
        return True
