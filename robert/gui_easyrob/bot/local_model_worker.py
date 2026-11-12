"""Background worker for downloading and loading the Local AI model."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

__all__ = ["LocalModelPrepareWorker"]


class LocalModelPrepareWorker(QThread):
    status_changed = Signal(str, str)
    prepared = Signal()
    failed = Signal(str)

    def __init__(self, local_manager, parent=None) -> None:
        super().__init__(parent)
        self.local_manager = local_manager
        self._last_status: str | None = None

    def _publish_status(self, status: str, detail: str) -> None:
        normalized_status = str(status)
        if normalized_status == self._last_status:
            return
        self._last_status = normalized_status
        self.status_changed.emit(normalized_status, str(detail))

    def _emit_progress(self, stage: str, message: str) -> None:
        normalized = str(stage).strip().lower()
        if normalized == "downloading":
            status = "Downloading..."
        elif normalized == "loading":
            status = "Loading..."
        else:
            status = "Preparing..."
        self._publish_status(status, str(message))

    def run(self) -> None:
        try:
            self._publish_status("Preparing...", "Checking the local model cache...")
            if not self.local_manager.model_exists():
                self._publish_status(
                    "Downloading...",
                    "Downloading the local model. This can take several minutes.",
                )
                self.local_manager.download_model(progress_callback=self._emit_progress)
            if hasattr(self.local_manager, "runtime_exists") and not self.local_manager.runtime_exists():
                self._publish_status(
                    "Downloading...",
                    "Downloading the local AI runtime for this platform...",
                )
                self.local_manager.download_runtime(progress_callback=self._emit_progress)
            self._publish_status("Loading...", "Loading the local model into memory...")
            self.local_manager.load_model()
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self.failed.emit(message)
            return

        self._publish_status("Ready", "Local AI is ready.")
        self.prepared.emit()
