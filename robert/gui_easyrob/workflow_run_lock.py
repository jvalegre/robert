"""Coordinate GUI workflow runs across EasyROB and CheckML processes."""

import tempfile
from pathlib import Path

from PySide6.QtCore import QLockFile


class WorkflowRunLock:
    """Hold a single user-level lock for workflows that write ROBERT results."""

    def __init__(self, path=None):
        lock_path = Path(path) if path is not None else Path(tempfile.gettempdir()) / "robert_gui_workflow.lock"
        self._lock = QLockFile(str(lock_path))
        self._lock.setStaleLockTime(0)
        self._held = False

    def try_acquire(self):
        if self._held:
            return False
        self._held = self._lock.tryLock(0)
        return self._held

    def release(self):
        if self._held:
            self._lock.unlock()
            self._held = False
