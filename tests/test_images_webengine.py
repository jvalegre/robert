"""One real Chromium check for the embedded image gallery."""

import os
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from gui_easyrob.tabs import images


pytestmark = pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    and os.environ.get("EASYROB_WEBENGINE_TEST") != "1",
    reason="Run this browser test separately with a windowing platform (Xvfb on Linux).",
)


def _wait_for(app, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _evaluate(page, script, app):
    values = []
    page.runJavaScript(script, values.append)
    assert _wait_for(app, lambda: bool(values))
    return values[0]


def test_gallery_renders_and_open_action_reaches_python(tmp_path, monkeypatch):
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)
    app = QApplication.instance() or QApplication([])
    output = tmp_path / "PREDICT"
    output.mkdir()
    image_path = output / "prediction.png"
    image = QImage(2, 2, QImage.Format.Format_RGB32)
    image.fill(0x336699)
    assert image.save(str(image_path))
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("target\n1\n", encoding="utf-8")
    opened = []
    monkeypatch.setattr(images.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)

    tab = images.ImagesTab(None, [], str(csv_path))
    try:
        tab.resize(1100, 750)
        tab.show()
        tab.activate_web()
        assert _wait_for(app, lambda: tab._web_ready)
        page = tab._web_view.page()
        assert _wait_for(
            app,
            lambda: _evaluate(page, "document.querySelectorAll('.image-card').length", app) == 1,
        )
        assert _evaluate(page, "document.querySelector('.image-card h2').textContent", app) == "prediction.png"
        assert _evaluate(page, "window.easyrobGallery.bridgeReady", app)
        screenshot = os.environ.get("EASYROB_WEBENGINE_SCREENSHOT")
        if screenshot:
            _wait_for(app, lambda: False, timeout=1)
            assert tab._web_view.grab().save(screenshot)
        _evaluate(page, "document.querySelector('[data-action=\"open\"]').click(); true", app)
        assert _wait_for(app, lambda: len(opened) == 1)
        assert Path(opened[0].toLocalFile()) == image_path
    finally:
        tab.close()
