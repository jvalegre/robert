"""Browser check for the dedicated CheckML navigation."""

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --no-sandbox")

from PySide6.QtCore import QUrl, QUrlQuery
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView


def _wait_for(app, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_checkml_web_mode_starts_on_checkml_and_hides_other_workflows():
    app = QApplication.instance() or QApplication([])
    view = QWebEngineView()
    loaded = []
    view.loadFinished.connect(loaded.append)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parent / "web" / "app.html"))
    query = QUrlQuery()
    query.addQueryItem("mode", "checkml")
    url.setQuery(query)
    try:
        view.setUrl(url)
        assert _wait_for(app, lambda: bool(loaded))
        assert loaded == [True]
        results = []
        view.page().runJavaScript("""JSON.stringify({
            title: document.title,
            active: document.querySelector('.page.active').id,
            hidden: getComputedStyle(document.querySelector('[data-page="workflow"]')).display,
            checkml: getComputedStyle(document.querySelector('[data-page="checkml"]')).display,
            documentation: document.querySelector('[data-action="openDocs"]').getClientRects().length > 0,
            youtube: document.querySelector('[data-action="openYouTube"]').getClientRects().length > 0,
            contact: document.querySelector('[data-action="openContact"]').getClientRects().length > 0,
            version: document.querySelector('[data-action="openVersion"]').getClientRects().length > 0
        })""", results.append)
        assert _wait_for(app, lambda: bool(results))
        assert '"title":"CheckML"' in results[0]
        assert '"active":"page-checkml"' in results[0]
        assert '"hidden":"none"' in results[0]
        assert '"checkml":"flex"' in results[0]
        for action in ("documentation", "youtube", "contact", "version"):
            assert f'"{action}":true' in results[0]
    finally:
        view.close()
