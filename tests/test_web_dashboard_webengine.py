"""Browser smoke tests for the local dashboard and standalone CheckML mode."""

import os
from pathlib import Path

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView


pytestmark = pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    and os.environ.get("EASYROB_WEBENGINE_TEST") != "1",
    reason="Run the dashboard in a windowing platform (Xvfb on Linux).",
)


@pytest.mark.parametrize("mode", ["easyrob", "checkml"])
def test_dashboard_loads_and_applies_app_mode(qtbot, mode):
    view = QWebEngineView()
    qtbot.addWidget(view)
    view.resize(1000, 700)
    view.show()

    page_path = Path(__file__).resolve().parents[1] / "robert" / "gui_easyrob" / "web" / "app.html"
    url = QUrl.fromLocalFile(str(page_path))
    if mode == "checkml":
        url.setQuery("mode=checkml")
    with qtbot.waitSignal(view.loadFinished, timeout=30000) as loaded:
        view.setUrl(url)
    assert loaded.args == [True]

    def evaluate(script):
        values = []
        view.page().runJavaScript(script, values.append)
        qtbot.waitUntil(lambda: bool(values), timeout=15000)
        return values[0]

    assert evaluate("Boolean(window.easyrobApp)")
    assert evaluate("document.body.classList.contains('checkml-mode')") is (mode == "checkml")
    assert evaluate("document.querySelector('.page.active').id") == (
        "page-checkml" if mode == "checkml" else "page-workflow"
    )
    bot_toggle_display = evaluate(
        "getComputedStyle(document.querySelector('.sidebar [data-bot-toggle]')).display"
    )
    assert (bot_toggle_display != "none") is (mode == "easyrob")
    assert evaluate("window.easyrobApp.openDrawer('bot'); document.getElementById('bot-drawer').hidden") is (
        mode == "checkml"
    )
