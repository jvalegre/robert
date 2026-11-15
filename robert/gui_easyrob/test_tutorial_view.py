"""Browser checks for readable tutorial images in the web workspace."""

import os
import time
import json
from pathlib import Path

import pytest
from PySide6.QtCore import QUrl, QUrlQuery
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --no-sandbox")


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


@pytest.fixture(scope="module")
def tutorial_page():
    app = QApplication.instance() or QApplication([])
    view = QWebEngineView()
    view.resize(1180, 790)
    view.show()
    loaded = []
    view.loadFinished.connect(loaded.append)
    source = Path(__file__).resolve().parent / "web" / "app.html"
    view.setUrl(QUrl.fromLocalFile(str(source)))
    assert _wait_for(app, lambda: bool(loaded))
    assert loaded == [True]
    page = view.page()
    _evaluate(page, "window.easyrobApp.openDrawer('tutorial'); "
                    "document.getElementById('tutorial-image').src = "
                    "'../tutorials/tutorial_images/overview/overview_1.png'; true", app)
    assert _wait_for(app, lambda: _evaluate(page, "document.getElementById('tutorial-image').naturalWidth", app) == 1160)
    yield app, view, page
    view.close()


def test_tutorial_image_is_readable_without_opening_zoom(tutorial_page):
    app, view, page = tutorial_page
    tutorial_text = (Path(__file__).resolve().parent / "tutorials" / "overview.md").read_text(encoding="utf-8").split("---")[4].strip()
    _evaluate(page, "document.getElementById('tutorial-copy').innerHTML = " + json.dumps(tutorial_text) + "; "
              "document.getElementById('tutorial-title').textContent = 'Overview'; "
              "document.getElementById('tutorial-count').textContent = '17 STEPS'; "
              "document.getElementById('tutorial-position').textContent = '5 / 17'; "
              "document.getElementById('tutorial-topics').innerHTML = "
              "['Overview', 'From CSV', 'From ChemDraw', 'New Predictions', 'Generate Descriptors', 'Check ML', 'robBOT']"
              ".map(label => `<button class='tutorial-topic'>${label}</button>`).join(''); true", app)

    def geometry():
        return json.loads(_evaluate(page, "JSON.stringify((() => {"
                                    "const drawer = document.getElementById('tutorial-drawer');"
                                    "const image = document.getElementById('tutorial-image');"
                                    "const copy = document.getElementById('tutorial-copy');"
                                    "const visual = document.querySelector('.tutorial-visual');"
                                    "const navigation = document.querySelector('.tutorial-navigation');"
                                    "const bounds = drawer.getBoundingClientRect();"
                                    "return {drawer: bounds.width, drawerHeight: bounds.height,"
                                    "scrollHeight: drawer.scrollHeight, clientHeight: drawer.clientHeight,"
                                    "image: image.getBoundingClientRect().width,"
                                    "height: image.getBoundingClientRect().height,"
                                    "copyBottom: copy.getBoundingClientRect().bottom,"
                                    "visualTop: visual.getBoundingClientRect().top,"
                                    "visualBottom: visual.getBoundingClientRect().bottom,"
                                    "navigationTop: navigation.getBoundingClientRect().top,"
                                    "navigationBottom: navigation.getBoundingClientRect().bottom,"
                                    "drawerBottom: bounds.bottom};})())", app))

    wide = geometry()
    assert 1000 <= wide["drawer"] <= 1100
    assert wide["drawerHeight"] <= 740
    assert wide["image"] >= 700
    assert wide["height"] >= 250
    assert wide["scrollHeight"] <= wide["clientHeight"] + 1
    assert wide["navigationBottom"] <= wide["drawerBottom"]
    assert wide["copyBottom"] <= wide["visualTop"] < wide["visualBottom"] <= wide["navigationTop"]
    view.resize(800, 700)
    assert _wait_for(app, lambda: _evaluate(page, "innerWidth", app) == 800)
    narrow = geometry()
    assert narrow["drawer"] <= 800
    assert narrow["image"] >= 550
    assert narrow["height"] >= 180
    assert narrow["scrollHeight"] <= narrow["clientHeight"] + 1
    assert narrow["navigationBottom"] <= narrow["drawerBottom"]
    assert narrow["copyBottom"] <= narrow["visualTop"] < narrow["visualBottom"] <= narrow["navigationTop"]

    view.resize(850, 600)
    assert _wait_for(app, lambda: _evaluate(page, "innerWidth === 850 && innerHeight === 600", app))
    compact = geometry()
    assert compact["height"] >= 120
    assert compact["scrollHeight"] <= compact["clientHeight"] + 1
    assert compact["navigationBottom"] <= compact["drawerBottom"]
    assert compact["copyBottom"] <= compact["visualTop"] < compact["visualBottom"] <= compact["navigationTop"]


def test_tutorial_image_does_not_open_a_second_view(tutorial_page):
    app, view, page = tutorial_page
    view.resize(1180, 790)
    _evaluate(page, "document.getElementById('tutorial-image').click(); true", app)
    assert _evaluate(page, "document.querySelectorAll('[aria-modal=true]:not([hidden])').length", app) == 1
    assert _evaluate(page, "document.getElementById('tutorial-zoom') === null", app)
    _evaluate(page, "document.dispatchEvent(new KeyboardEvent('keydown', {key:'Escape', bubbles:true})); true", app)
    assert _evaluate(page, "document.getElementById('tutorial-drawer').hidden", app)
    _evaluate(page, "window.easyrobApp.openDrawer('tutorial'); true", app)
    _evaluate(page, "document.getElementById('tutorial-backdrop').click(); true", app)
    assert _evaluate(page, "document.getElementById('tutorial-drawer').hidden && document.getElementById('tutorial-backdrop').hidden", app)


def test_tutorial_and_bot_use_static_backgrounds(tutorial_page):
    app, view, page = tutorial_page
    backgrounds = _evaluate(page, "JSON.stringify({main:getComputedStyle(document.querySelector('.main')).backgroundImage, "
                            "bot:getComputedStyle(document.getElementById('bot-drawer')).backgroundImage, "
                            "backdrop:getComputedStyle(document.getElementById('tutorial-backdrop')).backdropFilter})", app)
    import json
    styles = json.loads(backgrounds)
    assert "easyrob_background.gif" not in styles["main"]
    assert "easyrob_background.gif" not in styles["bot"]
    assert styles["backdrop"] == "none"


def test_bot_send_label_stays_on_one_line(tutorial_page):
    app, view, page = tutorial_page
    view.resize(1180, 790)
    line_count = _evaluate(page, "window.easyrobApp.openDrawer('bot'); "
                           "const button=document.getElementById('bot-send'); "
                           "const range=document.createRange(); range.selectNodeContents(button); "
                           "range.getClientRects().length", app)
    assert line_count == 1


def test_bot_rail_uses_its_width_and_keeps_settings_controls_inside(tutorial_page):
    app, view, page = tutorial_page

    def layout():
        return json.loads(_evaluate(page, "JSON.stringify((() => {"
                                    "const drawer = document.getElementById('bot-drawer');"
                                    "const conversation = drawer.querySelector('.bot-conversation');"
                                    "const settings = drawer.querySelector('.bot-settings');"
                                    "const bounds = drawer.getBoundingClientRect();"
                                    "const navigationSingleLine = [...document.querySelectorAll('.sidebar .nav-item')].every(button => {"
                                    "const range = document.createRange(); range.selectNodeContents(button.lastChild);"
                                    "return range.getClientRects().length <= 1;});"
                                    "const controls = [...drawer.querySelectorAll('button, input, select')].filter(node =>"
                                    "getComputedStyle(node).display !== 'none' && node.getBoundingClientRect().width);"
                                    "return {drawerWidth: bounds.width, scrollWidth: drawer.scrollWidth,"
                                    "clientWidth: drawer.clientWidth,"
                                    "conversationWidth: conversation.getBoundingClientRect().width,"
                                    "settingsWidth: settings.getBoundingClientRect().width,"
                                    "settingsBottom: settings.getBoundingClientRect().bottom,"
                                    "drawerBottom: bounds.bottom,"
                                    "navigationSingleLine,"
                                    "controlsInside: controls.every(node => {const rect = node.getBoundingClientRect();"
                                    "return rect.left >= bounds.left - 1 && rect.right <= bounds.right + 1;}),"
                                    "messageFont: parseFloat(getComputedStyle(drawer.querySelector('.bot-message p')).fontSize),"
                                    "messagesHeight: drawer.querySelector('.bot-messages').getBoundingClientRect().height};"
                                    "})())", app))

    for width, height in ((1180, 790), (1160, 720), (850, 600)):
        view.resize(width, height)
        assert _wait_for(app, lambda: _evaluate(page, f"innerWidth === {width} && innerHeight === {height}", app))
        _evaluate(page, "window.easyrobApp.openDrawer('bot'); "
                  "document.querySelector('.bot-settings').open = false; "
                  "document.getElementById('bot-messages').innerHTML = "
                  "'<div class=\"bot-message assistant\"><strong>robBOT</strong><p>Ask about your workflow, results, or AQME and ROBERT.</p></div>'; true", app)
        closed = layout()
        assert closed["settingsWidth"] >= closed["conversationWidth"] - 2
        assert closed["drawerBottom"] - closed["settingsBottom"] <= 24
        _evaluate(page, "document.querySelector('.bot-settings').open = true; true", app)
        measured = layout()
        assert measured["scrollWidth"] <= measured["clientWidth"] + 1
        assert measured["controlsInside"]
        assert measured["navigationSingleLine"]
        assert 12 <= measured["messageFont"] <= 13
        assert measured["messagesHeight"] <= height * 0.4


def test_checkml_web_mode_has_no_bot_drawer():
    app = QApplication.instance() or QApplication([])
    view = QWebEngineView()
    view.resize(1180, 790)
    view.show()
    loaded = []
    view.loadFinished.connect(loaded.append)
    source = Path(__file__).resolve().parent / "web" / "app.html"
    url = QUrl.fromLocalFile(str(source))
    query = QUrlQuery()
    query.addQueryItem("mode", "checkml")
    url.setQuery(query)
    view.setUrl(url)
    try:
        assert _wait_for(app, lambda: bool(loaded))
        assert loaded == [True]
        page = view.page()
        assert _evaluate(page, "getComputedStyle(document.querySelector('[data-bot-toggle]')).display", app) == "none"
        _evaluate(page, "window.easyrobApp.openDrawer('bot'); true", app)
        assert _evaluate(page, "document.getElementById('bot-drawer').hidden", app)
    finally:
        view.close()
