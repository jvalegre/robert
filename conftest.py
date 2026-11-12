"""Fixtures shared by repository and package-local tests."""

import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parent / "robert"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))


@pytest.fixture(autouse=True)
def isolate_gui_keyring(request, monkeypatch):
    """Use an in-memory keyring for headless tests that exercise the GUI."""
    test_path = Path(request.node.path)
    gui_test = (
        "gui_easyrob" in test_path.parts
        or "disposed_easyrob_window" in request.fixturenames
        or "easyrob_window" in request.fixturenames
        or test_path.name in {"test_web_shell.py", "test_bot_compact_gui.py"}
    )
    if sys.platform == "win32" or not gui_test:
        return

    import keyring

    saved = {}
    monkeypatch.setattr(keyring, "get_password", lambda service, account: saved.get((service, account)))
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda service, account, secret: saved.__setitem__((service, account), secret),
    )
    monkeypatch.setattr(
        keyring,
        "delete_password",
        lambda service, account: saved.pop((service, account), None),
    )
