"""Fixtures shared by repository and package-local tests."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parent / "robert"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))


@pytest.fixture(autouse=True)
def isolate_gui_keyring(request, monkeypatch):
    """Keep headless GUI tests independent of the desktop secret service."""
    test_path = Path(request.node.path)
    gui_test = (
        "gui_easyrob" in test_path.parts
        or test_path.name in {
            "test_easyrob.py",
            "test_web_shell.py",
            "test_bot_compact_gui.py",
            "test_web_dashboard_webengine.py",
        }
    )
    if sys.platform == "win32" or not gui_test:
        return

    saved = {}
    keyring = SimpleNamespace(
        get_password=lambda service, account: saved.get((service, account)),
        set_password=lambda service, account, secret: saved.__setitem__((service, account), secret),
        delete_password=lambda service, account: saved.pop((service, account), None),
    )
    for module_name in (
        "robert.gui_easyrob.bot.credential_store",
        "gui_easyrob.bot.credential_store",
        "bot.credential_store",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(
                module.CredentialStore,
                "_keyring",
                staticmethod(lambda backend=keyring: backend),
            )


