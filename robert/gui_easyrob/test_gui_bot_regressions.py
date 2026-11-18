"""Regression tests for workflow execution and robBOT lifecycle behavior."""

from pathlib import Path
from types import SimpleNamespace
import os
import shlex
import sys
import tempfile

import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.bot_rag import KnowledgeBase
from gui_easyrob.bot.bot_panel import BotPanel
from gui_easyrob.main.window import EasyROB
from gui_easyrob.utils.utils_gui import RobertWorker


class _Choice:
    def __init__(self, text):
        self.text = text

    def currentText(self):
        return self.text


def test_robbot_exposes_only_cloud_and_heuristic_modes():
    app = QApplication.instance() or QApplication([])
    panel = BotPanel()
    try:
        modes = [panel.mode_combo.itemText(index) for index in range(panel.mode_combo.count())]
        assert modes == ["Cloud AI", "Heuristic"]
        assert panel.mode_combo.currentText() == "Cloud AI"
    finally:
        panel.close()


def test_retired_local_mode_is_rejected_by_engine():
    engine = BotEngine(knowledge_base=KnowledgeBase.from_records([]))
    snapshot = GuiSnapshot(
        active_tab="ROBERT",
        workflow="Full Workflow",
        main_csv_path="",
        test_csv_path="",
        target_column="",
        prediction_type="Regression",
        process_running=False,
        run_enabled=False,
        stop_enabled=False,
        recent_console="",
        console_terms=[],
    )
    answer = engine.answer("Hello", snapshot, "Local AI", "Groq", "")
    assert answer == "Unsupported bot mode."


class _Toggle:
    def __init__(self, checked):
        self.checked = checked

    def isChecked(self):
        return self.checked

    def setChecked(self, value):
        self.checked = value


class _TextField:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value

    def clear(self):
        self.value = ""


class _SettingsPanel:
    def __init__(self, api_key=""):
        self.api_key_edit = _TextField(api_key)
        self.provider_combo = _Choice("Groq")
        self.provider_combo.findText = lambda value: 0 if value == "Groq" else -1
        self.provider_combo.setCurrentIndex = lambda index: None
        self.mode_combo = _Choice("Cloud AI")
        self.mode_combo.setCurrentText = lambda value: None
        self.web_search_checkbox = _Toggle(True)
        self.saved = False

    def set_saved_api_state(self, saved):
        self.saved = saved


class _CommandState:
    def __init__(self, workflow="Full Workflow"):
        self.workflow_selector = _Choice(workflow)
        self.all_models_toggle = _Toggle(False)
        self.y_dropdown = _Choice("target")
        self.names_dropdown = _Choice("name")
        self.type_dropdown = _Choice("Regression")
        self.ignore_list = SimpleNamespace(count=lambda: 0)
        self.aqme_workflow = _Toggle(False)
        self.tab_widget_aqme = SimpleNamespace(selected_atoms=[])
        self.csv_test_path = ""
        self.auto_type_value = True
        self.split_value = "even"
        self.categorical_value = "onehot"
        self.corr_filter_x_value = True
        self.corr_filter_y_value = False
        self.rfecv_filter_value = True
        self.pfi_filter_value = True
        self.auto_test_value = True

    def _resolve_python_executable(self):
        return "python"

    def __getattr__(self, name):
        return 0


def test_qdescp_atoms_and_solvent_are_passed_together():
    state = _CommandState()
    state.aqme_workflow = _Toggle(True)
    state.tab_widget_aqme.selected_atoms = [object()]
    state.descriptor_level_selected = "interpret"
    state.atoms_selected = ""
    state.solvent_selected = "water"

    command = EasyROB.build_robert_command(state, "train.csv")
    arguments = shlex.split(command)
    keyword_values = [
        arguments[index + 1]
        for index, value in enumerate(arguments[:-1])
        if value == "--qdescp_keywords"
    ]
    assert keyword_values == ["--qdescp_atoms [1] --qdescp_solvent water"]


def test_external_test_csv_keeps_its_absolute_path(tmp_path):
    state = _CommandState(workflow="PREDICT")
    test_path = tmp_path / "external" / "test.csv"
    state.csv_test_path = str(test_path)

    command = EasyROB.build_robert_command(state, str(tmp_path / "train" / "train.csv"))
    arguments = shlex.split(command)
    assert arguments[arguments.index("--csv_test") + 1] == str(test_path)


def test_worker_reports_process_start_failure_as_finished():
    worker = RobertWorker("missing_easyrob_executable_234981")
    errors = []
    exit_codes = []
    worker.error_received.connect(errors.append)
    worker.process_finished.connect(exit_codes.append)

    worker.run()

    assert errors
    assert exit_codes == [1]


def test_clearing_chat_discards_history():
    class Input:
        def clear(self):
            pass

        def setFocus(self, *_):
            pass

    panel = SimpleNamespace(input_edit=Input(), clear_history=lambda: None)
    state = SimpleNamespace(
        _bot_conversation_generation=0,
        bot_worker=None,
        bot_history=[{"role": "user", "content": "old question"}],
        bot_panel=panel,
    )

    EasyROB.clear_bot_history(state)

    assert state.bot_history == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI regression")
def test_saved_api_key_is_protected_and_can_be_deleted():
    from gui_easyrob.bot.credential_store import CredentialStore

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        settings_path = Path(temp_dir) / "settings.ini"
        settings = QSettings(str(settings_path), QSettings.IniFormat)
        store = CredentialStore(settings)

        store.save("example-secret-key")
        settings.sync()

        assert settings.value("bot/api_key") is None
        assert "example-secret-key" not in settings_path.read_text(encoding="utf-8")
        assert store.load() == "example-secret-key"

        store.delete()
        assert store.load() == ""


def test_non_windows_api_key_uses_system_keyring(monkeypatch):
    from gui_easyrob.bot import credential_store

    saved = {}
    keyring = SimpleNamespace(
        set_password=lambda service, account, secret: saved.__setitem__((service, account), secret),
        get_password=lambda service, account: saved.get((service, account)),
        delete_password=lambda service, account: saved.pop((service, account)),
    )
    monkeypatch.setattr(credential_store.sys, "platform", "linux")
    monkeypatch.setattr(credential_store.CredentialStore, "_keyring", staticmethod(lambda: keyring))

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        settings_path = Path(temp_dir) / "settings.ini"
        settings = QSettings(str(settings_path), QSettings.IniFormat)
        store = credential_store.CredentialStore(settings)

        store.save("example-secret-key")
        assert store.load() == "example-secret-key"
        assert settings.value("bot/api_key") is None
        assert settings.value("bot/api_key_protected") is None
        if settings_path.exists():
            assert "example-secret-key" not in settings_path.read_text(encoding="utf-8")

        store.delete()
        assert store.load() == ""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI regression")
def test_legacy_plaintext_api_key_is_migrated():
    from gui_easyrob.bot.credential_store import CredentialStore

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        settings_path = Path(temp_dir) / "settings.ini"
        settings = QSettings(str(settings_path), QSettings.IniFormat)
        settings.setValue("bot/api_key", "legacy-secret")
        settings.sync()

        store = CredentialStore(settings)

        assert store.load() == "legacy-secret"
        assert settings.value("bot/api_key") is None
        assert "legacy-secret" not in settings_path.read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI regression")
def test_loading_protected_api_key_removes_leftover_plaintext():
    from gui_easyrob.bot.credential_store import CredentialStore

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        settings = QSettings(str(Path(temp_dir) / "settings.ini"), QSettings.IniFormat)
        store = CredentialStore(settings)
        store.save("protected-secret")
        settings.setValue("bot/api_key", "stale-plaintext")
        settings.sync()

        assert store.load() == "protected-secret"
        assert settings.value("bot/api_key") is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI regression")
def test_gui_loads_api_key_from_protected_storage():
    from gui_easyrob.bot.credential_store import CredentialStore

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        settings = QSettings(str(Path(temp_dir) / "settings.ini"), QSettings.IniFormat)
        CredentialStore(settings).save("protected-secret")
        panel = _SettingsPanel()
        state = SimpleNamespace(bot_settings=settings, bot_panel=panel)

        EasyROB._load_bot_preferences(state)

        assert panel.api_key_edit.text() == "protected-secret"
        assert panel.saved is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI regression")
def test_gui_saves_api_key_without_plaintext(monkeypatch):
    from gui_easyrob.main import window as window_module

    monkeypatch.setattr(window_module.QMessageBox, "information", lambda *args: None)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        settings = QSettings(str(Path(temp_dir) / "settings.ini"), QSettings.IniFormat)
        panel = _SettingsPanel("new-secret")
        state = SimpleNamespace(bot_settings=settings, bot_panel=panel)

        EasyROB.save_bot_api_preferences(state)

        assert settings.value("bot/api_key") is None
        assert panel.saved is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI regression")
def test_api_key_save_reports_settings_write_failure():
    from gui_easyrob.bot.credential_store import CredentialStore, CredentialStoreError

    class UnwritableSettings:
        def setValue(self, *_):
            pass

        def remove(self, *_):
            pass

        def sync(self):
            pass

        def status(self):
            return QSettings.AccessError

    with pytest.raises(CredentialStoreError):
        CredentialStore(UnwritableSettings()).save("secret")
