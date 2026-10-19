from types import SimpleNamespace

from gui_easyrob.tabs.checkml import CheckMLTab
from gui_easyrob.main.window import EasyROB


def test_checkml_completion_refreshes_sibling_tabs_from_selected_csv():
    class MainWindow:
        def __init__(self):
            self.refreshed_paths = []

        def refresh_tabs(self, path):
            self.refreshed_paths.append(path)

    class CheckMLView:
        csv_path = "C:/runs/train.csv"

        def __init__(self, main_window):
            self.main_window = main_window

        def window(self):
            return self.main_window

    main_window = MainWindow()
    CheckMLTab._refresh_sibling_tabs(CheckMLView(main_window))

    assert main_window.refreshed_paths == ["C:/runs/train.csv"]


def test_chat_result_snapshot_reads_active_checkml_run():
    class Store:
        def __init__(self):
            self.calls = []

        def refresh(self, main_csv, test_csv, process_running):
            self.calls.append((main_csv, test_csv, process_running))
            return None

    store = Store()
    fake_window = SimpleNamespace(
        tab_widget=SimpleNamespace(currentIndex=lambda: 0, tabText=lambda index: "CheckML"),
        checkml_tab=SimpleNamespace(
            csv_path="C:/runs/checkml.csv",
            csv_test_path="C:/runs/external.csv",
            worker=object(),
        ),
        file_path="C:/runs/other.csv",
        csv_test_path="",
        worker=None,
        workflow_result_store=store,
        bot_panel=SimpleNamespace(set_workflow_summary_available=lambda *args, **kwargs: None),
    )

    EasyROB._refresh_workflow_result_snapshot(fake_window)

    assert store.calls == [("C:/runs/checkml.csv", "C:/runs/external.csv", True)]
