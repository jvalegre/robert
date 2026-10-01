"""All-model result discovery and shared Results navigation."""

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QTableView, QWidget
import shiboken6

from gui_easyrob.tabs.result_catalog import ResultCatalog


def make_all_models_run(tmp_path, pdf_bytes, png_bytes):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "input.csv"
    source.write_text("target\n1\n")
    predict = tmp_path / "PREDICT"
    external = predict / "csv_test"
    external.mkdir(parents=True)
    reports = tmp_path / "REPORT_models"
    reports.mkdir()
    for model in ("GB", "MVL", "RF"):
        for suffix in ("No_PFI", "PFI"):
            (predict / f"{model}_{suffix}.csv").write_text("target,target_pred\n1,2\n")
            (predict / f"Results_boundary_williams_{model}_{suffix}.csv").write_text(
                "leverage\n0.2\n"
            )
            (external / f"test_{model}_{suffix}.csv").write_text(
                "target,target_pred\n1,2\n"
            )
            (reports / f"ROBERT_report_{model}_{suffix}.pdf").write_bytes(pdf_bytes)
            (predict / f"Results_{model}_{suffix}.png").write_bytes(png_bytes)
            (external / f"Results_{model}_{suffix}_external.png").write_bytes(png_bytes)
    best_root = tmp_path / "GENERATE" / "Best_model"
    for suffix, filename in (("No_PFI", "GB.csv"), ("PFI", "RF_PFI.csv")):
        folder = best_root / suffix
        folder.mkdir(parents=True)
        (folder / filename).write_text("seed,repeat_kfolds,kfold\n0,10,5\n")
    return source


def test_catalog_defaults_to_best_variant_models_and_can_select_another_model(
    tmp_path, tiny_test_pdf_bytes, tiny_test_png_bytes,
):
    make_all_models_run(tmp_path, tiny_test_pdf_bytes, tiny_test_png_bytes)

    catalog = ResultCatalog.discover(tmp_path)

    assert catalog.is_all_models
    assert catalog.models == ("GB", "MVL", "RF")
    assert catalog.best_by_variant == {"No_PFI": "GB", "PFI": "RF"}
    assert [path.name for path in catalog.report_paths(None)] == [
        "ROBERT_report_GB_No_PFI.pdf", "ROBERT_report_RF_PFI.pdf"
    ]
    assert {key: path.name for key, path in catalog.external_predictions(None).items()} == {
        "No_PFI": "test_GB_No_PFI.csv", "PFI": "test_RF_PFI.csv"
    }
    assert [path.name for path in catalog.report_paths("MVL")] == [
        "ROBERT_report_MVL_No_PFI.pdf", "ROBERT_report_MVL_PFI.pdf"
    ]
    assert {key: path.name for key, path in catalog.external_predictions("MVL").items()} == {
        "No_PFI": "test_MVL_No_PFI.csv", "PFI": "test_MVL_PFI.csv"
    }
    assert catalog.include_image(tmp_path / "PREDICT" / "Results_GB_No_PFI.png", None)
    assert not catalog.include_image(tmp_path / "PREDICT" / "Results_MVL_No_PFI.png", None)
    assert catalog.include_image(tmp_path / "PREDICT" / "Results_MVL_No_PFI.png", "MVL")


def test_results_model_selector_updates_all_views_without_extra_tabs(
    tmp_path, disposed_easyrob_window, qapp, tiny_test_pdf_bytes, tiny_test_png_bytes,
):
    source = make_all_models_run(tmp_path, tiny_test_pdf_bytes, tiny_test_png_bytes)
    window = disposed_easyrob_window
    try:
        window._pending_refresh_path = str(source)
        window._execute_refresh_tabs()
        selector = window.results_workspace.model_selector
        assert selector.isVisibleTo(window.results_workspace)
        assert [selector.itemText(index) for index in range(selector.count())] == [
            "Best models", "GB", "MVL", "RF"
        ]
        assert {Path(path).name for path in window.results_tab.pdf_tabs} == {
            "ROBERT_report_GB_No_PFI.pdf", "ROBERT_report_RF_PFI.pdf"
        }
        assert [path.name for path in window.interactive_plots.paths] == [
            "GB_No_PFI.csv", "RF_PFI.csv", "test_GB_No_PFI.csv", "test_RF_PFI.csv"
        ]
        selector.setCurrentText("MVL")
        assert {Path(path).name for path in window.results_tab.pdf_tabs} == {
            "ROBERT_report_MVL_No_PFI.pdf", "ROBERT_report_MVL_PFI.pdf"
        }
        assert [path.name for path in window.interactive_plots.paths] == [
            "MVL_No_PFI.csv", "MVL_PFI.csv", "test_MVL_No_PFI.csv", "test_MVL_PFI.csv"
        ]
        assert window.images_tab.view_selector.count() == 2
        assert {Path(window.images_tab.folder_widgets["PREDICT"].itemAt(index).widget().image_path).name
                for index in range(window.images_tab.folder_widgets["PREDICT"].count())} == {
            "Results_MVL_No_PFI.png", "Results_MVL_PFI.png"
        }
        assert {key: path.name for key, path in window.predictions_tab.csv_paths.items()} == {
            "No_PFI": "test_MVL_No_PFI.csv", "PFI": "test_MVL_PFI.csv"
        }
        assert {key: path.name for key, path in window.predictions_tab._report_paths.items()} == {
            "No_PFI": "ROBERT_report_MVL_No_PFI.pdf",
            "PFI": "ROBERT_report_MVL_PFI.pdf",
        }
        assert window.predictions_tab.subtabs.count() == 2
        for _ in range(100):
            qapp.processEvents()
            if all(
                window.predictions_tab.subtabs.widget(index).findChild(QTableView)
                for index in range(2)
            ):
                break
            QTest.qWait(10)
        assert all(
            window.predictions_tab.subtabs.widget(index).findChild(QTableView)
            for index in range(2)
        )
        assert window.results_workspace.content.count() == 4

        for path in (tmp_path / "PREDICT" / "csv_test").glob("test_MVL_*.csv"):
            path.unlink()
        window._execute_refresh_tabs()
        assert selector.currentText() == "MVL"
        assert not window.results_workspace.buttons["Predictions"].isEnabled()
        assert window.results_workspace.buttons["Report"].isEnabled()
    finally:
        window.hide()


def test_switching_runs_resets_model_choice_and_hides_unused_selector(
    tmp_path, disposed_easyrob_window, tiny_test_pdf_bytes, tiny_test_png_bytes,
):
    source = make_all_models_run(
        tmp_path / "all", tiny_test_pdf_bytes, tiny_test_png_bytes,
    )
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    ordinary_source = ordinary / "input.csv"
    ordinary_source.write_text("target\n1\n")
    (ordinary / "ROBERT_report_No_PFI.pdf").write_bytes(tiny_test_pdf_bytes)
    window = disposed_easyrob_window
    try:
        window._pending_refresh_path = str(source)
        window._execute_refresh_tabs()
        window.results_workspace.model_selector.setCurrentText("MVL")
        window._pending_refresh_path = str(ordinary_source)
        window._execute_refresh_tabs()
        assert not window.results_workspace.model_bar.isVisibleTo(window.results_workspace)
        assert {Path(path).name for path in window.results_tab.pdf_tabs} == {
            "ROBERT_report_No_PFI.pdf"
        }
        window._pending_refresh_path = str(source)
        window._execute_refresh_tabs()
        assert window.results_workspace.model_selector.currentText() == "Best models"
    finally:
        window.hide()


def test_switching_models_releases_old_pdf_viewers(
    tmp_path, disposed_easyrob_window, qapp, tiny_test_pdf_bytes, tiny_test_png_bytes,
):
    source = make_all_models_run(tmp_path, tiny_test_pdf_bytes, tiny_test_png_bytes)
    window = disposed_easyrob_window
    def viewer_count():
        return sum(
            child.__class__.__name__ == "PDFViewer"
            for child in window.results_tab.findChildren(QWidget)
        )
    try:
        window._pending_refresh_path = str(source)
        window._execute_refresh_tabs()
        window.results_workspace.show_view("Report")
        window.tab_widget.setCurrentWidget(window.results_workspace)
        window.show()
        selector = window.results_workspace.model_selector
        viewer_counts = []
        for model in ("GB", "MVL", "RF", "GB", "MVL"):
            selector.setCurrentText(model)
            qapp.processEvents()
            viewer_counts.append(viewer_count())
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert max(viewer_counts) >= 1
        assert viewer_count() <= 2
    finally:
        window.hide()


def test_predictions_refresh_releases_old_table_pages(tmp_path, qapp):
    from gui_easyrob.tabs.predictions import PredictionsTab

    predictions = PredictionsTab()
    old_page = QWidget()
    predictions.subtabs.addTab(old_page, "No PFI")

    predictions.refresh_with_new_path(str(tmp_path / "input.csv"))
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    assert not shiboken6.isValid(old_page)
