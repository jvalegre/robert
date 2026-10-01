"""Data checks for interactive prediction figures."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PySide6.QtWidgets import QTabWidget

from gui_easyrob.tabs.interactive_predictions import (
    _source_frames,
    discover_prediction_csvs,
    prepare_prediction_data,
    InteractivePredictions,
)
from gui_easyrob.tabs.images import ImagesTab
from robert.utils import _williams_plot


def test_discovers_internal_and_optional_external_csvs(tmp_path):
    predict = tmp_path / "PREDICT"
    external = predict / "csv_test"
    external.mkdir(parents=True)
    (predict / "GB_No_PFI.csv").write_text("target,target_pred\n1,2\n")
    (predict / "GB_PFI.csv").write_text("target,target_pred\n1,2\n")
    (external / "new_GB_No_PFI.csv").write_text("target,target_pred\n1,2\n")
    (predict / "other.csv").write_text("x\n1\n")
    (predict / "Results_boundary_williams_GB_No_PFI.csv").write_text("leverage\n0.2\n")

    found = discover_prediction_csvs(tmp_path)

    assert [path.name for path in found] == [
        "GB_No_PFI.csv", "GB_PFI.csv", "new_GB_No_PFI.csv"
    ]


def test_prepares_diagnostics_and_joins_molecules_by_unique_identifier():
    predictions = pd.DataFrame({
        "code_name": [2, 1], "descriptor": [2.0, 1.0],
        "target": [2.0, 1.0], "target_pred": [2.5, 1.5],
        "target_pred_sd": [0.2, 0.1], "Set": ["Test", "CV"],
    })
    source = pd.DataFrame({"code_name": [1, 2], "SMILES": ["CCO", "CC(=O)O"]})

    result = prepare_prediction_data(predictions, [source])

    assert result["smiles"] == ["CC(=O)O", "CCO"]
    assert result["residual"].tolist() == [0.5, 0.5]
    assert result["target"] == "target"


def test_unknown_observations_do_not_produce_residuals():
    predictions = pd.DataFrame({
        "code_name": ["a"], "target": [float("nan")],
        "target_pred": [3.0], "target_pred_sd": [0.2],
    })

    result = prepare_prediction_data(predictions, [])

    assert pd.isna(result["residual"][0])
    assert result["smiles"] == [None]


def test_external_molecule_uses_external_input_when_ids_overlap(tmp_path):
    predict = tmp_path / "PREDICT"
    predict.mkdir()
    (tmp_path / "training.csv").write_text("code_name,SMILES\n1,CCO\n")
    (tmp_path / "external.csv").write_text("code_name,SMILES\n1,CC(=O)O\n")
    (predict / "PREDICT_data.dat").write_text(
        'Command line used in ROBERT: --csv_test "external.csv"'
    )
    frame = pd.DataFrame({"code_name": [1], "target": [1], "target_pred": [2]})

    sources = _source_frames(tmp_path, tmp_path / "training.csv", external=True)
    result = prepare_prediction_data(frame, sources)

    assert result["smiles"] == ["CC(=O)O"]


def test_all_models_external_molecule_uses_its_model_log(tmp_path):
    predict = tmp_path / "PREDICT"
    predict.mkdir()
    (tmp_path / "training.csv").write_text("code_name,SMILES\n1,CCO\n")
    (tmp_path / "external.csv").write_text("code_name,SMILES\n1,CC(=O)O\n")
    (predict / "PREDICT_GB_data.dat").write_text(
        'Command line used in ROBERT: --csv_test "external.csv"'
    )
    frame = pd.DataFrame({"code_name": [1], "target": [1], "target_pred": [2]})

    sources = _source_frames(
        tmp_path, tmp_path / "training.csv", external=True, model="GB"
    )
    result = prepare_prediction_data(frame, sources)

    assert result["smiles"] == ["CC(=O)O"]


def test_nonbest_model_plot_uses_its_own_cv_metadata(tmp_path, qapp):
    predict = tmp_path / "PREDICT"
    predict.mkdir()
    csv_path = predict / "MVL_No_PFI.csv"
    csv_path.write_text("target,target_pred,Set\n1,2,CV\n2,3,Test\n")
    metadata = tmp_path / "GENERATE" / "All_models" / "No_PFI" / "MVL"
    metadata.mkdir(parents=True)
    (metadata / "MVL.csv").write_text("seed,repeat_kfolds,kfold\n7,3,4\n")
    source = tmp_path / "input.csv"
    source.write_text("target\n1\n")

    panel = InteractivePredictions()
    panel.refresh(tmp_path, str(source), paths=[csv_path])

    assert panel.seed == 7
    assert panel.cv_label == "3x 4-fold CV"


def test_outlier_scores_match_robert_cv_absolute_error_scaling():
    frame = pd.DataFrame({
        "target": [0.0, 0.0, 0.0, 0.0],
        "target_pred": [1.0, -3.0, 5.0, -7.0],
        "Set": ["CV", "CV", "CV", "Test"],
    })

    data = prepare_prediction_data(frame, [])

    # CV absolute errors 1, 3, 5 have mean 3 and population SD sqrt(8/3).
    scale = (8 / 3) ** 0.5
    assert data["outlier_score"].tolist() == [
        -2 / scale, 0.0, 2 / scale, 4 / scale
    ]


def test_outlier_scores_use_internal_cv_reference_for_external_csv():
    reference = pd.DataFrame({
        "target": [0.0, 0.0, 0.0], "target_pred": [1.0, 3.0, 5.0],
        "Set": ["CV", "CV", "CV"],
    })
    external = pd.DataFrame({"target": [0.0], "target_pred": [-7.0]})

    data = prepare_prediction_data(external, [], reference)

    assert data["outlier_score"][0] == 4 / (8 / 3) ** 0.5


def test_prediction_chart_uses_cv_regression_instead_of_identity_line(qapp):
    frame = pd.DataFrame({
        "target": list(range(12)),
        "target_pred": [2 * value + 1 for value in range(12)],
        "Set": ["CV"] * 12,
    })
    panel = InteractivePredictions()
    panel.paths = [Path("PREDICT/model_No_PFI.csv")]
    panel.selector.addItem("PREDICT / model_No_PFI")
    panel.data = prepare_prediction_data(frame, [])

    panel._draw_charts()

    lines = panel.canvases[0].figure.axes[0].lines
    assert len(lines) == 1
    x, y = lines[0].get_data()
    assert abs((y[-1] - y[0]) / (x[-1] - x[0]) - 2) < 0.01


def test_molecule_preview_keeps_its_size_on_repeated_hover(qapp):
    frame = pd.DataFrame({"code_name": [1], "SMILES": ["CCO"],
                          "target": [1.0], "target_pred": [1.5]})
    panel = InteractivePredictions()
    panel.data = prepare_prediction_data(frame, [])
    panel.resize(900, 650)
    panel.show()
    qapp.processEvents()

    sizes = []
    for _ in range(6):
        panel._show_point(0)
        qapp.processEvents()
        sizes.append((panel.structure.width(), panel.structure.height()))

    assert len(set(sizes)) == 1


def test_interactive_charts_use_one_view_selector_instead_of_nested_tabs(qapp):
    panel = InteractivePredictions()

    assert not isinstance(panel.charts, QTabWidget)
    assert panel.charts.count() == 3
    assert [button.text() for button in panel.plot_buttons] == [
        "Observed vs predicted", "Outlier analysis", "Williams plot"
    ]


def test_images_navigation_contains_only_image_views(tmp_path, qapp):
    predict = tmp_path / "PREDICT"
    predict.mkdir()
    (predict / "GB_No_PFI.csv").write_text("target,target_pred\n1,2\n")
    selected = tmp_path / "input.csv"
    selected.write_text("target\n1\n")

    tab = ImagesTab(None, ["PREDICT"], str(selected))

    assert not isinstance(tab.folder_tabs, QTabWidget)
    assert tab.view_selector.count() == 0
    (predict / "figure.png").write_bytes(b"placeholder")
    tab.refresh_with_new_path(str(selected))
    assert [tab.view_selector.itemText(i) for i in range(tab.view_selector.count())] == ["PREDICT"]
    tab.refresh_with_new_path(str(selected))
    assert tab.view_selector.currentText() == "PREDICT"


def test_williams_generator_exports_exact_plot_coordinates(tmp_path):
    plot_base = tmp_path / "GB_No_PFI"
    _williams_plot(
        str(plot_base), np.array([0.1, 0.2]), np.array([-1.0, 0.5]),
        np.array([0.8]), np.array([3.4]), 0.6,
        point_ids=["a", "b", "c"], name_column="code_name",
        observed=[1.0, 2.0, 3.0], predicted=[2.0, 1.5, -0.4],
    )

    sidecar = pd.read_csv(tmp_path / "Results_boundary_williams_GB_No_PFI.csv")
    assert sidecar["point_id"].tolist() == ["a", "b", "c"]
    assert sidecar["name_column"].tolist() == ["code_name"] * 3
    assert sidecar["group"].tolist() == ["Typical 80%", "Typical 80%", "High leverage 20%"]
    assert sidecar["leverage"].tolist() == [0.1, 0.2, 0.8]
    assert sidecar["standardized_residual"].tolist() == [-1.0, 0.5, 3.4]
    assert sidecar["h_star"].tolist() == [0.6] * 3
    assert sidecar["observed"].tolist() == [1.0, 2.0, 3.0]
    assert sidecar["predicted"].tolist() == [2.0, 1.5, -0.4]


def test_williams_view_uses_matching_model_sidecar_and_falls_back_to_png(tmp_path, qapp):
    predict = tmp_path / "PREDICT"
    predict.mkdir()
    paths = [predict / "GB_No_PFI.csv", predict / "RF_No_PFI.csv"]
    for path in paths:
        path.write_text("code_name,target,target_pred,Set\na,1,2,CV\n")
    pd.DataFrame({
        "point_id": ["a"], "name_column": ["code_name"],
        "group": ["High leverage 20%"], "leverage": [0.8],
        "standardized_residual": [3.4], "h_star": [0.6],
        "observed": [1.0], "predicted": [2.0],
    }).to_csv(predict / "Results_boundary_williams_GB_No_PFI.csv", index=False)
    Figure(figsize=(2, 2)).savefig(predict / "Results_boundary_williams_RF_No_PFI.png")
    selected = tmp_path / "input.csv"
    selected.write_text("code_name,SMILES\na,CCO\n")

    panel = InteractivePredictions()
    panel.refresh(tmp_path, str(selected), paths=paths)

    assert panel.charts.count() == 3
    assert panel.williams_data["leverage"].tolist() == [0.8]
    assert panel.canvases[2].figure.axes[0].get_xlabel() == "Leverage"
    panel.selector.setCurrentIndex(1)
    assert panel.williams_data is None
    assert "interactive data unavailable" in panel.canvases[2].figure.axes[0].get_title().lower()


def test_williams_model_variant_and_external_selection_share_matching_sidecar(tmp_path, qapp):
    predict = tmp_path / "PREDICT"
    external = predict / "csv_test"
    external.mkdir(parents=True)
    paths = [predict / "GB_No_PFI.csv", predict / "GB_PFI.csv",
             external / "new_GB_No_PFI.csv"]
    for path in paths:
        path.write_text("code_name,target,target_pred,Set\na,1,2,CV\n")
    for stem, leverage in (("GB_No_PFI", 0.4), ("GB_PFI", 0.9)):
        pd.DataFrame({
            "point_id": ["a"], "name_column": ["code_name"],
            "group": ["Typical 80%"], "leverage": [leverage],
            "standardized_residual": [1.5], "h_star": [0.7],
            "observed": [1.0], "predicted": [2.0],
        }).to_csv(predict / f"Results_boundary_williams_{stem}.csv", index=False)
    selected = tmp_path / "input.csv"
    selected.write_text("code_name,SMILES\na,CCO\n")

    panel = InteractivePredictions()
    panel.refresh(tmp_path, str(selected), paths=paths)
    assert panel.williams_data["leverage"].tolist() == [0.4]
    panel._show_williams_point(0)
    assert "Leverage: 0.4" in panel.details.text()
    assert not panel.structure.pixmap().isNull()
    panel.selector.setCurrentIndex(1)
    assert panel.williams_data["leverage"].tolist() == [0.9]
    panel.selector.setCurrentIndex(2)
    assert panel.williams_data["leverage"].tolist() == [0.4]
