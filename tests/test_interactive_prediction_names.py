"""Regression checks for configured prediction identifiers."""

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from robert.gui_easyrob.tabs.interactive_predictions import InteractivePredictions


@pytest.mark.parametrize("name_source", ["metadata", "log"])
def test_interactive_outlier_uses_configured_identifier_and_molecule(
    tmp_path: Path, name_source: str,
):
    application = QApplication.instance() or QApplication([])
    prediction_dir = tmp_path / "PREDICT"
    prediction_dir.mkdir()
    prediction_path = prediction_dir / "MVL_No_PFI.csv"
    pd.DataFrame({
        "Compound": ["BTNC1_003", "BTNC1_007"],
        "target": [84.1, 12.0],
        "target_pred": [29.98, 13.0],
        "Set": ["CV", "CV"],
    }).to_csv(prediction_path, index=False)

    if name_source == "metadata":
        metadata_dir = tmp_path / "GENERATE" / "Best_model" / "No_PFI"
        metadata_dir.mkdir(parents=True)
        pd.DataFrame({
            "seed": [0], "repeat_kfolds": [10], "kfold": [5], "names": ["Compound"],
        }).to_csv(metadata_dir / "MVL.csv", index=False)
    else:
        (prediction_dir / "PREDICT_data.dat").write_text(
            'Command line used in ROBERT: --names "Compound"', encoding="utf-8",
        )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({
        "Compound": ["BTNC1_007", "BTNC1_003"],
        "SMILES": ["CCO", "CC(=O)O"],
    }).to_csv(input_path, index=False)

    panel = InteractivePredictions()
    panel.refresh(tmp_path, str(input_path), paths=[prediction_path])
    panel._show_point(0)

    assert panel.details.text().splitlines()[0] == "BTNC1_003"
    assert panel.data["smiles"][0] == "CC(=O)O"
    assert not panel.structure.pixmap().isNull()
    panel._show_point(1)
    assert panel.details.text().splitlines()[0] == "BTNC1_007"
    panel.close()
    application.processEvents()
