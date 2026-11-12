"""Focused regression checks for prediction data shown in the web workspace."""

from types import SimpleNamespace
import os

import pandas as pd

from gui_easyrob import web_results


def test_prediction_search_uses_numeric_code_name_when_run_metadata_is_missing(tmp_path, monkeypatch):
    csv_path = tmp_path / "predictions.csv"
    pd.DataFrame({"code_name": [6.0, 7.0, 10.0], "target_pred": [1.2, 2.3, 3.4]}).to_csv(
        csv_path, index=False,
    )
    tab = SimpleNamespace(
        csv_paths={"No_PFI": csv_path},
        _report_paths={"No_PFI": tmp_path / "report.pdf"},
        _base_path=tmp_path,
        _filter_prediction_dataframe=lambda frame: frame,
        _extract_names_column_from_predict=lambda: None,
    )
    monkeypatch.setattr(web_results, "evaluate_predictions_for_model", lambda *args, **kwargs: {})
    service = web_results.WebResultsService(SimpleNamespace(predictions_tab=tab))

    result = service.prediction_table("No_PFI", "10", "", False, 0)

    assert result["nameColumn"] == "code_name"
    assert result["total"] == 1
    assert result["rows"][0][0] == "10.0"


def test_prediction_assessment_refreshes_when_report_changes(tmp_path, monkeypatch):
    csv_path = tmp_path / "predictions.csv"
    report_path = tmp_path / "report.pdf"
    pd.DataFrame({"code_name": [1], "target_pred": [2.0]}).to_csv(csv_path, index=False)
    report_path.write_text("first", encoding="utf-8")
    tab = SimpleNamespace(
        csv_paths={"No_PFI": csv_path},
        _report_paths={"No_PFI": report_path},
        _base_path=tmp_path,
        _filter_prediction_dataframe=lambda frame: frame,
        _extract_names_column_from_predict=lambda: "code_name",
    )
    monkeypatch.setattr(
        web_results, "evaluate_predictions_for_model",
        lambda *args, **kwargs: {"scenario": {"messages": [report_path.read_text(encoding="utf-8")]}}
    )
    service = web_results.WebResultsService(SimpleNamespace(predictions_tab=tab))

    assert service.prediction_table("No_PFI", "", "", False, 0)["scenario"]["messages"] == ["first"]
    report_path.write_text("second", encoding="utf-8")
    modified = report_path.stat().st_mtime_ns + 2_000_000_000
    os.utime(report_path, ns=(modified, modified))
    assert service.prediction_table("No_PFI", "", "", False, 0)["scenario"]["messages"] == ["second"]
