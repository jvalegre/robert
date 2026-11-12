from unittest.mock import mock_open, patch

from gui_easyrob.utils.utils_gui import CSVPreflightWorker


def _validate_path(path: str) -> dict:
    worker = CSVPreflightWorker([("main CSV file", path)])
    results: list[dict] = []
    worker.validation_finished.connect(results.append)
    with patch.object(worker, "_resolve_path", return_value=path), patch("builtins.open", mock_open()):
        worker.run()
    assert len(results) == 1
    return results[0]


def test_csv_preflight_allows_paths_up_to_200_characters_and_rejects_longer_ones():
    allowed_result = _validate_path("a" * 200)
    rejected_result = _validate_path("b" * 201)

    assert allowed_result == {"ok": True}
    assert rejected_result["ok"] is False
    assert "longer than 200 characters" in rejected_result["message"]
