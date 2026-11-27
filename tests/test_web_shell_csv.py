"""CSV validation at the web ChemDraw bridge boundary."""

import csv
import json
from types import SimpleNamespace

from gui_easyrob import web_shell


def test_chemdraw_csv_rejects_duplicate_names_before_opening_save_dialog(monkeypatch):
    rows = [
        {"SMILES": "CCO", "code_name": "same", "target": "1.0"},
        {"SMILES": "CCC", "code_name": "same", "target": "2.0"},
    ]
    bridge = SimpleNamespace(_chemdraw_rows=[None, None], window=SimpleNamespace())
    monkeypatch.setattr(
        web_shell.QFileDialog,
        "getSaveFileName",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Save dialog opened for invalid rows")),
    )

    result = json.loads(web_shell.WorkflowBridge.saveChemdrawCsv(
        bridge, json.dumps({"rows": rows, "columns": [], "targetColumn": "target"})
    ))

    assert result == {"error": "code_name values must be unique"}


def test_chemdraw_csv_saves_valid_rows_and_loads_them_into_window(tmp_path, monkeypatch):
    path = tmp_path / "molecules.csv"
    loaded = []
    bridge = SimpleNamespace(
        _chemdraw_rows=[None],
        window=SimpleNamespace(set_file_path=lambda value: loaded.append(value) or True),
    )
    monkeypatch.setattr(
        web_shell.QFileDialog, "getSaveFileName", lambda *_args: (str(path), "CSV Files (*.csv)")
    )
    payload = json.dumps({
        "rows": [{"SMILES": "CCO", "code_name": "ethanol", "target": "1.5", "charge": "0"}],
        "columns": ["charge"],
        "targetColumn": "measurement",
    })

    result = json.loads(web_shell.WorkflowBridge.saveChemdrawCsv(bridge, payload))

    assert result == {"saved": str(path)}
    assert loaded == [str(path)]
    with path.open(newline="", encoding="utf-8") as source:
        assert list(csv.reader(source)) == [
            ["SMILES", "code_name", "measurement", "charge"],
            ["CCO", "ethanol", "1.5", "0"],
        ]
