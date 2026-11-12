from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import robert.aqme as aqme_module


def test_aqme_output_uses_csv_name_when_input_path_is_absolute(tmp_path, monkeypatch):
    """Keep an absolute input path out of the generated descriptor CSV name."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "external_test.csv"
    pd.DataFrame({
        "code_name": ["mol_1", "mol_2"],
        "SMILES": ["CCO", "CCC"],
        "target": [1.0, 2.0],
    }).to_csv(source, index=False)

    monkeypatch.setattr(
        aqme_module,
        "load_database",
        lambda _runner, csv_target, _job_type, print_info=False: (
            pd.read_csv(csv_target), None, None
        ),
    )

    def fake_run_aqme(_runner, command, _keywords):
        input_csv = command[command.index("--input") + 1]
        result = pd.read_csv(input_csv)
        result["HOMO"] = [-1.0, -2.0]
        result.to_csv("AQME-ROBERT_interpret_AQME_indiv.csv", index=False)

    monkeypatch.setattr(aqme_module.aqme, "run_aqme", fake_run_aqme)
    runner = object.__new__(aqme_module.aqme)
    runner.args = SimpleNamespace(
        descp_lvl="interpret",
        ignore=[],
        nprocs=1,
        qdescp_keywords="",
        log=SimpleNamespace(write=lambda _message: None),
    )

    runner.run_csearch_qdescp(str(source), aqme_test=True)

    output = tmp_path / "AQME-ROBERT_interpret_external_test.csv"
    assert output.is_file()
    assert pd.read_csv(output)["HOMO"].tolist() == [-1.0, -2.0]
