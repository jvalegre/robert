"""Regression checks for the AQME integration test workspace."""

from pathlib import Path

import pytest

from test_5aqme_n_full import aqme_test_workspace


def test_aqme_workspace_removes_outputs_after_success():
    original_cwd = Path.cwd()
    with aqme_test_workspace() as workspace:
        assert workspace.parent == Path(__file__).resolve().parent
        (workspace / "AQME-ROBERT_interpret_solubility.csv").touch()

    assert Path.cwd() == original_cwd
    assert not workspace.exists()


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_aqme_workspace_removes_outputs_after_failure(error):
    original_cwd = Path.cwd()
    with pytest.raises(error):
        with aqme_test_workspace() as workspace:
            assert workspace.parent == Path(__file__).resolve().parent
            assert Path.cwd() == workspace
            (workspace / "AQME-ROBERT_interpret_solubility.csv").touch()
            (workspace / "AQME-ROBERT_interpret_solubility_solvent.csv").touch()
            (workspace / "solubility_solvent.csv").touch()
            raise error()

    assert Path.cwd() == original_cwd
    assert not workspace.exists()
