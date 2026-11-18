"""Regression checks for CheckML's default column selection."""

from PySide6.QtWidgets import QApplication

from gui_easyrob.tabs.checkml import CheckMLTab


def test_smiles_is_ignored_when_it_is_the_first_csv_column(tmp_path):
    app = QApplication.instance() or QApplication([])
    csv_path = tmp_path / "molecules.csv"
    csv_path.write_text(
        "SMILES,code_name,target,MolWt\nCCO,compound_1,1.2,46.1\nCCC,compound_2,2.3,44.1\n",
        encoding="utf-8",
    )
    tab = CheckMLTab()
    try:
        tab.set_csv_path(str(csv_path))
        assert tab.y_dropdown.currentText() == "target"
        assert "SMILES" in tab.get_ignored_columns()
    finally:
        tab.close()


def test_smiles_is_ignored_even_when_csv_parses_it_as_numeric(tmp_path):
    app = QApplication.instance() or QApplication([])
    csv_path = tmp_path / "numeric_smiles.csv"
    csv_path.write_text(
        "smiles,code_name,activity\n123,compound_1,1.2\n456,compound_2,2.3\n",
        encoding="utf-8",
    )
    tab = CheckMLTab()
    try:
        tab.set_csv_path(str(csv_path))
        assert tab.y_dropdown.currentText() == "activity"
        assert "smiles" in tab.get_ignored_columns()
    finally:
        tab.close()
