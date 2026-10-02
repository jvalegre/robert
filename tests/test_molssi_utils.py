import unittest
from unittest.mock import patch

import pandas as pd

from gui_easyrob.utils.molssi_utils import _prepare_export


class TestMolssiUtils(unittest.TestCase):
    def setUp(self):
        self.df_work = pd.DataFrame(
            {
                "SMILES": ["CCO"],
                "_smiles_original": ["CCO"],
                "_smiles_canonical": ["CCO"],
            }
        )
        self.df_api = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "descriptor_a": [1.23],
            }
        )

    @patch("gui_easyrob.utils.molssi_utils._molssi_test_dataset_available", return_value=True)
    def test_prepare_export_disables_test_download_for_kraken(self, mocked_available):
        result = _prepare_export(
            self.df_work,
            self.df_api,
            "SMILES",
            "kraken",
            "ML",
            ["SMILES"],
        )

        self.assertTrue(result["available"])
        self.assertFalse(result["export_available"])
        mocked_available.assert_not_called()

    @patch("gui_easyrob.utils.molssi_utils._molssi_test_dataset_available", return_value=True)
    def test_prepare_export_keeps_test_download_for_supported_libraries(self, mocked_available):
        result = _prepare_export(
            self.df_work,
            self.df_api,
            "SMILES",
            "acids",
            "DFT",
            ["SMILES"],
        )

        self.assertTrue(result["available"])
        self.assertTrue(result["export_available"])
        mocked_available.assert_called_once_with("acids")


if __name__ == "__main__":
    unittest.main()
