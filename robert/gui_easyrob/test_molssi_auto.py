"""Regression checks for automatic MolSSI descriptor detection."""

from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest
import requests

from gui_easyrob.utils import molssi_utils
from scripts.create_molssi_smoke_csvs import KRAKEN_ML_SAMPLE, SAMPLES, TEST_WORKBOOKS


@pytest.fixture
def workspace_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        yield Path(directory)


class ApiResponse:
    def __init__(self, records, status_code=200):
        self.records = records
        self.status_code = status_code

    def json(self):
        return self.records


class DownloadResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ValueError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]


def workbook_bytes():
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", "<workbook/>")
    return buffer.getvalue()


def test_resolution_preserves_names_smiles_and_repeated_input_rows(monkeypatch):
    def fake_get(url, **_kwargs):
        if "/api/acids/" in url:
            return ApiResponse([{"molecule_id": "acid-1", "smiles": "CC(=O)O", "HOMO": -0.2}])
        return ApiResponse([])

    monkeypatch.setattr(molssi_utils.requests, "get", fake_get)
    monkeypatch.setattr(molssi_utils, "_molssi_test_dataset_available", lambda _slug, _type: False)
    source = pd.DataFrame(
        {"Name": ["first", "second"], "SMILES": ["CC(=O)O", "CC(=O)O"], "target": [1, 2]}
    )

    result = molssi_utils.resolve_molssi_descriptors(source)

    assert result["available"] is True
    assert result["export_df"]["Name"].tolist() == ["first", "second"]
    assert result["export_df"]["SMILES"].tolist() == ["CC(=O)O", "CC(=O)O"]
    assert result["export_df"]["target"].tolist() == [1, 2]
    assert result["export_df"]["HOMO"].tolist() == [-0.2, -0.2]
    assert "molecule_id" not in result["export_df"]
    assert result["descriptor_columns"] == ["HOMO"]


def test_invalid_smiles_cannot_be_reported_as_fully_covered(monkeypatch):
    monkeypatch.setattr(
        molssi_utils.requests, "get", lambda *_args, **_kwargs: ApiResponse(
            [{"molecule_id": "acid-1", "smiles": "CC(=O)O", "HOMO": -0.2}]
        )
    )
    source = pd.DataFrame({"SMILES": ["CC(=O)O", "not a smiles"]})

    result = molssi_utils.resolve_molssi_descriptors(source)

    assert result["available"] is False
    assert result["export_df"] is None


def test_new_aryl_bromide_connection_has_api_name_and_smiles(monkeypatch):
    def fake_get(url, **_kwargs):
        if "/api/aryl_bromide/" in url:
            return ApiResponse(
                [{"molecule_id": "arbr1547", "smiles": "Brc1ccccc1", "HOMO": -0.3}]
            )
        return ApiResponse([])

    monkeypatch.setattr(molssi_utils.requests, "get", fake_get)
    monkeypatch.setattr(molssi_utils, "_molssi_test_dataset_available", lambda _slug, _type: False)

    result = molssi_utils.resolve_molssi_descriptors(pd.DataFrame({"SMILES": ["Brc1ccccc1"]}))

    assert result["library"] == "aryl_bromide"
    assert result["export_df"]["code_name"].tolist() == ["arbr1547"]
    assert result["export_df"]["SMILES"].tolist() == ["Brc1ccccc1"]
    assert result["export_df"]["HOMO"].tolist() == [-0.3]


def test_kraken_ml_descriptors_do_not_offer_dft_test_workbook(monkeypatch):
    def fake_get(url, **_kwargs):
        if "/api/kraken/" in url and "data_type=ML" in url:
            return ApiResponse(
                [{"molecule_id": 17, "smiles": "c1ccc(P(c2ccccc2)c2ccccc2)cc1", "dipolemoment_max": 1.6}]
            )
        return ApiResponse([])

    monkeypatch.setattr(molssi_utils.requests, "get", fake_get)
    monkeypatch.setattr(molssi_utils, "_molssi_test_dataset_available", lambda _slug, _type: True)
    source = pd.DataFrame({"SMILES": ["P(c1ccccc1)(c1ccccc1)c1ccccc1"]})

    result = molssi_utils.resolve_molssi_descriptors(source)

    assert result["available"] is True
    assert result["library"] == "kraken"
    assert result["data_type"] == "ML"
    assert result["export_available"] is False
    assert result["export_df"]["code_name"].tolist() == [17]


def test_test_workbook_probe_rejects_html_and_accepts_xlsx(monkeypatch):
    monkeypatch.setattr(
        molssi_utils.requests, "get", lambda *_args, **_kwargs: DownloadResponse(b"<html>not found</html>")
    )
    assert molssi_utils._molssi_test_dataset_available("primary-amines", "DFT") is False

    monkeypatch.setattr(
        molssi_utils.requests, "get", lambda *_args, **_kwargs: DownloadResponse(workbook_bytes())
    )
    assert molssi_utils._molssi_test_dataset_available("primary-amines", "DFT") is True


def test_download_worker_rejects_html_without_replacing_existing_workbook(monkeypatch, tmp_path):
    target = tmp_path / "acids_library.xlsx"
    original_workbook = workbook_bytes()
    target.write_bytes(original_workbook)
    monkeypatch.setattr(
        molssi_utils.requests, "get", lambda *_args, **_kwargs: DownloadResponse(b"<html>site home</html>")
    )
    errors = []
    worker = molssi_utils.MolSSIDownloadWorker(
        "https://descriptor-libraries.molssi.org/acids/content/acids_library.xlsx", target
    )
    worker.error.connect(errors.append)

    worker.run()

    assert errors
    assert "XLSX" in errors[0]
    assert target.read_bytes() == original_workbook


def test_download_worker_streams_valid_workbook(monkeypatch):
    monkeypatch.setattr(
        molssi_utils.requests, "get", lambda *_args, **_kwargs: DownloadResponse(workbook_bytes())
    )
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        target = Path(directory) / "acids_library.xlsx"
        completed = []
        errors = []
        worker = molssi_utils.MolSSIDownloadWorker(
            "https://descriptor-libraries.molssi.org/acids/content/acids_library.xlsx", target
        )
        worker.finished.connect(completed.append)
        worker.error.connect(errors.append)

        worker.run()

        assert completed == [str(target)], errors
        assert target.read_bytes() == workbook_bytes()


def test_conversion_rejects_unmatched_smiles_in_identifier_sheet(tmp_path):
    workbook = tmp_path / "library.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"Compound Name": ["A", "B"], "HOMO": [-0.2, -0.3]}).to_excel(
            writer, sheet_name="Descriptors", index=False
        )
        pd.DataFrame({"Compound Name": ["A"], "SMILES": ["CCO"]}).to_excel(
            writer, sheet_name="Identifiers", index=False
        )
    worker = molssi_utils.ExcelToCSVWorker(str(workbook))
    errors = []
    worker.error.connect(errors.append)

    worker.run()

    assert errors
    assert not workbook.with_suffix(".csv").exists()


def test_conversion_keeps_corresponding_name_smiles_and_descriptors(workspace_tmp_path):
    workbook = workspace_tmp_path / "library.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"Compound Name": ["A", "B"], "HOMO": [-0.2, -0.3]}).to_excel(
            writer, sheet_name="Descriptors", index=False
        )
        pd.DataFrame({"Compound Name": ["B", "A"], "SMILES": ["CCC", "CCO"]}).to_excel(
            writer, sheet_name="Identifiers", index=False
        )
    completed = []
    worker = molssi_utils.ExcelToCSVWorker(str(workbook))
    worker.finished.connect(completed.append)

    worker.run()

    assert completed == [str(workbook.with_suffix(".csv"))]
    exported = pd.read_csv(completed[0])
    assert exported["code_name"].tolist() == ["A", "B"]
    assert exported["SMILES"].tolist() == ["CCO", "CCC"]
    assert exported["HOMO"].tolist() == [-0.2, -0.3]


def test_connection_failure_stops_further_library_requests(monkeypatch):
    calls = []

    def disconnected(url, **_kwargs):
        calls.append(url)
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(molssi_utils.requests, "get", disconnected)

    result = molssi_utils.resolve_molssi_descriptors(pd.DataFrame({"SMILES": ["CCO"]}))

    assert result["available"] is False
    assert "unavailable" in result["reason"].lower()
    assert len(calls) == 1


def test_response_without_descriptors_is_not_offered(monkeypatch):
    def fake_get(url, **_kwargs):
        if "/api/acids/" in url:
            return ApiResponse([{"molecule_id": "Ac1", "smiles": "CC(=O)O"}])
        return ApiResponse([])

    monkeypatch.setattr(molssi_utils.requests, "get", fake_get)
    monkeypatch.setattr(molssi_utils, "_molssi_test_dataset_available", lambda _slug, _type: False)

    result = molssi_utils.resolve_molssi_descriptors(pd.DataFrame({"SMILES": ["CC(=O)O"]}))

    assert result["available"] is False


def test_kraken_dft_can_offer_verified_workbook(monkeypatch):
    def fake_get(url, **_kwargs):
        if "/api/kraken/" in url and "data_type=DFT" in url:
            return ApiResponse(
                [{"molecule_id": 17, "smiles": "c1ccc(P(c2ccccc2)c2ccccc2)cc1", "dipolemoment_max": 1.4}]
            )
        return ApiResponse([])

    monkeypatch.setattr(molssi_utils.requests, "get", fake_get)
    monkeypatch.setattr(molssi_utils, "_molssi_test_dataset_available", lambda _slug, _type: True)

    result = molssi_utils.resolve_molssi_descriptors(
        pd.DataFrame({"SMILES": ["P(c1ccccc1)(c1ccccc1)c1ccccc1"]})
    )

    assert result["available"] is True
    assert result["data_type"] == "DFT"
    assert result["export_available"] is True


def test_conversion_accepts_kraken_dft_sheet_and_identifier(workspace_tmp_path):
    workbook = workspace_tmp_path / "kraken_library.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(
            {"Kraken_ID": [17], "SMILES": ["c1ccc(P(c2ccccc2)c2ccccc2)cc1"], "dipolemoment_max": [1.4]}
        ).to_excel(writer, sheet_name="Kraken_descriptors_2025_LATEST-", index=False)
    completed = []
    worker = molssi_utils.ExcelToCSVWorker(str(workbook))
    worker.finished.connect(completed.append)

    worker.run()

    assert completed == [str(workbook.with_suffix(".csv"))]
    exported = pd.read_csv(completed[0])
    assert exported["code_name"].tolist() == [17]
    assert exported["SMILES"].tolist() == ["c1ccc(P(c2ccccc2)c2ccccc2)cc1"]
    assert exported["dipolemoment_max"].tolist() == [1.4]


def test_conversion_maps_lowercase_id_to_code_name(workspace_tmp_path):
    workbook = workspace_tmp_path / "aryl_bromide_library.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"id": ["arbr6851"], "smiles": ["Brc1ccccc1"], "HOMO_Boltz": [-0.27]}).to_excel(
            writer, sheet_name="Sheet1", index=False
        )
    completed = []
    worker = molssi_utils.ExcelToCSVWorker(str(workbook))
    worker.finished.connect(completed.append)

    worker.run()

    exported = pd.read_csv(completed[0])
    assert exported["code_name"].tolist() == ["arbr6851"]
    assert exported["SMILES"].tolist() == ["Brc1ccccc1"]


@pytest.mark.parametrize(
    ("library", "smiles"),
    [
        ("acids", "CC(=O)O"),
        ("anilines", "Nc1ccccc1"),
        ("sulfides", "CCSC"),
        ("aryl_bromide", "Brc1ccccc1"),
        ("nn-ligands-hydride", "c1ccc(-c2ccccn2)nc1"),
        ("nn-ligands-fluoride", "c1ccc(-c2ccccn2)nc1"),
        ("nn-ligands-ligand-only", "c1ccc(-c2ccccn2)nc1"),
    ],
)
def test_descriptors_do_not_offer_incompatible_workbook(monkeypatch, library, smiles):
    def fake_get(url, **_kwargs):
        if f"/api/{library}/" in url:
            return ApiResponse(
                [{"molecule_id": "sample", "smiles": smiles, "HOMO": -0.2}]
            )
        return ApiResponse([])

    monkeypatch.setattr(molssi_utils.requests, "get", fake_get)
    monkeypatch.setattr(molssi_utils, "_molssi_test_dataset_available", lambda _slug, _type: True)

    result = molssi_utils.resolve_molssi_descriptors(
        pd.DataFrame({"SMILES": [smiles]})
    )

    assert result["available"] is True
    assert result["export_available"] is False


def test_external_test_csv_requires_matching_descriptor_columns(tmp_path):
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"SMILES": ["CCO"], "code_name": ["sample"], "HOMO": [-0.2]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="LUMO"):
        molssi_utils.validate_molssi_test_csv(csv_path, ["HOMO", "LUMO"])

    molssi_utils.validate_molssi_test_csv(csv_path, ["HOMO"])


def test_shutdown_tracks_download_and_conversion_workers():
    from gui_easyrob.main.window import EasyROB

    class Worker:
        def __init__(self, running):
            self.running = running

        def isRunning(self):
            return self.running

    resolver = Worker(True)
    downloader = Worker(True)
    converter = Worker(True)
    inactive = Worker(False)
    window = SimpleNamespace(
        _molssi_workers={resolver, inactive},
        _molssi_download_worker=downloader,
        molssi_tab=SimpleNamespace(_excel_worker=converter),
    )

    assert set(EasyROB._running_molssi_workers(window)) == {resolver, downloader, converter}


def test_rejected_test_csv_does_not_claim_success_or_remove_workbook(tmp_path, monkeypatch):
    from gui_easyrob.tabs import molssi as molssi_tab_module

    workbook_path = tmp_path / "library.xlsx"
    workbook_path.write_bytes(b"workbook")
    csv_path = tmp_path / "library.csv"
    csv_path.write_text("SMILES,code_name\nCCO,sample\n", encoding="utf-8")
    shown_popups = []
    main_window = SimpleNamespace(csv_test_path=None)

    class FakeMessageBox:
        Information = 1
        Ok = 2

        def __init__(self, _parent):
            pass

        def setIcon(self, _icon):
            pass

        def setWindowTitle(self, _title):
            pass

        def setText(self, _text):
            pass

        def setStandardButtons(self, _buttons):
            pass

    monkeypatch.setattr(molssi_tab_module, "QMessageBox", FakeMessageBox)
    tab = SimpleNamespace(
        load_test_requested=SimpleNamespace(emit=lambda _path: None),
        window=lambda: main_window,
        _exec_tracked_message_box=lambda *_args, **_kwargs: shown_popups.append("success"),
    )

    loaded = molssi_tab_module.MolSSIDatabasesTab.load_test_molssi(
        tab, str(csv_path), source=str(workbook_path)
    )

    assert loaded is False
    assert shown_popups == []
    assert workbook_path.exists()


def test_library_hint_selects_overlapping_library_without_exporting_hint(monkeypatch):
    calls = []
    smiles = "C[C@H]1COC(C2=N[C@@H](C)CO2)=N1"

    def fake_get(url, **_kwargs):
        calls.append(url)
        if "/api/nn-ligands-fluoride/" in url:
            return ApiResponse([{"molecule_id": "Lig1", "smiles": smiles, "F_descriptor": 1.2}])
        return ApiResponse([])

    monkeypatch.setattr(molssi_utils.requests, "get", fake_get)
    monkeypatch.setattr(molssi_utils, "_molssi_test_dataset_available", lambda *_args: False)
    source = pd.DataFrame({"SMILES": [smiles], "molssi_library": ["nn-ligands-fluoride"]})

    result = molssi_utils.resolve_molssi_descriptors(source)

    assert result["library"] == "nn-ligands-fluoride"
    assert result["export_df"]["code_name"].tolist() == ["Lig1"]
    assert "molssi_library" not in result["export_df"].columns
    assert len(calls) == 1


def test_unknown_library_hint_is_rejected_without_network_requests(monkeypatch):
    def unexpected_get(*_args, **_kwargs):
        raise AssertionError("The API must not be queried for an unknown library")

    monkeypatch.setattr(molssi_utils.requests, "get", unexpected_get)
    source = pd.DataFrame({"SMILES": ["CCO"], "molssi_library": ["unknown-library"]})

    result = molssi_utils.resolve_molssi_descriptors(source)

    assert result["available"] is False
    assert "unknown-library" in result["reason"]


def test_quinone_id_lookup_exports_corresponding_descriptors_and_test_offer(monkeypatch):
    smiles = "O=C1C(S(=O)(O)=O)=CC(C=C1)=O"
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        if "/molecules/search/" in url:
            return ApiResponse(
                [
                    {"molecule_id": "other", "smiles": "O=C1C=CC(=O)C=C1"},
                    {"molecule_id": "q10000", "smiles": smiles},
                ]
            )
        if "molecule_ids=q10000" in url:
            return ApiResponse([{"molecule_id": "q10000", "smiles": smiles, "HOMO_boltz": -0.3}])
        return ApiResponse([])

    monkeypatch.setattr(molssi_utils.requests, "get", fake_get)
    monkeypatch.setattr(molssi_utils, "_molssi_test_dataset_available", lambda *_args: True)
    source = pd.DataFrame({"SMILES": [smiles], "molssi_library": ["quinones"]})

    result = molssi_utils.resolve_molssi_descriptors(source)

    assert result["available"] is True
    assert result["library"] == "quinones"
    assert result["export_available"] is True
    assert result["export_df"]["code_name"].tolist() == ["q10000"]
    assert result["export_df"]["HOMO_boltz"].tolist() == [-0.3]
    assert len(calls) == 2


def test_id_lookup_rejects_ambiguous_exact_smiles(monkeypatch):
    smiles = "O=C1C(S(=O)(O)=O)=CC(C=C1)=O"

    def fake_get(url, **_kwargs):
        if "/molecules/search/" in url:
            return ApiResponse([
                {"molecule_id": "q1", "smiles": smiles},
                {"molecule_id": "q2", "smiles": smiles},
            ])
        raise AssertionError("Ambiguous IDs must not be exported")

    monkeypatch.setattr(molssi_utils.requests, "get", fake_get)
    source = pd.DataFrame({"SMILES": [smiles], "molssi_library": ["quinones"]})

    result = molssi_utils.resolve_molssi_descriptors(source)

    assert result["available"] is False


def test_excel_header_detection_reads_only_initial_rows(tmp_path, monkeypatch):
    workbook = tmp_path / "library.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"ID": ["sample"], "SMILES": ["CCO"]}).to_excel(
            writer, sheet_name="DFT", index=False
        )
    original_read_excel = molssi_utils.pd.read_excel
    reads = []

    def tracked_read_excel(*args, **kwargs):
        reads.append(kwargs.copy())
        return original_read_excel(*args, **kwargs)

    monkeypatch.setattr(molssi_utils.pd, "read_excel", tracked_read_excel)
    worker = molssi_utils.ExcelToCSVWorker(str(workbook))

    worker._read_descriptors("DFT")

    assert reads[0]["header"] is None
    assert reads[0]["nrows"] == 15


def test_failed_csv_conversion_preserves_existing_csv(tmp_path, monkeypatch):
    workbook = tmp_path / "library.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"ID": ["sample"], "SMILES": ["CCO"]}).to_excel(
            writer, sheet_name="DFT", index=False
        )
    existing_csv = workbook.with_suffix(".csv")
    existing_csv.write_text("previous export\n", encoding="utf-8")

    def fail_after_partial_write(_self, handle, **_kwargs):
        handle.write("partial export")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(molssi_utils.pd.DataFrame, "to_csv", fail_after_partial_write)
    worker = molssi_utils.ExcelToCSVWorker(str(workbook))
    errors = []
    worker.error.connect(errors.append)

    worker.run()

    assert errors and "disk failure" in errors[0]
    assert existing_csv.read_text(encoding="utf-8") == "previous export\n"
    assert not list(tmp_path.glob(".molssi-*.csv"))


def test_smoke_samples_cover_every_supported_connection():
    assert {sample[0] for sample in SAMPLES} == set(molssi_utils.ALL_MOLSSI_LIBRARIES)
    assert TEST_WORKBOOKS == molssi_utils.COMPATIBLE_TEST_WORKBOOK_LIBRARIES
    for _library, first_id, first_smiles, second_id, second_smiles in SAMPLES:
        assert first_id != second_id
        assert molssi_utils._canonical_smiles(first_smiles)
        assert molssi_utils._canonical_smiles(second_smiles)
        assert molssi_utils._canonical_smiles(first_smiles) != molssi_utils._canonical_smiles(second_smiles)
    assert KRAKEN_ML_SAMPLE[0] == "kraken"
    assert molssi_utils._canonical_smiles(KRAKEN_ML_SAMPLE[2])
    assert molssi_utils._canonical_smiles(KRAKEN_ML_SAMPLE[4])
