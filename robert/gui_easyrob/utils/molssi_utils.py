"""
MolSSI utilities for easyROB.

This module provides data-processing and background utilities for interacting
with MolSSI descriptor libraries and preparing datasets for model workflows.

Responsibilities:
- Download MolSSI datasets from remote APIs
- Resolve descriptor coverage for SMILES datasets
- Normalize and merge descriptor data into exportable formats
- Convert Excel-based MolSSI datasets into clean CSV files
- Run long operations in background threads to keep the GUI responsive

Architecture:
- Uses QThread-based workers for I/O-bound tasks (downloads, conversions)
- Uses RDKit for SMILES canonicalization and validation
- Communicates results back to the GUI via Qt signals

Notes:
- Designed to isolate data-processing logic from GUI components
- Handles partial failures gracefully when querying external services

"""

# ------------------------------------------------------------
# Standard library
# ------------------------------------------------------------
from pathlib import Path
import os
import re
import tempfile
import urllib.parse
import zipfile

# ------------------------------------------------------------
# Third-party libraries
# ------------------------------------------------------------
import pandas as pd
import requests

from rdkit import Chem

from PySide6.QtCore import QThread, Signal


MOLSSI_BASE_URL = "https://descriptor-libraries.molssi.org"
SMILES_EXPORT_LIBRARIES = (
    "acids",
    "primary-amines",
    "secondary-amines",
    "amines",
    "anilines",
    "sulfonimidamides",
    "cyanoarenes",
    "unactivated-primary-alkyl-bromides",
    "sulfonyl-fluorides",
    "sulfonates",
    "sulfides",
    "aryl_bromide",
    "nn-ligands-hydride",
    "nn-ligands-fluoride",
    "nn-ligands-ligand-only",
    "kraken",
)
ID_EXPORT_LIBRARIES = ("quinones", "cyclic-olefins")
ALL_MOLSSI_LIBRARIES = (*SMILES_EXPORT_LIBRARIES, *ID_EXPORT_LIBRARIES)
COMPATIBLE_TEST_WORKBOOK_LIBRARIES = frozenset({
    "primary-amines",
    "secondary-amines",
    "amines",
    "sulfonimidamides",
    "cyanoarenes",
    "sulfonyl-fluorides",
    "unactivated-primary-alkyl-bromides",
    "sulfonates",
    "kraken",
    "quinones",
})

class MolSSIWorker(QThread):
    """Background worker responsible for resolving MolSSI descriptors."""

    finished = Signal(dict)

    def __init__(self, df, file_path, should_abort, debug=False):
        super().__init__()
        self.df = df
        self.file_path = file_path
        self.should_abort = should_abort
        self.debug = debug

    def run(self):
        """Run the MolSSI descriptor resolution process in a background thread."""
        if self.should_abort():
            return

        result = resolve_molssi_descriptors(self.df)
        if self.should_abort():
            return

        self.finished.emit(result)


class MolSSIDownloadWorker(QThread):
    """Background worker responsible for downloading MolSSI datasets."""

    finished = Signal(str)
    error = Signal(str)

    def __init__(self, url, target_path, parent=None):
        super().__init__(parent)
        self.url = url
        self.target_path = Path(target_path)

    def run(self):
        """Download and validate an XLSX before replacing the target file."""
        temporary_path = None
        try:
            self.target_path.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(self.url, stream=True, timeout=(5, 60)) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=".molssi-", suffix=".xlsx",
                    dir=self.target_path.parent, delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            temporary_file.write(chunk)
            with zipfile.ZipFile(temporary_path) as workbook:
                members = set(workbook.namelist())
                if not {"[Content_Types].xml", "xl/workbook.xml"}.issubset(members):
                    raise ValueError("The MolSSI download is not an XLSX workbook")
            os.replace(temporary_path, self.target_path)
            temporary_path = None
            self.finished.emit(str(self.target_path))
        except (requests.RequestException, OSError, ValueError, zipfile.BadZipFile) as exc:
            self.error.emit(f"Unable to download a valid MolSSI XLSX dataset: {exc}")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def resolve_molssi_descriptors(df):
    """Resolve a full-coverage MolSSI descriptor dataset for a SMILES table."""
    smiles_col = next((col for col in df.columns if str(col).casefold() == "smiles"), None)

    if smiles_col is None:
        return {
            "available": False,
            "export_df": None,
            "library": None,
            "data_type": None,
            "reason": "No SMILES column found",
        }

    library_col = next(
        (col for col in df.columns if str(col).casefold() == "molssi_library"), None
    )
    libraries = ALL_MOLSSI_LIBRARIES
    if library_col is not None:
        requested = df[library_col].dropna().astype(str).str.strip().str.casefold().unique()
        if df[library_col].isna().any() or len(requested) != 1 or requested[0] not in libraries:
            return {
                "available": False,
                "export_df": None,
                "library": None,
                "data_type": None,
                "reason": f"Invalid MolSSI library selection: {', '.join(map(str, requested))}",
            }
        libraries = (requested[0],)

    df_work = df.drop(columns=[library_col]).copy() if library_col is not None else df.copy()
    original_input_columns = list(df_work.columns)
    df_work["_smiles_canonical"] = df_work[smiles_col].apply(_canonical_smiles)
    if df_work.empty or df_work["_smiles_canonical"].isna().any():
        return {
            "available": False,
            "export_df": None,
            "library": None,
            "data_type": None,
            "reason": "At least one SMILES is missing or invalid",
        }
    smiles_list = df_work["_smiles_canonical"].drop_duplicates().tolist()

    for library in libraries:
        data_types = ("DFT", "ML") if library == "kraken" else ("DFT",)
        for data_type in data_types:
            try:
                if library in ID_EXPORT_LIBRARIES:
                    df_api = _query_molssi_by_id(smiles_list, library, data_type)
                else:
                    df_api = _query_molssi_batched(smiles_list, library, data_type)
            except (requests.ConnectionError, requests.Timeout):
                return {
                    "available": False,
                    "export_df": None,
                    "library": None,
                    "data_type": None,
                    "reason": "MolSSI API is unavailable",
                }
            if df_api is None or "smiles" not in df_api.columns:
                continue
            canonical = df_api["smiles"].apply(_canonical_smiles)
            if not set(smiles_list).issubset(set(canonical.dropna())):
                continue
            result = _prepare_export(
                df_work, df_api, smiles_col, library, data_type, original_input_columns
            )
            if result["available"]:
                return result

    return {
        "available": False,
        "export_df": None,
        "library": None,
        "data_type": None,
        "reason": "MolSSI does not provide full coverage",
    }


def _canonical_smiles(value):
    """Return an isomeric canonical SMILES, or None for invalid input."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        molecule = Chem.MolFromSmiles(value.strip())
        return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None
    except (TypeError, ValueError):
        return None


def _query_molssi_batched(smiles, library, data_type, query_key="molecule_smiles"):
    """Fetch descriptor records without exceeding practical GET URL lengths."""
    if query_key not in {"molecule_smiles", "molecule_ids"}:
        raise ValueError(f"Unsupported MolSSI batch query key: {query_key}")
    frames = []
    encoded_smiles = [urllib.parse.quote(value, safe="") for value in smiles]
    batches = []
    batch = []
    for value in encoded_smiles:
        if batch and (len(batch) >= 100 or sum(map(len, batch)) + len(batch) + len(value) > 1600):
            batches.append(batch)
            batch = []
        batch.append(value)
    if batch:
        batches.append(batch)

    for batch in batches:
        url = (
            f"{MOLSSI_BASE_URL}/api/{library}/molecules/data/export/batch"
            f"?{query_key}={','.join(batch)}&data_type={data_type}&return_type=json"
        )
        try:
            response = requests.get(url, timeout=(5, 30))
            if response.status_code != 200:
                return None
            records = response.json()
            if not isinstance(records, list) or not records:
                return None
            frames.append(pd.DataFrame(records))
        except (requests.ConnectionError, requests.Timeout):
            raise
        except (requests.RequestException, ValueError):
            return None
    return pd.concat(frames, ignore_index=True)


def _query_molssi_by_id(smiles, library, data_type):
    """Resolve exact structures through search before ID-based batch export."""
    molecule_ids = []
    for canonical_smiles in smiles:
        exact_ids = set()
        for skip in (0, 100, 200):
            url = (
                f"{MOLSSI_BASE_URL}/api/{library}/molecules/search/"
                f"?substructure={urllib.parse.quote(canonical_smiles, safe='')}"
                f"&skip={skip}&limit=100"
            )
            try:
                response = requests.get(url, timeout=(5, 30))
                if response.status_code != 200:
                    return None
                candidates = response.json()
                if not isinstance(candidates, list):
                    return None
            except (requests.ConnectionError, requests.Timeout):
                raise
            except (requests.RequestException, ValueError):
                return None
            exact_ids.update(
                str(candidate["molecule_id"])
                for candidate in candidates
                if isinstance(candidate, dict)
                and candidate.get("molecule_id") is not None
                and _canonical_smiles(candidate.get("smiles")) == canonical_smiles
            )
            if len(exact_ids) > 1:
                return None
            if len(candidates) < 100:
                break
            if skip == 200:
                return None
        if len(exact_ids) != 1:
            return None
        molecule_ids.extend(exact_ids)
    return _query_molssi_batched(molecule_ids, library, data_type, query_key="molecule_ids")


def _molssi_workbook_url(library_slug):
    """Build the published workbook URL for a known library."""
    return f"{MOLSSI_BASE_URL}/{library_slug}/content/{library_slug}_library.xlsx"


def _molssi_test_dataset_available(library_slug, data_type):
    """Check that a published test workbook begins with an XLSX ZIP header."""
    if library_slug not in COMPATIBLE_TEST_WORKBOOK_LIBRARIES or (
        library_slug == "kraken" and data_type == "ML"
    ):
        return False
    try:
        with requests.get(
            _molssi_workbook_url(library_slug),
            headers={"Range": "bytes=0-3"},
            stream=True,
            timeout=(5, 15),
        ) as response:
            if response.status_code not in (200, 206):
                return False
            return next(response.iter_content(chunk_size=4), b"").startswith(b"PK\x03\x04")
    except (requests.RequestException, OSError):
        return False


def validate_molssi_test_csv(csv_path, descriptor_columns):
    """Reject a test CSV that cannot supply the selected API descriptors."""
    available_columns = set(pd.read_csv(csv_path, nrows=0).columns)
    required_columns = {"SMILES", "code_name", *descriptor_columns}
    missing = sorted(required_columns - available_columns)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"The MolSSI test CSV is missing {len(missing)} required columns: {preview}"
        )


def write_csv_atomic(df, target_path):
    """Replace a CSV only after its complete contents have been written."""
    target_path = Path(target_path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", prefix=".molssi-",
            suffix=".csv", dir=target_path.parent, delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            df.to_csv(temporary_file, index=False)
        os.replace(temporary_path, target_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _prepare_export(df_work, df_api, smiles_col, library, data_type, original_input_columns):
    """Prepare the merged MolSSI export DataFrame for use in easyROB."""
    try:
        df_api = df_api.copy()
        descriptor_columns = [
            column for column in df_api.columns if column not in {"molecule_id", "smiles"}
        ]
        if not descriptor_columns or df_api[descriptor_columns].isna().all(axis=1).any():
            raise ValueError("MolSSI returned no descriptors for at least one molecule")
        descriptor_export_columns = [
            fix_greek_caps_columns(
                f"{column}_molssi" if column in original_input_columns else column
            )
            for column in descriptor_columns
        ]
        df_api["_smiles_canonical"] = df_api["smiles"].apply(_canonical_smiles)
        df_api = df_api.dropna(subset=["_smiles_canonical"])
        if df_api["_smiles_canonical"].duplicated().any():
            raise ValueError("MolSSI returned ambiguous records for the same SMILES")

        has_name = any(str(column).casefold() in {"name", "code_name"} for column in original_input_columns)
        if not has_name and "molecule_id" in df_api.columns:
            df_api = df_api.rename(columns={"molecule_id": "code_name"})
        else:
            df_api = df_api.drop(columns=["molecule_id"], errors="ignore")
        df_api = df_api.drop(columns=["smiles"])
        export_df = df_work.merge(df_api, on="_smiles_canonical", how="left", sort=False, suffixes=("", "_molssi"))
        export_df = export_df.drop(columns=["_smiles_canonical"])
        cols = [smiles_col] + [column for column in export_df.columns if column != smiles_col]
        export_df = export_df.loc[:, cols].copy()
        export_df.columns = [
            fix_greek_caps_columns(c) if c != smiles_col else c
            for c in export_df.columns
        ]
        if library == "kraken" and data_type == "ML":
            test_dataset_status = "kraken_ml"
        elif library not in COMPATIBLE_TEST_WORKBOOK_LIBRARIES:
            test_dataset_status = "incompatible"
        elif _molssi_test_dataset_available(library, data_type):
            test_dataset_status = "available"
        else:
            test_dataset_status = "missing"
        return {
            "available": True,
            "export_available": test_dataset_status == "available",
            "test_dataset_status": test_dataset_status,
            "export_df": export_df,
            "descriptor_columns": descriptor_export_columns,
            "library": library,
            "data_type": data_type,
            "reason": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "export_available": False,
            "export_df": None,
            "library": None,
            "data_type": None,
            "reason": str(exc),
        }


def fix_greek_caps_columns(col: str) -> str:
    """Normalize Greek characters and canonical spelling in MolSSI headers."""
    greek_map = {
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
        "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
        "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
        "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
        "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
        "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
        "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta",
        "Ε": "epsilon", "Ζ": "zeta", "Η": "eta", "Θ": "theta",
        "Ι": "iota", "Κ": "kappa", "Λ": "lambda", "Μ": "mu",
        "Ν": "nu", "Ξ": "xi", "Ο": "omicron", "Π": "pi",
        "Ρ": "rho", "Σ": "sigma", "Τ": "tau", "Υ": "upsilon",
        "Φ": "phi", "Χ": "chi", "Ψ": "psi", "Ω": "omega",
    }

    for greek_char, latin in greek_map.items():
        col = col.replace(greek_char, latin)

    col = re.sub(r"(?<![A-Za-z0-9])low[_]?e(?![A-Za-z0-9])", "low_e", col, flags=re.IGNORECASE)
    col = re.sub(r"(?<![A-Za-z0-9])boltz(?![A-Za-z0-9])", "boltz", col, flags=re.IGNORECASE)
    return col

class ExcelToCSVWorker(QThread):
    """Convert MolSSI spreadsheets into normalized CSV files for easyROB."""

    finished = Signal(str)
    error = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def _normalize_columns(self, df):
        """Normalize column names by stripping whitespace, replacing spaces with underscores, and fixing Greek characters."""
        df = df.copy()
        normalized = []

        for column in df.columns:
            col = str(column).replace("\xa0", "").strip().replace(" ", "_")
            normalized.append(fix_greek_caps_columns(col))

        df.columns = normalized
        return df

    def _force_string_key(self, df, key):
        """Ensure the join key column is treated as a string and cleaned of non-breaking spaces."""
        if key in df.columns:
            df[key] = df[key].astype(str).replace("\xa0", "", regex=True).str.strip()
        return df

    def _detect_descriptor_sheet(self, sheets):
        """Determine which sheet contains the descriptors."""
        for name in ("Descriptors", "all_properties", "DFT"):
            if name in sheets:
                return name
        if len(sheets) == 1:
            return sheets[0]
        return None

    def _read_descriptors(self, sheet_name):
        """Read the descriptors from the spreadsheet."""
        raw = pd.read_excel(self.path, sheet_name=sheet_name, header=None, nrows=15)

        for i in range(min(15, len(raw))):
            row = raw.iloc[i].astype(str).str.lower().tolist()
            if any(key in row for key in ("id", "numerical_id", "compound_name", "smiles")):
                return pd.read_excel(self.path, sheet_name=sheet_name, header=i)

        return pd.read_excel(self.path, sheet_name=sheet_name)

    def _find_smiles_column(self, columns):
        """Find the SMILES column in the spreadsheet."""
        for col in columns:
            if col.lower() in ("smiles", "canonical_smiles"):
                return col
        return None

    def run(self):
        """Convert the spreadsheet into a CSV file."""
        try:
            with pd.ExcelFile(self.path) as xls:
                sheets = xls.sheet_names

            descriptor_sheet = self._detect_descriptor_sheet(sheets)
            if not descriptor_sheet:
                raise ValueError("No descriptor sheet found. Expected one of: Descriptors, all_properties, DFT")

            df_desc = self._normalize_columns(self._read_descriptors(descriptor_sheet))
            needs_merge = descriptor_sheet == "Descriptors"

            if not needs_merge:
                smiles_col = self._find_smiles_column(df_desc.columns)
                if not smiles_col:
                    raise ValueError(f"No SMILES column found in {descriptor_sheet} sheet")

                df_final = df_desc.rename(columns={smiles_col: "SMILES"})
                for key in ("Compound_Name", "Kraken_ID", "ID", "id", "Numerical_ID"):
                    if key in df_final.columns and "code_name" not in df_final.columns:
                        df_final.rename(columns={key: "code_name"}, inplace=True)
                        break
            else:
                smiles_col_desc = self._find_smiles_column(df_desc.columns)
                if smiles_col_desc:
                    df_final = df_desc.rename(columns={smiles_col_desc: "SMILES"})
                    for key in ("Compound_Name", "Kraken_ID", "ID", "id", "Numerical_ID"):
                        if key in df_final.columns and "code_name" not in df_final.columns:
                            df_final.rename(columns={key: "code_name"}, inplace=True)
                            break
                else:
                    if "Identifiers" not in sheets:
                        raise ValueError("Identifiers sheet not found")

                    df_id = self._normalize_columns(pd.read_excel(self.path, sheet_name="Identifiers"))
                    smiles_col_id = self._find_smiles_column(df_id.columns)
                    if not smiles_col_id:
                        raise ValueError("No SMILES column found in Identifiers")

                    join_key = None
                    for key in ("Compound_Name", "ID", "Numerical_ID"):
                        if key in df_desc.columns and key in df_id.columns:
                            join_key = key
                            break

                    if not join_key:
                        raise ValueError(
                            "No valid join key found between Descriptors and Identifiers "
                            "(expected ID, Numerical_ID, or Compound_Name)"
                        )

                    df_desc = self._force_string_key(df_desc, join_key)
                    df_id = self._force_string_key(df_id, join_key)
                    df_final = df_desc.merge(
                        df_id[[join_key, smiles_col_id]],
                        on=join_key,
                        how="left",
                        validate="many_to_one",
                    )

                    if "code_name" not in df_final.columns:
                        df_final.rename(columns={join_key: "code_name"}, inplace=True)

                    df_final.rename(columns={smiles_col_id: "SMILES"}, inplace=True)

            if df_final.empty or df_final["SMILES"].apply(_canonical_smiles).isna().any():
                raise ValueError("The MolSSI workbook contains missing or invalid SMILES")

            cols = ["SMILES"] + [c for c in df_final.columns if c != "SMILES"]
            df_final = df_final[cols]

            csv_path = Path(self.path).with_suffix(".csv")
            write_csv_atomic(df_final, csv_path)
            self.finished.emit(str(csv_path))
            
        except Exception as exc:
            self.error.emit(str(exc))
