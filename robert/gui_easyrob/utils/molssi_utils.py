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
import re
import urllib.parse

# ------------------------------------------------------------
# Third-party libraries
# ------------------------------------------------------------
import pandas as pd
import requests

from rdkit import Chem

from PySide6.QtCore import QThread, Signal

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

    def __init__(self, urls, target_path, parent=None):
        super().__init__(parent)
        self.urls = urls
        self.target_path = target_path

    def run(self):
        """Run the MolSSI dataset download process in a background thread."""
        for url in self.urls:
            try:
                response = requests.get(url, timeout=60)
                if response.status_code == 200 and response.content:
                    self.target_path.write_bytes(response.content)
                    self.finished.emit(str(self.target_path))
                    return
            except Exception:
                pass

        self.error.emit("Unable to download MolSSI dataset.")


def resolve_molssi_descriptors(df):
    """Resolve a full-coverage MolSSI descriptor dataset for a SMILES table."""
    original_input_columns = list(df.columns)
    smiles_col = next((col for col in df.columns if col.lower() == "smiles"), None)

    if smiles_col is None:
        return {
            "available": False,
            "export_df": None,
            "library": None,
            "data_type": None,
            "reason": "No SMILES column found",
        }

    def canonicalize(smiles):
        """Canonicalize a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    df_work = df.copy()
    df_work["_smiles_original"] = df_work[smiles_col]
    df_work["_smiles_canonical"] = df_work[smiles_col].apply(canonicalize)
    df_work = df_work.dropna(subset=["_smiles_canonical"])
    df_work = df_work.drop_duplicates(subset=["_smiles_canonical"])

    smiles_list = df_work["_smiles_canonical"].tolist()
    if not smiles_list:
        return {
            "available": False,
            "export_df": None,
            "library": None,
            "data_type": None,
            "reason": "No valid SMILES",
        }

    def chunked(lst, size):
        """Split a list into chunks of a specified size."""
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    def safe_query_batched(smiles, library, data_type, batch_size=200):
        """Query MolSSI API in batches and handle partial failures."""
        all_frames = []

        for batch in chunked(smiles, batch_size):
            try:
                encoded = [urllib.parse.quote(s) for s in batch]
                query = ",".join(encoded)
                url = (
                    f"https://descriptor-libraries.molssi.org/api/"
                    f"{library}/molecules/data/export/batch"
                    f"?molecule_smiles={query}"
                    f"&data_type={data_type}&return_type=json"
                )
                response = requests.get(url, timeout=60)

                if response.status_code != 200:
                    return None

                data = response.json()
                if not isinstance(data, list) or not data:
                    return None

                all_frames.append(pd.DataFrame(data))
            except Exception:
                return None

        return pd.concat(all_frames, ignore_index=True)

    def full_coverage(smiles, df_api):
        """Check if all SMILES are covered by the MolSSI API."""
        if df_api is None or df_api.empty:
            return False

        requested = {
            Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
            for s in smiles
            if Chem.MolFromSmiles(s)
        }
        returned = {
            Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
            for s in df_api["smiles"]
            if Chem.MolFromSmiles(s)
        }
        return not (requested - returned)

    dft_libraries = [
        "acids",
        "primary-amines",
        "secondary-amines",
        "amines",
        "anilines",
        "sulfonimidamides",
        "cyanoarenes",
        "unactivated-primary-alkyl-bromides",
        "sulfonyl-fluorides",
        "quinones",
        "kraken",
    ]

    for lib in dft_libraries:
        df_api = safe_query_batched(smiles_list, lib, "DFT")
        if full_coverage(smiles_list, df_api):
            return _prepare_export(df_work, df_api, smiles_col, lib, "DFT", original_input_columns)

    df_api = safe_query_batched(smiles_list, "kraken", "ML")
    if full_coverage(smiles_list, df_api):
        return _prepare_export(df_work, df_api, smiles_col, "kraken", "ML", original_input_columns)

    return {
        "available": False,
        "export_df": None,
        "library": None,
        "data_type": None,
        "reason": "MolSSI does not provide full coverage",
    }


def _molssi_test_dataset_available(library_slug):
    """Check whether the full MolSSI test dataset exists for a given library."""
    filename = f"{library_slug}_library.xlsx"
    url = f"https://descriptor-libraries.molssi.org/{library_slug}/content/{filename}"

    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except Exception as exc:
        print("HEAD request failed:", repr(exc))
        return False


def _prepare_export(df_work, df_api, smiles_col, library, data_type, original_input_columns):
    """Prepare the merged MolSSI export DataFrame for use in easyROB."""
    try:
        df_api = df_api.copy()
        df_api["_smiles_canonical"] = df_api["smiles"].apply(
            lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
            if Chem.MolFromSmiles(s)
            else None
        )

        df_merged = df_work.merge(df_api, on="_smiles_canonical", how="left")
        export_df = df_merged.drop(
            columns=[c for c in ["_smiles_canonical", "smiles"] if c in df_merged.columns]
        )
        export_df[smiles_col] = export_df["_smiles_original"]
        export_df = export_df.drop(columns=["_smiles_original"])

        cols = [smiles_col] + [c for c in export_df.columns if c != smiles_col]
        export_df = export_df[cols]
        export_df = export_df.copy()
        export_df.columns = [
            fix_greek_caps_columns(c) if c != smiles_col else c
            for c in export_df.columns
        ]

        if "molecule_id" in export_df.columns:
            only_smiles_input = len(original_input_columns) == 1 and original_input_columns[0].lower() == "smiles"
            if not only_smiles_input:
                export_df = export_df.drop(columns=["molecule_id"])

        export_available = _molssi_test_dataset_available(library)
        return {
            "available": True,
            "export_available": export_available,
            "export_df": export_df,
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
        return None

    def _read_descriptors(self, sheet_name):
        """Read the descriptors from the spreadsheet."""
        raw = pd.read_excel(self.path, sheet_name=sheet_name, header=None)

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
                for key in ("Compound_Name", "ID", "Numerical_ID"):
                    if key in df_final.columns and "code_name" not in df_final.columns:
                        df_final.rename(columns={key: "code_name"}, inplace=True)
                        break
            else:
                smiles_col_desc = self._find_smiles_column(df_desc.columns)
                if smiles_col_desc:
                    df_final = df_desc.rename(columns={smiles_col_desc: "SMILES"})
                    for key in ("Compound_Name", "ID", "Numerical_ID"):
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
                    df_final = df_desc.merge(df_id[[join_key, smiles_col_id]], on=join_key, how="left")

                    if "code_name" not in df_final.columns:
                        df_final.rename(columns={join_key: "code_name"}, inplace=True)

                    df_final.rename(columns={smiles_col_id: "SMILES"}, inplace=True)

            cols = ["SMILES"] + [c for c in df_final.columns if c != "SMILES"]
            df_final = df_final[cols]

            csv_path = str(Path(self.path).with_suffix(".csv"))
            df_final.to_csv(csv_path, index=False)
            self.finished.emit(csv_path)
            
        except Exception as exc:
            self.error.emit(str(exc))