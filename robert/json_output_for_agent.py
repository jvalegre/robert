"""Shared helpers for writing ROBERT evidence into JSON files.

These functions support ChatBob by turning ROBERT outputs, dataset facts,
and run artifacts into structured files that a user interface or LLM can read.

The helpers are intentionally additive and fail-soft:
they should never change ROBERT calculations, scoring, model selection,
standard output files, or normal run behavior.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd


_MANIFEST_SCHEMA_VERSION = "0.1"
_SUMMARY_SCHEMA_VERSION = "0.1"
_DEFAULT_MODULES = ["CURATE", "GENERATE", "VERIFY", "PREDICT", "REPORT", "AQME", "EVALUATE"]


def _to_unavailable() -> str:
    """Return the standard marker used when ROBERT evidence is not currently captured."""
    return "unavailable"


def _iso_timestamp_from_epoch(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _sha256_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    """Create a file fingerprint so ChatBob can tell whether an output file changed."""
    hasher = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def _extract_pdf_text_preview(file_path: Path, max_chars: int) -> str | None:
    """Try to extract a short text preview from a PDF without interrupting ROBERT if it fails."""
    try:
        import fitz

        doc = fitz.open(str(file_path))
        chunks: List[str] = []
        chars = 0
        for page in doc:
            if chars >= max_chars:
                break
            text = page.get_text("text")
            if not text:
                continue
            remaining = max_chars - chars
            fragment = text[:remaining]
            chunks.append(fragment)
            chars += len(fragment)
        doc.close()
        out = "\n".join(chunks).strip()
        return out if out else None
    except Exception:
        return None


def _extract_text_preview(file_path: Path, max_chars: int = 4000) -> str | None:
    """Read a short preview from text-like ROBERT output files for later search or display."""
    suffix = file_path.suffix.lower()
    try:
        if suffix in {".dat", ".txt", ".log", ".csv", ".json", ".yaml", ".yml"}:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            return text[:max_chars] if text else None
        if suffix == ".pdf":
            return _extract_pdf_text_preview(file_path, max_chars=max_chars)
    except Exception:
        return None
    return None


def _infer_artifact_type(file_path: Path) -> str:
    """Label a ROBERT output file as CSV, DAT, image, PDF, text, or other."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".dat":
        return "dat"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".tif", ".tiff", ".gif", ".webp"}:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".json", ".yaml", ".yml", ".txt", ".log"}:
        return "textual"
    return "binary_or_other"


def _build_file_record(file_path: Path, run_dir: Path) -> Dict[str, Any]:
    """Describe one output file using its path, type, size, timestamp, fingerprint, and preview."""
    stats = file_path.stat()
    rel = file_path.relative_to(run_dir)
    artifact_type = _infer_artifact_type(file_path)

    return {
        "path": str(file_path),
        "path_relative_to_run": str(rel).replace("\\", "/"),
        "artifact_type": artifact_type,
        "extension": file_path.suffix.lower(),
        "size_bytes": int(stats.st_size),
        "modified_utc": _iso_timestamp_from_epoch(stats.st_mtime),
        "sha256": _sha256_file(file_path),
        "text_preview": _extract_text_preview(file_path),
    }


def collect_module_manifest(module_name: str, run_dir: str | Path, max_files: int = 1000) -> Dict[str, Any]:
    """Build a structured inventory of files produced by one ROBERT module.

    This helps ChatBob answer questions such as which files were created,
    where they are stored, and which outputs are available for inspection.
    """

    run_root = Path(run_dir)
    module_dir = run_root / module_name
    payload: Dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "artifact_type": "module_manifest",
        "module": str(module_name),
        "run_dir": str(run_root),
        "module_dir": str(module_dir),
        "module_dir_exists": bool(module_dir.exists() and module_dir.is_dir()),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": 0,
        "files": [],
    }

    if not (module_dir.exists() and module_dir.is_dir()):
        return payload

    files: List[Path] = []
    for candidate in module_dir.rglob("*"):
        if candidate.is_file():
            files.append(candidate)

    files = sorted(files, key=lambda p: str(p).lower())
    if max_files > 0:
        files = files[:max_files]

    records = [_build_file_record(fp, run_root) for fp in files]
    payload["file_count"] = int(len(records))
    payload["files"] = records
    return _to_json_safe(payload)


def write_module_manifest(module_name: str, run_dir: str | Path) -> bool:
    """Write the file inventory for one ROBERT module as a JSON manifest."""

    run_root = Path(run_dir)
    output_path = run_root / f"{module_name}_manifest.json"
    manifest = collect_module_manifest(module_name, run_root)
    return write_json(manifest, output_path)


def write_archive_manifests(run_dir: str | Path, modules: List[str] | None = None) -> Dict[str, Any]:
    """Create file inventories for the ROBERT modules in one completed run.

    A ROBERT run can produce many outputs across folders such as CURATE,
    GENERATE, VERIFY, PREDICT, REPORT, AQME, and EVALUATE. This helper
    asks each module folder to write a manifest that lists its available
    files.

    The returned status tells ChatBob which module inventories were written
    successfully, so the interface can know what evidence is available before
    trying to explain the run.
    """

    run_root = Path(run_dir)
    module_names = modules or list(_DEFAULT_MODULES)

    status: Dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "artifact_type": "archive_manifest_write_status",
        "run_dir": str(run_root),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "module_results": [],
    }

    for module_name in module_names:
        output_path = run_root / f"{module_name}_manifest.json"
        succeeded = write_module_manifest(module_name, run_root)
        status["module_results"].append(
            {
                "module": str(module_name),
                "manifest_path": str(output_path),
                "succeeded": bool(succeeded),
            }
        )

    return _to_json_safe(status)


def collect_run_summary(
    run_dir: str | Path,
    command: List[str] | None = None,
    return_code: int | None = None,
    wrapper_metadata_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Create a high-level map of one completed ROBERT run.
    A completed ROBERT run can contain many module folders, manifest files,
    plots, CSV files, reports, and other outputs.
    This helper gathers the top-level information needed to understand
    what is available in the run. ChatBob can use this summary as a starting
    point before it reads the more detailed CURATE, GENERATE,
    VERIFY, PREDICT, or REPORT evidence.
    """

    run_root = Path(run_dir)
    manifest_paths = sorted(run_root.glob("*_manifest.json"))

    summary: Dict[str, Any] = {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "artifact_type": "run_summary",
        "run_dir": str(run_root),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": return_code,
        "command": command or [],
        "wrapper_metadata_path": str(wrapper_metadata_path) if wrapper_metadata_path is not None else None,
        "module_manifests": [],
        "top_level_files": [],
    }

    for manifest_path in manifest_paths:
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest_data = {}

        summary["module_manifests"].append(
            {
                "module": manifest_data.get("module"),
                "manifest_path": str(manifest_path),
                "file_count": manifest_data.get("file_count"),
                "module_dir_exists": manifest_data.get("module_dir_exists"),
            }
        )

    top_level_files = []
    for file_path in sorted(run_root.glob("*")):
        if file_path.is_file():
            top_level_files.append(_build_file_record(file_path, run_root))
    summary["top_level_files"] = top_level_files

    return _to_json_safe(summary)


def write_run_summary(
    run_dir: str | Path,
    command: List[str] | None = None,
    return_code: int | None = None,
    wrapper_metadata_path: str | Path | None = None,
) -> bool:
    """Write the high-level run summary as run_summary.json at the run root.
    This gives ChatBob a single file that points to the main outputs of a completed ROBERT run.
    The summary does not replace the original ROBERT files;
    it simply helps locate and organize them.
    """

    run_root = Path(run_dir)
    output_path = run_root / "run_summary.json"
    summary = collect_run_summary(
        run_root,
        command=command,
        return_code=return_code,
        wrapper_metadata_path=wrapper_metadata_path,
    )
    return write_json(summary, output_path)


def _read_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read the incoming CSV with separator autodetection to preserve raw intake facts.
    This is done before ROBERT changes the dataset.
    CURATE may later remove, reorder, transform, or filter parts of the data.
    This helper preserves the starting point so ChatBob can explain what the user originally gave to ROBERT.
    """

    return pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8")


def _to_json_safe(obj: Any) -> Any:
    """Convert scientific Python objects (pandas/numpy/path values) into values that can be written as JSON.
    
    ROBERT uses pandas, NumPy, file paths, timestamps, and missing values.
    Many of these objects cannot be written directly into JSON.
    This helper translates them into plain strings, numbers, lists,
    dictionaries, or nulls that ChatBob can read reliably.
    """

    if isinstance(obj, Path):
        return str(obj)

    if obj is None:
        return None

    if isinstance(obj, (pd.Series, pd.Index)):
        return [_to_json_safe(x) for x in obj.tolist()]

    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x) for x in obj.tolist()]

    if pd.api.types.is_scalar(obj) and pd.isna(obj):
        return None

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        if np.isfinite(obj):
            return float(obj)
        return None

    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}

    return obj


def _is_boolean_like(series: pd.Series) -> bool:
    """Detect columns that behave like booleans even when encoded as strings or 0/1.
    
    Detect whether a column behaves like yes/no or true/false data.
    Some datasets store boolean information as 0/1, yes/no, true/false,
    or similar text labels.
    This helper identifies those simple two-state columns so the
    dataset profile can describe them more clearly."""

    non_null = series.dropna()
    if non_null.empty:
        return False

    lowered = non_null.astype(str).str.strip().str.lower()
    allowed = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"}
    uniques = set(lowered.unique())
    return len(uniques) <= 2 and uniques.issubset(allowed)


def _type_label(series: pd.Series) -> str:
    """Assign a simple observable type label to one dataset column.
    This helper labels a column as numeric, categorical, text-like,
    boolean-like, or unknown based only on what is visible in the data.
    It does not make a scientific judgment; it creates a plain-language
    description that ChatBob can use when explaining the dataset."""

    if pd.api.types.is_bool_dtype(series) or _is_boolean_like(series):
        return "boolean_like"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    non_null = series.dropna()
    if non_null.empty:
        return "unknown"

    if pd.api.types.is_string_dtype(series) or series.dtype == "object":
        unique_count = non_null.nunique(dropna=True)
        ratio = unique_count / max(len(non_null), 1)
        avg_len = non_null.astype(str).str.len().mean()
        if ratio > 0.5 or avg_len > 30:
            return "text"
        return "categorical"

    return "categorical"


def measure_dataset_shape(raw_df: pd.DataFrame) -> Dict[str, Any]:
    
    """Record the basic size and layout of the original dataset.
    This captures the number of rows, number of columns, column names, column order,
    duplicate rows, and duplicate columns before CURATE changes anything.
    It helps ChatBob answer the basic question: what did the user start with?
    """
    
    return {
        "number_of_rows": int(len(raw_df)),
        "number_of_columns": int(len(raw_df.columns)),
        "column_names": [str(c) for c in raw_df.columns],
        "column_order": [str(c) for c in raw_df.columns],
        "duplicate_row_count": int(raw_df.duplicated().sum()),
        "duplicate_column_count": int(raw_df.T.duplicated().sum()),
    }


def measure_missingness(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Measure missing values in the original dataset.
    Missing values can affect how ROBERT curates and models a dataset.
    This helper records how many values are missing, which columns are affected,
    and whether missingness is small or substantial.
    """
    total_cells = int(raw_df.shape[0] * raw_df.shape[1])
    missing_by_col = raw_df.isna().sum()
    total_missing = int(missing_by_col.sum())
    rows_with_any_missing = int(raw_df.isna().any(axis=1).sum())

    def _cols_over(threshold: float) -> List[str]:
        pct = (missing_by_col / max(len(raw_df), 1)) * 100.0
        return [str(col) for col in pct.index if pct.loc[col] > threshold]

    return {
        "total_missing_values": total_missing,
        "percent_missing_overall": float((100.0 * total_missing / total_cells) if total_cells else 0.0),
        "columns_with_missing_values": [str(c) for c in missing_by_col.index if int(missing_by_col[c]) > 0],
        "missing_values_by_column": {str(c): int(v) for c, v in missing_by_col.items()},
        "rows_with_any_missing_value": rows_with_any_missing,
        "percent_rows_with_missing": float((100.0 * rows_with_any_missing / len(raw_df)) if len(raw_df) else 0.0),
        "columns_over_5_percent_missing": _cols_over(5.0),
        "columns_over_20_percent_missing": _cols_over(20.0),
        "columns_over_50_percent_missing": _cols_over(50.0),
    }


def measure_column_types(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Group the original dataset columns into simple observable categories.
    The helper identifies columns that appear numeric, categorical, text-like,
    boolean-like, constant, mostly constant, or mixed in type.
    This gives ChatBob a practical way to explain the character of the starting dataset.
    """
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    text_columns: List[str] = []
    boolean_like_columns: List[str] = []
    constant_columns: List[str] = []
    mostly_constant_columns: List[str] = []
    mixed_type_columns: List[str] = []

    for col in raw_df.columns:
        series = raw_df[col]
        label = _type_label(series)

        if label == "numeric":
            numeric_columns.append(str(col))
        elif label == "text":
            text_columns.append(str(col))
        elif label == "boolean_like":
            boolean_like_columns.append(str(col))
        else:
            categorical_columns.append(str(col))

        non_null = series.dropna()
        if not non_null.empty:
            unique_values = non_null.nunique(dropna=True)
            if unique_values == 1:
                constant_columns.append(str(col))
            else:
                top_freq = non_null.value_counts(dropna=True).iloc[0]
                if (top_freq / len(non_null)) >= 0.95:
                    mostly_constant_columns.append(str(col))

            python_types = {type(v).__name__ for v in non_null.head(1000).tolist()}
            if len(python_types) > 1:
                mixed_type_columns.append(str(col))

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "text_columns": text_columns,
        "boolean_like_columns": boolean_like_columns,
        "constant_columns": constant_columns,
        "mostly_constant_columns": mostly_constant_columns,
        "mixed_type_columns": mixed_type_columns,
    }


def measure_per_column_stats(raw_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Create a compact summary for each column in the original dataset.
    Each column summary includes its inferred type, number of non-missing values,
    number of unique values, and a few example values.
    This helps ChatBob explain individual columns without needing to display the full dataset.
    """
    per_column: Dict[str, Dict[str, Any]] = {}

    for col in raw_df.columns:
        series = raw_df[col]
        non_null = series.dropna()
        example_values = [_to_json_safe(v) for v in non_null.drop_duplicates().head(5).tolist()]

        per_column[str(col)] = {
            "inferred_type": _type_label(series),
            "non_null_count": int(non_null.shape[0]),
            "unique_value_count": int(non_null.nunique(dropna=True)),
            "example_values": example_values,
        }

    return per_column


def measure_target_profile(raw_df: pd.DataFrame, y: str) -> Dict[str, Any]:
    """Describe the target column before ROBERT changes the dataset.
    The target column is the property ROBERT is trying to model.
    This helper records missing values, unique values, numeric range, average, spread,
    likely problem type, and class counts when the target appears categorical.
    """
    if y not in raw_df.columns:
        return {
            "target_name": y,
            "target_missing_count": None,
            "target_unique_count": None,
            "target_min": None,
            "target_max": None,
            "target_mean": None,
            "target_median": None,
            "target_standard_deviation": None,
            "target_skew_estimate": None,
            "target_value_counts_if_categorical": {},
            "target_type_guess": "missing",
            "possible_problem_type": "unknown",
            "regression_likely": False,
            "classification_likely": False,
            "ambiguous": True,
        }

    series = raw_df[y]
    non_null = series.dropna()
    unique_count = int(non_null.nunique(dropna=True))
    target_label = _type_label(series)

    numeric_non_null = pd.to_numeric(non_null, errors="coerce").dropna()
    has_numeric = not numeric_non_null.empty and len(numeric_non_null) == len(non_null)

    classification_likely = bool((target_label in {"categorical", "boolean_like"}) or unique_count <= 10)
    regression_likely = bool(has_numeric and unique_count > 10)
    ambiguous = bool(classification_likely == regression_likely)

    if classification_likely:
        value_counts = {str(k): int(v) for k, v in non_null.value_counts(dropna=True).to_dict().items()}
    else:
        value_counts = {}

    if has_numeric and len(numeric_non_null) > 0:
        target_min = float(numeric_non_null.min())
        target_max = float(numeric_non_null.max())
        target_mean = float(numeric_non_null.mean())
        target_median = float(numeric_non_null.median())
        target_std = float(numeric_non_null.std()) if len(numeric_non_null) > 1 else 0.0
        target_skew = float(numeric_non_null.skew()) if len(numeric_non_null) > 2 else 0.0
    else:
        target_min = None
        target_max = None
        target_mean = None
        target_median = None
        target_std = None
        target_skew = None

    if classification_likely and not regression_likely:
        problem_type = "classification"
    elif regression_likely and not classification_likely:
        problem_type = "regression"
    else:
        problem_type = "ambiguous"

    return {
        "target_name": str(y),
        "target_missing_count": int(series.isna().sum()),
        "target_unique_count": unique_count,
        "target_min": target_min,
        "target_max": target_max,
        "target_mean": target_mean,
        "target_median": target_median,
        "target_standard_deviation": target_std,
        "target_skew_estimate": target_skew,
        "target_value_counts_if_categorical": value_counts,
        "target_type_guess": target_label,
        "possible_problem_type": problem_type,
        "regression_likely": regression_likely,
        "classification_likely": classification_likely,
        "ambiguous": ambiguous,
    }


def measure_descriptor_counts(raw_df: pd.DataFrame, y: str, ignore: List[str]) -> Dict[str, Any]:
    """Count the starting descriptor columns before CURATE begins.
    Descriptors are the input features available to build a model,
    excluding the target column and ignored columns.
    This helper records how many descriptors exist and how many appear numeric,
    categorical, text-like, constant, near-constant, or high in missing values.
    """
    ignore_set = set(ignore or [])
    descriptors = [c for c in raw_df.columns if c != y and c not in ignore_set]
    descriptor_df = raw_df[descriptors] if descriptors else pd.DataFrame(index=raw_df.index)

    numeric = []
    categorical = []
    text = []
    constant = []
    near_constant = []
    high_missing = []

    for col in descriptors:
        series = raw_df[col]
        label = _type_label(series)
        if label == "numeric":
            numeric.append(str(col))
        elif label == "text":
            text.append(str(col))
        else:
            categorical.append(str(col))

        non_null = series.dropna()
        if not non_null.empty:
            unique_vals = non_null.nunique(dropna=True)
            if unique_vals == 1:
                constant.append(str(col))
            else:
                top_freq = non_null.value_counts(dropna=True).iloc[0]
                if (top_freq / len(non_null)) >= 0.95:
                    near_constant.append(str(col))

        missing_pct = (float(series.isna().sum()) / len(raw_df) * 100.0) if len(raw_df) else 0.0
        if missing_pct > 20.0:
            high_missing.append(str(col))

    return {
        "initial_descriptor_count": int(len(descriptors)),
        "numeric_descriptor_count": int(len(numeric)),
        "categorical_descriptor_count": int(len(categorical)),
        "text_descriptor_count": int(len(text)),
        "constant_descriptor_count": int(len(constant)),
        "near_constant_descriptor_count": int(len(near_constant)),
        "high_missing_descriptor_count": int(len(high_missing)),
    }


def profile_input_dataset(csv_path: str | Path, y: str, ignore: List[str] | None = None) -> Dict[str, Any]:
    """Build the raw dataset profile used to explain the starting data.
    This helper re-reads the original CSV before CURATE changes it
    and gathers the dataset shape, missingness, column types, per-column summaries,
    target profile, and descriptor counts into one JSON-ready record.
    ChatBob can use this file to explain what the user gave ROBERT before
    any curation or model-building decisions were made.
    """

    raw_df = _read_raw_csv(csv_path)
    ignore = ignore or []

    payload = {
        "schema_version": "0.1",
        "artifact_type": "incoming_dataset_profile",
        "source_csv": str(csv_path),
        "dataset_shape": measure_dataset_shape(raw_df),
        "missingness": measure_missingness(raw_df),
        "column_types": measure_column_types(raw_df),
        "per_column": measure_per_column_stats(raw_df),
        "target_profile": measure_target_profile(raw_df, y),
        "descriptor_counts": measure_descriptor_counts(raw_df, y, ignore),
    }
    return _to_json_safe(payload)


def write_json(data: Dict[str, Any], output_path: str | Path) -> bool:
    """Write a JSON file without risking the ROBERT run.
    This helper writes structured evidence to disk and returns True if the write succeeds
    or False if it fails. A JSON failure should not stop ROBERT or alter any standard ROBERT output.
    """

    out = Path(output_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(_to_json_safe(data), handle, indent=2, sort_keys=False, ensure_ascii=False)
        return True
    except Exception:
        return False


def init_module_audit(
    module: str,
    artifact_type: str,
    source_files: List[str] | None = None,
    command_line: str | None = None,
) -> Dict[str, Any]:
    """Start a structured audit record for one ROBERT module.
    CURATE, GENERATE, VERIFY, PREDICT, and REPORT each produce evidence
    that ChatBob may need to explain later.
    This helper creates a common starting structure so each module can record what happened in a consistent way.
    """

    try:
        payload = {
            "schema_version": "0.1",
            "module": str(module),
            "artifact_type": str(artifact_type),
            "status": "in_progress",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": [str(p) for p in (source_files or [])],
            "command_line": str(command_line) if command_line is not None else None,
            "sections": {},
            "events": [],
            "notes": [],
        }
        return _to_json_safe(payload)
    except Exception:
        return {
            "schema_version": "0.1",
            "module": str(module),
            "artifact_type": str(artifact_type),
            "status": "in_progress",
            "sections": {},
            "events": [],
        }


def audit_event(
    audit: Dict[str, Any],
    event_type: str,
    payload: Dict[str, Any] | None = None,
    evidence_level: str = "direct",
    dat_text: str | None = None,
) -> Dict[str, Any]:
    """Add one event to a module audit without interrupting ROBERT.
    An event is something that happened during a module, such as a filtering step,
    warning, file creation, model check, or result.
    ChatBob can later use these events to explain the sequence of what ROBERT did.
    """

    try:
        if not isinstance(audit, dict):
            return audit

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": str(event_type),
            "evidence_level": str(evidence_level),
            "payload": _to_json_safe(payload or {}),
        }
        if dat_text is not None:
            event["dat_text_preview"] = str(dat_text)[:1000]

        events = audit.get("events", [])
        if not isinstance(events, list):
            events = []
        events.append(event)
        audit["events"] = events
        return audit
    except Exception:
        return audit


def audit_set(
    audit: Dict[str, Any],
    section: str,
    key: str,
    value: Any,
    evidence_level: str = "direct",
) -> Dict[str, Any]:
    """Store one named value inside a module audit section.
    This helper lets the JSON audit record important values in organized sections
    such as inputs, counts, warnings, model results, or files created.
    The evidence level marks whether the value came directly from ROBERT
    or was added as a simple helper summary for ChatBob.
    """

    try:
        if not isinstance(audit, dict):
            return audit

        sections = audit.get("sections", {})
        if not isinstance(sections, dict):
            sections = {}

        sec = sections.get(str(section), {})
        if not isinstance(sec, dict):
            sec = {}

        sec[str(key)] = {
            "value": _to_json_safe(value),
            "evidence_level": str(evidence_level),
        }
        sections[str(section)] = sec
        audit["sections"] = sections
        return audit
    except Exception:
        return audit


def finalize_module_audit(
    audit: Dict[str, Any],
    output_path: str | Path,
    status: str = "completed",
) -> bool:
    """Finish and write a module audit JSON file.
    This helper marks the audit as completed, adds an ending timestamp,
    and writes the JSON file using the fail-soft JSON writer.
    It gives each module a consistent way to close its evidence record.
    """

    try:
        if not isinstance(audit, dict):
            return False
        audit["status"] = str(status)
        audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
        return write_json(audit, output_path)
    except Exception:
        return False


def write_json_output_audit(
    audit_path: str | Path,
    module: str,
    attempted_output_path: str | Path,
    attempted: bool,
    succeeded: bool,
    artifact: str = "dataset_profile.json",
    error: Exception | None = None,
) -> bool:
    """Record whether the extra JSON evidence layer succeeded or failed.
    This audit is about the ChatBob-facing JSON files only.
    It records which JSON artifact was attempted, whether it succeeded,
    and what error occurred if it failed.
    It must not write messages into standard ROBERT outputs such as DAT, CSV, image, model, or report files.
    """

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": str(module),
        "layer": "json-output-for-agent",
        "layer_note": "This audit belongs to the JSON-output-for-agent layer and does not modify standard ROBERT outputs.",
        "artifact": str(artifact),
        "attempted": bool(attempted),
        "succeeded": bool(succeeded),
        "attempted_output_path": str(attempted_output_path),
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }

    out = Path(audit_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "schema_version": "0.1",
            "artifact_type": "json_output_audit",
            "module": str(module),
            "events": [],
        }
        if out.exists():
            try:
                with out.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if isinstance(existing, dict):
                    payload.update(existing)
            except Exception:
                pass

        events = payload.get("events", [])
        if not isinstance(events, list):
            events = []
        events.append(event)
        payload["events"] = events

        with out.open("w", encoding="utf-8") as handle:
            json.dump(_to_json_safe(payload), handle, indent=2, sort_keys=False, ensure_ascii=False)
        return True
    except Exception:
        return False


# Legacy fallback payload builder; runtime audit capture is preferred when available.
def build_curate_audit_payload(
    source_csv: str | Path,
    destination_dir: str | Path,
    target_column: str,
    names_column: str | None,
    ignored_columns: List[str] | None,
    discarded_columns_requested: List[str] | None,
    rows_before_curate: int | None,
    rows_after_curate: int | None,
    columns_before_curate: int | None,
    columns_after_curate: int | None,
) -> Dict[str, Any]:
    """Build the CURATE audit JSON from values that are directly observable.
    This helper records the input file, target column, ignored columns,
    requested discarded columns, before-and-after dataset counts, expected CURATE output files,
    and fields that are not yet available in structured form.
    It is intentionally conservative.
    If ROBERT has not exposed a value in a reliable structured way,
    the audit marks that value as unavailable rather than guessing.
    """

    src = Path(source_csv)
    dst = Path(destination_dir)
    ignored_columns = ignored_columns or []
    discarded_columns_requested = discarded_columns_requested or []

    csv_stem = src.stem
    files_written: List[Dict[str, Any]] = []

    candidate_paths: List[Path] = [
        dst / f"{csv_stem}_CURATE.csv",
        dst / "CURATE_options.csv",
        dst / "CURATE_data.dat",
        dst / "Pearson_heatmap.png",
    ]
    candidate_paths.extend(sorted(dst.glob(f"{csv_stem}_CURATE_*.csv")))

    seen: set[str] = set()
    for file_path in candidate_paths:
        key = str(file_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        files_written.append(
            {
                "path": str(file_path),
                "artifact_type": _infer_artifact_type(file_path),
                "exists": bool(file_path.exists()),
            }
        )

    payload = {
        "schema_version": "0.1",
        "artifact_type": "curate_audit",
        "module": "CURATE",
        "status": "completed",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "source_csv": str(source_csv),
            "target_column": str(target_column),
            "names_column": str(names_column) if names_column else None,
            "ignored_columns": [str(col) for col in ignored_columns],
            "discarded_columns_requested": [str(col) for col in discarded_columns_requested],
        },
        "observed_counts": {
            "rows_before_curate": rows_before_curate,
            "rows_after_curate": rows_after_curate,
            "columns_before_curate": columns_before_curate,
            "columns_after_curate": columns_after_curate,
            "descriptors_removed_duplicate_filter": _to_unavailable(),
            "descriptors_removed_missingness": _to_unavailable(),
            "descriptors_removed_categorical_transform": _to_unavailable(),
            "descriptors_removed_correlation_filter": _to_unavailable(),
            "descriptors_removed_other": _to_unavailable(),
        },
        "files_written": files_written,
        "standard_output_paths": {
            "curate_dat": str(dst / "CURATE_data.dat"),
            "curate_options_csv": str(dst / "CURATE_options.csv"),
            "curated_csvs": [
                str(fp)
                for fp in sorted(dst.glob(f"{csv_stem}_CURATE*.csv"))
            ],
        },
        "unavailable_fields": [
            "observed_counts.descriptors_removed_duplicate_filter",
            "observed_counts.descriptors_removed_missingness",
            "observed_counts.descriptors_removed_categorical_transform",
            "observed_counts.descriptors_removed_correlation_filter",
            "observed_counts.descriptors_removed_other",
        ],
        "notes": [
            "Step-specific descriptor removal counts are marked unavailable until direct in-memory counters are exposed.",
            "This artifact records observable CURATE evidence only and does not modify standard ROBERT outputs.",
        ],
    }

    return _to_json_safe(payload)
