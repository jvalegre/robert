"""Helpers for additive JSON artifacts used by the agent-facing UI layer.

These helpers are intentionally fail-soft and do not modify ROBERT calculations.
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
    return "unavailable"


def _iso_timestamp_from_epoch(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _sha256_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str | None:
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
    # Prefer simple best-effort extraction. Failures are acceptable and represented as None.
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
    """Collect an archive-only manifest for one module from an existing run directory."""

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
    """Write one module manifest JSON into the archive run root."""

    run_root = Path(run_dir)
    output_path = run_root / f"{module_name}_manifest.json"
    manifest = collect_module_manifest(module_name, run_root)
    return write_json(manifest, output_path)


def write_archive_manifests(run_dir: str | Path, modules: List[str] | None = None) -> Dict[str, Any]:
    """Write manifests for archive module folders and return a write-status payload."""

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
    """Collect a run-level summary that indexes module manifests and top-level outputs."""

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
    """Write run_summary.json at the run root."""

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
    """Read the incoming CSV with separator autodetection to preserve raw intake facts."""

    return pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8")


def _to_json_safe(obj: Any) -> Any:
    """Convert pandas/numpy/path values into standard JSON-serializable Python types."""

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
    """Detect columns that behave like booleans even when encoded as strings or 0/1."""

    non_null = series.dropna()
    if non_null.empty:
        return False

    lowered = non_null.astype(str).str.strip().str.lower()
    allowed = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"}
    uniques = set(lowered.unique())
    return len(uniques) <= 2 and uniques.issubset(allowed)


def _type_label(series: pd.Series) -> str:
    """Return a simple, observable type label for a column."""

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
    return {
        "number_of_rows": int(len(raw_df)),
        "number_of_columns": int(len(raw_df.columns)),
        "column_names": [str(c) for c in raw_df.columns],
        "column_order": [str(c) for c in raw_df.columns],
        "duplicate_row_count": int(raw_df.duplicated().sum()),
        "duplicate_column_count": int(raw_df.T.duplicated().sum()),
    }


def measure_missingness(raw_df: pd.DataFrame) -> Dict[str, Any]:
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
    """Build a raw-input dataset profile JSON payload from the incoming CSV file."""

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
    """Write JSON to disk in a fail-safe way; returns False when writing fails."""

    out = Path(output_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(_to_json_safe(data), handle, indent=2, sort_keys=False, ensure_ascii=False)
        return True
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
    """Write a JSON-layer audit record without touching standard ROBERT output files."""

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
    """Build CURATE module audit payload from directly observable values only."""

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
