"""Helpers for additive JSON artifacts used by the agent-facing UI layer.

These helpers are intentionally fail-soft and do not modify ROBERT calculations.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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
    error: Exception | None = None,
) -> bool:
    """Write a JSON-layer audit record without touching standard ROBERT output files."""

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": str(module),
        "layer": "json-output-for-agent",
        "layer_note": "This audit belongs to the JSON-output-for-agent layer and does not modify standard ROBERT outputs.",
        "artifact": "dataset_profile.json",
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
