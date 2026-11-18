"""Bounded, source-linked facts extracted from ROBERT prediction artifacts."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResultInsight:
    model: str
    variant: str
    category: str
    summary: str
    source: str


_SHAP_RE = re.compile(r"^\s*-\s+(.+?)\s+=\s+min:\s*([-+\deE.]+),\s*max:\s*([-+\deE.]+)\s*$")
_PFI_RE = re.compile(r"^\s*-\s+(.+?)\s+=\s+([-+\deE.]+)\s+\+-\s+([-+\deE.]+)\s*$")
_MAX_ROWS = 200_000
_AQME_SCAN_ROWS = 10_000
_MISSING_VALUES = {"", "nan", "none", "null", "n/a", "inf", "-inf"}


def extract_log_insights(text: str, model: str, source: str) -> tuple[ResultInsight, ...]:
    """Read logged SHAP ranges and PFI scores without interpreting chemistry."""
    variant = ""
    category = ""
    groups: dict[tuple[str, str], list[str]] = {}
    pfi_metrics: dict[str, str] = {}
    for line in text.splitlines():
        folded = line.casefold()
        if "starting model with all variables" in folded:
            variant, category = "No_PFI", ""
        elif "starting model with pfi" in folded:
            variant, category = "PFI", ""
        elif "shap plot saved" in folded:
            category = "shap"
        elif "pfi plot saved" in folded:
            category = "pfi"
        elif re.match(r"\s*o\s+", line):
            category = ""
        if not variant or not category:
            continue
        influence = re.match(r"^\s*Influence on\s+([A-Za-z0-9_-]+)\s*$", line, re.IGNORECASE)
        if category == "pfi" and influence:
            pfi_metrics[variant] = influence.group(1).upper()
        pattern = _SHAP_RE if category == "shap" else _PFI_RE
        match = pattern.match(line)
        if not match:
            continue
        name, first, second = match.groups()
        name = " ".join(name.split())[:100]
        value = (
            f"{name}: min={first}, max={second}"
            if category == "shap" else f"{name}: {first} ± {second}"
        )
        groups.setdefault((variant, category), []).append(value)
    insights = []
    for (variant, category), values in groups.items():
        prefix = (
            "Reported SHAP value ranges (not an importance ranking)"
            if category == "shap" else (
                f"Permutation importance for {pfi_metrics[variant]} (mean ± SD, reported order)"
                if variant in pfi_metrics else "Permutation importance (mean ± SD, reported order)"
            )
        )
        summary = f"{prefix}: {'; '.join(values[:5])}"
        if len(values) > 5:
            summary += f"; {len(values) - 5} more descriptors in source"
        insights.append(ResultInsight(model, variant, category, summary, source))
    return tuple(insights)


def _finite_float(value: str | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float) -> str:
    return f"{value:.3g}"


def extract_prediction_insight(path: Path, model: str, variant: str, source: str) -> ResultInsight | None:
    """Summarize regression residuals without treating prediction SD as an interval."""
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            prediction = next((name for name in fields if name.casefold().endswith("_pred")), None)
            if prediction is None:
                return None
            observed = prediction[:-5]
            observed_available = observed in fields
            identifier = next((name for name in ("code_name", "point_id", "name", "id") if name in fields), None)
            set_column = next((name for name in fields if name.casefold() == "set"), None)
            total = evaluated = missing = 0
            finite_predictions = 0
            prediction_min = prediction_max = None
            absolute_error_sum = 0.0
            largest: list[tuple[float, int, str, float, float, str]] = []
            set_totals: dict[str, tuple[int, float]] = {}
            for row in reader:
                total += 1
                if total > _MAX_ROWS:
                    break
                actual = _finite_float(row.get(observed))
                predicted = _finite_float(row.get(prediction))
                if predicted is not None:
                    finite_predictions += 1
                    prediction_min = predicted if prediction_min is None else min(prediction_min, predicted)
                    prediction_max = predicted if prediction_max is None else max(prediction_max, predicted)
                if actual is None or predicted is None:
                    missing += 1
                    continue
                error = abs(actual - predicted)
                evaluated += 1
                absolute_error_sum += error
                label = " ".join(str(row.get(identifier) or total).split())[:80]
                group = " ".join(str(row.get(set_column) or "").split())[:30] if set_column else ""
                largest.append((error, total, label, actual, predicted, group))
                largest.sort(key=lambda item: (-item[0], item[1]))
                del largest[3:]
                if group:
                    count, error_sum = set_totals.get(group, (0, 0.0))
                    set_totals[group] = (count + 1, error_sum + error)
    except (OSError, csv.Error, UnicodeError):
        return None
    if total == 0:
        return None
    fragments = [f"{min(total, _MAX_ROWS)} rows, {evaluated} evaluated"]
    if finite_predictions:
        fragments.append(
            f"{finite_predictions} finite predictions; predicted value range={_format_number(prediction_min)} "
            f"to {_format_number(prediction_max)}"
        )
    if evaluated:
        fragments.append(f"MAE={_format_number(absolute_error_sum / evaluated)} in target units")
        for group, (count, error_sum) in sorted(set_totals.items()):
            fragments.append(f"{group}: {count} evaluated, MAE={_format_number(error_sum / count)}")
        top = "; ".join(
            f"{label} ({group or 'unlabelled'}): observed={_format_number(actual)}, "
            f"predicted={_format_number(predicted)}, absolute error={_format_number(error)}"
            for error, _, label, actual, predicted, group in largest
        )
        fragments.append(f"largest observed errors: {top}")
    if not observed_available:
        fragments.append("observed target column absent; errors cannot be calculated")
    elif missing:
        fragments.append(f"{missing} rows without comparable observed/predicted values")
    if total > _MAX_ROWS:
        fragments.append(f"scan limited to first {_MAX_ROWS} rows")
    return ResultInsight(model, variant, "prediction_errors", "; ".join(fragments), source)


def extract_aqme_insight(path: Path, input_columns: int | None, source: str) -> ResultInsight | None:
    """Count missing AQME descriptor cells in a bounded scan of an output CSV."""
    if input_columns is None:
        return None
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if len(header) <= input_columns:
                return None
            missing = [0] * (len(header) - input_columns)
            rows = 0
            for row in reader:
                if rows >= _AQME_SCAN_ROWS:
                    break
                rows += 1
                for index in range(input_columns, len(header)):
                    value = row[index].strip().casefold() if index < len(row) else ""
                    if value in _MISSING_VALUES:
                        missing[index - input_columns] += 1
    except (OSError, csv.Error, UnicodeError):
        return None
    affected = [(header[input_columns + index], count) for index, count in enumerate(missing) if count]
    affected.sort(key=lambda item: (-item[1], item[0].casefold()))
    summary = (
        f"{rows} rows scanned; {len(missing)} descriptor columns; "
        f"{len(affected)} columns with missing values"
    )
    if affected:
        summary += "; most missing: " + "; ".join(
            f"{' '.join(name.split())[:80]}: {count}/{rows} missing"
            for name, count in affected[:5]
        )
    if rows >= _AQME_SCAN_ROWS:
        summary += f"; scan limited to first {_AQME_SCAN_ROWS} rows"
    return ResultInsight("", "", "aqme_completeness", summary, source)


def extract_dataset_overlap(main_path: Path, external_path: Path) -> ResultInsight | None:
    """Check literal SMILES overlap between the selected main and external inputs."""
    def read_smiles(path: Path) -> list[str] | None:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                column = next((name for name in (reader.fieldnames or []) if name.casefold() == "smiles"), None)
                if column is None:
                    return None
                return [str(row.get(column) or "").strip() for _, row in zip(range(_MAX_ROWS), reader)]
        except (OSError, csv.Error, UnicodeError):
            return None

    main_smiles = read_smiles(main_path)
    external_smiles = read_smiles(external_path)
    if main_smiles is None or not external_smiles:
        return None
    main_set = set(filter(None, main_smiles))
    matches = [value for value in external_smiles if value and value in main_set]
    summary = f"Exact SMILES overlap: {len(matches)}/{len(external_smiles)} external rows match the main input"
    if matches:
        summary += "; matching external rows are not independent by exact SMILES"
    else:
        summary += "; no exact string matches found"
    summary += "; SMILES were not chemically canonicalized"
    return ResultInsight("", "", "dataset_overlap", summary, f"{main_path.name}, {external_path.name}")
