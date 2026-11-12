"""Bounded structural checks for CSVs selected in the EasyROB GUI."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .result_insights import ResultInsight


_MAX_SCAN_ROWS = 500
_MAX_SCAN_COLUMNS = 500
_MISSING = {"", "nan", "none", "null", "n/a", "inf", "-inf"}


def is_selected_csv_question(question: str) -> bool:
    """Identify questions about the user's selected table, not CSV concepts."""
    folded = str(question or "").casefold()
    topic = re.search(r"\b(?:csv|dataset|datos?|data|tabla|archivo|file)\b", folded)
    contextual = re.search(
        r"\b(?:mi|mis|my|our|this|these|este|esta|estos|estas|selected|cargad\w*|subid\w*|"
        r"analiz\w*|revis\w*|inspeccion\w*|problemas?|fallas?|missing|faltan?|"
        r"duplic\w*|filas?|columnas?|rows?|columns?)\b",
        folded,
    )
    return bool(topic and contextual)


@dataclass(frozen=True, slots=True)
class _CsvProfile:
    insight: ResultInsight
    descriptors: tuple[str, ...] = ()
    smiles: tuple[str, ...] = ()
    inspected: bool = False


def _is_missing(value: str) -> bool:
    return value.strip().casefold() in _MISSING


def _is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _column_index(header: list[str], wanted: str) -> int | None:
    normalized = wanted.strip().casefold()
    return next((index for index, name in enumerate(header) if name.strip().casefold() == normalized), None)


def _profile_csv(
    path: Path,
    role: str,
    target_column: str,
    name_column: str,
    prediction_type: str,
    ignored_columns: set[str],
) -> _CsvProfile:
    source = path.name
    if not path.is_file():
        return _CsvProfile(ResultInsight("", "", "input_csv", f"{role} CSV is not readable; no rows inspected", source))
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(handle, dialect=dialect, strict=True)
            header = [item.strip() for item in next(reader, [])]
            if not header:
                return _CsvProfile(ResultInsight("", "", "input_csv", f"{role} CSV is empty; no columns inspected", source))
            scanned_header = header[:_MAX_SCAN_COLUMNS]
            target_index = _column_index(scanned_header, target_column) if target_column else None
            smiles_index = _column_index(scanned_header, "SMILES")
            identifier = name_column or next(
                (name for name in ("code_name", "name", "id") if _column_index(scanned_header, name) is not None),
                "",
            )
            identifier_index = _column_index(scanned_header, identifier) if identifier else None
            excluded = {"smiles", "code_name", "name", "id", target_column.strip().casefold(), *ignored_columns}
            descriptor_indexes = tuple(
                index for index, name in enumerate(scanned_header)
                if name and name.casefold() not in excluded
            )
            missing_counts = [0] * len(scanned_header)
            nonnumeric_counts = [0] * len(scanned_header)
            seen_identifiers: set[str] = set()
            seen_smiles: set[str] = set()
            smiles_values: list[str] = []
            repeated_identifiers = repeated_smiles = malformed_rows = rows = 0
            truncated = False
            for row in reader:
                if rows >= _MAX_SCAN_ROWS:
                    truncated = True
                    break
                rows += 1
                if len(row) != len(header):
                    malformed_rows += 1
                for index in range(len(scanned_header)):
                    value = row[index].strip() if index < len(row) else ""
                    if _is_missing(value):
                        missing_counts[index] += 1
                    elif index == target_index or index in descriptor_indexes:
                        if not _is_finite_number(value):
                            nonnumeric_counts[index] += 1
                if identifier_index is not None and identifier_index < len(row):
                    value = row[identifier_index].strip()
                    if value and value in seen_identifiers:
                        repeated_identifiers += 1
                    seen_identifiers.add(value)
                if smiles_index is not None and smiles_index < len(row):
                    value = row[smiles_index].strip()
                    if value:
                        repeated_smiles += value in seen_smiles
                        seen_smiles.add(value)
                        smiles_values.append(value)
    except (OSError, csv.Error, UnicodeError) as exc:
        message = f"{role} CSV could not be inspected ({type(exc).__name__}); no complete profile available"
        return _CsvProfile(ResultInsight("", "", "input_csv", message, source))

    header_counts: dict[str, int] = {}
    for name in header:
        key = name.casefold()
        header_counts[key] = header_counts.get(key, 0) + 1
    findings = [f"{role} CSV: {rows} rows scanned, {len(header)} columns"]
    if len(header) > _MAX_SCAN_COLUMNS:
        findings.append(f"column scan limited to first {_MAX_SCAN_COLUMNS} columns")
    if truncated:
        findings.append(f"scan limited to first {_MAX_SCAN_ROWS} rows")
    if malformed_rows:
        suffix = "row" if malformed_rows == 1 else "rows"
        findings.append(f"{malformed_rows} {suffix} with a different column count")
    duplicate_headers = [name for name in header if header_counts[name.casefold()] > 1]
    if duplicate_headers:
        findings.append(f"duplicate header: {duplicate_headers[0]}")
    if target_column and target_index is None:
        findings.append(
            "observed target column absent; no error metric can be computed"
            if role == "External" else f"target column {target_column} missing"
        )
    elif target_index is not None:
        count = missing_counts[target_index]
        if count:
            findings.append(f"{count} missing target value{'s' if count != 1 else ''}")
        count = nonnumeric_counts[target_index]
        if count and prediction_type.casefold().startswith("reg"):
            findings.append(f"{count} nonnumeric target value{'s' if count != 1 else ''}")
    for index in descriptor_indexes:
        count = nonnumeric_counts[index]
        if count:
            findings.append(
                f"{count} nonnumeric value{'s' if count != 1 else ''} in descriptor {scanned_header[index][:80]}"
            )
    if repeated_smiles:
        findings.append(f"{repeated_smiles} repeated exact SMILES")
    if repeated_identifiers:
        findings.append(f"{repeated_identifiers} repeated {identifier}")
    missing_descriptors = sorted(
        ((scanned_header[index], missing_counts[index]) for index in descriptor_indexes if missing_counts[index]),
        key=lambda item: (-item[1], item[0].casefold()),
    )
    findings.extend(
        f"{name[:80]}: {count}/{rows} missing" for name, count in missing_descriptors[:3]
    )
    if len(missing_descriptors) > 3:
        findings.append(f"{len(missing_descriptors) - 3} more descriptors with missing values")
    findings.append("structural checks only; chemical validity was not assessed")
    insight = ResultInsight("", "", "input_csv", "; ".join(findings), source)
    return _CsvProfile(insight, tuple(scanned_header[index] for index in descriptor_indexes), tuple(smiles_values), True)


def inspect_selected_csvs(
    main_path: str,
    external_path: str = "",
    *,
    target_column: str = "",
    name_column: str = "",
    prediction_type: str = "",
    ignored_columns: Iterable[str] = (),
) -> tuple[ResultInsight, ...]:
    """Inspect only selected CSV paths and return bounded, source-linked findings."""
    ignored = {str(name).strip().casefold() for name in ignored_columns}
    main = _profile_csv(Path(main_path), "Main", target_column, name_column, prediction_type, ignored) if main_path else None
    external = _profile_csv(Path(external_path), "External", target_column, name_column, prediction_type, ignored) if external_path else None
    profiles = tuple(profile for profile in (main, external) if profile is not None)
    insights = [profile.insight for profile in profiles]
    if main is not None and external is not None and main.inspected and external.inspected:
        external_headers = set(external.descriptors)
        absent = [name for name in main.descriptors if name not in external_headers]
        if absent:
            insight = external.insight
            detail = "; ".join(f"missing descriptor column: {name[:80]}" for name in absent[:3])
            if len(absent) > 3:
                detail += f"; {len(absent) - 3} more required descriptor columns absent"
            insights[1] = ResultInsight("", "", "input_csv", f"{insight.summary}; {detail}", insight.source)
        if main.smiles and external.smiles:
            main_smiles = set(main.smiles)
            overlap = sum(value in main_smiles for value in external.smiles)
            summary = (
                f"Exact SMILES overlap in scanned rows: {overlap}/{len(external.smiles)} external SMILES "
                "also occur in the main CSV; SMILES were not chemically canonicalized"
            )
            insights.append(ResultInsight("", "", "dataset_overlap", summary, f"{main.insight.source}, {external.insight.source}"))
    return tuple(insights)
