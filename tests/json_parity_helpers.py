import json
import os
import re
from io import StringIO
from typing import Dict, List

import pandas as pd
from sklearn.impute import KNNImputer


def first_int_after_dash(line: str) -> int:
    match = re.search(r"-\s+(\d+)", line)
    if not match:
        raise AssertionError(f"Could not parse integer from line: {line!r}")
    return int(match.group(1))


def find_line_index(lines: List[str], contains: str) -> int:
    for i, line in enumerate(lines):
        if contains in line:
            return i
    raise AssertionError(f"Could not find line containing: {contains}")


def parse_generate_load_counts_from_dat(dat_lines: List[str]) -> Dict[str, int]:
    anchor = find_line_index(dat_lines, "loaded successfully, including:")
    return {
        "datapoints_loaded": first_int_after_dash(dat_lines[anchor + 1]),
        "accepted_descriptors_loaded": first_int_after_dash(dat_lines[anchor + 2]),
        "ignored_descriptors_loaded": first_int_after_dash(dat_lines[anchor + 3]),
        "discarded_descriptors_loaded": first_int_after_dash(dat_lines[anchor + 4]),
    }


def parse_generate_model_scan_summary_from_dat(dat_lines: List[str]) -> Dict[str, object]:
    """
    Parse event-backed GENERATE model-scan summary lines from DAT text.

    Captures the first model-cycle line plus no-PFI and PFI combined metrics.
    """

    model_line = None
    for line in dat_lines:
        if "- ML model:" in line:
            model_line = line
            break
    if model_line is None:
        raise AssertionError("Could not find model cycle line in GENERATE DAT output")

    model_match = re.search(r"-\s*(\d+)\/(\d+)\s*-\s*ML model:\s*([^\s]+)", model_line)
    if not model_match:
        raise AssertionError(f"Could not parse model cycle line: {model_line!r}")

    no_pfi_line = None
    for line in dat_lines:
        if "Best combined" in line and "(no PFI filter)" in line:
            no_pfi_line = line
            break
    if no_pfi_line is None:
        raise AssertionError("Could not find no-PFI combined metric line in GENERATE DAT output")

    no_pfi_match = re.search(
        r"Best combined\s+([A-Za-z0-9_]+)\s+\(target\) found in BO for\s+([^\s]+)\s+\(no PFI filter\):\s*(-?\d+(?:\.\d+)?)",
        no_pfi_line,
    )
    if not no_pfi_match:
        raise AssertionError(f"Could not parse no-PFI combined metric line: {no_pfi_line!r}")

    pfi_line = None
    for line in dat_lines:
        if "Combined" in line and "(with PFI filter)" in line:
            pfi_line = line
            break
    if pfi_line is None:
        raise AssertionError("Could not find PFI combined metric line in GENERATE DAT output")

    pfi_match = re.search(
        r"Combined\s+([A-Za-z0-9_]+)\s+for\s+([^\s]+)\s+\(with PFI filter\):\s*(-?\d+(?:\.\d+)?)",
        pfi_line,
    )
    if not pfi_match:
        raise AssertionError(f"Could not parse PFI combined metric line: {pfi_line!r}")

    return {
        "cycle_number": int(model_match.group(1)),
        "cycle_total": int(model_match.group(2)),
        "model_name": model_match.group(3),
        "no_pfi_metric_label": no_pfi_match.group(1).lower(),
        "no_pfi_model_name": no_pfi_match.group(2),
        "no_pfi_combined_metric": float(no_pfi_match.group(3)),
        "pfi_metric_label": pfi_match.group(1).lower(),
        "pfi_model_name": pfi_match.group(2),
        "pfi_combined_metric": float(pfi_match.group(3)),
    }


def parse_labeled_counts_from_dat(
    dat_lines: List[str],
    anchor_text: str,
    label_to_json_key: Dict[str, str],
) -> Dict[str, int]:
    """
    Parse a contiguous "including:"-style block in a DAT file.

    The parser searches lines after the anchor for labels such as "datapoints"
    and maps each extracted count to a JSON key via label_to_json_key.
    """

    anchor = find_line_index(dat_lines, anchor_text)
    parsed: Dict[str, int] = {}

    for line in dat_lines[anchor + 1 :]:
        if not line.strip():
            break
        if "-" not in line:
            break
        for label, json_key in label_to_json_key.items():
            if label in line:
                parsed[json_key] = first_int_after_dash(line)

    missing = [json_key for json_key in label_to_json_key.values() if json_key not in parsed]
    if missing:
        raise AssertionError(
            f"Could not parse expected DAT labels after anchor {anchor_text!r}. Missing keys: {missing}"
        )

    return parsed


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise AssertionError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_database_events(audit: dict) -> List[dict]:
    return [
        event
        for event in audit.get("events", [])
        if event.get("event_type") == "load_database"
    ]


def assert_section_keys_present(audit: dict, keys: List[str]) -> None:
    section = audit.get("sections", {}).get("load_database", {})
    for key in keys:
        if key not in section:
            raise AssertionError(f"Missing load_database section key: {key}")


def any_event_matches(events: List[dict], expected: Dict[str, int], keys: List[str]) -> bool:
    for event in events:
        payload = event.get("payload", {})
        if all(payload.get(key) == expected[key] for key in keys):
            return True
    return False


def assert_event_payload_parity(
    audit: dict,
    event_type: str,
    expected: Dict[str, int],
    event_keys: List[str],
) -> None:
    events = [event for event in audit.get("events", []) if event.get("event_type") == event_type]
    if not events:
        raise AssertionError(f"No events found with type: {event_type}")
    if not any_event_matches(events, expected, event_keys):
        raise AssertionError(
            f"No {event_type} event matched expected payload for keys {event_keys}. Expected: {expected}"
        )


def parse_verify_summary_metrics_from_dat(dat_lines: List[str]) -> Dict[str, float]:
    """
    Parse the first VERIFY summary block metrics from DAT text.
    """

    marker = "Results of flawed models and sorted cross-validation:"
    start = find_line_index(dat_lines, marker)

    original_line = dat_lines[start + 1]
    y_mean_line = dat_lines[start + 2]
    y_shuffle_line = dat_lines[start + 3]
    onehot_line = dat_lines[start + 4]

    original_match = re.search(r"\)\s+(-?\d+(?:\.\d+)?)", original_line)
    if not original_match:
        raise AssertionError(f"Could not parse original CV metric from line: {original_line!r}")

    def _parse_test_metric(line: str) -> float:
        match = re.search(r"=\s*(-?\d+(?:\.\d+)?)", line)
        if not match:
            raise AssertionError(f"Could not parse VERIFY test metric from line: {line!r}")
        return float(match.group(1))

    return {
        "original_cv_metric": float(original_match.group(1)),
        "y_mean_result": _parse_test_metric(y_mean_line),
        "y_shuffle_result": _parse_test_metric(y_shuffle_line),
        "onehot_result": _parse_test_metric(onehot_line),
    }


def assert_event_payload_parity_rounded(
    audit: dict,
    event_type: str,
    expected: Dict[str, float],
    event_keys: List[str],
    ndigits: int = 2,
) -> None:
    """
    Assert parity by comparing rounded numeric values in event payloads.
    """

    events = [event for event in audit.get("events", []) if event.get("event_type") == event_type]
    if not events:
        raise AssertionError(f"No events found with type: {event_type}")

    for event in events:
        payload = event.get("payload", {})
        matched = True
        for key in event_keys:
            if key not in payload:
                matched = False
                break
            if round(float(payload[key]), ndigits) != round(float(expected[key]), ndigits):
                matched = False
                break
        if matched:
            return

    raise AssertionError(
        f"No {event_type} event matched rounded expected payload for keys {event_keys}. Expected: {expected}"
    )


def parse_predict_external_load_count_from_dat(dat_lines: List[str]) -> Dict[str, int]:
    """
    Parse the external test-set load count from PREDICT DAT text.
    """

    anchor = find_line_index(dat_lines, "External set")
    datapoints_line = dat_lines[anchor + 1]
    return {"datapoints_loaded": first_int_after_dash(datapoints_line)}


def parse_curate_correlation_filter_summary_from_dat(dat_lines: List[str]) -> Dict[str, object]:
    """
    Parse correlation-filter summary counts from CURATE DAT text.

    Returns values aligned with the correlation_filter event payload keys.
    """

    _ = find_line_index(dat_lines, "Correlation filter activated with these thresholds:")

    constant_removed = sum(1 for line in dat_lines if "all the values are the same" in line)
    low_y_corr_removed = sum(
        1
        for line in dat_lines
        if re.search(r"R\*\*2\s*=\s*-?\d+(?:\.\d+)?\s+with the\s+.+\s+values", line)
    )

    high_intercorr_removed = 0
    for line in dat_lines:
        if "descriptors removed due to high correlation with other descriptors" in line:
            match = re.search(r"Total:\s*(\d+)", line)
            if not match:
                raise AssertionError(f"Could not parse high-intercorrelation count from line: {line!r}")
            high_intercorr_removed = int(match.group(1))
            break

    rfecv_true = any("Recursive Feature Elimination with Cross-Validation (RFECV)" in line for line in dat_lines)
    rfecv_false = any("RFECV filter was not applied" in line for line in dat_lines)
    if rfecv_true and rfecv_false:
        raise AssertionError("Could not parse RFECV status unambiguously from CURATE DAT output")
    if not rfecv_true and not rfecv_false:
        raise AssertionError("Could not find RFECV status marker in CURATE DAT output")

    return {
        "constant_removed": int(constant_removed),
        "low_y_corr_removed": int(low_y_corr_removed),
        "high_intercorr_removed": int(high_intercorr_removed),
        "rfecv_applied": bool(rfecv_true),
    }


def parse_curate_categorical_transform_summary_from_dat(dat_lines: List[str]) -> Dict[str, object]:
    """
    Parse categorical-transform summary from CURATE DAT text.

    Returns values aligned with the categorical_transform event payload keys.
    """

    find_line_index(dat_lines, "Analyzing categorical variables")

    count_line = None
    for line in dat_lines:
        if "categorical variables were converted using the" in line and "mode in the categorical option" in line:
            count_line = line
            break

    if count_line is None:
        if any("No categorical variables were found" in line for line in dat_lines):
            return {
                "categorical_variables_count": 0,
                "generated_descriptors_count": 0,
                "mode": None,
            }
        raise AssertionError("Could not parse categorical-transform summary line from CURATE DAT output")

    count_match = re.search(r"A total of\s+(\d+)\s+categorical variables were converted", count_line)
    mode_match = re.search(r"using the\s+([^\s]+)\s+mode", count_line)
    if not count_match or not mode_match:
        raise AssertionError(f"Could not parse categorical-transform count/mode from line: {count_line!r}")

    categorical_variables_count = int(count_match.group(1))
    mode = mode_match.group(1)

    generated_descriptors_count = 0
    for i, line in enumerate(dat_lines):
        if "Generated descriptors:" in line:
            for desc_line in dat_lines[i + 1 :]:
                stripped = desc_line.strip()
                if not stripped:
                    break
                if stripped.startswith("o"):
                    break
                if stripped.startswith("-"):
                    generated_descriptors_count += 1
            break

    return {
        "categorical_variables_count": categorical_variables_count,
        "generated_descriptors_count": int(generated_descriptors_count),
        "mode": mode,
    }


def recompute_load_database_oracle(
    csv_load: str,
    y_col: str,
    ignore: List[str],
    discard: List[str],
    auto_fill: bool = False,
) -> Dict[str, int]:
    """
    Recompute load_database summary values independently from ROBERT runtime output.

    This mirrors the data-shape logic with standalone test code so DAT and JSON can
    each be validated against a third oracle.
    """

    with open(csv_load, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    # Keep parity with load_database CSV normalization behavior when semicolons are used.
    csv_text = "".join(lines)
    if len(lines) > 1 and lines[1].count(";") > 1:
        csv_text = csv_text.replace(",", ".").replace(";", ",")

    csv_df = pd.read_csv(StringIO(csv_text), encoding="utf-8")

    cols_to_drop: List[str] = []
    n_removed_rows = 0
    cols_with_missing: List[str] = []
    knn_applied = False

    descriptor_cols = [
        col
        for col in csv_df.columns
        if col not in ignore + discard and col != y_col
    ]
    min_count = int(0.9 * len(csv_df))

    cols_to_drop = [col for col in descriptor_cols if csv_df[col].notna().sum() < min_count]
    if cols_to_drop:
        csv_df = csv_df.drop(columns=cols_to_drop)
        descriptor_cols = [
            col
            for col in csv_df.columns
            if col not in ignore + discard and col != y_col
        ]

    rows_too_missing = csv_df[descriptor_cols].isna().sum(axis=1) > (0.5 * len(descriptor_cols))
    if rows_too_missing.any():
        n_removed_rows = int(rows_too_missing.sum())
        csv_df = csv_df[~rows_too_missing].reset_index(drop=True)
        descriptor_cols = [
            col
            for col in csv_df.columns
            if col not in ignore + discard and col != y_col
        ]

    if auto_fill:
        numeric_columns = csv_df.select_dtypes(include=["float"]).columns.drop(y_col, errors="ignore")
        if len(numeric_columns) > 0 and csv_df[numeric_columns].isna().any().any():
            imputer = KNNImputer(n_neighbors=5)
            csv_df[numeric_columns] = pd.DataFrame(
                imputer.fit_transform(csv_df[numeric_columns]),
                columns=numeric_columns,
                index=csv_df.index,
            )
            knn_applied = True
    else:
        cols_with_missing = [
            col
            for col in descriptor_cols
            if csv_df[col].isna().any() and col not in ignore + discard
        ]
        if cols_with_missing:
            csv_df = csv_df.drop(columns=cols_with_missing)

    discard_cols = [col for col in discard if col in csv_df.columns]
    if discard_cols:
        csv_df = csv_df.drop(discard_cols, axis=1)

    total_amount = len(csv_df.columns)
    ignored_descs = len(ignore)
    accepted_descs = total_amount - ignored_descs - 1
    if "Set" in csv_df.columns:
        accepted_descs -= 1
        ignored_descs += 1

    return {
        "datapoints_loaded": int(len(csv_df)),
        "accepted_descriptors_loaded": int(accepted_descs),
        "ignored_descriptors_loaded": int(ignored_descs),
        "discarded_descriptors_loaded": int(len(discard)),
        "columns_removed_lt90pct_data": int(len(cols_to_drop)),
        "rows_removed_gt50pct_missing": int(n_removed_rows),
        "columns_removed_any_missing": int(len(cols_with_missing)),
        "knn_imputer_applied": bool(knn_applied),
    }


def parse_predict_summary_metrics_from_dat(dat_lines: List[str]) -> Dict[str, float]:
    """
    Parse the first PREDICT summary block metrics from DAT text (regression/classification).
    """

    anchor = find_line_index(dat_lines, "Summary of results")

    metrics: Dict[str, float] = {}
    for line in dat_lines[anchor + 1 : anchor + 12]:
        if "R2 =" in line and "CV" in line:
            cv_match = re.search(r"R2\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*MAE\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*RMSE\s*=\s*(-?\d+(?:\.\d+)?)", line)
            if cv_match:
                metrics["cv_r2"] = float(cv_match.group(1))
                metrics["cv_mae"] = float(cv_match.group(2))
                metrics["cv_rmse"] = float(cv_match.group(3))
        elif "R2 =" in line and "Test" in line:
            test_match = re.search(r"R2\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*MAE\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*RMSE\s*=\s*(-?\d+(?:\.\d+)?)", line)
            if test_match:
                metrics["test_r2"] = float(test_match.group(1))
                metrics["test_mae"] = float(test_match.group(2))
                metrics["test_rmse"] = float(test_match.group(3))
        elif "Accur." in line and "CV" in line:
            cv_match = re.search(r"Accur\.\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*F1 score\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*MCC\s*=\s*(-?\d+(?:\.\d+)?)", line)
            if cv_match:
                metrics["cv_acc"] = float(cv_match.group(1))
                metrics["cv_f1"] = float(cv_match.group(2))
                metrics["cv_mcc"] = float(cv_match.group(3))
        elif "Accur." in line and "Test" in line:
            test_match = re.search(r"Accur\.\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*F1 score\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*MCC\s*=\s*(-?\d+(?:\.\d+)?)", line)
            if test_match:
                metrics["test_acc"] = float(test_match.group(1))
                metrics["test_f1"] = float(test_match.group(2))
                metrics["test_mcc"] = float(test_match.group(3))

    if not metrics:
        raise AssertionError("Could not parse any PREDICT summary metrics from DAT output")

    return metrics


def assert_predict_summary_event_parity(audit: dict, expected_summary: Dict[str, float], ndigits: int = 2) -> None:
    """
    Compare PREDICT DAT summary metrics against print_predict_summary event payload.
    """

    events = [event for event in audit.get("events", []) if event.get("event_type") == "print_predict_summary"]
    if not events:
        raise AssertionError("No events found with type: print_predict_summary")

    for event in events:
        payload = event.get("payload", {})
        cv_metrics = payload.get("cv_metrics", {})
        test_metrics = payload.get("test_metrics", {})

        matches = True
        for key, expected_value in expected_summary.items():
            if key.startswith("cv_"):
                metric_key = key.replace("cv_", "", 1)
                current = cv_metrics.get(metric_key)
            elif key.startswith("test_"):
                metric_key = key.replace("test_", "", 1)
                current = test_metrics.get(metric_key)
            else:
                matches = False
                break

            if current is None:
                matches = False
                break

            if round(float(current), ndigits) != round(float(expected_value), ndigits):
                matches = False
                break

        if matches:
            return

    raise AssertionError(
        f"No print_predict_summary event matched expected DAT summary metrics: {expected_summary}"
    )
