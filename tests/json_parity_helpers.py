import ast
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


def parse_verify_summary_metrics_from_dat(dat_lines: List[str]) -> Dict[str, object]:
    """
    Parse the first VERIFY summary block metrics from DAT text.
    """

    marker = "Results of flawed models and sorted cross-validation:"
    start = find_line_index(dat_lines, marker)

    original_line = dat_lines[start + 1]
    y_mean_line = dat_lines[start + 2]
    y_shuffle_line = dat_lines[start + 3]
    onehot_line = dat_lines[start + 4]
    sorted_line = dat_lines[start + 5]

    original_match = re.search(
        r"Original\s+(\w+)\s+\(([^)]+)\)\s+(-?\d+(?:\.\d+)?)\s+([+-])\s+(\d+)%\s+&\s+(\d+)%\s+threshold\s+=\s+(-?\d+(?:\.\d+)?)\s+&\s+(-?\d+(?:\.\d+)?)",
        original_line,
    )
    if not original_match:
        raise AssertionError(f"Could not parse original CV metric from line: {original_line!r}")

    def _parse_test_metric(line: str) -> float:
        match = re.search(r"=\s*(-?\d+(?:\.\d+)?)", line)
        if not match:
            raise AssertionError(f"Could not parse VERIFY test metric from line: {line!r}")
        return float(match.group(1))

    def _parse_test_status(line: str) -> str:
        match = re.search(r"\b(PASSED|UNCLEAR|FAILED)\b", line)
        if not match:
            raise AssertionError(f"Could not parse VERIFY test status from line: {line!r}")
        return match.group(1)

    sorted_metrics: Dict[str, List[float]]
    if "R2 =" in sorted_line:
        sorted_metrics = {
            "r2": ast.literal_eval(sorted_line.split("R2 = ", 1)[1].split(", MAE = ", 1)[0]),
            "mae": ast.literal_eval(sorted_line.split("MAE = ", 1)[1].split(", RMSE = ", 1)[0]),
            "rmse": ast.literal_eval(sorted_line.split("RMSE = ", 1)[1]),
        }
    else:
        sorted_metrics = {
            "acc": ast.literal_eval(sorted_line.split("Accuracy = ", 1)[1].split(", F1 score = ", 1)[0]),
            "f1": ast.literal_eval(sorted_line.split("F1 score = ", 1)[1].split(", MCC = ", 1)[0]),
            "mcc": ast.literal_eval(sorted_line.split("MCC = ", 1)[1]),
        }

    return {
        "error_type": original_match.group(1).lower(),
        "cv_type": original_match.group(2),
        "original_cv_metric": float(original_match.group(3)),
        "threshold_direction": "higher" if original_match.group(4) == "+" else "lower",
        "unclear_threshold_percent": int(original_match.group(5)),
        "pass_threshold_percent": int(original_match.group(6)),
        "unclear_threshold": float(original_match.group(7)),
        "pass_threshold": float(original_match.group(8)),
        "y_mean_result": _parse_test_metric(y_mean_line),
        "y_shuffle_result": _parse_test_metric(y_shuffle_line),
        "onehot_result": _parse_test_metric(onehot_line),
        "test_status": {
            "y_mean": _parse_test_status(y_mean_line),
            "y_shuffle": _parse_test_status(y_shuffle_line),
            "onehot": _parse_test_status(onehot_line),
        },
        "sorted_metrics": sorted_metrics,
    }


def parse_verify_branch_titles_from_dat(dat_lines: List[str]) -> List[str]:
    """Parse the VERIFY branch markers written before each model analysis block."""

    branch_titles = []
    for line in dat_lines:
        if "Starting model with all variables (No PFI)" in line:
            branch_titles.append("No_PFI")
        elif "Starting model with PFI filter" in line:
            branch_titles.append("PFI")
    if not branch_titles:
        raise AssertionError("Could not find VERIFY branch markers in DAT output")
    return branch_titles


def parse_verify_model_context_from_dat(dat_lines: List[str]) -> List[Dict[str, object]]:
    """Parse model and dataset context blocks written by VERIFY load_print."""

    contexts = []
    for index, line in enumerate(dat_lines):
        match = re.search(r"ML model\s+([^\s]+)\s+(.+) and Xy database were loaded", line)
        if not match:
            continue
        context = {
            "model_name": match.group(1),
            "suffix": match.group(2),
        }
        for context_line in dat_lines[index + 1 : index + 9]:
            label, separator, value = context_line.partition(":")
            if not separator:
                continue
            key = label.strip().lstrip("-").strip()
            value = value.strip()
            if key == "Target value":
                context["y_column"] = value
            elif key == "Names":
                context["names_column"] = value
            elif key == "Model":
                context["model_name"] = value
            elif key == "k-fold CV":
                context["kfold"] = int(value)
            elif key == "Repetitions CV":
                context["repeat_kfolds"] = int(value)
            elif key == "Descriptors":
                context["descriptor_list"] = ast.literal_eval(value)
            elif key == "Training points":
                context["train_datapoints"] = int(value)
            elif key == "Test points":
                context["test_datapoints"] = int(value)
        contexts.append(context)
    if not contexts:
        raise AssertionError("Could not find VERIFY model-context blocks in DAT output")
    return contexts


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
                "categorical_variables": [],
                "generated_descriptors": [],
                "categorical_variables_found": False,
                "mode": None,
            }
        raise AssertionError("Could not parse categorical-transform summary line from CURATE DAT output")

    count_match = re.search(r"A total of\s+(\d+)\s+categorical variables were converted", count_line)
    mode_match = re.search(r"using the\s+([^\s]+)\s+mode", count_line)
    if not count_match or not mode_match:
        raise AssertionError(f"Could not parse categorical-transform count/mode from line: {count_line!r}")

    categorical_variables_count = int(count_match.group(1))
    mode = mode_match.group(1)

    categorical_variables: List[str] = []
    generated_descriptors: List[str] = []

    count_index = dat_lines.index(count_line)
    section = None
    for line in dat_lines[count_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if mode.lower() == "numbers" and section == "categorical":
                break
            continue

        if stripped.startswith("o"):
            break

        if "Initial descriptors:" in line:
            section = "categorical"
            continue

        if "Generated descriptors:" in line:
            section = "generated"
            continue

        if stripped.startswith("-"):
            descriptor_name = stripped.lstrip("-").strip()
            if mode.lower() == "numbers" and section is None:
                categorical_variables.append(descriptor_name)
                continue
            if section == "categorical":
                categorical_variables.append(descriptor_name)
            elif section == "generated":
                generated_descriptors.append(descriptor_name)

    generated_descriptors_count = len(generated_descriptors)

    return {
        "categorical_variables_count": categorical_variables_count,
        "generated_descriptors_count": int(generated_descriptors_count),
        "categorical_variables": categorical_variables,
        "generated_descriptors": generated_descriptors,
        "categorical_variables_found": categorical_variables_count > 0,
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
        if "Points CV (train+valid.):Test =" in line:
            points = re.search(r"=\s*(\d+):(\d+)", line)
            if points:
                metrics["train_points"] = int(points.group(1))
                metrics["test_points"] = int(points.group(2))
        elif "Proportion CV (train+valid.):Test =" in line:
            proportions = re.search(r"=\s*(\d+):(\d+)", line)
            if proportions:
                metrics["train_proportion_percent"] = int(proportions.group(1))
                metrics["test_proportion_percent"] = int(proportions.group(2))
        elif "Number of descriptors =" in line:
            descriptor_count = re.search(r"=\s*(\d+)", line)
            if descriptor_count:
                metrics["descriptor_count"] = int(descriptor_count.group(1))
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
        point_counts = payload.get("point_counts", {})
        proportions = payload.get("train_test_proportion_percent", {})

        matches = True
        for key, expected_value in expected_summary.items():
            if key == "train_points":
                current = point_counts.get("train")
            elif key == "test_points":
                current = point_counts.get("test")
            elif key == "train_proportion_percent":
                current = proportions.get("train")
            elif key == "test_proportion_percent":
                current = proportions.get("test")
            elif key == "descriptor_count":
                current = payload.get("descriptor_count")
            elif key.startswith("cv_"):
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
