import glob
import os
import shutil

from robert.curate import curate
from robert.generate import generate
from robert.predict import predict
from robert.verify import verify

from tests.json_parity_helpers import (
    assert_predict_summary_event_parity,
    assert_event_payload_parity_rounded,
    assert_event_payload_parity,
    assert_section_keys_present,
    load_json,
    parse_generate_model_scan_summary_from_dat,
    parse_curate_categorical_transform_summary_from_dat,
    parse_curate_correlation_filter_summary_from_dat,
    parse_labeled_counts_from_dat,
    parse_predict_external_load_count_from_dat,
    parse_predict_summary_metrics_from_dat,
    parse_verify_branch_titles_from_dat,
    parse_verify_model_context_from_dat,
    recompute_load_database_oracle,
    recompute_categorical_transform_oracle,
    recompute_correlation_filter_oracle,
    parse_verify_summary_metrics_from_dat,
)


path_main = os.getcwd()
path_curate = os.path.join(path_main, "CURATE")
path_generate = os.path.join(path_main, "GENERATE")
path_predict = os.path.join(path_main, "PREDICT")
path_verify = os.path.join(path_main, "VERIFY")


def _clean_module_outputs() -> None:
    for folder in [path_curate, path_generate, path_predict, path_verify]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    for dat_file in glob.glob("*.dat"):
        if "CURATE" in dat_file or "GENERATE" in dat_file or "PREDICT" in dat_file or "VERIFY" in dat_file:
            os.remove(dat_file)


def test_generate_dat_json_parity_via_new_file_only():
    _clean_module_outputs()

    curate(
        csv_name=os.path.join("tests", "Robert_example.csv"),
        y="Target_values",
        names="Name",
        discard=["xtest"],
    )

    generate(
        generate=True,
        csv_name=os.path.join("CURATE", "Robert_example_CURATE.csv"),
        y="Target_values",
        model=["RF"],
        init_points=1,
        n_iter=1,
    )

    dat_path = os.path.join(path_generate, "GENERATE_data.dat")
    assert os.path.exists(dat_path)
    with open(dat_path, "r", encoding="utf-8") as handle:
        dat_lines = handle.readlines()

    expected = parse_labeled_counts_from_dat(
        dat_lines,
        "loaded successfully, including:",
        {
            "datapoints": "datapoints_loaded",
            "accepted descriptors": "accepted_descriptors_loaded",
            "ignored descriptors": "ignored_descriptors_loaded",
            "discarded descriptors": "discarded_descriptors_loaded",
        },
    )

    audit_path = os.path.join(path_main, "JSON", "generate_audit.json")
    audit = load_json(audit_path)

    assert_section_keys_present(
        audit,
        [
            "datapoints_loaded",
            "accepted_descriptors_loaded",
            "ignored_descriptors_loaded",
            "discarded_descriptors_loaded",
        ],
    )

    assert_event_payload_parity(
        audit,
        "load_database",
        expected,
        ["datapoints_loaded", "accepted_descriptors_loaded"],
    )

    section = audit["sections"]["load_database"]
    assert section["ignored_descriptors_loaded"]["value"] == expected["ignored_descriptors_loaded"]
    assert section["discarded_descriptors_loaded"]["value"] == expected["discarded_descriptors_loaded"]


def test_generate_model_scan_summary_dat_json_parity_via_new_file_only():
    _clean_module_outputs()

    curate(
        csv_name=os.path.join("tests", "Robert_example.csv"),
        y="Target_values",
        names="Name",
        discard=["xtest"],
    )

    generate(
        generate=True,
        csv_name=os.path.join("CURATE", "Robert_example_CURATE.csv"),
        y="Target_values",
        model=["RF"],
        init_points=1,
        n_iter=1,
    )

    dat_path = os.path.join(path_generate, "GENERATE_data.dat")
    assert os.path.exists(dat_path)
    with open(dat_path, "r", encoding="utf-8") as handle:
        dat_lines = handle.readlines()

    expected = parse_generate_model_scan_summary_from_dat(dat_lines)

    audit_path = os.path.join(path_main, "JSON", "generate_audit.json")
    audit = load_json(audit_path)

    model_start_events = [e for e in audit.get("events", []) if e.get("event_type") == "model_run_start"]
    assert model_start_events, "No model_run_start event found"
    assert any(
        e.get("payload", {}).get("cycle_number") == expected["cycle_number"]
        and e.get("payload", {}).get("cycle_total") == expected["cycle_total"]
        and e.get("payload", {}).get("model_name") == expected["model_name"]
        for e in model_start_events
    ), "No model_run_start event matched DAT model cycle summary"

    bo_events = [e for e in audit.get("events", []) if e.get("event_type") == "bo_workflow"]
    assert bo_events, "No bo_workflow event found"
    assert any(
        e.get("payload", {}).get("model") == expected["no_pfi_model_name"]
        and str(e.get("payload", {}).get("error_type", "")).lower() == expected["no_pfi_metric_label"]
        and round(float(e.get("payload", {}).get("combined_metric_value")), 2)
        == round(float(expected["no_pfi_combined_metric"]), 2)
        for e in bo_events
    ), "No bo_workflow event matched DAT no-PFI combined metric summary"

    pfi_events = [e for e in audit.get("events", []) if e.get("event_type") == "pfi_workflow"]
    assert pfi_events, "No pfi_workflow event found"
    assert any(
        e.get("payload", {}).get("model") == expected["pfi_model_name"]
        and str(e.get("payload", {}).get("error_type", "")).lower() == expected["pfi_metric_label"]
        and round(float(e.get("payload", {}).get("combined_metric_after_pfi")), 2)
        == round(float(expected["pfi_combined_metric"]), 2)
        for e in pfi_events
    ), "No pfi_workflow event matched DAT PFI combined metric summary"


def test_curate_load_database_third_oracle_parity_via_new_file_only():
    _clean_module_outputs()

    csv_input = os.path.join("tests", "Robert_example.csv")
    y_col = "Target_values"
    names_col = "Name"
    discard_cols = ["xtest"]

    curate(
        csv_name=csv_input,
        y=y_col,
        names=names_col,
        discard=discard_cols,
    )

    dat_path = os.path.join(path_curate, "CURATE_data.dat")
    assert os.path.exists(dat_path)
    with open(dat_path, "r", encoding="utf-8") as handle:
        dat_lines = handle.readlines()

    dat_counts = parse_labeled_counts_from_dat(
        dat_lines,
        "loaded successfully, including:",
        {
            "datapoints": "datapoints_loaded",
            "accepted descriptors": "accepted_descriptors_loaded",
            "ignored descriptors": "ignored_descriptors_loaded",
            "discarded descriptors": "discarded_descriptors_loaded",
        },
    )

    oracle = recompute_load_database_oracle(
        csv_load=csv_input,
        y_col=y_col,
        ignore=[names_col],
        discard=discard_cols,
        auto_fill=False,
    )

    assert dat_counts["datapoints_loaded"] == oracle["datapoints_loaded"]
    assert dat_counts["accepted_descriptors_loaded"] == oracle["accepted_descriptors_loaded"]
    assert dat_counts["ignored_descriptors_loaded"] == oracle["ignored_descriptors_loaded"]
    assert dat_counts["discarded_descriptors_loaded"] == oracle["discarded_descriptors_loaded"]

    audit_path = os.path.join(path_main, "JSON", "curate_audit.json")
    audit = load_json(audit_path)

    assert_section_keys_present(
        audit,
        [
            "datapoints_loaded",
            "accepted_descriptors_loaded",
            "ignored_descriptors_loaded",
            "discarded_descriptors_loaded",
            "columns_removed_lt90pct_data",
            "rows_removed_gt50pct_missing",
            "knn_imputer_applied",
        ],
    )

    assert_event_payload_parity(
        audit,
        "load_database",
        {
            "datapoints_loaded": oracle["datapoints_loaded"],
            "accepted_descriptors_loaded": oracle["accepted_descriptors_loaded"],
            "ignored_descriptors_loaded": oracle["ignored_descriptors_loaded"],
            "discarded_descriptors_loaded": oracle["discarded_descriptors_loaded"],
            "columns_removed_lt90pct_data": oracle["columns_removed_lt90pct_data"],
            "rows_removed_gt50pct_missing": oracle["rows_removed_gt50pct_missing"],
            "knn_imputer_applied": oracle["knn_imputer_applied"],
        },
        [
            "datapoints_loaded",
            "accepted_descriptors_loaded",
            "ignored_descriptors_loaded",
            "discarded_descriptors_loaded",
            "columns_removed_lt90pct_data",
            "rows_removed_gt50pct_missing",
            "knn_imputer_applied",
        ],
    )

    section = audit["sections"]["load_database"]
    assert section["ignored_descriptors_loaded"]["value"] == oracle["ignored_descriptors_loaded"]
    assert section["discarded_descriptors_loaded"]["value"] == oracle["discarded_descriptors_loaded"]


def test_curate_correlation_filter_dat_json_parity_via_new_file_only():
    _clean_module_outputs()

    curate(
        csv_name=os.path.join("tests", "Robert_example.csv"),
        y="Target_values",
        names="Name",
        discard=["xtest"],
    )

    dat_path = os.path.join(path_curate, "CURATE_data.dat")
    assert os.path.exists(dat_path)
    with open(dat_path, "r", encoding="utf-8") as handle:
        dat_lines = handle.readlines()

    expected = parse_curate_correlation_filter_summary_from_dat(dat_lines)
    oracle = recompute_correlation_filter_oracle(
        csv_load=os.path.join("tests", "Robert_example.csv"),
        y_col="Target_values",
        ignore=["Name"],
        discard=["xtest"],
    )
    assert expected["constant_removed"] == len(oracle["constant_removed"])
    assert expected["low_y_corr_removed"] == len(oracle["low_y_corr_removed"])
    assert expected["high_intercorr_removed"] == len(oracle["high_intercorr_removed"])
    assert expected["rfecv_applied"] == oracle["rfecv_applied"]

    audit_path = os.path.join(path_main, "JSON", "curate_audit.json")
    audit = load_json(audit_path)

    assert_event_payload_parity(
        audit,
        "correlation_filter",
        expected,
        ["constant_removed", "low_y_corr_removed", "high_intercorr_removed", "rfecv_applied"],
    )

    corr_section = audit["sections"]["correlation_filter"]
    corr_events = [e for e in audit.get("events", []) if e.get("event_type") == "correlation_filter"]
    assert corr_events, "No correlation_filter event found"

    assert any(
        payload.get("constant_descriptors_removed") == corr_section["constant_descriptors_removed"]["value"]
        and payload.get("low_y_correlation_descriptors_removed") == corr_section["low_y_correlation_descriptors_removed"]["value"]
        and payload.get("high_intercorrelation_removals") == corr_section["high_intercorrelation_removals"]["value"]
        and payload.get("descriptors_removed_correlation_filter") == corr_section["descriptors_removed_correlation_filter"]["value"]
        and payload.get("rfecv_selection_method_by_model") == corr_section["rfecv_selection_method_by_model"]["value"]
        and payload.get("rfecv_descriptors_selected_by_model") == corr_section["rfecv_descriptors_selected_by_model"]["value"]
        and payload.get("rfecv_skip_reason") == corr_section["rfecv_skip_reason"]["value"]
        for payload in (e.get("payload", {}) for e in corr_events)
    ), "No correlation_filter event retained the expected detailed payload fields"

    assert corr_section["constant_descriptor_count_removed"]["value"] == expected["constant_removed"]
    assert corr_section["low_y_correlation_descriptor_count_removed"]["value"] == expected["low_y_corr_removed"]
    assert corr_section["high_intercorrelation_descriptor_count_removed"]["value"] == expected["high_intercorr_removed"]
    assert corr_section["descriptors_removed_correlation_filter"]["value"] == expected["high_intercorr_removed"]
    assert corr_section["rfecv_applied"]["value"] == expected["rfecv_applied"]


def test_curate_categorical_transform_dat_json_parity_via_new_file_only():
    _clean_module_outputs()

    curate(
        csv_name=os.path.join("tests", "Robert_example.csv"),
        y="Target_values",
        names="Name",
        discard=["xtest"],
    )

    dat_path = os.path.join(path_curate, "CURATE_data.dat")
    assert os.path.exists(dat_path)
    with open(dat_path, "r", encoding="utf-8") as handle:
        dat_lines = handle.readlines()

    expected = parse_curate_categorical_transform_summary_from_dat(dat_lines)
    oracle = recompute_categorical_transform_oracle(
        csv_load=os.path.join("tests", "Robert_example.csv"),
        y_col="Target_values",
        ignore=["Name"],
        discard=["xtest"],
    )
    assert expected == oracle

    audit_path = os.path.join(path_main, "JSON", "curate_audit.json")
    audit = load_json(audit_path)

    event_expected = {
        "categorical_variables_count": expected["categorical_variables_count"],
        "categorical_variables": expected["categorical_variables"],
        "categorical_variables_found": expected["categorical_variables_found"],
        "generated_descriptors_count": expected["generated_descriptors_count"],
        "generated_descriptors": expected["generated_descriptors"],
        "mode": expected["mode"],
        "descriptors_removed_categorical_transform": expected["categorical_variables_count"],
    }
    assert_event_payload_parity(
        audit,
        "categorical_transform",
        event_expected,
        [
            "categorical_variables_count",
            "categorical_variables",
            "categorical_variables_found",
            "generated_descriptors_count",
            "generated_descriptors",
            "mode",
            "descriptors_removed_categorical_transform",
        ],
    )

    cat_section = audit["sections"]["categorical_transform"]
    assert cat_section["descriptors_removed_categorical_transform"]["value"] == expected["categorical_variables_count"]
    assert cat_section["categorical_variables_found"]["value"] == (expected["categorical_variables_count"] > 0)
    assert len(cat_section["generated_descriptors"]["value"]) == expected["generated_descriptors_count"]
    if expected["mode"] is not None:
        assert cat_section["mode"]["value"] == expected["mode"]


def test_verify_dat_json_parity_via_new_file_only():
    _clean_module_outputs()

    curate(
        csv_name=os.path.join("tests", "Robert_example.csv"),
        y="Target_values",
        names="Name",
        discard=["xtest"],
    )

    generate(
        generate=True,
        csv_name=os.path.join("CURATE", "Robert_example_CURATE.csv"),
        y="Target_values",
        model=["RF"],
        init_points=1,
        n_iter=1,
    )

    verify()

    dat_path = os.path.join(path_verify, "VERIFY_data.dat")
    assert os.path.exists(dat_path)
    with open(dat_path, "r", encoding="utf-8") as handle:
        dat_lines = handle.readlines()

    expected_load = parse_labeled_counts_from_dat(
        dat_lines,
        "loaded successfully, including:",
        {
            "datapoints": "datapoints_loaded",
            "accepted descriptors": "accepted_descriptors_loaded",
            "ignored descriptors": "ignored_descriptors_loaded",
            "discarded descriptors": "discarded_descriptors_loaded",
        },
    )
    expected_summary = parse_verify_summary_metrics_from_dat(dat_lines)
    expected_branches = parse_verify_branch_titles_from_dat(dat_lines)
    expected_contexts = parse_verify_model_context_from_dat(dat_lines)

    audit_path = os.path.join(path_main, "JSON", "verify_audit.json")
    audit = load_json(audit_path)

    assert_section_keys_present(
        audit,
        [
            "datapoints_loaded",
            "accepted_descriptors_loaded",
            "ignored_descriptors_loaded",
            "discarded_descriptors_loaded",
        ],
    )

    assert_event_payload_parity(
        audit,
        "load_database",
        expected_load,
        ["datapoints_loaded", "accepted_descriptors_loaded"],
    )

    assert_event_payload_parity_rounded(
        audit,
        "print_verify_summary",
        expected_summary,
        [
            "original_cv_metric",
            "unclear_threshold",
            "pass_threshold",
            "y_mean_result",
            "y_shuffle_result",
            "onehot_result",
        ],
        ndigits=2,
    )

    verify_summary_events = [
        event for event in audit.get("events", []) if event.get("event_type") == "print_verify_summary"
    ]
    assert any(
        event.get("payload", {}).get("error_type") == expected_summary["error_type"]
        and event.get("payload", {}).get("cv_type") == expected_summary["cv_type"]
        and event.get("payload", {}).get("threshold_direction") == expected_summary["threshold_direction"]
        and event.get("payload", {}).get("unclear_threshold_percent")
        == expected_summary["unclear_threshold_percent"]
        and event.get("payload", {}).get("pass_threshold_percent") == expected_summary["pass_threshold_percent"]
        for event in verify_summary_events
    ), "No print_verify_summary event matched DAT threshold metadata"

    assert any(
        event.get("payload", {}).get("sorted_cv_metrics", {}).get("regression")
        == expected_summary["sorted_metrics"]
        for event in verify_summary_events
    ), "No print_verify_summary event matched DAT sorted-CV metrics"

    analyze_events = [event for event in audit.get("events", []) if event.get("event_type") == "analyze_tests"]
    assert any(
        event.get("payload", {}).get("per_test_status") == expected_summary["test_status"]
        and all(
            round(float(event.get("payload", {}).get("per_test_metric", {}).get(test_name)), 2)
            == round(float(expected_summary[result_key]), 2)
            for test_name, result_key in {
                "y_mean": "y_mean_result",
                "y_shuffle": "y_shuffle_result",
                "onehot": "onehot_result",
            }.items()
        )
        for event in analyze_events
    ), "No analyze_tests event matched DAT test statuses and metrics"

    verify_test_events = [event for event in audit.get("events", []) if event.get("event_type") == "verify_test"]
    for test_name, result_key in {
        "y_mean": "y_mean_result",
        "y_shuffle": "y_shuffle_result",
        "onehot": "onehot_result",
    }.items():
        assert any(
            event.get("payload", {}).get("test_name") == test_name
            and round(float(event.get("payload", {}).get("resulting_metric")), 2)
            == round(float(expected_summary[result_key]), 2)
            for event in verify_test_events
        ), f"No verify_test event matched DAT result for {test_name}"

    verify_branch_events = [event for event in audit.get("events", []) if event.get("event_type") == "verify_branch"]
    actual_branches = [event.get("payload", {}).get("suffix_title") for event in verify_branch_events]
    assert actual_branches == expected_branches, "VERIFY branch events did not match DAT branch markers"

    model_context_events = [event for event in audit.get("events", []) if event.get("event_type") == "model_context"]
    assert len(model_context_events) == len(expected_contexts), "VERIFY model-context event count did not match DAT blocks"
    for event, expected_context in zip(model_context_events, expected_contexts):
        payload = event.get("payload", {})
        assert payload.get("model_name") == expected_context["model_name"]
        assert payload.get("y_column") == expected_context["y_column"]
        assert payload.get("names_column") == expected_context["names_column"]
        assert payload.get("kfold") == expected_context["kfold"]
        assert payload.get("repeat_kfolds") == expected_context["repeat_kfolds"]
        assert payload.get("descriptor_list") == expected_context["descriptor_list"]
        assert payload.get("train_datapoints") == expected_context["train_datapoints"]
        assert payload.get("total_datapoints_loaded") == (
            expected_context["train_datapoints"] + expected_context["test_datapoints"]
        )


def test_predict_dat_json_parity_via_new_file_only():
    _clean_module_outputs()

    curate(
        csv_name=os.path.join("tests", "Robert_example.csv"),
        y="Target_values",
        names="Name",
        discard=["xtest"],
    )

    generate(
        generate=True,
        csv_name=os.path.join("CURATE", "Robert_example_CURATE.csv"),
        y="Target_values",
        model=["RF"],
        init_points=1,
        n_iter=1,
    )

    predict(csv_test=os.path.join("tests", "Robert_example_test.csv"))

    dat_path = os.path.join(path_predict, "PREDICT_data.dat")
    assert os.path.exists(dat_path)
    with open(dat_path, "r", encoding="utf-8") as handle:
        dat_lines = handle.readlines()

    expected_external_load = parse_predict_external_load_count_from_dat(dat_lines)
    expected_summary = parse_predict_summary_metrics_from_dat(dat_lines)

    audit_path = os.path.join(path_main, "JSON", "predict_audit.json")
    audit = load_json(audit_path)

    assert_section_keys_present(audit, ["datapoints_loaded"])

    assert_event_payload_parity(
        audit,
        "load_database",
        expected_external_load,
        ["datapoints_loaded"],
    )

    assert_predict_summary_event_parity(audit, expected_summary, ndigits=2)
