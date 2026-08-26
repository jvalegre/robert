import pytest

from tests.json_parity_helpers import assert_event_payload_parity


def test_load_database_parity_rejects_mutated_expected_count():
    audit = {
        "events": [
            {
                "event_type": "load_database",
                "payload": {
                    "datapoints_loaded": 38,
                    "accepted_descriptors_loaded": 12,
                    "ignored_descriptors_loaded": 1,
                    "discarded_descriptors_loaded": 1,
                },
            }
        ]
    }
    mutated_expected = {
        "datapoints_loaded": 39,
        "accepted_descriptors_loaded": 12,
        "ignored_descriptors_loaded": 1,
        "discarded_descriptors_loaded": 1,
    }

    with pytest.raises(AssertionError):
        assert_event_payload_parity(
            audit,
            "load_database",
            mutated_expected,
            list(mutated_expected),
        )


def test_categorical_transform_parity_rejects_mutated_expected_descriptor():
    audit = {
        "events": [
            {
                "event_type": "categorical_transform",
                "payload": {
                    "categorical_variables_count": 1,
                    "categorical_variables": ["x4"],
                    "generated_descriptors_count": 4,
                    "generated_descriptors": ["Csub-Csub", "Csub-H", "Csub-O", "H-O"],
                    "categorical_variables_found": True,
                    "mode": "onehot",
                    "descriptors_removed_categorical_transform": 1,
                },
            }
        ]
    }
    mutated_expected = {
        "categorical_variables_count": 1,
        "categorical_variables": ["x4"],
        "generated_descriptors_count": 4,
        "generated_descriptors": ["Csub-Csub", "Csub-H", "Csub-O", "H-N"],
        "categorical_variables_found": True,
        "mode": "onehot",
        "descriptors_removed_categorical_transform": 1,
    }

    with pytest.raises(AssertionError):
        assert_event_payload_parity(
            audit,
            "categorical_transform",
            mutated_expected,
            list(mutated_expected),
        )


def test_correlation_filter_parity_rejects_mutated_expected_status():
    audit = {
        "events": [
            {
                "event_type": "correlation_filter",
                "payload": {
                    "constant_removed": 1,
                    "low_y_corr_removed": 0,
                    "high_intercorr_removed": 2,
                    "rfecv_applied": True,
                },
            }
        ]
    }
    mutated_expected = {
        "constant_removed": 1,
        "low_y_corr_removed": 0,
        "high_intercorr_removed": 2,
        "rfecv_applied": False,
    }

    with pytest.raises(AssertionError):
        assert_event_payload_parity(
            audit,
            "correlation_filter",
            mutated_expected,
            list(mutated_expected),
        )
