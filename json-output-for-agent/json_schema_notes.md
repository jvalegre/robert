# json_schema_notes.md

This file records working ideas for ROBERT JSON outputs.

The schema is experimental until confirmed through implementation and testing.

---

## Design Goals

The JSON files should be:

- easy for humans to read,
- easy for a UI to load,
- easy for an LLM to explain,
- traceable back to ROBERT outputs,
- stable enough to use across runs,
- explicit about what is known and unknown.

---

## Basic Module Summary Schema

```json
{
  "schema_version": "0.1",
  "module": "CURATE",
  "artifact_type": "module_summary",
  "status": "completed",
  "inputs": {},
  "outputs": {
    "files_created": []
  },
  "parameters": {},
  "direct_robert_values": {},
  "derived_helper_values": {},
  "warnings": [],
  "notes": []
}
```

---

## Basic Run Context Schema

```json
{
  "schema_version": "0.1",
  "artifact_type": "run_context",
  "robert_version": null,
  "run_id": null,
  "created_at": null,
  "command": null,
  "modules": {
    "curate": {},
    "generate": {},
    "verify": {},
    "predict": {},
    "report": {}
  },
  "score": {},
  "warnings": [],
  "file_manifest": []
}
```

---

## Basic File Manifest Schema

```json
{
  "schema_version": "0.1",
  "artifact_type": "file_manifest",
  "run_id": null,
  "created_at": null,
  "files": [
    {
      "path": "PREDICT/PREDICT_data.dat",
      "name": "PREDICT_data.dat",
      "extension": ".dat",
      "module": "PREDICT",
      "size_bytes": null,
      "modified_at": null,
      "description": "ROBERT prediction summary data"
    }
  ]
}
```

---

## Open Schema Questions

- Should each module write its own JSON file?
- Should REPORT assemble the full `run_context.json`?
- Should JSON export be controlled by a command-line option?
- Should JSON files include full values or only summaries?
- Should images be listed only, or should image metadata be captured?
- Should `.dat` file contents be copied into JSON or only referenced?
- Should file timestamps use local time or UTC?
- Should output folders be timestamped by run?

---

## Confirmed Baseline (2026-06-08)

- `JSON/dataset_profile.json` is the raw-intake artifact and is captured before ROBERT modifies the incoming dataset.
- Archive manifests and run summary JSONs are copy-only mirrors of generated outputs.
- JSON schema is still experimental (`schema_version: 0.1`).

## Confirmed Artifact Location Policy (2026-07-13)

- all ChatBob runtime JSON artifacts are written to the top-level `JSON/` folder;
- module-native runtime audit files in `JSON/` are distinct from wrapper-generated archive manifests;
- standard ROBERT scientific outputs remain in module folders (`CURATE/`, `GENERATE/`, `VERIFY/`, `PREDICT/`, `AQME/`, `EVALUATE/`, `REPORT/`).

---

## Long-Term Module Audit Pattern (Design Direction)

Planned pattern:
- Keep `JSON/dataset_profile.json` as raw incoming data evidence.
- Add one module-specific audit JSON per ROBERT module over time.
- Keep writes additive and fail-soft.
- Keep JSON-layer status only in JSON audit files.
- Never write JSON-layer status to standard `.dat`, `.csv`, image, or PDF outputs.

Proposed future module audit files:
- `JSON/curate_audit.json`
- `JSON/generate_audit.json`
- `JSON/verify_audit.json`
- `JSON/predict_audit.json`
- `JSON/aqme_audit.json`
- `JSON/evaluate_audit.json`
- `JSON/report_audit.json`

---

## CURATE Audit Schema: Original Design and Current Implementation

Purpose:
- Capture observable CURATE evidence without changing ROBERT behavior.
- Record unavailable values as unavailable rather than guessed.

```json
{
  "schema_version": "0.1",
  "artifact_type": "curate_audit",
  "module": "CURATE",
  "status": "completed_or_partial",
  "captured_utc": "ISO-8601",
  "inputs": {
    "source_csv": "path string",
    "target_column": "string",
    "names_column": "string or null",
    "ignored_columns": [],
    "discarded_columns_requested": []
  },
  "observed_counts": {
    "rows_before_curate": "int or unavailable",
    "rows_after_curate": "int or unavailable",
    "columns_before_curate": "int or unavailable",
    "columns_after_curate": "int or unavailable",
    "descriptors_removed_duplicate_filter": "int or unavailable",
    "descriptors_removed_missingness": "int or unavailable",
    "descriptors_removed_categorical_transform": "int or unavailable",
    "descriptors_removed_correlation_filter": "int or unavailable",
    "descriptors_removed_other": "int or unavailable"
  },
  "files_written": [
    {
      "path": "CURATE/...",
      "artifact_type": "csv|dat|png|json|other",
      "exists": true
    }
  ],
  "standard_output_paths": {
    "curate_dat": "CURATE/CURATE_data.dat",
    "curate_options_csv": "CURATE/CURATE_options.csv",
    "curated_csvs": []
  },
  "notes": [],
  "unavailable_fields": [],
  "json_layer_audit": {
    "attempted": true,
    "succeeded": true,
    "error_type": null,
    "error_message": null
  }
}
```

Design constraints:
- No guessed explanations for why descriptors were removed.
- If a reason/count is not observable from existing ROBERT state, mark as unavailable.
- Audit status belongs only in JSON audit artifacts.

Implementation status (2026-07-13):
- `JSON/curate_audit.json`, `JSON/generate_audit.json`, `JSON/verify_audit.json`, and `JSON/predict_audit.json` are implemented.
- `JSON/aqme_audit.json` and `JSON/evaluate_audit.json` are not implemented.
- REPORT has figure provenance capture in `JSON/report_figure_provenance.json`; a full `JSON/report_audit.json` is not implemented.
- Current implementation records direct observable fields and marks step-specific descriptor-removal counters as `unavailable`.
- JSON-layer write status for this artifact is recorded in `JSON/curate_json_output_audit.json` with `artifact="curate_audit.json"`.

---

## Standard Output Preservation Validation

A controlled regression comparison was completed on 2026-07-13 using:

- unmodified ROBERT 2.1.2, and
- ChatBob-modified ROBERT 2.1.2.

Results:

- all four primary `.dat` files were identical after normalizing only timestamps, paths, execution times, and trailing whitespace;
- all scientific values in 28 matching CSV files were identical;
- the only CSV text difference was the expected input path stored in `CURATE_options.csv`.

This confirms that the JSON-output implementation is additive for the tested regression case.

This result does not by itself prove that every JSON value is correct. JSON evidence must still be validated against the relevant ROBERT DAT, CSV, and in-memory source values.

---

## Question-Driven Schema Review

The next schema step should begin with a small set of user questions rather than a large universal schema.

Initial candidate questions:

1. What model was selected?
2. Which descriptors were used?
3. How well did the model perform?
4. Did the model pass the verification tests?
5. Which points were identified as outliers?

For each question, document:

- the required JSON artifact,
- the event type or section,
- the required fields,
- whether each field is direct, derived, or unavailable,
- the original ROBERT DAT or CSV source,
- the expected answer structure.

Only make schema changes that are needed to answer these initial questions reliably.

## Minimal Common Audit Envelope

A possible common structure for module-native audit files is:

```json
{
  "schema_version": "0.1",
  "module": "GENERATE",
  "artifact_type": "generate_audit",
  "status": "completed",
  "source_files": [],
  "events": [
    {
      "event_type": "prepare_sets",
      "evidence_level": "direct_or_derived",
      "source_reference": null,
      "payload": {}
    }
  ],
  "runtime": {},
  "unavailable_fields": []
}
