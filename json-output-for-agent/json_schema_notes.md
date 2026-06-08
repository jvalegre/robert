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

- `dataset_profile.json` is the raw-intake artifact and is captured before ROBERT modifies the incoming dataset.
- Archive manifests and run summary JSONs are copy-only mirrors of generated outputs.
- JSON schema is still experimental (`schema_version: 0.1`).

---

## Long-Term Module Audit Pattern (Design Direction)

Planned pattern:
- Keep `dataset_profile.json` as raw incoming data evidence.
- Add one module-specific audit JSON per ROBERT module over time.
- Keep writes additive and fail-soft.
- Keep JSON-layer status only in JSON audit files.
- Never write JSON-layer status to standard `.dat`, `.csv`, image, or PDF outputs.

Proposed future module audit files:
- `CURATE/curate_audit.json`
- `GENERATE/generate_audit.json`
- `VERIFY/verify_audit.json`
- `PREDICT/predict_audit.json`
- `REPORT/report_audit.json`

---

## Proposed CURATE Audit Schema (Design Only, Not Implemented)

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

Implementation status (2026-06-08):
- `CURATE/curate_audit.json` is now implemented.
- Current implementation records direct observable fields and marks step-specific descriptor-removal counters as `unavailable`.
- JSON-layer write status for this artifact is recorded in `CURATE/json_output_audit.json` with `artifact="curate_audit.json"`.
