# ROBERT JSON Output Project

This folder tracks the project to add structured JSON outputs to ROBERT.

The goal is to help future tools understand ROBERT runs without fragile parsing of PDFs or loosely formatted text files.

The project does not change ROBERT's scientific behavior.

## Important Files

- `AGENTS.md`  
  Instructions for AI coding agents working on this project.

- `PROJECT_RULES.md`  
  Process rules for how changes should be proposed, approved, implemented, tested, and documented.

- `TASKS.md`  
  Current project to-do list.

- `task_tracker_plain_english.md`  
  Plain-English record of completed work.

- `json_schema_notes.md`  
  Working notes about possible JSON structures.

## Current Priority

Preserve normal ROBERT output behavior while adding project-side JSON/export support.

Current wrapper policy:
- ROBERT still writes normal outputs in default root folders.
- A timestamped archive wrapper duplicates generated outputs into:
  - `json-output-for-agent/runs/<timestamp>_<input_csv_stem>/`
- The archive is copy-only and does not replace ROBERT output paths.

## Validation Source-Data Policy (Temporary)

For JSON-output validation runs, `databases/` is currently treated as a protected source-data folder.

Rules:
- Do not edit CSV files in `databases/`.
- Do not move or rename files in `databases/`.
- Do not overwrite source CSV files.
- Do not write generated ROBERT outputs into `databases/`.

Generated outputs must remain in normal root output folders and optional copy-only archives.

## 2026-06-08 Validation Snapshot

Validated with timestamped wrapper:
- Regression input: `databases/Regression/AQME-ROBERT_A_predict_solubility.csv`
- Classification input: `databases/Clasification/F_predict_outcome.csv`

Pass/fail summary:
- PASS: Standard root outputs are still produced for CURATE/GENERATE/VERIFY/PREDICT.
- PASS: JSON artifacts are produced (`CURATE/dataset_profile.json`, `CURATE/curate_audit.json`, `CURATE/json_output_audit.json`, archive manifests, `run_summary.json`).
- PASS: JSON files open and required top-level fields are present.
- PASS: `dataset_profile.json` includes `schema_version` and works for both regression and classification targets.
- PASS: No JSON-layer status text detected in standard `.dat` files.
- PASS: Copy-only archive behavior confirmed via `wrapper_run.json` copy-mode statement.
- PASS: Missing module folders are represented in manifests with `module_dir_exists=false` and `file_count=0`.
- PARTIAL: REPORT PDF was not generated because required WeasyPrint system libraries are missing in this environment.

## Git Ignore Guidance

Guidance for validation workflow:
- Do not ignore `databases/`.
- Ignore generated ROBERT run outputs in root folders and timestamped archives.

## Open TODO

Decide whether `databases/` should:
- remain in the repository,
- move to `tests/fixtures/`, or
- be removed after formal JSON unit tests are created.
