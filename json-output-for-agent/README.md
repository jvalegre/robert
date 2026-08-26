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

### Current Priority

The immediate priority is to prepare the JSON-output work for upstream review.

Current source-verified module-native runtime audit status:

- Implemented: CURATE, GENERATE, VERIFY, PREDICT
- Not implemented: AQME, EVALUATE
- REPORT: figure provenance only (`JSON/report_figure_provenance.json`), no full report audit yet

Current validation status (2026-08-26):

- CURATE, GENERATE, VERIFY, and PREDICT DAT-to-JSON parity checks are implemented in the isolated test files.
- CURATE also has independent third-oracle checks and mutation tests.
- The isolated parity suite passes with `7 passed`; the combined parity and CURATE mutation suite passes with `10 passed`.
- AQME provenance JSON is a possible future enhancement, not a current requirement.

Next steps:

1. Open a pull request from `json-output-for-agent`.
2. Agree with Juanvi on the integration branch and JSON output location.
3. Keep the documented implementation status of all module audit files synchronized with source and tests.
4. Review the current JSON structures against a small set of likely ChatBob questions.
5. Build deterministic question-to-evidence retrieval before adding an LLM answer layer.

## Validation Source-Data Policy (Temporary)

For JSON-output validation runs, `databases/` is currently treated as a protected source-data folder.

Rules:
- Do not edit CSV files in `databases/`.
- Do not move or rename files in `databases/`.
- Do not overwrite source CSV files.
- Do not write generated ROBERT outputs into `databases/`.

Generated outputs must remain in normal root output folders and optional copy-only archives.


## 2026-07-13 Controlled Run Comparison

The ChatBob JSON-output branch was synchronized with upstream ROBERT 2.1.2 and compared against an unmodified ROBERT 2.1.2 run.

Validation result:

- all four primary ROBERT `.dat` files were identical after normalizing only timestamps, file paths, execution times, and trailing whitespace;
- all scientific values in 28 matching CSV files were identical;
- the only expected CSV difference was the recorded input file path in `CURATE/CURATE_options.csv`.

For the tested regression case, the JSON-output implementation did not alter standard ROBERT scientific outputs.

## 2026-07-14 In-Repo Verification Update

Additional manual verification was completed in this repository after the 2026-07-13 comparison milestone.

Confirmed:

- structured JSON files were generated and reviewed for CURATE, GENERATE, VERIFY, PREDICT, and REPORT stages;
- when the updated ROBERT version in this repo was run and compared against DAT files from original ROBERT, no DAT differences were observed.

This confirmation supports the additive-behavior claim for the verified workflow scope.


## Comparison Notebook

`comparison/compare_robert_outputs.ipynb` compares two completed ROBERT runs.

The copied ROBERT output folders are local validation data and are not committed.

## 2026-06-08 Validation Snapshot

Validated with timestamped wrapper:
- Regression input: `databases/Regression/AQME-ROBERT_A_predict_solubility.csv`
- Classification input: `databases/Clasification/F_predict_outcome.csv`

Pass/fail summary:
- PASS: Standard root outputs are still produced for CURATE/GENERATE/VERIFY/PREDICT.
- PASS: JSON artifacts are produced (`JSON/dataset_profile.json`, `JSON/curate_audit.json`, `JSON/curate_json_output_audit.json`, archive manifests, `run_summary.json`).
- PASS: JSON files open and required top-level fields are present.
- PASS: `JSON/dataset_profile.json` includes `schema_version` and works for both regression and classification targets.
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
