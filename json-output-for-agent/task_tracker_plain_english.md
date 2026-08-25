# task_tracker_plain_english.md

This file records what has been done in plain language.

It should be updated after every meaningful project step.

The goal is that a chemist can read this file and understand the project history without opening the code.

---

## Current Project Goal

We are adding structured JSON outputs to ROBERT.

These JSON files should record information ROBERT already creates during a run.

The purpose is to make ROBERT results easier to inspect later through a user interface or optional explanation tool.

We are not changing ROBERT's scientific behavior.

---

## Running Log

### Entry 001

Date: YYYY-MM-DD

Status: Project setup started.

What we decided:
- We will work in small steps.
- The existing ROBERT code is the source of truth.
- We will inspect before changing.
- The agent must propose a plan before implementation.
- The user must approve before code changes.
- We will minimize changes to existing ROBERT files.
- We will use helper functions where possible.
- We will track progress in this file.

Files created or planned:
- `AGENTS.md`
- `PROJECT_RULES.md`
- `TASKS.md`
- `task_tracker_plain_english.md`

Confirmed results:
- No code has been changed yet.

Uncertainties:
- We have not yet inspected where all ROBERT outputs are created.
- We have not yet decided the final JSON schema.
- We have not yet decided whether JSON export should always run or be controlled by an option.

Next step:
- Inspect the ROBERT code to find where CURATE, GENERATE, VERIFY, PREDICT, and REPORT create outputs.

### Entry 002

Date: 2026-05-27

Goal of this step:
- Inspect ROBERT outputs and create a durable output map before any code modification.

What changed:
- We inspected CURATE, GENERATE, VERIFY, PREDICT, REPORT, and shared utility/report helper files.
- We documented where CSV files, image files, dat files, and the PDF report are created.
- We documented where ROBERT score values are calculated.
- We documented where values are already in memory before being written.
- We identified minimal and safe future points where JSON helper calls can be added later.

Files changed:
- `json-output-for-agent/output_map.md`
- `json-output-for-agent/task_tracker_plain_english.md`
- `json-output-for-agent/TASKS.md`

Why this change was made:
- To prepare a low-risk JSON export plan that does not change ROBERT scientific behavior.

How this was tested:
- Manual code inspection only.
- No ROBERT source code was edited.
- No runtime behavior was changed.

Confirmed results:
- Output creation points were mapped for all requested modules.
- Dat file creation flow was mapped through the shared Logger and finish_print functions.
- Image/plot creation points were mapped across CURATE, GENERATE, VERIFY, and PREDICT.
- PDF report creation and ROBERT score calculation flow were mapped in REPORT/report_utils.

What remains uncertain:
- Final JSON schema details are still undecided.
- Whether JSON export should always run or be optional is still undecided.
- Exact naming/location convention for JSON artifacts is still undecided.

Next suggested step:
- Propose one minimal first implementation: add a JSON helper module and one JSON write call in CURATE after `CURATE_options.csv` is written.

### Entry 003

Date: 2026-05-28

Goal of this step:
- Implement the first JSON artifact as an incoming dataset profile.
- Confirm that CURATE still finishes normally if JSON writing fails.

What changed:
- Added a new helper module: robert/json_output_for_agent.py.
- Added a fail-soft JSON hook in robert/curate.py immediately after load_variables().
- The helper now writes CURATE/dataset_profile.json during normal CURATE runs.
- The CURATE hook is wrapped in try/except and does not stop execution.

Files changed:
- `robert/json_output_for_agent.py`
- `robert/curate.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- To add the first machine-readable artifact requested for UI/agent workflows while preserving existing ROBERT behavior.

How this was tested:
- Baseline CURATE run:
	- `python -m robert --curate --csv_name tests/Robert_example.csv --y Target_values --ignore Name --names Name`
	- Confirmed normal CURATE outputs plus `CURATE/dataset_profile.json`.
- Fault-injection CURATE run:
	- Monkeypatched `write_json` to raise `RuntimeError('simulated json write failure')`.
	- Confirmed CURATE completed and generated standard outputs.
	- Confirmed JSON failure was recorded in `CURATE/json_output_audit.json`.

### Entry 004

Date: 2026-05-28

Goal of this step:
- Enforce the standard ROBERT output protection rule.
- Ensure JSON-layer status is never written into standard ROBERT files.

What changed:
- Removed JSON-layer status logging to the standard CURATE logger in robert/curate.py.
- Added write_json_output_audit(...) helper in robert/json_output_for_agent.py.
- Added dedicated audit artifact CURATE/json_output_audit.json for JSON-layer events.
- CURATE now records dataset profile success/failure only in the audit JSON file.

Files changed:
- `robert/curate.py`
- `robert/json_output_for_agent.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`
- `json-output-for-agent/output_map.md`

Why this change was made:
- To keep the JSON layer invisible to all standard ROBERT outputs.
- To guarantee that .dat/.csv/image and other standard files are not modified by JSON-layer status messages.

How this was tested:
- Baseline CURATE run confirmed:
	- Standard CURATE outputs are produced.
	- CURATE/dataset_profile.json is produced.
	- CURATE/json_output_audit.json records success.
	- CURATE/CURATE_data.dat contains no JSON-layer status text.
- Fault-injection run (write_json monkeypatched to raise) confirmed:
	- CURATE still completes.
	- Standard CURATE outputs are still produced.
	- CURATE/CURATE_data.dat still contains no JSON-layer status text.
	- CURATE/json_output_audit.json records failure with error type and message.

Confirmed results:
- JSON-layer status is now isolated to JSON audit artifacts.
- Standard ROBERT outputs are unchanged by JSON success/failure state.

What remains uncertain:
- An automated unit test dedicated to JSON audit edge cases is not yet added.

Next suggested step:
- Stop here per scope; do not extend beyond this protection fix.

### Entry 005

Date: 2026-05-28

Goal of this step:
- Add a timestamped archive wrapper without changing normal ROBERT output behavior.

What changed:
- Added a wrapper script: `json-output-for-agent/scripts/run_robert_timestamped.py`.
- Wrapper runs ROBERT from the repository root exactly as normal.
- After ROBERT finishes, wrapper copies generated outputs into:
	- `json-output-for-agent/runs/<timestamp>_<input_csv_stem>/`
- Added explicit copy-only policy text in `AGENTS.md` and `PROJECT_RULES.md`.

Files changed:
- `json-output-for-agent/scripts/run_robert_timestamped.py`
- `json-output-for-agent/AGENTS.md`
- `json-output-for-agent/PROJECT_RULES.md`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`
- `json-output-for-agent/README.md`

Why this change was made:
- To keep the established ROBERT audience and workflows fully intact while creating a project-side run archive for JSON/export tracking.

How this was tested:
- Wrapper dry-run check to confirm command build, folder creation, and metadata write without executing ROBERT.

Confirmed results:
- Wrapper policy is explicit: normal root outputs are preserved; archive is duplicate-copy only.

What remains uncertain:
- A full end-to-end wrapper run validation is still pending.

Next suggested step:
- Execute one full wrapper run and verify archive completeness for CURATE/GENERATE/VERIFY/PREDICT/REPORT artifacts.

### Entry 006

Date: 2026-05-28

Goal of this step:
- Start implementation of full-run JSON mirroring with minimal complexity and no change to standard ROBERT behavior.

What changed:
- Extended `robert/json_output_for_agent.py` with archive-manifest helpers.
- Added file metadata capture helpers (path, extension, size, modified time, sha256).
- Added safe text preview extraction for textual files and best-effort PDF text preview extraction.
- Added `collect_module_manifest(...)` and `write_module_manifest(...)`.
- Added `write_archive_manifests(...)` to generate one manifest per module folder in a run archive.
- Added `collect_run_summary(...)` and `write_run_summary(...)`.
- Updated wrapper script `json-output-for-agent/scripts/run_robert_timestamped.py` to generate manifests and run summary after copy.
- Added archive-level JSON write status artifact: `json_output_audit.json` in each run folder.

Files changed:
- `robert/json_output_for_agent.py`
- `json-output-for-agent/scripts/run_robert_timestamped.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- To deliver immediate full-run JSON mirrors from exact copied output artifacts while avoiding high-risk changes inside CURATE/GENERATE/VERIFY/PREDICT/REPORT internals.
- To keep standard outputs as source of truth and maintain copy-only archive behavior.

How this was tested:
- Wrapper dry run:
	- `python json-output-for-agent/scripts/run_robert_timestamped.py --wrapper-dry-run --csv_name tests/Robert_example.csv`
- Full wrapper run (CURATE example):
	- `python json-output-for-agent/scripts/run_robert_timestamped.py --curate --discard "['xtest']" --y Target_values --csv_name tests/Robert_example.csv --names Name`
- Verified archive folder contains:
	- `CURATE_manifest.json`
	- `GENERATE_manifest.json`
	- `VERIFY_manifest.json`
	- `PREDICT_manifest.json`
	- `REPORT_manifest.json`
	- `run_summary.json`
	- `json_output_audit.json`
	- `wrapper_run.json`

Confirmed results:
- Standard ROBERT execution and outputs remain unchanged.
- JSON mirrors are generated in timestamped archive only.
- Manifest records include hashes and file-level metadata suitable for exact comparisons.

### Entry 018

Date: 2026-07-28

Goal of this step:
- Complete parity sweep checkpoint for CURATE `load_database` using an anti-GIGO third-oracle strategy.

What changed:
- Added an independent recomputation helper in `tests/json_parity_helpers.py`:
	- `recompute_load_database_oracle(...)`
- Added a new isolated parity test in `tests/test_json_parity.py`:
	- `test_curate_load_database_third_oracle_parity_via_new_file_only`
- The new test now checks three-way agreement for CURATE load counts:
	1) DAT text parsing,
	2) JSON audit event/section values,
	3) independent recomputation from input CSV and options.

Why this change was made:
- To reduce garbage-in-garbage-out risk by validating DAT and JSON against an independent oracle rather than only against each other.

How this was tested:
- Ran focused parity suite:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest tests/test_json_parity.py -q`
- Result: `4 passed`.

Confirmed results:
- CURATE load-database parity checkpoint is complete and passing with third-oracle validation.

Next suggested step:
- Implement the next one-by-one checkpoint: CURATE `correlation_filter` DAT↔JSON parity.

### Entry 020

Date: 2026-07-29

Goal of this step:
- Complete parity sweep checkpoint for CURATE `correlation_filter` using isolated parity files only.

What changed:
- Added a new DAT parser helper in `tests/json_parity_helpers.py`:
	- `parse_curate_correlation_filter_summary_from_dat(...)`
- Added a new isolated parity test in `tests/test_json_parity.py`:
	- `test_curate_correlation_filter_dat_json_parity_via_new_file_only`
- The new test compares DAT summary values to JSON audit evidence for:
	1) `correlation_filter` event payload keys (`constant_removed`, `low_y_corr_removed`, `high_intercorr_removed`, `rfecv_applied`)
	2) `correlation_filter` section fallback keys for count-level parity.

Why this change was made:
- To close the next one-by-one DAT↔JSON trust checkpoint while keeping parity logic isolated from legacy test files.

How this was tested:
- Ran focused new checkpoint test:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest tests/test_json_parity.py::test_curate_correlation_filter_dat_json_parity_via_new_file_only -q`
- Ran full isolated parity suite:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest tests/test_json_parity.py -q`

Confirmed results:
- New `correlation_filter` parity checkpoint passed.
- Full isolated parity suite passed (`5 passed`).
- Scope rule preserved: parity changes remain only in `tests/json_parity_helpers.py` and `tests/test_json_parity.py`.

Next suggested step:
- Implement the next checkpoint: CURATE `categorical_transform` DAT↔JSON parity.

### Entry 021

Date: 2026-07-29

Goal of this step:
- Complete parity sweep checkpoint for CURATE `categorical_transform` using isolated parity files only.

What changed:
- Added a DAT parser helper in `tests/json_parity_helpers.py`:
	- `parse_curate_categorical_transform_summary_from_dat(...)`
- Added a new isolated parity test in `tests/test_json_parity.py`:
	- `test_curate_categorical_transform_dat_json_parity_via_new_file_only`
- The new test compares DAT summary values to JSON audit evidence for:
	1) `categorical_transform` event payload keys (`categorical_variables_count`, `generated_descriptors_count`, `mode`)
	2) `categorical_transform` section fallback checks (`descriptors_removed_categorical_transform`, `categorical_variables_found`, generated descriptor count).

Why this change was made:
- To close the next one-by-one DAT↔JSON trust checkpoint while keeping parity logic isolated from legacy test files.

How this was tested:
- Ran focused new checkpoint test:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest tests/test_json_parity.py::test_curate_categorical_transform_dat_json_parity_via_new_file_only -q`
- Ran full isolated parity suite:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest tests/test_json_parity.py -q`

Confirmed results:
- New `categorical_transform` parity checkpoint passed.
- Full isolated parity suite passed (`6 passed`).
- Scope rule preserved: parity changes remain only in `tests/json_parity_helpers.py` and `tests/test_json_parity.py`.

Next suggested step:
- Implement the next checkpoint: GENERATE model-scan summary parity.
- Missing module folders produce valid manifests with `module_dir_exists=false` and `file_count=0`.

What remains uncertain:
- Module-native write-time hooks are still not added.
- Automated tests for helper-manifest functions are still not added.

Next suggested step:
- Add small helper tests for manifest and run-summary functions, then (if still needed) add module-native hooks incrementally behind a low-risk option.

### Entry 022

Date: 2026-07-29

Goal of this step:
- Record the DAT-to-event payload audit and the approved follow-up sequence before any source code changes.

What changed:
- Completed a read-only inventory of DAT values that are not yet retained in matching JSON event payloads.
- Grouped the gaps by CURATE, GENERATE, VERIFY, and PREDICT.
- Marked the next approved implementation step as the narrow CURATE slice only.
- Left GENERATE, VERIFY, and PREDICT as later reviewable increments.

Why this change was made:
- To make the implementation order explicit before any source code changes.
- To keep the project focused on documentation first.

How this was tested:
- Documentation-only audit.
- No ROBERT source code was changed during this step.

Confirmed results:
- The audit found structured-event coverage gaps across CURATE, GENERATE, VERIFY, and PREDICT.
- The next approved implementation work is limited to CURATE load_database, categorical_transform, correlation_filter, and the associated focused parity, third-oracle, and mutation tests.
- No scientific calculations, thresholds, DAT output, CSV output, model behavior, or CLI behavior were changed.

Next suggested step:
- Implement only the approved CURATE slice in reviewable increments.

### Entry 023

Date: 2026-07-29

Goal of this step:
- Close the first approved CURATE event-field gap for `load_database`.

What changed:
- Added `ignored_descriptors_loaded` and `discarded_descriptors_loaded` to the existing `load_database` event payload using runtime values already calculated in `load_database()`.
- Extended the CURATE third-oracle parity test so the new event payload fields are required in the JSON audit check.

Why this change was made:
- To align the CURATE `load_database` event payload with the values already printed in the DAT summary.
- To keep the change narrow and limited to the first approved CURATE gap only.

How this was tested:
- Ran the focused CURATE parity test:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest 'tests/test_json_parity.py::test_curate_load_database_third_oracle_parity_via_new_file_only' -q`
- Ran the full isolated parity suite:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest tests/test_json_parity.py -q`

Confirmed results:
- The focused CURATE load_database parity test passed.
- The full isolated parity suite passed.
- Only the approved CURATE load_database event fields were added.

Next suggested step:
- Stop here until the next CURATE slice is explicitly approved.

### Entry 024

Date: 2026-07-31

Goal of this step:
- Close the approved CURATE `categorical_transform` event-field gap.

What changed:
- Expanded the CURATE `categorical_transform` event payload to retain fields already available at runtime:
	- `categorical_variables`
	- `categorical_variables_found`
	- `generated_descriptors`
	- `descriptors_removed_categorical_transform`
- Updated the isolated CURATE categorical parity test to require these event payload fields.
- Updated the DAT parser helper for categorical-transform parity so descriptor-name lists are parsed and compared at event level.

Why this change was made:
- To align CURATE DAT evidence and structured event payload coverage without changing scientific behavior.

How this was tested:
- Focused test:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest 'tests/test_json_parity.py::test_curate_categorical_transform_dat_json_parity_via_new_file_only' -q`
- Full isolated suite:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest tests/test_json_parity.py -q`

Confirmed results:
- Focused CURATE categorical-transform parity test passed.
- Full isolated parity suite passed (`7 passed`).
- Scope remained narrow to the approved CURATE slice.

Next suggested step:
- Stop here until the next CURATE slice is explicitly approved.

### Entry 025

Date: 2026-07-31

Goal of this step:
- Close the approved CURATE `correlation_filter` event-field gap.

What changed:
- Expanded the CURATE `correlation_filter` event payload to retain runtime values already available during filtering:
	- `constant_descriptors_removed`
	- `low_y_correlation_descriptors_removed`
	- `high_intercorrelation_removals`
	- `descriptors_removed_correlation_filter`
	- `rfecv_selection_method_by_model`
	- `rfecv_descriptors_selected_by_model`
	- `rfecv_skip_reason`
- Updated the isolated CURATE correlation parity test to require these detailed event payload fields.

Why this change was made:
- To complete the remaining approved CURATE event-field gap while preserving scientific behavior.

How this was tested:
- Focused test:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest 'tests/test_json_parity.py::test_curate_correlation_filter_dat_json_parity_via_new_file_only' -q`
- Full isolated suite:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest tests/test_json_parity.py -q`

Confirmed results:
- Focused CURATE correlation-filter parity test passed.
- Full isolated parity suite passed (`7 passed`).
- Scope remained narrow to the approved CURATE slice.

Next suggested step:
- Stop here until the next scope is explicitly approved.

### Entry 026

Date: 2026-08-25

Goal of this step:
- Close the approved GENERATE model-scan DAT-to-JSON parity checkpoint.

What changed:
- Added the total model-cycle count to each GENERATE `model_run_start` event.
- Added the PFI error/metric type to each GENERATE `pfi_workflow` event.
- Strengthened the isolated model-scan parity test so these fields must match the corresponding DAT values.

Files changed:
- `robert/generate.py`
- `robert/generate_utils.py`
- `tests/test_json_parity.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- To retain model-scan values ROBERT already writes to `GENERATE_data.dat` without changing model calculations or output behavior.

How this was tested:
- Focused GENERATE model-scan parity test: `1 passed`.
- Full isolated JSON parity suite: `7 passed`.

Confirmed results:
- GENERATE model-cycle total and PFI metric label now have direct JSON event coverage.
- DAT-to-JSON parity passed for the focused checkpoint and the complete isolated suite.
- Existing warnings were limited to dependency deprecations and known plotting/runtime warnings.

What remains uncertain:
- Other GENERATE DAT-to-event fields remain outside this checkpoint.

Next suggested step:
- Review and approve the next GENERATE parity gap before making another implementation change.

### Entry 027

Date: 2026-08-25

Goal of this step:
- Close the first VERIFY DAT-to-JSON parity increment after the GENERATE checkpoint.

What changed:
- Added VERIFY summary-event fields for the error type, CV configuration, threshold direction, threshold percentages, and threshold values already written to `VERIFY_data.dat`.
- Extended the isolated VERIFY parity test and DAT parser to require those fields.

Files changed:
- `robert/verify.py`
- `tests/json_parity_helpers.py`
- `tests/test_json_parity.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- To retain existing VERIFY summary evidence in structured JSON without changing calculations, thresholds, or standard DAT output.

How this was tested:
- Focused VERIFY parity test: `1 passed`.
- Full isolated JSON parity suite: `7 passed`.

Confirmed results:
- The new VERIFY summary metadata matches the corresponding DAT values.
- Existing parity coverage remains passing.
- Existing dependency and plotting warnings remain non-failing.

What remains uncertain:
- Other VERIFY DAT summary fields may still need event-level coverage.

Next suggested step:
- Review and approve the next VERIFY parity gap before making another implementation change.

### Entry 029

Date: 2026-08-25

Goal of this step:
- Close the next VERIFY DAT-to-JSON parity checkpoint for flawed-model test statuses.

What changed:
- Extended the isolated DAT parser to read the `PASSED`, `UNCLEAR`, or `FAILED` labels for the y-mean, y-shuffle, and one-hot tests.
- Required an existing `analyze_tests` JSON event to match those statuses and their DAT metrics.
- No VERIFY runtime code changed because these values were already present in the JSON audit event.

Files changed:
- `tests/json_parity_helpers.py`
- `tests/test_json_parity.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- To verify that the qualitative VERIFY test outcomes written to DAT remain synchronized with structured JSON evidence.

How this was tested:
- Focused VERIFY parity test: `1 passed`.
- Full isolated JSON parity suite: `7 passed`.

Confirmed results:
- The parity assertion now covers both status labels and numeric test metrics.
- The focused and full isolated parity tests passed.

What remains uncertain:
- Other VERIFY DAT summary fields may still need event-level coverage.

Next suggested step:
- Run the focused VERIFY parity test and then the full isolated parity suite.

### Entry 030

Date: 2026-08-25

Goal of this step:
- Verify that the JSON audit preserves the VERIFY branch markers written to DAT.

What changed:
- Added a DAT parser for the No_PFI and PFI branch markers.
- Required the isolated VERIFY parity test to compare the ordered DAT branch markers with `verify_branch` JSON events.
- No ROBERT runtime code changed because the JSON branch fields already existed.

Files changed:
- `tests/json_parity_helpers.py`
- `tests/test_json_parity.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- To confirm that JSON identifies the same VERIFY analysis branches as the standard DAT output.

How this was tested:
- Focused VERIFY parity test: `1 passed`.
- Full isolated JSON parity suite: `7 passed`.

Confirmed results:
- Branch-marker parity assertion added.
- The focused and full isolated parity tests passed.

What remains uncertain:
- Other VERIFY DAT summary fields may still need event-level coverage.

Next suggested step:
- Run the focused VERIFY parity test and then the full isolated parity suite.

### Entry 028

Date: 2026-08-25

Goal of this step:
- Close the next VERIFY DAT-to-JSON parity checkpoint for sorted cross-validation metrics.

What changed:
- Extended the isolated DAT parser to read the sorted regression or classification metrics from `VERIFY_data.dat`.
- Required the existing `print_verify_summary` JSON event to match those metrics.
- No VERIFY calculation or runtime event generation logic needed to change because the values were already present in the event payload.

Files changed:
- `tests/json_parity_helpers.py`
- `tests/test_json_parity.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- To verify that the sorted-CV values written by ROBERT to DAT remain synchronized with the corresponding JSON audit evidence.

How this was tested:
- Focused VERIFY sorted-CV parity test: `1 passed`.
- Full isolated JSON parity suite: `7 passed`.

Confirmed results:
- Sorted-CV regression metrics matched between DAT and JSON.
- Existing CURATE, GENERATE, VERIFY, and PREDICT parity checks remain passing.

What remains uncertain:
- Other VERIFY DAT summary fields may still need event-level coverage.

Next suggested step:
- Review and approve the next VERIFY parity gap before making another implementation change.

### Entry 007

Date: 2026-06-08

Goal of this step:
- Run a full validation checklist using protected source-data in `databases/`.
- Record pass/fail outcomes.
- Add a design-only proposal for a future `CURATE/curate_audit.json` hook.

What changed:
- Completed one regression wrapper validation run:
	- `databases/Regression/AQME-ROBERT_A_predict_solubility.csv`
	- `y=solubility`, `names=code_name`, `ignore=code_name`
- Completed one classification wrapper validation run:
	- `databases/Clasification/F_predict_outcome.csv`
	- `y=Outcome`, `names=Name`, `ignore=Name`, `type=clas`
- Verified no JSON-layer text appears in standard `.dat` files.
- Verified `CURATE/dataset_profile.json` and `CURATE/json_output_audit.json` exist and are valid JSON.
- Verified required top-level fields in archive `run_summary.json` and CURATE JSON artifacts.
- Verified copy-only archive behavior from `wrapper_run.json`.
- Verified missing module folders are represented safely in manifests with:
	- `module_dir_exists=false`
	- `file_count=0`
- Added design-only proposal for future `CURATE/curate_audit.json` (no code hook implemented yet).

Files changed:
- `json-output-for-agent/README.md`
- `json-output-for-agent/json_schema_notes.md`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`
- `.gitignore`

Why this change was made:
- To establish a clear, reproducible validation baseline before adding more module-native JSON hooks.
- To preserve ROBERT scientific behavior while preparing a low-risk future audit pattern.

How this was tested:
- Wrapper regression run and classification run from repository root.
- JSON validity and required-field checks by Python scripts.
- DAT pollution check by searching for JSON-layer markers in standard `.dat` files.
- Archive checks for copy-only behavior and missing-module handling.

Confirmed results:
- PASS: Standard root outputs for CURATE/GENERATE/VERIFY/PREDICT were produced.
- PASS: JSON artifacts were created in standard and archive paths.
- PASS: JSON files are valid and required fields are present.
- PASS: `dataset_profile.json` includes `schema_version` and works for both regression and classification input targets.
- PASS: No JSON-layer status messages were found in standard `.dat` files.
- PASS: Copy-only archive behavior confirmed.
- PASS: Missing-file/module behavior handled in manifests.
- PARTIAL: REPORT PDF generation failed in this environment due missing WeasyPrint system libraries.

What remains uncertain:
- We still need to install/report-support system libraries if PDF generation must be included in this validation matrix.
- Module-native audit hooks beyond dataset profile remain unimplemented by design.

Next suggested step:
- Propose the smallest safe implementation plan for `CURATE/curate_audit.json` and wait for explicit approval before coding.

### Entry 008

Date: 2026-06-08

Goal of this step:
- Implement the approved minimal hook for `CURATE/curate_audit.json`.
- Keep the implementation fail-soft and additive only.

What changed:
- Added reusable helper `build_curate_audit_payload(...)` in `robert/json_output_for_agent.py`.
- Generalized `write_json_output_audit(...)` to accept an `artifact` field so events can identify either:
	- `dataset_profile.json`, or
	- `curate_audit.json`.
- Added CURATE hook in `robert/curate.py` to write `CURATE/curate_audit.json`.
- Hook records only observable evidence and marks unavailable values explicitly.
- Hook is wrapped in try/except and writes status only to `CURATE/json_output_audit.json`.

Files changed:
- `robert/json_output_for_agent.py`
- `robert/curate.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- To start module-native audit pattern in the lowest-risk module (CURATE).
- To capture module-level evidence without changing ROBERT scientific behavior.

How this was tested:
- CURATE-only regression run:
	- `python -m robert --curate --csv_name databases/Regression/AQME-ROBERT_A_predict_solubility.csv --y solubility --names code_name --ignore code_name --model "['RF']"`
- CURATE-only classification run:
	- `python -m robert --curate --csv_name databases/Clasification/F_predict_outcome.csv --y Outcome --names Name --ignore Name --type clas --model "['RF']"`
- Checked artifacts created in `CURATE/`:
	- `dataset_profile.json`
	- `curate_audit.json`
	- `json_output_audit.json`
- Verified JSON audit events include both artifacts:
	- `artifact=dataset_profile.json`
	- `artifact=curate_audit.json`
- Verified no JSON-layer markers in `CURATE/CURATE_data.dat`.

Confirmed results:
- `curate_audit.json` is created for regression and classification CURATE runs.
- `curate_audit.json` includes `schema_version` and the approved top-level sections.
- Unknown step-level descriptor-removal counters are recorded as `unavailable` (not guessed).
- Standard CURATE outputs remain unchanged.
- JSON-layer status remains isolated to JSON audit artifacts.

What remains uncertain:
- Step-specific descriptor-removal counters per filtering stage are not yet directly exposed as in-memory counters.

Next suggested step:
- If desired, add low-risk direct counters for specific CURATE filter stages so future audit files can replace selected `unavailable` fields with observed values.

### Entry 009

Date: 2026-06-08

Goal of this step:
- Run fault-injection validation specifically for `curate_audit.json` write failure.

What changed:
- Ran CURATE with a temporary monkeypatch on the exact reference used by CURATE:
	- patched `robert.curate.write_json`
	- raised `RuntimeError('simulated curate_audit write failure')` only when output path ended with `curate_audit.json`
	- allowed all other JSON writes (`dataset_profile.json`, `json_output_audit.json`) to proceed

Files changed:
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- To confirm fail-soft behavior for `curate_audit.json` writing without affecting standard CURATE outputs.

How this was tested:
- Command run:
	- `python` inline script patching `robert.curate.write_json` and executing CURATE on `databases/Regression/AQME-ROBERT_A_predict_solubility.csv`.
- Verified:
	- CURATE completed (`FAULT_INJECTION_RUN_DONE`).
	- Standard outputs existed:
		- `CURATE/CURATE_data.dat`
		- `CURATE/CURATE_options.csv`
		- curated CSV files.
	- `CURATE/dataset_profile.json` existed with schema version `0.1`.
	- `CURATE/json_output_audit.json` recorded failed `curate_audit.json` event:
		- `error_type=RuntimeError`
		- `error_message=simulated curate_audit write failure`
	- `CURATE/CURATE_data.dat` contained no JSON-layer status text.

Confirmed results:
- Fail-soft behavior for `curate_audit.json` write failure is validated.
- Standard CURATE behavior/output remained unchanged.
- JSON-layer status remained isolated to JSON audit files.

What remains uncertain:
- None specific to this fault-injection case.

Next suggested step:
- Proceed to next approved module-native audit workstream when ready.

### Entry 010

Date: 2026-06-09

Goal of this step:

* Record progress after implementing module-native runtime audits for CURATE, GENERATE, and VERIFY.
* Reset the task list so the next workstream is clear before starting PREDICT.

What changed:

* `TASKS.md` was rewritten to reflect the current module-native audit workflow.
* The task list now separates archive-side manifests from module-native runtime audit JSON files.
* The task list now identifies PREDICT as the next module to implement.
* CURATE, GENERATE, and VERIFY are marked as implemented and validated module-native audits.

Files changed:

* `json-output-for-agent/TASKS.md`
* `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:

* The previous task list still reflected the early CURATE-first phase of the project.
* The project has now moved into a module-by-module runtime audit pattern.
* Updating the tracker now prevents confusion before starting PREDICT.

How this was tested:

* Documentation update only.
* No ROBERT source code was changed in this step.
* No ROBERT run was required for this documentation checkpoint.

Confirmed results:

* CURATE runtime audit has been implemented and validated.
* GENERATE runtime audit has been implemented and validated.
* VERIFY runtime audit has been implemented and validated.
* Standard `.dat` behavior was preserved during validation, with differences limited to timestamp/runtime or non-scientific formatting.
* JSON-layer status remains isolated to `json_output_audit.json`.
* No original ROBERT scientific variables, thresholds, descriptor logic, model logic, prediction logic, plotting logic, or CLI behavior were intentionally changed.

What remains uncertain:

* PREDICT module-native audit implementation has not started.
* AQME and EVALUATE module-native audits have not started.
* REPORT remains deferred and has a separate report-generation/parser issue to revisit later.
* The long-term role of archive manifests versus module-native audits still needs to be decided.

Next suggested step:

* Commit the documentation checkpoint.
* Begin PREDICT planning only, with no implementation until the plan is reviewed and approved.


### Entry 011

Date: 2026-07-13

Goal of this step:
- Synchronize the ChatBob JSON-output branch with the current upstream ROBERT version.
- Compare an unmodified ROBERT run with a ChatBob-modified ROBERT run.
- Confirm whether the JSON-output additions change standard ROBERT scientific outputs.

What changed:
- Fetched the current upstream ROBERT repository.
- Confirmed that upstream `master` contains ROBERT version 2.1.2.
- Merged `upstream/master` into the `json-output-for-agent` branch.
- Resolved one merge conflict in `robert/generate.py`.
- Preserved both:
  - the upstream classification handling added in ROBERT 2.1.2, and
  - the additive ChatBob GENERATE audit capture.
- Added a Jupyter notebook for comparing completed ROBERT output folders.
- Compared an unmodified ROBERT 2.1.2 run with the ChatBob-modified ROBERT 2.1.2 run.

Files changed:
- ROBERT files brought in through the merge from `upstream/master`.
- `robert/generate.py`, where the merge conflict was resolved.
- `comparison/compare_robert_outputs.ipynb`.
- Project documentation files associated with this update.

Why this change was made:
- An initial comparison mistakenly compared ROBERT 2.1.2 against the ChatBob branch while it was still based on ROBERT 2.1.0.
- That version mismatch produced differences in prediction values and outlier reporting.
- Synchronizing both runs to ROBERT 2.1.2 was necessary to isolate the effect of the ChatBob JSON additions.

How this was tested:
- Ran the same regression dataset through:
  - unmodified ROBERT 2.1.2, and
  - ChatBob-modified ROBERT 2.1.2.
- Compared the four primary `.dat` files:
  - `CURATE/CURATE_data.dat`
  - `GENERATE/GENERATE_data.dat`
  - `VERIFY/VERIFY_data.dat`
  - `PREDICT/PREDICT_data.dat`
- Normalized only expected run-specific values:
  - timestamps,
  - input paths,
  - output paths,
  - execution times,
  - trailing whitespace.
- Compared 28 matching CSV files.
- Numeric CSV values were compared within a strict floating-point tolerance.
- Text values were compared exactly.

Confirmed results:
- PASS: All four normalized `.dat` files were identical.
- PASS: All scientific CSV files were identical within numerical tolerance.
- PASS: CURATE outputs were unchanged.
- PASS: GENERATE model files and selected best-model files were unchanged.
- PASS: VERIFY outputs were unchanged.
- PASS: PREDICT values and uncertainty values were unchanged.
- EXPECTED DIFFERENCE: `CURATE/CURATE_options.csv` recorded a different input file path.
- EXPECTED DIFFERENCE: Raw `.dat` files contained different timestamps, paths, and execution times.
- The earlier PREDICT differences were caused by comparing ROBERT 2.1.0 with ROBERT 2.1.2, not by the ChatBob JSON additions.

What this establishes:
- For this regression test case, the ChatBob JSON-output implementation is additive.
- It does not change the standard ROBERT scientific `.dat` or CSV outputs.

What remains uncertain:
- The same controlled comparison has not yet been documented for classification.
- The current JSON structure still needs to be reviewed against the questions ChatBob should answer.
- The long-term location and integration of the JSON outputs should be agreed with Juanvi.
- A pull request should be reviewed before the branch diverges further from upstream.

Next suggested step:
- Open a pull request from `json-output-for-agent` for review.
- Share the validation result with Juanvi.
- Agree on the JSON output location and integration strategy.
- Inspect the current JSON structures and map a small set of likely user questions to the evidence required to answer them.


### Entry 012

Date: 2026-07-13

Goal of this step:
- Reconcile documentation with current source implementation status after post-session coding.
- Apply the confirmed architecture decision for artifact location.

What changed:
- Confirmed from source that module-native runtime audits are implemented for:
	- CURATE,
	- GENERATE,
	- VERIFY,
	- PREDICT.
- Confirmed from source that module-native runtime audits are not implemented for:
	- AQME,
	- EVALUATE.
- Confirmed REPORT currently writes figure provenance only (no full report audit).
- Confirmed ChatBob runtime JSON artifacts are written to the top-level `JSON/` folder.
- Updated project docs to remove stale status/path ambiguity and preserve the distinction between:
	- module-native runtime audits,
	- JSON write-status audits,
	- wrapper-generated archive manifests,
	- standard ROBERT outputs.

Files changed:
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/AGENTS.md`
- `json-output-for-agent/json_schema_notes.md`
- `json-output-for-agent/output_map.md`
- `json-output-for-agent/README.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- Earlier task/status documents were stale after implementation continued outside the previous agent session.
- Documentation now reflects source-verified implementation status and canonical JSON artifact location.

How this was tested:
- Source inspection only (no ROBERT run, no tests).

Confirmed results:
- Runtime audit paths are documented using `JSON/` examples.
- PREDICT is documented as implemented with validation still incomplete.
- AQME and EVALUATE remain documented as not implemented.
- REPORT is documented as figure-provenance-only at this stage.

What remains uncertain:
- Full audit-content validation for PREDICT remains pending.
- AQME and EVALUATE audit validation remains pending because implementation is not present.

Next suggested step:
- Prepare a draft pull request with explicit validation boundaries and open design decisions.


### Entry 013

Date: 2026-07-14

Goal of this step:
- Record newly completed manual verification performed outside the agent session.

What changed:
- Confirmed manual review that structured JSON files were generated for:
	- CURATE,
	- GENERATE,
	- VERIFY,
	- PREDICT,
	- REPORT.
- Confirmed manual in-repo run comparison against original ROBERT DAT files showed no DAT differences.
- Updated status docs so this verification is explicitly documented.

Files changed:
- `json-output-for-agent/README.md`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- The verification was completed outside the agent and needed to be captured as project evidence.

How this was tested:
- Documentation update only in this step.
- Verification evidence source: user-confirmed manual run and DAT comparison in this repository.

Confirmed results:
- Structured JSON generation was reviewed for major workflow stages.
- DAT comparison against original ROBERT reported no differences.

What remains uncertain:
- Automated tests for module-audit content are still pending.
- AQME and EVALUATE module-native audits remain not implemented.

Next suggested step:
- Prepare draft pull request text that includes both the 2026-07-13 controlled comparison and 2026-07-14 in-repo manual verification.


### Entry 014

Date: 2026-07-28

Goal of this step:
- Add an isolated DAT↔JSON parity test harness without changing legacy pytest scripts.

What changed:
- Added shared helper file for DAT parsing and JSON audit/event extraction.
- Added a new standalone pytest module that runs CURATE→GENERATE and checks DAT↔JSON parity.
- Implemented event-first parity checks for load counts where section values may be overwritten by repeated calls.
- Kept ignored/discarded checks section-based where event payload fields are not currently emitted.

Files changed:
- `tests/json_parity_helpers.py`
- `tests/test_json_parity.py`

Why this change was made:
- The project needed parity validation between human-readable DAT outputs and machine-readable JSON audits.
- The user requested these checks to live in new test files only, with no edits to existing pytest modules in this step.

How this was tested:
- Executed focused test run:
	- `python -m pytest tests/test_json_parity.py -q`

Confirmed results:
- PASS: `1 passed` for the new standalone parity test module.
- PASS: Parity checks are isolated to new files under `tests/`.
- PASS: Existing pytest scripts were not modified as part of this isolated parity step.

What remains uncertain:
- PREDICT and VERIFY parity coverage has not yet been added to the standalone harness.
- Some descriptor-related parity fields are still section-only because matching event payload keys are not currently emitted.

Next suggested step:
- Expand the standalone parity harness to include PREDICT and VERIFY with the same event-first assertion strategy.


### Entry 015

Date: 2026-07-28

Goal of this step:
- Start a tracked one-by-one sweep of all DAT↔JSON parallel evidence points across the project.

What changed:
- Mapped initial DAT↔JSON parity targets from source for CURATE, GENERATE, VERIFY, and PREDICT.
- Added a dedicated task phase for parity-sweep tracking in `TASKS.md` with explicit per-checkpoint checklist items.
- Upgraded parity helper utilities to support reusable labeled DAT-block parsing and generic event payload parity assertions.
- Refactored the standalone parity test to use the generalized helper pathway.

Files changed:
- `json-output-for-agent/TASKS.md`
- `tests/json_parity_helpers.py`
- `tests/test_json_parity.py`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- The project needs to identify every DAT-facing calculation that should align with JSON audit evidence.
- The user requested that parity work be tracked and completed one target at a time.
- A shared helper-first pattern reduces duplicated test code and makes new parity targets faster to add.

How this was tested:
- Focused pytest run of the standalone parity module (after helper refactor).

Confirmed results:
- The parity harness now supports generalized DAT label parsing and generic event payload matching.
- The first checked target (GENERATE load-database parity) remains covered through the generalized helper path.

What remains uncertain:
- VERIFY and PREDICT parity targets still need dedicated standalone tests.
- Some checks remain section-based where event payload keys are not yet present.

Next suggested step:
- Implement the next standalone parity target: VERIFY load-database and summary parity.


### Entry 016

Date: 2026-07-28

Goal of this step:
- Complete the next one-by-one parity target by validating VERIFY DAT↔JSON alignment.

What changed:
- Extended parity helpers to parse VERIFY summary metrics directly from DAT text.
- Added a rounded numeric event matcher for robust DAT↔JSON comparisons when DAT text is rounded for display.
- Added a new standalone VERIFY parity test (in existing new parity test module) that:
	- runs CURATE→GENERATE→VERIFY,
	- validates VERIFY `load_database` DAT counts against `verify_audit` `load_database` events,
	- validates VERIFY summary metrics in DAT against `verify_audit` `print_verify_summary` events.
- Updated task checklist status to mark VERIFY load and summary parity checkpoints complete.

Files changed:
- `tests/json_parity_helpers.py`
- `tests/test_json_parity.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- The project needs tracked closure of DAT↔JSON parity targets one at a time.
- VERIFY was selected as the next checkpoint after GENERATE.

How this was tested:
- Executed focused parity suite:
	- `python -m pytest tests/test_json_parity.py -q`

Confirmed results:
- PASS: parity suite now reports `2 passed`.
- PASS: VERIFY load-database parity is validated through standalone tests.
- PASS: VERIFY summary-metric parity is validated through standalone tests.

What remains uncertain:
- PREDICT parity targets are still pending.
- CURATE correlation-filter and categorical-transform parity targets are still pending.

Next suggested step:
- Implement PREDICT load-database and summary parity in the same helper-driven standalone framework.


### Entry 017

Date: 2026-07-28

Goal of this step:
- Complete the next one-by-one parity target by validating PREDICT DAT↔JSON alignment.

What changed:
- Extended parity helpers with PREDICT-specific DAT parsers for:
	- external-set load count lines,
	- summary metric lines in the PREDICT DAT report.
- Added helper logic to compare parsed PREDICT DAT summary metrics against `print_predict_summary` audit events.
- Added a standalone PREDICT parity test that runs CURATE→GENERATE→PREDICT (`csv_test` path) and asserts:
	- external-set `load_database` datapoint parity,
	- PREDICT summary metric parity.
- Updated the phase checklist to mark PREDICT load and summary parity checkpoints complete.

Files changed:
- `tests/json_parity_helpers.py`
- `tests/test_json_parity.py`
- `json-output-for-agent/TASKS.md`
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- The parity sweep requires explicit closure of DAT↔JSON targets one by one.
- PREDICT was the next tracked target after VERIFY.

How this was tested:
- Executed focused parity suite:
	- `python -m pytest tests/test_json_parity.py -q`

Confirmed results:
- PASS: parity suite now reports `3 passed`.
- PASS: PREDICT external-set load-database parity is validated.
- PASS: PREDICT summary-metric parity is validated.

What remains uncertain:
- CURATE `correlation_filter` and `categorical_transform` parity targets are still pending.
- GENERATE model-scan summary parity target is still pending.

Next suggested step:
- Implement CURATE `correlation_filter` and `categorical_transform` DAT↔JSON parity checks in the standalone framework.


### Entry 019

Date: 2026-07-28

Goal of this step:
- Re-establish trust by stating in plain English what parity checks do, what they do not do, and what confidence level is justified right now.

What changed:
- Added this trust-summary entry to make project status understandable without reading test code.
- Clarified the exact basis of current parity checks:
	1) Parse values from DAT text,
	2) read corresponding JSON audit values,
	3) compare those values for agreement.
- Clarified that one critical checkpoint now uses a third independent oracle:
	- CURATE `load_database` values are recomputed independently from input CSV and options,
	- then compared against both DAT and JSON.

Files changed:
- `json-output-for-agent/task_tracker_plain_english.md`

Why this change was made:
- The user requested a plain-English trust reset and clear explanation of what parity means.
- Without this, progress can look like "tests passing" without clear meaning.

How this was tested:
- Documentation update only for this entry.
- Technical basis referenced from passing parity suite:
	- `/Users/cjcscha/mambaforge/envs/cheminf/bin/python -m pytest tests/test_json_parity.py -q`
	- current result: `4 passed`.

Confirmed results:
- What we can trust now:
	- JSON artifacts are being produced for implemented modules.
	- For covered checkpoints, DAT and JSON currently agree on the tested fields.
	- CURATE `load_database` now has stronger anti-GIGO validation because both DAT and JSON are checked against an independent recomputation.
- What we cannot claim yet:
	- Parity is not full scientific correctness proof.
	- Remaining unchecked checkpoints can still hide mismatches.
	- Shared upstream logic could still produce matching but wrong values in places where third-oracle checks are not yet added.

What remains uncertain:
- CURATE `correlation_filter` parity still pending.
- CURATE `categorical_transform` parity still pending.
- GENERATE model-scan summary parity still pending.
- Third-oracle coverage is currently strong for CURATE `load_database`, but not yet expanded across all parity targets.

Next suggested step:
- Continue one-by-one with the next highest-value trust checkpoint:
	- CURATE `correlation_filter` parity,
	- then CURATE `categorical_transform`,
	- then GENERATE model-scan summary,
- while adding independent-oracle checks where feasible.


---

## Template for Future Entries

### Entry XXX

Date: YYYY-MM-DD

Goal of this step:
-

What changed:
-

Files changed:
-

Why this change was made:
-

How this was tested:
-

Confirmed results:
-

What remains uncertain:
-

Next suggested step:
-
