# TASKS.md — ROBERT JSON Output Tasks

## Project Status

Status: Active module-native audit implementation.

Goal: Add structured JSON outputs to ROBERT with minimal changes to existing code.

Core principle:

* ROBERT remains the scientific source of truth.
* JSON files record runtime evidence that ROBERT already creates.
* JSON writes are additive and fail-soft.
* Standard ROBERT `.dat`, `.csv`, image, and report behavior should remain unchanged.
* No ROBERT model logic, descriptor logic, thresholds, scoring logic, prediction logic, plotting logic, or CLI behavior should be changed unless explicitly approved.

---

## Current Priority

Prepare the additive JSON-output implementation for upstream review while keeping the schema experimental.

Immediate priorities:

1. Open a pull request before additional upstream conflicts accumulate.
2. Document the controlled ROBERT 2.1.2 run comparison.
3. Complete documentation reconciliation with source-verified module status.
4. Review the current JSON structures against a small set of ChatBob user questions.
5. Agree with Juanvi on output location, branch integration, and the scope of the first schema.

---

## Validation Standard for Each Module

For each module-native audit artifact in `JSON/`, confirm:

* [ ] Standard `.dat` output is unchanged except timestamp/runtime.
* [ ] Audit JSON opens as valid JSON.
* [ ] Module `*_json_output_audit.json` records successful write.
* [ ] JSON-layer text does not appear in standard `.dat` output.
* [ ] Git diff shows additive audit capture only.
* [ ] No ROBERT thresholds changed.
* [ ] No ROBERT scoring logic changed.
* [ ] No ROBERT descriptor logic changed.
* [ ] No ROBERT model-selection logic changed.
* [ ] No ROBERT prediction logic changed.
* [ ] No ROBERT plotting behavior changed.
* [ ] No ROBERT CLI behavior changed.

---

## Module-Native Audit Status

This section tracks JSON files written during the ROBERT module run in `JSON/`, not archive-side manifests created after copying outputs.

| Module   | Audit artifact                                       | Status          | Validation status | Notes                                                                                                                                   |
| -------- | ---------------------------------------------------- | --------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| CURATE   | `JSON/curate_audit.json`                             | Implemented     | Validated         | Captures CURATE runtime evidence and curated-output metadata.                                                                           |
| GENERATE | `JSON/generate_audit.json`                           | Implemented     | Validated         | Captures model generation, BO, PFI, best-model selection, outputs, and heatmap artifacts.                                               |
| VERIFY   | `JSON/verify_audit.json`                             | Implemented     | Validated         | Captures VERIFY branch context, model context, y-mean, y-shuffle, one-hot tests, thresholds, pass/fail status, plots, and summary text. |
| PREDICT  | `JSON/predict_audit.json`                            | Implemented     | Validated (manual) | Source implementation confirmed; manual artifact review and DAT parity verification completed in-repo.                                   |
| AQME     | `JSON/aqme_audit.json`                               | Not implemented | Not validated     | Later module.                                                                                                                           |
| EVALUATE | `JSON/evaluate_audit.json`                           | Not implemented | Not validated     | Later module.                                                                                                                           |
| REPORT   | `JSON/report_figure_provenance.json`                 | Partial         | Not validated     | Figure provenance is implemented; full module-native report audit is not implemented.                                                   |

Notes:

* Archive manifests and `run_summary.json` are copy-side artifacts.
* Module-native audits are runtime evidence artifacts.
* Module-native audits and JSON write-status files are written to the top-level `JSON/` folder.
* Both are useful, but they answer different questions.
* Archive manifests tell us what files were produced and copied.
* Module-native audits tell us what ROBERT knew while it was running.

---

## Phase 1 — Inspect Existing ROBERT Outputs

### To Do

* [x] Identify where CURATE writes output files.
* [x] Identify where GENERATE writes output files.
* [x] Identify where VERIFY writes output files.
* [x] Identify where PREDICT writes output files.
* [x] Identify where REPORT writes the PDF report.
* [x] Identify where `.dat` files are created.
* [x] Identify where images are created.
* [x] Identify where the ROBERT score is calculated.
* [x] Identify which values are already available in memory before being written to files.
* [x] Identify the smallest safe places to call JSON helper functions.

### Done

* [x] Output map inspection completed on 2026-05-27.
* [x] Created `json-output-for-agent/output_map.md` with:

  * module-by-module output write map,
  * `.dat` file creation map,
  * plot/image output map,
  * ROBERT score/report flow map,
  * safest future JSON insertion points,
  * risks and first implementation proposal.

---

## Phase 1.5 — Incoming Dataset Profile JSON

This phase added one JSON artifact that profiles observable facts from the incoming dataset before ROBERT modifies the data.

### To Do

* [x] Confirm insertion point in `curate.__init__`.
* [x] Confirm helper file location: `robert/json_output_for_agent.py`.
* [x] Confirm JSON output location: `JSON/dataset_profile.json`.
* [x] Enforce standard ROBERT output protection rule for JSON layer.
* [x] Implement `profile_input_dataset` helper function.
* [x] Implement dataset profile measurement helper functions.
* [x] Implement safe JSON writing helper function.
* [x] Implement JSON output audit helper function.
* [x] Add single JSON helper call in `curate.__init__`.
* [x] Confirm `JSON/dataset_profile.json` is created.
* [x] Confirm CURATE still completes when JSON write fails.
* [x] Confirm JSON status is written only to `JSON/curate_json_output_audit.json`.
* [x] Confirm `CURATE/CURATE_data.dat` contains no JSON-layer status messages.

### Done

* [x] Implemented `robert/json_output_for_agent.py`.
* [x] Added fail-soft hook in `robert/curate.py`.
* [x] Validated baseline run creates `JSON/dataset_profile.json`.
* [x] Injected simulated JSON failure and confirmed standard CURATE outputs still complete.
* [x] Removed JSON-layer writes to standard CURATE logger output.
* [x] Added `JSON/curate_json_output_audit.json` as the only JSON-layer status channel.

---

## Phase 2 — JSON Structure and Schema Notes

### To Do

* [ ] Decide final project folder name.
* [ ] Decide whether JSON export is always on or controlled by an option.
* [ ] Draft stable module-audit JSON schema after more modules are implemented.
* [ ] Draft stable run-level JSON schema after module-native audits are better understood.
* [ ] Decide how much full-value data should be included versus summarized.
* [ ] Decide long-term naming conventions for direct ROBERT values and JSON-only explanatory aliases.
* [ ] Decide whether timestamps should always use UTC.
* [ ] Decide whether module-native audit artifacts should eventually be indexed by a run-level context file.

### Done

* [x] Experimental schema notes started in `json_schema_notes.md`.
* [x] Current schema version marked as experimental: `schema_version: 0.1`.
* [x] Pattern established:

  * module name,
  * artifact type,
  * status,
  * input/settings section,
  * event stream,
  * runtime section,
  * JSON-layer write status in `json_output_audit.json`.

---

## Phase 3 — Build JSON Helper

### To Do

* [x] Create `robert/json_output_for_agent.py`.
* [x] Add safe JSON writing helper.
* [x] Add JSON-safe conversion helper for common Python, NumPy, pandas, and pathlib objects.
* [x] Add file manifest helper.
* [x] Add module summary helper.
* [x] Add run-level summary helper.
* [x] Add module audit helper pattern:

  * `init_module_audit`
  * `audit_set`
  * `audit_event`
  * `finalize_module_audit`
  * `write_json_output_audit`
* [ ] Add plain-English comments to helper functions where needed.
* [ ] Add simple tests for helper functions.

### Done

* [x] Core dataset-profile helper implemented.
* [x] Archive manifest and run summary helpers implemented.
* [x] Module-native audit helper pattern implemented and reused across CURATE, GENERATE, and VERIFY.

---

## Phase 4 — CURATE Module-Native Audit

### To Do

* [x] Choose CURATE as first module-native audit target.
* [x] Propose exact files and functions to change.
* [x] Wait for user approval.
* [x] Implement `JSON/curate_audit.json`.
* [x] Confirm existing ROBERT CURATE output is unchanged.
* [x] Confirm audit JSON output is created.
* [x] Confirm JSON output audit is created.
* [x] Confirm no JSON text appears in `CURATE/CURATE_data.dat`.

### Done

* [x] `JSON/curate_audit.json` implemented.
* [x] CURATE runtime evidence captured.
* [x] CURATE JSON write status isolated to `JSON/curate_json_output_audit.json`.
* [x] CURATE validation completed.
* [x] CURATE fault-injection validation completed for audit write failure.

---

## Phase 5 — GENERATE Module-Native Audit

### To Do

* [x] Propose implementation plan for GENERATE audit.
* [x] Wait for user approval.
* [x] Implement `JSON/generate_audit.json`.
* [x] Capture inputs and model-scan context.
* [x] Capture BO workflow evidence.
* [x] Capture PFI workflow evidence.
* [x] Capture best-model selection evidence.
* [x] Capture heatmap/image artifact metadata.
* [x] Capture outputs and runtime.
* [x] Confirm existing `GENERATE_data.dat` is unchanged except timestamp/runtime and non-scientific formatting differences.
* [x] Confirm key CSV outputs are unchanged.
* [x] Confirm audit JSON validates.
* [x] Confirm JSON output audit validates.
* [x] Confirm no JSON text appears in `GENERATE/GENERATE_data.dat`.

### Done

* [x] `JSON/generate_audit.json` implemented.
* [x] `JSON/generate_json_output_audit.json` implemented for audit-write status.
* [x] Standard GENERATE behavior validated against accepted baseline.
* [x] GENERATE audit captures:

  * input settings,
  * model list,
  * model-specific CURATE file use,
  * BO workflow,
  * optimized parameters,
  * PFI descriptor filtering,
  * best-model selection,
  * heatmap artifact metadata,
  * output file paths,
  * runtime.

---

## Phase 6 — VERIFY Module-Native Audit

### To Do

* [x] Propose implementation plan for VERIFY audit.
* [x] Wait for user approval.
* [x] Implement `JSON/verify_audit.json`.
* [x] Capture inputs and threshold constants.
* [x] Capture No_PFI and PFI branch context.
* [x] Capture model context.
* [x] Capture y-mean test evidence.
* [x] Capture y-shuffle test evidence.
* [x] Capture one-hot test evidence.
* [x] Capture threshold/status analysis.
* [x] Capture VERIFY plot artifact metadata.
* [x] Capture final `.dat` summary preview.
* [x] Add JSON-only explanatory descriptor fields where useful.
* [x] Confirm existing `VERIFY_data.dat` is unchanged except timestamp/runtime.
* [x] Confirm audit JSON validates.
* [x] Confirm JSON output audit validates.
* [x] Confirm no JSON text appears in `VERIFY/VERIFY_data.dat`.

### Done

* [x] `JSON/verify_audit.json` implemented.
* [x] `JSON/verify_json_output_audit.json` implemented for audit-write status.
* [x] Standard VERIFY behavior validated against accepted baseline.
* [x] VERIFY audit captures:

  * branch context,
  * model context,
  * active descriptor evidence,
  * cross-validation and sorted-CV metrics,
  * y-mean test,
  * y-shuffle test,
  * one-hot test,
  * thresholds,
  * PASS/UNCLEAR/FAILED status,
  * plot artifact metadata,
  * final summary text preview,
  * runtime.
* [x] Confirmed descriptor-related JSON fields are explanatory JSON metadata only and do not rename or change original ROBERT variables.

---

## Phase 7 — PREDICT Module-Native Audit

### To Do

* [x] Inspect current `predict.py` and `predict_utils.py` insertion points.
* [x] Propose implementation plan before coding.
* [x] Wait for user approval.
* [x] Implement `JSON/predict_audit.json`.
* [x] Capture input settings and model/source context.
* [x] Capture prediction CSV output metadata.
* [x] Capture prediction summary metrics.
* [x] Capture train/validation/test/external prediction evidence when available.
* [x] Capture uncertainty or variability evidence when available.
* [x] Capture SHAP artifact metadata when available.
* [x] Capture PFI artifact metadata when available.
* [x] Capture outlier evidence when available.
* [x] Capture y-distribution artifact metadata when available.
* [x] Capture Pearson heatmap metadata when available.
* [x] Capture generated plot/image paths.
* [x] Capture final `.dat` summary preview if useful.
* [ ] Confirm existing `PREDICT_data.dat` is unchanged except timestamp/runtime.
* [ ] Confirm prediction CSV outputs are unchanged.
* [ ] Confirm image outputs are unchanged or only differ by normal timestamp/rendering metadata.
* [ ] Confirm audit JSON validates.
* [ ] Confirm JSON output audit validates.
* [ ] Confirm no JSON text appears in `PREDICT/PREDICT_data.dat`.

### Done

* [x] `JSON/predict_audit.json` implemented.
* [x] `JSON/predict_json_output_audit.json` implemented for audit-write status.
* [x] Manual review confirmed structured JSON generation for PREDICT in the current repo.
* [x] Manual in-repo comparison against original ROBERT DAT files found no DAT differences.
* [ ] Add formal automated validation tests for PREDICT audit content.

---

## Phase 8 — AQME Module-Native Audit

### To Do

* [ ] Inspect current `aqme.py` insertion points.
* [ ] Propose implementation plan before coding.
* [ ] Wait for user approval.
* [ ] Implement `JSON/aqme_audit.json`.
* [ ] Capture input settings.
* [ ] Capture descriptor-generation provenance where available.
* [ ] Capture output file metadata.
* [ ] Capture any generated `.dat` summary preview if useful.
* [ ] Confirm existing AQME outputs are unchanged.
* [ ] Confirm audit JSON validates.
* [ ] Confirm JSON output audit validates.
* [ ] Confirm no JSON text appears in standard AQME outputs.

### Done

* [ ] Not started.

---

## Phase 9 — EVALUATE Module-Native Audit

### To Do

* [ ] Inspect current `evaluate.py` insertion points.
* [ ] Propose implementation plan before coding.
* [ ] Wait for user approval.
* [ ] Implement `JSON/evaluate_audit.json`.
* [ ] Capture input settings.
* [ ] Capture model evaluation context.
* [ ] Capture model parameter/output metadata.
* [ ] Capture evaluation metrics where available.
* [ ] Confirm existing EVALUATE outputs are unchanged.
* [ ] Confirm audit JSON validates.
* [ ] Confirm JSON output audit validates.
* [ ] Confirm no JSON text appears in standard EVALUATE outputs.

### Done

* [ ] Not started.

---

## Phase 10 — REPORT and Run-Level Context

### To Do

* [ ] Revisit REPORT only after upstream module-native audits are stable.
* [ ] Decide whether REPORT needs its own `JSON/report_audit.json`.
* [ ] Document the currently implemented report artifact `JSON/report_figure_provenance.json`.
* [ ] Decide whether a run-level context JSON can replace some fragile report parsing.
* [ ] Investigate current REPORT PDF/report parser failure separately.
* [ ] Confirm required WeasyPrint/system-library environment for PDF generation if PDF validation is needed.
* [ ] Decide whether REPORT should consume module-native JSON audits in a future design.

### Done

* [ ] REPORT module-native audit not started.
* [x] `JSON/report_figure_provenance.json` is implemented as report-figure provenance capture.
* [ ] Current REPORT issue identified as separate from CURATE, GENERATE, VERIFY, and PREDICT audit validation.

---

## Phase 11 — Timestamped Archive Wrapper

Goal:

* Keep normal ROBERT root outputs unchanged.
* Duplicate generated outputs into a timestamped project archive folder.

### To Do

* [x] Confirm feasibility outside ROBERT core.
* [x] Implement wrapper script in project folder.
* [x] Ensure wrapper runs ROBERT in normal root mode.
* [x] Ensure wrapper copies outputs into timestamped run folder.
* [x] Generate per-module JSON manifests from copied run artifacts.
* [x] Generate run-level `run_summary.json` in the archive.
* [x] Document explicit copy-only rule in AGENTS and project rules docs.
* [x] Validate on full runs and inspect copied archive contents.
* [ ] Decide long-term role of archive manifests now that module-native audits are being implemented.

### Done

* [x] Added wrapper script: `json-output-for-agent/scripts/run_robert_timestamped.py`.
* [x] Added policy language that standard output paths remain unchanged and archive behavior is copy-only.
* [x] Added archive-side manifest generation and run summary generation after copy.
* [x] Confirmed copy-only behavior.

---

## Phase 12 — Documentation and Tracking

### To Do

* [x] Document what JSON files are created.
* [x] Document where JSON files are saved.
* [x] Document what each top-level field means.
* [x] Document whether the JSON schema is experimental.
* [x] Add a short plain-English explanation for chemists.
* [x] Update `task_tracker_plain_english.md`.
* [x] Update `TASKS.md`.
* [ ] Update `json_schema_notes.md` to reflect CURATE, GENERATE, and VERIFY implementation status.
* [ ] Update `output_map.md` to mark implemented module-native hooks.
* [ ] Update `README.md` to reflect current module-native audit status.
* [ ] Update `PROJECT_RULES.md` only if process rules change.

### Done

* [x] Initial validation and schema documentation updated on 2026-06-08.
* [x] `TASKS.md` rewritten to reflect current module-native audit workflow.

---

## Phase 13 — Controlled ROBERT Run Equivalence Validation

### To Do

* [x] Confirm both comparison runs use the same ROBERT version.
* [x] Synchronize the ChatBob branch with upstream ROBERT 2.1.2.
* [x] Run the same regression dataset through original and ChatBob ROBERT.
* [x] Compare the four primary `.dat` files exactly.
* [x] Normalize only timestamps, paths, runtime, and trailing whitespace.
* [x] Confirm normalized `.dat` files are identical.
* [x] Compare matching CSV files by shape, columns, text, and numeric values.
* [x] Confirm best-model CSV files are identical.
* [x] Confirm prediction CSV values are identical.
* [x] Record the expected `csv_name` path difference in `CURATE_options.csv`.
* [ ] Repeat the controlled comparison for one classification example.
* [ ] Convert the comparison notebook checks into formal automated tests if appropriate.

### Done

* [x] Regression comparison completed using ROBERT 2.1.2 for both runs.
* [x] Four normalized `.dat` files were identical.
* [x] Scientific values in 28 matching CSV files were identical.
* [x] ChatBob JSON-output additions did not change standard ROBERT scientific outputs for this test case.
* [x] Additional manual in-repo verification (2026-07-14) reported no DAT differences versus original ROBERT.

---

## Phase 14 — Upstream Integration and Schema Review

### To Do

* [ ] Open a pull request from `json-output-for-agent`.
* [ ] Share the run-comparison evidence with Juanvi.
* [ ] Agree on whether the feature should merge into `master`, a development branch, or remain behind a feature branch temporarily.
* [ ] Agree on the official JSON output location.
* [ ] Inventory current JSON filenames, top-level keys, event types, and payload structures.
* [ ] Define five initial user questions ChatBob should answer.
* [ ] Map each question to its JSON evidence and original ROBERT source.
* [ ] Identify the minimum schema changes required.
* [ ] Avoid freezing a large schema before the question-to-evidence map is reviewed.
* [ ] Build deterministic evidence-selection functions before adding LLM answer generation.

---

## Phase 15 — Isolated DAT↔JSON Parity Test Harness

### To Do

* [x] Keep all new parity checks in newly created files under `tests/`.
* [x] Avoid edits to existing pytest scripts for this parity step.
* [x] Add reusable helper utilities for DAT parsing and JSON event extraction.
* [x] Add a standalone parity test that executes CURATE→GENERATE and validates DAT↔JSON agreement.
* [x] Prefer event-based assertions for repeated operations and use section checks only where event payload fields are not yet emitted.
* [x] Validate the new parity test with focused pytest execution.
* [x] Extend standalone parity coverage to PREDICT and VERIFY.
* [ ] Decide whether selected parity checks should eventually merge into legacy test suites after stabilization.

### Done

* [x] Added `tests/json_parity_helpers.py` for shared DAT parsing, JSON loading, and parity assertions.
* [x] Added `tests/test_json_parity.py` with isolated parity flow and local cleanup helpers.
* [x] Confirmed focused test success with `python -m pytest tests/test_json_parity.py -q` (`1 passed`).
* [x] Preserved non-invasive scope: no edits required to existing pytest scripts for this step.

---

## Phase 16 — DAT↔JSON Parity Sweep (Tracked One-by-One)

Goal:

* Inventory every location where ROBERT writes user-facing DAT evidence in parallel with JSON audit evidence.
* Track each parity checkpoint explicitly and close them one-by-one.
* Maximize reuse by expanding common parity helpers instead of adding bespoke test logic per checkpoint.

### To Do

* [x] Create an initial parity target inventory from source inspection.
* [x] Promote reusable DAT parsing helpers to support labeled block parsing.
* [x] Promote reusable JSON-event helpers for generic event payload parity assertions.
* [x] Track and validate `load_database` parity for CURATE.
* [x] Track and validate `load_database` parity for GENERATE.
* [x] Track and validate `load_database` parity for VERIFY.
* [x] Track and validate `load_database` parity for PREDICT.
* [x] Track and validate `correlation_filter` parity (CURATE).
* [x] Track and validate `categorical_transform` parity (CURATE).
* [x] Track and validate model-scan summary parity in GENERATE (`model_scan` / BO summary evidence).
* [x] Track and validate summary-metrics parity in VERIFY.
* [x] Track and validate summary-metrics parity in PREDICT.
* [ ] Decide whether missing event payload keys should be added where section-only checks are still required.
* [ ] Keep parity checks isolated to new test files until stabilization is complete.

### Initial Inventory (Source-Mapped Targets)

* `robert/utils.py::load_database` (DAT load counts + JSON `load_database` section/event)
* `robert/utils.py::correlation_filter` (DAT filter summary + JSON `correlation_filter` section/event)
* `robert/utils.py::categorical_transform` (DAT categorical summary + JSON `categorical_transform` section/event)
* `robert/generate.py` and `robert/generate_utils.py` (DAT model-scan progress/summary + JSON `generate_audit` events/sections)
* `robert/verify.py` (DAT verify-branch/test-result reporting + JSON `verify_audit` events/sections)
* `robert/predict.py` and `robert/predict_utils.py` (DAT prediction/result reporting + JSON `predict_audit` events/sections)

### Done

* [x] Added target-driven helper capabilities in `tests/json_parity_helpers.py` for reusable parity expansion.
* [x] Updated `tests/test_json_parity.py` to use generalized labeled DAT parsing and generic event payload checks.
* [x] Added standalone VERIFY parity checks in `tests/test_json_parity.py` for:

  * `load_database` DAT↔JSON count parity,
  * `print_verify_summary` DAT↔JSON summary metric parity.
* [x] Added standalone PREDICT parity checks in `tests/test_json_parity.py` for:

  * external-set `load_database` DAT↔JSON count parity (`csv_test` path),
  * `print_predict_summary` DAT↔JSON metric parity.
* [x] Added standalone CURATE third-oracle load-database parity checks in `tests/test_json_parity.py` that validate DAT and JSON against independent recomputation logic in `tests/json_parity_helpers.py`.
* [x] Added GENERATE model-scan parity coverage for model-cycle totals and PFI metric labels.
* [x] Added VERIFY summary-header parity coverage for error type, CV configuration, threshold direction, percentages, and values.
* [x] Added VERIFY sorted cross-validation metric parity coverage.
* [x] Added VERIFY flawed-test status and metric parity coverage.
* [x] Added VERIFY branch-marker parity coverage.
* [x] Added VERIFY model-context parity coverage.
* [x] Added VERIFY direct-test event parity coverage.
* [x] Added PREDICT summary-count and proportion parity coverage.

---

## Phase 17 — DAT-to-Event Payload Coverage Hardening

Status:

* DAT-to-event gap inventory identified.
* No runtime code implementation approved yet beyond the narrow CURATE slice below.
* Remaining module gaps are tracked as future reviewable increments.

Goal:

* Capture every meaningful DAT value that is not yet retained in the corresponding JSON event payload.
* Keep the review sequence narrow, explicit, and scientific-output safe.

### Identified Gaps

* CURATE
  * `load_database` event fields
  * `categorical_transform` event fields
  * `correlation_filter` event fields
* GENERATE
  * model-scan summary parity tied to `model_run_start`, `bo_workflow`, and `pfi_workflow` event payloads
  * model-cycle metadata shown in DAT but not yet fully retained in event payloads
* VERIFY
  * remaining DAT summary text fields not yet mirrored in the corresponding event payloads
* PREDICT
  * remaining DAT summary text fields not yet mirrored in the corresponding event payloads

### Currently Approved Implementation Work Only

* [x] CURATE `load_database` event fields
* [x] CURATE `categorical_transform` event fields
* [x] CURATE `correlation_filter` event fields
* [x] Focused parity tests for the CURATE slice above
* [x] Third-oracle tests for the CURATE slice above
* [ ] Mutation tests for the CURATE slice above

### Future Reviewable Increments

* [x] GENERATE model-scan gap closure completed for the approved cycle and metric-label fields
* [x] VERIFY summary-header gap closure completed for the approved metadata fields
* [x] VERIFY sorted cross-validation metric parity coverage completed
* [x] VERIFY flawed-test status and metric parity coverage completed
* [x] VERIFY branch-marker parity coverage completed
* [x] VERIFY model-context parity coverage completed
* [x] VERIFY direct-test event parity coverage completed
* [x] PREDICT summary-count and proportion parity coverage completed
* [ ] VERIFY gap closure after CURATE slice is complete and reviewed
* [ ] PREDICT gap closure after CURATE slice is complete and reviewed

### Guardrails

* No scientific calculations may change.
* No thresholds may change.
* No DAT output may change.
* No CSV output may change.
* No model behavior may change.
* No CLI behavior may change.

### Done

* [x] Read-only DAT-to-event inventory completed.
* [x] Module-level gap groups recorded for CURATE, GENERATE, VERIFY, and PREDICT.
* [x] Approved implementation sequence narrowed to the CURATE slice only.

---

## Validation Source-Data Policy

For JSON-output validation runs, `databases/` is currently treated as a protected source-data folder.

Rules:

* Do not edit CSV files in `databases/`.
* Do not move or rename files in `databases/`.
* Do not overwrite source CSV files.
* Do not write generated ROBERT outputs into `databases/`.

Generated outputs must remain in normal root output folders and optional copy-only archives.

Open decision:

* Decide whether `databases/` should:

  * remain in the repository,
  * move to `tests/fixtures/`, or
  * be removed after formal JSON unit tests are created.

---

## Parking Lot

Ideas to revisit later:

* UI architecture: Dash, EasyROB, Vercel, Streamlit, or another approach.
* Whether JSON export should become an official ROBERT feature.
* Whether JSON output should be optional or default.
* Whether future ChatBob should read only module-native audit JSON files, only run-level context JSON, or both.
* Whether images should be copied, referenced, or summarized in JSON.
* Whether output directories should use a timestamped run ID.
* Whether multiple ROBERT runs should be indexed by a central run registry.
* Whether REPORT should eventually read structured JSON rather than parsing `.dat` files.
* Whether formal unit tests should replace some manual validation checks.
