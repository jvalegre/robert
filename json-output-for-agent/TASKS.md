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

Continue module-native runtime audit implementation one module at a time.

Completed module-native audit artifacts:

* `CURATE/curate_audit.json`
* `GENERATE/generate_audit.json`
* `VERIFY/verify_audit.json`

Current next module:

* `PREDICT/predict_audit.json`

Later modules:

* `AQME/aqme_audit.json`
* `EVALUATE/evaluate_audit.json`

Deferred:

* `REPORT/report_audit.json` or a run-level report audit.
* REPORT should be revisited after the upstream evidence-producing modules are stable.

---

## Validation Standard for Each Module

For each module-native audit artifact, confirm:

* [ ] Standard `.dat` output is unchanged except timestamp/runtime.
* [ ] Audit JSON opens as valid JSON.
* [ ] Module `json_output_audit.json` records successful write.
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

This section tracks JSON files written during the ROBERT module run, not archive-side manifests created after copying outputs.

| Module   | Audit artifact                                       | Status          | Validation status | Notes                                                                                                                                   |
| -------- | ---------------------------------------------------- | --------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| CURATE   | `CURATE/curate_audit.json`                           | Implemented     | Validated         | Captures CURATE runtime evidence and curated-output metadata.                                                                           |
| GENERATE | `GENERATE/generate_audit.json`                       | Implemented     | Validated         | Captures model generation, BO, PFI, best-model selection, outputs, and heatmap artifacts.                                               |
| VERIFY   | `VERIFY/verify_audit.json`                           | Implemented     | Validated         | Captures VERIFY branch context, model context, y-mean, y-shuffle, one-hot tests, thresholds, pass/fail status, plots, and summary text. |
| PREDICT  | `PREDICT/predict_audit.json`                         | Not implemented | Not validated     | Next module.                                                                                                                            |
| AQME     | `AQME/aqme_audit.json`                               | Not implemented | Not validated     | Later module.                                                                                                                           |
| EVALUATE | `EVALUATE/evaluate_audit.json`                       | Not implemented | Not validated     | Later module.                                                                                                                           |
| REPORT   | `REPORT/report_audit.json` or run-level report audit | Deferred        | Not validated     | Revisit after upstream module audits are stable.                                                                                        |

Notes:

* Archive manifests and `run_summary.json` are copy-side artifacts.
* Module-native audits are runtime evidence artifacts.
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
* [x] Confirm JSON output location: `CURATE/dataset_profile.json`.
* [x] Enforce standard ROBERT output protection rule for JSON layer.
* [x] Implement `profile_input_dataset` helper function.
* [x] Implement dataset profile measurement helper functions.
* [x] Implement safe JSON writing helper function.
* [x] Implement JSON output audit helper function.
* [x] Add single JSON helper call in `curate.__init__`.
* [x] Confirm `CURATE/dataset_profile.json` is created.
* [x] Confirm CURATE still completes when JSON write fails.
* [x] Confirm JSON status is written only to `CURATE/json_output_audit.json`.
* [x] Confirm `CURATE/CURATE_data.dat` contains no JSON-layer status messages.

### Done

* [x] Implemented `robert/json_output_for_agent.py`.
* [x] Added fail-soft hook in `robert/curate.py`.
* [x] Validated baseline run creates `CURATE/dataset_profile.json`.
* [x] Injected simulated JSON failure and confirmed standard CURATE outputs still complete.
* [x] Removed JSON-layer writes to standard CURATE logger output.
* [x] Added `CURATE/json_output_audit.json` as the only JSON-layer status channel.

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
* [x] Implement `CURATE/curate_audit.json`.
* [x] Confirm existing ROBERT CURATE output is unchanged.
* [x] Confirm audit JSON output is created.
* [x] Confirm JSON output audit is created.
* [x] Confirm no JSON text appears in `CURATE/CURATE_data.dat`.

### Done

* [x] `CURATE/curate_audit.json` implemented.
* [x] CURATE runtime evidence captured.
* [x] CURATE JSON write status isolated to `CURATE/json_output_audit.json`.
* [x] CURATE validation completed.
* [x] CURATE fault-injection validation completed for audit write failure.

---

## Phase 5 — GENERATE Module-Native Audit

### To Do

* [x] Propose implementation plan for GENERATE audit.
* [x] Wait for user approval.
* [x] Implement `GENERATE/generate_audit.json`.
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

* [x] `GENERATE/generate_audit.json` implemented.
* [x] `GENERATE/json_output_audit.json` implemented for audit-write status.
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
* [x] Implement `VERIFY/verify_audit.json`.
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

* [x] `VERIFY/verify_audit.json` implemented.
* [x] `VERIFY/json_output_audit.json` implemented for audit-write status.
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

* [ ] Inspect current `predict.py` and `predict_utils.py` insertion points.
* [ ] Propose implementation plan before coding.
* [ ] Wait for user approval.
* [ ] Implement `PREDICT/predict_audit.json`.
* [ ] Capture input settings and model/source context.
* [ ] Capture prediction CSV output metadata.
* [ ] Capture prediction summary metrics.
* [ ] Capture train/validation/test/external prediction evidence when available.
* [ ] Capture uncertainty or variability evidence when available.
* [ ] Capture SHAP artifact metadata when available.
* [ ] Capture PFI artifact metadata when available.
* [ ] Capture outlier evidence when available.
* [ ] Capture y-distribution artifact metadata when available.
* [ ] Capture Pearson heatmap metadata when available.
* [ ] Capture generated plot/image paths.
* [ ] Capture final `.dat` summary preview if useful.
* [ ] Confirm existing `PREDICT_data.dat` is unchanged except timestamp/runtime.
* [ ] Confirm prediction CSV outputs are unchanged.
* [ ] Confirm image outputs are unchanged or only differ by normal timestamp/rendering metadata.
* [ ] Confirm audit JSON validates.
* [ ] Confirm JSON output audit validates.
* [ ] Confirm no JSON text appears in `PREDICT/PREDICT_data.dat`.

### Done

* [ ] Not started.

---

## Phase 8 — AQME Module-Native Audit

### To Do

* [ ] Inspect current `aqme.py` insertion points.
* [ ] Propose implementation plan before coding.
* [ ] Wait for user approval.
* [ ] Implement `AQME/aqme_audit.json`.
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
* [ ] Implement `EVALUATE/evaluate_audit.json`.
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
* [ ] Decide whether REPORT needs its own `REPORT/report_audit.json`.
* [ ] Decide whether a run-level context JSON can replace some fragile report parsing.
* [ ] Investigate current REPORT PDF/report parser failure separately.
* [ ] Confirm required WeasyPrint/system-library environment for PDF generation if PDF validation is needed.
* [ ] Decide whether REPORT should consume module-native JSON audits in a future design.

### Done

* [ ] REPORT module-native audit not started.
* [ ] Current REPORT issue identified as separate from CURATE, GENERATE, and VERIFY audit validation.

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
