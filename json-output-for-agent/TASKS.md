# TASKS.md — ROBERT JSON Output Tasks

## Project Status

Status: Starting new ROBERT core-development branch.

Goal: Add structured JSON outputs to ROBERT with minimal changes to existing code.

---

## Current Priority

Implement and validate the first additive JSON artifact in CURATE.

Also maintain a copy-only timestamped run archive outside ROBERT core behavior.

---

## Phase 1 — Inspect Existing ROBERT Outputs

### To Do

- [x] Identify where CURATE writes output files.
- [x] Identify where GENERATE writes output files.
- [x] Identify where VERIFY writes output files.
- [x] Identify where PREDICT writes output files.
- [x] Identify where REPORT writes the PDF report.
- [x] Identify where `.dat` files are created.
- [x] Identify where images are created.
- [x] Identify where the ROBERT score is calculated.
- [x] Identify which values are already available in memory before being written to files.
- [x] Identify the smallest safe places to call JSON helper functions.

### Done

- [x] Output map inspection completed (2026-05-27).
- [x] Created `json-output-for-agent/output_map.md` with:
	- module-by-module output write map,
	- `.dat` file creation map,
	- plot/image output map,
	- ROBERT score/report flow map,
	- safest future JSON insertion points,
	- risks and first implementation proposal.

---

## Phase 1.5 — Incoming Dataset Profile JSON (First Artifact)

This phase adds one JSON artifact that profiles observable facts from the incoming dataset.

### To Do

- [x] Confirm insertion point in curate.__init__ (between load_variables and load_database).
- [x] Confirm helper file location: robert/json_output_for_agent.py.
- [x] Confirm JSON output location: CURATE/dataset_profile.json.
- [x] Enforce standard ROBERT output protection rule for JSON layer.
- [x] Implement profile_input_dataset helper function.
- [x] Implement dataset profile measurement helper functions.
- [x] Implement safe JSON writing helper function.
- [x] Implement JSON output audit helper function.
- [x] Add single JSON helper call in curate.__init__.
- [x] Confirm CURATE/dataset_profile.json is created.
- [x] Confirm CURATE still completes when JSON write fails.
- [x] Confirm JSON status is written only to CURATE/json_output_audit.json.
- [x] Confirm CURATE/CURATE_data.dat contains no JSON-layer status messages.
- [x] Update task tracker.

### Done

- [x] Implemented robert/json_output_for_agent.py (2026-05-28).
- [x] Added fail-soft hook in robert/curate.py (2026-05-28).
- [x] Validated baseline run creates CURATE/dataset_profile.json.
- [x] Injected simulated JSON failure and confirmed standard CURATE outputs still complete.
- [x] Removed JSON-layer writes to standard CURATE logger output.
- [x] Added CURATE/json_output_audit.json as the only JSON-layer status channel.

---

## Phase 2 — Design JSON Structure

### To Do

- [ ] Decide final project folder name.
- [ ] Decide final helper file location.
- [ ] Decide whether JSON export is always on or controlled by an option.
- [ ] Draft module summary JSON schema.
- [ ] Draft run-level JSON schema.
- [ ] Draft file manifest JSON schema.
- [ ] Decide how to label direct ROBERT values vs derived helper values.
- [ ] Decide how timestamps should be formatted.

### Done

- [ ] Not started.

---

## Phase 3 — Build JSON Helper

### To Do

- [x] Create `robert/json_output_for_agent.py`.
- [x] Add safe JSON writing helper.
- [x] Add JSON-safe conversion helper for common Python, NumPy, pandas, and pathlib objects.
- [ ] Add file manifest helper.
- [ ] Add module summary helper.
- [ ] Add plain-English comments to helper functions.
- [ ] Add simple tests for helper functions.

### Done

- [x] Core dataset-profile helper implemented.

---

## Phase 4 — Add First Module JSON Output

### To Do

- [x] Choose first module to instrument.
- [x] Propose exact files and functions to change.
- [x] Wait for user approval.
- [x] Add minimal helper call.
- [x] Confirm existing ROBERT output is unchanged.
- [x] Confirm JSON output is created.
- [x] Update task tracker.

### Done

- [x] First module instrumented: CURATE incoming dataset profile.

---

## Phase 5 — Extend Across ROBERT Pipeline

### To Do

- [ ] Add CURATE JSON summary.
- [ ] Add GENERATE JSON summary.
- [ ] Add VERIFY JSON summary.
- [ ] Add PREDICT JSON summary.
- [ ] Add REPORT JSON summary.
- [ ] Add run-level context JSON if appropriate.
- [ ] Add generated file manifest.

### Done

- [ ] Not started.

---

## Phase 6 — Validate

### To Do

- [ ] Run a simple regression example.
- [ ] Run a simple classification example.
- [ ] Confirm normal ROBERT outputs still appear.
- [ ] Confirm JSON files are valid.
- [ ] Confirm JSON files contain expected fields.
- [ ] Confirm no scientific outputs changed.
- [ ] Test behavior when expected files are missing.
- [x] Test behavior when JSON writing fails or is unavailable.

### Done

- [x] Manual fault-injection validation completed for JSON write failure in CURATE.

---

## Phase 7 — Documentation

### To Do

- [ ] Document what JSON files are created.
- [ ] Document where JSON files are saved.
- [ ] Document what each top-level field means.
- [ ] Document whether the JSON schema is experimental.
- [ ] Add a short plain-English explanation for chemists.
- [ ] Update `task_tracker_plain_english.md`.

### Done

- [ ] Not started.

---

## Phase 8 — Timestamped Archive Wrapper (Copy-Only)

Goal:
- Keep normal ROBERT root outputs unchanged.
- Duplicate generated outputs into a timestamped project archive folder.

### To Do

- [x] Confirm feasibility outside ROBERT core.
- [x] Implement wrapper script in project folder.
- [x] Ensure wrapper runs ROBERT in normal root mode.
- [x] Ensure wrapper copies outputs (including JSON artifacts) to timestamped run folder.
- [x] Document explicit copy-only rule in AGENTS and project rules docs.
- [ ] Validate on one full run and inspect copied archive contents.

### Done

- [x] Added wrapper script: `json-output-for-agent/scripts/run_robert_timestamped.py`.
- [x] Added policy language that standard output paths remain unchanged and archive behavior is copy-only.

---

## Parking Lot

Ideas to revisit later:

- UI architecture: Dash, EasyROB, Vercel, Streamlit, or other.
- Whether JSON export should become an official ROBERT feature.
- Whether JSON output should be optional or default.
- Whether a future LLM layer should read only `run_context.json`.
- Whether images should be copied, referenced, or summarized in JSON.
- Whether output directories should use a timestamped run ID.
- Whether multiple ROBERT runs should be indexed by a central run registry.
