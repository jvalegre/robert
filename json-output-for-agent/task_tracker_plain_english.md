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
- Missing module folders produce valid manifests with `module_dir_exists=false` and `file_count=0`.

What remains uncertain:
- Module-native write-time hooks are still not added.
- Automated tests for helper-manifest functions are still not added.

Next suggested step:
- Add small helper tests for manifest and run-summary functions, then (if still needed) add module-native hooks incrementally behind a low-risk option.

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
