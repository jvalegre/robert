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
