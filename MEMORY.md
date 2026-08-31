# MEMORY.md — ROBERT / ChatBob Project State

## Current Project Goal

Build a lightweight interpretation layer for completed ROBERT runs.

The near-term technical goal is to make ROBERT generate stable, structured JSON evidence during a run so a future Dash app, ChatBob, can explain ROBERT results to chemists.

ChatBob is not intended to replace ROBERT or EasyROB.

- ROBERT = scientific engine
- EasyROB = setup / execution helper
- ChatBob = result interpretation and teaching layer

## Current Architecture

ROBERT still runs normally and creates its standard outputs.

The JSON layer adds extra files only. It must not change, append to, rename, delete, reorder, or otherwise modify standard ROBERT outputs.

Current JSON-related helper file:

```text
robert/json_output_for_agent.py
```

Current project planning folder:

```text
json-output-for-agent/
```

This folder may later be renamed to:

```text
json-output/
```

Future ChatBob code should live separately, probably in:

```text
chatbob/
```

## Non-Negotiable Rules

1. Plan before implementation.
2. Wait for explicit user approval before editing code.
3. Keep changes small, local, reversible, and easy to explain.
4. Do not change ROBERT scientific behavior.
5. Do not change ROBERT scoring logic.
6. Do not change model-selection logic.
7. Do not change thresholds.
8. Do not change existing CLI behavior.
9. Do not modify standard ROBERT output files.
10. JSON-layer status, warnings, success records, and failure records must go only into JSON-layer audit files.
11. Do not write JSON-layer messages into `.dat`, `.csv`, `.pdf`, image, model, or report files produced by standard ROBERT.
12. If evidence is unavailable, record it as unavailable. Do not guess.
13. Use plain-English comments for new code.
14. Avoid broad repository scans unless explicitly approved.

## Current JSON Artifacts

Current implemented artifacts:

```text
CURATE/dataset_profile.json
CURATE/curate_audit.json
CURATE/json_output_audit.json
```

Purpose:

```text
dataset_profile.json = what the raw input dataset looked like before ROBERT changed it
curate_audit.json = what CURATE did during the curation step
json_output_audit.json = whether JSON artifacts were attempted and whether they succeeded
```

The JSON layer is intended to be fail-soft. If JSON writing fails, ROBERT should continue normally.

## Current Validation Status

Completed validation:

* Regression CURATE validation passed.
* Classification CURATE validation passed.
* `dataset_profile.json` was created and validated.
* `curate_audit.json` was created and validated.
* `json_output_audit.json` records JSON-layer events.
* Standard `CURATE_data.dat` was checked and did not contain JSON-layer text.
* Timestamped wrapper archive behavior has been validated as copy-only.
* Missing module folders are represented in manifests with `module_dir_exists: false` and `file_count: 0`.

Known partial issue:

* Full REPORT PDF generation was partially blocked in the local environment by missing WeasyPrint system libraries. ROBERT runs still completed.

Still needed:

* Fault-injection validation specifically for `curate_audit.json` write failure.

## Current Source Data Folder

Temporary validation datasets currently live in:

```text
databases/
```

This folder is treated as protected source data.

Rules:

* Do not edit CSV files in `databases/`.
* Do not move them.
* Do not overwrite them.
* Do not add generated ROBERT outputs into this folder.
* Do not ignore `databases/` for now.
* Before committing database files, confirm they are safe to share, reasonably small, and not private or unpublished.

Later TODO:

* Decide whether `databases/` should remain in the repo, move to `tests/fixtures/`, or be removed after formal JSON unit tests are created.

## Current Documentation Files

Important project docs:

```text
json-output-for-agent/AGENTS.md
json-output-for-agent/PROJECT_RULES.md
json-output-for-agent/TASKS.md
json-output-for-agent/task_tracker_plain_english.md
json-output-for-agent/json_schema_notes.md
json-output-for-agent/output_map.md
json-output-for-agent/README.md
```

Update rules:

* Routine validation updates: update `task_tracker_plain_english.md`.
* Schema changes: update `json_schema_notes.md`.
* Task status changes: update `TASKS.md`.
* Project-direction changes: update `AGENTS.md` and `README.md`.

Avoid updating many docs for small routine checks.

## Current Next Task

Next task:

```text
Expand standalone DAT↔JSON parity tests while keeping legacy pytest files untouched.
```

Goal:

1. Keep parity checks isolated in new test files under `tests/`.
2. Reuse shared parity helpers for consistent event-first assertions.
3. Extend parity coverage from GENERATE to PREDICT and VERIFY.
4. Keep runtime behavior unchanged and avoid edits to existing pytest scripts unless explicitly approved.
5. Keep push scope clean by excluding local environment backups and generated folders.

## Future Module-Native Hook Direction

Module-native JSON hooks are important.

Current pattern:

```text
dataset_profile.json = raw input snapshot
curate_audit.json = CURATE module audit
```

Future likely artifacts:

```text
GENERATE/generate_audit.json
VERIFY/verify_audit.json
PREDICT/predict_audit.json
REPORT/report_audit.json
run_context.json
```

Current parity-testing note:

* A standalone parity harness has been added in new test files to compare DAT text and JSON audit evidence without modifying existing pytest modules.
* Event-level audit payloads are preferred for stable assertions when sections are mutable across repeated calls.

Each future hook must:

* capture information ROBERT already produces,
* write only new JSON files,
* never alter standard ROBERT outputs,
* include `schema_version`,
* be fail-soft,
* record JSON status only in JSON audit files,
* avoid invented explanations or thresholds.

## Future ChatBob Direction

ChatBob will likely be a Dash app.

Possible tabs:

```text
Home
Run Overview
Ask ChatBob
Teaching Mode
Next Steps
Troubleshooting
Evidence Viewer
```

ChatBob should:

* select a timestamped ROBERT run,
* read JSON evidence from the run,
* answer one-time questions using OpenAI Responses,
* support FAQ-style questions,
* support RAG over ROBERT docs and user-uploaded PDFs,
* explain ROBERT results for chemists,
* show evidence used in answers.

ChatBob should not:

* run ROBERT as its primary role,
* replace ROBERT,
* rescore ROBERT,
* modify ROBERT outputs,
* invent scientific conclusions.

## Files Not To Touch Without Approval

Do not touch these without explicit approval:

```text
standard ROBERT output files
standard ROBERT scoring code
standard ROBERT model-selection code
standard ROBERT CLI behavior
files inside databases/
files inside tests/
```

## Token-Saving Instructions For Agents

Before inspecting files, ask whether inspection is needed.

Prefer narrow file lists.

Do not scan the full repository unless explicitly approved.

For most tasks, follow this rhythm:

1. Plan only.
2. Wait for approval.
3. Execute only the approved plan.
4. Validate only the requested behavior.
5. Update only the necessary tracker file.

When starting a new chat, read only:

```text
AGENTS.md
MEMORY.md
TASKS.md if needed
```

Do not re-read the entire workspace.
