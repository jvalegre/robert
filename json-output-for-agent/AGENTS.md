# AGENTS.md — ROBERT JSON Output Agent Guide

## Mission

We are modifying ROBERT in the smallest safe way possible so that ROBERT can export structured JSON files during a run.

The goal is to capture information ROBERT already generates, including information written to `.dat` files, shown in the PDF report, saved as images, or produced during CURATE, GENERATE, VERIFY, PREDICT, and REPORT.

The JSON files will later support:

- a user interface,
- chemist-friendly result interpretation,
- easier retrieval of ROBERT run evidence,
- optional LLM explanations.

This project must not change ROBERT's scientific behavior.

ROBERT remains the authority.

The agent helps expose ROBERT evidence. It does not replace ROBERT, rescore ROBERT, or invent new diagnostics.

---

## Core Principle

Minimize changes to existing ROBERT code.

When changes are necessary, make them:

- local,
- small,
- reversible,
- easy to explain,
- additive,
- well commented,
- tested when practical.

The preferred strategy is to add helper functions in one project-specific helper file and then call those helpers from existing ROBERT modules with as few added lines as possible.

---

## Project Folder Convention

Use one dedicated project folder for this work:

```text
json-output-for-agent/
```

This folder may contain:

```text
json-output-for-agent/
  README.md
  PROJECT_RULES.md
  TASKS.md
  task_tracker_plain_english.md
  json_schema_notes.md
  example_outputs/
```

Python helper code should not use hyphens in the filename.

Use:

```text
robert/json_output_for_agent.py
```

Do not use:

```text
robert/json-output-for-agent.py
```

Reason: Python imports work cleanly with underscores, not hyphens.

---

## Source of Truth

The existing ROBERT codebase is the source of truth for:

- how ROBERT runs,
- how folders and output files are created,
- how reports are generated,
- how the ROBERT score is computed,
- existing command-line behavior,
- current tests and documentation.

Do not infer behavior if it can be determined from the code.

Inspect the relevant files first.

---

## Mandatory Planning Rule

Before modifying code, always propose a short plan that includes:

1. files to inspect,
2. files likely to change,
3. expected behavior,
4. risks,
5. how to test the change.

Do not implement until the user explicitly approves the plan.

If a change seems necessary, propose it first.

Do not implement without approval.

---

## Mandatory Execution Checkpoint

Before any implementation work begins, the agent must:

1. present the plan to the user,
2. receive explicit user approval,
3. only then execute edits or commands.

This applies to:

- code,
- notebooks,
- scripts,
- documentation,
- tests,
- folder reorganization.

---

## Communication Rule

Explain progress in plain language for a chemist.

Avoid unnecessary software jargon.

When technical terms are necessary, define them briefly.

For every meaningful change, explain:

- what changed,
- why it changed,
- what file changed,
- how we know it worked,
- what remains uncertain.

---

## Code Comment Rule

Every new code block added to existing ROBERT files must include one or preferably two plain-language comment lines.

The comments should explain what the new code does and why it is being added.

Example:

```python
# Save a small JSON summary of this ROBERT step for later UI or agent use.
# This does not change the ROBERT calculation; it only records information already produced.
```

Avoid clever or vague comments.

Use plain English.

---

## What We Are Trying to Capture

We want to capture structured information from the full ROBERT pipeline, including:

- input files,
- run settings,
- generated output files,
- `.dat` files,
- report values,
- images produced by ROBERT,
- model metrics,
- verification results,
- warning messages,
- score-related evidence,
- feature importance outputs,
- outlier information,
- file timestamps,
- run folder structure.

The goal is to create a structured JSON record associated with each ROBERT run.

---

## Desired JSON Behavior

JSON outputs should be:

- structured,
- stable,
- readable,
- useful for a UI,
- useful for later LLM explanation,
- traceable back to ROBERT outputs.

Each JSON artifact should include a schema version.

Example:

```json
{
  "schema_version": "0.1",
  "module": "VERIFY",
  "artifact_type": "module_summary",
  "status": "completed",
  "files_created": [],
  "values": {},
  "warnings": [],
  "notes": []
}
```

---

## Likely JSON Artifacts

Possible module-level files:

```text
CURATE/curate_summary.json
GENERATE/generate_summary.json
VERIFY/verify_summary.json
PREDICT/predict_summary.json
REPORT/report_summary.json
```

Possible run-level file:

```text
run_context.json
```

Possible file inventory:

```text
robert_file_manifest.json
```

The exact names should be confirmed after inspecting existing ROBERT conventions.

---

## File Timestamp Rule

For each ROBERT run, capture a structured inventory of generated files.

For each file, record when practical:

- file path,
- file name,
- module or folder,
- file extension,
- file size,
- last modified timestamp,
- whether it is JSON, DAT, CSV, PDF, PNG, SVG, TXT, or another type.

This manifest should help a future UI or agent retrieve the correct files later.

---

## Direct Values vs Derived Values

JSON values should be labeled conceptually as either:

1. Direct ROBERT values  
   Values already produced by ROBERT.

2. Derived convenience values  
   Simple helper values calculated only to organize ROBERT output.

Acceptable derived convenience values include:

- number of files created,
- list of available output files,
- timestamp inventory,
- whether a known output file exists,
- number of warnings captured from an existing ROBERT output.

Do not create new scientific diagnostics unless explicitly approved.

Do not create a new score.

Do not override the ROBERT score.

---

## Helper Function Strategy

Prefer a single helper file:

```text
robert/json_output_for_agent.py
```

This file should contain reusable helper functions for:

- safe JSON writing,
- timestamp capture,
- file manifest creation,
- conversion of NumPy/pandas values into JSON-safe values,
- module summary creation,
- run context assembly.

Existing ROBERT modules should call these helpers with minimal added code.

Avoid scattering JSON-writing logic throughout the existing codebase.

---

## Rules for Changing Existing ROBERT Code

Do not make broad edits.

Do not refactor unrelated code.

Do not change existing ROBERT behavior unless explicitly requested.

Do not remove existing functionality.

Do not rename public functions, CLI arguments, folders, or output files unless explicitly approved.

Do not change scoring logic.

Do not change model-selection logic.

Do not change default thresholds.

Do not introduce network calls into the standard ROBERT workflow.

Do not add required LLM or API dependencies.

Any LLM/API functionality must remain optional and outside the default ROBERT workflow.

---

## Testing Expectations

For each new JSON output, test that:

- the expected JSON file is created,
- the JSON file can be opened,
- the JSON file is valid,
- expected top-level fields exist,
- the normal ROBERT output still appears,
- existing behavior is unchanged.

If full tests are too slow, run the smallest meaningful test and explain what was not tested.

---

## Documentation Tracking Requirement

After meaningful progress, update:

```text
json-output-for-agent/task_tracker_plain_english.md
```

This file is the running plain-English record of accomplishments.

Also update:

```text
json-output-for-agent/TASKS.md
```

when tasks are completed, added, postponed, or changed.

The task tracker should be understandable to a chemist who does not want to inspect the code.

---

## First Development Task

Do not write code immediately.

First inspect the repository and identify:

1. where CURATE writes outputs,
2. where GENERATE writes outputs,
3. where VERIFY writes outputs,
4. where PREDICT writes outputs,
5. where REPORT writes the PDF,
6. where `.dat` files are created,
7. where images are created,
8. where the ROBERT score is calculated,
9. which outputs already exist and can be captured,
10. where the smallest JSON helper calls could be inserted.

Then propose the smallest safe implementation plan.

---

## Current Strategic Direction

The current goal is to make ROBERT easier to use through a future UI.

The immediate technical step is to make ROBERT produce structured JSON evidence as it runs.

The durable contribution of this branch should be:

> ROBERT can produce stable, structured, machine-readable evidence files without changing its scientific behavior.

The UI architecture can be decided later.
