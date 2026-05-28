# PROJECT_RULES.md — ROBERT JSON Output Project Rules

## Process Rules

1. Work in small steps, not giant rewrites.
2. Prefer readable code over clever code.
3. Every major change must be explained in plain language.
4. State assumptions explicitly.
5. Do not invent data or results.
6. If uncertain, mark uncertainty and suggest how to test it.
7. Update `task_tracker_plain_english.md` after meaningful progress.
8. Update `TASKS.md` when tasks change.
9. Keep scientific reasoning separate from speculation.
10. Preserve existing ROBERT behavior unless explicitly approved.

---

## Standard Workflow

For each step:

1. State the goal of the step.
2. State the scientific and coding assumptions.
3. Inspect relevant files before changing anything.
4. Propose a short implementation plan.
5. Wait for explicit approval.
6. Make one small, focused change.
7. Add plain-English comments to new code.
8. Test the change.
9. Explain what changed in plain language.
10. Record confirmed results separately from interpretation.
11. Update `task_tracker_plain_english.md`.
12. Update `TASKS.md` if needed.

---

## Planning Requirement

Before modifying code, provide:

1. files inspected,
2. files likely to change,
3. proposed change,
4. expected behavior,
5. risks,
6. testing plan.

Do not implement until the user approves.

---

## Code Change Rules

Allowed:

- small additive changes,
- helper functions,
- JSON export calls,
- file inventory creation,
- documentation updates,
- focused tests.

Not allowed without explicit approval:

- broad refactoring,
- changing ROBERT score logic,
- changing model-selection logic,
- changing thresholds,
- changing existing CLI behavior,
- deleting existing outputs,
- renaming public files or functions,
- introducing network/API calls,
- making LLM calls part of standard ROBERT execution.

For timestamped run archives:

- preserve standard ROBERT root output behavior,
- do not change default output destinations,
- use copy-only duplication into `json-output-for-agent/runs/<timestamp>_<input>/`.

---

## Comment Rule

Every new code section added to existing ROBERT files must include one or two plain-English comments.

The comments should explain:

- what the code does,
- why it is safe,
- whether it changes behavior or only records information.

Preferred style:

```python
# Record a JSON summary for later UI or agent use.
# This only saves information ROBERT already created; it does not change the calculation.
```

---

## JSON Output Rules

All JSON outputs should:

- include `schema_version`,
- include the ROBERT module name when relevant,
- include clear field names,
- avoid ambiguous abbreviations,
- distinguish direct values from derived helper values when practical,
- be valid JSON,
- be readable by humans,
- be useful to a future UI.

---

## File Manifest Rules

For each ROBERT run, aim to capture a file manifest that records generated files.

When practical, include:

- path,
- filename,
- extension,
- file size,
- modified timestamp,
- likely ROBERT module,
- short description if known.

The manifest should help future tools locate outputs without guessing.

---

## Testing Rules

For each new JSON artifact:

1. Confirm the file is created.
2. Confirm it opens as valid JSON.
3. Confirm expected top-level keys exist.
4. Confirm normal ROBERT outputs are still created.
5. Confirm no scientific result changed.

If a test is not run, say so clearly and explain why.

---

## Plain-English Update Rule

After meaningful progress, update:

```text
task_tracker_plain_english.md
```

Each update should include:

- date,
- what was done,
- files changed,
- what was tested,
- result,
- what remains uncertain,
- next suggested step.

Use plain language for a chemist.

---

## Scientific Integrity Rules

Do not claim that a model is good or bad unless the evidence supports it.

Do not invent explanations.

Do not invent thresholds.

Do not invent descriptor meanings.

Do not invent missing values.

If something is unknown, say it is unknown.

If something needs testing, propose a test.

---

## Current Project Priority

The priority is to identify where ROBERT already creates output information and then record that information in structured JSON with minimal changes to existing code.
