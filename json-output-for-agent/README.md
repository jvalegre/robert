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

## Current Priority

Preserve normal ROBERT output behavior while adding project-side JSON/export support.

Current wrapper policy:
- ROBERT still writes normal outputs in default root folders.
- A timestamped archive wrapper duplicates generated outputs into:
  - `json-output-for-agent/runs/<timestamp>_<input_csv_stem>/`
- The archive is copy-only and does not replace ROBERT output paths.
