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

### Current Priority

The current JSON-output milestone is ready for upstream review.

Implemented module-native runtime audits:

- CURATE
- GENERATE
- VERIFY
- PREDICT

Not currently required for this milestone:

- AQME audit JSON
- EVALUATE audit JSON
- full REPORT audit JSON

REPORT currently provides figure provenance through `JSON/report_figure_provenance.json`.

Validation completed for the current milestone:

- isolated DAT-to-JSON parity checks across CURATE, GENERATE, VERIFY, and PREDICT;
- independent third-oracle and mutation-test coverage for selected CURATE evidence;
- controlled comparison between pristine ROBERT 2.1.2 and the current ChatBob-modified ROBERT 2.1.2 using a regression dataset;
- normalized DAT comparison: PASS;
- scientific CSV comparison: PASS.

Next steps:

1. Open a pull request for upstream review.
2. Share the validation evidence with Juanvi.
3. Confirm the intended integration branch and long-term JSON output location.
4. Move into question-driven ChatBob design.
5. Add further JSON fields or parity coverage only where a real ChatBob user question exposes a need.

Broader multi-dataset parity validation remains future hardening rather than a blocker for the initial pull request or first ChatBob prototype.

## Validation Source-Data Policy (Temporary)

For JSON-output validation runs, `databases/` is currently treated as a protected source-data folder.

Rules:
- Do not edit CSV files in `databases/`.
- Do not move or rename files in `databases/`.
- Do not overwrite source CSV files.
- Do not write generated ROBERT outputs into `databases/`.

Generated outputs must remain in normal root output folders and optional copy-only archives.

## Controlled Original-vs-Modified ROBERT Validation

The purpose of this comparison is to confirm that the JSON-output additions do not change ROBERT's existing scientific results.

The validation workflow is:

1. Run the same dataset through pristine ROBERT 2.1.2.
2. Run the same dataset through the ChatBob-modified ROBERT 2.1.2 using the same scientific command-line options.
3. Preserve the completed output folders from both runs.
4. Compare them using `comparison/compare_robert_outputs.ipynb`.

The comparison notebook checks:

- the four primary ROBERT DAT files:
  - `CURATE/CURATE_data.dat`
  - `GENERATE/GENERATE_data.dat`
  - `VERIFY/VERIFY_data.dat`
  - `PREDICT/PREDICT_data.dat`
- matching scientific CSV files;
- model-selection outputs;
- prediction outputs.

For DAT comparison, only expected run-specific metadata is normalized:

- timestamps,
- absolute file paths,
- execution times,
- trailing whitespace.

Scientific values are not normalized away.

### Regression validation

A controlled regression comparison was revalidated on 2026-08-31 using `H_predict_ln-k.csv`.

The modified ROBERT run used:

`python -m robert --csv_name "H_predict_ln-k.csv" --y "ln(k)_rate" --names "Coupling" --ignore "Coupling"`

Results:

- Normalized DAT comparison: PASS
- Scientific CSV comparison: PASS
- The only expected CSV metadata difference was the recorded input CSV path in `CURATE/CURATE_options.csv`.
- macOS `.DS_Store` differences were non-scientific and ignored.

Conclusion:

For this controlled regression example, the ChatBob-modified ROBERT 2.1.2 preserves the standard scientific outputs of pristine ROBERT 2.1.2.

The regression dataset was supplied for development/testing and is not currently committed to the public branch pending confirmation that redistribution is appropriate.

## 2026-07-14 In-Repo Verification Update

Additional manual verification was completed in this repository after the 2026-07-13 comparison milestone.

Confirmed:

- structured JSON files were generated and reviewed for CURATE, GENERATE, VERIFY, PREDICT, and REPORT stages;
- when the updated ROBERT version in this repo was run and compared against DAT files from original ROBERT, no DAT differences were observed.

This confirmation supports the additive-behavior claim for the verified workflow scope.


## Comparison Notebook

`comparison/compare_robert_outputs.ipynb` compares two completed ROBERT runs.

The copied ROBERT output folders are local validation data and are not committed.

## 2026-06-08 Validation Snapshot

Validated with timestamped wrapper:
- Regression input: `databases/Regression/AQME-ROBERT_A_predict_solubility.csv`
- Classification input: `databases/Clasification/F_predict_outcome.csv`

Pass/fail summary:
- PASS: Standard root outputs are still produced for CURATE/GENERATE/VERIFY/PREDICT.
- PASS: JSON artifacts are produced (`JSON/dataset_profile.json`, `JSON/curate_audit.json`, `JSON/curate_json_output_audit.json`, archive manifests, `run_summary.json`).
- PASS: JSON files open and required top-level fields are present.
- PASS: `JSON/dataset_profile.json` includes `schema_version` and works for both regression and classification targets.
- PASS: No JSON-layer status text detected in standard `.dat` files.
- PASS: Copy-only archive behavior confirmed via `wrapper_run.json` copy-mode statement.
- PASS: Missing module folders are represented in manifests with `module_dir_exists=false` and `file_count=0`.
- PARTIAL: REPORT PDF was not generated because required WeasyPrint system libraries are missing in this environment.

## Validation Source-Data Policy

Some validation datasets were provided by Juanvi for development and testing.

Until redistribution is explicitly confirmed:

- treat these datasets as protected local source data;
- do not commit them to the public repository;
- do not modify or overwrite them during validation;
- document which dataset was used for a comparison when relevant;
- keep generated ROBERT outputs separate from the source datasets.

Generated ROBERT run outputs in root folders and timestamped archives should also remain excluded from Git.

If a dataset is later approved for public redistribution, it may be added as a documented validation fixture, for example under `tests/fixtures/` or another agreed location.