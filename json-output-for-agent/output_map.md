# ROBERT Output Map

## Plain-English Summary

This inspection mapped where ROBERT currently writes outputs without changing ROBERT code.

Main findings:
- CURATE writes curated CSV files and a CURATE options CSV in the CURATE folder, and also creates a Pearson heatmap image.
- GENERATE writes model databases and parameter CSV files in GENERATE/Raw_data, copies best files to GENERATE/Best_model, and creates heatmap images.
- VERIFY mainly writes a verification plot image and logs all test values to VERIFY/VERIFY_data.dat.
- PREDICT writes prediction CSVs, many analysis images (results, SHAP, PFI, outliers, distribution, Pearson), and logs metrics/warnings to PREDICT/PREDICT_data.dat.
- REPORT writes ROBERT_report.pdf in the working directory (plus temporary report.css and optional report_debug.txt).
- ROBERT score is calculated in report_utils.calc_score by combining parsed values from PREDICT_data.dat and VERIFY_data.dat.
- The values needed for JSON export already exist in memory in dictionaries/dataframes before file writes (for example: csv_df, options_df, bo_data, PFI_dict, verify_results, verify_metrics, Xy_data, data_score).

## Files Inspected

- robert/curate.py
- robert/generate.py
- robert/generate_utils.py
- robert/verify.py
- robert/predict.py
- robert/predict_utils.py
- robert/report.py
- robert/report_utils.py
- robert/utils.py

## Output Map

| ROBERT module | file/function that creates outputs | output folder | output files created | values that could be captured in JSON | possible JSON insertion point |
|---|---|---|---|---|---|
| CURATE | robert/curate.py -> save_curate | CURATE | <input>_CURATE.csv, optional <input>_CURATE_<MODEL>.csv, CURATE_options.csv | filtered descriptor list, model-specific descriptor lists, y, ignore, names, class label mapping, output file names | At end of save_curate right after each to_csv call and after options_df to_csv |
| CURATE | robert/utils.py -> pearson_map (called from curate.__init__) | CURATE | Pearson_heatmap.png (if descriptors <= 30) | descriptor count, correlation matrix summary, max abs correlation, generated file path | In pearson_map after plt.savefig and before log write |
| CURATE | robert/utils.py -> Logger via load_variables + finish_print | CURATE | CURATE_data.dat | all plain-text curation messages, counts, warnings, elapsed time | In finish_print just before log.finalize |
| GENERATE | robert/generate_utils.py -> BO_workflow | GENERATE/Raw_data/No_PFI | <MODEL>_db.csv, <MODEL>.csv | bo_data (model/type/kfold/repeat_kfolds/seed/error_type/y/names/X_descriptors/params/combined metric), set split labels | In BO_workflow immediately after db csv and params csv writes |
| GENERATE | robert/generate_utils.py -> save_pfi_csv | GENERATE/Raw_data/PFI | <MODEL>_PFI.csv, optional <MODEL>_PFI_db.csv | selected descriptors after PFI, combined metric with PFI, class label mapping, split | In save_pfi_csv after csv_PFI_df.to_csv and after _PFI_db write |
| GENERATE | robert/generate_utils.py -> detect_best | GENERATE/Best_model/(No_PFI or PFI) | copied best params csv and best _db.csv | winning model file name, best combined metric, selection criterion (min for rmse/mae, max otherwise) | In detect_best after best file copy operations |
| GENERATE | robert/utils.py -> create_heatmap (via heatmap_workflow) | GENERATE/Raw_data | Heatmap_ML_models_No_PFI.png, Heatmap_ML_models_PFI.png | per-model combined metric table used for heatmap, image path | In create_heatmap after plt.savefig |
| GENERATE | robert/utils.py -> Logger via load_variables + finish_print | GENERATE | GENERATE_data.dat | BO summary lines, per-model logs, elapsed time | In finish_print before finalize |
| VERIFY | robert/verify.py -> verify.__init__/analyze_tests/print_verify | VERIFY (log) | VERIFY_data.dat | verify_results dict (CV_score, sorted_CV_score, y_mean, y_shuffle, onehot, thresholds, pass/fail), verify_metrics dict | In verify.__init__ after analyze_tests and before print_verify |
| VERIFY | robert/utils.py -> plot_metrics | VERIFY | VERIFY_tests_<model>_<suffix>.png | verify_metrics bars/colors/thresholds, plotted metric values | In plot_metrics after plt.savefig |
| PREDICT | robert/predict_utils.py -> save_predictions | PREDICT and PREDICT/csv_test | <MODEL>_<suffix>.csv, csv_test/<external>_<MODEL>_<suffix>.csv | df_results table, external predictions table, conformal half-width, class label reconversion, set labels | In save_predictions immediately after each to_csv |
| PREDICT | robert/utils.py -> graph_reg/graph_clas | PREDICT and PREDICT/csv_test | Results_*.png, CV_variability_*.png, CV_train_valid_predict_*.png | y_true/y_pred arrays, prediction SD arrays, plotted set type, pred range tracked in Xy_data | In graph_reg/graph_clas after plt.savefig |
| PREDICT | robert/utils.py -> shap_analysis | PREDICT | SHAP_*.png | SHAP min/max per descriptor, descriptor ranking | In shap_analysis right after shap values computed and after plt.savefig |
| PREDICT | robert/utils.py -> PFI_plot | PREDICT | PFI_*.png | permutation importances means/stds, sorted descriptor importance | In PFI_plot after plt.savefig |
| PREDICT | robert/utils.py -> outlier_plot | PREDICT | Outliers_*.png | outlier SD values, outlier names, outlier counts/percentages | In outlier_plot after outlier_filter and after plt.savefig |
| PREDICT | robert/utils.py -> distribution_plot | PREDICT | y_distribution_*.png | quartile/class distribution counts, imbalance warnings | In distribution_plot after y_dist_dict built and after final file move |
| PREDICT | robert/utils.py -> pearson_map (via pearson_map_predict) | PREDICT | Pearson_heatmap_No_PFI.png, Pearson_heatmap_PFI.png (if <= 30 descriptors) | correlation matrix, max abs correlation pair, warning level | In pearson_map after plt.savefig |
| PREDICT | robert/utils.py -> Logger via load_variables + finish_print | PREDICT | PREDICT_data.dat | all prediction summaries, metrics, warnings, elapsed time | In finish_print before finalize |
| REPORT | robert/report_utils.py -> make_report | working directory | ROBERT_report.pdf | final assembled HTML string, css filename list, output path | In make_report after make_pdf and write_bytes |
| REPORT | robert/report.py -> report.__init__ | working directory | temporary report.css, optional report_debug.txt | report_html string, module section fragments, eval_only flag | In report.__init__ after report_html completion and before make_report |
| REPORT | robert/report.py + robert/report_utils.py -> print_score/calc_score | in-memory during report creation | no direct score file (score appears in report and section data) | data_score dict with all intermediate score components and final robert_score_<suffix> | In report.print_score after calc_score call for each suffix |

## .dat File Map

Primary creation mechanism:
- robert/utils.py defines Logger(filein, append, suffix="dat") which opens file path filein_append.dat in write mode.
- robert/utils.py load_variables creates Logger at module startup for all modules except REPORT.
- robert/utils.py finish_print writes elapsed time and closes the dat log with finalize().

Expected dat outputs from core modules:
- CURATE/CURATE_data.dat
- GENERATE/GENERATE_data.dat
- VERIFY/VERIFY_data.dat
- PREDICT/PREDICT_data.dat

Other dat usage:
- report_utils.repro_info reads <module>/<module>_data.dat files to build report sections.
- report.py reads PREDICT/PREDICT_data.dat in multiple sections (warnings, metrics, predictions).

## Plot/Image Output Map

Main producers and files:
- CURATE: Pearson_heatmap.png from robert/utils.py pearson_map.
- GENERATE: Heatmap_ML_models_No_PFI.png and Heatmap_ML_models_PFI.png from robert/utils.py create_heatmap.
- VERIFY: VERIFY_tests_*.png from robert/utils.py plot_metrics.
- PREDICT:
  - Results_*.png and/or CV_train_valid_predict_*.png from graph_reg/graph_clas.
  - CV_variability_*.png from graph_reg(sd_graph=True).
  - SHAP_*.png from shap_analysis.
  - PFI_*.png from PFI_plot.
  - Outliers_*.png from outlier_plot.
  - y_distribution_*.png from distribution_plot.
  - Pearson_heatmap_No_PFI.png and Pearson_heatmap_PFI.png from pearson_map.
  - External-set plots in PREDICT/csv_test from graph_reg/graph_clas with csv_test=True.

## ROBERT Score and Report Map

Where score is calculated:
- robert/report.py print_score calls report_utils.calc_score(dat_files, suffix, pred_type, data_score).
- robert/report_utils.py calc_score combines:
  - get_predict_scores output parsed from PREDICT_data.dat.
  - get_verify_scores output parsed from VERIFY_data.dat.
  - Additional arithmetic for regression/classification to compute robert_score_No PFI and robert_score_PFI.

Where PDF is written:
- robert/report_utils.py make_report writes ROBERT_report.pdf to current working directory.
- robert/report.py prepares report_html and temporary report.css before make_report.

## Safest Future JSON Export Points

Smallest safe insertion points with minimal behavioral risk:
- CURATE: end of save_curate after CSV writes (already has all dataframes/options in hand).
- GENERATE:
  - end of BO_workflow after writing model db and params CSV.
  - end of save_pfi_csv after writing PFI CSV(s).
  - end of detect_best after best-file copy.
- VERIFY: in verify.__init__ after analyze_tests returns verify_results and verify_metrics, before print_verify.
- PREDICT:
  - end of save_predictions after base and external CSV writes.
  - after each analytics function computes in-memory outputs (shap_analysis, PFI_plot, outlier_plot, distribution_plot), before/after image save.
- REPORT:
  - inside print_score after each calc_score call to capture fully assembled data_score.
  - inside make_report immediately after PDF write for final report artifact metadata.

Why these are safest:
- They are post-computation points where all needed values are already assembled.
- They are immediately adjacent to existing output writes.
- They avoid changing model training, prediction, and scoring algorithms.

## Risks

- REPORT score currently parses text from dat logs; changing dat text format later could silently break score extraction.
- Some outputs are conditional (for example Pearson heatmap skipped when descriptor count > 30, external csv_test artifacts only when csv_test provided).
- Ordering assumptions exist in report image pairing (No_PFI vs PFI order handling).
- If JSON writes are added directly into critical loops, accidental performance impact is possible.
- If JSON serialization is attempted on raw numpy/pandas objects without conversion, write failures can occur.

## Current Runtime Audit Status (2026-08-26)

- Canonical ChatBob runtime JSON location is the top-level `JSON/` folder.
- Implemented module-native runtime audits: `JSON/curate_audit.json`, `JSON/generate_audit.json`, `JSON/verify_audit.json`, `JSON/predict_audit.json`.
- Not implemented: `JSON/aqme_audit.json`, `JSON/evaluate_audit.json`.
- REPORT currently provides `JSON/report_figure_provenance.json` only (no full report audit).
- Wrapper-generated `*_manifest.json` and `run_summary.json` in timestamped archives are copy-side metadata, not module-native runtime audits.

## Current Validation Status

- CURATE, GENERATE, VERIFY, and PREDICT have isolated DAT-to-JSON parity coverage for the currently selected checkpoints.
- CURATE additionally has independent third-oracle checks and mutation tests.
- No AQME DAT-to-JSON parity target is currently defined because AQME does not produce a standard ROBERT DAT summary in the observed workflow.
- Future AQME work, if approved, should validate provenance, operation status, and generated artifacts instead of creating a synthetic DAT comparison.

## Proposed First Implementation Step

Implement one read-only helper module first (no hooks yet), for example robert/json_output_for_agent.py, with:
- safe_json_dump(data, path)
- to_json_safe(obj) for pandas/numpy/pathlib/scalars/lists/dicts
- write_module_artifact(module_name, artifact_type, payload, destination)

Then add only one hook in CURATE save_curate after CURATE_options.csv write, because that point is simple, deterministic, and already has clean metadata.

## Dataset Profile JSON Artifact (Implemented)

What was added:
- New helper module: robert/json_output_for_agent.py
- New CURATE hook: robert/curate.py in `curate.__init__`, immediately after `load_variables(...)` and before `load_database(...)`
- New output file in normal runs: JSON/dataset_profile.json

Why this insertion point was selected:
- It is the earliest practical place with access to `csv_name`, `y`, and `ignore`.
- The helper reads the input file independently and writes one additive artifact.
- The hook is fail-soft (try/except + JSON audit record), so CURATE behavior is unchanged if JSON writing fails.

Standard ROBERT output protection rule (implemented):
- The JSON layer does not write status messages to standard ROBERT outputs.
- JSON-layer success/failure is written only to `JSON/curate_json_output_audit.json`.
- Standard outputs such as `CURATE/CURATE_data.dat`, curated CSVs, options CSV, and images remain untouched by JSON-layer status.

Failure-tolerance validation completed:
- Baseline run created normal CURATE outputs plus `JSON/dataset_profile.json`.
- A simulated JSON write failure was injected by monkeypatching `write_json` to raise.
- CURATE still completed and produced standard outputs:
  - CURATE_data.dat
  - CURATE_options.csv
  - Pearson_heatmap.png
  - model-specific curated CSVs and general curated CSV
- JSON-layer status was recorded in `JSON/curate_json_output_audit.json`:
  - success events for normal runs
  - failure events with error type/message for simulated failures

## Open Questions

- Should JSON export be always on, or behind an option flag?
- JSON file location has been decided: use the top-level `JSON/` folder.
- Should each module write one summary JSON or multiple artifact JSON files?
- Do we need a run-level manifest file that indexes every generated artifact path?
- For REPORT, should we store only final data_score or also parsed intermediate values from get_predict_scores/get_verify_scores?
- Should JSON writes fail-soft (warn and continue) to guarantee no behavior change in scientific outputs?


## Controlled Output Comparison Update (2026-07-13)

A completed unmodified ROBERT 2.1.2 run was compared with a completed ChatBob-modified ROBERT 2.1.2 run.

Primary DAT comparison:

- `CURATE/CURATE_data.dat`: identical after expected metadata normalization
- `GENERATE/GENERATE_data.dat`: identical after expected metadata normalization
- `VERIFY/VERIFY_data.dat`: identical after expected metadata normalization
- `PREDICT/PREDICT_data.dat`: identical after expected metadata normalization

Expected normalized differences:

- run timestamps,
- absolute input and output paths,
- module execution times,
- trailing whitespace.

CSV comparison:

- 28 matching CSV files were compared.
- All scientific numeric and text values matched.
- The only expected text difference was `csv_name` in `CURATE/CURATE_options.csv`.
- Best-model and prediction CSV files were identical.

Conclusion:

For this regression test case, the additive JSON-output hooks did not change ROBERT's standard scientific outputs.


## Implementation Update (2026-05-28)

Implemented now (wrapper-side, archive-only):
- `robert/json_output_for_agent.py` now includes helpers to build module file manifests from copied archive outputs.
- `json-output-for-agent/scripts/run_robert_timestamped.py` now generates:
  - `<MODULE>_manifest.json` for CURATE, GENERATE, VERIFY, PREDICT, REPORT (and AQME/EVALUATE when present),
  - `run_summary.json` with command, return code, module manifest index, and top-level artifact manifest,
  - `json_output_audit.json` with manifest write status.

What this means for capture timing:
- Current implementation captures after ROBERT run completion and after copy into timestamped archive.
- This keeps normal ROBERT outputs untouched and ensures JSON reflects final saved artifacts.

What is not implemented yet as of 2026-05-28:
- Module-native simultaneous JSON writes during `.dat`/CSV/image write points inside CURATE/GENERATE/VERIFY/PREDICT/REPORT.
- That path is still optional for later if real-time event streaming is required.
