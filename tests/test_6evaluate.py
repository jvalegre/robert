#!/usr/bin/env python

######################################################.
# 	          Testing EVALUATE with pytest   	     #
######################################################.

import os
import sys
import glob
import json
import shutil
import subprocess
import pandas as pd

from robert.evaluate import evaluate
from robert.curate import curate
from robert.verify import verify
from robert.predict import predict
from robert.report import report

# saves the working directory
path_main = os.getcwd()

folders_to_reset = ['CURATE', 'GENERATE', 'VERIFY', 'PREDICT', 'EVALUATE']
files_to_reset = ['report_debug_No_PFI.txt', 'report_debug_PFI.txt', 'ROBERT_report_No_PFI.pdf', 'ROBERT_report_PFI.pdf']


def _clean_evaluate_run():
    for folder in folders_to_reset:
        folder_path = os.path.join(path_main, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    for file in files_to_reset:
        file_path = os.path.join(path_main, file)
        if os.path.exists(file_path):
            os.remove(file_path)


def test_EVALUATE():
    # leave the folders as they were initially to run a different batch of tests
    _clean_evaluate_run()

    # EVALUATE chains CURATE (Pearson only) + VERIFY + PREDICT + REPORT on top of the
    # provided train/valid databases, so a successful run is a reasonable smoke test that
    # the whole downstream pipeline still works, not just the EVALUATE module itself
    cmd_robert = [
        sys.executable,
        "-m",
        "robert",
        "--evaluate",
        "--y", "Target_values",
        "--names", "Name",
        "--csv_name", "tests/Evaluate_train.csv",
        # EVALUATE calls load_database() with print_info=False, which skips sanity_checks()
        # entirely - that's the only place the "auto" split default gets resolved into a real
        # split method (even/stratified), so the default "auto" currently reaches test_select() as a
        # literal, unhandled string there. Passing an explicit split avoids that (separate,
        # pre-existing bug, not touched here).
        "--split", "even",
        "--debug_report", "True",
    ]

    subprocess.run(cmd_robert)

    # EVALUATE's own direct output (independent of VERIFY/PREDICT/REPORT working)
    generate_no_pfi = os.path.join(path_main, "GENERATE", "Best_model", "No_PFI")
    assert os.path.exists(os.path.join(generate_no_pfi, "MVL_db.csv"))
    assert os.path.exists(os.path.join(generate_no_pfi, "MVL.csv"))
    # EVALUATE only ever produces a No PFI model
    assert not os.path.exists(os.path.join(path_main, "GENERATE", "Best_model", "PFI"))

    # VERIFY and PREDICT ran on top of the evaluated model
    assert len(glob.glob(os.path.join(path_main, "VERIFY", "*.dat"))) == 1
    assert len(glob.glob(os.path.join(path_main, "PREDICT", "*.dat"))) == 1

    # REPORT completed and produced the (No PFI only) PDF
    assert os.path.exists(os.path.join(path_main, "ROBERT_report_No_PFI.pdf"))
    assert not os.path.exists(os.path.join(path_main, "ROBERT_report_PFI.pdf"))

    # sanity check that the debug report log doesn't contain an unhandled traceback
    debug_report = os.path.join(path_main, "report_debug_No_PFI.txt")
    assert os.path.exists(debug_report)
    with open(debug_report, "r", encoding="utf-8") as f:
        debug_content = f.read()
    assert "Traceback (most recent call last)" not in debug_content

    # leave the folders as they were at the end of the test
    _clean_evaluate_run()


def test_EVALUATE_module():
    """
    Same EVALUATE workflow as test_EVALUATE() above, but calling evaluate() directly (the way
    a Jupyter/notebook user would) instead of through "python -m robert" via subprocess.run() -
    a subprocess is invisible to coverage tools, so this is what actually exercises
    evaluate.py (and the REPORT/VERIFY/PREDICT chain it drives) under coverage.

    Only regression is covered here: EVALUATE currently rejects type="clas" outright (see the
    "not valid in EVALUATE... 'clas' option will be added soon" check in utils.py) and its only
    supported eval_model (MVL/LinearRegression) is regression-only, so a classification variant
    of this test isn't possible until that support is added to the module itself.
    """
    _clean_evaluate_run()

    evaluate(
        y="Target_values",
        names="Name",
        csv_name="tests/Evaluate_train.csv",
        # see the comment on the "--split" cmd arg above: sidesteps the same pre-existing
        # "auto" split resolution bug, not touched here
        split="even",
    )

    # EVALUATE only builds the GENERATE/Best_model files - it doesn't run VERIFY/PREDICT/REPORT
    # itself. "python -m robert --evaluate" chains those afterward (see the "EVALUATE, only
    # evaluates models" branch in robert.py's main()); replicate that same chain here, calling
    # each module directly like the "--evaluate" CLI path does
    curate(y="Target_values", names="Name", csv_name="tests/Evaluate_train.csv")
    verify(ignore=["Set"])
    predict(ignore=["Set"])
    report(ignore=["Set"], debug_report=True)

    # EVALUATE's own direct output (independent of VERIFY/PREDICT/REPORT working)
    generate_no_pfi = os.path.join(path_main, "GENERATE", "Best_model", "No_PFI")
    assert os.path.exists(os.path.join(generate_no_pfi, "MVL_db.csv"))
    assert os.path.exists(os.path.join(generate_no_pfi, "MVL.csv"))
    # EVALUATE only ever produces a No PFI model
    assert not os.path.exists(os.path.join(path_main, "GENERATE", "Best_model", "PFI"))

    # VERIFY and PREDICT ran on top of the evaluated model
    assert len(glob.glob(os.path.join(path_main, "VERIFY", "*.dat"))) == 1
    assert len(glob.glob(os.path.join(path_main, "PREDICT", "*.dat"))) == 1

    # REPORT completed and produced the (No PFI only) PDF
    assert os.path.exists(os.path.join(path_main, "ROBERT_report_No_PFI.pdf"))
    assert not os.path.exists(os.path.join(path_main, "ROBERT_report_PFI.pdf"))

    debug_report = os.path.join(path_main, "report_debug_No_PFI.txt")
    assert os.path.exists(debug_report)
    with open(debug_report, "r", encoding="utf-8") as f:
        debug_content = f.read()
    assert "Traceback (most recent call last)" not in debug_content

    # deeper content checks (same spirit as the full-workflow pytests in
    # test_5aqme_n_full.py) - confirms the ROBERT score section actually rendered with real
    # data instead of just checking that the PDF file exists
    assert "Section A. ROBERT Score" in debug_content
    assert "Interpolation" in debug_content
    assert "Boundary robustness" in debug_content  # MVL is regression, so this column exists
    score_imgs = [
        line for line in debug_content.splitlines()
        if "report/score_" in line and "report/score_w" not in line
    ]
    assert len(score_imgs) == 2  # Interpolation + Boundary robustness

    # plain 'MVL' has no user-tunable hyperparameters, so the data-leakage banner (only
    # relevant when the user supplied their own hyperparameters) must NOT appear here
    assert "POSSIBLE DATA LEAKAGE" not in debug_content

    # leave the folders as they were at the end of the test
    _clean_evaluate_run()


def test_EVALUATE_custom_sklearn_model_and_user_split():
    """
    EVALUATE now accepts an arbitrary scikit-learn model (--model_params, restricted to
    sklearn's own registered estimators - see resolve_sklearn_estimator() in utils.py) instead
    of only the legacy 'MVL', and a user-provided 'Set' column to pick the test set directly
    instead of always auto-splitting with test_select().
    """
    _clean_evaluate_run()

    # combine train+valid into one CSV, relabeling the valid rows 'Test' - exercises the new
    # user-defined split (get_user_defined_test_points() in evaluate.py)
    combined = pd.concat(
        [pd.read_csv("tests/Evaluate_train.csv"), pd.read_csv("tests/Evaluate_valid.csv").assign(Set="Test")],
        ignore_index=True,
    )
    combined_path = os.path.join(path_main, "test_evaluate_combined.csv")
    combined.to_csv(combined_path, index=False)

    model_params_path = os.path.join(path_main, "test_evaluate_model_params.csv")
    pd.DataFrame({"param": ["model", "alpha", "random_state"], "value": ["Ridge", "0.5", "0"]}).to_csv(
        model_params_path, index=False
    )

    try:
        evaluate(y="Target_values", names="Name", csv_name=combined_path, model_params=model_params_path)
        curate(y="Target_values", names="Name", csv_name=combined_path)
        verify(ignore=["Set"])
        predict(ignore=["Set"])
        report(ignore=["Set"], debug_report=True)

        # the model resolved from model_params, not the legacy 'MVL' fallback
        params_csv = os.path.join(path_main, "GENERATE", "Best_model", "No_PFI", "Ridge.csv")
        assert os.path.exists(params_csv)
        params_row = pd.read_csv(params_csv).iloc[0]
        assert params_row["model"] == "Ridge"
        assert json.loads(params_row["params"])["alpha"] == 0.5

        # the 'Set' column was honored directly: 22 Training rows, 15 relabeled Test rows
        with open(os.path.join(path_main, "PREDICT", "PREDICT_data.dat"), encoding="utf-8") as f:
            predict_content = f.read()
        assert "- Training points: 22" in predict_content
        assert "- Test points: 15" in predict_content

        debug_report = os.path.join(path_main, "report_debug_No_PFI.txt")
        with open(debug_report, "r", encoding="utf-8") as f:
            debug_content = f.read()
        assert "Traceback (most recent call last)" not in debug_content
        assert "sklearn model: Ridge" in debug_content
        assert "alpha: 0.5" in debug_content

        # a user-supplied model (Ridge, via model_params) DOES carry the data-leakage risk:
        # its hyperparameters weren't derived from ROBERT's own held-out split, so the big
        # warning banner in Section A must appear (see print_warnings() in report.py)
        assert "POSSIBLE DATA LEAKAGE" in debug_content
        assert "both the CV and Test scores below" in debug_content
    finally:
        for path in (combined_path, model_params_path):
            if os.path.exists(path):
                os.remove(path)
        _clean_evaluate_run()


def test_EVALUATE_classification():
    """
    Classification support in EVALUATE (previously rejected outright - see the removed "not
    valid in EVALUATE... clas option will be added soon" check), driven through the same
    --model_params mechanism as the regression test above.
    """
    _clean_evaluate_run()

    combined = pd.concat(
        [pd.read_csv("tests/Evaluate_clas_train.csv"), pd.read_csv("tests/Evaluate_clas_valid.csv").assign(Set="Test")],
        ignore_index=True,
    )
    combined_path = os.path.join(path_main, "test_evaluate_clas_combined.csv")
    combined.to_csv(combined_path, index=False)

    model_params_path = os.path.join(path_main, "test_evaluate_clas_model_params.csv")
    pd.DataFrame(
        {"param": ["model", "n_estimators", "max_depth"], "value": ["RandomForestClassifier", "50", "5"]}
    ).to_csv(model_params_path, index=False)

    try:
        evaluate(
            y="Target_values", names="Name", csv_name=combined_path,
            model_params=model_params_path, type="clas",
        )
        curate(y="Target_values", names="Name", csv_name=combined_path, type="clas")
        verify(ignore=["Set"])
        predict(ignore=["Set"])
        report(ignore=["Set"], debug_report=True)

        params_csv = os.path.join(path_main, "GENERATE", "Best_model", "No_PFI", "RandomForestClassifier.csv")
        assert os.path.exists(params_csv)
        params_row = pd.read_csv(params_csv).iloc[0]
        assert params_row["type"] == "clas"
        assert params_row["error_type"] == "mcc"
        resolved_params = json.loads(params_row["params"])
        assert resolved_params["n_estimators"] == 50
        assert resolved_params["max_depth"] == 5

        with open(os.path.join(path_main, "PREDICT", "PREDICT_data.dat"), encoding="utf-8") as f:
            predict_content = f.read()
        assert "- Training points: 22" in predict_content
        assert "- Test points: 15" in predict_content
        assert "MCC" in predict_content

        debug_report = os.path.join(path_main, "report_debug_No_PFI.txt")
        with open(debug_report, "r", encoding="utf-8") as f:
            debug_content = f.read()
        assert "Traceback (most recent call last)" not in debug_content
        # classification has no Boundary robustness score (see score.rst) - only Interpolation
        # gets a real score column; the right column instead shows a short note explaining why
        # it's disabled (see print_score() in report.py) rather than being left blank
        assert "Boundary robustness" in debug_content
        assert "Disabled in classification problems" in debug_content
        score_imgs = [
            line for line in debug_content.splitlines()
            if "report/score_" in line and "report/score_w" not in line
        ]
        assert len(score_imgs) == 1  # Interpolation only - no Boundary robustness score bar/image
        # RandomForestClassifier's hyperparameters came from the user, same leakage risk as
        # the regression case above - the banner isn't regression-specific
        assert "POSSIBLE DATA LEAKAGE" in debug_content
    finally:
        for path in (combined_path, model_params_path):
            if os.path.exists(path):
                os.remove(path)
        _clean_evaluate_run()
