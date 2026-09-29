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

    This one covers regression with the default MVL model, and the classification variant is
    test_EVALUATE_classification() below.
    """
    _clean_evaluate_run()

    evaluate(
        y="Target_values",
        names="Name",
        csv_name="tests/Evaluate_train.csv",
    )

    # EVALUATE only builds the GENERATE/Best_model files - it doesn't run VERIFY/PREDICT/REPORT
    # itself. "python -m robert --evaluate" chains those afterward (see the "EVALUATE, only
    # evaluates models" branch in robert.py's main()); replicate that same chain here, calling
    # each module directly like the "--evaluate" CLI path does. The whole chain is needed
    # because the assertions below check the final report (score, warnings, banners), which
    # only exists once REPORT has run on top of VERIFY and PREDICT
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

    # plain 'MVL' has no user-tunable hyperparameters, so the optimistic-scores banner (only
    # relevant when the user supplied their own hyperparameters) must NOT appear here
    assert "SCORES MAY BE OPTIMISTIC" not in debug_content

    # the test set was picked by ROBERT's own systematic split, so the score is calibrated
    # and must be calculated normally
    assert "Score not available" not in debug_content

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
        # same EVALUATE -> CURATE -> VERIFY -> PREDICT -> REPORT chain as test_EVALUATE_module():
        # the assertions below inspect the final report, which needs the whole chain
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

        # a user-supplied model (Ridge, via model_params) DOES carry the optimistic-scores risk:
        # its hyperparameters weren't derived from ROBERT's own held-out split, so the big
        # warning banner in Section A must appear (see print_warnings() in report.py)
        assert "SCORES MAY BE OPTIMISTIC" in debug_content
        assert "both the CV and Test scores below" in debug_content

        # the user forced the test set (15 points instead of ROBERT's systematic ~20% split),
        # so the score isn't calibrated for this run and must NOT be calculated
        assert "Score not available" in debug_content
        assert "chosen by the user (Set column)" in debug_content
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
        # same EVALUATE -> CURATE -> VERIFY -> PREDICT -> REPORT chain as test_EVALUATE_module():
        # the assertions below inspect the final report, which needs the whole chain
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
        # RandomForestClassifier's hyperparameters came from the user, same optimistic-scores risk as
        # the regression case above - the banner isn't regression-specific
        assert "SCORES MAY BE OPTIMISTIC" in debug_content
        # same for the user-forced test set (15 points): no calibrated score
        assert "Score not available" in debug_content
        assert "chosen by the user (Set column)" in debug_content
    finally:
        for path in (combined_path, model_params_path):
            if os.path.exists(path):
                os.remove(path)
        _clean_evaluate_run()


def test_EVALUATE_user_set_of_standard_size_has_no_score():
    """
    A test set chosen by the user through the 'Set' column never gets a score, not even when it
    happens to have the standard ~20% size (7 of the 37 points) that would otherwise pass the
    test-set size check - it isn't ROBERT's own systematic split the score was calibrated for.
    """
    _clean_evaluate_run()

    combined = pd.concat(
        [pd.read_csv("tests/Evaluate_train.csv"), pd.read_csv("tests/Evaluate_valid.csv")],
        ignore_index=True,
    )
    combined["Set"] = ["Training"] * (len(combined) - 7) + ["Test"] * 7
    combined_path = os.path.join(path_main, "test_evaluate_std_set.csv")
    combined.to_csv(combined_path, index=False)

    try:
        # same EVALUATE -> CURATE -> VERIFY -> PREDICT -> REPORT chain as test_EVALUATE_module():
        # the assertions below inspect the final report, which needs the whole chain
        evaluate(y="Target_values", names="Name", csv_name=combined_path)
        curate(y="Target_values", names="Name", csv_name=combined_path)
        verify(ignore=["Set"])
        predict(ignore=["Set"])
        report(ignore=["Set"], debug_report=True)

        with open(os.path.join(path_main, "PREDICT", "PREDICT_data.dat"), encoding="utf-8") as f:
            assert "- Test points: 7" in f.read()

        with open(os.path.join(path_main, "report_debug_No_PFI.txt"), "r", encoding="utf-8") as f:
            debug_content = f.read()
        assert "Score not available" in debug_content
        assert "chosen by the user (Set column)" in debug_content
    finally:
        if os.path.exists(combined_path):
            os.remove(combined_path)
        _clean_evaluate_run()


def test_score_ratio_with_zero_error_baseline():
    """
    A near-perfect fit can round an error to exactly 0.00. A nonzero error over that zero
    baseline must be an unbounded ratio (worst case, never the best-case 0), and only 0/0
    means "no difference" - used by the CV-vs-test, train-vs-validation and degradation scores.
    """
    from robert.report_utils import safe_ratio

    assert safe_ratio(3.0, 2.0) == 1.5
    assert safe_ratio(0, 0) == 0
    assert safe_ratio(1.2, 0) == float("inf")
    # the resulting ratio falls in the worst scoring tier (> 1.5x) instead of the best (<= 1.25x)
    assert not safe_ratio(1.2, 0) <= 1.5


def test_EVALUATE_text_descriptor_column():
    """
    EVALUATE doesn't one-hot encode (unlike CURATE), so a text descriptor column (i.e. SMILES)
    must stop the program with a clear message instead of failing later inside scikit-learn.
    Ignoring that column lets it run normally.
    """
    import pytest

    _clean_evaluate_run()

    df = pd.read_csv("tests/Evaluate_train.csv")
    df["smiles"] = ["C"] * len(df)
    csv_path = os.path.join(path_main, "test_evaluate_text_col.csv")
    df.to_csv(csv_path, index=False)

    try:
        with pytest.raises(SystemExit):
            evaluate(y="Target_values", names="Name", csv_name=csv_path)
        with open(os.path.join(path_main, "EVALUATE", "EVALUATE_data.dat"), encoding="utf-8") as f:
            assert "contain text" in f.read()
        assert not os.path.exists(os.path.join(path_main, "GENERATE"))

        _clean_evaluate_run()
        evaluate(y="Target_values", names="Name", csv_name=csv_path, ignore=["smiles"])
        assert os.path.exists(os.path.join(path_main, "GENERATE", "Best_model", "No_PFI", "MVL.csv"))
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        _clean_evaluate_run()


def test_EVALUATE_missing_descriptor_value():
    """
    Unlike the full ROBERT pipeline (where CURATE fills in or removes missing data before
    GENERATE ever sees it), EVALUATE calls load_database() directly, which would otherwise
    silently drop/impute a descriptor with missing values with zero warning. This must stop
    the program instead, always (regardless of --auto_fill - see the design discussion in
    evaluate.py). Ignoring that column lets it run normally.
    """
    import pytest

    _clean_evaluate_run()

    df = pd.read_csv("tests/Evaluate_train.csv")
    df.loc[2, "x2"] = None
    csv_path = os.path.join(path_main, "test_evaluate_missing_x.csv")
    df.to_csv(csv_path, index=False)

    try:
        with pytest.raises(SystemExit):
            evaluate(y="Target_values", names="Name", csv_name=csv_path, auto_fill=True)
        with open(os.path.join(path_main, "EVALUATE", "EVALUATE_data.dat"), encoding="utf-8") as f:
            debug_content = f.read()
        assert "descriptor column (x2) has empty values in row(s) 4" in debug_content
        assert not os.path.exists(os.path.join(path_main, "GENERATE"))

        _clean_evaluate_run()
        evaluate(y="Target_values", names="Name", csv_name=csv_path, ignore=["x2"])
        assert os.path.exists(os.path.join(path_main, "GENERATE", "Best_model", "No_PFI", "MVL.csv"))
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        _clean_evaluate_run()
