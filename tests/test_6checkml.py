#!/usr/bin/env python

######################################################.
# 	          Testing CHECKML with pytest   	     #
######################################################.

import os
import sys
import glob
import json
import shutil
import subprocess
import pytest
import pandas as pd

from robert.checkml import checkml
from robert.curate import curate
from robert.verify import verify
from robert.predict import predict
from robert.report import report

# saves the working directory
path_main = os.getcwd()

folders_to_reset = ['CURATE', 'GENERATE', 'VERIFY', 'PREDICT', 'CHECKML']
files_to_reset = ['report_debug_No_PFI.txt', 'report_debug_PFI.txt', 'ROBERT_report_No_PFI.pdf', 'ROBERT_report_PFI.pdf']


def _clean_checkml_run():
    for folder in folders_to_reset:
        folder_path = os.path.join(path_main, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    for file in files_to_reset:
        file_path = os.path.join(path_main, file)
        if os.path.exists(file_path):
            os.remove(file_path)


def test_CHECKML():
    # leave the folders as they were initially to run a different batch of tests
    _clean_checkml_run()

    # CHECKML chains CURATE (Pearson only) + VERIFY + PREDICT + REPORT on top of the
    # provided train/valid databases, so a successful run is a reasonable smoke test that
    # the whole downstream pipeline still works, not just the CHECKML module itself
    cmd_robert = [
        sys.executable,
        "-m",
        "robert",
        "--checkml",
        "--y", "Target_values",
        "--names", "Name",
        "--csv_name", "tests/CheckML_train.csv",
        "--debug_report", "True",
    ]

    subprocess.run(cmd_robert)

    # CHECKML's own direct output (independent of VERIFY/PREDICT/REPORT working)
    generate_no_pfi = os.path.join(path_main, "GENERATE", "Best_model", "No_PFI")
    assert os.path.exists(os.path.join(generate_no_pfi, "MVL_db.csv"))
    assert os.path.exists(os.path.join(generate_no_pfi, "MVL.csv"))
    # CHECKML only ever produces a No PFI model
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
    _clean_checkml_run()


def test_CHECKML_module():
    """
    Same CHECKML workflow as test_CHECKML() above, but calling checkml() directly (the way
    a Jupyter/notebook user would) instead of through "python -m robert" via subprocess.run() -
    a subprocess is invisible to coverage tools, so this is what actually exercises
    checkml.py (and the REPORT/VERIFY/PREDICT chain it drives) under coverage.

    This one covers regression with the default MVL model, and the classification variant is
    test_CHECKML_classification() below.
    """
    _clean_checkml_run()

    checkml(
        y="Target_values",
        names="Name",
        csv_name="tests/CheckML_train.csv",
    )

    # CHECKML only builds the GENERATE/Best_model files - it doesn't run VERIFY/PREDICT/REPORT
    # itself. "python -m robert --checkml" chains those afterward (see the "CHECKML, only
    # evaluates models" branch in robert.py's main()); replicate that same chain here, calling
    # each module directly like the "--checkml" CLI path does. The whole chain is needed
    # because the assertions below check the final report (score, warnings, banners), which
    # only exists once REPORT has run on top of VERIFY and PREDICT
    curate(y="Target_values", names="Name", csv_name="tests/CheckML_train.csv")
    verify(ignore=["Set"])
    predict(ignore=["Set"])
    report(ignore=["Set"], debug_report=True)

    # CHECKML's own direct output (independent of VERIFY/PREDICT/REPORT working)
    generate_no_pfi = os.path.join(path_main, "GENERATE", "Best_model", "No_PFI")
    assert os.path.exists(os.path.join(generate_no_pfi, "MVL_db.csv"))
    assert os.path.exists(os.path.join(generate_no_pfi, "MVL.csv"))
    # CHECKML only ever produces a No PFI model
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
    # test_5aqme_n_full.py) - confirms the CheckML score section actually rendered with real
    # data instead of just checking that the PDF file exists
    assert "Section A. CheckML Score" in debug_content
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
    _clean_checkml_run()


def test_CHECKML_custom_sklearn_model_and_user_split():
    """
    CHECKML now accepts an arbitrary scikit-learn model (--model_params, restricted to
    sklearn's own registered estimators - see resolve_sklearn_estimator() in utils.py) instead
    of only the legacy 'MVL', and a user-provided 'Set' column to pick the test set directly
    instead of always auto-splitting with test_select().
    """
    _clean_checkml_run()

    # combine train+valid into one CSV, relabeling the valid rows 'Test' - exercises the new
    # user-defined split (get_user_defined_test_points() in checkml.py)
    combined = pd.concat(
        [pd.read_csv("tests/CheckML_train.csv"), pd.read_csv("tests/CheckML_valid.csv").assign(Set="Test")],
        ignore_index=True,
    )
    combined_path = os.path.join(path_main, "test_checkml_combined.csv")
    combined.to_csv(combined_path, index=False)

    model_params_path = os.path.join(path_main, "test_checkml_model_params.csv")
    pd.DataFrame({"param": ["model", "alpha", "random_state"], "value": ["Ridge", "0.5", "0"]}).to_csv(
        model_params_path, index=False
    )

    try:
        # same CHECKML -> CURATE -> VERIFY -> PREDICT -> REPORT chain as test_CHECKML_module():
        # the assertions below inspect the final report, which needs the whole chain
        checkml(y="Target_values", names="Name", csv_name=combined_path, model_params=model_params_path)
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

        # a user-supplied model (Ridge, via model_params) carries the optimistic-scores risk
        # (its hyperparameters weren't derived from ROBERT's own held-out split), but the big
        # warning banner in Section A is disabled on request (see print_warnings() in report.py)
        assert "SCORES MAY BE OPTIMISTIC" not in debug_content

        # the user forced the test set (15 points instead of ROBERT's systematic ~20% split),
        # so the score isn't calibrated for this run and must NOT be calculated
        assert "Score not available" in debug_content
        assert "chosen by the user (Set column)" in debug_content
    finally:
        for path in (combined_path, model_params_path):
            if os.path.exists(path):
                os.remove(path)
        _clean_checkml_run()


def test_CHECKML_classification():
    """
    Classification support in CHECKML (previously rejected outright - see the removed "not
    valid in CHECKML... clas option will be added soon" check), driven through the same
    --model_params mechanism as the regression test above.
    """
    _clean_checkml_run()

    combined = pd.concat(
        [pd.read_csv("tests/CheckML_clas_train.csv"), pd.read_csv("tests/CheckML_clas_valid.csv").assign(Set="Test")],
        ignore_index=True,
    )
    combined_path = os.path.join(path_main, "test_checkml_clas_combined.csv")
    combined.to_csv(combined_path, index=False)

    model_params_path = os.path.join(path_main, "test_checkml_clas_model_params.csv")
    pd.DataFrame(
        {"param": ["model", "n_estimators", "max_depth"], "value": ["RandomForestClassifier", "50", "5"]}
    ).to_csv(model_params_path, index=False)

    try:
        # same CHECKML -> CURATE -> VERIFY -> PREDICT -> REPORT chain as test_CHECKML_module():
        # the assertions below inspect the final report, which needs the whole chain
        checkml(
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
        # the regression case above, but the banner is disabled on request (see above)
        assert "SCORES MAY BE OPTIMISTIC" not in debug_content
        # same for the user-forced test set (15 points): no calibrated score
        assert "Score not available" in debug_content
        assert "chosen by the user (Set column)" in debug_content
    finally:
        for path in (combined_path, model_params_path):
            if os.path.exists(path):
                os.remove(path)
        _clean_checkml_run()


def test_CHECKML_user_set_of_standard_size_has_no_score():
    """
    A test set chosen by the user through the 'Set' column never gets a score, not even when it
    happens to have the standard ~20% size (7 of the 37 points) that would otherwise pass the
    test-set size check - it isn't ROBERT's own systematic split the score was calibrated for.
    """
    _clean_checkml_run()

    combined = pd.concat(
        [pd.read_csv("tests/CheckML_train.csv"), pd.read_csv("tests/CheckML_valid.csv")],
        ignore_index=True,
    )
    combined["Set"] = ["Training"] * (len(combined) - 7) + ["Test"] * 7
    combined_path = os.path.join(path_main, "test_checkml_std_set.csv")
    combined.to_csv(combined_path, index=False)

    try:
        # same CHECKML -> CURATE -> VERIFY -> PREDICT -> REPORT chain as test_CHECKML_module():
        # the assertions below inspect the final report, which needs the whole chain
        checkml(y="Target_values", names="Name", csv_name=combined_path)
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
        _clean_checkml_run()


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


def test_CHECKML_text_descriptor_column():
    """
    CHECKML doesn't one-hot encode (unlike CURATE), so a text descriptor column (i.e. SMILES)
    must stop the program with a clear message instead of failing later inside scikit-learn.
    Ignoring that column lets it run normally.
    """
    import pytest

    _clean_checkml_run()

    df = pd.read_csv("tests/CheckML_train.csv")
    df["smiles"] = ["C"] * len(df)
    csv_path = os.path.join(path_main, "test_checkml_text_col.csv")
    df.to_csv(csv_path, index=False)

    try:
        with pytest.raises(SystemExit):
            checkml(y="Target_values", names="Name", csv_name=csv_path)
        with open(os.path.join(path_main, "CHECKML", "CHECKML_data.dat"), encoding="utf-8") as f:
            assert "contain text" in f.read()
        assert not os.path.exists(os.path.join(path_main, "GENERATE"))

        _clean_checkml_run()
        checkml(y="Target_values", names="Name", csv_name=csv_path, ignore=["smiles"])
        assert os.path.exists(os.path.join(path_main, "GENERATE", "Best_model", "No_PFI", "MVL.csv"))
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        _clean_checkml_run()


def test_CHECKML_missing_descriptor_value():
    """
    Unlike the full ROBERT pipeline (where CURATE fills in or removes missing data before
    GENERATE ever sees it), CHECKML calls load_database() directly, which would otherwise
    silently drop/impute a descriptor with missing values with zero warning. This must stop
    the program instead, always (regardless of --auto_fill - see the design discussion in
    checkml.py). Ignoring that column lets it run normally.
    """
    import pytest

    _clean_checkml_run()

    df = pd.read_csv("tests/CheckML_train.csv")
    df.loc[2, "x2"] = None
    csv_path = os.path.join(path_main, "test_checkml_missing_x.csv")
    df.to_csv(csv_path, index=False)

    try:
        with pytest.raises(SystemExit):
            checkml(y="Target_values", names="Name", csv_name=csv_path, auto_fill=True)
        with open(os.path.join(path_main, "CHECKML", "CHECKML_data.dat"), encoding="utf-8") as f:
            debug_content = f.read()
        assert "descriptor column (x2) has empty values in row(s) 4" in debug_content
        assert not os.path.exists(os.path.join(path_main, "GENERATE"))

        _clean_checkml_run()
        checkml(y="Target_values", names="Name", csv_name=csv_path, ignore=["x2"])
        assert os.path.exists(os.path.join(path_main, "GENERATE", "Best_model", "No_PFI", "MVL.csv"))
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        _clean_checkml_run()


# ---------------------------------------------------------------------------------------------
# User-supplied scikit-learn models (--model_params / --model_file / model_obj)
# ---------------------------------------------------------------------------------------------

_CUSTOM_DESCRIPTORS = ["x2", "x5", "x6", "x7", "x9", "x10"]


def _write_custom_model_csvs():
    """Numeric-only regression/classification databases (CHECKML uses descriptors as they are)."""
    reg = pd.read_csv("tests/Robert_example.csv")[["Name", "Target_values"] + _CUSTOM_DESCRIPTORS]
    reg.drop_duplicates("Name").to_csv("custom_reg.csv", index=False)
    clas = pd.read_csv("tests/Robert_example_clas.csv")[["Name", "Target_values"] + _CUSTOM_DESCRIPTORS]
    clas.to_csv("custom_clas.csv", index=False)
    clas_txt = clas.copy()
    clas_txt["Target_values"] = clas_txt["Target_values"].map({0: "inactive", 1: "active"})
    clas_txt.to_csv("custom_clas_txt.csv", index=False)


def _write_model_params(rows):
    pd.DataFrame(rows, columns=["param", "value"]).to_csv("custom_model_params.csv", index=False)
    return "custom_model_params.csv"


def test_model_params_serialization_roundtrip():
    """
    The hyperparameters are saved as text in GENERATE/Best_model and read back by
    VERIFY/PREDICT/REPORT to rebuild the model - every type must come back exactly as it was
    """
    import numpy as np
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.tree import DecisionTreeRegressor
    from robert.utils import serialize_model_params, deserialize_model_params, dict_formating

    kernel = RBF(1.0) + WhiteKernel(0.1)
    params = {
        "kernel": kernel,  # object (GP kernel)
        "estimator": DecisionTreeRegressor(max_depth=2),  # object (base estimator)
        "class_weight": {0: 1, 1: 3},  # dict with integer keys
        "hidden_layer_sizes": (20, 10),  # tuple
        "max_depth": np.int64(3),  # numpy int (i.e. from a grid search)
        "alpha": np.float64(0.5),
        "criterion": "gini",
        "max_features": None,
        "bootstrap": True,
    }
    restored = deserialize_model_params(serialize_model_params(params))
    assert restored["kernel"] == kernel
    assert restored["estimator"].get_params()["max_depth"] == 2
    assert restored["class_weight"] == {0: 1, 1: 3}
    assert restored["hidden_layer_sizes"] == (20, 10)
    assert restored["max_depth"] == 3 and type(restored["max_depth"]) is int
    assert restored["alpha"] == 0.5 and type(restored["alpha"]) is float
    assert restored["criterion"] == "gini"
    assert restored["max_features"] is None
    assert restored["bootstrap"] is True

    # params saved before serialize_model_params() existed (plain JSON with text keys) still load
    old_format = {"params": json.dumps({"class_weight": {"0": 1, "1": 3}, "max_depth": 3})}
    assert dict_formating(old_format)["params"] == {"class_weight": {0: 1, 1: 3}, "max_depth": 3}
    # class names (non-numeric keys) are kept as they are
    names_format = {"params": json.dumps({"class_weight": {"active": 3, "inactive": 1}})}
    assert dict_formating(names_format)["params"]["class_weight"] == {"active": 3, "inactive": 1}


def test_model_params_csv_values():
    """Values typed in the --model_params CSV, including the ways Excel writes them"""
    from robert.utils import parse_param_value

    assert parse_param_value("3") == 3
    assert parse_param_value(" 0.1 ") == 0.1
    assert parse_param_value("(20, 10)") == (20, 10)
    assert parse_param_value("{0: 1, 1: 3}") == {0: 1, 1: 3}
    assert parse_param_value("[1, 2]") == [1, 2]
    assert parse_param_value("gini") == "gini"
    assert parse_param_value("'sqrt'") == "sqrt"
    for empty in [None, float("nan"), "", "  ", "None", "none", "NULL"]:
        assert parse_param_value(empty) is None
    for true in ["True", "TRUE", "true"]:
        assert parse_param_value(true) is True
    for false in ["False", "FALSE", "false"]:
        assert parse_param_value(false) is False


def _custom_model_options(scenario):
    import joblib
    import numpy as np

    opts = {"y": "Target_values", "names": "Name"}
    if scenario == "clas_class_weight_int":
        opts.update(csv_name="custom_clas.csv", type="clas", model_params=_write_model_params(
            [("model", "DecisionTreeClassifier"), ("max_depth", "3"), ("class_weight", "{0: 1, 1: 3}")]))
    elif scenario == "clas_class_weight_labels":
        opts.update(csv_name="custom_clas_txt.csv", type="clas", model_params=_write_model_params(
            [("model", "DecisionTreeClassifier"), ("max_depth", "3"), ("class_weight", "{'active': 3, 'inactive': 1}")]))
    elif scenario == "reg_model_file_gp_kernel":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        joblib.dump(GaussianProcessRegressor(kernel=RBF(1.0) + WhiteKernel(0.1), normalize_y=True), "custom_gp.joblib")
        opts.update(csv_name="custom_reg.csv", model_file="custom_gp.joblib")
    elif scenario == "reg_model_file_bagging":
        from sklearn.ensemble import BaggingRegressor
        from sklearn.tree import DecisionTreeRegressor
        joblib.dump(BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=10), "custom_bag.joblib")
        opts.update(csv_name="custom_reg.csv", model_file="custom_bag.joblib")
    elif scenario == "reg_model_obj_numpy_ints":
        from sklearn.ensemble import RandomForestRegressor
        opts.update(csv_name="custom_reg.csv",
                    model_obj=RandomForestRegressor(n_estimators=np.int64(20), max_depth=np.int64(3)))
    elif scenario == "reg_csv_quirks":
        opts.update(csv_name="custom_reg.csv", model_params=_write_model_params(
            [("model", "RandomForestRegressor"), (" n_estimators ", " 20 "), ("max_depth", None),
             ("bootstrap", "TRUE"), ("oob_score", "false"), ("max_features", "sqrt")]))
    elif scenario == "reg_csv_tuple_mlp":
        opts.update(csv_name="custom_reg.csv", model_params=_write_model_params(
            [("model", "MLPRegressor"), ("hidden_layer_sizes", "(20, 10)"), ("max_iter", "2000"), ("solver", "lbfgs")]))
    elif scenario == "clas_class_weight_bad_key":
        opts.update(csv_name="custom_clas.csv", type="clas", model_params=_write_model_params(
            [("model", "DecisionTreeClassifier"), ("class_weight", "{0: 1, 5: 3}")]))
    elif scenario == "clas_typo_value":
        opts.update(csv_name="custom_clas.csv", type="clas", model_params=_write_model_params(
            [("model", "DecisionTreeClassifier"), ("criterion", "ginni")]))
    return opts


@pytest.mark.parametrize(
    "scenario",
    [
        "clas_class_weight_int",  # class_weight={0: 1, 1: 3} (JSON used to turn the keys into text)
        "clas_class_weight_labels",  # class_weight typed with the original text labels
        "reg_model_file_gp_kernel",  # fitted model with a kernel object (nested get_params keys)
        "reg_model_file_bagging",  # model with a base estimator object
        "reg_model_obj_numpy_ints",  # numpy ints (i.e. hyperparameters from a grid search)
        "reg_csv_quirks",  # spaces, empty cell (None) and TRUE/false as Excel writes them
        "reg_csv_tuple_mlp",  # tuple hyperparameter
    ],
)
def test_CHECKML_custom_sklearn_models(scenario):
    """
    CHECKML saves the user's hyperparameters and VERIFY rebuilds the model from them and fits it
    in every CV fold - this used to fail for all of these models except the tuple one
    """
    _clean_checkml_run()
    _write_custom_model_csvs()
    opts = _custom_model_options(scenario)
    try:
        checkml(**opts)
        chain = {key: val for key, val in opts.items() if key not in ("model_params", "model_file", "model_obj")}
        curate(**chain)
        verify(ignore=["Set"])
        assert len(glob.glob(os.path.join(path_main, "VERIFY", "*.dat"))) == 1
        assert len(glob.glob(os.path.join(path_main, "VERIFY", "VERIFY_tests_*_No_PFI.png"))) == 1
        if scenario == "clas_class_weight_labels":
            with open(os.path.join(path_main, "CHECKML", "CHECKML_data.dat"), encoding="utf-8") as log_file:
                assert "class_weight keys converted to the class codes used by ROBERT" in log_file.read()
    finally:
        _clean_checkml_run()


@pytest.mark.parametrize(
    "scenario,expected_message",
    [
        ("clas_class_weight_bad_key", "The class_weight key 5 doesn't match any class of the y column"),
        ("clas_typo_value", "Invalid hyperparameters for DecisionTreeClassifier"),
    ],
)
def test_CHECKML_custom_sklearn_models_clear_errors(scenario, expected_message):
    """Wrong hyperparameters stop CHECKML with a clear message (not a traceback inside VERIFY)"""
    _clean_checkml_run()
    _write_custom_model_csvs()
    try:
        with pytest.raises(SystemExit):
            checkml(**_custom_model_options(scenario))
        with open(os.path.join(path_main, "CHECKML", "CHECKML_data.dat"), encoding="utf-8") as log_file:
            assert expected_message in log_file.read()
    finally:
        _clean_checkml_run()


# ---------------------------------------------------------------------------------------------
# Order of rows and columns
# ---------------------------------------------------------------------------------------------

def test_CHECKML_row_and_column_order_do_not_change_results():
    """
    ROBERT sorts rows by the names column and descriptors alphabetically when loading a database,
    so shuffling the rows/columns of the same CSV must give exactly the same test set, CV folds,
    VERIFY tests and PREDICT metrics/predictions (before, the row order changed which points
    landed in each fold, and therefore the scores)
    """
    base = pd.read_csv("tests/Robert_example.csv")[["Name", "Target_values"] + _CUSTOM_DESCRIPTORS].drop_duplicates("Name")
    shuffled = base.sample(frac=1, random_state=7)[["x9", "Name", "x10", "Target_values", "x2", "x7", "x5", "x6"]]
    assert list(shuffled["Name"]) != list(base["Name"])

    results = []
    try:
        for csv_name, df in [("order_original.csv", base), ("order_shuffled.csv", shuffled)]:
            _clean_checkml_run()
            df.to_csv(csv_name, index=False)
            model_params = _write_model_params(
                [("model", "RandomForestRegressor"), ("n_estimators", "20"), ("max_depth", "3")])
            checkml(y="Target_values", names="Name", csv_name=csv_name, model_params=model_params)
            verify(ignore=["Set"])
            predict(ignore=["Set"])

            best = os.path.join(path_main, "GENERATE", "Best_model", "No_PFI")
            db = pd.read_csv(os.path.join(best, "RandomForestRegressor_db.csv"))
            params = pd.read_csv(os.path.join(best, "RandomForestRegressor.csv"))
            with open(os.path.join(path_main, "VERIFY", "VERIFY_data.dat"), encoding="utf-8") as verify_file:
                verify_dat = verify_file.read()
            with open(os.path.join(path_main, "PREDICT", "PREDICT_data.dat"), encoding="utf-8") as predict_file:
                predict_dat = predict_file.read()
            preds = pd.read_csv(os.path.join(path_main, "PREDICT", "RandomForestRegressor_No_PFI.csv"))
            results.append({
                "sets": dict(zip(db["Name"], db["Set"])),
                "descriptors": params["X_descriptors"][0],
                "verify": [line.strip() for line in verify_dat.split("\n")
                           if any(tag in line for tag in ["y_mean", "y_shuffle", "onehot", "cluster", "Sorted"])],
                "predict": [line.strip() for line in predict_dat.split("\n")
                            if "10x 5-fold CV :" in line or "Test :" in line or "Sorted CV extremes" in line],
                "predictions": preds.sort_values("Name").reset_index(drop=True).round(10),
            })
    finally:
        _clean_checkml_run()

    original, reordered = results
    assert original["sets"] == reordered["sets"]
    assert original["descriptors"] == reordered["descriptors"]
    assert original["verify"] and original["verify"] == reordered["verify"]
    assert original["predict"] and original["predict"] == reordered["predict"]
    pd.testing.assert_frame_equal(original["predictions"][sorted(original["predictions"].columns)],
                                  reordered["predictions"][sorted(reordered["predictions"].columns)])


# ---------------------------------------------------------------------------------------------
# VERIFY onehot test with binary descriptors
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("with_binary_descriptor", [True, False])
def test_CHECKML_onehot_test_disabled_with_binary_descriptors(with_binary_descriptor):
    """
    Binarizing can't remove information from a descriptor that only has 2 values, so with one
    such column the onehot "flawed model" keeps that information and can predict as well as the
    original model - the test would fail without the model being flawed. It is disabled (N/A,
    like the cluster test in classification) as soon as one descriptor has only 2 values, and
    REPORT must score the remaining tests without crashing on the N/A line
    """
    _clean_checkml_run()
    df = pd.read_csv("tests/Robert_example.csv")[["Name", "Target_values", "x2", "x5", "x6"]].drop_duplicates("Name")
    if with_binary_descriptor:
        df["flag"] = (df["x2"] > df["x2"].median()).astype(int)
    df.to_csv("onehot_check.csv", index=False)
    try:
        checkml(y="Target_values", names="Name", csv_name="onehot_check.csv")
        verify(ignore=["Set"])
        predict(ignore=["Set"])
        report(ignore=["Set"], debug_report=True)

        with open(os.path.join(path_main, "VERIFY", "VERIFY_data.dat"), encoding="utf-8") as verify_file:
            onehot_line = next(line for line in verify_file if "onehot:" in line)
        if with_binary_descriptor:
            assert "onehot: N/A, descriptor(s) with only 2 values (flag)" in onehot_line
        else:
            assert "N/A" not in onehot_line and "% difference." in onehot_line

        # REPORT reads the flawed-model tests (skipping the N/A line) and finishes the PDF
        assert os.path.exists(os.path.join(path_main, "ROBERT_report_No_PFI.pdf"))
        with open(os.path.join(path_main, "report_debug_No_PFI.txt"), encoding="utf-8") as debug_file:
            assert '1. Model vs "flawed" models' in debug_file.read()
    finally:
        if os.path.exists("onehot_check.csv"):
            os.remove("onehot_check.csv")
        _clean_checkml_run()
