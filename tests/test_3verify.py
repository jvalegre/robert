#!/usr/bin/env python

######################################################.
# 	        Testing VERIFY with pytest 	             #
######################################################.

import subprocess
import sys

import pytest
import shutil

from robert.verify import verify

from tests.conftest import (
    clas_generate_layout,
    restore_regression_generate_layout,
)
from tests.platform_goldens import assert_verify_standard_rmse_line


# VERIFY tests
@pytest.mark.parametrize(
    "test_job",
    [
        ("clas"),  # test for clasification
        ("standard"),  # standard test
        ("standard_cmd"),  # standard test with command line
    ],
)
def test_VERIFY(test_job, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    path_verify = repo_root / "VERIFY"

    if path_verify.is_dir():
        shutil.rmtree(path_verify)
    for dat_file in repo_root.glob("*.dat"):
        if "VERIFY" in dat_file.name:
            dat_file.unlink()

    if test_job == "clas":
        with clas_generate_layout(repo_root):
            _run_verify(test_job, repo_root, path_verify)
    else:
        restore_regression_generate_layout(repo_root)
        _run_verify(test_job, repo_root, path_verify)


def _run_verify(test_job, repo_root, path_verify):
    if test_job == "standard_cmd":
        cmd_robert = [sys.executable, "-m", "robert", "--verify"]
        subprocess.run(cmd_robert, cwd=repo_root, check=False)
    else:
        verify()

    assert not (repo_root / "VERIFY_data.dat").is_file()
    verify_dat = path_verify / "VERIFY_data.dat"
    with verify_dat.open(encoding="utf-8") as outfile:
        outlines = outfile.readlines()
    assert "ROBERT v" in outlines[0]
    results_line, start_reading = False, False
    for i, line in enumerate(outlines):
        if "------- Starting model with PFI filter " in line:
            start_reading = True
        if start_reading:
            if "Results of flawed models and sorted cross-validation:" in line:
                results_line = True
                if test_job == "clas":
                    assert (
                        "Original MCC (10x 5-fold CV) 0.67 - 15% & 30% threshold = 0.57 & 0.47"
                        in outlines[i + 1]
                    )
                    assert (
                        "o y_mean: PASSED, MCC = 0.0, lower than thresholds"
                        in outlines[i + 2]
                    )
                    assert (
                        "o y_shuffle: PASSED, MCC = -0.014, lower than thresholds"
                        in outlines[i + 3]
                    )
                    assert (
                        "o onehot: PASSED, MCC = 0.0, lower than thresholds"
                        in outlines[i + 4]
                    )
                    assert (
                        "- Sorted CV : Accuracy = [0.67, 0.83, 1.0, 0.67, 0.6], F1 score = [0.75, 0.86, 1.0, 0.67, 0.5], MCC = [0.45, 0.71, 1.0, 0.5, 0.41]"
                        in outlines[i + 5]
                    )
                elif test_job in ["standard", "standard_cmd"]:
                    assert_verify_standard_rmse_line(outlines[i + 1])
                    assert "o y_mean: PASSED, RMSE = 0.7" in outlines[i + 2]
                    assert "o y_shuffle: PASSED, RMSE = 1.0" in outlines[i + 3]
                    assert "x onehot: FAILED, RMSE = 0.3" in outlines[i + 4]
                    assert (
                        "- Sorted 5-fold CV : R2 = [0.0, 0.48, 0.24, 0.07, 0.17], MAE = [0.17, 0.25, 0.09, 0.37, 0.45], RMSE = [0.22, 0.27, 0.11, 0.45, 0.51]"
                        in outlines[i + 5]
                    )
                break
    assert results_line

    assert len(list(path_verify.glob("*.png"))) == 2
    assert len(list(path_verify.glob("*.dat"))) == 1
