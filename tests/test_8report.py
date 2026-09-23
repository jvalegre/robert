#!/usr/bin/env python

######################################################.
# 	         Testing REPORT with pytest 	         #
######################################################.

import os
import glob
import pytest
import shutil
from robert.curate import curate
from robert.generate import generate
from robert.verify import verify
from robert.predict import predict
from robert.report import report

# saves the working directory
path_main = os.getcwd()


def _reset_pipeline_folders():
    # leave the folders as they were initially to run a different batch of tests
    for folder in ["CURATE", "GENERATE", "VERIFY", "PREDICT"]:
        folder_path = os.path.join(path_main, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    for pattern in ["*_data.dat", "report_debug*.txt", "ROBERT_report*.pdf", "report.css"]:
        for f in glob.glob(os.path.join(path_main, pattern)):
            os.remove(f)


# REPORT tests: builds a minimal CURATE->GENERATE->VERIFY->PREDICT pipeline and calls
# report() directly (in-process), unlike the full-workflow tests in test_5aqme_n_full.py
# which drive ROBERT through subprocess.run() - a subprocess isn't visible to coverage
# tools, so report.py/report_utils.py were otherwise never measured
@pytest.mark.parametrize(
    "test_job",
    [
        ("reg"),  # regression: Interpolation + Boundary robustness columns
        ("clas"),  # classification: Interpolation only, no Boundary robustness column
    ],
)
def test_REPORT(test_job):
    _reset_pipeline_folders()

    if test_job == "clas":
        curate_kwargs = {
            "y": "Target_values",
            "csv_name": os.path.join("tests", "Robert_example_clas.csv"),
            "names": "Name",
            "type": "clas",
        }
        curate_csv = os.path.join("CURATE", "Robert_example_clas_CURATE.csv")
    else:
        curate_kwargs = {
            "y": "Target_values",
            "csv_name": os.path.join("tests", "Robert_example.csv"),
            "names": "Name",
            "discard": ["xtest"],
        }
        curate_csv = os.path.join("CURATE", "Robert_example_CURATE.csv")

    curate(**curate_kwargs)

    generate_kwargs = {
        "csv_name": curate_csv,
        "y": "Target_values",
        "model": ["RF"],
        "init_points": 1,
        "n_iter": 1,
    }
    if test_job == "clas":
        generate_kwargs["type"] = "clas"

    generate(**generate_kwargs)
    verify()
    predict()
    report(debug_report=True)

    # both PDFs (No PFI / PFI) are always generated, regardless of type
    assert os.path.exists(os.path.join(path_main, "ROBERT_report_No_PFI.pdf"))
    assert os.path.exists(os.path.join(path_main, "ROBERT_report_PFI.pdf"))

    outfile = open(os.path.join(path_main, "report_debug_No_PFI.txt"), "r", encoding="utf-8")
    outlines = outfile.readlines()
    outfile.close()
    full_text = "".join(outlines)

    assert "Section A. ROBERT Score" in full_text

    # score images ("report/score_N.jpg") are the Interpolation/Boundary robustness score
    # bars in Section A - classification only ever gets one (Interpolation), since Boundary
    # robustness isn't defined for it (see docs/Report/score.rst); the right-hand column
    # shows a short "disabled" note instead (see print_score() in report.py) rather than
    # being left empty or stretching Interpolation to full width
    score_imgs = [line for line in outlines if "report/score_" in line and "report/score_w" not in line]
    if test_job == "clas":
        assert len(score_imgs) == 1
        assert "Boundary robustness" in full_text
        assert "Disabled in classification problems" in full_text
        assert "1. Consistency (sorted CV)" not in full_text
        assert (
            "Interpolation measures how reliably the model predicts within the range "
            "of data it was trained on.</i>" in full_text
        )
    else:
        assert len(score_imgs) == 2
        assert "Boundary robustness" in full_text
        assert "1. Sorted CV, top 20% (High)" in full_text

    _reset_pipeline_folders()
