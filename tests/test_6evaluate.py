#!/usr/bin/env python

######################################################.
# 	          Testing EVALUATE with pytest   	     #
######################################################.

import os
import sys
import glob
import shutil
import subprocess

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
