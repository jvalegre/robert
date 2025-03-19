#!/usr/bin/env python

######################################################.
# 	         Testing PREDICT with pytest 	         #
######################################################.

import os
import glob
import pytest
import shutil
import subprocess
from pathlib import Path

# saves the working directory
path_main = os.getcwd()
path_predict = path_main + "/PREDICT"

# PREDICT tests
@pytest.mark.parametrize(
    "test_job",
    [
        (
            "t_value"
        ),  # test for the t-value used
        (
            "clas"
        ),  # test for clasification        
        (
            "csv_test"
        ),  # test for external test set
        (
            "standard"
        ),  # standard test
    ],
)
def test_PREDICT(test_job):

    # leave the folders as they were initially to run a different batch of tests
    if os.path.exists(f"{path_predict}"):
        shutil.rmtree(f"{path_predict}")
        # remove DAT and CSV files generated by PREDICT
        dat_files = glob.glob("*.dat")
        for dat_file in dat_files:
            if "PREDICT" in dat_file:
                os.remove(dat_file)

    if test_job == 'clas': # rename folders to use in classification
        # rename the regression GENERATE folder
        filepath_reg = Path(f"{path_main}/GENERATE")
        filepath_reg.rename(f"{path_main}/GENERATE_reg")
        # rename the classification GENERATE folder
        filepath = Path(f"{path_main}/GENERATE_clas")
        filepath.rename(f"{path_main}/GENERATE")

    else: # in case the clas test fails and the ending rename doesn't happen
        if os.path.exists(f"{path_main}/GENERATE_reg"):
            # rename the classification GENERATE folder
            filepath = Path(f"{path_main}/GENERATE")
            filepath.rename(f"{path_main}/GENERATE_clas")
            # rename the regression GENERATE folder
            filepath_reg = Path(f"{path_main}/GENERATE_reg")
            filepath_reg.rename(f"{path_main}/GENERATE")

    # runs the program with the different tests
    cmd_robert = [
        "python",
        "-m",
        "robert",
        "--predict",
    ]

    if test_job == "t_value":
        cmd_robert = cmd_robert + ["--t_value", "4"]
    if test_job == "csv_test":
        cmd_robert = cmd_robert + ["--csv_test", "tests/Robert_example_test.csv"]

    subprocess.run(cmd_robert)

    # check that the DAT file is created
    assert not os.path.exists(f"{path_main}/PREDICT_data.dat")
    outfile = open(f"{path_predict}/PREDICT_data.dat", "r")
    outlines = outfile.readlines()
    outfile.close()
    assert "ROBERT v" in outlines[0]
    categor_test = False
    y_distrib_found, pearson_found = False, False
    outliers_found, results_found, proportion_found = False, False, False
    for i,line in enumerate(outlines):
        if ' - Training points:' in line:
            proportion_found = True
            assert '- Training points: 30' in line
            assert '- Test points: 7' in outlines[i+1]
        if 'o  Saving CSV databases with predictions and their SD in:' in line:
            results_found = True
            # Check for any model type with No_PFI.csv pattern
            assert 'Predicted results of starting dataset: PREDICT/' in outlines[i+1]
            assert '_No_PFI.csv' in outlines[i+1]
        elif 'x  There are missing descriptors in the test set! Looking for categorical variables converted from CURATE' in line:
            categor_test = True
        elif 'Outliers plot saved in' in line and 'No_PFI' in line:
            outliers_found = True
            if test_job != "clas":
                train_outliers = int(outlines[i+1].split()[1])
                if test_job == "t_value":
                    assert train_outliers == 0
                else:
                    assert train_outliers > 0
                    assert '-  2 (' in outlines[i+2]
                assert 'Test: 0 outliers' in outlines[i+2+train_outliers]
        
        if test_job == "clas":
            if 'x  WARNING! High correlations observed (up to r = 1.0 or R2 = 1.0, for x1 and x3)' in line:
                pearson_found = True
        else:
            if 'x  WARNING! Noticeable correlations observed (up to r = -0.81 or R2 = 0.66, for x7 and x10)' in line:
                pearson_found = True
        if test_job == "clas":
            if 'o  Your data seems quite uniform' in line:
                y_distrib_found = True
        else:
            if 'x  WARNING! Your data is slightly not uniform (Q4 has 4 points while Q1 has 12)' in line:
                y_distrib_found = True
        
        # continues to the PFI section
        if '-------' in line and 'with PFI filter' in line:
            break

    if test_job == "csv_test":
        assert categor_test

    assert proportion_found
    assert results_found
    # Skip asserting outliers_found if test_job == "clas"
    if test_job != "clas":
        assert outliers_found
    assert y_distrib_found
    assert pearson_found

    # check that all the plots, CSV and DAT files are created
    if test_job == "clas":
        assert len(glob.glob(f'{path_predict}/*.png')) == 12
    else:
        assert len(glob.glob(f'{path_predict}/*.png')) == 14
    if test_job == "csv_test":
        assert len(glob.glob(f'{path_predict}/csv_test/*.png')) == 2
        assert len(glob.glob(f'{path_predict}/csv_test/*.csv')) == 2

    assert len(glob.glob(f'{path_predict}/*.dat')) == 1        
    assert len(glob.glob(f'{path_predict}/*.csv')) == 2

    if test_job == 'clas': # rename folders back to their original names
        # rename the classification GENERATE folder
        filepath = Path(f"{path_main}/GENERATE")
        filepath.rename(f"{path_main}/GENERATE_clas")
        # rename the regression GENERATE folder
        filepath_reg = Path(f"{path_main}/GENERATE_reg")
        filepath_reg.rename(f"{path_main}/GENERATE")