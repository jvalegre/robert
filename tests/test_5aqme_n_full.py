#!/usr/bin/env python

######################################################.
# 	          Testing AQME with pytest   	         #
######################################################.

import os
import glob
import pytest
import shutil
import subprocess
import pandas as pd

# saves the working directory
path_main = os.getcwd()
path_aqme = path_main + "/AQME"

# AQME and full workflow tests
@pytest.mark.parametrize(
    "test_job",
    [
        (
            "full_workflow"
        ),  # test for a full workflow
        (
            "aqme"
        ),  # test for a full workflow starting from AQME
    ],
)
def test_AQME(test_job):

    # reset the folders
    folders = ['CURATE','GENERATE','GENERATE_reg','GENERATE_clas','PREDICT','VERIFY','AQME']
    for folder in folders:
        if os.path.exists(f"{path_main}/{folder}"):
            shutil.rmtree(f"{path_main}/{folder}")

    # runs the program with the different tests
    if test_job == 'full_workflow':
        y_var = 'Target_values'
        csv_var = "tests/Robert_example.csv"
        ignore_var = "['Name']"

    elif test_job == 'aqme':
        y_var = 'solub'
        # for AQME-ROBERT workflows, the CSV file must be in the working dir
        shutil.copy(f"{path_main}/tests/solubility.csv", f"{path_main}/solubility.csv")
        csv_var = "solubility.csv"
        ignore_var = "['smiles','code_name']"

    cmd_robert = [
        "python",
        "-m",
        "robert",
        "--csv_name", csv_var,
        '--y', y_var,
        "--ignore", ignore_var,
        "--epochs", "5"
    ]

    if test_job == 'full_workflow':
        cmd_robert = cmd_robert + ["--discard", "['xtest']"]

    elif test_job == 'aqme':
        cmd_robert = cmd_robert + ["--aqme","--csearch_keywords", "--sample 2", 
                    "--qdescp_keywords", "--qdescp_atoms ['C']", "--model", "['RF']",
                    "--train", "[60]"]

    subprocess.run(cmd_robert)

    # check that all the plots, CSV and DAT files are created
    # find ROBERT_report.pdf
    assert os.path.exists(f'{path_main}/ROBERT_report.pdf')
    
    # CURATE folder
    assert len(glob.glob(f'{path_main}/CURATE/*.png')) == 1
    assert len(glob.glob(f'{path_main}/CURATE/*.dat')) == 1
    assert len(glob.glob(f'{path_main}/CURATE/*.csv')) == 2

    # GENERATE folder
    folders_gen = ['No_PFI','PFI']
    for folder in folders_gen:
        csv_amount = glob.glob(f'{path_main}/GENERATE/Raw_data/{folder}/*.csv')
        if test_job == 'aqme':
            assert len(csv_amount) == 2
        else:
            assert len(csv_amount) == 32
        best_amount = glob.glob(f'{path_main}/GENERATE/Best_model/{folder}/*.csv')
        assert len(best_amount) == 2

    # VERIFY folder
    assert len(glob.glob(f'{path_main}/VERIFY/*.png')) == 2
    assert len(glob.glob(f'{path_main}/VERIFY/*.dat')) == 3

    # PREDICT folder
    assert len(glob.glob(f'{path_main}/PREDICT/*.png')) == 8
    assert len(glob.glob(f'{path_main}/PREDICT/*.dat')) == 9
    assert len(glob.glob(f'{path_main}/PREDICT/*.csv')) == 4

    if test_job == 'aqme':
        assert os.path.exists(f'{path_main}/AQME-ROBERT_solubility.csv')
        db_aqme = pd.read_csv(f'{path_main}/AQME-ROBERT_solubility.csv')
        descps = ['code_name','solub','smiles','C_FUKUI+','C_DBSTEP_Vbur','MolLogP']
        for descp in descps:
            assert descp in db_aqme.columns

        outfile = open(f"{path_aqme}/AQME_data.dat", "r")
        outlines = outfile.readlines()
        outfile.close()
        assert "ROBERT v" in outlines[0]
        assert os.path.exists(f'{path_aqme}/CSEARCH')
        assert os.path.exists(f'{path_aqme}/QDESCP')
        assert len(glob.glob(f'{path_aqme}/*.csv')) == 2
        assert len(glob.glob(f'{path_aqme}/*.dat')) == 3

    # reset the folder
    folders = ['CURATE','GENERATE','PREDICT','VERIFY','AQME']
    for folder in folders:
        if os.path.exists(f"{path_main}/{folder}"):
            shutil.rmtree(f"{path_main}/{folder}")
    if os.path.exists(f'{path_main}/AQME-ROBERT_solubility.csv'):
        os.remove(f'{path_main}/AQME-ROBERT_solubility.csv')
    
