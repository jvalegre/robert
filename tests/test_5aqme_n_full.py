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
            "full_workflow_test"
        ),  # test for a full workflow with test
        (
            "full_clas"
        ),  # test for a full workflow in classification
        (
            "full_clas_test"
        ),  # test for a full workflow in classification with test
        (
            "aqme"
        ),  # test for a full workflow starting from AQME
        (
            "2smiles_columns"
        ),  # test for a full workflow with 2 columns for SMILES
    ],
)
def test_AQME(test_job):

    # reset the folders (to avoid interferences with previous failed tests)
    folders = ['CURATE','GENERATE','GENERATE_reg','GENERATE_clas','PREDICT','VERIFY','AQME']
    for folder in folders:
        if os.path.exists(f"{path_main}/{folder}"):
            shutil.rmtree(f"{path_main}/{folder}")
    for file in ['report_debug.txt','ROBERT_report.pdf','AQME-ROBERT_solubility.csv','AQME-ROBERT_Robert_example_2smiles.csv','AQME-ROBERT_solubility_solvent.csv','Robert_example.csv','solubility.csv','solubility_solvent.csv']:
        if os.path.exists(f"{path_main}/{file}"):
            os.remove(f"{path_main}/{file}")

    # runs the program with the different tests
    if test_job in ['full_workflow','full_workflow_test']:
        y_var = 'Target_values'
        csv_var = "tests/Robert_example.csv"

    elif test_job in ['full_clas','full_clas_test']:
        y_var = 'Target_values'
        csv_var = "tests/Robert_example_clas.csv"

    elif test_job == 'aqme':
        y_var = 'solub'
        # for AQME-ROBERT workflows, the CSV file must be in the working dir
        shutil.copy(f"{path_main}/tests/solubility.csv", f"{path_main}/solubility.csv")
        csv_var = "solubility.csv"

    elif test_job == '2smiles_columns':
        y_var = 'solub'
        # for AQME-ROBERT workflows, the CSV file must be in the working dir
        shutil.copy(f"{path_main}/tests/solubility_solvent.csv", f"{path_main}/solubility_solvent.csv")
        csv_var = "solubility_solvent.csv"

    cmd_robert = [
        "python",
        "-m",
        "robert",
        "--csv_name", csv_var,
        '--y', y_var,
        "--epochs", "5",
        "--seed", "[0]",
        "--model", "['RF']",
        "--train", "[60]",
        "--pfi_epochs", "1",
        "--debug_report", "True"
    ]

    if test_job in ['full_workflow','full_workflow_test','full_clas','full_clas_test']:
        cmd_robert = cmd_robert + ["--ignore", "[Name]", "--names","Name"]
    
    if test_job in ['full_workflow','full_workflow_test']:
        cmd_robert = cmd_robert + ["--discard", "['xtest']"]
    
    if test_job == 'full_workflow_test':
        cmd_robert = cmd_robert + ["--csv_test", "tests/Robert_example_test.csv"]
    
    if test_job == 'full_clas_test':
        cmd_robert = cmd_robert + ["--csv_test", "tests/Robert_example_clas_test.csv"]

    if test_job in ['full_clas','full_clas_test']:
        cmd_robert = cmd_robert + ["--type", "clas"]

    if test_job == 'aqme':
        cmd_robert = cmd_robert + ["--aqme","--csearch_keywords", "--sample 1", 
                    "--qdescp_keywords", "--qdescp_atoms ['C'] --qdescp_acc 5 --qdescp_opt normal",
                    "--alpha", "0.5"]

    if test_job == '2smiles_columns':
        cmd_robert = cmd_robert  + ["--aqme","--csearch_keywords", "--sample 1", "--alpha", "0.5"]

    subprocess.run(cmd_robert)

    # check that all the plots, CSV and DAT files are created
    # find ROBERT_report.pdf
    assert os.path.exists(f'{path_main}/ROBERT_report.pdf')
    
    # CURATE folder
    if test_job != 'aqme': # in AQME, there are too many descriptors so the Pearson heatmap doesn't show
        assert len(glob.glob(f'{path_main}/CURATE/*.png')) == 1
    assert len(glob.glob(f'{path_main}/CURATE/*.dat')) == 1
    assert len(glob.glob(f'{path_main}/CURATE/*.csv')) == 2

    # GENERATE folder
    folders_gen = ['No_PFI','PFI']
    for folder in folders_gen:
        csv_amount = glob.glob(f'{path_main}/GENERATE/Raw_data/{folder}/*.csv')
        assert len(csv_amount) == 2
        best_amount = glob.glob(f'{path_main}/GENERATE/Best_model/{folder}/*.csv')
        assert len(best_amount) == 2

    # VERIFY folder
    assert len(glob.glob(f'{path_main}/VERIFY/*.png')) == 4
    assert len(glob.glob(f'{path_main}/VERIFY/*.dat')) == 3
    assert len(glob.glob(f'{path_main}/VERIFY/*.csv')) == 2

    # PREDICT folder
    if test_job in ['full_clas','full_clas_test']:
        assert len(glob.glob(f'{path_main}/PREDICT/*.dat')) == 7 # missing the 4 dat files from outliers and CV variability
        assert len(glob.glob(f'{path_main}/PREDICT/*.png')) == 12
    else:
        assert len(glob.glob(f'{path_main}/PREDICT/*.dat')) == 11
        assert len(glob.glob(f'{path_main}/PREDICT/*.png')) == 14

    if test_job == 'full_clas_test':
        assert len(glob.glob(f'{path_main}/PREDICT/csv_test/*.csv')) == 2 # 2 extra CSV files for the test set
        assert len(glob.glob(f'{path_main}/PREDICT/csv_test/*.png')) == 2 # 2 extra PNG for test confusion matrices
    elif test_job == 'full_workflow_test':
        assert len(glob.glob(f'{path_main}/PREDICT/csv_test/*.csv')) == 2 # 2 extra CSV files for the test set
        assert len(glob.glob(f'{path_main}/PREDICT/csv_test/*.png')) == 2 # 2 extra PNG for predictions ± SD
    else:
        assert len(glob.glob(f'{path_main}/PREDICT/*.csv')) == 4

    if test_job == 'aqme':
        assert os.path.exists(f'{path_main}/AQME-ROBERT_solubility.csv')
        db_aqme = pd.read_csv(f'{path_main}/AQME-ROBERT_solubility.csv')
        descps = ['code_name','solub','C_FUKUI+','MolLogP']
        for descp in descps:
            assert descp in db_aqme.columns
        assert 'smiles' in db_aqme.columns
        assert 'C_DBSTEP_Vbur' not in db_aqme.columns

        outfile = open(f"{path_aqme}/AQME_data.dat", "r")
        outlines = outfile.readlines()
        outfile.close()
        assert "ROBERT v" in outlines[0]
        assert os.path.exists(f'{path_aqme}/CSEARCH')
        assert os.path.exists(f'{path_aqme}/QDESCP')
        assert len(glob.glob(f'{path_aqme}/*.csv')) == 1
        assert len(glob.glob(f'{path_aqme}/*.dat')) == 3

    if test_job == '2smiles_columns':
        assert os.path.exists(f'{path_main}/AQME-ROBERT_solubility_solvent.csv')
        db_aqme = pd.read_csv(f'{path_main}/AQME-ROBERT_solubility_solvent.csv')
        assert 'smiles_sub' and 'smiles_solvent' in db_aqme.columns

    # find important parts in ROBERT_report
    outfile = open(f"{path_main}/report_debug.txt", "r")
    outlines = outfile.readlines()
    outfile.close()

    find_pearson,find_heatmap,find_verify = 0,0,0
    find_shap,find_pfi,find_outliers = 0,0,0
    find_results_reg,find_results_train_clas,find_results_valid_clas = 0,0,0
    find_results_test_clas,find_test = 0,0

    for line in outlines:
        if 'Pearson_heatmap.png' in line:
            find_pearson += 1
        if 'Heatmap ML models no PFI filter.png' in line:
            find_heatmap += 1
        if 'VERIFY_tests_RF_60_PFI.png' in line:
            find_verify += 1
        if 'SHAP_RF_60_PFI.png' in line:
            find_shap += 1
        if 'PFI_RF_60_PFI.png' in line:
            find_pfi += 1
        if 'Outliers_RF_60_No_PFI.png' in line:
            find_outliers += 1
        if 'Results_RF_60_No_PFI.png' in line:
            find_results_reg += 1
        if 'Results_RF_60_No_PFI_train.png' in line:
            find_results_train_clas += 1
        if 'Results_RF_60_No_PFI_valid.png' in line:
            find_results_valid_clas += 1
        if 'Results_RF_60_No_PFI_csv_test.png' in line:
            find_results_test_clas += 1
        if 'csv_test :' in line:
            find_test += 1
    
    # more specific tests to check content from the ROBERT score section
    if test_job == 'full_workflow':
        robert_score,points_desc = [],[]
        ml_model_count,partition_count,metrics_count = 0,0,0
        flawed_models,pred_ability,cv_r2_models,cv_sd_models,points_descp = [],[],[],[],[]
        predict_graphs,flawed_image,cv_r2_image,cv_sd_image = False,False,False,False

        for line in outlines:
            if 'robert/report/score_' in line and 'robert/report/score_w' not in line:
                robert_score.append(line.split('robert/report/score_')[1][0])
            if 'Model = RF' in line:
                ml_model_count += 1
            if 'Train:Validation = 59:41' in line:
                partition_count += 1
            if 'Points(train+valid.):descriptors = ' in line:
                points_desc.append(line.split('Points(train+valid.):descriptors = ')[1].split('</p>')[0])
            if 'Results_RF_60_No_PFI.png' in line:
                predict_graphs = True
            if 'Train : R<sup>2</sup> = ' in line:
                metrics_count += 1
            if '1. Model vs "flawed" models' in line:
                flawed_models.append(line)
            if 'VERIFY/VERIFY_tests_RF_60_No_PFI.png' in line:
                flawed_image = True
            if '2. Predictive ability of the model' in line:
                pred_ability.append(line)
            if '3a. CV predictions train + valid.' in line:
                cv_r2_models.append(line)
            if 'VERIFY/CV_train_valid_predict_RF_60_No_PFI.png' in line:
                cv_r2_image = True
            if '3b. Avg. standard deviation (SD)' in line:
                cv_sd_models.append(line)
            if 'PREDICT/CV_variability_RF_60_No_PFI.png' in line:
                cv_sd_image = True
            if '4. Points(train+valid.):descriptors' in line:
                points_descp.append(line)
            if 'How to predict new values with these models?' in line:
                break

        # model summary, robert score, predict graphs and model metrics
        assert robert_score[0] == '3'
        assert robert_score[1] == '8'
        assert ml_model_count == 2
        assert partition_count == 2
        assert points_desc[0] == '37:12'
        assert points_desc[1] == '37:3'
        assert predict_graphs
        assert metrics_count == 2
        # advanced analysis, flawed models section 1
        assert '1 / 3' in flawed_models[0]
        assert 'robert/report/score_w_3_1.jpg' in flawed_models[0]
        assert '3 / 3' in flawed_models[1]
        assert 'robert/report/score_w_3_3.jpg' in flawed_models[1]
        assert flawed_image
        # advanced analysis, predictive ability section 2
        assert '1 / 2' in pred_ability[0]
        assert 'robert/report/score_w_2_1.jpg' in pred_ability[0]
        assert '2 / 2' in pred_ability[1]
        assert 'robert/report/score_w_2_2.jpg' in pred_ability[1]
        # advanced analysis, CV predictive ability section 3a
        assert '1 / 2' in cv_r2_models[0]
        assert 'robert/report/score_w_2_1.jpg' in cv_r2_models[0]
        assert '2 / 2' in cv_r2_models[1]
        assert 'robert/report/score_w_2_2.jpg' in cv_r2_models[1]
        assert cv_r2_image
        # advanced analysis, CV variability section 3b
        assert '0 / 2' in cv_sd_models[0]
        assert 'robert/report/score_w_2_0.jpg' in cv_sd_models[0]
        assert '0 / 2' in cv_sd_models[1]
        assert 'robert/report/score_w_2_0.jpg' in cv_sd_models[1]
        assert cv_sd_image
        # advanced analysis, predictive ability section 2
        assert '0 / 1' in points_descp[0]
        assert 'robert/report/score_w_1_0.jpg' in points_descp[0]
        assert '1 / 1' in points_descp[1]
        assert 'robert/report/score_w_1_1.jpg' in points_descp[1]

    if test_job in ['full_workflow','full_workflow_test','aqme','2smiles_columns']:
        assert find_outliers > 0
        assert find_results_reg > 0
        assert find_results_train_clas == 0
        assert find_results_valid_clas == 0
    else:
        assert find_outliers == 0
        assert find_results_reg == 0
        assert find_results_train_clas > 0
        assert find_results_valid_clas > 0

    if test_job in ['full_workflow_test','full_clas_test']:
        assert find_test > 0
        if test_job == 'full_clas_test':
            assert find_results_test_clas > 0
        else:
            assert find_results_test_clas == 0
    else:
        assert find_test == 0

    # common to all reports
    if test_job != 'aqme': # too many descriptors so no Pearon heatmap
        assert find_pearson > 0
    assert find_heatmap > 0
    assert find_verify > 0
    assert find_shap > 0
    assert find_pfi > 0

    # reset the folder
    folders = ['CURATE','GENERATE','PREDICT','VERIFY','AQME']
    for folder in folders:
        if os.path.exists(f"{path_main}/{folder}"):
            shutil.rmtree(f"{path_main}/{folder}")
    for file_discard in ['report_debug.txt','ROBERT_report.pdf','AQME-ROBERT_solubility.csv','AQME-ROBERT_Robert_example_2smiles.csv','AQME-ROBERT_solubility_solvent.csv','Robert_example.csv','solubility.csv','solubility_solvent.csv']:
        if os.path.exists(f'{path_main}/{file_discard}'):
            os.remove(f'{path_main}/{file_discard}')