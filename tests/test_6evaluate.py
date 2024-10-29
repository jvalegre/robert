#!/usr/bin/env python

######################################################.
# 	          Testing EVALUATE with pytest   	     #
######################################################.

import os
import glob
import pytest
import shutil
import subprocess

# saves the working directory
path_main = os.getcwd()
path_tests = os.getcwd() + "/tests"

# EVALUATE tests
@pytest.mark.parametrize(
    "test_job",
    [
        (
            "standard"
        ),  # test for a train+valid. evaluation
        (
            "test"
        ),  # test for a train+valid.+test evaluation
        # (
        #     "clas"
        # ),  # test for classification models, currently EVALUATE only works for MVL, but this test will be implemented in the future
        (
            "missing_input"
        ),  # test that if the --csv_train or --csv_valid options are empty, a prompt pops up and asks for them 
    ],
)
def test_EVALUATE(test_job):

    # reset the folders (to avoid interferences with previous failed tests)
    folders = ['CURATE','GENERATE','GENERATE_reg','GENERATE_clas','PREDICT','VERIFY','AQME','EVALUATE']
    for folder in folders:
        if os.path.exists(f"{path_main}/{folder}"):
            shutil.rmtree(f"{path_main}/{folder}")
    for file in ['report_debug.txt','ROBERT_report.pdf','Evaluate_train.csv','Evaluate_valid.csv','Evaluate_test.csv']:
        if os.path.exists(f"{path_main}/{file}"):
            os.remove(f"{path_main}/{file}")

    # runs the program with the different tests
    cmd_robert = [
        "python",
        "-m",
        "robert",
        "--evaluate",
        '--y', 'Target_values',
        "--debug_report", "True",
        "--names","Name"
    ]

    if test_job in ['test']:
        cmd_robert = cmd_robert + ["--csv_test", "tests/Evaluate_test.csv"]

    if test_job != 'clas':
        cmd_robert = cmd_robert + ["--csv_train", "tests/Evaluate_train.csv"]
    else:
        cmd_robert = cmd_robert + ["--type", "clas"]
        cmd_robert = cmd_robert + ["--csv_train", "tests/Evaluate_clas_train.csv"]

    if test_job != 'missing_input':
        if test_job != 'clas':
            cmd_robert = cmd_robert + ["--csv_valid", "tests/Evaluate_valid.csv"]
        else:
            cmd_robert = cmd_robert + ["--csv_valid", "tests/Evaluate_clas_valid.csv"]
        subprocess.run(cmd_robert)
    
    else:
        cmd_missing = f'{" ".join(cmd_robert)} < {path_tests}/csv_valid.txt'
        os.system(cmd_missing)

    # check that all the plots, CSV and DAT files are created (only No PFI models)
    
    # CURATE folder
    assert len(glob.glob(f'{path_main}/CURATE/*.png')) == 1
    assert len(glob.glob(f'{path_main}/CURATE/*.dat')) == 1
    assert len(glob.glob(f'{path_main}/CURATE/*.csv')) == 2

    # GENERATE folder
    assert len(glob.glob(f'{path_main}/GENERATE/Best_model/No_PFI/*.csv')) == 2
    assert not os.path.exists(f'{path_main}/GENERATE/Best_model/PFI')

    # VERIFY folder
    assert len(glob.glob(f'{path_main}/VERIFY/*.png')) == 2
    assert len(glob.glob(f'{path_main}/VERIFY/*.dat')) == 1
    assert len(glob.glob(f'{path_main}/VERIFY/*.csv')) == 1

    # PREDICT folder
    if test_job in ['clas']:
        assert len(glob.glob(f'{path_main}/PREDICT/*.png')) == 6
    else:
        assert len(glob.glob(f'{path_main}/PREDICT/*.png')) == 7
    if test_job != 'test':
        assert len(glob.glob(f'{path_main}/PREDICT/*.csv')) == 2
    else:
        assert len(glob.glob(f'{path_main}/PREDICT/*.csv')) == 3
    assert len(glob.glob(f'{path_main}/PREDICT/*.dat')) == 1

    # find important parts in ROBERT_report
    assert os.path.exists(f'{path_main}/ROBERT_report.pdf')

    outfile = open(f"{path_main}/report_debug.txt", "r")
    outlines = outfile.readlines()
    outfile.close()

    find_heatmap,find_verify = 0,0
    find_shap,find_pfi,find_outliers = 0,0,0
    find_results_reg,find_results_valid_clas = 0,0
    find_results_test_clas,find_test = 0,0
    y_distrib_image,pearson_pred_image = False,False

    for line in outlines:
        if 'Heatmap_ML_models_No_PFI.png' in line:
            find_heatmap += 1
        if 'VERIFY_tests_MVL_59_No_PFI.png' in line:
            find_verify += 1
        if 'SHAP_MVL_59_No_PFI.png' in line:
            find_shap += 1
        if 'PFI_MVL_59_No_PFI.png' in line:
            find_pfi += 1
        if 'Outliers_MVL_59_No_PFI.png' in line:
            find_outliers += 1
        if 'Results_MVL_59_No_PFI.png' in line:
            find_results_reg += 1
        if 'Results_MVL_59_No_PFI_valid.png' in line:
            find_results_valid_clas += 1
        if 'Results_MVL_59_No_PFI_csv_test.png' in line:
            find_results_test_clas += 1
        if 'csv_test :' in line:
            find_test += 1
    
    # more specific tests to check content from the ROBERT score section
    robert_score,points_desc = [],[]
    ml_model_count,partition_count,metrics_count = 0,0,0
    flawed_models,pred_ability,cv_r2_models,cv_sd_models,points_descp = [],[],[],[],[]
    predict_graphs,flawed_image,cv_r2_image,cv_sd_image = False,False,False,False
    partition_count_test = 0

    for line in outlines:
        if 'robert/report/score_' in line and 'robert/report/score_w' not in line:
            robert_score.append(line.split('robert/report/score_')[1][0])
        if 'Model = MVL' in line:
            ml_model_count += 1
        if test_job != 'test':
            if 'Train:Validation = 59:41' in line:
                partition_count += 1
        else:
            if 'Train:Validation:Test = 47:32:21' in line:
                partition_count_test += 1
        if 'Points(train+valid.):descriptors = ' in line:
            points_desc.append(line.split('Points(train+valid.):descriptors = ')[1].split('</p>')[0])
        if 'Results_MVL_59_No_PFI.png' in line:
            predict_graphs = True
        if 'Train : R<sup>2</sup> = ' in line:
            metrics_count += 1
        if '1. Model vs "flawed" models' in line:
            flawed_models.append(line)
        if 'VERIFY/VERIFY_tests_MVL_59_No_PFI.png' in line:
            flawed_image = True
        if '2. Predictive ability of the model' in line:
            pred_ability.append(line)
        if '3a. CV predictions train + valid.' in line:
            cv_r2_models.append(line)
        if 'VERIFY/CV_train_valid_predict_MVL_59_No_PFI.png' in line:
            cv_r2_image = True
        if test_job == 'clas':
            if '3b. MCC difference (model vs CV)' in line:
                cv_sd_models.append(line)
        else:
            if '3b. Avg. standard deviation (SD)' in line:
                cv_sd_models.append(line)
        if 'PREDICT/CV_variability_MVL_59_No_PFI.png' in line:
            cv_sd_image = True
        if '4. Points(train+valid.):descriptors' in line:
            points_descp.append(line)
        if 'y_distribution_MVL_59_No_PFI.png' in line:
            y_distrib_image = True
        if 'Pearson_heatmap_No_PFI.png' in line:
            pearson_pred_image = True
        if 'How to predict new values with these models?' in line:
            break

    # model summary, robert score, predict graphs and model metrics
    assert len(robert_score) == 1
    if test_job == 'test':
        assert robert_score[0] == '2'
        assert partition_count == 0
    elif test_job == 'clas':
        assert robert_score[0] == '5'
        assert partition_count == 1
    else:
        assert robert_score[0] == '4'
        assert partition_count == 1
    assert ml_model_count == 1
    assert len(points_desc) == 1
    if test_job == 'clas':
        assert points_desc[0] == '37:7'
    else:
        assert points_desc[0] == '37:12'
    assert predict_graphs
    assert metrics_count == 1
    # advanced analysis, flawed models section 1
    assert '1 / 3' in flawed_models[0]
    assert 'robert/report/score_w_3_1.jpg' in flawed_models[0]
    assert flawed_image
    # advanced analysis, predictive ability section 2
    if test_job == 'test':
        assert '0 / 2' in pred_ability[0]
        assert 'robert/report/score_w_2_0.jpg' in pred_ability[0]
    elif test_job == 'clas':
        assert '1 / 2' in pred_ability[0]
        assert 'robert/report/score_w_2_1.jpg' in pred_ability[0]
    else:    
        assert '2 / 2' in pred_ability[0]
        assert 'robert/report/score_w_2_2.jpg' in pred_ability[0]
    # advanced analysis, CV predictive ability section 3a
    if test_job == 'clas':
        assert '0 / 2' in cv_r2_models[0]
        assert 'robert/report/score_w_2_0.jpg' in cv_r2_models[0]
    else:
        assert '1 / 2' in cv_r2_models[0]
        assert 'robert/report/score_w_2_1.jpg' in cv_r2_models[0]
    assert cv_r2_image
    # advanced analysis, CV variability section 3b 
    # for reg, CV R2 - for clas, MCC difference
    if test_job == 'clas':
        assert '2 / 2' in cv_sd_models[0]
        assert 'robert/report/score_w_2_2.jpg' in cv_sd_models[0]
        assert not cv_sd_image
    else:
        assert '0 / 2' in cv_sd_models[0]
        assert 'robert/report/score_w_2_0.jpg' in cv_sd_models[0]
        assert cv_sd_image
    # advanced analysis, predictive ability section 2
    if test_job == 'clas':
        assert '1 / 1' in points_descp[0]
        assert 'robert/report/score_w_1_1.jpg' in points_descp[0]
    else:
        assert '0 / 1' in points_descp[0]
        assert 'robert/report/score_w_1_0.jpg' in points_descp[0]

    if test_job not in ['clas']:
        assert find_outliers == 1
        assert find_results_reg == 1
        assert find_results_valid_clas == 0
    else:
        assert find_outliers == 0
        assert find_results_reg == 0
        assert find_results_valid_clas == 1

    if test_job in ['test']:
        assert find_test == 0 # the csv_test from EVALUATE should not count as the external test set from standard ROBERT workflows
        assert partition_count_test

    # common to all reports
    assert find_heatmap == 0 # EVALUATE skips GENERATE model selection
    assert find_verify == 1
    assert find_shap == 1
    assert find_pfi == 1
    # y distribution and Pearson images
    assert y_distrib_image
    assert pearson_pred_image

    # reset the folder
    folders = ['CURATE','GENERATE','GENERATE_reg','GENERATE_clas','PREDICT','VERIFY','AQME','EVALUATE']
    for folder in folders:
        if os.path.exists(f"{path_main}/{folder}"):
            shutil.rmtree(f"{path_main}/{folder}")
    for file_discard in ['report_debug.txt','ROBERT_report.pdf','Evaluate_train.csv','Evaluate_valid.csv','Evaluate_test.csv']:
        if os.path.exists(f'{path_main}/{file_discard}'):
            os.remove(f'{path_main}/{file_discard}')