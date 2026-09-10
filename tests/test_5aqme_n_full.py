#!/usr/bin/env python

######################################################.
# 	          Testing AQME with pytest   	         #
######################################################.

import os
import sys
import glob
import pytest
import shutil
import subprocess
import pandas as pd

# saves the working directory
path_main = os.getcwd()
path_aqme = os.path.join(path_main, "AQME")


# AQME and full workflow tests
@pytest.mark.parametrize(
    "test_job",
    [
        ("full_workflow"),  # test for a full workflow
        ("full_workflow_test"),  # test for a full workflow with test
        ("full_clas"),  # test for a full workflow in classification
        ("full_clas_test"),  # test for a full workflow in classification with test
        ("aqme"),  # test for a full workflow starting from AQME
        ("2smiles_columns"),  # test for a full workflow with 2 columns for SMILES
    ],
)
def test_AQME(test_job):
    # reset the folders (to avoid interferences with previous failed tests)
    folders = [
        "CURATE",
        "GENERATE",
        "GENERATE_reg",
        "GENERATE_clas",
        "PREDICT",
        "VERIFY",
        "AQME",
    ]
    for folder in folders:
        if os.path.exists(f"{path_main}/{folder}"):
            shutil.rmtree(f"{path_main}/{folder}")
    for file in [
        "report_debug_No_PFI.txt",
        "report_debug_PFI.txt",
        "ROBERT_report_No_PFI.pdf",
        "ROBERT_report_PFI.pdf",
        "AQME-ROBERT_solubility.csv",
        "AQME-ROBERT_Robert_example_2smiles.csv",
        "AQME-ROBERT_solubility_solvent.csv",
        "Robert_example.csv",
        "solubility.csv",
        "solubility_solvent.csv",
    ]:
        if os.path.exists(f"{path_main}/{file}"):
            os.remove(f"{path_main}/{file}")

    # runs the program with the different tests
    if test_job in ["full_workflow", "full_workflow_test"]:
        y_var = "Target_values"
        csv_var = "tests/Robert_example.csv"

    elif test_job in ["full_clas", "full_clas_test"]:
        y_var = "Target_values"
        csv_var = "tests/Robert_example_clas.csv"

    elif test_job == "aqme":
        y_var = "solub"
        # for AQME-ROBERT workflows, the CSV file must be in the working dir
        shutil.copy(f"{path_main}/tests/solubility.csv", f"{path_main}/solubility.csv")
        csv_var = "solubility.csv"

    elif test_job == "2smiles_columns":
        y_var = "solub"
        # for AQME-ROBERT workflows, the CSV file must be in the working dir
        shutil.copy(
            f"{path_main}/tests/solubility_solvent.csv",
            f"{path_main}/solubility_solvent.csv",
        )
        csv_var = "solubility_solvent.csv"

    cmd_robert = [
        sys.executable,
        "-m",
        "robert",
        "--csv_name",
        csv_var,
        "--y",
        y_var,
        "--init_points",
        "1",
        "--n_iter",
        "1",
        "--model",
        "['RF']",
        "--pfi_epochs",
        "1",
        "--debug_report",
        "True",
    ]

    if test_job in [
        "full_workflow",
        "full_workflow_test",
        "full_clas",
        "full_clas_test",
    ]:
        cmd_robert = cmd_robert + ["--names", "Name"]

    if test_job in ["full_workflow", "full_workflow_test"]:
        cmd_robert = cmd_robert + ["--discard", "['xtest']"]

    if test_job == "full_workflow_test":
        cmd_robert = cmd_robert + ["--csv_test", "tests/Robert_example_test.csv"]

    if test_job == "full_clas_test":
        cmd_robert = cmd_robert + ["--csv_test", "tests/Robert_example_clas_test.csv"]

    if test_job in ["full_clas", "full_clas_test"]:
        cmd_robert = cmd_robert + ["--type", "clas"]

    if test_job == "aqme":
        cmd_robert = cmd_robert + [
            "--aqme",
            "--qdescp_keywords",
            "--qdescp_atoms ['C'] --qdescp_acc 5 --qdescp_opt normal",
            "--alpha",
            "0.5",
        ]

    if test_job == "2smiles_columns":
        cmd_robert = cmd_robert + ["--aqme", "--alpha", "0.5"]

    subprocess.run(cmd_robert)

    # check that all the plots, CSV and DAT files are created
    # find ROBERT_report_No_PFI.pdf and ROBERT_report_PFI.pdf
    assert os.path.exists(f"{path_main}/ROBERT_report_No_PFI.pdf")
    assert os.path.exists(f"{path_main}/ROBERT_report_PFI.pdf")

    # CURATE folder
    if (
        test_job != "aqme"
    ):  # in AQME, there are too many descriptors so the Pearson heatmap doesn't show
        assert len(glob.glob(f"{path_main}/CURATE/*.png")) == 1
    assert len(glob.glob(f"{path_main}/CURATE/*.dat")) == 1
    assert len(glob.glob(f"{path_main}/CURATE/*.csv")) == 3

    # GENERATE folder
    folders_gen = ["No_PFI", "PFI"]
    for folder in folders_gen:
        csv_amount = glob.glob(f"{path_main}/GENERATE/Raw_data/{folder}/*.csv")
        assert len(csv_amount) == 2
        best_amount = glob.glob(f"{path_main}/GENERATE/Best_model/{folder}/*.csv")
        assert len(best_amount) == 2

    # VERIFY folder
    assert len(glob.glob(f"{path_main}/VERIFY/*.png")) == 2
    assert len(glob.glob(f"{path_main}/VERIFY/*.dat")) == 1

    # PREDICT folder
    if test_job in ["full_clas", "full_clas_test"]:
        assert len(glob.glob(f"{path_main}/PREDICT/*.png")) == 12
    else:
        assert len(glob.glob(f"{path_main}/PREDICT/*.png")) == 24
    assert len(glob.glob(f"{path_main}/PREDICT/*.dat")) == 1

    if test_job == "full_clas_test":
        assert (
            len(glob.glob(f"{path_main}/PREDICT/csv_test/*.csv")) == 2
        )  # 2 extra CSV files for the test set
        assert (
            len(glob.glob(f"{path_main}/PREDICT/csv_test/*.png")) == 2
        )  # 2 extra PNG for test confusion matrices
    elif test_job == "full_workflow_test":
        assert (
            len(glob.glob(f"{path_main}/PREDICT/csv_test/*.csv")) == 2
        )  # 2 extra CSV files for the test set
        assert (
            len(glob.glob(f"{path_main}/PREDICT/csv_test/*.png")) == 2
        )  # 2 extra PNG for predictions ± SD
    else:
        assert len(glob.glob(f"{path_main}/PREDICT/*.csv")) == 2

    if test_job == "aqme":
        assert os.path.exists(f"{path_main}/AQME-ROBERT_interpret_solubility.csv")
        db_aqme = pd.read_csv(f"{path_main}/AQME-ROBERT_interpret_solubility.csv")
        descps = ["code_name", "solub", "HOMO", "C_Partial charge", "C_Buried volume"]
        for descp in descps:
            assert descp in db_aqme.columns
        assert "smiles" in db_aqme.columns
        assert "C_DBSTEP_Vbur" not in db_aqme.columns

        outfile = open(f"{path_aqme}/AQME_data.dat", "r")
        outlines = outfile.readlines()
        outfile.close()
        assert "ROBERT v" in outlines[0]
        assert os.path.exists(f"{path_aqme}/CSEARCH")
        assert os.path.exists(f"{path_aqme}/QDESCP")
        assert len(glob.glob(f"{path_aqme}/*.csv")) == 0
        assert len(glob.glob(f"{path_aqme}/*.dat")) == 3

    if test_job == "2smiles_columns":
        assert os.path.exists(
            f"{path_main}/AQME-ROBERT_interpret_solubility_solvent.csv"
        )
        db_aqme = pd.read_csv(
            f"{path_main}/AQME-ROBERT_interpret_solubility_solvent.csv"
        )
        descps_2smiles = [
            "code_name",
            "smiles_sub",
            "smiles_solvent",
            "solub",
            "HOMO_sub",
            "C_Partial charge_sub",
            "HOMO_solvent",
        ]
        for descp in descps_2smiles:
            assert descp in db_aqme.columns

    # find important parts in ROBERT_report
    outfile = open(f"{path_main}/report_debug_No_PFI.txt", "r")
    outlines = outfile.readlines()
    outfile.close()

    find_heatmap, find_verify = 0, 0
    find_shap, find_pfi, find_outliers = 0, 0, 0
    find_results_reg, find_results_test_clas = 0, 0
    find_results_test_clas, find_test, find_results_external = 0, 0, 0

    for line in outlines:
        if "Heatmap_ML_models_No_PFI.png" in line:
            find_heatmap += 1
        if "VERIFY_tests_RF_No_PFI.png" in line:
            find_verify += 1
        if "SHAP_RF_No_PFI.png" in line:
            find_shap += 1
        if "PFI_RF_No_PFI.png" in line:
            find_pfi += 1
        if "Outliers_RF_No_PFI.png" in line:
            find_outliers += 1
        if "Results_RF_No_PFI.png" in line:
            find_results_reg += 1
        if "Results_RF_No_PFI_test.png" in line:
            find_results_test_clas += 1
        if "Results_RF_No_PFI_external.png" in line:  # name in clas
            find_results_external += 1
        elif "CV_variability_RF_No_PFI_external.png" in line:  # name in reg
            find_results_external += 1
        if "External test metrics" in line:
            find_test += 1

    # more specific tests to check content from the ROBERT score section
    if test_job in ["full_workflow", "full_clas"]:
        robert_score, points_desc = [], []
        ml_model_count, partition_count, metrics_train_count, metrics_test_count = (
            0,
            0,
            0,
            0,
        )
        (
            flawed_models,
            pred_ability,
            pred_test_ability,
            cv_sd_models,
            cv_vs_test_models,
            bound_ability,
            bound_ability_clas,
        ) = [], [], [], [], [], [], []
        predict_graphs, flawed_image, cv_sd_image = False, False, False
        y_distrib_image, pearson_pred_image = False, False
        find_severe_red = False
        find_moder_correl, find_moder_y_dist = False, False
        find_assess_red = False

        for i, line in enumerate(outlines):
            if "report/score_" in line and "report/score_w" not in line:
                robert_score.append(line.split("report/score_")[1][0])
            if "Model = RF" in line:
                ml_model_count += 1
            if "CV (train+valid.)" in line and "81:19" in line:
                partition_count += 1
            if "Points(train+validation):descriptors = " in line:
                points_desc.append(
                    line.split("Points(train+validation):descriptors = ")[1].split(
                        "</p>"
                    )[0]
                )
            if "Results_RF_No_PFI.png" in line:
                predict_graphs = True
            if "10x 5-fold CV : R<sup>2</sup> = " in line:
                metrics_train_count += 1
            if "Test : R<sup>2</sup> = " in line:
                metrics_test_count += 1
            if '1. Model vs "flawed" models' in line:
                flawed_models.append(line)
            if (
                "VERIFY/VERIFY_tests_RF_No_PFI.png" in line
                or "VERIFY\\VERIFY_tests_RF_No_PFI.png" in line
            ):
                flawed_image = True
            if "2. CV predictions of the model" in line:
                pred_ability.append(line)
            if "3. Test set predictions" in line:
                pred_test_ability.append(line)
            if "5. CV vs test consistency" in line:
                cv_vs_test_models.append(line)
            if "6. Prediction stability" in line:
                cv_sd_models.append(line)
            if (
                "PREDICT/CV_variability_RF_No_PFI.png" in line
                or "PREDICT\\CV_variability_RF_No_PFI.png" in line
            ):
                cv_sd_image = True
            if "1. Sorted CV, top 20% (High)" in line:
                bound_ability.append(line)
            elif "1. Consistency (sorted CV)" in line:
                bound_ability_clas.append(line)
            if "y_distribution_RF_No_PFI.png" in line:
                y_distrib_image = True
            if "Pearson_heatmap_RF_No_PFI.png" in line:
                pearson_pred_image = True
            if (
                "Failing required tests (Section B.1)" in line
                and "color: #c56666" in outlines[i - 1]
            ):
                find_severe_red = True
            if (
                "Moderately correlated features (Section D)" in line
                and "color: #c5c57d" in outlines[i - 1]
            ):
                find_moder_correl = True
            if (
                "Slightly uneven y distribution (Section C)" in line
                and "color: #c5c57d" in outlines[i - 1]
            ):
                find_moder_y_dist = True
            if (
                "The model is unreliable" in line
                and "color: #c56666" in outlines[i - 1]
            ):
                find_assess_red = True
            if "How to predict new values with these models?" in line:
                break

        if test_job == "full_workflow":
            # model summary, robert score, predict graphs and model metrics
            # NOTE: report_debug_{suffix}.txt now holds a SINGLE model/suffix's report (one
            # PDF per suffix - see report.py's "generate one PDF per model (No PFI / PFI)"
            # loop), not a combined No_PFI+PFI report like before 2.2.0 - so every field below
            # that used to be checked as a [0]/[1] pair (one per model) now appears exactly
            # once. The only field that legitimately has 2 entries is robert_score, since each
            # single-suffix report still shows 2 columns (Interpolation/Boundary robustness)
            assert robert_score[0] == "4"
            assert robert_score[1] == "0"
            assert ml_model_count == 1
            assert partition_count == 1
            assert points_desc[0] == "30:11"
            assert predict_graphs
            assert metrics_train_count == 1
            assert metrics_test_count == 1
            # advanced analysis, flawed models section 1
            assert "-3 / 0" in flawed_models[0]
            assert flawed_image
            # advanced analysis, predictive ability section 2
            assert "0 / 2" in pred_ability[0]
            assert "report/score_w_2_0.jpg" in pred_ability[0]
            # advanced analysis, predictive ability of external test set, item 3
            assert "1 / 2" in pred_test_ability[0]
            assert "report/score_w_2_1.jpg" in pred_test_ability[0]
            # advanced analysis, CV vs test consistency, item 5
            assert "2 / 2" in cv_vs_test_models[0]
            assert "report/score_w_2_2.jpg" in cv_vs_test_models[0]
            # advanced analysis, prediction stability, item 6
            assert "2 / 2" in cv_sd_models[0]
            assert "report/score_w_2_2.jpg" in cv_sd_models[0]
            assert cv_sd_image
            # advanced analysis, boundary robustness, sub-item 1 (Sorted CV, top 20% / High)
            assert "0 / 2" in bound_ability[0]
            assert "report/score_w_2_0.jpg" in bound_ability[0]
            # y distribution and Pearson images
            assert y_distrib_image
            assert pearson_pred_image
            # warnings
            assert find_severe_red
            assert find_moder_y_dist
            assert find_assess_red

        elif test_job == "full_clas":
            # model summary, robert score, predict graphs and model metrics
            # (see the NOTE above the full_workflow block: report_debug_{suffix}.txt now
            # holds a single model/suffix's report, so every field below appears once,
            # except robert_score which still has 2 entries - Interpolation/Boundary)
            assert robert_score[0] == "4"
            assert robert_score[1] == "1"
            assert ml_model_count == 1
            assert points_desc[0] == "30:6"
            # advanced analysis, flawed models section 1
            assert "-2 / 0" in flawed_models[0]
            assert flawed_image
            # advanced analysis, predictive ability section 2
            assert "1 / 3" in pred_ability[0]
            assert "report/score_w_3_1.jpg" in pred_ability[0]
            # advanced analysis, predictive ability of external test set, item 3
            assert "1 / 3" in pred_test_ability[0]
            assert "report/score_w_3_1.jpg" in pred_test_ability[0]
            # advanced analysis, CV vs test consistency, item 5
            assert "2 / 2" in cv_vs_test_models[0]
            assert "report/score_w_2_2.jpg" in cv_vs_test_models[0]
            # advanced analysis, boundary robustness, sub-item 1 (Consistency, sorted CV)
            assert "1 / 2" in bound_ability_clas[0]
            assert "report/score_w_2_1.jpg" in bound_ability_clas[0]
            # y distribution and Pearson images
            assert y_distrib_image
            assert pearson_pred_image
            # warnings
            assert find_severe_red
            assert find_moder_correl
            assert find_assess_red

    if test_job in ["full_workflow", "full_workflow_test", "aqme", "2smiles_columns"]:
        assert find_outliers > 0
        assert find_results_reg > 0
        assert find_results_test_clas == 0
    else:
        assert find_outliers == 0
        assert find_results_reg == 0
        assert find_results_test_clas > 0

    if test_job in ["full_workflow_test", "full_clas_test"]:
        assert find_test > 0
        assert find_results_external > 0
        if test_job == "full_clas_test":
            assert find_results_test_clas > 0
        else:
            assert find_results_test_clas == 0
    else:
        assert find_test == 0
        assert find_results_external == 0

    # common to all reports
    assert find_heatmap > 0
    assert find_verify > 0
    assert find_shap > 0
    assert find_pfi > 0

    # reset the folder
    folders = ["CURATE", "GENERATE", "PREDICT", "VERIFY", "AQME"]
    for folder in folders:
        if os.path.exists(f"{path_main}/{folder}"):
            shutil.rmtree(f"{path_main}/{folder}")
    for file_discard in [
        "report_debug_No_PFI.txt",
        "report_debug_PFI.txt",
        "ROBERT_report_No_PFI.pdf",
        "ROBERT_report_PFI.pdf",
        "AQME-ROBERT_interpret_solubility.csv",
        "AQME-ROBERT_interpret_Robert_example_2smiles.csv",
        "AQME-ROBERT_interpret_solubility_solvent.csv",
        "Robert_example.csv",
        "solubility.csv",
        "solubility_solvent.csv",
    ]:
        if os.path.exists(f"{path_main}/{file_discard}"):
            os.remove(f"{path_main}/{file_discard}")
