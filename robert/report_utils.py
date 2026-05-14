#####################################################.
#     This file stores functions from REPORT        #
#####################################################.

import os
import sys
import glob
import pandas as pd
import numpy as np
import textwrap
from pathlib import Path
import ast


title_no_pfi = 'No PFI (standard descriptor filter):'
title_pfi = 'PFI (only important descriptors):'

def get_csv_names(self,command_line):
    """
    Detects the options from a command line or add them from manual inputs
    """
    
    csv_name = ''
    if '--csv_name' in command_line:
        csv_name = command_line.split('--csv_name')[1].split()[0]
        csv_name = remove_quot(csv_name)
    
    csv_test = ''
    if '--csv_test' in command_line:
        csv_test = command_line.split('--csv_test')[1].split()[0]
        csv_test = remove_quot(csv_test)

    if self.args.csv_name == '':
        self.args.csv_name = csv_name

    if self.args.csv_test == '':
        self.args.csv_test = csv_test

    return self


def remove_quot(name):
    '''
    Remove initial and final quotations from names
    '''
    
    if name[0] in ['"',"'"]:
        name = name[1:]
    if name[-1] in ['"',"'"]:
        name = name[:-1]
    
    return name


def get_outliers(file,suffix,spacing):
    """
    Retrieve the summary of results from the PREDICT and VERIFY dat files
    """
    
    with open(file, 'r', encoding='utf-8') as datfile:
        lines = datfile.readlines()
        train_outliers,test_outliers = [],[]
        for i,line in enumerate(lines):
            if suffix == 'No PFI':
                if 'o  Outliers plot saved' in line and 'No_PFI.png' in line:
                    train_outliers,test_outliers = locate_outliers(i,lines)
            if suffix == 'PFI':
                if 'o  Outliers plot saved' in line and 'No_PFI.png' not in line:
                    train_outliers,test_outliers = locate_outliers(i,lines)

        summary = []
        # add the outlier part
        summary.append(f'\n{spacing*2}<u>Outliers (max. 10 shown)</u>\n')
        summary = summary + train_outliers + test_outliers

    summary = f'{spacing*2}'.join(summary)

    # add columns
    if suffix == 'No PFI':
        title_col = title_no_pfi
    elif suffix == 'PFI':
        title_col = title_pfi

    column = f"""
    <p style='margin-top:-18px'><span style="font-weight:bold;">{spacing*2}{title_col}</span></p>
    <pre style="text-align: justify;">{summary}<br></pre>
    """

    return column


def get_metrics(file,suffix,spacing):
    """
    Retrieve the summary of results from the PREDICT dat files
    """
    
    with open(file, 'r', encoding='utf-8') as datfile:
        lines = datfile.readlines()
        start_results,stop_results = 0,0
        for i,line in enumerate(lines):
            if suffix == 'No PFI':
                if 'o  Summary of results' in line and 'No_PFI:' in line:
                    start_results = i+1
                    stop_results = i+6
            if suffix == 'PFI':
                if 'o  Summary of results' in line and 'No_PFI:' not in line:
                    start_results = i+1
                    stop_results = i+6

        # add the summary of results of PREDICT
        start_results += 4 # skip informaton that aren't metrics
        summary = []
        for line in lines[start_results:stop_results+1]:
            if 'R2' in line:
                line = line.replace('R2','R<sup>2</sup>')

            if suffix == 'No PFI':
                summary.append(line[8:])
            elif suffix == 'PFI':
                summary.append(f'{spacing}{spacing}{line[8:]}')

    summary = ''.join(summary)

    column = f"""
    <pre style="text-align: justify; margin-top: 10px;">{summary}</pre>
    """

    return column


def get_csv_metrics(file,suffix,spacing):
    """
    Retrieve the csv_test results from the PREDICT dat file
    """
    
    results_line = ''
    with open(file, 'r', encoding='utf-8') as datfile:
        lines = datfile.readlines()
        for i,line in enumerate(lines):
            if suffix == 'No PFI':
                if 'o  Summary of results' in line and 'No_PFI:' in line:
                    for j in range(i,i+15):
                        if 'o  SHAP' in lines[j]:
                            break
                        elif '-  External test : ' in lines[j]:
                            results_line = lines[j][25:]
            if suffix == 'PFI':
                if 'o  Summary of results' in line and 'No_PFI:' not in line:
                    for j in range(i,i+15):
                        if 'o  SHAP' in lines[j]:
                            break
                        elif '-  External test : ' in lines[j]:
                            results_line = lines[j][25:]

    # start the csv_test section
    metrics_dat = f'<p style="text-align: justify; margin-top: -15px; margin-bottom: -3px;">{spacing*2}<u>External test metrics</u></p>'

    # add line with model metrics (if any)
    if results_line != '':
        metrics_dat += f'<p style="text-align: justify; margin-bottom: 35px;">{spacing*2}{results_line}</p>'
    
        return metrics_dat
    
    else:
        return ''


def get_csv_pred(suffix,path_csv_test,y_value,names,spacing):
    """
    Retrieve the csv_test results from the PREDICT dat file
    """
    
    pred_line = ''
    csv_test_folder = f'{os.getcwd()}/{os.path.dirname(path_csv_test)}'
    csv_test_list = glob.glob(f'{csv_test_folder}/*.csv')
    for file in csv_test_list:
        if suffix == 'No PFI':
            if '_No_PFI.csv' in file:
                csv_test_file = file
        if suffix == 'PFI':
            if '_No_PFI.csv' not in file and '_PFI.csv' in file:
                csv_test_file = file

    csv_test_df = pd.read_csv(csv_test_file, encoding='utf-8')

    # start the csv_test section
    pred_line = f'<p style="text-align: justify; margin-top: -15px; margin-bottom: -3px;">{spacing*2}<u>External test predictions (sorted, max. 20 shown)</u></p>'

    if suffix == 'No PFI':
        pred_line += f'<p style="text-align: justify; margin-bottom: 20px;">{spacing*2}From /PREDICT/csv_test/...No_PFI.csv</p>'
    elif suffix == 'PFI':
        pred_line += f'<p style="text-align: justify; margin-bottom: 20px;">{spacing*2}From /PREDICT/csv_test/..._PFI.csv</p>'

    pred_line += '''<style>
    th, td {
    border:0.75px solid black;
    border-collapse: collapse;
    padding: 2px;
    text-align: justify;
    }
    </style>
    '''

    y_val_exist = False
    if f'{y_value}' in csv_test_df.columns:
        y_val_exist = True

    # adjust format of headers
    names_head = names
    if len(str(names_head)) > 12:
        names_head = f'{str(names_head[:9])}...'
    y_value_head = y_value
    if len(str(y_value_head)) > 12:
        y_value_head = f'{str(y_value_head[:9])}...'

    if pred_line != '':
        if suffix == 'No PFI':
            margin_left = 0
        else:
            margin_left = 27

    pred_line += f'''
    <table style="width:91%; margin-left: {margin_left}px; margin-top: 20px">
        <tr>
            <td><strong>{names_head}</strong></td>'''
    if y_val_exist:
        pred_line += f'''
            <td><strong>{y_value_head}</strong></td>'''
    if f'{y_value}_pred_sd' in csv_test_df:
        pred_line += f'''
                <td><strong>{y_value_head}_pred ± sd</strong></td>
            </tr>'''
    else:
        pred_line += f'''
                <td><strong>{y_value_head}_pred</strong></td>
            </tr>'''
    
    # retrieve and sort the values
    if not y_val_exist:
        csv_test_df[y_value] = csv_test_df[f'{y_value}_pred']

    # in clas problems, there are no SD in the predictions (we use a list of 0s)
    if f'{y_value}_pred_sd' in csv_test_df:
        sd_list = csv_test_df[f'{y_value}_pred_sd']
    else:
        sd_list = [0] * len(csv_test_df[f'{y_value}_pred'])

    y_pred_sorted, y_sorted, names_sorted, sd_sorted = (list(t) for t in zip(*sorted(zip(csv_test_df[f'{y_value}_pred'], csv_test_df[y_value], csv_test_df[names], sd_list), reverse=True)))

    max_table = False
    if len(y_pred_sorted) > 20:
        max_table = True

    count_entries = 0
    for y_val_pred, y_val, name, sd in zip(y_pred_sorted, y_sorted, names_sorted, sd_sorted):
        # adjust format of entries
        if len(str(name)) > 12:
            name = f'{str(name[:9])}...'
        y_val_pred = round(y_val_pred, 2)
        y_val = round(y_val, 2)
        sd = round(sd, 2)
        if f'{y_value}_pred_sd' in csv_test_df:
            y_val_pred_formatted = f'{y_val_pred} ± {sd}'
        else:
            y_val_pred_formatted = f'{y_val_pred}'
        add_entry = True
        # if there are more than 20 predictions, only 20 values will be shown
        if max_table and count_entries >= 10:
            add_entry = False
            if count_entries == 10:
                pred_line += f'''
                <tr>
                    <td>...</td>'''
                if y_val_exist:
                    pred_line += f'''
                    <td>...</td>'''
                pred_line += f'''
                    <td>...</td>
                </tr>'''
            elif count_entries >= (len(y_pred_sorted) - 10):
                add_entry = True
        if add_entry:
            pred_line += f'''
            <tr>
                <td>{name}</td>'''
            if y_val_exist:
                pred_line += f'''
                <td>{y_val}</td>'''
            pred_line += f'''
                <td>{y_val_pred_formatted}</td>
            </tr>'''
        count_entries += 1

    pred_line += f'''
    </table>
    <p style="margin-bottom: 30px"></p>'''

    return pred_line


def detect_predictions(module_file):
    """
    Check whether there are predictions from an external test set
    """
    
    csv_test_exists = False

    # summary of the external CSV test set (if any)
    y_value, names, path_csv_test = '','',''
    with open(module_file, 'r', encoding= 'utf-8') as datfile:
        lines = datfile.readlines()
        for _,line in enumerate(lines):
            if '- Target value:' in line:
                y_value = ' '.join(line.split(':')[1:]).strip()
            elif '- Names:' in line:
                names = line.split()[-1]
            elif 'External set with predicted results:' in line:
                path_csv_test = line.split()[-1]
                csv_test_exists = True

    return csv_test_exists, y_value, names, path_csv_test


def locate_outliers(i,lines):
    """
    Returns the start and end of the PREDICT summary in the dat file
    """
    
    train_outliers,test_outliers = [],[]
    len_line = 54
    for j in range(i+1,len(lines)):
        if 'Train:' in lines[j]:
            for k in range(j,len(lines)):
                if 'Test:' in lines[k]:
                    break
                elif len(train_outliers) <= 10: # 10 outliers and the line with the % of outliers
                    if len(lines[k][6:]) > len_line:
                        outlier_line = f'{lines[k][6:len_line+6]}\n{lines[k][len_line+6:]}'
                    else:
                        outlier_line = lines[k][6:]
                    train_outliers.append(outlier_line)
        elif 'Test:' in lines[j]:
            for k in range(j,len(lines)):
                if len(lines[k].split()) == 0:
                    break
                elif len(test_outliers) <= 10: # 10 outliers and the line with the % of outliers
                    if len(lines[k][6:]) > len_line:
                        outlier_line = f'{lines[k][6:len_line+6]}\n{lines[k][len_line+6:]}'
                    else:
                        outlier_line = lines[k][6:]
                    test_outliers.append(outlier_line)

        if len(lines[j].split()) == 0:
            break

    return train_outliers,test_outliers

   
def combine_cols(columns):
    """
    Makes a string with multi-column lines
    """
    
    column_data = ''
    for column in columns:
        column_data += f'<div style="flex: 1;">{column}</div>'

    combined_data = f"""
    <div style="display: flex;">
    {column_data}
    </div>
    """

    return combined_data


def revert_list(list_tuple):
    """
    Reverts the order of a list of two components
    """

    if len(list_tuple) == 2 and 'No_PFI' in list_tuple[1]:
        new_sort = [] # for some reason reverse() gives a weird issue when reverting lists
        new_sort.append(list_tuple[1])
        new_sort.append(list_tuple[0])
        list_tuple = new_sort

    return list_tuple


def get_col_score(score_info,data_score,suffix,spacing,eval_only):
    """
    Gather the information regarding the score of the No PFI and PFI models
    """
    
    ML_line_format = f'<p style="text-align: justify; margin-top: -10px; margin-bottom: 0px;">{spacing}'
    part_line_format = f'<p style="text-align: justify; margin-top: 1px; margin-bottom: 0px;">{spacing}'

    score_title = f'''&nbsp;&nbsp;·&nbsp;&nbsp;Score  {data_score[f'robert_score_{suffix}']}'''
    if suffix == 'No PFI':
        caption = f'{spacing}{title_no_pfi.replace(":",score_title)}'

    elif suffix == 'PFI':
        caption = f'{spacing}{title_pfi.replace(":",score_title)}'

    partitions_ratio = data_score['proportion_ratio_print'].split('-  Proportion ')[1]

    if not eval_only:
        title_line = f'{caption}'
    else:
        title_line = 'Summary and score of your model (No PFI)'

    column = f"""<p style="margin-top:-18px;"><span style="font-weight:bold;">{title_line}</span></p>
    {ML_line_format}Model = {data_score['ML_model']}&nbsp;&nbsp;·&nbsp;&nbsp;{partitions_ratio}</p>
    {part_line_format}Points(train+validation):descriptors = {data_score[f'points_descp_ratio_{suffix}']}</p>
    <p style="margin-top: 4px;">{score_info}
    <p style="margin-bottom: 18px;"></p>
    """

    return column


def adv_flawed(self,suffix,data_score,spacing):
    """
    Gather the advanced analysis of flawed models
    """

    score_flawed = data_score[f'flawed_mod_score_{suffix}']

    if score_flawed == 0:
        flaw_result = f'The model predicts right for the right reasons.'
    else:
        flaw_result = f'Warning! The model probably has important flaws.'

    # adds a bit more space if there is no test set
    score_adv_flawed = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    init_spacing = f'<p style="text-align: justify; margin-top: -14px; margin-bottom: 0px;">{spacing}'
    column = f"""
    {init_spacing}<span style="font-weight:bold;">1. Model vs "flawed" models</span> &nbsp;({score_flawed} / 0)</p>
    {score_adv_flawed}{flaw_result}<br>{spacing}<i>· Scoring from -6 to 0 ·</i><br>{spacing}Pass: 0, Unclear: -1, Fail: -2.</p>
    """

    return column


def adv_predict(self,suffix,data_score,spacing,pred_type):
    """
    Gather the advanced analysis of predictive ability

    Updated for classification:
      - Instead of awarding up to 2 points, we now award up to 3.
      - We define new thresholds for MCC:
            if MCC > 0.75 => 3, if 0.50 < MCC <= 0.75 => 2,
            if 0.30 < MCC <= 0.50 => 1, else => 0
    """

    score_predict = data_score.get(f'cv_score_combined_{suffix}', 0)
    cv_type = data_score.get(f"cv_type_{suffix}", "10x 5-fold CV")

    if pred_type == 'reg':
        predict_image = f'{self.args.path_icons}/score_w_2_{score_predict}.jpg'
        metric_type = ['Scaled RMSE','R<sup>2</sup>']
        scaled_rmse_cv = data_score.get(f'scaled_rmse_cv_{suffix}', 0)
        r2_cv = data_score.get(f'r2_cv_{suffix}', 0)

        predict_result = f'{metric_type[0]} ({cv_type}) = {scaled_rmse_cv}%.'
        predict_result += f'<br>{spacing}{metric_type[1]} ({cv_type}) = {r2_cv}.'
        thres_line = 'Scaled RMSE ≤ 10%: +2, Scaled RMSE ≤ 20%: +1.'
        thres_line += f'<br>{spacing}R<sup>2</sup> < 0.5: -2, R<sup>2</sup> < 0.7: -1'
        init_sep = f'<p style="text-align: justify; margin-top: 17px; margin-bottom: 0px;">{spacing}'
        score_adv_pred = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
        column = f"""{init_sep}<span style="font-weight:bold;">2. CV predictions of the model</span> &nbsp;({score_predict} / 2 &nbsp;<img src="file:///{predict_image}" alt="score" style="width: 13%">)</p>
        {score_adv_pred}{predict_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}{thres_line}</p>
        """
        return column
    else:
        # Classification: award up to 3 points
        mcc_cv = data_score.get(f'r2_cv_{suffix}', 0)

        if mcc_cv > 0.75:
            display_score = 3
        elif mcc_cv > 0.5:  # up to 0.75
            display_score = 2
        elif mcc_cv > 0.3:  # up to 0.5
            display_score = 1
        else:
            display_score = 0

        predict_image = f'{self.args.path_icons}/score_w_3_{display_score}.jpg'
        metric_type = ['MCC']
        predict_result = f'{metric_type[0]} ({cv_type}) = {mcc_cv}.'
        thres_line = "MCC >0.75: +3; 0.50-0.75: +2; 0.30-0.50: +1"

        init_sep = f'<p style="text-align: justify; margin-top: 17px; margin-bottom: 0px;">{spacing}'
        score_adv_pred = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
        column = f"""{init_sep}<span style="font-weight:bold;">2. CV predictions of the model</span> &nbsp;({display_score} / 3 &nbsp;<img src="file:///{predict_image}" alt="score" style="width: 13%">)</p>
        {score_adv_pred}{predict_result}<br>{spacing}<i>· Scoring from 0 to 3 ·</i><br>{spacing}{thres_line}</p>
        """
        return column


def adv_test(self,suffix,data_score,spacing,pred_type):
    """
    Gather the advanced analysis of predictive ability with the test set
    """

    score_test = data_score.get(f'test_score_combined_{suffix}', 0)

    if pred_type == 'reg':
        test_image = f'{self.args.path_icons}/score_w_2_{score_test}.jpg'
        metric_type = ['Scaled RMSE','R<sup>2</sup>']
        predict_result = f'{metric_type[0]} (test set) = {data_score.get(f"scaled_rmse_test_{suffix}", 0)}%.'
        predict_result += f'<br>{spacing}{metric_type[1]} (test set) = {data_score.get(f"r2_test_{suffix}", 0)}.'
        thres_line = 'Scaled RMSE ≤ 10%: +2, Scaled RMSE ≤ 20%: +1.'
        thres_line += f'<br>{spacing}R<sup>2</sup> < 0.5: -2, R<sup>2</sup> < 0.7: -1'
        score_adv_cv = f'<p style="text-align: justify; margin-top: 5px; margin-bottom: 0px;">{spacing}'
        column = f"""{score_adv_cv}<br>{spacing}<span style="font-weight:bold;">3. Predictive ability & overfitting</span></p>
        <p style="text-align: justify; margin-top: 15px; margin-bottom: 0px;">{spacing}<u>3a. Predictions test set</u> &nbsp;({score_test} / 2 &nbsp;<img src="file:///{test_image}" alt="score" style="width: 13%">)</p>
        {score_adv_cv}{predict_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}{thres_line}</p><br>
        """
        return column
    else:
        # Classification: award up to 3 points
        test_mcc = data_score.get(f"r2_test_{suffix}", 0)

        if test_mcc > 0.75:
            display_score = 3
        elif test_mcc > 0.5:
            display_score = 2
        elif test_mcc > 0.3:
            display_score = 1
        else:
            display_score = 0

        test_image = f'{self.args.path_icons}/score_w_3_{display_score}.jpg'
        metric_type = ['MCC']
        predict_result = f'{metric_type[0]} (test set) = {test_mcc}.'
        thres_line = ('MCC >0.75: +3; 0.50-0.75: +2; 0.30-0.50: +1')

        score_adv_cv = f'<p style="text-align: justify; margin-top: 5px; margin-bottom: 0px;">{spacing}'
        column = f"""{score_adv_cv}<br>{spacing}<span style="font-weight:bold;">3. Predictive ability & overfitting</span></p>
        <p style="text-align: justify; margin-top: 15px; margin-bottom: 0px;">{spacing}<u>3a. Predictions test set</u> &nbsp;({display_score} / 3 &nbsp;<img src="file:///{test_image}" alt="score" style="width: 13%">)</p>
        {score_adv_cv}{predict_result}<br>{spacing}<i>· Scoring from 0 to 3 ·</i><br>{spacing}{thres_line}</p><br>
        """
        return column


def adv_diff_test(self,suffix,data_score,spacing,pred_type):
    """
    Gather the advanced analysis of difference in model performance between CV and test set.
    For regression, we compare scaled RMSE. For classification, we compare Δ MCC.
    """
    
    if pred_type == 'reg':
        # Regression: use diff_scaled_rmse_score
        score_diff_test = data_score[f'diff_scaled_rmse_score_{suffix}']
        diff_test_image = f'{self.args.path_icons}/score_w_2_{score_diff_test}.jpg'
        
        diff_result = f'RMSE in test is {round(data_score[f"factor_scaled_rmse_{suffix}"],2)}*scaled RMSE (CV).'
        
        thres_line = 'Scaled RMSE (test) ≤ 1.25*scaled RMSE (CV): +2.'
        thres_line += f'<br>{spacing}Scaled RMSE (test) ≤ 1.50*scaled RMSE (CV): +1.'
    else:
        # Classification: use diff_mcc_score instead
        score_diff_test = data_score.get(f'diff_mcc_score_{suffix}', 0)
        diff_test_image = f'{self.args.path_icons}/score_w_2_{score_diff_test}.jpg'
        
        # Calculate the absolute difference between CV MCC and test MCC
        mcc_cv = data_score.get(f'r2_cv_{suffix}', 0)
        mcc_test = data_score.get(f'r2_test_{suffix}', 0)
        diff_mcc = round(abs(mcc_test - mcc_cv), 2)
        
        diff_result = f'The ΔMCC between CV and test is {diff_mcc}.'
        
        thres_line = 'ΔMCC ≤ 0.15: +2, ΔMCC ≤ 0.30: +1'

    score_adv_diff = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<u>3b. Prediction accuracy test vs CV</u> &nbsp;({score_diff_test} / 2 &nbsp;<img src="file:///{diff_test_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_diff}<i>Relative differences in values from sections 2 and 3a.</i><br>
    {spacing}{diff_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}{thres_line}</p><br>
    """

    return column


def adv_cv_sd(self,suffix,data_score,spacing):
    """
    Gather the advanced analysis of test predictions regarding variation
    """

    score_cv_sd = data_score[f'cv_sd_score_{suffix}']
    cv_r2_image = f'{self.args.path_icons}/score_w_2_{score_cv_sd}.jpg'
    y_range_covered = round(data_score[f"cv_range_cov_{suffix}"]*100)
    cv_4sd = round(data_score[f"cv_4sd_{suffix}"],1)

    if score_cv_sd == 0:
        cv_sd_result = f'High variation, 4*SD = {cv_4sd} ({y_range_covered}% y-range).'
    elif score_cv_sd == 1:
        cv_sd_result = f'Moderate variation, 4*SD = {cv_4sd} ({y_range_covered}% y-range).'
    elif score_cv_sd == 2:
        cv_sd_result = f'Low variation, 4*SD = {cv_4sd} ({y_range_covered}% y-range).'

    score_adv_pred = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<u>3c. Avg. standard deviation (SD)</u> &nbsp;({score_cv_sd} / 2 &nbsp;<img src="file:///{cv_r2_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_pred}{cv_sd_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}4*SD ≤ 25% y-range: +2, 4*SD ≤ 50% y-range: +1.</p>
    """

    return column


def adv_cv_diff(self,suffix,data_score,spacing,pred_type,test_set=False):
    """
    Gather the advanced analysis of cross-validation regarding variation
    """

    if pred_type == 'clas':
        # Skip entirely for classification
        return ""

    score_cv_diff = data_score.get(f'r2_diff_score_{suffix}', 0)
    cv_diff_image = f'{self.args.path_icons}/score_w_2_{score_cv_diff}.jpg'
    cv_diff = round(data_score.get(f'r2_diff_{suffix}', 0), 2)

    if test_set:
        sd_set = 'test'
    else:
        sd_set = 'valid.'

    # Build R2 difference text
    if score_cv_diff == 0:
        cv_diff_result = f'High variation ({sd_set} and CV), ΔR² = {cv_diff}.'
    elif score_cv_diff == 1:
        cv_diff_result = f'Moderate variation ({sd_set} and CV), ΔR² = {cv_diff}.'
    elif score_cv_diff == 2:
        cv_diff_result = f'Low variation ({sd_set} and CV), ΔR² = {cv_diff}.'
            
    metric_label = "ΔR²"
    threshold_text = f'{metric_label} 0.15-0.30: +1, {metric_label} < 0.15: +2.'
    title = f"<u>3b. R² difference (model vs CV)</u>"

    score_adv_pred = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}{title} &nbsp;({score_cv_diff} / 2 &nbsp;<img src="file:///{cv_diff_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_pred}{cv_diff_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}{threshold_text}<br>
    """    
    return column


def adv_sorted_cv(self, suffix, data_score, spacing, pred_type):
    """
    Gather the advanced analysis of sorted CV (fold-by-fold variation).
    For regression, we look at scaled_rmse_sorted_{suffix}.
    For classification, we look at sorted_mcc_{suffix}, awarding points
    based on how many folds stay near the maximum, indicating consistent performance.
    """

    score_sorted = data_score.get(f'sorted_cv_score_{suffix}', 0)
    sorted_cv_image = f'{self.args.path_icons}/score_w_2_{score_sorted}.jpg'

    error_keyword = "rmse" if pred_type.lower() == 'reg' else "mcc"

    
    # "3d. Extrapolation/Consistency (sorted CV)"
    if f'scaled_{error_keyword}_sorted_{suffix}' not in data_score:
        data_score[f'scaled_{error_keyword}_sorted_{suffix}'] = []

    if pred_type == 'reg':
        title_cap = '3d. Extrapolation'
        sorted_rmse = [f'{val}%' for val in data_score[f'scaled_{error_keyword}_sorted_{suffix}']]
    else:
        title_cap = '3c. Consistency'
        sorted_rmse = [f'{val}' for val in data_score[f'scaled_{error_keyword}_sorted_{suffix}']]

    sorted_rmse_str = str(sorted_rmse).replace("'", '')

    column = f"""
    <p style="text-align: justify; margin-top: 35px; margin-bottom: 0px;">{spacing}<u>{title_cap} (sorted CV)</u> &nbsp;({score_sorted} / 2 &nbsp;<img src="file:///{sorted_cv_image}" alt="ROBERT Score" style="width: 13%">)</p>
    <p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}Scaled {error_keyword.upper()}s across 5-fold CV:
    <br>{spacing}{sorted_rmse_str}
    <br>{spacing}<i>· Scoring from 0 to 2 ·</i>
    <br>{spacing}Every two folds with {error_keyword.upper()}s ≤ 1.25*min {error_keyword.upper()}: +1.</p>
    """
    return column


def get_col_text(type_thres):
    """
    Gather the information regarding the thresholds used in the score and abbreviation sections
    """

    reduced_line = '<p style="text-align: justify; margin-top: -8px;">' # reduces line separation separation

    first_line = '<p style="text-align: justify; margin-top: -25px;">'
    if type_thres == 'abbrev_1':
        abbrev_list = ['<strong>ACC:</strong> accuracy',
                        '<strong>ADAB:</strong> AdaBoost',
                        '<strong>CSV:</strong> comma separated values',
                        '<strong>CLAS:</strong> classification',
                        '<strong>CV:</strong> cross-validation',
                        '<strong>F1 score:</strong> balanced F-score',
                        '<strong>GB:</strong> gradient boosting',
                        '<strong>GP:</strong> gaussian process'
        ]

    elif type_thres == 'abbrev_2':
        abbrev_list = ['<strong>KN:</strong> k-nearest neighbors',
                        '<strong>MAE:</strong> root-mean-square error', 
                        "<strong>MCC:</strong> Matthew's correl. coefficient",
                        '<strong>ML:</strong> machine learning',                          
                        '<strong>MVL:</strong> multivariate lineal models',
                        '<strong>NN:</strong> neural network',
                        '<strong>PFI:</strong> permutation feature importance',
                        '<strong>R2:</strong> coefficient of determination'
        ]
    elif type_thres == 'abbrev_3':
        abbrev_list = ['<strong>REG:</strong> Regression',
                        '<strong>RF:</strong> random forest',
                        '<strong>RMSE:</strong> root mean square error',
                        '<strong>RND:</strong> random',
                        '<strong>SHAP:</strong> Shapley additive explanations',
                        '<strong>VR:</strong> voting regressor',
        ]

    column = ''
    for i,ele in enumerate(abbrev_list):
        if i == 0:
            column += f"""{first_line}{ele}</p>
"""
        else:
            column += f"""{reduced_line}{ele}</p>
"""

    return column


def get_col_transpa(params_dict,suffix,section,spacing):
    """
    Gather the information regarding the model parameters represented in the Reproducibility section
    """

    first_line = f'<p style="text-align: justify; margin-top: -40px;">{spacing*2}' # reduces line separation separation
    reduced_line = f'<p style="text-align: justify; margin-top: -8px;">{spacing*2}' # reduces line separation separation

    if suffix == 'No PFI':
        caption = f'{title_no_pfi}'

    elif suffix == 'PFI':
        caption = f'{title_pfi}'

    excluded_params = [f"combined_{params_dict['error_type']}", 'train', 'X_descriptors', 'y', 'error_train', 'cv_error', 'names']
    misc_params = ['type','error_type','split','kfold','repeat_kfolds','seed']
    if params_dict['type'] == 'reg':
        model_type = 'Regressor'
    elif params_dict['type'] == 'clas':
        model_type = 'Classifier'
    models_dict = {'RF': f'RandomForest{model_type}',
                    'MVL': 'LinearRegression',
                    'GB': f'GradientBoosting{model_type}',
                    'NN': f'MLP{model_type}',
                    'GP': f'GaussianProcess{model_type}',
                    'ADAB': f'AdaBoost{model_type}',
                    }

    col_info,sklearn_model = '',''
    for _,ele in enumerate(params_dict.keys()):
        if ele not in excluded_params:
            if ele == 'model' and section == 'model_section':
                sklearn_model = models_dict[params_dict[ele].upper()]
                sklearn_model = f"""{first_line}sklearn model: {sklearn_model}</p>"""
            elif section == 'model_section' and ele.lower() not in misc_params:
                if ele == 'params':
                    model_params = ast.literal_eval(params_dict['params'])
                    for param in model_params:
                        col_info += f"""{reduced_line}{param}: {model_params[param]}</p>"""
            elif section == 'misc_section' and ele.lower() in misc_params:
                if col_info == '':
                    col_info += f"""{first_line}{ele}: {params_dict[ele]}</p>"""
                else:
                    col_info += f"""{reduced_line}{ele}: {params_dict[ele]}</p>"""
    
    column = f"""<p style="margin-top: -30px;"><span style="font-weight:bold;">{spacing*2}{caption}</span></p>
    {sklearn_model}{col_info}
    """

    return column


def calc_score(dat_files,suffix,pred_type,data_score):
    '''
    Calculates ROBERT score
    '''

    data_score = get_predict_scores(dat_files['PREDICT'],suffix,pred_type,data_score)

    data_score = get_verify_scores(dat_files['VERIFY'],suffix,pred_type,data_score)

    if pred_type == 'reg':
        robert_score = data_score.get(f'cv_score_combined_{suffix}', 0) + data_score.get(f'test_score_combined_{suffix}', 0) \
                + data_score.get(f'cv_sd_score_{suffix}', 0) + data_score.get(f'diff_scaled_rmse_score_{suffix}', 0) \
                + data_score.get(f'flawed_mod_score_{suffix}', 0) + data_score.get(f'sorted_cv_score_{suffix}', 0)
        # Adjustment to avoid negative values
        if robert_score < 0:
            robert_score = 0
        # Assign the final value
        data_score[f'robert_score_{suffix}'] = robert_score

    elif pred_type == 'clas':
        # Calculate the difference between CV MCC and test MCC
        mcc_cv = data_score.get(f'r2_cv_{suffix}', 0)
        mcc_test = data_score.get(f'r2_test_{suffix}', 0)
        diff_mcc = round(np.abs(mcc_test - mcc_cv), 2)

        # Assign a score based on the MCC gap (e.g., ±2, ±1, 0)
        data_score[f'diff_mcc_score_{suffix}'] = 0
        if diff_mcc < 0.15:
            data_score[f'diff_mcc_score_{suffix}'] += 2
        elif diff_mcc <= 0.30:
            data_score[f'diff_mcc_score_{suffix}'] += 1

        # Sum scores similarly to regression:
        robert_score = (
            data_score.get(f'cv_score_combined_{suffix}', 0)
            + data_score.get(f'test_score_combined_{suffix}', 0)
            + data_score.get(f'flawed_mod_score_{suffix}', 0)
            + data_score.get(f'sorted_cv_score_{suffix}', 0)
            + data_score.get(f'diff_mcc_score_{suffix}', 0)
            + data_score.get(f'descp_score_{suffix}', 0)
        )
        # Adjustment to avoid negative values
        if robert_score < 0:
            robert_score = 0

        # Assign the final value
        data_score[f'robert_score_{suffix}'] = robert_score

    return data_score
    

def get_verify_scores(dat_verify,suffix,pred_type,data_score):
    """
    Calculates scores that come from the VERIFY module (VERIFY tests)
    """

    start_data = False
    flawed_score = 0
    failed_tests = 0
    sorted_cv_score = 0
    for i,line in enumerate(dat_verify):
        # set starting points for No PFI and PFI models
        if suffix == 'No PFI':
            if '------- ' in line and '(No PFI)' in line:
                start_data = True
            elif '------- ' in line and 'with PFI' in line:
                start_data = False
        if suffix == 'PFI':
            if '------- ' in line and 'with PFI' in line:
                start_data = True
        
        if start_data:
            error_keyword = "rmse" if pred_type.lower() == 'reg' else "mcc"
            if f"Original {error_keyword.upper()} (" in line:
                for j in range(i+1,i+4): # y-mean, y-shuffle and onehot tests
                    if 'UNCLEAR' in dat_verify[j]:
                        flawed_score -= 1
                    elif 'FAILED' in dat_verify[j]:
                        flawed_score -= 2
                        failed_tests += 1
                if '- Sorted ' in dat_verify[i+4]:
                    sorted_cv_results = dat_verify[i+4].split(f'{error_keyword.upper()} = ')[-1]
                    sorted_cv_results = ast.literal_eval(sorted_cv_results)
                    if pred_type.lower() == 'reg':
                        data_score[f'scaled_{error_keyword}_sorted_{suffix}'] = [round((val/data_score[f'y_range_{suffix}'])*100,2) for val in sorted_cv_results]
                    else:
                        data_score[f'scaled_{error_keyword}_sorted_{suffix}'] = sorted_cv_results # no scaling for MCC

                    # define min and max values
                    data_score[f'min_scaled_{error_keyword}_{suffix}'] = min(data_score[f'scaled_{error_keyword}_sorted_{suffix}'])
                    idx_min_scaled_rmse = data_score[f'scaled_{error_keyword}_sorted_{suffix}'].index(data_score[f'min_scaled_{error_keyword}_{suffix}'])
                    data_score[f'max_scaled_{error_keyword}_{suffix}'] = max(data_score[f'scaled_{error_keyword}_sorted_{suffix}'])
                    idx_max_scaled_rmse = data_score[f'scaled_{error_keyword}_sorted_{suffix}'].index(data_score[f'max_scaled_{error_keyword}_{suffix}'])

                    data_score[f'scaled_{error_keyword}_results_sorted_{suffix}'] = []
                    for idx,err in enumerate(data_score[f'scaled_{error_keyword}_sorted_{suffix}']):
                        if pred_type.lower() == 'reg':
                            if idx == idx_min_scaled_rmse:
                                data_score[f'scaled_{error_keyword}_results_sorted_{suffix}'].append('min')
                            elif err <= (data_score[f'min_scaled_{error_keyword}_{suffix}']*1.25):
                                data_score[f'scaled_{error_keyword}_results_sorted_{suffix}'].append('pass')
                            else:
                                data_score[f'scaled_{error_keyword}_results_sorted_{suffix}'].append('fail')
                        else:
                            if idx == idx_max_scaled_rmse:
                                data_score[f'scaled_{error_keyword}_results_sorted_{suffix}'].append('max')
                            elif err >= (data_score[f'max_scaled_{error_keyword}_{suffix}']*0.75):
                                data_score[f'scaled_{error_keyword}_results_sorted_{suffix}'].append('pass')
                            else:
                                data_score[f'scaled_{error_keyword}_results_sorted_{suffix}'].append('fail')

                    sorted_cv_score = int(data_score[f'scaled_{error_keyword}_results_sorted_{suffix}'].count('pass')/2)

    # adjust max 1 point for flawed tests
    if flawed_score > 1:
        flawed_score = 1
  
    # stores data
    data_score[f'flawed_mod_score_{suffix}'] = flawed_score
    data_score[f'failed_tests_{suffix}'] = failed_tests
    data_score[f'sorted_cv_score_{suffix}'] = sorted_cv_score

    return data_score


def get_predict_scores(dat_predict,suffix,pred_type,data_score):
    """
    Calculates scores that come from the PREDICT module (R2 or accuracy, datapoints:descriptors ratio, outlier proportion)
    """

    start_data = False
    data_score[f'rmse_score_{suffix}'] = 0
    data_score[f'cv_type_{suffix}'] = "10x 5-fold CV"

    for i,line in enumerate(dat_predict):

        # set starting points for No PFI and PFI models
        if suffix == 'No PFI':
            if '------- ' in line and '(No PFI)' in line:
                start_data = True
            elif '------- ' in line and 'with PFI' in line:
                start_data = False
        if suffix == 'PFI':
            if '------- ' in line and 'with PFI' in line:
                start_data = True
        
        if start_data:
            # model type
            if line.startswith('   - Model:'):
                data_score['ML_model'] = line.split()[-1]
            # R2 and proportion
            if 'o  Summary of results' in line:
                data_score['proportion_ratio_print'] = dat_predict[i+2]
                data_score[f'points_descp_ratio_{suffix}'] = dat_predict[i+4].split()[-1]

                # scaled RMSE/MCC from test (if any) or validation
                if pred_type == 'reg':
                    if '-fold CV : R2 =' in dat_predict[i+5]:
                        data_score[f'rmse_cv_{suffix}'] = float(dat_predict[i+5].split()[-1])
                        data_score[f"cv_type_{suffix}"] = ' '.join([ele for ele in dat_predict[i+5].split()[1:4]])
                        data_score[f'r2_cv_{suffix}'] = float(dat_predict[i+5].split(',')[0].split()[-1])
                    if 'Test : R2 =' in dat_predict[i+6]:
                        data_score[f'rmse_test_{suffix}'] = float(dat_predict[i+6].split()[-1])
                        data_score[f'r2_test_{suffix}'] = float(dat_predict[i+6].split(',')[0].split()[-1])
                    if '-  y range of dataset' in dat_predict[i+8]:
                        data_score[f'y_range_{suffix}'] = float(dat_predict[i+8].split()[-1])

                    data_score[f'scaled_rmse_cv_{suffix}'] = round((data_score[f'rmse_cv_{suffix}']/data_score[f'y_range_{suffix}'])*100,2)
                    data_score[f'scaled_rmse_test_{suffix}'] = round((data_score[f'rmse_test_{suffix}']/data_score[f'y_range_{suffix}'])*100,2)

                    data_score[f'cv_score_rmse_{suffix}'] = score_rmse_mcc(pred_type,data_score[f'scaled_rmse_cv_{suffix}'])
                    data_score[f'test_score_rmse_{suffix}'] = score_rmse_mcc(pred_type,data_score[f'scaled_rmse_test_{suffix}'])

                    # get penalties for R2
                    data_score[f'cv_penalty_r2_{suffix}'] = calc_penalty_r2(data_score[f'r2_cv_{suffix}'])
                    data_score[f'test_penalty_r2_{suffix}'] = calc_penalty_r2(data_score[f'r2_test_{suffix}'])

                    # combined scores RMSE/R2 (min 0)
                    data_score[f'cv_score_combined_{suffix}'] = data_score[f'cv_score_rmse_{suffix}'] + data_score[f'cv_penalty_r2_{suffix}']
                    if data_score[f'cv_score_combined_{suffix}'] < 0:
                        data_score[f'cv_score_combined_{suffix}'] = 0
                    data_score[f'test_score_combined_{suffix}'] = data_score[f'test_score_rmse_{suffix}'] + data_score[f'test_penalty_r2_{suffix}']
                    if data_score[f'test_score_combined_{suffix}'] < 0:
                        data_score[f'test_score_combined_{suffix}'] = 0

                    diff_score = 0
                    # relative difference between RMSE from test and CV
                    data_score[f'factor_scaled_rmse_{suffix}'] = data_score[f'scaled_rmse_test_{suffix}'] / data_score[f'scaled_rmse_cv_{suffix}']
                    if data_score[f'factor_scaled_rmse_{suffix}'] <= 1.25:
                        diff_score += 2
                    elif data_score[f'factor_scaled_rmse_{suffix}'] <= 1.5:
                        diff_score += 1
                    data_score[f'diff_scaled_rmse_score_{suffix}'] = diff_score
    
                elif pred_type == 'clas':  # Process classification: using MCC extracted from CV and Test results
                    # Extract MCC from the 10x 5-fold CV line
                    if '5-fold' in dat_predict[i+5]:
                        parts = dat_predict[i+5].split(',')
                        mcc_cv = None
                        for part in parts:
                            if 'MCC' in part:
                                mcc_cv = float(part.split('=')[-1])
                                break
                        if mcc_cv is not None:
                            data_score[f'r2_cv_{suffix}'] = mcc_cv  # storing MCC in a key keyed as r2_cv for consistency
                    # Extract MCC from the Test line
                    if '-  Test :' in dat_predict[i+6]:
                        parts = dat_predict[i+6].split(',')
                        mcc_test = None
                        for part in parts:
                            if 'MCC' in part:
                                mcc_test = float(part.split('=')[-1])
                                break
                        if mcc_test is not None:
                            data_score[f'r2_test_{suffix}'] = mcc_test
                    # Compute CV and Test scores using the classification thresholds in score_rmse_mcc
                    data_score[f'cv_score_rmse_{suffix}'] = score_rmse_mcc(pred_type, data_score.get(f'r2_cv_{suffix}', 0))
                    data_score[f'test_score_rmse_{suffix}'] = score_rmse_mcc(pred_type, data_score.get(f'r2_test_{suffix}', 0))
    
                    # For classification, the combined score is simply the score from MCC (no additional penalty)
                    data_score[f'cv_score_combined_{suffix}'] = data_score[f'cv_score_rmse_{suffix}']
                    data_score[f'test_score_combined_{suffix}'] = data_score[f'test_score_rmse_{suffix}']

            # SD from CV
            if pred_type == 'reg':
                if '-  Average SD in test set' in line:
                    cv_sd = float(line.split()[-1])
                    cv_4sd = 4*cv_sd
                    y_range_covered = cv_4sd/data_score[f'y_range_{suffix}']

                    cv_sd_score = 0
                    if y_range_covered <= 0.25:
                        cv_sd_score += 2
                    elif y_range_covered <= 0.50:
                        cv_sd_score += 1

                    data_score[f"cv_4sd_{suffix}"] = cv_4sd
                    data_score[f"cv_range_cov_{suffix}"] = y_range_covered
                    data_score[f'cv_sd_score_{suffix}'] = cv_sd_score

    return data_score


def score_rmse_mcc(pred_type,scaledrmse_mcc_val):
    '''
    Calculate scores for R2 and MCC using predetermined thresholds
    
    For regression (scaled RMSE): 0-2 points
    For classification (MCC): 0-3 points
    '''

    r2_mcc_score = 0

    if pred_type == 'reg': # scaled RMSE
        if scaledrmse_mcc_val <= 10:
            r2_mcc_score += 2
        elif scaledrmse_mcc_val <= 20:
            r2_mcc_score += 1

    else: # MCC
        if scaledrmse_mcc_val > 0.75:
            r2_mcc_score += 3
        elif scaledrmse_mcc_val > 0.5:
            r2_mcc_score += 2
        elif scaledrmse_mcc_val > 0.3:
            r2_mcc_score += 1
    
    return r2_mcc_score


def calc_penalty_r2(r2_val):
    '''
    Calculate scores for R2 and MCC using predetermined thresholds
    '''

    penalty_r2 = 0

    if r2_val < 0.5:
        penalty_r2 -= 2
    elif r2_val < 0.7:
        penalty_r2 -= 1
    
    return penalty_r2


def repro_info(modules):
    """
    Retrieves variables used in the Reproducibility section
    """

    version_n_date, citation, command_line = '','',''
    python_version, total_time = '',0
    dat_files = {}
    for module in modules:
        path_file = Path(f'{os.getcwd()}/{module}/{module}_data.dat')
        if os.path.exists(path_file):
            datfile = open(path_file, 'r', encoding= 'utf-8', errors="replace")
            txt_file = []
            for line in datfile:
                txt_file.append(line)
                if 'Time' in line and 'seconds' in line:
                    total_time += float(line.split()[2])
                if 'How to cite: ' in line:
                    citation = line.split('How to cite: ')[1]
                if 'ROBERT v' == line[:8]:
                    version_n_date = line
                if 'Command line used in ROBERT: ' in line:
                    if '--csv_name' not in command_line: # ensures that the value for --csv_name is stored
                        command_line = line.split('Command line used in ROBERT: ')[1]
            total_time = round(total_time,2)
            dat_files[module] = txt_file
            datfile.close()
 
    try:
        import platform
        python_version = platform.python_version()
    except:
        python_version = '(version could not be determined)'
    
    return version_n_date, citation, command_line, python_version, total_time, dat_files


def make_report(report_html, HTML):
    """
    Generate a css file that will be used to make the PDF file
    """

    css_files = ["report.css"]
    outfile = f"{os.getcwd()}/ROBERT_report.pdf"
    if os.path.exists(outfile):
        try:
            os.remove(outfile)
        except PermissionError:
            print('\nx  ROBERT_report.pdf is open! Please, close the PDF file and run ROBERT again with --report (i.e., "python -m robert --report").')
            sys.exit()
    pdf = make_pdf(report_html, HTML, css_files)
    _ = Path(outfile).write_bytes(pdf)


def make_pdf(html, HTML, css_files):
    """Generate a PDF file from a string of HTML"""
    htmldoc = HTML(string=html, base_url="")
    if css_files:
        htmldoc = htmldoc.write_pdf(stylesheets=css_files)
    else:
        htmldoc = htmldoc.write_pdf()
    return htmldoc


def css_content(csv_name,robert_version):
    """
    Obtain ROBERT version and CSV name to use it on top of the PDF report
    """

    css_content = f"""
    body {{
    font-size: 12px;
    line-height: 1.5;
    }}
    @page {{
        size: A4;
        margin: 2cm;
        @bottom-right {{
            content: "Page "counter(page) " of " counter(pages);
            font-size: 8pt;
            position: fixed;
            right: 0;
            bottom: 0;
            transform: translateY(-8pt);
            white-space: nowrap;
        }}  
        @bottom-left {{
            content: "ROBERT v {robert_version}";
            font-size: 8pt;
            position: fixed;
            left: 0;
            bottom: 0;
            transform: translateY(-8pt);
            white-space: nowrap;
        }}
        @bottom-center {{
            content: "";
            border-top: 3px solid black;
            width: 100%;
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0pt;
            transform: translateY(8pt);
        }}  
        @top-center {{
            content: "";
            border-top: 3px solid black;
            width: 100%;
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0pt;
            transform: translateY(40pt);
        }}  
        @top-left {{
            content: "ROBERT Report";
            font-size: 8pt;
            font-weight:bold;
            position: fixed;
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0pt;
            transform: translateY(2pt);
            white-space: nowrap;
        }} 
        @top-right {{
            content: "{csv_name}";
            font-size: 8pt;
            font-style: italic;
            position: fixed;
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0pt;
            transform: translateY(2pt);
            white-space: nowrap;
        }} 
    }}
    * {{
        font-family: "Helvetica", Arial, sans-serif;
    }}
    .dat-content {{
        width: 50%;
        max-width: 595pt;
        overflow-x: auto;
        line-height: 1.2;
    }}

    img[src="Robert_logo.jpg"] {{
        float: center;
    }}
    img[src*="Pearson"] {{
        display: inline-block;
        vertical-align: bottom;
        max-width: 48%;
        margin-left: 10px;
        margin-bottom: -5px;
    }}
    img[src*="PFI"] {{
        display: inline-block;
        vertical-align: bottom;
        max-width: 48%;
        margin-left: 10px;
        margin-bottom: -5px;
    }}

    img[src*="PFI"]:first-child {{
        margin-right: 10%;
    }}
    .img-PREDICT {{
        margin-top: 20px;
    }}
    
    hr.black {{
    border: none;
    height: 3px;
    background-color: black;
    }}
    
    hr {{
    border: none;
    height: 1px;
    background-color: gray;
    }}

    body:before {{
    top: 1.2cm;
    }}
    """
    return css_content


def format_lines(module_data, max_width=122, cmd_line=False, one_column=False, spacing=''):
    """
    Reads a file and returns a formatted string between two markers
    """

    formatted_lines = []
    lines = module_data.split('\n')
    for i,line in enumerate(lines):
        if 'R2' in line:
            line = line.replace('R2','R<sup>2</sup>')
        if cmd_line:
            formatted_line = textwrap.fill(line, width=max_width-5, subsequent_indent='')
        else:
            formatted_line = textwrap.fill(line, width=max_width, subsequent_indent='')
        if i > 0:
            formatted_lines.append(f'<pre style="text-align: justify;">\n{formatted_line}</pre>')
        else:
            formatted_lines.append(f'<pre style="text-align: justify;">{formatted_line}</pre>\n')

    # for two columns
    if not one_column:
        return ''.join(formatted_lines)
    
    # for one column
    one_col_lines = ''
    for line in ''.join(formatted_lines).split('\n'):
        if line.startswith('<pre style="text-align: justify;">') and line != '<pre style="text-align: justify;">':
            one_col_lines += line.replace('<pre style="text-align: justify;">',f'<pre style="text-align: justify;">{spacing*3}')
        elif not line.startswith('<'):
            one_col_lines += f'\n{spacing*3}{line}'
        else:
            one_col_lines += f'\n{line}'
    return one_col_lines



def get_spacing_col(suffix,spacing_PFI):
    '''
    Assign spacing of column
    '''
    
    if suffix == 'No PFI':
        spacing = ''
    elif suffix == 'PFI':
        spacing = spacing_PFI
    
    return spacing