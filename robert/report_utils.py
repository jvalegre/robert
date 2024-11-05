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
    
    with open(file, 'r') as datfile:
        lines = datfile.readlines()
        train_outliers,valid_outliers,test_outliers = [],[],[]
        for i,line in enumerate(lines):
            if suffix == 'No PFI':
                if 'o  Outliers plot saved' in line and 'No_PFI.png' in line:
                    train_outliers,valid_outliers,test_outliers = locate_outliers(i,lines)
            if suffix == 'PFI':
                if 'o  Outliers plot saved' in line and 'No_PFI.png' not in line:
                    train_outliers,valid_outliers,test_outliers = locate_outliers(i,lines)

        summary = []

        # add the outlier part
        summary.append(f'\n{spacing*2}<u>Outliers (max. 10 shown)</u>\n')
        summary = summary + train_outliers + valid_outliers + test_outliers

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
    Retrieve the summary of results from the PREDICT and VERIFY dat files
    """
    
    with open(file, 'r') as datfile:
        lines = datfile.readlines()
        start_results,stop_results = 0,0
        for i,line in enumerate(lines):
            if suffix == 'No PFI':
                if 'o  Results saved in' in line and 'No_PFI.dat' in line:
                    start_results,stop_results = locate_results(i,lines)
            if suffix == 'PFI':
                if 'o  Results saved in' in line and 'No_PFI.dat' not in line:
                    start_results,stop_results = locate_results(i,lines)

        # add the summary of results of PREDICT
        start_results += 4
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
    with open(file, 'r') as datfile:
        lines = datfile.readlines()
        for i,line in enumerate(lines):
            if suffix == 'No PFI':
                if 'o  Results saved in' in line and 'No_PFI.dat' in line:
                    for j in range(i,i+15):
                        if 'o  SHAP' in lines[j]:
                            break
                        elif '-  csv_test : ' in lines[j]:
                            results_line = lines[j][9:]
            if suffix == 'PFI':
                if 'o  Results saved in' in line and 'No_PFI.dat' not in line:
                    for j in range(i,i+15):
                        if 'o  SHAP' in lines[j]:
                            break
                        elif '-  csv_test : ' in lines[j]:
                            results_line = lines[j][9:]

    # start the csv_test section
    metrics_dat = f'<p style="text-align: justify; margin-top: -15px; margin-bottom: -3px;">{spacing*2}<u>csv_test metrics</u></p>'

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
            if '_predicted_No_PFI' in file:
                csv_test_file = file
        if suffix == 'PFI':
            if '_predicted_PFI' in file:
                csv_test_file = file

    csv_test_df = pd.read_csv(csv_test_file, encoding='utf-8')

    # start the csv_test section
    pred_line = f'<p style="text-align: justify; margin-top: -15px; margin-bottom: -3px;">{spacing*2}<u>csv_test predictions (sorted, max. 20 shown)</u></p>'

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
    with open(module_file, 'r') as datfile:
        lines = datfile.readlines()
        for _,line in enumerate(lines):
            if '- Target value:' in line:
                y_value = line.split()[-1]
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
    
    train_outliers,valid_outliers,test_outliers = [],[],[]
    len_line = 54
    for j in range(i+1,len(lines)):
        if 'Train:' in lines[j]:
            for k in range(j,len(lines)):
                if 'Validation:' in lines[k]:
                    break
                elif len(train_outliers) <= 10: # 10 outliers and the line with the % of outliers
                    if len(lines[k][6:]) > len_line:
                        outlier_line = f'{lines[k][6:len_line+6]}\n{lines[k][len_line+6:]}'
                    else:
                        outlier_line = lines[k][6:]
                    train_outliers.append(outlier_line)
        elif 'Validation:' in lines[j]:
            for k in range(j,len(lines)):
                if 'Test:' in lines[k] or len(lines[k].split()) == 0:
                    break
                elif len(valid_outliers) <= 10: # 10 outliers and the line with the % of outliers
                    if len(lines[k][6:]) > len_line:
                        outlier_line = f'{lines[k][6:len_line+6]}\n{lines[k][len_line+6:]}'
                    else:
                        outlier_line = lines[k][6:]
                    valid_outliers.append(outlier_line)
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

    return train_outliers,valid_outliers,test_outliers


def locate_results(i,lines):
    """
    Returns the start and end of the outliers section in the PREDICT dat file
    """
    
    start_results = i+1
    stop_results = i+6
    for j in range(i+1,i+10):
        if '-  Test : ' in lines[j]:
            stop_results = i+7
    
    return start_results,stop_results

   
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
    if score_flawed <= 0:
        verify_image = f'{self.args.path_icons}/score_w_3_0.jpg'
    else:
        verify_image = f'{self.args.path_icons}/score_w_3_{score_flawed}.jpg'

    if score_flawed <= 0 or data_score[f'failed_tests_{suffix}'] > 0:
       flaw_result = f'DO NOT USE THIS MODEL! It has important flaws.'
    elif score_flawed in [1,2]:
        flaw_result = f'WARNING! The model might have important flaws.'
    elif score_flawed == 3:
        flaw_result = f'The model predicts right for the right reasons.'

    # adds a bit more space if there is no test set
    score_adv_flawed = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    init_spacing = f'<p style="text-align: justify; margin-top: -14px; margin-bottom: 0px;">{spacing}'
    column = f"""
    {init_spacing}<span style="font-weight:bold;">1. Model vs "flawed" models</span> &nbsp;({score_flawed} / 3 &nbsp;<img src="file:///{verify_image}" alt="ROBERT Score" style="width: 19%">)</p>
    {score_adv_flawed}{flaw_result}<br>{spacing}Pass: +1, Unclear: 0, Fail: -1. <i><a href="https://robert.readthedocs.io/en/latest/Report/score.html" style="text-decorations:none; color:inherit; text-decoration:none;">Details here.</a></i></p>
    """

    return column


def adv_predict(self,suffix,data_score,spacing,test_set,pred_type):
    """
    Gather the advanced analysis of predictive ability
    """

    score_predict = data_score[f'r2_score_{suffix}']
    predict_image = f'{self.args.path_icons}/score_w_2_{score_predict}.jpg'

    if test_set:
        r2_set = 'test'
    else:
        r2_set = 'valid.'

    if pred_type == 'reg':
        metric_type = 'R<sup>2</sup>'
    else:
        metric_type = 'MCC'

    if score_predict == 0:
        predict_result = f'Low predictive ability with {metric_type} ({r2_set}) = {data_score[f"r2_valid_{suffix}"]}.'
    elif score_predict == 1:
        predict_result = f'Moderate predict. ability with {metric_type} ({r2_set}) = {data_score[f"r2_valid_{suffix}"]}.'
    elif score_predict == 2:
        predict_result = f'Good predictive ability with {metric_type} ({r2_set}) = {data_score[f"r2_valid_{suffix}"]}.'
    
    if pred_type == 'reg':
        thres_line = 'R<sup>2</sup> 0.70-0.85: +1, R<sup>2</sup> >0.85: +2.'
    else:
        thres_line = 'MCC 0.50-0.75: +1, MCC >0.75: +2.'

    init_sep = f'<p style="text-align: justify; margin-top: 17px; margin-bottom: 0px;">{spacing}'
    score_adv_pred = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""{init_sep}<span style="font-weight:bold;">2. Predictive ability of the model</span> &nbsp;({score_predict} / 2 &nbsp;<img src="file:///{predict_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_pred}{predict_result}<br>{spacing}{thres_line}</p>
    """

    return column


def adv_cv_r2(self,suffix,data_score,spacing,pred_type):
    """
    Gather the advanced analysis of cross-validation regarding predictive ability
    """

    score_cv_r2 = data_score[f'cv_r2_score_{suffix}']
    cv_r2_image = f'{self.args.path_icons}/score_w_2_{score_cv_r2}.jpg'
    cv_type = data_score[f'cv_type_{suffix}']

    if pred_type == 'reg':
        metric_type = 'R<sup>2</sup>'
    else:
        metric_type = 'MCC'

    if score_cv_r2 == 0:
        cv_result = f'Low predictive ability with {metric_type} ({cv_type}) = {data_score[f"cv_r2_{suffix}"]}.'
    elif score_cv_r2 == 1:
        cv_result = f'Moderate predict. ability with {metric_type} ({cv_type}) = {data_score[f"cv_r2_{suffix}"]}.'
    elif score_cv_r2 == 2:
        cv_result = f'Good predictive ability with {metric_type} ({cv_type}) = {data_score[f"cv_r2_{suffix}"]}.'

    if pred_type == 'reg':
        thres_line = 'R<sup>2</sup> 0.70-0.85: +1, R<sup>2</sup> >0.85: +2.'
    else:
        thres_line = 'MCC 0.50-0.75: +1, MCC >0.75: +2.'

    score_adv_cv = f'<p style="text-align: justify; margin-top: 5px; margin-bottom: 0px;">{spacing}'
    column = f"""{score_adv_cv}<br>{spacing}<span style="font-weight:bold;">3. Cross-validation ({cv_type}) of the model</span></p>
    {score_adv_cv}Overfitting analysis on the model with 3a and 3b:</p>
    <p style="text-align: justify; margin-top: 15px; margin-bottom: 0px;">{spacing}<u>3a. CV predictions train + valid.</u> &nbsp;({score_cv_r2} / 2 &nbsp;<img src="file:///{cv_r2_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_cv}{cv_result}<br>{spacing}{thres_line}</p>
    """

    return column


def adv_cv_sd(self,suffix,data_score,spacing,test_set):
    """
    Gather the advanced analysis of cross-validation regarding variation
    """

    score_cv_sd = data_score[f'cv_sd_score_{suffix}']
    cv_r2_image = f'{self.args.path_icons}/score_w_2_{score_cv_sd}.jpg'
    y_range_covered = round(data_score[f"cv_range_cov_{suffix}"]*100)
    cv_4sd = round(data_score[f"cv_4sd_{suffix}"],1)
    if test_set:
        sd_set = 'test'
    else:
        sd_set = 'valid.'

    if score_cv_sd == 0:
        cv_sd_result = f'High variation, 4*SD ({sd_set}) = {cv_4sd} ({y_range_covered}% y-range).'
    elif score_cv_sd == 1:
        cv_sd_result = f'Moderate variation, 4*SD ({sd_set}) = {cv_4sd} ({y_range_covered}% y-range).'
    elif score_cv_sd == 2:
        cv_sd_result = f'Low variation, 4*SD ({sd_set}) = {cv_4sd} ({y_range_covered}% y-range).'

    score_adv_pred = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<u>3b. Avg. standard deviation (SD)</u> &nbsp;({score_cv_sd} / 2 &nbsp;<img src="file:///{cv_r2_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_pred}{cv_sd_result}<br>{spacing}4*SD 25-50% y-range: +1, 4*SD < 25% y-range: +2. <i>
    <br>{spacing}<a href="https://robert.readthedocs.io/en/latest/Report/score.html" style="text-decorations:none; color:inherit; text-decoration:none;">Details here.</a></i></p>
    """

    return column


def adv_cv_diff(self,suffix,data_score,spacing,test_set):
    """
    Gather the advanced analysis of cross-validation regarding variation
    """

    score_cv_diff = data_score[f'r2_diff_score_{suffix}']
    cv_diff_image = f'{self.args.path_icons}/score_w_2_{score_cv_diff}.jpg'
    cv_diff = round(data_score[f'r2_diff_{suffix}'],2)
    if test_set:
        sd_set = 'test'
    else:
        sd_set = 'valid.'

    if score_cv_diff == 0:
        cv_diff_result = f'High variation ({sd_set} and CV), ΔMCC = {cv_diff}.'
    elif score_cv_diff == 1:
        cv_diff_result = f'Moderate variation ({sd_set} and CV), ΔMCC = {cv_diff}.'
    elif score_cv_diff == 2:
        cv_diff_result = f'Low variation ({sd_set} and CV), ΔMCC = {cv_diff}.'

    score_adv_pred = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<u>3b. MCC difference (model vs CV)</u> &nbsp;({score_cv_diff} / 2 &nbsp;<img src="file:///{cv_diff_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_pred}{cv_diff_result}<br>{spacing}ΔMCC 0.15-0.30: +1, ΔMCC < 0.15: +2.
    """

    return column


def adv_descp(self,suffix,data_score,spacing):
    """
    Gather the advanced analysis of predictive ability
    """

    score_descp = data_score[f'descp_score_{suffix}']
    points_descp_ratio = data_score[f'points_descp_ratio_{suffix}']
    predict_image = f'{self.args.path_icons}/score_w_1_{score_descp}.jpg'
    if score_descp == 0:
        predict_result = f'Number of descps. could be lower (ratio {points_descp_ratio}).'
    elif score_descp == 1:
        predict_result = f'Decent number of descps. (ratio {points_descp_ratio}).'

    score_adv_pred = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""{score_adv_pred}<br>{spacing}<span style="font-weight:bold;">4. Points(train+valid.):descriptors</span> &nbsp;({score_descp} / 1 &nbsp;<img src="file:///{predict_image}" alt="ROBERT Score" style="width: 6.6%">)</p>
    {score_adv_pred}{predict_result}<br>{spacing}5 or more points per descriptor: +1.</p>
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

    excluded_params = [params_dict['error_type'],'train','y']
    misc_params = ['split','type','error_type']
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
                    'VR': f'Voting{model_type} (combining RF, GB and NN)'
                    }

    col_info,sklearn_model = '',''
    for _,ele in enumerate(params_dict.keys()):
        if ele not in excluded_params:
            if ele == 'model' and section == 'model_section':
                sklearn_model = models_dict[params_dict[ele].upper()]
                sklearn_model = f"""{first_line}sklearn model: {sklearn_model}</p>"""
            elif section == 'model_section' and ele.lower() not in misc_params:
                if ele != 'X_descriptors':
                    if ele == 'seed':
                        col_info += f"""{reduced_line}random_state: {params_dict[ele]}</p>"""
                    else:
                        col_info += f"""{reduced_line}{ele}: {params_dict[ele]}</p>"""
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

    descp_warning = 0
    data_score,test_set = get_predict_scores(dat_files['PREDICT'],suffix,pred_type,data_score)

    if data_score[f'descp_score_{suffix}'] == 0:
        descp_warning += 1

    data_score = get_verify_scores(dat_files['VERIFY'],suffix,pred_type,data_score)

    if pred_type == 'reg':
        robert_score = data_score[f'r2_score_{suffix}'] + data_score[f'cv_sd_score_{suffix}'] + data_score[f'cv_r2_score_{suffix}'] + data_score[f'flawed_mod_score_{suffix}'] + data_score[f'descp_score_{suffix}']
    elif pred_type == 'clas':
        # MCC difference between model and CV (the variables say r2 for uniformity with reg)
        r2_diff = round(np.abs(data_score[f'r2_valid_{suffix}']-data_score[f'cv_r2_{suffix}']),2)
        data_score[f'r2_diff_{suffix}'] = r2_diff
        data_score[f'r2_diff_score_{suffix}'] = 0
        if r2_diff < 0.15:
            data_score[f'r2_diff_score_{suffix}'] += 2
        elif r2_diff <= 0.30:
            data_score[f'r2_diff_score_{suffix}'] += 1
        robert_score = data_score[f'r2_score_{suffix}'] + data_score[f'cv_r2_score_{suffix}'] + data_score[f'flawed_mod_score_{suffix}'] + data_score[f'r2_diff_score_{suffix}'] + data_score[f'descp_score_{suffix}']
    
    if robert_score < 0:
        robert_score = 0

    data_score[f'robert_score_{suffix}'] = robert_score

    return data_score,test_set
    

def get_verify_scores(dat_verify,suffix,pred_type,data_score):
    """
    Calculates scores that come from the VERIFY module (VERIFY tests)
    """

    start_data = False
    cv_type = ''
    flawed_score,cv_r2 = 0,0
    failed_tests = 0
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
            if 'Original ' in line and '(valid' in line:
                for j in range(i+1,i+4): # y-mean, y-shuffle and onehot tests
                    if 'PASSED' in dat_verify[j]:
                        flawed_score += 1
                    elif 'FAILED' in dat_verify[j]:
                        flawed_score -= 1
                        failed_tests += 1
                if 'LOOCV' in dat_verify[i+4]:
                    cv_type = 'LOOCV'
                else:
                    cv_type = f'{dat_verify[i+4].split()[1]} CV'
                if pred_type == 'reg':
                    cv_r2 = float(dat_verify[i+4].split('R2 = ')[1].split(',')[0])
                else:
                    cv_r2 = float(dat_verify[i+4].split()[-1])

    # calculate CV scores
    cv_r2_score = score_r2_mcc(pred_type,cv_r2)

    # stores data
    data_score[f'flawed_mod_score_{suffix}'] = flawed_score
    data_score[f'cv_type_{suffix}'] = cv_type
    data_score[f'cv_r2_{suffix}'] = cv_r2
    data_score[f'cv_r2_score_{suffix}'] = cv_r2_score
    data_score[f'failed_tests_{suffix}'] = failed_tests

    return data_score


def get_predict_scores(dat_predict,suffix,pred_type,data_score):
    """
    Calculates scores that come from the PREDICT module (R2 or accuracy, datapoints:descriptors ratio, outlier proportion)
    """

    start_data, test_set = False, False
    data_score[f'r2_score_{suffix}'] = 0
    data_score[f'descp_score_{suffix}'] = 0

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
            if 'o  Results saved in PREDICT/' in line:
                data_score['proportion_ratio_print'] = dat_predict[i+2]
                # R2/MCC from test (if any) or validation
                if pred_type == 'reg':
                    if '-  Test : R2' in dat_predict[i+7]:
                        data_score[f'r2_valid_{suffix}'] = float(dat_predict[i+7].split()[5].split(',')[0])
                        test_set = True
                    elif '-  Valid. : R2' in dat_predict[i+6]:
                        data_score[f'r2_valid_{suffix}'] = float(dat_predict[i+6].split()[5].split(',')[0])

                    data_score[f'r2_score_{suffix}'] = score_r2_mcc(pred_type,data_score[f'r2_valid_{suffix}'])

                elif pred_type == 'clas': # it stores MCC but the label says R2 for consistency with reg
                    if '-  Test : Accuracy' in dat_predict[i+7]:
                        data_score[f'r2_valid_{suffix}'] = float(dat_predict[i+7].split()[-1])
                        test_set = True
                    elif '-  Valid. : Accuracy' in dat_predict[i+6]:
                        data_score[f'r2_valid_{suffix}'] = float(dat_predict[i+6].split()[-1])

                    data_score[f'r2_score_{suffix}'] = score_r2_mcc(pred_type,data_score[f'r2_valid_{suffix}'])

                # proportion of datapoints and descriptors
                data_score[f'points_descp_ratio_{suffix}'] = dat_predict[i+4].split()[-1]
                proportion = int(data_score[f'points_descp_ratio_{suffix}'].split(':')[0]) / int(data_score[f'points_descp_ratio_{suffix}'].split(':')[1])
                if proportion >= 5:
                    data_score[f'descp_score_{suffix}'] += 1            

            # SD from CV
            if pred_type == 'reg':
                if 'o  Cross-validation variation' in line:
                    cv_sd = float(dat_predict[i+2].split()[-1])
                    cv_4sd = 4*cv_sd
                    y_range = float(dat_predict[i+3].split()[-1])
                    y_range_covered = cv_4sd/y_range

                    cv_sd_score = 0
                    if y_range_covered < 0.25:
                        cv_sd_score += 2
                    elif y_range_covered <= 0.50:
                        cv_sd_score += 1

                    data_score[f"cv_4sd_{suffix}"] = cv_4sd
                    data_score[f"cv_range_cov_{suffix}"] = y_range_covered
                    data_score[f'cv_sd_score_{suffix}'] = cv_sd_score

    return data_score,test_set


def score_r2_mcc(pred_type,r2_mcc_val):
    '''
    Calculate scores for R2 and MCC using predetermined thresholds
    '''

    r2_mcc_score = 0

    if pred_type == 'reg': # R2
        if r2_mcc_val > 0.85:
            r2_mcc_score += 2
        elif r2_mcc_val >= 0.7:
            r2_mcc_score += 1

    else: # MCC
        if r2_mcc_val > 0.75:
            r2_mcc_score += 2
        elif r2_mcc_val >= 0.5:
            r2_mcc_score += 1
    
    return r2_mcc_score


def repro_info(modules):
    """
    Retrieves variables used in the Reproducibility and Transparency section
    """

    version_n_date, citation, command_line = '','',''
    python_version, intelex_version, total_time = '','',0
    intelex_installed = True
    dat_files = {}
    for module in modules:
        path_file = Path(f'{os.getcwd()}/{module}/{module}_data.dat')
        if os.path.exists(path_file):
            datfile = open(path_file, 'r', errors="replace")
            txt_file = []
            for line in datfile:
                txt_file.append(line)
                if module.upper() in ['GENERATE','VERIFY','PREDICT']:
                    if 'The scikit-learn-intelex accelerator is not installed' in line:
                        intelex_installed = False
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
    if intelex_installed:
        try:
            import pkg_resources
            intelex_version = pkg_resources.get_distribution("scikit-learn-intelex").version
        except:
            intelex_version = '(version could not be determined)'
    else:
        intelex_version = 'not installed'
    
    return version_n_date, citation, command_line, python_version, intelex_version, total_time, dat_files


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