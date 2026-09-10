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

# shared left indent for the Boundary robustness column in Section A, so the caption, the
# description text (even once it wraps) and the metrics line below all share one left edge
BOUNDARY_INDENT = '18px'

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

            summary.append(f'{spacing}{line[8:]}')

    summary = ''.join(summary)

    column = f"""
    <pre style="text-align: justify; margin-top: 10px;">{summary}</pre>
    """

    return column


def get_boundary_metrics(data_score,suffix,pred_type,spacing):
    """
    Gather the Low/High (sorted CV) boundary robustness metrics for the ROBERT score section
    """

    if pred_type.lower() == 'reg':
        # repeated-CV values from boundary_plot() (same methodology as interpolation) -
        # not VERIFY's old single-pass sorted-CV array, which is kept only for classification
        low_val = data_score.get(f'scaled_rmse_low_{suffix}')
        high_val = data_score.get(f'scaled_rmse_high_{suffix}')
        if low_val is None or high_val is None:
            return ''
    else:
        sorted_vals = data_score.get('scaled_mcc_sorted_' + suffix, [])
        if not sorted_vals:
            return ''
        low_val = sorted_vals[0]
        high_val = sorted_vals[-1]

    if pred_type == 'reg':
        metric_line = f'Scaled RMSE (Low, sorted CV) = {low_val}%.'
        metric_line += f'<br>Scaled RMSE (High, sorted CV) = {high_val}%.'
    else:
        metric_line = f'MCC (Low, sorted CV) = {low_val}.'
        metric_line += f'<br>MCC (High, sorted CV) = {high_val}.'

    indent = f'padding-left: {BOUNDARY_INDENT};' if spacing else ''
    column = f"""
    <pre style="margin-top: 10px; {indent} box-sizing: border-box;">{metric_line}</pre>
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
    csv_test_file = None
    csv_test_folder = f'{os.getcwd()}/{os.path.dirname(path_csv_test)}'
    csv_test_list = glob.glob(f'{csv_test_folder}/*.csv')
    for file in csv_test_list:
        if suffix == 'No PFI':
            if '_No_PFI.csv' in file:
                csv_test_file = file
        if suffix == 'PFI':
            if '_No_PFI.csv' not in file and '_PFI.csv' in file:
                csv_test_file = file

    # this suffix has no matching csv_test predictions file (e.g. a model with no PFI variant
    # under --all_models), skip the section instead of crashing with a NameError
    if csv_test_file is None:
        return ''

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

    margin_left = 0

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

   
def combine_cols(columns,align_top=False):
    """
    Makes a string with multi-column lines

    align_top=True uses align-items:flex-start instead of the flex default (stretch), so a
    short column isn't forced to visually stretch down to match a much taller sibling (e.g.
    Section B's rows, where one side can carry far more content/images than the other) -
    default stays False (unchanged) everywhere else, since other sections haven't been
    re-verified against this behavior change.
    """

    column_data = ''
    for column in columns:
        column_data += f'<div style="flex: 1;">{column}</div>'

    align_style = 'align-items: flex-start;' if align_top else ''
    combined_data = f"""
    <div style="display: flex; {align_style}">
    {column_data}
    </div>
    """

    return combined_data


def get_col_score(score_info,data_score,suffix,col,spacing):
    """
    Gather the information regarding the score of the model, for the Interpolation (left)
    or Boundary robustness (right) column
    """

    ML_line_format = f'<p style="text-align: justify; margin-top: -10px; margin-bottom: 0px;">{spacing}'
    part_line_format = f'<p style="text-align: justify; margin-top: 1px; margin-bottom: 0px;">{spacing}'

    score_key = 'interp_score' if col == 'interpolation' else 'extrap_score'
    score_val = data_score.get(f'{score_key}_{suffix}', 0)
    score_title = f'''&nbsp;&nbsp;·&nbsp;&nbsp;Score  {score_val}'''

    if col == 'interpolation':
        caption = f'{spacing}Interpolation{score_title}'
        partitions_ratio = data_score['proportion_ratio_print'].split('-  Proportion ')[1]

        column = f"""<p style="margin-top:-18px;"><span style="font-weight:bold;">{caption}</span></p>
        {ML_line_format}Model = {data_score['ML_model']}&nbsp;&nbsp;·&nbsp;&nbsp;{partitions_ratio}</p>
        {part_line_format}Points(train+validation):descriptors = {data_score[f'points_descp_ratio_{suffix}']}</p>
        <p style="margin-top: 4px;">{score_info}
        <p style="margin-bottom: 18px;"></p>
        """

    else:
        # single shared padding-left (not leading nbsp) so the caption, the description text
        # (even once it wraps to a 2nd line) and the score bar all start at the same left edge
        indent = f'padding-left: {BOUNDARY_INDENT};' if spacing else ''

        # 2nd line (mirrors interpolation's "Points(...):descriptors" line so both score bars
        # sit at the same height) - reports how many points fall in the bottom/top 20% (Low/
        # High) folds that the boundary robustness score is actually computed from, out of the
        # total. Uses total_points (train+validation+test), since the Low/High sorted CV is
        # run over the whole dataset (see sort_n_load()), not just the train+validation pool.
        total_points = data_score.get(f'total_points_{suffix}')
        extreme_line = ''
        if total_points:
            # approximate: total_points isn't always evenly divisible by 5, and array-splitting
            # distributes any remainder unevenly across folds, so the real Low/High fold sizes
            # can differ from this by a point or two
            fold_points = round(total_points/5)
            extreme_line = f'Points in bottom+top 20% ~= {2*fold_points}:{total_points}.'

        column = f"""<div style="{indent} box-sizing: border-box;">
        <p style="margin-top:-18px;"><span style="font-weight:bold;">Boundary robustness{score_title}</span></p>
        <p style="margin-top: -10px; margin-bottom: 0px;">Performance at the extremes of the y-range.</p>
        <p style="margin-top: 1px; margin-bottom: 0px;">{extreme_line}</p>
        <p style="margin-top: 4px;">{score_info}
        <p style="margin-bottom: 18px;"></p>
        </div>
        """

    return column


def adv_flawed(suffix,data_score,spacing):
    """
    Gather the advanced analysis of flawed models
    """

    score_flawed = data_score.get(f'flawed_mod_score_{suffix}', 0)

    if score_flawed == 0:
        flaw_result = f'The model predicts right for the right reasons.'
    else:
        flaw_result = f'Warning! The model probably has important flaws.'

    # adds a bit more space if there is no test set
    score_adv_flawed = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    init_spacing = f'<p style="text-align: justify; margin-top: -14px; margin-bottom: 0px;">{spacing}'
    column = f"""
    {init_spacing}<span style="font-weight:bold;">1. Model vs "flawed" models</span> &nbsp;({score_flawed} / 0)</p>
    {score_adv_flawed}{flaw_result}<br>{spacing}<i>· Scoring from -8 to 0 ·</i><br>{spacing}Pass: 0, Unclear: -1, Fail: -2.</p>
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
        predict_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_predict}.jpg'
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
        # Classification: award up to 3 points - reuse the same thresholds already computed
        # into cv_score_combined_{suffix} via score_rmse_mcc(), instead of a second hand-rolled
        # copy that could silently drift from the score actually feeding interp_score
        mcc_cv = data_score.get(f'r2_cv_{suffix}', 0)
        display_score = score_predict

        predict_image = f'{self._posix_uri(self.args.path_icons)}/score_w_3_{display_score}.jpg'
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
        test_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_test}.jpg'
        metric_type = ['Scaled RMSE','R<sup>2</sup>']
        predict_result = f'{metric_type[0]} (test set) = {data_score.get(f"scaled_rmse_test_{suffix}", 0)}%.'
        predict_result += f'<br>{spacing}{metric_type[1]} (test set) = {data_score.get(f"r2_test_{suffix}", 0)}.'
        thres_line = 'Scaled RMSE ≤ 10%: +2, Scaled RMSE ≤ 20%: +1.'
        thres_line += f'<br>{spacing}R<sup>2</sup> < 0.5: -2, R<sup>2</sup> < 0.7: -1'
        score_adv_cv = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
        column = f"""<p style="text-align: justify; margin-top: 17px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">3. Test set predictions</span> &nbsp;({score_test} / 2 &nbsp;<img src="file:///{test_image}" alt="score" style="width: 13%">)</p>
        {score_adv_cv}{predict_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}{thres_line}</p>
        """
        return column
    else:
        # Classification: award up to 3 points - reuse the same thresholds already computed
        # into test_score_combined_{suffix} via score_rmse_mcc(), instead of a second hand-rolled
        # copy that could silently drift from the score actually feeding interp_score
        test_mcc = data_score.get(f"r2_test_{suffix}", 0)
        display_score = score_test

        test_image = f'{self._posix_uri(self.args.path_icons)}/score_w_3_{display_score}.jpg'
        metric_type = ['MCC']
        predict_result = f'{metric_type[0]} (test set) = {test_mcc}.'
        thres_line = ('MCC >0.75: +3; 0.50-0.75: +2; 0.30-0.50: +1')

        score_adv_cv = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
        column = f"""<p style="text-align: justify; margin-top: 17px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">3. Test set predictions</span> &nbsp;({display_score} / 3 &nbsp;<img src="file:///{test_image}" alt="score" style="width: 13%">)</p>
        {score_adv_cv}{predict_result}<br>{spacing}<i>· Scoring from 0 to 3 ·</i><br>{spacing}{thres_line}</p>
        """
        return column


def adv_diff_test(self,suffix,data_score,spacing,pred_type):
    """
    Gather the advanced analysis of difference in model performance between CV and test set.
    For regression, we compare scaled RMSE. For classification, we compare Δ MCC.
    """
    
    if pred_type == 'reg':
        # Regression: use diff_scaled_rmse_score
        score_diff_test = data_score.get(f'diff_scaled_rmse_score_{suffix}', 0)
        diff_test_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_diff_test}.jpg'
        
        diff_result = f'RMSE in test is {round(data_score[f"factor_scaled_rmse_{suffix}"],2)}*scaled RMSE (CV).'
        
        thres_line = 'Scaled RMSE (test) ≤ 1.25*scaled RMSE (CV): +2.'
        thres_line += f'<br>{spacing}Scaled RMSE (test) ≤ 1.50*scaled RMSE (CV): +1.'
    else:
        # Classification: use diff_mcc_score instead
        score_diff_test = data_score.get(f'diff_mcc_score_{suffix}', 0)
        diff_test_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_diff_test}.jpg'
        
        # Calculate the absolute difference between CV MCC and test MCC
        mcc_cv = data_score.get(f'r2_cv_{suffix}', 0)
        mcc_test = data_score.get(f'r2_test_{suffix}', 0)
        diff_mcc = round(abs(mcc_test - mcc_cv), 2)
        
        diff_result = f'The ΔMCC between CV and test is {diff_mcc}.'
        
        thres_line = 'ΔMCC ≤ 0.15: +2, ΔMCC ≤ 0.30: +1'

    score_adv_diff = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">5. CV vs test consistency</span> &nbsp;({score_diff_test} / 2 &nbsp;<img src="file:///{diff_test_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_diff}<i>Relative differences in values from sections 2 and 3.</i><br>
    {spacing}{diff_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}{thres_line}</p>
    """

    return column


def adv_cv_sd(self,suffix,data_score,spacing,pred_type='reg'):
    """
    Interpolation item 6: prediction stability under resampling, combining three facets - each
    scored 0-2 with its own threshold, final score is their average, rounded.

    Regression:
    (a) SD of the repeated-CV predictions in the test set (unchanged from before)
    (b) SD of the out-of-fold CV predictions themselves in train+validation (new)
    (c) how much the aggregate RMSE itself varies from repeat to repeat (new) - a
        dataset-wide view of the same "how much does this depend on the random split"
        question (a)/(b) ask per-point

    Classification: the same three facets, adapted for a discrete label (see the 'clas'
    branch in get_predict_scores() and print_predict() in predict_utils.py):
    (a) disagreement rate between individual CV repeats and the majority-voted class, test set
    (b) same, for the out-of-fold train+validation predictions
    (c) coefficient of variation of the per-repeat MCC (instead of per-repeat RMSE)
    """

    score_cv_sd = data_score[f'cv_sd_score_{suffix}']
    cv_r2_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_cv_sd}.jpg'

    if pred_type == 'reg':
        y_range_covered = round(data_score.get(f"cv_range_cov_{suffix}",0)*100)
        cv_4sd = round(data_score.get(f"cv_4sd_{suffix}",0),1)
        y_range_covered_train = round(data_score.get(f"cv_range_cov_train_{suffix}",0)*100)
        cv_4sd_train = round(data_score.get(f"cv_4sd_train_{suffix}",0),1)
        rmse_cv_pct = data_score.get(f'rmse_cv_pct_{suffix}',0)

        cv_sd_result = (
            f'(a) Test SD, 4*SD = {cv_4sd} ({y_range_covered}% y-range).'
            f'<br>{spacing}(b) CV SD, 4*SD = {cv_4sd_train} ({y_range_covered_train}% y-range).'
            f'<br>{spacing}(c) RMSE coef. of variation = {rmse_cv_pct}%.'
        )
        thres_line = '(a)/(b) 4*SD ≤25%/≤50%. (c) ≤15%/≤25%. Avg., rounded.'
    else:
        disagree_a = data_score.get(f'clas_disagree_a_{suffix}',0)
        disagree_b = data_score.get(f'clas_disagree_b_{suffix}',0)
        mcc_cv_pct = data_score.get(f'mcc_cv_pct_{suffix}',0)

        cv_sd_result = (
            f'(a) Test disagreement (10 repeats) = {disagree_a}%.'
            f'<br>{spacing}(b) CV disagreement (10 repeats) = {disagree_b}%.'
            f'<br>{spacing}(c) MCC coef. of variation = {mcc_cv_pct}%.'
        )
        thres_line = '(a)/(b)/(c) ≤15%/≤25%. Avg., rounded.'

    score_adv_pred = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">6. Prediction stability</span> &nbsp;({score_cv_sd} / 2 &nbsp;<img src="file:///{cv_r2_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_pred}{cv_sd_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}{thres_line}</p>
    """

    return column


def adv_sorted_cv(self, suffix, data_score, spacing, pred_type):
    """
    Gather the advanced analysis of sorted CV fold-by-fold consistency, for classification
    only (regression's boundary robustness column uses the 5 dedicated functions below
    instead: adv_sorted_cv_high, adv_sorted_cv_low, adv_spearman, adv_bound_sd,
    adv_applicability_domain).
    """

    score_sorted = data_score.get(f'sorted_cv_score_{suffix}', 0)
    sorted_cv_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_sorted}.jpg'

    if 'scaled_mcc_sorted_' + suffix not in data_score:
        data_score[f'scaled_mcc_sorted_{suffix}'] = []
    sorted_mcc = [f'{val}' for val in data_score[f'scaled_mcc_sorted_{suffix}']]
    sorted_mcc_str = str(sorted_mcc).replace("'", '')

    column = f"""
    <p style="text-align: justify; margin-top: -14px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">1. Consistency (sorted CV)</span> &nbsp;({score_sorted} / 2 &nbsp;<img src="file:///{sorted_cv_image}" alt="ROBERT Score" style="width: 13%">)</p>
    <p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}MCCs across 5-fold CV:
    <br>{spacing}{sorted_mcc_str}
    <br>{spacing}<i>· Scoring from 0 to 2 ·</i>
    <br>{spacing}Every two folds with MCCs ≥ 0.75*max MCC: +1.</p>
    """
    return column


def adv_train_val_gap(self,suffix,data_score,spacing):
    """
    Interpolation sub-metric 4: gap between out-of-fold validation error and the same fold's
    own in-fold training fit (a large gap indicates the model memorizes each fold's training
    split instead of generalizing to its own held-out validation split)
    """

    score_gap = data_score.get(f'train_val_gap_score_{suffix}', 0)
    gap_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_gap}.jpg'

    scaled_trainfit = data_score.get(f'scaled_rmse_trainfit_{suffix}')
    factor_trainfit = data_score.get(f'factor_trainfit_{suffix}')

    if scaled_trainfit is None:
        gap_result = 'Not available.'
    else:
        gap_result = f'Validation is {round(factor_trainfit,2)}*train RMSE.'

    thres_line = 'Validation ≤ 1.25*train RMSE: +2.'
    thres_line += f'<br>{spacing}Validation ≤ 1.50*train RMSE: +1.'

    score_adv_gap = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">4. Train vs validation gap</span> &nbsp;({score_gap} / 2 &nbsp;<img src="file:///{gap_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_gap}{gap_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}{thres_line}</p>
    """

    return column


def adv_sorted_cv_high(self,suffix,data_score,spacing):
    """
    Boundary robustness sub-metric 1: sorted-CV fold for the top 20% (High, highest y)
    """

    score_high = data_score.get(f'sorted_cv_high_score_{suffix}', 0)
    high_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_high}.jpg'
    scaled_high = data_score.get(f'scaled_rmse_high_{suffix}', 0)
    crossing_high = data_score.get(f'crossing_high_{suffix}', 0)

    score_adv_high = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: -14px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">1. Sorted CV, top 20% (High)</span> &nbsp;({score_high} / 2 &nbsp;<img src="file:///{high_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_high}Scaled RMSE (High, sorted CV) = {scaled_high}%.<br>{spacing}Beyond training range (High) = {round(crossing_high*100)}%.<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}Scaled RMSE ≤ 10%: +2, ≤ 20%: +1.<br>{spacing}Crossing < 50%: -1, < 20%: -2.</p>
    """

    return column


def adv_sorted_cv_low(self,suffix,data_score,spacing):
    """
    Boundary robustness sub-metric 2: sorted-CV fold for the bottom 20% (Low, lowest y)
    """

    score_low = data_score.get(f'sorted_cv_low_score_{suffix}', 0)
    low_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_low}.jpg'
    scaled_low = data_score.get(f'scaled_rmse_low_{suffix}', 0)
    crossing_low = data_score.get(f'crossing_low_{suffix}', 0)

    score_adv_low = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">2. Sorted CV, bottom 20% (Low)</span> &nbsp;({score_low} / 2 &nbsp;<img src="file:///{low_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_low}Scaled RMSE (Low, sorted CV) = {scaled_low}%.<br>{spacing}Beyond training range (Low) = {round(crossing_low*100)}%.<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}Scaled RMSE ≤ 10%: +2, ≤ 20%: +1.<br>{spacing}Crossing < 50%: -1, < 20%: -2.</p>
    """

    return column


def adv_spearman(self,suffix,data_score,spacing):
    """
    Boundary robustness sub-metric 3: Spearman rank correlation within each extreme fold's own
    points - does the model at least rank the Low/High fold's points correctly (biggest actual
    value -> biggest predicted value), even if the absolute error (scored in items 1/2) is off?
    Scored per side (Low, High) independently, separate from the RMSE-based items 1/2, which
    are penalized by the "real range extension" crossing rate instead (see
    get_verify_scores() and calc_penalty_crossing())
    """

    score_spearman = data_score.get(f'spearman_score_{suffix}', 0)
    spearman_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_spearman}.jpg'
    spearman_low = data_score.get(f'spearman_low_{suffix}')
    spearman_high = data_score.get(f'spearman_high_{suffix}')

    if spearman_low is None or spearman_high is None:
        spearman_result = 'Not available.'
    else:
        spearman_result = f'Spearman rank : Low = {spearman_low}, High = {spearman_high}.'

    score_adv_spearman = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">3. Spearman rank</span> &nbsp;({score_spearman} / 2 &nbsp;<img src="file:///{spearman_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_spearman}{spearman_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}Low ≥ 0.5: +1. High ≥ 0.5: +1.</p>
    """

    return column


def adv_bound_sd(self,suffix,data_score,spacing):
    """
    Boundary robustness sub-metric 4: degradation ratio. Compares each extreme fold's
    (Low/High) scaled RMSE against the RMSE of its own "remaining 80%" baseline from the same
    sorted-CV run - how much worse does the model get at the edges of the y-range compared to
    its own typical performance? Scored independently per side (+1 each), so one
    badly-degraded side cannot hide behind a fine one.
    """

    degradation_score = data_score.get(f'degradation_score_{suffix}', 0)
    degradation_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{degradation_score}.jpg'
    ratio_high = data_score.get(f'degradation_ratio_high_{suffix}')
    ratio_low = data_score.get(f'degradation_ratio_low_{suffix}')

    if ratio_high is None or ratio_low is None:
        degradation_result = 'Not available.'
    else:
        degradation_result = f'Degradation vs 80% RMSE: High {round(ratio_high,2)}x, Low {round(ratio_low,2)}x.'

    score_adv_degradation = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">4. Degradation ratio</span> &nbsp;({degradation_score} / 2 &nbsp;<img src="file:///{degradation_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_degradation}{degradation_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}Low ≤ 1.5x: +1. High ≤ 1.5x: +1.</p>
    """

    return column


def adv_applicability_domain(self,suffix,data_score,spacing):
    """
    Boundary robustness sub-metric 5: applicability domain (leverage) - boundary robustness in
    X-descriptor space (train on the 80% "typical" points, test on the 20% highest-leverage,
    most structurally/descriptor-wise extreme points), complementing sub-metrics 1-4 (all
    about boundary robustness in y-space)
    """

    score_ad = data_score.get(f'applicability_domain_score_{suffix}', 0)
    ad_image = f'{self._posix_uri(self.args.path_icons)}/score_w_2_{score_ad}.jpg'
    scaled_rmse_ad = data_score.get(f'scaled_rmse_ad_{suffix}')

    if scaled_rmse_ad is None:
        ad_result = 'Not available.'
    else:
        ad_result = f'Scaled RMSE (high-leverage 20%) = {scaled_rmse_ad}%.'

    score_adv_ad = f'<p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}'
    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">5. Applicability domain (leverage)</span> &nbsp;({score_ad} / 2 &nbsp;<img src="file:///{ad_image}" alt="ROBERT Score" style="width: 13%">)</p>
    {score_adv_ad}{ad_result}<br>{spacing}<i>· Scoring from 0 to 2 ·</i><br>{spacing}Scaled RMSE ≤ 10%: +2, ≤ 20%: +1.</p>
    """

    return column


def adv_bound_diagnostic(suffix,data_score,spacing):
    """
    Boundary robustness item 6 (unscored): additional diagnostic kept for context - global Spearman
    rank correlation over the whole sorted dataset (the local, per-extreme Spearman is already
    scored in item 3). Not scored, but still useful signal, so kept visible as its own item
    instead of dropped - mirrors Interpolation's item 6 slot so both columns keep the same row
    count
    """

    spearman_global = data_score.get(f'spearman_global_{suffix}')

    if spearman_global is None:
        return ''

    result = f'Global Spearman rank (whole dataset) = {spearman_global}.'

    column = f"""<p style="text-align: justify; margin-top: 20px; margin-bottom: 0px;">{spacing}<span style="font-weight:bold;">6. Additional diagnostics</span> &nbsp;(unscored)</p>
    <p style="text-align: justify; margin-top: 3px; margin-bottom: 0px;">{spacing}{result}</p>
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
    Calculates the Interpolation and Boundary robustness scores (independent, 0-10 each)
    '''

    data_score = get_predict_scores(dat_files['PREDICT'],suffix,pred_type,data_score)

    data_score = get_verify_scores(dat_files['VERIFY'],suffix,pred_type,data_score)

    if pred_type == 'reg':
        # Interpolation: flawed-model gate + CV + test + train-vs-validation gap +
        # CV-vs-test consistency + avg SD (max 10: 5 sub-points x 0/2, flawed only subtracts)
        interp_score = data_score.get(f'flawed_mod_score_{suffix}', 0) \
                + data_score.get(f'cv_score_combined_{suffix}', 0) + data_score.get(f'test_score_combined_{suffix}', 0) \
                + data_score.get(f'train_val_gap_score_{suffix}', 0) + data_score.get(f'diff_scaled_rmse_score_{suffix}', 0) \
                + data_score.get(f'cv_sd_score_{suffix}', 0)
        if interp_score < 0:
            interp_score = 0

        # Boundary robustness: High + Low sorted-CV + Spearman rank + degradation ratio +
        # applicability domain (max 10: 5 sub-points x 0/2)
        extrap_score = data_score.get(f'sorted_cv_high_score_{suffix}', 0) + data_score.get(f'sorted_cv_low_score_{suffix}', 0) \
                + data_score.get(f'spearman_score_{suffix}', 0) + data_score.get(f'degradation_score_{suffix}', 0) \
                + data_score.get(f'applicability_domain_score_{suffix}', 0)
        if extrap_score < 0:
            extrap_score = 0

        data_score[f'interp_score_{suffix}'] = interp_score
        data_score[f'extrap_score_{suffix}'] = extrap_score

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

        # Interpolation: flawed-model gate + CV + test + MCC-diff (item 5) + prediction
        # stability (item 6, see the 'clas' branch in get_predict_scores()) - Low/High/
        # degradation/bias still aren't well-defined for MCC, so Boundary robustness keeps
        # only the existing fold-consistency score. Max achievable is 10: cv_score_combined(3)
        # + test_score_combined(3) + flawed_mod_score(0, never positive) + diff_mcc_score(2)
        # + cv_sd_score(2) - see interp_max in print_assessment()
        interp_score = (
            data_score.get(f'cv_score_combined_{suffix}', 0)
            + data_score.get(f'test_score_combined_{suffix}', 0)
            + data_score.get(f'flawed_mod_score_{suffix}', 0)
            + data_score.get(f'diff_mcc_score_{suffix}', 0)
            + data_score.get(f'cv_sd_score_{suffix}', 0)
        )
        if interp_score < 0:
            interp_score = 0

        extrap_score = data_score.get(f'sorted_cv_score_{suffix}', 0)
        if extrap_score < 0:
            extrap_score = 0

        data_score[f'interp_score_{suffix}'] = interp_score
        data_score[f'extrap_score_{suffix}'] = extrap_score

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
                for j in range(i+1,i+5): # y-mean, y-shuffle, onehot and cluster tests
                    if 'UNCLEAR' in dat_verify[j]:
                        flawed_score -= 1
                    elif 'FAILED' in dat_verify[j]:
                        flawed_score -= 2
                        failed_tests += 1
                if '- Sorted ' in dat_verify[i+5]:
                    sorted_cv_results = dat_verify[i+5].split(f'{error_keyword.upper()} = ')[-1]
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

                    if pred_type.lower() == 'reg':
                        # boundary robustness sub-metrics: Low (bottom fold) and High (top fold)
                        # scored individually, plus the degradation ratio vs the main CV error.
                        # calc_score() runs get_predict_scores() before get_verify_scores(), so
                        # scaled_rmse_cv_{suffix}/scaled_rmse_low_{suffix}/scaled_rmse_high_{suffix}
                        # are already available here - the latter two come from PREDICT's
                        # boundary_plot() repeated-CV run (same methodology as interpolation),
                        # NOT from VERIFY's old single-pass sorted-CV array
                        # (data_score[f'scaled_{error_keyword}_sorted_{suffix}']), which is kept
                        # around only for classification's adv_sorted_cv() display
                        scaled_low = data_score[f'scaled_rmse_low_{suffix}']
                        scaled_high = data_score[f'scaled_rmse_high_{suffix}']

                        # R2 (a variance-ratio) is unstable on a ~7-point, narrow-range fold
                        # and ends up punishing models regardless of actual quality (see
                        # discussion with the user) - replaced with a penalty based on the
                        # "real range extension" crossing rate (does the model actually predict
                        # beyond the training range it saw, or does it clip toward the
                        # boundary?) instead of Spearman, which now gets its own dedicated item
                        # (see adv_spearman()). Parsed in get_predict_scores() since that's
                        # logged by boundary_plot() into PREDICT_data.dat, and available
                        # here because calc_score() runs get_predict_scores() first
                        crossing_low = data_score.get(f'crossing_low_{suffix}', 0)
                        crossing_high = data_score.get(f'crossing_high_{suffix}', 0)

                        low_score = score_rmse_mcc(pred_type,scaled_low) + calc_penalty_crossing(crossing_low)
                        high_score = score_rmse_mcc(pred_type,scaled_high) + calc_penalty_crossing(crossing_high)
                        data_score[f'sorted_cv_low_score_{suffix}'] = max(low_score,0)
                        data_score[f'sorted_cv_high_score_{suffix}'] = max(high_score,0)

                        # boundary robustness sub-metric 4: degradation ratio - each extreme
                        # fold is compared against the RMSE of its OWN "remaining 80%" from the
                        # same sorted-CV run (not the 10x 5-fold CV RMSE, which is a different
                        # CV procedure over a different dataset scope - see boundary_plot() in
                        # utils.py). Scored independently per side (+1 each), so one
                        # badly-degraded side cannot hide behind a fine one. This has to live
                        # here (not in get_predict_scores(), where every other Low/High
                        # sub-metric is computed) because calc_score() runs get_predict_scores()
                        # BEFORE get_verify_scores() - scaled_rmse_bottom80/top80 only exist
                        # once this function's own loop reaches this point, so computing the
                        # score in get_predict_scores() would always read them as None and
                        # silently score 0/2 regardless of the actual ratios
                        scaled_rmse_bottom80 = data_score.get(f'scaled_rmse_bottom80_{suffix}')
                        scaled_rmse_top80 = data_score.get(f'scaled_rmse_top80_{suffix}')
                        if scaled_rmse_bottom80 and scaled_rmse_top80:
                            data_score[f'degradation_ratio_high_{suffix}'] = scaled_high / scaled_rmse_bottom80
                            data_score[f'degradation_ratio_low_{suffix}'] = scaled_low / scaled_rmse_top80
                            data_score[f'degradation_score_{suffix}'] = int(data_score[f'degradation_ratio_low_{suffix}'] <= 1.5) + int(data_score[f'degradation_ratio_high_{suffix}'] <= 1.5)

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

                # total dataset size (train+validation AND test) - the boundary robustness
                # Low/High folds are drawn from the whole dataset (see sort_n_load()), not just
                # the train+validation pool, so the "points in extreme folds" line needs this
                if 'Points CV (train+valid.):Test' in dat_predict[i+1]:
                    n_train_str,n_test_str = dat_predict[i+1].split('=')[-1].split(':')
                    data_score[f'total_points_{suffix}'] = int(n_train_str) + int(n_test_str)

                    # the score's thresholds (test_score, CV-vs-test consistency, etc.) were
                    # calibrated assuming the standard test_set=0.2 split (see the note in
                    # score.rst) - reconstruct what test_select()/prepare_sets() would have
                    # carved out for that split (same round()+4-point-floor formula as
                    # utils.py) and compare against what actually happened. A mismatch means a
                    # non-default --test_set, including 0 (i.e. no internal test set at all) -
                    # in that case the score isn't meaningful, so it's hidden in the PDF instead
                    # of silently showing a misleadingly low value. --csv_test does NOT affect
                    # this check: it no longer forces the internal split to 0 (see
                    # prepare_sets()), so a run using both an internal split and an external
                    # --csv_test file scores normally
                    n_test = int(n_test_str)
                    expected_test_pts = max(round(0.2 * data_score[f'total_points_{suffix}']), 4)
                    data_score[f'n_test_{suffix}'] = n_test
                    data_score[f'score_available_{suffix}'] = (n_test == expected_test_pts)

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
                    # relative difference between RMSE from test and CV (guarded the same way
                    # as factor_trainfit below: a near-perfect CV fit can round its scaled RMSE
                    # to 0.00, which would otherwise raise ZeroDivisionError here)
                    if data_score[f'scaled_rmse_cv_{suffix}'] > 0:
                        data_score[f'factor_scaled_rmse_{suffix}'] = data_score[f'scaled_rmse_test_{suffix}'] / data_score[f'scaled_rmse_cv_{suffix}']
                    else:
                        data_score[f'factor_scaled_rmse_{suffix}'] = 0
                    if data_score[f'factor_scaled_rmse_{suffix}'] <= 1.25:
                        diff_score += 2
                    elif data_score[f'factor_scaled_rmse_{suffix}'] <= 1.5:
                        diff_score += 1
                    data_score[f'diff_scaled_rmse_score_{suffix}'] = diff_score

                    # train-vs-validation gap (interpolation): how much worse out-of-fold
                    # validation RMSE is compared to the same fold's own in-fold training fit
                    data_score[f'train_val_gap_score_{suffix}'] = 0
                    for j in range(i,i+12):
                        if 'Train fit (in-fold)' in dat_predict[j]:
                            rmse_trainfit = float(dat_predict[j].split('RMSE = ')[-1].strip())
                            data_score[f'scaled_rmse_trainfit_{suffix}'] = round((rmse_trainfit/data_score[f'y_range_{suffix}'])*100,2)
                            if data_score[f'scaled_rmse_trainfit_{suffix}'] > 0:
                                factor_trainfit = data_score[f'scaled_rmse_cv_{suffix}'] / data_score[f'scaled_rmse_trainfit_{suffix}']
                            else:
                                factor_trainfit = 0
                            data_score[f'factor_trainfit_{suffix}'] = factor_trainfit
                            train_val_gap_score = 0
                            if factor_trainfit <= 1.25:
                                train_val_gap_score += 2
                            elif factor_trainfit <= 1.5:
                                train_val_gap_score += 1
                            data_score[f'train_val_gap_score_{suffix}'] = train_val_gap_score
                            break

                    # boundary robustness: RMSE of the extreme folds themselves (Low/High), from
                    # the repeated-CV predictions computed by boundary_plot() in utils.py - this
                    # is the headline "Scaled RMSE (Low/High, sorted CV)" number shown in the
                    # report, replacing VERIFY's old single-pass sorted-CV RMSE (see get_verify_scores)
                    for j in range(i,i+25):
                        if 'Sorted CV extremes (repeated)' in dat_predict[j]:
                            rmse_low = float(dat_predict[j].split('RMSE Low = ')[-1].split(',')[0].strip())
                            rmse_high = float(dat_predict[j].split('RMSE High = ')[-1].strip())
                            data_score[f'scaled_rmse_low_{suffix}'] = round((rmse_low/data_score[f'y_range_{suffix}'])*100,2)
                            data_score[f'scaled_rmse_high_{suffix}'] = round((rmse_high/data_score[f'y_range_{suffix}'])*100,2)
                            break

                    # real range extension - does the model actually predict beyond the range
                    # it was trained on (High: prediction > max value seen in training; Low:
                    # prediction < min value seen), rather than clipping toward the training
                    # boundary? No longer its own scored item - now the penalty inside items
                    # 1/2 (Low/High RMSE), replacing Spearman there (see get_verify_scores() and
                    # calc_penalty_crossing()); still logged by boundary_plot() in utils.py
                    for j in range(i,i+26):
                        if 'Real range extension (Low/High)' in dat_predict[j]:
                            range_line = dat_predict[j]
                            data_score[f'crossing_low_{suffix}'] = float(range_line.split('Low crossing = ')[-1].split(',')[0].strip())
                            data_score[f'crossing_high_{suffix}'] = float(range_line.split('High crossing = ')[-1].strip())
                            break

                    # degradation ratio baseline: RMSE of each extreme fold's own "remaining
                    # 80%" from the same sorted-CV run (see boundary_plot() in utils.py) -
                    # used instead of the 10x 5-fold CV RMSE so the ratio compares within the
                    # same CV procedure/dataset scope rather than across two different ones
                    for j in range(i,i+29):
                        if 'Degradation baseline (remaining 80%)' in dat_predict[j]:
                            rmse_top80 = float(dat_predict[j].split('vs Low (top 80%) RMSE = ')[-1].split(',')[0].strip())
                            rmse_bottom80 = float(dat_predict[j].split('vs High (bottom 80%) RMSE = ')[-1].strip())
                            data_score[f'scaled_rmse_top80_{suffix}'] = round((rmse_top80/data_score[f'y_range_{suffix}'])*100,2)
                            data_score[f'scaled_rmse_bottom80_{suffix}'] = round((rmse_bottom80/data_score[f'y_range_{suffix}'])*100,2)
                            break

                    # boundary robustness sub-metric 3: Spearman rank correlation - internal
                    # (within each fold's own points), logged by boundary_plot() in utils.py.
                    # Now its own dedicated item (used to be a penalty inside items 1/2,
                    # replaced there by the range-extension crossing rate - see get_verify_scores())
                    spearman_global = 0
                    for j in range(i,i+30):
                        if 'Spearman rank (Low/High)' in dat_predict[j]:
                            spearman_line = dat_predict[j]
                            spearman_low = float(spearman_line.split('Low (internal) = ')[-1].split(',')[0].strip())
                            spearman_high = float(spearman_line.split('High (internal) = ')[-1].split(',')[0].strip())
                            data_score[f'spearman_low_{suffix}'] = spearman_low
                            data_score[f'spearman_high_{suffix}'] = spearman_high
                            spearman_global = float(spearman_line.split('global = ')[-1].strip())
                            data_score[f'spearman_global_{suffix}'] = spearman_global
                            data_score[f'spearman_score_{suffix}'] = int(spearman_low >= 0.5) + int(spearman_high >= 0.5)
                            break

                    # boundary robustness sub-metric 4 (degradation ratio) is scored in
                    # get_verify_scores() instead of here - degradation_ratio_low/high_{suffix}
                    # are computed from VERIFY's own sorted-CV data, which calc_score() only
                    # parses AFTER this function returns, so they aren't available yet at this
                    # point (see get_verify_scores() for the actual score computation)

                    # boundary robustness sub-metric 5: applicability domain (leverage) -
                    # boundary robustness in X-descriptor space rather than y-space, logged by
                    # applicability_domain_plot() in utils.py. RMSE-only scoring (no rank
                    # penalty here, unlike items 1/2/4), since this is a single 80/20 split
                    # rather than a small ~7-point CV fold
                    data_score[f'applicability_domain_score_{suffix}'] = 0
                    for j in range(i,i+42):
                        if 'Applicability domain (leverage)' in dat_predict[j]:
                            ad_line = dat_predict[j]
                            rmse_ad = float(ad_line.split('RMSE = ')[-1].split(',')[0].strip())
                            scaled_rmse_ad = round((rmse_ad/data_score[f'y_range_{suffix}'])*100,2)
                            data_score[f'scaled_rmse_ad_{suffix}'] = scaled_rmse_ad
                            data_score[f'applicability_domain_score_{suffix}'] = score_rmse_mcc(pred_type,scaled_rmse_ad)
                            break

                elif pred_type == 'clas':  # Process classification: using MCC extracted from CV and Test results
                    # Extract MCC from the CV line (generic match, since the fold count is a
                    # user-settable option via --kfold and isn't always 5)
                    if '-fold CV :' in dat_predict[i+5]:
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

            # Interpolation item 6 - three facets of stability under resampling, each scored
            # 0-2 with its own threshold, then averaged and rounded into the final item score
            # (facet (a) unchanged from before; (b)/(c) new, logged by print_predict() in
            # predict_utils.py right after (a))
            if pred_type == 'reg':
                # (a) SD of the repeated-CV predictions in the test set
                if '-  Average SD in test set' in line:
                    cv_sd = float(line.split()[-1])
                    cv_4sd = 4*cv_sd
                    y_range_covered = cv_4sd/data_score[f'y_range_{suffix}']

                    cv_sd_score_a = 0
                    if y_range_covered <= 0.25:
                        cv_sd_score_a += 2
                    elif y_range_covered <= 0.50:
                        cv_sd_score_a += 1

                    data_score[f"cv_4sd_{suffix}"] = cv_4sd
                    data_score[f"cv_range_cov_{suffix}"] = y_range_covered
                    data_score[f'cv_sd_score_a_{suffix}'] = cv_sd_score_a

                # (b) SD of the out-of-fold CV predictions themselves (train+validation) - same
                # 25%/50% thresholds as (a), since both use the "4*SD as % of y-range"
                # convention, just on a different set of predictions
                if '-  Average SD in train+validation (out-of-fold)' in line:
                    cv_sd_train = float(line.split()[-1])
                    cv_4sd_train = 4*cv_sd_train
                    y_range_covered_train = cv_4sd_train/data_score[f'y_range_{suffix}']

                    cv_sd_score_b = 0
                    if y_range_covered_train <= 0.25:
                        cv_sd_score_b += 2
                    elif y_range_covered_train <= 0.50:
                        cv_sd_score_b += 1

                    data_score[f"cv_4sd_train_{suffix}"] = cv_4sd_train
                    data_score[f"cv_range_cov_train_{suffix}"] = y_range_covered_train
                    data_score[f'cv_sd_score_b_{suffix}'] = cv_sd_score_b

                # (c) how much the AGGREGATE RMSE itself varies from repeat to repeat (a
                # dataset-wide view of the same "how much does this depend on the random
                # split" question (a)/(b) ask per-point) - always the last of the three lines
                # printed, so (a)/(b) are already available here to combine into the final score
                if '-  RMSE coefficient of variation (10 repeats)' in line:
                    rmse_cv_pct = float(line.split()[-1])

                    cv_sd_score_c = 0
                    if rmse_cv_pct <= 15:
                        cv_sd_score_c += 2
                    elif rmse_cv_pct <= 25:
                        cv_sd_score_c += 1

                    data_score[f'rmse_cv_pct_{suffix}'] = rmse_cv_pct
                    data_score[f'cv_sd_score_c_{suffix}'] = cv_sd_score_c

                    facet_scores = [data_score.get(f'cv_sd_score_a_{suffix}',0),
                                     data_score.get(f'cv_sd_score_b_{suffix}',0),
                                     cv_sd_score_c]
                    data_score[f'cv_sd_score_{suffix}'] = round(sum(facet_scores)/3)

            elif pred_type == 'clas':
                # Interpolation item 6 for classification: same "how much does this depend
                # on the random split" question as the regression facets (a)/(b)/(c), all
                # expressed directly as percentages here (no y-range scaling needed for a
                # discrete label). (a)/(b) use the disagreement rate between individual CV
                # repeats and the already majority-voted class (logged by print_predict() in
                # predict_utils.py); (c) uses the coefficient of variation of the per-repeat
                # MCC instead of the per-repeat RMSE. Reuses the same 15%/25% thresholds as
                # regression facet (c), since all three are now "percent deviation" metrics -
                # a first-pass choice mirroring the existing scale, not independently
                # benchmarked the way the rest of the score was (see score.rst note)
                if '-  Average agreement in test set' in line:
                    disagreement_a = 100 - float(line.split()[-1])
                    cv_sd_score_a = 0
                    if disagreement_a <= 15:
                        cv_sd_score_a += 2
                    elif disagreement_a <= 25:
                        cv_sd_score_a += 1
                    data_score[f'clas_disagree_a_{suffix}'] = round(disagreement_a,1)
                    data_score[f'cv_sd_score_a_{suffix}'] = cv_sd_score_a

                if '-  Average agreement in train+validation (out-of-fold)' in line:
                    disagreement_b = 100 - float(line.split()[-1])
                    cv_sd_score_b = 0
                    if disagreement_b <= 15:
                        cv_sd_score_b += 2
                    elif disagreement_b <= 25:
                        cv_sd_score_b += 1
                    data_score[f'clas_disagree_b_{suffix}'] = round(disagreement_b,1)
                    data_score[f'cv_sd_score_b_{suffix}'] = cv_sd_score_b

                if '-  MCC coefficient of variation (10 repeats)' in line:
                    mcc_cv_pct = float(line.split()[-1])
                    cv_sd_score_c = 0
                    if mcc_cv_pct <= 15:
                        cv_sd_score_c += 2
                    elif mcc_cv_pct <= 25:
                        cv_sd_score_c += 1

                    data_score[f'mcc_cv_pct_{suffix}'] = mcc_cv_pct
                    data_score[f'cv_sd_score_c_{suffix}'] = cv_sd_score_c

                    facet_scores = [data_score.get(f'cv_sd_score_a_{suffix}',0),
                                     data_score.get(f'cv_sd_score_b_{suffix}',0),
                                     cv_sd_score_c]
                    data_score[f'cv_sd_score_{suffix}'] = round(sum(facet_scores)/3)

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


def calc_penalty_crossing(crossing_rate):
    '''
    Penalty for the "real range extension" crossing rate (fraction of Low/High predictions that
    actually land beyond the training range, rather than clipping toward the training
    boundary), used for the Low/High sub-metrics - same 0.5/0.2 tier shape as a Spearman-based
    penalty would use, just swapped to the crossing rate metric (see discussion with the
    user: Spearman rank gets its own dedicated item instead of penalizing here).
    '''

    penalty_crossing = 0

    if crossing_rate < 0.2:
        penalty_crossing -= 2
    elif crossing_rate < 0.5:
        penalty_crossing -= 1

    return penalty_crossing


def repro_info(modules,model_suffix=''):
    """
    Retrieves variables used in the Reproducibility section. model_suffix (e.g. '_RF') reads
    the per-model log split off by VERIFY/PREDICT under --all_models instead of the normal
    shared VERIFY_data.dat/PREDICT_data.dat - only applies to those two modules, since
    CURATE/GENERATE run once for the whole dataset and are never split per model.
    """

    version_n_date, citation, command_line = '','',''
    python_version, total_time = '',0
    dat_files = {}
    for module in modules:
        file_suffix = model_suffix if module in ('VERIFY','PREDICT') else ''
        path_file = Path(f'{os.getcwd()}/{module}/{module}{file_suffix}_data.dat')
        if os.path.exists(path_file):
            txt_file = []
            with open(path_file, 'r', encoding= 'utf-8', errors="replace") as datfile:
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
 
    try:
        import platform
        python_version = platform.python_version()
    except:
        python_version = '(version could not be determined)'
    
    return version_n_date, citation, command_line, python_version, total_time, dat_files


def make_report(report_html, HTML, pdf_name='ROBERT_report.pdf'):
    """
    Generate a css file that will be used to make the PDF file
    """

    css_files = ["report.css"]
    outfile = f"{os.getcwd()}/{pdf_name}"
    if os.path.exists(outfile):
        try:
            os.remove(outfile)
        except PermissionError:
            print(f'\nx  {pdf_name} is open! Please, close the PDF file and run ROBERT again with --report (i.e., "python -m robert --report").')
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