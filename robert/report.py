"""
Parameters
----------

    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    report_modules : list of str, default=['CURATE','GENERATE','VERIFY','PREDICT']
        List of the modules to include in the report.
    debug_report : bool, default=False
        Debug mode using during the pytests of report.py

"""
#####################################################.
#        This file stores the REPORT class          #
#    used for generating the final PDF report       #
#####################################################.

import os
import sys
import glob
import json
import platform
import pandas as pd
from pathlib import Path
from robert.utils import (load_variables,
    pd_to_dict,
)
from robert.report_utils import (
    get_csv_names,
    get_col_score,
    calc_score,
    adv_flawed,
    adv_predict,
    adv_cv_r2,
    adv_cv_sd,
    adv_cv_diff,
    adv_descp,
    get_col_text,
    repro_info,
    make_report,
    css_content,
    format_lines,
    combine_cols,
    revert_list,
    get_metrics,
    get_col_transpa,
    get_spacing_col,
    get_outliers,
    detect_predictions,
    get_csv_metrics,
    get_csv_pred
)

class report:
    """
    Class containing all the functions from the REPORT module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the REPORT module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):
        # check if there is a problem with weasyprint (required for this module)
        try:
            from weasyprint import HTML
        except (OSError, ModuleNotFoundError):
            print(f"\n  x The REPORT module requires weasyprint but this module is missing, the PDF with the summary of the results has not been created. Try installing ROBERT with 'conda install -c conda-forge robert'")
            sys.exit()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "report")

        eval_only = False
        # if EVALUATE is activated, no PFI models are generated
        path_eval = Path(f'{os.getcwd()}/EVALUATE/EVALUATE_data.dat')
        if os.path.exists(path_eval):
            eval_only = True

        # get spacing between No PFI and PFI columns
        spacing_PFI = f'{("&nbsp;")*4}'

        # Reproducibility section (these functions only gather information, the sections
        # will be print later in the report)
        citation_dat, repro_dat, dat_files, csv_name, robert_version = self.get_repro(eval_only)

        # Transparency section 
        transpa_dat,params_df = self.get_transparency(spacing_PFI)
        pred_type = params_df['type'][0].lower()

        # print header
        report_html = self.print_header(citation_dat)

        # print ROBERT score section
        score_dat,data_score,test_set = self.print_score(dat_files,pred_type,eval_only,spacing_PFI)
        report_html += score_dat

        # print warnings in ROBERT score section
        warnings_dat,warnings_dict = self.print_warnings(pred_type,eval_only,data_score)
        report_html += warnings_dat

        # print advanced score analysis
        report_html += self.print_adv_anal(pred_type,eval_only,spacing_PFI,data_score,test_set)

        # print y distribution
        report_html += self.print_y_distrib(pred_type,eval_only,spacing_PFI,warnings_dict)

        # print feature importances
        report_html += self.print_features(warnings_dict,eval_only,spacing_PFI)

        # print outlier analysis
        report_html += self.print_outliers(pred_type,eval_only,spacing_PFI)

        # print model screening
        report_html += self.print_generate(pred_type,eval_only)

        # print reproducibility section
        report_html += repro_dat

        # print transparency section
        report_html += transpa_dat

        # print abbreviation section
        report_html += self.get_abbrev()

        # print new predictions
        report_html += self.print_predictions(pred_type,eval_only,spacing_PFI)

        # print miscellaneous section
        report_html += self.print_misc()

        if self.args.debug_report:
            with open("report_debug.txt", "w") as debug_text:
                debug_text.write(report_html)

        # create css
        with open("report.css", "w") as cssfile:
            cssfile.write(css_content(csv_name,robert_version))

        _ = make_report(report_html,HTML)

        # Remove report.css file
        os.remove("report.css")
        
        print('\no  ROBERT_report.pdf was created successfully in the working directory!')


    def print_header(self,citation_dat):
        """
        Retrieves the header for the HTML string
        """

        # combines the top image with the other sections of the header
        header_lines = f"""
            <h1 style="text-align: center; margin-bottom: 0.5em;">
                <img src="file:///{self.args.path_icons}/Robert_logo.jpg" alt="" style="display: block; margin-left: auto; margin-right: auto; width: 50%; margin-top: -12px;" />
                <span style="font-weight:bold;"></span>
            </h1>
            {citation_dat}
            """

        return header_lines


    def print_score(self,dat_files,pred_type,eval_only,spacing_PFI):
        """
        Generates the ROBERT score section
        """
        
        # starts with the icon of ROBERT score
        score_dat = ''
        score_dat = self.module_lines('score',score_dat) 

        # calculates the ROBERT scores (R2 is analogous for accuracy in classification)
        data_score = {}

        columns_score,columns_summary = [],[]
        # get two columns to combine and print
        for suffix in ['No PFI','PFI']:
            spacing = get_spacing_col(suffix,spacing_PFI)

            if eval_only and suffix == 'PFI':
                columns_score.append('')
            else:
                # calculate score
                data_score,test_set = calc_score(dat_files,suffix,pred_type,data_score)

                # initial two-column ROBERT score summary
                score_info = f"""{spacing}<img src="file:///{self.args.path_icons}/score_{data_score[f'robert_score_{suffix}']}.jpg" style="width: 330px; margin-top:7px; margin-bottom:-18px;"></p>"""
                columns_score.append(get_col_score(score_info,data_score,suffix,spacing,eval_only))

        # Combine both columns
        score_dat += combine_cols(columns_score)
        
        # add corresponding images
        diff_height = 25 # account for different graph sizes in reg and clas

        height = 221
        if pred_type == 'clas':
            if test_set:
                height += 17 # otherwise the first graph doesn't fit
            else:
                height += diff_height
        score_dat += self.print_img('Results',-5,height,'PREDICT',pred_type,eval_only,test_set=test_set,diff_names=True)

        for suffix in ['No PFI','PFI']:
            spacing = get_spacing_col(suffix,spacing_PFI)

            if eval_only and suffix == 'PFI':
                columns_summary.append('')
            else:
                # metrics of the models
                module_file = f'{os.getcwd()}/PREDICT/PREDICT_data.dat'
                columns_summary.append(get_metrics(module_file,suffix,spacing))

        # Combine both columns
        score_dat += combine_cols(columns_summary)

        return score_dat,data_score,test_set


    def print_warnings(self,pred_type,eval_only,data_score):
        """
        Generates the warning boxes in the ROBERT score section
        """

        # load spacing, colors, and line and table formats
        space,color_dict,style_lines,warnings_dat = self.get_warning_params()

        # gather the lines from PREDICT where the potential warnings are print
        warnings_dict = self.get_warning_lines(pred_type)

        columns_warnings = []
        # get two columns to combine and print
        warnings_dict['severe_warnings_No PFI'], warnings_dict['severe_warnings_PFI'] = [],[]
        warnings_dict['moderate_warnings_No PFI'], warnings_dict['moderate_warnings_PFI'] = [],[]
        for suffix in ['No PFI','PFI']:

            if eval_only and suffix == 'PFI':
                columns_warnings.append('')
            else:
                if suffix == 'No PFI':
                    margin_left = 0
                else:
                    margin_left = 29

                # analyze and append warnings
                warnings_dict = self.analyze_warnings(data_score,suffix,warnings_dict,pred_type)

                # add table in the corresponding column
                warning_print = f'''
                <table style="width:91%; margin-left: {margin_left}px; margin-top: 20px;">
                    <tr style="height:270px; vertical-align:top;">
                        <td>'''

                # add severe warnings
                warning_print += f'''
                <p style="margin-bottom: -10px; margin-top: 5px;"><strong>{space}Severe warnings</strong></p>'''
                if len(warnings_dict[f'severe_warnings_{suffix}']) == 0:
                    warning_print += self.print_line_warning(
                        'No severe warnings detected',
                        style_lines,color_dict['blue'],space)
                else:
                    for sev_warning in warnings_dict[f'severe_warnings_{suffix}']:
                        warning_print += self.print_line_warning(
                            sev_warning,
                            style_lines,color_dict['red'],space)

                # add moderate warnings
                warning_print += f'''
                <p style="margin-bottom: -10px; margin-top: 35px;"><strong>{space}Moderate warnings</strong></p>'''
                if len(warnings_dict[f'moderate_warnings_{suffix}']) == 0:
                    warning_print += self.print_line_warning(
                        'No moderate warnings detected',
                        style_lines,color_dict['blue'],space)
                else:
                    for mode_warning in warnings_dict[f'moderate_warnings_{suffix}']:
                        warning_print += self.print_line_warning(
                            mode_warning,
                            style_lines,color_dict['yellow'],space)

                # add overall assessment
                warning_print += self.print_assessment(space,suffix,data_score,style_lines,warnings_dict,color_dict,pred_type)
                           
                # end table
                warning_print += f'''</td>
                        </tr></table>'''
            
                columns_warnings.append(warning_print)

        # Combine both columns
        warnings_dat += combine_cols(columns_warnings)

        # page break
        warnings_dat += f"""<p style="page-break-after: always;"></p>"""

        return warnings_dat,warnings_dict


    def get_warning_params(self):
        '''
        Load spacing, colors, and line and table formats
        '''
        
        space = '&nbsp;'
        color_dict = {
            'red': '#c56666',
            'yellow': '#c5c57d',
            'blue': '#9ba5e3'
        }
        style_lines = '<p style="margin-bottom: -10px;">'

        # table style
        warnings_dat = '''<style>
            th, td {
            border:0.5px solid Gray;
            padding-top: 4px;
            padding-left: 4px;
            text-align: justify;
            }
            </style>
            '''

        return space,color_dict,style_lines,warnings_dat

    def analyze_warnings(self,data_score,suffix,warnings_dict,pred_type):
        '''
        Analyze and append warnings
        '''
        
        # tests from flawed models
        if data_score[f'flawed_mod_score_{suffix}'] < 3:
            if data_score[f'failed_tests_{suffix}'] > 0:
                warnings_dict[f'severe_warnings_{suffix}'].append('Failing required tests (Section B.1)')
            else:
                warnings_dict[f'moderate_warnings_{suffix}'].append('Some tests are unclear (Section B.1)')

        # variation in CV
        if pred_type == 'reg':
            if data_score[f'cv_sd_score_{suffix}'] == 0:
                warnings_dict[f'moderate_warnings_{suffix}'].append('Imprecise predictions (Section B.3b)')
        elif pred_type == 'clas':
            if data_score[f'r2_diff_score_{suffix}'] == 0:
                warnings_dict[f'moderate_warnings_{suffix}'].append('Imprecise predictions (Section B.3b)')

        # y distribution
        if 'WARNING! Your data is not uniform' in warnings_dict[f'y_dist_info_{suffix}']:
            if pred_type == 'reg':
                warnings_dict[f'moderate_warnings_{suffix}'].append('Uneven y distribution (Section C)')
            elif pred_type == 'clas': # it's severe in clasification
                warnings_dict[f'severe_warnings_{suffix}'].append('Very uneven class distribution (Section C)')
        elif 'WARNING! Your data is slightly not uniform' in warnings_dict[f'y_dist_info_{suffix}']:
            if pred_type == 'reg':
                warnings_dict[f'moderate_warnings_{suffix}'].append('Slightly uneven y distribution (Section C)')
            elif pred_type == 'clas':
                warnings_dict[f'moderate_warnings_{suffix}'].append('Uneven class distribution (Section C)')

        # feature correlation
        if 'WARNING! High correlations' in warnings_dict[f'pearson_info_{suffix}']:
            warnings_dict[f'moderate_warnings_{suffix}'].append('Highly correlated features (Section D)')
        elif 'WARNING! Noticeable correlations' in warnings_dict[f'pearson_info_{suffix}']:
            warnings_dict[f'moderate_warnings_{suffix}'].append('Moderately correlated features (Section D)')

        # outliers (threshold is set above 6.5 SD, around 99.9 CI)
        if pred_type == 'reg':
            if warnings_dict[f'max_sd_{suffix}'] > 6.5:
                warnings_dict[f'moderate_warnings_{suffix}'].append('Potential "faulty" outliers (Section E)')

        return warnings_dict


    def get_warning_lines(self,pred_type):
        '''
        Gather the lines from PREDICT where the potential warnings are print
        '''
        
        warnings_dict = {}

        # get lines with warnings from PREDICT
        file_pred = f'{os.getcwd()}/PREDICT/PREDICT_data.dat'
        with open(file_pred, 'r') as datfile:
            lines = datfile.readlines()
            pfi_section_pearson = False # to get both No PFI and PFI information
            pfi_section_y_dist = False
            pfi_section_outlier = False
            for i,line in enumerate(lines):
                if 'Ideally, variables should show low' in line and not pfi_section_pearson:
                    warnings_dict['pearson_info_No PFI'] = lines[i+1][6:]
                    pfi_section_pearson = True # the next line found will correspond to the PFI section
                elif 'Ideally, variables should show low' in line and pfi_section_pearson:
                    warnings_dict['pearson_info_PFI'] = lines[i+1][6:]
                if 'Ideally, the number of datapoints in' in line and not pfi_section_y_dist:
                    warnings_dict['y_dist_info_No PFI'] = lines[i+2][6:]
                    pfi_section_y_dist = True
                elif 'Ideally, the number of datapoints in' in line and pfi_section_y_dist:
                    warnings_dict['y_dist_info_PFI'] = lines[i+2][6:]
                if pred_type == 'reg':
                    if 'Outliers plot saved' in line and not pfi_section_outlier:
                        max_SD = 0
                        for j in range(i,len(lines)):
                            if '-------' in lines[j]:
                                break
                            elif 'SDs' in lines[j]:
                                sd_line = float(lines[j].split()[2][1:])
                                if sd_line > max_SD:
                                    max_SD = sd_line
                        warnings_dict['max_sd_No PFI'] = max_SD
                        pfi_section_outlier = True
                    elif 'Outliers plot saved' in line and pfi_section_outlier:
                        max_SD = 0
                        for j in range(i,len(lines)):
                            if '-------' in lines[j]:
                                break
                            elif 'SDs' in lines[j]:
                                sd_line = float(lines[j].split()[2][1:])
                                if sd_line > max_SD:
                                    max_SD = sd_line
                        warnings_dict['max_sd_PFI'] = max_SD

            return warnings_dict


    def print_line_warning(self,message,style_lines,color,space):
        '''
        Add line with warning
        '''
        
        return f'''
        {style_lines}<span style='font-size:15px; color: {color};'>{space}&#9673;</span>
        {space}{message}</p>'''


    def print_assessment(self,space,suffix,data_score,style_lines,warnings_dict,color_dict,pred_type):
        '''
        Add overall assessment to the ROBERT score section
        '''
        
        assessment_print = f'''
<p style="margin-bottom: -10px; margin-top: 35px;"><strong>{space}Overall assessment</strong></p>'''

        if len(warnings_dict[f'severe_warnings_{suffix}']) > 0 or data_score[f'robert_score_{suffix}'] < 7:
            assessment_print += self.print_line_warning(
                'The model is unreliable',
                style_lines,color_dict['red'],space)

        elif data_score[f'robert_score_{suffix}'] in [9,10]:
            if pred_type == 'reg' and len(warnings_dict[f'moderate_warnings_{suffix}']) >= 3:
                assessment_print += self.print_line_warning(
                    'Reliable model, but examine warnings',
                    style_lines,color_dict['yellow'],space)
            elif pred_type == 'clas' and len(warnings_dict[f'moderate_warnings_{suffix}']) >= 2:
                assessment_print += self.print_line_warning(
                    'Reliable model, but examine warnings',
                    style_lines,color_dict['yellow'],space)
            else:
                assessment_print += self.print_line_warning(
                    f'The model seems reliable',
                    style_lines,color_dict['blue'],space)

        elif data_score[f'robert_score_{suffix}'] in [7,8]:
            assessment_print += self.print_line_warning(
                'Decent model, but it has limitations',
                style_lines,color_dict['yellow'],space)
        
        return assessment_print


    def print_adv_anal(self,pred_type,eval_only,spacing_PFI,data_score,test_set):
        """
        Generates the advanced score analysis section
        """

        adv_score_dat = ''

        adv_score_dat += self.module_lines('adv_anal',adv_score_dat)

        # parts of the robert score section
        score_sections = ['adv_flawed']
        score_sections.append('adv_flawed_extra')
        score_sections.append('adv_predict')
        score_sections.append('adv_cv_r2')
        score_sections.append('adv_cv_sd')
        score_sections.append('adv_cv_diff')
        score_sections.append('adv_descp')

        for section in score_sections:
            columns_score = []
            # get two columns to combine and print
            for suffix in ['No PFI','PFI']:

                # add spacing of PFI column
                if suffix == 'No PFI':
                    spacing = ''
                elif suffix == 'PFI':
                    spacing = spacing_PFI

                if eval_only and suffix == 'PFI':
                    columns_score.append('')
                else:

                    if section == 'adv_flawed':
                        # advanced score analysis 1, flawed models
                        columns_score.append(adv_flawed(self,suffix,data_score,spacing*2))

                    elif section == 'adv_predict':
                        # advanced score analysis 2, predictive ability
                        columns_score.append(adv_predict(self,suffix,data_score,spacing*2,test_set,pred_type))

                    elif section == 'adv_cv_r2':
                        # advanced score analysis 3 and 3a, predictive ability of CV
                        columns_score.append(adv_cv_r2(self,suffix,data_score,spacing*2,pred_type))

                    elif section == 'adv_cv_sd' and pred_type == 'reg':
                        # advanced score analysis 3b, SD of CV
                        columns_score.append(adv_cv_sd(self,suffix,data_score,spacing*2,test_set))

                    elif section == 'adv_cv_diff' and pred_type == 'clas':
                        # advanced score analysis 3b, difference of MCC in model and CV
                        columns_score.append(adv_cv_diff(self,suffix,data_score,spacing*2,test_set))

                    elif section == 'adv_descp':
                        # advanced score analysis 4, descriptor proportion
                        columns_score.append(adv_descp(self,suffix,data_score,spacing*2))

            # Combine both columns
            adv_score_dat += combine_cols(columns_score)

            # add corresponding images
            diff_height = 25 # account for different graph sizes in reg and clas
            section_separator = f'<hr style="height: 0.5px; margin-top: 22px; margin-bottom: 0px; background-color:LightGray">'

            if section == 'adv_flawed':
                height = 238
                if pred_type == 'clas':
                    height -= 15
                adv_score_dat += self.print_img('VERIFY_tests',13,height,'VERIFY',pred_type,eval_only)
                # page break to second page
                adv_score_dat += '<hr style="height: 0.5px; margin-top: 30px; background-color:LightGray">'

            elif section == 'adv_predict':
                adv_score_dat += section_separator

            elif section == 'adv_cv_r2':
                height = 221
                if pred_type == 'clas':
                    height += diff_height
                adv_score_dat += self.print_img('CV_train_valid_predict',10,height,'VERIFY',pred_type,eval_only)

            elif section == 'adv_cv_sd' and pred_type == 'reg':
                adv_score_dat += self.print_img('CV_variability',10,221,'PREDICT',pred_type,eval_only)
                adv_score_dat += section_separator

            elif section == 'adv_cv_diff' and pred_type == 'clas':
                adv_score_dat += section_separator

            elif section == 'adv_descp':
                adv_score_dat += '<p style="margin-bottom: 50px;"></p>'

        return adv_score_dat


    def print_misc(self):
        """
        Generates the miscellaneous section
        """

        misc_dat = ''
        misc_dat += self.module_lines('misc',misc_dat)

        # get some tips
        style_line = '<p style="text-align: justify; margin-top: -5px;">' # reduces line separation separation
        misc_dat += f"""<p style="text-align: justify; margin-top: -3px; margin-bottom: -3px;"><u>Some general tips to improve the score</u></p>"""
        misc_dat += f'<p style="text-align: justify;">1. Adding meaningful datapoints might help to improve the model. Also, using a uniform population of datapoints across the whole range of y values usually helps to obtain reliable predictions across the whole range. More information about the range of y values used is available in Section C.</p>'
        misc_dat += f'{style_line}2. Adding meaningful descriptors or replacing/deleting the least useful descriptors used might help. Feature importances are gathered in Section D.</p>'


        # how to predict new values
        misc_dat += f"""
        <br><p style="text-align: justify; margin-top: -3px; margin-bottom: -3px;"><u>How to predict new values with these models?</u></p>
<p style="text-align: justify;">1. Create a CSV database with the new points, including the necessary descriptors.</p>
{style_line}2. Place the CSV file in the parent folder (i.e., where the module folders were created)</p>
{style_line}3. Run the PREDICT module as 'python -m robert --predict --csv_test FILENAME.csv'.</p>
{style_line}4. The predictions will be shown at the end of the resulting PDF report and will be stored in the last column of two CSV files called MODEL_SIZE_test(_No)_PFI.csv, which are in the PREDICT folder.</p>"""

        # add separator line
        misc_dat += '<hr style="margin-top: 20px;">'

        return misc_dat
    

    def print_outliers(self,pred_type,eval_only,spacing_PFI):
        """
        Generates the outliers section
        """
        
        # starts with the icon of outliers
        outlier_dat = ''
        outlier_dat = self.module_lines('outliers',outlier_dat,pred_type=pred_type) 

        if pred_type == 'reg':
            columns_outlier = []
            # get two columns to combine and print
            for suffix in ['No PFI','PFI']:
                spacing = get_spacing_col(suffix,spacing_PFI)

                if eval_only and suffix == 'PFI':
                    columns_outlier.append('')
                else:
                    # get information about outliers
                    module_file = f'{os.getcwd()}/PREDICT/PREDICT_data.dat'
                    columns_outlier.append(get_outliers(module_file,suffix,spacing))

            # Combine both columns
            outlier_dat += combine_cols(columns_outlier)
            
            # add corresponding images
            height = 217
            outlier_dat += self.print_img('Outliers',-5,height,'PREDICT',pred_type,eval_only)

        # add separator line and page break
        outlier_dat += '<hr style="margin-top: 20px;">'
        outlier_dat += f"""<p style="page-break-after: always;"></p>"""

        return outlier_dat


    def print_y_distrib(self,pred_type,eval_only,spacing_PFI,warnings_dict):
        """
        Generates the y distribution section
        """
        
        # starts with the icon of outliers
        distrib_dat = ''
        distrib_dat = self.module_lines('y_distrib',distrib_dat) 
        
        # add corresponding images
        height = 220
        distrib_dat += self.print_img('y_distribution',-5,height,'PREDICT',pred_type,eval_only)

        columns_y_distrib = []
        # get two columns to combine and print
        for suffix in ['No PFI','PFI']:
            spacing = get_spacing_col(suffix,spacing_PFI)

            if eval_only and suffix == 'PFI':
                columns_y_distrib.append('')
            else:
                # split the sentence into 1 column size and add spacing line by line
                y_distrib_sentence = format_lines(warnings_dict[f'y_dist_info_{suffix}'],max_width=55,one_column=True,spacing=spacing)

                column = f"""
                <p style='margin-top:25px; margin-bottom:-6px'><span style="font-weight:bold;">{spacing*3}y distribution analysis</span></p>
                {y_distrib_sentence}
                """
                columns_y_distrib.append(column)

        # Combine both columns
        distrib_dat += combine_cols(columns_y_distrib)

        distrib_dat += '<p style="margin-bottom: 30px;"></p>'

        # add separator line and page break
        distrib_dat += '<hr style="margin-top: 20px;">'
        distrib_dat += f"""<p style="page-break-after: always;"></p>"""

        return distrib_dat


    def print_features(self,warnings_dict,eval_only,spacing_PFI):
        """
        Generates the feature analysis section
        """
        
        # starts with the icon of feature importances
        feature_dat = ''
        feature_dat = self.module_lines('features',feature_dat) 
        
        # add corresponding images
        module_path = Path(f'{os.getcwd()}/PREDICT')
                
        shap_images = glob.glob(f'{module_path}/SHAP_*.png')
        pfi_images = glob.glob(f'{module_path}/PFI_*.png')
        pearson_images = glob.glob(f'{module_path}/Pearson_*.png')

        shap_images = revert_list(shap_images)
        pfi_images = revert_list(pfi_images)
        pearson_images = revert_list(pearson_images)

        image_pair_list = [shap_images, pfi_images, pearson_images]

        margin_top, margin_bottom = -10,30
        for _,image_pair in enumerate(image_pair_list):
            if len(image_pair) < 2 and not eval_only: # Pearson graphs aren't created when >30 descriptors
                pair_list = f'<p style="width: 91%; margin-bottom: {margin_bottom}px; margin-top: {margin_top}px">Pearson maps not created if >30 descriptors.'
                pair_list += f'{("&nbsp;")*15}'
                if len(image_pair) == 1:
                    pair_list += f'<img src="file:///{image_pair[0]}" style="margin: 0; width: 100%;"/></p>'
                elif len(image_pair) == 0:
                    pair_list += f'{("&nbsp;")*15}'
                    pair_list += f'Pearson maps not created if >30 descriptors.</p>'
            elif eval_only:
                if len(image_pair) == 1:
                    pair_list = f'<p style="width: 91%; margin-bottom: {margin_bottom}px; margin-top: {margin_top}px"><img src="file:///{image_pair[0]}" style="margin: 0; width: 100%;"/></p>'
                elif len(image_pair) == 0:
                    pair_list = f'<p style="width: 91%; margin-bottom: {margin_bottom}px;  margin-top: {margin_top}px">Pearson maps not created if >30 descriptors.</p>'
            else:
                pair_list = f'<p style="width: 91%; margin-bottom: {margin_bottom}px; margin-top: {margin_top}px"><img src="file:///{image_pair[0]}" style="margin: 0; width: 100%;"/>'
                pair_list += f'{("&nbsp;")*22}'
                pair_list += f'<img src="file:///{image_pair[1]}" style="margin: 0; width: 100%;"/></p>'
            feature_dat += pair_list

        columns_pearson = []
        # get two columns to combine and print
        for suffix in ['No PFI','PFI']:
            spacing = get_spacing_col(suffix,spacing_PFI)

            if eval_only and suffix == 'PFI':
                columns_pearson.append('')
            else:
                # split the sentence into 1 column size and add spacing line by line
                pearson_sentence = format_lines(warnings_dict[f'pearson_info_{suffix}'],max_width=55,one_column=True,spacing=spacing)

                column = f"""
                <p style='margin-top:-10px; margin-bottom:-6px'><span style="font-weight:bold;">{spacing*3}Correlation analysis</span></p>
                {pearson_sentence}
                """
                columns_pearson.append(column)

        # Combine both columns
        feature_dat += combine_cols(columns_pearson)

        # add separator line and page break
        feature_dat += '<hr style="margin-top: 10px;">'
        feature_dat += f"""<p style="page-break-after: always;"></p>"""

        return feature_dat


    def print_generate(self,pred_type,eval_only):
        """
        Generates the GENERATE hyperoptimization section
        """
        
        # starts with the icon of feature importances
        generate_dat = ''
        generate_dat = self.module_lines('generate',generate_dat,eval_only=eval_only) 
        
        # add corresponding images
        if not eval_only:
            height = 217
            generate_dat += self.print_img('Heatmap',-5,height,'GENERATE',pred_type,eval_only)

        generate_dat += '<p style="margin-bottom: 50px;"></p>'

        return generate_dat


    def get_repro(self,eval_only):
        """
        Generates the reproducibility section
        """
        
        version_n_date, citation, command_line, python_version, intelex_version, total_time, dat_files = repro_info(self.args.report_modules)
        robert_version = version_n_date.split()[2]

        if self.args.csv_name == '' or self.args.csv_test == '':
            self = get_csv_names(self,command_line)

        if eval_only and self.args.csv_train != '':
            self.args.csv_name = self.args.csv_train

        repro_dat,citation_dat = '',''
        
        # version, date and citation
        citation_dat += f"""<p style="text-align: justify; margin-top: -9px;"><br>{version_n_date}</p>
        <p style="text-align: justify;  margin-top: -10px;"><span style="font-weight:bold;">How to cite:</span> {citation}</p>"""

        aqme_workflow,aqme_updated = False,True
        crest_workflow = False
        if '--aqme' in command_line:
            original_command = command_line
            aqme_workflow = True
            command_line = command_line.replace('AQME-ROBERT_','')
            self.args.csv_name = f'{self.args.csv_name}'.replace('AQME-ROBERT_','')
            if self.args.csv_test != '':
                self.args.csv_test = f'{self.args.csv_test}'.replace('AQME-ROBERT_','')

        if '--program crest' in command_line.lower():
            crest_workflow = True

        # make the text more compact if --aqme is used (more lines are included)
        if aqme_workflow:
            first_line = f'<p style="text-align: justify; margin-bottom: 10px; margin-top: -16px;">' # reduces line separation separation
        else:
            first_line = f'<p style="text-align: justify; margin-bottom: 10px; margin-top: -8px;">' # reduces line separation separation
        reduced_line = f'<p style="text-align: justify; margin-top: -5px;">' # reduces line separation separation        
        space = ('&nbsp;')*4

        # just in case the command lines are so long
        command_line = format_lines(command_line,cmd_line=True)

        # reproducibility section, starts with the icon of reproducibility  
        repro_dat += f"""{first_line}<br><strong>1. Download these files <i>(the authors should have uploaded the files as supporting information!)</i>:</strong></p>"""
        if not eval_only:
            repro_dat += f"""{reduced_line}{space}- CSV database ({self.args.csv_name})</p>"""
            if self.args.csv_test != '':
                repro_dat += f"""{reduced_line}{space}- External test set ({self.args.csv_test})</p>"""
        else:
            repro_dat += f"""{reduced_line}{space}- Training CSV database ({self.args.csv_train})</p>"""
            repro_dat += f"""{reduced_line}{space}- Validation CSV database ({self.args.csv_valid})</p>"""
            if self.args.csv_test != '':
                repro_dat += f"""{reduced_line}{space}- Test CSV database ({self.args.csv_test})</p>"""  

        if aqme_workflow:
            try:
                path_aqme = Path(f'{os.getcwd()}/AQME/CSEARCH_data.dat')
                datfile = open(path_aqme, 'r', errors="replace")
                outlines = datfile.readlines()
                aqme_version = outlines[0].split()[2]
                datfile.close()
                find_aqme = True
            except:
                find_aqme = False
                aqme_version = '0.0' # dummy number
            if int(aqme_version.split('.')[0]) in [0,1] and int(aqme_version.split('.')[1]) < 6:
                aqme_updated = False
                repro_dat += f"""{reduced_line}{space}<i>Warning! This workflow might not be exactly reproducible, update to AQME v1.6.0+ (pip install aqme --upgrade)</i></p>"""
                repro_dat += f"""{reduced_line}{space}To obtain the same results, download the descriptor database (AQME-ROBERT_{self.args.csv_name}) and run:</p>"""
                repro_line = []
                original_command = original_command.replace(self.args.csv_name,f'AQME-ROBERT_{self.args.csv_name}')
                for i,keyword in enumerate(original_command.split('"')):
                    if i == 0:
                        if '--aqme' not in keyword and '--qdescp_keywords' not in keyword and '--csearch_keywords' not in keyword:
                                repro_line.append(keyword)
                        else:
                            repro_line.append('python -m robert ')
                    if i > 0:
                        if '--qdescp_keywords' not in original_command.split('"')[i-1] and '--csearch_keywords' not in original_command.split('"')[i-1]:
                            if '--aqme' not in keyword and '--qdescp_keywords' not in keyword and '--csearch_keywords' not in keyword and keyword != '\n':
                                repro_line.append(keyword) 
                repro_line = '"'.join(repro_line)
                repro_line += '"'
                if '--names ' not in repro_line:
                    repro_line += ' --names "code_name"'
                repro_line = f'{reduced_line}{space}- Run: {repro_line}'
                repro_line = format_lines(repro_line,cmd_line=True)
                repro_dat += f"""{reduced_line}{repro_line}</p>"""

        if aqme_workflow and not aqme_updated:
            # I use a very reduced line in this title because the formatted command_line comes with an extra blank line
            # (if AQME is not updated the PDF contains a reproducibility warning)
            repro_dat += f"""<p style="text-align: justify; margin-top: -44px;"><br><strong>2. Install and adjust the versions of the following Python modules:</strong></p>"""
        else:
            repro_dat += f"""{first_line}<br><strong>2. Install and adjust the versions of the following Python modules:</strong></p>"""
        repro_dat += f"""{reduced_line}{space}- Install ROBERT and its dependencies: conda install -c conda-forge robert</p>"""
        repro_dat += f"""{reduced_line}{space}- Adjust ROBERT version: pip install robert=={robert_version}</p>"""

        if intelex_version != 'not installed':
            repro_dat += f"""{reduced_line}{space}- Install scikit-learn-intelex: pip install scikit-learn-intelex=={intelex_version}</p>"""
            repro_dat += f"""{reduced_line}{space}<i>(if scikit-learn-intelex is not installed, slightly different results might be obtained)</i></p>"""
        else:
            repro_dat += f"""{reduced_line}{space}- scikit-learn-intelex: not installed</p>"""
            repro_dat += f"""{reduced_line}{space}<i>(if scikit-learn-intelex is installed, slightly different results might be obtained)</i></p>"""

        if aqme_workflow:
            if not find_aqme:
                repro_dat += f"""{reduced_line}{space}- AQME is required, but no version was found:</p>"""
            repro_dat += f"""{reduced_line}{space}- Install AQME and its dependencies: conda install -c conda-forge aqme</p>"""
            if find_aqme:
                repro_dat += f"""{reduced_line}{space}- Adjust AQME version: pip install aqme=={aqme_version}</p>"""

            try:
                path_xtb = Path(f'{os.getcwd()}/AQME/QDESCP')
                xtb_json = glob.glob(f'{path_xtb}/*.json')[0]
                f = open(xtb_json, "r")  # Opening JSON file
                data = json.loads(f.read())  # read file
                f.close()
                xtb_version = data['xtb version'].split()[0]
                find_xtb = True
            except:
                find_xtb = False
            if not find_xtb:
                repro_dat += f"""{reduced_line}{space}- xTB is required, but no version was found:</p>"""
            repro_dat += f"""{reduced_line}{space}- Install xTB: conda install -c conda-forge xtb</p>"""
            if find_xtb:
                repro_dat += f"""{reduced_line}{space}- Adjust xTB version (if possible): conda install -c conda-forge xtb={xtb_version}</p>"""

        if crest_workflow:
            try:
                import pkg_resources
                crest_version = pkg_resources.get_distribution('crest').version
                find_crest = True
            except:
                find_crest = False
            if not find_crest:
                repro_dat += f"""{reduced_line}{space}- CREST is required, but no version was found:</p>"""
            repro_dat += f"""{reduced_line}{space}- Install CREST: conda install -c conda-forge crest</p>"""
            if find_crest:
                repro_dat += f"""{reduced_line}{space}- Adjust CREST version: conda install -c conda-forge crest={crest_version})</p>"""

        character_line = ''
        if self.args.csv_test != '':
            character_line += 's'

        repro_dat += f"""{first_line}<br><strong>3. Run ROBERT using this command line in the folder with the CSV database{character_line}:</strong></p>{reduced_line}{command_line}</p>"""

        # I use a very reduced line in this title because the formatted command_line comes with an extra blank line
        if aqme_workflow:
            repro_dat += f"""<p style="text-align: justify; margin-top: -44px;"><br><strong>4. Execution time, Python version and OS:</strong></p>"""
        else:
            repro_dat += f"""<p style="text-align: justify; margin-top: -37px;"><br><strong>4. Execution time, Python version and OS:</strong></p>"""
            
        # add total execution time
        repro_dat += f"""{reduced_line}Originally run in Python {python_version} using {platform.system()} {platform.version()}</p>"""
        repro_dat += f"""{reduced_line}Total execution time: {total_time} seconds <i>(the number of processors should be specified by the user)</i></p>"""

        # add separator line and page break
        repro_dat += '<hr style="margin-top: 20px;">'
        repro_dat += f"""<p style="page-break-after: always;"></p>"""

        repro_dat = self.module_lines('repro',repro_dat) 

        return citation_dat, repro_dat, dat_files, self.args.csv_name, robert_version


    def get_transparency(self,spacing_PFI):
        """
        Generates the transparency section
        """

        transpa_dat = ''
        titles_line = f'<p style="text-align: justify; margin-top: -12px; margin-bottom: 3px">' # reduces line separation separation

        # add params of the models
        transpa_dat += f"""{titles_line}<br><strong>1. Parameters of the scikit-learn models (same keywords as used in scikit-learn):</strong></p>"""
        
        model_dat, params_df = self.transpa_model_misc('model_section',spacing_PFI)
        transpa_dat += model_dat

        # add misc params
        transpa_dat += f"""<p style="text-align: justify; margin-top: -95px; margin-bottom: 3px;"><br><strong>2. ROBERT options for data split (KN or RND), predict type (REG or CLAS) and hyperopt error (RMSE, etc.):</strong></p>"""
        
        section_dat, params_df = self.transpa_model_misc('misc_section',spacing_PFI)
        transpa_dat += section_dat

        transpa_dat = self.module_lines('transpa',transpa_dat) 


        return transpa_dat,params_df


    def transpa_model_misc(self,section,spacing_PFI):
        """
        Collects the data for model parameters and misc options in the Reproducibility section
        """

        columns_repro = []
        for suffix in ['No PFI','PFI']:
            spacing = get_spacing_col(suffix,spacing_PFI)

            # set the parameters for each ML model
            params_dir = f'{self.args.params_dir}/{"_".join(suffix.split())}'
            files_param = glob.glob(f'{params_dir}/*.csv')
            for file_param in files_param:
                if '_db' not in file_param:
                    params_df = pd.read_csv(file_param, encoding='utf-8')
            params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

            columns_repro.append(get_col_transpa(params_dict,suffix,section,spacing))

        section_dat = combine_cols(columns_repro)

        section_dat += '<p style="text-align: justify; margin-top: -70px;">'

        return section_dat,params_df


    def get_abbrev(self):
        """
        Generates the abbreviations section
        """

        # starts with the icon of abbreviation
        abbrev_dat = ''
        abbrev_dat = self.module_lines('abbrev',abbrev_dat) 

        columns_abbrev = []
        columns_abbrev.append(get_col_text('abbrev_1'))
        columns_abbrev.append(get_col_text('abbrev_2'))
        columns_abbrev.append(get_col_text('abbrev_3'))

        abbrev_dat += combine_cols(columns_abbrev)
        abbrev_dat +=f'<hr style="margin-top: 15px;">'

        abbrev_dat += f"""<p style="page-break-after: always;"></p>"""

        return abbrev_dat


    def print_predictions(self,pred_type,eval_only,spacing_PFI):
        """
        Generates the outliers section
        """
        
        # detects whether there are predictions from an external set
        module_file = f'{os.getcwd()}/PREDICT/PREDICT_data.dat'
        csv_test_exists, y_value, names, path_csv_test = detect_predictions(module_file)

        if csv_test_exists:
            pred_dat = ''
            pred_dat = self.module_lines('pred',pred_dat,pred_type=pred_type) 

            columns_metrics = []
            # add metrics
            for suffix in ['No PFI','PFI']:
                spacing = get_spacing_col(suffix,spacing_PFI)

                if eval_only and suffix == 'PFI':
                    columns_metrics.append('')
                else:
                    columns_metrics.append(get_csv_metrics(module_file,suffix,spacing))

            # Combine both columns
            pred_dat += combine_cols(columns_metrics)

            columns_pred = []
            # add predictions table
            for suffix in ['No PFI','PFI']:
                spacing = get_spacing_col(suffix,spacing_PFI)

                if eval_only and suffix == 'PFI':
                    columns_pred.append('')
                else:
                    # add metrics
                    module_file = f'{os.getcwd()}/PREDICT/PREDICT_data.dat'
                    columns_pred.append(get_csv_pred(suffix,path_csv_test,y_value,names,spacing))

            # Combine both columns
            pred_dat += combine_cols(columns_pred)

            # add corresponding images
            height = 217
            if pred_type == 'reg':
                prefix_img = 'CV_variability'
            elif pred_type == 'clas':
                prefix_img = 'Results'
                height += 17
            if len(glob.glob(f'{os.getcwd()}/PREDICT/csv_test/{prefix_img}*.png')) > 0:
                pred_dat += self.print_img(prefix_img,-5,height,'PREDICT/csv_test',pred_type,eval_only)

            # add separator line and page break
            pred_dat += '<hr style="margin-top: 20px;">'
            pred_dat += f"""<p style="page-break-after: always;"></p>"""

            return pred_dat

        else:
            return ''


    def module_lines(self,module,module_data,pred_type='reg',eval_only=False):
        """
        Returns the line with icon and module for section titles
        """
        
        if module == 'score':
            module_name = 'Section A. ROBERT Score'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This score is designed to evaluate the models using different metrics.</i>'
        elif module == 'adv_anal':
            module_name = 'Section B. Advanced Score Analysis'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section explains each component that comprises the ROBERT score.</i>'
        elif module == 'y_distrib':
            module_name = 'Section C. Distribution of y Values'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section shows the distribution of y values within the training and validation sets.</i>'
        elif module == 'features':
            module_name = 'Section D. Feature Importances'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section presents feature importances measured using the validation set.</i>'
        elif module == 'outliers':
            module_name = 'Section E. Outlier Analysis'
            if pred_type == 'clas':
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This feature is disabled in classification problems.</i>'
            else:
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section detects outliers using the standard deviation (SD) of errors from the training set.</i>'
        elif module == 'generate':
            module_name = 'Section F. Model Screening'
            if eval_only:
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">The screening of models is disabled when using the EVALUATE module.</i>'
            else:
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section compares different combinations of hyperoptimized algorithms and partition sizes.</i>'
        elif module == 'repro':
            module_name = 'Section G. Reproducibility'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section provides all the instructions to reproduce the results presented.</i>'
        elif module == 'transpa':
            module_name = 'Section H. Transparency'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section contains important parameters used in scikit-learn models and ROBERT.</i>'
        elif module == 'abbrev':
            module_name = 'Section I. Abbreviations'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">Reference section for the abbreviations used.</i>'
        elif module == 'pred':
            module_name = 'Section J. New Predictions'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">Predictions of the external test set added with the csv_test option.</i>'
        elif module == 'misc':
            module_name = 'Miscellaneous'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">General tips to improve the models and instructions to predict new values.</i>'

        if module not in ['repro','transpa','misc']:
            module_data = format_lines(module_data)
        module_data = '<div class="aqme-content"><pre>' + module_data + '</pre></div>'
        
        separator_section = '<hr><p style="margin-top:25px;"></p>'

        title_line = f"""
            {separator_section}
            <p><span style="font-weight:bold;">
                <img src="file:///{self.args.path_icons}/{module}.png" alt="" style="width:20px; height:20px; margin-right:5px;">
                {module_name}
            </span></p>{section_explain}
            {module_data}
            </p>
            """
        
        return title_line


    def print_img(self,file_name,margin_top,height,module,pred_type,eval_only,test_set=False,diff_names=False):
        """
        Generates the string that includes couples of images to print
        """
        
        module_path = Path(f'{os.getcwd()}/{module}')

        # detect test
        set_types = ['train','valid']
        if test_set:
            set_types.append('test')

        # different names for reg and clas problems, only for results images from PREDICT
        if diff_names:
            if pred_type.lower() == 'reg':
                results_images = [str(file_path) for file_path in module_path.rglob(f'{file_name}_*.png')]
            elif pred_type.lower() == 'clas':
                results_images  = []
                all_images = [str(file_path) for file_path in module_path.rglob('*.png')]
                for img in all_images:
                    if 'test' in set_types:
                        if file_name in img and '_test.png' in img:
                            results_images.append(img)
                    else:
                        if file_name in img and '_valid.png' in img:
                            results_images.append(img)
        # images with no suffixes in the names
        else:
            results_images = [str(file_path) for file_path in module_path.rglob(f'{file_name}_*.png')]

        # keep the ordering (No_PFI in the left, PFI in the right of the PDF)
        results_images = revert_list(results_images)            
        
        # add the graphs
        width = 100

        pair_list = f'<p style="width: {width}%; margin-bottom: -2px;  margin-top: {margin_top}px"><img src="file:///{results_images[0]}" style="margin: 0; width: 270px; height: {height}px; object-fit: cover; object-position: 0 100%;"/>'
        if not eval_only:
            pair_list += f'{("&nbsp;")*22}'
            pair_list += f'<img src="file:///{results_images[1]}" style="margin: 0; width: 270px; height: {height}px; object-fit: cover; object-position: 0 100%;"/></p>'

        html_png = f'{pair_list}'

        return html_png    
