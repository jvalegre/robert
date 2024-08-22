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
    get_images,
    get_time,
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
    get_summary,
    get_col_transpa,
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

        # create report
        report_html,csv_name,robert_version,params_df = self.get_header(self.args.report_modules)
        report_html += self.get_data(self.args.report_modules,params_df)

        if self.args.debug_report:
            with open("report_debug.txt", "w") as debug_text:
                debug_text.write(report_html)

        # create css
        with open("report.css", "w") as cssfile:
            cssfile.write(css_content(csv_name,robert_version))

        _ = make_report(report_html,HTML)

        # Remove report.css file and _REPORT graphs created
        os.remove("report.css")
        graph_path = Path(f'{os.getcwd()}/PREDICT')
        graph_remove = [str(file_path) for file_path in graph_path.rglob('*_REPORT.png')]
        for file in graph_remove:
            os.remove(file)
        
        print('\no  ROBERT_report.pdf was created successfully in the working directory!')

     
    def get_data(self, modules, params_df):
        """
        Get information, times and images of the modules
        """

        data_lines = ""

        # AQME section
        if 'AQME' in modules:
            aqme_file = f'{os.getcwd()}/AQME/AQME_data.dat'
            if os.path.exists(aqme_file):
                aqme_time = get_time(aqme_file)
                aqme_data = f"""<i>This module performs RDKit conformer generation from SMILES, followed by the creation of 200+ molecular and atomic descriptors using RDKit, xTB and DBSTEP (saved as AQME-ROBERT_FILENAME.csv).</i>
The complete output (AQME_data.dat) and raw data are stored in the AQME folder.
{aqme_time}
"""
                data_lines += self.module_lines('AQME',aqme_data) 

        # CURATE section
        if 'CURATE' in modules:
            curate_file = f'{os.getcwd()}/CURATE/CURATE_data.dat'
            if os.path.exists(curate_file):
                # section header
                curate_time = get_time(curate_file)
                curate_data = f"""<i>This module takes care of data curation, including filters for correlated descriptors, noise, and duplicates, as well as conversion of categorical descriptors.</i>
The complete output (CURATE_data.dat) and curated database are stored in the CURATE folder.
{curate_time}
"""

                data_lines += self.module_lines('CURATE',curate_data)

                # include images
                data_lines += get_images('CURATE')

        # GENERATE section
        if 'GENERATE' in modules:
            generate_file = f'{os.getcwd()}/GENERATE/GENERATE_data.dat'
            if os.path.exists(generate_file):
                # section header
                generate_time = get_time(generate_file)
                generate_data = f"""<i>This module carries out a screening of ML models and selects the most accurate ones. It includes a comparison of multiple hyperoptimized models and training sizes.</i>
The complete output (GENERATE_data.dat) and heatmaps are stored in the GENERATE folder.
{generate_time}
"""

                data_lines += self.module_lines('GENERATE',generate_data)

                # include images
                data_lines += get_images('GENERATE')

        # VERIFY section
        if 'VERIFY' in modules:
            verify_file = f'{os.getcwd()}/VERIFY/VERIFY_data.dat'
            if os.path.exists(verify_file):
                # section header
                verify_time = get_time(verify_file)
                verify_data = f"""<i>Determination of predictive ability of models using four tests: cross-validation, y-mean (error against the mean y baseline), y-shuffle (predict with shuffled y values), and one-hot (predict using one-hot encoding instead of the X values).</i>
The complete output (VERIFY_data.dat) and donut plot are stored in the VERIFY folder.
{verify_time}
"""

                data_lines += self.module_lines('VERIFY',verify_data)

                # include images
                data_lines += get_images('VERIFY')

        # PREDICT section
        if 'PREDICT' in modules:
            predict_file = f'{os.getcwd()}/PREDICT/PREDICT_data.dat'
            if os.path.exists(predict_file):
                # section header
                predict_time = get_time(predict_file)
                predict_data = f"""<i>This module predicts and plots the results of training and validation sets from GENERATE, as well as from external test sets (if any). Feature importances from SHAP and PFI, and outlier analysis are also represented.</i>
The complete output (PREDICT_data.dat) and heatmaps are stored in the PREDICT folder.
{predict_time}
"""

                data_lines += self.module_lines('PREDICT',predict_data)

                # include images and summary for No PFI and PFI models
                data_lines += get_images('PREDICT',file=predict_file,pred_type=params_df['type'][0].lower())
                data_lines +=f'<hr style="margin-top: 15px;">'

        return data_lines


    def get_header(self,modules):
        """
        Retrieves the header for the HTML string
        """

        # Reproducibility section
        citation_dat, repro_dat, dat_files, csv_name, robert_version = self.get_repro(modules)

        # Transparency section
        transpa_dat,params_df = self.get_transparency()

        # ROBERT score section
        score_dat = self.print_score_section(dat_files,params_df['type'][0].lower())

        # abbreviation section
        abbrev_dat = self.get_abbrev()

        # combines the top image with the other sections of the header
        header_lines = f"""
            <h1 style="text-align: center; margin-bottom: 0.5em;">
                <img src="file:///{self.args.path_icons}/Robert_logo.jpg" alt="" style="display: block; margin-left: auto; margin-right: auto; width: 50%; margin-top: -18px;" />
                <span style="font-weight:bold;"></span>
            </h1>
            {citation_dat}
            {score_dat}
            {repro_dat}
            {transpa_dat}
            {abbrev_dat}
            </p>
            """

        return header_lines,csv_name,robert_version,params_df


    def print_score_section(self,dat_files,pred_type):
        """
        Generates the ROBERT score section
        """
        
        # starts with the icon of ROBERT score
        score_dat = ''
        score_dat = self.module_lines('score',score_dat) 

        # calculates the ROBERT scores (R2 is analogous for accuracy in classification)
        robert_score_list = []
        data_score = {}
        # parts of the robert score section
        score_sections = ['score_main']
        score_sections.append('metrics_predict')
        score_sections.append('adv_flawed')
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
                    spacing_PFI = ''
                elif suffix == 'PFI':
                    spacing_PFI = '&nbsp;&nbsp;&nbsp;&nbsp;'

                if section == 'score_main':
                    # calculate score
                    robert_score,test_set,descp_warning = calc_score(dat_files,suffix,pred_type,data_score)
                    robert_score_list.append(robert_score)

                    # initial two-column ROBERT score summary
                    score_info = f"""{spacing_PFI}<img src="file:///{self.args.path_icons}/score_{robert_score}.jpg" alt="ROBERT Score" style="width: 100%; margin-top:7px; margin-bottom:-18px;">"""
                    columns_score.append(get_col_score(score_info,data_score,suffix,spacing_PFI))

                elif section == 'metrics_predict':
                    # metrics of the models
                    module_file = f'{os.getcwd()}/PREDICT/PREDICT_data.dat'
                    columns_score.append(get_summary('PREDICT',module_file,suffix,titles=False,pred_type=pred_type))

                elif section == 'adv_flawed':
                    # advanced score analysis 1, flawed models
                    columns_score.append(adv_flawed(self,suffix,data_score,spacing_PFI,test_set))

                elif section == 'adv_predict':
                    # advanced score analysis 2, predictive ability
                    columns_score.append(adv_predict(self,suffix,data_score,spacing_PFI,test_set,pred_type))

                elif section == 'adv_cv_r2':
                    # advanced score analysis 3 and 3a, predictive ability of CV
                    columns_score.append(adv_cv_r2(self,suffix,data_score,spacing_PFI,pred_type))

                elif section == 'adv_cv_sd' and pred_type == 'reg':
                    # advanced score analysis 3b, SD of CV
                    columns_score.append(adv_cv_sd(self,suffix,data_score,spacing_PFI,test_set))

                elif section == 'adv_cv_diff' and pred_type == 'clas':
                    # advanced score analysis 3b, difference of MCC in model and CV
                    columns_score.append(adv_cv_diff(self,suffix,data_score,spacing_PFI,test_set))

                elif section == 'adv_descp':
                    # advanced score analysis 4, descriptor proportion
                    columns_score.append(adv_descp(self,suffix,data_score,spacing_PFI))

            # Combine both columns
            score_dat += combine_cols(columns_score)

            # add corresponding images
            diff_height = 25 # account for different graph sizes in reg and clas
            margin_bottom = -5 # margin for section separator
            section_separator = f'<hr style="height: 0.5px; margin-top: 15px; margin-bottom: {margin_bottom}px; background-color:LightGray">'

            if section == 'score_main':
                height = 215
                if pred_type == 'clas':
                    if test_set:
                        height += 17 # otherwise the first graph doesn't fit
                    else:
                        height += diff_height
                score_dat += self.print_img('Results',-5,height,'PREDICT',pred_type,test_set=test_set,diff_names=True)

            # add separator line between main and advanced analysis
            elif section == 'metrics_predict':
                score_dat += f'<hr style="height: 0.5px; margin-top: -2px; margin-bottom: 8px; background-color:LightGray">'

            elif section == 'adv_flawed':
                height = 238
                if pred_type == 'clas':
                    height -= 15
                score_dat += self.print_img('VERIFY_tests',13,height,'VERIFY',pred_type)
                # page break to second page
                score_dat += f"""<p style="page-break-after: always;"></p>"""
                score_dat += f'<hr style="height: 0.5px; margin-bottom: 13px; background-color:LightGray">'

            elif section == 'adv_predict':
                score_dat += section_separator

            elif section == 'adv_cv_r2':
                height = 215
                if pred_type == 'clas':
                    height += diff_height
                score_dat += self.print_img('CV_train_valid_predict',10,height,'VERIFY',pred_type)

            elif section == 'adv_cv_sd' and pred_type == 'reg':
                score_dat += self.print_img('CV_variability',10,215,'PREDICT',pred_type)
                score_dat += section_separator

            elif section == 'adv_cv_diff' and pred_type == 'clas':
                score_dat += section_separator

            elif section == 'adv_descp':
                score_dat += '<hr style="margin-top: 20px;">'
        
        score_dat += f"""<p style="page-break-after: always;"></p>"""

        # get some tips
        score_dat += f"""<hr style="margin-top: 10px; margin-bottom: 35px">
<p style="text-align: justify; margin-top: -8px; margin-bottom: -2px;"><u>Some tips to improve the score</u></p>"""
        
        n_scoring = 1
        style_line = '<p style="text-align: justify;">'
        reduced_line = '<p style="text-align: justify; margin-top: -5px;">' # reduces line separation separation
        last_line = '<p style="text-align: justify; margin-top: -5px; margin-bottom: 30px;">'
        if max(robert_score_list) >= 9:
            score_dat += f'{style_line}&#10004;&nbsp;&nbsp; A ROBERT score of 9 or 10 suggests that the predictive ability of your model is strong, congratulations!</p>'
            n_scoring += 1
            style_line = reduced_line
        else:
            datapoints = int(data_score[f'points_descp_ratio_{suffix}'].split(':')[0])
            if datapoints <= 50:
                score_dat += f'{style_line}&#9888;&nbsp; The model uses only {datapoints} datapoints, adding meaningful datapoints might help to improve the model.</p>'
                n_scoring += 1
                style_line = reduced_line
            if descp_warning > 0:
                score_dat += f'{style_line}&#9888;&nbsp; Adding meaningful descriptors or replacing/deleting the least useful descriptors used might help. Feature importances are gathered in the SHAP and PFI sections of the /PREDICT/PREDICT_data.dat file.</p>'
                n_scoring += 1
                style_line = reduced_line
            else:
                score_dat += f'{style_line}&#9888;&nbsp; Replacing or deleting the least useful descriptors used might help to improve the model. Feature importances are gathered in the SHAP and PFI sections of the /PREDICT/PREDICT_data.dat file.</p>'
                n_scoring += 1
                style_line = reduced_line

        # how to predict new values
        score_dat += f"""
        <br><p style="text-align: justify; margin-top: -3px; margin-bottom: -3px;"><u>How to predict new values with these models?</u></p>
<p style="text-align: justify;">1. Create a CSV database with the new points, including the necessary descriptors.</p>
{reduced_line}2. Place the CSV file in the parent folder (i.e., where the module folders were created)</p>
{reduced_line}3. Run the PREDICT module as 'python -m robert --predict --csv_test FILENAME.csv'.</p>
{last_line}4. The predictions will be shown at the end of the resulting PDF report and will be stored in the last column of two CSV files called MODEL_SIZE_test(_No)_PFI.csv, which are in the PREDICT folder.</p>"""

        return score_dat


    def get_repro(self,modules):
        """
        Generates the data printed in the Reproducibility section
        """
        
        version_n_date, citation, command_line, python_version, intelex_version, total_time, dat_files = repro_info(modules)
        robert_version = version_n_date.split()[2]

        if self.args.csv_name == '' or self.args.csv_test == '':
            self = get_csv_names(self,command_line)

        repro_dat,citation_dat = '',''
        
        # version, date and citation
        citation_dat += f"""<p style="text-align: justify; margin-top: -15px;"><br>{version_n_date}</p>
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
        repro_dat += f"""{reduced_line}{space}- CSV database ({self.args.csv_name})</p>"""
        if self.args.csv_test != '':
            repro_dat += f"""{reduced_line}{space}- External test set ({self.args.csv_test})</p>"""

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
        repro_dat +=f'<hr style="margin-top: 25px;">'

        repro_dat += f"""<p style="page-break-after: always;"></p>"""

        repro_dat = self.module_lines('repro',repro_dat) 

        return citation_dat, repro_dat, dat_files, self.args.csv_name, robert_version


    def get_transparency(self):
        """
        Generates the data printed in the Transparency section
        """

        transpa_dat = ''
        titles_line = f'<p style="text-align: justify; margin-top: -12px; margin-bottom: 3px">' # reduces line separation separation

        # add params of the models
        transpa_dat += f"""{titles_line}<br><strong>1. Parameters of the scikit-learn models (same keywords as used in scikit-learn):</strong></p>"""
        
        model_dat, params_df = self.transpa_model_misc('model_section')
        transpa_dat += model_dat

        # add misc params
        transpa_dat += f"""<p style="text-align: justify; margin-top: -95px; margin-bottom: 3px;"><br><strong>2. ROBERT options for data split (KN or RND), predict type (REG or CLAS) and hyperopt error (RMSE, etc.):</strong></p>"""
        
        section_dat, params_df = self.transpa_model_misc('misc_section')
        transpa_dat += section_dat

        transpa_dat = self.module_lines('transpa',transpa_dat) 


        return transpa_dat,params_df


    def transpa_model_misc(self,section):
        """
        Collects the data for model parameters and misc options in the Reproducibility section
        """

        columns_repro = []
        for suffix in ['No_PFI','PFI']:
            # set the parameters for each ML model
            params_dir = f'{self.args.params_dir}/{suffix}'
            files_param = glob.glob(f'{params_dir}/*.csv')
            for file_param in files_param:
                if '_db' not in file_param:
                    params_df = pd.read_csv(file_param, encoding='utf-8')
            params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

            columns_repro.append(get_col_transpa(params_dict,suffix,section))

        section_dat = combine_cols(columns_repro)

        section_dat += '<p style="text-align: justify; margin-top: -70px;">'

        return section_dat,params_df


    def get_abbrev(self):
        """
        Generates the Abbreviations section
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


    def module_lines(self,module,module_data):
        """
        Returns the line with icon and module for section titles
        """
        
        if module == 'repro':
            module_name = 'REPRODUCIBILITY'
            section_explain = f'<i style="text-align: justify;">This section provides all the instructions to reproduce the results presented.</i>'
        elif module == 'transpa':
            module_name = 'TRANSPARENCY'
            section_explain = f'<i style="text-align: justify;">This section contains important parameters used in scikit-learn models and ROBERT.</i>'
        elif module == 'score':
            module_name = 'ROBERT SCORE'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This score is designed to evaluate the models using different metrics.</i>'
        elif module == 'abbrev':
            module_name = 'ABBREVIATIONS'
            section_explain = f'<i style="text-align: justify;">Reference section for the abbreviations used.</i>'
        else:
            module_name = module
            section_explain = ''
        if module not in ['repro','transpa']:
            module_data = format_lines(module_data)
        module_data = '<div class="aqme-content"><pre>' + module_data + '</pre></div>'
        if module == 'score':
            separator_section = '<hr><p style="margin-top:23px;"></p>'
        else:
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


    def print_img(self,file_name,margin_top,height,module,pred_type,test_set=False,diff_names=False):
        """
        Generate the string that includes the results images from PREDICT
        """
        
        module_path = Path(f'{os.getcwd()}/{module}')

        # detect test
        set_types = ['train','valid']
        if test_set:
            set_types.append('test')

        module_path = Path(f'{os.getcwd()}/{module}')

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
        
        # define widths of the graphs
        width = 91
        pair_list = f'<p style="width: {width}%; margin-bottom: -2px;  margin-top: {margin_top}px"><img src="file:///{results_images[0]}" style="margin: 0; width: 270px; height: {height}px; object-fit: cover; object-position: 0 100%;"/>'
        pair_list += f'{("&nbsp;")*17}'
        pair_list += f'<img src="file:///{results_images[1]}" style="margin: 0; width: 270px; height: {height}px; object-fit: cover; object-position: 0 100%;"/></p>'

        html_png = f'{pair_list}'

        return html_png    
