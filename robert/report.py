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
    get_graph_style,
)
from robert.predict_utils import graph_reg
from robert.report_utils import (
    get_csv_names,
    get_images,
    get_time,
    get_col_score,
    get_col_text,
    get_verify_scores,
    get_predict_scores,
    repro_info,
    make_report,
    css_content,
    format_lines,
    combine_cols,
    get_y_values,
    revert_list,
    get_summary,
    get_col_transpa,
    get_pts
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
                generate_data = f"""<i>This module carries out a screening of ML models and selects the most accurate one. It includes a comparison of multiple hyperoptimized models and training sizes.</i>
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
                verify_data = f"""<i>Determination of predictive ability of models using four tests: 5-fold CV, y-mean (error against the mean y baseline), y-shuffle (predict with shuffled y values), and one-hot (predict using one-hot encoding instead of the X values).</i>
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

        return data_lines


    def get_header(self,modules):
        """
        Retrieves the header for the HTML string
        """

        # Reproducibility section
        citation_dat, repro_dat, dat_files, csv_name, csv_test, robert_version = self.get_repro(modules)

        # Transparency section
        transpa_dat,params_df = self.get_transparency()

        # ROBERT score section
        score_dat = self.get_score(dat_files,csv_test,params_df['type'][0].lower())

        # abbreviation section
        abbrev_dat = self.get_abbrev()

        # combines the top image with the other sections of the header
        header_lines = f"""
            <h1 style="text-align: center; margin-bottom: 0.5em;">
                <img src="file:///{self.args.path_icons}/Robert_logo.jpg" alt="" style="display: block; margin-left: auto; margin-right: auto; width: 50%; margin-top: -12px;" />
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


    def get_score(self,dat_files,csv_test,pred_type):
        """
        Generates the ROBERT score section
        """
        
        # starts with the icon of ROBERT score
        score_dat = ''
        score_dat = self.module_lines('score',score_dat) 

        # calculates the ROBERT scores (R2 is analogous for accuracy in classification)
        outliers_warnings, r2_verify_warnings, descp_warning = 0,0,0
        robert_score_list,columns_score = [],[]
        
        for suffix in ['No PFI','PFI']:
            data_score,test_set = get_predict_scores(dat_files['PREDICT'],suffix,pred_type)
            if data_score['outliers_score'] < 2:
                outliers_warnings += 1
            if data_score['descp_score'] < 2:
                descp_warning += 1

            data_score['verify_score'],data_score['verify_extra_score'] = get_verify_scores(dat_files['VERIFY'],suffix,pred_type)
            if pred_type == 'reg':
                if data_score['r2_score'] < 2 or data_score['verify_score'] < 4:
                    r2_verify_warnings += 1
                robert_score = data_score['r2_score'] + data_score['outliers_score'] + data_score['verify_score'] + data_score['descp_score']

            elif pred_type == 'clas':
                if data_score['r2_score'] < 2 or data_score['verify_score'] < 2 or data_score['verify_extra_score'] < 2:
                    r2_verify_warnings += 1
                robert_score = data_score['r2_score'] + data_score['verify_score'] + data_score['verify_extra_score'] + data_score['descp_score']
            
            robert_score_list.append(robert_score)

            # add some spacing to the PFI column fo be aligned with the second half of the threshold section
            if suffix == 'No PFI':
                spacing_PFI = ''
            elif suffix == 'PFI':
                spacing_PFI = '&nbsp;&nbsp;&nbsp;&nbsp;'

            # prints the two-column ROBERT score summary
            score_info = f"""{spacing_PFI}<img src="file:///{self.args.path_icons}/score_{robert_score}.jpg" alt="ROBERT Score" style="width: 100%; margin-top:5px;">
<p style="text-align: justify; margin-top: -2px; margin-bottom: -2px;"><strong>{spacing_PFI}The model has a score of {robert_score}/10</strong><p>"""

            # get amount of points or lines to add
            if suffix == 'No PFI':
                columns_score.append(get_col_score(score_info,data_score,suffix,csv_test,spacing_PFI,pred_type,test_set))
            elif suffix == 'PFI':
                columns_score.append(get_col_score(score_info,data_score,suffix,csv_test,spacing_PFI,pred_type,test_set))

        # Combine both columns
        score_dat += combine_cols(columns_score)

        # gets the result images from PREDICT
        score_dat += self.get_results_img()

        # gets errors from PREDICT
        columns_summary = []
        module_file = f'{os.getcwd()}/PREDICT/PREDICT_data.dat'
        for suffix in ['No PFI','PFI']:
            # in the SCORE section, we only take the lines showing errors
            columns_summary.append(get_summary('PREDICT',module_file,suffix,titles=False,pred_type=pred_type))
        
        # Combine both columns
        score_dat += combine_cols(columns_summary)
        
        # represents the thresholds
        score_dat += f"""
<br><p style="text-align: justify; margin-top: -6px; margin-bottom: -2px;"></p>"""
        
        space_title = '&nbsp;'*21
        threshold_cols = ''
        if pred_type == 'reg':
            lines_R2 = [f'{"&nbsp;"*1}<strong>R<sup>2</sup></strong>&nbsp;&nbsp;<u>{"&nbsp;"*27}</u>',
                        f'<br>{"&nbsp;"*1}{get_pts(2)}{"&nbsp;"*2}  R<sup>2</sup> > 0.85{"&nbsp;"*11}',
                        f'<br>{"&nbsp;"*1}{get_pts(1)}{"&nbsp;"*4}    0.85 > R<sup>2</sup> > 0.70',
                        f'<br>{"&nbsp;"*1}{get_pts(0)}{"&nbsp;"*5}     R<sup>2</sup> < 0.70{"&nbsp;"*12}']
            lines_outliers = [f'<strong>Outliers</strong>&nbsp;&nbsp;<u>{"&nbsp;"*27}</u>',
                        f'{get_pts(2)}{"&nbsp;"*2}  < 7.5% of outliers{"&nbsp;"*7}',
                        f'{get_pts(1)}{"&nbsp;"*4}    7.5% < outliers < 15%',
                        f'{get_pts(0)}{"&nbsp;"*5}     > 15% of outliers{"&nbsp;"*8}']
            lines_descp = [f'<strong>Points:descriptors</strong>&nbsp;&nbsp;<u>{"&nbsp;"*6}</u>',
                        f'{get_pts(2)}{"&nbsp;"*2}  > 10:1 p:d ratio',
                        f'{get_pts(1)}{"&nbsp;"*4}    10:1 > p:d ratio > 3:1',
                        f'{get_pts(0)}{"&nbsp;"*5}     p:d ratio < 3:1']
            lines_verify = [f'<strong>VERIFY tests</strong>&nbsp;&nbsp;<u>{"&nbsp;"*13}</u>',
                        f'{"&nbsp;"*7}  Up to {get_pts(4)} (tests pass)',
                        f'{get_pts(0)}&nbsp; (all tests failed)',
                        f'']
            for i,_ in enumerate(lines_R2):
                threshold_cols += f'{lines_R2[i]}{"&nbsp;"*10}{lines_outliers[i]}{"&nbsp;"*10}{lines_descp[i]}{"&nbsp;"*9}{lines_verify[i]}'


        elif pred_type == 'clas':
            lines_R2 = [f'{"&nbsp;"*1}<strong>Accuracy</strong>&nbsp;&nbsp;<u>{"&nbsp;"*21}</u>',
                        f'<br>{"&nbsp;"*1}{get_pts(2)}{"&nbsp;"*2}  Accuracy > 0.85{"&nbsp;"*7}',
                        f'<br>{"&nbsp;"*1}{get_pts(1)}{"&nbsp;"*4}    0.85 > Accur. > 0.70',
                        f'<br>{"&nbsp;"*2}{get_pts(0)}{"&nbsp;"*5}    Accur. < 0.70{"&nbsp;"*4}']
            lines_outliers = [f'<strong>Outliers</strong>&nbsp;&nbsp;<u>{"&nbsp;"*19}</u>',
                        f'{get_pts(0)}{"&nbsp;"*2}Excluded in classif.',
                        f'{"&nbsp;"*35}',
                        f'{"&nbsp;"*42}']
            lines_descp = [f'<strong>Points:descriptors</strong>&nbsp;&nbsp;<u>{"&nbsp;"*7}</u>',
                        f'{get_pts(2)}{"&nbsp;"*2}  > 10:1 p:d ratio{"&nbsp;"*1}',
                        f'{get_pts(1)}{"&nbsp;"*4}    10:1 > p:d ratio > 3:1',
                        f'{get_pts(0)}{"&nbsp;"*5}     p:d ratio < 3:1']
            lines_verify = [f'<strong>VERIFY tests</strong>&nbsp;&nbsp;<u>{"&nbsp;"*13}</u>',
                        f'{"&nbsp;"*8}{get_pts(2)}{"&nbsp;"*2}y-shuffle & y-mean',
                        f'{get_pts(1)}{"&nbsp;"*4}5-fold CV & onehot',
                        f'{"&nbsp;"*11}{get_pts(0)}{"&nbsp;"*5}(all tests failed)']

            for i,_ in enumerate(lines_R2):
                threshold_cols += f'{lines_R2[i]}{"&nbsp;"*10}{lines_outliers[i]}{"&nbsp;"*10}{lines_descp[i]}{"&nbsp;"*10}{lines_verify[i]}'

        score_dat += '''<style>
        th, td {
        border:0.75px solid black;
        border-collapse: collapse;
        padding: 5px;
        text-align: justify;
        }
        '''
        score_dat += f'''
        </style>
        <table style="width:100%">
            <tr>
                <td colspan="3">{space_title}<strong>Score thresholds</strong> <i>(detailed in https://robert.readthedocs.io/en/latest/Score/score.html)</i></td>
            </tr>
            <tr>
                <td colspan="3">{threshold_cols}</td>
            </tr>
            <tr>
        </table>'''
        
        score_dat += f"""<p style="page-break-after: always;"></p>"""

        # get some tips
        score_dat += f"""
<br><p style="text-align: justify; margin-top: -8px; margin-bottom: -2px;"><u>Some tips to improve the score</u></p>"""
        
        n_scoring = 1
        style_line = '<p style="text-align: justify;">'
        reduced_line = '<p style="text-align: justify; margin-top: -5px;">' # reduces line separation separation
        last_line = '<p style="text-align: justify; margin-top: -5px; margin-bottom: 30px;">'
        if max(robert_score_list) >= 9:
            score_dat += f'{style_line}&#10004;&nbsp;&nbsp; A ROBERT score of 9 or 10 suggests that the predictive ability of your model is strong, congratulations!</p>'
            n_scoring += 1
            style_line = reduced_line
        else:
            datapoints = int(data_score['proportion_ratio'].split(':')[0])
            if datapoints <= 50:
                score_dat += f'{style_line}&#9888;&nbsp; The model uses only {datapoints} datapoints, adding meaningful datapoints might help to improve the model.</p>'
                n_scoring += 1
                style_line = reduced_line
            if outliers_warnings > 0 and pred_type == 'reg':
                if outliers_warnings == 1:
                    outliers_warnings = 'One'
                elif outliers_warnings == 2:
                    outliers_warnings = 'Two'
                score_dat += f'{style_line}&#9888;&nbsp; {outliers_warnings} of your models have more than 7.5% of outliers (5% is expected for a normal distribution with the t-value of 2 that ROBERT uses), using a more homogeneous distribution of results might help.</p>'
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
        citation_dat += f"""<p style="text-align: justify;"><br>{version_n_date}</p>
        <p style="text-align: justify;  margin-top: -8px;"><span style="font-weight:bold;">How to cite:</span> {citation}</p>"""

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

        repro_dat += f"""<p style="page-break-after: always;"></p>"""

        repro_dat = self.module_lines('repro',repro_dat) 

        return citation_dat, repro_dat, dat_files, self.args.csv_name, self.args.csv_test, robert_version


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
            section_explain = f'<i style="text-align: justify;">This score is designed to analyze the predictive ability of the models using different metrics.</i>'
        elif module == 'abbrev':
            module_name = 'ABBREVIATIONS'
            section_explain = f'<i style="text-align: justify;">Reference section for the abbreviations used.</i>'
        else:
            module_name = module
            section_explain = ''
        if module not in ['repro','transpa']:
            module_data = format_lines(module_data)
        module_data = '<div class="aqme-content"><pre>' + module_data + '</pre></div>'
        separator_section = ''
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


    def get_results_img(self):
        """
        Generate the string that includes the results images from PREDICT
        """
        
        # first, generates graphs with no titles

        # get all the y and y_pred
        module_path = Path(f'{os.getcwd()}/PREDICT')
        csv_files = [str(file_path) for file_path in module_path.rglob('*.csv')]
        graph_style = get_graph_style()

        for suffix in ['No_PFI','PFI']:
            # set the parameters for each ML model
            params_dir = f'{self.args.params_dir}/{suffix}'
            files_param = glob.glob(f'{params_dir}/*.csv')
            for file_param in files_param:
                if '_db' not in file_param:
                    params_df = pd.read_csv(file_param, encoding='utf-8')
            params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

            # get y and y_pred values
            Xy_data = {}
            for file in csv_files:
                plot_graph = False
                if suffix == 'No_PFI' and suffix in file:
                    plot_graph = True
                elif suffix == 'PFI' and 'No_PFI' not in file:
                    plot_graph = True
                if plot_graph:
                    if f'_train_{suffix}.csv' in file:
                        Xy_data["y_train"], Xy_data["y_pred_train"] = get_y_values(file,params_dict["y"])
                    elif f'_valid_{suffix}.csv' in file:
                        Xy_data["y_valid"], Xy_data["y_pred_valid"] = get_y_values(file,params_dict["y"])
                    elif f'_test_{suffix}.csv' in file:
                        Xy_data["y_test"], Xy_data["y_pred_test"] = get_y_values(file,params_dict["y"])

            set_types = ['train','valid']
            if 'y_test' in Xy_data:
                set_types.append('test')
            
            path_n_suffix = f'{module_path}/Results_{suffix}_REPORT' # I add "_REPORT" to the file so I can delete these files later

            if params_dict['type'].lower() == 'reg':
                _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style,print_fun=False)

        module_path = Path(f'{os.getcwd()}/PREDICT')
        if params_dict['type'].lower() == 'reg':
            results_images = [str(file_path) for file_path in module_path.rglob('*_REPORT.png')]
        elif params_dict['type'].lower() == 'clas':
            results_images  = []
            all_images = [str(file_path) for file_path in module_path.rglob('*.png')]
            for img in all_images:
                if 'test' in set_types:
                    if 'Results' in img and '_test.png' in img:
                        results_images.append(img)
                else:
                    if 'Results' in img and '_valid.png' in img:
                        results_images.append(img)

        # keep the ordering (No_PFI in the left, PFI in the right of the PDF)
        if 'No_PFI' in results_images[1]:
            results_images = revert_list(results_images)
        
        # define widths the graphs
        width = 91
        pair_list = f'<p style="width: {width}%; margin-bottom: -2px;  margin-top: -5px"><img src="file:///{results_images[0]}" style="margin: 0"/>'
        pair_list += f'{("&nbsp;")*15}'
        pair_list += f'<img src="file:///{results_images[1]}" style="margin: 0"/></p>'

        html_png = f'{pair_list}'

        return html_png    
