"""
Parameters
----------

    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    report_modules : list of str, default=['CURATE','GENERATE','VERIFY','PREDICT']
        List of the modules to include in the report.

"""
#####################################################.
#        This file stores the REPORT class          #
#    used for generating the final PDF report       #
#####################################################.

import os
import sys
from robert.utils import load_variables
from robert.report_utils import (
    get_csv_names,
    get_images,
    get_time,
    get_col_score,
    get_col_thres,
    get_verify_scores,
    get_predict_scores,
    repro_info,
    make_report,
    css_content,
    format_lines,
    combine_cols
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
        report_html,csv_name,robert_version = self.get_header(self.args.report_modules)
        report_html += self.get_data(self.args.report_modules)

        # create css
        with open("report.css", "w") as cssfile:
            cssfile.write(css_content(csv_name,robert_version))

        _ = make_report(report_html,HTML)

        # Remove report.css file
        os.remove("report.css")
                
    def get_data(self, modules):
        """
        Get information, times and images of the modules
        """

        data_lines = ""

        # AQME section
        if 'AQME' in modules:
            aqme_file = f'{os.getcwd()}/AQME/AQME_data.dat'
            if os.path.exists(aqme_file):
                aqme_time = get_time(aqme_file)
                aqme_data = f"""<i>This module performs RDKit-based conformer generation from SMILES databases in CSV files, followed by the generation of 200+ molecular and atomic descriptors using RDKit, xTB and DBSTEP (saved as AQME-ROBERT_FILENAME.csv).</i>
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
                data_lines += get_images('PREDICT')

        return data_lines


    def get_header(self,modules):
        """
        Retrieves the header for the HTML string
        """

        # Reproducibility and Transparency section
        citation_dat, repro_dat, dat_files, csv_name, csv_test, robert_version = self.get_repro(modules)

        # ROBERT score section
        score_dat = self.get_score(dat_files,csv_test)

        # combines the top image with the other sections of the header
        header_lines = f"""
            <h1 style="text-align: center; margin-bottom: 0.5em;">
                <img src="file:///{self.args.path_icons}/Robert_logo.jpg" alt="" style="display: block; margin-left: auto; margin-right: auto; width: 50%; margin-top: -12px;" />
                <span style="font-weight:bold;"></span>
            </h1>
            <p style="text-align: justify;">
            {citation_dat}
            {score_dat}
            {repro_dat}
            </p>
            """

        return header_lines,csv_name,robert_version


    def get_score(self,dat_files,csv_test):
        """
        Generates the ROBERT score section
        """
        
        # starts with the icon of ROBERT score
        score_dat = ''
        score_dat = self.module_lines('score',score_dat) 

        # calculates the ROBERT scores
        outliers_warnings, r2_verify_warnings, descp_warning = 0,0,0
        robert_score_list,columns_score = [],[]
        data_score = {}
        for suffix in ['No PFI','PFI']:
            data_score['r2_score'], data_score['r2_valid'], data_score['outliers_score'], data_score['outliers_prop'], data_score['descp_score'], data_score['proportion_ratio'] = get_predict_scores(dat_files['PREDICT'],suffix)
            if data_score['outliers_score'] < 2:
                outliers_warnings += 1
            if data_score['descp_score'] < 2:
                descp_warning += 1

            data_score['verify_score'] = get_verify_scores(dat_files['VERIFY'],suffix)
            if data_score['r2_score'] < 2 or data_score['verify_score'] < 2:
                r2_verify_warnings += 1
            
            robert_score = data_score['r2_score'] + data_score['outliers_score'] + data_score['verify_score'] + data_score['descp_score']
            robert_score_list.append(robert_score)

            # prints the two-column ROBERT score summary
            score_info = f"""<img src="file:///{self.args.path_icons}/score_{robert_score}.jpg" alt="ROBERT Score" style="width: 100%; margin-top:5px;">
<u>Your model has a score of <span style="font-weight:bold;">{robert_score}/10</span></u>"""
            
            # get amount of points or lines to add
            if suffix == 'No PFI':
                columns_score.append(get_col_score(score_info,data_score,suffix))
            elif suffix == 'PFI':
                columns_score.append(get_col_score(score_info,data_score,suffix))

        # Combine both columns
        score_dat += combine_cols(columns_score)

        # represents the thresholds
        if csv_test != '':
            score_set = f'(based on results from the external test set)'
        else:
            score_set = '(based on results from the validation set)'
        score_dat += f"""
<br><u>Score thresholds {score_set}</u>"""
        
        columns_thres = []
        columns_thres.append(get_col_thres('R<sup>2</sup>'))
        columns_thres.append(get_col_thres('outliers'))
        columns_thres.append(get_col_thres('descps'))
        columns_thres.append(get_col_thres('VERIFY'))
        score_dat += combine_cols(columns_thres)
        
        # get some tips
        score_dat += f"""
<br><u>Some tips to improve the score</u>"""
        
        n_scoring = 1
        if min(robert_score_list) >= 9:
            score_dat += f'<p style="text-align: justify;">&#10004;&nbsp;&nbsp; A ROBERT score of 9 or 10 suggests that the predictive ability of your model is strong, congratulations!</p>'
            n_scoring += 1
        else:
            datapoints = int(data_score['proportion_ratio'].split(':')[0])
            if datapoints <= 50:
                score_dat += f'<p style="text-align: justify;">&#9888;&nbsp; The model uses only {datapoints} datapoints, adding meaningful datapoints might help to improve the model.</p>'
                n_scoring += 1
            if outliers_warnings > 0:
                if outliers_warnings == 1:
                    outliers_warnings = 'One'
                elif outliers_warnings == 2:
                    outliers_warnings = 'Two'
                score_dat += f'<p style="text-align: justify;">&#9888;&nbsp; {outliers_warnings} of your models have more than 5% of outliers (expected for a t-value of 2), using a more homogeneous distribution of results might help. For example, avoid using many points with similar y values and only a few points with distant y values.</p>'
                n_scoring += 1
            if descp_warning > 0:
                score_dat += f'<p style="text-align: justify;">&#9888;&nbsp; Adding meaningful descriptors or replacing/deleting the least useful descriptors used might help. Feature importances are gathered in the SHAP and PFI sections of the /PREDICT/PREDICT_data.dat file.</p>'
                n_scoring += 1
            else:
                score_dat += f'<p style="text-align: justify;">&#9888;&nbsp; Replacing or deleting the least useful descriptors used might help to improve the model. Feature importances are gathered in the SHAP and PFI sections of the /PREDICT/PREDICT_data.dat file.</p>'
                n_scoring += 1

        # how to predict new values
        score_dat += f"""
        <p><br><u>How to predict new values?</u></p>
<p style="text-align: justify;"><span style="font-weight:bold;">1.</span> Create a CSV database with the new points, including the necessary descriptors.</p>
<p style="text-align: justify;"><span style="font-weight:bold;">2.</span> Place the CSV file in the parent folder (i.e., where the module folders were created)</p>
<p style="text-align: justify;"><span style="font-weight:bold;">3.</span> Run the PREDICT module as 'python -m robert --predict --csv_test FILENAME.csv'.</p>
<p style="text-align: justify;"><span style="font-weight:bold;">4.</span> The predictions will be stored in the last column of two CSV files called MODEL_SIZE_test(_No)_PFI.csv, which are stored in the PREDICT folder.</p>"""

        score_dat += f"""<p style="page-break-after: always;"></p>"""

        return score_dat


    def get_repro(self,modules):
        """
        Generates the data printed in the Reproducibility and Transparency section
        """
        
        version_n_date, citation, command_line, python_version, intelex_version, total_time, dat_files = repro_info(modules)
        robert_version = version_n_date.split()[2]

        csv_name,csv_test = get_csv_names(command_line,dat_files)

        repro_dat,citation_dat = '',''
        
        # version, date and citation
        citation_dat += f"""<p><br>{version_n_date}<br><span style="font-weight:bold;">How to cite:</span> {citation}</p>"""

        # reproducibility section, starts with the icon of reproducibility        
        repro_dat += f"""
<u>1. Upload these files to the supporting information:</u>
  - Report with results (ROBERT_report.pdf)
  - CSV database ({csv_name})
"""
        
        if csv_test != '':
            repro_dat += f"""  - External test set ({csv_test})
"""

        repro_dat += f"""
<br><u>2. Install the following Python modules:</u>
  - ROBERT: conda install -c conda-forge robert={robert_version} (or pip install robert=={robert_version})
"""
        if intelex_version != 'not installed':
            repro_dat += f"""  - scikit-learn-intelex: (pip install scikit-learn-intelex=={intelex_version})
"""
        else:
            repro_dat += f"""  - scikit-learn-intelex: not installed (make sure you do not have it installed)
"""

        repro_dat += f"""
<br><u>3. Run ROBERT with the following command line (originally run in Python {python_version}):</u>
{command_line}
"""

        # add total execution time
        repro_dat += f"""
    <span style="font-weight:bold;">\nTotal execution time:</span> {total_time} seconds
        """

        repro_dat = self.module_lines('repro',repro_dat) 

        return citation_dat, repro_dat, dat_files, csv_name, csv_test, robert_version


    def module_lines(self,module,module_data):
        """
        Returns the line with icon and module for section titles
        """
        
        if module == 'repro':
            module_name = 'REPRODUCIBILITY AND TRANSPARENCY'
        elif module == 'score':
            module_name = 'ROBERT SCORE'
        else:
            module_name = module
        module_data = format_lines(module_data)
        module_data = '<div class="aqme-content"><pre>' + module_data + '</pre></div>'
        separator_section = ''
        if module != 'repro':
            if module == 'score':
                separator_section = '<hr><p style="margin-top:30px;"></p>'
            else:
                separator_section = '<hr>'

        title_line = f"""
            {separator_section}
            <p><span style="font-weight:bold;">
                <img src="file:///{self.args.path_icons}/{module}.png" alt="" style="width:20px; height:20px; margin-right:5px;">
                {module_name}
            </span></p>
            {module_data}
            </p>
            """
        
        return title_line
