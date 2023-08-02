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
import textwrap
from pathlib import Path
import sys
from robert.utils import load_variables

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

                # include images and summary
                data_lines += get_images('PREDICT')
 
        #         if module != 'AQME':
        #             if module == 'PREDICT':
        #                 # Column 1: No PFI
        #                 column1 = f"""
        #                 <p><span style="font-weight:bold;">No PFI (all descriptors):</span></p>
        #                 <pre style="text-align: justify;">{module_data_predict}</pre>
        #                 """

        #                 # Column 2: PFI
        #                 column2 = f"""
        #                 <p><span style="font-weight:bold;">PFI (only important descriptors):</span></p>
        #                 <pre style="text-align: justify;">{module_data_predict_2}</pre>
        #                 """

        #                 # Combine both columns
        #                 column_div = f"""
        #                 <div style="display: flex;">
        #                     <div style="flex: 1;">{column1}</div>
        #                     <div style="flex: 1;">{column2}</div>
        #                 </div>
        #                 """
        #                 data_lines += f"""
        #                 <p>&nbsp;</p>
        #                 <p><span style="font-weight:bold;">
        #                 ------- Images and summary generated by the {module} module -------
        #                 </span></p>
        #                 </p>
        #                 {column_div}
        #                 </p>
        #                 {html_png} 
        #                 """


        return data_lines


    def get_header(self,modules):
        """
        Retrieves the header for the HTML string
        """

        # Reproducibility and Transparency section
        citation_dat, repro_dat, dat_files, csv_name, robert_version = self.get_repro(modules)

        # ROBERT score section
        score_dat = self.get_score(dat_files)

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


    def get_score(self,dat_files):
        """
        Generates the ROBERT score section
        """
        
        # starts with the icon of ROBERT score
        score_dat = ''
        score_dat = self.module_lines('score',score_dat) 

        # calculates the ROBERT scores
        outliers_warnings, r2_verify_warnings, descp_warning = 0,0,0
        robert_score_list = []
        data_score = {}
        for suffix in ['No PFI','PFI']:
            data_score['r2_score'], data_score['r2_valid'], data_score['outliers_score'], data_score['outliers_prop'], data_score['descp_score'], data_score['proportion_ratio'] = get_predict_scores(dat_files['PREDICT'],suffix)
            robert_score_list.append(data_score['r2_score'])
            if data_score['outliers_score'] < 2:
                outliers_warnings += 1
            if data_score['descp_score'] < 2:
                descp_warning += 1

            data_score['verify_score'] = get_verify_scores(dat_files['VERIFY'],suffix)
            if data_score['r2_score'] < 2 or data_score['verify_score'] < 2:
                r2_verify_warnings += 1
            
            robert_score = data_score['r2_score'] + data_score['outliers_score'] + data_score['verify_score'] + data_score['descp_score']

            # prints the two-column ROBERT score summary
            score_info = f"""<img src="file:///{self.args.path_icons}/score_{robert_score}.jpg" alt="ROBERT Score" style="width: 100%;">
<u>Your model has a score of <span style="font-weight:bold;">{robert_score}/10</span></u>"""
            
            # get amount of points or lines to add
            if suffix == 'No PFI':
                column1 = get_col_score(score_info,data_score,suffix)
            elif suffix == 'PFI':
                column2 = get_col_score(score_info,data_score,suffix)

        # Combine both columns
        score_dat += f"""
        <div style="display: flex;">
            <div style="flex: 1;">{column1}</div>
            <div style="flex: 1;">{column2}</div>
        </div>
        """

        # represents the thresholds
        score_dat += f"""
<u>Score thresholds</u>"""
        thres_1 = get_col_thres('R<sup>2</sup>')
        thres_2 = get_col_thres('outliers')
        thres_3 = get_col_thres('VERIFY')
        thres_4 = get_col_thres('descps')
        score_dat += f"""
        <div style="display: flex;">
            <div style="flex: 1;">{thres_1}</div>
            <div style="flex: 1;">{thres_2}</div>
            <div style="flex: 1;">{thres_3}</div>
            <div style="flex: 1;">{thres_4}</div>            
        </div>
        """
        
        # get some tips
        score_dat += f"""
<u>Some tips to improve the score</u>"""
        
        if min(robert_score_list) in [9,10]:
            score_dat += f"\n- A ROBERT score of 9 or 10 suggests that the predictive ability of your model is strong, congratulations!"
        else:
            datapoints = int(data_score['proportion_ratio'].split(':')[0])
            if datapoints <= 50:
                score_dat += f"\n- The model uses only {datapoints} datapoints, adding meaningful datapoints might help to improve the model."
            if outliers_warnings > 0:
                if outliers_warnings == 1:
                    outliers_warnings = 'One'
                elif outliers_warnings == 2:
                    outliers_warnings = 'Two'
                score_dat += f"\n- {outliers_warnings} of your models have more than 5% of outliers (expected for a t-value of 2), using a more homogeneous distribution of results might help. For example, avoid using many points with similar y values and only a few points with distant y values."
            if descp_warning > 0:
                score_dat += f"\n- Adding meaningful descriptors or replacing/deleting the least useful descriptors used might help. Feature importances are gathered in the SHAP and PFI sections of the /PREDICT/PREDICT_data.dat file."
            else:
                score_dat += f"\n- Replacing or deleting the least useful descriptors used might help to improve the model. Feature importances are gathered in the SHAP and PFI sections of the /PREDICT/PREDICT_data.dat file."

        score_dat += f"""</p>"""

        return score_dat


    def get_repro(self,modules):
        """
        Generates the data printed in the Reproducibility and Transparency section
        """
        
        version_n_date, citation, command_line, python_version, intelex_version, total_time, dat_files = repro_info(modules)
        robert_version = version_n_date.split()[2]
        
        csv_name = ''
        if 'csv_name' in command_line:
            csv_name = command_line.split('csv_name')[1].split()[0]
            if csv_name[0] in ['"',"'"]:
                csv_name = csv_name[1:]
            if csv_name[-1] in ['"',"'"]:
                csv_name = csv_name[:-1]
        else:
            for module in ['CURATE','GENERATE']:
                if module in dat_files:
                    for line in dat_files[module]:
                        if 'csv_name option set to ' in line:
                            csv_name = line.split('csv_name option set to ')[1].split()[0]
                            break
                    if csv_name != '':
                        break
        
        csv_test = ''
        if 'csv_test' in command_line:
            csv_test = command_line.split('csv_test')[1].split()[0]
            if csv_test[0] in ['"',"'"]:
                csv_test = csv_test[1:]
            if csv_test[-1] in ['"',"'"]:
                csv_test = csv_test[:-1]

        repro_dat,citation_dat = '',''
        
        # version, date and citation
        citation_dat += f"""<br>{version_n_date}<span style="font-weight:bold;"><br>How to cite:</span> {citation}"""

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
        repro_dat += f"""
<br><u>3. Run ROBERT with the following command line (originally run in Python {python_version}):</u>
{command_line}
"""

        # add total execution time
        repro_dat += f"""
    <span style="font-weight:bold;">\nTotal execution time:</span> {total_time} seconds
        """

        repro_dat = self.module_lines('repro',repro_dat) 

        return citation_dat, repro_dat, dat_files, csv_name, robert_version


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
        title_line = f"""
            <p>
            <hr>
            <p><span style="font-weight:bold;">
                <img src="file:///{self.args.path_icons}/{module}.png" alt="" style="width:20px; height:20px; margin-right:5px;">
                {module_name}
            </span></p>
            {module_data}
            </p>
            """
        
        return title_line


def get_images(module):
    """
    Generate the string that includes images
    """
    
    module_path = Path(f'{os.getcwd()}/{module}')
    module_images = [str(file_path) for file_path in module_path.rglob('*.png')]

    if module not in ['PREDICT','VERIFY']:
        image_caption = f'<br>------- Images generated by the {module} module -------'
    else:
        image_caption = f'<br>------- Images and summary generated by the {module} module -------'

    if module == 'VERIFY':
        if len(module_images) == 2 and 'No_PFI' in module_images[1]:
            module_images = revert_list(module_images)

    if module == 'PREDICT':
        results_images = []
        shap_images = []
        pfi_images = []
        outliers_images = []

        for image_path in module_images:
            filename = Path(image_path).stem
            if "Results" in filename:
                results_images.append(image_path)
            elif "SHAP" in filename:
                shap_images.append(image_path)
            elif "Outliers" in filename:
                outliers_images.append(image_path)
            else:
                pfi_images.append(image_path)

        # keep the ordering (No_PFI in the left, PFI in the right of the PDF)
        if len(results_images) == 2 and 'No_PFI' in results_images[1]:
            results_images = revert_list(results_images)
        if len(shap_images) == 2 and 'No_PFI' in shap_images[1]:
            shap_images = revert_list(shap_images)
        if len(pfi_images) == 2 and 'No_PFI' in pfi_images[1]:
            pfi_images = revert_list(pfi_images)
        if len(outliers_images) == 2 and 'No_PFI' in outliers_images[1]:
            outliers_images = revert_list(outliers_images)
        
        html_png = ''
        for _,image_pair in enumerate([results_images, shap_images, pfi_images, outliers_images]):
            
            pair_list = '   '.join([f'<img src="file:///{image_path}" style="margin-bottom: 10px; margin-top: 10px;"/>' for image_path in image_pair])
            html_png += f'{pair_list}'

    if module != 'PREDICT':
        html_png = '   '.join([f'<img src="file:///{image_path}" style="margin-bottom: 10px; margin-top: 10px;"/>' for image_path in module_images])

    imag_lines = f"""
<p style="text-align: center; margin: 0;"><span style="font-weight:bold;">
{image_caption}
</span></p>
{html_png} 
"""
    
    return imag_lines


def revert_list(list_tuple):
    """
    Reverts the order of a list of two components
    """

    new_sort = [] # for some reason reverse() gives a weird issue when reverting lists
    new_sort.append(list_tuple[1])
    new_sort.append(list_tuple[0])

    return new_sort


def get_time(file):
    """
    Returns the execution time of the modules
    """

    module_time = 'x  ROBERT did not save any execution time for this module'
    with open(file, 'r') as datfile:
        for line in reversed(datfile.readlines()):
            if 'Time' in line and 'seconds' in line:
                module_time = line
                break

    return module_time


def get_col_score(score_info,data_score,suffix):
    """
    Gather the information regarding the score of the No PFI and PFI models
    """
    
    r2_pts = get_pts(data_score['r2_score'])
    outliers_pts = get_pts(data_score['outliers_score'])
    descp_pts = get_pts(data_score['descp_score'])
    verify_pts = get_pts(data_score['verify_score'])

    spacing_r2 = get_spacing(data_score['r2_score'])
    spacing_outliers = get_spacing(data_score['outliers_score'])
    spacing_descp = get_spacing(data_score['descp_score'])
    spacing_verify = get_spacing(data_score['verify_score'])

    score_info += f"""
{r2_pts}{spacing_r2}  Your model shows an R<sup>2</sup> of {data_score['r2_valid']}
{outliers_pts}{spacing_outliers}  Your model has {data_score['outliers_prop']}% of outliers
{descp_pts}{spacing_descp}  Your model uses {data_score['proportion_ratio']} points:descriptors
{verify_pts}{spacing_verify}  Your model passes {data_score['verify_score']} VERIFY tests
"""

    # Column 1: No PFI
    if suffix == 'No PFI':
        column = f"""
        <p><span style="font-weight:bold;">No PFI (all descriptors):</span></p>
        <pre style="text-align: justify;">{score_info}</pre>
        """
    elif suffix == 'PFI':
        # Column 2: PFI
        column = f"""
        <p><span style="font-weight:bold;">PFI (only important descriptors):</span></p>
        <pre style="text-align: justify;">{score_info}</pre>
        """

    return column


def get_col_thres(type_thres):
    """
    Gather the information regarding the thresholds used in the score
    """

    if type_thres == 'R<sup>2</sup>':
        column = f"""<span style="font-weight:bold;">R<sup>2</sup></span><pre style="text-align: justify;">
{get_pts(2)}  R<sup>2</sup> > 0.85
{get_pts(1)}    0.85 > R<sup>2</sup> > 0.70
{get_pts(0)}     R<sup>2</sup> < 0.70</pre>
"""

    elif type_thres == 'outliers':
        column = f"""<span style="font-weight:bold;">Outliers</span><pre style="text-align: justify;">
{get_pts(2)}  < 5% of outliers
{get_pts(1)}    5% < outliers < 10%
{get_pts(0)}     > 10% of outliers</pre>
"""

    elif type_thres == 'VERIFY':
        column = f"""<span style="font-weight:bold;">VERIFY tests</span><pre style="text-align: justify;">
Up to {get_pts(4)} (tests pass)
{get_pts(0)}  (all tests failed)</pre>
"""

    elif type_thres == 'descps':
        column = f"""<span style="font-weight:bold;">Points:descriptors</span><pre style="text-align: justify;">
{get_pts(2)}  > 10:1 ratio
{get_pts(1)}    10:1 > ratio > 3:1
{get_pts(0)}     ratio < 3:1</pre>
"""

    return column


def get_pts(score):
    """
    Get a string with the numbers of points or lines to add
    """

    if score > 0:
        str_pts = f"{score*'&#9679;'}"
    else:
        str_pts = f"-"

    return str_pts


def get_spacing(score):
    """
    Get a string with the number of spaces needed
    """

    if score == 4:
        spacing = f""
    if score == 3:
        spacing = f"  "
    if score == 2:
        spacing = f"    "
    if score == 1:
        spacing = f"      "
    if score == 0:
        spacing = f"       "

    return spacing


def get_verify_scores(dat_verify,suffix):
    """
    Calculates scores that come from the VERIFY module (VERIFY tests)
    """

    start_data = False
    verify_score = 0
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
            if 'Original ' in line and '(validation set) =' in line:
                for j in range(i+1,i+5):
                    if 'PASSED' in dat_verify[j]:
                        verify_score += 1

    return verify_score


def get_predict_scores(dat_predict,suffix):
    """
    Calculates scores that come from the PREDICT module (R2, datapoints:descriptors ratio, outlier proportion)
    """

    start_data, test_set = False, False
    r2_score, outliers_score, descp_score = 0,0,0
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
            # R2 and proportion
            if 'o  Results saved in PREDICT/' in line:
                # R2 from test (if any) or validation
                if '-  Test : R2' in dat_predict[i+7]:
                    r2_valid = float(dat_predict[i+7].split()[5].split(',')[0])
                    test_set = True
                elif '-  Validation : R2' in dat_predict[i+6]:
                    r2_valid = float(dat_predict[i+6].split()[5].split(',')[0])
                if r2_valid > 0.85:
                    r2_score += 2
                elif r2_valid > 0.7:
                    r2_score += 1

                # proportion
                proportion_ratio = dat_predict[i+4].split()[-1]
                proportion = int(proportion_ratio.split(':')[0]) / int(proportion_ratio.split(':')[1])
                if proportion > 10:
                    descp_score += 2
                elif proportion > 3:
                    descp_score += 1

            # outliers
            if 'o  Outlier values saved in' in line:
                for j in range(i,len(dat_predict)):
                    if test_set and 'Test:' in dat_predict[j]:
                        outliers_prop = dat_predict[j].split()[-1]
                        outliers_prop = outliers_prop.split('%)')[0]
                        outliers_prop = float(outliers_prop.split('(')[-1])
                    elif not test_set and 'Validation:' in dat_predict[j]:
                        outliers_prop = dat_predict[j].split()[-1]
                        outliers_prop = outliers_prop.split('%)')[0]
                        outliers_prop = float(outliers_prop.split('(')[-1])
                    elif len(dat_predict[j].split()) == 0:
                        break
                if outliers_prop < 5:
                    outliers_score += 2
                elif outliers_prop < 10:
                    outliers_score += 1

    return r2_score, r2_valid, outliers_score, outliers_prop, descp_score, proportion_ratio


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
        os.remove(outfile)
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


def format_lines(module_data, max_width=122):
    """
    Reads a file and returns a formatted string between two markers
    """

    formatted_lines = []
    lines = module_data.split('\n')
    for i,line in enumerate(lines):
        formatted_line = textwrap.fill(line, width=max_width, subsequent_indent='')
        if i > 0:
            formatted_lines.append(f'<pre style="text-align: justify;">\n{formatted_line}</pre>')
        else:
            formatted_lines.append(f'<pre style="text-align: justify;">{formatted_line}</pre>\n')

    return ''.join(formatted_lines)
