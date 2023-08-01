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

        # create css
        with open("report.css", "w") as cssfile:
            cssfile.write(css_content(self.args.report_modules))

        # create report
        report_html = self.get_header(self.args.report_modules)
        report_html += self.get_data(self.args.report_modules)
        _ = make_report(report_html,HTML)

        # Remove report.css file
        os.remove("report.css")
                
    def get_data(self, modules):
        data_lines = ""
        starting_word = ["Starting", "Starting", "Starting", "Starting tests", "Representation"]
        
        for i, module in enumerate(modules):
            dat_file = Path(f'{os.getcwd()}/{module}/{module}_data.dat')
            module_dir = Path(f'{os.getcwd()}/{module}')
            module_images = [str(file_path) for file_path in module_dir.rglob('*.png') if file_path.is_file()]
            # keep the ordering (No_PFI in the left, PFI in the right of the PDF)
            if len(module_images) == 2 and 'No_PFI' in module_images[1]:
                new_sort = [] # for some reason reverse() gives a weird issue when reverting lists
                new_sort.append(module_images[1])
                new_sort.append(module_images[0])
                module_images = new_sort
            # Group PNGs in pairs based on their names within the "PREDICT" module
            if module == "PREDICT":
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
                for imag_list in [results_images, shap_images, pfi_images, outliers_images]:
                    if len(imag_list) == 2 and 'No_PFI' in imag_list[1]:
                        imag_list = imag_list.reverse()

                module_images = results_images + shap_images + pfi_images + outliers_images
            
            html_png = ''.join([f'<img src="file:///{image_path}" alt="" class="img-{module}"/>' for image_path in module_images])

            if os.path.exists(dat_file):
                start_st = starting_word[modules.index(module)]
                module_data = format_data(dat_file, start_st, f"Time {module}:")
                module_data = '<div class="aqme-content"><pre>' + module_data + '</pre></div>'
                module_data_predict = ""
                
                if module == 'PREDICT':
                    start_phrase = "Results saved in PREDICT/"
                    end_phrase = "\n\n"
                    start_pos = module_data.find(start_phrase)
                    end_pos = module_data.find(end_phrase, start_pos)

                    if start_pos != -1 and end_pos != -1:
                        module_data_predict = module_data[start_pos + len(start_phrase):end_pos].strip()

                        start_pos_2 = module_data.find(start_phrase, end_pos)
                        end_pos_2 = module_data.find(end_phrase, start_pos_2)

                        if start_pos_2 != -1 and end_pos_2 != -1:
                            module_data_predict_2 = module_data[start_pos_2 + len(start_phrase):end_pos_2].strip()

                data_lines += f"""
                    <p>
                    <hr>
                    <p><span style="font-weight:bold;">
                        <img src="file:///{self.args.path_icons}/{module}.png" alt="" style="width:20px; height:20px; margin-right:5px;">
                        {module}
                    </span></p>
                    {module_data}
                    </p>
                    """
                
                if module != 'AQME':
                    if module == 'PREDICT':
                        # Column 1: No PFI
                        column1 = f"""
                        <p><span style="font-weight:bold;">No PFI (all descriptors):</span></p>
                        <pre style="text-align: justify;">{module_data_predict}</pre>
                        """

                        # Column 2: PFI
                        column2 = f"""
                        <p><span style="font-weight:bold;">PFI (only important descriptors):</span></p>
                        <pre style="text-align: justify;">{module_data_predict_2}</pre>
                        """

                        # Combine both columns
                        column_div = f"""
                        <div style="display: flex;">
                            <div style="flex: 1;">{column1}</div>
                            <div style="flex: 1;">{column2}</div>
                        </div>
                        """
                        data_lines += f"""
                        <p>&nbsp;</p>
                        <p><span style="font-weight:bold;">
                        ------- Images and summary generated by the {module} module -------
                        </span></p>
                        </p>
                        {column_div}
                        </p>
                        {html_png} 
                        """
                    else:
                        data_lines += f"""
                        <p>&nbsp;</p>
                        <p><span style="font-weight:bold;">
                        ------- Images generated by the {module} module -------
                        </span></p>
                        </p>
                        {html_png} 
                        """
                if i == len(modules) - 1:
                    data_lines += f"""
                        <p>
                    """

        return data_lines


    def get_header(self,modules):
        """
        Retrieves the header for the HTML string
        """

        # Reproducibility and Transparency section
        citation_dat, repro_dat, dat_files = self.get_repro(modules)

        # ROBERT score section
        score_dat = self.get_score(dat_files)

        # combines the top image with the other sections of the header
        header_lines = f"""
            <h1 style="text-align: center; margin-bottom: 0.5em;">
                <img src="file:///{self.args.path_icons}/Robert_logo.jpg" alt="" style="display: block; margin-left: auto; margin-right: auto; width: 50%; margin-top: -12px;" />
                <span style="font-weight:bold;"></span>
            <p>
            </h1>
            <p style="text-align: center;"></p>
            <p style="text-align: justify;">
            <div class="dat-content"><pre>{citation_dat}</pre></div>
            <div class="dat-content"><pre>{score_dat}</pre></div>
            <div class="dat-content"><pre>{repro_dat}</pre></div>
            """

        return header_lines


    def get_score(self,dat_files):
        """
        Generates the ROBERT score section
        """
        
        # starts with the icon of ROBERT score
        score_dat = f"""<p><span style="font-weight:bold;"><img src="file:///{self.args.path_icons}/score.png" alt="" style="width:20px; height:20px; margin-right:5px;">ROBERT SCORE</span></p>"""

        # calculates the ROBERT scores
        outliers_warnings,r2_verify_warnings, descp_warning = 0,0,0
        robert_score_list = []
        for suffix in ['No PFI','PFI']:
            r2_score, r2_valid, outliers_score, outliers_prop, descp_score, proportion_ratio = get_predict_scores(dat_files['PREDICT'],suffix)
            robert_score_list.append(r2_score)
            if outliers_score < 2:
                outliers_warnings += 1
            if descp_score < 2:
                descp_warning += 1

            verify_score = get_verify_scores(dat_files['VERIFY'],suffix)
            if r2_score < 2 or verify_score < 2:
                r2_verify_warnings += 1
            
            robert_score = r2_score + outliers_score + verify_score + descp_score

            # prints the two-column ROBERT score summary
            score_info = f"""<img src="file:///{self.args.path_icons}/score_{robert_score}.jpg" alt="ROBERT Score" style="width: 100%;">
<u>Your model has a score of <span style="font-weight:bold;">{robert_score}/10</span></u>
{r2_score}/2 points: Your model shows an R<sup>2</sup> of {r2_valid}
{outliers_score}/2 points: Your model has {outliers_prop}% of outliers
{verify_score}/4 points: Your model passed {verify_score} VERIFY tests
{descp_score}/2 points: Your model uses {proportion_ratio} points:descriptors"""

            # Column 1: No PFI
            if suffix == 'No PFI':
                column1 = f"""
                <p><span style="font-weight:bold;">No PFI (all descriptors):</span></p>
                <pre style="text-align: justify;">{score_info}</pre>
                """
            elif suffix == 'PFI':
                # Column 2: PFI
                column2 = f"""
                <p><span style="font-weight:bold;">PFI (only important descriptors):</span></p>
                <pre style="text-align: justify;">{score_info}</pre>
                """

        # Combine both columns
        score_dat += f"""
        <div style="display: flex;">
            <div style="flex: 1;">{column1}</div>
            <div style="flex: 1;">{column2}</div>
        </div>
        """

        # reprsents the thresholds
        score_dat += f"""
<u>Score thresholds</u>
2 points for R<sup>2</sup> > 0.85 in validation, 1 point if 0.70 < R<sup>2</sup> < 0.85, 0 points if R<sup>2</sup> < 0.70.
2 points if < 5% outliers in validation, 1 point if < 10% outliers, 0 points if > 10% outliers.
1 point for each PASSED test in the VERIFY analysis
2 points if > 10:1 points:descriptors, 1 point if > 3:1, 0 points if < 3:1
"""
        
        # get some tips
        score_dat += f"""
<u>Some tips to improve the score</u>"""
        
        if min(robert_score_list) in [9,10]:
            score_dat += f"\n- A ROBERT score of 9 or 10 suggests that the predictive ability of your model is strong, congratulations!"
        else:
            datapoints = int(proportion_ratio.split(':')[0])
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
        
        if 'csv_name' in command_line:
            csv_name = command_line.split('csv_name')[1].split()[0]
            if csv_name[0] in ['"',"'"]:
                csv_name = csv_name[1:]
            if csv_name[-1] in ['"',"'"]:
                csv_name = csv_name[:-1]
        if 'csv_test' in command_line:
            csv_test = command_line.split('csv_test')[1].split()[0]
            if csv_test[0] in ['"',"'"]:
                csv_test = csv_test[1:]
            if csv_test[-1] in ['"',"'"]:
                csv_test = csv_test[:-1]

        repro_dat,citation_dat = '',''
        
        # version, date and citation
        citation_dat += f"""{version_n_date}<span style="font-weight:bold;">How to cite:</span> {citation}"""

        # reproducibility section, starts with the icon of reproducibility
        repro_dat += f"""

                
<p><span style="font-weight:bold;"><img src="file:///{self.args.path_icons}/repro.png" alt="" style="width:20px; height:20px; margin-right:5px;">REPRODUCIBILITY AND TRANSPARENCY</span></p>
"""
        
        repro_dat += f"""
<u>1. Upload these files to the supporting information:</u>
  - Report with results (ROBERT_report.pdf)
  - CSV database ({csv_name})
  - External test set ({csv_test})

<u>2. Install the following Python modules:</u>
  - ROBERT: conda install -c conda-forge robert={robert_version} (or pip install robert=={robert_version})
"""
        if intelex_version != 'not installed':
            repro_dat += f"""  - scikit-learn-intelex: (pip install scikit-learn-intelex=={intelex_version})
"""
        repro_dat += f"""
<u>3. Run ROBERT with the following command line (originally run in Python {python_version}):</u>
{command_line}
"""

        # add total execution time
        repro_dat += f"""
    <span style="font-weight:bold;">\nTotal execution time:</span> {total_time} seconds
        """

        return citation_dat, repro_dat, dat_files


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
            if 'Original score (validation set):' in line:
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


def css_content(modules):
    """
    Obtain ROBERT version and CSV name to use it on top of the PDF report
    """

    version = None
    csv_name = None

    for module in modules:
        dat_file = Path(f'{os.getcwd()}/{module}/{module}_data.dat')
        if os.path.exists(dat_file):
            with open(dat_file, 'r', errors="replace") as datfile:
                for line in datfile:
                    if '--csv_name' in line:
                        csv_name = line.split('--csv_name')[1].strip().split()[0]
                        csv_name = csv_name.replace('"','').replace("'",'')
                        break
                    elif version is None:
                        version = line[:14].strip()

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
            content: "{version}";
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


def format_data(file_path, start_str, end_str, every=110):
    """
    Reads a file and returns a formatted string between two markers
    """

    with open(file_path, 'r', errors="replace") as file:
        content = file.read()
        start_pos = content.find(start_str) - 3
        end_pos = content.find(end_str, start_pos)
        end_pos = content.find("\n", end_pos)
        data = content[start_pos:end_pos]
        lines = data.split('\n')
        formatted_lines = []
        max_width = every
        remove_lines = False

        for i,line in enumerate(lines):
            if line.strip().startswith('Heatmap generation:'):
                for word in lines[i+1].split():
                    if '/' in word:
                        n_tries = word.split('/')[1]
                        break
                remove_lines = True
                formatted_lines.append(f'   - {n_tries} models were tested, for more information check the {os.path.basename(file_path)} file in the GENERATE folder')
            if line.strip().startswith('o  Heatmap ML models no PFI filter succesfully created'):
                remove_lines = False
                formatted_lines.append('')
            if not remove_lines:
                if line.startswith('- '):
                    line = '    ' + line[4:]
                if '-------' in line:
                    formatted_line = '<strong>' + line + '</strong>'
                else:
                    formatted_line = textwrap.fill(line, width=max_width, subsequent_indent=' '*4)
                formatted_lines.append(formatted_line)

        return '\n'.join(formatted_lines)
