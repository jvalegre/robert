"""
Parameters
----------

General
+++++++

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
            cssfile.write(css_content())

        # create report
        report_html = ''
        report_html += self.get_header(self.args.report_modules)
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
            
            # Group PNGs in pairs based on their names within the "PREDICT" module
            if module == "PREDICT":
                results_images = []
                shap_images = []
                pfi_images = []

                for image_path in module_images:
                    filename = Path(image_path).stem
                    if "Results" in filename:
                        results_images.append(image_path)
                    elif "SHAP" in filename:
                        shap_images.append(image_path)
                    elif "PFI" in filename:
                        pfi_images.append(image_path)

                module_images = results_images + shap_images + pfi_images
            
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
                        <p><span style="font-weight:bold;">No PFI:</span></p>
                        <pre style="text-align: justify;">{module_data_predict}</pre>
                        """

                        # Column 2: PFI
                        column2 = f"""
                        <p><span style="font-weight:bold;">PFI:</span></p>
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
        for module in modules:
            dat_file = Path(f'{os.getcwd()}/{module}/{module}_data.dat')
            if os.path.exists(dat_file):
                with open(dat_file, 'r', errors="replace") as datfile:
                    header_dat = datfile.readline() + datfile.readline() + datfile.readline()
                    last_line = datfile.readline()
                    formatted_last_line = insert_newlines(last_line.strip(), 105)
                    header_dat += formatted_last_line
                    break

        header_lines = f"""
            <h1 style="text-align: center; margin-bottom: 0.5em;">
                <img src="file:///{self.args.path_icons}/Robert_logo.jpg" alt="" style="display: block; margin-left: auto; margin-right: auto; width: 50%; margin-top: -12px;" />
                <span style="font-weight:bold;"></span>
            <p>
            </h1>
            <p style="text-align: center;"></p>
            <p style="text-align: justify;">
            <div class="dat-content"><pre>{header_dat}</pre></div>
            """
        return header_lines


def make_report(report_html, HTML):
    css_files = ["report.css"]
    outfile = f"{os.getcwd()}/ROBERT_report.pdf"
    if os.path.exists(outfile):
        os.remove(outfile)
    pdf = make_pdf(report_html, HTML, css_files)
    _ = Path(outfile).write_bytes(pdf)


def make_pdf(html, HTML, css_files):
    """Generate a PDF file from a string of HTML."""
    htmldoc = HTML(string=html, base_url="")
    if css_files:
        htmldoc = htmldoc.write_pdf(stylesheets=css_files)
    else:
        htmldoc = htmldoc.write_pdf()
    return htmldoc


def css_content():
    # Obtain ROBERT version and CSV name
    dat_file = Path(f'{os.getcwd()}/GENERATE/GENERATE_data.dat')
    version = None
    csv_name = None

    if os.path.exists(dat_file):
        with open(dat_file, 'r', errors="replace") as datfile:
            for line in datfile:
                if '--csv_name' in line:
                    csv_name = line.split('--csv_name')[1].strip().split()[0]
                    if csv_name.endswith('.csv'):
                        break
                    else:
                        csv_name = None
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


def insert_newlines(string, every=105):
    """Inserts a newline character every `every` characters in the string."""
    wrapped_lines = textwrap.wrap(string, width=every)
    return '\n'.join(wrapped_lines)


def format_data(file_path, start_str, end_str, every=105):
    """Reads a file and returns a formatted string between two markers."""
    with open(file_path, 'r', errors="replace") as file:
        content = file.read()
        start_pos = content.find(start_str) - 3
        end_pos = content.find(end_str, start_pos)
        end_pos = content.find("\n", end_pos)
        data = content[start_pos:end_pos]
        lines = data.split('\n')
        formatted_lines = []
        max_width = 110
        remove_lines = False

        for i,line in enumerate(lines):
            if line.strip().startswith('Heatmap generation:'):
                for word in lines[i+1].split():
                    if '/' in word:
                        n_tries = word.split('/')[1]
                        break
                remove_lines = True
                formatted_lines.append(f'   - {n_tries} models were tested, for more information check the {os.path.basename(file_path)} file in the GENERATE folder')
            elif line.strip().startswith('Heatmap ML models no PFI filter succesfully created'):
                remove_lines = False
            elif not remove_lines:
                if line.startswith('- '):
                    line = '    ' + line[4:]
                if '-------' in line:
                    formatted_line = '<strong>' + line + '</strong>'
                else:
                    formatted_line = textwrap.fill(line, width=max_width, subsequent_indent=' '*4)
                formatted_lines.append(formatted_line)

        return '\n'.join(formatted_lines)
