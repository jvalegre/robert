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
from pathlib import Path
from weasyprint import HTML
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
        _ = self.init_report()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "report")

        # create css
        with open("report.css", "w") as cssfile:
            cssfile.write(css_content())

        # create report
        report_html = ''
        report_html += self.get_header(self.args.report_modules)
        report_html += self.get_data(self.args.report_modules)
        _ = make_report(report_html)

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
                data_lines += f"""
                    <p>
                    <hr>
                    <p><span style="font-weight:bold;">
                        <img src="file:///{self.args.path_icons}/{module}.png" alt="" style="width:20px; height:20px; margin-right:5px;">
                        {module}
                    </span></p>
                    {module_data}
                    {html_png}
                    """

                if i == len(modules) - 1:
                    data_lines += f"""
                        <p>
                        <hr class="black">
                    """

        return data_lines



    def init_report(self):
        '''
        Checks whether weasyprint works to make the report
        '''
        try:
            from weasyprint import HTML
        except (OSError, ModuleNotFoundError):
            print(f"\n  x The REPORT module requires weasyprint but this module is missing, the PDF with the summary of the results has not been created. Try installing ROBERT with 'conda install -c conda-forge robert'")


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
            <h1 style="text-align: left; margin-bottom: 0.5em;">
                Report
                <img src="file:///{self.args.path_icons}/Robert_logo.jpg" alt="" style="float:right; width:25%;" />
                <br>
                <hr class="black">
                <span style="font-weight:bold;"></span>
            </h1>
            <p style="text-align: center;"></p>
            <p style="text-align: justify;"
            <div class="dat-content"><pre>{header_dat}</pre></div>
            """

        return header_lines


def make_pdf(html, css_files=None):
    """Generate a PDF file from a string of HTML."""
    htmldoc = HTML(string=html, base_url="")
    if css_files:
        htmldoc = htmldoc.write_pdf(stylesheets=css_files)
    else:
        htmldoc = htmldoc.write_pdf()
    return htmldoc


def make_report(report_html):
    css_files = ["report.css"]
    outfile = f"{os.getcwd()}/ROBERT_report.pdf"
    if os.path.exists(outfile):
        os.remove(outfile)
    pdf = make_pdf(report_html, css_files=css_files)
    _ = Path(outfile).write_bytes(pdf)


def css_content():
    css_content = """
    body {
    font-size: 12px;
    }
    @page {
        size: A4;
        margin: 2cm;
        @bottom-right {
            content: counter(page);
            font-size: 8pt;
            position: fixed;
            right: 0;
            bottom: 0;
        }
    }
    * {
        font-family: Arial, sans-serif;
    }
    .dat-content {
        width: 50%;
        max-width: 595pt;
        overflow-x: auto;
        line-height: 1.2;
    }

    img[src="Robert_logo.jpg"] {
        float: right;
        width: 50%;
    }
    img[src*="Pearson"] {
        display: inline-block;
        vertical-align: bottom;
        max-width: 48%;
        margin-left: 10px;
        margin-bottom: -5px;
    }

    img[src*="PFI"] {
        display: inline-block;
        vertical-align: bottom;
        max-width: 48%;
        margin-left: 10px;
        margin-bottom: -5px;
    }

    img[src*="PFI"]:first-child {
        margin-right: 10%;
    }
    .img-PREDICT {
        margin-top: 20px;
    }
    
    hr.black {
    border: none;
    height: 1px;
    background-color: black;
    }
    hr {
    border: none;
    height: 1px;
    background-color: gray;
    }

    body:before {
    top: 1.2cm;
    }
    """
    return css_content


def insert_newlines(string, every=100):
    """Inserts a newline character every `every` characters in the string."""
    return '\n'.join(string[i:i+every] for i in range(0, len(string), every))


def format_data(file_path, start_str, end_str, every=100):
    """Reads a file and returns a formatted string between two markers."""
    with open(file_path, 'r', errors="replace") as file:
        content = file.read()
        start_pos = content.find(start_str) - 3
        end_pos = content.find(end_str, start_pos)
        end_pos = content.find("\n", end_pos)
        data = content[start_pos:end_pos]
        lines = data.split('\n')
        formatted_lines = [
            insert_newlines('    ' + line[4:] if line.startswith('- ') else line, every)
            for line in lines
        ]
        return '\n'.join(formatted_lines)