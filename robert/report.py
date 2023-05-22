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

import time
from robert.utils import (load_variables,
    finish_print
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
        _ = self.init_report()

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "report")
            
        # # load data from DAT files
        # def data_content(self):
        #     """
        #     Reads a file and returns a formatted string between two markers
        #     """

        #     start_str = X
        #     end_str = X
        #     insert_newlines = X
        #     for module in enumerate(self.args.report_modules):
        #         data_file = f'{os.getcwd()}/{module}_data.dat'
        #         if os.path.exists(data_file):
        #             XX si es el primer dat, haz el header XX
        #             header_content()
        #             module_content()
        #             with open(data_file, 'r', errors="replace") as file:
        #                 content = file.read()
        #                 start_pos = content.find(start_str) - 3
        #                 end_pos = content.find(end_str, start_pos)
        #                 end_pos = content.find("\n", end_pos)
        #                 data = content[start_pos:end_pos]
        #                 lines = data.split('\n')
        #                 formatted_lines = [
        #                     insert_newlines('    ' + line[4:] if line.startswith('- ') else line, every)
        #                     for line in lines
        #                 ]

        #     return '\n'.join(formatted_lines)


        # # load css format
        css_content = css_content_fun()

        _ = finish_print(self,start_time,'REPORT')


    def init_report(self):
        '''
        Checks whether weasyprint works to make the report
        '''

        try:
            from weasyprint import HTML
        except OSError:
            self.log.write(f"\n  x The REPORT module requires weasyprint, and it is not compatible with your installation. Try installing ROBERT with 'conda install -c conda-forge robert'")


def css_content_fun():
    '''
    Create content for css
    '''
    
    css_content = """
    body {
    font-size: 12px;
    }
    @page :first {
        @top-left {
            content: "";
        }
        @top-right {
            content: "";
        }
        @bottom-left {
            content: "ROBERT v 0.0.1";
            font-size: 8pt;
            position: fixed;
            left: 0;
            bottom: 0;
        }
        @bottom-right {
            content: counter(page) " of " counter(pages);
            font-size: 8pt;
            position: fixed;
            right: 0;
            bottom: 0;
        }
    }
    @page {
        size: A4;
        margin: 2cm;
        @top-left {
            content: "ROBERT report";
            font-size: 8pt;
            position: fixed;
            left: 0;
            top: 0;
        }
        @top-right {
            content: "CSV file name";
            font-size: 8pt;
            position: fixed;
            left: 0;
            top: 0;
        }
        @bottom-left {
            content: "ROBERT v 0.0.1";
            font-size: 8pt;
            position: fixed;
            left: 0;
            bottom: 0;
        }
        @bottom-right {
            content: counter(page) " of " counter(pages);
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
    .img-predict {
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
