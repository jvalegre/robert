"""
Parameters
----------

General
+++++++

    # define csv file that contains the database (without the .csv extension) and the response value
    w_dir = os.getcwd()

    # name of the csv containing the database without the CSV extension. For example: csv_name = 'Phenolic_data' 
    csv_name = 'Robert_example'

    # name of the csv file that will contain the optimal parameters
    name_csv_hyperopt = 'Predictor_parameters'

    # specify the response value (y), for example: response_value = 'activation_barrier_kcal/mol'
    response_value = 'Target_values'

    # specify columns of the csv to drop from the descriptors but to keep in the final database
    # (i.e. reaction names). For example: fixed_descriptors = ['Name','SMILES','YSI/MW','YSI','CN','MW','weakest_bondtype'].
    # If there are not descriptors to discard, just use fixed_descriptors = []
    fixed_descriptors = ['Name']

    # convert columns with strings into categorical values using 1,2,3... (alternative
    # to one-hot encoding that the code uses by default)
    categorical_mode = False

    # activate with correlation_filter = True
    correlation_filter = True

    # threshold values for the correlation filters (if correlation_filter = True)
    correlation_y_threshold = 0.02 # (only use descriptors that correlate with R**2 > 0.02 with the response value)
    correlation_x_threshold = 0.85 # (only use descriptors that don't correlate with R**2 > 0.85 with other descriptors)

   files : str or list of str, default=None
     Input files. Formats accepted: XYZ, SDF, GJF, COM and PDB. Also, lists can
     be used (i.e. [FILE1.sdf, FILE2.sdf] or \*.FORMAT such as \*.sdf).  
   program : str, default=None
     Program required in the conformational refining. 
     Current options: 'xtb', 'ani'
"""
#####################################################.
#        This file stores the PREDICT class         #
#              used in the predictor                #
#####################################################.

import os
import sys
import time
from pathlib import Path
from scipy import stats
from robert.utils import load_variables


class predict:
    """
    Class containing all the functions from the PREDICT module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the PREDICT module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time_overall = time.time()
        # load default and user-specified variables
        self.args = load_variables(kwargs, "curate")

        cmin_program = True
        if self.args.program is None:
            cmin_program = False
        if cmin_program:
            if self.args.program.lower() not in ["xtb", "ani"]:
                cmin_program = False
        if not cmin_program:
            self.args.log.write('\nx  Program not supported for CMIN refinement! Specify: program="xtb" (or "ani")')
            self.args.log.finalize()
            sys.exit()

        try:
            os.chdir(self.args.w_dir_main)
        except FileNotFoundError:
            self.args.w_dir_main = Path(f"{os.getcwd()}/{self.args.w_dir_main}")
            os.chdir(self.args.w_dir_main)

        # retrieves the different files to run in CMIN
        if len(self.args.files) == 0:
            self.args.log.write('\nx  No files were found! Make sure you use quotation marks if you are using * (i.e. --files "*.sdf")')
            self.args.log.finalize()
            sys.exit()


        elapsed_time = round(time.time() - start_time_overall, 2)
        self.args.log.write(f"\nTime CMIN: {elapsed_time} seconds\n")
        self.args.log.finalize()

        # this is added to avoid path problems in jupyter notebooks
        os.chdir(self.args.initial_dir)


