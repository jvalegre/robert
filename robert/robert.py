#!/usr/bin/env python

######################################################################.
######################################################################
###                                                                ###
###  ROBERT is a tool that allows to carry out automated:          ###
###  (CURATE) Curate the data                                      ###
###  (GENERATE) Optimize the ML model                              ###
###  (VERIFY) ML model analysis                                    ###
###  (PREDICT) Predict new data                                    ###
###  (AQME) AQME-ROBERT workflow                                   ###
###  (REPORT) Creates a report with the results                    ###
###  (CHEERS) Acknowledgements                                     ###
###                                                                ###
######################################################################
###                                                                ###
###  Authors: Juan V. Alegre Requena, David Dalmau Ginesta         ###
###                                                                ###
###  Please, report any bugs or suggestions to:                    ###
###  jv.alegre@csic.es                                             ###
###                                                                ###
######################################################################
######################################################################.


import os
from pathlib import Path
import pandas as pd
from robert.curate import curate
from robert.generate import generate
from robert.verify import verify
from robert.predict import predict
from robert.report import report
from robert.aqme import aqme
from robert.utils import (command_line_args,missing_inputs)


def main():
    """
    Main function of ROBERT, acts as the starting point when the program is run through a terminal
    """

    # load user-defined arguments from command line
    args = command_line_args()
    args.command_line = True

    # if no modules are called, the full workflow is activated
    full_workflow = False
    if not args.curate and not args.generate and not args.predict:
        if not args.cheers and not args.verify and not args.report:
            full_workflow = True
    
    if args.aqme:
        full_workflow = True

        # adjust argument names after running AQME
        args = set_aqme_args(args)

    # save the csv_name, y and names values from full workflows
    if full_workflow:
        args = missing_inputs(args,'full_workflow',print_err=True)

    # AQME
    if args.aqme:
        aqme(**vars(args))

    # CURATE
    if args.curate or full_workflow:
        curate(**vars(args))

    if full_workflow:
        args.y = '' # this ensures GENERATE communicates with CURATE (see the load_variables() function in utils.py)
        args.discard = [] # avoids an error since the variable(s) are removed in CURATE

    # GENERATE
    if args.generate or full_workflow:
        generate(**vars(args))

    # VERIFY
    if args.verify or full_workflow:
        verify(**vars(args))

    # PREDICT
    if args.predict or full_workflow:
        predict(**vars(args))

    # REPORT
    if args.report or full_workflow:
        report(**vars(args))
    
    # CHEERS
    if args.cheers:
        print('o  This module was designed to thank ROBERT Paton, who was a mentor to me throughout my years at Colorado State University, and who introduced me to the field of cheminformatics. Cheers mate!\n')


def set_aqme_args(args):
    """
    Changes arguments to couple AQME with ROBERT
    """

    # set the path to the database created by AQME to continue in the full_workflow
    args.csv_name = Path(os.path.dirname(args.csv_name)).joinpath(f'AQME-ROBERT_{os.path.basename(args.csv_name)}')
    aqme_df = pd.read_csv(args.csv_name)

    # ignore the names and SMILES of the molecules
    for column in aqme_df.columns:
        if column.lower() in ['smiles','code_name','charge','mult','complex_type','geom','constraints_atoms','constraints_dist','constraints_angle','constraints_dihedral'] and column not in args.ignore:
            args.ignore.append(column)

    # set the names for the outlier analysis
    for column in aqme_df.columns:
        if column.lower() == 'code_name' and args.names == '':
            args.names = column

    return args

if __name__ == "__main__":
    main()
