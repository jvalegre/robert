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
import sys
import shutil
from pathlib import Path
import pandas as pd
from robert.curate import curate
from robert.generate import generate
from robert.verify import verify
from robert.predict import predict
from robert.report import report
from robert.aqme import aqme
from robert.evaluate import evaluate
from robert.utils import (command_line_args,missing_inputs)


def main(exe_type='command',sys_args=None):
    """
    Main function of ROBERT, acts as the starting point when the program is run through a terminal
    """

    # load user-defined arguments from command line
    args = command_line_args(exe_type,sys_args)
    args.command_line = True

    if not args.evaluate:
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
            # remove the EVALUATE folder to avoid issues when generating the report PDF
            eval_folder = Path(f'{os.getcwd()}/EVALUATE')
            if os.path.exists(eval_folder):
                shutil.rmtree(eval_folder)

            args = missing_inputs(args,'full_workflow',print_err=True)

        # AQME
        if args.aqme:
            aqme(**vars(args))
            # set the path to the database created by AQME to continue in the full_workflow
            args.csv_name = Path(os.path.dirname(args.csv_name)).joinpath(f'AQME-ROBERT_{args.descp_lvl}_{os.path.basename(args.csv_name)}')
            if args.csv_test != '':
                args.csv_test = Path(os.path.dirname(args.csv_test)).joinpath(f'AQME-ROBERT_{args.descp_lvl}_{os.path.basename(args.csv_test)}')

        # CURATE
        if args.curate or full_workflow:
            curate(**vars(args))

        if full_workflow:
            # this ensures GENERATE communicates with CURATE (see the load_variables() function in utils.py)
            args.y = ''
            args.discard = [] # avoids an error since the variable(s) are removed in CURATE
            args.csv_name = '' # force GENERATE to use the curated database

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

    # EVALUATE, only evaluates models
    else:
        # EVALUATE, which also creates the files of GENERATE
        evaluate(**vars(args))

        # adjust variables to make them compatible with regular ROBERT workflows
        args = set_eval_vars(args)

        # CURATE, only for the Pearson corr matrix (print if there are 30 or less points)
        curate(**vars(args))

        # VERIFY
        verify(**vars(args))

        # PREDICT
        predict(**vars(args))

        # REPORT
        report(**vars(args))


def set_aqme_args(args):
    """
    Changes arguments to couple AQME with ROBERT
    """

    if os.path.exists(args.csv_name):
        aqme_df = pd.read_csv(args.csv_name, encoding='utf-8')
    else:
        print(f'\nx  The path of your CSV file doesn\'t exist! You specified: {args.csv_name}')
        sys.exit()

    # list of potential arguments from CSV inputs in AQME
    aqme_args = ['smiles','charge','mult','complex_type','geom','constraints_atoms','constraints_dist','constraints_angle','constraints_dihedral']

    # ignore the names and SMILES of the molecules
    remove = []
    for column in args.ignore:
        if column.lower() in aqme_args:
            remove.append(column)
    for column in remove:
        args.ignore.remove(column)
    if 'code_name' in args.ignore:
        args.ignore.remove('code_name')
    for column in aqme_df.columns:
        if column.lower() == 'code_name' and args.names == '':
            args.names = column

    return args


def set_eval_vars(args):
    '''
    Adjust variables to make them compatible with regular ROBERT workflows
    '''
    
    args.ignore.append('Set')
    csv_basename = os.path.basename(f'{args.csv_train}').split('.')[0]
    args.csv_name = f'{csv_basename}_EVAL_db.csv'
    args.csv_test = ''

    return args


if __name__ == "__main__":
    main()
