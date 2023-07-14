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

    # AQME
    if args.aqme:
        # save the csv_name and y values from AQME workflows
        args = missing_inputs(args,print_err=True)

        full_workflow = True
        aqme(
            csv_name=args.csv_name,
            varfile=args.varfile,
            y=args.y,
            command_line=args.command_line,
            destination=args.destination,
            qdescp_keywords=args.qdescp_keywords,
            csearch_keywords=args.csearch_keywords,
            discard=args.discard,
            ignore=args.ignore
        )

        # adjust argument names after running AQME
        args = set_aqme_args(args)

    # CURATE
    if args.curate or full_workflow:
        curate(
            varfile=args.varfile,
            command_line=args.command_line,
            destination=args.destination,
            csv_name=args.csv_name,
            y=args.y,
            discard=args.discard,
            ignore=args.ignore,
            categorical=args.categorical,
            corr_filter=args.corr_filter,
            desc_thres=args.desc_thres,
            thres_x=args.thres_x,
            thres_y=args.thres_y,
        )

    if full_workflow:
        args.y = '' # this ensures GENERATE communicates with CURATE (see the load_variables() function in utils.py)
        args.discard = [] # avoids an error since the variable(s) are removed in CURATE

    # GENERATE
    if args.generate or full_workflow:
        generate(
            varfile=args.varfile,
            command_line=args.command_line,
            destination=args.destination,
            csv_name=args.csv_name,
            y=args.y,
            discard=args.discard,
            ignore=args.ignore,
            train=args.train,
            split=args.split,
            model=args.model,
            type=args.type,
            seed=args.seed,
            generate_acc=args.generate_acc,
            filter_train=args.filter_train,
            epochs=args.epochs,
            error_type=args.error_type,
            custom_params=args.custom_params,
            pfi_epochs=args.pfi_epochs,
            pfi_threshold=args.pfi_threshold,
            pfi_filter=args.pfi_filter,
            pfi_max=args.pfi_max,
        )

    # VERIFY
    if args.verify or full_workflow:
        verify(
            varfile=args.varfile,
            command_line=args.command_line,
            destination=args.destination,
            params_dir=args.params_dir,
            thres_test=args.thres_test,
            kfold=args.kfold,
        )

    # PREDICT
    if args.predict or full_workflow:
        predict(
            varfile=args.varfile,
            command_line=args.command_line,
            destination=args.destination,
            params_dir=args.params_dir,
            csv_test=args.csv_test,
            t_value=args.t_value,
            shap_show=args.shap_show,
            pfi_epochs=args.pfi_epochs,
            pfi_show=args.pfi_show,
            names=args.names
        )

    # REPORT
    if args.report or full_workflow:
        report(
            varfile=args.varfile,
            command_line=args.command_line,
            destination=args.destination,
            report_modules=args.report_modules
        )
    
    # CHEERS
    if args.cheers:
        print('o  Blimey! This module was designed to thank my mate ROBERT Paton, who was a mentor to me throughout my years at Colorado State University, and who introduced me to the field of cheminformatics.\n')


def set_aqme_args(args):
    """
    Changes arguments to couple AQME with ROBERT
    """

    # set the path to the database created by AQME to continue in the full_workflow
    args.csv_name = Path(os.path.dirname(args.csv_name)).joinpath(f'AQME-ROBERT_{os.path.basename(args.csv_name)}')
    aqme_df = pd.read_csv(args.csv_name)

    # ignore the names and SMILES of the molecules
    for column in aqme_df.columns:
        if column.lower() in ['smiles','code_name'] and column not in args.ignore:
            args.ignore.append(column)

    # set the names for the outlier analysis
    for column in aqme_df.columns:
        if column.lower() == 'code_name' and args.names == '':
            args.names = column

    return args

if __name__ == "__main__":
    main()
