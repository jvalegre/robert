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
###  (CHEERS) Acknowledgement                                      ###
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


from robert.curate import curate
from robert.generate import generate
from robert.verify import verify
from robert.predict import predict
from robert.utils import command_line_args


def main():
    """
    Main function of AQME, acts as the starting point when the program is run through a terminal
    """

    # load user-defined arguments from command line
    args = command_line_args()
    args.command_line = True

    full_workflow = False
    if not args.curate and not args.generate and not args.predict and not args.verify:
        if not args.cheers:
            full_workflow = True

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
            thres_x=args.thres_x,
            thres_y=args.thres_y,
        )

    if full_workflow:
        args.y = '' # this ensures GENERATE communicates with CURATE (see the load_variables() function in utils.py)

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
            error_type=args.error_type,
            seed=args.seed
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
            seed=args.seed,
            shap_show=args.shap_show,
            pfi_epochs=args.pfi_epochs,
            pfi_show=args.pfi_show,
            names=args.names
        )


    # CHEERS
    if args.cheers:
        print('o  Blimey! This module was designed to thank my mate ROBERT Paton, who was a mentor to me throughout my years at Colorado State University, and who introduced me to the field of cheminformatics.\n')


if __name__ == "__main__":
    main()
