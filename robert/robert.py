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
###  (CHEERS) Acknowledges                                         ###
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
# from robert.outliers import outliers
# from robert.predict import predict
from robert.utils import command_line_args


def main():
    """
    Main function of AQME, acts as the starting point when the program is run through a terminal
    """

    # load user-defined arguments from command line
    args = command_line_args()
    args.command_line = True

    if not args.curate and not args.generate and not args.predict and not args.verify:
        print('x  The --module option was not specified in the command line! Options: "curate", "generate", "verify", "predict".\n')

    # CURATE
    if args.curate:
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

    # GENERATE
    if args.generate:
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
            mode=args.mode,
            seed=args.seed,
            epochs=args.epochs,
            hyperopt_target=args.hyperopt_target,
            custom_params=args.custom_params,
            PFI_epochs=args.PFI_epochs,
            PFI_threshold=args.PFI_threshold,
            PFI=args.PFI
        )

    # CHEERS
    if args.cheers:
        print('o  Blimey, this module was designed to thank my mate ROBERT Paton, who was a mentor to me throughout my years at Colorado State University, and who introduced me to the field of cheminformatics.\n')


if __name__ == "__main__":
    main()
