#####################################################.
#      This file contains the argument parser       #
#####################################################.

import os

var_dict = {
    "varfile": None,
    "command_line": False,
    "curate": False,
    "generate": False,
    "outliers": False,
    "predict": False,
    "seed": 62609,
    "destination": None,
    "csv_name" : '',
    "csv_params" : 'ML_params',
    "y" : '',
    "discard" : [],
    "ignore" : [],
    "categorical" : "onehot",
    "corr_filter" : True,
    "thres_y" : 0.02,
    "thres_x" : 0.85,
    "train" : [60,70,80,90],
    "split" : "KN",
    "model" : ['RF','GB','NN', 'VR'],
    "mode" : "reg",
    "seed" : 0,
    "epochs" : 500,
    "hyperopt_target" : "rmse",
    "PFI_epochs" : 30,
    "PFI_threshold" : 0.04,
    "PFI" : True,
}

# part for using the options in a script or jupyter notebook
class options_add:
    pass


def set_options(kwargs):
    # set default options and options provided
    options = options_add()
    # dictionary containing default values for options

    for key in var_dict:
        vars(options)[key] = var_dict[key]
    for key in kwargs:
        if key in var_dict:
            vars(options)[key] = kwargs[key]
        elif key.lower() in var_dict:
            vars(options)[key.lower()] = kwargs[key.lower()]
        else:
            print("Warning! Option: [", key,":",kwargs[key],"] provided but no option exists, try the online documentation to see available options for each module.",)

    return options
