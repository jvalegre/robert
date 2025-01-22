#####################################################.
#      This file contains the argument parser       #
#####################################################.

import sys

var_dict = {
    "varfile": None,
    "command_line": False,
    "extra_cmd": '',
    "curate": False,
    "generate": False,
    "predict": False,
    "verify": False,
    "aqme": False,
    "report": False,
    "cheers": False,
    "evaluate": False,
    "seed": 0,
    "destination": None,
    "csv_name" : '',
    "csv_train": '',
    "csv_valid": '',
    "csv_test": '',
    "y" : '',
    "discard" : [],
    "ignore" : [],
    "categorical" : "onehot",
    "corr_filter" : True,
    "std" : True,
    "desc_thres" : 25,
    "thres_y" : 0.001,
    "thres_x" : 0.7,
    "test_set" : 0.1,
    "auto_test" : True,
    "auto_type": True,
    "model" : ['RF','GB','NN','MVL'],
    "eval_model" : 'MVL',
    "custom_params" : None,
    "type" : "reg",
    "split" : "even",
    "nprocs": 8,
    "error_type" : "rmse",
    "pfi_epochs" : 5,
    "pfi_threshold" : 0.04,
    "pfi_filter" : True,
    "pfi_max" : 0,
    "init_points" : 10,
    "n_iter" : 10,
    "expect_improv" : 0.05,
    "kfold" : 5,
    "repeat_kfolds" : 'auto',
    "alpha" : 0.05,
    "params_dir" : '',
    "csv_test" : '',
    "t_value" : 2,
    "shap_show" : 10,
    "pfi_show" : 10,
    "names" : '',
    "qdescp_keywords" : '',
    "descp_lvl": "interpret",
    "report_modules" : ['AQME','CURATE','GENERATE','VERIFY','PREDICT'],
    "debug_report": False
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
            sys.exit()
 
    return options
