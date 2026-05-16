#####################################################.
#      This file contains the argument parser       #
#####################################################.

import copy
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
    "csv_test": '',
    "y" : '',
    "discard" : [],
    "ignore" : [],
    "categorical" : "onehot",
    "corr_filter_x" : True,
    "corr_filter_y" : False,
    "std" : True,
    "desc_thres" : 25,
    "thres_y" : 0.001,
    "thres_x" : 0.7,
    "test_set" : 0.2,
    "auto_test" : True,
    "auto_type": True,
    "auto_fill": True,
    "model" : ['RF','GB','NN','MVL'],
    "eval_model" : 'MVL',
    "custom_params" : None,
    "type" : "reg",
    "split" : "auto",
    "nprocs": 8,
    "error_type" : "rmse",
    "pfi_epochs" : 5,
    "pfi_threshold" : 0.2,
    "pfi_filter" : True,
    "pfi_max" : 0,
    "init_points" : 10,
    "n_iter" : 10,
    "expect_improv" : 0.05,
    "kfold" : 5,
    "repeat_kfolds" : 10,
    "alpha" : 0.05,
    "params_dir" : '',
    "t_value" : 2,
    "shap_show" : 10,
    "pfi_show" : 10,
    "names" : '',
    "qdescp_keywords" : '',
    "descp_lvl": "interpret",
    "report_modules" : ['AQME','CURATE','GENERATE','VERIFY','PREDICT'],
    "debug_report": False,
    # Split conformal (regression): symmetric interval half-width.
    "conformal_enable": True,
    "conformal_calib_frac": 0.15,
    "conformal_coverage": 0.9,
    # Top-k meta-model uncertainty (Python API / optional PREDICT path).
    "uq_enable_meta": False,
    "uq_top_k_models": 3,
    "uq_model_weighting": "score_weighted",
    # Auto uncertainty selection (regression; opt-in).
    "uq_auto_enable": False,
    "uq_auto_candidates": ["cv_sd", "conformal", "meta_total"],
    "uq_auto_scaler": "global_multiplicative",
    "uq_auto_metric_weights": {"coverage": 1.0, "sharpness": 0.25, "nll": 0.5},
    "uq_auto_min_samples": 12,
    "uq_auto_random_state": 0,
    "uq_auto_clas_mode": "error",
    # When False, skip SHAP/PFI/heatmap/outlier/distribution plots (PREDICT only).
    "predict_diagnostics": True,
    # Plot verbosity: 0 = no figures, 1 = workflow summaries (CURATE/GENERATE/VERIFY +
    # main PREDICT result plots when predict_diagnostics is True), 2 = full diagnostics.
    "plot_verbosity": 2,
}


# part for using the options in a script or jupyter notebook
class options_add:
    pass


def set_options(kwargs):
    # set default options and options provided
    options = options_add()
    # dictionary containing default values for options

    for key in var_dict:
        vars(options)[key] = copy.deepcopy(var_dict[key])
    for key in kwargs:
        if key in var_dict:
            vars(options)[key] = kwargs[key]
        elif key.lower() in var_dict:
            vars(options)[key.lower()] = kwargs[key.lower()]
        else:
            print("Warning! Option: [", key,":",kwargs[key],"] provided but no option exists, try the online documentation to see available options for each module.",)
            sys.exit()

    return options
