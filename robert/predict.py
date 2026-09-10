"""
Parameters
----------

    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    params_dir : str, default=''
        Folder containing the database and parameters of the ML model.
    csv_test : str, default=''
        Name of the CSV file containing the test set (if any). A path can be provided (i.e. 
        'C:/Users/FOLDER/FILE.csv'). 
    t_value : float, default=2
        t-value that will be the threshold to identify outliers (check tables for t-values elsewhere).
        The higher the t-value the more restrictive the analysis will be (i.e. there will be more 
        outliers with t-value=1 than with t-value = 4).
    alpha : float, default=0.05
        Significance level, or probability of making a wrong decision. This parameter is related to
        the confidence intervals (i.e. 1-alpha is the confidence interval). By default, an alpha value
        of 0.05 is used, which corresponds to a confidence interval of 95%.
    shap_show : int, default=10,
        Number of descriptors shown in the plot of the SHAP analysis.
    pfi_show : int, default=10,
        Number of descriptors shown in the plot of the PFI analysis.
    pfi_epochs : int, default=5,
        Sets the number of times a feature is randomly shuffled during the PFI analysis
        (standard from sklearn webpage: 5).
    names : str, default=''
        Column of the names for each datapoint. Names are used to print outliers.

"""
#####################################################.
#        This file stores the PREDICT class         #
#    used to analyze and generate ML predictors     #
#####################################################.

import os
import time
from robert.predict_utils import (plot_predictions,
    save_predictions,
    print_predict,
    pearson_map_predict
    )
from robert.utils import (load_variables,
    load_db_n_params,
    load_n_predict,
    load_model,
    _apply_full_refit_split_conformal,
    finish_print,
    print_pfi,
    PFI_plot,
    linear_equation,
    shap_analysis,
    outlier_plot,
    distribution_plot,
    boundary_plot,
    applicability_domain_plot,
    build_params_dirs,
    Logger,
)

class predict:
    """
    Class containing all the functions from the PREDICT module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the PREDICT module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "predict")

        # if params_dir = '', the program performs the tests for the No_PFI and PFI folders
        # (or, with --all_models, for every model staged under GENERATE/All_models)
        params_dirs, suffixes, suffix_titles, model_names = build_params_dirs(self)

        # --all_models: each model gets its own PREDICT_{model}_data.dat instead of merging
        # every model's output into one shared log, where REPORT would have no reliable way
        # to tell which lines belong to which model. The original shared log (already holds
        # the run header from load_variables) is closed right away in that case.
        default_log = self.args.log
        if any(m is not None for m in model_names):
            default_log.finalize()

        # model_names repeats each model once per PFI variant (No_PFI then PFI) - only
        # open a new Logger when the model actually changes, otherwise the PFI pass would
        # re-open (and truncate, Logger uses mode 'w') the same file the No_PFI pass just
        # wrote, silently discarding the No_PFI block
        current_model = None
        model_time_start = None
        model_durations = {}
        for (params_dir,suffix,suffix_title,model_name) in zip(params_dirs,suffixes,suffix_titles,model_names):
            if os.path.exists(params_dir):

                if model_name is not None and model_name != current_model:
                    if self.args.log is not default_log:
                        self.args.log.finalize()
                    if current_model is not None:
                        model_durations[current_model] = time.time() - model_time_start
                    self.args.log = Logger(self.args.destination / f'PREDICT_{model_name}', 'data')
                    current_model = model_name
                    model_time_start = time.time()

                _ = print_pfi(self,params_dir)

                # load the Xy databse and model parameters
                Xy_data, model_data, suffix_title = load_db_n_params(self,params_dir,suffix,suffix_title,"verify",True) # module 'verify' since PREDICT follows similar protocols
                
                # get results from training, test and external test (if any)
                Xy_data = load_n_predict(self, model_data, Xy_data, BO_opt=False)
                if getattr(self.args, "_api_predict", False):
                    loaded_model = load_model(self, model_data['model'], **model_data['params'])
                    Xy_data = _apply_full_refit_split_conformal(
                        self,
                        model_data,
                        Xy_data,
                        loaded_model,
                        y_cv_mean_train=Xy_data["y_pred_train"],
                        overwrite_predictions=True,
                    )

                # save predictions for all sets
                path_n_suffix, name_points, Xy_data = save_predictions(self,Xy_data,model_data,suffix_title)

                # represent y vs predicted y
                colors = plot_predictions(self,model_data,Xy_data,path_n_suffix)

                # print results
                _ = print_predict(self,Xy_data,model_data,suffix_title)

                # boundary robustness plots (Low/High sorted CV + bias) for the ROBERT report -
                # logged right after "Summary of results" so report_utils.get_predict_scores()
                # can find them nearby
                if model_data['type'].lower() == 'reg':
                    _ = boundary_plot(self,model_data,Xy_data,path_n_suffix)
                    _ = applicability_domain_plot(self,model_data,Xy_data,path_n_suffix)

                # fit once and reuse for the linear equation/SHAP/PFI analyses below instead of
                # each of them independently loading and refitting the same model on the same data
                fitted_model = load_model(self, model_data['model'], **model_data['params'])
                fitted_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])

                # linear model equation (MVL only) - must be logged before SHAP/PFI,
                # REPORT's print_features() reads it back out of the log to show it above those plots
                if model_data['model'].upper() == 'MVL':
                    _ = linear_equation(self,Xy_data,model_data,fitted_model)

                # SHAP analysis
                _ = shap_analysis(self,Xy_data,model_data,path_n_suffix,fitted_model)

                # PFI analysis
                _ = PFI_plot(self,Xy_data,model_data,path_n_suffix,fitted_model)

                # create Pearson heatmap - unlike path_n_suffix-based outputs above, this
                # builds its own filename directly from suffix_title, so the model name has
                # to be included here explicitly (matches the same f"{model}_{suffix_title}"
                # convention used everywhere else, e.g. save_predictions())
                _ = pearson_map_predict(self,Xy_data,params_dir,f"{model_data['model']}_{suffix_title}")

                # Outlier analysis
                if model_data['type'].lower() == 'reg':
                    _ = outlier_plot(self,Xy_data,path_n_suffix,name_points,colors)

                # y distribution
                _ = distribution_plot(self,Xy_data,path_n_suffix,model_data)

        # --all_models: each model gets its own "Time PREDICT" line reflecting only the time
        # spent processing that model (not the cumulative time for every model), since each
        # model's own PDF report should show its own PREDICT duration. finish_print() can't be
        # reused here because it always measures from the original start_time (the whole
        # multi-model run), and it also only writes into whichever log is CURRENTLY open (the
        # last model processed) - every other model's log needs its line appended separately
        # to its own (already-closed) file
        if current_model is not None:
            model_durations[current_model] = time.time() - model_time_start
            for m, duration in model_durations.items():
                if m == current_model:
                    continue
                with open(self.args.destination / f'PREDICT_{m}_data.dat', 'a', encoding='utf-8') as f:
                    f.write(f"\nTime PREDICT: {round(duration, 2)} seconds\n")
            self.args.log.write(f"\nTime PREDICT: {round(model_durations[current_model], 2)} seconds\n")
            self.args.log.finalize()
        else:
            _ = finish_print(self,start_time,'PREDICT')
