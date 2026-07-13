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
import math
from robert.predict_utils import (plot_predictions,
    save_predictions,
    print_predict,
    pearson_map_predict
    )
from robert.json_output_for_agent import (
    init_module_audit,
    audit_set,
    audit_event,
    finalize_module_audit,
    agent_json_path,
    write_json_output_audit,
)
from robert.utils import (load_variables,
    load_db_n_params,
    load_n_predict,
    load_model,
    _apply_full_refit_split_conformal,
    finish_print,
    print_pfi,
    PFI_plot,
    shap_analysis,
    outlier_plot,
    distribution_plot,
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

        source_files = [str(self.args.params_dir)]
        if self.args.csv_test != '':
            source_files.append(str(self.args.csv_test))

        self.args.predict_audit = init_module_audit(
            module="PREDICT",
            artifact_type="predict_audit",
            source_files=source_files,
            command_line=getattr(self.args, "command_line", None),
        )
        predict_audit_path = agent_json_path("predict_audit.json")
        json_audit_path = agent_json_path("predict_json_output_audit.json")

        self.args.predict_audit = audit_set(self.args.predict_audit, "inputs", "params_dir", self.args.params_dir)
        self.args.predict_audit = audit_set(self.args.predict_audit, "inputs", "csv_test", self.args.csv_test)
        self.args.predict_audit = audit_set(self.args.predict_audit, "inputs", "t_value", self.args.t_value)
        self.args.predict_audit = audit_set(self.args.predict_audit, "inputs", "alpha", self.args.alpha)
        self.args.predict_audit = audit_set(self.args.predict_audit, "inputs", "shap_show", self.args.shap_show)
        self.args.predict_audit = audit_set(self.args.predict_audit, "inputs", "pfi_show", self.args.pfi_show)
        self.args.predict_audit = audit_set(self.args.predict_audit, "inputs", "pfi_epochs", self.args.pfi_epochs)
        self.args.predict_audit = audit_set(self.args.predict_audit, "inputs", "names", self.args.names)
        self.args.predict_audit = audit_set(self.args.predict_audit, "inputs", "destination", str(self.args.destination))

        # if params_dir = '', the program performs the tests for the No_PFI and PFI folders
        if 'GENERATE/Best_model' in self.args.params_dir:
            params_dirs = [f'{self.args.params_dir}/No_PFI',f'{self.args.params_dir}/PFI']
            suffixes = ['(with no PFI filter)','(with PFI filter)']
            suffix_titles = ['No_PFI','PFI']
        else:
            params_dirs = [self.args.params_dir]
            suffixes = ['custom']
            suffix_titles = ['custom']

        for (params_dir,suffix,suffix_title) in zip(params_dirs,suffixes,suffix_titles):
            branch_key = f"{suffix_title}::{params_dir}"
            params_dir_exists = bool(os.path.exists(params_dir))
            is_pfi_filtered = bool(str(suffix_title) == 'PFI')

            self.args.predict_audit = audit_event(
                self.args.predict_audit,
                event_type="predict_branch",
                payload={
                    "branch_key": str(branch_key),
                    "params_dir_used": str(params_dir),
                    "suffix": str(suffix),
                    "suffix_title": str(suffix_title),
                    "params_dir_exists": params_dir_exists,
                    "branch_is_pfi_filtered": is_pfi_filtered,
                    "workflow_includes_pfi_branch": bool('GENERATE/Best_model' in self.args.params_dir),
                },
                evidence_level="direct",
            )
            if params_dir_exists:

                _ = print_pfi(self,params_dir)

                # load the Xy databse and model parameters
                Xy_data, model_data, suffix_title = load_db_n_params(self,params_dir,suffix,suffix_title,"verify",True) # module 'verify' since PREDICT follows similar protocols

                active_descriptor_list = [str(desc) for desc in model_data.get('X_descriptors', [])] if isinstance(model_data.get('X_descriptors', []), list) else [str(desc) for desc in model_data.get('X_descriptors', [])]
                active_descriptor_count = int(len(active_descriptor_list))

                train_datapoints = None
                test_datapoints = None
                total_datapoints_loaded = None
                if 'y_train' in Xy_data:
                    train_datapoints = int(len(Xy_data['y_train']))
                elif 'X_train' in Xy_data and hasattr(Xy_data['X_train'], 'shape'):
                    train_datapoints = int(Xy_data['X_train'].shape[0])

                if 'y_test' in Xy_data:
                    test_datapoints = int(len(Xy_data['y_test']))
                elif 'X_test' in Xy_data and hasattr(Xy_data['X_test'], 'shape'):
                    test_datapoints = int(Xy_data['X_test'].shape[0])

                if 'X' in Xy_data and hasattr(Xy_data['X'], 'shape'):
                    total_datapoints_loaded = int(Xy_data['X'].shape[0])
                elif 'y' in Xy_data:
                    total_datapoints_loaded = int(len(Xy_data['y']))

                external_available = bool(
                    'y_external' in Xy_data
                    and hasattr(Xy_data['y_external'], 'isnull')
                    and not Xy_data['y_external'].isnull().values.any()
                    and len(Xy_data['y_external']) > 0
                )

                self.args.predict_audit = audit_event(
                    self.args.predict_audit,
                    event_type="model_context",
                    payload={
                        "branch_key": str(branch_key),
                        "params_dir_used": str(params_dir),
                        "suffix": str(suffix),
                        "suffix_title": str(suffix_title),
                        "model_name": str(model_data['model']),
                        "model_type": str(model_data['type']),
                        "error_type": str(model_data['error_type']),
                        "y_column": str(model_data['y']),
                        "names_column": str(model_data['names']),
                        "active_descriptor_count": active_descriptor_count,
                        "active_descriptor_list": active_descriptor_list,
                        "active_descriptor_source": "model_data['X_descriptors']",
                        "train_datapoints": train_datapoints,
                        "test_datapoints": test_datapoints,
                        "total_datapoints_loaded": total_datapoints_loaded,
                        "external_test_available": external_available,
                    },
                    evidence_level="direct",
                )
                
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

                def _mean_safe(values):
                    if values is None:
                        return None
                    try:
                        if len(values) == 0:
                            return None
                        mean_val = sum(values) / len(values)
                        if isinstance(mean_val, float) and (math.isnan(mean_val) or math.isinf(mean_val)):
                            return None
                        return float(mean_val)
                    except Exception:
                        return None

                measured_y_range = None
                if all(key in Xy_data for key in ['pred_min', 'pred_max', 'pred_range']):
                    measured_y_range = {
                        "min": float(Xy_data['pred_min']),
                        "max": float(Xy_data['pred_max']),
                        "range": float(Xy_data['pred_range']),
                    }

                self.args.predict_audit = audit_event(
                    self.args.predict_audit,
                    event_type="prediction_result_context",
                    payload={
                        "branch_key": str(branch_key),
                        "model_type": str(model_data['type']),
                        "regression_metrics": {
                            "cv": {
                                "r2": Xy_data.get('r2_train', None),
                                "mae": Xy_data.get('mae_train', None),
                                "rmse": Xy_data.get('rmse_train', None),
                            },
                            "test": {
                                "r2": Xy_data.get('r2_test', None),
                                "mae": Xy_data.get('mae_test', None),
                                "rmse": Xy_data.get('rmse_test', None),
                            },
                            "external": {
                                "r2": Xy_data.get('r2_external', None),
                                "mae": Xy_data.get('mae_external', None),
                                "rmse": Xy_data.get('rmse_external', None),
                            } if external_available else None,
                        },
                        "classification_metrics": {
                            "cv": {
                                "acc": Xy_data.get('acc_train', None),
                                "f1": Xy_data.get('f1_train', None),
                                "mcc": Xy_data.get('mcc_train', None),
                            },
                            "test": {
                                "acc": Xy_data.get('acc_test', None),
                                "f1": Xy_data.get('f1_test', None),
                                "mcc": Xy_data.get('mcc_test', None),
                            },
                            "external": {
                                "acc": Xy_data.get('acc_external', None),
                                "f1": Xy_data.get('f1_external', None),
                                "mcc": Xy_data.get('mcc_external', None),
                            } if external_available else None,
                        },
                        "prediction_sd_summary": {
                            "train_mean_sd": _mean_safe(Xy_data.get('y_pred_train_sd', None)),
                            "test_mean_sd": _mean_safe(Xy_data.get('y_pred_test_sd', None)),
                            "external_mean_sd": _mean_safe(Xy_data.get('y_pred_external_sd', None)) if external_available else None,
                        },
                        "conformal_half_width": Xy_data.get('conformal_half_width', None),
                        "measured_y_range": measured_y_range,
                    },
                    evidence_level="direct",
                )

                # save predictions for all sets
                path_n_suffix, name_points, Xy_data, save_metadata = save_predictions(self,Xy_data,model_data,suffix_title)

                self.args.predict_audit = audit_event(
                    self.args.predict_audit,
                    event_type="save_predictions",
                    payload={
                        "branch_key": str(branch_key),
                        "metadata": save_metadata,
                    },
                    evidence_level="direct",
                )

                # represent y vs predicted y
                colors = plot_predictions(self,model_data,Xy_data,path_n_suffix)

                if all(key in Xy_data for key in ['pred_min', 'pred_max', 'pred_range']):
                    self.args.predict_audit = audit_event(
                        self.args.predict_audit,
                        event_type="measured_y_range_post_plot",
                        payload={
                            "branch_key": str(branch_key),
                            "measured_y_range": {
                                "min": float(Xy_data['pred_min']),
                                "max": float(Xy_data['pred_max']),
                                "range": float(Xy_data['pred_range']),
                            },
                        },
                        evidence_level="direct",
                    )
                else:
                    self.args.predict_audit = audit_event(
                        self.args.predict_audit,
                        event_type="measured_y_range_post_plot",
                        payload={
                            "branch_key": str(branch_key),
                            "state": "skipped",
                            "reason": "pred_range_keys_not_available_after_plot_predictions",
                        },
                        evidence_level="derived",
                    )

                base_name = os.path.basename(path_n_suffix)
                base_dir = os.path.dirname(path_n_suffix)
                graph_type = 'regression' if model_data['type'].lower() == 'reg' else 'classification'
                external_plots_applicable = bool(
                    'y_external' in Xy_data
                    and hasattr(Xy_data['y_external'], 'isnull')
                    and not Xy_data['y_external'].isnull().values.any()
                    and len(Xy_data['y_external']) > 0
                )

                if graph_type == 'regression':
                    expected_plots = {
                        "results": f"{base_dir}/Results_{base_name}.png",
                        "cv_variability": f"{base_dir}/CV_variability_{base_name}.png",
                        "external": f"{base_dir}/csv_test/CV_variability_{base_name}_external.png" if external_plots_applicable else None,
                    }
                else:
                    expected_plots = {
                        "cv_train_valid": f"{base_dir}/CV_train_valid_predict_{base_name}.png",
                        "test": f"{base_dir}/Results_{base_name}_test.png",
                        "external": f"{base_dir}/csv_test/Results_{base_name}_external.png" if external_plots_applicable else None,
                    }

                plot_status = {}
                for plot_key, plot_path in expected_plots.items():
                    if plot_path is None:
                        plot_status[plot_key] = {"path": None, "state": "not_applicable", "exists": False}
                    else:
                        exists = bool(os.path.exists(plot_path))
                        plot_status[plot_key] = {
                            "path": str(plot_path),
                            "state": "created" if exists else "skipped",
                            "exists": exists,
                        }

                self.args.predict_audit = audit_event(
                    self.args.predict_audit,
                    event_type="predict_plots",
                    payload={
                        "branch_key": str(branch_key),
                        "graph_type": graph_type,
                        "plots": plot_status,
                    },
                    evidence_level="derived",
                )

                # print results
                print_results_text = print_predict(self,Xy_data,model_data,suffix_title)

                self.args.predict_audit = audit_event(
                    self.args.predict_audit,
                    event_type="print_predict_summary",
                    payload={
                        "branch_key": str(branch_key),
                        "model_type": str(model_data['type']),
                        "cv_metrics": {
                            "r2": Xy_data.get('r2_train', None),
                            "mae": Xy_data.get('mae_train', None),
                            "rmse": Xy_data.get('rmse_train', None),
                            "acc": Xy_data.get('acc_train', None),
                            "f1": Xy_data.get('f1_train', None),
                            "mcc": Xy_data.get('mcc_train', None),
                        },
                        "test_metrics": {
                            "r2": Xy_data.get('r2_test', None),
                            "mae": Xy_data.get('mae_test', None),
                            "rmse": Xy_data.get('rmse_test', None),
                            "acc": Xy_data.get('acc_test', None),
                            "f1": Xy_data.get('f1_test', None),
                            "mcc": Xy_data.get('mcc_test', None),
                        },
                        "external_metrics": {
                            "r2": Xy_data.get('r2_external', None),
                            "mae": Xy_data.get('mae_external', None),
                            "rmse": Xy_data.get('rmse_external', None),
                            "acc": Xy_data.get('acc_external', None),
                            "f1": Xy_data.get('f1_external', None),
                            "mcc": Xy_data.get('mcc_external', None),
                        } if external_plots_applicable else None,
                        "dat_text_preview_deferred": bool(not isinstance(print_results_text, str) or print_results_text == ''),
                    },
                    evidence_level="direct",
                    dat_text=print_results_text if isinstance(print_results_text, str) and print_results_text != '' else None,
                )

                # SHAP analysis
                _ = shap_analysis(self,Xy_data,model_data,path_n_suffix)

                shap_path = f"{base_dir}/SHAP_{base_name}.png"
                self.args.predict_audit = audit_event(
                    self.args.predict_audit,
                    event_type="shap_artifact",
                    payload={
                        "branch_key": str(branch_key),
                        "path": str(shap_path),
                        "exists": bool(os.path.exists(shap_path)),
                        "state": "created" if os.path.exists(shap_path) else "skipped",
                    },
                    evidence_level="derived",
                )

                # PFI analysis
                _ = PFI_plot(self,Xy_data,model_data,path_n_suffix)

                pfi_path = f"{base_dir}/PFI_{base_name}.png"
                self.args.predict_audit = audit_event(
                    self.args.predict_audit,
                    event_type="pfi_artifact",
                    payload={
                        "branch_key": str(branch_key),
                        "path": str(pfi_path),
                        "exists": bool(os.path.exists(pfi_path)),
                        "state": "created" if os.path.exists(pfi_path) else "skipped",
                    },
                    evidence_level="derived",
                )

                # create Pearson heatmap
                _ = pearson_map_predict(self,Xy_data,params_dir)

                pearson_path = None
                pearson_state = "not_applicable"
                if str(suffix_title) in ['No_PFI', 'PFI']:
                    pearson_path = str(self.args.destination.joinpath(f"Pearson_heatmap_{suffix_title}.png"))
                    pearson_exists = bool(os.path.exists(pearson_path))
                    pearson_state = "created" if pearson_exists else "skipped"
                else:
                    pearson_exists = False

                self.args.predict_audit = audit_event(
                    self.args.predict_audit,
                    event_type="pearson_artifact",
                    payload={
                        "branch_key": str(branch_key),
                        "path": pearson_path,
                        "exists": pearson_exists,
                        "state": pearson_state,
                    },
                    evidence_level="derived",
                )

                # Outlier analysis
                if model_data['type'].lower() == 'reg':
                    _ = outlier_plot(self,Xy_data,path_n_suffix,name_points,colors)
                    outlier_path = f"{base_dir}/Outliers_{base_name}.png"
                    outlier_exists = bool(os.path.exists(outlier_path))
                    self.args.predict_audit = audit_event(
                        self.args.predict_audit,
                        event_type="outlier_artifact",
                        payload={
                            "branch_key": str(branch_key),
                            "path": str(outlier_path),
                            "exists": outlier_exists,
                            "state": "created" if outlier_exists else "skipped",
                        },
                        evidence_level="derived",
                    )
                else:
                    self.args.predict_audit = audit_event(
                        self.args.predict_audit,
                        event_type="outlier_artifact",
                        payload={
                            "branch_key": str(branch_key),
                            "path": None,
                            "exists": False,
                            "state": "not_applicable",
                        },
                        evidence_level="derived",
                    )

                # y distribution
                _ = distribution_plot(self,Xy_data,path_n_suffix,model_data)

                distribution_path = f"{base_dir}/y_distribution_{base_name}.png"
                distribution_exists = bool(os.path.exists(distribution_path))
                self.args.predict_audit = audit_event(
                    self.args.predict_audit,
                    event_type="distribution_artifact",
                    payload={
                        "branch_key": str(branch_key),
                        "path": str(distribution_path),
                        "exists": distribution_exists,
                        "state": "created" if distribution_exists else "skipped",
                    },
                    evidence_level="derived",
                )

        self.args.predict_audit = audit_set(
            self.args.predict_audit,
            "runtime",
            "module_runtime_seconds",
            round(time.time() - start_time, 2),
        )

        try:
            audit_write_ok = finalize_module_audit(self.args.predict_audit, predict_audit_path, status="completed")
            if not audit_write_ok:
                _ = write_json_output_audit(
                    json_audit_path,
                    module="PREDICT",
                    attempted_output_path=predict_audit_path,
                    attempted=True,
                    succeeded=False,
                    artifact="predict_audit.json",
                    error=RuntimeError("predict_audit_json_write_returned_false"),
                )
            else:
                _ = write_json_output_audit(
                    json_audit_path,
                    module="PREDICT",
                    attempted_output_path=predict_audit_path,
                    attempted=True,
                    succeeded=True,
                    artifact="predict_audit.json",
                    error=None,
                )
        except Exception as json_error:
            _ = write_json_output_audit(
                json_audit_path,
                module="PREDICT",
                attempted_output_path=predict_audit_path,
                attempted=True,
                succeeded=False,
                artifact="predict_audit.json",
                error=json_error,
            )

        _ = finish_print(self,start_time,'PREDICT')
