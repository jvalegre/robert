"""
Parameters
----------

    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    params_dir : str, default=''
        Folder containing the database and parameters of the ML model to analyze.
    seed : int, default=0
        Random seed used in the ML predictor models and other protocols.
    kfold : int, default=5
        Number of random data splits for the cross-validation of the models. 
    repeat_kfolds : int, default=10
        Number of repetitions for the k-fold cross-validation of the models.

"""
#####################################################.
#        This file stores the VERIFY class          #
#           used for ML model analysis              #
#####################################################.

import os
import time
import numpy as np
from statistics import mode
from robert.json_output_for_agent import (
    init_module_audit,
    audit_set,
    audit_event,
    finalize_module_audit,
    write_json_output_audit,
)
from robert.utils import (load_variables,
    load_db_n_params,
    load_n_predict,
    finish_print,
    get_prediction_results,
    print_pfi,
    plot_metrics
)


# thresholds for passing tests in VERIFY
thres_test_pass = 0.3
thres_test_unclear = 0.15

class verify:
    """
    Class containing all the functions from the VERIFY module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the VERIFY module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "verify")

        command_line_value = getattr(self.args, "command_line", None)
        if not isinstance(command_line_value, str) or command_line_value.strip() == "":
            command_line_value = None

        self.args.verify_audit = init_module_audit(
            module="VERIFY",
            artifact_type="verify_audit",
            source_files=[str(self.args.params_dir)],
            command_line=command_line_value,
        )
        verify_audit_path = self.args.destination.joinpath("verify_audit.json")
        json_audit_path = self.args.destination.joinpath("json_output_audit.json")

        self.args.verify_audit = audit_set(self.args.verify_audit, "inputs", "params_dir", self.args.params_dir)
        self.args.verify_audit = audit_set(self.args.verify_audit, "inputs", "seed", self.args.seed)
        self.args.verify_audit = audit_set(self.args.verify_audit, "inputs", "kfold", self.args.kfold)
        self.args.verify_audit = audit_set(self.args.verify_audit, "inputs", "repeat_kfolds", self.args.repeat_kfolds)
        self.args.verify_audit = audit_set(self.args.verify_audit, "inputs", "destination", str(self.args.destination))
        self.args.verify_audit = audit_set(self.args.verify_audit, "inputs", "thres_test_pass", thres_test_pass)
        self.args.verify_audit = audit_set(self.args.verify_audit, "inputs", "thres_test_unclear", thres_test_unclear)

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
            is_pfi_filtered = bool(str(suffix_title) == 'PFI')
            self.args.verify_audit = audit_event(
                self.args.verify_audit,
                event_type="verify_branch",
                payload={
                    "params_dir_used": str(params_dir),
                    "suffix": str(suffix),
                    "suffix_title": str(suffix_title),
                    "params_dir_exists": bool(os.path.exists(params_dir)),
                    "pfi_section_active": is_pfi_filtered,
                    "branch_is_pfi_filtered": is_pfi_filtered,
                    "workflow_includes_pfi_branch": bool('GENERATE/Best_model' in self.args.params_dir),
                },
                evidence_level="direct",
            )
            if os.path.exists(params_dir):

                _ = print_pfi(self,params_dir)

                # load the Xy databse and model parameters
                Xy_data, model_data, suffix_title = load_db_n_params(self,params_dir,suffix,suffix_title,"verify",True)

                train_datapoints = None
                total_datapoints_loaded = None
                descriptor_list = None
                accepted_descriptor_count = None
                ignored_descriptor_count = None
                discarded_descriptor_count = None

                if isinstance(Xy_data, dict):
                    if 'X_train' in Xy_data and hasattr(Xy_data['X_train'], 'shape'):
                        train_datapoints = int(Xy_data['X_train'].shape[0])
                        if hasattr(Xy_data['X_train'], 'columns'):
                            descriptor_list = [str(col) for col in Xy_data['X_train'].columns]
                    elif 'y_train' in Xy_data and hasattr(Xy_data['y_train'], '__len__'):
                        train_datapoints = int(len(Xy_data['y_train']))

                    if 'X' in Xy_data and hasattr(Xy_data['X'], 'shape'):
                        total_datapoints_loaded = int(Xy_data['X'].shape[0])
                    elif 'y' in Xy_data and hasattr(Xy_data['y'], '__len__'):
                        total_datapoints_loaded = int(len(Xy_data['y']))

                if isinstance(model_data, dict):
                    if 'X_descriptors' in model_data and model_data['X_descriptors'] is not None:
                        accepted_descriptor_count = int(len(model_data['X_descriptors']))
                        if descriptor_list is None:
                            descriptor_list = [str(col) for col in model_data['X_descriptors']]
                    if 'X_descriptors_ignored' in model_data and model_data['X_descriptors_ignored'] is not None:
                        ignored_descriptor_count = int(len(model_data['X_descriptors_ignored']))
                    if 'X_descriptors_discarded' in model_data and model_data['X_descriptors_discarded'] is not None:
                        discarded_descriptor_count = int(len(model_data['X_descriptors_discarded']))

                self.args.verify_audit = audit_event(
                    self.args.verify_audit,
                    event_type="model_context",
                    payload={
                        "params_dir_used": str(params_dir),
                        "suffix": str(suffix),
                        "suffix_title": str(suffix_title),
                        "model_name": str(model_data['model']),
                        "model_type": str(model_data['type']),
                        "error_type": str(model_data['error_type']),
                        "descriptor_count": int(len(model_data['X_descriptors'])),
                        "y_column": str(model_data['y']),
                        "names_column": str(model_data['names']),
                        "kfold": int(model_data['kfold']),
                        "repeat_kfolds": int(model_data['repeat_kfolds']),
                        "seed": int(model_data['seed']),
                        "train_datapoints": train_datapoints,
                        "total_datapoints_loaded": total_datapoints_loaded,
                        "accepted_descriptor_count": accepted_descriptor_count,
                        "ignored_descriptor_count": ignored_descriptor_count,
                        "discarded_descriptor_count": discarded_descriptor_count,
                        "descriptor_list": descriptor_list,
                        "active_descriptor_count": accepted_descriptor_count,
                        "active_descriptor_list": descriptor_list,
                        "active_descriptor_source": "model_data['X_descriptors']",
                    },
                    evidence_level="direct",
                )
                
                # this dictionary will keep the results of the tests
                verify_results = {'error_type': model_data['error_type']}

                # get data about repeated and sorted CVs
                Xy_data = load_n_predict(self, model_data, Xy_data, BO_opt=True, verify_job=True)
                verify_results['CV_score'] = Xy_data[f'{verify_results["error_type"]}_train']
                verify_results['sorted_CV_score'] = Xy_data[f'{model_data["error_type"]}_train_sorted_CV']
                if model_data['type'].lower() == 'reg':
                    verify_results[f'r2_train_sorted_CV'] = [float(f"{val:.2f}") for val in Xy_data[f'r2_train_sorted_CV']]
                    verify_results[f'mae_train_sorted_CV'] = [float(f"{val:.2f}") for val in Xy_data[f'mae_train_sorted_CV']]
                    verify_results[f'rmse_train_sorted_CV'] = [float(f"{val:.2f}") for val in Xy_data[f'rmse_train_sorted_CV']]
                elif model_data['type'].lower() == 'clas':
                    verify_results[f'acc_train_sorted_CV'] = [float(f"{val:.2f}") for val in Xy_data[f'acc_train_sorted_CV']]
                    verify_results[f'f1_train_sorted_CV'] = [float(f"{val:.2f}") for val in Xy_data[f'f1_train_sorted_CV']]
                    verify_results[f'mcc_train_sorted_CV'] = [float(f"{val:.2f}") for val in Xy_data[f'mcc_train_sorted_CV']]

                self.args.verify_audit = audit_event(
                    self.args.verify_audit,
                    event_type="repeated_sorted_cv",
                    payload={
                        "params_dir_used": str(params_dir),
                        "suffix_title": str(suffix_title),
                        "original_cv_score": verify_results['CV_score'],
                        "sorted_cv_score": verify_results['sorted_CV_score'],
                        "sorted_metrics_regression": {
                            "r2": verify_results.get('r2_train_sorted_CV', None),
                            "mae": verify_results.get('mae_train_sorted_CV', None),
                            "rmse": verify_results.get('rmse_train_sorted_CV', None),
                        },
                        "sorted_metrics_classification": {
                            "acc": verify_results.get('acc_train_sorted_CV', None),
                            "f1": verify_results.get('f1_train_sorted_CV', None),
                            "mcc": verify_results.get('mcc_train_sorted_CV', None),
                        },
                    },
                    evidence_level="direct",
                )

                # load the Xy databse and model parameters
                Xy_data, model_data, suffix_title = load_db_n_params(self,params_dir,suffix,suffix_title,"verify",False)

                # calculate scores for the y-mean test
                verify_results = self.ymean_test(verify_results,Xy_data,model_data)

                # load the Xy databse and model parameters
                Xy_data, model_data, suffix_title = load_db_n_params(self,params_dir,suffix,suffix_title,"verify",False)

                # calculate scores for the y-shuffle test
                verify_results = self.yshuffle_test(verify_results,Xy_data,model_data)

                # load the Xy databse and model parameters
                Xy_data, model_data, suffix_title = load_db_n_params(self,params_dir,suffix,suffix_title,"verify",False)

                # one-hot test (check that if a value isnt 0, the value assigned is 1)
                verify_results = self.onehot_test(verify_results,Xy_data,model_data)

                # analysis of results
                results_print,verify_results,verify_metrics = self.analyze_tests(verify_results)

                # plot a bar graph with the results
                print_ver = plot_metrics(model_data,suffix_title,verify_metrics,verify_results)
                csv_name = os.path.basename(model_data['model']).split('_db.csv')[0]
                verify_plot_file = f"{os.getcwd()}/VERIFY/VERIFY_tests_{csv_name}_{suffix_title}.png"
                self.args.verify_audit = audit_event(
                    self.args.verify_audit,
                    event_type="verify_plot",
                    payload={
                        "plot_type": "VERIFY flawed-model test plot",
                        "suffix_title": str(suffix_title),
                        "image_path": str(verify_plot_file),
                        "image_generation_completed": bool(os.path.exists(verify_plot_file)),
                        "short_interpretation": "comparison of original model vs flawed-model VERIFY tests",
                    },
                    evidence_level="derived",
                )

                # print and save results
                _ = self.print_verify(results_print,verify_results,print_ver,model_data)

        self.args.verify_audit = audit_set(
            self.args.verify_audit,
            "runtime",
            "module_runtime_seconds",
            round(time.time() - start_time, 2),
        )

        try:
            audit_write_ok = finalize_module_audit(self.args.verify_audit, verify_audit_path, status="completed")
            if not audit_write_ok:
                _ = write_json_output_audit(
                    json_audit_path,
                    module="VERIFY",
                    attempted_output_path=verify_audit_path,
                    attempted=True,
                    succeeded=False,
                    artifact="verify_audit.json",
                    error=RuntimeError("verify_audit_json_write_returned_false"),
                )
            else:
                _ = write_json_output_audit(
                    json_audit_path,
                    module="VERIFY",
                    attempted_output_path=verify_audit_path,
                    attempted=True,
                    succeeded=True,
                    artifact="verify_audit.json",
                    error=None,
                )
        except Exception as json_error:
            _ = write_json_output_audit(
                json_audit_path,
                module="VERIFY",
                attempted_output_path=verify_audit_path,
                attempted=True,
                succeeded=False,
                artifact="verify_audit.json",
                error=json_error,
            )

        _ = finish_print(self,start_time,'VERIFY')


    def ymean_test(self,verify_results,Xy_data,model_data):
        '''
        Calculate the accuracy of the model when using a flat line of predicted y values. For 
        regression, the mean of the y values is used. For classification, the value that is
        predicted more often is used.
        '''

        Xy_ymean = Xy_data.copy()   
        if model_data['type'].lower() == 'reg':
            y_mean_array = np.ones(len(Xy_ymean['y_train']))*(Xy_ymean['y_train'].mean())
            Xy_ymean['r2_train'], Xy_ymean['mae_train'], Xy_ymean['rmse_train'] = get_prediction_results(model_data,Xy_ymean['y_train'],y_mean_array)
        
        elif model_data['type'].lower() == 'clas':
            y_mean_array = np.ones(len(Xy_ymean['y_train']))*mode(Xy_ymean['y_train'])
            Xy_ymean['acc_train'], Xy_ymean['f1_train'], Xy_ymean['mcc_train'] = get_prediction_results(model_data,Xy_ymean['y_train'],y_mean_array)

        verify_results['y_mean'] = Xy_ymean[f'{verify_results["error_type"]}_train']

        baseline_prediction = float(Xy_ymean['y_train'].mean()) if model_data['type'].lower() == 'reg' else int(mode(Xy_ymean['y_train']))
        self.args.verify_audit = audit_event(
            self.args.verify_audit,
            event_type="verify_test",
            payload={
                "test_name": "y_mean",
                "baseline_prediction_used": baseline_prediction,
                "resulting_metric": verify_results['y_mean'],
                "error_type": verify_results['error_type'],
            },
            evidence_level="direct",
        )

        return verify_results


    def yshuffle_test(self,verify_results,Xy_data,model_data):
        '''
        Calculate the accuracy of the model when the y values are randomly shuffled in the validation set
        For example, a y array of 1.3, 2.1, 4.0, 5.2 might become 2.1, 1.3, 5.2, 4.0.
        '''

        Xy_yshuffle = Xy_data.copy()
        Xy_yshuffle['y_train'] = Xy_yshuffle['y_train'].sample(frac=1,random_state=model_data['seed'],axis=0)
        Xy_yshuffle = load_n_predict(self, model_data, Xy_yshuffle, BO_opt=False)

        verify_results['y_shuffle'] = Xy_yshuffle[f'{verify_results["error_type"]}_train']

        self.args.verify_audit = audit_event(
            self.args.verify_audit,
            event_type="verify_test",
            payload={
                "test_name": "y_shuffle",
                "random_seed_used": int(model_data['seed']),
                "resulting_metric": verify_results['y_shuffle'],
                "error_type": verify_results['error_type'],
            },
            evidence_level="direct",
        )

        return verify_results


    def onehot_test(self,verify_results,Xy_data,model_data):
        '''
        Calculate the accuracy of the model when using one-hot models. All X values that are
        not 0 are considered to be 1 (NaN from missing values are converted to 0).
        '''

        Xy_onehot = Xy_data.copy()
        for desc in Xy_onehot['X_train']:
            new_vals = []
            for val in Xy_onehot['X_train'][desc]:
                if val == 0:
                    new_vals.append(0)
                else:
                    new_vals.append(1)
            Xy_onehot['X_train_scaled'][desc] = new_vals

        for desc in Xy_onehot['X_train']:
            new_vals = []
            for val in Xy_onehot['X_train'][desc]:
                if val == 0:
                    new_vals.append(0)
                else:
                    new_vals.append(1)
            Xy_onehot['X_train_scaled'][desc] = new_vals

        Xy_onehot = load_n_predict(self, model_data, Xy_onehot, BO_opt=False)
        verify_results['onehot'] = Xy_onehot[f'{verify_results["error_type"]}_train']

        self.args.verify_audit = audit_event(
            self.args.verify_audit,
            event_type="verify_test",
            payload={
                "test_name": "onehot",
                "descriptor_count_tested": int(len(Xy_onehot['X_train'].columns)),
                "resulting_metric": verify_results['onehot'],
                "error_type": verify_results['error_type'],
            },
            evidence_level="direct",
        )
        return verify_results


    def analyze_tests(self,verify_results):
        '''
        Function to check whether the tests pass and retrieve the corresponding colors:
        1. Blue for passing tests
        2. Red for failing tests
        '''

        blue_color = '#1f77b4'
        red_color = '#cd5c5c'
        yellow_color = '#c5c57d'
        colors = [None,None,None]
        results_print = [None,None,None]
        metrics = [None,None,None]

        # the threshold uses validation results to compare in the tests
        verify_results['higher_thres'] = (1+thres_test_pass)*verify_results['CV_score']
        verify_results['unclear_higher_thres'] = (1+thres_test_unclear)*verify_results['CV_score']
        verify_results['lower_thres'] = (1-thres_test_pass)*verify_results['CV_score']
        verify_results['unclear_lower_thres'] = (1-thres_test_unclear)*verify_results['CV_score']

        # determine whether the tests pass
        test_names = ['y_mean','y_shuffle','onehot']
        for i,test_ver in enumerate(test_names):
            metrics[i] = verify_results[test_ver]
            if verify_results['error_type'].lower() in ['mae','rmse']:
                if verify_results[test_ver] <= verify_results['unclear_higher_thres']:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, lower than threshold'
                elif verify_results[test_ver] <= verify_results['higher_thres']:
                        colors[i] = yellow_color
                        results_print[i] = f'\n         - {test_ver}: UNCLEAR, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, higher than original, but close to fail'
                else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, higher than thresholds'

            else:
                if verify_results[test_ver] >= verify_results['unclear_lower_thres']:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, higher than thresholds'
                elif verify_results[test_ver] >= verify_results['lower_thres']:
                        colors[i] = yellow_color
                        results_print[i] = f'\n         - {test_ver}: UNCLEAR, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, lower than original, but close to fail'
                else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, lower than thresholds'

        # store metrics and colors to represent in comparison graph, adding the metrics of the 
        # original model first
        test_names = ['Model'] + test_names
        colors = [blue_color] + colors
        metrics = [verify_results['CV_score']] + metrics
        verify_metrics = {'test_names': test_names,
                          'colors': colors,
                          'metrics': metrics,
                          'higher_thres': verify_results['higher_thres'],
                          'lower_thres': verify_results['lower_thres'],
                          'unclear_higher_thres': verify_results['unclear_higher_thres'],
                          'unclear_lower_thres': verify_results['unclear_lower_thres'],
                          }        

        per_test_status = {}
        for test_ver, result_text in zip(['y_mean', 'y_shuffle', 'onehot'], results_print):
            if 'FAILED' in result_text:
                per_test_status[test_ver] = 'FAILED'
            elif 'UNCLEAR' in result_text:
                per_test_status[test_ver] = 'UNCLEAR'
            else:
                per_test_status[test_ver] = 'PASSED'

        self.args.verify_audit = audit_event(
            self.args.verify_audit,
            event_type="analyze_tests",
            payload={
                "higher_threshold": verify_results['higher_thres'],
                "unclear_higher_threshold": verify_results['unclear_higher_thres'],
                "lower_threshold": verify_results['lower_thres'],
                "unclear_lower_threshold": verify_results['unclear_lower_thres'],
                "per_test_status": per_test_status,
                "per_test_metric": {
                    "y_mean": verify_results['y_mean'],
                    "y_shuffle": verify_results['y_shuffle'],
                    "onehot": verify_results['onehot'],
                },
                "scoring_direction": "lower_is_better" if verify_results['error_type'].lower() in ['mae', 'rmse'] else "higher_is_better",
                "colors": verify_metrics['colors'],
            },
            evidence_level="derived",
        )
        
        return results_print,verify_results,verify_metrics


    def print_verify(self,results_print,verify_results,print_ver,model_data):
        '''
        Print and store the results of VERIFY
        '''

        print_ver += f'\n      Results of flawed models and sorted cross-validation:'
        CV_type = f"{model_data['repeat_kfolds']}x {model_data['kfold']}-fold CV"
        # the printing order should be y-mean, y-shuffle and one-hot
        if verify_results['error_type'].lower() in ['mae','rmse']:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} ({CV_type}) {verify_results["CV_score"]:.2} + {int(thres_test_unclear*100)}% & {int(thres_test_pass*100)}% threshold = {verify_results["unclear_higher_thres"]:.2} & {verify_results["higher_thres"]:.2}'
        else:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} ({CV_type}) {verify_results["CV_score"]:.2} - {int(thres_test_unclear*100)}% & {int(thres_test_pass*100)}% threshold = {verify_results["unclear_lower_thres"]:.2} & {verify_results["lower_thres"]:.2}'
        print_ver += results_print[0]
        print_ver += results_print[1]
        print_ver += results_print[2]
        if model_data['type'].lower() == 'reg':
            print_ver += f"\n         - Sorted {model_data['kfold']}-fold CV : R2 = {verify_results['r2_train_sorted_CV']}, MAE = {verify_results['mae_train_sorted_CV']}, RMSE = {verify_results['rmse_train_sorted_CV']}"
        elif model_data['type'].lower() == 'clas':
            print_ver += f"\n         - Sorted CV : Accuracy = {verify_results['acc_train_sorted_CV']}, F1 score = {verify_results['f1_train_sorted_CV']}, MCC = {verify_results['mcc_train_sorted_CV']}"

        self.args.verify_audit = audit_event(
            self.args.verify_audit,
            event_type="print_verify_summary",
            payload={
                "original_cv_metric": verify_results['CV_score'],
                "y_mean_result": verify_results['y_mean'],
                "y_shuffle_result": verify_results['y_shuffle'],
                "onehot_result": verify_results['onehot'],
                "sorted_cv_metrics": {
                    "regression": {
                        "r2": verify_results.get('r2_train_sorted_CV', None),
                        "mae": verify_results.get('mae_train_sorted_CV', None),
                        "rmse": verify_results.get('rmse_train_sorted_CV', None),
                    },
                    "classification": {
                        "acc": verify_results.get('acc_train_sorted_CV', None),
                        "f1": verify_results.get('f1_train_sorted_CV', None),
                        "mcc": verify_results.get('mcc_train_sorted_CV', None),
                    },
                },
            },
            evidence_level="direct",
            dat_text=print_ver,
        )

        self.args.log.write(print_ver)
