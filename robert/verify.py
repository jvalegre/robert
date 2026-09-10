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
import copy
import numpy as np
from statistics import mode
from sklearn.cluster import KMeans
from robert.utils import (load_variables,
    load_db_n_params,
    load_n_predict,
    finish_print,
    get_prediction_results,
    print_pfi,
    plot_metrics,
    repeated_kfold_cv,
    build_params_dirs,
    Logger
)


class _ClusterMeanBaseline:
    '''
    Trivial baseline used by verify.cluster_test(): clusters the X descriptors into two groups
    with KMeans and predicts each point with its own cluster's training-y mean. Follows the
    same fit()/predict() interface as the real sklearn models so it can be dropped straight
    into repeated_kfold_cv() and evaluated out-of-fold exactly like the real model, making its
    RMSE directly comparable to CV_score.
    '''

    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, y):
        self.kmeans = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        labels = self.kmeans.fit_predict(X)
        y = np.asarray(y)
        self.cluster_means = {label: y[labels == label].mean() for label in set(labels)}
        return self

    def predict(self, X):
        labels = self.kmeans.predict(X)
        return np.array([self.cluster_means[label] for label in labels])


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

        # if params_dir = '', the program performs the tests for the No_PFI and PFI folders
        # (or, with --all_models, for every model staged under GENERATE/All_models)
        params_dirs, suffixes, suffix_titles, model_names = build_params_dirs(self)

        # --all_models: each model gets its own VERIFY_{model}_data.dat instead of merging
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
                    self.args.log = Logger(self.args.destination / f'VERIFY_{model_name}', 'data')
                    current_model = model_name
                    model_time_start = time.time()

                _ = print_pfi(self,params_dir)

                # load the Xy databse and model parameters
                Xy_data, model_data, suffix_title = load_db_n_params(self,params_dir,suffix,suffix_title,"verify",True)

                # ymean_test/yshuffle_test/onehot_test/cluster_test below each need their own
                # untouched copy of the freshly-loaded data (onehot_test in particular mutates
                # X_train_scaled in place, and load_n_predict() right below adds CV-result keys
                # to Xy_data) - snapshot it now with a deep copy instead of re-reading and
                # reprocessing the same CSVs (plus the external test set, if any) from disk
                # 4 more times just to get a clean copy
                Xy_data_pristine = copy.deepcopy(Xy_data)

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

                # calculate scores for the y-mean test
                verify_results = self.ymean_test(verify_results,copy.deepcopy(Xy_data_pristine),model_data)

                # calculate scores for the y-shuffle test
                verify_results = self.yshuffle_test(verify_results,copy.deepcopy(Xy_data_pristine),model_data)

                # one-hot test (check that if a value isnt 0, the value assigned is 1)
                verify_results = self.onehot_test(verify_results,copy.deepcopy(Xy_data_pristine),model_data)

                # two-cluster-mean test (flags datasets that are really just two separated
                # point clouds, where a model can look good while only learning "which side
                # of the gap is this point on")
                verify_results = self.cluster_test(verify_results,copy.deepcopy(Xy_data_pristine),model_data)

                # analysis of results
                results_print,verify_results,verify_metrics = self.analyze_tests(verify_results)

                # plot a bar graph with the results (kept with its title - the report crops
                # the title out via a precisely-measured container height, see report.py's
                # verify_height - but the saved PNG itself keeps the full title)
                print_ver = plot_metrics(model_data,suffix_title,verify_metrics,verify_results)

                # print and save results
                _ = self.print_verify(results_print,verify_results,print_ver,model_data)

        # --all_models: each model gets its own "Time VERIFY" line reflecting only the time
        # spent processing that model (not the cumulative time for every model), since each
        # model's own PDF report should show its own VERIFY duration. finish_print() can't be
        # reused here because it always measures from the original start_time (the whole
        # multi-model run), and it also only writes into whichever log is CURRENTLY open (the
        # last model processed) - every other model's log needs its line appended separately
        # to its own (already-closed) file
        if current_model is not None:
            model_durations[current_model] = time.time() - model_time_start
            for m, duration in model_durations.items():
                if m == current_model:
                    continue
                with open(self.args.destination / f'VERIFY_{m}_data.dat', 'a', encoding='utf-8') as f:
                    f.write(f"\nTime VERIFY: {round(duration, 2)} seconds\n")
            self.args.log.write(f"\nTime VERIFY: {round(model_durations[current_model], 2)} seconds\n")
            self.args.log.finalize()
        else:
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

        return verify_results


    def onehot_test(self,verify_results,Xy_data,model_data):
        '''
        Calculate the accuracy of the model when using one-hot models. Continuous descriptors
        are binarized by their own median (value >= median -> 1, else -> 0) - not by "== 0",
        which only binarizes descriptors that already contain literal zeros and silently
        collapses any strictly-positive continuous descriptor (e.g. a molecular weight) into
        a constant, useless column instead of a genuine two-way split.

        Descriptors that already have 2 or fewer unique values are left untouched instead of
        being median-split: for an imbalanced already-binary descriptor (e.g. a flag that only
        30% of points have), the median can fall on the majority value, so "value >= median"
        is true for both classes and collapses the column to a constant - the same failure
        mode the median split was meant to fix for continuous descriptors, just triggered by
        class imbalance instead of the literal-zeros case.

        If every descriptor already has 2 or fewer unique values, there's no continuous
        variation left to destroy anywhere in the dataset: the "onehot" model ends up trained
        on essentially the same information as the original, so its error comes out
        indistinguishable from the original CV score, which would always trip the "FAILED"
        threshold (no degradation from binarizing) regardless of whether the model is
        actually flawed. In that case the test is skipped (marked N/A) instead of being
        forced to a spurious fail.
        '''

        already_binary = all(
            Xy_data['X_train'][desc].dropna().nunique() <= 2
            for desc in Xy_data['X_train']
        )
        if already_binary:
            verify_results['onehot'] = verify_results['CV_score']
            verify_results['onehot_skipped'] = True
            return verify_results

        Xy_onehot = Xy_data.copy()
        constant_descs = []
        for desc in Xy_onehot['X_train']:
            # already binary/categorical: median split could collapse an imbalanced
            # minority class to a constant, so this descriptor is kept as-is
            if Xy_onehot['X_train'][desc].dropna().nunique() <= 2:
                continue

            desc_median = Xy_onehot['X_train'][desc].median()
            new_vals = [1 if val >= desc_median else 0 for val in Xy_onehot['X_train'][desc]]

            # a discrete descriptor with many ties at the median can still collapse to a
            # constant after the split - drop just that descriptor instead of letting it
            # invalidate the whole test
            if len(set(new_vals)) <= 1:
                constant_descs.append(desc)
                continue

            Xy_onehot['X_train_scaled'][desc] = new_vals

        if constant_descs:
            Xy_onehot['X_train_scaled'] = Xy_onehot['X_train_scaled'].drop(columns=constant_descs)
            if 'X_test_scaled' in Xy_onehot:
                Xy_onehot['X_test_scaled'] = Xy_onehot['X_test_scaled'].drop(columns=constant_descs)
            if 'X_external_scaled' in Xy_onehot:
                Xy_onehot['X_external_scaled'] = Xy_onehot['X_external_scaled'].drop(columns=constant_descs)

        if Xy_onehot['X_train_scaled'].shape[1] == 0:
            # every descriptor collapsed to a constant after binarization: nothing left to
            # evaluate, same rationale as the already-binary skip above
            verify_results['onehot'] = verify_results['CV_score']
            verify_results['onehot_skipped'] = True
            return verify_results

        Xy_onehot = load_n_predict(self, model_data, Xy_onehot, BO_opt=False)
        verify_results['onehot'] = Xy_onehot[f'{verify_results["error_type"]}_train']
        return verify_results


    def cluster_test(self,verify_results,Xy_data,model_data):
        '''
        Calculate the accuracy of a simple two-cluster-mean baseline (_ClusterMeanBaseline):
        clusters X into two groups with KMeans and predicts each point with its own cluster's
        training-y mean, evaluated out-of-fold via the same repeated_kfold_cv() used for the
        real model - so its RMSE is directly comparable to CV_score, unlike a naive in-sample
        baseline (which would almost always look artificially good on any small dataset,
        regardless of whether it is actually bimodal, and give a false positive here).

        A dataset where the points sit in two point clouds separated by a wide gap in y (i.e.
        barely any points cover the middle of the range) can make a model look deceptively good
        even if it has only learned "which of the two point clouds is this compound in" - not
        the underlying relationship. If the real model barely beats this trivial baseline, that
        is a sign of exactly that failure mode, regardless of how good its raw CV error looks
        in isolation. Classification is not affected by this failure mode the same way (already
        covered by the y_mean test's majority-class baseline), so this test reuses that result
        there.
        '''

        if model_data['type'].lower() != 'reg':
            verify_results['cluster'] = verify_results['y_mean']
            return verify_results

        baseline_model = _ClusterMeanBaseline(random_state=model_data['seed'])
        Xy_cluster = repeated_kfold_cv(model_data, baseline_model, Xy_data.copy(), BO_opt=True)

        y_all_list,y_pred_all_list = [],[]
        for y_val,y_pred_vals in zip(Xy_cluster['y_train'], Xy_cluster['y_pred_train_all']):
            for y_pred_val in y_pred_vals:
                y_all_list.append(y_val)
                y_pred_all_list.append(y_pred_val)

        r2_cluster, mae_cluster, rmse_cluster = get_prediction_results(model_data, y_all_list, y_pred_all_list)
        verify_results['cluster'] = {'r2': r2_cluster, 'mae': mae_cluster, 'rmse': rmse_cluster}[verify_results['error_type']]

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
        gray_color = '#a9a9a9'
        colors = [None,None,None,None]
        results_print = [None,None,None,None]
        metrics = [None,None,None,None]

        # the threshold uses validation results to compare in the tests
        verify_results['higher_thres'] = (1+thres_test_pass)*verify_results['CV_score']
        verify_results['unclear_higher_thres'] = (1+thres_test_unclear)*verify_results['CV_score']
        # for higher-is-better metrics (r2/acc/mcc/f1, the only ones that use lower_thres/
        # unclear_lower_thres below), CV_score can be negative (e.g. a poor-fitting R2) - scaling
        # by (1-thres) then moves the threshold TOWARD zero, i.e. easier to pass, the opposite of
        # the intended "thres fraction worse than CV_score". Subtracting a fraction of
        # abs(CV_score) instead keeps the threshold on the "worse" side regardless of sign
        verify_results['lower_thres'] = verify_results['CV_score'] - thres_test_pass*abs(verify_results['CV_score'])
        verify_results['unclear_lower_thres'] = verify_results['CV_score'] - thres_test_unclear*abs(verify_results['CV_score'])

        # determine whether the tests pass
        test_names = ['y_mean','y_shuffle','onehot','cluster']
        for i,test_ver in enumerate(test_names):
            metrics[i] = verify_results[test_ver]

            # onehot test skipped (all descriptors already binary - see onehot_test()): not
            # scored, shown as N/A instead of being forced through the pass/fail thresholds
            if test_ver == 'onehot' and verify_results.get('onehot_skipped',False):
                colors[i] = gray_color
                results_print[i] = f'\n         - {test_ver}: N/A, all descriptors are already binary (0/1)'
                continue

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
        
        return results_print,verify_results,verify_metrics


    def print_verify(self,results_print,verify_results,print_ver,model_data):
        '''
        Print and store the results of VERIFY
        '''

        print_ver += f'\n      Results of flawed models and sorted cross-validation:'
        CV_type = f"{model_data['repeat_kfolds']}x {model_data['kfold']}-fold CV"
        # the printing order should be y-mean, y-shuffle, one-hot and cluster
        if verify_results['error_type'].lower() in ['mae','rmse']:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} ({CV_type}) {verify_results["CV_score"]:.2} + {int(thres_test_unclear*100)}% & {int(thres_test_pass*100)}% threshold = {verify_results["unclear_higher_thres"]:.2} & {verify_results["higher_thres"]:.2}'
        else:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} ({CV_type}) {verify_results["CV_score"]:.2} - {int(thres_test_unclear*100)}% & {int(thres_test_pass*100)}% threshold = {verify_results["unclear_lower_thres"]:.2} & {verify_results["lower_thres"]:.2}'
        print_ver += results_print[0]
        print_ver += results_print[1]
        print_ver += results_print[2]
        print_ver += results_print[3]
        if model_data['type'].lower() == 'reg':
            print_ver += f"\n         - Sorted {model_data['kfold']}-fold CV : R2 = {verify_results['r2_train_sorted_CV']}, MAE = {verify_results['mae_train_sorted_CV']}, RMSE = {verify_results['rmse_train_sorted_CV']}"
        elif model_data['type'].lower() == 'clas':
            print_ver += f"\n         - Sorted CV : Accuracy = {verify_results['acc_train_sorted_CV']}, F1 score = {verify_results['f1_train_sorted_CV']}, MCC = {verify_results['mcc_train_sorted_CV']}"

        self.args.log.write(print_ver)
