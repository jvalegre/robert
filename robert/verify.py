"""
Parameters
----------

    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    params_dir : str, default=''
        Folder containing the database and parameters of the ML model to analyze.
    thres_test : float, default=0.25,
        Threshold used to determine if a test pasess. It is determined in % units of diference between
        the RMSE (MCC in classificators) of the model and the test (i.e., 0.25 = 25% difference with the 
        original value). Test passes if:
        1. y-mean, y-shuffle and one-hot encoding tests: the error is greater than 25% (from original MAE and RMSE for regressors)
        2. K-fold cross validation: the error is lower than 25%
    kfold : int, default=5,
        The training set is split into a K number of folds in the cross-validation test (i.e. 5-fold CV).

"""
#####################################################.
#        This file stores the VERIFY class          #
#           used for ML model analysis              #
#####################################################.

import os
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sb
from statistics import mode
# for users with no intel architectures. This part has to be before the sklearn imports
try:
    from sklearnex import patch_sklearn
    patch_sklearn(verbose=False)
except (ModuleNotFoundError,ImportError):
    pass
from sklearn.model_selection import KFold
from robert.utils import (load_variables,
    load_db_n_params,
    load_model,
    pd_to_dict,
    load_n_predict,
    finish_print,
    get_prediction_results,
    print_pfi
)


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
        if 'GENERATE/Best_model' in self.args.params_dir:
            params_dirs = [f'{self.args.params_dir}/No_PFI',f'{self.args.params_dir}/PFI']
        else:
            params_dirs = [self.args.params_dir]

        for params_dir in params_dirs:
            if os.path.exists(params_dir):

                _ = print_pfi(self,params_dir)

                # load and ML model parameters, and add standardized descriptors
                Xy_data, params_df, params_path, suffix_title, _ = load_db_n_params(self,params_dir,"verify",True)
                
                # set the parameters for each ML model of the hyperopt optimization
                params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

                # this dictionary will keep the results of the tests
                verify_results = {'error_type': params_df['error_type'][0]}

                # get original score
                Xy_orig = Xy_data.copy()
                Xy_orig = load_n_predict(params_dict, Xy_orig)  
                verify_results['original_score_train'] = Xy_orig[f'{verify_results["error_type"]}_train']
                verify_results['original_score_valid'] = Xy_orig[f'{verify_results["error_type"]}_valid']

                # calculate R2 for k-fold cross validation (if needed)
                verify_results = self.cv_test(verify_results,Xy_data,params_dict)

                # calculate scores for the y-mean test
                verify_results = self.ymean_test(verify_results,Xy_data,params_dict)

                # load and ML model parameters again (to avoid weird memory issues on Windows, for some 
                # reason the Xy_data dataframe changes when changing X descriptors in copy() objects)
                Xy_data, params_df, params_path, suffix_title, _ = load_db_n_params(self,params_dir,"verify",False)

                # calculate scores for the y-shuffle test
                verify_results = self.yshuffle_test(verify_results,Xy_data,params_dict)

                # load and ML model parameters again (to avoid weird memory issues on Windows, for some 
                # reason the Xy_data dataframe changes when changing X descriptors in copy() objects)
                Xy_data, params_df, params_path, suffix_title, _ = load_db_n_params(self,params_dir,"verify",False)

                # one-hot test (check that if a value isnt 0, the value assigned is 1)
                verify_results = self.onehot_test(verify_results,Xy_data,params_dict)

                # load and ML model parameters again (to avoid weird memory issues on Windows, for some 
                # reason the Xy_data dataframe changes when changing X descriptors in copy() objects)
                Xy_data, params_df, params_path, suffix_title, _ = load_db_n_params(self,params_dir,"verify",False)

                # analysis of results
                colors,color_codes,results_print,verify_results = self.analyze_tests(verify_results)

                # plot a donut plot with the results
                print_ver,path_n_suffix = self.plot_donut(colors,color_codes,params_path,suffix_title)

                # print and save results
                _ = self.print_verify(results_print,verify_results,print_ver,path_n_suffix)

        _ = finish_print(self,start_time,'VERIFY')


    def cv_test(self,verify_results,Xy_data,params_dict):
        '''
        Performs a K-fold cross-validation on the training set.
        '''      

        # Fit the original model with the training set
        loaded_model = load_model(params_dict)
        loaded_model.fit(np.asarray(Xy_data['X_train_scaled']).tolist(), np.asarray(Xy_data['y_train']).tolist())
        data_cv = load_n_predict(params_dict, Xy_data)
        
        cv_score = []
        data_cv = {}
        kf = KFold(n_splits=self.args.kfold,shuffle=True)
        
        # combine training and validation for CV
        X_combined = pd.concat([Xy_data['X_train_scaled'],Xy_data['X_valid_scaled']], axis=0).reset_index(drop=True)
        y_combined = pd.concat([Xy_data['y_train'],Xy_data['y_valid']], axis=0).reset_index(drop=True)
        for _, (train_index, valid_index) in enumerate(kf.split(X_combined)):
            XY_cv = {}
            XY_cv['X_train_scaled'] = X_combined.loc[train_index]
            XY_cv['y_train'] = y_combined.loc[train_index]
            XY_cv['X_valid_scaled'] = X_combined.loc[valid_index]
            XY_cv['y_valid'] = y_combined.loc[valid_index]
            data_cv = load_n_predict(params_dict, XY_cv)

            cv_score.append(data_cv[f'{verify_results["error_type"].lower()}_valid'])

        verify_results['cv_score'] = np.mean(cv_score)
        verify_results['cv_std'] = np.std(cv_score)
        
        return verify_results


    def ymean_test(self,verify_results,Xy_data,params_dict):
        '''
        Calculate the accuracy of the model when using a flat line of predicted y values. For 
        regression, the mean of the y values is used. For classification, the value that is
        predicted more often is used.
        '''

        Xy_ymean = Xy_data.copy()   
        if params_dict['type'].lower() == 'reg':
            y_mean_array = np.ones(len(Xy_ymean['y_valid']))*(Xy_ymean['y_valid'].mean())
            Xy_ymean['r2_valid'], Xy_ymean['mae_valid'], Xy_ymean['rmse_valid'] = get_prediction_results(params_dict,Xy_ymean['y_valid'],y_mean_array)
        
        elif params_dict['type'].lower() == 'clas':
            y_mean_array = np.ones(len(Xy_ymean['y_valid']))*mode(Xy_ymean['y_valid'])
            Xy_ymean['acc_valid'], Xy_ymean['f1_valid'], Xy_ymean['mcc_valid'] = get_prediction_results(params_dict,Xy_ymean['y_valid'],y_mean_array)

        verify_results['y_mean'] = Xy_ymean[f'{verify_results["error_type"]}_valid']

        return verify_results


    def yshuffle_test(self,verify_results,Xy_data,params_dict):
        '''
        Calculate the accuracy of the model when the y values are randomly shuffled in the validation set
        For example, a y array of 1.3, 2.1, 4.0, 5.2 might become 2.1, 1.3, 5.2, 4.0.
        '''

        Xy_yshuffle = Xy_data.copy()
        Xy_yshuffle['y_valid'] = Xy_yshuffle['y_valid'].sample(frac=1,random_state=params_dict['seed'],axis=0)
        Xy_yshuffle = load_n_predict(params_dict, Xy_yshuffle)  
        verify_results['y_shuffle'] = Xy_yshuffle[f'{verify_results["error_type"]}_valid']

        return verify_results


    def onehot_test(self,verify_results,Xy_data,params_dict):
        '''
        Calculate the accuracy of the model when using one-hot models. All X values that are
        not 0 are considered to be 1 (NaN from missing values are converted to 0).
        '''

        Xy_onehot = Xy_data.copy()
        for desc in Xy_onehot['X_train']:
            new_vals = []
            for val in Xy_onehot['X_train'][desc]:
                if int(val) == 0:
                    new_vals.append(0)
                else:
                    new_vals.append(1)
            Xy_onehot['X_train_scaled'][desc] = new_vals

        for desc in Xy_onehot['X_valid']:
            new_vals = []
            for val in Xy_onehot['X_valid'][desc]:
                if int(val) == 0:
                    new_vals.append(0)
                else:
                    new_vals.append(1)
            Xy_onehot['X_valid_scaled'][desc] = new_vals

        Xy_onehot = load_n_predict(params_dict, Xy_onehot)  
        verify_results['onehot'] = Xy_onehot[f'{verify_results["error_type"]}_valid']

        return verify_results


    def analyze_tests(self,verify_results):
        '''
        Function to check whether the tests pass and retrieve the corresponding colors:
        1. Blue for passing tests
        2. Red for failing tests
        '''

        blue_color = '#1f77b4'
        red_color = '#cd5c5c'
        color_codes = {'blue' : blue_color,
                        'red' : red_color}
        colors = [None,None,None,None]
        results_print = [None,None,None,None]
        # these thresholds use validation results to compare in the tests
        verify_results['higher_thres'] = (1+self.args.thres_test)*verify_results['original_score_valid']
        verify_results['lower_thres'] = (1-self.args.thres_test)*verify_results['original_score_valid']

        for i,test_ver in enumerate(['y_mean', 'cv_score', 'onehot', 'y_shuffle']):
            if verify_results['error_type'].lower() in ['mae','rmse']:
                if verify_results[test_ver] <= verify_results['higher_thres']:
                    if test_ver != 'cv_score':
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, lower than thres.'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {self.args.kfold}-fold CV: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, lower than thres.'
                else:
                    if test_ver != 'cv_score':
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, higher than thres.'
                    else:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {self.args.kfold}-fold CV: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, higher than thres.'

            else:
                if verify_results[test_ver] >= verify_results['lower_thres']:
                    if test_ver != 'cv_score':
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, higher than thres.'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {self.args.kfold}-fold CV: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, higher than thres.'
                else:
                    if test_ver != 'cv_score':
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, lower than thres.'
                    else:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {self.args.kfold}-fold CV: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2}, lower than thres.'
        
        return colors,color_codes,results_print,verify_results


    def plot_donut(self,colors,color_codes,params_path,suffix_title):
        '''
        Creates a donut plot with the results of VERIFY
        '''

        # set PATH names and plot
        base_csv_name = '_'.join(os.path.basename(params_path).replace('.csv','_').split('_')[0:2])
        base_csv_name = f'VERIFY/{base_csv_name}'
        base_csv_path = f"{Path(os.getcwd()).joinpath(base_csv_name)}"
        path_n_suffix = f'{base_csv_path}_{suffix_title}'

        sb.reset_defaults()
        _, ax = plt.subplots(figsize=(7.45,6), subplot_kw=dict(aspect="equal"))
        
        recipe = ["y_mean",
                f"{self.args.kfold}-fold CV",
                "onehot",
                "y_shuffle"]
                
        # make 4 even parts in the donut plot
        data = [25, 25, 25, 25]
        explode = [None, None, None, None]
        
        # failing or undetermined tests will lead to pieces that are outside the regular donut
        for i,color in enumerate(colors):
            if color == color_codes['red']:
                explode[i] = 0.05
            else:
                explode[i] = 0
        
        wedgeprops = {'width':0.4, 'edgecolor':'black', 'lw':0.72}
        wedges, _ = ax.pie(data, wedgeprops=wedgeprops, startangle=0, colors=colors, explode=explode)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")
        fontsize = 14

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(recipe[i], xy=(x, y), xytext=(1.15*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, fontsize=fontsize, **kw)
        title_verify = f"VERIFY tests of {os.path.basename(path_n_suffix)}"
        plt.title(title_verify, y=1.04, fontsize = fontsize, fontweight="bold")

        # save plot
        verify_plot_file = f'{os.path.dirname(path_n_suffix)}/VERIFY_tests_{os.path.basename(path_n_suffix)}.png'
        plt.savefig(verify_plot_file, dpi=300, bbox_inches='tight')
        plt.clf()

        path_reduced = '/'.join(f'{verify_plot_file}'.replace('\\','/').split('/')[-2:])
        print_ver = f"\n   o  VERIFY donut plots saved in {path_reduced}"

        return print_ver, path_n_suffix


    def print_verify(self,results_print,verify_results,print_ver,path_n_suffix):
        '''
        Print and store the results of VERIFY
        '''
        
        verify_results_file = f'{os.path.dirname(path_n_suffix)}/VERIFY_tests_{os.path.basename(path_n_suffix)}.dat'
        path_reduced = '/'.join(f'{verify_results_file}'.replace('\\','/').split('/')[-2:])
        print_ver += f"\n   o  VERIFY test values saved in {path_reduced}"
        print_ver += f'\n      Results of the VERIFY tests:'
        # the printing order should be CV, y-mean, y-shuffle and one-hot
        if verify_results['error_type'].lower() in ['mae','rmse']:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} (valid. set) {verify_results["original_score_valid"]:.2} + {int(self.args.thres_test*100)}% thres. = {verify_results["higher_thres"]:.2}'
        else:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} (valid. set) {verify_results["original_score_valid"]:.2} - {int(self.args.thres_test*100)}% thres. = {verify_results["lower_thres"]:.2}'
        print_ver += results_print[1]
        print_ver += results_print[0]
        print_ver += results_print[3]
        print_ver += results_print[2]
        self.args.log.write(print_ver)
        dat_results = open(verify_results_file, "w")
        dat_results.write(print_ver)
        dat_results.close()
