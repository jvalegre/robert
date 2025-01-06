"""
Parameters
----------

    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    params_dir : str, default=''
        Folder containing the database and parameters of the ML model to analyze.
    kfold : int, default='auto',
        Number of folds for the k-fold cross-validation. If 'auto', the program does 
        a LOOCV for databases with less than 50 points, and 5-fold CV for larger databases 

"""
#####################################################.
#        This file stores the VERIFY class          #
#           used for ML model analysis              #
#####################################################.

import os
import time
import numpy as np
from statistics import mode
from robert.utils import (load_variables,
    load_db_n_params,
    cv_test,
    pd_to_dict,
    load_n_predict,
    finish_print,
    get_prediction_results,
    print_pfi,
    plot_metrics
)


#thresholds for passing tests in VERIFY
thres_test_pass = 0.25
thres_test_unclear = 0.1

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

                # load database and ML model parameters, and add standardized descriptors
                Xy_data, params_df, params_path, suffix_title, _ = load_db_n_params(self,params_dir,"verify",True)
                
                # set the parameters for each ML model of the hyperopt optimization
                params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

                # this dictionary will keep the results of the tests
                verify_results = {'error_type': params_df['error_type'][0]}

                # get original score
                Xy_orig = Xy_data.copy()
                Xy_orig,_ = load_n_predict(self, params_dict, Xy_orig)  
                verify_results['original_score_train'] = Xy_orig[f'{verify_results["error_type"]}_train']
                verify_results['original_score_valid'] = Xy_orig[f'{verify_results["error_type"]}_valid']

                # calculate cross-validation
                verify_results,type_cv,path_n_suffix = cv_test(self,verify_results,Xy_data,params_dict,params_path,suffix_title,'verify')

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
                results_print,verify_results,verify_metrics = self.analyze_tests(verify_results,type_cv)

                # plot a bar graph with the results
                print_ver = plot_metrics(path_n_suffix,verify_metrics,verify_results)

                # print and save results
                _ = self.print_verify(results_print,verify_results,print_ver,path_n_suffix,verify_metrics,params_dict)

        _ = finish_print(self,start_time,'VERIFY')


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
        Xy_yshuffle,_ = load_n_predict(self, params_dict, Xy_yshuffle)  
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

        Xy_onehot,_ = load_n_predict(self, params_dict, Xy_onehot)  
        verify_results['onehot'] = Xy_onehot[f'{verify_results["error_type"]}_valid']

        return verify_results


    def analyze_tests(self,verify_results,type_cv):
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
        verify_results['higher_thres'] = (1+thres_test_pass)*verify_results['original_score_valid']
        verify_results['unclear_higher_thres'] = (1+thres_test_unclear)*verify_results['original_score_valid']
        verify_results['lower_thres'] = (1-thres_test_pass)*verify_results['original_score_valid']
        verify_results['unclear_lower_thres'] = (1-thres_test_unclear)*verify_results['original_score_valid']

        # determine whether the tests pass
        test_names = ['y_mean','y_shuffle','onehot']
        for i,test_ver in enumerate(test_names):
            metrics[i] = verify_results[test_ver]
            if verify_results['error_type'].lower() in ['mae','rmse']:
                if verify_results[test_ver] <= verify_results['unclear_higher_thres']:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, lower than threshold'
                elif verify_results[test_ver] <= verify_results['higher_thres']:
                        colors[i] = yellow_color
                        results_print[i] = f'\n         - {test_ver}: UNCLEAR, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, higher than original, but close to fail'
                else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, higher than thresholds'

            else:
                if verify_results[test_ver] >= verify_results['unclear_lower_thres']:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, higher than thresholds'
                elif verify_results[test_ver] >= verify_results['lower_thres']:
                        colors[i] = yellow_color
                        results_print[i] = f'\n         - {test_ver}: UNCLEAR, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, lower than original, but close to fail'
                else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, lower than thresholds'

        # store metrics and colors to represent in comparison graph, adding the metrics of the 
        # original model first
        test_names = ['Model'] + test_names
        colors = [blue_color] + colors
        metrics = [verify_results['original_score_valid']] + metrics
        verify_metrics = {'test_names': test_names,
                          'colors': colors,
                          'metrics': metrics,
                          'higher_thres': verify_results['higher_thres'],
                          'lower_thres': verify_results['lower_thres'],
                          'unclear_higher_thres': verify_results['unclear_higher_thres'],
                          'unclear_lower_thres': verify_results['unclear_lower_thres'],
                          'type_cv': type_cv
                          }        
        
        return results_print,verify_results,verify_metrics


    def print_verify(self,results_print,verify_results,print_ver,path_n_suffix,verify_metrics,params_dict):
        '''
        Print and store the results of VERIFY
        '''

        print_ver += f'\n      Results of flawed models and cross-validation:'
        # the printing order should be y-mean, y-shuffle and one-hot
        if verify_results['error_type'].lower() in ['mae','rmse']:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} (valid. set) {round(verify_results["original_score_valid"],2)} + {int(thres_test_unclear*100)}% & {int(thres_test_pass*100)}% threshold = {round(verify_results["unclear_higher_thres"],2)} & {round(verify_results["higher_thres"],2)}'
        else:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} (valid. set) {round(verify_results["original_score_valid"],2)} - {int(thres_test_unclear*100)}% & {int(thres_test_pass*100)}% threshold = {round(verify_results["unclear_lower_thres"],2)} & {round(verify_results["lower_thres"],2)}'
        print_ver += results_print[0]
        print_ver += results_print[1]
        print_ver += results_print[2]
        if params_dict['type'].lower() == 'reg':
            print_ver += f"\n         - {verify_metrics['type_cv']} : R2 = {round(verify_results['cv_r2'],2)}, MAE = {round(verify_results['cv_mae'],2)}, RMSE = {round(verify_results['cv_rmse'],2)}"
        elif params_dict['type'].lower() == 'clas':
            print_ver += f"\n         - {verify_metrics['type_cv']} : Accuracy = {round(verify_results['cv_acc'],2)}, F1 score = {round(verify_results['cv_f1'],2)}, MCC = {round(verify_results['cv_mcc'],2)}"

        self.args.log.write(print_ver)
