"""
Parameters
----------

General
+++++++

     destination : str, default=None,
         Directory to create the output file(s).
     varfile : str, default=None
         Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
     params_dir : str, default=''
         Folder containing the database and parameters of the ML model to analyze.
     thres_test : float, default=0.2,
         Threshold used to determine if a test pasess. It is determined in % units of diference between
         the R2 (MCC in classificators) of the model and the test (i.e., 0.2 = 20% difference with the 
         original value). Test passes if:
            1. x- and y-shuffle tests: decreases more than X% (from original R2, regressors, or MCC, 
            classificators) or the error is greated than X% (from original MAE and RMSE for regressors)
            2. One-hot encoding test: decreases more than X%
            3. K-fold cross validation: decreases less than X%
     kfold : int, default=5,
         The training set is split into a K number of folds in the cross-validation test (i.e. 5-fold CV).
     error_type : str, default: rmse (regression), acc (classification)
         Target value used during the hyperopt optimization. Options:
         Regression:
            1. rmse (root-mean-square error)
            2. mae (mean absolute error)
            3. r2 (R-squared)
         Classification:
            1. mcc (Matthew's correlation coefficient)
            2. f1 (F1 score)
            3. acc (accuracy, fraction of correct predictions)
     seed : int, default=8,
         Random seed used in the ML predictor models, data splitting and other protocols.


"""
#####################################################.
#        This file stores the VERIFY class          #
#           used for ML model analysis              #
#####################################################.

import os
import time
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sb
from sklearn.model_selection import cross_val_score
from robert.utils import (load_variables,
    load_db_n_params,
    load_model,
    pd_to_dict,
    load_n_predict,
    finish_print
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

                # load and ML model parameters, and add standardized descriptors
                Xy_data, params_df, params_path, suffix_title = load_db_n_params(self,params_dir,"verify")
                
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

                # calculate scores for the X-shuffle test
                verify_results = self.xshuffle_test(verify_results,Xy_data,params_dict)

                # load and ML model parameters again (to avoid weird memory issues on Windows, for some 
                # reason the Xy_data dataframe changes when changing X descriptors in copy() objects)
                Xy_data, params_df, params_path, suffix_title = load_db_n_params(self,params_dir,"verify")

                # calculate scores for the y-shuffle test
                verify_results = self.yshuffle_test(verify_results,Xy_data,params_dict)

                # load and ML model parameters again (to avoid weird memory issues on Windows, for some 
                # reason the Xy_data dataframe changes when changing X descriptors in copy() objects)
                Xy_data, params_df, params_path, suffix_title = load_db_n_params(self,params_dir,"verify")

                # one-hot test (check that if a value isnt 0, the value assigned is 1)
                verify_results = self.onehot_test(verify_results,Xy_data,params_dict)

                # load and ML model parameters again (to avoid weird memory issues on Windows, for some 
                # reason the Xy_data dataframe changes when changing X descriptors in copy() objects)
                Xy_data, params_df, params_path, suffix_title = load_db_n_params(self,params_dir,"verify")

                # analysis of results
                colors,color_codes,results_print = self.analyze_tests(verify_results,params_dict)

                # plot a donut plot with the results
                print_ver,path_n_suffix = self.plot_donut(colors,color_codes,params_path,suffix_title)

                # print and save results
                _ = self.print_verify(results_print,verify_results,print_ver,path_n_suffix)

        _ = finish_print(self,start_time,'VERIFY')


    def cv_test(self,verify_results,Xy_data,params_dict):
        '''
        Performs a K-fold cross-validation on the training set.
        '''

        # adjust the scoring type
        if params_dict['type'] == 'reg':
            if verify_results['error_type'] == 'r2':
                scoring = "r2"
            elif verify_results['error_type'] == 'mae':
                scoring = "neg_mean_absolute_error"
            elif verify_results['error_type'] == 'rmse':
                scoring = "neg_root_mean_squared_error"
        elif params_dict['type'] == 'clas':
            if verify_results['error_type'] == 'acc':
                scoring = "accuracy"
            elif verify_results['error_type'] == 'f1':
                scoring = "f1"
            elif verify_results['error_type'] == 'mcc':
                scoring = "mcc"        
        
        loaded_model = load_model(params_dict)
        # Fit the model with the training set
        loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])  

        cv_score = cross_val_score(loaded_model, Xy_data['X_train_scaled'], 
                    Xy_data['y_train'], cv=self.args.kfold, scoring=scoring)
        # for MAE and RMSE, sklearn takes negative values
        if verify_results['error_type'] in ['mae','rmse']:
            cv_score = -cv_score
        verify_results['cv_score'] = cv_score.mean()
        verify_results['cv_std'] = cv_score.std()
        
        return verify_results


    def xshuffle_test(self,verify_results,Xy_data,params_dict):
        '''
        Calculate the accuracy of the model when the X data is randomly shuffled (columns are 
        randomly shuffled). For example, a descriptor array of X1, X2, X3, X4 might become 
        X2, X4, X1, X3.
        '''

        Xy_xshuffle = Xy_data.copy()
        random_state_xshuff = self.args.seed
        for _,column in enumerate(Xy_xshuffle['X_train_scaled']):
            Xy_xshuffle['X_train_scaled'][column] = Xy_xshuffle['X_train_scaled'][column].sample(frac=1,random_state=random_state_xshuff,ignore_index=True,axis=0)
            random_state_xshuff += 1
        for _,column in enumerate(Xy_xshuffle['X_valid_scaled']):
            Xy_xshuffle['X_valid_scaled'][column] = Xy_xshuffle['X_valid_scaled'][column].sample(frac=1,random_state=random_state_xshuff,ignore_index=True,axis=0)
            random_state_xshuff += 1
        Xy_xshuffle = load_n_predict(params_dict, Xy_xshuffle)  
        verify_results['X_shuffle'] = Xy_xshuffle[f'{verify_results["error_type"]}_valid']

        return verify_results


    def yshuffle_test(self,verify_results,Xy_data,params_dict):
        '''
        Calculate the accuracy of the model when the y values are randomly shuffled (rows are randomly 
        shuffled). For example, a y array of 1.3, 2.1, 4.0, 5.2 might become 2.1, 1.3, 5.2, 4.0.
        '''

        Xy_yshuffle = Xy_data.copy()
        Xy_yshuffle['y_train'] = Xy_yshuffle['y_train'].sample(frac=1,random_state=self.args.seed,axis=0)
        Xy_yshuffle['y_valid'] = Xy_yshuffle['y_valid'].sample(frac=1,random_state=self.args.seed,axis=0)
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


    def analyze_tests(self,verify_results,params_dict):
        '''
        Function to check whether the tests pass and retrieve the corresponding colors:
        1. Blue for passing tests
        2. Red for failing tests
        3. Grey for the CV test when using KN-based data splitting (to account for misleading results)
        '''

        blue_color = '#1f77b4'
        red_color = '#cd5c5c'
        grey_color = '#c1cdd3'
        color_codes = {'blue' : blue_color,
                        'red' : red_color,
                        'grey' : grey_color}
        colors = [None,None,None,None]
        results_print = [None,None,None,None]
        # these thresholds use train results to compare in the CV
        higher_thres_train = (1+self.args.thres_test)*verify_results['original_score_train']
        lower_thres_train = (1-self.args.thres_test)*verify_results['original_score_train']
        # these thresholds use validation results to compare in the CV
        higher_thres_valid = (1+self.args.thres_test)*verify_results['original_score_valid']
        lower_thres_valid = (1-self.args.thres_test)*verify_results['original_score_valid']

        for i,test_ver in enumerate(['X_shuffle', 'cv_score', 'onehot', 'y_shuffle']):
            # the CV test should give values as good as the originals, while the other tests
            # should give worse results. MAE and RMSE go in the opposite direction as R2,
            # F1 scores and MCC
            if test_ver == 'cv_score':
                if verify_results['error_type'] in ['mae','rmse']:
                    if verify_results[test_ver] >= higher_thres_train:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {self.args.kfold}-fold CV: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2} is higher than the threshold ({higher_thres_train:.2})'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {self.args.kfold}-fold CV: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2} is lower than the threshold ({higher_thres_train:.2})'
                else:
                    if verify_results[test_ver] <= lower_thres_train:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {self.args.kfold}-fold CV: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2} is lower than the threshold ({lower_thres_train:.2})'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {self.args.kfold}-fold CV: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2} is higher than the threshold ({lower_thres_train:.2})'
                
                # the CV test also fails if there is too much variation (+- 50% of the CV result)
                if verify_results['cv_std'] >= 0.5*verify_results['cv_score']:
                    colors[i] = red_color
                    results_print[i] = f'\n         x {self.args.kfold}-fold CV: FAILED, SD 50% higher than the CV score. CV result: {verify_results["error_type"].upper()} = {verify_results["cv_score"]:.2} +- {verify_results["cv_std"]:.2}'

                # when using K-neighbours to select the training data, classical K-fold CV might not
                # be very useful. We mark this part in grey
                if params_dict['split'] == 'KN':
                    colors[i] = grey_color
                    results_print[i] = f'\n         - {self.args.kfold}-fold CV: NOT DETERMINED, data splitting was done with KN. CV result: {verify_results["error_type"].upper()} = {verify_results["cv_score"]:.2}'

            elif test_ver in ['X_shuffle', 'y_shuffle', 'onehot']:
                if verify_results['error_type'] in ['mae','rmse']:
                    if verify_results[test_ver] <= higher_thres_valid:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2} is lower than the threshold ({higher_thres_valid:.2})'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2} is higher than the threshold ({higher_thres_valid:.2})'
                else:
                    if verify_results[test_ver] >= lower_thres_valid:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2} is higher than the threshold ({lower_thres_valid:.2})'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {verify_results[test_ver]:.2} is lower than the threshold ({lower_thres_valid:.2})'

        return colors,color_codes,results_print


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
        
        recipe = ["X-shuffle",
                f"{self.args.kfold}-fold CV",
                "One-hot",
                "y-shuffle"]
                
        # make 4 even parts in the donut plot
        data = [25, 25, 25, 25]
        explode = [None, None, None, None]
        
        # failing or undetermined tests will lead to pieces that are outside the regular donut
        for i,color in enumerate(colors):
            if color in [color_codes['red'], color_codes['grey']]:
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
        plt.title(title_verify, y=1.04, fontsize = fontsize)

        # save plot
        verify_plot_file = f'{os.path.dirname(path_n_suffix)}/VERIFY_tests_{os.path.basename(path_n_suffix)}.png'
        plt.savefig(verify_plot_file, dpi=300, bbox_inches='tight')
        plt.clf()
        print_ver = f"\n   o  VERIFY donut plots saved in {verify_plot_file}"

        return print_ver, path_n_suffix


    def print_verify(self,results_print,verify_results,print_ver,path_n_suffix):
        '''
        Print and store the results of VERIFY
        '''
        
        verify_results_file = f'{os.path.dirname(path_n_suffix)}/VERIFY_tests_{os.path.basename(path_n_suffix)}.dat'
        print_ver += f"\n   o  VERIFY test values saved in {verify_results_file}:"
        print_ver += f'\n      Results of the VERIFY tests:'
        # the printing order should be CV, X-shuffle, y-shuffle and one-hot
        print_ver += f'\n      Original score (train set for CV): {verify_results["error_type"].upper()} = {verify_results["original_score_train"]:.2}, with a +- threshold (thres_test option) of {self.args.thres_test*100}%:'
        print_ver += results_print[1]
        print_ver += f'\n      Original score (validation set): {verify_results["error_type"].upper()} = {verify_results["original_score_valid"]:.2}, with a +- threshold (thres_test option) of {self.args.thres_test*100}%:'
        print_ver += results_print[0]
        print_ver += results_print[3]
        print_ver += results_print[2]
        self.args.log.write(print_ver)
        dat_results = open(verify_results_file, "w")
        dat_results.write(print_ver)
        dat_results.close()
