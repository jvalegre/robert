"""
Parameters
----------

General
+++++++

     destination : str, default=None,
         Directory to create the output file(s).
     varfile : str, default=None
         Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
     model_dir : str, default=''
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

"""
#####################################################.
#        This file stores the VERIFY class          #
#           used for ML model analysis              #
#####################################################.

import os
import sys
import time
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import glob
from scipy import stats
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

        # if model_dir = '', the program performs the tests for the No_PFI and PFI folders
        if 'GENERATE/Best_model' in self.args.model_dir:
            model_dirs = [f'{self.args.model_dir}/No_PFI',f'{self.args.model_dir}/PFI']
        else:
            model_dirs = [self.args.model_dir]

        for model_dir in model_dirs:
            if os.path.exists(model_dir):

                # load and ML model parameters, and add standardized descriptors
                Xy_data, params_df = load_db_n_params(self,model_dir,"verify")
                
                # set the parameters for each ML model of the hyperopt optimization
                params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

                # this dictionary will keep the results of the tests
                verify_results = {'error_type': self.args.error_type} 

                # get original score
                Xy_orig = Xy_data.copy()
                Xy_orig = load_n_predict(params_dict, Xy_orig, 'valid')  
                verify_results['original_score'] = Xy_orig[self.args.error_type]

                # calculate R2 for k-fold cross validation (if needed)
                verify_results = self.cv_test(verify_results,Xy_data,params_dict)

                # calculate scores for the X-shuffle test
                verify_results = self.xshuffle_test(verify_results,Xy_data,params_dict)

                # calculate scores for the y-shuffle test
                verify_results = self.yshuffle_test(verify_results,Xy_data,params_dict)

                # one-hot test (check that if a value isnt 0, the value assigned is 1)
                verify_results = self.onehot_test(verify_results,Xy_data,params_dict)

                # analysis of results
                colors,color_codes,results_print = self.analyze_tests(verify_results,params_dict)

                # plot a donut plot with the results
                _ = self.plot_donut(colors,color_codes,model_dir)

                # print results
                _ = self.print_verify(results_print,verify_results)

        _ = finish_print(self,start_time,'VERIFY')


    def cv_test(self,verify_results,Xy_data,params_dict):
        '''
        Performs a K-fold cross-validation on the training set.
        '''

        # adjust the scoring type
        if params_dict['mode'] == 'reg':
            if self.args.error_type == 'r2':
                scoring = "r2"
            elif self.args.error_type == 'mae':
                scoring = "neg_mean_absolute_error"
            elif self.args.error_type == 'rmse':
                scoring = "neg_root_mean_squared_error"
        elif params_dict['mode'] == 'clas':
            if self.args.error_type == 'acc':
                scoring = "accuracy"
            elif self.args.error_type == 'f1':
                scoring = "f1"
            elif self.args.error_type == 'mcc':
                scoring = "mcc"        
        
        loaded_model = load_model(params_dict)
        # Fit the model with the training set
        loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])  

        cv_score = cross_val_score(loaded_model, Xy_data['X_train_scaled'], 
                    Xy_data['y_train'], cv=self.args.kfold, scoring=scoring)
        # for MAE and RMSE, sklearn takes negative values
        if self.args.error_type in ['mae','rmse']:
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
        Xy_xshuffle['X_train_scaled'] = Xy_xshuffle['X_train_scaled'].sample(frac=1,random_state=self.args.seed,axis=1)
        Xy_xshuffle['X_valid_scaled'] = Xy_xshuffle['X_valid_scaled'].sample(frac=1,random_state=self.args.seed,axis=1)
        Xy_xshuffle = load_n_predict(params_dict, Xy_xshuffle, 'valid')  
        verify_results['X_shuffle'] = Xy_xshuffle[self.args.error_type]

        return verify_results


    def yshuffle_test(self,verify_results,Xy_data,params_dict):
        '''
        Calculate the accuracy of the model when the y values are randomly shuffled (rows are randomly 
        shuffled). For example, a y array of 1.3, 2.1, 4.0, 5.2 might become 2.1, 1.3, 5.2, 4.0.
        '''

        Xy_yshuffle = Xy_data.copy()
        Xy_yshuffle['y_train'] = Xy_yshuffle['y_train'].sample(frac=1,random_state=self.args.seed,axis=0)
        Xy_yshuffle['y_valid'] = Xy_yshuffle['y_valid'].sample(frac=1,random_state=self.args.seed,axis=0)
        Xy_yshuffle = load_n_predict(params_dict, Xy_yshuffle, 'valid')  
        verify_results['y_shuffle'] = Xy_yshuffle[self.args.error_type]

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

        Xy_onehot = load_n_predict(params_dict, Xy_onehot, 'valid')  
        verify_results['onehot'] = Xy_onehot[self.args.error_type]

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
        higher_thres = (1+self.args.thres_test)*verify_results['original_score']
        lower_thres = (1-self.args.thres_test)*verify_results['original_score']

        for i,test_ver in enumerate(['X_shuffle', 'cv_score', 'onehot', 'y_shuffle']):
            # the CV test should give values as good as the originals, while the other tests
            # should give worse results. MAE and RMSE go in the opposite direction as R2,
            # F1 scores and MCC
            if test_ver == 'cv_score':
                if self.args.error_type in ['mae','rmse']:
                    if verify_results[test_ver] >= higher_thres:
                        colors[i] = red_color
                        results_print[i] = f'\n      x {self.args.kfold}-fold CV: FAILED, {self.args.error_type.upper()} = {verify_results[test_ver]:.2} is higher than the threshold ({higher_thres:.2})'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n      o {self.args.kfold}-fold CV: PASSED, {self.args.error_type.upper()} = {verify_results[test_ver]:.2} is lower than the threshold ({higher_thres:.2})'
                else:
                    if verify_results[test_ver] <= lower_thres:
                        colors[i] = red_color
                        results_print[i] = f'\n      x {self.args.kfold}-fold CV: FAILED, {self.args.error_type.upper()} = {verify_results[test_ver]:.2} is lower than the threshold ({lower_thres:.2})'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n      o {self.args.kfold}-fold CV: PASSED, {self.args.error_type.upper()} = {verify_results[test_ver]:.2} is higher than the threshold ({lower_thres:.2})'
                
                # the CV test also fails if there is too much variation (+- 50% of the CV result)
                if verify_results['cv_std'] >= 0.5*verify_results['cv_score']:
                    colors[i] = red_color
                    results_print[i] = f'\n      x {self.args.kfold}-fold CV: FAILED, SD 50% higher than the CV score. CV result : {self.args.error_type.upper()} = {verify_results["cv_score"]:.2} +- {verify_results["cv_std"]:.2}'

                # when using K-neighbours to select the training data, classical K-fold CV might not
                # be very useful. We mark this part in grey
                if params_dict['split'] == 'KN':
                    colors[i] = grey_color
                    results_print[i] = f'\n      - {self.args.kfold}-fold CV: NOT DETERMINED, data splitting was done with k-neighbours (KN). CV result : {self.args.error_type.upper()} = {verify_results["cv_score"]:.2}'

            elif test_ver in ['X_shuffle', 'y_shuffle', 'onehot']:
                if self.args.error_type in ['mae','rmse']:
                    if verify_results[test_ver] <= higher_thres:
                        colors[i] = red_color
                        results_print[i] = f'\n      x {test_ver}: FAILED, {self.args.error_type.upper()} = {verify_results[test_ver]:.2} is lower than the threshold ({higher_thres:.2})'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n      o {test_ver}: PASSED, {self.args.error_type.upper()} = {verify_results[test_ver]:.2} is higher than the threshold ({higher_thres:.2})'
                else:
                    if verify_results[test_ver] >= lower_thres:
                        colors[i] = red_color
                        results_print[i] = f'\n      x {test_ver}: FAILED, {self.args.error_type.upper()} = {verify_results[test_ver]:.2} is higher than the threshold ({lower_thres:.2})'
                    else:
                        colors[i] = blue_color
                        results_print[i] = f'\n      o {test_ver}: PASSED, {self.args.error_type.upper()} = {verify_results[test_ver]:.2} is lower than the threshold ({lower_thres:.2})'

        return colors,color_codes,results_print


    def plot_donut(self,colors,color_codes,model_dir):
        '''
        Creates a donut plot with the results of VERIFY
        '''

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
        
        # in the pie(): wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' }
        wedgeprops = {'width':0.4, 'edgecolor':'black', 'lw':0.72}
        # wedgeprops=dict(width=0.4)
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

        suffix = '(with no PFI filter)'
        label = 'No_PFI'
        if 'PFI' in model_dir and 'No_PFI' not in model_dir:
            suffix = '(with PFI filter)'
            label = 'PFI'
        ax.set_title(f"VERIFY tests {suffix}", fontsize=fontsize)
        plt.savefig(f'VERIFY_tests_{label}.png', dpi=300, bbox_inches='tight')


    def print_verify(self,results_print,verify_results):
        txt_ver = f'\n   Results of the verify tests. Original score: {self.args.error_type.upper()} = {verify_results["original_score"]:.2}, with a +- threshold (thres_test option) of {self.args.thres_test}:'
        # the printing order should be CV, X-shuffle, y-shuffle and one-hot
        txt_ver += results_print[1]
        txt_ver += results_print[0]
        txt_ver += results_print[3]
        txt_ver += results_print[2]
        self.args.log.write(txt_ver)
