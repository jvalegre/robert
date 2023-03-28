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
     thres_test : int, default=0.2,
         Threshold used to determine if a test pasess. It is determined in % units of diference between
         the R2 (MCC in classificators) of the model and the test (i.e., 0.2 = 20% difference with the 
         original value). Test passes if:
            1. x- and y-shuffle tests: decreases more than X% (from original R2, regressors, or MCC, classificators)
            2. One-hot encoding test: decreases more than X%
            3. K-fold cross validation: decreases less than X%
     kfold : int, default=5,
         The training set is split into a K number of folds in the cross-validation test (i.e. 5-fold CV).
     error_type : str, default: r2 (regression), mcc (classification)
         Target value used during the hyperopt optimization. Options:
         Regression:
            1. rmse (root-mean-square error)
            2. mae (mean absolute error)
            3. r2 (R-squared)
         Classification:
            1. mcc (Matthew's correlation coefficient)
            2. f1_score (F1 score)
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
    pd_to_dict
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

        # load and ML model parameters, and add standardized descriptors
        if self.args.model_dir == '':
            model_dir = 'GENERATE/Best_model/No_PFI'
        Xy_data, params_df = load_db_n_params(self,model_dir,"verify")
        
        # set the parameters for each ML model of the hyperopt optimization
        params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)
        loaded_model = load_model(params_dict)

        # Fit the model with the training set
        loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])  

        # this dictionary will keep the results of the tests
        verify_results = {} 

        # calculate R2 for k-fold cross validation (if needed)
        verify_results = self.cv_test(verify_results,Xy_data,loaded_model,params_dict)
        print(verify_results)

        # # load, fit model and predict values
        # Xy_xshuffle = Xy_data.copy()
        # shuffleing
        # Xy_xshuffle = load_n_predict(params, Xy_xshuffle, 'valid')
        # save this dataset and print graph with R2, RMSE, MAE below graph

        # Xy_yshuffle = Xy_data.copy()
        # shuffleing
        # Xy_yshuffle = load_n_predict(params, Xy_yshuffle, 'valid')
        # save this dataset and print graph with R2, RMSE, MAE below graph

        # one-hot test (check that if a value isnt 0, the value assigned is 1)

        # at the end, incluye si se pasan o se fallan los tests (X shuffle: PASSED (X vs X R2, X vs X RMSE, X vs X MAE))
        # opcion para cambiar los thresholds de pasar tests (20% de todos los valores de R2, RMSE, y MAE del modelo normal?)

    # # make donut plot
    # fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    # recipe = ["X-shuffle: passing",
    #         "5-fold CV: failing",
    #         "One-hot: passing",
    #         "y-shuffle: passing",
    #         ]
    # data = [25, 25, 25, 25]
    # blue_color = 'tab:blue'
    # red_color = 'indianred'
    # # red_color = 'lightsteelblue'
    # # red_color = '#CD4447'
    # colors = [blue_color, red_color, blue_color,blue_color]
    # explode = (0, 0.05, 0.0, 0)  
    
    # # in the pie(): wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' }
    # wedgeprops = {'width':0.4, 'edgecolor':'black', 'lw':0.72}
    # # wedgeprops=dict(width=0.4)
    # wedges, texts = ax.pie(data, wedgeprops=wedgeprops, startangle=0, colors=colors, explode=explode)

    # bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    # kw = dict(arrowprops=dict(arrowstyle="-"),
    #         bbox=bbox_props, zorder=0, va="center")

    # for i, p in enumerate(wedges):
    #     ang = (p.theta2 - p.theta1)/2. + p.theta1
    #     y = np.sin(np.deg2rad(ang))
    #     x = np.cos(np.deg2rad(ang))
    #     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    #     connectionstyle = f"angle,angleA=0,angleB={ang}"
    #     kw["arrowprops"].update({"connectionstyle": connectionstyle})
    #     ax.annotate(recipe[i], xy=(x, y), xytext=(1.15*np.sign(x), 1.4*y),
    #                 horizontalalignment=horizontalalignment, **kw)

    # ax.set_title("Statistical tests")
    # plt.savefig(f'Statistical tests.png', dpi=300, bbox_inches='tight')
    # plt.show()

    def cv_test(self,verify_results,Xy_data,loaded_model,params_dict):
        '''
        Performs a K-fold cross-validation on the training set.
        '''

        if params_dict['split'] == 'KN':
            self.args.log.write(f"\nx  The k-neighbours splitting (KN) was detected! Skipping cross validation since this analysis might show misleading results.")
        # else:
        #     # adjust the scoring type
        #     if self.args.error_type == 'r2':
        #         scoring = XX
        #     etc.
        #     cv_score = cross_val_score(loaded_model, Xy_data['X_train_scaled'], 
        #                 Xy_data['y_train'], cv=self.args.kfold, scoring=scoring)
        #     verify_results['cv_score'] = cv_score.mean()
        #     verify_results['cv_std'] = cv_score.std()
        
        return verify_results
