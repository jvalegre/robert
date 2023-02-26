"""
Parameters
----------

General
+++++++

     destination : str, default=None,
         Directory to create the output file(s).
     varfile : str, default=None
         Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
     seed : int, default=8,
         Random seed used in the ML predictor models, data splitting and other protocols.
     thres_test : int, default=0.2,
         Threshold used to determine if a test pasess. It is determined in % units of diference between
         the R2 (MCC in classificators) of the model and the test (i.e., 0.2 = 20% difference with the 
         original value). Test passes if:
            1. x- and y-shuffle tests: decreases more than X% (from original R2, regressors, or MCC, classificators)
            2. One-hot encoding test: decreases more than X%
            3. K-fold cross validation: decreases less than X%
     kfold : int, default=5,
         The training set is split into a K number of folds in the cross-validation test (i.e. 5-fold CV).

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
    load_database,
    standardize,
    load_db_n_params
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
        Xy_data, params_df = load_db_n_params(self,'Best_model/No_PFI',"verify")
        
    def load_db_n_params(self,folder_model,module):
        path_no_PFI = f"{self.args.destination.joinpath(folder_model)}"
        if os.exists(path_no_PFI):
            csv_files = glob.glob(path_no_PFI.joinpath("*.csv").as_posix())
            if len(csv_files) != 2:
                self.args.log.write(f"\nx  There are too many CSV files in the {path_no_PFI} folder! Only two CSV files should be there, one with the model parameters and the other with the Xy database.")
                self.args.log.finalize()
                sys.exit()
            for csv_file in csv_files:
                if '_db' not in csv_file:
                    Xy_data_df = load_database(self,csv_file,module)
                else:
                    params_df = load_database(self,csv_file,module)
                    params_name = os.path.basename(csv_file)
        else:
            self.args.log.write(f"\nx  The folder with the model and database ({path_no_PFI}) does not exist! Did you use the destination=PATH option in the other modules?")

        # load only the descriptors used in the model and standardize X
        Xy_train_df = Xy_data_df[Xy_data_df.Set == 'Training']
        Xy_valid_df = Xy_data_df[Xy_data_df.Set == 'Validation']

        X_train,X_valid = {},{} # (using a dict to keep the same format of load_model)
        for column in params_df['X_descriptors']:
            X_train[column] = Xy_train_df[column]
            X_valid[column] = Xy_valid_df[column]
        y_train = Xy_train_df[params_df['y'][0]]
        y_valid = Xy_valid_df[params_df['y'][0]]

        Xy_data = {'X_train': X_train,
                   'X_valid': X_valid,
                   'y_train': y_train,
                   'y_valid': y_valid}

        Xy_data['X_train_scaled'], Xy_data['X_valid_scaled'] = standardize(Xy_data['X_train'],Xy_data['X_valid'])
        
        _ = csv_load_info(self,params_name,'no PFI filter',params_df)

        return Xy_data, params_df

    def csv_load_info(self,params_name,suffix,params_df):
        txt_load = f'\no  ML model {params_name} (with {suffix}) and its corresponding Xy database were loaded successfully, including:'
        txt_load += f'\n   - Target value:{params_df['y'][0]}'
        txt_load += f'\n   - Model:{XX}'
        txt_load += f'\n   - Descriptors:{XX}'
        txt_load += f'\n   - Training points:{XX}'
        txt_load += f'\n   - Validation points:{XX}'
        self.args.log.write(txt_load)


        # # set the parameters for each ML model of the hyperopt optimization
        # loaded_model = load_model(params)

        # # Fit the model with the training set
        # loaded_model.fit(data['X_train_scaled'], data['y_train'])  

        # te da un array, haz media y SD
        # cv_score = cross_val_score(loaded_model, Xy_data['X_train_scaled'], Xy_data['y_train'], cv=self.args.cv_kfold)
        # crossvalid (WARN that this test might not work correctly when using small datasets with Kneigh spliting)

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
