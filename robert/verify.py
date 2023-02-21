"""
Parameters
----------

General
+++++++

# Specify t-value that will be the threshold to identify outliers
# (check tables for t-values elsewhere). The higher the t-value 
# the more restrictive the analysis will be (i.e. there will 
# be more outliers with t-value=1 than with t-value = 4)
t_value = 2

   files : str or list of str, default=None
     Input files. Formats accepted: XYZ, SDF, GJF, COM and PDB. Also, lists can
     be used (i.e. [FILE1.sdf, FILE2.sdf] or \*.FORMAT such as \*.sdf).  
   program : str, default=None
     Program required in the conformational refining. 
     Current options: 'xtb', 'ani'
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
from scipy import stats
from sklearn.model_selection import cross_val_score
from robert.utils import load_variables


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

        # load database and ML model parameters
        path_no_PFI = f"{self.args.destination.joinpath('Best_model/No_PFI')}"
        if os.exists(path_no_PFI):
            for file in glob.glob(*.csv)
                csv_df = load_database(self,"verify")


        txt_load = f'\no  Database {self.args.csv_name} (with no {suffix}) and its corresponding ML model parameters loaded successfully, including:'
        txt_load += f'\n   - Target value:{self.args.y}'
        txt_load += f'\n   - Model:{XX}'
        txt_load += f'\n   - Descriptors:{XX}'
        txt_load += f'\n   - Training points:{XX}'
        txt_load += f'\n   - Validation points:{XX}'

        # load only the descriptors used in the model and standardize X
        Xy_train_df = csv_df[csv_df.Set == 'Training']
        Xy_valid_df = csv_df[csv_df.Set == 'Validation']

        X_train,X_valid = {},{} # (using a dict to keep the same format of load_model)
        for column in csv_df['X_descriptors']:
            X_train[column] = Xy_train_df[column]
            X_valid[column] = Xy_valid_df[column]
        y_train = Xy_train_df[csv_df['y'][0]]
        y_valid = Xy_valid_df[csv_df['y'][0]]

        Xy_data = {'X_train': X_train,
                   'X_valid': X_valid,
                   'y_train': y_train,
                   'y_valid': y_valid}

        Xy_data['X_train_scaled'], Xy_data['X_valid_scaled'] = standardize(Xy_data['X_train'],Xy_data['X_valid'])

        # set the parameters for each ML model of the hyperopt optimization
        loaded_model = load_model(params)

        # Fit the model with the training set
        loaded_model.fit(data['X_train_scaled'], data['y_train'])  

        te da un array, haz media y SD
        cv_score = cross_val_score(loaded_model, Xy_data['X_train_scaled'], Xy_data['y_train'], cv=self.args.cv_kfold)
        crossvalid (WARN that this test might not work correctly when using small datasets with Kneigh spliting)

        # load, fit model and predict values
        Xy_xshuffle = Xy_data.copy()
        shuffleing
        Xy_xshuffle = load_n_predict(params, Xy_xshuffle, 'valid')
        save this dataset and print graph with R2, RMSE, MAE below graph

        Xy_yshuffle = Xy_data.copy()
        shuffleing
        Xy_yshuffle = load_n_predict(params, Xy_yshuffle, 'valid')
        save this dataset and print graph with R2, RMSE, MAE below graph

        one-hot test (check that if a value isnt 0, the value assigned is 1)

        at the end, incluye si se pasan o se fallan los tests (X shuffle: PASSED (X vs X R2, X vs X RMSE, X vs X MAE))
        opcion para cambiar los thresholds de pasar tests (20% de todos los valores de R2, RMSE, y MAE del modelo normal?)
