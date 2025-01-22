"""
Parameters
----------

GENERAL
+++++++

    csv_train : str, default=''
        Name of the CSV file containing the train set. A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv'). 
    csv_valid : str, default=''
        Name of the CSV file containing the validation set. A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv'). 
    csv_test : str, default=''
        Name of the CSV file containing the test set (if any). A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv'). 
    y : str, default=''
        Name of the column containing the response variable in the input CSV file (i.e. 'solubility'). 
    names : str, default=''
        Column of the names for each datapoint. Names are used to print outliers.
    eval_model : str, default='MVL'
        ML models that can be evaluated (for now, only models from sklearn are accepted, more options will be added): 
        1. 'MVL' (Multivariate lineal models, LinearRegression() in sklearn)
    type : str, default='reg'
        Type of the pedictions. Options: 
        1. 'reg' (Regressor)
        2. 'clas' (Classifier)
    seed : int, default=0
        Random seed used in the ML predictor models and other protocols.
    destination : str, default=None,
        Directory to create the output file(s).
    
Affect VERIFY
+++++++++++++

    error_type : str, default: rmse (regression), mcc (classification)
        Target value used during the VERIFY evaluation. Options:
        Regression:
        1. rmse (root-mean-square error)
        2. mae (mean absolute error)
        3. r2 (R-squared, not recommended since R2 might be good even with high errors in small datasets)
        Classification:
        1. mcc (Matthew's correlation coefficient)
        2. f1 (F1 score)
        3. acc (accuracy, fraction of correct predictions)

Affect PREDICT
++++++++++++++

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

Affect VERIFY and PREDICT
+++++++++++++++++++++++++

    kfold : int, default=5
        Number of random data splits for the cross-validation of the models. 
    repeat_kfolds : int, default='auto'
        Number of repetitions for the k-fold cross-validation of the models. If 'auto',
        repeat_kfolds = 10 for <50 datapoints and 5 otherwise.

"""
#####################################################.
#        This file stores the EVALUATE class        #
#          used to evaluate existing models         #
#####################################################.

import os
import sys
import shutil
import time
import pandas as pd
from pathlib import Path
from robert.utils import load_variables, finish_print, load_database


class evaluate:
    """
    Class containing all the functions from the EVALUATE module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the EVALUATE module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "evaluate")

        # load databases, merge them and save CSV for ROBERT workflow
        csv_df_train,csv_df_valid = self.load_sets()

        # saves database and model params in the /GENERATE/Best_model/No_PFI folder
        _ = self.save_generate(csv_df_train,csv_df_valid)

        # finish the printing of the EVALUATE info file
        _ = finish_print(self,start_time,'EVALUATE')


    def load_sets(self):
        '''
        Load databases, merge them and save CSV (tracking the test types using the Set column)
        '''
        
        csv_df_train,_,_ = load_database(self,self.args.csv_train,"evaluate",print=False)
        csv_df_train['Set'] = ['Training'] * len(csv_df_train[self.args.y])
        csv_df_valid,_,_ = load_database(self,self.args.csv_valid,"evaluate",print=False)
        csv_df_valid['Set'] = ['Validation'] * len(csv_df_valid[self.args.y])
        if self.args.csv_test != '':
            csv_df_test,_,_ = load_database(self,self.args.csv_test,"evaluate",print=False)
            csv_df_test['Set'] = ['Test'] * len(csv_df_test[self.args.y])

        self.args.ignore.append('Set')

        # ensure that the datasets have the same number of columns and save combined database
        if len(csv_df_train.columns) != len(csv_df_valid.columns):
            self.args.log.write(f"\nx  Training and validation CSV files do not have the same number of columns!")
            self.args.log.finalize()
            sys.exit()
        elif self.args.csv_test != '' and len(csv_df_train.columns) != len(csv_df_test.columns):
            self.args.log.write(f"\nx  Training and test CSV files do not have the same number of columns!")
            self.args.log.finalize()
            sys.exit()

        csv_df = pd.concat([csv_df_train,csv_df_valid], axis=0).reset_index(drop=True)
        if self.args.csv_test != '':
            csv_df = pd.concat([csv_df,csv_df_test], axis=0).reset_index(drop=True)
        
        csv_basename = os.path.basename(f'{self.args.csv_train}').split('.')[0]
        self.args.csv_name = f'{csv_basename}_EVAL_db.csv'
        _ = csv_df.to_csv(f'{self.args.csv_name}', index = None, header=True)

        return csv_df_train,csv_df_valid


    def save_generate(self,csv_df_train,csv_df_valid):
        '''
        Saves database and model params in the /GENERATE/Best_model/No_PFI folder
        '''
        
        # copy database
        generate_folder = Path('GENERATE/Best_model/No_PFI')
        if os.path.exists(generate_folder):
            shutil.rmtree(generate_folder)
        Path(generate_folder).mkdir(exist_ok=True, parents=True)

        train_points = len(csv_df_train[self.args.y])
        valid_points = len(csv_df_valid[self.args.y])
        train_proportion = round((train_points/(train_points+valid_points))*100)
        adapted_name = f'{self.args.eval_model}_{train_proportion}'

        shutil.copy(f"{self.args.csv_name}", f"{generate_folder}/{adapted_name}_db.csv")

        # save all the parameters of the model
        df_params = pd.DataFrame()
        train_points = len(csv_df_train[self.args.y])
        valid_points = len(csv_df_valid[self.args.y])
        df_params['train'] = [train_proportion]
        df_params['kfold'] = [self.args.kfold]
        df_params['repeat_kfolds'] = [self.args.repeat_kfolds]
        df_params['model'] = [self.args.eval_model]
        df_params['type'] = [self.args.type]
        df_params['seed'] = [self.args.seed]
        df_params['y'] = [self.args.y]
        df_params['names'] = [self.args.names]
        df_params['error_type'] = [self.args.error_type]
        # self.arg.names and "Set" are already in self.args.ignore
        descriptors = csv_df_train.drop(self.args.ignore+[self.args.y], axis=1).columns
        df_params['X_descriptors'] = [list(descriptors)]

        _ = df_params.to_csv(f'{generate_folder}/{adapted_name}.csv', index = None, header=True)
