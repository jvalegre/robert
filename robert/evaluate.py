"""
Parameters
----------

GENERAL
+++++++

    csv_name : str, default=''
        Name of the CSV file containing all the points used in the model (combining train + valid + test).
        A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv'). 
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

Affect VERIFY and PREDICT
+++++++++++++++++++++++++

    kfold : int, default=5
        Number of random data splits for the cross-validation of the models. 
    repeat_kfolds : int, default=10
        Number of repetitions for the k-fold cross-validation of the models.

"""
#####################################################.
#        This file stores the EVALUATE class        #
#          used to evaluate existing models         #
#####################################################.

import os
import shutil
import time
import pandas as pd
from pathlib import Path
from robert.utils import load_variables, finish_print, load_database, prepare_sets
from robert.generate_utils import set_sets


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

        # clean folders from previous runs
        _ = self.clean_eval()

        # load database, discard user-defined descriptors and perform data checks
        csv_df, csv_X, csv_y = load_database(self,self.args.csv_name,"generate",print_info=False)

        # standardizes and separates an external test set
        Xy_data = prepare_sets(self,csv_df,csv_X,csv_y,None,self.args.names,None,None,None,BO_opt=True)

        # saves database and model params in the /GENERATE/Best_model/No_PFI folder
        _ = self.save_generate(csv_df,Xy_data)

        # finish the printing of the EVALUATE info file
        _ = finish_print(self,start_time,'EVALUATE')

    def clean_eval(self):
        '''
        Cleans folders from previous runs
        '''

        for folder in ['CURATE','GENERATE','VERIFY','PREDICT']:
            eval_folder = f'{Path(os.getcwd()).joinpath(folder)}'
            if os.path.exists(eval_folder):
                shutil.rmtree(eval_folder)

    def save_generate(self,csv_df,Xy_data):
        '''
        Saves database and model params in the /GENERATE/Best_model/No_PFI folder
        '''

        # copy database with Set column
        generate_folder = Path('GENERATE/Best_model/No_PFI')
        if os.path.exists(generate_folder):
            shutil.rmtree(generate_folder)
        Path(generate_folder).mkdir(exist_ok=True, parents=True)

        # include the Set column to differentiate between train and test sets (and external test, if any)
        csv_df = set_sets(csv_df,Xy_data)

        _ = csv_df.to_csv(f'{generate_folder}/{self.args.eval_model}_db.csv', index = None, header=True)

        # save all the parameters of the model
        df_params = pd.DataFrame()
        df_params['kfold'] = [self.args.kfold]
        df_params['repeat_kfolds'] = [self.args.repeat_kfolds]
        df_params['model'] = [self.args.eval_model]
        df_params['type'] = [self.args.type]
        df_params['seed'] = [self.args.seed]
        df_params['y'] = [self.args.y]
        df_params['names'] = [self.args.names]
        df_params['error_type'] = [self.args.error_type]
        df_params['params'] = '{}'
        df_params['X_descriptors'] = [list(Xy_data['X_descriptors'])]

        _ = df_params.to_csv(f'{generate_folder}/{self.args.eval_model}.csv', index = None, header=True)
