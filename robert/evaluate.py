"""
Parameters
----------

GENERAL
+++++++

    csv_name : str, default=''
        Name of the CSV file containing all the points used in the model (combining train + valid + test).
        A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv'). Include a 'Set' column with
        the value 'Test' on the rows that should be held out as the test set to control the
        split yourself - otherwise ROBERT splits it automatically (see the 'split'/'test_set'
        options), same as GENERATE.
    y : str, default=''
        Name of the column containing the response variable in the input CSV file (i.e. 'solubility').
    names : str, default=''
        Column of the names for each datapoint. Names are used to print outliers.
    eval_model : str, default='MVL'
        Legacy option, only used as a fallback when none of model_obj/model_file/model_params
        are provided. 'MVL' (Multivariate linear model, LinearRegression() in sklearn) is the
        only value accepted here - use model_obj/model_file/model_params for any other model.
    model_obj : object, default=None
        An already-fitted scikit-learn estimator object (Python API only, e.g. from a Jupyter
        notebook) to evaluate. Only its class and hyperparameters (via get_params()) are used -
        VERIFY/PREDICT's repeated cross-validation always refits fresh copies per fold, so the
        object's already-fitted weights themselves aren't reused.
    model_file : str, default=''
        Path to a fitted scikit-learn estimator saved with joblib/pickle. Same idea as
        model_obj, loaded from disk instead of passed directly.
    model_params : str, default=''
        Path to a CSV with two columns, 'param' and 'value', listing the exact scikit-learn
        class to use (a row with param='model', e.g. value='RandomForestRegressor') and its
        hyperparameters (one row each, e.g. param='n_estimators', value=200). Only scikit-learn's
        own registered estimators are accepted (see sklearn.utils.all_estimators()) - this never
        imports or executes anything outside sklearn.
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
import sys
import shutil
import time
import json
import ast
import pandas as pd
from pathlib import Path
from robert.utils import (load_variables, finish_print, load_database, prepare_sets,
    check_clas_problem, resolve_sklearn_estimator, model_adjust_params)
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

        # adjust options of classification problems and detects whether the right type of problem was used
        self = check_clas_problem(self,csv_df)
        # check_clas_problem() converts csv_df's y column to integer class codes in place;
        # csv_y was captured before that conversion, so it needs refreshing to stay in sync
        csv_y = csv_df[self.args.y]

        # resolve which model to evaluate (an already-fitted object/file, a CSV of
        # hyperparameters, or the legacy 'MVL' default) and validate it against --type
        self.model_params_dict = self.resolve_eval_model()

        # a user-provided 'Set' column picks the test set directly instead of the usual
        # automatic split (see get_user_defined_test_points()); BO_opt=True re-enables the
        # normal test_select()-based split when no such column/values are present
        test_points = self.get_user_defined_test_points(csv_df)
        Xy_data = prepare_sets(self,csv_df,csv_X,csv_y,test_points,self.args.names,None,None,None,BO_opt=(test_points is None))

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

    def resolve_eval_model(self):
        '''
        Decide which model EVALUATE will evaluate, and its hyperparameters. Priority order
        (most specific/explicit first): an already-fitted model object (model_obj, Python API
        only) -> a fitted model saved with joblib/pickle (model_file) -> a CSV naming a
        scikit-learn class and its hyperparameters (model_params) -> the legacy eval_model
        option (default 'MVL').

        A fitted model's own get_params() is read directly, so the user never has to retype
        hyperparameters by hand - only its class and hyperparameters are kept, since
        repeated_kfold_cv() always refits fresh copies of the model per fold (the object's
        already-fitted weights aren't reused as-is).

        Sets self.args.eval_model to the resolved class name (used everywhere downstream -
        VERIFY/PREDICT load models the same way regardless of where the name came from) and
        returns the resolved hyperparameters dict.
        '''

        if self.args.model_obj is not None:
            model_obj = self.args.model_obj
            model_name = type(model_obj).__name__
            params = model_obj.get_params()
            self.args.log.write(f"\no  Using the model object passed directly (model_obj): {model_name}")

        elif self.args.model_file:
            import joblib
            if not os.path.exists(self.args.model_file):
                self.args.log.write(f"\nx  WARNING! The model_file specified ({self.args.model_file}) does not exist!")
                sys.exit()
            model_obj = joblib.load(self.args.model_file)
            model_name = type(model_obj).__name__
            params = model_obj.get_params()
            self.args.log.write(f"\no  Loaded a pre-fitted model from {self.args.model_file}: {model_name}")

        elif self.args.model_params:
            if not os.path.exists(self.args.model_params):
                self.args.log.write(f"\nx  WARNING! The model_params CSV specified ({self.args.model_params}) does not exist!")
                sys.exit()
            params_df = pd.read_csv(self.args.model_params, encoding='utf-8')
            if not {'param','value'}.issubset(params_df.columns):
                self.args.log.write(f"\nx  WARNING! The model_params CSV must have 'param' and 'value' columns!")
                sys.exit()
            rows = dict(zip(params_df['param'], params_df['value']))
            if 'model' not in rows:
                self.args.log.write(f"\nx  WARNING! The model_params CSV must include a row with param='model' naming the scikit-learn class to use (e.g. value='RandomForestRegressor')!")
                sys.exit()
            model_name = str(rows.pop('model'))
            # values come from a CSV as plain text - recover their real type (int/float/bool/
            # None/list) where possible, otherwise keep the literal string (e.g. solver names)
            params = {}
            for param, value in rows.items():
                try:
                    params[param] = ast.literal_eval(str(value))
                except (ValueError, SyntaxError):
                    params[param] = value
            self.args.log.write(f"\no  Using the model and hyperparameters from {self.args.model_params}: {model_name}")

        else:
            # legacy default - MVL/LinearRegression, unchanged behavior
            model_name = self.args.eval_model
            params = {}

        # validate the resolved model against --type before committing to it - a dynamic
        # scikit-learn model (anything other than the legacy 'MVL') must exist as a real
        # sklearn estimator of the requested type (see resolve_sklearn_estimator())
        if model_name != 'MVL':
            if resolve_sklearn_estimator(model_name, self.args.type) is None:
                other_type = 'clas' if self.args.type.lower() == 'reg' else 'reg'
                if resolve_sklearn_estimator(model_name, other_type) is not None:
                    self.args.log.write(f"\nx  WARNING! '{model_name}' is a {other_type} estimator, but type='{self.args.type}' was used. Set type='{other_type}' instead.")
                else:
                    self.args.log.write(f"\nx  WARNING! '{model_name}' is not a valid scikit-learn {self.args.type} estimator name (or isn't available in this scikit-learn version). Check the exact class name (e.g., 'RandomForestRegressor').")
                sys.exit()

        self.args.eval_model = model_name

        # fills in random_state from --seed only if the class accepts one AND the user didn't
        # already set their own - an explicit value is respected as-is (see model_adjust_params())
        params = model_adjust_params(self, model_name, params)

        return params

    def get_user_defined_test_points(self,csv_df):
        '''
        Detects a user-provided 'Set' column marking which points to hold out as the test set
        (rows with value 'Test', case-insensitive), instead of auto-splitting with
        test_select(). Returns None if no such column/values are present, so the caller falls
        back to the normal automatic split.
        '''

        if 'Set' not in csv_df.columns:
            return None

        set_col = csv_df['Set'].astype(str).str.strip().str.lower()
        test_mask = set_col == 'test'
        if not test_mask.any():
            return None

        test_points = csv_df.index[test_mask].tolist()
        self.args.log.write(f"\no  Using the 'Set' column from the input CSV to select the test set ({len(test_points)} points) instead of an automatic split.")
        return test_points

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
        df_params['params'] = json.dumps(self.model_params_dict, default=str)
        df_params['X_descriptors'] = [list(Xy_data['X_descriptors'])]

        _ = df_params.to_csv(f'{generate_folder}/{self.args.eval_model}.csv', index = None, header=True)
