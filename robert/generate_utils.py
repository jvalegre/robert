#####################################################.
#     This file stores functions from GENERATE      #
#####################################################.

import os
import sys
import shutil
from pathlib import Path
import random
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
import matplotlib.colors as mcolor
# for users with no intel architectures. This part has to be before the sklearn imports
try:
    from sklearnex import patch_sklearn
    patch_sklearn(verbose=False)
except (ModuleNotFoundError,ImportError):
    pass
from sklearn.cluster import KMeans
import yaml
import json
import glob
from pkg_resources import resource_filename
from sklearn.inspection import permutation_importance
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from robert.utils import (
    load_model,
    load_n_predict,
    standardize,
    pd_to_dict)


# hyperopt workflow
def hyperopt_workflow(self, csv_df, ML_model, size, Xy_data_hp, seed):

    # edit this function to modify the hyperopt parameter optimization (i.e. the 
    # lists represent values to include in the grid search)
    space4rf = hyperopt_params(self, ML_model)

    # Run hyperopt
    trials = Trials()

    # This best high initial number is a dummy value used as the start of the optimization
    # (otherwise it does not run more than once since best becomes a dictionary)
    best = float('inf')

    hyperopt_data = {'best': best, 'model': ML_model.upper(),
                'type': self.args.type.lower(),
                'split': self.args.split.upper(),
                'train': size, 
                'seed': seed,
                'error_type': self.args.error_type.lower(),
                'y': self.args.y,
                'X_descriptors': Xy_data_hp['X_descriptors'],
                'destination': self.args.destination.as_posix()}

    # adjust the format for the sklearn models and add the data to the dict
    for desc in ['X_train_scaled','y_train','X_valid_scaled','y_valid']:
        Xy_data_hp[desc] = np.asarray(Xy_data_hp[desc]).tolist()
        hyperopt_data[desc] = Xy_data_hp[desc]

    # save the initial json
    with open('hyperopt.json', 'w') as outfile:
        json.dump(hyperopt_data, outfile)

    if ML_model.upper() == 'MVL':
        max_evals = 1 # use all the parameters, it is not possible to adjust parameters for MVL
    else:
        max_evals = self.args.epochs
    try:
        # this part allows users to modify exploitation/exploration, replaces the fmin() function below
        # from hyperopt import partial, mix, anneal, rand
        # best = fmin(f, space4rf, algo=partial(mix.suggest,
        #         p_suggest=[
        #         (.1, rand.suggest),
        #         (.1, anneal.suggest),
        #         (.8, tpe.suggest),]),
        #         max_evals=max_evals, trials=trials, rstate=np.random.default_rng(hyperopt_data['seed']))
        best = fmin(f, space4rf, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(hyperopt_data['seed']))
        if os.path.exists('hyperopt.json'):
            os.remove('hyperopt.json')
    except ValueError:
        self.args.log.write('\nx  There is an error in the hyperopt module, 1) are you using type ="clas" for regression y values instead of type="reg"? or 2) are you using very small partition sizes for validation sets (fix with train="[60,70]" for example)?')

    if os.path.exists('hyperopt.json'):
        os.remove('hyperopt.json')

    # copy the database used
    # set a new column for the set
    set_column = []
    n_points = len(csv_df[csv_df.columns[0]])
    for i in range(0,n_points):
        if i in Xy_data_hp['training_points']:
            set_column.append('Training')
        else:
            set_column.append('Validation')
    csv_df['Set'] = set_column

    # save the csv file
    if os.path.exists(self.args.destination.joinpath(f"Raw_data/No_PFI/{ML_model}_{size}_{seed}.csv")):
        db_name = self.args.destination.joinpath(f"Raw_data/No_PFI/{ML_model}_{size}_{seed}_db")
        _ = csv_df.to_csv(f'{db_name}.csv', index = None, header=True)

# generates initial parameters for the hyperopt optimization
def hyperopt_params(self, model_type):
    # load the parameters of the models from their corresponding yaml files
    params = load_params(self,model_type)

    if model_type.upper() == 'RF':
        space4rf_hyperopt = {'max_depth': hp.choice('max_depth', params['max_depth']),
                'max_features': hp.choice('max_features', params['max_features']),
                'n_estimators': hp.choice('n_estimators', params['n_estimators']),
                'min_samples_split': hp.choice('min_samples_split', params['min_samples_split']),
                'min_samples_leaf': hp.choice('min_samples_leaf', params['min_samples_leaf']),
                'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', params['min_weight_fraction_leaf']),
                'oob_score': hp.choice('oob_score', params['oob_score']),
                'ccp_alpha': hp.choice('ccp_alpha', params['ccp_alpha']),
                'max_samples': hp.choice('max_samples', params['max_samples']),
                }  

    elif model_type.upper() == 'GB':
        space4rf_hyperopt = {'max_depth': hp.choice('max_depth', params['max_depth']),
                'max_features': hp.choice('max_features', params['max_features']),
                'n_estimators': hp.choice('n_estimators', params['n_estimators']),
                'learning_rate': hp.choice('learning_rate', params['learning_rate']),
                'validation_fraction': hp.choice('validation_fraction', params['validation_fraction']),
                'subsample': hp.choice('subsample', params['subsample']),
                'min_samples_split': hp.choice('min_samples_split', params['min_samples_split']),
                'min_samples_leaf': hp.choice('min_samples_leaf', params['min_samples_leaf']),
                'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', params['min_weight_fraction_leaf']),
                'ccp_alpha': hp.choice('ccp_alpha', params['ccp_alpha'])}

    elif model_type.upper() == 'NN':
        space4rf_hyperopt = {'batch_size': hp.choice('batch_size', params['batch_size']),
                'hidden_layer_sizes': hp.choice('hidden_layer_sizes', params['hidden_layer_sizes']),
                'learning_rate_init': hp.choice('learning_rate_init', params['learning_rate_init']),
                'max_iter': hp.choice('max_iter', params['max_iter']),
                'validation_fraction': hp.choice('validation_fraction', params['validation_fraction']),
                'alpha': hp.choice('alpha', params['alpha']),
                'shuffle': hp.choice('shuffle', params['shuffle']),
                'tol': hp.choice('tol', params['tol']),
                'early_stopping': hp.choice('early_stopping', params['early_stopping']),
                'beta_1': hp.choice('beta_1', params['beta_1']),
                'beta_2': hp.choice('beta_2', params['beta_2']),
                'epsilon': hp.choice('epsilon', params['epsilon'])}

    elif model_type.upper() == 'ADAB':
        space4rf_hyperopt = {'n_estimators': hp.choice('n_estimators', params['n_estimators']),
            'learning_rate': hp.choice('learning_rate', params['learning_rate'])}  

    elif model_type.upper() == 'GP':
        space4rf_hyperopt = {'n_restarts_optimizer': hp.choice('n_restarts_optimizer', params['n_restarts_optimizer'])}  

    elif model_type.upper() == 'VR':
        space4rf_hyperopt = {'max_depth': hp.choice('max_depth', params['max_depth']),
                'max_features': hp.choice('max_features', params['max_features']),
                'n_estimators': hp.choice('n_estimators', params['n_estimators']),
                'min_samples_split': hp.choice('min_samples_split', params['min_samples_split']),
                'min_samples_leaf': hp.choice('min_samples_leaf', params['min_samples_leaf']),
                'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', params['min_weight_fraction_leaf']),
                'oob_score': hp.choice('oob_score', params['oob_score']),
                'ccp_alpha': hp.choice('ccp_alpha', params['ccp_alpha']),
                'subsample': hp.choice('subsample', params['subsample']),
                'max_samples': hp.choice('max_samples', params['max_samples']),
                'learning_rate': hp.choice('learning_rate', params['learning_rate']),
                'validation_fraction': hp.choice('validation_fraction', params['validation_fraction']),
                'batch_size': hp.choice('batch_size', params['batch_size']),
                'hidden_layer_sizes': hp.choice('hidden_layer_sizes', params['hidden_layer_sizes']),
                'learning_rate_init': hp.choice('learning_rate_init', params['learning_rate_init']),
                'max_iter': hp.choice('max_iter', params['max_iter']),
                'alpha': hp.choice('alpha', params['alpha']),
                'shuffle': hp.choice('shuffle', params['shuffle']),
                'tol': hp.choice('tol', params['tol']),
                'early_stopping': hp.choice('early_stopping', params['early_stopping']),
                'beta_1': hp.choice('beta_1', params['beta_1']),
                'beta_2': hp.choice('beta_2', params['beta_2']),
                'epsilon': hp.choice('epsilon', params['epsilon'])}  

    elif model_type.upper() == 'MVL':
        space4rf_hyperopt = {'max_features': hp.choice('max_features', params['max_features'])}

    return space4rf_hyperopt


# f function of hyperopt. The params variables is the space4rf used in fmin()
def f(params):

    # this json file is used to: 1) keep track of the best value, 2) store the X and y values,
    # 3) store other general options (i.e. type, error_type, etc.) during the hyperopt process
    with open('hyperopt.json') as json_file:
        hyperopt_data = json.load(json_file)
    best = hyperopt_data['best']

    # complete the lsit of missing parameters
    params['type'] = hyperopt_data['type']
    params['model'] = hyperopt_data['model']
    params['train'] = hyperopt_data['train']
    params['seed'] = hyperopt_data['seed']
    params['error_type'] = hyperopt_data['error_type']

    # correct for a problem when loading arrays in json
    if params['model'].upper() in ['NN','VR']:
        layer_arrays = []
        if not isinstance(params['hidden_layer_sizes'],int):
            for _,ele in enumerate(params['hidden_layer_sizes'].split(',')):
                if ele != '':
                    layer_arrays = int(ele)
        else:
            layer_arrays = ele
        params['hidden_layer_sizes'] = (layer_arrays)

    try:
        opt_target,data = load_n_predict(params, hyperopt_data, hyperopt=True)
        # this avoids weird models with R2 very close to 1 and 0, which are selected sometimes
        # because the errors of the validation sets are low
        if data['r2_train'] > 0.99 or data['r2_train'] < 0.01:
            opt_target = float('inf')

        # since the hyperoptimizer aims to minimize the target values, the code needs to use negative
        # values for R2, accuracy, F1 score and MCC (these values are inverted again before storing them)
        if params['error_type'].lower() in ['r2', 'mcc', 'f1', 'acc']:
            opt_target = -opt_target
    except RuntimeError:
        opt_target = float('inf')

    if opt_target < best:
        # The "best" optimizing value is updated in an external JSON file after each hyperopt cycle
        # (using opt_target), and the parameters of the best model found are kept in a CSV file
        os.remove('hyperopt.json')
        hyperopt_data['best'] = opt_target
        with open('hyperopt.json', 'w') as outfile:
            json.dump(hyperopt_data, outfile)

        best = opt_target

        # returns values to normal if inverted during hyperoptimization
        if params['error_type'].lower() in ['r2', 'mcc', 'f1', 'acc']:
            opt_target = -opt_target

        # create csv_hyperopt dataframe
        csv_hyperopt = {'train': hyperopt_data["train"],
                        'split': hyperopt_data['split'],
                        'model': hyperopt_data['model'],
                        'type': hyperopt_data['type'],
                        'seed': hyperopt_data['seed'],
                        'y': hyperopt_data['y'],
                        'error_type': hyperopt_data['error_type'],
                        'X_descriptors': hyperopt_data['X_descriptors']}

        if hyperopt_data['model'].upper() in ['RF','GB','VR']:
            csv_hyperopt['n_estimators'] = params['n_estimators']
            csv_hyperopt['max_depth'] = params['max_depth']
            csv_hyperopt['max_features'] = params['max_features']
            csv_hyperopt['min_samples_split'] = params['min_samples_split']
            csv_hyperopt['min_samples_leaf'] = params['min_samples_leaf']
            csv_hyperopt['min_weight_fraction_leaf'] = params['min_weight_fraction_leaf']
            csv_hyperopt['ccp_alpha'] = params['ccp_alpha']

            if hyperopt_data['model'].upper() in ['RF','VR']:
                csv_hyperopt['oob_score'] = params['oob_score']
                csv_hyperopt['max_samples'] = params['max_samples']

            if hyperopt_data['model'].upper() in ['GB','VR']:
                csv_hyperopt['learning_rate'] = params['learning_rate']
                csv_hyperopt['subsample'] = params['subsample']

            if hyperopt_data['model'].upper() == 'GB':
                csv_hyperopt['validation_fraction'] = params['validation_fraction']

        if hyperopt_data['model'].upper() in ['NN','VR']:
                csv_hyperopt['batch_size'] = params['batch_size']
                csv_hyperopt['hidden_layer_sizes'] = params['hidden_layer_sizes']
                csv_hyperopt['learning_rate_init'] = params['learning_rate_init']
                csv_hyperopt['max_iter'] = params['max_iter']
                csv_hyperopt['validation_fraction'] = params['validation_fraction']
                csv_hyperopt['alpha'] = params['alpha']
                csv_hyperopt['shuffle'] = params['shuffle']
                csv_hyperopt['tol'] = params['tol']
                csv_hyperopt['early_stopping'] = params['early_stopping']
                csv_hyperopt['beta_1'] = params['beta_1']
                csv_hyperopt['beta_2'] = params['beta_2']
                csv_hyperopt['epsilon'] = params['epsilon']
    
        elif hyperopt_data['model'].upper() == 'ADAB':
                csv_hyperopt['n_estimators'] = params['n_estimators']
                csv_hyperopt['learning_rate'] = params['learning_rate']

        elif hyperopt_data['model'].upper() == 'GP':
                csv_hyperopt['n_restarts_optimizer'] = params['n_restarts_optimizer']

        csv_hyperopt[hyperopt_data['error_type']] = opt_target

        # save into a csv file
        csv_hyperopt_df = pd.DataFrame.from_dict(csv_hyperopt, orient='index')
        csv_hyperopt_df = csv_hyperopt_df.transpose()
        
        destination = Path(hyperopt_data['destination'])
        name_csv_hyperopt = destination.joinpath(f"Raw_data/No_PFI/{hyperopt_data['model']}_{hyperopt_data['train']}_{hyperopt_data['seed']}.csv")
        _ = csv_hyperopt_df.to_csv(name_csv_hyperopt, index = None, header=True)
 
    return {'loss': best, 'status': STATUS_OK}


def load_params(self,model_type):
    """
    Loads the parameters for the calculation from a yaml if specified. Otherwise
    does nothing.
    """

    # the model params are downloaded with ROBERT
    if self.args.custom_params is None:
        models_path = Path(resource_filename("robert", "model_params"))
    else:
        models_path = self.initial_dir.joinpath(self.args.custom_params)
    varfile = models_path.joinpath(f'{model_type.upper()}_params.yaml')

    # Variables will be updated from YAML file
    with open(varfile, "r") as file:
        try:
            params = yaml.load(file, Loader=yaml.SafeLoader)
        except (yaml.scanner.ScannerError,yaml.parser.ParserError):
            self.args.log.write(f'\nx  Error while reading {varfile}. Edit the yaml file and try again (i.e. use ":" instead of "=" to specify variables)')
            sys.exit()

    for param in params:
        if params[param] == 'True':
            params[param] = True
        elif params[param] == 'False':
            params[param] = False

    return params


def prepare_sets(self,csv_X,csv_y,size,seed):
    # split into training and validation sets
    try:
        Xy_data = data_split(self,csv_X,csv_y,size,seed)
    except TypeError:
        self.args.log.write(f'   x The data split process failed! This is probably due to using strings/words as values (use --curate to curate the data first)')
        sys.exit()

    # standardization of X values using the mean and SD of the training set
    X_train_scaled, X_valid_scaled = standardize(self,Xy_data['X_train'],Xy_data['X_valid'])
    Xy_data['X_train_scaled'] = X_train_scaled
    Xy_data['X_valid_scaled'] = X_valid_scaled

    # also store the descriptors used (the labels disappear after data_split() )
    Xy_data['X_descriptors'] = csv_X.columns.tolist()

    # discard descriptors with NaN values after standardization. This might happen when using descriptor
    # columns that after the data split contain only one unique value in one of the sets (i.e., 0,0,0,0,0)
    columns_nan = []
    for column in Xy_data['X_train_scaled'].columns:
        if Xy_data['X_train_scaled'][column].isnull().values.any():
            columns_nan.append(column)
        elif Xy_data['X_valid_scaled'][column].isnull().values.any():
            columns_nan.append(column)

    if len(columns_nan) > 0:
        nan_print = '   x  Variables removed for having NaN values after standardization:'
        for column_nan in columns_nan:
            nan_print += f'\n      - {column_nan}'
            Xy_data['X_descriptors'].remove(column_nan)
            Xy_data['X_train_scaled'] = Xy_data['X_train_scaled'].drop(column_nan, axis=1)
            Xy_data['X_valid_scaled'] = Xy_data['X_valid_scaled'].drop(column_nan, axis=1)
            Xy_data['X_train'] = Xy_data['X_train'].drop(column_nan, axis=1)
            Xy_data['X_valid'] = Xy_data['X_valid'].drop(column_nan, axis=1)
            csv_X = csv_X.drop(column_nan, axis=1)

        self.args.log.write(nan_print)

    return Xy_data


def data_split(self,csv_X,csv_y,size,seed):

    if size == 100:
        # if there is no validation set, use all the points
        training_points = np.arange(0,len(csv_X),1)
    else:
        if self.args.split.upper() == 'KN':
            # k-neighbours data split
            # standardize the data before k-neighbours-based data splitting
            Xmeans = csv_X.mean(axis=0)
            Xstds = csv_X.std(axis=0)
            X_scaled = (csv_X - Xmeans) / Xstds

            training_points = k_neigh(self,X_scaled,csv_y,size,seed)

        elif self.args.split.upper() == 'RND':
            n_of_points = int(len(csv_X)*(size/100))

            random.seed(seed)
            training_points = random.sample(range(len(csv_X)), n_of_points)

    Xy_data = Xy_split(csv_X,csv_y,training_points)

    return Xy_data


# returns a dictionary with the database divided into train and validation
def Xy_split(csv_X,csv_y,training_points):
    Xy_data =  {}
    Xy_data['X_train'] = csv_X.iloc[training_points]
    Xy_data['y_train'] = csv_y.iloc[training_points]
    Xy_data['X_valid'] = csv_X.drop(training_points)
    Xy_data['y_valid'] = csv_y.drop(training_points)
    Xy_data['training_points'] = training_points
    return Xy_data


def k_neigh(self,X_scaled,csv_y,size,seed):
    
    # number of clusters in the training set from the k-neighbours clustering (based on the
    # training set size specified above)
    X_scaled_array = np.asarray(X_scaled)
    number_of_clusters = int(len(X_scaled)*(size/100))

    # to avoid points from the validation set outside the training set, the 2 first training
    # points are automatically set as the 2 points with minimum/maximum response value
    training_points = [csv_y.idxmin(),csv_y.idxmax()]
    number_of_clusters -= 2
    
    # runs the k-neighbours algorithm and keeps the closest point to the center of each cluster
    kmeans = KMeans(n_clusters=number_of_clusters,random_state=seed)
    try:
        kmeans.fit(X_scaled_array)
    except ValueError:
        self.args.log.write("\nx  The K-means clustering process failed! This might be due to having NaN or strings as descriptors (curate the data first with CURATE) or having too few datapoints!")
        sys.exit()
    _ = kmeans.predict(X_scaled_array)
    centers = kmeans.cluster_centers_
    for i in range(number_of_clusters):
        results_cluster = 1000000
        for k in range(len(X_scaled_array[:, 0])):
            if k not in training_points:
                # calculate the Euclidean distance in n-dimensions
                points_sum = 0
                for l in range(len(X_scaled_array[0])):
                    points_sum += (X_scaled_array[:, l][k]-centers[:, l][i])**2
                if np.sqrt(points_sum) < results_cluster:
                    results_cluster = np.sqrt(points_sum)
                    training_point = k
        
        training_points.append(training_point)
    training_points.sort()

    return training_points


def PFI_workflow(self, csv_df, ML_model, size, Xy_data, seed):
    # filter off parameters with low PFI (not relevant in the model)
    name_csv_hyperopt = f"Raw_data/No_PFI/{ML_model}_{size}_{seed}"
    path_csv = self.args.destination.joinpath(f'{name_csv_hyperopt}.csv')
    PFI_df = pd.read_csv(path_csv)
    PFI_dict = pd_to_dict(PFI_df) # (using a dict to keep the same format of load_model)
    PFI_discard_cols = PFI_filter(self,Xy_data,PFI_dict,seed)

    # generate new X datasets and store the descriptors used for the PFI-filtered model
    desc_keep = len(Xy_data['X_train'].columns)
    
    # if the filter does not remove any descriptors based on the PFI threshold, or the 
    # proportion of descriptors:total datapoints is higher than 1:3, then the filter takes
    # the minimum value of 1 and 2:
    #   1. 25% less descriptors than the No PFI original model
    #   2. Proportion of 1:3 of descriptors:total datapoints (training + validation)
    total_points = len(Xy_data['y_train'])+len(Xy_data['y_valid'])
    n_descp_PFI = desc_keep-len(PFI_discard_cols)
    if n_descp_PFI > 0.33*total_points or n_descp_PFI == desc_keep or n_descp_PFI == 0:
        option_one = int(0.75*len(Xy_data['X_train'].columns))
        option_two = int(0.33*total_points)
        pfi_max = min(option_one,option_two)
    else:
        pfi_max = self.args.pfi_max
    
    discard_idx, descriptors_PFI = [],[]
    # just in case none of the descriptors passed the PFI filter
    # select only the most important deascriptors until the pfi_max limit
    if n_descp_PFI == 0:
        descriptors_PFI = PFI_discard_cols[:pfi_max]
        discard_idx = PFI_discard_cols[len(PFI_discard_cols)-pfi_max:]

    else:
        if pfi_max != 0:
            desc_keep = pfi_max
        for _,column in enumerate(Xy_data['X_train'].columns):
            if column not in PFI_discard_cols and len(descriptors_PFI) < desc_keep:
                descriptors_PFI.append(column)
            else:
                discard_idx.append(column)

    Xy_data_PFI = Xy_data.copy()
    Xy_data_PFI['X_train'] = Xy_data['X_train'].drop(discard_idx, axis=1)
    Xy_data_PFI['X_valid'] = Xy_data['X_valid'].drop(discard_idx, axis=1)
    Xy_data_PFI['X_train_scaled'], Xy_data_PFI['X_valid_scaled'] = standardize(self,Xy_data_PFI['X_train'],Xy_data_PFI['X_valid'])
    PFI_dict['X_descriptors'] = descriptors_PFI
    if 'max_features' in  PFI_dict and PFI_dict['max_features'] > len(descriptors_PFI):
        PFI_dict['max_features'] = len(descriptors_PFI)

    # updates the model's error and descriptors used from the corresponding No_PFI CSV file 
    # (the other parameters remain the same)
    opt_target,_ = load_n_predict(PFI_dict, Xy_data_PFI, hyperopt=True)
    PFI_dict[PFI_dict['error_type']] = opt_target

    # save CSV file
    name_csv_hyperopt_PFI = name_csv_hyperopt.replace('No_PFI','PFI')
    path_csv_PFI = self.args.destination.joinpath(f'{name_csv_hyperopt_PFI}_PFI')
    csv_PFI_df = pd.DataFrame.from_dict(PFI_dict, orient='index')
    csv_PFI_df = csv_PFI_df.transpose()
    _ = csv_PFI_df.to_csv(f'{path_csv_PFI}.csv', index = None, header=True)

    # copy the database used
    # set a new column for the set
    set_column = []
    n_points = len(csv_df[csv_df.columns[0]])
    for i in range(0,n_points):
        if i in Xy_data['training_points']:
            set_column.append('Training')
        else:
            set_column.append('Validation')
    csv_df['Set'] = set_column

    # save the csv file
    if os.path.exists(self.args.destination.joinpath(f"Raw_data/PFI/{ML_model}_{size}_{seed}_PFI.csv")):
        db_name = self.args.destination.joinpath(f"Raw_data/PFI/{ML_model}_{size}_{seed}_PFI_db")
        _ = csv_df.to_csv(f'{db_name}.csv', index = None, header=True)


def PFI_filter(self,Xy_data,PFI_dict,seed):

    # load and fit model
    loaded_model = load_model(PFI_dict)
    loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])

    # we use the validation set during PFI as suggested by the sklearn team:
    # "Using a held-out set makes it possible to highlight which features contribute the most to the 
    # generalization power of the inspected model. Features that are important on the training set 
    # but not on the held-out set might cause the model to overfit."
    score_model = loaded_model.score(Xy_data['X_valid_scaled'], Xy_data['y_valid'])
    perm_importance = permutation_importance(loaded_model, Xy_data['X_valid_scaled'], Xy_data['y_valid'], n_repeats=self.args.pfi_epochs, random_state=seed)
    # transforms the values into a list and sort the PFI values with the descriptor names
    descp_cols, PFI_values, PFI_sd = [],[],[]
    for i,desc in enumerate(Xy_data['X_train'].columns):
        descp_cols.append(desc) # includes lists of descriptors not column names!
        PFI_values.append(perm_importance.importances_mean[i])
        PFI_sd.append(perm_importance.importances_std[i])
  
    PFI_values, PFI_sd, descp_cols = (list(t) for t in zip(*sorted(zip(PFI_values, PFI_sd, descp_cols), reverse=True)))

    # PFI filter
    PFI_discard_cols = []
    PFI_thres = abs(self.args.pfi_threshold*score_model)
    for i in range(len(PFI_values)):
        if PFI_values[i] < PFI_thres:
            PFI_discard_cols.append(descp_cols[i])

    return PFI_discard_cols


def filter_seed(self, name_csv):
    '''
    Check which seed led to the best results
    '''

    # track errors fo all seeds
    errors = []
    for seed in self.args.seed:
        if 'No_PFI' in f'{name_csv}':
            file_seed = f'{name_csv}_{seed}.csv'
        else:
            file_seed = f'{name_csv}_{seed}_PFI.csv'
        if os.path.exists(file_seed):
            results_model = pd.read_csv(f'{file_seed}')
            errors.append(results_model[results_model['error_type'][0]][0])
        else:
            errors.append(np.nan)

    # keep best result, take out seed from name, and delete the other CSV files
    if results_model['type'][0].lower() == 'reg' and results_model['error_type'][0].lower() in ['mae','rmse']:
        min_idx = errors.index(np.nanmin(errors))
    else:
        min_idx = errors.index(np.nanmax(errors))

    for i,seed in enumerate(self.args.seed):
        if 'No_PFI' in f'{name_csv}':
            file_seed = f'{name_csv}_{seed}.csv'
            file_seed_db = f'{name_csv}_{seed}_db.csv'
            new_file = f'{name_csv}'
        else:
            file_seed = f'{name_csv}_{seed}_PFI.csv'
            file_seed_db = f'{name_csv}_{seed}_PFI_db.csv'   
            new_file = f'{name_csv}_PFI'     
        if os.path.exists(file_seed):
            if i == min_idx:
                os.rename(file_seed,f'{new_file}.csv')
            else:
                os.remove(file_seed)
        if os.path.exists(file_seed_db):
            if i == min_idx:
                os.rename(file_seed_db,f'{new_file}_db.csv')
            else:
                os.remove(file_seed_db)


def detect_best(folder):
    '''
    Check which combination led to the best results
    '''

    # detect files
    file_list = glob.glob(f'{folder}/*.csv')
    errors = []
    for file in file_list:
        if '_db' not in file:
            results_model = pd.read_csv(f'{file}')
            errors.append(results_model[results_model['error_type'][0]][0])
        else:
            errors.append(np.nan)

    # detect best result and copy files to the Best_model folder
    if results_model['type'][0].lower() == 'reg' and results_model['error_type'][0].lower() in ['mae','rmse']:
        min_idx = errors.index(np.nanmin(errors))
    else:
        min_idx = errors.index(np.nanmax(errors))
    best_name = file_list[min_idx]
    best_db = f'{os.path.dirname(file_list[min_idx])}/{os.path.basename(file_list[min_idx]).split(".csv")[0]}_db.csv'

    shutil.copyfile(f'{best_name}', f'{best_name}'.replace('Raw_data','Best_model'))
    shutil.copyfile(f'{best_db}', f'{best_db}'.replace('Raw_data','Best_model'))


def heatmap_workflow(self,folder_hm):
    # create matrix of ML models, training sizes and errors/precision (no PFI filter)
    path_raw = self.args.destination.joinpath(f"Raw_data")
    csv_data,model_list,size_list = {},[],[]
    for csv_file in glob.glob(path_raw.joinpath(f"{folder_hm}/*.csv").as_posix()):
        if '_db' not in csv_file:
            basename = os.path.basename(csv_file)
            csv_model = basename.replace('.','_').split('_')[0]
            if csv_model not in model_list:
                model_list.append(csv_model)
                csv_data[csv_model] = {}
            csv_size = basename.replace('.','_').split('_')[1]
            if csv_model not in size_list:
                size_list.append(csv_size)
            csv_value = pd.read_csv(csv_file)
            csv_data[csv_model][csv_size] = csv_value[self.args.error_type][0]
    # pass dictionary into a dataframe
    csv_df = pd.DataFrame()
    for csv_model in csv_data:
        csv_df[csv_model] = csv_data[csv_model]
    
    # plot heatmap
    if folder_hm == "No_PFI":
        suffix = 'no PFI filter'
    elif folder_hm == "PFI":
        suffix = 'with PFI filter'
    _ = create_heatmap(self,csv_df,suffix,path_raw)

def create_heatmap(self,csv_df,suffix,path_raw):
    csv_df = csv_df.sort_index(ascending=False)
    sb.set(font_scale=1.2, style='ticks')
    _, ax = plt.subplots(figsize=(7.45,6))
    cmap_blues_75_percent_512 = [mcolor.rgb2hex(c) for c in plt.cm.Blues(np.linspace(0, 0.8, 512))]
    ax = sb.heatmap(csv_df, annot=True, linewidth=1, cmap=cmap_blues_75_percent_512, cbar_kws={'label': f'{self.args.error_type.upper()} validation set'})
    fontsize = 14
    ax.set_xlabel("ML Model",fontsize=fontsize)
    ax.set_ylabel("Training Size",fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    title_fig = f'Heatmap ML models {suffix}'
    plt.title(title_fig, y=1.04, fontsize = fontsize, fontweight="bold")
    sb.despine(top=False, right=False)
    plt.savefig(f'{path_raw.joinpath(title_fig)}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    path_reduced = '/'.join(f'{path_raw}'.replace('\\','/').split('/')[-2:])
    self.args.log.write(f'\no  {title_fig} succesfully created in {path_reduced}')
