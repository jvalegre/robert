#####################################################.
#     This file stores functions from GENERATE      #
#####################################################.

import os
import sys
import time
import shutil
from pathlib import Path
from scipy import stats
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import yaml
import json
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from robert.utils import load_model, model_stats


# initial function for hyperopt
def run_hyperopt(self, ML_model, size, Xy_data):

    # edit this function to modify the hyperopt parameter optimization (i.e. the 
    # lists represent values to include in the grid search)
    space4rf,Predictor_params = hyperopt_params(ML_model, Xy_data['X_train_scaled'])

    if ML_model == 'MVL':
        n_epochs = 1

    # Run hyperopt
    trials = Trials()

    # This best high initial number is a dummy value used as the start of the optimization
    # (otherwise it does not run more than once since best becomes a dictionary)
    best = 100000

    hyperopt_data = {'best': best, 'model': ML_model,
                'mode': self.args.mode,
                'Predictor_params': Predictor_params,
                'Xy_data': Xy_data,
                'size': size, 
                'seed': self.args.seed,
                'hyperopt_data': self.args.hyperopt_target}

    with open('hyperopt.json', 'w') as outfile:
        json.dump(hyperopt_data, outfile)

    try:
        best = fmin(f, space4rf, algo=tpe.suggest, max_evals=n_epochs, trials=trials, rstate=np.random.default_rng(hyperopt_data['seed']))

    except ValueError:
        print('There is an error in the hyperopt module, are you using mode = \'clas\' for regression y values instead of mode = \'reg\'?')

    try:
        os.remove('hyperopt.json')
    except FileNotFoundError:
        pass


# generates initial parameters for the hyperopt optimization
def hyperopt_params(self, model_type, X_hyperopt):
    # load the parameters of the models from their corresponding yaml files
    params = load_from_yaml(self,model_type,X_hyperopt)

    if model_type == 'RF':
        space4rf_hyperopt = {'max_depth': hp.choice('max_depth', params['max_depth']),
                'max_features': hp.choice('max_features', params['max_features']),
                'n_estimators': hp.choice('n_estimators', params['n_estimators'])}  

    elif model_type == 'GB':
        space4rf_hyperopt = {'max_depth': hp.choice('max_depth', params['max_depth']),
                'max_features': hp.choice('max_features', params['max_features']),
                'n_estimators': hp.choice('n_estimators', params['n_estimators']),
                'learning_rate': hp.choice('learning_rate', params['learning_rate']),
                'validation_fraction': hp.choice('validation_fraction', ['validation_fraction'])}  

    elif model_type == 'AdaB':
        space4rf_hyperopt = {'n_estimators': hp.choice('n_estimators', params['n_estimators']),
            'learning_rate': hp.choice('learning_rate', params['learning_rate'])}  

    elif model_type == 'NN':
        space4rf_hyperopt = {'batch_size': hp.choice('batch_size', params['batch_size']),
                'hidden_layer_sizes': hp.choice('hidden_layer_sizes', params['hidden_layer_sizes']),
                'learning_rate_init': hp.choice('learning_rate_init', params['learning_rate_init']),
                'max_iter': hp.choice('max_iter', params['max_iter']),
                'validation_fraction': hp.choice('validation_fraction', params['validation_fraction'])}

    elif model_type == 'VR':
        space4rf_hyperopt = {'max_depth': hp.choice('max_depth', params['max_depth']),
                'max_features': hp.choice('max_features', params['max_features']),
                'n_estimators': hp.choice('n_estimators', params['n_estimators']),
                'learning_rate': hp.choice('learning_rate', params['learning_rate']),
                'validation_fraction': hp.choice('validation_fraction', params['validation_fraction']),
                'batch_size': hp.choice('batch_size', params['batch_size']),
                'hidden_layer_sizes': hp.choice('hidden_layer_sizes', params['hidden_layer_sizes']),
                'learning_rate_init': hp.choice('learning_rate_init', params['learning_rate_init']),
                'max_iter': hp.choice('max_iter', params['max_iter'])}  

    elif model_type == 'MVL':
        space4rf_hyperopt = {'max_features': hp.choice('max_features', params['max_features'])}

    return space4rf_hyperopt,params


# f function of hyperopt. The params variables is the space4rf used in fmin()
def f(params):

    with open('hyperopt.json') as json_file:
        hyperopt_data = json.load(json_file)

    opt_target = hyperopt_epoch(params, hyperopt_data)
    
    # The "best" optimizing value is updated in an external JSON file after each hyperopt cycle
    # (using opt_target), and the parameters of the best model found are kept in a CSV file
    hyperopt_data['best'] = opt_target
    os.remove('hyperopt.json')
    with open('hyperopt.json', 'w') as outfile:
        json.dump(hyperopt_data, outfile)

    if opt_target < best:
        best = opt_target

        # create csv_hyperopt dataframe
        csv_hyperopt = {'size': hyperopt_data["size"],
                        'model': hyperopt_data['model'],
                        'mode': hyperopt_data['mode'],
                        'seed': hyperopt_data['seed']}

        if hyperopt_data['model'].upper() in ['RF','GB','VR']:
            csv_hyperopt['n_estimators'] = params['n_estimators']
            csv_hyperopt['max_depth'] = params['max_depth']
            csv_hyperopt['max_features'] = params['max_features']

            if hyperopt_data['model'].upper() in ['GB','VR']:
                csv_hyperopt['learning_rate'] = params['learning_rate']

            if hyperopt_data['model'].upper() == 'GB':
                csv_hyperopt['validation_fraction'] = params['validation_fraction']

        elif hyperopt_data['model'].upper() in ['NN','VR']:
                csv_hyperopt['batch_size'] = params['batch_size']
                csv_hyperopt['hidden_layer_sizes'] = params['hidden_layer_sizes']
                csv_hyperopt['learning_rate_init'] = params['learning_rate_init']
                csv_hyperopt['max_iter'] = params['max_iter']
                csv_hyperopt['validation_fraction'] = params['validation_fraction']
    
        elif hyperopt_data['model'].upper() == 'ADAB':
                csv_hyperopt['n_estimators'] = params['n_estimators']
                csv_hyperopt['learning_rate'] = params['learning_rate']
            
        if hyperopt_data['mode'] == 'reg':
            csv_hyperopt[hyperopt_data['hyperopt_target']] = best
            
        elif hyperopt_data['mode'] == 'clas':
            # need to reconvert the value (it was converted into a negative value in hyperopt_epoch())
            csv_hyperopt[hyperopt_data['hyperopt_target']] = -best

        # save into a csv file
        csv_hyperopt_df = pd.DataFrame.from_dict(csv_hyperopt, orient='index')
        csv_hyperopt_df = csv_hyperopt_df.transpose()
        
        include (raw data) folder
        name_csv_hyperopt = f"{hyperopt_data['model']}_{hyperopt_data['size']}"

        _ = csv_hyperopt_df.to_csv(name_csv_hyperopt+'.csv', index = None, header=True)
 
    return {'loss': best, 'status': STATUS_OK}


# calculates RMSE of the validation set with the parameters of the corresponding
# hyperopt optimization cycle
def hyperopt_epoch(params, hyperopt_data):

    # set the parameters for each ML model of the hyperopt optimization
    loaded_model = load_model(params, hyperopt_data)

    # Fit the model with the training set
    loaded_model.fit(hyperopt_data['X_train_scaled'], hyperopt_data['y_train'])  

    if hyperopt_data['size'] == 100:
        # if there is not test set, only used values from training
        y_pred_valid = loaded_model.predict(hyperopt_data['X_train_scaled'])
        y_valid = hyperopt_data['y_train']
    
    else:
        # Predicted values using the model for validation
        y_pred_valid = loaded_model.predict(hyperopt_data['X_valid_scaled'])
        y_valid = hyperopt_data['y_valid']

    # Validation stats
    if hyperopt_data['mode'] == 'reg':
        r2_valid, mae_valid, rmse_valid = model_stats(y_valid,y_pred_valid)
        if hyperopt_data['hyperopt_target'] == 'rmse':
            opt_target = rmse_valid
        elif hyperopt_data['hyperopt_target'] == 'mae_valid':
            opt_target = mae_valid
        elif hyperopt_data['hyperopt_target'] == 'r2':
            opt_target = r2_valid

    elif hyperopt_data['mode'] == 'clas':
        # I make these scores negative so the optimizer is consistent to finding 
        # a minima as in the case of error in regression
        acc_valid = -accuracy_score(y_valid,y_pred_valid)
        f1_score_valid = -f1_score(y_valid,y_pred_valid)
        mcc_valid = -matthews_corrcoef(y_valid,y_pred_valid)
        if hyperopt_data['hyperopt_target'] == 'mcc':
            opt_target = mcc_valid
        elif hyperopt_data['hyperopt_target'] == 'f1_score':
            opt_target = f1_score_valid
        elif hyperopt_data['hyperopt_target'] == 'acc':
            opt_target = acc_valid

    return opt_target


def load_from_yaml(self,model_type,X_hyperopt):
    """
    Loads the parameters for the calculation from a yaml if specified. Otherwise
    does nothing.
    """
    varfile = f'{model_type.upper()}_params.yaml'
    # Variables will be updated from YAML file
    with open(varfile, "r") as file:
        try:
            params = yaml.load(file, Loader=yaml.SafeLoader)
        except yaml.scanner.ScannerError:
            self.args.log.write(f'\nx  Error while reading {varfile}. Edit the yaml file and try again (i.e. use ":" instead of "=" to specify variables)')
            sys.exit()

    # number of descriptors to scan, from 1 to the max amount of descriptors using the interval
    # specified with interval_descriptors
    if model_type in ['RF','GB','VR','MVL']:
        max_features = [1]
        n_descriptors = len(X_hyperopt.columns)
        interval_features = int((n_descriptors+1)/params["interval_features"])
        for num in range(max_features[0],n_descriptors,interval_features):
            max_features.append(num)
        max_features.append(n_descriptors)
        params['max_features'] = max_features

    return params

def data_split(self,csv_X,csv_y,size):
        
    if size == 100:
        # if there is no validation set, use all the points
        training_points = np.arange(0,len(csv_X),1)
    else:
        if self.args.split == 'KN':
            # k-neighbours data split
            # standardize the data before k-neighbours-based data splitting
            Xmeans = csv_X.mean(axis=0)
            Xstds = csv_X.std(axis=0)
            X_scaled = (csv_X - Xmeans) / Xstds

            training_points = k_neigh(self,X_scaled,csv_y,size)

        elif self.args.split == 'RND':
            n_of_points = int(len(csv_X)*(size/100))

            random.seed(self.args.seed)
            training_points = random.sample(range(len(csv_X)), n_of_points)

    Xy_data =  {}
    Xy_data['X_train'] = csv_X.iloc[training_points]
    Xy_data['y_train'] = csv_y.iloc[training_points]
    Xy_data['X_valid'] = csv_X.drop(training_points)
    Xy_data['y_valid'] = csv_y.drop(training_points)
            
    return Xy_data


def k_neigh(self,X_scaled,csv_y,size):
    
    # number of clusters in the training set from the k-neighbours clustering (based on the
    # training set size specified above)
    X_scaled_array = np.asarray(X_scaled)
    number_of_clusters = int(len(X_scaled)*(size/100))

    # to avoid points from the validation set outside the training set, the 2 first training
    # points are automatically set as the 2 points with minimum/maximum response value
    training_points = [csv_y.idxmin(),csv_y.idxmax()]
    number_of_clusters -= 2
    
    # runs the k-neighbours algorithm and keeps the closest point to the center of each cluster
    kmeans = KMeans(n_clusters=number_of_clusters,random_state=self.args.seed)
    kmeans.fit(X_scaled_array)
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