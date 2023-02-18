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
import glob
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from robert.utils import load_model, model_stats


# initial function for hyperopt
def run_hyperopt(self, ML_model, size, Xy_data_hp):

    # edit this function to modify the hyperopt parameter optimization (i.e. the 
    # lists represent values to include in the grid search)
    space4rf = hyperopt_params(ML_model, Xy_data_hp['X_train_scaled'])

    if ML_model == 'MVL':
        n_epochs = 1

    # Run hyperopt
    trials = Trials()

    # This best high initial number is a dummy value used as the start of the optimization
    # (otherwise it does not run more than once since best becomes a dictionary)
    best = 100000

    hyperopt_data = {'best': best, 'model': ML_model,
                'mode': self.args.mode,
                'split': self.args.split,
                'size': size, 
                'seed': self.args.seed,
                'hyperopt_target': self.args.hyperopt_target}

    # adjust the format for the sklearn models and add the data to the dict
    for desc in ['X_train_scaled','y_train','X_valid_scaled','y_valid']:
        Xy_data_hp[desc] = np.asarray(Xy_data_hp[desc]).tolist()
        hyperopt_data[desc] = Xy_data_hp[desc]

    # save the initial json
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

    return space4rf_hyperopt


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
                        'split': hyperopt_data['split'],
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
        
        name_csv_hyperopt = Path(f"/Raw_data/No_PFI/{hyperopt_data['model']}_{hyperopt_data['train']}.csv")
        _ = csv_hyperopt_df.to_csv(name_csv_hyperopt, index = None, header=True)
 
    return {'loss': best, 'status': STATUS_OK}


# calculates RMSE of the validation set with the parameters of the corresponding
# hyperopt optimization cycle
def hyperopt_epoch(params, hyperopt_data):

    # set the parameters for each ML model of the hyperopt optimization
    loaded_model = load_model(params, hyperopt_data)

    # Fit the model with the training set
    loaded_model.fit(hyperopt_data['X_train_scaled'], hyperopt_data['y_train'])  

    if hyperopt_data['train'] == 100:
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
    Xy_data['training_points'] = training_points
            
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


def PFI_workflow(self,PFI_df,Xy_data):

    # load and fit model
    loaded_model = load_model(PFI_df, PFI_df)
    loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])

    # we use the validation set during PFI as suggested by the sklearn team:
    # "Using a held-out set makes it possible to highlight which features contribute the most to the 
    # generalization power of the inspected model. Features that are important on the training set 
    # but not on the held-out set might cause the model to overfit."
    score_model = loaded_model.score(Xy_data['X_valid_scaled'], Xy_data['y_valid'])
    perm_importance = permutation_importance(loaded_model, Xy_data['X_valid_scaled'], Xy_data['y_valid'], n_repeats=self.args.PFI_epochs, random_state=self.args.seed)

    # transforms the values into a list and sort the PFI values with the descriptors names
    combined_descriptor_list = []
    for column in Xy_data['X_train']:
        combined_descriptor_list.append(column)
    PFI_values, PFI_SD = [],[]
    for value in perm_importance.importances_mean:
        PFI_values.append(value)
    for sd in perm_importance.importances_std:
        PFI_SD.append(sd)
    PFI_values, PFI_SD, combined_descriptor_list = (list(t) for t in zip(*sorted(zip(PFI_values, PFI_SD, combined_descriptor_list), reverse=True)))

    # PFI filter
    PFI_discard = []
    PFI_filter = self.args.PFI_threshold*score_model
    for i in reversed(range(len(PFI_values))):
        if PFI_values[i] < PFI_filter:
            PFI_discard.append(combined_descriptor_list[i])
    Xy_data['X_train_PFI'] = Xy_data['X_train'].drop(PFI_discard, axis=1)
    Xy_data['X_valid_PFI'] = Xy_data['X_valid'].drop(PFI_discard, axis=1)

    guarda en best/PFI
    en el paso de antes, guarda graph en best_results/No_PFI

        # printing and representing the results
        print(f"\nPermutation feature importances of the descriptors in the {PFI_df['model']}_{PFI_df['train']}_PFI model (for the validation set). Only showing values that drop the original score at least by {self.args.PFI_threshold*100}%:\n")
        print('Original score = '+f'{score_model:.2f}')
        for i in range(len(PFI_values)):
            print(combined_descriptor_list[i]+': '+f'{PFI_values[i]:.2f}'+' '+u'\u00B1'+ ' ' + f'{PFI_SD[i]:.2f}')

        y_ticks = np.arange(0, len(PFI_values))
        fig, ax = plt.subplots()
        ax.barh(y_ticks, PFI_values[::-1])
        ax.set_yticklabels(combined_descriptor_list[::-1])
        ax.set_yticks(y_ticks)
        ax.set_title(model_type_PFI_fun+" permutation feature importances (PFI)")
        fig.tight_layout()
        plot = ax.set(ylabel=None, xlabel='PFI')

        plt.savefig(f'PFI/{model_type_PFI_fun}+ permutation feature importances (PFI)', dpi=600, bbox_inches='tight')

        plt.show()
    
    return combined_descriptor_list

def update_best(self,csv_df,Xy_data,name_csv):

    # check if the results of the new model are better than the previous best model
    results_model = pd.read_csv(name_csv)
    new_error = results_model[results_model['hyperopt_target']]

    if 'No_PFI' in name_csv:
        folder_suf = 'No_PFI'
    else:
        folder_suf = 'PFI'

    # detects previos best file with results
    folder_best = f"/Best_model/{folder_suf}"
    csv_files = glob.glob(f'{folder_best}/*.csv')
    for csv_file in csv_files:
        file_split = os.path.basename(csv_file).replace('.','_').split('_')
        if len(file_split) == 3 and file_split[0] in ['rf','mvl','gb','adab','nn','vr']:
            name_best = csv_file
    results_best = pd.read_csv(name_best)
    best_error = results_best[results_best['hyperopt_target']]

    # error for regressors
    replace_best = False
    if results_model['mode'].lower() == 'reg' and results_best['hyperopt_target'].lower() in ['rmse','mae']:
        if new_error < best_error:
            replace_best = True
    # precision for classificators and R2
    else:
        if new_error > best_error:
            replace_best = True

    if replace_best:
        for file in glob.glob(f'{folder_best}/*.*'):
            os.remove(file)
        shutil.copyfile(name_csv, f'{folder_best}/os.path.basename({name_csv})')


        # set two new columns, one for the predicted y values and the other for the set
        csv_df[f'Predicted_{self.args.y}'] = results_model['']


        set_column = []
        for _,num in enumerate(Xy_data['y_train']):
            if num in Xy_data['training_points']:
                set_column.append('Training')
            else:
                set_column.append('Validation')
        csv_df['Set'] = set_column
    

usa xy_data to create first training and then validation, then concatenate as below. This way,
you can add also predicted y_values for valid and train
csv_df anade una columna de training valid en base a Xy_data
anade la colunma de predicted_y (anade primero en hyperopt y luego carga)

    X_train_csv = best_model[21].copy()
    X_validation_csv = best_model[22].copy()

    X_train_csv[response_value] = best_model[18]
    X_validation_csv[response_value] = best_model[19]

    X_train_csv[f'Predicted {response_value}'] = best_model[11]
    X_validation_csv[f'Predicted {response_value}'] =  best_model[12]
    
    X_train_csv = pd.concat([X_train_csv, best_model[9]], axis=1)
    X_validation_csv = pd.concat([X_validation_csv, best_model[14]], axis=1)

    X_train_csv['Set'] = 'Training'
    X_validation_csv['Set'] = 'Validation'

    df_csv = pd.concat([X_train_csv, X_validation_csv], axis=0)