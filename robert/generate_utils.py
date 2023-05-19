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
from sklearn.cluster import KMeans
import yaml
import json
import glob
import warnings # this avoids warnings from sklearn
warnings.filterwarnings("ignore")
from pkg_resources import resource_filename
from sklearn.inspection import permutation_importance
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from robert.utils import (
    load_model,
    load_n_predict,
    standardize,
    pd_to_dict)


# hyperopt workflow
def hyperopt_workflow(self, csv_df, ML_model, size, Xy_data_hp):

    # edit this function to modify the hyperopt parameter optimization (i.e. the 
    # lists represent values to include in the grid search)
    space4rf = hyperopt_params(self, ML_model, Xy_data_hp['X_train_scaled'])

    # Run hyperopt
    trials = Trials()

    # This best high initial number is a dummy value used as the start of the optimization
    # (otherwise it does not run more than once since best becomes a dictionary)
    best = 100000

    hyperopt_data = {'best': best, 'model': ML_model,
                'type': self.args.type,
                'split': self.args.split,
                'train': size, 
                'seed': self.args.seed,
                'error_type': self.args.error_type,
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
        best = fmin(f, space4rf, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(hyperopt_data['seed']))
        if os.path.exists('hyperopt.json'):
            os.remove('hyperopt.json')
    except ValueError:
        self.args.log.write('\nx  There is an error in the hyperopt module, 1) are you using type ="clas" for regression y values instead of type="reg"? or 2) are you using very small partition sizes for validation sets (fix with train="[60,70]" for example)?')
        self.args.log.finalize()
        sys.exit()
    try:
        os.remove('hyperopt.json')
    except FileNotFoundError:
        pass

    # check if this combination is the best model and replace data in the Best_model folder
    name_csv = self.args.destination.joinpath(f"Raw_data/No_PFI/{ML_model}_{size}")
    _ = update_best(self, csv_df, Xy_data_hp,name_csv)


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
                'validation_fraction': hp.choice('validation_fraction', params['validation_fraction'])}  

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

    opt_target = load_n_predict(params, hyperopt_data, hyperopt=True)

    # since the hyperoptimizer aims to minimize the target values, the code needs to use negative
    # values for R2, accuracy, F1 score and MCC (these values are inverted again before storing them)
    if params['error_type'].lower() in ['r2', 'mcc', 'f1', 'acc']:
        opt_target = -opt_target

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

            if hyperopt_data['model'].upper() in ['GB','VR']:
                csv_hyperopt['learning_rate'] = params['learning_rate']

            if hyperopt_data['model'].upper() == 'GB':
                csv_hyperopt['validation_fraction'] = params['validation_fraction']

        if hyperopt_data['model'].upper() in ['NN','VR']:
                csv_hyperopt['batch_size'] = params['batch_size']
                csv_hyperopt['hidden_layer_sizes'] = params['hidden_layer_sizes']
                csv_hyperopt['learning_rate_init'] = params['learning_rate_init']
                csv_hyperopt['max_iter'] = params['max_iter']
                csv_hyperopt['validation_fraction'] = params['validation_fraction']
    
        elif hyperopt_data['model'].upper() == 'ADAB':
                csv_hyperopt['n_estimators'] = params['n_estimators']
                csv_hyperopt['learning_rate'] = params['learning_rate']
            
        csv_hyperopt[hyperopt_data['error_type']] = opt_target

        # save into a csv file
        csv_hyperopt_df = pd.DataFrame.from_dict(csv_hyperopt, orient='index')
        csv_hyperopt_df = csv_hyperopt_df.transpose()
        
        destination = Path(hyperopt_data['destination'])
        name_csv_hyperopt = destination.joinpath(f"Raw_data/No_PFI/{hyperopt_data['model']}_{hyperopt_data['train']}.csv")
        _ = csv_hyperopt_df.to_csv(name_csv_hyperopt, index = None, header=True)
 
    return {'loss': best, 'status': STATUS_OK}


def load_from_yaml(self,model_type,X_hyperopt):
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

    # number of descriptors to scan, from 1 to the max amount of descriptors using the interval
    # specified with interval_descriptors
    if model_type in ['RF','GB','VR','MVL']:
        max_features = [1]
        n_descriptors = len(X_hyperopt.columns)
        interval_descriptors = int((n_descriptors+1)/params["interval_descriptors"])
        for num in range(max_features[0],n_descriptors,interval_descriptors):
            max_features.append(num)
        max_features.append(n_descriptors)
        params['max_features'] = max_features

    return params


def prepare_sets(self,csv_X,csv_y,size):
    # split into training and validation sets
    try:
        Xy_data = data_split(self,csv_X,csv_y,size)
    except TypeError:
        self.args.log.write(f'   x The data split process failed! This is probably due to using strings/words as values (use --curate to curate the data first)')
        sys.exit()

    # standardization of X values using the mean and SD of the training set
    X_train_scaled, X_valid_scaled = standardize(self,Xy_data['X_train'],Xy_data['X_valid'])
    Xy_data['X_train_scaled'] = X_train_scaled
    Xy_data['X_valid_scaled'] = X_valid_scaled

    # also store the descriptors used (the labels disappear after data_split() )
    Xy_data['X_descriptors'] = csv_X.columns.tolist()

    return Xy_data


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
    try:
        kmeans.fit(X_scaled_array)
    except ValueError:
        self.args.log.write("\nx  The K-means clustering process failed! This is probably due to having NaN or strings as descriptors. To avoid this issue, curate the data first with the CURATE module!")
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


def PFI_workflow(self, csv_df, ML_model, size, Xy_data):
    # filter off parameters with low PFI (not relevant in the model)
    name_csv_hyperopt = f"Raw_data/No_PFI/{ML_model}_{size}"
    path_csv = self.args.destination.joinpath(f'{name_csv_hyperopt}.csv')
    PFI_df = pd.read_csv(path_csv)
    PFI_dict = pd_to_dict(PFI_df) # (using a dict to keep the same format of load_model)
    PFI_discard = PFI_filter(self,Xy_data,PFI_dict,ML_model,size)

    # generate new X datasets and store the descriptors used for the PFI-filtered model
    discard_idx, descriptors_PFI = [],[]
    desc_keep = len(Xy_data['X_train'])
    if self.args.pfi_max != 0:
        desc_keep = self.args.pfi_max
    for _,column in enumerate(Xy_data['X_train']):
        if column not in PFI_discard and len(descriptors_PFI) < desc_keep:
            descriptors_PFI.append(column)
        else:
            discard_idx.append(column)
    Xy_data_PFI = Xy_data.copy()

    Xy_data_PFI['X_train'] = Xy_data['X_train'].drop(discard_idx, axis=1)
    Xy_data_PFI['X_valid'] = Xy_data['X_valid'].drop(discard_idx, axis=1)
    Xy_data_PFI['X_train_scaled'], Xy_data_PFI['X_valid_scaled'] = standardize(self,Xy_data_PFI['X_train'],Xy_data_PFI['X_valid'])
    PFI_dict['X_descriptors'] = descriptors_PFI

    # updates the model's error and descriptors used from the corresponding No_PFI CSV file 
    # (the other parameters remain the same)
    opt_target = load_n_predict(PFI_dict, Xy_data_PFI, hyperopt=True)
    PFI_dict[PFI_dict['error_type']] = opt_target
    
    # save CSV file
    name_csv_hyperopt_PFI = name_csv_hyperopt.replace('No_PFI','PFI')
    path_csv_PFI = self.args.destination.joinpath(f'{name_csv_hyperopt_PFI}_PFI')
    csv_PFI_df = pd.DataFrame.from_dict(PFI_dict, orient='index')
    csv_PFI_df = csv_PFI_df.transpose()
    _ = csv_PFI_df.to_csv(f'{path_csv_PFI}.csv', index = None, header=True)

    # check if this combination is the best model and replace data in the Best_model folder
    _ = update_best(self,csv_df,Xy_data_PFI,path_csv_PFI)


def PFI_filter(self,Xy_data,PFI_dict,ML_model,size):

    # load and fit model
    loaded_model = load_model(PFI_dict)
    loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])

    # we use the validation set during PFI as suggested by the sklearn team:
    # "Using a held-out set makes it possible to highlight which features contribute the most to the 
    # generalization power of the inspected model. Features that are important on the training set 
    # but not on the held-out set might cause the model to overfit."
    score_model = loaded_model.score(Xy_data['X_valid_scaled'], Xy_data['y_valid'])
    perm_importance = permutation_importance(loaded_model, Xy_data['X_valid_scaled'], Xy_data['y_valid'], n_repeats=self.args.pfi_epochs, random_state=self.args.seed)

    # transforms the values into a list and sort the PFI values with the descriptors names
    desc_list, PFI_values, PFI_sd = [],[],[]
    for i,desc in enumerate(Xy_data['X_train']):
        desc_list.append(desc)
        PFI_values.append(perm_importance.importances_mean[i])
        PFI_sd.append(perm_importance.importances_std[i])
  
    PFI_values, PFI_sd, desc_list = (list(t) for t in zip(*sorted(zip(PFI_values, PFI_sd, desc_list), reverse=True)))

    # PFI filter
    PFI_discard = []
    PFI_thres = abs(self.args.pfi_threshold*score_model)
    for i in reversed(range(len(PFI_values))):
        if PFI_values[i] < PFI_thres:
            PFI_discard.append(desc_list[i])

    # disconnect the PFI filter if none of the variables pass the filter
    if len(PFI_discard) == len(PFI_values):
        PFI_discard = []
        self.args.log.write(f'   x The PFI filter was disabled for model {ML_model}_{size} (no variables passed)')

    return PFI_discard


def update_best(self, csv_df, Xy_data, name_csv):

    if 'No_PFI' in name_csv.as_posix():
        folder_suf = 'No_PFI'
        split_n = 3
    else:
        folder_suf = 'PFI'
        split_n = 4

    # detects previous best file with results
    folder_raw = self.args.destination.joinpath(f"Raw_data/{folder_suf}")
    folder_best = self.args.destination.joinpath(f"Best_model/{folder_suf}")
    csv_files = glob.glob(f'{folder_best}/*.csv')
    if len(csv_files) > 0:
        for csv_file in csv_files:
            file_split = os.path.basename(csv_file).replace('.','_').split('_')
            # split_n is added to differentiate between params and db files
            if len(file_split) == split_n and file_split[0].lower() in ['rf','mvl','gb','adab','nn','vr']:
                name_best = csv_file
        results_best = pd.read_csv(name_best)
        best_error = results_best[results_best['error_type'][0]][0]

        # check if the results of the new model are better than the previous best model
        results_model = pd.read_csv(f'{name_csv}.csv')
        new_error = results_model[results_model['error_type'][0]][0]

        # error for current regressor
        replace_best = False
        if results_model['type'][0].lower() == 'reg' and results_best['error_type'][0].lower() in ['rmse','mae']:
            if new_error < best_error:
                replace_best = True
        # precision for current classificator and R2
        else:
            if new_error > best_error:
                replace_best = True
    # first model
    else:
        replace_best = True
    
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
    db_name = Path(f'{folder_raw.joinpath(os.path.basename(name_csv))}_db')
    _ = csv_df.to_csv(f'{db_name}.csv', index = None, header=True)

    # update the new best model (if the current model shows better proficiency)
    if replace_best:
        for file in glob.glob(f'{folder_best}/*.*'):
            os.remove(file)

        # copy the ML model params and database
        shutil.copyfile(f'{name_csv}.csv', f'{folder_best.joinpath(os.path.basename(name_csv))}.csv')
        shutil.copyfile(f'{db_name}.csv', f'{folder_best.joinpath(os.path.basename(db_name))}.csv')


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
    self.args.log.write(f'\no  {title_fig} succesfully created in {path_raw}')
