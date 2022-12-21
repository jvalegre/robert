######################################################.
#     This file stores all Robert functions used     #
######################################################.

import os,sys
import numpy as np
import pandas as pd
import json
import random
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, matthews_corrcoef, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from scipy import stats
from matplotlib import pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# k-neighbours clustering
def k_neigh(X_scaled_split,training_folds_split,random_seed,y_split):
    
    # number of clusters in the training set from the k-neighbours clustering (based on the
    # training set size specified above)
    X_scaled_array = np.asarray(X_scaled_split)
    number_of_clusters = int(len(X_scaled_split)*(training_folds_split/100))

    # to avoid points from the validation set outside the training set, the 2 first training
    # points are automatically set as the 2 points with minimum/maximum response value
    training_points = [y_split.idxmin(),y_split.idxmax()]
    number_of_clusters -= 2
    
    # runs the k-neighbours algorithm and keeps the closest point to the center of each cluster
    kmeans = KMeans(n_clusters=number_of_clusters,random_state=random_seed)
    kmeans.fit(X_scaled_array)
    cluster = kmeans.predict(X_scaled_array)
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

# function to split the data into training, validation and test sets based on k-neighbours clustering
def data_split(X_split,y_split,train_partition,random_seed,split_mode):
    
    training_folds_split = train_partition
    
    if train_partition == 100:
        # if there is no validation set, use all the points
        training_points = np.arange(0,len(X_split),1)
    else:
        if split_mode == 'KN':
            # k-neighbours data split

            # standardize the data before k-neighbours-based data splitting
            Xmeans = X_split.mean(axis=0)
            Xstds = X_split.std(axis=0)
            X_scaled_split = (X_split - Xmeans) / Xstds

            training_points = k_neigh(X_scaled_split,training_folds_split,random_seed,y_split)

        elif split_mode == 'RND':
            n_of_points = int(len(X_split)*(training_folds_split/100))

            random.seed(random_seed)
            training_points = random.sample(range(len(X_split)), n_of_points)

    X_train_kneigh = X_split.iloc[training_points]
    y_train_kneigh = y_split.iloc[training_points]
    X_validation_kneigh = X_split.drop(training_points)
    y_validation_kneigh = y_split.drop(training_points)
              
    return X_train_kneigh, y_train_kneigh, X_validation_kneigh, y_validation_kneigh, training_points


# generates initial parameters for the hyperopt optimization
def hyperopt_params(model_type_hyperopt, X_hyperopt):
    interval_DFT_descriptors = int((len(X_hyperopt.columns)+1)/3)
    if model_type_hyperopt == 'RF':
        # for Random Forest regressor
        Predictor_parameters = ['max_depth', 'max_features', 'n_estimators']

        range_max_depth = [20,40,80,160,320]
        range_DFT_descriptors = [1,len(X_hyperopt.columns)+1] # from 1 to the max amount of DFT descriptors in x_values
        range_n_estimators = [1,3,20,40,80]

        space4rf_hyperopt = {Predictor_parameters[0]: hp.choice(Predictor_parameters[0], range_max_depth),
                Predictor_parameters[1]: hp.choice(Predictor_parameters[1], range(range_DFT_descriptors[0],range_DFT_descriptors[1],interval_DFT_descriptors)),
                Predictor_parameters[2]: hp.choice(Predictor_parameters[2], range_n_estimators)}  

    elif model_type_hyperopt == 'GB':
        # for Gradient boosting regressor
        Predictor_parameters = ['max_depth', 'max_features', 'n_estimators', 'learning_rate', 'validation_fraction']

        range_learning_rate = [0.01,0.05,0.1,0.2,0.5]
        range_validation_fraction = [0.1,0.2]
        range_max_depth = [20,40,80,160,320]
        range_DFT_descriptors = [1,len(X_hyperopt.columns)+1] # from 1 to the max amount of DFT descriptors in x_values
        range_n_estimators = [20,40,80,160,320]

        space4rf_hyperopt = {Predictor_parameters[0]: hp.choice(Predictor_parameters[0], range_max_depth),
                Predictor_parameters[1]: hp.choice(Predictor_parameters[1], range(range_DFT_descriptors[0],range_DFT_descriptors[1],interval_DFT_descriptors)),
                Predictor_parameters[2]: hp.choice(Predictor_parameters[2], range_n_estimators),
                Predictor_parameters[3]: hp.choice(Predictor_parameters[3], range_learning_rate),
                Predictor_parameters[4]: hp.choice(Predictor_parameters[4], range_validation_fraction)}  

    elif model_type_hyperopt == 'AdaB':
        # for AdaBoost regressor
        Predictor_parameters = ['n_estimators', 'learning_rate']

        range_learning_rate = [0.1,0.2,0.5,1,2,5]
        range_n_estimators = [10,20,40,80,160,320]

        space4rf_hyperopt = {Predictor_parameters[0]: hp.choice(Predictor_parameters[0], range_n_estimators),
            Predictor_parameters[1]: hp.choice(Predictor_parameters[1], range_learning_rate)}  

    elif model_type_hyperopt == 'NN':
        # for MLP regressor
        Predictor_parameters = ['batch_size', 'hidden_layer_sizes', 'learning_rate_init', 'max_iter', 'validation_fraction']

        range_batch_size = [20,40,80,160,320]
        range_hidden_layer_sizes = [(32),(64),(128),(64,32),(64,32,16)]
        range_learning_rate_init = [0.0001,0.0005,0.001,0.005,0.01]
        range_max_iter = [200,500]
        range_validation_fraction = [0.1,0.2]

        space4rf_hyperopt = {Predictor_parameters[0]: hp.choice(Predictor_parameters[0], range_batch_size),
                Predictor_parameters[1]: hp.choice(Predictor_parameters[1], range_hidden_layer_sizes),
                Predictor_parameters[2]: hp.choice(Predictor_parameters[2], range_learning_rate_init),
                Predictor_parameters[3]: hp.choice(Predictor_parameters[3], range_max_iter),
                Predictor_parameters[4]: hp.choice(Predictor_parameters[4], range_validation_fraction)}

    elif model_type_hyperopt == 'VR':
        # for Random Forest regressor
        Predictor_parameters = ['max_depth', 'max_features', 'n_estimators', 'learning_rate', 'validation_fraction', 'batch_size', 'hidden_layer_sizes', 'learning_rate_init', 'max_iter']

        range_max_depth = [20,40,160,320]
        range_DFT_descriptors = [1,len(X_hyperopt.columns)+1] # from 1 to the max amount of DFT descriptors in x_values
        range_n_estimators = [5,20,40,160,320]
        range_learning_rate = [0.01,0.05,0.1,0.2,0.5]
        range_validation_fraction = [0.1,0.2]
        range_batch_size = [20,40,160,320]
        range_hidden_layer_sizes = [(32),(64),(64,32),(64,32,16)]
        range_learning_rate_init = [0.0001,0.0005,0.001,0.01]
        range_max_iter = [200,500]

        space4rf_hyperopt = {Predictor_parameters[0]: hp.choice(Predictor_parameters[0], range_max_depth),
                Predictor_parameters[1]: hp.choice(Predictor_parameters[1], range(range_DFT_descriptors[0],range_DFT_descriptors[1],interval_DFT_descriptors)),
                Predictor_parameters[2]: hp.choice(Predictor_parameters[2], range_n_estimators),
                Predictor_parameters[3]: hp.choice(Predictor_parameters[3], range_learning_rate),
                Predictor_parameters[4]: hp.choice(Predictor_parameters[4], range_validation_fraction),
                Predictor_parameters[5]: hp.choice(Predictor_parameters[5], range_batch_size),
                Predictor_parameters[6]: hp.choice(Predictor_parameters[6], range_hidden_layer_sizes),
                Predictor_parameters[7]: hp.choice(Predictor_parameters[7], range_learning_rate_init),
                Predictor_parameters[8]: hp.choice(Predictor_parameters[8], range_max_iter)}  

    elif model_type_hyperopt == 'MVL':
        # for Multivariate lineal models
        Predictor_parameters = ['max_features']
        space4rf_hyperopt = {Predictor_parameters[0]: hp.choice(Predictor_parameters[0], interval_DFT_descriptors)} # from 1 to the max amount of DFT descriptors in x_values

    return space4rf_hyperopt,Predictor_parameters


# prints the results of the hyperopt optimization process
def print_hyperopt_params(model_type_hyperopt,best_parameters_df_hyperopt,training_size_hyperopt,w_dir_hyperopt):
    print('\nThe best parameters for the',model_type_hyperopt,'model are:')

    if model_type_hyperopt in ['RF','GB','VR']:    
        print('\nNumber of estimators:',str(int(best_parameters_df_hyperopt['n_estimators'][0])),
            '\nMax. depth:',str(int(best_parameters_df_hyperopt['max_depth'][0])),
            '\nMax. features:',str( int(best_parameters_df_hyperopt['max_features'][0])))

        if model_type_hyperopt in ['GB','VR']: 
            print('\nLearning rate:',str(float(best_parameters_df_hyperopt['learning_rate'][0])),
                '\nValidation fraction:',str(float(best_parameters_df_hyperopt['validation_fraction'][0])))

    elif model_type_hyperopt == 'AdaB':
        print('\nNumber of estimators:',str(int(best_parameters_df_hyperopt['n_estimators'][0])),
            '\nLearning rate:',str(float(best_parameters_df_hyperopt['learning_rate'][0])))

    if model_type_hyperopt in ['NN','VR']:
        print('\nBatch size:',str(int(best_parameters_df_hyperopt['batch_size'][0])),
            '\nHidden layer sizes:',str(eval(str(best_parameters_df_hyperopt['hidden_layer_sizes'][0]))),
            '\nLearning rate init:',str(float(best_parameters_df_hyperopt['learning_rate_init'][0])),
            '\nMax iterations:',str(int(best_parameters_df_hyperopt['max_iter'][0])))

        if model_type_hyperopt == 'NN':
            print('\nValidation fraction:',str(float(best_parameters_df_hyperopt['validation_fraction'][0])))

    print('\nTraining set proportion:',str(training_size_hyperopt)+'%',
        '\n\nThe optimal parameters have been stored in',
        w_dir_hyperopt)

# calculates RMSE of the validation set with the parameters of the corresponding
# hyperopt optimization cycle
def hyperopt_train_test(params, model_type, prediction_type, Predictor_parameters, X_train_scaled, y_train, training_size, X_validation_scaled, y_validation):

    # keep seed constant for all the hyperopt process, otherwise the program
    # does not reach convergence efficiently
    random_seed = 0

    # set the parameters for each cycle of the hyperopt optimization
    if model_type == 'RF':
        max_depth_gen = params[Predictor_parameters[0]]
        max_features_gen = params[Predictor_parameters[1]]
        n_estimators_gen = params[Predictor_parameters[2]]
        
        if prediction_type == 'reg':
            model_hyperopt = RandomForestRegressor(max_depth=max_depth_gen,
                                    max_features=max_features_gen,
                                    n_estimators=n_estimators_gen,
                                    random_state=random_seed)

        elif prediction_type == 'clas':
            model_hyperopt = RandomForestClassifier(max_depth=max_depth_gen,
                                    max_features=max_features_gen,
                                    n_estimators=n_estimators_gen,
                                    random_state=random_seed)

    elif model_type == 'GB':
        max_depth_gen = params[Predictor_parameters[0]]
        max_features_gen = params[Predictor_parameters[1]]
        n_estimators_gen = params[Predictor_parameters[2]]
        learning_rate_gen = params[Predictor_parameters[3]]
        validation_fraction_gen = params[Predictor_parameters[4]]
    
        if prediction_type == 'reg':
            model_hyperopt = GradientBoostingRegressor(max_depth=max_depth_gen, 
                                    max_features=max_features_gen,
                                    n_estimators=n_estimators_gen,
                                    learning_rate=learning_rate_gen,
                                    validation_fraction=validation_fraction_gen,
                                    random_state=random_seed)

        elif prediction_type == 'clas':
            model_hyperopt = GradientBoostingClassifier(max_depth=max_depth_gen, 
                                    max_features=max_features_gen,
                                    n_estimators=n_estimators_gen,
                                    learning_rate=learning_rate_gen,
                                    validation_fraction=validation_fraction_gen,
                                    random_state=random_seed)

    elif model_type == 'AdaB':
        n_estimators_gen = params[Predictor_parameters[0]]
        learning_rate_gen = params[Predictor_parameters[1]] 

        if prediction_type == 'reg':
            model_hyperopt = AdaBoostRegressor(n_estimators=n_estimators_gen,
                                    learning_rate=learning_rate_gen,
                                    random_state=random_seed)

        elif prediction_type == 'clas':
            model_hyperopt = AdaBoostClassifier(n_estimators=n_estimators_gen,
                                    learning_rate=learning_rate_gen,
                                    random_state=random_seed)

    elif model_type == 'NN':
        batch_size_gen = params[Predictor_parameters[0]]
        hidden_layer_sizes_gen = params[Predictor_parameters[1]]
        learning_rate_init_gen = params[Predictor_parameters[2]]
        max_iter_gen = params[Predictor_parameters[3]]
        validation_fraction_gen = params[Predictor_parameters[4]]

        if prediction_type == 'reg':
            model_hyperopt = MLPRegressor(batch_size=batch_size_gen,
                                    hidden_layer_sizes=hidden_layer_sizes_gen,
                                    learning_rate_init=learning_rate_init_gen,
                                    max_iter=max_iter_gen,
                                    validation_fraction=validation_fraction_gen,
                                    random_state=random_seed)
                                    
        elif prediction_type == 'clas':
            model_hyperopt = MLPClassifier(batch_size=batch_size_gen,
                                    hidden_layer_sizes=hidden_layer_sizes_gen,
                                    learning_rate_init=learning_rate_init_gen,
                                    max_iter=max_iter_gen,
                                    validation_fraction=validation_fraction_gen,
                                    random_state=random_seed)
            
    elif model_type == 'VR':
        max_depth_gen = params[Predictor_parameters[0]]
        max_features_gen = params[Predictor_parameters[1]]
        n_estimators_gen = params[Predictor_parameters[2]]
        learning_rate_gen = params[Predictor_parameters[3]]
        validation_fraction_gen = params[Predictor_parameters[4]]
        batch_size_gen = params[Predictor_parameters[5]]
        hidden_layer_sizes_gen = params[Predictor_parameters[6]]
        learning_rate_init_gen = params[Predictor_parameters[7]]
        max_iter_gen = params[Predictor_parameters[8]]

        if prediction_type == 'reg':
            r1_gen = GradientBoostingRegressor(max_depth=max_depth_gen, 
                                    max_features=max_features_gen,
                                    n_estimators=n_estimators_gen,
                                    learning_rate=learning_rate_gen,
                                    validation_fraction=validation_fraction_gen,
                                    random_state=random_seed)

            r2_gen = RandomForestRegressor(max_depth=max_depth_gen,
                                max_features=max_features_gen,
                                n_estimators=n_estimators_gen,
                                random_state=random_seed)

            r3_gen = MLPRegressor(batch_size=batch_size_gen,
                                    hidden_layer_sizes=hidden_layer_sizes_gen,
                                    learning_rate_init=learning_rate_init_gen,
                                    max_iter=max_iter_gen,
                                    validation_fraction=validation_fraction_gen,
                                    random_state=random_seed)

            model_hyperopt = VotingRegressor([('gb', r1_gen), ('rf', r2_gen), ('nn', r3_gen)])

        elif prediction_type == 'clas':
            r1_gen = GradientBoostingClassifier(max_depth=max_depth_gen, 
                                    max_features=max_features_gen,
                                    n_estimators=n_estimators_gen,
                                    learning_rate=learning_rate_gen,
                                    validation_fraction=validation_fraction_gen,
                                    random_state=random_seed)

            r2_gen = RandomForestClassifier(max_depth=max_depth_gen,
                                max_features=max_features_gen,
                                n_estimators=n_estimators_gen,
                                random_state=random_seed)

            r3_gen = MLPClassifier(batch_size=batch_size_gen,
                                    hidden_layer_sizes=hidden_layer_sizes_gen,
                                    learning_rate_init=learning_rate_init_gen,
                                    max_iter=max_iter_gen,
                                    validation_fraction=validation_fraction_gen,
                                    random_state=random_seed)

            model_hyperopt = VotingClassifier([('gb', r1_gen), ('rf', r2_gen), ('nn', r3_gen)])

    elif model_type == 'MVL':
        max_features_gen = params[Predictor_parameters[0]] # waiting for sklearn to implement this option

        if prediction_type == 'reg':
            model_hyperopt = LinearRegression()

        elif prediction_type == 'clas':
            print('Multivariate models (model_type = \'MVL\') are not compatible with classifiers (prediction_type = \'clas\')')
            sys.exit()

    # Fit the model with the training set
    model_hyperopt.fit(X_train_scaled, y_train)  

    if training_size == 100:
        # if there is not test set, only used values from training
        y_pred_validation = model_hyperopt.predict(X_train_scaled)
        y_validation = y_train
    
    else:
        # Predicted values using the model for training, valid. and test
        y_pred_validation = model_hyperopt.predict(X_validation_scaled)

    # Validation stats
    if prediction_type == 'reg':
        # using RMSE as the value to minimize in the hyperopt process
        _, _, rmse_validation_ind = model_stats(y_validation,y_pred_validation)
        
    elif prediction_type == 'clas':
        # get Matthews correlation coefficient. I make it negative so the optimizer 
        # is consistent to finding a minima as in the case of error in regression
        rmse_validation_ind = -matthews_corrcoef(y_validation,y_pred_validation)

    if model_type == 'RF':
        return rmse_validation_ind,max_features_gen,max_depth_gen,n_estimators_gen
    elif model_type == 'GB':
        return rmse_validation_ind,max_features_gen,max_depth_gen,n_estimators_gen,learning_rate_gen,validation_fraction_gen
    elif model_type == 'AdaB':
        return rmse_validation_ind,n_estimators_gen,learning_rate_gen      
    elif model_type == 'NN':
        return rmse_validation_ind,batch_size_gen,hidden_layer_sizes_gen,learning_rate_init_gen,max_iter_gen,validation_fraction_gen
    elif model_type == 'VR':   
         return rmse_validation_ind,max_features_gen,max_depth_gen,n_estimators_gen,learning_rate_gen,validation_fraction_gen,batch_size_gen,hidden_layer_sizes_gen,learning_rate_init_gen,max_iter_gen
    elif model_type == 'MVL':
        return rmse_validation_ind,max_features_gen


# initial function for hyperopt
def run_hyperopt(n_epochs, model_type, X, training_size, prediction_type, random_init, w_dir, X_train_scaled, y_train, X_validation_scaled, y_validation, name_csv_hyperopt):

    # edit this function to modify the hyperopt parameter optimization (i.e. the 
    # lists represent values to include in the grid search)
    space4rf,Predictor_parameters = hyperopt_params(model_type, X)

    if model_type == 'MVL':
        n_epochs = 1

    # Run hyperopt
    trials = Trials()

    # This best high initial number is a dummy value used as the start of the optimization
    # (otherwise it does not run more than once since best becomes a dictionary)
    best = 100000

    X_train_scaled = np.asarray(X_train_scaled).tolist()
    y_train = np.asarray(y_train).tolist()
    X_validation_scaled = np.asarray(X_validation_scaled).tolist()
    y_validation = np.asarray(y_validation).tolist()

    hp_data = {'best': best, 'model_type': model_type, 'prediction_type': prediction_type,
                'Predictor_parameters': Predictor_parameters, 'X_train_scaled': X_train_scaled,
                'y_train': y_train, 'training_size': training_size, 
                'X_validation_scaled': X_validation_scaled, 'y_validation': y_validation,
                'name_csv_hyperopt': name_csv_hyperopt, 'random_init': random_init}

    with open('hp.json', 'w') as outfile:
        json.dump(hp_data, outfile)

    try:
        best = fmin(f, space4rf, algo=tpe.suggest, max_evals=n_epochs, trials=trials, rstate=np.random.default_rng(random_init))

    except ValueError:
        print('There is an error in the hyperopt module, are you using prediction_type = \'clas\' for regression instead of prediction_type = \'reg\'?')

    try:
        os.remove('hp.json')
    except FileNotFoundError:
        pass

# f function of hyperopt
def f(params):

    with open('hp.json') as json_file:
        hp_data = json.load(json_file)
    
    best = hp_data['best']
    model_type = hp_data['model_type']
    prediction_type = hp_data['prediction_type']
    Predictor_parameters = hp_data['Predictor_parameters']
    X_train_scaled = hp_data['X_train_scaled']
    y_train = hp_data['y_train']
    training_size = hp_data['training_size']
    X_validation_scaled = hp_data['X_validation_scaled']
    y_validation = hp_data['y_validation']
    name_csv_hyperopt = hp_data['name_csv_hyperopt']
    random_init = hp_data['random_init']

    # I need to trick the function because hypergrid retrieves the
    # last parameters that it analyzes after it has completed
    # the number of epochs (not the optimal parameters found)
    acc = hyperopt_train_test(params, model_type, prediction_type, Predictor_parameters, X_train_scaled, y_train, training_size, X_validation_scaled, y_validation)
    
    # The parameters are stored in an external CSV/json files
    # as the optimizer finds better results
    hp_data['best'] = acc[0]
    os.remove('hp.json')
    with open('hp.json', 'w') as outfile:
        json.dump(hp_data, outfile)

    if acc[0] < best:
        best = acc[0]
        train_proportion = str(training_size)+'% k-neighbours split'

        if model_type in ['RF','GB','VR']:
            max_features = acc[1]
            max_depth = acc[2]
            n_estimators = acc[3]

            best_parameters_dict = {'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'max_features': max_features,
                            'train_proportion': train_proportion,
                            'model_type': model_type,
                            'prediction_type': prediction_type,
                            'random_init': random_init}

            if model_type in ['GB','VR']:
                learning_rate = acc[4]
                validation_fraction = acc[5]

                best_parameters_dict['learning_rate'] = learning_rate
                best_parameters_dict['validation_fraction'] = validation_fraction

            if model_type == 'VR':
                batch_size = acc[6]
                hidden_layer_sizes = acc[7]
                learning_rate_init = acc[8]
                max_iter = acc[9]

                best_parameters_dict['batch_size'] = batch_size
                best_parameters_dict['hidden_layer_sizes'] = hidden_layer_sizes
                best_parameters_dict['learning_rate_init'] = learning_rate_init
                best_parameters_dict['max_iter'] = max_iter


        elif model_type == 'AdaB':
            n_estimators = acc[1]
            learning_rate = acc[2]

            best_parameters_dict = {'train_proportion': train_proportion,
                                    'model_type': model_type,
                                    'n_estimators': n_estimators,
                                    'learning_rate': learning_rate,
                                    'prediction_type': prediction_type,
                                    'random_init': random_init}

        elif model_type == 'NN':
            batch_size = acc[1]
            hidden_layer_sizes = acc[2]
            learning_rate_init = acc[3]
            max_iter = acc[4]
            validation_fraction = acc[5]

            best_parameters_dict = {'train_proportion': train_proportion,
                                    'model_type': model_type,
                                    'batch_size': batch_size,
                                    'hidden_layer_sizes': hidden_layer_sizes,
                                    'learning_rate_init': learning_rate_init,
                                    'max_iter': max_iter,
                                    'validation_fraction': validation_fraction,
                                    'prediction_type': prediction_type,
                                    'random_init': random_init}

        elif model_type == 'MVL':
            best_parameters_dict = {'train_proportion': train_proportion,
                                    'model_type': model_type,
                                    'prediction_type': prediction_type,
                                    'random_init': random_init}
            
        if prediction_type == 'reg':
            best_parameters_dict['rmse'] = best
            
        elif prediction_type == 'clas':
            best_parameters_dict['MCC'] = -best

        best_parameters_df = pd.DataFrame.from_dict(best_parameters_dict, orient='index')
        best_parameters_df = best_parameters_df.transpose()
        
        export_param_excel = best_parameters_df.to_csv(name_csv_hyperopt+'.csv', index = None, header=True)
 
    return {'loss': best, 'status': STATUS_OK}


# function to get stats from the model
def model_stats(y,y_pred):   
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    _, _, r_value, _, _ = stats.linregress(y, y_pred)
    r2 = r_value**2

    return r2,mae,rmse


# function to calculate k-fold cross validation
def cross_val_calc(random_init_cv,model_type_cv,df_cv,X_train_cv_scaled,y_train_cv,prediction_type_cv,cv_kfold):
    predictor_model = predictor_model_fun(model_type_cv, df_cv, random_init_cv, prediction_type_cv)
    cv_score = cross_val_score(predictor_model, X_train_cv_scaled, y_train_cv, cv=cv_kfold)

    return cv_score


# correlation filter
def correlation_filter_fun(DFT_parameters_fun,correlation_y_threshold_fun,correlation_x_threshold_fun,fixed_descriptors_fun,descriptors_to_drop_fun,response_value):
    print('\nExcluded parameters:')
    for i,column in enumerate(DFT_parameters_fun.columns):
        if column not in fixed_descriptors_fun and column not in descriptors_to_drop_fun and column != response_value:
            # discards the descriptors with low correlation to the response values
            _, _, r_value, _, _ = stats.linregress(DFT_parameters_fun[column],DFT_parameters_fun[response_value])
            rsquared = r_value**2
            if rsquared < correlation_y_threshold_fun:
                descriptors_to_drop_fun.append(column)
                print(column,': only R**2 = '+str(round(rsquared,2))+' with the experimental dr values')
            # discards correlated descriptors
            if column != DFT_parameters_fun.columns[-1]:
                for j,column2 in enumerate(DFT_parameters_fun.columns):
                    if j > i and column2 not in fixed_descriptors_fun and column2 not in descriptors_to_drop_fun and column2 != response_value:
                        _, _, r_value, _, _ = stats.linregress(DFT_parameters_fun[column],DFT_parameters_fun[column2])
                        rsquared = r_value**2
                        if rsquared > correlation_x_threshold_fun:
                            descriptors_to_drop_fun.append(column)
                            print(column,': R**2 = '+str(round(rsquared,2))+' with '+column2)

    return descriptors_to_drop_fun

# Obtain the model/training size with the minimum rmse_train_PFI
def optimal_model(size_data):
    min_rmse=10000000000
    rmse_no_PFI = []
    for models_data in size_data:
        for rmse_validation in models_data[1]:
            # rmse from PFI models
            if rmse_validation[7] < min_rmse:   
                min_rmse=rmse_validation[7]
                best_model=rmse_validation
            # rmse from models without the PFI filter
            rmse_no_PFI.append(rmse_validation[8])
    
    if min(rmse_no_PFI) < min_rmse:
        # print(size_data[0][1][0][9].columns)
        len_PFI = len(size_data[0][1][0][9].columns)
        len_no_PFI = len(size_data[0][1][0][10].columns)
        print(f"x  Warning! Error lower without PFI filter (no PFI: RMSE = {round(min(rmse_no_PFI),2)} with {len_no_PFI} variables; with PFI filter: {round(min_rmse,2)} with {len_PFI} variables) consider using PFI_filtering=False")    
        
    return best_model

# predictor workflow 
def predictor_workflow(random_init_fun,model_type_fun,df_fun,X_train_scaled_fun,y_train_fun,X_validation_scaled_fun,y_validation_fun,prediction_type_fun,train_partition):

    # just for simplicity and saving code lines, in classification:
    # 1. r2 = accuracy classification score (proportion of correct predictions)
    # 2. mae = F1 score (balanced F-score)
    # 3. rmse = Matthews correlation coefficient (MCC)

    predictor_model = predictor_model_fun(model_type_fun, df_fun, random_init_fun, prediction_type_fun)

    predictor_model.fit(X_train_scaled_fun, y_train_fun)
    
    y_pred_train_fun = predictor_model.predict(X_train_scaled_fun)

    # Get some stats from the performance on the different partitions
    # Training
    if prediction_type_fun == 'reg':
        r2_train_fun, mae_train_fun, rmse_train_fun = model_stats(y_train_fun,y_pred_train_fun)

    elif prediction_type_fun == 'clas':
        r2_train_fun = accuracy_score(y_train_fun,y_pred_train_fun)
        mae_train_fun = f1_score(y_train_fun,y_pred_train_fun)
        rmse_train_fun = matthews_corrcoef(y_train_fun,y_pred_train_fun)
    
    if train_partition == 100:
        r2_validation_fun, mae_validation_fun, rmse_validation_fun = 0, 0, 0

    else:
        # Validation
        y_pred_validation_fun = predictor_model.predict(X_validation_scaled_fun)

        if prediction_type_fun == 'reg':
            r2_validation_fun, mae_validation_fun, rmse_validation_fun = model_stats(y_validation_fun,y_pred_validation_fun)
        elif prediction_type_fun == 'clas':
            r2_validation_fun = accuracy_score(y_validation_fun,y_pred_validation_fun)
            mae_validation_fun = f1_score(y_validation_fun,y_pred_validation_fun)
            rmse_validation_fun = matthews_corrcoef(y_validation_fun,y_pred_validation_fun)

    return r2_train_fun,mae_train_fun,rmse_train_fun,r2_validation_fun,mae_validation_fun,rmse_validation_fun,y_pred_train_fun,y_pred_validation_fun


# function to print STATS from the predictor model
def print_model_stats(model_type_print,X_train_scaled_print,X_validation_scaled_print,r2_train_print,mae_train_print,rmse_train_print,r2_validation_print,mae_validation_print,rmse_validation_print,prediction_type_fun,cv_score_print,cv_kfold_print,results_file):
# def print_model_stats(dict_best):
    print_line = f'Model: {model_type_print}\n'
    print_line += 'k-neighbours-based training, validation and test sets have been created with this distribution:\n'
    print_line += f'Training points: {len(X_train_scaled_print)}\n'
    if results_file in ['Robert_results.txt','Robert_results_x-shuffle.txt','Robert_results_y-shuffle.txt']:
        print_line += f'Validation points: {len(X_validation_scaled_print)}\n\n'
    elif results_file == 'Robert_results_test_set.txt':
        print_line += f'Test points: {len(X_validation_scaled_print)}\n\n'

    if prediction_type_fun == 'reg':
        print_line += f'k-neighbours-based training: R2 = {round(r2_train_print,2)}; MAE = {round(mae_train_print,2)}; RMSE = {round(rmse_train_print,2)}\n'
        if cv_kfold_print is not None:
            print_line += f'{cv_kfold_print}-fold cross validation: {round(cv_score_print.mean(),2)} ' + u'\u00B1' + f' {round(cv_score_print.std(),2)}\n'
        if results_file in ['Robert_results.txt','Robert_results_x-shuffle.txt','Robert_results_y-shuffle.txt']:
            print_line += f'k-neighbours-based validation: R2 = {round(r2_validation_print,2)}; MAE = {round(mae_validation_print,2)}; RMSE = {round(rmse_validation_print,2)}'
        elif results_file == 'Robert_results_test_set.txt':
              print_line += f'k-neighbours-based test: R2 = {round(r2_validation_print,2)}; MAE = {round(mae_validation_print,2)}; RMSE = {round(rmse_validation_print,2)}'
          
    if prediction_type_fun == 'clas':
        print_line += f'k-neighbours-based training: Accuracy = {round(r2_train_print,2)}; F1 score = {round(mae_train_print,2)}; MCC = {round(rmse_train_print,2)}\n'
        if cv_kfold_print is not None:
            print_line += f'{cv_kfold_print}-fold cross validation: {round(cv_score_print.mean(),2)} ' + u'\u00B1' + f' {round(cv_score_print.std(),2)}\n'
        if results_file in ['Robert_results.txt','Robert_results_x-shuffle.txt','Robert_results_y-shuffle.txt']:
            print_line += f'k-neighbours-based validation: Accuracy = {round(r2_validation_print,2)}; F1 score = {round(mae_validation_print,2)}; MCC = {round(rmse_validation_print,2)}'
        elif results_file == 'Robert_results_test_set.txt':
            print_line += f'k-neighbours-based test: Accuracy = {round(r2_validation_print,2)}; F1 score = {round(mae_validation_print,2)}; MCC = {round(rmse_validation_print,2)}'

    print(print_line)
    if results_file != None:
        log = open(results_file, "w")
        log.write(print_line)
        log.close()

# function to generate the predictor model
def predictor_model_fun(model_type_fun, best_parameters_df, random_state, prediction_type_fun):
    if prediction_type_fun == 'reg':
        if model_type_fun == 'RF':   
            predictor_model_fun = RandomForestRegressor(random_state=random_state,
                n_estimators=int(best_parameters_df['n_estimators'][0]), max_features= int(best_parameters_df['max_features'][0]),
                max_depth=int(best_parameters_df['max_depth'][0]))

        elif model_type_fun == 'GB':
            predictor_model_fun = GradientBoostingRegressor(max_depth=int(best_parameters_df['max_depth'][0]), 
                                max_features= int(best_parameters_df['max_features'][0]), n_estimators=int(best_parameters_df['n_estimators'][0]),
                                learning_rate=float(best_parameters_df['learning_rate'][0]), random_state=random_state,
                                validation_fraction=float(best_parameters_df['validation_fraction'][0]))

        elif model_type_fun == 'AdaB':
            predictor_model_fun = AdaBoostRegressor(n_estimators=int(best_parameters_df['n_estimators'][0]),
                                learning_rate=float(best_parameters_df['learning_rate'][0]), random_state=random_state)

        elif model_type_fun == 'NN':
            predictor_model_fun = MLPRegressor(batch_size=int(best_parameters_df['batch_size'][0]),
                                hidden_layer_sizes=eval(str(best_parameters_df['hidden_layer_sizes'][0])),
                                learning_rate_init=float(best_parameters_df['learning_rate_init'][0]),
                                max_iter=int(best_parameters_df['max_iter'][0]), random_state=random_state,
                                validation_fraction=float(best_parameters_df['validation_fraction'][0]))

        elif model_type_fun == 'VR':
            r1 = GradientBoostingRegressor(max_depth=int(best_parameters_df['max_depth'][0]), 
                                        max_features= int(best_parameters_df['max_features'][0]),
                                        n_estimators=int(best_parameters_df['n_estimators'][0]),
                                        learning_rate=float(best_parameters_df['learning_rate'][0]),
                                        validation_fraction=float(best_parameters_df['validation_fraction'][0]),
                                        random_state=random_state)

            r2 = RandomForestRegressor(max_depth=int(best_parameters_df['max_depth'][0]),
                                    max_features= int(best_parameters_df['max_features'][0]),
                                    n_estimators=int(best_parameters_df['n_estimators'][0]),
                                    random_state=random_state)

            r3 = MLPRegressor(batch_size=int(best_parameters_df['batch_size'][0]),
                                        hidden_layer_sizes=eval(str(best_parameters_df['hidden_layer_sizes'][0])),
                                        learning_rate_init=float(best_parameters_df['learning_rate_init'][0]),
                                        max_iter=int(best_parameters_df['max_iter'][0]),
                                        validation_fraction=float(best_parameters_df['validation_fraction'][0]),
                                        random_state=random_state)

            predictor_model_fun = VotingRegressor([('gb', r1), ('rf', r2), ('nn', r3)])

        elif model_type_fun == 'MVL':
            predictor_model_fun = LinearRegression()

    elif prediction_type_fun == 'clas':
        if model_type_fun == 'RF':   
            predictor_model_fun = RandomForestClassifier(random_state=random_state,
                n_estimators=int(best_parameters_df['n_estimators'][0]), max_features= int(best_parameters_df['max_features'][0]),
                max_depth=int(best_parameters_df['max_depth'][0]))

        elif model_type_fun == 'GB':
            predictor_model_fun = GradientBoostingClassifier(max_depth=int(best_parameters_df['max_depth'][0]), 
                                max_features= int(best_parameters_df['max_features'][0]), n_estimators=int(best_parameters_df['n_estimators'][0]),
                                learning_rate=float(best_parameters_df['learning_rate'][0]), random_state=random_state,
                                validation_fraction=float(best_parameters_df['validation_fraction'][0]))

        elif model_type_fun == 'AdaB':
            predictor_model_fun = AdaBoostClassifier(n_estimators=int(best_parameters_df['n_estimators'][0]),
                                learning_rate=float(best_parameters_df['learning_rate'][0]), random_state=random_state)

        elif model_type_fun == 'NN':
            predictor_model_fun = MLPClassifier(batch_size=int(best_parameters_df['batch_size'][0]),
                                hidden_layer_sizes=eval(str(best_parameters_df['hidden_layer_sizes'][0])),
                                learning_rate_init=float(best_parameters_df['learning_rate_init'][0]),
                                max_iter=int(best_parameters_df['max_iter'][0]), random_state=random_state,
                                validation_fraction=float(best_parameters_df['validation_fraction'][0]))

        elif model_type_fun == 'VR':
            r1 = GradientBoostingClassifier(max_depth=int(best_parameters_df['max_depth'][0]), 
                                        max_features= int(best_parameters_df['max_features'][0]),
                                        n_estimators=int(best_parameters_df['n_estimators'][0]),
                                        learning_rate=float(best_parameters_df['learning_rate'][0]),
                                        validation_fraction=float(best_parameters_df['validation_fraction'][0]),
                                        random_state=random_state)

            r2 = RandomForestClassifier(max_depth=int(best_parameters_df['max_depth'][0]),
                                    max_features= int(best_parameters_df['max_features'][0]),
                                    n_estimators=int(best_parameters_df['n_estimators'][0]),
                                    random_state=random_state)

            r3 = MLPClassifier(batch_size=int(best_parameters_df['batch_size'][0]),
                                        hidden_layer_sizes=eval(str(best_parameters_df['hidden_layer_sizes'][0])),
                                        learning_rate_init=float(best_parameters_df['learning_rate_init'][0]),
                                        max_iter=int(best_parameters_df['max_iter'][0]),
                                        validation_fraction=float(best_parameters_df['validation_fraction'][0]),
                                        random_state=random_state)

            predictor_model_fun = VotingClassifier([('gb', r1), ('rf', r2), ('nn', r3)])

        elif model_type_fun == 'MVL':
            print('Multivariate models (model_type = \'MVL\') are not compatible with classifiers (prediction_type = \'clas\')')
            sys.exit()

    return predictor_model_fun


# functions to perform the permutation feature importances (PFI) analysis
def PFI_workflow(X_PFI_fun,model_type_PFI_fun,best_parameters_df_PFI_fun,X_train_scaled_PFI_fun,y_train_PFI_fun,X_validation_scaled_PFI_fun,y_validation_PFI_fun,n_repeats_PFI_fun,per_cent_PFI_filter_PFI_fun,save_PFI,prediction_type_fun,PFI_filtering):
    combined_descriptor_list = []
    for column in X_PFI_fun:
        combined_descriptor_list.append(column)

    model_perm = predictor_model_fun(model_type_PFI_fun, best_parameters_df_PFI_fun, 0, prediction_type_fun)

    model_perm.fit(X_train_scaled_PFI_fun, y_train_PFI_fun)
    score_model = model_perm.score(X_validation_scaled_PFI_fun, y_validation_PFI_fun)

    perm_importance = permutation_importance(model_perm, X_validation_scaled_PFI_fun, y_validation_PFI_fun, n_repeats=n_repeats_PFI_fun, random_state=0)

    # transforms the values into a list and sort the PFI values with the descriptors names
    PFI_values, PFI_SD = [],[]
    for value in perm_importance.importances_mean:
        PFI_values.append(value)

    for sd in perm_importance.importances_std:
        PFI_SD.append(sd)
        
    PFI_values, PFI_SD, combined_descriptor_list = (list(t) for t in zip(*sorted(zip(PFI_values, PFI_SD, combined_descriptor_list), reverse=True)))

    # PFI filter
    if PFI_filtering:
        PFI_filter = per_cent_PFI_filter_PFI_fun*score_model
        for i in reversed(range(len(PFI_values))):
            if PFI_values[i] < PFI_filter:
                del PFI_values[i]
                del PFI_SD[i]
                del combined_descriptor_list[i]

    if save_PFI:
        # printing and representing the results
        print('\nPermutation feature importances of the descriptors in the '+model_type_PFI_fun+' model (for the validation set). Only showing values that drop the original score at least by '+f'{per_cent_PFI_filter_PFI_fun*100}'+'%:\n')
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