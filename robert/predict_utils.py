#####################################################.
#     This file stores functions from PREDICT       #
#####################################################.

import os
import sys
import ast
from pathlib import Path
import pandas as pd
import numpy as np
from robert.utils import (
    standardize,
    load_dfs,
    load_database,
    categorical_transform,
    get_graph_style,
    pearson_map,
    graph_reg,
    graph_clas,
    )


def load_test(self, Xy_data, params_df, Xy_test_df):
    ''''
    Loads Xy data of the test set
    '''

    descs_model = ast.literal_eval(params_df['X_descriptors'][0])
    Xy_test_csv, X_test_csv, y_test_csv = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

    # test points coming from the files specified in csv_test
    if self.args.csv_test != '':
        Xy_test_csv = load_database(self, self.args.csv_test, "predict")
        X_test_csv, y_test_csv = test_csv(self,Xy_test_csv,descs_model,params_df)
        Xy_data['X_csv_test'] = X_test_csv
        Xy_data['y_csv_test'] = y_test_csv
        _, Xy_data['X_csv_test_scaled'] = standardize(self,Xy_data['X_train'],Xy_data['X_csv_test'])
        Xy_test_df = Xy_test_csv

    # test points coming from the test_set option (from GENERATE)
    if len(Xy_data['X_test']) > 0:
        _, Xy_data['X_test_scaled'] = standardize(self,Xy_data['X_train'],Xy_data['X_test'])
        
    return Xy_data, Xy_test_df


def test_csv(self,Xy_test_df,descs_model,params_df):
    """
    Separates the test databases into X and y. This allows to merge test external databases that 
    contain different columns with internal test databases coming from GENERATE
    """

    y_test_df = pd.DataFrame()
    
    try:
        X_test_df = Xy_test_df[descs_model]
    except KeyError:
        # this might fail if the initial categorical variables have not been transformed
        try:
            self.args.log.write(f"\n   x  There are missing descriptors in the test set! Looking for categorical variables converted from CURATE")
            Xy_test_df = categorical_transform(self,Xy_test_df,'predict')
            X_test_df = Xy_test_df[descs_model]
            self.args.log.write(f"   o  The missing descriptors were successfully created")
        except KeyError:
            self.args.log.write(f"   x  There are still missing descriptors in the test set! The following descriptors are needed: {descs_model}")
            self.args.log.finalize()
            sys.exit()

    if params_df['y'][0] in Xy_test_df:
        y_test_df = Xy_test_df[params_df['y'][0]]

    return X_test_df, y_test_df


def plot_predictions(self, params_dict, Xy_data, path_n_suffix):
    '''
    Plot graphs of predicted vs actual values for train, validation and test sets
    '''

    set_types = ['train','valid']
    if 'y_pred_test' in Xy_data and not Xy_data['y_test'].isnull().values.any() and len(Xy_data['y_test']) > 0:
        set_types.append('test')
    
    graph_style = get_graph_style()
    
    self.args.log.write(f"\n   o  Saving graphs and CSV databases in:")
    if params_dict['type'].lower() == 'reg':
        # Plot graph with all sets
        _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style)
        # Plot CV average ± SD graph of validation or test set
        _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style,cv_mapie_graph=True)
        if 'y_pred_csv_test' in Xy_data and not Xy_data['y_csv_test'].isnull().values.any() and len(Xy_data['y_csv_test']) > 0:
            # Plot CV average ± SD graph of validation or test set
            _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style,csv_test=True,cv_mapie_graph=True)

    elif params_dict['type'].lower() == 'clas':
        for set_type in set_types:
            _ = graph_clas(self,Xy_data,params_dict,set_type,path_n_suffix)
        if 'y_pred_csv_test' in Xy_data and not Xy_data['y_csv_test'].isnull().values.any() and len(Xy_data['y_csv_test']) > 0:
            set_type = 'csv_test'
            _ = graph_clas(self,Xy_data,params_dict,set_type,path_n_suffix,csv_test=True)

    return graph_style


def save_predictions(self,Xy_data,params_dir,Xy_test_df,params_dict):
    '''
    Saves CSV files with the different sets and their predicted results
    '''

    Xy_orig_df, Xy_path, params_df, _, _, suffix_title = load_dfs(self,params_dir,'no_print')
    Xy_orig_df = Xy_orig_df[0] # Now Xy_orig_df is a list because we do CV after GENERATE (to select the best model) but in this case we only have one element
    params_df = params_df[0] # Now params_df is a list because we do CV after GENERATE (to select the best model) but in this case we only have one element
    base_csv_name = '_'.join(os.path.basename(Path(Xy_path)).replace('.csv','_').split('_')[0:2])
    base_csv_name = f'PREDICT/{base_csv_name}'
    base_csv_path = f"{Path(os.getcwd()).joinpath(base_csv_name)}"
    Xy_orig_train = Xy_orig_df[Xy_orig_df.Set == 'Training']
    Xy_orig_train[f'{params_df["y"][0]}_pred'] = Xy_data['y_pred_train']
    train_path = f'{base_csv_path}_train_{suffix_title}.csv'
    _ = Xy_orig_train.to_csv(train_path, index = None, header=True)
    print_preds = f'      -  Train set with predicted results: PREDICT/{os.path.basename(train_path)}'
    Xy_orig_valid = Xy_orig_df[Xy_orig_df.Set == 'Validation']
    Xy_orig_valid[f'{params_df["y"][0]}_pred'] = Xy_data['y_pred_valid']
    # Search in the csv file for model type to avoid problems when using only the predict module with a classification model (reg model is the default)
    if params_df['type'].values[0] == 'reg' and 'y_pred_valid_sd' in Xy_data:
        Xy_orig_valid[f'{params_df["y"][0]}_pred_sd'] = Xy_data['y_pred_valid_sd']
    valid_path = f'{base_csv_path}_valid_{suffix_title}.csv'
    _ = Xy_orig_valid.to_csv(valid_path, index = None, header=True)
    print_preds += f'\n      -  Validation set with predicted results: PREDICT/{os.path.basename(valid_path)}'
    Xy_data['csv_pred_name'] = os.path.basename(valid_path)
    # saves test predictions
    Xy_orig_test = None
    if 'X_test_scaled' in Xy_data:
        Xy_orig_test = Xy_orig_df[Xy_orig_df.Set == 'Test']
        Xy_orig_test[f'{params_df["y"][0]}_pred'] = Xy_data['y_pred_test']
        if params_df['type'].values[0] == 'reg' and 'y_pred_test_sd' in Xy_data:
            Xy_orig_test[f'{params_df["y"][0]}_pred_sd'] = Xy_data['y_pred_test_sd']
        test_path = f'{base_csv_path}_test_{suffix_title}.csv'
        _ = Xy_orig_test.to_csv(test_path, index = None, header=True)
        print_preds += f'\n      -  Test set with predicted results: PREDICT/{os.path.basename(test_path)}'
        Xy_data['csv_pred_name'] = os.path.basename(test_path)
        
    # saves prediction for external test in --csv_test
    if self.args.csv_test != '':
        Xy_test_df[f'{params_df["y"][0]}_pred'] = Xy_data['y_pred_csv_test']
        if params_df['type'].values[0] == 'reg':
            Xy_test_df[f'{params_df["y"][0]}_pred_sd'] = Xy_data['y_pred_csv_test_sd']
        folder_csv = f'{os.path.dirname(base_csv_path)}/csv_test'
        Path(folder_csv).mkdir(exist_ok=True, parents=True)
        csv_name = f'{os.path.basename(self.args.csv_test)}'.split(".csv")[0]
        csv_name += f'_predicted_{suffix_title}.csv'
        csv_test_path = f'{folder_csv}/{csv_name}'
        _ = Xy_test_df.to_csv(csv_test_path, index = None, header=True)
        print_preds += f'\n      -  External set with predicted results: PREDICT/csv_test/{os.path.basename(csv_test_path)}'
        Xy_data['csv_pred_name'] = f'csv_test/{os.path.basename(csv_test_path)}'

    self.args.log.write(print_preds)

    path_n_suffix = f'{base_csv_path}_{suffix_title}'

    # store the names of the datapoints
    name_points = {}
    if params_dict['names'] != '':
        if params_dict['names'].lower() in Xy_orig_train: # accounts for upper/lowercase mismatches
            params_dict['names'] = params_dict['names'].lower()
        if params_dict['names'].upper() in Xy_orig_train:
            params_dict['names'] = params_dict['names'].upper()
        if params_dict['names'] in Xy_orig_train:
            name_points['train'] = Xy_orig_train[params_dict['names']]
            name_points['valid'] = Xy_orig_valid[params_dict['names']]
        if Xy_orig_test is not None:
            name_points['test'] = Xy_orig_test[params_dict['names']]

    return path_n_suffix, name_points, Xy_data


def print_predict(self,Xy_data,params_dict,path_n_suffix,loaded_model):
    '''
    Prints results of the predictions for all the sets
    '''
    
    dat_file = f'{os.path.dirname(path_n_suffix)}/Results_{os.path.basename(path_n_suffix)}.dat'
    path_reduced = '/'.join(f'{dat_file}'.replace('\\','/').split('/')[-2:])
    print_results = f"\n   o  Results saved in {path_reduced}:"
    set_print = 'Train:Validation'

    # get number of points and proportions
    n_train = len(Xy_data['X_train'])
    n_valid = len(Xy_data['X_valid'])
    n_test = 0
    n_points = f'{n_train}:{n_valid}'
    if 'X_test' in Xy_data and len(Xy_data['X_test']) > 0:
        set_print += ':Test'
        n_test = len(Xy_data['X_test'])
        n_points += f':{n_test}'
    total_points = n_train + n_valid + n_test
    print_results += f"\n      -  Points {set_print} = {n_points}"

    prop_train = round(n_train*100/total_points)
    prop_valid = round(n_valid*100/total_points)
    prop_test = round(n_test*100/total_points)
    prop_print = f'{prop_train}:{prop_valid}'
    if 'X_test' in Xy_data and len(Xy_data['X_test']) > 0:
        prop_print += f':{prop_test}'
    print_results += f"\n      -  Proportion {set_print} = {prop_print}"
    
    n_descps = len(Xy_data['X_train'].keys())
    print_results += f"\n      -  Number of descriptors = {n_descps}"
    print_results += f"\n      -  Proportion (train+valid.) points:descriptors = {n_train+n_valid}:{n_descps}"

    # print results and save dat file
    if params_dict['type'].lower() == 'reg':
        print_results += f"\n      -  Train : R2 = {Xy_data['r2_train']:.2}, MAE = {Xy_data['mae_train']:.2}, RMSE = {Xy_data['rmse_train']:.2}"
        print_results += f"\n      -  Valid. : R2 = {Xy_data['r2_valid']:.2}, MAE = {Xy_data['mae_valid']:.2}, RMSE = {Xy_data['rmse_valid']:.2}"
        if 'y_pred_test' in Xy_data and not Xy_data['y_test'].isnull().values.any() and len(Xy_data['y_test']) > 0:
            print_results += f"\n      -  Test : R2 = {Xy_data['r2_test']:.2}, MAE = {Xy_data['mae_test']:.2}, RMSE = {Xy_data['rmse_test']:.2}"
        if 'y_pred_csv_test' in Xy_data and not Xy_data['y_csv_test'].isnull().values.any() and len(Xy_data['y_csv_test']) > 0:
            print_results += f"\n      -  csv_test : R2 = {Xy_data['r2_csv_test']:.2}, MAE = {Xy_data['mae_csv_test']:.2}, RMSE = {Xy_data['rmse_csv_test']:.2}"
        # add equation for linear models
        if params_dict['model'].upper() == 'MVL' or self.args.evaluate == 'True':
            desc_mvl = ast.literal_eval(params_dict['X_descriptors'])
            print_results += f"\n\n   o  Linear model equation, with coefficients obtained using standardized data (coefficient values/importances can be compared):"
            print_results += f"\n      - {params_dict['y']} = {loaded_model.intercept_:.2} "
            for i, coeff in enumerate(loaded_model.coef_):
                if float(coeff) >= 0:
                    print_results += f"+ ({coeff:.2} * {desc_mvl[i]}) "
                else:
                    print_results += f"- ({np.abs(coeff):.2} * {desc_mvl[i]}) "

    elif params_dict['type'].lower() == 'clas':
        print_results += f"\n      -  Train : Accuracy = {Xy_data['acc_train']:.2}, F1 score = {Xy_data['f1_train']:.2}, MCC = {Xy_data['mcc_train']:.2}"
        print_results += f"\n      -  Valid. : Accuracy = {Xy_data['acc_valid']:.2}, F1 score = {Xy_data['f1_valid']:.2}, MCC = {Xy_data['mcc_valid']:.2}"
        if 'y_pred_test' in Xy_data and not Xy_data['y_test'].isnull().values.any() and len(Xy_data['y_test']) > 0:
            print_results += f"\n      -  Test : Accuracy = {Xy_data['acc_test']:.2}, F1 score = {Xy_data['f1_test']:.2}, MCC = {Xy_data['mcc_test']:.2}"
        if 'y_pred_csv_test' in Xy_data and not Xy_data['y_csv_test'].isnull().values.any() and len(Xy_data['y_csv_test']) > 0:
            print_results += f"\n      -  csv_test : Accur. = {Xy_data['acc_csv_test']:.2}, F1 score = {Xy_data['f1_csv_test']:.2}, MCC = {Xy_data['mcc_csv_test']:.2}"

    self.args.log.write(print_results)


def print_cv_var(self,Xy_data,params_dict,path_n_suffix):
    '''
    Prints results of the predictions for all the sets
    '''

    shap_plot_file = f'{os.path.dirname(path_n_suffix)}/CV_variability_{os.path.basename(path_n_suffix)}.png'
    path_reduced = '/'.join(f'{shap_plot_file}'.replace('\\','/').split('/')[-2:])
    if Xy_data['cv_type'] == 'loocv':
        cv_type = f'LOOCV'
    else:
        kfold = Xy_data['cv_type'].split('_')[-3]
        cv_type = f'{kfold}-fold CV'

    print_cv_var = f"\n   o  Cross-validation variation (with {cv_type}) graph saved in {path_reduced}:"
    print_cv_var += f"\n      -  Standard deviations saved in PREDICT/{Xy_data['csv_pred_name']} in the {params_dict['y']}_pred_sd column"
    print_cv_var += f"\n      -  Average SD = {round(Xy_data['avg_sd'],2)}"
    print_cv_var += f"\n      -  y range of dataset (train+valid.) = {round(Xy_data['pred_min'],2)} to {round(Xy_data['pred_max'],2)}, total {round(Xy_data['pred_range'],2)}"

    self.args.log.write(print_cv_var)


def pearson_map_predict(self,Xy_data,params_dir):
    '''
    Plots the Pearson map and analyzes correlation of descriptors.
    '''

    X_combined = pd.concat([Xy_data['X_train'],Xy_data['X_valid']], axis=0).reset_index(drop=True)
    corr_matrix = pearson_map(self,X_combined,'predict',params_dir=params_dir)

    corr_dict = {'descp_1': [],
                 'descp_2': [],
                 'r': []
    }
    for i,descp in enumerate(corr_matrix.columns):
        for j,val in enumerate(corr_matrix[descp]):
            if i < j and np.abs(val) > 0.8:
                corr_dict['descp_1'].append(corr_matrix.columns[i])
                corr_dict['descp_2'].append(corr_matrix.columns[j])
                corr_dict['r'].append(val)

    print_corr = f'      Ideally, variables should show low correlations.' # no initial \n, it's a new log.write
    if len(corr_dict['descp_1']) == 0:
        print_corr += f"\n      o  Correlations between variables are acceptable"
    else:
        abs_r_list = list(np.abs(corr_dict['r']))
        abs_max_r = max(abs_r_list)
        max_r = corr_dict['r'][abs_r_list.index(abs_max_r)]
        max_descp_1 = corr_dict['descp_1'][abs_r_list.index(abs_max_r)]
        max_descp_2 = corr_dict['descp_2'][abs_r_list.index(abs_max_r)]
        if abs_max_r > 0.84:
            print_corr += f"\n      x  WARNING! High correlations observed (up to r = {round(max_r,2)} or R2 = {round(max_r*max_r,2)}, for {max_descp_1} and {max_descp_2})"
        elif abs_max_r > 0.71:
            print_corr += f"\n      x  WARNING! Noticeable correlations observed (up to r = {round(max_r,2)} or R2 = {round(max_r*max_r,2)}, for {max_descp_1} and {max_descp_2})"

    self.args.log.write(print_corr)
