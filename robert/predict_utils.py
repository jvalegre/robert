#####################################################.
#     This file stores functions from PREDICT       #
#####################################################.

import os
import ast
import json
from pathlib import Path
import pandas as pd
import numpy as np
from robert.utils import (
    get_graph_style,
    pearson_map,
    graph_reg,
    graph_clas,
    get_error_labels,
    mcc_scorer_clf,
    )


def _get_class_mapping_reverse(model_data):
    mapping = model_data.get('class_mapping_reverse')
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except (json.JSONDecodeError, TypeError):
            mapping = ast.literal_eval(mapping)
    if isinstance(mapping, dict):
        return {int(key): value for key, value in mapping.items()}
    if 'class_0_label' in model_data and 'class_1_label' in model_data:
        return {
            0: model_data['class_0_label'],
            1: model_data['class_1_label']
        }
    return None


def plot_predictions(self, params_dict, Xy_data, path_n_suffix):
    '''
    Plot graphs of predicted vs actual values for train, validation and test sets
    '''

    set_types = [f"{params_dict['repeat_kfolds']}x {params_dict['kfold']}-fold CV",'test']
    
    graph_style = get_graph_style()
    
    self.args.log.write(f"\n   o  Saving graphs in:")

    if params_dict['type'].lower() == 'reg':
        # Plot graph with all sets (kept with its title - the saved PNG stays fully titled;
        # the report crops the title out via a precisely-measured container height instead,
        # see report.py's print_score_images height)
        _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style)
        # Plot CV average ± SD graph of validation or test set
        _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style,sd_graph=True)
        # Plot CV average ± SD graph of the out-of-fold train+validation predictions (item 6's
        # 2nd stability facet - see print_predict()'s "Average SD in train+validation" line)
        _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style,sd_graph=True,sd_set='train')
        if 'y_external' in Xy_data and not Xy_data['y_external'].isnull().values.any() and len(Xy_data['y_external']) > 0:
            # Plot CV average ± SD graph of external set
            set_type = 'external'
            _ = graph_reg(self,Xy_data,params_dict,set_type,path_n_suffix,graph_style,csv_test=True,sd_graph=True)

    elif params_dict['type'].lower() == 'clas':
        for set_type in set_types:
            _ = graph_clas(self,Xy_data,params_dict,set_type,path_n_suffix)
        if 'y_external' in Xy_data and not Xy_data['y_external'].isnull().values.any() and len(Xy_data['y_external']) > 0:
            set_type = 'external'
            _ = graph_clas(self,Xy_data,params_dict,set_type,path_n_suffix,csv_test=True)

    return graph_style


def save_predictions(self,Xy_data,model_data,suffix_title):
    '''
    Saves CSV files with the different sets and their predicted results
    '''

    # Check if we need to reconvert class labels (for classification with string labels)
    class_mapping_reverse = _get_class_mapping_reverse(model_data)
    reconvert_labels = class_mapping_reverse is not None

    # save CV and test results as a single df
    Xy_train, Xy_test = pd.DataFrame(Xy_data['names_train']), pd.DataFrame(Xy_data['names_test'])
    for col in Xy_data['X_train']:
        Xy_train[col] = Xy_data['X_train'][col].tolist()
        Xy_test[col] = Xy_data['X_test'][col].tolist()
    
    # Store y values and predictions, reconverting if needed
    y_col = model_data['y']
    
    # For training set
    y_train_values = Xy_data['y_train'].tolist()
    y_pred_train_values = Xy_data['y_pred_train']
    if reconvert_labels:
        y_train_values = [class_mapping_reverse[int(y)] for y in y_train_values]
        y_pred_train_values = [class_mapping_reverse[int(y)] for y in y_pred_train_values]
    
    Xy_train[y_col] = y_train_values
    Xy_train[f"{y_col}_pred"] = y_pred_train_values
    Xy_train[f"{y_col}_pred_sd"] = Xy_data['y_pred_train_sd']
    include_conformal = bool(getattr(self.args, "_api_predict", False))
    if include_conformal:
        hw_scalar = float(Xy_data.get("conformal_half_width", float("nan")))
        if model_data["type"].lower() != "reg":
            hw_scalar = float("nan")
        Xy_train[f"{y_col}_pred_conformal_hw"] = [hw_scalar] * len(Xy_train)

    # For test set
    y_test_values = Xy_data['y_test'].tolist()
    y_pred_test_values = Xy_data['y_pred_test']
    if reconvert_labels:
        y_test_values = [class_mapping_reverse[int(y)] for y in y_test_values]
        y_pred_test_values = [class_mapping_reverse[int(y)] for y in y_pred_test_values]
    
    Xy_test[y_col] = y_test_values
    Xy_test[f"{y_col}_pred"] = y_pred_test_values
    Xy_test[f"{y_col}_pred_sd"] = Xy_data['y_pred_test_sd']
    if include_conformal:
        Xy_test[f"{y_col}_pred_conformal_hw"] = [hw_scalar] * len(Xy_test)

    df_results = pd.concat([Xy_train, Xy_test], axis=0)

    # add column with sets
    train_list = ['CV' for _ in Xy_data['y_train']]
    test_list = ['Test' for _ in Xy_data['y_test']]
    col_set = train_list + test_list
    df_results['Set'] = col_set

    # save results as CSV
    base_csv_name = f"PREDICT/{model_data['model']}_{suffix_title}"
    base_csv_path = f"{Path(os.getcwd()).joinpath(base_csv_name)}"
    path_n_suffix = f'{base_csv_path}'
    _ = df_results.to_csv(f'{base_csv_path}.csv', index = None, header=True)
    
    # also save results for performance of individual folds (useful for t-tests and Wilcoxon tests between the folds)
    error1, error2, error3 = get_error_labels(model_data['type'])

    # df_folds = pd.DataFrame()
    # df_folds['Fold'] = [f'{i+1}' for i in range(len(Xy_data['idx_valid']))]
    # df_folds['idx_valid'] = Xy_data['idx_valid']
    # df_folds[f'{error1}_valid'] = Xy_data[f'fold_{error1}_valid']
    # df_folds[f'{error2}_valid'] = Xy_data[f'fold_{error2}_valid']
    # df_folds[f'{error3}_valid'] = Xy_data[f'fold_{error3}_valid']
    # df_folds[f'{error1}_test'] = Xy_data[f'fold_{error1}_test']
    # df_folds[f'{error2}_test'] = Xy_data[f'fold_{error2}_test']
    # df_folds[f'{error3}_test'] = Xy_data[f'fold_{error3}_test']

    # path_folds = f'{base_csv_path}_CV_folds'
    # _ = df_folds.to_csv(f'{path_folds}.csv', index = None, header=True)

    # prints
    print_preds = f'   o  Saving CSV databases with predictions and their SD in:'   
    print_preds += f'\n      -  Predicted results of starting dataset: {base_csv_name}.csv'

    if self.args.csv_test != '':            
        # saves prediction for external test in --csv_test
        Xy_external = pd.DataFrame(Xy_data['names_external'])

        for col in Xy_data['X_external']:
            Xy_external[col] = Xy_data['X_external'][col].tolist()

        # Reconvert external set labels if needed (external targets may be legitimately
        # unknown/NaN, e.g. when predicting on new compounds, so those are left as-is)
        if 'y_external' in Xy_data:
            y_external_values = Xy_data['y_external'].tolist()
            if reconvert_labels:
                y_external_values = [class_mapping_reverse[int(y)] if pd.notna(y) else y for y in y_external_values]
            Xy_external[model_data['y']] = y_external_values
        
        y_pred_external_values = Xy_data['y_pred_external']
        if reconvert_labels:
            y_pred_external_values = [class_mapping_reverse[int(y)] for y in y_pred_external_values]
        
        Xy_external[f"{model_data['y']}_pred"] = y_pred_external_values
        Xy_external[f"{model_data['y']}_pred_sd"] = Xy_data['y_pred_external_sd']
        if include_conformal:
            Xy_external[f"{model_data['y']}_pred_conformal_hw"] = [hw_scalar] * len(Xy_external)

        path_external = Path(os.getcwd()).joinpath('PREDICT/csv_test/')
        Path(path_external).mkdir(exist_ok=True, parents=True)
        csv_name_external = f'{os.path.basename(self.args.csv_test).split(".csv")[0]}_{model_data["model"]}_{suffix_title}.csv'
        name_external = f"{path_external}/{csv_name_external}"

        _ = Xy_external.to_csv(name_external, index = None, header=True)
        print_preds += f'\n      -  External set with predicted results: PREDICT/csv_test/{csv_name_external}'

    self.args.log.write(print_preds)

    # store the names of the datapoints
    name_points = {}
    if model_data['names'] != '':
        if model_data['names'].lower() in Xy_train: # accounts for upper/lowercase mismatches
            model_data['names'] = model_data['names'].lower()
        if model_data['names'].upper() in Xy_train:
            model_data['names'] = model_data['names'].upper()
        if model_data['names'] in Xy_train:
            name_points['train'] = df_results[model_data['names']][df_results.Set == 'CV']
            name_points['test'] = df_results[model_data['names']][df_results.Set == 'Test']

    return path_n_suffix, name_points, Xy_data


def print_predict(self,Xy_data,model_data,suffix_title):
    '''
    Prints results of the predictions for all the sets
    '''

    print_results = f"\n   o  Summary of results {model_data['model']}_{suffix_title}:"
    set_print = 'CV (train+valid.):Test'

    # get number of points and proportions
    n_train = len(Xy_data['y_train'])
    n_test = len(Xy_data['y_test'])
    n_points = f'{n_train}:{n_test}'
    print_results += f"\n      -  Points {set_print} = {n_points}"

    total_points = n_train + n_test
    prop_train = round(n_train*100/total_points)
    prop_test = round(n_test*100/total_points)
    prop_print = f'{prop_train}:{prop_test}'
    print_results += f"\n      -  Proportion {set_print} = {prop_print}"

    n_descps = len(Xy_data['X_train'].keys())
    print_results += f"\n      -  Number of descriptors = {n_descps}"
    print_results += f"\n      -  Proportion (train+valid.) points:descriptors = {n_train}:{n_descps}"

    # print results and save dat file
    CV_type = f"{model_data['repeat_kfolds']}x {model_data['kfold']}-fold CV"
    if model_data['type'].lower() == 'reg':
        print_results += f"\n      -  {CV_type} : R2 = {Xy_data['r2_train']:.2}, MAE = {Xy_data['mae_train']:.2}, RMSE = {Xy_data['rmse_train']:.2}"
        print_results += f"\n      -  Test : R2 = {Xy_data['r2_test']:.2}, MAE = {Xy_data['mae_test']:.2}, RMSE = {Xy_data['rmse_test']:.2}"
        print_results += f"\n      -  Average SD in test set = {np.mean(Xy_data['y_pred_test_sd']):.2}"
        print_results += f"\n      -  y range of dataset (train+valid.) = {float(Xy_data['pred_min']):.2} to {float(Xy_data['pred_max']):.2}, total {float(Xy_data['pred_range']):.2}"
        if 'r2_train_infold' in Xy_data:
            print_results += f"\n      -  Train fit (in-fold) : R2 = {Xy_data['r2_train_infold']:.2}, MAE = {Xy_data['mae_train_infold']:.2}, RMSE = {Xy_data['rmse_train_infold']:.2}"

        # Interpolation item 6, facets (b)/(c): stability of the out-of-fold CV predictions
        # themselves (not just the test set - facet (a) above), across the repeated-CV
        # repetitions. (b) average per-point SD of the out-of-fold predictions in
        # train+validation; (c) how much the AGGREGATE RMSE itself varies from repeat to
        # repeat (coefficient of variation = SD/mean of the 10 per-repeat RMSEs) - a
        # complementary, dataset-wide view of the same "how much does this depend on which
        # random split we happened to get" question that (a)/(b) ask per-point
        print_results += f"\n      -  Average SD in train+validation (out-of-fold) = {np.mean(Xy_data['y_pred_train_sd']):.2}"
        if 'y_pred_train_all' in Xy_data:
            y_pred_train_all = np.array(Xy_data['y_pred_train_all'])
            y_train_actual = np.array(Xy_data['y_train'])
            repeat_rmses = np.sqrt(np.mean((y_train_actual[:,None]-y_pred_train_all)**2,axis=0))
            if len(repeat_rmses) > 1 and np.mean(repeat_rmses) > 0:
                rmse_cv_pct = 100*np.std(repeat_rmses,ddof=1)/np.mean(repeat_rmses)
            else:
                rmse_cv_pct = 0
            print_results += f"\n      -  RMSE coefficient of variation (10 repeats) = {rmse_cv_pct:.1f}"

        if 'r2_external' in Xy_data:
            print_results += f"\n      -  External test : R2 = {Xy_data['r2_external']:.2}, MAE = {Xy_data['mae_external']:.2}, RMSE = {Xy_data['rmse_external']:.2}"

    elif model_data['type'].lower() == 'clas':
        print_results += f"\n      -  {CV_type} : Accur. = {Xy_data['acc_train']:.2}, F1 score = {Xy_data['f1_train']:.2}, MCC = {Xy_data['mcc_train']:.2}"
        if 'y_pred_test' in Xy_data and not Xy_data['y_test'].isnull().values.any() and len(Xy_data['y_test']) > 0:
            print_results += f"\n      -  Test : Accur. = {Xy_data['acc_test']:.2}, F1 score = {Xy_data['f1_test']:.2}, MCC = {Xy_data['mcc_test']:.2}"

        # Interpolation item 6 for classification: the same "how much does this depend on
        # the random split" question as the regression facets, adapted for a discrete label.
        # (a)/(b) use the repeat-to-repeat agreement rate with the already-computed
        # majority-voted class instead of a numeric SD; (c) uses the coefficient of
        # variation of the per-repeat MCC instead of the per-repeat RMSE
        if 'y_pred_test_all' in Xy_data and len(Xy_data['y_pred_test_all']) > 0:
            test_agree = [np.mean(np.array(reps) == vote) for reps,vote in zip(Xy_data['y_pred_test_all'],Xy_data['y_pred_test'])]
            print_results += f"\n      -  Average agreement in test set (10 repeats) = {100*np.mean(test_agree):.1f}"

        if 'y_pred_train_all' in Xy_data:
            train_agree = [np.mean(np.array(reps) == vote) for reps,vote in zip(Xy_data['y_pred_train_all'],Xy_data['y_pred_train'])]
            print_results += f"\n      -  Average agreement in train+validation (out-of-fold) (10 repeats) = {100*np.mean(train_agree):.1f}"

            y_pred_train_all = np.array(Xy_data['y_pred_train_all'])
            y_train_actual = np.array(Xy_data['y_train'])
            if y_pred_train_all.shape[1] > 1:
                repeat_mccs = [mcc_scorer_clf(y_train_actual,y_pred_train_all[:,i]) for i in range(y_pred_train_all.shape[1])]
                if np.mean(repeat_mccs) != 0:
                    mcc_cv_pct = 100*np.std(repeat_mccs,ddof=1)/abs(np.mean(repeat_mccs))
                else:
                    mcc_cv_pct = 0
                print_results += f"\n      -  MCC coefficient of variation (10 repeats) = {mcc_cv_pct:.1f}"

        if 'acc_external' in Xy_data:
            print_results += f"\n      -  External test : Accur. = {Xy_data['acc_external']:.2}, F1 score = {Xy_data['f1_external']:.2}, MCC = {Xy_data['mcc_external']:.2}"

    self.args.log.write(print_results)


def pearson_map_predict(self,Xy_data,params_dir,suffix_title):
    '''
    Plots the Pearson map and analyzes correlation of descriptors.
    '''

    X_combined = pd.concat([Xy_data['X_train'], Xy_data['X_test']], axis=0, ignore_index=True)
    corr_matrix = pearson_map(self,X_combined,'predict',params_dir=params_dir,suffix_title=suffix_title)

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
            print_corr += f"\n      x  WARNING! High correlations observed (up to r = {max_r:.2} or R2 = {max_r*max_r:.2}, for {max_descp_1} and {max_descp_2})"
        elif abs_max_r > 0.71:
            print_corr += f"\n      x  WARNING! Noticeable correlations observed (up to r = {max_r:.2} or R2 = {max_r*max_r:.2}, for {max_descp_1} and {max_descp_2})"

    self.args.log.write(print_corr)
