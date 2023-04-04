#####################################################.
#     This file stores functions from PREDICT       #
#####################################################.

import os
import sys
import ast
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import shap
from robert.curate import curate
from robert.utils import (
    load_model,
    standardize,
    load_dfs,
    load_database,
    )


def load_test(self, Xy_data, params_df):
    ''''
    Loads Xy data of the test set
    '''

    Xy_test_df = load_database(self, self.args.csv_test, "predict")
    descs_model = ast.literal_eval(params_df['X_descriptors'][0])
    try:
        Xy_data['X_test'] = Xy_test_df[descs_model]
    except KeyError:
        # this might fail if the initial categorical variables have not been transformed
        try:
            self.args.log.write(f"\n   x  There are missing descriptors in the test set! Looking for categorical variables converted from CURATE")
            Xy_test_df = curate.categorical_transform(self,Xy_test_df,'predict')
            Xy_data['X_test'] = Xy_test_df[descs_model]
            self.args.log.write(f"   o  The missing descriptors were successfully created")
        except KeyError:
            self.args.log.write(f"   x  There are still missing descriptors in the test set! The following descriptors are needed: {descs_model}")
            self.args.log.finalize()
            sys.exit()

    if params_df['y'][0] in Xy_test_df:
        Xy_data['y_test'] = Xy_test_df[params_df['y'][0]]
    _, Xy_data['X_test_scaled'] = standardize(Xy_data['X_train'],Xy_data['X_test'])

    return Xy_data


def plot_predictions(self, params_dict, Xy_data, path_n_suffix):
    '''
    Plot graphs of predicted vs actual values for train, validation and test sets
    '''

    if params_dict['mode'] == 'clas':
        # load the ML model
        loaded_model = load_model(params_dict)
        # Fit the model with the training set
        loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])  

    set_types = ['train','valid']
    if 'y_test' in Xy_data:
        set_types.append('test')
    
    self.args.log.write(f"\n   o  Saving graphs and CSV databases in {Path(os.getcwd()).joinpath('PREDICT')}:")
    if params_dict['mode'] == 'reg':
        _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix)

    elif params_dict['mode'] == 'clas':
        for set_type in set_types:
            _ = graph_clas(self,loaded_model,Xy_data,params_dict,set_type,path_n_suffix)


def graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix):
    '''
    Plot regression graphs of predicted vs actual values for train, validation and test sets
    '''

    color_train, color_validation, color_test = 'b', 'orange', 'r'
    dot_size = 50
    alpha = 1 # from 0 (transparent) to 1 (opaque)

    # Create graph
    sb.set(style="ticks")
    _, ax = plt.subplots(figsize=(7.45,6))

    # Set styling preferences
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Title of the axis
    plt.ylabel(f'Predicted {params_dict["y"]}', fontsize=14)
    plt.xlabel(f'{params_dict["y"]}', fontsize=14)
    
    title_graph = f'Predictions_train_valid'
    if 'test' in set_types:
        title_graph += '_test'
    plt.text(0.5, 1.08, f'{title_graph} of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
        fontsize=14, fontweight='bold', transform = ax.transAxes)

    # Plot the data
    _ = ax.scatter(Xy_data["y_train"], Xy_data["y_pred_train"],
                c = color_train, s = dot_size, edgecolor = 'k', linewidths = 0.8, alpha = alpha, zorder=2)

    _ = ax.scatter(Xy_data["y_valid"], Xy_data["y_pred_valid"],
                c = color_validation, s = dot_size, edgecolor = 'k', linewidths = 0.8, alpha = alpha, zorder=2)
    if 'y_test' in Xy_data:
        _ = ax.scatter(Xy_data["y_test"], Xy_data["y_pred_test"],
                    c = color_test, s = dot_size, edgecolor = 'k', linewidths = 0.8, alpha = alpha, zorder=2)

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
            fancybox=True, shadow=True, ncol=5, labels=set_types, fontsize=14)

    # Add the regression line with a confidence interval based on the training sets
    Xy_data_df = pd.DataFrame()
    Xy_data_df["y_train"] = Xy_data["y_train"]
    Xy_data_df["y_pred_train"] = Xy_data["y_pred_train"]
    _ = sb.regplot(x="y_train", y="y_pred_train", data=Xy_data_df, scatter=False, color=".1", 
                    truncate = True, ax=ax, seed=self.args.seed)

    # Add gridlines
    ax.grid(linestyle='--', linewidth=1)

    # set limits
    min_value_graph, max_value_graph = set_lim_reg(Xy_data,set_types)

    plt.xlim(min_value_graph, max_value_graph)
    plt.ylim(min_value_graph, max_value_graph)

    reg_plot_file = f'{os.path.dirname(path_n_suffix)}/Results_{os.path.basename(path_n_suffix)}.png'
    plt.savefig(f'{reg_plot_file}', dpi=300, bbox_inches='tight')
    self.args.log.write(f"      -  Graph in: {reg_plot_file}")
    plt.clf()


def set_lim_reg(Xy_data,set_types):
    '''
    Set axis limits for regression plots
    '''
    
    size_space = 0.1*abs(min(Xy_data["y_train"])-max(Xy_data["y_train"]))
    if 'test' in set_types:
        if min(Xy_data["y_train"]) < min(Xy_data["y_valid"]) and min(Xy_data["y_train"]) < min(Xy_data["y_test"]):
            min_value_graph = min(Xy_data["y_train"])-size_space
        elif min(Xy_data["y_valid"]) < min(Xy_data["y_train"]) and min(Xy_data["y_valid"]) < min(Xy_data["y_test"]):
            min_value_graph = min(Xy_data["y_valid"])-size_space
        else:
            min_value_graph = min(Xy_data["y_test"])-size_space
            
        if max(Xy_data["y_train"]) > max(Xy_data["y_valid"]) and max(Xy_data["y_train"]) > max(Xy_data["y_test"]):
            max_value_graph = max(Xy_data["y_train"])+size_space
        elif max(Xy_data["y_valid"]) > max(Xy_data["y_train"]) and max(Xy_data["y_valid"]) > max(Xy_data["y_test"]):
            max_value_graph = max(Xy_data["y_valid"])+size_space
        else:
            max_value_graph = max(Xy_data["y_test"])+size_space
    else:
        if min(Xy_data["y_train"]) < min(Xy_data["y_valid"]):
            min_value_graph = min(Xy_data["y_train"])-size_space
        else:
            min_value_graph = min(Xy_data["y_valid"])-size_space
            
        if max(Xy_data["y_train"]) > max(Xy_data["y_valid"]):
            max_value_graph = max(Xy_data["y_train"])+size_space
        else:
            max_value_graph = max(Xy_data["y_valid"])+size_space
    
    return min_value_graph, max_value_graph


def graph_clas(self,loaded_model,Xy_data,params_dict,set_type,path_n_suffix):
    '''
    Plot a confusion matrix with the prediction vs actual values
    '''
    
    matrix = ConfusionMatrixDisplay.from_estimator(loaded_model, Xy_data[f'X_{set_type}_scaled'], Xy_data[f'y_{set_type}'], normalize=("Normalized confusion matrix", "true"), cmap='Blues') 
    matrix.ax_.set_title(f'Confusion Matrix for the {set_type} set of {os.path.basename(path_n_suffix)}', fontsize=14)
    plt.xlabel(f'Predicted {params_dict["y"]}', fontsize=14)
    plt.ylabel(f'{params_dict["y"]}', fontsize=14)
    plt.gcf().axes[0].tick_params(fontsize=14)
    plt.gcf().axes[1].tick_params(fontsize=14)
    clas_plot_file = f'{os.path.dirname(path_n_suffix)}/Results_{os.path.basename(path_n_suffix)}.png'

    plt.savefig(f'{clas_plot_file}', dpi=300, bbox_inches='tight')
    self.args.log.write(f"      -  Graph in: {clas_plot_file}")
    plt.clf()


def save_predictions(self,Xy_data,model_dir):
    '''
    Saves CSV files with the different sets and their predicted results
    '''
    
    Xy_orig_df, Xy_path, params_df, _, _, suffix_title = load_dfs(self,model_dir,'no_print')
    base_csv_name = '_'.join(os.path.basename(Xy_path).split('_')[0:2])
    base_csv_name = f'PREDICT/{base_csv_name}'
    base_csv_path = f"{Path(os.getcwd()).joinpath(base_csv_name)}"
    Xy_orig_train = Xy_orig_df[Xy_orig_df.Set == 'Training']
    Xy_orig_train[f'{params_df["y"][0]}_pred'] = Xy_data['y_pred_train']
    train_path = f'{base_csv_path}_train_{suffix_title}.csv'
    _ = Xy_orig_train.to_csv(train_path, index = None, header=True)
    print_preds = f'      -  Train set with predicted results: {os.path.basename(train_path)}'
    Xy_orig_valid = Xy_orig_df[Xy_orig_df.Set == 'Validation']
    Xy_orig_valid[f'{params_df["y"][0]}_pred'] = Xy_data['y_pred_valid']
    valid_path = f'{base_csv_path}_valid_{suffix_title}.csv'
    _ = Xy_orig_valid.to_csv(valid_path, index = None, header=True)
    print_preds += f'\n      -  Validation set with predicted results: {os.path.basename(valid_path)}'
    if self.args.csv_test != '':
        Xy_orig_test = load_database(self, self.args.csv_test, "no_print")
        Xy_orig_test[f'{params_df["y"][0]}_pred'] = Xy_data['y_pred_test']
        test_path = f'{base_csv_path}_test_{suffix_title}.csv'
        _ = Xy_orig_test.to_csv(test_path, index = None, header=True)
        print_preds += f'\n      -  Test set with predicted results: {os.path.basename(test_path)}'
    self.args.log.write(print_preds)

    path_n_suffix = f'{base_csv_path}_{suffix_title}'
    return path_n_suffix


def print_predict(self,Xy_data,params_dict,path_n_suffix):
    '''
    Prints results of the predictions for all the sets
    '''
    
    dat_file = f'{os.path.dirname(path_n_suffix)}/Results_{os.path.basename(path_n_suffix)}.dat'
    print_results = f"\n   o  Results saved in {dat_file}:"
    set_print = 'Train:Validation'

    # get number of points and proportions
    n_train = len(Xy_data['X_train'])
    n_valid = len(Xy_data['X_valid'])
    n_test = 0
    n_points = f'{n_train}:{n_valid}'
    if 'X_test' in Xy_data:
        set_print += ':Test'
        n_test = len(Xy_data['X_test'])
        n_points += f':{n_test}'
    total_points = n_train + n_valid + n_test
    print_results += f"\n      -  Points {set_print} = {n_points}"

    prop_train = round(n_train*100/total_points)
    prop_valid = round(n_valid*100/total_points)
    prop_test = round(n_test*100/total_points)
    prop_print = f'{prop_train}:{prop_valid}'
    if 'X_test' in Xy_data:
        prop_print += f':{prop_test}'
    print_results += f"\n      -  Proportion {set_print} = {prop_print}"

    # print results and save dat file
    if params_dict['mode'] == 'reg':
        print_results += f"\n      -  Train : R2 = {Xy_data['r2_train']:.2}, MAE = {Xy_data['mae_train']:.2}, RMSE = {Xy_data['rmse_train']:.2}"
        print_results += f"\n      -  Validation : R2 = {Xy_data['r2_valid']:.2}, MAE = {Xy_data['mae_valid']:.2}, RMSE = {Xy_data['rmse_valid']:.2}"
        if 'y_test' in Xy_data:
            print_results += f"\n      -  Test : R2 = {Xy_data['r2_test']:.2}, MAE = {Xy_data['mae_test']:.2}, RMSE = {Xy_data['rmse_test']:.2}"

    elif params_dict['mode'] == 'clas':
        print_results += f"\n      -  Train : Accuracy = {Xy_data['acc_train']:.2}, F1 score = {Xy_data['f1_train']:.2}, MCC = {Xy_data['mcc_train']:.2}"
        print_results += f"\n      -  Validation : Accuracy = {Xy_data['acc_valid']:.2}, F1 score = {Xy_data['f1_valid']:.2}, MCC = {Xy_data['mcc_valid']:.2}"
        if 'y_test' in Xy_data:
            print_results += f"\n      -  Test : Accuracy = {Xy_data['acc_test']:.2}, F1 score = {Xy_data['f1_test']:.2}, MCC = {Xy_data['mcc_test']:.2}"

    self.args.log.write(print_results)
    dat_results = open(dat_file, "w")
    dat_results.write(print_results)
    dat_results.close()


def shap_analysis(self,Xy_data,params_dict,path_n_suffix):
    '''
    Plots and prints the results of the SHAP analysis
    '''

    shap_plot_file = f'{os.path.dirname(path_n_suffix)}/SHAP_{os.path.basename(path_n_suffix)}.png'

    # load and fit the ML model
    loaded_model = load_model(params_dict)
    loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train']) 

    # run the SHAP analysis and save the plot
    explainer = shap.Explainer(loaded_model.predict, Xy_data['X_valid_scaled'], seed=self.args.seed)
    shap_values = explainer(Xy_data['X_valid_scaled'])
    # explainer = shap.TreeExplainer(loaded_model) # in case the standard version doesn't work
    _ = shap.summary_plot(shap_values, Xy_data['X_valid_scaled'], max_display=self.args.shap_show,show=False)
    
    plt.title(f'SHAP analysis of {os.path.basename(path_n_suffix)}',fontweight="bold")
    
    plt.savefig(f'{shap_plot_file}', dpi=300, bbox_inches='tight')
    plt.clf()
    print_shap = f"\n   o  SHAP plot saved in {shap_plot_file}"

    # collect SHAP values and print
    shap_results_file = f'{os.path.dirname(path_n_suffix)}/SHAP_{os.path.basename(path_n_suffix)}.dat'
    desc_list, min_list, max_list = [],[],[]
    for i,desc in enumerate(Xy_data['X_train_scaled']):
        desc_list.append(desc)
        val_list_indiv= []
        for _,val in enumerate(shap_values.values):
            val_list_indiv.append(val[i])
        min_indiv = min(val_list_indiv)
        max_indiv = max(val_list_indiv)
        min_list.append(min_indiv)
        max_list.append(max_indiv)
    
    if max(max_list, key=abs) > max(min_list, key=abs):
        max_list, min_list, desc_list = (list(t) for t in zip(*sorted(zip(max_list, min_list, desc_list), reverse=True)))
    else:
        min_list, max_list, desc_list = (list(t) for t in zip(*sorted(zip(min_list, max_list, desc_list), reverse=False)))

    print_shap += f"\n   o  SHAP values saved in {shap_results_file}:"
    for i,desc in enumerate(desc_list):
        print_shap += f"\n      -  {desc} = min: {min_list[i]:.2}, max: {max_list[i]:.2}"

    self.args.log.write(print_shap)
    dat_results = open(shap_results_file, "w")
    dat_results.write(print_shap)
    dat_results.close()


def PFI_plot(self,Xy_data,params_dict,path_n_suffix):
    '''
    Plots and prints the results of the PFI analysis
    '''

    pfi_plot_file = f'{os.path.dirname(path_n_suffix)}/PFI_{os.path.basename(path_n_suffix)}.png'

    # load and fit the ML model
    loaded_model = load_model(params_dict)
    loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train']) 

    score_model = loaded_model.score(Xy_data['X_valid_scaled'], Xy_data['y_valid'])
    perm_importance = permutation_importance(loaded_model, Xy_data['X_valid_scaled'], Xy_data['y_valid'], n_repeats=self.args.pfi_epochs, random_state=self.args.seed)

    # sort descriptors and results from PFI
    desc_list, PFI_values, PFI_sd = [],[],[]
    for i,desc in enumerate(Xy_data['X_train_scaled']):
        desc_list.append(desc)
        PFI_values.append(perm_importance.importances_mean[i])
        PFI_sd.append(perm_importance.importances_std[i])
        if len(desc_list) == self.args.pfi_show:
            break
  
    PFI_values, PFI_sd, desc_list = (list(t) for t in zip(*sorted(zip(PFI_values, PFI_sd, desc_list), reverse=False)))

    # plot and print results
    fig, ax = plt.subplots(figsize=(7.45,6))
    y_ticks = np.arange(0, len(desc_list))
    ax.barh(desc_list, PFI_values)
    ax.set_yticks(y_ticks,labels=desc_list,fontsize=14)
    plt.text(0.5, 1.08, f'Permutation feature importances (PFIs) of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
        fontsize=14, fontweight='bold', transform = ax.transAxes)
    fig.tight_layout()
    ax.set(ylabel=None, xlabel='PFI')

    plt.savefig(f'{pfi_plot_file}', dpi=300, bbox_inches='tight')
    plt.clf()
    print_PFI = f"\n   o  PFI plot saved in {pfi_plot_file}"

    pfi_results_file = f'{os.path.dirname(path_n_suffix)}/PFI_{os.path.basename(path_n_suffix)}.dat'
    print_PFI += f"\n   o  PFI values saved in {pfi_results_file}:"
    if params_dict['mode'] == 'reg':
        print_PFI += f'\n      Original score (from model.score, R2) = {score_model:.2}'
    elif params_dict['mode'] == 'clas':
        print_PFI += f'\n      Original score (from model.score, accuracy) = {score_model:.2}'
    # shown from higher to lower values
    PFI_values, PFI_sd, desc_list = (list(t) for t in zip(*sorted(zip(PFI_values, PFI_sd, desc_list), reverse=True)))
    for i,desc in enumerate(desc_list):
        print_PFI += f"\n      -  {desc} = {PFI_values[i]:.2} +- {PFI_sd[i]:.2}"
    
    self.args.log.write(print_PFI)
    dat_results = open(pfi_results_file, "w")
    dat_results.write(print_PFI)
    dat_results.close()


def outlier_plot(self,Xy_data,params_dict,path_n_suffix):
    '''
    Plots and prints the results of the outlier analysis
    '''

    # calculate absolute errors between predicted y and actual values
    outliers_train = [abs(x-y) for x,y in zip(Xy_data['y_train'],Xy_data['y_pred_train'])]
    outliers_valid = [abs(x-y) for x,y in zip(Xy_data['y_valid'],Xy_data['y_pred_valid'])]
    if 'y_test' in Xy_data:
        outliers_test = [abs(x-y) for x,y in zip(Xy_data['y_test'],Xy_data['y_pred_test'])]

    # the errors are scaled using standard deviation units. When the absolute
    # error is larger than the t-value, the point is considered an outlier
    outliers_mean = np.mean(outliers_train)
    outliers_sd = np.std(outliers_train)
    # print(outliers_mean,outliers_sd)

    outliers_train_scaled = (outliers_train-outliers_mean)/outliers_sd
    outliers_valid_scaled = (outliers_valid-outliers_mean)/outliers_sd
    outliers_test_scaled = (outliers_test-outliers_mean)/outliers_sd

    # for i,val in enumerate(outliers_train):
    #     error_normal.append((error_abs[i]-Mean)/SD)
    #     if np.absolute(error_normal[i]) > t_value:
    #         error_outlier.append(error_normal[i])
    #         outlier_names.append(Ir_cat_names[i])


    #     # plot data in SD units
    #       fig, ax = plt.subplots(figsize=(7.45,6))
    #     plt.text(0.5, 1.08, f'Outlier analysis of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
    #     fontsize=14, fontweight='bold', transform = ax.transAxes)

    #     plt.grid(linestyle='--', linewidth=1)
    #     Plot_outliers = {'error_outlier': error_outlier}
    #     Plot_no_outliers = {'error_no_outlier': error_no_outlier}
    #     df_outliers = pd.DataFrame.from_dict(Plot_outliers)
    #     df_no_outliers = pd.DataFrame.from_dict(Plot_no_outliers)
    #     plt.scatter(df_no_outliers["error_no_outlier"], df_no_outliers["error_no_outlier"],
    #                  c='b', edgecolors='none', alpha=0.4,)  # Include border to the points
    #     plt.scatter(df_outliers["error_outlier"], df_outliers["error_outlier"],
    #                  c='r', edgecolors='none', alpha=0.4,)  # Include border to the points
# copy scatter and style (ticks, labels, etc) from above
    #     # Set styling preferences
    #       ADD TITLES!
    #     sb.set(style="ticks")
    #     plt.xlabel('SD of the errors',fontsize=14)
    #     plt.xticks(fontsize=14)
    #     plt.ylabel('SD of the errors',fontsize=14)
    #     plt.yticks(fontsize=14)
        
    #     # Set plot limits
    #     axis_limit = math.ceil(np.absolute(error_normal).max() + 0.5)
    #     plt.ylim(-(axis_limit), (axis_limit))
    #     plt.xlim(-(axis_limit), (axis_limit))

    #     # plot rectangles in corners
    #     diff_tvalue = axis_limit - t_value
    #     Rectangle_top = mpatches.Rectangle(xy=(axis_limit, axis_limit), width=-diff_tvalue, height=-diff_tvalue, facecolor='grey', alpha=0.3)
    #     Rectangle_bottom = mpatches.Rectangle(xy=(-(axis_limit), -(axis_limit)), width=diff_tvalue, height=diff_tvalue, facecolor='grey', alpha=0.3)
    #     ax.add_patch(Rectangle_top)
    #     ax.add_patch(Rectangle_bottom)
         
        
    #     # Calculate % of outliers discarded

    #     plt.savefig('DFT_vs_Experimental_Outliers_'+folder+'.png', dpi=400, bbox_inches='tight')
        
    #     plt.show()
        # print
        # Train: XX of XX
            #    - XX
            #    - XX