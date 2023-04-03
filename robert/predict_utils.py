#####################################################.
#     This file stores functions from PREDICT       #
#####################################################.

import os
import sys
import ast
from pathlib import Path
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from robert.curate import curate
from robert.utils import (
    load_model,
    standardize,
    load_database,
    standardize,
    load_model
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


def plot_predictions(self, params_dict, Xy_data, model_dir):
    '''
    Plot graphs of predicted vs actual values for train, validation and test sets
    '''

    if params_dict['mode'] == 'clas':
        # set the parameters for each ML model of the hyperopt optimization
        loaded_model = load_model(params_dict)
        # Fit the model with the training set
        loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])  

    suffix = '(with no PFI filter)'
    label = 'No_PFI'
    if 'PFI' in model_dir and 'No_PFI' not in model_dir:
        suffix = '(with PFI filter)'
        label = 'PFI'

    set_types = ['train','valid']
    if 'y_test' in Xy_data:
        set_types.append('test')
    
    if params_dict['mode'] == 'reg':
        _ = graph_reg(self,Xy_data,params_dict,set_types,suffix,label)

    elif params_dict['mode'] == 'clas':
        for set_type in set_types:
            _ = graph_clas(self,loaded_model,Xy_data,params_dict,set_type,suffix,label)


def graph_reg(self,Xy_data,params_dict,set_types,suffix,label):
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
    plt.text(0.5, 1.08, f'{title_graph} {suffix}', horizontalalignment='center',
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
                    truncate = True, ax=ax)

    # Add gridlines
    ax.grid(linestyle='--', linewidth=1)

    # set limits
    min_value_graph, max_value_graph = set_lim_reg(Xy_data,set_types)

    plt.xlim(min_value_graph, max_value_graph)
    plt.ylim(min_value_graph, max_value_graph)

    path_raw = f'PREDICT/{title_graph}_{label}'
    full_path = Path(os.getcwd()).joinpath(path_raw)
    self.args.log.write(f"\n   o  Saving graph in: {full_path}.png")
    plt.savefig(f'{full_path}.png', dpi=300, bbox_inches='tight')


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


def graph_clas(self,loaded_model,Xy_data,params_dict,set_type,suffix,label):
    '''
    Plot a confusion matrix with the prediction vs actual values
    '''
    
    matrix = ConfusionMatrixDisplay.from_estimator(loaded_model, Xy_data[f'X_{set_type}_scaled'], Xy_data[f'y_{set_type}'], normalize=("Normalized confusion matrix", "true"), cmap='Blues') 
    matrix.ax_.set_title(f'Confusion Matrix for the {set_type} set {suffix}', fontsize=14)
    plt.xlabel(f'Predicted {params_dict["y"]}', fontsize=14)
    plt.ylabel(f'{params_dict["y"]}', fontsize=14)
    plt.gcf().axes[0].tick_params(fontsize=14)
    plt.gcf().axes[1].tick_params(fontsize=14)
    path_raw = f'PREDICT/Confusion_matrix_{set_type}_{label}'
    full_path = Path(os.getcwd()).joinpath(path_raw)
    self.args.log.write(f"\n   o  Saving graph in: {full_path}.png")
    plt.savefig(f'{full_path}.png', dpi=300, bbox_inches='tight')


