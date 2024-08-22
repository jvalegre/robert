"""
Parameters
----------

    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    params_dir : str, default=''
        Folder containing the database and parameters of the ML model to analyze.
    thres_test : float, default=0.25,
        Threshold used to determine if a test pasess. It is determined in % units of diference between
        the RMSE (MCC in classificators) of the model and the test (i.e., 0.25 = 25% difference with the 
        original value). Test passes if:
        1. y-mean, y-shuffle and one-hot encoding tests: the error is greater than 25% (from original MAE and RMSE for regressors)
        2. Cross-validation: the error is lower than 25%
    kfold : int, default='auto',
        Number of folds for the k-fold cross-validation. If 'auto', the program does 
        a LOOCV for databases with less than 50 points, and 5-fold CV for larger databases 

"""
#####################################################.
#        This file stores the VERIFY class          #
#           used for ML model analysis              #
#####################################################.

import os
import time
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, ArrowStyle
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sb
from statistics import mode
# for users with no intel architectures. This part has to be before the sklearn imports
try:
    from sklearnex import patch_sklearn
    patch_sklearn(verbose=False)
except (ModuleNotFoundError,ImportError):
    pass
from sklearn.model_selection import KFold
from robert.utils import (load_variables,
    load_db_n_params,
    load_model,
    pd_to_dict,
    load_n_predict,
    finish_print,
    get_prediction_results,
    print_pfi,
    get_graph_style
)
from robert.predict_utils import graph_reg,graph_clas


class verify:
    """
    Class containing all the functions from the VERIFY module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the VERIFY module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "verify")

        # if params_dir = '', the program performs the tests for the No_PFI and PFI folders
        if 'GENERATE/Best_model' in self.args.params_dir:
            params_dirs = [f'{self.args.params_dir}/No_PFI',f'{self.args.params_dir}/PFI']
        else:
            params_dirs = [self.args.params_dir]

        for params_dir in params_dirs:
            if os.path.exists(params_dir):

                _ = print_pfi(self,params_dir)

                # load database and ML model parameters, and add standardized descriptors
                Xy_data, params_df, params_path, suffix_title, _ = load_db_n_params(self,params_dir,"verify",True)
                
                # set the parameters for each ML model of the hyperopt optimization
                params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

                # this dictionary will keep the results of the tests
                verify_results = {'error_type': params_df['error_type'][0]}

                # get original score
                Xy_orig = Xy_data.copy()
                Xy_orig = load_n_predict(self, params_dict, Xy_orig)  
                verify_results['original_score_train'] = Xy_orig[f'{verify_results["error_type"]}_train']
                verify_results['original_score_valid'] = Xy_orig[f'{verify_results["error_type"]}_valid']

                # calculate cross-validation
                verify_results,type_cv,path_n_suffix = self.cv_test(verify_results,Xy_data,params_dict,params_path,suffix_title)

                # calculate scores for the y-mean test
                verify_results = self.ymean_test(verify_results,Xy_data,params_dict)

                # load and ML model parameters again (to avoid weird memory issues on Windows, for some 
                # reason the Xy_data dataframe changes when changing X descriptors in copy() objects)
                Xy_data, params_df, params_path, suffix_title, _ = load_db_n_params(self,params_dir,"verify",False)

                # calculate scores for the y-shuffle test
                verify_results = self.yshuffle_test(verify_results,Xy_data,params_dict)

                # load and ML model parameters again (to avoid weird memory issues on Windows, for some 
                # reason the Xy_data dataframe changes when changing X descriptors in copy() objects)
                Xy_data, params_df, params_path, suffix_title, _ = load_db_n_params(self,params_dir,"verify",False)

                # one-hot test (check that if a value isnt 0, the value assigned is 1)
                verify_results = self.onehot_test(verify_results,Xy_data,params_dict)

                # load and ML model parameters again (to avoid weird memory issues on Windows, for some 
                # reason the Xy_data dataframe changes when changing X descriptors in copy() objects)
                Xy_data, params_df, params_path, suffix_title, _ = load_db_n_params(self,params_dir,"verify",False)

                # analysis of results
                results_print,verify_results,verify_metrics = self.analyze_tests(verify_results,type_cv)

                # plot a bar graph with the results
                print_ver = self.plot_metrics(path_n_suffix,verify_metrics,verify_results)

                # print and save results
                _ = self.print_verify(results_print,verify_results,print_ver,path_n_suffix,verify_metrics,params_dict)

        _ = finish_print(self,start_time,'VERIFY')


    def cv_test(self,verify_results,Xy_data,params_dict,params_path,suffix_title):
        '''
        Performs a cross-validation on the training+validation dataset.
        '''      

        # set PATH names and plot
        base_csv_name = '_'.join(os.path.basename(params_path).replace('.csv','_').split('_')[0:2])
        base_csv_name = f'VERIFY/{base_csv_name}'
        base_csv_path = f"{Path(os.getcwd()).joinpath(base_csv_name)}"
        path_n_suffix = f'{base_csv_path}_{suffix_title}'

        # Fit the original model with the training set
        loaded_model = load_model(params_dict)
        loaded_model.fit(np.asarray(Xy_data['X_train_scaled']).tolist(), np.asarray(Xy_data['y_train']).tolist())
        data_cv = load_n_predict(self, params_dict, Xy_data)
        
        cv_y_list,cv_y_pred_list = [],[]
        data_cv = {}
        # combine training and validation for CV
        X_combined = pd.concat([Xy_data['X_train_scaled'],Xy_data['X_valid_scaled']], axis=0).reset_index(drop=True)
        y_combined = pd.concat([Xy_data['y_train'],Xy_data['y_valid']], axis=0).reset_index(drop=True)

        if self.args.kfold == 'auto':
            # LOOCV for relatively small datasets (less than 50 datapoints)
            if len(y_combined) < 50:
                type_cv = f'LOOCV'
                kf = KFold(n_splits=len(y_combined))
            # k-fold CV with the same training/validation proportion used for fitting the model, using 5 splits
            else:
                type_cv = '5-fold CV'
                kf = KFold(n_splits=5, shuffle=True, random_state=params_dict['seed'])
        # k-fold CV with the same training/validation proportion used for fitting the model, with k different data splits
        else:
            type_cv = f'{self.args.kfold}-fold CV'
            kf = KFold(n_splits=self.args.kfold, shuffle=True, random_state=params_dict['seed'])

        # separate into folds, then store the predictions
        for _, (train_index, valid_index) in enumerate(kf.split(X_combined)):
            XY_cv = {}
            XY_cv['X_train_scaled'] = X_combined.loc[train_index]
            XY_cv['y_train'] = y_combined.loc[train_index]
            XY_cv['X_valid_scaled'] = X_combined.loc[valid_index]
            XY_cv['y_valid'] = y_combined.loc[valid_index]
            data_cv = load_n_predict(self, params_dict, XY_cv)

            for y_cv,y_pred_cv in zip(data_cv['y_valid'],data_cv['y_pred_valid']):
                cv_y_list.append(y_cv)
                cv_y_pred_list.append(y_pred_cv)

        # calculate metrics and plot graphs
        graph_style = get_graph_style()
        Xy_data["y_cv_valid"] = cv_y_list
        Xy_data["y_pred_cv_valid"] = cv_y_pred_list

        if params_dict['type'].lower() == 'reg':
            verify_results['cv_r2'], verify_results['cv_mae'], verify_results['cv_rmse'] = get_prediction_results(params_dict,cv_y_list,cv_y_pred_list)
            set_types = [type_cv]
            _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style)

        elif params_dict['type'].lower() == 'clas':
            verify_results['cv_acc'], verify_results['cv_f1'], verify_results['cv_mcc'] = get_prediction_results(params_dict,cv_y_list,cv_y_pred_list)
            set_type = f'{type_cv} train+valid.'
            _ = graph_clas(self,Xy_data,params_dict,set_type,path_n_suffix)

        verify_results['cv_score'] = verify_results[f'cv_{verify_results["error_type"].lower()}']

        # save CSV with results
        Xy_cv_df = pd.DataFrame()
        Xy_cv_df[f'{params_dict["y"]}'] = Xy_data["y_cv_valid"]
        Xy_cv_df[f'{params_dict["y"]}_pred'] = Xy_data["y_pred_cv_valid"]
        csv_test_path = f'{os.path.dirname(path_n_suffix)}/CV_predictions_{os.path.basename(path_n_suffix)}.csv'
        _ = Xy_cv_df.to_csv(csv_test_path, index = None, header=True)

        return verify_results,type_cv,path_n_suffix


    def ymean_test(self,verify_results,Xy_data,params_dict):
        '''
        Calculate the accuracy of the model when using a flat line of predicted y values. For 
        regression, the mean of the y values is used. For classification, the value that is
        predicted more often is used.
        '''

        Xy_ymean = Xy_data.copy()   
        if params_dict['type'].lower() == 'reg':
            y_mean_array = np.ones(len(Xy_ymean['y_valid']))*(Xy_ymean['y_valid'].mean())
            Xy_ymean['r2_valid'], Xy_ymean['mae_valid'], Xy_ymean['rmse_valid'] = get_prediction_results(params_dict,Xy_ymean['y_valid'],y_mean_array)
        
        elif params_dict['type'].lower() == 'clas':
            y_mean_array = np.ones(len(Xy_ymean['y_valid']))*mode(Xy_ymean['y_valid'])
            Xy_ymean['acc_valid'], Xy_ymean['f1_valid'], Xy_ymean['mcc_valid'] = get_prediction_results(params_dict,Xy_ymean['y_valid'],y_mean_array)

        verify_results['y_mean'] = Xy_ymean[f'{verify_results["error_type"]}_valid']

        return verify_results


    def yshuffle_test(self,verify_results,Xy_data,params_dict):
        '''
        Calculate the accuracy of the model when the y values are randomly shuffled in the validation set
        For example, a y array of 1.3, 2.1, 4.0, 5.2 might become 2.1, 1.3, 5.2, 4.0.
        '''

        Xy_yshuffle = Xy_data.copy()
        Xy_yshuffle['y_valid'] = Xy_yshuffle['y_valid'].sample(frac=1,random_state=params_dict['seed'],axis=0)
        Xy_yshuffle = load_n_predict(self, params_dict, Xy_yshuffle)  
        verify_results['y_shuffle'] = Xy_yshuffle[f'{verify_results["error_type"]}_valid']

        return verify_results


    def onehot_test(self,verify_results,Xy_data,params_dict):
        '''
        Calculate the accuracy of the model when using one-hot models. All X values that are
        not 0 are considered to be 1 (NaN from missing values are converted to 0).
        '''

        Xy_onehot = Xy_data.copy()
        for desc in Xy_onehot['X_train']:
            new_vals = []
            for val in Xy_onehot['X_train'][desc]:
                if int(val) == 0:
                    new_vals.append(0)
                else:
                    new_vals.append(1)
            Xy_onehot['X_train_scaled'][desc] = new_vals

        for desc in Xy_onehot['X_valid']:
            new_vals = []
            for val in Xy_onehot['X_valid'][desc]:
                if int(val) == 0:
                    new_vals.append(0)
                else:
                    new_vals.append(1)
            Xy_onehot['X_valid_scaled'][desc] = new_vals

        Xy_onehot = load_n_predict(self, params_dict, Xy_onehot)  
        verify_results['onehot'] = Xy_onehot[f'{verify_results["error_type"]}_valid']

        return verify_results


    def analyze_tests(self,verify_results,type_cv):
        '''
        Function to check whether the tests pass and retrieve the corresponding colors:
        1. Blue for passing tests
        2. Red for failing tests
        '''

        blue_color = '#1f77b4'
        red_color = '#cd5c5c'
        colors = [None,None,None]
        results_print = [None,None,None]
        metrics = [None,None,None]
        thres_test = self.args.thres_test                

        # the threshold uses validation results to compare in the tests
        verify_results['higher_thres'] = (1+thres_test)*verify_results['original_score_valid']
        verify_results['lower_thres'] = (1-thres_test)*verify_results['original_score_valid']

        # determine whether the tests pass
        test_names = ['y_mean','y_shuffle','onehot']
        for i,test_ver in enumerate(test_names):
            metrics[i] = verify_results[test_ver]
            if verify_results['error_type'].lower() in ['mae','rmse']:
                if verify_results[test_ver] <= verify_results['higher_thres']:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, lower than thres.'
                else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, higher than thres.'

            else:
                if verify_results[test_ver] >= verify_results['lower_thres']:
                        colors[i] = red_color
                        results_print[i] = f'\n         x {test_ver}: FAILED, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, higher than thres.'
                else:
                        colors[i] = blue_color
                        results_print[i] = f'\n         o {test_ver}: PASSED, {verify_results["error_type"].upper()} = {round(verify_results[test_ver],2)}, lower than thres.'

        # store metrics and colors to represent in comparison graph, adding the metrics of the 
        # original model first
        test_names = ['Model'] + test_names
        colors = [blue_color] + colors
        metrics = [verify_results['original_score_valid']] + metrics
        verify_metrics = {'test_names': test_names,
                          'colors': colors,
                          'metrics': metrics,
                          'thres_test': thres_test,
                          'higher_thres': verify_results['higher_thres'],
                          'lower_thres': verify_results['lower_thres'],
                          'type_cv': type_cv
                          }        
        
        return results_print,verify_results,verify_metrics


    def plot_metrics(self,path_n_suffix,verify_metrics,verify_results):
        '''
        Creates a plot with the results of the flawed models in VERIFY
        '''

        sb.reset_defaults()
        sb.set(style="ticks")
        _, ax = plt.subplots(figsize=(7.45,6))

        # axis limits
        max_val = max(verify_metrics['metrics'])
        min_val = min(verify_metrics['metrics'])
        range_vals = np.abs(max_val - min_val)
        if verify_results['error_type'].lower() in ['mae','rmse']:
            max_lim = 1.2*max_val
            min_lim = 0
        else:
            max_lim = max_val + (0.2*range_vals)
            min_lim = min_val - (0.1*range_vals)
        plt.ylim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)

        width_bar = 0.55
        label_count = 0
        for test_metric,test_name,test_color in zip(verify_metrics['metrics'],verify_metrics['test_names'],verify_metrics['colors']):
            rects = ax.bar(test_name, round(test_metric,2), label=test_name, 
                   width=width_bar, linewidth=1, edgecolor='k', 
                   color=test_color, zorder=2)
            # plot whether the tests pass or fail
            if test_name != 'Model':
                if test_metric >= 0:
                    offset_txt = test_metric+(0.05*range_vals)
                else:
                    offset_txt = test_metric-(0.05*range_vals)
                if test_color == '#1f77b4':
                    txt_bar = 'pass'
                elif test_color == '#cd5c5c':
                    txt_bar = 'fail'
                ax.text(label_count, offset_txt, txt_bar, color=test_color, 
                        fontstyle='italic', horizontalalignment='center')
            label_count += 1

        # Set tick sizes
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # title and labels of the axis
        plt.ylabel(f'{verify_results["error_type"].upper()}', fontsize=14)

        title_graph = f"VERIFY tests of {os.path.basename(path_n_suffix)}"
        plt.text(0.5, 1.08, f'{title_graph} of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
            fontsize=14, fontweight='bold', transform = ax.transAxes)

        # add threshold line and arrow indicating passed test direction
        arrow_length = np.abs(max_lim-min_lim)/11
        
        if verify_results['error_type'].lower() in ['mae','rmse']:
            thres_line = verify_metrics['higher_thres']            
        else:
            thres_line = verify_metrics['lower_thres']
            arrow_length = -arrow_length

        width = 2
        xmin = 0.237
        thres = ax.axhline(thres_line,xmin=xmin, color='black',ls='--', label='thres', zorder=0)

        x_arrow = 0.5
        style = ArrowStyle('simple', head_length=4.5*width, head_width=3.5*width, tail_width=width)
        arrow = FancyArrowPatch((x_arrow, thres_line), (x_arrow, thres_line+arrow_length), 
                                arrowstyle=style, color='k')  # (x1,y1), (x2,y2) vector direction                   
        ax.add_patch(arrow)

        # invisible "dummy" arrows to make the graph wider so the real arrows fit in the right place
        ax.arrow(x_arrow, thres_line, 0, 0, width=0, fc='k', ec='k') # x,y,dx,dy format

        # legend and regression line with 95% CI considering all possible lines (not CI of the points)
        def make_legend_arrow(legend, orig_handle,
                            xdescent, ydescent,
                            width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5*height, width, 0, width=1.5, length_includes_head=True, head_width=0.58*height )
            return p

        arrow = plt.arrow(0, 0, 0, 0, label='arrow', width=0, fc='k', ec='k') # arrow for the legend
        plt.figlegend([thres,arrow], [f'Threshold ({verify_results["error_type"].upper()} = {round(thres_line,2)})','Passing condition'], handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
                      loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=True, fontsize=14)

        # Add gridlines
        ax.grid(linestyle='--', linewidth=1)

        # save plot
        verify_plot_file = f'{os.path.dirname(path_n_suffix)}/VERIFY_tests_{os.path.basename(path_n_suffix)}.png'
        plt.savefig(verify_plot_file, dpi=300, bbox_inches='tight')
        plt.clf()

        path_reduced = '/'.join(f'{verify_plot_file}'.replace('\\','/').split('/')[-2:])
        print_ver = f"\n   o  VERIFY plot saved in {path_reduced}"

        return print_ver


    def print_verify(self,results_print,verify_results,print_ver,path_n_suffix,verify_metrics,params_dict):
        '''
        Print and store the results of VERIFY
        '''
        
        verify_results_file = f'{os.path.dirname(path_n_suffix)}/VERIFY_tests_{os.path.basename(path_n_suffix)}.dat'
        path_reduced = '/'.join(f'{verify_results_file}'.replace('\\','/').split('/')[-2:])
        print_ver += f"\n   o  VERIFY test values saved in {path_reduced}"
        print_ver += f'\n      Results of flawed models and cross-validation:'
        # the printing order should be y-mean, y-shuffle and one-hot
        if verify_results['error_type'].lower() in ['mae','rmse']:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} (valid. set) {round(verify_results["original_score_valid"],2)} + {int(verify_metrics["thres_test"]*100)}% thres. = {round(verify_results["higher_thres"],2)}'
        else:
            print_ver += f'\n      Original {verify_results["error_type"].upper()} (valid. set) {round(verify_results["original_score_valid"],2)} - {int(verify_metrics["thres_test"]*100)}% thres. = {round(verify_results["lower_thres"],2)}'
        print_ver += results_print[0]
        print_ver += results_print[1]
        print_ver += results_print[2]
        if params_dict['type'].lower() == 'reg':
            print_ver += f"\n         - {verify_metrics['type_cv']} : R2 = {round(verify_results['cv_r2'],2)}, MAE = {round(verify_results['cv_mae'],2)}, RMSE = {round(verify_results['cv_rmse'],2)}"
        elif params_dict['type'].lower() == 'clas':
            print_ver += f"\n         - {verify_metrics['type_cv']} : Accuracy = {round(verify_results['cv_acc'],2)}, F1 score = {round(verify_results['cv_f1'],2)}, MCC = {round(verify_results['cv_mcc'],2)}"

        self.args.log.write(print_ver)
        dat_results = open(verify_results_file, "w")
        dat_results.write(print_ver)
        dat_results.close()
