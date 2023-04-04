"""
Parameters
----------

General
+++++++

     destination : str, default=None,
         Directory to create the output file(s).
     varfile : str, default=None
         Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
     model_dir : str, default=''
         Folder containing the database and parameters of the ML model.
     csv_test : str, default=''
         Name of the CSV file containing the test set (if any). A path can be provided (i.e. 
         'C:/Users/FOLDER/FILE.csv'). 
     t_value : float, default=2
         t-value that will be the threshold to identify outliers (check tables for t-values elsewhere).
         The higher the t-value the more restrictive the analysis will be (i.e. there will be more 
         outliers with t-value=1 than with t-value = 4).
     seed : int, default=8,
         Random seed used in the ML predictor models, data splitting and other protocols.
     shap_show : int, default=10,
         Number of descriptors shown in the plot of the SHAP analysis.
     pfi_show : int, default=10,
         Number of descriptors shown in the plot of the PFI analysis.
     pfi_epochs : int, default=30,
         Sets the number of times a feature is randomly shuffled during the PFI analysis
         (standard from Sklearn webpage: 30).

"""
#####################################################.
#        This file stores the PREDICT class         #
#    used to analyze and generate ML predictors     #
#####################################################.

import os
import time
from robert.predict_utils import (plot_predictions,
    load_test,
    save_predictions,
    print_predict,
    shap_analysis,
    PFI_plot
    )
from robert.utils import (load_variables,
    load_db_n_params,
    pd_to_dict,
    load_n_predict,
)

class predict:
    """
    Class containing all the functions from the PREDICT module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the PREDICT module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "predict")

        # if model_dir = '', the program performs the tests for the No_PFI and PFI folders
        if 'GENERATE/Best_model' in self.args.model_dir:
            model_dirs = [f'{self.args.model_dir}/No_PFI',f'{self.args.model_dir}/PFI']
        else:
            model_dirs = [self.args.model_dir]

        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                # load and ML model parameters, and add standardized descriptors
                Xy_data, params_df = load_db_n_params(self,model_dir,"verify")

                # load test set and add standardized descriptors
                if self.args.csv_test != '':
                    Xy_data = load_test(self, Xy_data, params_df)
                    
                # set the parameters for each ML model of the hyperopt optimization
                params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

                # get results from training, validation and test (if any)
                Xy_data = load_n_predict(params_dict, Xy_data)
                
                # save predictions for all sets
                path_n_suffix = save_predictions(self,Xy_data,model_dir)

                # represent y vs predicted y
                _ = plot_predictions(self, params_dict, Xy_data, path_n_suffix)

                # print results
                _ = print_predict(self,Xy_data,params_dict,path_n_suffix)  

                # SHAP analysis
                _ = shap_analysis(self,Xy_data,params_dict,path_n_suffix)

                # PFI analysis
                _ = PFI_plot(self,Xy_data,params_dict,path_n_suffix)


# PFI analysis (valid)
# outlier analysis (train+valid)

# def plot_outliers(t_value,Experim_values_init,DFT_values_init,Ir_cat_names):
    
#     n_initial_points = len(Experim_values_init)
    
#     # Calculation of different parameters for regression model
#     # to calculate the errors
#     slope, intercept, r_value, p_value, std_err = stats.linregress(Experim_values_init,DFT_values_init)
#     predict_y = intercept + (slope * Experim_values_init)
    
#     # Outlayer detector graph
#     # Normalize the data
#     error_abs, error_normal = [], []
#     error_no_outlier = []
#     error_outlier = []
#     outlier_names = []
#     list_to_remove_Experim_values, list_to_remove_DFT_values = [], []
#     outlier_indexes = []

#     # Calculate absolute errors between predicted y and actual 
#     # DFT values
#     for i in range(len(predict_y)):
#         error_abs.append(np.absolute(predict_y[i] - DFT_values_init[i]))
#     Mean = np.mean(error_abs)
#     SD = np.std(error_abs)

#     # Since the data is normalized, the errors are in standard
#     # deviation units. When the absolute error is larger than 
#     # assigned t-value, it is considered an outlier
#     for i in range(len(Experim_values_init)):
#         error_normal.append((error_abs[i]-Mean)/SD)
#         if np.absolute(error_normal[i]) > t_value:
#             error_outlier.append(error_normal[i])
#             outlier_names.append(Ir_cat_names[i])
#             list_to_remove_Experim_values.append(Experim_values_init[i])
#             list_to_remove_DFT_values.append(DFT_values_init[i])
#         else:
#             error_no_outlier.append(error_normal[i])

#     # Make a copy of the values without the outliers
#     for i in reversed(range(len(Experim_values_init))):
#         if Ir_cat_names[i] in outlier_names:
#             del Experim_values_init[i]
#             del DFT_values_init[i]
#             outlier_indexes.append(i)
    
#     # Plot outliers in red
#     fig, ax = plt.subplots()
#     plt.grid(linestyle='--', linewidth=1)
#     Plot_outliers = {'error_outlier': error_outlier}
#     Plot_no_outliers = {'error_no_outlier': error_no_outlier}
#     df_outliers = pd.DataFrame.from_dict(Plot_outliers)
#     df_no_outliers = pd.DataFrame.from_dict(Plot_no_outliers)
#     plt.scatter(df_no_outliers["error_no_outlier"], df_no_outliers["error_no_outlier"],
#                  c='b', edgecolors='none', alpha=0.4,)  # Include border to the points
#     plt.scatter(df_outliers["error_outlier"], df_outliers["error_outlier"],
#                  c='r', edgecolors='none', alpha=0.4,)  # Include border to the points

#     # Set styling preferences
#     sb.set(font_scale=1.2, style="ticks")
#     plt.xlabel('$\sigma$$_e$$_r$$_r$$_o$$_r$',fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.ylabel('$\sigma$$_e$$_r$$_r$$_o$$_r$',fontsize=14)
#     plt.yticks(fontsize=14)
    
#     # Set plot limits depending on the max normal absolute error
#     axis_limit = math.ceil(np.absolute(error_normal).max() + 0.5)
#     plt.ylim(-(axis_limit), (axis_limit))
#     plt.xlim(-(axis_limit), (axis_limit))

#     # Plot rectangles in corners. First, I get the difference
#     # between t_value and the axis_limit since that is gonna be
#     # the size of the rectangle
#     diff_tvalue = axis_limit - t_value
#     Rectangle_top = mpatches.Rectangle(xy=(axis_limit, axis_limit), width=-diff_tvalue, height=-diff_tvalue, facecolor='grey', alpha=0.3)
#     Rectangle_bottom = mpatches.Rectangle(xy=(-(axis_limit), -(axis_limit)), width=diff_tvalue, height=diff_tvalue, facecolor='grey', alpha=0.3)
    
#     # ax.add_patch(Rectangle)
#     ax.add_patch(Rectangle_top)
#     ax.add_patch(Rectangle_bottom)

#     print('Outliers for '+folder+':')
    
#     outliers_printed_list = ''
#     for i in outlier_names:
#         outliers_printed_list += i+', '
#     print(outliers_printed_list[:-2])   
    
#     # Calculate % of outliers discarded

#     discarded_outliers = (len(outlier_indexes))/(n_initial_points)*100
#     print('\nDiscarding '+str(len(outlier_indexes))+' outliers out of '+str(n_initial_points)+' points ('+str(round(discarded_outliers,2))+'%).')
    
#     plt.savefig('DFT_vs_Experimental_Outliers_'+folder+'.png', dpi=400, bbox_inches='tight')
    
#     plt.show()
    
#     return Experim_values_init,DFT_values_init,outlier_indexes

        # printing and representing the results
        # print(f"\nPermutation feature importances of the descriptors in the {PFI_df['model']}_{PFI_df['train']}_PFI model (for the validation set). Only showing values that drop the original score at least by {self.args.PFI_threshold*100}%:\n")
        # print('Original score = '+f'{score_model:.2f}')
        # for i in range(len(PFI_values)):
        #     print(combined_descriptor_list[i]+': '+f'{PFI_values[i]:.2f}'+' '+u'\u00B1'+ ' ' + f'{PFI_SD[i]:.2f}')

        # y_ticks = np.arange(0, len(PFI_values))
        # fig, ax = plt.subplots()
        # ax.barh(y_ticks, PFI_values[::-1])
        # ax.set_yticklabels(combined_descriptor_list[::-1])
        # ax.set_yticks(y_ticks)
        # ax.set_title(model_type_PFI_fun+" permutation feature importances (PFI)")
        # fig.tight_layout()
        # plot = ax.set(ylabel=None, xlabel='PFI')

        # plt.savefig(f'PFI/{model_type_PFI_fun}+ permutation feature importances (PFI)', dpi=600, bbox_inches='tight')

        # plt.show()