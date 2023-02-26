"""
Parameters
----------

General
+++++++

# Specify t-value that will be the threshold to identify outliers
# (check tables for t-values elsewhere). The higher the t-value 
# the more restrictive the analysis will be (i.e. there will 
# be more outliers with t-value=1 than with t-value = 4)
t_value = 2

   files : str or list of str, default=None
     Input files. Formats accepted: XYZ, SDF, GJF, COM and PDB. Also, lists can
     be used (i.e. [FILE1.sdf, FILE2.sdf] or \*.FORMAT such as \*.sdf).  
   program : str, default=None
     Program required in the conformational refining. 
     Current options: 'xtb', 'ani'
"""
#####################################################.
#        This file stores the PREDICT class         #
#              used in the predictor                #
#####################################################.

import os
import sys
import time
from pathlib import Path
from scipy import stats
from robert.utils import load_variables


class predict:
    """
    Class containing all the functions from the PREDICT module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the PREDICT module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time_overall = time.time()
        # load default and user-specified variables
        self.args = load_variables(kwargs, "predict")





da la opcion de guardar 1 grafico con todo o separar en 3 graficos
graph with trainng, valid y test si existe, optional (como en el articulo de phneols)
RMSE, MAE, R2 prints
SHAP analysis (valid)
PFI analysis (valid)

        outlier analysis (train+valid)

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