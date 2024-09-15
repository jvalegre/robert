"""
Parameters
----------

    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    params_dir : str, default=''
        Folder containing the database and parameters of the ML model.
    csv_test : str, default=''
        Name of the CSV file containing the test set (if any). A path can be provided (i.e. 
        'C:/Users/FOLDER/FILE.csv'). 
    t_value : float, default=2
        t-value that will be the threshold to identify outliers (check tables for t-values elsewhere).
        The higher the t-value the more restrictive the analysis will be (i.e. there will be more 
        outliers with t-value=1 than with t-value = 4).
    alpha : float, default=0.05
        Significance level, or probability of making a wrong decision. This parameter is related to
        the confidence intervals (i.e. 1-alpha is the confidence interval). By default, an alpha value
        of 0.05 is used, which corresponds to a confidence interval of 95%.
    shap_show : int, default=10,
        Number of descriptors shown in the plot of the SHAP analysis.
    pfi_show : int, default=10,
        Number of descriptors shown in the plot of the PFI analysis.
    pfi_epochs : int, default=5,
        Sets the number of times a feature is randomly shuffled during the PFI analysis
        (standard from sklearn webpage: 5).
    names : str, default=''
        Column of the names for each datapoint. Names are used to print outliers.

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
    PFI_plot,
    outlier_plot,
    print_cv_var,
    distribution_plot,
    pearson_map_predict
    )
from robert.utils import (load_variables,
    load_db_n_params,
    pd_to_dict,
    load_n_predict,
    finish_print,
    print_pfi
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

        # if params_dir = '', the program performs the tests for the No_PFI and PFI folders
        if 'GENERATE/Best_model' in self.args.params_dir:
            params_dirs = [f'{self.args.params_dir}/No_PFI',f'{self.args.params_dir}/PFI']
        else:
            params_dirs = [self.args.params_dir]

        for params_dir in params_dirs:
            if os.path.exists(params_dir):

                _ = print_pfi(self,params_dir)

                # load and ML model parameters, and add standardized descriptors
                Xy_data, params_df, _, _, Xy_test_df = load_db_n_params(self,params_dir,"verify",True) # module 'verify' since PREDICT follows similar protocols

                # load test set (if any) and add standardized descriptors
                Xy_data, Xy_test_df = load_test(self, Xy_data, params_df, Xy_test_df)
                
                # set the parameters for each ML model
                params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)
                
                # get results from training, validation and test (if any)
                Xy_data,loaded_model = load_n_predict(self, params_dict, Xy_data, mapie=True)

                # save predictions for all sets
                path_n_suffix, name_points, Xy_data = save_predictions(self,Xy_data,params_dir,Xy_test_df,params_dict)

                # represent y vs predicted y
                colors = plot_predictions(self,params_dict,Xy_data,path_n_suffix)

                # print results
                _ = print_predict(self,Xy_data,params_dict,path_n_suffix,loaded_model)  

                # print CV variation (for regression)
                if params_dict['type'].lower() == 'reg':
                    _ = print_cv_var(self,Xy_data,params_dict,path_n_suffix)

                # SHAP analysis
                _ = shap_analysis(self,Xy_data,params_dict,path_n_suffix)

                # PFI analysis
                _ = PFI_plot(self,Xy_data,params_dict,path_n_suffix)

                # create Pearson heatmap
                _ = pearson_map_predict(self,Xy_data,params_dir)

                # Outlier analysis
                if params_dict['type'].lower() == 'reg':
                    _ = outlier_plot(self,Xy_data,path_n_suffix,name_points,colors)

                # y distribution
                _ = distribution_plot(self,Xy_data,path_n_suffix,params_dict)

        _ = finish_print(self,start_time,'PREDICT')
