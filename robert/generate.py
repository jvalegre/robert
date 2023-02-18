"""
Parameters
----------

General
+++++++
     csv_name : str, default=''
         Name of the CSV file containing the database. A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv'). 
     csv_params : str, default='ML_params'
         Name of the CSV file that will store the parameters of the ML models. 
     y : str, default=''
         Name of the column containing the response variable in the input CSV file (i.e. 'solubility'). 
     discard : list, default=[]
         List containing the columns of the input CSV file that will not be included as descriptors
         in the curated CSV file (i.e. ['name','SMILES']).
     ignore : list, default=[]
         List containing the columns of the input CSV file that will be ignored during the curation process
         (i.e. ['name','SMILES']). The descriptors will be included in the curated CSV file. The y value
         is automatically ignored.
     destination : str, default=None,
         Directory to create the output file(s).
     varfile : str, default=None
         Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
     train : list, default=[60,70,80,90],
         Proportions of the training set to use in the ML scan. The numbers are relative to the training 
         set proportion (i.e. 40 = 40% training data).
     split : str, default='KN'
         Mode for splitting data. Options: 
            1. 'KN' (k-neighbours clustering-based splitting)
            2. 'RND' (random splitting).  
     model : list, default=['RF','GB','NN', 'VR'],
         ML models used in the ML scan. Options: 
            1. 'RF' (Random forest)
            2. 'MVL' (Multivariate lineal models)
            3. 'GB' (Gradient boosting)
            4. 'AdaB' (AdaBoost regressor)
            5. 'NN' (MLP regressor neural network)
            6. 'VR' (Voting regressor combining RF, GB and NN )
     mode : str, default='reg',
         Type of the pedictions. Options: 
            1. 'reg' (Regressor)
            1. 'clas' (Classifier)
     seed : int, default=0,
         Random seed used in the ML predictor models, data splitting and other protocols.
     epochs : int, default=500,
         Number of epochs for the hyperopt optimization.
     hyperopt_target : str, default: rmse (regression), mcc (classification)
         Target value used during the hyperopt optimization. Options:
         Regression:
            1. rmse (root-mean-square error)
            2. mae (mean absolute error)
            3. r2 (R-squared)
         Classification:
            1. mcc (Matthew's correlation coefficient)
            2. f1_score (F1 score)
            3. acc (accuracy, fraction of correct predictions)
     PFI : bool, default=True
         Activate the PFI filter of descriptors.
     PFI_epochs : int, default=30,
         Sets the number of times a feature is randomly shuffled during the PFI analysis
         (standard from Sklearn webpage: 30).
     PFI_threshold : float, default=0.04,
         The PFI filter is X% of the model's score (% adjusted, 0.04 = 4% of the total score during PFI).
         For regression, a value of 0.04 is recommended. For classification, the filter is turned off
         by default if PFI_threshold is 0.04.

"""
#####################################################.
#        This file stores the GENERATE class        #
#             used in model generation              #
#####################################################.

import os
import sys
import time
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from progress.bar import IncrementalBar
from robert.utils import (
    load_variables, 
    destination_folder, 
    sanity_checks, 
    load_database, 
    create_folders,
    standardize,
)
from robert.generate_utils import data_split, run_hyperopt, update_best


class generate:
    """
    Class containing all the functions from the GENERATE module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the GENERATE module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time_overall = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "generate")

        # creates destination folder
        _ = destination_folder(self,"GENERATE")

        # initial sanity checks
        _ = sanity_checks(self, 'initial', "generate", None)

        # turn off PFI_filter for classification
        if self.args.PFI_threshold == 0.04 and self.args.mode == 'clas':
            self.args.log.write("\nx  The PFI filter was disabled for classification models")
            self.args.PFI = False

        # adjust the default value of hyperopt_target for classification
        if self.args.mode == 'clas':
            self.args.hyperopt_target = 'mcc'

        # load database, discard user-defined descriptors and perform data checks
        csv_df = load_database(self, "generate")

        # ignore user-defined descriptors and assign X and y values
        csv_df = csv_df.drop(self.args.ignore, axis=1)
        csv_X = csv_df.drop([self.args.y], axis=1)
        csv_y = csv_df[self.args.y]

        # Check if the folders exist and if they do, delete and replace them
        folder_names = ['Best_model/No_PFI', 'Raw_data/No_PFI']
        if self.args.PFI:
            folder_names.append('Best_model/PFI')
            folder_names.append('Raw_data/PFI')
        _ = create_folders(folder_names)

        # scan different training set sizes
        self.args.log.write(f"\no  Starting heatmap generation with {len(self.args.model)} ML models {self.args.model} and {len(self.args.train)} training sizes {self.args.train}.")

        bar = IncrementalBar("\no  Heatmap generation", max=len(self.args.files))
        for size in self.args.train:
            # split into training and validation sets
            Xy_data = data_split(self,csv_X,csv_y,size)

            # standardization of X values using the mean and SD of the training set
            X_train_scaled, X_valid_scaled = standardize(Xy_data['X_train'],Xy_data['X_valid'])
            Xy_data['X_train_scaled'] = X_train_scaled
            Xy_data['X_valid_scaled'] = X_valid_scaled

            # scan different ML models
            for ML_model in self.args.model:
                # hyperopt process including k-neighbours-based splitting of the data
                _ = run_hyperopt(self, ML_model, size, Xy_data)

                # check if this combination is the best model and replace data in the Best_model folder
                name_csv = Path(f"/Raw_data/No_PFI/{ML_model}_{size}.csv")
                _ = update_best(self,csv_df,Xy_data,name_csv)

                # apply the PFI filter if it is activated
                if self.args.PFI:

                    # read the CSV file with the optimal parameters of this model/size combination
                    name_csv_hyperopt = Path(f"/Raw_data/{ML_model}_{size}")
                    PFI_df = pd.read_csv(name_csv_hyperopt+'.csv')

                    _ = PFI_workflow(PFI_df,Xy_data)
            




                bar.next()
        bar.finish()


        #     # calculate the permutation feature importance (PFI) of the descriptors in the 
        #         # model and generates a new dataset
  

        #         # PFI function
        #         combined_descriptor_list = PFI_workflow(X,MODEL,PFI_df,X_train_scaled,y_train,X_valid_scaled,y_valid,n_repeats,PFI_threshold,False,mode,PFI)

        #         # creates X and y sets
        #         # creates a database with the most important descriptors after PFI

        #         df_PFI_model = pd.DataFrame()
        #         df_PFI_model[response_value] = DFT_parameters_filtered[response_value]

        #         for i,column in enumerate(DFT_parameters_filtered.columns):
        #             if column in combined_descriptor_list:
        #                 df_PFI_model[column] = DFT_parameters_filtered[column]

        #         X_PFI = df_PFI_model.drop([response_value], axis=1)
        #         y_PFI = df_PFI_model[response_value]

        #         # k-neighbours-based data splitting using previous training points
        #         X_train_PFI = X_PFI.iloc[training_points]
        #         y_train_PFI = y_PFI.iloc[training_points]
        #         X_valid_PFI = X_PFI.drop(training_points)
        #         y_valid_PFI = y_PFI.drop(training_points)

        #         # standardizes the data sets using the mean and standard dev from the train set
        #         Xmean = X_train_PFI.mean(axis=0)
        #         Xstd = X_train_PFI.std(axis=0)
        #         X_train_PFI_scaled = (X_train_PFI - Xmean) / Xstd
        #         X_valid_PFI_scaled = (X_valid_PFI - Xmean) / Xstd
        #         # run the best model from hyperopt and calculates its efficiency using only 
        #         # the most important features from the PFI analysis
        #         try:
        #             if int(PFI_df['max_features'][0]) > len(X_PFI.columns):
        #                 PFI_df.at[0,'max_features'] = len(X_PFI.columns)
        #                 # replace the value in the parameters csv
        #                 export_param_excel = PFI_df.to_csv(csv_params+'.csv', index = None, header=True)
        #         except KeyError:
        #             pass

        #         if mode == 'reg':
        #             # calculate R2, MAE and RMSE for train and validation sets
        #             r2_train_PFI,mae_train_PFI,rmse_train_PFI,r2_valid_PFI,mae_valid_PFI,rmse_valid_PFI,y_pred_train_PFI,y_pred_valid_PFI = predictor_workflow(seed,MODEL,PFI_df,X_train_PFI_scaled,y_train_PFI,X_valid_PFI_scaled,y_valid_PFI,mode,size)
        #             # calculates k-fold cross validation
        #             cv_score = cross_val_calc(seed,MODEL,PFI_df,X_train_PFI_scaled,y_train_PFI,mode,cv_kfold)
        #             # print stats
        #             #print_model_stats(MODEL,X_train_PFI_scaled,X_valid_PFI_scaled,r2_train_PFI,mae_train_PFI,rmse_train_PFI,r2_valid_PFI,mae_valid_PFI,rmse_valid_PFI,mode,cv_score,cv_kfold,'Robert_results.txt')
        #             # data of the model
        #             dict_model_PFI = {
        #                 "MODEL": MODEL,
        #                 "size": size,
        #                 "PFI_df": PFI_df,
        #                 "r2_train_PFI": r2_train_PFI,
        #                 "mae_train_PFI": mae_train_PFI,
        #                 "rmse_train_PFI": rmse_train_PFI,
        #                 "r2_valid_PFI": r2_valid_PFI,
        #                 "mae_valid_PFI": mae_valid_PFI,
        #                 "rmse_valid_PFI": rmse_valid_PFI,
        #                 "rmse_valid": rmse_valid,
        #                 "X_train_PFI_scaled": X_train_PFI_scaled,
        #                 "X_train_scaled": X_train_scaled,
        #                 "y_pred_train_PFI": y_pred_train_PFI,
        #                 "y_pred_valid_PFI": y_pred_valid_PFI,
        #                 "cv_score": cv_score,
        #                 "X_valid_PFI_scaled": X_valid_PFI_scaled,
        #                 "mode": mode,
        #                 "cv_kfold": cv_kfold,
        #                 "Robert_results": "Robert_results.txt",
        #                 "y_train_PFI": y_train_PFI,
        #                 "y_valid_PFI": y_valid_PFI,
        #                 "X_PFI": X_PFI,
        #                 "fixed_data_train": fixed_data_train,
        #                 "fixed_data_valid": fixed_data_valid
        #             }
        #             models_data_indiv = [MODEL, PFI_df, r2_train_PFI,mae_train_PFI,rmse_train_PFI,r2_valid_PFI,mae_valid_PFI,rmse_valid_PFI,rmse_valid,X_train_PFI_scaled,X_train_scaled,y_pred_train_PFI,y_pred_valid_PFI, cv_score,X_valid_PFI_scaled,mode,cv_kfold,'Robert_results.txt',y_train_PFI,y_valid_PFI, X_PFI,fixed_data_train,fixed_data_valid]
        #         elif mode == 'clas':
        #             # calculate accuracy, F1 score and MCC for train and validation sets
        #             accuracy_train_PFI,f1score_train_PFI,mcc_train_PFI,accuracy_valid_PFI,f1score_valid_PFI,mcc_valid_PFI,y_pred_train_PFI,y_pred_valid_PFI = predictor_workflow(seed,MODEL,PFI_df,X_train_PFI_scaled,y_train_PFI,X_valid_PFI_scaled,y_valid_PFI,mode,size)
        #             # calculates k-fold cross validation
        #             cv_score = cross_val_calc(seed,MODEL,PFI_df,X_train_PFI_scaled,y_train_PFI,mode,cv_kfold)
        #             # print stats
        #             #print_model_stats(MODEL,X_train_PFI_scaled,X_valid_PFI_scaled,accuracy_train_PFI,f1score_train_PFI,mcc_train_PFI,accuracy_valid_PFI,f1score_valid_PFI,mcc_valid_PFI,mode,cv_score,cv_kfold,'Robert_results.txt')
        #             # data of the model
        #             models_data_indiv = [MODEL, PFI_df, accuracy_train_PFI,f1score_train_PFI,mcc_train_PFI,accuracy_valid_PFI,f1score_valid_PFI,mcc_valid_PFI,y_pred_train_PFI,y_pred_valid_PFI, cv_score]
            
        #         #Create csv files for all model and training sizes
        #         dict_model_PFI_pd = pd.DataFrame.from_dict(dict_model_PFI, orient='index')
        #         dict_model_PFI_pd=dict_model_PFI_pd.transpose()
        #         dict_model_PFI_excel = dict_model_PFI_pd.to_csv(f'Raw_data/Model_params/{dict_model_PFI["MODEL"]}_{size}_PFI.csv', index = None, header=True)

        #         models_data.append(models_data_indiv)
        
        #     size_data_indiv = [size,models_data]
        #     size_data.append(size_data_indiv)
            
        # train_and_evaluate_models(X, y, train, split, model, mode, seed, w_dir, csv_params, cv_kfold)
        # def find_min_column_value(csv_files, column_name, output_directory):
        #     min_value = float('inf')
        #     min_file = None
        #     for csv_file in csv_files:
        #         with open(csv_file, 'r') as f:
        #             reader = csv.DictReader(f)
        #             for row in reader:
        #                 value = float(row[column_name])
        #                 if value < min_value:
        #                     min_value = value
        #                     min_file = csv_file
            
        #     shutil.copy(min_file, output_directory)
        #     return min_value, min_file

        # csv_files = glob.glob('Raw_data/Model_params/*[!_PFI]*.csv')
        # column_name = 'rmse_valid'
        # min_value, min_file = find_min_column_value(csv_files, column_name, 'Raw_data/Best_Model/Best_Model.csv')

        # csv_files_PFI = glob.glob('Raw_data/Model_params/*_PFI*.csv')
        # column_name_PFI = 'rmse_valid_PFI'
        # min_value_PFI, min_file_PFI = find_min_column_value(csv_files_PFI, column_name_PFI, 'Raw_data/Best_Model/Best_Model_PFI.csv')

        # # Read the CSV file into a Pandas DataFrame
        # df_min = pd.read_csv(min_file)

        # # Get the 'MODEL' and 'size' values from the first row of the DataFrame
        # model_value = df_min['MODEL'].iloc[0]
        # size_value = df_min['size'].iloc[0]

        # # Read the CSV file into a Pandas DataFrame
        # df_min_PFI = pd.read_csv(min_file_PFI)

        # # Get the 'MODEL' and 'size' values from the first row of the DataFrame
        # model_value_PFI = df_min_PFI['MODEL'].iloc[0]
        # size_value_PFI = df_min_PFI['size'].iloc[0]

        # # Warning if there is a model without PFI with rmse < than model with PFI with the less rmse value  
        # if min_value < min_value_PFI:
        #     print('\n'f"x  Warning! Error lower without PFI filter (no PFI: RMSE = {round(min_value,2)} using {model_value}_{size_value} ; with PFI filter: {round(min_value_PFI,2)} using {model_value_PFI}_{size_value_PFI}) consider using PFI=False")      

        # print('\n'f"The optimal model using PFI={PFI} is {model_value_PFI} with training size {size_value_PFI}%"'\n')

        # #Obtain the best model (<rmse_valid value)
        # #best_model = optimal_model(size_data)

        # # List to store the rmse_valid_PFI values of files without _PFI in the name
        # rmse_list_1 = []
        # # List to store the rmse_valid_PFI values of the files with _PFI in the name
        # rmse_list_2 = []
        # # Iterate over the csv files in the directory Raw_data/Model_params
        # for filename in os.listdir("Raw_data/Model_params"):
        #     # If the file does not have _PFI in its name
        #     if "_PFI" not in filename:
        #         # Read the file with pandas and select the value of rmse_valid_PFI
        #         df = pd.read_csv(f"Raw_data/Model_params/{filename}")
        #         rmse = df["rmse_valid"].values[0]
        #         # Add the value to the list rmse_list_1
        #         rmse_list_1.append(rmse)
        #     # If the file does have _PFI in its name
        #     else:
        #         # Read the file with pandas and select the value of rmse_valid_PFI
        #         df = pd.read_csv(f"Raw_data/Model_params/{filename}")
        #         rmse = df["rmse_valid_PFI"].values[0]
        #         # Add the value to the list rmse_list_2
        #         rmse_list_2.append(rmse)

        # def create_dataframe(data, column_name):
        #     num_columns = len(model)
        #     num_rows = len(train)
        #     # Creates a list of column names using model
        #     column_names = sorted(model)
        #     # Creates a list of row names using train
        #     row_names = train
        #     values_matrix = np.array(data).reshape(num_columns, num_rows)
        #     # Creates the DataFrame using pd.DataFrame() and providing the array of values, the number of columns and rows, and the lists of column and row names
        #     df = pd.DataFrame(data=values_matrix, columns=row_names, index=column_names)
        #     df = df.transpose()
        #     df.columns.name = column_name
        #     return df
        # plot_data_1 = create_dataframe(rmse_list_1, 'Model Type')
        # plot_data_2 = create_dataframe(rmse_list_2, 'Model Type')

        # def create_heatmap(data, title, output_file):
        #     df_plot = pd.DataFrame(data)
        #     df_plot.columns = [model]
        #     df_plot.index = [train]
        #     df_plot = df_plot.sort_index(ascending=False)
        #     fig, ax = plt.subplots(figsize=(7.45,6))
        #     sb.set(font_scale=1.2, style='ticks')
        #     cmap_blues_75_percent_512 =  [mcolor.rgb2hex(c) for c in plt.cm.Blues(np.linspace(0, 0.8, 512))]
        #     ax = sb.heatmap(df_plot, annot=True, linewidth=1, cmap=cmap_blues_75_percent_512, cbar_kws={'label': 'RMSE Validation'})
        #     ax.set(xlabel="Model Type", ylabel="Training Size")
        #     plt.title(title)
        #     sb.despine(top=False, right=False)
        #     plt.savefig(output_file, dpi=600, bbox_inches='tight')
        #     ax.plot()

        # if PFI:
        #     create_heatmap(plot_data_1, 'NO_PFI', 'Benchmark_methods/NO_PFI.png')
        #     create_heatmap(plot_data_2, 'PFI', 'Benchmark_methods/PFI.png')
        # else:
        #     create_heatmap(plot_data_1, 'NO_PFI', 'Benchmark_methods/NO_PFI.png')










        # saves the curated CSV
        txt_csv = f'\no  {len(csv_df.columns)} descriptors remaining after applying correlation filters:\n'
        txt_csv += '\n'.join(f'   - {var}' for var in csv_df.columns)
        self.args.log.write(txt_csv)

        csv_curate_name = f'{self.args.csv_name.split(".")[0]}_CURATE.csv'
        csv_curate_name = self.curate_folder.joinpath(csv_curate_name)
        _ = csv_df.to_csv(f'{csv_curate_name}', index = None, header=True)
        self.args.log.write(f'\no  The curated database was stored in {csv_curate_name}.')

        elapsed_time = round(time.time() - start_time_overall, 2)
        self.args.log.write(f"\nTime CURATE: {elapsed_time} seconds\n")
        self.args.log.finalize()

        # this is added to avoid path problems in jupyter notebooks
        os.chdir(self.args.initial_dir)



