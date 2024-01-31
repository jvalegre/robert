"""
Parameters
----------

    csv_name : str, default=''
        Name of the CSV file containing the database. A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv'). 
    y : str, default=''
        Name of the column containing the response variable in the input CSV file (i.e. 'solubility'). 
    discard : list, default=[]
        List containing the columns of the input CSV file that will not be included as descriptors
        in the curated CSV file (i.e. ['name','SMILES']).
    ignore : list, default=[]
        List containing the columns of the input CSV file that will be ignored during the curation process
        (i.e. ['name','SMILES']). The descriptors will be included in the curated CSV file. The y value
        is automatically ignored.
    destination : str, default=None
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    train : list, default=[60,70,80,90]
        Proportions of the training set to use in the ML scan. The numbers are relative to the training 
        set proportion (i.e. 40 = 40% training data).
    auto_kn : bool, default=True
        Changes random splitting to KN splitting in databases with less than 100 datapoints.
    filter_train : bool, default=True
        Disables the 90% training size in databases with less than 50 datapoints, and the 80% in less than 30.
    split : str, default='RND'
        Mode for splitting data. Options: 
        1. 'KN' (k-neighbours clustering-based splitting)
        2. 'RND' (random splitting).  
    model : list, default=['RF','GB','NN','MVL'] (regression) and default=['RF','GB','NN','AdaB'] (classification) 
        ML models available: 
        1. 'RF' (Random forest)
        2. 'MVL' (Multivariate lineal models)
        3. 'GB' (Gradient boosting)
        4. 'NN' (MLP neural network)
        5. 'GP' (Gaussian Process)
        6. 'AdaB' (AdaBoost)
        7. 'VR' (Voting regressor combining RF, GB and NN)
    custom_params : str, default=None
        Define new parameters for the ML models used in the hyperoptimization workflow. The path
        to the folder containing all the yaml files should be specified (i.e. custom_params='YAML_FOLDER')
    type : str, default='reg'
        Type of the pedictions. Options: 
        1. 'reg' (Regressor)
        2. 'clas' (Classifier)
    seed : list, default=[]
        Random seeds used in the ML predictor models, data splitting and other protocols. If seed 
        is not adjusted manually, the generate_acc option will set the values for seed.
    epochs : int, default=0
        Number of epochs for the hyperopt optimization. If epochs is not adjusted manually, the 
        generate_acc option will set the values for epochs.
    generate_acc : str, default='mid'
        Accuracy of the workflow performed in GENERATE in terms of seed and epochs. Options:
        1. 'low', fastest and least accurate protocol (seed = [0,8,19], epochs = 20)
        2. 'mid', compromise between 'low' and 'high' accurate protocol (seed = [0,8,19,43,70,233], 
        epochs = 40)
        3. 'high', slowest and most accurate protocol (seed = [0,8,19,43,70,233,1989,9999,20394,3948301], 
        epochs = 100)
    error_type : str, default: rmse (regression), acc (classification)
        Target value used during the hyperopt optimization. Options:
        Regression:
        1. rmse (root-mean-square error)
        2. mae (mean absolute error)
        3. r2 (R-squared, not recommended since R2 might be good even with high errors in small datasets)
        Classification:
        1. mcc (Matthew's correlation coefficient)
        2. f1 (F1 score)
        3. acc (accuracy, fraction of correct predictions)
    pfi_filter : bool, default=True
        Activate the PFI filter of descriptors.
    pfi_epochs : int, default=5
        Sets the number of times a feature is randomly shuffled during the PFI analysis
        (standard from sklearn webpage: 5).
    pfi_threshold : float, default=0.04
        The PFI filter is X% of the model's score (% adjusted, 0.04 = 4% of the total score during PFI).
        For regression, a value of 0.04 is recommended. For classification, the filter is turned off
        by default if pfi_threshold is 0.04.
    pfi_max : int, default=0
        Number of features to keep after the PFI filter. If pfi_max is 0, all the features that pass the PFI
        filter are used.
    test_set : float, default=0.1
        Amount of datapoints to separate as external test set. These points will not be used during the
        hyperoptimization, and PREDICT will be used the points as test set during ROBERT workflows. The separation
        of test points occurs at random before any data splits are carried out inside the GENErATE screening.
    auto_test : bool, default=True
        Removes test sets in databases with less than 50 datapoints and raises % of test points to 10% if 
        test_set is lower than that.
        
"""
#####################################################.
#        This file stores the GENERATE class        #
#             used in model generation              #
#####################################################.

import time
import pandas as pd
import random
from robert.utils import (
    load_variables, 
    finish_print,
    load_database,
)
from robert.generate_utils import (
    prepare_sets,
    hyperopt_workflow,
    PFI_workflow,
    heatmap_workflow,
    filter_seed,
    detect_best
)


class generate:
    """
    Class containing all the functions from the GENERATE module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the GENERATE module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "generate")

        # load database, discard user-defined descriptors and perform data checks
        csv_df, csv_X, csv_y = load_database(self,self.args.csv_name,"generate")

        # separates an external test set (if applicable)
        csv_df, csv_X, csv_y, csv_df_test = self.separate_test(csv_df, csv_X, csv_y)
  
        # changes from random to KN data splitting in some cases
        if self.args.auto_kn:
            self = self.adjust_split(csv_df)

        # if there are less than 50 and 30 datapoints, the 90% and 80% training sizes are disabled by default
        if self.args.filter_train:
            self = self.adjust_train(csv_df)
        
        # scan different ML models
        txt_heatmap = f"\no  Starting heatmap scan with {len(self.args.model)} ML models ({self.args.model}) and {len(self.args.train)} training sizes ({self.args.train})."

        # scan different training partition sizes
        cycle = 1
        txt_heatmap += f'\n   Heatmap generation:'
        self.args.log.write(txt_heatmap)
        for size in self.args.train:

            # scan different ML models
            for ML_model in self.args.model:

                for seed in self.args.seed:
                    seed = int(seed)

                    self.args.log.write(f'   - {cycle}/{len(self.args.model)*len(self.args.train)*len(self.args.seed)} - Training size: {size}, ML model: {ML_model}, seed: {seed} ')

                    # splits the data into training and validation sets, and standardize the X sets
                    Xy_data = prepare_sets(self,csv_X,csv_y,size,seed)

                    # restore defaults
                    Xy_data_hyp = Xy_data.copy()
                    csv_df_hyp = csv_df.copy()
                    csv_df_PFI = csv_df.copy()

                    # hyperopt process for ML models
                    _ = hyperopt_workflow(self, csv_df_hyp, ML_model, size, Xy_data_hyp, seed, csv_df_test)

                    # apply the PFI descriptor filter if it is activated
                    if self.args.pfi_filter:
                        try:
                            _ = PFI_workflow(self, csv_df_PFI, ML_model, size, Xy_data_hyp, seed, csv_df_test)
                        except (FileNotFoundError,ValueError): # in case the model/train/seed combination failed
                            pass

                    cycle += 1

                # only select best seed for each train/model combination
                try: # in case there are no models passing generate
                    name_csv = self.args.destination.joinpath(f"Raw_data/No_PFI/{ML_model}_{size}")
                    _ = filter_seed(self, name_csv)
                    if self.args.pfi_filter:
                        name_csv_pfi = self.args.destination.joinpath(f"Raw_data/PFI/{ML_model}_{size}")
                        _ = filter_seed(self, name_csv_pfi)
                except UnboundLocalError:
                    pass

        # detects best combinations
        dir_csv = self.args.destination.joinpath(f"Raw_data")
        _ = detect_best(f'{dir_csv}/No_PFI')
        if self.args.pfi_filter:
            _ = detect_best(f'{dir_csv}/PFI')

        # create heatmap plot(s)
        _ = heatmap_workflow(self,"No_PFI")

        if self.args.pfi_filter:
            _ = heatmap_workflow(self,"PFI")

        _ = finish_print(self,start_time,'GENERATE')


    def separate_test(self, csv_df, csv_X, csv_y):
        """
        Separates (if applies) a test set from the database before the model scan
        """
        
        csv_df_test = pd.DataFrame()
        test_points = []

        if self.args.csv_test != '':
            self.args.test_set = 0
        
        if self.args.auto_test:
            if self.args.test_set != 0:
                if len(csv_df[self.args.y]) < 50:
                    self.args.test_set = 0
                    self.args.log.write(f'\nx    WARNING! The database contains {len(csv_df[self.args.y])} datapoints, the data will be split in training and validation with no points separated as external test set (too few points to reach a reliable splitting). You can bypass this option and include test points with "--auto_test False".')
                    self.args.test_set
                elif self.args.test_set < 0.1:
                    self.args.test_set = 0.1
                    self.args.log.write(f'\nx    WARNING! The test_set option was set to {self.args.test_set}, this value will be raised to 0.1 to include a meaningful amount of points in the test set. You can bypass this option and include less test points with "--auto_test False".')

                if self.args.test_set > 0:
                    n_of_points = int(len(csv_X)*(self.args.test_set))

                    random.seed(self.args.seed[0])
                    test_points = random.sample(range(len(csv_X)), n_of_points)

                    # separates the test set and reset_indexes
                    csv_df_test = csv_df.iloc[test_points].reset_index(drop=True)
                    csv_df = csv_df.drop(test_points, axis=0).reset_index(drop=True)
                    csv_X = csv_X.drop(test_points, axis=0).reset_index(drop=True)
                    csv_y = csv_y.drop(test_points, axis=0).reset_index(drop=True)
        
        else:
            self.args.test_set = 0

        return csv_df, csv_X, csv_y, csv_df_test


    def adjust_split(self, csv_df):
        '''
        Changes the split to KN when small or imbalanced databases are used
        '''
        
        # when using databases with a small number of points
        if len(csv_df[self.args.y]) < 100 and self.args.split.lower() == 'rnd':
            self.args.split = 'KN'
            self.args.log.write(f'\nx    WARNING! The database contains {len(csv_df[self.args.y])} datapoints, KN data splitting will replace the default random splitting (too few points to reach a reliable splitting). You can use random splitting with "--auto_kn False".')
        
        # when using unbalanced databases (using an arbitrary imbalance ratio of 10 with the two halves of the data)
        mid_value = max(csv_df[self.args.y])-((max(csv_df[self.args.y])-min(csv_df[self.args.y]))/2)
        high_vals = len([i for i in csv_df[self.args.y] if i >= mid_value])
        low_vals = len([i for i in csv_df[self.args.y] if i < mid_value])
        imb_ratio_high = high_vals/low_vals
        imb_ratio_low = low_vals/high_vals
        if max(imb_ratio_high,imb_ratio_low) > 10:
            self.args.split = 'KN'
            if imb_ratio_high > 10:
                range_type = 'high'
            elif imb_ratio_low > 10:
                range_type = 'low'
            self.args.log.write(f'\nx    WARNING! The database is imbalanced (imbalance ratio > 10, more values in the {range_type} half of values), KN data splitting will replace the default random splitting. You can use random splitting with "--auto_kn False".')
        
        return self


    def adjust_train(self,csv_df):
        '''
        Removes 90% and 80% training sizes from model screenings when using small databases
        '''
        
        removed = []
        if len(csv_df[self.args.y]) < 50 and 90 in self.args.train:
            self.args.train.remove(90)
            removed.append('90%')
        if len(csv_df[self.args.y]) < 30 and 80 in self.args.train:
            self.args.train.remove(80)
            removed.append('80%')
        if len(removed) > 0:
            self.args.log.write(f'\nx    WARNING! The database contains {len(csv_df[self.args.y])} datapoints, the {", ".join(removed)} training size(s) will be excluded (too few validation points to reach a reliable result). You can include this size(s) using "--filter_train False".')
        
        return self
