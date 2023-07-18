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
    filter_train : bool, default=True
        Disables the 90% training size in databases with less than 50 entries.
    split : str, default='KN'
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

"""
#####################################################.
#        This file stores the GENERATE class        #
#             used in model generation              #
#####################################################.

import time
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

        # if there are less than 50 datapoints, the 90% training size is disabled by default
        if self.args.filter_train:
            if len(csv_df[self.args.y]) < 50 and 90 in self.args.train:
                self.args.train.remove(90)
                self.args.log.write(f'\nx    WARNING! The database contains {len(csv_df[self.args.y])} datapoints, the 90% training size will be excluded (too few validation points to reach a reliable result). You can include this size using "--filter_train False".')

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
                    _ = hyperopt_workflow(self, csv_df_hyp, ML_model, size, Xy_data_hyp, seed)

                    # apply the PFI descriptor filter if it is activated
                    if self.args.pfi_filter:
                        try:
                            _ = PFI_workflow(self, csv_df_PFI, ML_model, size, Xy_data_hyp, seed)
                        except (FileNotFoundError,ValueError): # in case the model/train/seed combination failed
                            pass

                    cycle += 1

                # only select best seed for each train/model combination
                name_csv = self.args.destination.joinpath(f"Raw_data/No_PFI/{ML_model}_{size}")
                _ = filter_seed(self, name_csv)
                if self.args.pfi_filter:
                    name_csv_pfi = self.args.destination.joinpath(f"Raw_data/PFI/{ML_model}_{size}")
                    _ = filter_seed(self, name_csv_pfi)

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
