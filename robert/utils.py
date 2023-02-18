######################################################.
#          This file stores functions used           #
#                in multiple modules                 #
######################################################.

import os
import sys
import time
import getopt
import glob
import yaml
import ast
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from robert.argument_parser import set_options, var_dict
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression


robert_version = "0.0.1"
time_run = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
robert_ref = f"ROBERT v {robert_version}, Alegre-Requena, J. V.; Dalmau, D., 2023. https://github.com/jvalegre/robert"


# load paramters from yaml file
def load_from_yaml(self):
    """
    Loads the parameters for the calculation from a yaml if specified. Otherwise
    does nothing.
    """

    txt_yaml = f"\no  Importing ROBERT parameters from {self.varfile}"
    error_yaml = False
    # Variables will be updated from YAML file
    try:
        if os.path.exists(self.varfile):
            if os.path.splitext(self.varfile)[1] in [".yaml", ".yml", ".txt"]:
                with open(self.varfile, "r") as file:
                    try:
                        param_list = yaml.load(file, Loader=yaml.SafeLoader)
                    except yaml.scanner.ScannerError:
                        txt_yaml = f'\nx  Error while reading {self.varfile}. Edit the yaml file and try again (i.e. use ":" instead of "=" to specify variables)'
                        error_yaml = True
        if not error_yaml:
            for param in param_list:
                if hasattr(self, param):
                    if getattr(self, param) != param_list[param]:
                        setattr(self, param, param_list[param])

    except UnboundLocalError:
        txt_yaml = "\nx  The specified yaml file containing parameters was not found! Make sure that the valid params file is in the folder where you are running the code."

    return self, txt_yaml


# class for logging
class Logger:
    """
    Class that wraps a file object to abstract the logging.
    """

    # Class Logger to writargs.input.split('.')[0] output to a file
    def __init__(self, filein, append, suffix="dat"):
        self.log = open(f"{filein}_{append}.{suffix}", "w")

    def write(self, message):
        """
        Appends a newline character to the message and writes it into the file.

        Parameters
        ----------
        message : str
           Text to be written in the log file.
        """
        self.log.write(f"{message}\n")
        print(f"{message}\n")

    def fatal(self, message):
        """
        Writes the message to the file. Closes the file and raises an error exit

        Parameters
        ----------
        message : str
           text to be written in the log file.
        """
        self.write(message)
        self.finalize()
        raise SystemExit(1)

    def finalize(self):
        """
        Closes the file
        """
        self.log.close()


def move_file(destination, source, file):
    """
    Moves files from the source folder to the destination folder and creates
    the destination folders when needed.

    Parameters
    ----------
    destination : str
        Path to the destination folder
    src : str
        Path to the source folder
    file : str
        Full name of the file (file + extension)
    """

    destination.mkdir(exist_ok=True, parents=True)
    filepath = source / file
    try:
        filepath.rename(destination / file)
    except FileExistsError:
        filepath.replace(destination / file)


def command_line_args():
    """
    Load default and user-defined arguments specified through command lines. Arrguments are loaded as a dictionary
    """

    # First, create dictionary with user-defined arguments
    kwargs = {}
    available_args = ["help"]
    bool_args = [
        "curate",
        "generate",
        "outliers",
        "predict",
    ]

    for arg in var_dict:
        if arg in bool_args:
            available_args.append(f"{arg}")
        else:
            available_args.append(f"{arg} =")

    try:
        opts, _ = getopt.getopt(sys.argv[1:], "h", available_args)
    except getopt.GetoptError as err:
        print(err)
        sys.exit()

    for arg, value in opts:
        if arg.find("--") > -1:
            arg_name = arg.split("--")[1].strip()
        elif arg.find("-") > -1:
            arg_name = arg.split("-")[1].strip()
        if arg_name in bool_args:
            value = True
        if value == "None":
            value = None
        if arg_name in ("h", "help"):
            print(f"o  ROBERT v {robert_version} is installed correctly! For more information about the available options, see the documentation in https://github.com/jvalegre/robert")
            sys.exit()
        else:
            # this "if" allows to use * to select multiple files in multiple OS
            if arg_name.lower() == 'files' and value.find('*') > -1:
                kwargs[arg_name] = glob.glob(value)
            else:
                # this converts the string parameters to lists
                if arg_name.lower() in ["discard","ignore","train","model"]:
                    if not isinstance(value, list):
                        try:
                            value = ast.literal_eval(value)
                        except (SyntaxError, ValueError):
                            # this line fixes issues when using "[X]" or ["X"] instead of "['X']" when using lists
                            value = value.replace('[',']').replace(',',']').split(']')
                            while('' in value):
                                value.remove('')
                kwargs[arg_name] = value

    # Second, load all the default variables as an "add_option" object
    args = load_variables(kwargs, "command")

    return args


def load_variables(kwargs, robert_module, create_dat=True):
    """
    Load default and user-defined variables
    """

    # first, load default values and options manually added to the function
    self = set_options(kwargs)

    # this part loads variables from yaml files (if varfile is used)
    txt_yaml = ""
    if self.varfile is not None:
        self, txt_yaml = load_from_yaml(self)
    if robert_module != "command":
        self.initial_dir = Path(os.getcwd())

        # start a log file
        if create_dat:
            logger_1 = 'ROBERT'
            logger_1, logger_2 = robert_module.upper(), "data"

            if txt_yaml not in [
                "",
                f"\no  Importing ROBERT parameters from {self.varfile}",
                "\nx  The specified yaml file containing parameters was not found! Make sure that the valid params file is in the folder where you are running the code.\n",
            ]:
                self.log = Logger(self.initial_dir / logger_1, logger_2)
                self.log.write(txt_yaml)
                self.log.finalize()
                os.chdir(self.initial_dir)
                sys.exit()

            if not self.command_line:
                self.log = Logger(self.initial_dir / logger_1, logger_2)
            else:
                # prevents errors when using command lines and running to remote directories
                path_command = Path(f"{os.getcwd()}")
                self.log = Logger(path_command / logger_1, logger_2)

            self.log.write(f"ROBERT v {robert_version} {time_run} \nCitation: {robert_ref}\n")

            if self.command_line:
                self.log.write(f"Command line used in ROBERT: robert {' '.join([str(elem) for elem in sys.argv[1:]])}\n")

    return self


def destination_folder(self,dest_module):
    if self.args.destination is None:
        self.curate_folder = Path(self.args.initial_dir).joinpath(dest_module)
    else:
        if Path(f"{self.args.destination}").exists():
            self.curate_folder = Path(self.args.destination)
        else:
            self.curate_folder = Path(self.args.initial_dir).joinpath(self.args.destination)

    self.curate_folder.mkdir(exist_ok=True, parents=True)


def sanity_checks(self, type_checks, module, columns_csv):
    """
    Check that different variables are set correctly
    """

    curate_valid = True
    if type_checks == 'initial':
        if self.args.csv_name is '':
            self.args.log.write('\nx  Specify the name of your CSV file with the csv_name option!')
            curate_valid = False

        elif not os.path.exists(self.args.csv_name):
            self.args.log.write(f'\nx  The path of your CSV file doesn\'t exist! You specified: {self.args.csv_name}')
            curate_valid = False
            
        if self.args.y == '':
            self.args.log.write(f"\nx  Specify a y value (column name) with the y option! (i.e. y='solubility')")
            curate_valid = False

        if module == 'curate':
            if self.args.categorical.lower() not in ['onehot','numbers']:
                self.args.log.write(f"\nx  The categorical option used is not valid! Options: 'onehot', 'numbers'")
                curate_valid = False

            elif float(self.args.thres_x) > 1 or float(self.args.thres_x) < 0:
                self.args.log.write(f"\nx  The thres_x option should be between 0 and 1!")
                curate_valid = False

            elif float(self.args.thres_y) > 1 or float(self.args.thres_y) < 0:
                self.args.log.write(f"\nx  The thres_y option should be between 0 and 1!")
                curate_valid = False
        
        elif module == 'generate':
            if self.args.split.lower() not in ['kn','rnd']:
                self.args.log.write(f"\nx  The split option used is not valid! Options: 'KN', 'RND'")
                curate_valid = False

            for model in self.args.model:
                if model.lower() not in ['rf','mvl','gb','adab','nn','vr']:
                    self.args.log.write(f"\nx  The model option used is not valid! Options: 'RF', 'MVL', 'GB', 'AdaB', 'NN', 'VR'")
                    curate_valid = False

            if len(self.args.model.lower()) == 0:
                self.args.log.write(f"\nx  Choose an ML model in the model option!")
                curate_valid = False

            if len(self.args.train.lower()) == 0:
                self.args.log.write(f"\nx  Choose train proportion(s) in the train option!")
                curate_valid = False

            if self.args.mode.lower() not in ['reg','clas']:
                self.args.log.write(f"\nx  The mode option used is not valid! Options: 'reg', 'clas'")
                curate_valid = False

            if int(self.args.epochs) <= 0:
                self.args.log.write(f"\nx  The number of epochs must be higher than 0!")
                curate_valid = False
            
            if self.args.mode.lower() == 'reg' and self.args.hyperopt_target not in ['rmse','mae','r2']:
                self.args.log.write(f"\nx  The hyperopt_target option is not valid! Options for regression: 'rmse', 'mae', 'r2'")
                curate_valid = False

            if self.args.mode.lower() == 'clas' and self.args.hyperopt_target not in ['mcc','f1_score','acc']:
                self.args.log.write(f"\nx  The hyperopt_target option is not valid! Options for classification: 'mcc', 'f1_score', 'acc'")
                curate_valid = False

            if int(self.args.PFI_epochs) <= 0:
                self.args.log.write(f"\nx  The number of PFI_epochs must be higher than 0!")
                curate_valid = False

    elif type_checks == 'csv_db':
        if self.args.y not in columns_csv:
            self.args.log.write(f"\nx  The y option specified ({self.args.y}) is not a columnd in the csv selected ({self.args.csv_name})!")
            curate_valid = False

        for val in self.args.discard:
            if val not in columns_csv:
                self.args.log.write(f"\nx  Descriptor {val} specified in the discard option is not a columnd in the csv selected ({self.args.csv_name})!")
                curate_valid = False

        for val in self.args.ignore:
            if val not in columns_csv:
                self.args.log.write(f"\nx  Descriptor {val} specified in the ignore option is not a columnd in the csv selected ({self.args.csv_name})!")
                curate_valid = False

    if not curate_valid:
        self.args.log.finalize()
        sys.exit()


def load_database(self,module):
    csv_df = pd.read_csv(self.args.csv_name)
    sanity_checks(self,'csv_db',module,csv_df.columns)
    csv_df = csv_df.drop(self.args.discard, axis=1)
    total_amount = len(csv_df.columns)
    ignored_descs = len(self.args.ignore)
    accepted_descs = total_amount - ignored_descs
    txt_load = f'\no  Database {self.args.csv_name} loaded successfully, including:'
    txt_load += f'\n   - {len(csv_df[self.args.y])} datapoints'
    txt_load += f'\n   - {accepted_descs} accepted descriptors'
    txt_load += f'\n   - {ignored_descs} ignored descriptors'
    txt_load += f'\n   - {len(self.args.discard)} discarded descriptors'
    self.args.log.write(txt_load)

    return csv_df


def create_folders(folder_names):
    for folder in folder_names:
        try:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
        except Exception as e:
            print(f'Error while deleting folder "{folder}": {e}')


def standardize(X_train,X_valid):
    
    # standardizes the data sets using the mean and standard dev from the train set
    Xmean = X_train.mean(axis=0)
    Xstd = X_train.std(axis=0)
    X_train_scaled = (X_train - Xmean) / Xstd
    X_valid_scaled = (X_valid - Xmean) / Xstd

    return X_train_scaled, X_valid_scaled


def load_model(params, model_data):

    # load regressor models
    if model_data['mode'] == 'reg':
        loaded_model = load_model_reg(params, model_data)

    # load classifier models
    elif model_data['mode'] == 'clas':
        loaded_model = load_model_clas(params, model_data)

    return loaded_model


def load_model_reg(params, model_data):
    if model_data['model'] == 'RF':     
        loaded_model = RandomForestRegressor(max_depth=params['max_depth'],
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                random_state=model_data['seed'])

    elif model_data['model']  == 'GB':    
        loaded_model = GradientBoostingRegressor(max_depth=params['max_depth'], 
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                learning_rate=params['learning_rate'],
                                validation_fraction=params['validation_fraction'],
                                random_state=model_data['seed'])

    elif model_data['model']  == 'AdaB':
        loaded_model = AdaBoostRegressor(n_estimators=params['n_estimators'],
                                learning_rate=params['learning_rate'],
                                random_state=model_data['seed'])

    elif model_data['model']  == 'NN':
        loaded_model = MLPRegressor(batch_size=params['batch_size'],
                                hidden_layer_sizes=params['hidden_layer_sizes'],
                                learning_rate_init=params['learning_rate_init'],
                                max_iter=params['max_iter'],
                                validation_fraction=params['validation_fraction'],
                                random_state=model_data['seed'])                    
            
    elif model_data['model']  == 'VR':
        r1 = GradientBoostingRegressor(max_depth=params['max_depth'], 
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                learning_rate=params['learning_rate'],
                                validation_fraction=params['validation_fraction'],
                                random_state=model_data['seed'])
        r2 = RandomForestRegressor(max_depth=params['max_depth'],
                            max_features=params['max_features'],
                            n_estimators=params['n_estimators'],
                            random_state=model_data['seed'])
        r3 = MLPRegressor(batch_size=params['batch_size'],
                                hidden_layer_sizes=params['hidden_layer_sizes'],
                                learning_rate_init=params['learning_rate_init'],
                                max_iter=params['max_iter'],
                                validation_fraction=params['validation_fraction'],
                                random_state=model_data['seed'])
        loaded_model = VotingRegressor([('gb', r1), ('rf', r2), ('nn', r3)])

    elif model_data['model']  == 'MVL':
        loaded_model = LinearRegression(n_features_in_=params['max_features'])

    return loaded_model


def load_model_clas(params, model_data):

    if model_data['model']  == 'RF':     
        loaded_model = RandomForestClassifier(max_depth=params['max_depth'],
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                random_state=model_data['seed'])

    elif model_data['model']  == 'GB':    
        loaded_model = GradientBoostingClassifier(max_depth=params['max_depth'], 
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                learning_rate=params['learning_rate'],
                                validation_fraction=params['validation_fraction'],
                                random_state=model_data['seed'])

    elif model_data['model']  == 'AdaB':
            loaded_model = AdaBoostClassifier(n_estimators=params['n_estimators'],
                                    learning_rate=params['learning_rate'],
                                    random_state=model_data['seed'])

    elif model_data['model']  == 'NN':
        loaded_model = MLPClassifier(batch_size=params['batch_size'],
                                hidden_layer_sizes=params['hidden_layer_sizes'],
                                learning_rate_init=params['learning_rate_init'],
                                max_iter=params['max_iter'],
                                validation_fraction=params['validation_fraction'],
                                random_state=model_data['seed'])

    elif model_data['model']  == 'VR':
        r1 = GradientBoostingClassifier(max_depth=params['max_depth'], 
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                learning_rate=params['learning_rate'],
                                validation_fraction=params['validation_fraction'],
                                random_state=model_data['seed'])
        r2 = RandomForestClassifier(max_depth=params['max_depth'],
                            max_features=params['max_features'],
                            n_estimators=params['n_estimators'],
                            random_state=model_data['seed'])
        r3 = MLPClassifier(batch_size=params['batch_size'],
                                hidden_layer_sizes=params['hidden_layer_sizes'],
                                learning_rate_init=params['learning_rate_init'],
                                max_iter=params['max_iter'],
                                validation_fraction=params['validation_fraction'],
                                random_state=model_data['seed'])

        loaded_model = VotingClassifier([('gb', r1), ('rf', r2), ('nn', r3)])

    elif model_data['model']  == 'MVL':
        print('Multivariate linear models (model = \'MVL\') are not compatible with classifiers (mode = \'clas\')')
        sys.exit()

    return loaded_model


# function to get stats from the model
def model_stats(y,y_pred):   
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    _, _, r_value, _, _ = stats.linregress(y, y_pred)
    r2 = r_value**2

    return r2,mae,rmse