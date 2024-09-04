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
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
from scipy import stats
# for users with no intel architectures. This part has to be before the sklearn imports
try:
    from sklearnex import patch_sklearn
    patch_sklearn(verbose=False)
except (ModuleNotFoundError,ImportError):
    pass
from pkg_resources import resource_filename
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
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
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression
from mapie.regression import MapieRegressor
from mapie.conformity_scores import AbsoluteConformityScore
import warnings # this avoids warnings from sklearn
warnings.filterwarnings("ignore")

robert_version = "1.2.0"
time_run = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
robert_ref = "Dalmau, D.; Alegre Requena, J. V. ChemRxiv, 2023, DOI: 10.26434/chemrxiv-2023-k994h"


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
            if os.path.basename(Path(self.varfile)).split('.')[1] in ["yaml", "yml", "txt"]:
                with open(self.varfile, "r") as file:
                    try:
                        param_list = yaml.load(file, Loader=yaml.SafeLoader)
                    except (yaml.scanner.ScannerError,yaml.parser.ParserError):
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

    def finalize(self):
        """
        Closes the file
        """
        self.log.close()


def command_line_args(exe_type,sys_args):
    """
    Load default and user-defined arguments specified through command lines. Arrguments are loaded as a dictionary
    """

    if exe_type == 'exe':
        sys.argv = sys_args
    
    # First, create dictionary with user-defined arguments
    kwargs = {}
    available_args = ["help"]
    bool_args = [
        "curate",
        "generate",
        "verify",
        "predict",
        "aqme",
        "report",
        "cheers",
        "evaluate"
    ]
    list_args = [
        "discard",
        "ignore",
        "train",
        "model",
        "report_modules",
        "seed"
    ]
    int_args = [
        'pfi_epochs',
        'epochs',
        'nprocs',
        'pfi_max',
        'kfold',
        'shap_show',
        'pfi_show',
    ]
    float_args = [
        'pfi_threshold',
        'thres_test',
        't_value',
        'thres_x',
        'thres_y',
        'test_set',
        'desc_thres',
        'alpha',
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

        if arg_name in ("h", "help"):
            print(f"""\n
###########################################################################################
#                                                                                         #
#     ROBERT v {robert_version} is installed correctly, thanks for using the program!     #
#                                                                                         #
###########################################################################################


o How to run a full workflow with ROBERT in the command line?
-------------------------------------------------------------

* From a CSV database: python -m robert --ignore "[COL1,COL2,etc]" --names "names_COL" --y "y_COL" --csv_name "FILENAME.csv"
* From a CSV with SMILES: python -m robert --aqme --y "y_COL" --csv_name "FILENAME.csv"


o Required options:
-------------------

--ignore "[COL1,COL2,etc]" (default=[]) : CSV columns that will be ignored (i.e., names, ID, etc.)
--names "names_COL" (default="") : CSV columns containing the names of the datapoints
--y "y_COL" (default="") : CSV column containing the y values
--csv_name "FILENAME.csv" (default="") : name of the input CSV


o Other common options:
-----------------------

* General:
  --csv_test "FILENAME.csv" (default="") : name of the CSV containing the external test set
  --discard "[COL1,COL2,etc]" (default=[]) : CSV columns that will be removed

* Affecting data curation in CURATE:
  --kfold INT (default='auto') : number of folds for k-fold cross-validation of the RFECV feature selector. If 'auto', the program does a LOOCV for databases with less than 50 points, and 5-fold CV for larger databases 
  --categorical "onehot" or "numbers" (default="onehot") : type of conversion for categorical variables
  --corr_filter BOOL (default=True) : disable the correlation filter

* Affecting model screening in GENERATE:
  --train "[SIZE1,SIZE2,etc]" (default=[60,70,80,90]) : training set % sizes to use in the ML scan (i.e., "[60,70]")
  --model "[MODEL1,MODEL2,etc]" (default=["RF","GB","NN","MVL"]) : ML models to use in the ML scan (i.e., "[RF,GB]")
  --type "reg" or "clas" (default="reg") : regression or classification models
  --generate_acc "low", "mid" or "high" (default="mid") : use more or less epochs and seeds during model hyperoptimization
  --pfi_max INT (default=0) : number of features to keep in the PFI models

* Affecting tests, VERIFY:
  --kfold INT (default='auto') : number of folds for k-fold cross-validation. If 'auto', the program does a LOOCV for databases with less than 50 points, and 5-fold CV for larger databases 
  --thres_test FLOAT (default=0.25) : threshold to determine whether tests pass

* Affecting predictions, PREDICT:
  --t_value INT (default=2) : t-value threshold to identify outliers
  --shap_show INT (default=10) : maximum number of descriptors shown in the SHAP plot

* Affecting SMILES workflows, AQME:
  --qdescp_keywords STR (default="") : extra keywords in QDESCP (i.e. "--qdescp_atoms [Ir] --alpb h2o") 
  --csearch_keywords STR (default="--sample 50") : extra keywords in CSEARCH


o How to cite ROBERT:
---------------------

{robert_ref}


o Complete documentation:
-------------------------

For more information, see the complete documentation in https://robert.readthedocs.io""")
            sys.exit()
        else:
            # this "if" allows to use * to select multiple files in multiple OS
            if arg_name.lower() == 'files' and value.find('*') > -1:
                kwargs[arg_name] = glob.glob(value)
            else:
                # converts the string parameters from command line to the right format
                if arg_name in bool_args:
                    value = True                    
                elif arg_name.lower() in list_args:
                    value = format_lists(value)
                elif arg_name.lower() in int_args:
                    value = int(value)
                elif arg_name.lower() in float_args:
                    value = float(value)
                elif value == "None":
                    value = None
                elif value == "False":
                    value = False
                elif value == "True":
                    value = True

                kwargs[arg_name] = value

    # Second, load all the default variables as an "add_option" object
    args = load_variables(kwargs, "command")

    return args


def format_lists(value):
    '''
    Transforms strings into a list
    '''

    if not isinstance(value, list):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            # this line fixes issues when using "[X]" or ["X"] instead of "['X']" when using lists
            value = value.replace('[',']').replace(',',']').replace("'",']').split(']')
            while('' in value):
                value.remove('')
    return value


def load_variables(kwargs, robert_module):
    """
    Load default and user-defined variables
    """

    # first, load default values and options manually added to the function
    self = set_options(kwargs)

    # this part loads variables from yaml files (if varfile is used)
    txt_yaml = ""
    if self.varfile is not None:
        self, txt_yaml = load_from_yaml(self)

    # check if user used .csv in csv_name
    if not os.path.exists(f"{self.csv_name}") and os.path.exists(f'{self.csv_name}.csv'):
        self.csv_name = f'{self.csv_name}.csv'

    if robert_module != "command":
        self.initial_dir = Path(os.getcwd())

        # adds --names to --ignore
        if self.names not in self.ignore and self.names != '':
            self.ignore.append(self.names)

        # creates destination folder
        if robert_module.upper() != 'REPORT':
            self = destination_folder(self,robert_module)

            # start a log file
            logger_1 = 'ROBERT'
            logger_1, logger_2 = robert_module.upper(), "data"

            if txt_yaml not in [
                "",
                f"\no  Importing ROBERT parameters from {self.varfile}",
                "\nx  The specified yaml file containing parameters was not found! Make sure that the valid params file is in the folder where you are running the code.\n",
            ]:
                self.log = Logger(self.destination / logger_1, logger_2)
                self.log.write(txt_yaml)
                self.log.finalize()
                sys.exit()

            self.log = Logger(self.destination / logger_1, logger_2)
            self.log.write(f"ROBERT v {robert_version} {time_run} \nHow to cite: {robert_ref}\n")

            if self.command_line:
                cmd_print = ''
                cmd_args = sys.argv[1:]
                if self.extra_cmd != '':
                    for arg in self.extra_cmd.split():
                        cmd_args.append(arg)
                for i,elem in enumerate(cmd_args):
                    if elem[0] in ['"',"'"]:
                        elem = elem[1:]
                    if elem[-1] in ['"',"'"]:
                        elem = elem[:-1]
                    if elem != '-h' and elem.split('--')[-1] not in var_dict:
                        # parse single elements of the list as strings (otherwise the commands cannot be reproduced)
                        if '--qdescp_atoms' in elem:
                            new_arg = []
                            list_qdescp = elem.replace(', ',',').replace(' ,',',').split()
                            for j,qdescp_elem in enumerate(list_qdescp):
                                if list_qdescp[j-1] == '--qdescp_atoms':
                                    qdescp_elem = qdescp_elem[1:-1]
                                    new_elem = []
                                    for smarts_strings in qdescp_elem.split(','):
                                        new_elem.append(f'{smarts_strings}'.replace("'",''))
                                    new_arg.append(f'{new_elem}'.replace(" ",""))
                                else:
                                    new_arg.append(qdescp_elem)
                            new_arg = ' '.join(new_arg)
                            elem = new_arg
                        if cmd_args[i-1].split('--')[-1] in var_dict: # check if the previous word is an arg
                            cmd_print += f'"{elem}'
                        if i == len(cmd_args)-1 or cmd_args[i+1].split('--')[-1] in var_dict: # check if the next word is an arg, or last word in command
                            cmd_print += f'"'
                    else:
                        cmd_print += f'{elem}'
                    if i != len(cmd_args)-1:
                        cmd_print += ' '

                self.log.write(f"Command line used in ROBERT: python -m robert {cmd_print}\n")

        elif robert_module.upper() == 'REPORT':
            self.path_icons = Path(resource_filename("robert", "report"))

        # using or not the intelex accelerator might affect the results
        if robert_module.upper() in ['GENERATE','VERIFY','PREDICT']:
            try:
                from sklearnex import patch_sklearn
                pass
            except (ModuleNotFoundError,ImportError):
                self.log.write(f"\nWARNING! The scikit-learn-intelex accelerator is not installed, the results might vary if it is installed and the execution times might become much longer (if available, use 'pip install scikit-learn-intelex')")

        if robert_module.upper() in ['GENERATE', 'VERIFY']:
            # adjust the default value of error_type for classification
            if self.type.lower() == 'clas':
                if self.error_type not in ['acc', 'mcc', 'f1']:
                    self.error_type = 'mcc'

        if robert_module.upper() in ['PREDICT','VERIFY','REPORT']:
            if self.params_dir == '':
                self.params_dir = 'GENERATE/Best_model'

        if robert_module.upper() == 'CURATE':
            self.log.write(f"\no  Starting data curation with the CURATE module")

        elif robert_module.upper() == 'GENERATE':
            self.log.write(f"\no  Starting generation of ML models with the GENERATE module")
            
            # Check if the folders exist and if they do, delete and replace them
            folder_names = [self.initial_dir.joinpath('GENERATE/Best_model/No_PFI'), self.initial_dir.joinpath('GENERATE/Raw_data/No_PFI')]
            if self.pfi_filter:
                folder_names.append(self.initial_dir.joinpath('GENERATE/Best_model/PFI'))
                folder_names.append(self.initial_dir.joinpath('GENERATE/Raw_data/PFI'))
            _ = create_folders(folder_names)

            for i,val in enumerate(self.train):
                self.train[i] = int(val)

            # set seeds and epochs depending on precision
            if len(self.seed) == 0:
                if self.generate_acc == 'low':
                    self.seed = [0,8,19]
                elif self.generate_acc == 'mid':
                    self.seed = [0,8,19,43,70,233]
                elif self.generate_acc == 'high':
                    self.seed = [0,8,19,43,70,233,1989,9999,20394,3948301]
            
            if self.epochs == 0:
                if self.generate_acc == 'low':
                    self.epochs = 20
                elif self.generate_acc == 'mid':
                    self.epochs = 40
                elif self.generate_acc == 'high':
                    self.epochs = 100
        
            if self.type.lower() == 'clas':
                if self.model == ['RF','GB','NN','MVL']:
                    self.model = ['RF','GB','NN','AdaB']
            
            models_gen = [] # use capital letters in all the models
            for model_type in self.model:
                models_gen.append(model_type.upper())
            self.model = models_gen

            # if there are missing options, look for them from a previous CURATE job (if any)
            options_dict = {
                'y': self.y,
                'names': self.names,
                'ignore': self.ignore,
                'csv_name': self.csv_name
            }
            curate_folder = Path(self.initial_dir).joinpath('CURATE')
            curate_csv = f'{curate_folder}/CURATE_options.csv'
            if os.path.exists(curate_csv):
                curate_df = pd.read_csv(curate_csv, encoding='utf-8')

                for option in options_dict:
                    if options_dict[option] == '':
                        if option == 'y':
                            self.y = curate_df['y'][0]
                        elif option == 'names':
                            self.names = curate_df['names'][0]
                        elif option == 'ignore':
                            self.ignore = curate_df['ignore'][0]
                            self.ignore  = format_lists(self.ignore)
                        elif option == 'csv_name':
                            self.csv_name = curate_df['csv_name'][0]

        elif robert_module.upper() == 'VERIFY':
            self.log.write(f"\no  Starting tests to verify the prediction ability of the ML models with the VERIFY module")

        elif robert_module.upper() == 'PREDICT':
            self.log.write(f"\no  Representation of predictions and analysis of ML models with the PREDICT module")
            if self.names == '':
                # tries to get names from GENERATE
                if 'GENERATE/Best_model' in self.params_dir:
                    params_dirs = [f'{self.params_dir}/No_PFI',f'{self.params_dir}/PFI']
                else:
                    params_dirs = [self.params_dir]
                _, params_df, _, _, _ = load_db_n_params(self,params_dirs[0],"verify",False)
                self.names = params_df["names"][0]

        elif robert_module.upper() in ['AQME', 'AQME_TEST']: 
            # Check if the csv has 2 columns named smiles or smiles_Suffix. The file is read as text because pandas assigns automatically
            # .1 to duplicate columns. (i.e. SMILES and SMILES.1 if there are two columns named SMILES)
            unique_columns=[]
            with open(self.csv_name, 'r') as datfile:
                lines = datfile.readlines()
                for column in lines[0].split(','):
                    if column in unique_columns:
                        print(f"\nWARNING! The CSV file contains duplicate columns ({column}). Please, rename or remove these columns. If you want to use more than one SMILES column, use _Suffix (i.e. SMILES_1, SMILES_2, ...)")
                        sys.exit()
                    else:
                        unique_columns.append(column)
            
            # Check if there is a column with the name "smiles" or "smiles_" followed by any characters
            if not any(col.lower().startswith("smiles") for col in unique_columns):
                print("\nWARNING! The CSV file does not contain a column with the name 'smiles' or a column starting with 'smiles_'. Please make sure the column exists.")
                sys.exit()

            # Check if there are duplicate names in code_names in the csv file.
            df = pd.read_csv(self.csv_name, encoding='utf-8')
            unique_entries=[]
            for entry in df['code_name']:
                if entry in unique_entries:
                    print(f"\nWARNING! The code_name column in the CSV file contains duplicate entries ({entry}). Please, rename or remove these entries.")
                    sys.exit()
                else:
                    unique_entries.append(entry)

            self.log.write(f"\no  Starting the generation of AQME descriptors with the AQME module")

        # initial sanity checks
        if robert_module.upper() != 'REPORT':
            _ = sanity_checks(self, 'initial', robert_module, None)

    return self

def destination_folder(self,dest_module):
    if self.destination is None:
        self.destination = Path(self.initial_dir).joinpath(dest_module.upper())
    else:
        self.log.write(f"\nx  The destination option has not been implemented yet! Please, remove it from your input and stay tuned.")
        sys.exit()
        # this part does not work for know
        # if Path(f"{self.destination}").exists():
        #     self.destination = Path(self.destination)
        # else:
        #     self.destination = Path(self.initial_dir).joinpath(self.destination)

    if os.path.exists(self.destination):
        shutil.rmtree(self.destination)
    self.destination.mkdir(exist_ok=True, parents=True)

    return self


def missing_inputs(self,module,print_err=False):
    """
    Gives the option to input missing variables in the terminal
    """

    if module.lower() not in ['predict','verify','report','aqme_test']:
        if module.lower() == 'evaluate':
            if self.csv_train == '':
                self = check_csv_option(self,'csv_train',print_err)
            if self.csv_valid == '':
                self = check_csv_option(self,'csv_valid',print_err)

        else:
            if self.csv_name == '':
                self = check_csv_option(self,'csv_name',print_err)

    if module.lower() not in ['predict','verify','report','aqme_test']:
        if self.y == '':
            if print_err:
                print(f'\nx  Specify a y value (column name) with the y option! (i.e. y="solubility")')
            else:
                self.log.write(f'\nx  Specify a y value (column name) with the y option! (i.e. y="solubility")')
            self.y = input('Enter the column with y values: ')
            self.extra_cmd += f' --y {self.y}'
            if not print_err:
                self.log.write(f"   -  y option set to {self.y} by the user")

    if module.lower() in ['full_workflow','predict','curate','generate','evaluate']:
        if self.names == '':
            if print_err:
                print(f'\nx  Specify the column with the entry names! (i.e. names="code_name")')
            else:
                self.log.write(f'\nx  Specify the column with the entry names! (i.e. names="code_name")')
            self.names = input('Enter the column with the entry names: ')
            self.extra_cmd += f' --names {self.names}'
            if not print_err:
                self.log.write(f"   -  names option set to {self.names} by the user")
        if self.names != '' and self.names not in self.ignore:
            self.ignore.append(self.names)

    return self


def check_csv_option(self,csv_option,print_err):
    '''
    Checks missing values in input CSV options
    '''
    
    if csv_option == 'csv_name':
        line_print = f'\nx  Specify the CSV name for the {csv_option} option!'
    elif csv_option == 'csv_train':
        line_print = f'\nx  Specify the CSV name containing the TRAINING set!'
    elif csv_option == 'csv_valid':
        line_print = f'\nx  Specify the CSV name containing the VALIDATION set!'

    if print_err:
        print(line_print)
    else:
        self.log.write(line_print)
    val_option = input('Enter the name of your CSV file: ')
    self.extra_cmd += f' --{csv_option} {val_option}'
    if not print_err:
        self.log.write(f"   -  {csv_option} option set to {val_option} by the user")

    if csv_option == 'csv_name':
        self.csv_name = val_option    
    elif csv_option == 'csv_train':
        self.csv_train = val_option
    elif csv_option == 'csv_valid':
        self.csv_valid = val_option

    return self


def sanity_checks(self, type_checks, module, columns_csv):
    """
    Check that different variables are set correctly
    """

    curate_valid = True
    # adds manual inputs missing from the command line
    self = missing_inputs(self,module)

    if module.lower() == 'evaluate':
        curate_valid = locate_csv(self,self.csv_train,'csv_train',curate_valid)
        curate_valid = locate_csv(self,self.csv_valid,'csv_valid',curate_valid)
        if self.csv_test != '':
            curate_valid = locate_csv(self,self.csv_test,'csv_test',curate_valid)

        if self.eval_model.lower() not in ['mvl']:
            self.log.write(f"\nx  The eval_model option used is not valid! Options: 'MVL' (more options will be added soon)")
            curate_valid = False

        if self.type.lower() not in ['reg']:
            self.log.write(f"\nx  The type option used is not valid in EVALUATE! Options: 'reg' (the 'clas' option will be added soon)")
            curate_valid = False

    elif type_checks == 'initial' and module.lower() not in ['verify','predict']:

        curate_valid = locate_csv(self,self.csv_name,'csv_name',curate_valid)

        if module.lower() == 'curate':
            if self.categorical.lower() not in ['onehot','numbers']:
                self.log.write(f"\nx  The categorical option used is not valid! Options: 'onehot', 'numbers'")
                curate_valid = False

            for thres,thres_name in zip([self.thres_x,self.thres_y],['thres_x','thres_y']):
                if float(thres) > 1 or float(thres) < 0:
                    self.log.write(f"\nx  The {thres_name} option should be between 0 and 1!")
                    curate_valid = False
        
        elif module.lower() == 'generate':
            if self.split.lower() not in ['kn','rnd']:
                self.log.write(f"\nx  The split option used is not valid! Options: 'KN', 'RND'")
                curate_valid = False

            if self.generate_acc.lower() not in ['low','mid','high']:
                self.log.write(f"\nx  The generate_acc option used is not valid! Options: 'low', 'mid', 'high'")
                curate_valid = False

            for model_type in self.model:
                if model_type.upper() not in ['RF','MVL','GB','GP','ADAB','NN','VR'] or len(self.model) == 0:
                    self.log.write(f"\nx  The model option used is not valid! Options: 'RF', 'MVL', 'GB', 'ADAB', 'NN', 'VR'")
                    curate_valid = False
                if model_type.upper() == 'MVL' and self.type.lower() == 'clas':
                    self.log.write(f"\nx  Multivariate linear models (MVL in the model_type option) are not compatible with classificaton!")                 
                    curate_valid = False

            for option,option_name in zip([self.model,self.train],['model','train']):
                if len(option) == 0:
                    self.log.write(f"\nx  Add parameters to the {option_name} option!")
                    curate_valid = False

            if self.type.lower() not in ['reg','clas']:
                self.log.write(f"\nx  The type option used is not valid! Options: 'reg', 'clas'")
                curate_valid = False

            for option,option_name in zip([self.epochs,self.pfi_epochs],['epochs','pfi_epochs']):
                if option <= 0:
                    self.log.write(f"\nx  The number of {option_name} must be higher than 0!")
                    curate_valid = False

    if type_checks == 'initial' and module.lower() in ['generate','verify','predict','report']:

        if type_checks == 'initial' and module.lower() in ['generate','verify']:
            if self.type.lower() == 'reg' and self.error_type.lower() not in ['rmse','mae','r2']:
                self.log.write(f"\nx  The error_type option is not valid! Options for regression: 'rmse', 'mae', 'r2'")
                curate_valid = False

            if self.type.lower() == 'clas' and self.error_type.lower() not in ['mcc','f1','acc']:
                self.log.write(f"\nx  The error_type option is not valid! Options for classification: 'mcc', 'f1', 'acc'")
                curate_valid = False
        
        if module.lower() == 'verify':
            if float(self.thres_test) < 0:
                self.log.write(f"\nx  The thres_test should be higher than 0!")
                curate_valid = False

        if module.lower() in ['verify','predict']:
            if os.getcwd() in f"{self.params_dir}":
                path_db = self.params_dir
            else:
                path_db = f"{Path(os.getcwd()).joinpath(self.params_dir)}"

            if not os.path.exists(path_db):
                self.log.write(f'\nx  The path of your CSV files doesn\'t exist! Set the folder containing the two CSV files with 1) the parameters of the model and 2) the Xy database with the params_dir option')
                curate_valid = False

        if module.lower() == 'predict':
            if self.t_value < 0:
                self.log.write(f"\nx  t_value ({self.t_value}) should be higher 0!")
                curate_valid = False

            if self.csv_test != '':
                if os.getcwd() in f"{self.csv_test}":
                    path_test = self.csv_test
                else:
                    path_test = f"{Path(os.getcwd()).joinpath(self.csv_test)}"
                if not os.path.exists(path_test):
                    self.log.write(f'\nx  The path of your CSV file with the test set doesn\'t exist! You specified: {self.csv_test}')
                    curate_valid = False

        if module.lower() == 'report':
            if len(self.report_modules) == 0:
                self.log.write(f'\nx  No modules were provided in the report_modules option! Options: "CURATE", "GENERATE", "VERIFY", "PREDICT"')
                curate_valid = False

            for module in self.report_modules:
                if module.upper() not in ['CURATE','GENERATE','VERIFY','PREDICT','AQME']:
                    self.log.write(f'\nx  Module {module} specified in the report_modules option is not a valid module! Options: "CURATE", "GENERATE", "VERIFY", "PREDICT", "AQME"')
                    curate_valid = False
  
    elif type_checks == 'csv_db':
        if module.lower() != 'predict':
            if self.y not in columns_csv:
                if self.y.lower() in columns_csv: # accounts for upper/lowercase mismatches
                    self.y = self.y.lower()
                elif self.y.upper() in columns_csv:
                    self.y = self.y.upper()
                else:
                    self.log.write(f"\nx  The y option specified ({self.y}) is not a column in the csv selected ({self.csv_name})! If you are using command lines, make sure you add quotation marks like --y \"VALUE\"")
                    curate_valid = False

            for option,option_name in zip([self.discard,self.ignore],['discard','ignore']):
                for val in option:
                    if val not in columns_csv:
                        self.log.write(f"\nx  Descriptor {val} specified in the {option_name} option is not a column in the csv selected ({self.csv_name})!")
                        curate_valid = False

    if not curate_valid:
        self.log.finalize()
        sys.exit()


def locate_csv(self,csv_input,csv_type,curate_valid):
    '''
    Assesses whether the input CSV databases can be located
    '''

    path_csv = ''
    if os.path.exists(f"{csv_input}"):
        path_csv = csv_input
    elif os.path.exists(f"{Path(os.getcwd()).joinpath(csv_input)}"):
        path_csv = f"{Path(os.getcwd()).joinpath(csv_input)}"
    if not os.path.exists(path_csv) or csv_input == '':
        self.log.write(f'\nx  The path of your CSV file doesn\'t exist! You specified: --{csv_type} {csv_input}')
        curate_valid = False
    
    return curate_valid


def load_database(self,csv_load,module):
    '''
    Loads a database in CSV format
    '''
    
    txt_load = ''
    # this part fixes CSV files that use ";" as separator
    with open(csv_load, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    if lines[1].count(';') > 1:
        new_csv_name = os.path.basename(csv_load).split('.csv')[0].split('.CSV')[0]+'_original.csv'
        shutil.move(csv_load, Path(os.path.dirname(csv_load)).joinpath(new_csv_name))
        new_csv_file = open(csv_load, "w")
        for line in lines:
            line = line.replace(',','.')
            line = line.replace(';',',')
            # line = line.replace(':',',')
            new_csv_file.write(line)
        new_csv_file.close()
        txt_load += f'\nx  WARNING! The original database was not a valid CSV (i.e., formatting issues from Microsoft Excel?). A new database using commas as separators was created and used instead, and the original database was stored as {new_csv_name}. To prevent this issue from happening again, you should use commas as separators: https://support.edapp.com/change-csv-separator.\n\n'

    csv_df = pd.read_csv(csv_load, encoding='utf-8')
    # Fill missing values with zeros
    csv_df = csv_df.fillna(0)

    if module.lower() not in ['verify','no_print','evaluate']:
        sanity_checks(self.args,'csv_db',module,csv_df.columns)
        csv_df = csv_df.drop(self.args.discard, axis=1)
        total_amount = len(csv_df.columns)
        ignored_descs = len(self.args.ignore)
        accepted_descs = total_amount - ignored_descs - 1 # the y column is substracted
        if module.lower() not in ['aqme']:
            csv_name = os.path.basename(csv_load)
            if module.lower() not in ['predict']:
                txt_load += f'\no  Database {csv_name} loaded successfully, including:'
                txt_load += f'\n   - {len(csv_df[self.args.y])} datapoints'
                txt_load += f'\n   - {accepted_descs} accepted descriptors'
                txt_load += f'\n   - {ignored_descs} ignored descriptors'
                txt_load += f'\n   - {len(self.args.discard)} discarded descriptors'
            else:
                txt_load += f'\n   o  Test set {csv_name} loaded successfully, including:'
                txt_load += f'\n      - {len(csv_df[csv_df.columns[0]])} datapoints'
            self.args.log.write(txt_load)

    if module.lower() != 'generate':
        return csv_df
    else:
        # ignore user-defined descriptors and assign X and y values (but keeps the original database)
        csv_df_ignore = csv_df.drop(self.args.ignore, axis=1)
        csv_X = csv_df_ignore.drop([self.args.y], axis=1)
        csv_y = csv_df_ignore[self.args.y]
        return csv_df,csv_X,csv_y


def create_folders(folder_names):
    for folder in folder_names:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        folder.mkdir(exist_ok=True, parents=True)


def finish_print(self,start_time,module):
    elapsed_time = round(time.time() - start_time, 2)
    self.args.log.write(f"\nTime {module.upper()}: {elapsed_time} seconds\n")
    self.args.log.finalize()


def standardize(self,X_train,X_valid):
    
    # standardizes the data sets using the mean and standard dev from the train set
    try: # this fails if there are strings as values
        Xmean = X_train.mean(axis=0)
        Xstd = X_train.std(axis=0)
        X_train_scaled = (X_train - Xmean) / Xstd
        X_valid_scaled = (X_valid - Xmean) / Xstd
    except TypeError:
        self.args.log.write(f'   x The standardization process failed! This is probably due to using strings/words as values (use --curate to curate the data first)')
        sys.exit()

    return X_train_scaled, X_valid_scaled


def load_model(params):

    # load regressor models
    if params['type'].lower() == 'reg':
        loaded_model = load_model_reg(params)

    # load classifier models
    elif params['type'].lower() == 'clas':
        loaded_model = load_model_clas(params)

    return loaded_model


def load_model_reg(params):
    if params['model'].upper() in ['RF','VR']:
        loaded_model = RandomForestRegressor(max_depth=params['max_depth'],
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                min_samples_split=params['min_samples_split'],
                                min_samples_leaf=params['min_samples_leaf'],
                                min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                oob_score=params['oob_score'],
                                ccp_alpha=params['ccp_alpha'],
                                max_samples=params['max_samples'],
                                random_state=params['seed'],
                                n_jobs=None)

        if params['model'].upper() == 'VR':
            r1 = loaded_model

    if params['model'].upper() in ['GB','VR']:    
        loaded_model = GradientBoostingRegressor(max_depth=params['max_depth'], 
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                learning_rate=params['learning_rate'],
                                validation_fraction=params['validation_fraction'],
                                subsample=params['subsample'],
                                min_samples_split=params['min_samples_split'],
                                min_samples_leaf=params['min_samples_leaf'],
                                min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                ccp_alpha=params['ccp_alpha'],
                                random_state=params['seed'])

        if params['model'].upper() == 'VR':
            r2 = loaded_model

    if params['model'].upper() in ['NN','VR']:

        # correct for a problem with the 'hidden_layer_sizes' parameter when loading arrays from JSON
        params = correct_hidden_layers(params)

        loaded_model = MLPRegressor(batch_size=params['batch_size'],
                                hidden_layer_sizes=params['hidden_layer_sizes'],
                                learning_rate_init=params['learning_rate_init'],
                                max_iter=params['max_iter'],
                                validation_fraction=params['validation_fraction'],
                                alpha=params['alpha'],
                                shuffle=params['shuffle'],
                                tol=params['tol'],
                                early_stopping=params['early_stopping'],
                                beta_1=params['beta_1'],
                                beta_2=params['beta_2'],
                                epsilon=params['epsilon'],
                                random_state=params['seed'])  

        if params['model'].upper() == 'VR':
            r3 = loaded_model      
            
    if params['model'].upper() == 'ADAB':
        loaded_model = AdaBoostRegressor(n_estimators=params['n_estimators'],
                                learning_rate=params['learning_rate'],
                                random_state=params['seed'])

    if params['model'].upper() == 'GP':
        loaded_model = GaussianProcessRegressor(n_restarts_optimizer=params['n_restarts_optimizer'],
                                random_state=params['seed'])

    if params['model'].upper() == 'VR':
        loaded_model = VotingRegressor([('rf', r1), ('gb', r2), ('nn', r3)])

    if params['model'].upper() == 'MVL':
        loaded_model = LinearRegression(n_jobs=None)

    return loaded_model


def load_model_clas(params):

    if params['model'].upper() in ['RF','VR']:     
        loaded_model = RandomForestClassifier(max_depth=params['max_depth'],
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                min_samples_split=params['min_samples_split'],
                                min_samples_leaf=params['min_samples_leaf'],
                                min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                oob_score=params['oob_score'],
                                ccp_alpha=params['ccp_alpha'],
                                max_samples=params['max_samples'],
                                random_state=params['seed'],
                                n_jobs=None)

        if params['model'].upper() == 'VR':
            r1 = loaded_model

    if params['model'].upper() in ['GB','VR']:    
        loaded_model = GradientBoostingClassifier(max_depth=params['max_depth'], 
                                max_features=params['max_features'],
                                n_estimators=params['n_estimators'],
                                learning_rate=params['learning_rate'],
                                validation_fraction=params['validation_fraction'],
                                subsample=params['subsample'],
                                min_samples_split=params['min_samples_split'],
                                min_samples_leaf=params['min_samples_leaf'],
                                min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                ccp_alpha=params['ccp_alpha'],
                                random_state=params['seed'])

        if params['model'].upper() == 'VR':
            r2 = loaded_model

    if params['model'].upper() == 'ADAB':
            loaded_model = AdaBoostClassifier(n_estimators=params['n_estimators'],
                                learning_rate=params['learning_rate'],
                                random_state=params['seed'])

    if params['model'].upper() in ['NN','VR']:

        # correct for a problem with the 'hidden_layer_sizes' parameter when loading arrays from JSON
        params = correct_hidden_layers(params)

        loaded_model = MLPClassifier(batch_size=params['batch_size'],
                                hidden_layer_sizes=params['hidden_layer_sizes'],
                                learning_rate_init=params['learning_rate_init'],
                                max_iter=params['max_iter'],
                                validation_fraction=params['validation_fraction'],
                                alpha=params['alpha'],
                                shuffle=params['shuffle'],
                                tol=params['tol'],
                                early_stopping=params['early_stopping'],
                                beta_1=params['beta_1'],
                                beta_2=params['beta_2'],
                                epsilon=params['epsilon'],
                                random_state=params['seed'])

        if params['model'].upper() == 'VR':
            r3 = loaded_model

    if params['model'].upper() == 'GP':
        loaded_model = GaussianProcessClassifier(n_restarts_optimizer=params['n_restarts_optimizer'],
                                random_state=params['seed'],
                                n_jobs=None)

    if params['model'].upper() == 'VR':
        loaded_model = VotingClassifier([('rf', r1), ('gb', r2), ('nn', r3)])

    return loaded_model


# calculates errors/precision and predicted values of the ML models
def load_n_predict(self, params, data, hyperopt=False, mapie=False):

    # set the parameters for each ML model of the hyperopt optimization
    loaded_model = load_model(params)

    # Fit the model with the training set
    loaded_model.fit(np.asarray(data['X_train_scaled']).tolist(), np.asarray(data['y_train']).tolist())

    # store the predicted values for training
    data['y_pred_train'] = loaded_model.predict(np.asarray(data['X_train_scaled']).tolist()).tolist()

    if params['train'] < 100:
        # Predicted values using the model for out-of-bag (oob) sets (validation or test)
        data['y_pred_valid'] = loaded_model.predict(np.asarray(data['X_valid_scaled']).tolist()).tolist()
        if 'X_test_scaled' in data:
            data['y_pred_test'] = loaded_model.predict(np.asarray(data['X_test_scaled']).tolist()).tolist()
        if 'X_csv_test_scaled' in data:
            data['y_pred_csv_test'] = loaded_model.predict(np.asarray(data['X_csv_test_scaled']).tolist()).tolist()

    # get metrics for the different sets
    if params['type'].lower() == 'reg':
        data['r2_train'], data['mae_train'], data['rmse_train'] = get_prediction_results(params,data['y_train'],data['y_pred_train'])
        if params['train'] == 100:
            data['r2_valid'], data['mae_valid'], data['rmse_valid'] = data['r2_train'], data['mae_train'], data['rmse_train']
        else:
            data['r2_valid'], data['mae_valid'], data['rmse_valid'] = get_prediction_results(params,data['y_valid'],data['y_pred_valid'])
        if 'y_pred_test' in data and not data['y_test'].isnull().values.any() and len(data['y_test']) > 0:
            data['r2_test'], data['mae_test'], data['rmse_test'] = get_prediction_results(params,data['y_test'],data['y_pred_test'])
        if 'y_pred_csv_test' in data and not data['y_csv_test'].isnull().values.any() and len(data['y_csv_test']) > 0:
            data['r2_csv_test'], data['mae_csv_test'], data['rmse_csv_test'] = get_prediction_results(params,data['y_csv_test'],data['y_pred_csv_test'])
            
        if hyperopt:
            opt_target = data[f'{params["error_type"].lower()}_valid']
            if params['error_type'].lower() == 'r2':
                # avoids problems with regression lines with good R2 in validation that go in a different
                # direction of the regression in train
                score_model = loaded_model.score(data['X_valid_scaled'], data['y_valid'])
                if score_model < 0:
                    opt_target = 0
            return opt_target,data
        else:
            if mapie:
                if 'X_csv_test_scaled' in data:
                    data = calc_ci_n_sd(self,loaded_model,data,'csv_test')

                if 'X_test_scaled' in data:
                    data = calc_ci_n_sd(self,loaded_model,data,'test')
                
                elif 'X_valid_scaled' in data:
                    data = calc_ci_n_sd(self,loaded_model,data,'valid')

            return data

    elif params['type'].lower() == 'clas':
        data['acc_train'], data['f1_train'], data['mcc_train'] = get_prediction_results(params,data['y_train'],data['y_pred_train'])
        if params['train'] == 100:
            data['acc_valid'], data['f1_valid'], data['mcc_valid'] = data['acc_train'], data['f1_train'], data['mcc_train']
        else:
            data['acc_valid'], data['f1_valid'], data['mcc_valid'] = get_prediction_results(params,data['y_valid'],data['y_pred_valid'])
        if 'y_pred_test' in data and not data['y_test'].isnull().values.any() and len(data['y_test']) > 0:
            data['acc_test'], data['f1_test'], data['mcc_test'] = get_prediction_results(params,data['y_test'],data['y_pred_test'])
        if 'y_pred_csv_test' in data and not data['y_csv_test'].isnull().values.any() and len(data['y_csv_test']) > 0:
            data['acc_csv_test'], data['f1_csv_test'], data['mcc_csv_test'] = get_prediction_results(params,data['y_csv_test'],data['y_pred_csv_test'])
        if hyperopt:
            opt_target = data[f'{params["error_type"].lower()}_valid']
            return opt_target,data
        else:
            return data


def correct_hidden_layers(params):
    '''
    Correct for a problem with the 'hidden_layer_sizes' parameter when loading arrays from JSON
    '''
    
    layer_arrays = []

    if not isinstance(params['hidden_layer_sizes'],int):
        if params['hidden_layer_sizes'][0] == '[':
            params['hidden_layer_sizes'] = params['hidden_layer_sizes'][1:]
        if params['hidden_layer_sizes'][-1] == ']':
            params['hidden_layer_sizes'] = params['hidden_layer_sizes'][:-1]
        if not isinstance(params['hidden_layer_sizes'],list):
            for _,ele in enumerate(params['hidden_layer_sizes'].split(',')):
                if ele != '':
                    layer_arrays.append(int(ele))
        else:
            for _,ele in enumerate(params['hidden_layer_sizes']):
                if ele != '':
                    layer_arrays.append(int(ele))
    else:
        layer_arrays = ele

    params['hidden_layer_sizes'] = (layer_arrays)

    return params


def calc_ci_n_sd(self,loaded_model,data,set_mapie):
    """
    Calculate confidence intervals and standard deviations of each datapoint with MAPIE.
    """

    # mapie for obtaining prediction intervals
    my_conformity_score = AbsoluteConformityScore()
    my_conformity_score.consistency_check = False

    # LOOCV for relatively small training sets (less than 50 datapoints), 5-fold CV otherwise
    if self.args.kfold == 'auto':
        if len(data['X_train_scaled']) < 50:
            cv_type = 'loocv'
            kfold_type = -1 # -1 for LOOCV in MAPIE
        else:
            cv_type = '5_fold_cv'
            kfold_type = 5
    else:
        cv_type = f'{self.args.kfold}_fold_cv'
        kfold_type = self.args.kfold

    mapie_model = MapieRegressor(loaded_model, method="plus", cv=kfold_type, agg_function="median", conformity_score=my_conformity_score, n_jobs=-1, random_state=0)
    mapie_model.fit(data['X_train_scaled'].values, data['y_train'].values) # .values is needed to avoid an sklearn warning

    # Check if 1/alpha is lower than the number of samples
    if 1 / self.args.alpha >= len(data[f'X_{set_mapie}_scaled']):
        self.args.alpha = 0.1
        if 1 / self.args.alpha >= len(data[f'X_{set_mapie}_scaled']):
            self.args.alpha = 0.5
    
    # Predict test set and obtain prediction intervals
    y_pred, y_pis = mapie_model.predict(data[f'X_{set_mapie}_scaled'].values, alpha=[self.args.alpha]) # .values is needed to avoid an sklearn warning

    # Calculate the width of the prediction intervals
    y_interval_width = np.abs(y_pis[:, 0, :] - y_pis[:, 1, :])

    # NOTE: the middle of the prediction intervals is very close to the predicted value from
    # the original model. Therefore, we will use the originally predicted values with the 
    # calculated SD from the intervals

    # Estimate the standard deviation of the prediction intervals (assuming symmetric prediction intervals and approximately normal distribution of errors)
    # assuming normal population doesn't add very significant errors even in low-data regimes (i.e. for 20 points,
    # Student's t value is 2.086 instead of 1.96) 
    dict_alpha  = {0.05: 1.96, 0.1: 1.645, 0.5: 0.674}
    y_pred_sd = y_interval_width / (2 * dict_alpha[self.args.alpha])
    avg_sd = np.mean(y_pred_sd) # average SD

    # Add 'y_pred_SET_cv' and 'y_pred_SET_sd' entry to data dictionary
    data[f'y_pred_{set_mapie}_sd'] = y_pred_sd
    data['avg_sd'] = avg_sd
    data['cv_type'] = cv_type

    return data


def get_prediction_results(params,y,y_pred):
    if params['type'].lower() == 'reg':
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        if len(np.unique(y)) > 1 and len(np.unique(y_pred)) > 1:
            res = stats.linregress(y, y_pred)
            r2 = res.rvalue**2
        else:
            r2 = 0.0
        return r2, mae, rmse

    elif params['type'].lower() == 'clas':
        acc = accuracy_score(y,y_pred)
        # F1 by default uses average='binnary', to deal with predictions with more than 2 ouput values we use average='micro'
        # if len(set(y))==2:
        try:
            f1_score_val = f1_score(y,y_pred)
        except ValueError:
            f1_score_val = f1_score(y,y_pred,average='micro')
        mcc = matthews_corrcoef(y,y_pred)
        return acc, f1_score_val, mcc


def load_db_n_params(self,folder_model,module,print_load):
    '''
    Loads the parameters and Xy databases from a folder, add scaled X data and print information
    about the databases
    '''

    Xy_data_df, _, params_df, params_path, suffix, suffix_title = load_dfs(self,folder_model,module)

    # load only the descriptors used in the model and standardize X
    Xy_train_df = Xy_data_df[Xy_data_df.Set == 'Training'].reset_index(drop=True)
    Xy_valid_df = Xy_data_df[Xy_data_df.Set == 'Validation'].reset_index(drop=True)
    Xy_test_df = Xy_data_df[Xy_data_df.Set == 'Test'].reset_index(drop=True)

    Xy_data = {} # (using a dict to keep the same format of load_model() )
    descs_model = ast.literal_eval(params_df['X_descriptors'][0])
    Xy_data['X_train'] = Xy_train_df[descs_model]
    Xy_data['X_valid'] = Xy_valid_df[descs_model]
    Xy_data['y_train'] = Xy_train_df[params_df['y'][0]]
    Xy_data['y_valid'] = Xy_valid_df[params_df['y'][0]]
    Xy_data['X_train_scaled'], Xy_data['X_valid_scaled'] = standardize(self,Xy_data['X_train'],Xy_data['X_valid'])
    # test sets are scaled later in PREDICT

    Xy_data['X_test'] = Xy_test_df[descs_model]
    Xy_data['y_test'] = Xy_test_df[params_df['y'][0]]

    point_count = {}
    point_count['train'] = len(Xy_data['X_train_scaled'])
    point_count['valid'] = len(Xy_data['X_valid_scaled'])
    point_count['test'] = len(Xy_data['X_test'])

    params_name = os.path.basename(params_path)
    if print_load:
        _ = load_print(self,params_name,suffix,params_df,point_count)

    return Xy_data, params_df, params_path, suffix_title, Xy_test_df


def load_dfs(self,folder_model,module):
    '''
    Loads the parameters and Xy databases from a folder as dataframes
    '''
    
    if os.getcwd() in f"{folder_model}":
        path_db = folder_model
    else:
        path_db = f"{Path(os.getcwd()).joinpath(folder_model)}"
    suffix = '(with no PFI filter)'
    suffix_title = 'No_PFI'
    if os.path.exists(path_db):
        csv_files = glob.glob(f'{Path(path_db).joinpath("*.csv")}')
        if len(csv_files) != 2:
            self.args.log.write(f"\nx  There are not two CSV files in the {path_db} folder! Only two CSV files should be there, one with the model parameters and the other with the Xy database.")
            self.args.log.finalize()
            sys.exit()
        for csv_file in csv_files:
            if 'PFI' in os.path.basename(csv_file).replace('.csv','_').split('_'):
                suffix = '(with PFI filter)'
                suffix_title = 'PFI'
            if '_db' in csv_file:
                Xy_data_df = load_database(self,csv_file,module)
                Xy_path = csv_file
            else:
                params_df = load_database(self,csv_file,module)
                params_path = csv_file
    else:
        self.args.log.write(f"\nx  The folder with the model and database ({path_db}) does not exist! Did you use the destination=PATH option in the other modules?")
        sys.exit()

    return Xy_data_df, Xy_path, params_df, params_path, suffix, suffix_title


def load_print(self,params_name,suffix,params_df,point_count):
    if '.csv' in params_name:
        params_name = params_name.split('.csv')[0]
    txt_load = f'\no  ML model {params_name} {suffix} and Xy database were loaded, including:'
    txt_load += f'\n   - Target value: {params_df["y"][0]}'
    txt_load += f'\n   - Names: {params_df["names"][0]}'
    txt_load += f'\n   - Model: {params_df["model"][0]}'
    txt_load += f'\n   - Descriptors: {params_df["X_descriptors"][0]}'
    txt_load += f'\n   - Training points: {point_count["train"]}'
    txt_load += f'\n   - Validation points: {point_count["valid"]}'
    if 'test' in point_count:
        txt_load += f'\n   - Test points: {point_count["test"]}'
    self.args.log.write(txt_load)


def pd_to_dict(PFI_df):
    PFI_df_dict = {}
    for column in PFI_df.columns:
        PFI_df_dict[column] = PFI_df[column][0]
    return PFI_df_dict


def print_pfi(self,params_dir):
    if 'No_PFI' in params_dir:
        self.args.log.write('\n\n------- Starting model with all variables (No PFI) -------')
    else:
        self.args.log.write('\n\n------- Starting model with PFI filter (only important descriptors used) -------')


def get_graph_style():
    """
    Retrieves the graph style for regression plots
    """
    
    graph_style = {'color_train' : 'b',
        'color_valid' : 'orange',
        'color_test' : 'r',
        'dot_size' : 50,
        'alpha' : 1 # from 0 (transparent) to 1 (opaque)
        }

    return graph_style


def pearson_map(self,csv_df_pearson,module,params_dir=None):
    '''
    Creates Pearson heatmap
    '''

    if module.lower() == 'curate': # only represent the final descriptors in CURATE
        csv_df_pearson = csv_df_pearson.drop([self.args.y] + self.args.ignore, axis=1)

    corr_matrix = csv_df_pearson.corr()
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)]= True
    
    # no representatoins when there are more than 30 descriptors
    if len(csv_df_pearson.columns) > 30:
        disable_plot = True
    else:
        disable_plot = False
        _, ax = plt.subplots(figsize=(7.45,6))
        size_title = 14
        size_font = 14-2*((len(csv_df_pearson.columns)/5))

    if disable_plot:
        if module.lower() == 'curate':
            self.args.log.write(f'\nx  The Pearson heatmap was not generated because the number of features and the y value ({len(csv_df_pearson.columns)}) is higher than 30.')
        if module.lower() == 'predict':
            self.args.log.write(f'\n   x  The Pearson heatmap was not generated because the number of features and the y value ({len(csv_df_pearson.columns)}) is higher than 30.')

    else:
        sb.set(font_scale=1.2, style='ticks')

        _ = sb.heatmap(corr_matrix,
                        mask = mask,
                        square = True,
                        linewidths = .5,
                        cmap = 'coolwarm',
                        cbar = False,
                        cbar_kws = {'shrink': .4,
                                    'ticks' : [-1, -.5, 0, 0.5, 1]},
                        vmin = -1,
                        vmax = 1,
                        annot = True,
                        annot_kws = {'size': size_font})

        plt.tick_params(labelsize=size_font)
        #add the column names as labels
        ax.set_yticklabels(corr_matrix.columns, rotation = 0)
        ax.set_xticklabels(corr_matrix.columns)

        title_fig = 'Pearson\'s r heatmap'
        if module.lower() == 'predict':
            if os.path.basename(Path(params_dir)) == 'No_PFI':
                suffix_title = 'No_PFI'
            elif os.path.basename(Path(params_dir)) == 'PFI':
                suffix_title = 'PFI'
            title_fig += f'_{suffix_title}'

        plt.title(title_fig, y=1.04, fontsize = size_title, fontweight="bold")
        sb.set_style({'xtick.bottom': True}, {'ytick.left': True})

        if module.lower() == 'curate':
            heatmap_name = 'Pearson_heatmap.png'
        elif module.lower() == 'predict':
            heatmap_name = f'Pearson_heatmap_{suffix_title}.png'

        heatmap_path = self.args.destination.joinpath(heatmap_name)
        plt.savefig(f'{heatmap_path}', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        path_reduced = '/'.join(f'{heatmap_path}'.replace('\\','/').split('/')[-2:])
        if module.lower() == 'curate':
            self.args.log.write(f'\no  The Pearson heatmap was stored in {path_reduced}.')
        elif module.lower() == 'predict':
            self.args.log.write(f'\n   o  The Pearson heatmap was stored in {path_reduced}.')

    return corr_matrix