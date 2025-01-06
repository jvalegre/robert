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
import importlib
import matplotlib.patches as mpatches
import matplotlib.colors as mcolor
from matplotlib.legend_handler import HandlerPatch
import shap
import seaborn as sb
from scipy import stats
from pkg_resources import resource_filename
# for users with no intel architectures. This part has to be before the sklearn imports
try:
    from sklearnex import patch_sklearn
    patch_sklearn(verbose=False)
except (ModuleNotFoundError,ImportError):
    pass
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             matthews_corrcoef, accuracy_score, f1_score, make_scorer, ConfusionMatrixDisplay)
from sklearn.feature_selection import RFECV
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
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from mapie.regression import MapieRegressor
from mapie.conformity_scores import AbsoluteConformityScore
from robert.argument_parser import set_options, var_dict
import warnings # this avoids warnings from sklearn
warnings.filterwarnings("ignore")


robert_version = "1.3.1"
time_run = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
robert_ref = "Dalmau, D.; Alegre Requena, J. V. WIREs Comput Mol Sci. 2024, 14, e1733."


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
        self.log = open(f"{filein}_{append}.{suffix}", "w", encoding="utf-8")

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
        "evaluate",
    ]
    list_args = [
        "discard",
        "ignore",
        "train",
        "model",
        "report_modules",
        "seed",
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

* Affecting predictions, PREDICT:
  --t_value INT (default=2) : t-value threshold to identify outliers
  --shap_show INT (default=10) : maximum number of descriptors shown in the SHAP plot

* Affecting SMILES workflows, AQME:
  --qdescp_keywords STR (default="") : extra keywords in QDESCP (i.e. "--qdescp_atoms [Ir] --alpb h2o") 
  --csearch_keywords STR (default="--sample 50") : extra keywords in CSEARCH
  --descp_lvl (default="interpret") "interpret", "denovo" or "full" : type of descriptor calculation


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

    # remove extra spaces that sometimes are included by mistake
    value = [ele.strip() if isinstance(ele, str) else ele for ele in value]

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

    # check if user used .csv in csv_test
    if self.csv_test and not os.path.exists(f"{self.csv_test}") and os.path.exists(f'{self.csv_test}.csv'):
        self.csv_test = f'{self.csv_test}.csv'

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
                self.log.write(f"\nx  WARNING! The scikit-learn-intelex accelerator is not installed, the results might vary if it is installed and the execution times might become much longer (if available, use 'pip install scikit-learn-intelex')")

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
                if ('MVL' or 'mvl') in self.model:
                    self.model = [x if x.upper() != 'MVL' else 'AdaB' for x in self.model]
            
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


def correlation_filter(self, csv_df):
    """
    Discards a) correlated variables and b) variables that do not correlate with the y values, based
    on R**2 values c) reduces the number of descriptors to one third of the datapoints using RFECV.
    """

    txt_corr = ''

    # loosen correlation filters if there are too few descriptors
    n_descps = len(csv_df.columns)-len(self.args.ignore)-1 # all columns - ignored - y
    if self.args.desc_thres and n_descps*self.args.desc_thres < len(csv_df[self.args.y]):
        self.args.thres_x = 0.95
        self.args.thres_y = 0.0001
        txt_corr += f'\nx  WARNING! The number of descriptors ({n_descps}) is {self.args.desc_thres} times lower than the number of datapoints ({len(csv_df[self.args.y])}), the correlation filters are loosen to thres_x = 0.95 and thres_y = 0.0001! Default thresholds (0.9 and 0.001) can be used with "--desc_thres False"'

    txt_corr += f'\no  Correlation filter activated with these thresholds: thres_x = {self.args.thres_x}, thres_y = {self.args.thres_y}'

    descriptors_drop = []
    txt_corr += f'\n   Excluded descriptors:'
    for i,column in enumerate(csv_df.columns):
        if column not in descriptors_drop and column not in self.args.ignore and column != self.args.y:
            # finds the descriptors with low correlation to the response values
            try:
                res_y = stats.linregress(csv_df[column],csv_df[self.args.y])
                rsquared_y = res_y.rvalue**2
                if rsquared_y < self.args.thres_y:
                    descriptors_drop.append(column)
                    txt_corr += f'\n   - {column}: R**2 = {round(rsquared_y,2)} with the {self.args.y} values'
            except ValueError: # this avoids X descriptors where the majority of the values are the same
                descriptors_drop.append(column)
                txt_corr += f'\n   - {column}: error in R**2 with the {self.args.y} values (are all the values the same?)'

            # finds correlated descriptors
            if column != csv_df.columns[-1] and column not in descriptors_drop:
                for j,column2 in enumerate(csv_df.columns):
                    if j > i and column2 not in self.args.ignore and column not in descriptors_drop and column2 not in descriptors_drop and column2 != self.args.y:
                        res_x = stats.linregress(csv_df[column],csv_df[column2])
                        rsquared_x = res_x.rvalue**2
                        if rsquared_x > self.args.thres_x:
                            # discard the column with less correlation with the y values
                            res_xy = stats.linregress(csv_df[column2],csv_df[self.args.y])
                            rsquared_y2 = res_xy.rvalue**2
                            if rsquared_y >= rsquared_y2:
                                descriptors_drop.append(column2)
                                txt_corr += f'\n   - {column2}: R**2 = {round(rsquared_x,2)} with {column}'
                            else:
                                descriptors_drop.append(column)
                                txt_corr += f'\n   - {column}: R**2 = {round(rsquared_x,2)} with {column2}'
    
    # drop descriptors that did not pass the filters
    csv_df_filtered = csv_df.drop(descriptors_drop, axis=1)

    # Check if descriptors are more than one third of datapoints
    n_descps = len(csv_df_filtered.columns)-len(self.args.ignore)-1 # all columns - ignored - y
    if len(csv_df[self.args.y]) > 100 and self.args.auto_test ==True:
        datapoints = len(csv_df[self.args.y])*0.9
    else:
        datapoints = len(csv_df[self.args.y])
    if n_descps > datapoints / 3:
        num_descriptors = int(datapoints / 3)
        # Avoid situations where the number of descriptors is equal to the number of datapoints/3
        if len(csv_df[self.args.y]) / 3 == num_descriptors:
            num_descriptors -= 1
        # Use RFECV with a simple RandomForestRegressor to select the most important descriptors
        if self.args.type.lower() == 'reg':
            estimator = RandomForestRegressor(random_state=0, n_estimators=30, max_depth=10,  n_jobs=None)
        elif self.args.type.lower() == 'clas':
            estimator = RandomForestClassifier(random_state=0, n_estimators=30, max_depth=10,  n_jobs=None)
        if self.args.kfold == 'auto':
            # LOOCV for relatively small datasets (less than 50 datapoints)
            if len(csv_df[self.args.y]) < 50:
                n_splits = len(csv_df[self.args.y])
                cv_type = 'LOOCV'
            # k-fold CV with the same training/validation proportion used for fitting the model, using 5 splits
            else:
                n_splits = 5
                cv_type = '5-fold CV'
        else:
            n_splits = self.args.kfold
            cv_type = f'{n_splits}-fold CV'
        txt_corr += f'\n\no  Descriptors reduced to one third of datapoints using RFECV with {cv_type}: {num_descriptors} descriptors remaining'

        selector = RFECV(estimator, min_features_to_select=num_descriptors, cv=KFold(n_splits=n_splits, shuffle=True, random_state=0), n_jobs=None)
        X = csv_df_filtered.drop([self.args.y] + self.args.ignore, axis=1)
        y = csv_df_filtered[self.args.y]
        # Convert column names to strings to avoid any issues
        X.columns = X.columns.astype(str)
        selector.fit(X, y)
        # Sort the descriptors by their importance scores
        descriptors_importances = list(zip(X.columns, selector.estimator_.feature_importances_))
        sorted_descriptors = sorted(descriptors_importances, key=lambda x: x[1], reverse=True)
        selected_descriptors = [descriptor for descriptor, _ in sorted_descriptors[:num_descriptors]]
        # Find the descriptors to drop
        descriptors_drop = [descriptor for descriptor in csv_df_filtered.columns if descriptor not in selected_descriptors and descriptor not in self.args.ignore and descriptor != self.args.y]
        csv_df_filtered = csv_df_filtered.drop(descriptors_drop, axis=1)

    if len(descriptors_drop) == 0:
        txt_corr += f'\n   -  No descriptors were removed'

    self.args.log.write(txt_corr)

    txt_csv = f'\no  {len(csv_df_filtered.columns)} columns remaining after applying duplicate, correlation filters and RFECV:\n'
    txt_csv += '\n'.join(f'   - {var}' for var in csv_df_filtered.columns)
    self.args.log.write(txt_csv)

    return csv_df_filtered


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
            if self.split.lower() not in ['kn','rnd','stratified','even']:
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


def check_clas_problem(self,csv_df):
    '''
    Changes type to classification if there are only two different y values
    '''

    # changes type to classification if there are only two different y values
    if self.args.type.lower() == 'reg' and self.args.auto_type:
        if len(set(csv_df[self.args.y])) in [1,2,3,4]:
            self.args.type = 'clas'
            if self.args.error_type not in ['acc', 'mcc', 'f1']:
                self.args.error_type = 'mcc'
            if ('MVL' or 'mvl') in self.args.model:
                self.args.model = [x if x.upper() != 'MVL' else 'AdaB' for x in self.args.model]

            y_val_detect = f'{list(set(csv_df[self.args.y]))[0]} and {list(set(csv_df[self.args.y]))[1]}'
            self.args.log.write(f'\no  Only two different y values were detected ({y_val_detect})! The program will consider classification models (same effect as using "--type clas"). This option can be disabled with "--auto_type False"')

    if self.args.type.lower() == 'clas':
        if len(set(csv_df[self.args.y])) == 2:
            for target_val in set(csv_df[self.args.y]):
                if target_val not in [0,'0',1,'1']:
                    y_val_detect = f'{list(set(csv_df[self.args.y]))[0]} and {list(set(csv_df[self.args.y]))[1]}'
                    self.args.log.write(f'\nx  Only 0s and 1s values are currently allowed for classification problems! {y_val_detect} were used as values')
                    self.args.log.finalize()
                    sys.exit()

        if len(set(csv_df[self.args.y])) != 2:
            self.args.log.write(f'\nx  Only two different y values are currently allowed! {len(set(csv_df[self.args.y]))} different values were used {set(csv_df[self.args.y])}')
            self.args.log.finalize()
            sys.exit()

    return self
    

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
    # Fill missing values using KNN imputer, excluding target column
    csv_df = csv_df.dropna(axis=1, thresh=int(0.7 * len(csv_df)))  # Remove columns with <70% data
    target_col = self.args.y
    int_columns = csv_df.select_dtypes(include=['int']).columns.drop(target_col, errors='ignore')
    numeric_columns = csv_df.select_dtypes(include=['float']).columns.drop(target_col, errors='ignore')
    imputer = KNNImputer(n_neighbors=5)
    if csv_df[numeric_columns].isna().any().any():
        csv_df[numeric_columns] = pd.DataFrame(imputer.fit_transform(csv_df[numeric_columns]), columns=numeric_columns)
    csv_df[int_columns] = csv_df[int_columns]

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


def categorical_transform(self,csv_df,module):
    ''' converts all columns with strings into categorical values (one hot encoding
    by default, can be set to numerical 1,2,3... with categorical = True).
    Troubleshooting! For one-hot encoding, don't use variable names that are
    also column headers! i.e. DESCRIPTOR "C_atom" contain C2 as a value,
    but C2 is already a header of a different column in the database. Same applies
    for multiple columns containing the same variable names.
    '''

    if module.lower() == 'curate':
        txt_categor = f'\no  Analyzing categorical variables'

    descriptors_to_drop, categorical_vars, new_categor_desc = [],[],[]
    for column in csv_df.columns:
        if column not in self.args.ignore and column != self.args.y:
            if(csv_df[column].dtype == 'object'):
                descriptors_to_drop.append(column)
                categorical_vars.append(column)
                if self.args.categorical.lower() == 'numbers':
                    csv_df[column] = csv_df[column].astype('category')
                    csv_df[column] = csv_df[column].cat.codes
                else:
                    _ = csv_df[column].unique() # is this necessary?
                    categor_descs = pd.get_dummies(csv_df[column])
                    csv_df = csv_df.drop(column, axis=1)
                    csv_df = pd.concat([csv_df, categor_descs], axis=1)
                    for desc in categor_descs:
                        new_categor_desc.append(desc)

    if module.lower() == 'curate':
        if len(categorical_vars) == 0:
            txt_categor += f'\n   - No categorical variables were found'
        else:
            if self.args.categorical.lower() == 'numbers':
                txt_categor += f'\n   A total of {len(categorical_vars)} categorical variables were converted using the {self.args.categorical} mode in the categorical option:\n'
                txt_categor += '\n'.join(f'   - {var}' for var in categorical_vars)
            else:
                txt_categor += f'\n   A total of {len(categorical_vars)} categorical variables were converted using the {self.args.categorical} mode in the categorical option'
                txt_categor += f'\n   Initial descriptors:\n'
                txt_categor += '\n'.join(f'   - {var}' for var in categorical_vars)
                txt_categor += f'\n   Generated descriptors:\n'
                txt_categor += '\n'.join(f'   - {var}' for var in new_categor_desc)

        self.args.log.write(f'{txt_categor}')

    return csv_df


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
                                solver='lbfgs',
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
                                solver='lbfgs',
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

            return data,loaded_model

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
            return data,loaded_model


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


def k_means(self,X_scaled,csv_y,size,seed,idx_list):
    '''
    Returns the data points that will be used as training set based on the k-means clustering
    '''
    
    # number of clusters in the training set from the k-means clustering (based on the
    # training set size specified above)
    X_scaled_array = np.asarray(X_scaled)
    number_of_clusters = int(len(csv_y)*(size/100))

    # to avoid points from the validation set outside the training set, the 2 first training
    # points are automatically set as the 2 points with minimum/maximum response value
    if self.args.type.lower() == 'reg':
        training_points = [csv_y.idxmin(),csv_y.idxmax()]
        training_idx = [csv_y.idxmin(),csv_y.idxmax()]
        number_of_clusters -= 2
    else:
        training_points = []
        training_idx = []
    
    # runs the k-means algorithm and keeps the closest point to the center of each cluster
    kmeans = KMeans(n_clusters=number_of_clusters,random_state=seed)
    try:
        kmeans.fit(X_scaled_array)
    except ValueError:
        self.args.log.write("\nx  The K-means clustering process failed! This might be due to having NaN or strings as descriptors (curate the data first with CURATE) or having too few datapoints!")
        sys.exit()
    centers = kmeans.cluster_centers_
    for i in range(number_of_clusters):
        results_cluster = 1000000
        for k in range(len(X_scaled_array[:, 0])):
            if k not in training_idx:
                # calculate the Euclidean distance in n-dimensions
                points_sum = 0
                for l in range(len(X_scaled_array[0])):
                    points_sum += (X_scaled_array[:, l][k]-centers[:, l][i])**2
                if np.sqrt(points_sum) < results_cluster:
                    results_cluster = np.sqrt(points_sum)
                    training_point = k
        training_idx.append(training_point)
        training_points.append(idx_list[training_point])
    
    training_points.sort()

    return training_points


def PFI_filter(self,Xy_data,PFI_dict,seed):
    '''
    Performs the PFI calculation and returns a list of the descriptors that are not important
    '''

    # load and fit model
    loaded_model = load_model(PFI_dict)
    loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])

    # we use the validation set during PFI as suggested by the sklearn team:
    # "Using a held-out set makes it possible to highlight which features contribute the most to the 
    # generalization power of the inspected model. Features that are important on the training set 
    # but not on the held-out set might cause the model to overfit."
    score_model = loaded_model.score(Xy_data['X_valid_scaled'], Xy_data['y_valid'])
    error_type = PFI_dict['error_type'].lower()
    
    if PFI_dict['type'].lower() == 'reg':
        scoring = {
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_median_absolute_error',
            'r2': 'r2'
        }.get(error_type)
    else:
        scoring = {
            'mcc': make_scorer(matthews_corrcoef),
            'f1': 'f1',
            'acc': 'accuracy'
        }.get(error_type)

    perm_importance = permutation_importance(loaded_model, Xy_data['X_valid_scaled'], Xy_data['y_valid'], scoring=scoring, n_repeats=self.args.pfi_epochs, random_state=seed)
            
    # transforms the values into a list and sort the PFI values with the descriptor names
    descp_cols, PFI_values, PFI_sd = [],[],[]
    for i,desc in enumerate(Xy_data['X_train'].columns):
        descp_cols.append(desc) # includes lists of descriptors not column names!
        PFI_values.append(perm_importance.importances_mean[i])
        PFI_sd.append(perm_importance.importances_std[i])
  
    PFI_values, PFI_sd, descp_cols = (list(t) for t in zip(*sorted(zip(PFI_values, PFI_sd, descp_cols), reverse=True)))
    #Esto hay que hacerlo cuando init_curate =False, sino hay que hacer PFI_discard_cols=descp_cols.
    # PFI filter
    PFI_discard_cols = []
    PFI_thres = abs(self.args.pfi_threshold*score_model)
    for i in range(len(PFI_values)):
        if PFI_values[i] < PFI_thres:
            PFI_discard_cols.append(descp_cols[i])

    return PFI_discard_cols


def data_split(self,csv_X,csv_y,size,seed):

    if size == 100:
        # if there is no validation set, use all the points
        training_points = np.arange(0,len(csv_X),1)
    else:
        if self.args.split.upper() == 'KN':
            # k-neighbours data split
            # standardize the data before k-neighbours-based data splitting
            Xmeans = csv_X.mean(axis=0)
            Xstds = csv_X.std(axis=0)
            X_scaled = (csv_X - Xmeans) / Xstds

            # removes NaN missing values after standardization
            for column in X_scaled.columns:
                if X_scaled[column].isnull().values.any():
                    X_scaled = X_scaled.drop(column, axis=1)

            # selects representative training points for each target value in classification problems
            if self.args.type == 'clas':
                class_0_idx = list(csv_y[csv_y == 0].index)
                class_1_idx = list(csv_y[csv_y == 1].index)
                class_0_size = int(len(class_0_idx)/len(csv_y)*size)
                class_1_size = size-class_0_size

                train_class_0 = k_means(self,X_scaled.iloc[class_0_idx],csv_y,class_0_size,seed,class_0_idx)
                train_class_1 = k_means(self,X_scaled.iloc[class_1_idx],csv_y,class_1_size,seed,class_1_idx)
                training_points = train_class_0+train_class_1

            else:
                idx_list = csv_y.index
                training_points = k_means(self,X_scaled,csv_y,size,seed,idx_list)

        elif self.args.split.upper() == 'RND':
            X_train, _, _, _ = train_test_split(csv_X, csv_y, train_size=size/100, random_state=seed)
            training_points = X_train.index.tolist()

        elif self.args.split.upper() == 'STRATIFIED':
            # Calculate the number of bins based on the validation size
            stratified_quantiles = max(2, round(len(csv_y) * (1 - (size / 100))))
            y_binned = pd.qcut(csv_y, q=stratified_quantiles, labels=False, duplicates='drop')
            
            # Adjust the number of bins until each class has at least 2 members
            while y_binned.value_counts().min() < 2 and stratified_quantiles > 2:
                stratified_quantiles -= 1
                y_binned = pd.qcut(csv_y, q=stratified_quantiles, labels=False, duplicates='drop')
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=(100 - size) / 100, random_state=seed)
            for train_idx, _ in splitter.split(csv_X, y_binned):
                training_points = train_idx.tolist()

            # Ensure the extremes are in the training set
            for idx in [csv_y.idxmin(), csv_y.idxmax()]:
                if idx not in training_points:
                    training_points.append(idx)

            # Ensure the second smallest and second largest are not in the training set
            for idx in [csv_y.nsmallest(2).index[1], csv_y.nlargest(2).index[1]]:
                if idx in training_points:
                    training_points.remove(idx)

        elif self.args.split.upper() == 'EVEN':
            # Number of bins based on validation size
            stratified_quantiles = max(2, round(len(csv_y) * (1 - (size / 100))))
            y_binned = pd.qcut(csv_y, q=stratified_quantiles, labels=False, duplicates='drop')

            # Adjust bin count if any bin has fewer than two elements
            while y_binned.value_counts().min() < 2 and stratified_quantiles > 2:
                stratified_quantiles -= 1
                y_binned = pd.qcut(csv_y, q=stratified_quantiles, labels=False, duplicates='drop')

            # Determine central validation points for each bin
            training_points = []
            for bin_label in y_binned.unique():
                bin_indices = y_binned[y_binned == bin_label].index
                sorted_indices = sorted(bin_indices, key=lambda idx: csv_y[idx])
                n_excluded = int(round(len(sorted_indices) * (100 - size) / 100))
                mid = len(sorted_indices) // 2
                excluded_points = sorted_indices[mid - n_excluded // 2 : mid + (n_excluded + 1) // 2]
                training_points.extend(idx for idx in sorted_indices if idx not in excluded_points)

            # Ensure extremes are in the training set
            for idx in [csv_y.idxmin(), csv_y.idxmax()]:
                if idx not in training_points:
                    training_points.append(idx)

            # Remove second smallest and second largest from training
            for idx in [csv_y.nsmallest(2).index[1], csv_y.nlargest(2).index[1]]:
                if idx in training_points:
                    training_points.remove(idx)

    training_points.sort()
    Xy_data = Xy_split(csv_X,csv_y,training_points)

    return Xy_data


def Xy_split(csv_X,csv_y,training_points):
    '''
    Returns a dictionary with the database divided into train and validation
    '''

    Xy_data =  {}
    Xy_data['X_train'] = csv_X.iloc[training_points]
    Xy_data['y_train'] = csv_y.iloc[training_points]
    Xy_data['X_valid'] = csv_X.drop(training_points)
    Xy_data['y_valid'] = csv_y.drop(training_points)
    Xy_data['training_points'] = training_points

    return Xy_data


def create_heatmap(self,csv_df,suffix,path_raw):
    """
    Graph the heatmap
    """

    importlib.reload(plt) # needed to avoid threading issues
    csv_df = csv_df.sort_index(ascending=False)
    sb.set(font_scale=1.2, style='ticks')
    _, ax = plt.subplots(figsize=(7.45,6))
    cmap_blues_75_percent_512 = [mcolor.rgb2hex(c) for c in plt.cm.Blues(np.linspace(0, 0.8, 512))]
    # Replace inf values with NaN for proper heatmap visualization
    csv_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    ax = sb.heatmap(csv_df, annot=True, linewidth=1, cmap=cmap_blues_75_percent_512, cbar_kws={'label': f'Combined {self.args.error_type.upper()}'}, mask=csv_df.isnull())
    fontsize = 14
    ax.set_xlabel("ML Model",fontsize=fontsize)
    ax.set_ylabel("Training Size",fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    title_fig = f'Heatmap ML models {suffix}'
    plt.title(title_fig, y=1.04, fontsize = fontsize, fontweight="bold")
    sb.despine(top=False, right=False)
    name_fig = '_'.join(title_fig.split())
    plt.savefig(f'{path_raw.joinpath(name_fig)}.png', dpi=300, bbox_inches='tight')

    path_reduced = '/'.join(f'{path_raw}'.replace('\\','/').split('/')[-2:])
    self.args.log.write(f'\no  {name_fig} succesfully created in {path_reduced}')


def graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style,csv_test=False,print_fun=True,cv_mapie_graph=False):
    '''
    Plot regression graphs of predicted vs actual values for train, validation and test sets
    '''

    # Create graph
    importlib.reload(plt) # needed to avoid threading issues
    sb.set(style="ticks")

    _, ax = plt.subplots(figsize=(7.45,6))

    # Set tick sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Title and labels of the axis
    plt.ylabel(f'Predicted {params_dict["y"]}', fontsize=14)
    plt.xlabel(f'{params_dict["y"]}', fontsize=14)
    
    error_bars = "valid"
    if 'y_pred_test' in Xy_data and not Xy_data['y_test'].isnull().values.any() and len(Xy_data['y_test']) > 0:
        error_bars = "test"

    title_graph = graph_title(self,csv_test,set_types,cv_mapie_graph,error_bars,Xy_data)

    if print_fun:
        plt.text(0.5, 1.08, f'{title_graph} of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
            fontsize=14, fontweight='bold', transform = ax.transAxes)

    # Plot the data
    # CV graphs from VERIFY
    if 'CV' in set_types[0]:
        _ = ax.scatter(Xy_data["y_cv_valid"], Xy_data["y_pred_cv_valid"],
                    c = graph_style['color_valid'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)

    # other graphs
    elif not csv_test:
        if not cv_mapie_graph:
            _ = ax.scatter(Xy_data["y_train"], Xy_data["y_pred_train"],
                        c = graph_style['color_train'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)   
                    
        if not cv_mapie_graph or error_bars == 'valid':
            _ = ax.scatter(Xy_data["y_valid"], Xy_data["y_pred_valid"],
                        c = graph_style['color_valid'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)

        if error_bars == 'test':
            _ = ax.scatter(Xy_data["y_test"], Xy_data["y_pred_test"],
                        c = graph_style['color_test'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=3)

    else:
        error_bars = "test"
        _ = ax.scatter(Xy_data["y_csv_test"], Xy_data["y_pred_csv_test"],
                        c = graph_style['color_test'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)

    # average CV (MAPIE) ± SD graphs 
    if cv_mapie_graph:
        if not csv_test:   
            # Plot the data with the error bars
            _ = ax.errorbar(Xy_data[f"y_{error_bars}"], Xy_data[f"y_pred_{error_bars}"], yerr=Xy_data[f"y_pred_{error_bars}_sd"].flatten(), fmt='none', ecolor="gray", capsize=3, zorder=1)
            # Adjust labels from legend
            set_types=[error_bars,f'± SD']

        else:
            _ = ax.errorbar(Xy_data[f"y_csv_{error_bars}"], Xy_data[f"y_pred_csv_{error_bars}"], yerr=Xy_data[f"y_pred_csv_{error_bars}_sd"].flatten(), fmt='none', ecolor="gray", capsize=3, zorder=1)
            set_types=['External test',f'± SD']

    # legend and regression line with 95% CI considering all possible lines (not CI of the points)
    if 'CV' in set_types[0]: # CV in VERIFY
        if 'LOOCV' in set_types[0]:
            legend_coords = (0.835, 0.15) # LOOCV
        else:
            if len(set_types[0].split('-')[0]) == 1: # 1- to 9-fold CV
                legend_coords = (0.82, 0.15)
            elif len(set_types[0].split('-')[0]) == 2: # => 10-fold CV
                legend_coords = (0.807, 0.15)
    elif len(set_types) == 3: # train + valid + test
        legend_coords = (0.63, 0.15)
    elif len(set_types) == 2: # train + valid (or sets with ± SD)
        if 'External test' in set_types:
            legend_coords = (0.66, 0.15)
        else:
            legend_coords = (0.735, 0.15)
    ax.legend(loc='upper center', bbox_to_anchor=legend_coords,
            fancybox=True, shadow=True, ncol=5, labels=set_types, fontsize=14)

    Xy_data_df = pd.DataFrame()
    if 'CV' in set_types[0]:
        line_suff = 'cv_valid'
    elif not csv_test:
        line_suff = 'train'
    else:
        line_suff = 'csv_test'

    Xy_data_df[f"y_{line_suff}"] = Xy_data[f"y_{line_suff}"]
    Xy_data_df[f"y_pred_{line_suff}"] = Xy_data[f"y_pred_{line_suff}"]
    if len(Xy_data_df[f"y_pred_{line_suff}"]) >= 10:
        _ = sb.regplot(x=f"y_{line_suff}", y=f"y_pred_{line_suff}", data=Xy_data_df, scatter=False, color=".1", 
                        truncate = True, ax=ax, seed=params_dict['seed'])

    # set axis limits and graph PATH
    min_value_graph,max_value_graph,reg_plot_file,path_reduced = graph_vars(Xy_data,set_types,csv_test,path_n_suffix,cv_mapie_graph)

    # track the range of predictions (used in ROBERT score)
    pred_min = min(min(Xy_data["y_train"]),min(Xy_data["y_valid"]))
    pred_max = max(max(Xy_data["y_train"]),max(Xy_data["y_valid"]))
    pred_range = np.abs(pred_max-pred_min)
    Xy_data['pred_min'] = pred_min
    Xy_data['pred_max'] = pred_max
    Xy_data['pred_range'] = pred_range

    # Add gridlines
    ax.grid(linestyle='--', linewidth=1)

    # set axis limits
    plt.xlim(min_value_graph, max_value_graph)
    plt.ylim(min_value_graph, max_value_graph)

    # save graph
    plt.savefig(f'{reg_plot_file}', dpi=300, bbox_inches='tight')
    if print_fun:
        self.args.log.write(f"      -  Graph in: {path_reduced}")


def graph_title(self,csv_test,set_types,cv_mapie_graph,error_bars,Xy_data):
    '''
    Retrieves the corresponding graph title.
    '''

    # set title for regular graphs
    if not cv_mapie_graph:
        # title for k-fold CV graphs
        if 'CV' in set_types[0]:
            title_graph = f'{set_types[0]} for train+valid. sets'
        elif not csv_test:
            # regular graphs
            title_graph = f'Predictions_train_valid'
            if 'test' in set_types:
                title_graph += '_test'
        else:
            title_graph = f'{os.path.basename(self.args.csv_test)}'
            if len(title_graph) > 30:
                title_graph = f'{title_graph[:27]}...'

    # set title for averaged CV ± SD graphs
    else:
        if not csv_test:
            sets_title = error_bars
        else:
            sets_title = 'external test'
        if Xy_data['cv_type'] == 'loocv':
            title_graph = f'{sets_title} set ± SD (LOOCV)'
        else:
            kfold = Xy_data['cv_type'].split('_')[-3]
            title_graph = f'{sets_title} set ± SD ({kfold}-fold CV)'

    return title_graph


def graph_vars(Xy_data,set_types,csv_test,path_n_suffix,cv_mapie_graph):
    '''
    Set axis limits for regression plots and PATH to save the graphs
    '''

    # x and y axis limits for graphs with multiple sets
    if not csv_test and 'CV' not in set_types[0]:
        size_space = 0.1*abs(min(Xy_data["y_train"])-max(Xy_data["y_train"]))
        min_value_graph = min(min(Xy_data["y_train"]),min(Xy_data["y_pred_train"]),min(Xy_data["y_valid"]),min(Xy_data["y_pred_valid"]))
        if 'test' in set_types:
            min_value_graph = min(min_value_graph,min(Xy_data["y_test"]),min(Xy_data["y_pred_test"]))
        min_value_graph = min_value_graph-size_space
            
        max_value_graph = max(max(Xy_data["y_train"]),max(Xy_data["y_pred_train"]),max(Xy_data["y_valid"]),max(Xy_data["y_pred_valid"]))
        if 'test' in set_types:
            max_value_graph = max(max_value_graph,max(Xy_data["y_test"]),max(Xy_data["y_pred_test"]))
        max_value_graph = max_value_graph+size_space

    else: # limits for graphs with only one set
        if 'CV' in set_types[0]: # for CV graphs
            set_type = 'cv_valid'
        else: # for external test sets
            set_type = 'csv_test'
        size_space = 0.1*abs(min(Xy_data[f'y_{set_type}'])-max(Xy_data[f'y_{set_type}']))
        min_value_graph = min(min(Xy_data[f'y_{set_type}']),min(Xy_data[f'y_pred_{set_type}']))
        min_value_graph = min_value_graph-size_space
        max_value_graph = max(max(Xy_data[f'y_{set_type}']),max(Xy_data[f'y_pred_{set_type}']))
        max_value_graph = max_value_graph+size_space

    # PATH of the graph
    if not csv_test:
        if 'CV' in set_types[0]:
            reg_plot_file = f'{os.path.dirname(path_n_suffix)}/CV_train_valid_predict_{os.path.basename(path_n_suffix)}.png'
        elif not cv_mapie_graph:
            reg_plot_file = f'{os.path.dirname(path_n_suffix)}/Results_{os.path.basename(path_n_suffix)}.png'
        else:
            reg_plot_file = f'{os.path.dirname(path_n_suffix)}/CV_variability_{os.path.basename(path_n_suffix)}.png'
        path_reduced = '/'.join(f'{reg_plot_file}'.replace('\\','/').split('/')[-2:])

    else:
        folder_graph = f'{os.path.dirname(path_n_suffix)}/csv_test'
        if not cv_mapie_graph:
            reg_plot_file = f'{folder_graph}/Results_{os.path.basename(path_n_suffix)}.png'
        else:
            reg_plot_file = f'{folder_graph}/CV_variability_{os.path.basename(path_n_suffix)}.png'
        path_reduced = '/'.join(f'{reg_plot_file}'.replace('\\','/').split('/')[-3:])
    
    return min_value_graph,max_value_graph,reg_plot_file,path_reduced


def graph_clas(self,Xy_data,params_dict,set_type,path_n_suffix,csv_test=False,print_fun=True):
    '''
    Plot a confusion matrix with the prediction vs actual values
    '''

    # get confusion matrix
    importlib.reload(plt) # needed to avoid threading issues
    if 'CV' in set_type: # CV graphs
        matrix = ConfusionMatrixDisplay.from_predictions(Xy_data[f'y_cv_valid'], Xy_data[f'y_pred_cv_valid'], normalize=None, cmap='Blues') 
    else: # other graphs
        matrix = ConfusionMatrixDisplay.from_predictions(Xy_data[f'y_{set_type}'], Xy_data[f'y_pred_{set_type}'], normalize=None, cmap='Blues') 

    # transfer it to the same format and size used in reg graphs
    _, ax = plt.subplots(figsize=(7.45,6))
    matrix.plot(ax=ax, cmap='Blues')

    if print_fun:
        if 'CV' not in set_type:
            title_set = f'{set_type} set'
        else:
            title_set = set_type
        plt.text(0.5, 1.08, f'{title_set} of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
            fontsize=14, fontweight='bold', transform = ax.transAxes)

    plt.xlabel(f'Predicted {params_dict["y"]}', fontsize=14)
    plt.ylabel(f'{params_dict["y"]}', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # save fig
    if 'CV' in set_type: # CV graphs
        clas_plot_file = f'{os.path.dirname(path_n_suffix)}/CV_train_valid_predict_{os.path.basename(path_n_suffix)}.png'
        path_reduced = '/'.join(f'{clas_plot_file}'.replace('\\','/').split('/')[-2:])

    elif not csv_test:
        clas_plot_file = f'{os.path.dirname(path_n_suffix)}/Results_{os.path.basename(path_n_suffix)}_{set_type}.png'
        path_reduced = '/'.join(f'{clas_plot_file}'.replace('\\','/').split('/')[-2:])

    else:
        folder_graph = f'{os.path.dirname(path_n_suffix)}/csv_test'
        clas_plot_file = f'{folder_graph}/Results_{os.path.basename(path_n_suffix)}_{set_type}.png'
        path_reduced = '/'.join(f'{clas_plot_file}'.replace('\\','/').split('/')[-3:])

    plt.savefig(f'{clas_plot_file}', dpi=300, bbox_inches='tight')

    if print_fun:
        self.args.log.write(f"      -  Graph in: {path_reduced}")


def shap_analysis(self,Xy_data,params_dict,path_n_suffix):
    '''
    Plots and prints the results of the SHAP analysis
    '''

    importlib.reload(plt) # needed to avoid threading issues
    _, _ = plt.subplots(figsize=(7.45,6))

    shap_plot_file = f'{os.path.dirname(path_n_suffix)}/SHAP_{os.path.basename(path_n_suffix)}.png'

    # load and fit the ML model
    loaded_model = load_model(params_dict)
    loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train']) 

    # run the SHAP analysis and save the plot
    explainer = shap.Explainer(loaded_model.predict, Xy_data['X_valid_scaled'], seed=params_dict['seed'])
    try:
        shap_values = explainer(Xy_data['X_valid_scaled'])
    except ValueError:
        shap_values = explainer(Xy_data['X_valid_scaled'],max_evals=(2*len(Xy_data['X_valid_scaled'].columns))+1)

    shap_show = [self.args.shap_show,len(Xy_data['X_valid_scaled'].columns)]
    aspect_shap = 25+((min(shap_show)-2)*5)
    height_shap = 1.2+min(shap_show)/4

    # explainer = shap.TreeExplainer(loaded_model) # in case the standard version doesn't work
    _ = shap.summary_plot(shap_values, Xy_data['X_valid_scaled'], max_display=self.args.shap_show,show=False, plot_size=[7.45,height_shap])

    # set title
    plt.title(f'SHAP analysis of {os.path.basename(path_n_suffix)}', fontsize = 14, fontweight="bold")

    path_reduced = '/'.join(f'{shap_plot_file}'.replace('\\','/').split('/')[-2:])
    print_shap = f"\n   o  SHAP plot saved in {path_reduced}"

    # collect SHAP values and print
    desc_list, min_list, max_list = [],[],[]
    for i,desc in enumerate(Xy_data['X_train_scaled']):
        desc_list.append(desc)
        val_list_indiv= []
        for _,val in enumerate(shap_values.values):
            val_list_indiv.append(val[i])
        min_indiv = min(val_list_indiv)
        max_indiv = max(val_list_indiv)
        min_list.append(min_indiv)
        max_list.append(max_indiv)
    
    if max(max_list, key=abs) > max(min_list, key=abs):
        max_list, min_list, desc_list = (list(t) for t in zip(*sorted(zip(max_list, min_list, desc_list), reverse=True)))
    else:
        min_list, max_list, desc_list = (list(t) for t in zip(*sorted(zip(min_list, max_list, desc_list), reverse=False)))

    for i,desc in enumerate(desc_list):
        print_shap += f"\n      -  {desc} = min: {min_list[i]:.2}, max: {max_list[i]:.2}"

    self.args.log.write(print_shap)

    # adjust width of the colorbar
    plt.gcf().axes[-1].set_aspect(aspect_shap)
    plt.gcf().axes[-1].set_box_aspect(aspect_shap)
    
    plt.savefig(f'{shap_plot_file}', dpi=300, bbox_inches='tight')


def PFI_plot(self,Xy_data,params_dict,path_n_suffix):
    '''
    Plots and prints the results of the PFI analysis
    '''

    importlib.reload(plt) # needed to avoid threading issues
    pfi_plot_file = f'{os.path.dirname(path_n_suffix)}/PFI_{os.path.basename(path_n_suffix)}.png'

    # load and fit the ML model
    loaded_model = load_model(params_dict)
    loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train']) 

    score_model = loaded_model.score(Xy_data['X_valid_scaled'], Xy_data['y_valid'])
    error_type = params_dict['error_type'].lower()
    
    # select scoring function for PFI analysis based on the error type
    if params_dict['type'].lower() == 'reg':
        scoring = {
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_median_absolute_error',
            'r2': 'r2'
        }.get(error_type)
    else:
        scoring = {
            'mcc': make_scorer(matthews_corrcoef),
            'f1': 'f1',
            'acc': 'accuracy'
        }.get(error_type)

    perm_importance = permutation_importance(loaded_model, Xy_data['X_valid_scaled'], Xy_data['y_valid'], scoring=scoring, n_repeats=self.args.pfi_epochs, random_state=params_dict['seed'])

    # sort descriptors and results from PFI
    desc_list, PFI_values, PFI_sd = [],[],[]
    for i,desc in enumerate(Xy_data['X_train_scaled']):
        desc_list.append(desc)
        PFI_values.append(perm_importance.importances_mean[i])
        PFI_sd.append(perm_importance.importances_std[i])

    # sort from higher to lower values and keep only the top self.args.pfi_show descriptors
    PFI_values, PFI_sd, desc_list = (list(t) for t in zip(*sorted(zip(PFI_values, PFI_sd, desc_list), reverse=True)))
    PFI_values_plot = PFI_values[:self.args.pfi_show][::-1]
    desc_list_plot = desc_list[:self.args.pfi_show][::-1]

    # plot and print results
    _, ax = plt.subplots(figsize=(7.45,6))
    y_ticks = np.arange(0, len(desc_list_plot))
    ax.barh(desc_list_plot, PFI_values_plot)
    ax.set_yticks(y_ticks,labels=desc_list_plot,fontsize=14)
    plt.text(0.5, 1.08, f'Permutation feature importances (PFIs) of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
        fontsize=14, fontweight='bold', transform = ax.transAxes)
    ax.set(ylabel=None, xlabel='PFI')

    plt.savefig(f'{pfi_plot_file}', dpi=300, bbox_inches='tight')

    path_reduced = '/'.join(f'{pfi_plot_file}'.replace('\\','/').split('/')[-2:])
    print_PFI = f"\n   o  PFI plot saved in {path_reduced}"

    if params_dict['type'].lower() == 'reg':
        print_PFI += f'\n      Original score (from model.score, R2) = {score_model:.2}'
    elif params_dict['type'].lower() == 'clas':
        print_PFI += f'\n      Original score (from model.score, MCC) = {score_model:.2}'

    for i,desc in enumerate(desc_list):
        print_PFI += f"\n      -  {desc} = {PFI_values[i]:.2} ± {PFI_sd[i]:.2}"
    
    self.args.log.write(print_PFI)


def outlier_plot(self,Xy_data,path_n_suffix,name_points,graph_style):
    '''
    Plots and prints the results of the outlier analysis
    '''

    importlib.reload(plt) # needed to avoid threading issues
    # detect outliers
    outliers_data, print_outliers = outlier_filter(self, Xy_data, name_points)

    # plot data in SD units
    sb.set(style="ticks")

    _, ax = plt.subplots(figsize=(7.45,6))
    plt.text(0.5, 1.08, f'Outlier analysis of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
    fontsize=14, fontweight='bold', transform = ax.transAxes)

    plt.grid(linestyle='--', linewidth=1)
    _ = ax.scatter(outliers_data['train_scaled'], outliers_data['train_scaled'],
            c = graph_style['color_train'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)
    _ = ax.scatter(outliers_data['valid_scaled'], outliers_data['valid_scaled'],
            c = graph_style['color_valid'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)
    if 'test_scaled' in outliers_data:
        _ = ax.scatter(outliers_data['test_scaled'], outliers_data['test_scaled'],
            c = graph_style['color_test'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)
    
    # Set styling preferences and graph limits
    plt.xlabel('SD of the errors',fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('SD of the errors',fontsize=14)
    plt.yticks(fontsize=14)
    
    axis_limit = max(outliers_data['train_scaled'], key=abs)
    if max(outliers_data['valid_scaled'], key=abs) > axis_limit:
        axis_limit = max(outliers_data['valid_scaled'], key=abs)
    if 'test_scaled' in outliers_data:
        if max(outliers_data['test_scaled'], key=abs) > axis_limit:
            axis_limit = max(outliers_data['test_scaled'], key=abs)
    axis_limit = axis_limit+0.5
    if axis_limit < 2.5: # this fixes a problem when representing rectangles in graphs with low SDs
        axis_limit = 2.5
    plt.ylim(-axis_limit, axis_limit)
    plt.xlim(-axis_limit, axis_limit)

    # plot rectangles in corners
    diff_tvalue = axis_limit - self.args.t_value
    Rectangle_top = mpatches.Rectangle(xy=(axis_limit, axis_limit), width=-diff_tvalue, height=-diff_tvalue, facecolor='grey', alpha=0.3)
    Rectangle_bottom = mpatches.Rectangle(xy=(-(axis_limit), -(axis_limit)), width=diff_tvalue, height=diff_tvalue, facecolor='grey', alpha=0.3)
    ax.add_patch(Rectangle_top)
    ax.add_patch(Rectangle_bottom)

    # save plot and print results
    outliers_plot_file = f'{os.path.dirname(path_n_suffix)}/Outliers_{os.path.basename(path_n_suffix)}.png'
    plt.savefig(f'{outliers_plot_file}', dpi=300, bbox_inches='tight')
    
    path_reduced = '/'.join(f'{outliers_plot_file}'.replace('\\','/').split('/')[-2:])
    print_outliers += f"\n   o  Outliers plot saved in {path_reduced}"

    if 'train' not in name_points:
        print_outliers += f'\n      x  No names option (or var missing in CSV file)! Outlier names will not be shown'
    else:
        if 'test_scaled' in outliers_data and 'test' not in name_points:
            print_outliers += f'\n      x  No names option (or var missing in CSV file in the test file)! Outlier names will not be shown'

    print_outliers = outlier_analysis(print_outliers,outliers_data,'train')
    print_outliers = outlier_analysis(print_outliers,outliers_data,'valid')
    if 'test_scaled' in outliers_data:
        print_outliers = outlier_analysis(print_outliers,outliers_data,'test')
    
    self.args.log.write(print_outliers)


def outlier_analysis(print_outliers,outliers_data,outliers_set):
    '''
    Analyzes the outlier results
    '''
    
    if outliers_set == 'train':
        label_set = 'Train'
        outliers_label = 'outliers_train'
        n_points_label = 'train_scaled'
        outliers_name = 'names_train'
    elif outliers_set == 'valid':
        label_set = 'Validation'
        outliers_label = 'outliers_valid'
        n_points_label = 'valid_scaled'
        outliers_name = 'names_valid'
    elif outliers_set == 'test':
        label_set = 'Test'
        outliers_label = 'outliers_test'
        n_points_label = 'test_scaled'
        outliers_name = 'names_test'

    per_cent = (len(outliers_data[outliers_label])/len(outliers_data[n_points_label]))*100
    print_outliers += f"\n      {label_set}: {len(outliers_data[outliers_label])} outliers out of {len(outliers_data[n_points_label])} datapoints ({per_cent:.1f}%)"
    for val,name in zip(outliers_data[outliers_label], outliers_data[outliers_name]):
        print_outliers += f"\n      -  {name} ({val:.2} SDs)"
    return print_outliers


def outlier_filter(self, Xy_data, name_points):
    '''
    Calculates and stores absolute errors in SD units for all the sets
    '''
    
    # calculate absolute errors between predicted y and actual values
    outliers_train = [abs(x-y) for x,y in zip(Xy_data['y_train'],Xy_data['y_pred_train'])]
    outliers_valid = [abs(x-y) for x,y in zip(Xy_data['y_valid'],Xy_data['y_pred_valid'])]
    if 'y_pred_test' in Xy_data and not Xy_data['y_test'].isnull().values.any() and len(Xy_data['y_test']) > 0:
        outliers_test = [abs(x-y) for x,y in zip(Xy_data['y_test'],Xy_data['y_pred_test'])]

    # the errors are scaled using standard deviation units. When the absolute
    # error is larger than the t-value, the point is considered an outlier. All the sets
    # use the mean and SD of the train set
    outliers_mean = np.mean(outliers_train)
    outliers_sd = np.std(outliers_train)

    outliers_data = {}
    outliers_data['train_scaled'] = (outliers_train-outliers_mean)/outliers_sd
    outliers_data['valid_scaled'] = (outliers_valid-outliers_mean)/outliers_sd
    if 'y_pred_test' in Xy_data and not Xy_data['y_test'].isnull().values.any() and len(Xy_data['y_test']) > 0:
        outliers_data['test_scaled'] = (outliers_test-outliers_mean)/outliers_sd

    print_outliers, naming, naming_test = '', False, False
    if 'train' in name_points:
        naming = True
        if 'test' in name_points:
            naming_test = True

    outliers_data['outliers_train'], outliers_data['names_train'] = detect_outliers(self, outliers_data['train_scaled'], name_points, naming, 'train')
    outliers_data['outliers_valid'], outliers_data['names_valid'] = detect_outliers(self, outliers_data['valid_scaled'], name_points, naming, 'valid')
    if 'y_pred_test' in Xy_data and not Xy_data['y_test'].isnull().values.any() and len(Xy_data['y_test']) > 0:
        outliers_data['outliers_test'], outliers_data['names_test'] = detect_outliers(self, outliers_data['test_scaled'], name_points, naming_test, 'test')
    
    return outliers_data, print_outliers


def detect_outliers(self, outliers_scaled, name_points, naming_detect, set_type):
    '''
    Detects and store outliers with their corresponding datapoint names
    '''

    val_outliers = []
    name_outliers = []
    if naming_detect:
        name_points_list = name_points[set_type].to_list()
    for i,val in enumerate(outliers_scaled):
        if val > self.args.t_value or val < -self.args.t_value:
            val_outliers.append(val)
            if naming_detect:
                name_outliers.append(name_points_list[i])

    return val_outliers, name_outliers


def distribution_plot(self,Xy_data,path_n_suffix,params_dict):
    '''
    Plots histogram (reg) or bin plot (clas).
    '''

    # make graph
    importlib.reload(plt) # needed to avoid threading issues
    sb.set(style="ticks")

    _, ax = plt.subplots(figsize=(7.45,6))
    plt.text(0.5, 1.08, f'y-values distribution (train+valid.) of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
    fontsize=14, fontweight='bold', transform = ax.transAxes)

    plt.grid(linestyle='--', linewidth=1)

    # combine train and validation sets
    y_combined = pd.concat([Xy_data['y_train'],Xy_data['y_valid']], axis=0).reset_index(drop=True)

    # plot histogram, quartile lines and the points in each quartile
    if params_dict['type'].lower() == 'reg':
        y_dist_dict,ax = plot_quartiles(y_combined,ax)
    
    # plot a bar plot with the count of each y type
    elif params_dict['type'].lower() == 'clas':
        y_dist_dict,ax = plot_y_count(y_combined,ax)

    # set styling preferences and graph limits
    plt.xlabel(f'{params_dict["y"]} values',fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('Frequency',fontsize=14)
    plt.yticks(fontsize=14)

    # set limits
    if params_dict['type'].lower() == 'reg':
        border_y_range = 0.1*np.abs(max(y_combined)-min(y_combined))
        plt.xlim(min(y_combined)-border_y_range, max(y_combined)+border_y_range)

    # save plot and print results
    orig_distrib_file = f'y_distribution_{os.path.basename(path_n_suffix)}.png'
    plt.savefig(f'{orig_distrib_file}', dpi=300, bbox_inches='tight')
    # for a VERY weird reason, I need to save the figure in the working directory and then move it into PREDICT
    final_distrib_file = f'{os.path.dirname(path_n_suffix)}/y_distribution_{os.path.basename(path_n_suffix)}.png'
    shutil.move(orig_distrib_file, final_distrib_file)

    path_reduced = '/'.join(f'{final_distrib_file}'.replace('\\','/').split('/')[-2:])
    print_distrib = f"\n   o  y-values distribution plot saved in {path_reduced}"

    # print the quartile results
    if params_dict['type'].lower() == 'reg':
        print_distrib += f"\n      Ideally, the number of datapoints in the four quartiles of the y-range should be uniform (25% population in each quartile) to have similar confidence intervals in the predictions across the y-range"
        quartile_pops = [len(y_dist_dict['q1_points']),len(y_dist_dict['q2_points']),len(y_dist_dict['q3_points']),len(y_dist_dict['q4_points'])]
        print_distrib += f"\n      -  The number of points in each quartile is Q1: {quartile_pops[0]}, Q2: {quartile_pops[1]}, Q3: {quartile_pops[2]}, Q4: {quartile_pops[3]}"
        quartile_min_idx = quartile_pops.index(min(quartile_pops))
        quartile_max_idx = quartile_pops.index(max(quartile_pops))
        if 4*min(quartile_pops) < max(quartile_pops):
            print_distrib += f"\n      x  WARNING! Your data is not uniform (Q{quartile_min_idx+1} has {min(quartile_pops)} points while Q{quartile_max_idx+1} has {max(quartile_pops)})"
        elif 2*min(quartile_pops) < max(quartile_pops):
            print_distrib += f"\n      x  WARNING! Your data is slightly not uniform (Q{quartile_min_idx+1} has {min(quartile_pops)} points while Q{quartile_max_idx+1} has {max(quartile_pops)})"
        else:
            print_distrib += f"\n      o  Your data seems quite uniform"

    elif params_dict['type'].lower() == 'clas':
        if len(y_dist_dict['count_labels']) > 2:
            self.args.log.write(f"\n      ADAPT THIS PART for 3+ prediction classes!!")
            sys.exit()
        print_distrib += f"\n      Ideally, the number of datapoints in each prediction class should be uniform (50% population per class) to have similar reliability in the predictions across classes"
        distrib_counts = [y_dist_dict['count_labels'][0],y_dist_dict['count_labels'][1]]
        print_distrib += f"\n      - The number of points in each class is {y_dist_dict['type_labels'][0]}: {y_dist_dict['count_labels'][0]}, {y_dist_dict['type_labels'][1]}: {y_dist_dict['count_labels'][1]}"
        class_min_idx = distrib_counts.index(min(distrib_counts))
        class_max_idx = distrib_counts.index(max(distrib_counts))
        if 3*min(distrib_counts) < max(distrib_counts):
            print_distrib += f"\n      x  WARNING! Your data is not uniform (class {y_dist_dict['type_labels'][class_min_idx]} has {min(distrib_counts)} points while class {y_dist_dict['type_labels'][class_max_idx]} has {max(distrib_counts)})"
        elif 1.5*min(distrib_counts) < max(distrib_counts):
            print_distrib += f"\n      x  WARNING! Your data is slightly not uniform (class {y_dist_dict['type_labels'][class_min_idx]} has {min(distrib_counts)} points while class {y_dist_dict['type_labels'][class_max_idx]} has {max(distrib_counts)})"
        else:
            print_distrib += f"\n      o  Your data seems quite uniform"

    self.args.log.write(print_distrib)


def plot_quartiles(y_combined,ax):
    '''
    Plot histogram, quartile lines and the points in each quartile.
    '''

    bins = max([round(len(y_combined)/5),5]) # at least 5 bins until 25 points
    # histogram
    y_hist, _, _ = ax.hist(y_combined, bins=bins,
                color='#1f77b4', edgecolor='k', linewidth=1, alpha=1)

    # uniformity lines to plot
    separation_range = np.abs(max(y_combined)-min(y_combined))/4
    quart_dict = {'line_1': min(y_combined),
                    'line_2': min(y_combined) + separation_range,
                    'line_3': min(y_combined) + (2*separation_range),
                    'line_4': min(y_combined) + (3*separation_range),
                    'line_5': max(y_combined)}

    lines_plot = [quart_dict[line] for line in quart_dict]
    ax.vlines([lines_plot], ymin=max(y_hist)*1.05, ymax=max(y_hist)*1.3, colors='crimson', linestyles='--')

    # points in each quartile
    quart_dict['q1_points'] = []
    quart_dict['q2_points'] = []
    quart_dict['q3_points'] = []
    quart_dict['q4_points'] = []

    for val in y_combined:
        if val < quart_dict['line_2']:
            quart_dict['q1_points'].append(val)
        elif quart_dict['line_2'] < val < quart_dict['line_3']:
            quart_dict['q2_points'].append(val)
        elif quart_dict['line_3'] < val < quart_dict['line_4']:
            quart_dict['q3_points'].append(val)
        elif val >= quart_dict['line_4']:
            quart_dict['q4_points'].append(val)

    x_quart = 0.185
    for quart in quart_dict:
        if 'points' in quart:
            plt.text(x_quart, 0.845, f'Q{quart[1]}\n{len(quart_dict[quart])} points', horizontalalignment='center',
                    fontsize=12, transform = ax.transAxes, backgroundcolor='w')
            x_quart += 0.209

    return quart_dict,ax


def plot_y_count(y_combined,ax):
    '''
    Plot a bar plot with the count of each y type.
    '''

    # get the number of times that each y type is included
    labels_used = set(y_combined)
    type_labels,count_labels = [],[]
    for label in labels_used:
        type_labels.append(label)
        count_labels.append(len(y_combined[y_combined == label]))

    _ = ax.bar(type_labels, count_labels, tick_label=type_labels,
                color='#1f77b4', edgecolor='k', linewidth=1, alpha=1,
                width=0.4)

    y_dist_dict = {'type_labels': type_labels,
                   'count_labels': count_labels}

    return y_dist_dict,ax


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
            kfold_type = -1 # -1 for Jackknife+ in MAPIE
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

    if len(Xy_data_df) == 1:
        Xy_data_df = Xy_data_df[0]
        params_df = params_df[0]
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
    else:
        Xy_data = []
        for i in range(len(Xy_data_df)):
            Xy_data_df_i = Xy_data_df[i]
            params_df_i = params_df[i]
            # load only the descriptors used in the model and standardize X
            Xy_train_df = Xy_data_df_i[Xy_data_df_i.Set == 'Training'].reset_index(drop=True)
            Xy_valid_df = Xy_data_df_i[Xy_data_df_i.Set == 'Validation'].reset_index(drop=True)
            Xy_test_df = Xy_data_df_i[Xy_data_df_i.Set == 'Test'].reset_index(drop=True)

            descs_model = ast.literal_eval(params_df_i['X_descriptors'][0])
            Xy_data_i = {}
            Xy_data_i['X_train'] = Xy_train_df[descs_model]
            Xy_data_i['X_valid'] = Xy_valid_df[descs_model]
            Xy_data_i['y_train'] = Xy_train_df[params_df_i['y'][0]]
            Xy_data_i['y_valid'] = Xy_valid_df[params_df_i['y'][0]]
            Xy_data_i['X_train_scaled'], Xy_data_i['X_valid_scaled'] = standardize(self, Xy_data_i['X_train'], Xy_data_i['X_valid'])
            # test sets are scaled later in PREDICT

            Xy_data_i['X_test'] = Xy_test_df[descs_model]
            Xy_data_i['y_test'] = Xy_test_df[params_df_i['y'][0]]

            point_count = {}
            point_count['train'] = len(Xy_data_i['X_train_scaled'])
            point_count['valid'] = len(Xy_data_i['X_valid_scaled'])
            point_count['test'] = len(Xy_data_i['X_test'])

            Xy_data.append(Xy_data_i)

            params_name = os.path.basename(params_path)

    return Xy_data, params_df, params_path, suffix_title, Xy_test_df


def load_dfs(self,folder_model,module):
    '''
    Loads the parameters and Xy databases from the GENERATE folder as dataframes
    '''
    
    if os.getcwd() in f"{folder_model}":
        path_db = folder_model
    else:
        path_db = f"{Path(os.getcwd()).joinpath(folder_model)}"
    suffix = '(with no PFI filter)'
    suffix_title = 'No_PFI'
    if os.path.exists(path_db):
        csv_files = glob.glob(f'{Path(path_db).joinpath("*.csv")}')
        Xy_data_dfs = []
        params_dfs = []
        for csv_file in csv_files:
            if 'PFI' in os.path.basename(csv_file).replace('.csv','_').split('_'):
                suffix = '(with PFI filter)'
                suffix_title = 'PFI'
            if csv_file.endswith('_db.csv'):
                Xy_data_df = load_database(self,csv_file,module)
                Xy_data_dfs.append(Xy_data_df)
                Xy_path = csv_file
            else:
                params_df = load_database(self,csv_file,module)
                params_dfs.append(params_df)
                params_path = csv_file
    else:
        self.args.log.write(f"\nx  The folder with the model and database ({path_db}) does not exist! Did you use the destination=PATH option in the other modules?")
        sys.exit()

    return Xy_data_dfs, Xy_path, params_dfs, params_path, suffix, suffix_title


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

    importlib.reload(plt) # needed to avoid threading issues
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

        path_reduced = '/'.join(f'{heatmap_path}'.replace('\\','/').split('/')[-2:])
        if module.lower() == 'curate':
            self.args.log.write(f'\no  The Pearson heatmap was stored in {path_reduced}.')
        elif module.lower() == 'predict':
            self.args.log.write(f'\n   o  The Pearson heatmap was stored in {path_reduced}.')

    return corr_matrix

def cv_test(self,verify_results,Xy_data,params_dict,params_path,suffix_title,module):
    '''
    Performs a cross-validation on the training+validation dataset.
    '''      

    if module.lower() == 'verify':
        # set PATH names and plot
        base_csv_name = '_'.join(os.path.basename(params_path).replace('.csv','_').split('_')[0:2])
        base_csv_name = f'VERIFY/{base_csv_name}'
        base_csv_path = f"{Path(os.getcwd()).joinpath(base_csv_name)}"
        path_n_suffix = f'{base_csv_path}_{suffix_title}'

    # Fit the original model with the training set
    loaded_model = load_model(params_dict)
    loaded_model.fit(np.asarray(Xy_data['X_train_scaled']).tolist(), np.asarray(Xy_data['y_train']).tolist())
    data_cv,_ = load_n_predict(self, params_dict, Xy_data)
    
    cv_y_list,cv_y_pred_list = [],[]
    data_cv = {}
    # combine training and validation for CV
    X_combined = pd.concat([Xy_data['X_train'],Xy_data['X_valid']], axis=0).reset_index(drop=True)
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
        XY_cv['X_train_scaled'], XY_cv['X_valid_scaled'] = standardize(self,X_combined.loc[train_index],X_combined.loc[valid_index])
        XY_cv['y_train'] = y_combined.loc[train_index]
        XY_cv['y_valid'] = y_combined.loc[valid_index]
        data_cv,_ = load_n_predict(self, params_dict, XY_cv)

        for y_cv,y_pred_cv in zip(data_cv['y_valid'],data_cv['y_pred_valid']):
            cv_y_list.append(y_cv)
            cv_y_pred_list.append(y_pred_cv)

    # calculate metrics (and plot graphs for VERIFY)
    if module.lower() == 'verify':
        graph_style = get_graph_style()
    Xy_data["y_cv_valid"] = cv_y_list
    Xy_data["y_pred_cv_valid"] = cv_y_pred_list

    if params_dict['type'].lower() == 'reg':
        verify_results['cv_r2'], verify_results['cv_mae'], verify_results['cv_rmse'] = get_prediction_results(params_dict,cv_y_list,cv_y_pred_list)
        if module.lower() == 'verify':
            set_types = [type_cv]
            _ = graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style)

    elif params_dict['type'].lower() == 'clas':
        verify_results['cv_acc'], verify_results['cv_f1'], verify_results['cv_mcc'] = get_prediction_results(params_dict,cv_y_list,cv_y_pred_list)
        if module.lower() == 'verify':
            set_type = f'{type_cv} train+valid.'
            _ = graph_clas(self,Xy_data,params_dict,set_type,path_n_suffix)

    verify_results['cv_score'] = verify_results[f'cv_{verify_results["error_type"].lower()}']

    if module.lower() == 'verify':
        # save CSV with results
        Xy_cv_df = pd.DataFrame()
        Xy_cv_df[f'{params_dict["y"]}'] = Xy_data["y_cv_valid"]
        Xy_cv_df[f'{params_dict["y"]}_pred'] = Xy_data["y_pred_cv_valid"]
        csv_test_path = f'{os.path.dirname(path_n_suffix)}/CV_predictions_{os.path.basename(path_n_suffix)}.csv'
        _ = Xy_cv_df.to_csv(csv_test_path, index = None, header=True)

    else:
        path_n_suffix,type_cv = None,None

    return verify_results,type_cv,path_n_suffix


def plot_metrics(path_n_suffix,verify_metrics,verify_results):
    '''
    Creates a plot with the results of the flawed models in VERIFY
    '''

    importlib.reload(plt) # needed to avoid threading issues
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
            elif test_color == '#c5c57d':
                txt_bar = 'unclear'
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
        unclear_thres_line = verify_metrics['unclear_higher_thres']  
    else:
        thres_line = verify_metrics['lower_thres']
        unclear_thres_line = verify_metrics['unclear_lower_thres']
        arrow_length = -arrow_length

    width = 2
    xmin = 0.237
    thres = ax.axhline(thres_line,xmin=xmin, color='black',ls='--', label='thres', zorder=0)
    thres = ax.axhline(unclear_thres_line,xmin=xmin, color='black',ls='--', label='thres', zorder=0)

    x_arrow = 0.5
    style = mpatches.ArrowStyle('simple', head_length=4.5*width, head_width=3.5*width, tail_width=width)
    arrow = mpatches.FancyArrowPatch((x_arrow, thres_line), (x_arrow, thres_line+arrow_length), 
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
    plt.figlegend([thres,arrow], [f'Limits: {round(thres_line,2)} (pass), {round(unclear_thres_line,2)} (unclear)','Pass test'], handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
                    loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, fontsize=14)

    # Add gridlines
    ax.grid(linestyle='--', linewidth=1)

    # save plot
    verify_plot_file = f'{os.path.dirname(path_n_suffix)}/VERIFY_tests_{os.path.basename(path_n_suffix)}.png'
    plt.savefig(verify_plot_file, dpi=300, bbox_inches='tight')

    path_reduced = '/'.join(f'{verify_plot_file}'.replace('\\','/').split('/')[-2:])
    print_ver = f"\n   o  VERIFY plot saved in {path_reduced}"

    return print_ver

