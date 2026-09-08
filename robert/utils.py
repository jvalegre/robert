######################################################.
#          This file stores functions used           #
#                in multiple modules                 #
######################################################.

import os
import sys
import time
import getopt
import glob
import json
import yaml
import ast
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Force single-threaded execution for reproducibility across platforms
# This prevents numerical differences between Windows/Ubuntu in parallel operations
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
from matplotlib import pyplot as plt
import importlib
import matplotlib.patches as mpatches
import matplotlib.colors as mcolor
from matplotlib.legend_handler import HandlerPatch
from matplotlib.ticker import FormatStrFormatter
import shap
import seaborn as sb
from scipy import stats
from scipy.spatial.distance import cdist
from importlib.resources import files
# sklearnex was deactivated in ROBERT v2.1 because it only accelerated RF
# try:
#     from sklearnex import patch_sklearn
#     patch_sklearn(verbose=True)
# except (ModuleNotFoundError,ImportError):
#     pass
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             matthews_corrcoef, accuracy_score, f1_score, make_scorer,
                             ConfusionMatrixDisplay)
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    )
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, RepeatedKFold, KFold, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
from robert.argument_parser import set_options, var_dict
from bayes_opt import BayesianOptimization
import warnings # this avoids warnings from sklearn
warnings.filterwarnings("ignore")


robert_version = "2.2.0"
time_run = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
robert_ref = "Dalmau, D.; Alegre Requena, J. V. WIREs Comput Mol Sci. 2024, 14, e1733. Dalmau, D.; Sigman, M. S.; Alegre-Requena, J. V. Chem. Sci. 2025, 16, 8555-8560."


def _parse_class_mapping_reverse(mapping):
    """
    Return a class-code -> original-label mapping with integer keys.
    """

    if mapping is None or (isinstance(mapping, float) and pd.isna(mapping)):
        return None
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except (json.JSONDecodeError, TypeError):
            mapping = ast.literal_eval(mapping)
    if not isinstance(mapping, dict):
        return None
    return {int(key): value for key, value in mapping.items()}


def _sorted_class_values(values):
    """
    Sort class labels deterministically, including mixed string/numeric labels.
    """

    return sorted(pd.unique(values), key=lambda value: (str(type(value)), str(value)))


def _integer_like_class_values(values):
    try:
        numeric_values = pd.to_numeric(pd.Series(values), errors="raise")
    except (TypeError, ValueError):
        return False
    return np.all(np.isfinite(numeric_values)) and np.all(
        np.equal(np.mod(numeric_values, 1), 0)
    )


def _classification_vote(values):
    """
    Combine repeated classification predictions with majority voting.
    """

    class_values, counts = np.unique(np.asarray(values), return_counts=True)
    return int(class_values[np.argmax(counts)])


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
        # Simulate sys.argv for use in an executable environment
        sys.argv = ["launcher.exe"]
        for k, v in sys_args.items():
            sys.argv.append(k)
            if v is not None:
                sys.argv.append(str(v))

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
    ]
    int_args = [
        'pfi_epochs',
        'epochs',
        'nprocs',
        'pfi_max',
        'kfold',
        'repeat_kfolds',
        'shap_show',
        'pfi_show',
        "seed",
        "init_points",
        "n_iter",
    ]
    float_args = [
        'pfi_threshold',
        't_value',
        'thres_x',
        'thres_y',
        'test_set',
        'desc_thres',
        'alpha',
        'expect_improv'
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
  --kfold INT (default=5) : number of folds for k-fold cross-validation of the RFECV feature selector
  --categorical "onehot" or "numbers" (default="onehot") : type of conversion for categorical variables
  --corr_filter_x BOOL (default=True) : activate/disable the correlation filter of descriptors X
  --rfecv_filter BOOL (default=True) : activate/disable RFECV/PFI descriptor selection (runs
    independently of --corr_filter_x, only when descriptors outnumber one-third of the datapoints
    after the correlation filter)

* Affecting model screening in GENERATE:
  --model "[MODEL1,MODEL2,etc]" (default=["RF","GB","NN","MVL"]) : ML models to use in the ML scan (i.e., "[RF,GB]")
  --type "reg" or "clas" (default="reg") : regression or classification models
  --pfi_max INT (default=0) : number of features to keep in the PFI models

* Affecting tests, VERIFY:
  --kfold INT (default=5) : number of folds for k-fold cross-validation

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


def build_params_dirs(self):
    '''
    Builds the (params_dirs, suffixes, suffix_titles, model_names) quadruplet that
    VERIFY/PREDICT loop over. Normally this is just the winning No_PFI/PFI model
    (GENERATE/Best_model), and model_names is [None, None] (one shared log file, as before).
    With --all_models active, GENERATE also stages every model's own files into
    GENERATE/All_models/{No_PFI,PFI}/{MODEL}/ (see stage_all_models() in generate_utils.py) -
    this discovers whichever of those per-model folders actually exist (not every model
    necessarily has a PFI variant) and loops over all of them instead, so each model gets its
    own full VERIFY/PREDICT pass. model_names then carries the actual model name per entry,
    which the caller uses to split the log file per model (see verify.py/predict.py) instead
    of merging every model's output into one shared VERIFY_data.dat/PREDICT_data.dat, where
    REPORT would have no reliable way to tell which lines belong to which model.
    '''

    if 'GENERATE/Best_model' in self.args.params_dir:
        if getattr(self.args, 'all_models', False):
            params_dirs, suffixes, suffix_titles, model_names = [], [], [], []
            all_models_dir = self.args.params_dir.replace('Best_model', 'All_models')
            # group entries by MODEL first (No_PFI then PFI for that model), not by PFI
            # variant first - verify.py/predict.py write both entries of a model into the
            # SAME per-model log file, only opening a new Logger (which truncates via mode
            # 'w') when the model name changes. Grouping by PFI variant first would revisit
            # each model a second, non-consecutive time (once per variant), so the later
            # visit would reopen and wipe out the earlier variant's already-written block
            discovered = {}
            for pfi_variant in ['No_PFI', 'PFI']:
                for model_dir in sorted(glob.glob(f'{all_models_dir}/{pfi_variant}/*')):
                    discovered.setdefault(os.path.basename(model_dir), {})[pfi_variant] = model_dir
            for model_name in sorted(discovered):
                for pfi_variant, pfi_label in [('No_PFI', 'no PFI filter'), ('PFI', 'PFI filter')]:
                    if pfi_variant in discovered[model_name]:
                        params_dirs.append(discovered[model_name][pfi_variant])
                        suffixes.append(f'(with {pfi_label}, model {model_name})')
                        # NOT f'{model_name}_{pfi_variant}': every downstream filename/path
                        # builder (save_predictions, plot_metrics, etc.) already independently
                        # prepends model_data['model'] to suffix_title (see e.g. predict_utils.py's
                        # base_csv_name = f"PREDICT/{model_data['model']}_{suffix_title}") - adding
                        # the model name here too would double it up (e.g. "GB_GB_No_PFI")
                        suffix_titles.append(pfi_variant)
                        model_names.append(model_name)
        else:
            params_dirs = [f'{self.args.params_dir}/No_PFI',f'{self.args.params_dir}/PFI']
            suffixes = ['(with no PFI filter)','(with PFI filter)']
            suffix_titles = ['No_PFI','PFI']
            model_names = [None, None]
    else:
        params_dirs = [self.args.params_dir]
        suffixes = ['custom']
        suffix_titles = ['custom']
        model_names = [None]

    return params_dirs, suffixes, suffix_titles, model_names


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

    # check for spaces in csv file names
    if " " in str(self.csv_name):
        print("\nx  ERROR: The input CSV file name contains spaces. Please remove spaces from the file name and try again. Spaces in file names can cause problems. Example: use 'my_data.csv' instead of 'my data.csv'.")
        sys.exit()
    if self.csv_test and " " in str(self.csv_test):
        print("\nx  ERROR: The test CSV file name contains spaces. Please remove spaces from the file name and try again. Spaces in file names can cause problems. Example: use 'test_data.csv' instead of 'test data.csv'.")
        sys.exit()

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
            self.path_icons = files("robert").joinpath("report")

        # sklearnex was deactivated in ROBERT v2.1 because it only accelerated RF
        # using or not the intelex accelerator might affect the results
        # if robert_module.upper() in ['GENERATE','VERIFY','PREDICT']:
            # try:
            #     import sklearnex
            #     pass
            # except (ModuleNotFoundError,ImportError):
            #     self.log.write(f"\nx  WARNING! The scikit-learn-intelex accelerator is not installed, the results might vary if it is installed and the execution times might become much longer (if available, use 'pip install scikit-learn-intelex')")

        if robert_module.upper() in ['GENERATE', 'VERIFY']:
            # adjust the default value of error_type for classification
            if self.type.lower() == 'clas':
                if self.error_type not in ['acc', 'mcc', 'f1']:
                    self.error_type = 'mcc'

        if robert_module.upper() in ['PREDICT','VERIFY','REPORT']:
            if self.params_dir == '':
                self.params_dir = 'GENERATE/Best_model'

        if robert_module.upper() in ['CURATE','GENERATE']:
            if self.type.lower() == 'clas':
                if any(x.upper() == 'MVL' for x in self.model):
                    self.model = [x if x.upper() != 'MVL' else 'AdaB' for x in self.model]
            
            models_gen = [] # use capital letters in all the models
            for model_type in self.model:
                models_gen.append(model_type.upper())
            self.model = models_gen

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
                
                # Load class labels if they exist (for classification with string labels)
                if 'class_mapping_reverse' in curate_df.columns:
                    self.class_mapping_reverse = _parse_class_mapping_reverse(curate_df['class_mapping_reverse'][0])
                if 'class_0_label' in curate_df.columns and 'class_1_label' in curate_df.columns:
                    self.class_0_label = curate_df['class_0_label'][0]
                    self.class_1_label = curate_df['class_1_label'][0]

        elif robert_module.upper() in ['PREDICT','VERIFY']:
            if robert_module.upper() == 'PREDICT':
                self.log.write(f"\no  Representation of predictions and analysis of ML models with the PREDICT module")
            elif robert_module.upper() == 'VERIFY':
                self.log.write(f"\no  Starting tests to verify the prediction ability of the ML models with the VERIFY module")

            if '' in [self.names,self.y,self.csv_name]:
                # tries to get names from GENERATE
                if 'GENERATE/Best_model' in self.params_dir:
                    params_dirs = [f'{self.params_dir}/No_PFI',f'{self.params_dir}/PFI']
                else:
                    params_dirs = [self.params_dir]
                self.args = self
                _,_,_,model_data,csv_name = load_dfs(self,params_dirs[0],'predict',sanity_check=True)

                self.names = model_data["names"]
                self.y = model_data["y"]
                self.csv_name = csv_name
                # Load error_type if present
                if "error_type" in model_data:
                    self.error_type = model_data["error_type"]
                # Ensure we also load the type if present
                if "type" in model_data:
                    self.type = model_data["type"]

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
    
    REPRODUCIBILITY GUARANTEES:
    - Columns are sorted alphabetically before any operation
    - Rows are sorted by y value to ensure consistent ordering
    - Tie-breaking in correlations uses alphabetical order
    - RFECV descriptor selection uses sorted feature importances with alphabetical tie-breaking
    """

    txt_corr = ''
    
    # Sort columns alphabetically and rows by y value for reproducibility
    descriptor_cols = [col for col in csv_df.columns if col not in self.args.ignore and col != self.args.y]
    descriptor_cols_sorted = sorted(descriptor_cols)
    other_cols = [col for col in csv_df.columns if col in self.args.ignore or col == self.args.y]
    csv_df = csv_df[descriptor_cols_sorted + other_cols].copy()
    csv_df = csv_df.reset_index(drop=True).sort_values(by=self.args.y, kind='stable').reset_index(drop=True)

    txt_corr += f'\no  Correlation filter activated with these thresholds: thres_x = {self.args.thres_x}'
    if self.args.corr_filter_y:
        txt_corr += f', thres_y = {self.args.thres_y}'

    descriptors_drop = []
    txt_corr += f'\n   Excluded descriptors:'
    
    # First pass: remove constant descriptors and those with low correlation to y
    for _,column in enumerate(csv_df.columns):
        if column not in descriptors_drop and column not in self.args.ignore and column != self.args.y:
            # Remove descriptors where all values are the same
            if len(set(csv_df[column])) == 1:
                descriptors_drop.append(column)
                txt_corr += f'\n   - {column}: all the values are the same'

            # Remove descriptors with low correlation to the response values
            # (skip columns already dropped above as constant - no fresh rsquared_y to check)
            if self.args.corr_filter_y and column not in descriptors_drop:
                # Calculate correlation with y for remaining descriptors
                res_y = stats.linregress(csv_df[column],csv_df[self.args.y])
                rsquared_y = res_y.rvalue**2
                if rsquared_y < self.args.thres_y:
                    descriptors_drop.append(column)
                    txt_corr += f'\n   - {column}: R**2 = {rsquared_y:.2} with the {self.args.y} values'

    self.args.log.write(txt_corr)

    # Second pass: remove highly correlated descriptors (always removing the most correlated first)
    txt_corr = ''
    csv_df_filtered = csv_df.drop(descriptors_drop, axis=1)
    csv_df_X_filtered = csv_df_filtered.drop([self.args.y] + self.args.ignore, axis=1)
    if self.args.corr_filter_x and len(csv_df_X_filtered.columns) > 1:        
        # Calculate R2 correlation matrix between descriptors
        corr_matrix = csv_df_X_filtered.corr().abs()
        corr_matrix_r2 = corr_matrix ** 2
        upper = corr_matrix_r2.where(np.triu(np.ones(corr_matrix_r2.shape), k=1).astype(bool))
        
        # Calculate R2 of each descriptor with y (for deciding which one to drop)
        r2_with_y = {}
        for col in csv_df_X_filtered.columns:
            res_y = stats.linregress(csv_df_filtered[col], csv_df_filtered[self.args.y])
            r2_with_y[col] = res_y.rvalue**2
        
        # Iteratively remove the most correlated descriptors
        while True:
            # Find the maximum R2 correlation
            max_r2 = upper.max().max()
            if max_r2 <= self.args.thres_x or str(max_r2).lower() == 'nan':
                break
            
            # Get ALL pairs with maximum correlation, round to avoid floating point issues
            upper_rounded = upper.round(10)
            max_r2_rounded = upper_rounded.max().max()
            row_idx, col_idx = np.where(upper_rounded == max_r2_rounded)
            
            # Sort pairs alphabetically for deterministic selection
            pairs = [(upper.index[row_idx[i]], upper.columns[col_idx[i]]) for i in range(len(row_idx))]
            pairs.sort()
            col_name_1, col_name_2 = pairs[0]
            
            # Drop the descriptor with lower R2 to y, round for comparison
            r2_1 = round(r2_with_y[col_name_1], 10)
            r2_2 = round(r2_with_y[col_name_2], 10)
            
            if r2_1 == r2_2:
                # Tied on R2 with y, drop alphabetically later one
                drop_col = col_name_1 if col_name_1 > col_name_2 else col_name_2
                keep_col = col_name_2 if col_name_1 > col_name_2 else col_name_1
            elif r2_1 < r2_2:
                drop_col = col_name_1
                keep_col = col_name_2
            else:
                drop_col = col_name_2
                keep_col = col_name_1
            
            descriptors_drop.append(drop_col)
            txt_corr += f'\n   - {drop_col} removed (R2 = {max_r2:.2f} with {keep_col}), kept more predictive descriptor'
            
            upper = upper.drop(index=drop_col, columns=drop_col)
            del r2_with_y[drop_col]

        # Collect all descriptors that were removed
        to_drop = [col for col in csv_df_X_filtered.columns if col not in upper.columns]

        if to_drop:
            txt_corr += f'\n   - Total: {len(to_drop)} descriptors removed due to high correlation with other descriptors'

    # drop descriptors that did not pass the filters
    csv_df_filtered = csv_df.drop(descriptors_drop, axis=1)

    # recompute n_descps AFTER the correlation filter (not before it, at line ~731) - the RFECV/
    # PFI check below decides whether descriptor selection is still needed on what's left post-
    # correlation-filter, not on the original pre-filter count. Using the pre-filter count here
    # triggered RFECV/PFI even when the correlation filter alone had already brought the
    # descriptor count under the one-third threshold, running an expensive per-model selection
    # step for no benefit
    n_descps = len(csv_df_filtered.columns)-len(self.args.ignore)-1

    if len(descriptors_drop) == 0:
        txt_corr += f'\n   -  No descriptors were removed'

    self.args.log.write(txt_corr)

    # Check if descriptors are more than one third of datapoints
    txt_corr = ''
    descriptors_used = {}
    csv_df_per_model = {}

    num_descriptors = round(len(csv_df[self.args.y]) / 3)
    if self.args.rfecv_filter and n_descps > num_descriptors:
        cv_type = f'{self.args.repeat_kfolds}x {self.args.kfold}_fold_cv'
        txt_corr += f'\no  There are more descriptors than one-third of the data points. A Recursive Feature Elimination with Cross-Validation (RFECV) or permutation feature importance (PFI) using {cv_type} will be performed to select the most relevant descriptors for each model'
        self.args.log.write(txt_corr)
        txt_corr = ''

        # Perform RFECV for each model specified by the user
        X_df = csv_df_filtered.drop([self.args.y] + self.args.ignore, axis=1)
        X_scaled_df,_ = scale_df(X_df,None)
        y_df = csv_df_filtered[self.args.y]

        for model in sorted(self.args.model):
            # Load the parameters of the minimalist model used for RFECV
            rfecv_params = load_minimal_model(model)
            rfecv_params = model_adjust_params(self, model, rfecv_params)

            estimator = load_model(self, model, **rfecv_params)

            # Repeated kfold-CV type
            cv_model = RepeatedKFold(n_splits=self.args.kfold, n_repeats=self.args.repeat_kfolds, random_state=self.args.seed)

            # Select scoring function for RFECV analysis based on the error type
            scoring = get_scoring_key(self.args.type,self.args.error_type)

            # Use different strategies for models without feature_importances_
            if model.upper() in ['NN', 'GP']:
                # For NN and GP, use a simpler approach: select top features by correlation with y
                # after initial fit, then use permutation importance to rank them
                
                # Train the model once on all features
                estimator.fit(X_scaled_df, y_df)
                
                # Use permutation importance to rank features
                perm_result = permutation_importance(estimator, X_scaled_df, y_df, 
                                      n_repeats=self.args.pfi_epochs, 
                                      random_state=self.args.seed,
                                      scoring=scoring,
                                      n_jobs=1)  # Force single thread for reproducibility
                
                # Round to reduce floating point variance
                feature_importances = np.round(perm_result.importances_mean, decimals=10)
                
                # Create list of (importance, name) tuples for ALL features
                importance_with_names = [(feature_importances[i], X_scaled_df.columns[i]) 
                                        for i in range(len(feature_importances))]
                
                # Sort by importance (descending) and break ties alphabetically for determinism
                importance_with_names.sort(key=lambda x: (-x[0], x[1]))  # Sort by importance DESC, then name ASC
                
                # Select top num_descriptors features (or all with positive importance if fewer)
                positive_features = [(imp, name) for imp, name in importance_with_names if imp > 0]
                if len(positive_features) > num_descriptors:
                    descriptors_used[model] = [name for _, name in positive_features[:num_descriptors]]
                else:
                    descriptors_used[model] = [name for _, name in positive_features]
                
                # Sort final list alphabetically for consistent ordering in output
                descriptors_used[model] = sorted(descriptors_used[model])
                
                txt_corr += f'\n   - {model}: {len(descriptors_used[model])} descriptors selected (using PFI)'
            
            else:
                # MVL, RF, GB, ADAB: use RFECV with feature importances
                # Set step=1 for most stable/deterministic feature elimination
                selector = RFECV(estimator, scoring=scoring, min_features_to_select=2, cv=cv_model, step=1, n_jobs=1)
                selector.fit(X_scaled_df, y_df)
                
                # Get selected features
                selected_mask = selector.support_
                selected_features_list = list(X_scaled_df.columns[selected_mask])
                
                # Get feature importances for selected features only
                if model.upper() == 'MVL':
                    # For MVL, use absolute coefficients as importance
                    feature_importances = np.abs(selector.estimator_.coef_)
                else: 
                    # RF, GB, ADAB have feature_importances_
                    feature_importances = selector.estimator_.feature_importances_
                
                # Round importances to reduce floating point variance
                feature_importances = np.round(feature_importances, decimals=10)
                
                # Create (importance, name) pairs for selected features
                importance_with_names = list(zip(feature_importances, selected_features_list))
                
                # Sort by importance (descending) with alphabetical tie-breaking for determinism
                importance_with_names.sort(key=lambda x: (-x[0], x[1]))  # Sort by importance DESC, then name ASC
                
                # Select top num_descriptors (or all if fewer selected)
                n_to_select = min(num_descriptors, len(importance_with_names))
                descriptors_used[model] = [name for _, name in importance_with_names[:n_to_select]]
                
                # Sort final list alphabetically for consistent ordering in output
                descriptors_used[model] = sorted(descriptors_used[model])
                
                txt_corr += f'\n   - {model}: {len(descriptors_used[model])} descriptors selected (using RFECV)'
            
            # Create model-specific dataframe with sorted columns for reproducibility
            keep_cols = descriptors_used[model] + [self.args.y] + self.args.ignore
            keep_cols = list(dict.fromkeys(keep_cols))  # Remove duplicates preserving order
            keep_cols = [col for col in keep_cols if col in csv_df_filtered.columns]
            
            # Sort descriptor columns alphabetically, keep y and ignore at the end
            descriptor_cols = [col for col in keep_cols if col not in self.args.ignore and col != self.args.y]
            other_cols = [col for col in keep_cols if col in self.args.ignore or col == self.args.y]
            sorted_cols = sorted(descriptor_cols) + sorted([col for col in other_cols if col in self.args.ignore]) + [self.args.y]
            
            csv_df_per_model[model] = csv_df_filtered[sorted_cols].copy()

    else:
        if not self.args.rfecv_filter:
            txt_corr += f'\n   x The RFECV/PFI filter was disabled (--rfecv_filter False), descriptor selection was skipped'
        else:
            txt_corr += f'\n   x The RFECV filter was not applied, there are less descriptors than one-third of the data points ({n_descps} <= {num_descriptors})'
        # If RFECV is not applied, all models use the same filtered dataframe
        for model in self.args.model:
            csv_df_per_model[model] = csv_df_filtered

    self.args.log.write(txt_corr)

    # Return both the general filtered dataframe and model-specific dataframes
    return csv_df_filtered, csv_df_per_model


def load_minimal_model(model):
    '''
    Load the parameters of the minimalist models used for REFCV
    '''

    minimal_params = {
        'RF' : {
        'n_estimators': 30,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0,
        'max_features': 1,
        'ccp_alpha': 0.0,
        'max_samples': None
        },
        'GB': {
        'n_estimators': 30,
        'learning_rate': 0.1,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 1.0,
        'max_features': None,
        'validation_fraction': 0.2,
        'min_weight_fraction_leaf': 0.0,
        'ccp_alpha': 0.0
        },
        'NN': {
        'hidden_layer_1': 4,
        'hidden_layer_2': 4,
        'max_iter': 200,
        'alpha': 0.01,
        'tol': 0.0001
        },
        'ADAB': {
        'learning_rate': 1.0,
        'n_estimators': 30
        },
        'GP': {
        'n_restarts_optimizer': 30,
        },
        'MVL': {
        }
    }

    return minimal_params[model]

def mcc_scorer_clf(y_true,y_pred):
    """Forces classification predictions to integer for MCC."""
    # Even if .predict() returns floats, coerce them to integer:
    y_pred = np.round(y_pred).astype(int)
    
    return matthews_corrcoef(y_true, y_pred)

def f1_scorer_clf(y_true,y_pred):
    """Use binary F1 when possible and micro F1 for multiclass."""
    y_pred = np.round(y_pred).astype(int)
    try:
        return f1_score(y_true, y_pred)
    except ValueError:
        return f1_score(y_true, y_pred, average='micro')

def get_scoring_key(problem_type,error_type):
    '''
    Load scoring function for evaluating models
    '''

    if problem_type.lower() == 'reg':
        scoring = {
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_median_absolute_error',
            'r2': 'r2'
        }.get(error_type)
    else:
        # For classification
        if error_type == 'mcc':
            # Use the custom MCC scorer that ensures integer predictions
            scoring = make_scorer(mcc_scorer_clf)
        elif error_type == 'f1':
            scoring = make_scorer(f1_scorer_clf)
        else:
            scoring = {
                'acc': 'accuracy'
            }.get(error_type)
   
    return scoring


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
        curate_valid = locate_csv(self,self.csv_name,curate_valid)

        if self.eval_model.lower() not in ['mvl']:
            self.log.write(f"\nx  The eval_model option used is not valid! Options: 'MVL' (more options will be added soon)")
            curate_valid = False

        if self.type.lower() not in ['reg']:
            self.log.write(f"\nx  The type option used is not valid in EVALUATE! Options: 'reg' (the 'clas' option will be added soon)")
            curate_valid = False

    elif type_checks == 'initial' and module.lower() not in ['verify','predict']:

        curate_valid = locate_csv(self,self.csv_name,curate_valid)

        if module.lower() == 'curate':
            if self.categorical.lower() not in ['onehot','numbers']:
                self.log.write(f"\nx  The categorical option used is not valid! Options: 'onehot', 'numbers'")
                curate_valid = False

            for thres,thres_name in zip([self.thres_x,self.thres_y],['thres_x','thres_y']):
                if float(thres) > 1 or float(thres) < 0:
                    self.log.write(f"\nx  The {thres_name} option should be between 0 and 1!")
                    curate_valid = False
        
        elif module.lower() == 'generate':
            if self.split.lower() not in ['kn','rnd','stratified','even','extra_q1','extra_q5','auto']:
                self.log.write(f"\nx  The split option used is not valid! Options: 'KN', 'RND', 'stratified', 'even', 'extra_q1', 'extra_q5', 'auto'")
                curate_valid = False

            if self.split == 'auto':
                if self.type.lower() == 'reg':
                    self.split = 'even'
                elif self.type.lower() == 'clas':
                    # stratified guarantees both train and test keep the same class
                    # proportions as the full dataset - a plain random split can, by chance,
                    # leave a class under/over-represented, especially on small datasets
                    self.split = 'stratified'

            for model_type in self.model:
                if model_type.upper() not in ['RF','MVL','GB','GP','ADAB','NN'] or len(self.model) == 0:
                    self.log.write(f"\nx  The model option used is not valid! Options: 'RF', 'MVL', 'GB', 'ADAB', 'NN'")
                    curate_valid = False
                if model_type.upper() == 'MVL' and self.type.lower() == 'clas':
                    self.log.write(f"\nx  Multivariate linear models (MVL in the model_type option) are not compatible with classificaton!")                 
                    curate_valid = False

            if self.type.lower() not in ['reg','clas']:
                self.log.write(f"\nx  The type option used is not valid! Options: 'reg', 'clas'")
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


def locate_csv(self,csv_input,curate_valid):
    '''
    Assesses whether the input CSV databases can be located
    '''

    path_csv = ''
    if os.path.exists(f"{csv_input}"):
        path_csv = csv_input
    elif os.path.exists(f"{Path(os.getcwd()).joinpath(csv_input)}"):
        path_csv = f"{Path(os.getcwd()).joinpath(csv_input)}"
    if not os.path.exists(path_csv) or csv_input == '':
        self.log.write(f'\nx  The path of your CSV file doesn\'t exist! You specified: --csv_name {csv_input}')
        curate_valid = False
    
    return curate_valid


def check_clas_problem(self,csv_df):
    '''
    Changes type to classification if there are only two different y values.
    Automatically converts non-integer class labels to integer codes.
    Stores the original labels for later reconversion in outputs.
    '''

    # changes type to classification if there are only two different y values
    if self.args.type.lower() == 'reg' and self.args.auto_type:
        num_unique = len(set(csv_df[self.args.y]))
        if num_unique == 2:
            self.args.type = 'clas'
            if self.args.error_type not in ['acc', 'mcc', 'f1']:
                self.args.error_type = 'mcc'
            if any(x.upper() == 'MVL' for x in self.args.model):
                self.args.model = [x if x.upper() != 'MVL' else 'ADAB' for x in self.args.model]

            # sorted (not list(set(...))) for deterministic output regardless of string-hash
            # randomization, matching this file's reproducibility guarantees elsewhere
            unique_vals = _sorted_class_values(csv_df[self.args.y])
            y_val_detect = f'{unique_vals[0]} and {unique_vals[1]}'
            self.args.log.write(f'\no  Only two different y values were detected ({y_val_detect})! The program will consider classification models (same effect as using "--type clas"). This option can be disabled with "--auto_type False"')

    if self.args.type.lower() == 'clas':
        unique_values = _sorted_class_values(csv_df[self.args.y])
        if _integer_like_class_values(unique_values):
            csv_df[self.args.y] = pd.to_numeric(csv_df[self.args.y]).astype(int)
        else:
            self.args.class_mapping = {
                original_label: class_code
                for class_code, original_label in enumerate(unique_values)
            }
            self.args.class_mapping_reverse = {
                class_code: original_label
                for original_label, class_code in self.args.class_mapping.items()
            }
            if len(unique_values) >= 2:
                self.args.class_0_label = str(unique_values[0])
                self.args.class_1_label = str(unique_values[1])

            csv_df[self.args.y] = csv_df[self.args.y].map(self.args.class_mapping).astype(int)

            mapping_txt = ', '.join(
                f'{original_label} -> {class_code}'
                for original_label, class_code in self.args.class_mapping.items()
            )
            self.args.log.write(f'\no  Classification labels converted: {mapping_txt}')
            self.args.log.write(f'   Original labels will be restored in output files')

        # Check that each class has at least 5 points
        class_counts = csv_df[self.args.y].value_counts()
        min_class_count = class_counts.min()
        min_class_label = class_counts.idxmin()
        
        if min_class_count < 5:
            # Get original label if available
            if hasattr(self.args, 'class_mapping_reverse') and min_class_label in self.args.class_mapping_reverse:
                original_label = self.args.class_mapping_reverse[min_class_label]
            else:
                original_label = min_class_label

            class_dist = {}
            for class_label, count in class_counts.items():
                if hasattr(self.args, 'class_mapping_reverse') and class_label in self.args.class_mapping_reverse:
                    original_count_label = self.args.class_mapping_reverse[class_label]
                else:
                    original_count_label = class_label
                class_dist[str(original_count_label)] = int(count)
            
            self.args.log.write(f'\nx  Insufficient data for classification! One of the classes has only {min_class_count} datapoints (class "{original_label}")')
            self.args.log.write(f'   Each class must have at least 5 datapoints to ensure robust train/validation/test splits')
            self.args.log.write(f'   Current distribution: {class_dist}')
            self.args.log.write(f'   Please add more datapoints for the minority class or consider a different approach')
            self.args.log.finalize()
            sys.exit()

    return self
    

def load_database(self,csv_load,module,print_info=True,external_test=False):
    '''
    Loads either a Xy (params=False) or a parameter (params=True) database from a CSV file
    '''
    
    # adjust external set in AQME workflows
    if module.lower() == 'aqme_test':
        external_test = True

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

    # Missing data handling: robust strategy for columns and rows (optional KNN imputer)
    target_col = self.args.y

    # Missing target (y) values crash later steps (e.g. data splitting), so remove them upfront
    # and warn the user instead of letting the program fail. External test sets are skipped since
    # their target column is often unknown on purpose (that's what's being predicted)
    if not external_test and target_col in csv_df.columns:
        rows_missing_y = csv_df[target_col].isna()
        if rows_missing_y.any():
            n_missing_y = int(rows_missing_y.sum())
            csv_df = csv_df[~rows_missing_y].reset_index(drop=True)
            self.args.log.write(f'\nx  WARNING! {n_missing_y} row(s) had missing values in the target column ({target_col}) and were removed, since the target value cannot be predicted or imputed. Please fill in these values manually (or with a reasonable value, e.g. 0) if you want to keep them in the database.\n')

    descriptor_cols = [col for col in csv_df.columns if col not in self.args.ignore+self.args.discard and col != self.args.y]
    min_count = int(0.9 * len(csv_df))

    # Remove columns with <90% data
    cols_to_drop = [col for col in descriptor_cols if csv_df[col].notna().sum() < min_count]
    if cols_to_drop:
        csv_df = csv_df.drop(columns=cols_to_drop)
        if module.lower() == 'curate':
            txt_load += f"\n   - Removed {len(cols_to_drop)} column(s) with <90% data\n"
        descriptor_cols = [col for col in csv_df.columns if col not in self.args.ignore+self.args.discard and col != self.args.y]
    
    # Remove rows with <50% data
    rows_too_missing = csv_df[descriptor_cols].isna().sum(axis=1) > (0.5 * len(descriptor_cols))
    if rows_too_missing.any():
        n_removed_rows = rows_too_missing.sum()
        csv_df = csv_df[~rows_too_missing].reset_index(drop=True)
        descriptor_cols = [col for col in csv_df.columns if col not in self.args.ignore+self.args.discard and col != self.args.y]
        if module.lower() == 'curate':
            txt_load += f"\n   - Removed {n_removed_rows} row(s) with >50% missing descriptors\n"
    
    # Apply KNN imputer only to numeric columns with missing values (when auto_fill is activated)
    if self.args.auto_fill:
        numeric_columns = csv_df.select_dtypes(include=['float']).columns.drop(target_col, errors='ignore')
        if csv_df[numeric_columns].isna().any().any():
            imputer = KNNImputer(n_neighbors=5)
            csv_df[numeric_columns] = pd.DataFrame(imputer.fit_transform(csv_df[numeric_columns]), columns=numeric_columns, index=csv_df.index)
            if module.lower() == 'curate':
                txt_load += f"\n   - Applied KNN imputer to columns with missing values\n"
    else:
        # Remove columns with ANY missing value
        cols_with_missing = [col for col in descriptor_cols if csv_df[col].isna().any() and col not in self.args.ignore+self.args.discard]
        if cols_with_missing:
            csv_df = csv_df.drop(columns=cols_with_missing)
            if module.lower() == 'curate':
                txt_load += f"\n   - Removed {len(cols_with_missing)} column(s) with missing values\n"

    if print_info:
        sanity_checks(self.args,'csv_db',module,csv_df.columns)

    discard_cols = [col for col in self.args.discard if col in csv_df.columns]
    if discard_cols:
        csv_df = csv_df.drop(discard_cols, axis=1)

    if print_info:
        total_amount = len(csv_df.columns)
        ignored_descs = len(self.args.ignore)
        accepted_descs = total_amount - ignored_descs - 1 # the y column is substracted
        if 'Set' in csv_df.columns: # removes the column that tracks sets
            accepted_descs -= 1
            ignored_descs += 1
        if module.lower() not in ['aqme','aqme_test']:
            csv_name = os.path.basename(csv_load)
            if module.lower() not in ['predict']:
                txt_load += f'\no  Database {csv_name} loaded successfully, including:'
                txt_load += f'\n   - {len(csv_df[self.args.y])} datapoints'
                txt_load += f'\n   - {accepted_descs} accepted descriptors'
                txt_load += f'\n   - {ignored_descs} ignored descriptors'
                txt_load += (
                    f"\n   - {len(self.args.discard)} discarded descriptors"
                )
            else:
                txt_load += (
                    f"\no  External set {csv_name} loaded successfully, "
                    "including:"
                )
                txt_load += f"\n   - {len(csv_df)} datapoints (rows)"
            self.args.log.write(txt_load)
            if accepted_descs == 0:
                self.args.log.write(f"\nx  The aren't any valid descriptors! Check the messages above to see whether the filters have discarded descriptors")
                sys.exit()

    # Sort columns alphabetically for reproducibility across ALL modules
    if module.lower() not in ['aqme', 'aqme_test']:
        # Get descriptor columns (excluding y and ignore)
        descriptor_cols = [col for col in csv_df.columns if col not in self.args.ignore and col != self.args.y]
        descriptor_cols_sorted = sorted(descriptor_cols)
        # Get other columns (y and ignore)
        other_cols = [col for col in csv_df.columns if col in self.args.ignore or col == self.args.y]
        # Reorder dataframe with sorted descriptors + other columns
        sorted_all_cols = descriptor_cols_sorted + other_cols
        csv_df = csv_df[sorted_all_cols]
    
    # ignore user-defined descriptors and assign X and y values (but keeps the original database)
    if module.lower() == 'generate':
        # Only drop columns that actually exist in the dataframe (for model-specific CSVs from CURATE)
        cols_to_ignore = [col for col in self.args.ignore if col in csv_df.columns]
        # Also drop 'Set' column if it exists (for model-specific CSVs)
        if 'Set' in csv_df.columns and 'Set' not in cols_to_ignore:
            cols_to_ignore.append('Set')
        csv_df_ignore = csv_df.drop(cols_to_ignore, axis=1)
        csv_X = csv_df_ignore.drop([self.args.y], axis=1)
        csv_y = csv_df_ignore[self.args.y]
        
        # Columns are already sorted from above, just extract them
        csv_X = csv_X[sorted([col for col in csv_X.columns])]
        
    else:
        if external_test and self.args.y not in csv_df.columns:
            csv_X = csv_df
            csv_y = None
        else:
            csv_X = csv_df.drop([self.args.y], axis=1)
            csv_y = csv_df[self.args.y]

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


def scale_df(csv_X,csv_X_external):
    '''
    Scale the X matrix for the training set and the external test set (if any)
    '''
    
    scaler = StandardScaler()
    _ = scaler.fit(csv_X)
    X_scaled = scaler.transform(csv_X)
    X_scaled_df = pd.DataFrame(X_scaled, columns = csv_X.columns)

    X_scaled_external_df = None
    if csv_X_external is not None:
        X_scaled_external = scaler.transform(csv_X_external)
        X_scaled_external_df = pd.DataFrame(X_scaled_external, columns = csv_X_external.columns)

    return X_scaled_df,X_scaled_external_df


def Xy_split(csv_df,csv_X,X_scaled_df,csv_y,csv_external_df,csv_X_external,X_scaled_external_df,csv_y_external,test_points,column_names):
    '''
    Returns a dictionary with the database divided into train and validation
    '''

    Xy_data =  {}

    if len(test_points) == 0:
        Xy_data['X_train'] = csv_X
        Xy_data['X_train_scaled'] = X_scaled_df
        Xy_data['y_train'] = csv_y
        Xy_data['names_train'] = csv_df[column_names]
        # keep the X_test/y_test keys present (as empty) even without an internal test split -
        # this is a real, supported configuration (--csv_test with --auto_test False keeps the
        # whole database as training data), but most of this file reads Xy_data['X_test_scaled']
        # etc. unconditionally, so omitting these keys turns a supported option combo into a
        # KeyError deep inside VERIFY/PREDICT instead of a graceful "no internal test set"
        Xy_data['X_test'] = csv_X.iloc[0:0]
        Xy_data['X_test_scaled'] = X_scaled_df.iloc[0:0]
        Xy_data['y_test'] = csv_y.iloc[0:0]
        Xy_data['names_test'] = csv_df[column_names].iloc[0:0]

    else:
        Xy_data['X_train'] = csv_X.drop(test_points)
        Xy_data['X_train_scaled'] = X_scaled_df.drop(test_points)
        Xy_data['y_train'] = csv_y.drop(test_points)
        Xy_data['X_test'] = csv_X.iloc[test_points]
        Xy_data['X_test_scaled'] = X_scaled_df.iloc[test_points]
        Xy_data['y_test'] = csv_y.iloc[test_points]
        Xy_data['names_train'] = csv_df.drop(test_points)[column_names]
        Xy_data['names_test'] = csv_df.iloc[test_points][column_names]

    Xy_data['test_points'] = test_points

    if X_scaled_external_df is not None:
        Xy_data['X_external'] = csv_X_external
        Xy_data['X_external_scaled'] = X_scaled_external_df
        if csv_y_external is not None:
            Xy_data['y_external'] = csv_y_external 
        Xy_data['names_external'] = csv_external_df[column_names]

    return Xy_data


def test_select(self,X_scaled,csv_y):
    '''
    Selection of test set (if any)
    '''

    # adjusts size of the test_set to include at least 4 points regardless of the number of datapoints
    test_input_size = round(self.args.test_set * len(csv_y))
    min_test_size = 4
    selected_size = max(test_input_size,min_test_size)

    # KN's classification branch only handles exactly 2 classes (hardcoded class_0/class_1
    # groups below); EVEN/EXTRA_Q1/EXTRA_Q5 rely on a continuous, ordered target value (the
    # "lowest/highest 20% of y", quantile bins, etc.) which has no meaningful equivalent for a
    # discrete class label, regardless of how many classes there are. RND is always safe.
    # STRATIFIED (below) stratifies directly by the class labels via StratifiedShuffleSplit, so
    # it doesn't need a fallback here - it works for any number of classes
    if self.args.type.lower() == 'clas':
        split_upper = self.args.split.upper()
        if split_upper in ('EVEN', 'EXTRA_Q1', 'EXTRA_Q5'):
            self.args.log.write(f'\nx  WARNING! The {self.args.split} split relies on a continuous, ordered target value and isn\'t defined for classification, the STRATIFIED split was used instead.')
            self.args.split = 'STRATIFIED'
        elif split_upper == 'KN' and len(set(csv_y)) != 2:
            self.args.log.write(f'\nx  WARNING! The KN split is only available for binary classification, the STRATIFIED split was used instead.')
            self.args.split = 'STRATIFIED'

    if self.args.split.upper() == 'KN':
        # k-neighbours data split

        # selects representative training points for each target value in classification problems
        if self.args.type.lower() == 'clas':
            class_0_idx = list(csv_y[csv_y == 0].index)
            class_1_idx = list(csv_y[csv_y == 1].index)
            class_0_test_size = round((len(class_0_idx)/len(csv_y))*selected_size)
            class_1_test_size = selected_size-class_0_test_size
            class_0_train_size = len(class_0_idx) - class_0_test_size
            class_1_train_size = len(class_1_idx) - class_1_test_size

            # the k-means function internally selects the training points to be as diverse as possible, 
            # but it returns the test points
            test_class_0 = k_means(self,X_scaled.iloc[class_0_idx],csv_y,class_0_train_size,self.args.seed,class_0_idx)
            test_class_1 = k_means(self,X_scaled.iloc[class_1_idx],csv_y,class_1_train_size,self.args.seed,class_1_idx)
            test_points = test_class_0+test_class_1

        else:
            idx_list = csv_y.index
            training_size = len(csv_y)-selected_size
            test_points = k_means(self,X_scaled,csv_y,training_size,self.args.seed,idx_list)

    elif self.args.split.upper() == 'RND':
        size = round(selected_size * 100 / (len(csv_y)))
        _, X_test, _, _ = train_test_split(X_scaled, csv_y, test_size=size/100, random_state=self.args.seed)
        test_points = X_test.index.tolist()

    elif self.args.split.upper() == 'STRATIFIED':

        if self.args.type.lower() == 'clas':
            # classification: the class labels themselves are already the natural strata, so
            # stratify directly on them - no capping extremes or quantile-binning a continuous
            # value (that's the regression path below). Works for any number of classes, not
            # just binary, since StratifiedShuffleSplit itself is class-count-agnostic
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=selected_size, random_state=self.args.seed)
            # splitter.split() yields (train_idx, test_idx), in that order
            for _, test_idx in splitter.split(X_scaled, csv_y):
                test_points = csv_y.index[test_idx].tolist()

        else:
            size = np.ceil(selected_size * 100 / (len(csv_y)))
            # Remove the max and min values so they don't end up in the training set
            # Calculate the number of bins based on the number of points
            csv_y_capped = csv_y.drop([csv_y.idxmin(), csv_y.idxmax()])
            X_scaled_capped = X_scaled.loc[csv_y_capped.index]

            # StratifiedShuffleSplit needs at least 1 sample per class/bin in the smaller partition
            # (which has ~size% of the points), so the bin count can never exceed that partition's
            # size - otherwise it raises "train_size should be greater or equal to the number of classes"
            max_bins = max(2, int(size / 100 * len(csv_y_capped)))
            selected_size = min(selected_size, max_bins)

            y_binned = pd.qcut(csv_y_capped, q=selected_size, labels=False, duplicates='drop')

            # Adjust the number of bins until each class has at least 2 members
            while y_binned.value_counts().min() < 2 and selected_size > 2:
                selected_size -= 1
                y_binned = pd.qcut(csv_y_capped, q=selected_size, labels=False, duplicates='drop')
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=size / 100, random_state=self.args.seed)
            # splitter.split() yields (train_idx, test_idx), in that order. test_idx are
            # positions within the capped (min/max removed) subset, so map them back to the
            # original row labels via csv_y_capped's index instead of using them directly
            for _, test_idx in splitter.split(X_scaled_capped, y_binned):
                test_points = csv_y_capped.index[test_idx].tolist()

    elif self.args.split.upper() == 'EVEN':
        # Remove the max and min values so they don't end up in the training set
        csv_y_capped = csv_y.drop([csv_y.idxmin(), csv_y.idxmax()])
        # Calculate the number of bins based on the number of points
        y_binned = pd.qcut(csv_y_capped, q=selected_size, labels=False, duplicates='drop')

        # Adjust bin count if any bin has fewer than two elements (happens in imbalanced data, see comment below)
        temp_size = selected_size
        while y_binned.value_counts().min() < 2 and temp_size > 2:
            temp_size -= 1
            y_binned = pd.qcut(csv_y_capped, q=temp_size, labels=False, duplicates='drop')

        # Determine central validation points for each bin
        test_points = []
        for bin_label in y_binned.unique():
            bin_indices = y_binned[y_binned == bin_label].index
            sorted_indices = sorted(bin_indices, key=lambda idx: csv_y[idx])
            test_points.append(sorted_indices[round(len(sorted_indices)/2)])

        # in umbalanced databases, the points cannot be selected entirely even (i.e., if a database
        # contains 10 points in th 0-10 range, and 1000 points in the 10-90 range, choosing 100
        # points might cause that all the 10 points in the 0-10 range are selected as test)
        # For this issue, the points discarded in pd.qcut() are selected randomly to complete the target
        # amount of points in the test set
        random_seed = self.args.seed
        while len(test_points) < selected_size:
            new_test_point = int(csv_y.sample(n=1, random_state=random_seed).index[0])
            if new_test_point not in [csv_y.idxmin(), csv_y.idxmax()] + test_points:
                test_points.append(new_test_point)
            random_seed += 1

    elif self.args.split.upper() == 'EXTRA_Q1':
        # 20% lowest points
        portion = max(1, round(0.2 * len(csv_y)))
        test_points = csv_y.nsmallest(portion).index.tolist()
    
    elif self.args.split.upper() == 'EXTRA_Q5':
        # 20%% highest points
        portion = max(1, round(0.2 * len(csv_y)))
        test_points = csv_y.nlargest(portion).index.tolist()

    test_points.sort()

    return test_points


def round_int_param(value, bounds):
    '''
    Rounds a continuous BO suggestion to a valid integer value for an integer-type
    hyperparameter, snapping to the nearest allowed step (4th element of bounds, i.e.
    (lower, upper, int, step) - defaults to step=1, i.e. every integer, if not given) instead
    of every single integer. A large range like n_estimators (10-200) rarely needs single-unit
    resolution, so a coarser step lets each BO iteration cover more meaningfully different
    configurations instead of two iterations landing on e.g. 71 and 72 trees.
    '''
    lower, upper = bounds[0], bounds[1]
    step = bounds[3] if len(bounds) == 4 else 1
    n_steps = round((value - lower) / step)
    rounded = lower + n_steps * step
    return int(max(lower, min(upper, rounded)))


def generate_lhs_points(pbounds, n_points, random_state=None):
    """
    Generate initial points using Latin Hypercube Sampling for better space coverage.
    LHS ensures uniform distribution across all dimensions of the hyperparameter space.
    
    Args:
        pbounds: Dictionary with parameter bounds from BO_hyperparams
        n_points: Number of initial points to generate
        random_state: Random seed for reproducibility
    
    Returns:
        List of dictionaries with parameter values
    """
    # use a local RNG instead of np.random.seed(), which would reset the process-global NumPy
    # random state and could silently perturb any other code in the process that relies on it
    rng = np.random.RandomState(random_state)

    param_names = list(pbounds.keys())
    n_params = len(param_names)

    # Generate LHS samples in [0, 1]^n_params
    # Each dimension is divided into n_points intervals, and one point is sampled from each interval
    samples = np.zeros((n_points, n_params))
    for i in range(n_params):
        # Create intervals and sample within each
        intervals = np.linspace(0, 1, n_points + 1)
        samples[:, i] = rng.uniform(intervals[:-1], intervals[1:])
        # Shuffle to break correlation between dimensions
        rng.shuffle(samples[:, i])
    
    # Scale samples to actual parameter bounds
    initial_points = []
    for sample in samples:
        point = {}
        for i, param_name in enumerate(param_names):
            bounds = pbounds[param_name]
            lower, upper = bounds[:2]
            # Scale from [0, 1] to [lower, upper]
            value = lower + sample[i] * (upper - lower)
            if len(bounds) >= 3 and bounds[2] is int:
                value = round_int_param(value, bounds)
            point[param_name] = value
        initial_points.append(point)
    
    return initial_points


def BO_optimizer(self,bo_data,Xy_data):
    # bayes_opt expects plain (lower, upper) bounds, without the int type marker
    # used by generate_lhs_points() to round integer hyperparameters
    pbounds = {name: bounds[:2] for name, bounds in BO_hyperparams(bo_data['model']).items()}

    # Initialize Bayesian optimization. No acquisition_function is passed on purpose: bayes_opt
    # then defaults to UpperConfidenceBound(kappa=2.576), which scales its exploration with the
    # model's own uncertainty (kappa*std) rather than a fixed absolute margin like Expected
    # Improvement's xi - the latter needs to be hand-tuned to each dataset's error scale to behave
    # consistently, which isn't practical across the arbitrary datasets ROBERT is run on
    optimizer = BayesianOptimization(
        f=lambda **p: BO_iteration(self, bo_data, Xy_data, **p),
        pbounds=pbounds,
        verbose=2,
        random_state=self.args.seed
    )

    # Generate initial points using Latin Hypercube Sampling for better space coverage
    if self.args.init_points > 0:
        initial_points = generate_lhs_points(
            pbounds=BO_hyperparams(bo_data['model']),
            n_points=self.args.init_points,
            random_state=self.args.seed
        )
        # Probe the initial points (evaluated immediately, not lazily queued, since the
        # manual suggest/probe loop below replaces optimizer.maximize() and needs these
        # already registered to base its first suggestion on)
        for params in initial_points:
            optimizer.probe(params=params, lazy=False)

    # int-type hyperparameters, so they can be rounded below. bayes_opt's own suggest() has no
    # notion of integer dimensions (pbounds only carries (lower, upper)) - only the LHS points
    # above get rounded, by generate_lhs_points(). Using optimizer.maximize() directly would
    # let every iteration from here on suggest and register raw continuous values for these,
    # even though model_adjust_params() rounds them right before actually building the model -
    # so the model itself was always fine, but the registered/reported point (and the "best
    # params" ultimately saved) would misleadingly show decimals for e.g. n_estimators.
    int_params = {name: bounds for name, bounds in BO_hyperparams(bo_data['model']).items()
                  if len(bounds) >= 3 and bounds[2] is int}

    # Run the optimization (with warnings suppressed for Convergence issues)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for _ in range(self.args.n_iter):
            next_point = optimizer.suggest()
            for name, bounds in int_params.items():
                next_point[name] = round_int_param(next_point[name], bounds)
            optimizer.probe(params=next_point, lazy=False)

    if bo_data['error_type'].upper() in ['RMSE','MAE']:
        BO_target = -optimizer.max['target']
    else:
        BO_target = optimizer.max['target']
    self.args.log.write(f"   o Best combined {bo_data['error_type'].upper()} (target) found in BO for {bo_data['model']} (no PFI filter): {BO_target:.2}")

    # Retrieve best parameters and best result
    return optimizer.max['params'], BO_target


def BO_iteration(self, bo_data, Xy_data, **params):
    '''
    Evaluate a model with given parameters using cross-validation.
    Returns the mean negative root mean squared error (higher is better).
    '''

    bo_data['params'] = model_adjust_params(self, bo_data['model'], params)
    BO_iter_score = load_n_predict(self, bo_data, Xy_data, BO_opt=True)

    return BO_iter_score


def BO_hyperparams(model_name):
    '''
    Hyperparameter search space per model, as (lower, upper) for continuous parameters or
    (lower, upper, int) / (lower, upper, int, step) for integer ones (step defaults to 1,
    i.e. every integer, if omitted - see round_int_param()).

    Bounds are tuned for ROBERT's typical dataset sizes (mostly 20-100 datapoints, up to
    ~10,000 at most), which is far smaller than the sample sizes scikit-learn's own defaults
    assume - so several ranges deliberately sit in a more heavily-regularized/lower-capacity
    region than scikit-learn's generic defaults (e.g. NN's hidden_layer_1/2, RF/GB's shallower
    max_depth). A few parameters are intentionally left out of the search entirely:
    - GB's validation_fraction: only has any effect if n_iter_no_change is set, which ROBERT
      never does, so scikit-learn silently ignores it - searching it wastes BO budget on a
      parameter with zero effect on the fitted model.
    - RF/GB's min_weight_fraction_leaf: constrains leaf size by weighted sample fraction, but
      ROBERT never passes sample_weight to fit(), so with implicit uniform weights this is a
      weaker, redundant echo of min_samples_leaf/min_samples_split (already searched).
    - NN's tol (lbfgs convergence tolerance): affects when the solver stops, not the quality of
      the fit for the max_iter budget already given, so it's left at scikit-learn's default.
    '''

    model_BO_params = {
        'RF' : {
        'n_estimators': (10, 200, int, 10),
        'max_depth': (5, 20, int),
        'min_samples_split': (2, 10, int),
        'min_samples_leaf': (2, 10, int),
        'max_features': (0.1, 1.0),
        'ccp_alpha': (0, 0.05),
        'max_samples': (0.5, 1.0)
        },
        'GB': {
        'n_estimators': (10, 200, int, 10),
        'learning_rate': (0.01, 0.3),
        'max_depth': (2, 10, int), # boosting needs shallower trees than bagging (sklearn's own GB default is 3, vs RF's unlimited)
        'min_samples_split': (2, 10, int),
        'min_samples_leaf': (2, 10, int),
        'subsample': (0.7, 1.0),
        'max_features': (0.1, 1.0),
        'ccp_alpha': (0, 0.05)
        },
        'NN': {
        'hidden_layer_1': (1, 10, int),
        'hidden_layer_2': (0, 10, int),
        'max_iter': (200, 500, int, 25),
        'alpha': (0.0001, 0.1) # widened down to sklearn's own default (0.0001) - the previous
                                # floor of 0.01 excluded that entire region from the search
        },
        'ADAB': {
        'learning_rate': (0.1, 5),
        'n_estimators': (10, 100, int, 5)
        },
        'GP': {
        'n_restarts_optimizer': (0, 20, int), # scikit-learn's own guidance: 5-10 restarts is
                                               # typical in practice, the old upper bound of 100
                                               # went far past where more restarts help
        }
    }

    return model_BO_params[model_name]


def BO_metrics(self, bo_data, Xy_data):
    '''
    Get combined score for repeated k-fold and top-bottom sorted CVs (used in BO)
    '''

    metric_combined = load_n_predict(self, bo_data, Xy_data, BO_opt=True)
    if bo_data['error_type'].upper() in ['RMSE','MAE']:
        metric_combined  = -metric_combined
    bo_data[f"combined_{bo_data['error_type']}"] = metric_combined

    return bo_data


def model_adjust_params(self,model_name,params):
    '''
    Add seed and convert parameters to integers, since they come as floats with decimals in the iterations

    '''

    if model_name != 'MVL':
        params['random_state'] = self.args.seed

        if model_name in ['RF','GB']:
            params['n_estimators'] = round(params['n_estimators'])
            params['max_depth'] = round(params['max_depth'])
            params['min_samples_split'] = round(params['min_samples_split'])
            params['min_samples_leaf'] = round(params['min_samples_leaf'])

        elif model_name == 'NN':
            # add solver first
            params['solver'] = 'lbfgs'
            params['max_iter'] = round(params['max_iter'])
            params['hidden_layer_1'] = round(params['hidden_layer_1'])
            params['hidden_layer_2'] = round(params['hidden_layer_2'])

        elif model_name == 'ADAB':
            params['n_estimators'] = round(params['n_estimators'])

        elif model_name == 'GP':
            params['n_restarts_optimizer'] = round(params['n_restarts_optimizer'])

    return params


def load_model(self, model_name, **params):
    """
    Load models with their corresponding parameters.
    """

    if model_name == 'RF':
        # Ensure n_jobs=1 for reproducibility if not already in params
        if 'n_jobs' not in params:
            params['n_jobs'] = 1
        if self.args.type.lower() == 'reg':
            loaded_model = RandomForestRegressor(**params)
        else:
            loaded_model = RandomForestClassifier(**params)

    elif model_name == 'GB':
        # GradientBoosting doesn't have n_jobs parameter, it's already deterministic
        if self.args.type.lower() == 'reg':
            loaded_model = GradientBoostingRegressor(**params)
        else:
            loaded_model = GradientBoostingClassifier(**params)

    elif model_name == 'NN':
        # create the hidden layers architecture first
        params = setup_hidden_layers(params)

        if self.args.type.lower() == 'reg':
            loaded_model = MLPRegressor(**params)
        else:
            loaded_model = MLPClassifier(**params)

    elif model_name == 'ADAB':
        if self.args.type.lower() == 'reg':
            loaded_model = AdaBoostRegressor(**params)
        else:
            loaded_model = AdaBoostClassifier(**params)

    elif model_name == 'GP':        
        if self.args.type.lower() == 'reg':
            loaded_model = GaussianProcessRegressor(**params)
        else:
            loaded_model = GaussianProcessClassifier(**params)

    elif model_name == 'MVL':
        loaded_model = LinearRegression(**params)
    
    return loaded_model


def setup_hidden_layers(params):
    '''
    Build hidden layer structure from provided parameters
    '''

    hidden_layer_sizes = []
    hidden_layer_1 = params.pop('hidden_layer_1')
    hidden_layer_2 = params.pop('hidden_layer_2')
    if hidden_layer_1 > 0:
        hidden_layer_sizes.append(hidden_layer_1)
    if hidden_layer_2 > 0:
        hidden_layer_sizes.append(hidden_layer_2)
    hidden_layer_sizes = tuple(hidden_layer_sizes) if hidden_layer_sizes else (1,)

    params['hidden_layer_sizes'] = hidden_layer_sizes

    return params


def _conformal_abs_residual_quantile(abs_residuals, coverage):
    """
    Finite-sample quantile for split-style conformal symmetric intervals
    (nonconformity score = absolute residual).
    """
    abs_r = np.asarray(abs_residuals, dtype=float)
    n = int(abs_r.size)
    if n < 1:
        return float("nan")
    alpha = 1.0 - float(coverage)
    if not (0.0 < alpha < 1.0):
        return float("nan")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(abs_r, level, method="higher"))


def _apply_full_refit_split_conformal(
        self,
        model_data,
        Xy_data,
        loaded_model,
        y_cv_mean_train=None,
        overwrite_predictions=False,
        output_prefix="full_refit"):
    """
    Optional full-refit uncertainty path for API/BO experiments.

    The normal ROBERT workflow uses repeated-CV predictions from
    ``repeated_kfold_cv``. This helper only overwrites ``y_pred_*`` when an
    explicit caller requests it; otherwise results are stored under
    ``{output_prefix}_*`` keys so the standard outputs remain untouched.
    """
    reg = model_data["type"].lower() == "reg"
    m_full = clone(loaded_model)
    X_tr = Xy_data["X_train_scaled"]
    y_tr = Xy_data["y_train"]

    def select_rows(data, idx):
        if hasattr(data, "iloc"):
            return data.iloc[idx]
        return np.asarray(data)[idx]

    def format_predictions(pred):
        pred_arr = np.asarray(pred).ravel()
        if reg:
            return np.asarray(pred_arr, dtype=float).tolist()
        return [int(round(float(v))) for v in pred_arr]

    def pred_key(set_name):
        if overwrite_predictions:
            return f"y_pred_{set_name}"
        return f"{output_prefix}_y_pred_{set_name}"

    m_full.fit(X_tr, y_tr)
    Xy_data[pred_key("train")] = format_predictions(m_full.predict(X_tr))

    if "X_test_scaled" in Xy_data:
        Xy_data[pred_key("test")] = format_predictions(
            m_full.predict(Xy_data["X_test_scaled"])
        )

    if "X_external_scaled" in Xy_data:
        Xy_data[pred_key("external")] = format_predictions(
            m_full.predict(Xy_data["X_external_scaled"])
        )

    hw = float("nan")
    if reg and bool(getattr(self.args, "conformal_enable", True)):
        cov = float(getattr(self.args, "conformal_coverage", 0.9))
        cal_frac = float(getattr(self.args, "conformal_calib_frac", 0.15))
        y_flat = np.asarray(y_tr, dtype=float).ravel()
        n = len(y_flat)
        min_fit = 5
        min_cal = 3
        n_cal_target = max(min_cal, int(round(n * cal_frac)))
        n_fit = n - n_cal_target
        if n_cal_target >= min_cal and n_fit >= min_fit:
            idx = np.arange(n)
            fit_ix, cal_ix = train_test_split(
                idx,
                test_size=n_cal_target / n,
                random_state=int(self.args.seed),
                shuffle=True,
            )
            est_cal = clone(loaded_model)
            X_fit = select_rows(X_tr, fit_ix)
            X_cal = select_rows(X_tr, cal_ix)
            y_fit = select_rows(y_tr, fit_ix)
            y_cal_arr = np.asarray(select_rows(y_tr, cal_ix), dtype=float).ravel()
            est_cal.fit(X_fit, y_fit)
            pred_cal = np.asarray(est_cal.predict(X_cal), dtype=float).ravel()
            abs_res = np.abs(y_cal_arr - pred_cal)
            hw = _conformal_abs_residual_quantile(abs_res, cov)
        elif y_cv_mean_train is not None:
            y_cv = np.asarray(y_cv_mean_train, dtype=float).ravel()
            if y_cv.size == y_flat.size:
                finite_mask = np.isfinite(y_flat) & np.isfinite(y_cv)
                abs_res = np.abs(y_flat[finite_mask] - y_cv[finite_mask])
                hw = _conformal_abs_residual_quantile(abs_res, cov)

    hw_key = "conformal_half_width" if overwrite_predictions else f"{output_prefix}_conformal_half_width"
    Xy_data[hw_key] = hw
    return Xy_data


def load_n_predict(self, model_data, Xy_data, BO_opt=False, verify_job=False):
    '''
    Load model and calculate errors/precision and predicted values of the ML models
    '''

    # set the parameters for the ML model and load it
    loaded_model = load_model(self, model_data['model'], **model_data['params'])

    # calculate predicted y values using repeated k-fold CV
    Xy_data = repeated_kfold_cv(model_data,loaded_model,Xy_data,BO_opt)

    # combine all the predictions from the repeated CV (metrics of the train set)
    y_all_list,y_pred_all_list = [],[]
    for y_val,y_pred_vals in zip(Xy_data['y_train'],Xy_data['y_pred_train_all']):
        for y_pred_val in y_pred_vals:
            y_all_list.append(y_val)
            y_pred_all_list.append(y_pred_val)
        
    # get metrics for the different sets
    error_labels = {'reg': ['r2','mae','rmse'],
                    'clas': ['acc','f1','mcc']
                    }

    error1 = error_labels[model_data['type']][0]
    error2 = error_labels[model_data['type']][1]
    error3 = error_labels[model_data['type']][2]
    Xy_data[f'{error1}_train'], Xy_data[f'{error2}_train'], Xy_data[f'{error3}_train'] = get_prediction_results(model_data,y_all_list,y_pred_all_list)
    if not BO_opt:
        Xy_data[f'{error1}_test'], Xy_data[f'{error2}_test'], Xy_data[f'{error3}_test'] = get_prediction_results(model_data,Xy_data['y_test'],Xy_data['y_pred_test'])
        if 'y_external' in Xy_data and len(Xy_data['y_external']) > 0 and Xy_data['y_external'].notna().any():
            # external targets can be partially known (e.g. predicting on some new, unlabeled
            # compounds) - score only the rows with a known target instead of skipping every
            # external metric just because a few rows are NaN
            y_ext_mask = Xy_data['y_external'].notna().to_numpy()
            y_ext_known = Xy_data['y_external'][y_ext_mask]
            y_pred_ext_known = np.asarray(Xy_data['y_pred_external'])[y_ext_mask]
            Xy_data[f'{error1}_external'], Xy_data[f'{error2}_external'], Xy_data[f'{error3}_external'] = get_prediction_results(model_data,y_ext_known,y_pred_ext_known)
        if 'y_pred_train_infold' in Xy_data:
            # in-fold training fit (used for the train-vs-validation gap score)
            Xy_data[f'{error1}_train_infold'], Xy_data[f'{error2}_train_infold'], Xy_data[f'{error3}_train_infold'] = get_prediction_results(model_data,Xy_data['y_train'],Xy_data['y_pred_train_infold'])
    if BO_opt:
        # calculate sorted CV and its metrics
        # print the target that is above the BO
        # print the final result of the BO just after finishing all the iterations
        # include_test=verify_job: the Low/High report metrics (VERIFY's call, verify_job=True)
        # need the sorted CV over the whole dataset, but GENERATE's BO/PFI hyperparameter
        # search (verify_job=False here) must never let the test set leak into tuning
        Xy_data = sorted_kfold_cv(loaded_model, model_data, Xy_data, error_labels, include_test=verify_job)
        combined_score = (Xy_data[f'{model_data["error_type"]}_train'] + Xy_data[f'{model_data["error_type"]}_up_bottom']) / 2

        # Return if this is part of a verify job
        if verify_job:
            return Xy_data

        # Return negative score for MAE/RMSE in BO
        if model_data["error_type"].lower() in ['mae', 'rmse']:
            return -combined_score
        else:
            return combined_score
    else:
        return Xy_data


def repeated_kfold_cv(model_data,loaded_model,Xy_data,BO_opt):
    '''
    Performs a repeated k-fold cross-validation on the Xy dataset
    '''

    # create a list of lists with the same number of entries as y
    y_global,y_pred_global = [],[]
    y_pred_global_train_infold = []
    for _ in range(len(Xy_data['y_train'])):
        y_pred_global.append([])
        y_global.append([])
        y_pred_global_train_infold.append([])

    y_pred_global_test = []
    for _ in range(len(Xy_data['y_test'])):
        y_pred_global_test.append([])

    y_global_external,y_pred_global_external = [],[]
    if 'X_external' in Xy_data: # if there is an external test set
        for _ in range(len(Xy_data['X_external'])):
            y_pred_global_external.append([])
            y_global_external.append([])

    # start the repeated CV
    for CV_repeat in range(int(model_data['repeat_kfolds'])):
        _,y_pred_global,y_pred_global_test,y_pred_global_external,y_pred_global_train_infold = kfold_cv(y_global,y_pred_global,
                                        y_pred_global_test,
                                        y_pred_global_external,
                                        model_data,loaded_model,
                                        Xy_data,CV_repeat,BO_opt=BO_opt,
                                        y_pred_global_train_infold=y_pred_global_train_infold)

    y_train_pred, y_train_std = [],[]
    for y_val in y_pred_global:
        if model_data['type'].lower() == 'reg':
            y_train_pred.append(np.mean(y_val))
        elif model_data['type'].lower() == 'clas':
            y_train_pred.append(_classification_vote(y_val))
        y_train_std.append(round(np.std(y_val),2))

    Xy_data['y_pred_train_all'] = y_pred_global
    Xy_data['y_pred_train'] = y_train_pred
    Xy_data['y_pred_train_sd'] = y_train_std

    if not BO_opt:
        y_test_pred, y_test_std = [],[]
        for y_val_test in y_pred_global_test:
            if model_data['type'].lower() == 'reg':
                y_test_pred.append(np.mean(y_val_test))
            elif model_data['type'].lower() == 'clas':
                y_test_pred.append(_classification_vote(y_val_test))
            y_test_std.append(round(np.std(y_val_test),2))

        Xy_data['y_pred_test_all'] = y_pred_global_test
        Xy_data['y_pred_test'] = y_test_pred
        Xy_data['y_pred_test_sd'] = y_test_std

        if 'X_external' in Xy_data: # if there is an external test set
            y_external_pred, y_external_std = [],[]
            for y_val_external in y_pred_global_external:
                if model_data['type'].lower() == 'reg':
                    y_external_pred.append(np.mean(y_val_external))
                elif model_data['type'].lower() == 'clas':
                    y_external_pred.append(_classification_vote(y_val_external))
                y_external_std.append(round(np.std(y_val_external),2))

            Xy_data['y_pred_external_all'] = y_pred_global_external
            Xy_data['y_pred_external'] = y_external_pred
            Xy_data['y_pred_external_sd'] = y_external_std

        # in-fold training fit (each point is predicted by the fold(s) where it was part of
        # the training portion, i.e. NOT held out) - used for the train-vs-validation gap score
        y_train_infold_pred = []
        for y_val_trin in y_pred_global_train_infold:
            if model_data['type'].lower() == 'reg':
                y_train_infold_pred.append(np.mean(y_val_trin))
            elif model_data['type'].lower() == 'clas':
                y_train_infold_pred.append(_classification_vote(y_val_trin))
        Xy_data['y_pred_train_infold'] = y_train_infold_pred

    return Xy_data


def kfold_cv(y_global,y_pred_global,
             y_pred_global_test,
             y_pred_global_external,
             model_data,loaded_model,Xy_data,random_state,
             BO_opt=False,shuffle=True,kfold_cv_type='repeated',
             y_pred_global_train_infold=None,include_test=False):
    '''
    Perform a k-fold CV
    Uses StratifiedKFold for classification problems to maintain class distribution
    '''

    # load CV scheme
    if model_data['type'].lower() == 'clas':
        # Use StratifiedKFold for classification to maintain class distribution
        cv = StratifiedKFold(n_splits=int(model_data['kfold']), shuffle=shuffle, random_state=random_state)
    else:
        cv = KFold(n_splits=int(model_data['kfold']), shuffle=shuffle, random_state=random_state)

    # # load Xy values and sort using y_train as the sorting reference
    if kfold_cv_type == 'sorted':
        X_init,y_init = sort_n_load(Xy_data,include_test=include_test) # do not use, currently it doesn't sort indices for X_train as well

    else:
        # convert Xy values of training and validation for CV
        X_init = np.array(Xy_data['X_train_scaled'])
        y_init = np.array(Xy_data['y_train'])

        # convert Xy values for the test set and external test set (if any) - only needed
        # outside BO, where these conversions would otherwise run (and be discarded unused)
        # on every fold of every repeat of every BO iteration
        if not BO_opt:
            X_test = np.array(Xy_data['X_test_scaled'])
            # sklearn estimators reject predict() on a 0-row array (ensure_min_samples=1),
            # so an empty internal test set (e.g. --csv_test with --auto_test False) needs an
            # explicit skip here rather than just being an empty-but-present array
            has_test = len(X_test) > 0
            if 'X_external_scaled' in Xy_data:
                X_external = np.array(Xy_data['X_external_scaled'])

    ix_training, ix_valid = [], []
    # Loop through each fold and append the training & test indices to the empty lists above
    if model_data['type'].lower() == 'clas':
        # For classification, we need to pass y values to ensure stratification
        for fold in cv.split(X_init, y_init):
            ix_training.append(fold[0]), ix_valid.append(fold[1])
    else:
        for fold in cv.split(X_init):
            ix_training.append(fold[0]), ix_valid.append(fold[1])

    # Loop through each outer fold, and extract predicted vs actual values and SHAP feature analysis
    for (train_outer_ix, test_outer_ix) in zip(ix_training, ix_valid):
        X_train, X_valid = X_init[train_outer_ix, :], X_init[test_outer_ix, :]
        y_train, y_valid = y_init[train_outer_ix], y_init[test_outer_ix]

        fit = loaded_model.fit(X_train, y_train)
        y_pred_valid = fit.predict(X_valid)
        if not BO_opt:
            y_pred_test = fit.predict(X_test) if has_test else np.array([])
            if 'X_external_scaled' in Xy_data:
                y_pred_external = fit.predict(X_external)
            if kfold_cv_type == 'repeated' and y_pred_global_train_infold is not None:
                y_pred_train_infold = fit.predict(X_train)

        if kfold_cv_type == 'repeated':
            for y_val,y_pred_val,idx in zip(y_valid,y_pred_valid,test_outer_ix):
                y_global[idx].append(y_val)
                y_pred_global[idx].append(y_pred_val)
            if not BO_opt:
                for idx,y_pred_val_test in enumerate(y_pred_test):
                    y_pred_global_test[idx].append(y_pred_val_test)
                if 'X_external_scaled' in Xy_data:
                    for idx,y_pred_val_external in enumerate(y_pred_external):
                        y_pred_global_external[idx].append(y_pred_val_external)
                if y_pred_global_train_infold is not None:
                    for y_pred_val_trin,idx in zip(y_pred_train_infold,train_outer_ix):
                        y_pred_global_train_infold[idx].append(y_pred_val_trin)

        elif kfold_cv_type == 'sorted':
            y_global.append(y_valid)
            y_pred_global.append(y_pred_valid)

    return y_global,y_pred_global,y_pred_global_test,y_pred_global_external,y_pred_global_train_infold


def sort_n_load(Xy_data,include_test=False):
    '''
    Sort Xy data values to enhance reproducibility in cases where same databases are loaded
    with different row order, ensuring stable sorting across OS with kind='stable'.

    include_test=True combines train+validation with the test set first, since the
    boundary robustness (Low/High) analysis needs the true bottom/top 20% of the whole dataset, not
    just of the train+validation split - otherwise the true extremes of y could be sitting in
    the (excluded) test set. This must stay False (the default) during GENERATE's Bayesian
    Optimization / PFI hyperparameter search (kfold_cv's BO_opt=True path also runs there,
    not just for the final VERIFY/PREDICT report analysis) - the test set must never leak into
    hyperparameter selection, and mid-PFI train/test can also briefly have different columns.
    '''

    if include_test:
        X_all = np.concatenate([np.array(Xy_data['X_train_scaled']), np.array(Xy_data['X_test_scaled'])])
        y_all = np.concatenate([np.array(Xy_data['y_train']), np.array(Xy_data['y_test'])])
    else:
        X_all = np.array(Xy_data['X_train_scaled'])
        y_all = np.array(Xy_data['y_train'])

    sorted_indices = np.argsort(y_all, kind='stable')
    sorted_X_all = X_all[sorted_indices]
    sorted_y_all = y_all[sorted_indices]

    return sorted_X_all, sorted_y_all


def sorted_kfold_cv(loaded_model,model_data,Xy_data,error_labels,include_test=False):
    '''
    Performs a sorted k-fold cross-validation on the Xy dataset. Returns the average of the two results

    include_test=True runs it over the whole dataset (train+validation+test), for the final
    VERIFY/PREDICT report analysis. Must stay False for GENERATE's BO/PFI hyperparameter
    search (see sort_n_load()).
    '''

    # perform sorted 5-fold CV
    Xy_data['y_sorted_cv'],Xy_data['y_pred_sorted_cv'] = [],[]
    Xy_data['y_sorted_cv'],Xy_data['y_pred_sorted_cv'],_,_,_ = kfold_cv(Xy_data['y_sorted_cv'],Xy_data['y_pred_sorted_cv'],
                                                None,
                                                None,
                                                model_data,loaded_model,Xy_data,None,BO_opt=True,shuffle=False,kfold_cv_type='sorted',
                                                include_test=include_test)
    error1 = error_labels[model_data['type']][0]
    error2 = error_labels[model_data['type']][1]
    error3 = error_labels[model_data['type']][2]
    if model_data['type'].lower() == 'reg':
        Xy_data[f'{error1}_train_sorted_CV'], Xy_data[f'{error2}_train_sorted_CV'], Xy_data[f'{error3}_train_sorted_CV'] = [],[],[]
        for y_cv,y_pred_cd in zip(Xy_data['y_sorted_cv'],Xy_data['y_pred_sorted_cv']):
            r2_train_sorted_CV, mae_train_sorted_CV, rmse_train_sorted_CV = get_prediction_results(model_data,y_cv,y_pred_cd)
            Xy_data[f'{error1}_train_sorted_CV'].append(r2_train_sorted_CV)
            Xy_data[f'{error2}_train_sorted_CV'].append(mae_train_sorted_CV)
            Xy_data[f'{error3}_train_sorted_CV'].append(rmse_train_sorted_CV)

        # take the worst performing predictions from the top and bottom folds
        if model_data["error_type"].lower() in ['mae','rmse']:
            Xy_data[f'{model_data["error_type"]}_up_bottom'] = max(Xy_data[f'{model_data["error_type"]}_train_sorted_CV'][0], Xy_data[f'{model_data["error_type"]}_train_sorted_CV'][-1])
            Xy_data['r2_up_bottom'] = min(Xy_data['r2_train_sorted_CV'][0], Xy_data['r2_train_sorted_CV'][-1])
        else:  # r2
            Xy_data[f'{model_data["error_type"]}_up_bottom'] = min(Xy_data[f'{model_data["error_type"]}_train_sorted_CV'][0], Xy_data[f'{model_data["error_type"]}_train_sorted_CV'][-1])

    else:  # classification
        Xy_data[f'{error1}_train_sorted_CV'], Xy_data[f'{error2}_train_sorted_CV'], Xy_data[f'{error3}_train_sorted_CV'] = [],[],[]
        for y_cv, y_pred_cd in zip(Xy_data['y_sorted_cv'], Xy_data['y_pred_sorted_cv']):
            acc_fold, f1_fold, mcc_fold = get_prediction_results(model_data, y_cv, y_pred_cd)
            Xy_data[f'{error1}_train_sorted_CV'].append(acc_fold)
            Xy_data[f'{error2}_train_sorted_CV'].append(f1_fold)
            Xy_data[f'{error3}_train_sorted_CV'].append(mcc_fold)

        # Measure fold stability by difference between best and worst fold
        Xy_data[f'{model_data["error_type"]}_up_bottom'] = np.mean(np.abs(Xy_data[f'{model_data["error_type"]}_train_sorted_CV']))

    return Xy_data


def _cv_within_partition(self,model_data,X_train_pool,y_train_pool,X_test_pool,y_test_pool):
    '''
    Runs the SAME 10x repeated 5-fold CV used for interpolation (repeated_kfold_cv), scoped to
    an arbitrary 80/20 partition instead of the "official" train/test split - so boundary
    robustness predictions (Low/High, applicability domain) are computed exactly like
    interpolation's "10x 5-fold CV"/"test" pair: each point in the 80% pool is averaged over
    its ~10 out-of-fold CV appearances, and each point in the 20% held-out pool is averaged
    over all repeat*kfold (e.g. 10*5=50) models fitted during the run - instead of a single
    train-once/test-once fit. This makes the reported RMSE directly comparable to
    interpolation's, both being averages over the same kind of repeated-CV procedure rather
    than one being averaged and the other a single noisy draw.
    '''

    loaded_model = load_model(self, model_data['model'], **model_data['params'])
    temp_Xy_data = {
        'X_train_scaled': X_train_pool,
        'y_train': y_train_pool,
        'X_test_scaled': X_test_pool,
        'y_test': y_test_pool,
    }
    temp_Xy_data = repeated_kfold_cv(model_data,loaded_model,temp_Xy_data,BO_opt=False)

    # y_pred_test_sd: per-point SD computed directly over all repeat*kfold raw predictions
    # (same statistic interpolation's item 6 "Avg. standard deviation (SD)" already uses, via
    # Xy_data['y_pred_test_sd'] in repeated_kfold_cv) - kept in the same units/convention here
    # so boundary robustness's stability sub-metric is directly comparable to interpolation's
    return np.array(temp_Xy_data['y_pred_train']), np.array(temp_Xy_data['y_pred_test']), np.array(temp_Xy_data['y_pred_test_sd'])


def boundary_plot(self,model_data,Xy_data,path_n_suffix):
    '''
    Plots predicted vs actual values for the High extreme (top 20% of y) and Low extreme
    (bottom 20%), each held out from its own 10x repeated 5-fold CV run on the OTHER 80% - the
    same methodology interpolation's "10x 5-fold CV"/"test" plot uses (repeated_kfold_cv), just
    scoped to each extreme's own 80/20 partition of the whole (sorted) dataset instead of the
    official train/test split. Both the 80%-pool points and the 20%-extreme points are
    therefore averaged over repeated CV appearances, not a single train-once/test-once fit -
    making the reported RMSE directly comparable to interpolation's own averaged RMSE.
    '''

    X_all,y_all = sort_n_load(Xy_data,include_test=True)
    n_extreme = max(1, round(len(X_all) * 0.2))

    X_bottom80,y_bottom = X_all[:-n_extreme], y_all[:-n_extreme]
    X_high,y_high = X_all[-n_extreme:], y_all[-n_extreme:]
    X_top80,y_top = X_all[n_extreme:], y_all[n_extreme:]
    X_low,y_low = X_all[:n_extreme], y_all[:n_extreme]

    y_pred_bottom,y_pred_high,s_high = _cv_within_partition(self,model_data,X_bottom80,y_bottom,X_high,y_high)
    y_pred_top,y_pred_low,s_low = _cv_within_partition(self,model_data,X_top80,y_top,X_low,y_low)

    # RMSE of each extreme's own "remaining 80%" (from the same repeated-CV run), used as the
    # degradation ratio baseline - comparing against interpolation's own RMSE would still mix
    # two different dataset scopes (Low/High are drawn from the whole dataset, interpolation
    # only from train+validation), even though both are now the same repeated-CV procedure
    _,_,rmse_bottom80 = get_prediction_results(model_data,y_bottom,y_pred_bottom)
    _,_,rmse_top80 = get_prediction_results(model_data,y_top,y_pred_top)

    # RMSE of the extreme folds themselves (Low/High vs their repeated-CV predictions) - this is
    # the actual "how well does the model do on the extreme 20%" number shown in the report,
    # replacing VERIFY's old single-pass sorted-CV RMSE now that the predictions come from the
    # same repeated-CV procedure as interpolation
    _,_,rmse_high = get_prediction_results(model_data,y_high,y_pred_high)
    _,_,rmse_low = get_prediction_results(model_data,y_low,y_pred_low)

    # Spearman rank correlation replaces R2 as the "is the fit trustworthy" check for the
    # extreme folds - R2 (a variance-ratio) is unstable on a small, narrow-range subset and
    # ends up punishing models regardless of actual quality, whereas rank correlation is far
    # more robust at small N. Two complementary checks:
    # - internal: within the Low/High fold's own points, does the model rank them correctly
    #   (biggest actual value -> biggest predicted value)?
    # - global: over the WHOLE (sorted) dataset, does the model's ranking correctly place the
    #   true extremes as extremes at all, rather than compressing them toward the middle?
    #   Complements Bias (mean shift) with a rank-based view of the same failure mode.
    # fallbacks use 0.0 (float), not 0 (int) - the f"{value:.2}" format used when logging these
    # below raises ValueError ("Precision not allowed in integer format specifier") if value
    # ends up as a plain Python int, which happened on small/degenerate datasets (e.g. a
    # 2-descriptor, ~35-point dataset where a Low/High fold can have too few points or ties)
    spearman_high = stats.spearmanr(y_high,y_pred_high).correlation if len(y_high) > 1 else 0.0
    spearman_low = stats.spearmanr(y_low,y_pred_low).correlation if len(y_low) > 1 else 0.0
    spearman_high = 0.0 if np.isnan(spearman_high) else spearman_high
    spearman_low = 0.0 if np.isnan(spearman_low) else spearman_low

    # bottom80 (indices [0:-n_extreme]) + high (indices [-n_extreme:]) covers the whole sorted
    # dataset exactly once, with no gap or overlap - no need to also pull in the Low run's data
    y_all_sorted = np.concatenate([y_bottom,y_high])
    y_pred_all_sorted = np.concatenate([y_pred_bottom,y_pred_high])
    spearman_global = stats.spearmanr(y_all_sorted,y_pred_all_sorted).correlation
    spearman_global = 0.0 if np.isnan(spearman_global) else spearman_global

    # sub-metric 3: real range extension - does the model actually dare to predict beyond the
    # range it was trained on, or does it just clip toward the training boundary (a subtler
    # form of regression-to-the-mean that scaled RMSE/Spearman alone don't catch, since a model
    # that never leaves the training range can still rank/fit reasonably within it)?
    y_max_train = y_bottom.max() # High's model only ever saw up to this value while training
    y_min_train = y_top.min() # Low's model only ever saw down to this value while training
    crossing_high = np.mean(y_pred_high > y_max_train)
    crossing_low = np.mean(y_pred_low < y_min_train)

    # sub-metric 4: prediction uncertainty usefulness - s_high/s_low (per-point SD across all
    # repeat*kfold raw predictions, same statistic as interpolation's item 6) define a
    # +-2*SD "repeated-CV prediction band" per point (not a calibrated confidence interval).
    # 4a. coverage: does the true value actually fall inside that band?
    y_extreme = np.concatenate([y_high,y_low])
    pred_extreme = np.concatenate([y_pred_high,y_pred_low])
    s_extreme = np.concatenate([s_high,s_low])
    covered = (y_extreme >= pred_extreme-2*s_extreme) & (y_extreme <= pred_extreme+2*s_extreme)
    coverage = np.mean(covered)

    # 4b. does a higher SD actually flag the harder (higher-error) points? split the extreme
    # points by their own median SD and compare RMSE between the high-SD and low-SD halves
    err_extreme = np.abs(y_extreme-pred_extreme)
    median_s = np.median(s_extreme)
    low_sd_mask,high_sd_mask = s_extreme <= median_s, s_extreme > median_s
    if low_sd_mask.sum() > 0 and high_sd_mask.sum() > 0:
        rmse_low_sd = np.sqrt(np.mean(err_extreme[low_sd_mask]**2))
        rmse_high_sd = np.sqrt(np.mean(err_extreme[high_sd_mask]**2))
        sd_rmse_ratio = rmse_high_sd/rmse_low_sd if rmse_low_sd > 0 else float('inf')
    else: # not enough spread in SD to split into two groups
        sd_rmse_ratio = 0.0

    _boundary_scatter(self,model_data,path_n_suffix,'high_',
        y_bottom,y_pred_bottom,y_high,y_pred_high,'Bottom 80%','Top 20%')
    _boundary_scatter(self,model_data,path_n_suffix,'low_',
        y_top,y_pred_top,y_low,y_pred_low,'Top 80%','Bottom 20%')

    self.args.log.write(f"      -  Sorted CV extremes (repeated) : RMSE Low = {rmse_low:.2}, RMSE High = {rmse_high:.2}")
    self.args.log.write(f"      -  Real range extension (Low/High) : Low crossing = {crossing_low:.2}, High crossing = {crossing_high:.2}")
    self.args.log.write(f"      -  Degradation baseline (remaining 80%) : vs Low (top 80%) RMSE = {rmse_top80:.2}, vs High (bottom 80%) RMSE = {rmse_bottom80:.2}")
    self.args.log.write(f"      -  Spearman rank (Low/High) : Low (internal) = {spearman_low:.2}, High (internal) = {spearman_high:.2}, global = {spearman_global:.2}")
    self.args.log.write(f"      -  Uncertainty coverage (+/-2SD band) : {coverage:.2}")
    self.args.log.write(f"      -  Uncertainty RMSE ratio (high-SD/low-SD) : {sd_rmse_ratio:.3}")


def applicability_domain_plot(self,model_data,Xy_data,path_n_suffix):
    '''
    Boundary robustness in X-descriptor space (Section B sub-metric 5), complementing the
    Low/High sorted-CV analysis (boundary robustness in y-space). Ranks every point in the whole dataset by
    leverage - h_i = x_i^T (X_train^T X_train)^-1 x_i, the standard QSAR/regression
    "applicability domain" metric, which (unlike a per-descriptor or plain Euclidean check)
    accounts for correlations between descriptors - holds out the top 20% highest-leverage
    (most structurally/descriptor-wise extreme) points as a test set, trains on the remaining
    80% "typical" points, and scores how well the model extrapolates to those descriptor
    outliers.

    Leverage is computed using ONLY the training-candidate 80% (X_train above), not the whole
    dataset, so the holdout points never contribute to defining their own "how extreme am I"
    reference domain. Resolved in two passes: an initial leverage over the whole dataset picks
    a provisional 80/20 split, then leverage is recomputed using only that provisional 80% and
    the split is redone from those (now contamination-free) values.
    '''

    X_all = np.concatenate([np.array(Xy_data['X_train_scaled']), np.array(Xy_data['X_test_scaled'])])
    y_all = np.concatenate([np.array(Xy_data['y_train']), np.array(Xy_data['y_test'])])
    n_test = max(1, round(len(X_all) * 0.2))

    def _leverage(X_ref,X_query):
        XtX_inv = np.linalg.pinv(X_ref.T @ X_ref)
        return np.einsum('ij,jk,ik->i', X_query, XtX_inv, X_query)

    # pass 1: provisional split using leverage over the whole dataset (self-contaminated)
    leverage_provisional = _leverage(X_all,X_all)
    provisional_train_idx = np.argsort(leverage_provisional, kind='stable')[:-n_test]

    # pass 2: recompute leverage using only the provisional 80% training candidates, then
    # re-split from these contamination-free values
    leverage = _leverage(X_all[provisional_train_idx],X_all)
    sorted_idx = np.argsort(leverage, kind='stable')
    train_idx, test_idx = sorted_idx[:-n_test], sorted_idx[-n_test:]

    X_train_ad, y_train_ad = X_all[train_idx], y_all[train_idx]
    X_test_ad, y_test_ad = X_all[test_idx], y_all[test_idx]

    # 10x repeated 5-fold CV within the 80% "typical" pool (same methodology as interpolation
    # and the Low/High boundary robustness plots - see _cv_within_partition()), instead of a
    # single train-once/test-once fit, so this RMSE is directly comparable to the others
    y_pred_train_ad,y_pred_test_ad,_ = _cv_within_partition(self,model_data,X_train_ad,y_train_ad,X_test_ad,y_test_ad)

    _,_,rmse_ad = get_prediction_results(model_data,y_test_ad,y_pred_test_ad)

    # two complementary plots: an actual-vs-predicted scatter (same style as Low/High) shows how
    # close predictions are to actual values, tying visually to the Scaled RMSE this item is
    # scored on; a Williams plot (leverage vs standardized residual, with the standard QSAR
    # warning-leverage h* = 3(p+1)/n and +-3 SD bounds) shows the whole dataset at once and
    # flags which points are structurally outside the domain / have suspicious residuals
    _boundary_scatter(self,model_data,path_n_suffix,'ad_',
        y_train_ad,y_pred_train_ad,y_test_ad,y_pred_test_ad,'Typical 80%','High leverage 20%',
        title='Applicability domain (leverage)',legend_fontsize=12)

    n_train_ad, p = X_train_ad.shape
    h_star = 3*(p+1)/n_train_ad
    residuals_all = np.concatenate([y_train_ad-y_pred_train_ad, y_test_ad-y_pred_test_ad])
    resid_sd = np.std(residuals_all) if np.std(residuals_all) > 0 else 1
    std_resid_train = (y_train_ad-y_pred_train_ad)/resid_sd
    std_resid_test = (y_test_ad-y_pred_test_ad)/resid_sd

    _williams_plot(path_n_suffix,
        leverage[train_idx],std_resid_train,leverage[test_idx],std_resid_test,h_star)

    self.args.log.write(f"      -  Applicability domain (leverage) : RMSE = {rmse_ad:.2}, h* = {h_star:.2}")


def _place_legend(ax,all_x,all_y,labels,fontsize=14):
    '''
    Places the legend at the same fixed bottom-right anchor (0.70, 0.15) used by the main
    "10x 5-fold CV / test" plot (graph_reg), so the margin matches across the report, unless a
    data point actually falls behind it, in which case (and only then) it falls back to
    matplotlib's loc='best', which auto-avoids overlapping data.
    '''

    legend_kwargs = dict(handletextpad=0, fancybox=True, shadow=True, ncol=5, labels=labels, fontsize=fontsize)

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.70, 0.15), **legend_kwargs)

    # get_window_extent() needs an explicit renderer on the Agg backend (used for saving
    # figures without a display) - without it, it silently returns a stale/degenerate bbox and
    # the overlap check below never triggers, which is why the auto-move wasn't working
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)

    inv = ax.transData.inverted()
    (x0,y0),(x1,y1) = inv.transform([[bbox.x0,bbox.y0],[bbox.x1,bbox.y1]])
    x_lo,x_hi = sorted([x0,x1])
    y_lo,y_hi = sorted([y0,y1])

    overlap = any(x_lo <= x <= x_hi and y_lo <= y <= y_hi for x,y in zip(all_x,all_y))
    if overlap:
        legend.remove()
        legend = ax.legend(loc='best', **legend_kwargs)

    return legend


def _williams_plot(path_n_suffix,leverage_train,resid_train,leverage_test,resid_test,h_star,print_fun=True):
    '''
    Standard QSAR "Williams plot": leverage (x) vs standardized residual (y), marking the
    warning leverage h* and the +-3 SD bounds that define the applicability domain.
    print_fun=False skips the in-image title - used for the small (270px-wide) Section B
    thumbnails, where the container height crops it anyway (see print_img_tag()) and the
    surrounding item text already says what the plot shows
    '''

    importlib.reload(plt) # needed to avoid threading issues
    sb.set(style="ticks")
    graph_style = get_graph_style()

    _, ax = plt.subplots(figsize=(7.45,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if print_fun:
        plt.text(0.5, 1.08, f'Applicability domain (Williams plot) of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
            fontsize=14, fontweight='bold', transform = ax.transAxes)

    _ = ax.scatter(leverage_train, resid_train,
                c = graph_style['color_train'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)
    _ = ax.scatter(leverage_test, resid_test,
                c = graph_style['color_test'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=3)

    ax.axhline(3, color='dimgray', linestyle='--', linewidth=1, zorder=1)
    ax.axhline(-3, color='dimgray', linestyle='--', linewidth=1, zorder=1)
    ax.axvline(h_star, color='dimgray', linestyle='--', linewidth=1, zorder=1)

    _place_legend(ax,np.concatenate([leverage_train,leverage_test]),np.concatenate([resid_train,resid_test]),
        ['Typical 80%','High leverage 20%'],fontsize=12)

    plt.ylabel('Standardized residual', fontsize=14)
    plt.xlabel('Leverage', fontsize=14)

    all_leverage = np.concatenate([leverage_train, leverage_test])
    all_resid = np.concatenate([resid_train, resid_test])
    x_max = max(max(all_leverage)*1.1, h_star*1.2)
    y_max = max(max(np.abs(all_resid))*1.1, 3.5)

    ax.grid(linestyle='--', linewidth=1)
    plt.xlim(0, x_max)
    plt.ylim(-y_max, y_max)

    williams_plot_file = f'{os.path.dirname(path_n_suffix)}/Results_boundary_williams_{os.path.basename(path_n_suffix)}.png'
    plt.savefig(f'{williams_plot_file}', dpi=300, bbox_inches='tight')
    plt.close()


def _boundary_scatter(self,model_data,path_n_suffix,file_prefix,y_rest,y_pred_rest,y_extreme,y_pred_extreme,rest_label,extreme_label,title='Boundary robustness (sorted CV)',legend_fontsize=14,print_fun=True):
    '''
    Shared plotting logic for boundary_plot()'s High and Low scatter plots: "rest" of the
    data in blue, the highlighted extreme fold in red.
    print_fun=False skips the in-image title - used for the small (270px-wide) Section B
    thumbnails, where the container height crops it anyway (see print_img_tag()) and the
    surrounding item text already says what the plot shows
    '''

    importlib.reload(plt) # needed to avoid threading issues
    sb.set(style="ticks")
    graph_style = get_graph_style()

    _, ax = plt.subplots(figsize=(7.45,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if print_fun:
        plt.text(0.5, 1.08, f'{title} of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
            fontsize=14, fontweight='bold', transform = ax.transAxes)

    _ = ax.scatter(y_rest, y_pred_rest,
                c = graph_style['color_train'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)
    _ = ax.scatter(y_extreme, y_pred_extreme,
                c = graph_style['color_test'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=3)

    _place_legend(ax,np.concatenate([y_rest,y_extreme]),np.concatenate([y_pred_rest,y_pred_extreme]),
        [rest_label,extreme_label],fontsize=legend_fontsize)

    Xy_data_df = pd.DataFrame()
    Xy_data_df['y_rest'] = y_rest
    Xy_data_df['y_pred_rest'] = y_pred_rest
    if len(y_rest) >= 10:
        _ = sb.regplot(x='y_rest', y='y_pred_rest', data=Xy_data_df, scatter=False, color=".1",
                        truncate = True, ax=ax, seed=model_data['seed'])

    plt.ylabel(f'Predicted {model_data["y"]}', fontsize=14)
    plt.xlabel(f'{model_data["y"]}', fontsize=14)

    y_all = np.concatenate([y_rest, y_extreme])
    y_pred_all = np.concatenate([y_pred_rest, y_pred_extreme])
    size_space = 0.1*abs(min(y_all)-max(y_all)) if max(y_all) != min(y_all) else 1
    min_value_graph = min(min(y_all),min(y_pred_all))-size_space
    max_value_graph = max(max(y_all),max(y_pred_all))+size_space

    ax.grid(linestyle='--', linewidth=1)
    plt.xlim(min_value_graph, max_value_graph)
    plt.ylim(min_value_graph, max_value_graph)

    boundary_plot_file = f'{os.path.dirname(path_n_suffix)}/Results_boundary_{file_prefix}{os.path.basename(path_n_suffix)}.png'
    plt.savefig(f'{boundary_plot_file}', dpi=300, bbox_inches='tight')
    plt.close()

    path_reduced = '/'.join(f'{boundary_plot_file}'.replace('\\','/').split('/')[-2:])
    self.args.log.write(f"      -  Graph in: {path_reduced}")




def k_means(self,X_scaled,csv_y,size,seed,idx_list):
    '''
    
    Uses k-means clustering to select the test points to be as diverse as possible, 
    but it returns the test pointsReturns the data points that will be used as training set based on the k-means clustering
    
    '''
    
    # number of clusters in the training set from the k-means clustering (based on the
    # training set size specified above)
    X_scaled_array = np.asarray(X_scaled)
    number_of_clusters = size

    # to avoid points from the validation set outside the training set, the 2 first training
    # points are automatically set as the 2 points with minimum/maximum response value
    if self.args.type.lower() == 'reg':
        test_points = []
        training_idx = [csv_y.idxmin(),csv_y.idxmax()]
        number_of_clusters -= 2
    else:
        test_points = []
        training_idx = []
    
    # runs the k-means algorithm and keeps the closest point to the center of each cluster
    kmeans = KMeans(n_clusters=number_of_clusters,random_state=seed)
    try:
        kmeans.fit(X_scaled_array)
    except ValueError:
        self.args.log.write("\nx  The K-means clustering process failed! This might be due to having NaN or strings as descriptors (curate the data first with CURATE) or having too few datapoints!")
        sys.exit()
    centers = kmeans.cluster_centers_

    # Euclidean distance from every point to every cluster center, computed once with a
    # vectorized pairwise distance call instead of a per-point/per-descriptor Python triple loop
    dists_to_centers = cdist(X_scaled_array, centers)

    claimed = np.zeros(len(X_scaled_array), dtype=bool)
    claimed[list(training_idx)] = True
    for i in range(number_of_clusters):
        dists_i = np.where(claimed, np.inf, dists_to_centers[:, i])
        training_point = int(np.argmin(dists_i))
        if np.isinf(dists_i[training_point]):
            # every point is already claimed by an earlier cluster (e.g. more clusters than
            # remaining unclaimed points) - skip instead of silently re-appending a duplicate
            continue
        training_idx.append(training_point)
        claimed[training_point] = True

    test_idx = [idx for idx in range(len(X_scaled_array[:, 0])) if idx not in training_idx]
    test_points = [idx_list[i] for i in test_idx]
    test_points.sort()

    return test_points


def PFI_filter(self, Xy_data, model_data):
    '''
    Performs the PFI calculation and returns a list of the descriptors that are not important
    '''

    # load and fit model
    loaded_model = load_model(self,model_data['model'],**model_data['params'])
    loaded_model.fit(Xy_data['X_train_scaled'], Xy_data['y_train'])

    # select scoring function for PFI analysis based on the error type
    scoring, score_model, _ = scoring_n_score(self,model_data,Xy_data,loaded_model)
    
    perm_importance = permutation_importance(loaded_model, Xy_data['X_train_scaled'], Xy_data['y_train'], scoring=scoring, n_repeats=self.args.pfi_epochs, random_state=self.args.seed, n_jobs=1)

    # transforms the values into a list and sort the PFI values with the descriptor names
    descp_cols_pfi, PFI_values, PFI_sd = [],[],[]
    for i,desc in enumerate(Xy_data['X_train_scaled'].columns):
        descp_cols_pfi.append(desc) # includes lists of descriptors not column names!
        PFI_values.append(perm_importance.importances_mean[i])
        PFI_sd.append(perm_importance.importances_std[i])
  
    PFI_values, PFI_sd, descp_cols_pfi = (list(t) for t in zip(*sorted(zip(PFI_values, PFI_sd, descp_cols_pfi), reverse=True)))

    # PFI filter
    PFI_discard_cols = []
    # the threshold is based either on the RMSE of the model or the importance of the most important descriptor
    PFI_thres = max([abs(self.args.pfi_threshold*score_model),abs(self.args.pfi_threshold*PFI_values[0])])
    for i in range(len(PFI_values)):
        if PFI_values[i] < PFI_thres:
            PFI_discard_cols.append(descp_cols_pfi[i])

    return PFI_discard_cols,descp_cols_pfi


def scoring_n_score(self,model_data,Xy_data,loaded_model):
    '''
    Get scoring system and score of the original model with CV
    '''

    error_type = model_data['error_type'].lower()
    scoring = get_scoring_key(model_data['type'],error_type)
    cv_model = RepeatedKFold(n_splits=self.args.kfold, n_repeats=self.args.repeat_kfolds, random_state=self.args.seed)
    score_model = cross_val_score(estimator = loaded_model, X=Xy_data['X_train_scaled'], y=Xy_data['y_train'],scoring=scoring, cv =cv_model)
    score_model = score_model.mean()

    if model_data['error_type'].lower() in ['rmse','mae']:
        score_model = -score_model

    return scoring, score_model, error_type


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
    ax.set_ylabel("",fontsize=fontsize)
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    title_fig = f'Heatmap ML models {suffix}'
    plt.title(title_fig, y=1.04, fontsize = fontsize, fontweight="bold")
    sb.despine(top=False, right=False)
    name_fig = '_'.join(title_fig.split())
    plt.savefig(f'{path_raw.joinpath(name_fig)}.png', dpi=300, bbox_inches='tight')
    plt.close()

    path_reduced = '/'.join(f'{path_raw}'.replace('\\','/').split('/')[-2:])
    self.args.log.write(f'\no  {name_fig} succesfully created in {path_reduced}')


def create_score_heatmap(csv_df,save_path):
    '''
    Heatmap of each model's final score (0-10) for ONE metric (Interpolation or
    Boundary robustness - one call per side), used in Section F of the report instead of
    GENERATE's combined-RMSE heatmap when --all_models is active - the raw BO objective
    (combined RMSE) only reflects the search criterion used to tune each model's
    hyperparameters, not the richer final score (test performance, stability, boundary
    behavior, etc.) that the rest of the report is built around, so it isn't the most useful
    basis for comparing models against each other. Kept as two single-row images (not one
    combined 2-row image) so Interpolation can sit in the report's left column and Boundary
    robustness in the right, matching the same left/right separation used by every other
    section
    '''

    importlib.reload(plt)
    sb.set(font_scale=1.2, style='ticks')
    _, ax = plt.subplots(figsize=(7.45,4))
    cmap_blues_75_percent_512 = [mcolor.rgb2hex(c) for c in plt.cm.Blues(np.linspace(0, 0.8, 512))]
    ax = sb.heatmap(csv_df, annot=True, linewidth=1, cmap=cmap_blues_75_percent_512,
                     vmin=0, vmax=10, cbar_kws={'label': 'Score (0-10)'}, mask=csv_df.isnull())
    fontsize = 14
    ax.set_xlabel("ML Model",fontsize=fontsize)
    ax.set_ylabel("",fontsize=fontsize)
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # no in-image title: the report embeds this at a fixed width with auto height (no crop),
    # and the report page already prints an "Interpolation"/"Boundary robustness" bold caption
    # right above it (see print_generate() in report.py), so a second, redundant title would
    # just eat into the image's vertical space
    sb.despine(top=False, right=False)
    plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight')
    plt.close()


def graph_reg(self,Xy_data,params_dict,set_types,path_n_suffix,graph_style,csv_test=False,print_fun=True,sd_graph=False,sd_set='test',show_title=True):
    '''
    Plot regression graphs of predicted vs actual values for train, validation and test sets.
    sd_set (only used when sd_graph=True and csv_test=False) picks which averaged-CV ± SD set
    to plot: 'test' (default, unchanged) or 'train' (out-of-fold CV predictions in
    train+validation, used for Interpolation item 6's train+validation SD sub-metric).
    show_title=False skips the in-image title only (print_fun still controls the "Graph in:"
    log line independently) - used for the small (270px-wide) Section B thumbnails, where the
    container height crops the title anyway (see print_img_tag()) and the surrounding item
    text already says what the plot shows
    '''

    # Create graph
    importlib.reload(plt) # needed to avoid threading issues
    sb.set(style="ticks")

    _, ax = plt.subplots(figsize=(7.45,6))

    # Set tick sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    error_bars = sd_set if (sd_graph and not csv_test) else "test"

    title_graph = graph_title(self,csv_test,sd_graph,error_bars)

    if show_title:
        plt.text(0.5, 1.08, f'{title_graph} of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
            fontsize=14, fontweight='bold', transform = ax.transAxes)

    # Plot the data (also used for the train/out-of-fold ± SD variant, sd_set='train' - same
    # blue "10x 5-fold CV" dots as the regular plot, with the ± SD error bars added on top
    # below, instead of the plain "fmt='none'" error bars showing with no dots underneath)
    if not sd_graph or sd_set == 'train':
        _ = ax.scatter(Xy_data["y_train"], Xy_data["y_pred_train"],
                    c = graph_style['color_train'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)

    if csv_test:
        error_bars = "external"
        _ = ax.scatter(Xy_data["y_external"], Xy_data["y_pred_external"],
                        c = graph_style['color_test'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)

    # skipped for the train/out-of-fold ± SD variant (sd_set='train') - that variant only
    # shows the train+validation errorbar plot below, not the (unrelated) test points
    elif not (sd_graph and sd_set == 'train'):
        _ = ax.scatter(Xy_data["y_test"], Xy_data["y_pred_test"],
                    c = graph_style['color_test'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=3)

    # average CV ± SD graphs 
    if sd_graph:
        if not csv_test:   
            # Plot the data with the error bars
            _ = ax.errorbar(Xy_data[f"y_{error_bars}"], Xy_data[f"y_pred_{error_bars}"], yerr=Xy_data[f"y_pred_{error_bars}_sd"], fmt='none', ecolor="gray", capsize=3, zorder=1)
            # Adjust labels from legend
            set_types=[error_bars,f'± SD']

        else:
            _ = ax.errorbar(Xy_data[f"y_{error_bars}"], Xy_data[f"y_pred_{error_bars}"], yerr=Xy_data[f"y_pred_{error_bars}_sd"], fmt='none', ecolor="gray", capsize=3, zorder=1)
            set_types=['External test',f'± SD']

    # legend and regression line with 95% CI considering all possible lines (not CI of the points)
    legend_coords = None
    if 'CV' in set_types[0]: # CV in VERIFY
        legend_coords = (0.70, 0.15)
    elif len(set_types) == 2: # external test or sets with ± SD
        if 'External test' in set_types:
            legend_coords = (0.66, 0.15)
        else:
            legend_coords = (0.735, 0.15)
    ax.legend(loc='upper center', bbox_to_anchor=legend_coords,
            handletextpad=0,
            fancybox=True, shadow=True, ncol=5, labels=set_types, fontsize=14)

    Xy_data_df = pd.DataFrame()
    if not sd_graph:
        line_suff = 'train'
    elif not csv_test:
        line_suff = error_bars # 'test' or 'train', matches sd_set
    else:
        line_suff = 'external'

    Xy_data_df[f"y_{line_suff}"] = Xy_data[f"y_{line_suff}"]
    Xy_data_df[f"y_pred_{line_suff}"] = Xy_data[f"y_pred_{line_suff}"]
    if len(Xy_data_df[f"y_pred_{line_suff}"]) >= 10:
        _ = sb.regplot(x=f"y_{line_suff}", y=f"y_pred_{line_suff}", data=Xy_data_df, scatter=False, color=".1", 
                        truncate = True, ax=ax, seed=params_dict['seed'])

    # Title and labels of the axis
    plt.ylabel(f'Predicted {params_dict["y"]}', fontsize=14)
    plt.xlabel(f'{params_dict["y"]}', fontsize=14)

    # set axis limits and graph PATH
    min_value_graph,max_value_graph,reg_plot_file,path_reduced = graph_vars(Xy_data,set_types,csv_test,path_n_suffix,sd_graph,sd_set)

    # track the range of predictions (used in ROBERT score) - an empty internal test set
    # (e.g. --csv_test with --auto_test False) has no y_test values to fold in
    if len(Xy_data["y_test"]) > 0:
        pred_min = min(min(Xy_data["y_train"]),min(Xy_data["y_test"]))
        pred_max = max(max(Xy_data["y_train"]),max(Xy_data["y_test"]))
    else:
        pred_min = min(Xy_data["y_train"])
        pred_max = max(Xy_data["y_train"])
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
    plt.close()
    if print_fun:
        self.args.log.write(f"      -  Graph in: {path_reduced}")


def graph_title(self,csv_test,sd_graph,error_bars):
    '''
    Retrieves the corresponding graph title.
    '''

    # set title for regular graphs
    if not sd_graph:
        if not csv_test:
            # regular graphs
            title_graph = f'Predictions CV and test set'
        else:
            title_graph = f'{os.path.basename(self.args.csv_test)}'
            if len(title_graph) > 30:
                title_graph = f'{title_graph[:27]}...'

    # set title for averaged CV ± SD graphs
    else:
        if not csv_test:
            # 'train' here means the out-of-fold CV predictions in train+validation, not a
            # literal in-fold training fit - name it accordingly to avoid confusion
            sets_title = 'out-of-fold CV' if error_bars == 'train' else error_bars
        else:
            sets_title = 'external test'

        title_graph = f'{sets_title} set ± SD (CV)'

    return title_graph


def graph_vars(Xy_data,set_types,csv_test,path_n_suffix,sd_graph,sd_set='test'):
    '''
    Set axis limits for regression plots and PATH to save the graphs
    '''

    # x and y axis limits for graphs with multiple sets
    if not csv_test:
        # an empty internal test set (e.g. --csv_test with --auto_test False) has no y_test
        # values to fold into the axis limits - min()/max() reject empty sequences outright
        has_test = len(Xy_data["y_test"]) > 0

        size_space = 0.1*abs(min(Xy_data["y_train"])-max(Xy_data["y_train"]))
        min_value_graph = min(min(Xy_data["y_train"]),min(Xy_data["y_pred_train"]))
        if has_test:
            min_value_graph = min(min_value_graph,min(Xy_data["y_test"]),min(Xy_data["y_pred_test"]))
        min_value_graph = min_value_graph-size_space

        max_value_graph = max(max(Xy_data["y_train"]),max(Xy_data["y_pred_train"]))
        if has_test:
            max_value_graph = max(max_value_graph,max(Xy_data["y_test"]),max(Xy_data["y_pred_test"]))
        max_value_graph = max_value_graph+size_space

    else: # limits for graphs with only one set
        set_type = 'external'
        size_space = 0.1*abs(min(Xy_data[f'y_{set_type}'])-max(Xy_data[f'y_{set_type}']))
        min_value_graph = min(min(Xy_data[f'y_{set_type}']),min(Xy_data[f'y_pred_{set_type}']))
        min_value_graph = min_value_graph-size_space
        max_value_graph = max(max(Xy_data[f'y_{set_type}']),max(Xy_data[f'y_pred_{set_type}']))
        max_value_graph = max_value_graph+size_space

    # PATH of the graph
    if not csv_test:
        if not sd_graph:
            reg_plot_file = f'{os.path.dirname(path_n_suffix)}/Results_{os.path.basename(path_n_suffix)}.png'
        elif sd_set == 'train':
            reg_plot_file = f'{os.path.dirname(path_n_suffix)}/CV_variability_cv_{os.path.basename(path_n_suffix)}.png'
        else:
            reg_plot_file = f'{os.path.dirname(path_n_suffix)}/CV_variability_{os.path.basename(path_n_suffix)}.png'
        path_reduced = '/'.join(f'{reg_plot_file}'.replace('\\','/').split('/')[-2:])

    else:
        folder_graph = f'{os.path.dirname(path_n_suffix)}/csv_test'
        if not sd_graph:
            reg_plot_file = f'{folder_graph}/Results_{os.path.basename(path_n_suffix)}_{set_type}.png'
        else:
            reg_plot_file = f'{folder_graph}/CV_variability_{os.path.basename(path_n_suffix)}_{set_type}.png'
        path_reduced = '/'.join(f'{reg_plot_file}'.replace('\\','/').split('/')[-3:])
    
    return min_value_graph,max_value_graph,reg_plot_file,path_reduced


def graph_clas(self,Xy_data,params_dict,set_type,path_n_suffix,csv_test=False,print_fun=True):
    '''
    Plot a confusion matrix with the prediction vs actual values
    '''

    importlib.reload(plt) # needed to avoid threading issues

    # get confusion matrix
    if 'CV' in set_type: # CV graphs
        y_train_binary = np.round(Xy_data[f'y_train']).astype(int)
        y_pred_train_binary = np.round(Xy_data[f'y_pred_train']).astype(int)
        class_labels = sorted(np.unique(np.concatenate([y_train_binary, y_pred_train_binary])))
        y_actual = y_train_binary
        y_pred = y_pred_train_binary
    else: # other graphs
        y_binary = np.round(Xy_data[f'y_{set_type}']).astype(int)
        y_pred_binary = np.round(Xy_data[f'y_pred_{set_type}']).astype(int)
        class_labels = sorted(np.unique(np.concatenate([y_binary, y_pred_binary])))
        y_actual = y_binary
        y_pred = y_pred_binary

    display_labels = None
    class_mapping_reverse = params_dict.get('class_mapping_reverse')
    if class_mapping_reverse:
        class_mapping_reverse = _parse_class_mapping_reverse(class_mapping_reverse)
        class_labels = sorted(class_mapping_reverse)
        display_labels = [class_mapping_reverse[label] for label in class_labels]
    elif 'class_0_label' in params_dict and 'class_1_label' in params_dict:
        class_labels = [0, 1]
        display_labels = [params_dict['class_0_label'], params_dict['class_1_label']]

    # plot directly onto a pre-created figure (same format/size used in reg graphs) instead
    # of letting from_predictions() create and leak its own internal figure when ax=None
    _, ax = plt.subplots(figsize=(7.45,6))
    matrix = ConfusionMatrixDisplay.from_predictions(
        y_actual,
        y_pred,
        labels=class_labels,
        normalize=None,
        cmap='Blues',
        display_labels=display_labels,
        ax=ax,
    )

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
    plt.close()

    if print_fun:
        self.args.log.write(f"      -  Graph in: {path_reduced}")


def linear_equation(self,Xy_data,model_data,loaded_model):
    '''
    Prints the linear regression equation for MVL models. Must be logged before shap_analysis()
    and PFI_plot(), since REPORT's print_features() reads this equation back out of the log file
    to display it above the SHAP/PFI plots. loaded_model must already be fitted on
    Xy_data['X_train_scaled']/['y_train'] (shared with shap_analysis()/PFI_plot() so the same
    model isn't refit 3 times in a row for the same data)
    '''

    terms = [f'{coef:.3f}*{desc}' for coef,desc in zip(loaded_model.coef_, Xy_data['X_train_scaled'].columns)]
    equation = f"y = {loaded_model.intercept_:.3f} + " + ' + '.join(terms)
    equation = equation.replace('+ -', '- ')

    print_eq = f"\n   o  Linear model equation"
    print_eq += f"\n      -  {equation}"

    self.args.log.write(print_eq)


def shap_analysis(self,Xy_data,model_data,path_n_suffix,loaded_model):
    '''
    Plots and prints the results of the SHAP analysis. loaded_model must already be fitted on
    Xy_data['X_train_scaled']/['y_train']
    '''

    importlib.reload(plt) # needed to avoid threading issues
    _, _ = plt.subplots(figsize=(7.45,6))

    shap_plot_file = f'{os.path.dirname(path_n_suffix)}/SHAP_{os.path.basename(path_n_suffix)}.png'

    # run the SHAP analysis and save the plot
    if model_data['model'].upper() in ['RF', 'GB']:
        # exact Shapley values for tree ensembles in polynomial time, instead of
        # the Exact/Permutation fallback that shap.Explainer(model.predict, X) uses
        # (that fallback can't detect the tree structure and gets very slow on
        # datasets with many rows and/or descriptors)
        explainer = shap.TreeExplainer(loaded_model)
        shap_values = explainer(Xy_data['X_train_scaled'])
    elif model_data['model'].upper() == 'MVL':
        # exact Shapley values in closed form for linear models (no repeated
        # model evaluations), instead of the black-box Exact/Permutation fallback
        explainer = shap.LinearExplainer(loaded_model, Xy_data['X_train_scaled'])
        shap_values = explainer(Xy_data['X_train_scaled'])
    else:
        # black-box models (NN, GP, AdaB) still need the generic Exact/Permutation
        # fallback, which gets slow because its cost scales with the size of the
        # background data. Summarizing the background speeds this up a lot on large
        # datasets; below 100 points summarizing barely helps and only adds
        # approximation error, so the full training set is used as-is in that case
        background = Xy_data['X_train_scaled']
        if len(background) > 100:
            background = shap.sample(background, nsamples=100, random_state=model_data['seed'])
        explainer = shap.Explainer(loaded_model.predict, background, seed=model_data['seed'])
        try:
            shap_values = explainer(Xy_data['X_train_scaled'])
        except ValueError:
            shap_values = explainer(Xy_data['X_train_scaled'],max_evals=(2*len(Xy_data['X_train_scaled'].columns))+1)

    if np.asarray(shap_values.values).ndim == 3:
        # classifiers return SHAP values per class (n_samples, n_features, n_classes);
        # shap's own plotting code isn't reliable across class-axis sizes (e.g. size 1
        # for some binary fits), so pick the last class (the positive class in binary
        # problems) explicitly and plot/summarize a plain 2D array throughout
        shap_values = shap_values[..., -1]

    shap_show = [self.args.shap_show,len(Xy_data['X_train_scaled'].columns)]
    aspect_shap = 25+((min(shap_show)-2)*5)
    height_shap = 1.2+min(shap_show)/4

    # explainer = shap.TreeExplainer(loaded_model) # in case the standard version doesn't work
    _ = shap.summary_plot(shap_values, Xy_data['X_train_scaled'], max_display=self.args.shap_show,show=False, plot_size=[7.45,height_shap])

    # set title
    plt.title(f'SHAP analysis of {os.path.basename(path_n_suffix)}', fontsize = 14, fontweight="bold")

    path_reduced = '/'.join(f'{shap_plot_file}'.replace('\\','/').split('/')[-2:])
    print_shap = f"\n   o  SHAP plot saved in {path_reduced}"

    # collect SHAP values and print (vectorized instead of looping over every row per descriptor)
    desc_list = list(Xy_data['X_train_scaled'].columns)
    min_list = shap_values.values.min(axis=0).tolist()
    max_list = shap_values.values.max(axis=0).tolist()
    
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
    plt.close()


def PFI_plot(self,Xy_data,model_data,path_n_suffix,loaded_model):
    '''
    Plots and prints the results of the PFI analysis. loaded_model must already be fitted on
    Xy_data['X_train_scaled']/['y_train']
    '''

    importlib.reload(plt) # needed to avoid threading issues
    pfi_plot_file = f'{os.path.dirname(path_n_suffix)}/PFI_{os.path.basename(path_n_suffix)}.png'

    # select scoring function for PFI analysis based on the error type
    scoring, _, error_type = scoring_n_score(self,model_data,Xy_data,loaded_model)

    perm_importance = permutation_importance(loaded_model, Xy_data['X_train_scaled'], Xy_data['y_train'], scoring=scoring, n_repeats=self.args.pfi_epochs, random_state=model_data['seed'], n_jobs=1)

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
    plt.close()

    path_reduced = '/'.join(f'{pfi_plot_file}'.replace('\\','/').split('/')[-2:])
    print_PFI = f"\n   o  PFI plot saved in {path_reduced}"

    print_PFI += f'\n      Influence on {error_type.upper()}'

    for i,desc in enumerate(desc_list):
        print_PFI += f"\n      -  {desc} = {PFI_values[i]:.2} +- {PFI_sd[i]:.2}"
    
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
    _ = ax.scatter(outliers_data['test_scaled'], outliers_data['test_scaled'],
        c = graph_style['color_test'], s = graph_style['dot_size'], edgecolor = 'k', linewidths = 0.8, alpha = graph_style['alpha'], zorder=2)
    
    # Set styling preferences and graph limits
    plt.xlabel('SD of the errors',fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('SD of the errors',fontsize=14)
    plt.yticks(fontsize=14)
    
    axis_limit = max(outliers_data['train_scaled'], key=abs)
    if outliers_data.get('test_scaled') is not None and len(outliers_data['test_scaled']) > 0:
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
    plt.close()
    
    path_reduced = '/'.join(f'{outliers_plot_file}'.replace('\\','/').split('/')[-2:])
    print_outliers += f"\n   o  Outliers plot saved in {path_reduced}"

    if 'train' not in name_points:
        print_outliers += f'\n      x  No names option (or var missing in CSV file)! Outlier names will not be shown'
    else:
        if 'test_scaled' in outliers_data and 'test' not in name_points:
            print_outliers += f'\n      x  No names option (or var missing in CSV file in the test file)! Outlier names will not be shown'

    print_outliers = outlier_analysis(print_outliers,outliers_data,'train')
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

    n_points = len(outliers_data[n_points_label])
    per_cent = (len(outliers_data[outliers_label])/n_points)*100 if n_points > 0 else 0.0
    print_outliers += f"\n      {label_set}: {len(outliers_data[outliers_label])} outliers out of {n_points} datapoints ({per_cent:.1f}%)"
    for val,name in zip(outliers_data[outliers_label], outliers_data[outliers_name]):
        print_outliers += f"\n      -  {name} ({val:.2} SDs)"
    return print_outliers


def outlier_filter(self, Xy_data, name_points):
    '''
    Calculates and stores absolute errors in SD units for all the sets
    '''
    
    # calculate absolute errors between predicted y and actual values
    outliers_train = [abs(x-y) for x,y in zip(Xy_data['y_train'],Xy_data['y_pred_train'])]
    outliers_test = [abs(x-y) for x,y in zip(Xy_data['y_test'],Xy_data['y_pred_test'])]

    # the errors are scaled using standard deviation units. When the absolute
    # error is larger than the t-value, the point is considered an outlier. All the sets
    # use the mean and SD of the train set
    outliers_mean = np.mean(outliers_train)
    outliers_sd = np.std(outliers_train)
    # a train set with (near-)identical errors across all points gives SD = 0, which would
    # turn every scaled error into inf/nan instead of correctly showing "no outliers"
    if outliers_sd == 0:
        outliers_sd = 1

    outliers_data = {}
    outliers_data['train_scaled'] = (outliers_train-outliers_mean)/outliers_sd
    outliers_data['test_scaled'] = (outliers_test-outliers_mean)/outliers_sd

    print_outliers, naming, naming_test = '', False, False
    if 'train' in name_points:
        naming = True
        if 'test' in name_points:
            naming_test = True

    outliers_data['outliers_train'], outliers_data['names_train'] = detect_outliers(self, outliers_data['train_scaled'], name_points, naming, 'train')
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
    plt.text(0.5, 1.08, f'y-values distribution (CV + test) of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
    fontsize=14, fontweight='bold', transform = ax.transAxes)

    plt.grid(linestyle='--', linewidth=1)

    # combine train and validation sets
    y_combined = pd.concat([Xy_data['y_train'],Xy_data['y_test']], axis=0).reset_index(drop=True)

    # plot histogram, quartile lines and the points in each quartile
    if params_dict['type'].lower() == 'reg':
        y_dist_dict,ax = plot_quartiles(y_combined,ax)
    
    # plot a bar plot with the count of each y type
    elif params_dict['type'].lower() == 'clas':
        class_mapping_reverse = params_dict.get('class_mapping_reverse')
        if class_mapping_reverse:
            class_mapping_reverse = _parse_class_mapping_reverse(class_mapping_reverse)
        y_dist_dict,ax = plot_y_count(y_combined,ax,class_mapping_reverse)

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
    plt.close()
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
        n_classes = len(y_dist_dict['count_labels'])
        ideal_population = round(100 / n_classes, 1) if n_classes else 0
        print_distrib += f"\n      Ideally, the number of datapoints in each prediction class should be uniform ({ideal_population}% population per class) to have similar reliability in the predictions across classes"
        distrib_counts = list(y_dist_dict['count_labels'])
        class_counts_txt = ', '.join(
            f"{label}: {count}"
            for label, count in zip(y_dist_dict['type_labels'], y_dist_dict['count_labels'])
        )
        print_distrib += f"\n      - The number of points in each class is {class_counts_txt}"
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


def plot_y_count(y_combined,ax,class_mapping_reverse=None):
    '''
    Plot a bar plot with the count of each y type.
    '''

    # get the number of times that each y type is included
    labels_used = sorted(pd.unique(y_combined))
    type_labels,count_labels = [],[]
    for label in labels_used:
        if class_mapping_reverse:
            type_labels.append(class_mapping_reverse.get(int(label), label))
        else:
            type_labels.append(label)
        count_labels.append(len(y_combined[y_combined == label]))

    x_labels = list(range(len(type_labels)))
    _ = ax.bar(x_labels, count_labels, tick_label=type_labels,
                color='#1f77b4', edgecolor='k', linewidth=1, alpha=1,
                width=0.4)

    y_dist_dict = {'type_labels': type_labels,
                   'count_labels': count_labels}

    return y_dist_dict,ax


def get_prediction_results(model_data,y,y_pred_all):
    '''
    Calculate metrics based on y and y_pred
    '''

    # an empty set (e.g. no internal test set when --csv_test is used with --auto_test False)
    # has no metrics to compute - sklearn's metric functions reject 0-sample arrays outright
    if len(y) == 0:
        return float('nan'), float('nan'), float('nan')

    if model_data['type'].lower() == 'reg':
        mae = mean_absolute_error(y,y_pred_all)
        rmse = np.sqrt(mean_squared_error(y,y_pred_all))
        if len(np.unique(y)) > 1 and len(np.unique(y_pred_all)) > 1:
            res = stats.linregress(y,y_pred_all)
            r2 = res.rvalue**2
        else:
            r2 = 0.0

        return r2, mae, rmse

    elif model_data['type'].lower() == 'clas':
        # ensure true and predicted labels are integers
        acc = accuracy_score(y,np.round(y_pred_all).astype(int))
        # F1 by default uses average='binnary', to deal with predictions with more than 2 ouput values we use average='micro'
        # if len(set(y))==2:
        try:
            f1_score_val = f1_score(y,np.round(y_pred_all).astype(int))
        except ValueError:
            f1_score_val = f1_score(y,np.round(y_pred_all).astype(int),average='micro')
        mcc = matthews_corrcoef(y,np.round(y_pred_all).astype(int))
        return acc, f1_score_val, mcc


def get_error_labels(model_type):
    """
    Returns the three error metric labels for the given model type.
    
    Parameters
    ----------
    model_type : str
        The type of model: 'reg' for regression or 'clas' for classification
        
    Returns
    -------
    tuple of str
        Three error labels appropriate for the model type:
        - Regression: ('r2', 'mae', 'rmse')
        - Classification: ('acc', 'f1', 'mcc')
    """
    error_labels = {
        'reg': ('r2', 'mae', 'rmse'),
        'clas': ('acc', 'f1', 'mcc')
    }
    
    model_type_lower = model_type.lower()
    
    return error_labels[model_type_lower]


def load_db_n_params(self,params_dir,suffix,suffix_title,module,print_load):
    '''
    Loads the parameters and Xy databases from a folder, add scaled X data and print information
    about the databases
    '''

    # load databases from CSV
    csv_df,csv_X,csv_y,model_data,_ = load_dfs(self,params_dir,module,print_info=print_load)

    # detect points in the test set
    test_points = csv_X[csv_X['Set'] == 'Test'].index.tolist()
    csv_X = csv_X.drop(columns=['Set'])

    # keep only the descriptors used in the model
    csv_X = csv_X[model_data['X_descriptors']]

    # load and adjust external set (if any)
    csv_external_df, csv_X_external,csv_y_external = None,None,None
    if self.args.csv_test != '':
        csv_external_df,csv_X_external,csv_y_external = load_database(self,self.args.csv_test,'predict',external_test=True)
        if (
            model_data['type'].lower() == 'clas'
            and csv_y_external is not None
            and 'class_mapping_reverse' in model_data
        ):
            class_mapping_reverse = _parse_class_mapping_reverse(model_data['class_mapping_reverse'])
            if class_mapping_reverse:
                class_mapping = {str(label): code for code, label in class_mapping_reverse.items()}
                csv_y_external = csv_y_external.map(
                    lambda value: class_mapping.get(str(value), value)
                ).astype(int)
        try:
            csv_X_external = csv_X_external[model_data['X_descriptors']]
        except KeyError:
            # this might fail if the initial categorical variables have not been transformed
            try:
                self.args.log.write(f"\n   x  There are missing descriptors in the test set! Looking for categorical variables converted from CURATE")
                csv_X_external = categorical_transform(self,csv_X_external,'predict')
                csv_X_external = csv_X_external[model_data['X_descriptors']]
                self.args.log.write(f"   o  The missing descriptors were successfully created")
            except KeyError:
                self.args.log.write(f"   x  There are still missing descriptors in the test set! The following descriptors are needed: {model_data['X_descriptors']}")
                self.args.log.finalize()
                sys.exit()

    # split tests
    Xy_data = prepare_sets(self,csv_df,csv_X,csv_y,test_points,model_data['names'],csv_external_df,csv_X_external,csv_y_external,BO_opt=False)

    # print information of loaded database
    params_name = os.path.basename(params_dir)
    if print_load:
        _ = load_print(self,params_name,suffix,model_data,Xy_data)

    return Xy_data, model_data, suffix_title


def prepare_sets(self,csv_df,csv_X,csv_y,test_points,column_names,csv_external_df,csv_X_external,csv_y_external,BO_opt=False):
    '''
    Standardizes and separate test set
    '''

    # test_points stays None when csv_test is set and auto_test doesn't force a test_set (e.g.
    # --csv_test with --auto_test False); Xy_split() expects a list, an empty one meaning "no
    # test points carved out of the training data"
    if test_points is None:
        test_points = []

    X_scaled_df,X_scaled_external_df = scale_df(csv_X,csv_X_external)

    # separate test set and save it in the Xy data
    if BO_opt:
        # --csv_test is for getting predictions on new/external points, which may or may not
        # have known y values (e.g. compounds not yet made) - it is NOT a substitute for the
        # internal test split the score is calibrated on, so it must not suppress it. ROBERT
        # still carves its own internal test set as usual; the external file is used
        # separately, on top of that, wherever csv_test is actually consumed
        if self.args.auto_test:
            # test_set == 0 is treated as a deliberate "no test set" choice (train on 100% of
            # the data as train+valid), not a value that needs the safety net - only small,
            # possibly-accidental nonzero values get raised to 0.2 (--auto_test False bypasses
            # this entirely, but explicit 0 no longer needs it just for this specific case)
            if 0 < self.args.test_set < 0.2:
                self.args.test_set = 0.2
                self.args.log.write(f'\nx  WARNING! The test_set option was set to {self.args.test_set}, this value will be raised to 0.2 to include a meaningful amount of points in the test set. You can bypass this option and include less test points with "--auto_test False".')

        if self.args.test_set > 0:
            self.args.log.write(f'\no  Before hyperoptimization, {int(self.args.test_set*100)}% of the data (or 4 points at least) was separated as test set, using an even distribution of data points across the range of y values.')
            try:
                test_points = test_select(self,X_scaled_df,csv_y)
            except TypeError:
                self.args.log.write(f'   x The data split process failed! This is probably due to using strings/words as values (use --curate to curate the data first)')
                sys.exit()

    # load predefined sets and save the info in Xy data
    Xy_data = Xy_split(csv_df,csv_X,X_scaled_df,csv_y,csv_external_df,csv_X_external,X_scaled_external_df,csv_y_external,test_points,column_names)

    # also store the descriptors used (the labels disappear after test_select() )
    Xy_data['X_descriptors'] = csv_X.columns.tolist()

    return Xy_data


def load_dfs(self,folder_model,module,sanity_check=False,print_info=True):
    '''
    Loads the parameters and Xy databases from the GENERATE folder as dataframes
    '''
    
    if os.getcwd() in f"{folder_model}":
        path_db = folder_model
    else:
        path_db = f"{Path(os.getcwd()).joinpath(folder_model)}"
    if os.path.exists(path_db):
        csv_files = glob.glob(f'{Path(path_db).joinpath("*.csv")}')
        csv_files.sort(key=lambda f: f.endswith('_db.csv')) # sort the database file to be the last one, depending on the OS was taking first the dabatase and then the parameters
        for csv_file in csv_files:
            if csv_file.endswith('_db.csv'):
                if not sanity_check:
                    csv_df,csv_X,csv_y = load_database(self,csv_file,module,print_info=print_info)
                csv_name = csv_file
            else:
                csv_df,csv_X,csv_y = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
                # convert df to dict, then adjust params to a valid format
                model_data = load_params(self,csv_file)
    else:
        self.args.log.write(f"\nx  The folder with the model and database ({path_db}) does not exist! Did you use the destination=PATH option in the other modules?")
        sys.exit()

    return csv_df,csv_X,csv_y,model_data,csv_name


def load_params(self,path_csv):
    '''
    Load parameters from a CSV and adjust the format
    '''
    
    PFI_df = pd.read_csv(path_csv, encoding='utf-8')
    PFI_dict = pd_to_dict(PFI_df)
    PFI_dict = dict_formating(PFI_dict)
    PFI_dict['params'] = model_adjust_params(self, PFI_dict['model'], PFI_dict['params'])

    return PFI_dict


def load_print(self,params_name,suffix,model_data,Xy_data):
    '''
    Print information of the database loaded and type of model used
    '''

    if '.csv' in params_name:
        params_name = params_name.split('.csv')[0]
    txt_load = f'\no  ML model {params_name} {suffix} and Xy database were loaded, including:'
    txt_load += f'\n   - Target value: {model_data["y"]}'
    txt_load += f'\n   - Names: {model_data["names"]}'
    txt_load += f'\n   - Model: {model_data["model"]}'
    txt_load += f'\n   - k-fold CV: {model_data["kfold"]}'
    txt_load += f'\n   - Repetitions CV: {model_data["repeat_kfolds"]}'
    txt_load += f'\n   - Descriptors: {model_data["X_descriptors"]}'
    txt_load += f'\n   - Training points: {len(Xy_data["y_train"])}'
    txt_load += f'\n   - Test points: {len(Xy_data["y_test"])}'
    if 'X_external' in Xy_data:
        txt_load += f'\n   - External test points: {len(Xy_data["X_external"])}'
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


def pearson_map(self,csv_df_pearson,module,params_dir=None,suffix_title=None):
    '''
    Creates Pearson heatmap
    '''

    importlib.reload(plt) # needed to avoid threading issues
    if module.lower() == 'curate': # only represent the final descriptors in CURATE
        csv_df_pearson = csv_df_pearson.drop([self.args.y] + self.args.ignore, axis=1)

    # numeric_only=True: in EVALUATE mode, CURATE's categorical_transform() is skipped (only
    # this heatmap is drawn), so csv_df_pearson can still contain raw non-numeric descriptor
    # columns - .corr() raises ValueError outright on those instead of just excluding them
    corr_matrix = csv_df_pearson.corr(numeric_only=True)
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)]= True

    # with only the lower triangle unmasked, the first row (only cell = the diagonal, itself
    # masked) and the last column (would only be unmasked by a row below the last one, which
    # doesn't exist) are always entirely blank. bbox_inches='tight' does not crop them away
    # (they are still valid, if invisible, axes content), which left a large empty gap above
    # the heatmap - most noticeable with very few descriptors, where it could leave just a
    # single visible cell. Trimming them keeps every remaining cell meaningful. This is only
    # for the plot - the function still returns the full, untrimmed corr_matrix, since callers
    # (e.g. pearson_map_predict()'s correlation-warning scan) index it by original position and
    # trimming would silently shift some pairs onto the (skipped) diagonal
    if len(corr_matrix.columns) > 1:
        corr_matrix_plot = corr_matrix.iloc[1:, :-1]
        mask_plot = mask[1:, :-1]
    else:
        corr_matrix_plot = corr_matrix
        mask_plot = mask

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

        _ = sb.heatmap(corr_matrix_plot,
                        mask = mask_plot,
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
        # Explicitly align tick positions with the full correlation matrix.
        # Newer matplotlib/seaborn combinations can auto-reduce the locator
        # ticks, which then makes set_*ticklabels fail if all labels are passed.
        tick_positions_y = np.arange(len(corr_matrix_plot.index)) + 0.5
        tick_positions_x = np.arange(len(corr_matrix_plot.columns)) + 0.5
        ax.set_yticks(tick_positions_y)
        ax.set_xticks(tick_positions_x)
        ax.set_yticklabels(corr_matrix_plot.index, rotation=0)
        ax.set_xticklabels(corr_matrix_plot.columns)

        title_fig = 'Pearson\'s r heatmap'
        if module.lower() == 'predict':
            # prefer the caller-provided suffix_title (always correct, including per-model
            # folders like GENERATE/All_models/No_PFI/RF, whose basename is the model name,
            # not "No_PFI"/"PFI") - fall back to guessing from the folder name only if it
            # wasn't given
            if suffix_title is None:
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
        plt.close()

        path_reduced = '/'.join(f'{heatmap_path}'.replace('\\','/').split('/')[-2:])
        if module.lower() == 'curate':
            self.args.log.write(f'\no  The Pearson heatmap was stored in {path_reduced}.')
        elif module.lower() == 'predict':
            self.args.log.write(f'\n   o  The Pearson heatmap was stored in {path_reduced}.')

    return corr_matrix


def plot_metrics(model_data,suffix_title,verify_metrics,verify_results,show_title=True):
    '''
    Creates a plot with the results of the flawed models in VERIFY.
    show_title=False skips the in-image title - used for the small (270px-wide) Section B
    thumbnail, where the container height crops it anyway (see print_img_tag()) and the
    surrounding item text already says what the plot shows
    '''

    importlib.reload(plt) # needed to avoid threading issues
    sb.reset_defaults()
    sb.set(style="ticks")
    _, ax = plt.subplots(figsize=(7.45,6))

    # define names
    csv_name = os.path.basename(model_data['model']).split('_db.csv')[0]
    base_csv_name = f'VERIFY/{csv_name}'
    base_csv_path = f"{Path(os.getcwd()).joinpath(base_csv_name)}"
    path_n_suffix = f'{base_csv_path}_{suffix_title}'

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

    # adjust number of significative numbers shown
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    width_bar = 0.55
    label_count = 0
    for test_metric,test_name,test_color in zip(verify_metrics['metrics'],verify_metrics['test_names'],verify_metrics['colors']):
        rects = ax.bar(test_name, test_metric, label=test_name, 
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
            elif test_color == '#a9a9a9':
                txt_bar = 'N/A'
            ax.text(label_count, offset_txt, txt_bar, color=test_color, 
                    fontstyle='italic', horizontalalignment='center')
        label_count += 1

    # Set tick sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # title and labels of the axis
    plt.ylabel(f'{verify_results["error_type"].upper()}', fontsize=14)

    if show_title:
        plt.text(0.5, 1.08, f'VERIFY tests of {os.path.basename(path_n_suffix)}', horizontalalignment='center',
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
    plt.figlegend([thres,arrow], [f'Limits: {thres_line:.2} (pass), {unclear_thres_line:.2} (unclear)','Pass test'], handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
                    loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, fontsize=14)

    # Add gridlines
    ax.grid(linestyle='--', linewidth=1)

    # save plot
    verify_plot_file = f'{os.path.dirname(path_n_suffix)}/VERIFY_tests_{os.path.basename(path_n_suffix)}.png'
    plt.savefig(verify_plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    path_reduced = '/'.join(f'{verify_plot_file}'.replace('\\','/').split('/')[-2:])
    print_ver = f"\n   o  VERIFY plot saved in {path_reduced}"

    return print_ver


def dict_formating(dict_csv):
    '''
    Adapt format of dictionaries that come from dataframes loaded from CSV
    '''
    
    import json

    if 'X_descriptors' in dict_csv:
        # Try JSON first (new format), fall back to ast.literal_eval (old format)
        try:
            dict_csv['X_descriptors'] = json.loads(dict_csv['X_descriptors'])
        except (json.JSONDecodeError, TypeError):
            dict_csv['X_descriptors'] = ast.literal_eval(dict_csv['X_descriptors'])
            
    if 'params' in dict_csv:
        # Try JSON first (new format), fall back to ast.literal_eval (old format)
        try:
            dict_csv['params'] = json.loads(dict_csv['params'])
        except (json.JSONDecodeError, TypeError):
            dict_csv['params'] = ast.literal_eval(dict_csv['params'])

    if 'class_mapping_reverse' in dict_csv:
        dict_csv['class_mapping_reverse'] = _parse_class_mapping_reverse(
            dict_csv['class_mapping_reverse']
        )

    return dict_csv
