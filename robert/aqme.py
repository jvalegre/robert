"""
Parameters
----------

    csv_name : str, default=''
        Name of the CSV file containing the database with SMILES and code_name columns. A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv'). 
    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    y : str, default=''
        Name of the column containing the response variable in the input CSV file (i.e. 'solubility'). 
    qdescp_keywords : str, default=''    
        Add extra keywords to the AQME-QDESCP run (i.e. qdescp_keywords="--qdescp_atoms ['Ir']")
    descp_lvl : str, default='interpret'
        Type of descriptor to be used in the AQME-ROBERT workflow. Options are 'interpret', 'denovo' or 'full'.

"""
#####################################################.
#         This file stores the AQME class           #
#     used to perform the AQME-ROBERT workflow      #
#####################################################.

import os
import glob
import subprocess
import time
import shutil
import sys
from pathlib import Path
import pandas as pd
from robert.utils import (load_variables,
    finish_print,
    load_database
    )

# list of potential arguments from CSV inputs in AQME
aqme_args = ['charge','mult','complex_type','geom','constraints_atoms','constraints_dist','constraints_angle','constraints_dihedral','sample']

class aqme:
    """
    Class containing all the functions from the AQME module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the AQME module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "aqme")

        # check if AQME is installed (required for this module)
        _ = self.init_aqme()

        # run an AQME workflow, which includes CSEARCH and QDESCP jobs
        self = self.run_csearch_qdescp(self.args.csv_name)

        # run an AQME workflow for the test set (if any)
        if self.args.csv_test != '':
            _ = self.run_csearch_qdescp(self.args.csv_test,aqme_test=True)

        # move AQME output files (remove previous runs as well)
        _ = move_aqme()

        # finish the printing of the AQME info file
        _ = finish_print(self,start_time,'AQME')


    def run_csearch_qdescp(self,csv_target,aqme_test=False):
        '''
        Runs CSEARCH and QDESCP jobs in AQME
        '''
        
        # load database just to perform data checks (i.e. no need to run AQME if the specified y is not
        # in the database, since the program would crush in the subsequent CURATE job)
        job_type = 'aqme'
        if aqme_test:
            job_type = 'aqme_test'

        # move the SDF files from the csv_name run (otherwise, the xTB calcs are repeated in csv_test)
        else:
            path_sdf = Path(f'{os.getcwd()}/CSEARCH/sdf_temp')
            if os.path.exists(path_sdf):
                shutil.rmtree(path_sdf)
            path_sdf.mkdir(exist_ok=True, parents=True)
            for sdf_file in glob.glob(f'{os.getcwd()}/CSEARCH/*.sdf'):
                new_sdf = path_sdf.joinpath(os.path.basename(sdf_file))
                if os.path.exists(new_sdf):
                    os.remove(new_sdf)
                shutil.move(sdf_file, new_sdf)
                      
        # Load database
        csv_df,_,_ = load_database(self,csv_target,job_type,print_info=False)

         # avoid running calcs with special signs (i.e. *)
        for name_csv_indiv in csv_df['code_name']:
            if '*' in f'{name_csv_indiv}':
                self.args.log.write(f"\nx  WARNING! The names provided in the CSV contain * (i.e. {name_csv_indiv}). Please, remove all the * characters.")
                self.args.log.finalize()
                sys.exit()

        # find if there is more than one SMILES column in the CSV file
        for column in csv_df.columns:
            if "SMILES" == column.upper() or "SMILES_" in column.upper():
                
                self.args.ignore.append(column)

                # create individual csv file for each SMILES column
                csv_temp = csv_df[['code_name', column] + [col for col in csv_df.columns if col.lower() in aqme_args]]
                csv_temp.columns = ['code_name', 'SMILES'] + [col for col in csv_temp.columns if col.lower() in aqme_args]
                
                if column.upper() == "SMILES":
                    smi_suffix = None
                    csv_temp.to_csv('AQME_indiv.csv', index=False)
                    aqme_indv_name = 'AQME_indiv'
                else:
                    smi_suffix = column.split("_")[1]
                    csv_temp['code_name'] = csv_temp['code_name'] + '_' + smi_suffix
                    csv_temp.to_csv(f'AQME_indiv_{smi_suffix}.csv', index=False)
                    aqme_indv_name = f'AQME_indiv_{smi_suffix}'

                # run AQME-QDESCP to generate descriptors
                cmd_qdescp = ['python','-u', '-m', 'aqme', '--qdescp', '--input', f'{aqme_indv_name}.csv', '--program', 'xtb', '--csv_name', f'{aqme_indv_name}.csv', '--nprocs', f'{self.args.nprocs}', '--robert']
                _ = self.run_aqme(cmd_qdescp, self.args.qdescp_keywords)

                if smi_suffix is not None:
                    # Change column names by adding suffix
                    try:
                        df_temp = pd.read_csv(f'AQME-ROBERT_{self.args.descp_lvl}_{aqme_indv_name}.csv', encoding='utf-8')
                    except FileNotFoundError:
                        self.args.log.write("x  WARNING! ROBERT stopped due to a problem with the AQME job. Please, check the previous AQME warnings.")
                        sys.exit()
                    df_temp.columns = [f'{col}_{smi_suffix}' if col not in ['code_name','SMILES'] and col not in aqme_args else col for col in df_temp.columns]
                    df_temp.to_csv(f'AQME-ROBERT_{self.args.descp_lvl}_{aqme_indv_name}.csv', index=False)

                    # Check if there are missing rows in the AQME-ROBERT_{aqme_indv_name}.csv
                    if len(df_temp) < len(csv_temp):
                        missing_rows = csv_temp.loc[~csv_temp['code_name'].isin(df_temp['code_name'])]
                        missing_rows[['code_name', 'SMILES']].to_csv(f'AQME-ROBERT_{self.args.descp_lvl}_{aqme_indv_name}.csv', mode='a', header=False, index=False)

                    # Get the order of code_name in aqme_indv_name
                    order = csv_temp['code_name'].tolist()

                    # Sort the rows in 'AQME-ROBERT_{aqme_indv_name}.csv' based on the order
                    df_temp = pd.read_csv(f'AQME-ROBERT_{self.args.descp_lvl}_{aqme_indv_name}.csv', encoding='utf-8')
                    df_temp = df_temp.sort_values(by='code_name', key=lambda x: x.map({v: i for i, v in enumerate(order)}))

                    # Fill missing values with corresponding SMILES row
                    df_temp = df_temp.fillna(df_temp.groupby('SMILES').transform('first'))

                    df_temp.to_csv(f'AQME-ROBERT_{self.args.descp_lvl}_{aqme_indv_name}.csv', index=False)

                # return SDF files after csv_test
                if aqme_test:
                    for sdf_file in glob.glob(f'{path_sdf}/*.sdf'):
                        new_sdf = Path(f'{os.getcwd()}/CSEARCH').joinpath(os.path.basename(sdf_file))
                        shutil.move(sdf_file, new_sdf)
                        shutil.rmtree(path_sdf)

        # if AQME-ROBERT_AQME_indiv_n.csv >0 in folder:
        if len(glob.glob(f'AQME-ROBERT_{self.args.descp_lvl}_AQME_indiv*.csv')) > 0:

            df_concat = pd.DataFrame()

            # Read and concatenate CSV files 
            for file in sorted(glob.glob(f'AQME-ROBERT_{self.args.descp_lvl}_AQME_indiv*.csv'), key=os.path.getmtime,reverse=True):
                columns_to_drop = ['code_name', 'SMILES'] + aqme_args
                df_temp = pd.read_csv(file, encoding='utf-8')
                columns_to_drop = [col for col in columns_to_drop if col in df_temp.columns]
                df_temp = df_temp.drop(columns=columns_to_drop)
                df_concat = pd.concat([df_temp, df_concat], axis=1)
            df_concat = pd.concat([csv_df, df_concat], axis=1)
            df_concat.to_csv(f'AQME-ROBERT_{self.args.descp_lvl}_{csv_target}', index=False)


        # if no qdesc_atom is set, only keep molecular properties and discard atomic properties
        aqme_db = f'AQME-ROBERT_{self.args.descp_lvl}_{csv_target}'

        # ensure that the AQME database was successfully created
        if not os.path.exists(aqme_db):
            self.args.log.write(f"\nx  The initial AQME descriptor protocol did not create any CSV output!")
            sys.exit()
        
        # remove atomic properties if no SMARTS patterns were selected in qdescp
        if 'qdescp_atoms' not in self.args.qdescp_keywords:
            _ = filter_atom_prop(aqme_db,csv_df)

        # remove arguments from CSV inputs in AQME
        _ = filter_aqme_args(aqme_db)
        
        # delete AQME_indiv*.csv files
        for file in glob.glob('*QME_indiv*.csv'):
            os.remove(file)
        
        # this returns stores options just in case csv_test is included
        return self

    def run_aqme(self,command,extra_keywords):
        '''
        Function that runs the AQME jobs
        '''

        if extra_keywords != '':
            for keyword in extra_keywords.split():
                command.append(keyword)

        subprocess.run(command)


    def init_aqme(self):
        '''
        Checks whether AQME is installed
        '''
        
        try:
            from aqme.qprep import qprep
        except ModuleNotFoundError:
            self.args.log.write("x  AQME is not installed (required for the --aqme option)! The program is typically installed within 2-5 minutes (https://aqme.readthedocs.io, see the Installation section)")
            sys.exit()


def filter_atom_prop(aqme_db, csv_df):
    '''
    Function that filters off atomic properties if no atom was selected in the --qdescp_atoms option
    '''
    
    aqme_df = pd.read_csv(aqme_db, encoding='utf-8')
    for column in aqme_df.columns:
        if column == 'DBSTEP_Vbur':
            aqme_df = aqme_df.drop(column, axis=1)
        # remove lists of atomic properties (skip columns from AQME arguments)
        elif aqme_df[column].dtype == object and column.lower() not in aqme_args:
            if '[' in aqme_df[column][0] and column not in csv_df.columns:
                aqme_df = aqme_df.drop(column, axis=1)
    os.remove(aqme_db)
    _ = aqme_df.to_csv(f'{aqme_db}', index=None, header=True)


def filter_aqme_args(aqme_db):
    '''
    Function that filters off AQME arguments in CSV inputs
    '''
    
    aqme_df = pd.read_csv(aqme_db, encoding='utf-8')
    for column in aqme_df.columns:
        if column.lower() in aqme_args:
            aqme_df = aqme_df.drop(column, axis=1)
    os.remove(aqme_db)
    _ = aqme_df.to_csv(f'{aqme_db}', index = None, header=True)


def move_aqme():
    '''
    Move raw data from AQME-CSEARCH and -QDESCP runs into the AQME folder
    '''
    
    for file in glob.glob(f'*'):
        if 'CSEARCH' in file or 'QDESCP' in file:
            if os.path.exists(f'AQME/{file}'):
                if len(os.path.basename(Path(file)).split('.')) == 1:
                    shutil.rmtree(f'AQME/{file}')
                else:
                    os.remove(f'AQME/{file}')
            shutil.move(file, f'AQME/{file}')