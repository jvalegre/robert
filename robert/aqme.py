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
    csearch_keywords : str, default='--sample 50'
        Add extra keywords to the AQME-CSEARCH run (i.e. csearch_keywords='--sample 10'). CREST can be
        used instead of RDKit by adding "--program crest".

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
        _ = self.run_csearch_qdescp(self.args.csv_name)

        # run an AQME workflow for the test set (if any)
        if self.args.csv_test != '':
            _ = self.run_csearch_qdescp(self.args.csv_test,aqme_test=False)

        # move AQME output files (remove previous runs as well)
        _ = move_aqme()

        # finish the printing of the AQME info file
        _ = finish_print(self,start_time,'AQME')


    def run_csearch_qdescp(self,csv_target,aqme_test=True):
        '''
        Runs CSEARCH and QDESCP jobs in AQME
        '''
        
        # load database just to perform data checks (i.e. no need to run AQME if the specified y is not
        # in the database, since the program would crush in the subsequent CURATE job)
        if aqme_test:
            _ = load_database(self,csv_target,"aqme")

        # run the initial AQME-CSEARCH conformational search with RDKit (default) or CREST
        if '--program crest' not in self.args.csearch_keywords.lower():
            cmd_csearch = ['python', '-m', 'aqme', '--csearch', '--program', 'rdkit', '--input', f'{csv_target}']
        else:
            self.args.log.write(f"\no  CREST detected in the csearch_keywords option, it will be used for conformer sampling")
            cmd_csearch = ['python', '-m', 'aqme', '--csearch', '--input', f'{csv_target}']
        _ = self.run_aqme(cmd_csearch,self.args.csearch_keywords)

        # run QDESCP to generate descriptors
        cmd_qdescp = ['python', '-m', 'aqme', '--qdescp', '--files', 'CSEARCH/*.sdf', '--program', 'xtb', '--csv_name', f'{csv_target}']
        _ = self.run_aqme(cmd_qdescp,self.args.qdescp_keywords)

        # if no qdesc_atom is set, only keep molecular properties and discard atomic properties
        aqme_db = f'AQME-ROBERT_{csv_target}'

        # ensure that the AQME database was successfully created
        if not os.path.exists(aqme_db):
            self.args.log.write(f"\nx  The initial AQME descriptor protocol did not create any CSV output!")
            sys.exit()

        if 'qdescp_atoms' not in self.args.qdescp_keywords:
            _ = filter_atom_prop(aqme_db)


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
            self.args.log.write("x  AQME is not installed (required for the --aqme option)! You can install the program with 'conda install -c conda-forge aqme'")
            sys.exit()


def filter_atom_prop(aqme_db):
    '''
    Function that filters off atomic properties if no atom was selected in the --qdescp_atoms option
    '''
    
    aqme_df = pd.read_csv(aqme_db)
    for column in aqme_df.columns:
        if column == 'DBSTEP_Vbur':
            aqme_df = aqme_df.drop(column, axis=1)
        # remove lists of atomic properties (skip columns from AQME arguments)
        elif aqme_df[column].dtype == object and column.lower() not in ['smiles','code_name','charge','mult','complex_type','geom','constraints_atoms','constraints_dist','constraints_angle','constraints_dihedral']:
            if '[' in aqme_df[column][0]:
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