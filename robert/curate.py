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
    names : str, default=''
        Column of the names for each datapoint. Names are used to print outliers.
    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    categorical : str, default='onehot'
        Mode to convert data from columns with categorical variables. As an example, a variable containing 4
        types of C atoms (i.e. primary, secondary, tertiary, quaternary) will be converted into categorical
        variables. Options: 
        1. 'onehot' (for one-hot encoding, ROBERT will create a descriptor for each type of
        C atom using 0s and 1s to indicate whether the C type is present)
        2. 'numbers' (to describe the C atoms with numbers: 1, 2, 3, 4).
    corr_filter : bool, default=True
        Activate the correlation filters of descriptors. Two filters will be performed based on the correlation
        of the descriptors with other descriptors (x filter) and the y values (y filter).
    desc_thres : float, default=25
        Threshold for the descriptor-to-datapoints ratio to loose the correlation filter. By default,
        the correlation filter is loosen if there are 25 times more datapoints than descriptors.
    thres_x : float, default=0.7
        Thresolhold to discard descriptors based on high R**2 correlation with other descriptors (i.e. 
        if thres_x=0.7, variables that show R**2 > 0.7 will be discarded).
    thres_y : float, default=0.001
        Thresolhold to discard descriptors with poor correlation with the y values based on R**2 (i.e.
        if thres_y=0.001, variables that show R**2 < 0.001 will be discarded).
    seed : int, default=0
        Random seed used in RFECV feature selector and other protocols.
    kfold : int, default=5
        Number of random data splits for the cross-validation of the RFECV feature selector. 
    repeat_kfolds : int, default=10
        Number of repetitions for the k-fold cross-validation of the RFECV feature selector.
    auto_type : bool, default=True
        If there are only two y values, the program automatically changes the type of problem to classification.


"""
#####################################################.
#         This file stores the CURATE class         #
#               used in data curation               #
#####################################################.

import time
import os
import pandas as pd
from robert.utils import (load_variables, finish_print, load_database, pearson_map,
                          check_clas_problem, categorical_transform, correlation_filter)


class curate:
    """
    Class containing all the functions from the CURATE module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the CURATE module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "curate")

        # load database, discard user-defined descriptors and perform data checks
        csv_df,_,_ = load_database(self,self.args.csv_name,"curate")

        # adjust options of classification problems and detects whether the right type of problem was used
        self = check_clas_problem(self,csv_df)

        if not self.args.evaluate:
            # transform categorical descriptors
            csv_df = categorical_transform(self,csv_df,'curate')

            # apply duplicate filters (i.e., duplication of datapoints or descriptors)
            csv_df = self.dup_filter(csv_df)

            # apply the correlation filters and returns the database without correlated descriptors
            if self.args.corr_filter:
                csv_df = correlation_filter(self,csv_df)

        # create Pearson heatmap
        _ = pearson_map(self,csv_df,'curate')

        # save the curated CSV
        _ = self.save_curate(csv_df)

        # finish the printing of the CURATE info file
        _ = finish_print(self,start_time,'CURATE')


    def dup_filter(self,csv_df_dup):
        '''
        Removes duplicated datapoints and descriptors
        '''

        txt_dup = f'\no  Duplication filters activated'
        txt_dup += f'\n   Excluded datapoints:'

        # remove duplicated entries
        datapoint_drop = []
        for i,datapoint in enumerate(csv_df_dup.duplicated()):
            if datapoint:
                datapoint_drop.append(i)
        for datapoint in datapoint_drop:
            txt_dup += f'\n   - Datapoint number {datapoint}'

        if len(datapoint_drop) == 0:
            txt_dup += f'\n   -  No datapoints were removed'

        csv_df_dup = csv_df_dup.drop(datapoint_drop, axis=0)

        csv_df_dup.reset_index(drop=True)
        self.args.log.write(txt_dup)

        return csv_df_dup


    def save_curate(self,csv_df):
        '''
        Saves the curated database and options used in CURATE
        '''
        
        # saves curated database
        csv_basename = os.path.basename(f'{self.args.csv_name}').split('.')[0]
        csv_curate_name = f'{csv_basename}_CURATE.csv'
        csv_curate_name = self.args.destination.joinpath(csv_curate_name)
        _ = csv_df.to_csv(f'{csv_curate_name}', index = None, header=True)
        path_reduced = '/'.join(f'{csv_curate_name}'.replace('\\','/').split('/')[-2:])
        self.args.log.write(f'\no  The curated database was stored in {path_reduced}.')

        # saves important options used in CURATE
        options_name = f'CURATE_options.csv'
        options_name = self.args.destination.joinpath(options_name)
        options_df = pd.DataFrame()
        options_df['y'] = [self.args.y]
        options_df['ignore'] = [self.args.ignore]
        options_df['names'] = [self.args.names]
        options_df['csv_name'] = [csv_curate_name]
        _ = options_df.to_csv(f'{options_name}', index = None, header=True)
