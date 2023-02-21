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
     thres_x : float, default=0.85
         Thresolhold to discard descriptors based on high R**2 correlation with other descriptors (i.e. 
         if thres_x=0.85, variables that show R**2 > 0.85 will be discarded).
     thres_y : float, default=0.02
         Thresolhold to discard descriptors with poor correlation with the y values based on R**2 (i.e.
         if thres_y=0.02, variables that show R**2 < 0.02 will be discarded).
     destination : str, default=None,
         Directory to create the output file(s).
     varfile : str, default=None
         Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  


"""
#####################################################.
#         This file stores the CURATE class         #
#               used in data curation               #
#####################################################.

import time
import pandas as pd
from scipy import stats
from robert.utils import load_variables, finish_print, load_database


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
        csv_df = load_database(self,self.args.csv_name,"curate")

        # transform categorical descriptors
        csv_df = self.categorical_transform(csv_df)

        # applies the correlation filters and returns the database without correlated descriptors
        if self.args.corr_filter:
            csv_df = self.correlation_filter(csv_df)

        # saves the curated CSV
        csv_curate_name = f'{self.args.csv_name.split(".")[0]}_CURATE.csv'
        csv_curate_name = self.destination.joinpath(csv_curate_name)
        _ = csv_df.to_csv(f'{csv_curate_name}', index = None, header=True)
        self.args.log.write(f'\no  The curated database was stored in {csv_curate_name}.')

        # finish the printing of the CURATE info file
        _ = finish_print(self,start_time,'CURATE')


    def categorical_transform(self,csv_df):
        # converts all columns with strings into categorical values (one hot encoding
        # by default, can be set to numerical 1,2,3... with categorical = True).
        # Troubleshooting! For one-hot encoding, don't use variable names that are
        # also column headers! i.e. DESCRIPTOR "C_atom" contain C2 as a value,
        # but C2 is already a header of a different column in the database. Same applies
        # for multiple columns containing the same variable names.

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

        if len(categorical_vars) == 0:
            txt_categor += f'\n   - No categorical variables were found.'
        else:
            if self.args.categorical == 'numbers':
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


    def correlation_filter(self, csv_df):
        """
        Discards a) correlated variables and b) variables that do not correlate with the y values, based
        on R**2 values.
        """

        txt_corr = f'\no  Correlation filter activated with these thresholds: thres_x = {self.args.thres_x}, thres_y = {self.args.thres_y}'
        txt_corr += f'\n   Excluded descriptors:'

        descriptors_drop = []
        for i,column in enumerate(csv_df.columns):
            if column not in descriptors_drop and column not in self.args.ignore and column != self.args.y:
                # finds the descriptors with low correlation to the response values
                _, _, r_value_y, _, _ = stats.linregress(csv_df[column],csv_df[self.args.y])
                rsquared_y = r_value_y**2
                if rsquared_y < self.args.thres_y:
                    descriptors_drop.append(column)
                    txt_corr += f'\n   - {column}: R**2 = {round(rsquared_y,2)} with the {self.args.y} values'
                # finds correlated descriptors
                if column != csv_df.columns[-1]:
                    for j,column2 in enumerate(csv_df.columns):
                        if j > i and column2 and column2 not in self.args.ignore not in descriptors_drop and column2 != self.args.y:
                            _, _, r_value_x, _, _ = stats.linregress(csv_df[column],csv_df[column2])
                            rsquared_x = r_value_x**2
                            if rsquared_x > self.args.thres_x:
                                # discard the column with less correlation with the y values
                                _, _, r_value_y2, _, _ = stats.linregress(csv_df[column2],csv_df[self.args.y])
                                rsquared_y2 = r_value_y2**2
                                if rsquared_y >= rsquared_y2:
                                    descriptors_drop.append(column2)
                                    txt_corr += f'\n   - {column2}: R**2 = {round(rsquared_x,2)} with {column}'
                                else:
                                    descriptors_drop.append(column)
                                    txt_corr += f'\n   - {column}: R**2 = {round(rsquared_x,2)} with {column2}'
        
        if len(descriptors_drop) == 0:
            txt_corr += f'\no  No descriptors were removed'
    
        self.args.log.write(txt_corr)

        # drop descriptors that did not pass the filters
        csv_df_filtered = csv_df.drop(descriptors_drop, axis=1)
        txt_csv = f'\no  {len(csv_df_filtered.columns)} descriptors remaining after applying correlation filters:\n'
        txt_csv += '\n'.join(f'   - {var}' for var in csv_df_filtered.columns)
        self.args.log.write(txt_csv)

        return csv_df_filtered
