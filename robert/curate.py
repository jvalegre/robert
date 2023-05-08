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
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sb
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
        csv_df = self.categorical_transform(csv_df,'curate')

        # apply duplicate filters (i.e., duplication of datapoints or descriptors)
        csv_df = self.dup_filter(csv_df)

        # apply the correlation filters and returns the database without correlated descriptors
        if self.args.corr_filter:
            csv_df = self.correlation_filter(csv_df)

        # create Pearson heatmap
        _ = self.pearson_map(csv_df)

        # save the curated CSV
        _ = self.save_curate(csv_df)

        # finish the printing of the CURATE info file
        _ = finish_print(self,start_time,'CURATE')


    def categorical_transform(self,csv_df,module):
        # converts all columns with strings into categorical values (one hot encoding
        # by default, can be set to numerical 1,2,3... with categorical = True).
        # Troubleshooting! For one-hot encoding, don't use variable names that are
        # also column headers! i.e. DESCRIPTOR "C_atom" contain C2 as a value,
        # but C2 is already a header of a different column in the database. Same applies
        # for multiple columns containing the same variable names.

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
                try:
                    _, _, r_value_y, _, _ = stats.linregress(csv_df[column],csv_df[self.args.y])
                    rsquared_y = r_value_y**2
                    if rsquared_y < self.args.thres_y:
                        descriptors_drop.append(column)
                        txt_corr += f'\n   - {column}: R**2 = {round(rsquared_y,2)} with the {self.args.y} values'
                except ValueError: # this avoids X descriptors where the majority of the values are the same
                    descriptors_drop.append(column)
                    txt_corr += f'\n   - {column}: error in R**2 with the {self.args.y} values (are all the values the same?)'

                # finds correlated descriptors
                if column != csv_df.columns[-1] and column not in descriptors_drop:
                    for j,column2 in enumerate(csv_df.columns):
                        if j > i and column2 not in self.args.ignore and column2 not in descriptors_drop and column2 != self.args.y:
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
            txt_corr += f'\n   -  No descriptors were removed'
    
        self.args.log.write(txt_corr)

        # drop descriptors that did not pass the filters
        csv_df_filtered = csv_df.drop(descriptors_drop, axis=1)
        txt_csv = f'\no  {len(csv_df_filtered.columns)} columns remaining after applying duplicate and correlation filters:\n'
        txt_csv += '\n'.join(f'   - {var}' for var in csv_df_filtered.columns)
        self.args.log.write(txt_csv)

        return csv_df_filtered


    def save_curate(self,csv_df):
        '''
        Saves the curated database and options used in CURATE
        '''
        
        # saves curated database
        csv_basename = f'{self.args.csv_name}'.split('.')[0]
        csv_curate_name = f'{csv_basename}_CURATE.csv'
        csv_curate_name = self.args.destination.joinpath(csv_curate_name)
        _ = csv_df.to_csv(f'{csv_curate_name}', index = None, header=True)
        self.args.log.write(f'\no  The curated database was stored in {csv_curate_name}.')

        # saves important options used in CURATE
        options_name = f'CURATE_options.csv'
        options_name = self.args.destination.joinpath(options_name)
        options_df = pd.DataFrame()
        options_df['y'] = [self.args.y]
        options_df['ignore'] = [self.args.ignore]
        options_df['csv_name'] = [csv_curate_name]
        _ = options_df.to_csv(f'{options_name}', index = None, header=True)


    def pearson_map(self,csv_df_pearson):
        '''
        Creates Pearson heatmap
        '''

        csv_df_pearson = csv_df_pearson.drop(self.args.ignore,axis=1)
        corr_matrix = csv_df_pearson.corr()

        mask = np.zeros_like(corr_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)]= True

        _, ax = plt.subplots(figsize=(7.45,6))

        sb.set(font_scale=1.2, style='ticks')

        # determines size of the letters inside the boxes (approx.)
        size_font = 14-2*((len(csv_df_pearson.columns)/5))

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
        plt.title(title_fig, y=1.04, fontsize = 14, fontweight="bold")
        sb.set_style({'xtick.bottom': True}, {'ytick.left': True})

        heatmap_name = 'Pearson_heatmap.png'
        heatmap_path = self.args.destination.joinpath(heatmap_name)
        plt.savefig(f'{heatmap_path}', dpi=300, bbox_inches='tight')
        plt.clf()

        self.args.log.write(f'\no  The Pearson heatmap was stored in {heatmap_path}.')
        