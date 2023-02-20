"""
Parameters
----------

General
+++++++

   files : str or list of str, default=None
     Input files. Formats accepted: XYZ, SDF, GJF, COM and PDB. Also, lists can
     be used (i.e. [FILE1.sdf, FILE2.sdf] or \*.FORMAT such as \*.sdf).  
   program : str, default=None
     Program required in the conformational refining. 
     Current options: 'xtb', 'ani'
"""
#####################################################.
#        This file stores the PREDICT class         #
#              used in the predictor                #
#####################################################.

import os
import sys
import time
from pathlib import Path
from scipy import stats
from robert.utils import load_variables


class predict:
    """
    Class containing all the functions from the PREDICT module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the PREDICT module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time_overall = time.time()
        # load default and user-specified variables
        self.args = load_variables(kwargs, "predict")






graph with trainng, valid y test si existe, optional (como en el articulo de phneols)
RMSE, MAE, R2 prints
SHAP analysis (valid)
PFI analysis (valid)
        # printing and representing the results
        print(f"\nPermutation feature importances of the descriptors in the {PFI_df['model']}_{PFI_df['train']}_PFI model (for the validation set). Only showing values that drop the original score at least by {self.args.PFI_threshold*100}%:\n")
        print('Original score = '+f'{score_model:.2f}')
        for i in range(len(PFI_values)):
            print(combined_descriptor_list[i]+': '+f'{PFI_values[i]:.2f}'+' '+u'\u00B1'+ ' ' + f'{PFI_SD[i]:.2f}')

        y_ticks = np.arange(0, len(PFI_values))
        fig, ax = plt.subplots()
        ax.barh(y_ticks, PFI_values[::-1])
        ax.set_yticklabels(combined_descriptor_list[::-1])
        ax.set_yticks(y_ticks)
        ax.set_title(model_type_PFI_fun+" permutation feature importances (PFI)")
        fig.tight_layout()
        plot = ax.set(ylabel=None, xlabel='PFI')

        plt.savefig(f'PFI/{model_type_PFI_fun}+ permutation feature importances (PFI)', dpi=600, bbox_inches='tight')

        plt.show()