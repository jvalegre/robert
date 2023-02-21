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


class aqme:
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





por default, genera fingerprints a partir de SMILES y añade descriptores
elimina columnas de 0s y 1s
guarda que significan cada fp