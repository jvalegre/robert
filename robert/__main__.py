#!/usr/bin/env python

###############################################.
#          __main__ file for the code         #
###############################################.

from __future__ import absolute_import

import sys
from robert import robert

# If we are running from a wheel, add the wheel to sys.path
# This allows the usage python pip-*.whl/pip install pip-*.whl

if __package__ != 'robert':
    print('ROBERT is not installed! Use: pip install robert or conda install -y -c conda-forge robert.')

if __name__ == '__main__':
    robert.main()
    sys.exit()
