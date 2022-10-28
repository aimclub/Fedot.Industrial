# coding=utf-8
# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy.stats.mstats import mode
from sklearn.utils.validation import check_array


"""
This file contains the implementation of different aggregation functions to
combine the outputs of the base
classifiers to give the final decision.

References
----------
Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
John Wiley & Sons, 2004.

J. Kittler, M. Hatef, R. P. W. Duin, J. Matas, On combining classifiers, IEEE
Transactions on Pattern Analysis and Machine Intelligence 20 (1998) 226–239.
"""



