
# The following is a Python translation of a MATLAB file originally written principally by Mike Tipping
# as part of his SparseBayes software library. Initially published on GitHub on July 21st, 2015.

# SB2_PREPROCESSBASIS  Do basis matrix pre-processing for SPARSEBAYES
#
# [BASIS, SCALES] = SB2_PREPROCESSBASIS(BASIS)
#
# OUTPUT ARGUMENTS:
#
#    BASIS    NxM matrix of basis vectors appropriately pre-processed
#            (scaled to unit length per column) 
# 
#    SCALES    Vector of scaling factors applied
#
# 
# INPUT ARGUMENTS:
# 
#    BASIS    NxM matrix of basis vectors (one column per basis function)
#
# NOTES: 
# 
# Simply normalises the basis vectors to unit length, returning the
# original lengths so that the weights can be rescaled appropriately
# before returning to the user.
#

#
# Copyright 2009, Vector Anomaly Ltd
#
# This file is part of the SPARSEBAYES library for Matlab (V2.0).
#
# SPARSEBAYES is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# SPARSEBAYES is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with SPARSEBAYES in the accompanying file "licence.txt"; if not, write to
# the Free Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,
# MA 02110-1301 USA
#
# Contact the author: m a i l [at] m i k e t i p p i n g . c o m
#

import numpy as np

def SB2_PreProcessBasis(BASIS):
    
    [N,M]    = BASIS.shape
    
    # Compute "lengths" of basis vectors (columns of BASIS)
    
    Scales = np.power(BASIS, 2)         #Square all elements of BASIS
    Scales = np.sum(Scales, axis=0)     #Sum columns
    Scales = np.power(Scales, 0.5)      #Take sqrt
    
    # Work-around divide-by-zero inconvenience

    Scales[Scales == 0] = 1

    for m in range(M):
        BASIS[:,m] /= Scales[0,m]

    return BASIS, Scales
