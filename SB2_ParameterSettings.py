
# The following is a Python translation of a MATLAB file originally written principally by Mike Tipping
# as part of his SparseBayes software library. Initially published on GitHub on July 21st, 2015.

# SB2_PARAMETERSETTINGS  User parameter initialisation for SPARSEBAYES
#
# SETTINGS = SB2_PARAMETERSETTINGS(parameter1, value1, parameter2, value2,...)
#
# OUTPUT ARGUMENTS:
# 
#    SETTINGS    An initialisation structure to pass to SPARSEBAYES
# 
# INPUT ARGUMENTS:
# 
#    Optional number of parameter-value pairs to specify    some, all, or
#    none of the following:
# 
#    BETA        (Gaussian) noise precision (inverse variance)
#    NOISESTD    (Gaussian) noise standard deviation
#    RELEVANT    Indices of columns of basis matrix to use at start-up
#    MU (WEIGHTS)        Corresponding vector of weights to RELEVANT
#    ALPHA        Corresponding vector of hyperparameter values to RELEVANT
#
# EXAMPLE:
#
#    SETTINGS = SB2_ParameterSettings('NoiseStd',0.1)
# 
# NOTES:
#
# 1.    If no input arguments are supplied, defaults (effectively an
#        empty structure) will be returned.
#
# 2.    If both BETA and NOISESTD are specified, BETA will take
#        precedence.
#
# 3.    RELEVANT may be specified without WEIGHTS or ALPHA (these will be
#        sensibly initialised later).    
# 
# 4.    If RELEVANT is specified, WEIGHTS may be specified also without ALPHA.
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

def SB2_ParameterSettings(*args):
    
    # Ensure arguments are supplied in pairs

    if len(args) % 2 != 0:
        raise Exception('Arguments to SB2_ParameterSettings should be (property, value) pairs')

    
    # Any settings specified?
    
    numSettings    = len(args)/2
    
    ## Defaults - over-ridden later if requested

    # Two options for setting noise level (purely for convenience)
    # - if 'beta' set, 'noiseStdDev' will be over-ridden
    
    SETTINGS = {
        'BETA'          : [],
        'NOISESTD'      : [],
        
        'RELEVANT'      : [],
        'MU'            : [],
        'ALPHA'         : []  
                }
    
    ## Requested overrides

    # Parse string/variable pairs
    
    for n in range(numSettings):
        property_    = args[n*2]
        value        = args[n*2 + 1]
        
        if property_ not in SETTINGS:
            raise Exception('Unrecognised initialisation property: {0}'.format(property_))
        else:
            SETTINGS[property_] = value
            
    return SETTINGS
