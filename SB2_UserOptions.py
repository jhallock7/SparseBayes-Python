
# The following is a Python translation of a MATLAB file originally written principally by Mike Tipping
# as part of his SparseBayes software library. Initially published on GitHub on July 21st, 2015.

# SB2_USEROPTIONS  User option specification for SPARSEBAYES
#
# OPTIONS = SB2_USEROPTIONS(parameter1, value1, parameter2, value2,...)
#
# OUTPUT ARGUMENTS:
# 
#    OPTIONS        An options structure to pass to SPARSEBAYES
# 
# INPUT ARGUMENTS:
# 
#    Optional number of parameter-value pairs to specify the following:
# 
#    ITERATIONS    Number of interations to run for.
# 
#    TIME        Time limit to run for, expressed as a space-separated 
#                string. e.g. '1.5 hours', '30 minutes', '1 second'.
# 
#    DIAGNOSTICLEVEL    Integer [0,4] or string to determine the verbosity of
#                    diagnostic output.
#                    0 or 'ZERO' or 'NONE'    No output
#                    1 or 'LOW'                Low level of output
#                    2 or 'MEDIUM'            etc...
#                    3 or 'HIGH'
#                    4 or 'ULTRA'
#
#    DIAGNOSTICFILE    Filename to write diagnostics to file instead of
#                    the default stdout.
# 
#    MONITOR        Integer number: diagnostic information is output
#                every MONITOR iterations.
# 
#    FIXEDNOISE    True/false whether the Gaussian noise is to be fixed
#                (default: false.
#
#    FREEBASIS    Indices of basis vectors considered "free" and not 
#                constrained by the Bayesian prior (e.g. the "bias").
# 
#    CALLBACK    External function to call each iteration of the algorithm
#                (string). Intended to facilitate graphical demos etc.
# 
#    CALLBACKDATA    Arbitrary additional data to pass to the CALLBACK
#                    function.
#
# EXAMPLE:
#
#    OPTIONS = SB2_UserOptions('diagnosticLevel','medium',...
#                  'monitor',25,...
#                  'diagnosticFile', 'logfile.txt');
#
# NOTES:
#
# Each option (field of OPTIONS) is given a default value in
# SB2_USEROPTIONS. Any supplied property-value pairs over-ride those
# defaults.
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

def SB2_UserOptions(*args):
    
    # Ensure arguments are supplied in pairs
    
    if len(args) % 2 != 0:
        raise Exception('Arguments to SB2_UserOptions should be (property, value) pairs')
    
    # Any options specified?
    
    numSettings = len(args)/2
    

    ###########################################################################
    
    # Set defaults
    
    OPTIONS = {}
    
    # Assume we will infer the noise in the Gaussian case
    
    OPTIONS['FIXEDNOISE'] = False
  
    # Option to allow subset of the basis (e.g. bias) to be unregularised
    
    OPTIONS['FREEBASIS'] = []
    
    # Option to set max iterations to run for
    
    OPTIONS['ITERATIONS'] = 10000

    # Option to set max time to run for
    
    OPTIONS['TIME'] = 10000  # seconds

    # Set options for monitoring and recording the algorithm's progress
    
    OPTIONS['MONITOR']              = 0
    OPTIONS['DIAGNOSTICLEVEL']      = 0
    OPTIONS['DIAGNOSTICFID']        = 1  # stdout
    OPTIONS['DIAGNOSTICFILE']       = []
    
    # Option to call a function during each iteration (to create demos etc)
     
    OPTIONS['CALLBACK']             = False
    OPTIONS['CALLBACKFUNC']         = []
    OPTIONS['CALLBACKDATA']         = {}


    ###########################################################################

    # Parse string/variable pairs
    
    for n in range(numSettings):
        property_   = args[n*2]
        value       = args[n*2 + 1]

        if property_ not in OPTIONS:
            raise Exception('Unrecognised user option: {0}'.format(property_))
        
        OPTIONS[property_] = value
        
        if property_ == 'DIAGNOSTICLEVEL':
            if type(value) is str:
                if value == 'ZERO' or value == 'NONE':
                    OPTIONS['DIAGNOSTICLEVEL'] = 0
                elif value == 'LOW':
                    OPTIONS['DIAGNOSTICLEVEL'] = 1
                elif value == 'MEDIUM':
                    OPTIONS['DIAGNOSTICLEVEL'] = 2
                elif value == 'HIGH':
                    OPTIONS['DIAGNOSTICLEVEL'] = 3
                elif value == 'ULTRA':
                    OPTIONS['DIAGNOSTICLEVEL'] = 4
                else:
                    raise Exception('Unrecognised textual diagnostic level: {0}'.format(value))
            elif type(value) is int:
                if value < 0 or value > 4:
                    raise Exception('Supplied level should be integer in [0,4], or one of ZERO/LOW/MEDIUM/HIGH/ULTRA')
        
        if property_ == 'DIAGNOSTICFILE':
            OPTIONS['DIAGNOSTICFID'] = -1  #  "It will be opened later"
        
        if property_ == 'CALLBACK':
            OPTIONS['CALLBACK'] = True
            OPTIONS['CALLBACKFUNC'] = value
            if OPTIONS['CALLBACKFUNC'] not in locals():             #UNCERTAIN ABOUT THIS
                raise Exception('Callback function {0} does not appear to exist'.format(value))
         
        if property_ == 'TIME':
            OPTIONS['TIME'] = timeInSeconds(value)
             
              
    return OPTIONS




##### Support function: parse time specification

def timeInSeconds(value_):
    
    args = value_.split()
    args[1] = args[1].upper()
    
    v = int(args[0])
    
    if args[1] == 'SECONDS' or args[1] == 'SECOND':
        pass
    
    elif args[1] == 'MINUTES' or args[1] == 'MINUTE':
        v *= 60
        
    elif args[1] == 'HOURS' or args[1] == 'HOUR':
        v *= 3600
        
    else:
        raise Exception('Badly formed time string: {0}'.format(value_))
    
    return v  # Returns time in seconds
