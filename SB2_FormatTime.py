
# The following is a Python translation of a MATLAB file originally written principally by Mike Tipping
# as part of his SparseBayes software library. Initially published on GitHub on July 21st, 2015.

# SB2_FORMATTIME  Pretty output of diagnostic SPARSEBAYES time information
#
# STRING = SB2_FORMATTIME(ELAPSEDTIMEINSECONDS)
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
import math

def SB2_FormatTime(elapsedTime):
    if elapsedTime >= 3600:
        # More than an hour...
        h               = math.floor(elapsedTime/float(3600))
        m               = math.floor(np.remainder(elapsedTime,3600)/float(60))
        s               = math.floor(np.remainder(elapsedTime,60))
        
        timeString_     = '{0}h {1}m {2}s'.format(h,m,s)
    
    elif elapsedTime >= 60:
        # More than one minute (but less than an hour)
        m               = math.floor(elapsedTime/float(60))
        s               = math.floor(np.remainder(elapsedTime,60))
        
        timeString_     = '{0}m {1}s'.format(m,s)
    
    else:
        # Under a minute
        s               = elapsedTime
        
        timeString_    = '{0} secs'.format(s)
        
    return timeString_
