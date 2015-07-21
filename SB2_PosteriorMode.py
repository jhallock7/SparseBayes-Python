
# The following is a Python translation of a MATLAB file originally written principally by Mike Tipping
# as part of his SparseBayes software library. Initially published on GitHub on July 21st, 2015.

# SB2_POSTERIORMODE  Posterior mode-finder for the SPARSEBAYES algorithm
#
# [MU, U, BETA, LIKELIHOODMODE, BADHESS] = ...
#    SB2_POSTERIORMODE(LIKELIHOOD,BASIS,TARGETS,ALPHA,MU,ITSMAX,OPTIONS)
#
# OUTPUT ARGUMENTS:
# 
#    MU                Parameter values at the mode
#    U                Cholesky factor of the covariance at the mode
#    BETA            Vector of pseudo-noise variances at the mode
#    LIKELIHOODMODE    Data likelihood at the mode
#    BADHESS            Returns true if Hessian is "bad" (becomes
#                    non-positive-definite during maximisation)
# 
# INPUT ARGUMENTS:
# 
#    LIKELIHOOD    LIKELIHOOD structure
#    BASIS        Current relevant basis matrix
#    TARGETS        N-vector with target output values
#    ALPHA        Current hyperparameters
#    MU            Current weights
#    ITSMAX        Maximum number of iterations to run
#    OPTIONS        Standard OPTIONS structure (only affects diagnostics)
# 
# NOTES:
# 
# SB2_POSTERIORMODE finds the posterior mode (with respect to the
# weights) of the likelihood function in the non-Gaussian case to
# facilitate subsequent Laplace approximation.
#
# This function is intended for internal use by SPARSEBAYES only (within
# SB2_FullStatistics).
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
from SB2_Sigmoid import SB2_Sigmoid
from SB2_Diagnostic import SB2_Diagnostic

def SB2_PosteriorMode(LIKELIHOOD, BASIS, Targets, Alpha, Mu, itsMax, OPTIONS):
    
    # TOLERANCES
    
    # Termination criterion for each gradient dimension
    
    GRADIENT_MIN    = 1e-6
    
    # Minimum fraction of the full Newton step considered
    
    STEP_MIN        = 1/float(2^8)


    [N, M]          = BASIS.shape
    A               = np.diagflat(Alpha)
    
    # NB: for historical reasons, we work in term of error here (negative
    # log-liklihood) and minimise


    # Get current model output and data error
    
    BASIS_Mu        = BASIS * Mu # Linear output
    [dataError, y]  = SB2_DataError(LIKELIHOOD, BASIS_Mu, Targets)
    
    # Add on the weight penalty
    
    regulariser     = (Alpha.T * np.power(Mu, 2))/2
    newTotalError   = dataError + regulariser
 
    badHess        = False
    errorLog       = np.zeros((itsMax,1))
    
    for iteration in range(0, itsMax):
        
        # Log the error value each iteration
        
        errorLog[iteration]    = newTotalError
        
        SB2_Diagnostic(OPTIONS,4,'PM cycle: {:d}\t error: {%:.6f}\n', iteration, errorLog[iteration])
        
        # Construct the gradient
        
        e    = (Targets-y)
        g    = BASIS.T*e - np.multiply(Alpha, Mu)
        
        # Compute the likelihood-dependent analogue of the noise precision.
        # NB: Beta now a vector.
        
        if LIKELIHOOD['InUse'] == LIKELIHOOD['Bernoulli']:
            Beta    = np.multiply(y,(1-y))
            
        if LIKELIHOOD['InUse'] == LIKELIHOOD['Poisson']:
            Beta    = y
            
        # Compute the Hessian
    
        BASIS_B    = np.multiply(BASIS, (Beta * np.ones((1,M))))
        H            = (BASIS_B.T*BASIS + A)
        # Invert Hessian via Cholesky, watching out for ill-conditioning
        pdErr = False
        try:
            U    = np.linalg.cholesky(H).T
        
        except np.linalg.linalg.LinAlgError:
            pdErr = True

        if pdErr:
            # If you see this, it's *probably* the result of a bad choice of
            # basis. e.g. a kernel with too large a "width"
            SB2_Diagnostic(OPTIONS, 1, '** Warning ** Ill-conditioned Hessian ({0})\n'.format(1/np.linalg.cond(H)))
            badHess             = True
            U                   = []
            Beta                = []
            likelihoodMode      = []
            return
    
        # Before progressing, check for termination based on the gradient norm
        if np.all(np.abs(g)<GRADIENT_MIN):
            errorLog    = errorLog[range(0,iteration)]
            SB2_Diagnostic(OPTIONS, 4, ['PM convergence (<{0}) after {1} iterations, |g| = {2}\n'.format(GRADIENT_MIN,iteration, np.max(np.abs(g)))])
            break
    
        # If all OK, compute full Newton step: H^{-1} * g
        
        DeltaMu = np.linalg.solve(   U,   np.linalg.solve(U.T, g)    )
        step = 1
    
        while step > STEP_MIN:
            # Follow gradient to get new value of parameters
            Mu_new        = Mu + step*DeltaMu
            BASIS_Mu    = BASIS*Mu_new
            
            # Compute outputs and error at new point
        
            [dataError,y]       = SB2_DataError(LIKELIHOOD,BASIS_Mu,Targets)
            regulariser         = (Alpha.T * (np.power(Mu_new,2)))/2
            newTotalError       = dataError + regulariser
            
            # Test that we haven't made things worse
        
            if newTotalError >= errorLog[iteration]:
                # If so, back off!
                step    = step/2
                SB2_Diagnostic(OPTIONS, 4,['PM error increase! Backing off to l= {0}\n'.format(step)])
            else:
                Mu    = Mu_new
                step    = 0            # this will force exit from the "while" loop
        
        
        # If we get here with non-zero "step", it means that the smallest
        # offset from the current point along the "downhill" direction did not
        # lead to a decrease in error. In other words, we must be
        # infinitesimally close to a minimum (which is OK).
    
        if step != 0:
            SB2_Diagnostic(OPTIONS, 4, 'PM stopping due to back-off limit (|g| = {0})\n'.format(np.max(np.abs(g))))
            break
    
    # Simple computation of return value of log likelihood at mode
    
    likelihoodMode    = -dataError
    
    return [Mu, U, Beta, likelihoodMode, badHess]
    
    



    ######################
    #
    # Support function
    #
    ######################


def SB2_DataError(LIKELIHOOD, BASIS_Mu,Targets):
    
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Bernoulli']:
        y    = SB2_Sigmoid(BASIS_Mu)
        
        # Handle probability zero cases
        y0    = (y==0)
        y1    = (y==1)
        
        if np.any(y0[Targets>0]) or np.any(y1[Targets<1]):
            # Error infinite when model gives zero probability in
            # contradiction to data
            e    = np.inf
            
        else:
            # Any y=0 or y=1 cases must now be accompanied by appropriate
            # output=0 or output=1 values, so can be excluded.
            e    = -(Targets[np.logical_not(y0)] * np.log(y[np.logical_not(y0)].T) + (1-Targets[np.logical_not(y1)].T).T * np.log(1-y[np.logical_not(y1)].T))[0,0]

    if LIKELIHOOD['InUse'] == LIKELIHOOD['Poisson']:
        y    = np.exp(BASIS_Mu)
        e    = -np.sum(np.multiply(Targets, BASIS_Mu) - y)
            
    return [e, y]
    