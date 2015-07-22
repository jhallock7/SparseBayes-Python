
# The following is a Python translation of a MATLAB file originally written principally by Mike Tipping
# as part of his SparseBayes software library. Initially published on GitHub on July 21st, 2015.

# SPARSEBAYESDEMO  Simple demonstration of the SPARSEBAYES algorithm
#
#    SPARSEBAYESDEMO(LIKELIHOOD, DIMENSION, NOISETOSIGNAL)
#
# OUTPUT ARGUMENTS: None
# 
# INPUT ARGUMENTS:
# 
#    LIKELIHOOD        Text string, one of 'Gaussian' or 'Bernoulli'
#    DIMENSION        Integer, 1 or 2
#    NOISETOSIGNAL    An optional positive number to specify the
#                    noise-to-signal (standard deviation) fraction.
#                    (Optional: default value is 0.2).
# 
# EXAMPLES:
# 
#    SPARSEBAYESDEMO("Bernoulli",2)
#    SPARSEBAYESDEMO("Gaussian",1,0.5)
#
# NOTES: 
# 
# This program offers a simple demonstration of how to use the
# SPARSEBAYES (V2) Matlab software.
# 
# Synthetic data is generated from an underlying linear model based
# on a set of "Gaussian" basis functions, with the generator being
# "sparse" such that 10% of potential weights are non-zero. Data may be
# generated in an input space of one or two dimensions.
# 
# This generator is then used either as the basis for real-valued data with
# additive Gaussian noise (whose level may be varied), or for binary
# class-labelled data based on probabilities given by a sigmoid link
# function.
# 
# The SPARSEBAYES algorithm is then run on the data, and results and
# diagnostic information are graphed.
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from SB2_Likelihoods import SB2_Likelihoods
from SB2_UserOptions import SB2_UserOptions
from SB2_ParameterSettings import SB2_ParameterSettings
from SparseBayes import SparseBayes
from SB2_Sigmoid import SB2_Sigmoid

def SparseBayesDemo(*args):
    likelihood_ = args[0]
    dimension = args[1]
    if len(args) == 3:
        noiseToSignal = args[2]
    
    # Fix the random seed for reproducibility of results

    rseed    = 1
    np.random.seed(rseed)

    #######################################################################
    #
    # --- VALIDATE INPUT ARGUMENTS ---
    #
    #######################################################################
    
    # (1) likelihood

    LIKELIHOOD      = SB2_Likelihoods(likelihood_) 
    
    # (2) dimension
    
    if (dimension != 1) and (dimension != 2):
        raise Exception('Specified dimension should be 1 or 2')

    # Set up default for "noiseToSignal" variable.
    # For ease of use, we'll just ignore it in the case of a non-Gaussian
    # likelihood model.
    
    if 'noiseToSignal' not in locals():
        noiseToSignal = 0.2
    
    #######################################################################
    # 
    # --- SET UP DEMO PARAMETERS ---
    # 
    #######################################################################
    # 
    # Experiment with these values to vary the demo if you wish
    #

    if dimension == 1:
        N = 100    # Number of points
    else:
        N = 900    # Gives a nice square grid of decent size

    basisWidth      = 0.05        # NB: data is in [0,1]
    
    # Define probability of a basis function NOT being used by the generative
    # model. i.e. if pSparse=0.90, only 10% of basis functions (on average) will
    # be used to synthesise the data.
    
    pSparse         = 0.90
    iterations      = 500
    
    # Heuristically adjust basis width to account for 
    # distance scaling with dimension.
  
    basisWidth = basisWidth ** (1/float(dimension))
    
    #######################################################################
    #
    # --- SYNTHETIC DATA GENERATION ---
    #
    #######################################################################
    # 
    # First define the input data over a regular grid
    # 
    
 
    if (dimension == 1):
        X = (np.matrix(range(N)) / float(N)).T
        
    else: # dimension is 2
        sqrtN        = math.floor(math.sqrt(N))
        N            = sqrtN*sqrtN
        x            = (np.matrix(range(int(sqrtN))) / sqrtN).T
        X            = np.zeros((sqrtN ** 2, 2), float)
        for i in range(int(sqrtN)):
            for j in range(int(sqrtN)):
                    X[i*int(sqrtN) + j, 0] = x[i]
                    X[i*int(sqrtN) + j, 1] = x[j]
        X = np.matrix(X)
   
      
    # Now define the basis 
    # 
    # Locate basis functions at data points
    
    C = X
    
    # Compute ("Gaussian") basis (design) matrix
    
    BASIS = np.exp(-distSquared(X,C)/(basisWidth**2))
    
    # Randomise some weights, then make each weight sparse with probability
    # pSparse 
    
    M = BASIS.shape[1]
    w = np.random.randn(M,1)*100 / float((M*(1-pSparse)))
    sparse        = np.random.rand(M,1)<pSparse
    w[sparse] = 0
    
    # Now we have the basis and weights, compute linear model
    
    z = BASIS*w   
    
    # Finally generate the data according to the likelihood model
    
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
        # Generate our data by adding some noise on to the generative function
        noise = np.std(z, ddof=1) * noiseToSignal
        Outputs    = z + noise*np.random.randn(N,1)
    
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Bernoulli']:
        # Generate random [0,1] labels given by the log-odds 'z'
        Outputs    = np.random.rand(N,1) < SB2_Sigmoid(z)
        
    
    #######################################################################
    # 
    # --- SET UP GRAPHING PARAMETERS ---
    # 
    #######################################################################

    fRows       = 2
    fCols       = 3
      
    SP_DATA     = 1
    SP_LIKELY   = 2
    SP_LINEAR   = 3
    SP_COMPARE  = 4
    SP_WEIGHTS  = 5
    SP_GAMMA    = 6
        
    fig = plt.figure()
    plt.clf()
    TITLE_SIZE    = 12
        
    if (dimension == 1):
        plt.subplot(fRows,fCols,SP_DATA)
        plt.plot(X,Outputs,'k.', clip_on=False)
    else: 
        ax = fig.add_subplot(fRows,fCols,SP_DATA, projection='3d')
        #ax.scatter(X[:,0],X[:,1],Outputs, 'k.')
    
    t_    = 'Generated data ({0} points)'.format(N)
    plt.title(t_,fontsize=TITLE_SIZE)
         
    
    #######################################################################
    # 
    # --- SPARSE BAYES INFERENCE SECTION ---
    # 
    #######################################################################
    #
    # The section of code below is the main section required to run the
    # SPARSEBAYES algorithm.
    # 
    #######################################################################
    #
    # Set up the options:
    # 
    # - we set the diagnostics level to 2 (reasonable)
    # - we will monitor the progress every 10 iterations
    # 
    
    OPTIONS = SB2_UserOptions('ITERATIONS', iterations, 'DIAGNOSTICLEVEL', 2, 'MONITOR', 10)
    
    # Set initial parameter values:
    # 
    # - this specification of the initial noise standard deviation is not
    # necessary, but included here for illustration. If omitted, SPARSEBAYES
    # will call SB2_PARAMETERSETTINGS itself to obtain an appropriate default
    # for the noise (and other SETTINGS fields).
    
    SETTINGS    = SB2_ParameterSettings('NOISESTD', 0.1)
    
    # Now run the main SPARSEBAYES function
    
    B = np.matrix(BASIS)

    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = SparseBayes(likelihood_, BASIS, Outputs, OPTIONS, SETTINGS)

    BASIS = B

    print('\nPARAMETER = \n')
    for key, value in PARAMETER.items():
        try:
            print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0,0])))
        except IndexError:
            try:
                print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0])))
            except IndexError:
                print('\t{0} : {1}'.format(key,value))
        except AttributeError:
            print('\t{0} : {1}'.format(key,value))


    print('\nHYPERPARAMETER = \n')
    for key, value in HYPERPARAMETER.items():
        try:
            print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0,0])))
        except IndexError:
            try:
                print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0])))
            except IndexError:
                print('\t{0} : {1}'.format(key,value))
        except AttributeError:
            print('\t{0} : {1}'.format(key,value))

    print('\nDIAGNOSTIC = \n')
    for key, value in DIAGNOSTIC.items():
        try:
            print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0,0])))
        except IndexError:
            try:
                print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0])))
            except IndexError:
                print('\t{0} : {1}'.format(key,value))
        except AttributeError:
            print('\t{0} : {1}'.format(key,value))
    print('\n')
    
    # Manipulate the returned weights for convenience later
    
    w_infer                             = np.zeros((M,1))
    w_infer[PARAMETER['RELEVANT']]      = PARAMETER['VALUE']

    # Compute the inferred prediction function
    
    y           = BASIS*w_infer

    # Convert the output according to the likelihood (i.e. apply link function)
    
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
        y_l     = y
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Bernoulli']:
        y_l     = SB2_Sigmoid(y) > 0.5
        
    
    

    #######################################################################
    # 
    # --- PLOT THE RESULTS ---
    #
    #######################################################################
        
        
    # Likelihood trace (and Gaussian noise info)
        
    plt.subplot(fRows,fCols,SP_LIKELY)
    lsteps    = np.size(DIAGNOSTIC['LIKELIHOOD'])
    plt.plot(range(0,lsteps), DIAGNOSTIC['LIKELIHOOD'], 'g-')
    plt.xlim(0, lsteps+1)
    plt.title('Log marginal likelihood trace',fontsize=TITLE_SIZE)
        
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
        ax    = plt.axis()
        dx    = ax[1]-ax[0]
        dy    = ax[3]-ax[2]
        t_    = 'Actual noise:   {:.5f}'.format(noise)
        plt.text(ax[0]+0.1*dx,ax[2]+0.6*dy,t_,fontname='Courier')
        t_    = 'Inferred noise: {:.5f}'.format( 1/math.sqrt(HYPERPARAMETER['BETA']) )
        plt.text(ax[0]+0.1*dx,ax[2]+0.5*dy,t_,fontname='Courier')
            
            
    # Compare the generative and predictive linear models
    if dimension == 1:
        plt.subplot(fRows,fCols,SP_LINEAR)
            
        if dimension == 1:
            plt.plot(X,z,'b-', linewidth=4, label='Actual') 
            plt.plot(X,y,'r-', linewidth=3, label='Model')
        else:
            pass
        plt.title('Generative function and linear model',fontsize=TITLE_SIZE)
        legend = plt.legend(loc=2, shadow=False, fontsize='small', frameon=False)
    
    
    # Compare the data and the predictive model (post link-function)
    if dimension == 1:
        plt.subplot(fRows,fCols,SP_COMPARE)
        if dimension == 1:
            plt.plot(X,Outputs,'k.', linewidth=4)
            plt.plot(X,y_l,'r-', linewidth=3)
            plt.plot(X[PARAMETER['RELEVANT']],Outputs[PARAMETER['RELEVANT']],'yo', markersize=8, clip_on=False)
        else:
            pass
        plt.title('Data and predictor',fontsize=TITLE_SIZE)
    
    
    # Show the inferred weights
        
    plt.subplot(fRows,fCols,SP_WEIGHTS)
    markerline, stemlines, baseline = plt.stem(w_infer, 'r-.')
    plt.setp(markerline, markerfacecolor='r')
    plt.setp(baseline, color='k', linewidth=1)
    plt.xlim(0, N+1)
    t_    = 'Inferred weights ({:d})'.format(len(PARAMETER['RELEVANT']))
    plt.title(t_,fontsize=TITLE_SIZE)
    
    
    # Show the "well-determinedness" factors
        
    plt.subplot(fRows,fCols,SP_GAMMA)
    ind = np.arange(len(DIAGNOSTIC['GAMMA'])) + 1
    plt.bar(ind, DIAGNOSTIC['GAMMA'], 0.7, color='g', align = 'center')
    plt.xlim(0, len(PARAMETER['RELEVANT'])+1)
    plt.ylim(0, 1.1)
    plt.title('Well-determinedness (gamma)',fontsize=TITLE_SIZE)
    
    plt.show()



#######################################################################
#
# Support function to compute basis
#


def distSquared(X,Y):
    nx = np.size(X, 0)
    ny = np.size(Y, 0)
    D2 =  ( np.multiply(X, X).sum(1)  * np.ones((1, ny)) ) + ( np.ones((nx, 1)) * np.multiply(Y, Y).sum(1).T  ) - 2*X*Y.T
    return D2
