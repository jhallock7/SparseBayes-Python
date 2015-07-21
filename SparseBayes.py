
# SPARSEBAYES  Sparse Bayesian modelling: main estimation algorithm
#
# [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
#    SPARSEBAYES(LIKELIHOOD, BASIS, TARGETS, OPTIONS, SETTINGS)
#
# OUTPUT ARGUMENTS:
# 
#    PARAMETER    Structure specifying inferred primary parameters:
# 
#    .Value        Vector of weight values
#    .Relevant    Vector of corresponding indices of relevant
#                columns of BASIS matrix (sorted ascending)
#
#    HYPERPARAMETER    Structure specifying inferred hyperparameters:
# 
#    .Alpha            Vector of weight precision values
#    .beta            Noise precision (Gaussian likelihood case)
# 
#    DIAGNOSTIC    Structure containing various diagnostics:
# 
#    .Gamma        Vector of "well-determined" factors [0,1] for
#                relevant weights 
#    .Likelihood    Vector of evolving log-marginal-likelihood
#    .iterations    Number of iterations run
#    .S_Factor    Vector of S ("Sparsity") values for relevant weights
#    .Q_Factor    Vector of Q ("Quality") values for relevant weights
# 
# INPUT ARGUMENTS:
# 
#    LIKELIHOOD    String comprising one of 'Gaussian', 'Bernoulli' or 'Poisson'
#
#    BASIS        NxM matrix of basis vectors (one column per basis function)
#
#    TARGETS        N-vector with target output values
# 
#    OPTIONS        User-specified settings via SB2_USEROPTIONS [Optional]
# 
#    SETTINGS    Initialisation of main parameter values via
#                SB2_PARAMETERSETTINGS [Optional]
#
# NOTES: 
#
# SPARSEBAYES is the implementation of the main algorithm for parameter
# inference in "sparse Bayesian" models.
# 
# Given inputs (BASIS), desired outputs (TARGETS) and an appropriate
# LIKELIHOOD function, SPARSEBAYES will optimise the log-marginal-likelihood
# of the corresponding sparse Bayesian model and should return (given
# reasonable choice of basis) a sparse vector of model parameters.
# 
# OPTIONS and SETTINGS arguments may be omitted, and if so, will assume
# sensible default values.
# 
# SEE ALSO:
#
#    SB2_USEROPTIONS: available user-definable options, such as the
#    number of iterations to run for, level of diagnostic output etc.
# 
#    SB2_PARAMETERSETTINGS: facility to change default initial values for
#    parameters and hyperparameters.
# 
#    SB2_CONTROLSETTINGS: hard-wired internal algorithm settings.
#
# The main algorithm is based upon that outlined in "Fast marginal
# likelihood maximisation for sparse Bayesian models", by Tipping & Faul, in
# Proceedings of AISTATS 2003. That paper may be downloaded from
# www.relevancevector.com or via the conference online proceedings site at
# http://research.microsoft.com/conferences/aistats2003/proceedings/.
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
import time
from SB2_ParameterSettings  import SB2_ParameterSettings
from SB2_UserOptions  import SB2_UserOptions
from SB2_ControlSettings import SB2_ControlSettings
from SB2_Diagnostic import SB2_Diagnostic
from SB2_Initialisation import SB2_Initialisation
from SB2_FullStatistics import SB2_FullStatistics
from SB2_FormatTime import SB2_FormatTime


def SparseBayes(*args):

    likelihood_     = args[0]
    BASIS           = args[1]
    Targets         = args[2]
    if len(args) > 3:
        OPTIONS     = args[3]
    if len(args) > 4:
        SETTINGS    = args[4]
    
    
     
    ###########################################################################
    ##
    ## SET UP INITIAL PARAMETERS, USER OPTIONS AND ALGORITHM CONTROL SETTINGS
    ##
    ###########################################################################
    
        
    ## If no initial parameter setting structure passed, import the defaults
    
    if 'SETTINGS' not in locals():
        SETTINGS    = SB2_ParameterSettings()

    ## If no user options passed, import defaults
        
    if 'OPTIONS' not in locals():
        OPTIONS    = SB2_UserOptions()
        

    ## Any sanity checks on options and initialisation here
      
    # Error if fixed noise specified but value not set
    
    if OPTIONS['FIXEDNOISE'] and not SETTINGS['BETA'] and not SETTINGS['NOISESTD']:
        raise Exception('Option to fix noise variance is set but value is not supplied.')

    
    ## Get the default algorithm control settings
    
    CONTROLS    = SB2_ControlSettings()

    # Start the clock now for diagnostic purposes (and to allow the algorithm to
    # run for a fixed time)

    t0 = time.time()
    
    ###########################################################################

    ##
    ## INITIALISATION
    ##
    ## Pre-process basis, set up internal parameters according to initial
    ## settings and likelihod specification
    ##
    
    ###########################################################################
    

    # Kick off diagnostics (primarily, open log file if specified)
    
    OPTIONS = SB2_Diagnostic(OPTIONS, 'start')
    
    # Initialise everything, based on SETTINGS and OPTIONS
    
    [LIKELIHOOD, BASIS, BasisScales, Alpha, beta, Mu, PHI, Used] = SB2_Initialisation(likelihood_, BASIS, Targets, SETTINGS, OPTIONS)  
    
    # Cache some values for later efficiency
    
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
        # It will be computationally advantageous to "cache" this quantity 
        # in the Gaussian case
        BASIS_PHI    = BASIS.T * PHI
    else:
        BASIS_PHI    = []
    
    BASIS_Targets    = BASIS.T * Targets

    # FULL COMPUTATION
    # 
    # Initialise with a full explicit computation of the statistics
    # 
    # NOTE: The AISTATS paper uses "S/Q" (upper case) to denote the key
    # "sparsity/quality" Factors for "included" basis functions, and "s/q"
    # (lower case) for the factors calculated when the relevant basis
    # functions are "excluded".
    # 
    # Here, for greater clarity:
    # 
    #    All S/Q are denoted by vectors S_in, Q_in
    #    All s/q are denoted by vectors S_out, Q_out

    [SIGMA,Mu,S_in,Q_in,S_out,Q_out,Factor,logML,Gamma,BASIS_B_PHI,beta] = SB2_FullStatistics(LIKELIHOOD,BASIS,PHI,Targets,Used,Alpha,beta,Mu,BASIS_PHI,BASIS_Targets,OPTIONS)
    
    # Avoid falling over in pathological case of zero iterations
    
    if OPTIONS['ITERATIONS'] == 0:
        PARAMETER               = []
        HYPERPARAMETER          = []
        #DIAGNOSTIC['Likelihood']   = logML THIS IS A BUG IN TH ORIGINAL PROGRAM
        return

    [N, M_full]         = BASIS.shape
    M                   = PHI.shape[1]
    
    # Some diagnostics
    
    addCount            = 0
    deleteCount         = 0
    updateCount         = 0

    # Create storage to record evolution of log marginal likelihood
    
    maxLogSize          = OPTIONS['ITERATIONS'] + CONTROLS['BetaUpdateStart'] + math.ceil(OPTIONS['ITERATIONS'] / float(CONTROLS['BetaUpdateFrequency']))
    logMarginalLog      = np.zeros((maxLogSize,1))
    count               = 0

    # If we're doing basis alignment testing, we'll need to maintain lists of
    # those functions that are near identical, both in and out of the current
    # model.

    if CONTROLS['BasisAlignmentTest']:
        Aligned_out         = []
        Aligned_in          = []
        alignDeferCount     = 0
        
    # ACTION CODES
    #
    # Assign an integer code to the basic action types
    
    ACTION_REESTIMATE       = 0
    ACTION_ADD              = 1
    ACTION_DELETE           = -1
    
    # Some extra types
    
    ACTION_TERMINATE        = 10
    ACTION_NOISE_ONLY       = 11
    ACTION_ALIGNMENT_SKIP   = 12
    
    # Before kicking off the main loop, call the specified "callback" function
    # with "ACTION_ADD" to take account of the initialisation of the model
    
    if OPTIONS['CALLBACK']:
        eval(OPTIONS['CALLBACKFUNC'])(0, ACTION_ADD,logML/N, Used, Mu / BasisScales[Used].T, SIGMA, Alpha, beta, Gamma, PHI, OPTIONS['CALLBACKDATA'].flatten())
        
    
    ###########################################################################
    ##
    ## MAIN LOOP
    ##
    ###########################################################################
    
    i                   = 0    # Iteration number
    LAST_ITERATION      = False

    while not LAST_ITERATION:
        
        i    = i+1   

        # "UpdateIteration": set to true if this is an iteration where fast matrix
        # update rules can be used compute the appropriate quantities
   
        # This can be done if:
   
        # -    we are using a Gaussian likelihood
        # -    we are using other likelihoods and we have not specified a full
        #        posterior mode computation this cycle
   
        UpdateIteration    = LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian'] or np.remainder(i,CONTROLS['PosteriorModeFrequency'])
        
        #######################################################################
  
        ## DECISION PHASE
  
        ## Assess all potential actions
  

        # Compute change in likelihood for all possible updates
  
        DeltaML         = np.zeros((M_full, 1))
        Action          = ACTION_REESTIMATE*np.ones((M_full, 1)) # Default
        
        # 'Relevance Factor' (Q^S-S) values for basis functions in model
        UsedFactor      = Factor[Used]
        
        # RE-ESTIMATION: must be a POSITIVE 'factor' and already IN the model
        
        iu              = np.array(UsedFactor > CONTROLS['ZeroFactor']).flatten()
        index           = Used[iu]
        NewAlpha        = np.divide(np.power(S_out[index],2),  Factor[index].astype(float))
        Delta           = np.divide(float(1), NewAlpha) - np.divide(float(1), Alpha[iu]) # Temp vector
        
        # Quick computation of change in log-likelihood given all re-estimations
        
        DeltaML[index]  =  ( np.divide(np.multiply(Delta, (np.power(Q_in[index], 2))) , ( np.multiply(Delta, S_in[index]) + 1)) - np.log(1 + np.multiply(S_in[index], Delta)) )/2

        # DELETION: if NEGATIVE factor and IN model
        
        # But don't delete:
        #        - any "free" basis functions (e.g. the "bias")
        #        - if there is only one basis function (M=1)
         
        # (In practice, this latter event ought only to happen with the Gaussian
        # likelihood when initial noise is too high. In that case, a later beta
        # update should 'cure' this.
        
        iu              = np.logical_not(iu)     # iu = UsedFactor <= CONTROLS.ZeroFactor
        index           = Used[iu]
        anyToDelete     = ( np.setdiff1d(index, OPTIONS['FREEBASIS'])).size != 0 and M > 1
        
        if anyToDelete:
            # Quick computation of change in log-likelihood given all deletions
            DeltaML[index]  = -(np.divide(np.power(Q_out[index], 2), (S_out[index] + Alpha[iu])) - np.log(1 + np.divide(S_out[index], Alpha[iu])))/2
            Action[index]   = ACTION_DELETE
            # Note: if M==1, DeltaML will be left as zero, which is fine
            
        # ADDITION: must be a POSITIVE factor and OUT of the model
        
        # Find ALL good factors ...
        GoodFactor          = Factor > CONTROLS['ZeroFactor']
        # ... then mask out those already in model
        GoodFactor[Used]    = 0
        # ... and then mask out any that are aligned with those in the model
        if CONTROLS['BasisAlignmentTest']:
            GoodFactor[Aligned_out]    = 0
        index               = np.array(GoodFactor).squeeze().nonzero()
        anyToAdd            = np.size(index) != 0
        
        if anyToAdd:   
            # Quick computation of change in log-likelihood given all additions
            quot            = np.divide(np.power(Q_in[index], 2) , S_in[index])
            DeltaML[index]  = (quot - 1 - np.log(quot))/2
            Action[index]   = ACTION_ADD
            

        #######################################################################
        
        # Post-process action results to take account of preferences
  
        # Ensure that nothing happens with "free basis" functions
        
        DeltaML[OPTIONS['FREEBASIS']]    = 0
        
        # If we prefer ADD or DELETE actions over RE-ESTIMATION
        
        if (anyToAdd and CONTROLS['PriorityAddition']) or (anyToDelete and CONTROLS['PriorityDeletion']):
            # We won't perform re-estimation this iteration, which we achieve by
            # zero-ing out the delta
            DeltaML[Action == ACTION_REESTIMATE]    = 0
            
            # Furthermore, we should enforce ADD if preferred and DELETE is not
            # - and vice-versa

            if anyToAdd and CONTROLS['PriorityAddition'] and not CONTROLS['PriorityDeletion']:
                DeltaML[Action == ACTION_DELETE]    = 0
                
            if anyToDelete and CONTROLS['PriorityDeletion'] and not CONTROLS['PriorityAddition']:
                DeltaML[Action == ACTION_ADD]       = 0
                
        # Finally...we choose the action that results 
        # in the greatest change in likelihood
        
        deltaLogMarginal, nu    = DeltaML.max(), DeltaML.argmax()
        selectedAction          = Action[nu]
        anyWorthwhileAction     = deltaLogMarginal > 0
        
        # We need to note if basis nu is already in the model, and if so,
        # find its interior index, denoted by "j"

        if selectedAction == ACTION_REESTIMATE or selectedAction == ACTION_DELETE:   
            j       = np.array(Used == nu).squeeze().nonzero()[0]
        
        # Get the individual basis vector for update and compute its optimal alpha,
        # according to equation (20): alpha = S_out^2 / (Q_out^2 - S_out)
    
        Phi         = BASIS[:,nu]
        newAlpha    = S_out[nu]**2 / float(Factor[nu])

        #########################################################################
    
        # TERMINATION CONDITIONS
        #
        # Propose to terminate if:
        #
        # 1.    there is no worthwhile (likelihood-increasing) action, OR
        #
        # 2a.    the best action is an ACTION_REESTIMATE but this would only lead to
        #        an infinitesimal alpha change, AND
        # 2b.    at the same time there are no potential awaiting deletions
    
        if not anyWorthwhileAction or (selectedAction == ACTION_REESTIMATE and abs( np.log(newAlpha) - np.log(Alpha[j]) ) < CONTROLS['MinDeltaLogAlpha'] and not anyToDelete):
        
            selectedAction  = ACTION_TERMINATE
            act_            = 'potential termination'

        #########################################################################
  
        # ALIGNMENT CHECKS
        #
        # If we're checking "alignment", we may have further processing to do
        # on addition and deletion
    
        if CONTROLS['BasisAlignmentTest']:
    
        # Addition - rule out addition (from now onwards) if the new basis
        # vector is aligned too closely to one or more already in the model
    
            if selectedAction == ACTION_ADD:
                # Basic test for correlated basis vectors
                # (note, Phi and columns of PHI are normalised)
        
                p               = Phi.T * PHI
                findAligned     = np.array(p > CONTROLS['AlignmentMax']).squeeze().nonzero()
                numAligned      = np.size(findAligned)
            
                if numAligned > 0:
                    # The added basis function is effectively indistinguishable from
                    # one present already
                    selectedAction      = ACTION_ALIGNMENT_SKIP
                    act_                = 'alignment-deferred addition'
                    alignDeferCount     = alignDeferCount + 1
                
                    # Make a note so we don't try this next time
                    # May be more than one in the model, which we need to note was
                    # the cause of function 'nu' being rejected
                
                    Aligned_out     = np.vstack([Aligned_out , nu * np.ones((numAligned,1))])
                    Aligned_in      = np.vstack([Aligned_in , Used[findAligned]])
                
            # Deletion: reinstate any previously deferred basis functions
            # resulting from this basis function
        
            if selectedAction == ACTION_DELETE:
                findAligned     = np.array(Aligned_in == nu).squeeze().nonzero()
                numAligned      = np.size(findAligned)
            
                if numAligned > 0:
                    reinstated                  = Aligned_out[findAligned]
                    Aligned_in[findAligned]     = []
                    Aligned_out[findAligned]    = []
                
                
                    r_ = ''
                
                    for i in np.nditer(reinstated): #Mimics MATLAB's sprintf function
                        if (len(r_) == 0):
                            r_ = r_ + str(i)   
                        else:
                            r_ = r_ + ' ' + str(i)
                
                    SB2_Diagnostic(OPTIONS,3,'Alignment reinstatement of %s', r_)
                
        #########################################################################
    
        ## ACTION PHASE
        ##
        ## Implement above decision
    
        # We'll want to note if we've made a change which necessitates later
        # updating of the statistics
    
        UPDATE_REQUIRED    = False
    
        if selectedAction == ACTION_REESTIMATE:
        
            # Basis function 'nu' is already in the model, 
            # and we're re-estimating its corresponding alpha
            # 
            # - should correspond to Appendix A.3
        
            oldAlpha    = Alpha[j]
            Alpha[j]    = newAlpha
            s_j         = SIGMA[:,j]
            deltaInv    = 1 / float((newAlpha - oldAlpha))
            kappa       = 1 / float((SIGMA[j,j] + deltaInv))
            tmp         = kappa * s_j
            SIGMANEW    = SIGMA - tmp * s_j.T
            deltaMu     = -Mu[j][0,0] * tmp
            Mu          = Mu + deltaMu
        
            if UpdateIteration:
                S_in    = S_in + kappa * np.power((BASIS_B_PHI * s_j), 2)
                Q_in    = Q_in - (BASIS_B_PHI * deltaMu)
            
            updateCount     = updateCount + 1
            act_            = 're-estimation'
        
            UPDATE_REQUIRED    = True
        
        if selectedAction == ACTION_ADD:
        
            # Basis function nu is not in the model, and we're adding it in
            # 
            # - should correspond to Appendix A.2
        
            if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
                BASIS_Phi       = BASIS.T * Phi
                BASIS_PHI       = np.hstack([BASIS_PHI, BASIS_Phi])
                B_Phi           = beta * Phi
                BASIS_B_Phi     = beta * BASIS_Phi
        
            else:
                B_Phi           = np.multiply(Phi, beta)
                BASIS_B_phi     = BASIS.T * B_Phi
            
            tmp         = ((B_Phi.T * PHI) * SIGMA).T
        
            Alpha       = np.vstack([Alpha , newAlpha])
            PHI         = np.hstack([PHI, Phi])
        
            s_ii        = np.matrix(1 / float((newAlpha + S_in[nu])))
            s_i         = -s_ii[0,0] * tmp
            TAU         = -s_i * tmp.T
            SIGMANEW    = np.vstack([ np.hstack([SIGMA+TAU, s_i]) , np.hstack([s_i.T, s_ii]) ])
            mu_i        = s_ii[0,0] * Q_in[nu]
            deltaMu     = np.vstack([-mu_i[0,0]*tmp , mu_i])
            Mu          = np.vstack([Mu , 0]) + deltaMu
        
            if UpdateIteration:
                mCi     = BASIS_B_Phi - BASIS_B_PHI*tmp
                S_in    = S_in - s_ii[0,0] * np.power(mCi, 2)
                Q_in    = Q_in - mu_i[0,0] * mCi
            
            Used        = np.hstack([Used, nu])
            addCount    = addCount + 1
            act_        = 'addition'
        
            UPDATE_REQUIRED    = True
        
    
        if selectedAction == ACTION_DELETE:
        
            # Basis function nu is in the model, but we're removing it
            # 
            # - should correspond to Appendix A.4
        
            if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
                BASIS_PHI = np.delete(BASIS_PHI,j,1) #Deletes jth column
                
            
            PHI     = np.delete(PHI,j,1)
            Alpha   = np.delete(Alpha,j,0)
            
            s_jj        = SIGMA[j,j]
            s_j         = SIGMA[:,j]
            tmp         = s_j / float(s_jj)
            SIGMANEW    = SIGMA - tmp * s_j.T
            SIGMANEW    = np.delete(SIGMANEW,j,0)
            SIGMANEW    = np.delete(SIGMANEW,j,1)
            deltaMu     = -Mu[j][0,0] * tmp
            mu_j        = Mu[j]
            Mu          = Mu + deltaMu
            Mu          = np.delete(Mu,j,0)
        
            if UpdateIteration:
                jPm     = BASIS_B_PHI * s_j
                S_in    = S_in + np.power(jPm, 2) / float(s_jj)
                Q_in    = Q_in + jPm * mu_j / float(s_jj)
            
            Used = np.delete(Used,j)
            deleteCount         = deleteCount + 1
            act_                = 'deletion'
        
            UPDATE_REQUIRED    = True
        
        M        = np.size(Used)

        SB2_Diagnostic(OPTIONS, 3, 'ACTION: %s of %d (%g)', act_, nu, deltaLogMarginal)
    
    
        #########################################################################
    
        ## UPDATE STATISTICS
    
        # If we've performed a meaningful action,
        # update the relevant variables
    
        if UPDATE_REQUIRED:
        
            # S_in & Q_in values were calculated earlier
            # 
            # Here: update the S_out/Q_out values and relevance factors
        
            if UpdateIteration:
            
                # Previous "update" statisics calculated earlier are valid
            
                S_out           = np.matrix(S_in)
                Q_out           = np.matrix(Q_in)
                tmp             = np.divide(Alpha, (Alpha - S_in[Used]))
                S_out[Used]     = np.multiply(tmp, S_in[Used])
                Q_out[Used]     = np.multiply(tmp, Q_in[Used])
                Factor          = np.multiply(Q_out, Q_out) - S_out
                SIGMA           = SIGMANEW
                Gamma           = 1 - np.multiply(Alpha, SIGMA.diagonal().T)

                if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
                    BASIS_B_PHI    = beta * BASIS_PHI
                else:
                    BASIS_B_PHI    = ((np.multiply(PHI , (beta * np.ones((1,M))))).T *BASIS).T
        
            else:
                # Compute all statistics in "full" form (non-Gaussian likelihoods)
            
                [SIGMA,Mu,S_in,Q_in,S_out,Q_out,Factor,newLogM,Gamma,BASIS_B_PHI,beta] = SB2_FullStatistics(LIKELIHOOD,BASIS,PHI,Targets,Used,Alpha,beta,Mu,BASIS_PHI,BASIS_Targets,OPTIONS)
                
                deltaLogMarginal    = newLogM - logML
            
            if (UpdateIteration and deltaLogMarginal < 0):
                SB2_Diagnostic(OPTIONS, 1, '** Alert **  DECREASE IN LIKELIHOOD !! (%g)', deltaLogMarginal)
            
            logML                   = logML + deltaLogMarginal
            count                   = count + 1
            logMarginalLog[count - 1]   = logML
       
        
        # GAUSSIAN NOISE ESTIMATE
        # 
        # For Gaussian likelihood, re-estimate noise variance if:
        # 
        # - not fixed, AND
        # - an update is specified this cycle as normal, OR
        # - we're considering termination
    
        if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian'] and not OPTIONS['FIXEDNOISE'] and (selectedAction == ACTION_TERMINATE or i<=CONTROLS['BetaUpdateStart']  or np.remainder(i,CONTROLS['BetaUpdateFrequency'])==0):
        
            betaZ1      = beta
            y           = PHI * Mu
            e           = Targets - y
            beta        = (N - np.sum(Gamma)) / float(e.T*e)
        
            # Work-around zero-noise issue
            beta        = np.min(np.hstack([beta , (CONTROLS['BetaMaxFactor'] / float(np.var(Targets))) ]))
        
            deltaLogBeta    = np.log(beta) - np.log(betaZ1)
        
            if abs(deltaLogBeta) > CONTROLS['MinDeltaLogBeta']:
            
                # Full re-computation of statistics required after beta update
            
                [SIGMA,Mu,S_in,Q_in,S_out,Q_out,Factor,logML,Gamma,BASIS_B_PHI] = SB2_FullStatistics(LIKELIHOOD,BASIS,PHI,Targets,Used,Alpha,beta,Mu,BASIS_PHI,BASIS_Targets,OPTIONS)[0:10]
                count                       = count + 1
                logMarginalLog[count - 1]   = logML
                
                if selectedAction == ACTION_TERMINATE:
                
                    # We considered terminating above as no alpha update seemed
                    # worthwhile. However, a beta update has made a non-trivial
                    # increase in the likelihood, so we continue.
                
                    selectedAction = ACTION_NOISE_ONLY
                
                    SB2_Diagnostic(OPTIONS,3,'Noise update (termination deferred)')
                
                
        # CALLBACK
        #
        # Call callback function if specified
        # -    this can be useful for demos etc where it is desired to display
        #        graphical information at each iteration
    
        if OPTIONS['CALLBACK']:
        
            eval(OPTIONS['CALLBACKFUNC'])(i,selectedAction,logMarginalLog[1:count]/float(N),Used,np.divide(Mu, BasisScales[Used].T),SIGMA,Alpha,beta,Gamma,np.multiply(PHI,(np.ones((N,1))*BasisScales[Used])),OPTIONS['CALLBACKDATA'].flatten())


        #########################################################################
  
        # END OF CYCLE PROCESSING
        # 
        # Check if termination still specified, and output diagnostics
    
        if selectedAction == ACTION_TERMINATE:
        
            # If we're here, then no update to the model was considered worthwhile
        
            SB2_Diagnostic(OPTIONS,2, '** Stopping at iteration {:d} (Max_delta_ml={:g}) **'.format(i, deltaLogMarginal))
            
            if LIKELIHOOD['InUse'] != LIKELIHOOD['Gaussian']:
                SB2_Diagnostic(OPTIONS,2,'{:4d}> L = {:.6f}\t Gamma = {:.2f} (M = {:d})'.format( i, logML[0,0]/float(N), np.sum(Gamma), M))
            else:
                SB2_Diagnostic(OPTIONS,2, '{:4d}> L = {:.6f}\t Gamma = {:.2f} (M = {:d})\t s={:.3f}'.format(i,logML[0,0]/float(N),np.sum(Gamma),M,np.sqrt(1/float(beta))))
            break
    
        # Check for "natural" termination
    
        ITERATION_LIMIT     = i == OPTIONS['ITERATIONS']
        TIME_LIMIT          = time.time() - t0  > OPTIONS['TIME']
        LAST_ITERATION      = ITERATION_LIMIT or TIME_LIMIT
    
        if ( OPTIONS['MONITOR'] and not np.remainder(i,OPTIONS['MONITOR']) ) or LAST_ITERATION:
        
            # Output diagnostic progress info if desired
        
            if LIKELIHOOD['InUse'] != LIKELIHOOD['Gaussian']:
                SB2_Diagnostic(OPTIONS,2,'{:5d}> L = {:.6f}\t Gamma = {:.2f} (M = {:d})'.format(i,logML[0,0]/float(N),np.sum(Gamma),M))
            else:
                SB2_Diagnostic(OPTIONS,2,'{:5d}> L = {:.6f}\t Gamma = {:.2f} (M = {:d})\t s={:.3f}'.format(i,logML[0,0]/float(N),np.sum(Gamma),M,np.sqrt(1/float(beta))))
                
                
    ###########################################################################

    ##
    ## POST-PROCESSING
    ##
    
    #
    # Warn if we exited the main loop without terminating automatically
    
    if selectedAction != ACTION_TERMINATE:
        
        if ITERATION_LIMIT:
            SB2_Diagnostic(OPTIONS,1,'** Iteration limit: algorithm did not converge')
        elif TIME_LIMIT:
            SB2_Diagnostic(OPTIONS,1,'** Time limit: algorithm did not converge')
            
    
    
    # Output action summary if desired
    
    if OPTIONS['DIAGNOSTICLEVEL'] > 1:
        # Stop timer
        t1    = time.time() - t0
        total    = addCount + deleteCount + updateCount
        if CONTROLS['BasisAlignmentTest']:
            total    = total + alignDeferCount
        
        SB2_Diagnostic(OPTIONS,2,'Action Summary')
        SB2_Diagnostic(OPTIONS,2,'==============')
        SB2_Diagnostic(OPTIONS,2,'Added\t\t{:6d} ({:.0f}%)'.format(addCount,100*addCount/float(total)))
        SB2_Diagnostic(OPTIONS,2,'Deleted\t\t{:6d} ({:.0f}%)'.format(deleteCount,100*deleteCount/float(total)))
        SB2_Diagnostic(OPTIONS,2,'Reestimated\t{:6d} ({:.0f}%)'.format(updateCount, 100*updateCount/float(total)))
        
        if CONTROLS['BasisAlignmentTest'] and alignDeferCount:
            SB2_Diagnostic(OPTIONS,2,'--------------')
            SB2_Diagnostic(OPTIONS,2,'Deferred\t{:6d} (%.0f%)'.format(alignDeferCount,100*alignDeferCount/float(total)))
            
        SB2_Diagnostic(OPTIONS,2,'==============')
        SB2_Diagnostic(OPTIONS,2,'Total of {:d} likelihood updates'.format(count))
        SB2_Diagnostic(OPTIONS,2,'Time to run: {:s}'.format(SB2_FormatTime(t1)))
        
    
    # Terminate diagnostics
    OPTIONS = SB2_Diagnostic(OPTIONS, 'end')
    
    ############################################################
    
    ##
    ## OUTPUT VARIABLES
    ##
     
    PARAMETER       = {}
    HYPERPARAMETER  = {}
    DIAGNOSTIC      = {}
     
    # We also choose to sort here - it can't hurt and may help later
        
    PARAMETER['RELEVANT'], index    = np.sort(Used), np.argsort(Used)
        
    # Not forgetting to correct for normalisation too
    
    PARAMETER['VALUE']              = np.divide(Mu[index], BasisScales[:,Used[index]].T)
    
    HYPERPARAMETER['ALPHA']         = np.divide(Alpha[index], (np.power(BasisScales[:,Used[index]].T, 2)))
    HYPERPARAMETER['BETA']          = beta
    
    DIAGNOSTIC['GAMMA']             = Gamma[index]
    DIAGNOSTIC['LIKELIHOOD']        = logMarginalLog[0:count]
    DIAGNOSTIC['ITERATIONS']        = i
    DIAGNOSTIC['S_FACTOR']          = S_out
    DIAGNOSTIC['Q_FACTOR']          = Q_out
    DIAGNOSTIC['M_FULL']            = M_full
    

    return [PARAMETER, HYPERPARAMETER, DIAGNOSTIC]
