"""
Flow Equations for Many-Body Quantum Systems
S. J. Thomson
Dahlem Centre for Complex Quantum Systems, FU Berlin
steven.thomson@fu-berlin.de
steventhomson.co.uk / @PhysicsSteve
https://orcid.org/0000-0001-9065-9842
---------------------------------------------

This work is licensed under a Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International License. This work may
be edited and shared by others, provided the author of this work is credited 
and any future authors also freely share and make their work available. This work
may not be used or modified by authors who do not make any derivative work 
available under the same conditions. This work may not be modified and used for
any commercial purpose, in whole or in part. For further information, see the 
license details at https://creativecommons.org/licenses/by-nc-sa/4.0/.

This work is distributed without any form of warranty or committment to provide technical 
support, nor the guarantee that it is free from bugs or technical incompatibilities
with all possible computer systems. Use, modify and troubleshoot at your own risk.

---------------------------------------------

This file contains all of the matrix/tensor contraction routines used to compute flow equations numerically.

"""

import os
from psutil import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of OpenMP threads
os.environ['MKL_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of MKL threads
os.environ['NUMBA_NUM_THREADS'] = str(int(cpu_count(logical=False))) # Set number of Numba threads
import numpy as np
from numba import jit,prange,float64
# from numba import get_num_threads,threading_layer
#import numpy

#------------------------------------------------------------------------------
# jit functions which return a matrix
    
@jit(float64[:,:](float64[:,:],float64[:,:],float64[:,:]),nopython=True,parallel=True,fastmath=True,cache=True,nogil=True)
# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit(A,B,C):
    """ Contract two square matrices. Computes upper half only and then symmetrises. """
    m,_=A.shape
    for i in prange(m):
        for j in range(i,m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
            C[j,i] = C[i,j]
    return C

@jit(float64[:,:](float64[:,:],float64[:,:],float64[:,:]),nopython=True,parallel=True,fastmath=True,cache=True,nogil=True)
def con_jit_anti(A,B,C):
    """ Contract two square matrices. Computes upper half only and then anti-symmetrises. """
    m,_=A.shape
    for i in prange(m):
        for j in range(i,m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
            C[j,i] = -C[i,j]
    return C

@jit(nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit_comp(A,B):
    """ Contract two square complex matrices. Computes upper half only and then symmetrises. """
    C = np.zeros(A.shape,dtype=np.complex64)
    m,_=A.shape
    for i in prange(m):
        for j in range(i,m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
            C[j,i] = np.conj(C[i,j])

    return C

@jit(nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit_anti_comp(A,B):
    """ Contract two square complex matrices. Computes upper half only and then anti-symmetrises. """
    C = np.zeros(A.shape,dtype=np.complex64)
    m,_=A.shape
    for i in prange(m):
        for j in range(i,m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
            C[j,i] = -np.conj(C[i,j])

    return C

@jit(float64[:,:](float64[:,:,:,:],float64[:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit42_NO(A,B,state):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    C = np.zeros(B.shape,dtype=np.float64)
    m,_=B.shape
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])
                        C[i,j] += A[k,q,i,j]*B[q,k]*(state[k]-state[q])
                        C[i,j] += -A[k,j,i,q]*B[q,k]*(state[k]-state[q])
                        C[i,j] += A[i,q,k,j]*B[q,k]*(state[k]-state[q])

    return C

@jit(float64[:,:](float64[:,:,:,:],float64[:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit42_NO_secondpair(A,B,state):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    C = np.zeros(B.shape,dtype=np.float64)
    m,_=B.shape
    for i in range(m):
        for j in range(i):
            # C[i,j] = 0.
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])
        C[j,i] = C[i,j]
    return C

@jit(float64[:,:](float64[:,:,:,:],float64[:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit42_NO_firstpair(A,B,state):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    C = np.zeros(B.shape,dtype=np.float64)
    m,_=B.shape
    for i in range(m):
        for j in range(i):
            # C[i,j] = 0.
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[k,q,i,j]*B[q,k]*(state[k]-state[q])
        C[j,i] = C[i,j]
    return C

@jit(float64[:,:](float64[:,:,:,:],float64[:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit42_comp_NO(A,B,state):
    """ 2-point contractions of a (complex) rank-4 tensor with a (complex) square matrix. Computes upper half only and then symmetrises. """
    C = np.zeros(B.shape,dtype=np.float64)
    m,_=B.shape
    for i in prange(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])
                        C[i,j] += A[k,q,i,j]*B[q,k]*(state[k]-state[q])
                        C[i,j] += -A[k,j,i,q]*B[q,k]*(state[k]-state[q])
                        C[i,j] += A[i,q,k,j]*B[q,k]*(state[k]-state[q])

    return C


#------------------------------------------------------------------------------
# jit functions which return a rank-4 tensor

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:]),nopython=True,parallel=True,fastmath=True,cache=True,nogil=True)
def con_jit42(A,B):
    C = np.zeros(A.shape,dtype=np.float64)
    m,_,_,_=A.shape
    for i in prange(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    C[i,j,k,q] = 0.
                    for l in range(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 

    return C

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:]),nopython=True,parallel=True,fastmath=True,cache=True,nogil=True)
def con_jit42_firstpair(A,B):
    C = np.zeros(A.shape,dtype=np.float64)
    m,_,_,_=A.shape
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    C[i,j,k,q] = 0.
                    for l in range(m):
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 

    return C

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:]),nopython=True,parallel=True,fastmath=True,cache=True,nogil=True)
def con_jit42_secondpair(A,B):
    C = np.zeros(A.shape,dtype=np.float64)
    m,_,_,_=A.shape
    for i in prange(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    C[i,j,k,q] = 0.
                    for l in range(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]

    return C

@jit(nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit42_comp(A,B):
    C = np.zeros(A.shape,dtype=np.complex64)
    m,_,_,_=A.shape
    for i in prange(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    for l in range(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 

    return C

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:,:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit44_NO(A,B,state):
    C = np.zeros(A.shape,dtype=np.float64)
    m0,_,_,_=A.shape
    for i in prange(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += 0.25*A[i,j,l,m]*(B[m,l,k,q]+B[k,q,m,l]-B[m,q,k,l]+B[k,l,m,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,m,i,j]*(B[m,l,k,q]+B[k,q,m,l]-B[m,q,k,l]+B[k,l,m,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,j,i,m]*(B[m,l,k,q]+B[k,l,m,q]-B[m,q,k,l]+B[k,q,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -0.25*A[i,l,m,j]*(B[k,m,l,q]+B[k,q,l,m]+B[l,m,k,q]-B[l,q,k,m])*(state[l]-state[m]) #-
                                C[i,j,k,q] +=  0.25*A[l,j,m,q]*(B[i,m,k,l]+B[i,l,k,m])*(state[l]+state[m]) #--
                                C[i,j,k,q] +=  0.25*A[i,l,k,m]*(B[m,j,l,q]+B[l,j,m,q])*(state[l]+state[m]) #--

                                if state[l] != state[m]:
                                    C[i,j,k,q] += -0.25*A[i,q,l,m]*(B[m,l,k,j]+B[k,j,m,l]-B[m,j,k,l]+B[k,l,m,j])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,m,i,q]*(B[m,l,k,j]+B[k,j,m,l]-B[m,j,k,l]+B[k,l,m,j])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,q,i,m]*(B[m,l,k,j]+B[k,l,m,j]-B[m,j,k,l]+B[k,j,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += 0.25*A[i,l,m,q]*(B[k,m,l,j]+B[k,j,l,m]+B[l,m,k,j]-B[l,j,k,m])*(state[l]-state[m]) #-
                                C[i,j,k,q] +=  -0.25*A[l,q,m,j]*(B[i,m,k,l]+B[i,l,k,m])*(state[l]+state[m]) #--
                                C[i,j,k,q] +=  -0.25*A[i,l,k,m]*(B[m,q,l,j]+B[l,q,m,j])*(state[l]+state[m]) #--

                                if state[l] != state[m]:
                                    C[i,j,k,q] += -0.25*A[k,j,l,m]*(B[m,l,i,q]+B[i,q,m,l]-B[m,q,i,l]+B[i,l,m,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,m,k,j]*(B[m,l,i,q]+B[i,q,m,l]-B[m,q,i,l]+B[i,l,m,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,j,k,m]*(B[m,l,i,q]+B[i,l,m,q]-B[m,q,i,l]+B[i,q,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += 0.25*A[k,l,m,j]*(B[i,m,l,q]+B[i,q,l,m]+B[l,m,i,q]-B[l,q,i,m])*(state[l]-state[m]) #-
                                C[i,j,k,q] +=  -0.25*A[l,j,m,q]*(B[k,m,i,l]+B[k,l,i,m])*(state[l]+state[m]) #--
                                C[i,j,k,q] +=  -0.25*A[k,l,i,m]*(B[m,j,l,q]+B[l,j,m,q])*(state[l]+state[m]) #--

                                if state[l] != state[m]:
                                    C[i,j,k,q] += 0.25*A[k,q,l,m]*(B[m,l,i,j]+B[i,j,m,l]-B[m,j,i,l]+B[i,l,m,j])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,m,k,q]*(B[m,l,i,j]+B[i,j,m,l]-B[m,j,i,l]+B[i,l,m,j])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,q,k,m]*(B[m,l,i,j]+B[i,l,m,j]-B[m,j,i,l]+B[i,j,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -0.25*A[k,l,m,q]*(B[i,m,l,j]+B[i,j,l,m]+B[l,m,i,j]-B[l,j,i,m])*(state[l]-state[m]) #-
                                C[i,j,k,q] +=  0.25*A[l,q,m,j]*(B[k,m,i,l]+B[k,l,i,m])*(state[l]+state[m]) #--
                                C[i,j,k,q] +=  0.25*A[k,l,i,m]*(B[m,q,l,j]+B[l,q,m,j])*(state[l]+state[m]) #--

    return C

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:,:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit44_NO_up_mixed(A,B,state):
    C = np.zeros(A.shape,dtype=np.float64)
    m0,_,_,_=A.shape

    for i in prange(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += 0.25*A[i,j,l,m]*(B[m,l,k,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,m,i,j]*(B[m,l,k,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,j,i,m]*(B[m,l,k,q])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -0.25*A[i,l,m,j]*(B[l,m,k,q])*(state[l]-state[m]) #-

                                    C[i,j,k,q] += -0.25*A[i,q,l,m]*(B[m,l,k,j])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,m,i,q]*(B[m,l,k,j])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,q,i,m]*(B[m,l,k,j])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += 0.25*A[i,l,m,q]*(B[l,m,k,j])*(state[l]-state[m]) #-

                                    C[i,j,k,q] += -0.25*A[k,j,l,m]*(B[m,l,i,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,m,k,j]*(B[m,l,i,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,j,k,m]*(B[m,l,i,q])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += 0.25*A[k,l,m,j]*(B[l,m,i,q])*(state[l]-state[m]) #-

                                    C[i,j,k,q] += 0.25*A[k,q,l,m]*(B[m,l,i,j])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,m,k,q]*(B[m,l,i,j])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,q,k,m]*(B[m,l,i,j])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -0.25*A[k,l,m,q]*(B[l,m,i,j])*(state[l]-state[m]) #-

    return C

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:,:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit44_NO_down_mixed(A,B,state):
    C = np.zeros(A.shape,dtype=np.float64)
    m0,_,_,_=A.shape
    for i in prange(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += 0.25*A[i,j,l,m]*(B[k,q,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,m,i,j]*(B[k,q,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,j,i,m]*(B[k,q,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -0.25*A[i,l,m,j]*(B[k,q,l,m])*(state[l]-state[m]) #-

                                    C[i,j,k,q] += -0.25*A[i,q,l,m]*(B[k,j,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,m,i,q]*(B[k,j,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,q,i,m]*(B[k,j,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += 0.25*A[i,l,m,q]*(B[k,j,l,m])*(state[l]-state[m]) #-

                                    C[i,j,k,q] += -0.25*A[k,j,l,m]*(B[i,q,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,m,k,j]*(B[i,q,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,j,k,m]*(B[i,q,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += 0.25*A[k,l,m,j]*(B[i,q,l,m])*(state[l]-state[m]) #-

                                    C[i,j,k,q] += 0.25*A[k,q,l,m]*(B[i,j,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += 0.25*A[l,m,k,q]*(B[i,j,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,q,k,m]*(B[i,j,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -0.25*A[k,l,m,q]*(B[i,j,l,m])*(state[l]-state[m]) #-

    return C

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit44_NO_mixed(A,B,upstate,downstate):
    C = np.zeros(A.shape,dtype=np.float64)
    m0,_,_,_=A.shape
    for i in prange(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                # if state[l] != state[m]:
                                C[i,j,k,q] += A[l,j,k,m]*(B[i,l,m,q])*(upstate[l]-downstate[m]) #+
                                C[i,j,k,q] += A[i,l,m,q]*B[l,j,k,m]*(-upstate[l]+downstate[m])
                                C[i,j,k,q] +=  -A[l,j,m,q]*(B[i,l,k,m])*(upstate[l]+downstate[m]) #--
                                C[i,j,k,q] +=  A[i,l,k,m]*(B[l,j,m,q])*(upstate[l]+downstate[m]) #--

                                
    return C

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:,:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit44_NO_mixed_mixed_up(A,B,state):
    C = np.zeros(A.shape,dtype=np.float64)
    m0,_,_,_=A.shape
    for i in prange(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += 0.25*A[i,j,l,m]*(B[k,q,m,l])*(state[l]-state[m]) #+ 
                                    C[i,j,k,q] += -0.25*A[i,q,l,m]*B[k,j,m,l]*(state[l]-state[m])
                                    C[i,j,k,q] += -0.25*A[k,j,l,m]*(B[i,q,m,l])*(state[l]-state[m]) 
                                    C[i,j,k,q] += 0.25*A[k,q,l,m]*(B[i,j,m,l])*(state[l]-state[m])
               
    return C

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:,:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit44_NO_mixed_mixed_down(A,B,state):
    C = np.zeros(A.shape,dtype=np.float64)
    m0,_,_,_=A.shape
    # count = 0
    for i in prange(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += 0.25*A[l,m,i,j]*(B[m,l,k,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -0.25*A[l,m,i,q]*(B[m,l,k,j])*(state[l]-state[m])
                                    C[i,j,k,q] += -0.25*A[l,m,k,j]*(B[m,l,i,q])*(state[l]-state[m])
                                    C[i,j,k,q] += 0.25*A[l,m,k,q]*(B[m,l,i,j])*(state[l]-state[m])

    return C

@jit(float64[:,:,:,:](float64[:,:,:,:],float64[:,:,:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit44_anti_NO(A,B,state):
    C = np.zeros(A.shape,dtype=np.float64)
    m0,_,_,_=A.shape
    for i in prange(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                    # Indices to be summed over
                    for l in range(m0):
                        for m in range(m0):
                            if state[l] != state[m]:
                                C[i,j,k,q] += A[i,j,l,m]*(B[m,l,k,q]+B[k,q,m,l]-B[m,q,k,l]+B[k,l,m,q])*(state[l]-state[m]) #+
                                C[i,j,k,q] += A[l,m,i,j]*(B[m,l,k,q]+B[k,q,m,l]-B[m,q,k,l]+B[k,l,m,q])*(state[l]-state[m]) #+
                                C[i,j,k,q] += -A[l,j,i,m]*(B[m,l,k,q]+B[k,l,m,q]-B[m,q,k,l]+B[k,q,m,l])*(state[l]-state[m]) #-
                                C[i,j,k,q] += A[i,l,m,j]*(B[k,m,l,q]+B[k,q,l,m]+B[l,m,k,q]-B[l,q,k,m])*(state[l]-state[m]) #-
                            C[i,j,k,q] += A[l,j,m,q]*(B[i,m,k,l]+B[i,l,k,m])*(state[l]+state[m]) #--
                            C[i,j,k,q] += A[i,l,k,m]*(B[m,j,l,q]+B[l,j,m,q])*(state[l]+state[m]) #--
    return C
         
