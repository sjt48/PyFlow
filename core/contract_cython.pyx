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
cimport cython
import numpy as np
cimport numpy as np
# from cython.parallel import prange
DTYPE = np.float64
       
#------------------------------------------------------------------------------

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:],float64[:,:],float64[:,:])],'(n,n),(n,n)->(n,n)',target='cpu',nopython=True)
def cycon22(double[:,:] A,double[:,:] B, double[:,:]C):
    cdef int i,j,k,m
    m = len(A[0])
    for i in range(m):
        for j in range(m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]

    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:],float64[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='cpu',nopython=True)
def cycon_42(double[:,:,:,:] A,double[:,:] B,double[:,:,:,:] C):
    cdef int i,j,k,q,m,l
    m = len(A[0,0,0])
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                        for l in range(m):
                            C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                            C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                            C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                            C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
    return np.asarray(C,dtype=np.float64)


@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:],float64[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='cpu',nopython=True)
def cycon_42_firstpair(double[:,:,:,:] A,double[:,:] B,double[:,:,:,:] C):
    cdef int i,j,k,q,m,l
    m = len(A[0,0,0])
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    for l in range(m):
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:],float64[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='cpu',nopython=True)
def cycon_42_secondpair(double[:,:,:,:] A, double[:,:] B, double[:,:,:,:] C):
    cdef int i,j,k,q,m,l
    # m,_,_,_=A.shape
    m = len(A[0,0,0])
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    for l in range(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:],float64[:],float64[:,:])],'(n,n,n,n),(n,n),(n)->(n,n)',target='cpu',nopython=True)
def cycon_42_NO(double[:,:,:,:] A, double[:,:] B, double[:] state, double[:,:] C):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    cdef int i,j,k,q,m
    m = len(B[0])
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])
                        C[i,j] += A[k,q,i,j]*B[q,k]*(state[k]-state[q])
                        C[i,j] += -A[k,j,i,q]*B[q,k]*(state[k]-state[q])
                        C[i,j] += A[i,q,k,j]*B[q,k]*(state[k]-state[q])
    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:],float64[:],float64[:,:])],'(n,n,n,n),(n,n),(n)->(n,n)',target='cpu',nopython=True)
def cycon_42_NO_secondpair(double[:,:,:,:] A, double[:,:] B, double[:] state, double[:,:] C):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    cdef int i,j,k,q,m
    m = len(B[0])
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])
    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:],float64[:],float64[:,:])],'(n,n,n,n),(n,n),(n)->(n,n)',target='cpu',nopython=True)
def cycon_42_NO_firstpair(double[:,:,:,:] A, double[:,:] B, double[:] state, double[:,:] C):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    cdef int i,j,k,q,m
    m = len(B[0])
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[k,q,i,j]*B[q,k]*(state[k]-state[q])
    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def cycon_44_NO(double[:,:,:,:] A,double[:,:,:,:] B,double[:] state,double[:,:,:,:] C):
    cdef int i,j,k,q,l,m,m0
    m0 = len(A[0,0,0])
    for i in range(m0):
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

    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def cycon_44_NO_up_mixed(double[:,:,:,:] A,double[:,:,:,:] B,double[:] state,double[:,:,:,:] C):
    cdef int i,j,k,q,l,m,m0
    m0 = len(A[0,0,0])
    for i in range(m0):
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
    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def cycon_44_NO_down_mixed(double[:,:,:,:] A,double[:,:,:,:] B,double[:] state,double[:,:,:,:] C):
    cdef int i,j,k,q,l,m,m0
    m0 = len(A[0,0,0])
    for i in range(m0):
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
    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def cycon_44_NO_mixed(double[:,:,:,:] A,double[:,:,:,:] B,double[:] upstate,double[:] downstate,double[:,:,:,:] C):
    cdef int i,j,k,q,l,m,m0
    m0 = len(A[0,0,0])
    for i in range(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                C[i,j,k,q] += A[l,j,k,m]*(B[i,l,m,q])*(upstate[l]-downstate[m]) #+
                                C[i,j,k,q] += A[i,l,m,q]*B[l,j,k,m]*(-upstate[l]+downstate[m])
                                C[i,j,k,q] +=  -A[l,j,m,q]*(B[i,l,k,m])*(upstate[l]+downstate[m]) #--
                                C[i,j,k,q] +=  A[i,l,k,m]*(B[l,j,m,q])*(upstate[l]+downstate[m]) #--
    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def cycon_44_NO_mixed_mixed_up(double[:,:,:,:] A,double[:,:,:,:] B,double[:] state,double[:,:,:,:] C):
    cdef int i,j,k,q,l,m,m0
    m0 = len(A[0,0,0])
    for i in range(m0):
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
    return np.asarray(C,dtype=np.float64)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.cdivision(True)
# @guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def cycon_44_NO_mixed_mixed_down(double[:,:,:,:] A,double[:,:,:,:] B,double[:] state,double[:,:,:,:] C):
    cdef int i,j,k,q,l,m,m0
    m0 = len(A[0,0,0])
    for i in range(m0):
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
    return np.asarray(C,dtype=np.float64)