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
from numba import jit,prange,guvectorize,float64,complex128
# from numba import get_num_threads,threading_layer
#------------------------------------------------------------------------------
# Tensor contraction subroutines

# General contraction function
def contract(A,B,method='jit',comp=False,eta=False,pair=None):
    """ General contract function: gets shape and calls appropriate contraction function. """
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    if A.ndim == B.ndim == 2:
        con = con22(A,B,method,comp,eta)
    if A.ndim != B.ndim:
        if A.ndim == 4:
            if B.ndim == 2:
                if pair == None:
                    con = con42(A,B,method,comp)
                elif pair == 'first':
                    con = con42_firstpair(A,B,method,comp)
                elif pair == 'second':
                    con = con42_secondpair(A,B,method,comp)
        if A.ndim == 2:
            if B.ndim == 4:
                if pair == None:
                    con = con24(A,B,method,comp)
                elif pair == 'first':
                    con = con24_firstpair(A,B,method,comp)
                elif pair == 'second':
                    con = con24_secondpair(A,B,method,comp)
    # print(get_num_threads())
    # print("Threading layer: %s" % threading_layer())
    return con

# Normal-ordering contraction function
def contractNO(A,B,method='jit',comp=False,eta=False,state=[],upstate=[],downstate=[],pair=None):
    """ General normal-ordering function: gets shape and calls appropriate contraction function. """

    if A.ndim == B.ndim == 2:
        con = 0
    elif A.ndim == B.ndim == 4 and method == 'jit':
        if A.ndim == B.ndim == 4 and pair==None:
            con = con44_NO(A,B,method=method,comp=comp,eta=eta,state=state)
        elif A.ndim == B.ndim == 4 and pair=='up-mixed':
            con = con_jit44_NO_up_mixed(A,B,state=state)
        elif A.ndim == B.ndim == 4 and pair=='down-mixed':
            con = con_jit44_NO_down_mixed(A,B,state=state)
        elif A.ndim == B.ndim == 4 and pair=='mixed-mixed-up':
            con = con_jit44_NO_mixed_mixed_up(A,B,state=state)
        elif A.ndim == B.ndim == 4 and pair=='mixed-mixed-down':
            con = con_jit44_NO_mixed_mixed_down(A,B,state=state)
        elif A.ndim == B.ndim == 4 and pair=='mixed-up':
            con = -1*con_jit44_NO_up_mixed(A,B,state=state)
        elif A.ndim == B.ndim == 4 and pair=='mixed-down':
            con = -1*con_jit44_NO_down_mixed(A,B,state=state)
        elif A.ndim == B.ndim == 4 and pair=='mixed':
            con = con_jit44_NO_mixed(A,B,upstate=upstate,downstate=downstate)
        
    elif A.ndim == B.ndim == 4 and method == 'vec':
        if A.ndim == B.ndim == 4 and pair == None:
            con = con44_NO(A,B,method=method,comp=comp,eta=eta,state=state)
        elif A.ndim == B.ndim == 4 and pair=='up-mixed':
            con = np.zeros(A.shape,dtype=np.float64)
            con_vec44_NO_up_mixed(A,B,state,con)
        elif A.ndim == B.ndim == 4 and pair=='down-mixed':
            con = con = np.zeros(A.shape,dtype=np.float64)
            con_vec44_NO_down_mixed(A,B,state,con)
        elif A.ndim == B.ndim == 4 and pair=='mixed-mixed-up':
            con = np.zeros(A.shape,dtype=np.float64)
            con_vec44_NO_mixed_mixed_up(A,B,state,con)
        elif A.ndim == B.ndim == 4 and pair=='mixed-mixed-down':
            con = np.zeros(A.shape,dtype=np.float64)
            con_vec44_NO_mixed_mixed_down(A,B,state,con)
        elif A.ndim == B.ndim == 4 and pair=='mixed-up':
            con = np.zeros(A.shape,dtype=np.float64)
            con_vec44_NO_up_mixed(A,B,state,con)
            con *= -1
        elif A.ndim == B.ndim == 4 and pair=='mixed-down':
            con = np.zeros(A.shape,dtype=np.float64)
            con_vec44_NO_down_mixed(A,B,state,con)
            con *= -1
        elif A.ndim == B.ndim == 4 and pair=='mixed':
            con = np.zeros(A.shape,dtype=np.float64)
            con_vec44_NO_mixed(A,B,upstate,downstate,con)

    elif A.ndim != B.ndim:
        if A.ndim == 4:
            if B.ndim == 2:
                con = con42_NO(A,B,method=method,comp=comp,state=state,pair=pair)
        elif A.ndim == 2:
            if B.ndim == 4:
                con = con24_NO(A,B,method=method,comp=comp,state=state,pair=pair)
    return con

def contractNO2(A,B,method='jit',comp=False,eta=False,state=[],pair=None):
    """ General normal-ordering function: gets shape and calls appropriate contraction function. """

    if A.ndim != B.ndim:
        if A.ndim == 4:
            if B.ndim == 2:
                con = con42_NO(A,B,method=method,comp=comp,state=state,pair=pair)
        elif A.ndim == 2:
            if B.ndim == 4:
                con = con24_NO(A,B,method=method,comp=comp,state=state,pair=pair)
    return con

# Contract square matrices (matrix multiplication)
def con22(A,B,method='jit',comp=False,eta=False):
    """ Contraction function for matrices.
    
        Takes two input matrices, A and B, and contracts them according to the specified method.

        Parameters
        ----------
        A : array
            Input matrix.
        B : array
            Input matrix.
        method : string, optional
            Defines which contraction method to use.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
            (Specifially, they only compute half of the contraction and (anti)symmetrically copy it 
            to the other half.)
        comp : Bool, optional
            If method is 'jit' or 'vectorize' and either matrix is complex, comp=True will call a 
            contraction subroutine that complex conjugates appropriate terms without computing them.
        eta : Bool, optional
            If method is 'jit' or 'vectorize' and eta=True, the resulting matrix will be antisymmetrised.
            Otherwise, the result will be symmetric. Methods 'einsum' and 'tensordot' compute the full 
            matrix contraction and do not make use of symmetries, so this parameter does not affect them.
    
    """

    if method == 'einsum':
        return np.einsum('ij,jk->ik',A,B,optimize=True) - np.einsum('ki,ij->kj',B,A,optimize=True)
    elif method == 'tensordot':
        return np.tensordot(A,B,axes=1) - np.tensordot(B,A,axes=1)
    elif method == 'jit' and comp==False:
        con = np.zeros(A.shape,dtype=np.float64)
        if eta==False:
            return con_jit(A,B,con)
        elif eta==True:
            return con_jit_anti(A,B,con)
    elif method == 'jit' and comp==True:
        if eta == False:
            return con_jit_comp(A,B)
        else:
            return con_jit_anti_comp(A,B)
    elif method == 'vec' and comp == False:
        con = np.zeros(A.shape)
        con_vec(A,B,con)
        return con
    elif method == 'vec' and comp == True:
        if A.dtype==np.complex128 and B.dtype==np.complex128:
            con = np.zeros(A.shape,dtype=np.complex128)
            con_vec_comp2(A,B,con)
        elif A.dtype == np.complex128 and B.dtype == np.float64:
            con = np.zeros(A.shape,dtype=np.complex128)
            con_vec_comp(A,B,con)
        elif A.dtype == np.float64 and B.dtype == np.complex128:
            con = np.zeros(A.shape,dtype=np.complex128)
            con_vec_comp3(A,B,con)
        return con
    
# Contract rank-4 tensor with square matrix
def con42(A,B,method='jit',comp=False):
    """ Contraction function for a rank-4 tensor and a matrix (rank-2 tensor).

    Takes two input arrays, A and B, and contracts them according to the specified method.
    Note that the first array is the rank-4 tensor, and the second is the rank-2 matrix.

    Parameters
    ----------
    A : array
        Input matrix.
    B : array
        Input matrix.
    method : string, optional
        Defines which contraction method to use.
        Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
        The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        (Specifially, they only compute half of the contraction and (anti)symmetrically copy it 
        to the other half.)
    comp : Bool, optional
        If method is 'jit' or 'vectorize' and either matrix is complex, comp=True will call a 
        contraction subroutine that complex conjugates appropriate terms without computing them.
    eta : Bool, optional
        If method is 'jit' or 'vectorize' and eta=True, the resulting matrix will be antisymmetrised.
        Otherwise, the result will be symmetric. Methods 'einsum' and 'tensordot' compute the full 
        matrix contraction and do not make use of symmetries, so this parameter does not affect them.
    
    """

    if method == 'einsum':
        con = np.einsum('abcd,df->abcf',A,B,optimize=True) 
        con += -np.einsum('abcd,ec->abed',A,B,optimize=True)
        con += np.einsum('abcd,bf->afcd',A,B,optimize=True)
        con += -np.einsum('abcd,ea->ebcd',A,B,optimize=True)
    elif method == 'tensordot':
        con = - np.moveaxis(np.tensordot(A,B,axes=[0,1]),[0,1,2,3],[1,2,3,0])
        con += - np.moveaxis(np.tensordot(A,B,axes=[2,1]),[0,1,2,3],[0,1,3,2])
        con += np.moveaxis(np.tensordot(A,B,axes=[1,0]),[0,1,2,3],[0,2,3,1])
        con += np.tensordot(A,B,axes=[3,0])
    elif method == 'jit' and comp == False:
        con = con_jit42(A,B)
    elif method == 'jit' and comp == True:
        con = con_jit42_comp(A,B)
    elif method == 'vec' and comp == False:
        con = np.zeros(A.shape,dtype=np.float64)
        con_vec42(A,B,con)
    elif method == 'vec' and comp == True:
        con = np.zeros(A.shape,dtype=np.complex128)
        if A.dtype == np.float64 and B.dtype==np.complex128:
            con_vec42_comp(A,B,con)
        elif A.dtype == np.complex128 and B.dtype==np.float64:
            con_vec42_comp2(A,B,con)
        elif A.dtype == np.complex128 and B.dtype==np.complex128:
            con_vec42_comp3(A,B,con)
    
    return con

# Contract square matrix with rank-4 tensor
def con24(A,B,method='jit',comp=False,eta=False):
    """ Utility function to flip the order of matrix and tensor and re-use the function con42. """
    return -con42(B,A,method,comp)

# Double-contract rank-4 tensor with square matrix
def con42_NO(A,B,method='jit',comp=False,state=[],pair=None):
    """ Normal-ordering correction function for a rank-4 tensor and a matrix (rank-2 tensor).

    Takes two input arrays, A and B, and performs all 2-body contractions according to the 
    specified method and with respect to the specified state, which must be supplied to the function.
    Note that the first array is the rank-4 tensor, and the second is the rank-2 matrix.

    Parameters
    ----------
    A : array
        Input matrix.
    B : array
        Input matrix.:
    method : string, optional
        Defines which contraction method to use.
        Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
        The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        (Specifially, they only compute half of the contraction and (anti)symmetrically copy it 
        to the other half.)
    comp : Bool, optional
        If method is 'jit' or 'vectorize' and either matrix is complex, comp=True will call a 
        contraction subroutine that complex conjugates appropriate terms without computing them.
    eta : Bool, optional
        If method is 'jit' or 'vectorize' and eta=True, the resulting matrix will be antisymmetrised.
        Otherwise, the result will be symmetric. Methods 'einsum' and 'tensordot' compute the full 
        matrix contraction and do not make use of symmetries, so this parameter does not affect them.
    state : array
        Reference state for the computation of normal-ordering corretions.
    
    """

    if method == 'einsum':
        print('N/O CORRECTIONS NOT POSSIBLE WITH EINSUM')
        print('*** Switching method to jit ***')
        method = 'jit'
    elif method == 'tensordot':
        print('N/O CORRECTIONS NOT POSSIBLE WITH TENSORDOT')
        print('*** Switching method to jit ***')
        method = 'jit'
    elif method == 'jit' and comp == False:
        # print('jit')
        if pair == None:
            con = con_jit42_NO(A,B,state)
        elif pair == 'first':
            con = con_jit42_NO_firstpair(A,B,state)
        elif pair == 'second':
            con = con_jit42_NO_secondpair(A,B,state)
    elif method == 'jit' and comp == True:
        con = con_jit42_comp_NO(A,B,state)
    elif method == 'vec' and comp == False:
        con = np.zeros(B.shape,dtype=np.float64)
        if pair == None:
            con=con_jit42_NO(A,B,state)
        elif pair == 'first':
            con_vec42_NO_firstpair(A,B,state,con)
        elif pair == 'second':
            con_vec42_NO_secondpair(A,B,state,con)
    return con

# Double square matrix with rank-4 tensor
def con24_NO(A,B,method='jit',comp=False,eta=False,state=[],pair=None):
    """ Utility function to flip the order of matrix and tensor and re-use the function con42_NO. """
    return -con42_NO(B,A,method,comp,state,pair)

# Double-contract rank-4 tensor with square matrix
def con44_NO(A,B,method='jit',comp=False,eta=False,state=[]):
    """ Normal-ordering correction function for two rank-4 tensors.

    Takes two input arrays, A and B, and performs all 2-body contractions according to the 
    specified method and with respect to the specified state, which must be supplied to the function.

    Parameters
    ----------
    A : array
        Input matrix.
    B : array
        Input matrix.
    method : string, optional
        Defines which contraction method to use.
        Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
        The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        (Specifially, they only compute half of the contraction and (anti)symmetrically copy it 
        to the other half.)
    comp : Bool, optional
        If method is 'jit' or 'vectorize' and either matrix is complex, comp=True will call a 
        contraction subroutine that complex conjugates appropriate terms without computing them.
    eta : Bool, optional
        If method is 'jit' or 'vectorize' and eta=True, the resulting matrix will be antisymmetrised.
        Otherwise, the result will be symmetric. Methods 'einsum' and 'tensordot' compute the full 
        matrix contraction and do not make use of symmetries, so this parameter does not affect them.
    state : array
        Reference state for the computation of normal-ordering corretions.
    
    """

    if method == 'einsum':
        print('N/O CORRECTIONS NOT POSSIBLE WITH EINSUM')
    elif method == 'tensordot':
        print('N/O CORRECTIONS NOT POSSIBLE WITH TENSORDOT')
    elif method == 'jit' and comp == False:
        # if eta == False:
        con = con_jit44_NO(A,B,state)
    elif method == 'vec' and comp == False:
        con = np.zeros(A.shape,dtype=np.float64)
        con_vec44_NO(A,B,state,con)
        # elif eta==True:
        # con = con_jit44_anti_NO(A,B,state)

    return con

# Contract rank-4 tensor with square matrix
def con42_firstpair(A,B,method='jit',comp=False,eta=False):
    #print(psutil.cpu_percent(percpu=True))    
# print('con42',A.dtype,B.dtype)
    if method == 'einsum':
        # print('einsum')
        # con = np.einsum('abcd,df->abcf',A,B,optimize=True) 
        # con += -np.einsum('abcd,ec->abed',A,B,optimize=True)
        con = np.einsum('abcd,bf->afcd',A,B,optimize=True)
        con += -np.einsum('abcd,ea->ebcd',A,B,optimize=True)
    elif method == 'tensordot':
        # print('tensordot')
        con = - np.moveaxis(np.tensordot(A,B,axes=[0,1]),[0,1,2,3],[1,2,3,0])
        # con += - np.moveaxis(np.tensordot(A,B,axes=[2,1]),[0,1,2,3],[0,1,3,2])
        con += np.moveaxis(np.tensordot(A,B,axes=[1,0]),[0,1,2,3],[0,2,3,1])
        # con += np.tensordot(A,B,axes=[3,0])
    elif method == 'jit':
        con = con_jit42_firstpair(A,B)
    elif method == 'vec':
        con = np.zeros(A.shape,dtype=np.float64)
        con = con_vec42_firstpair(A,B,con)
    return con

# Contract rank-4 tensor with square matrix
def con42_secondpair(A,B,method='jit',comp=False,eta=False):

    if method == 'einsum':
        # print('einsum')
        con = np.einsum('abcd,df->abcf',A,B,optimize=True) 
        con += -np.einsum('abcd,ec->abed',A,B,optimize=True)
        # con += np.einsum('abcd,bf->afcd',A,B,optimize=True)
        # con += -np.einsum('abcd,ea->ebcd',A,B,optimize=True)
    elif method == 'tensordot':
        # print('tensordot')
        # con = - np.moveaxis(np.tensordot(A,B,axes=[0,1]),[0,1,2,3],[1,2,3,0])
        con = - np.moveaxis(np.tensordot(A,B,axes=[2,1]),[0,1,2,3],[0,1,3,2])
        # con += np.moveaxis(np.tensordot(A,B,axes=[1,0]),[0,1,2,3],[0,2,3,1])
        con += np.tensordot(A,B,axes=[3,0])
    elif method == 'jit':
        con = con_jit42_secondpair(A,B)
    elif method == 'vec':
        con = np.zeros(A.shape,dtype=np.float64)
        con_vec42_secondpair(A,B,con)
    return con


def con24_firstpair(A,B,method='jit',comp=False,eta=False):
    return -con42_firstpair(B,A,method,comp,eta)

def con24_secondpair(A,B,method='jit',comp=False,eta=False):
    return -con42_secondpair(B,A,method,comp,eta)
#------------------------------------------------------------------------------
# jit functions which return a matrix
    
@jit(float64[:,:](float64[:,:],float64[:,:],float64[:,:]),nopython=True,parallel=True,fastmath=True,cache=True,nogil=True)
# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit(A,B,C):
    """ Contract two square matrices. Computes upper half only and then symmetrises. """
    m,_=A.shape
    for i in prange(m):
        for j in range(m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]

    return C

@jit(float64[:,:](float64[:,:],float64[:,:],float64[:,:]),nopython=True,parallel=True,fastmath=True,cache=True,nogil=True)
def con_jit_anti(A,B,C):
    """ Contract two square matrices. Computes upper half only and then anti-symmetrises. """
    m,_=A.shape
    for i in prange(m):
        for j in range(m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
            # C[j,i] = -C[i,j]
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
    for i in prange(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])

    return C

@jit(float64[:,:](float64[:,:,:,:],float64[:,:],float64[:]),nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit42_NO_firstpair(A,B,state):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    C = np.zeros(B.shape,dtype=np.float64)
    m,_=B.shape
    for i in prange(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])
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
                                    C[i,j,k,q] += A[i,j,l,m]*(B[m,l,k,q]+B[k,q,m,l]-B[m,q,k,l]+B[k,l,m,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += A[l,m,i,j]*(B[m,l,k,q]+B[k,q,m,l]-B[m,q,k,l]+B[k,l,m,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -A[l,j,i,m]*(B[m,l,k,q]+B[k,l,m,q]-B[m,q,k,l]+B[k,q,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -A[i,l,m,j]*(B[k,m,l,q]+B[k,q,l,m]+B[l,m,k,q]-B[l,q,k,m])*(state[l]-state[m]) #-
                                C[i,j,k,q] +=  A[l,j,m,q]*(B[i,m,k,l]+B[i,l,k,m])*(state[l]+state[m]) #--
                                C[i,j,k,q] +=  A[i,l,k,m]*(B[m,j,l,q]+B[l,j,m,q])*(state[l]+state[m]) #--
       
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
                                    C[i,j,k,q] += A[i,j,l,m]*(B[m,l,k,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += A[l,m,i,j]*(B[m,l,k,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -A[l,j,i,m]*(B[m,l,k,q])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -A[i,l,m,j]*(B[l,m,k,q])*(state[l]-state[m]) #-

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
                                    C[i,j,k,q] += A[i,j,l,m]*(B[k,q,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += A[l,m,i,j]*(B[k,q,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -A[l,j,i,m]*(B[k,q,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -A[i,l,m,j]*(B[k,q,l,m])*(state[l]-state[m]) #-
                                
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
                                    C[i,j,k,q] += A[i,j,l,m]*(B[k,q,m,l])*(state[l]-state[m]) #+               
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
                                    C[i,j,k,q] += A[l,m,i,j]*(B[m,l,k,q])*(state[l]-state[m]) #+

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
         
#------------------------------------------------------------------------------
# guvectorize functions
# Note the different functions for different combinations of Re/Im inputs
# needed because of the explicit type declarations, not needed for jit (above)

@guvectorize([(float64[:,:],float64[:,:],float64[:,:])],'(n,n),(n,n)->(n,n)',target='cpu',nopython=True)
def con_vec(A,B,C):
    m,_=A.shape
    for i in range(m):
        for j in range(m):
            C[i,j] = 0.
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]

@guvectorize([(float64[:,:],complex128[:,:],complex128[:,:])],'(n,n),(n,n)->(n,n)',target='cpu',nopython=True)
def con_vec_comp(A,B,C):
    m,_=A.shape
    for i in range(m):
        for j in range(m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]

@guvectorize([(complex128[:,:],complex128[:,:],complex128[:,:])],'(n,n),(n,n)->(n,n)',target='cpu',nopython=True)
def con_vec_comp2(A,B,C):
    m,_=A.shape
    for i in range(m):
        for j in range(m):
            C[i,j] = 0.
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
    
@guvectorize([(float64[:,:],complex128[:,:],complex128[:,:])],'(n,n),(n,n)->(n,n)',target='cpu',nopython=True)
def con_vec_comp3(A,B,C):
    m,_=A.shape
    for i in range(m):
        for j in range(m):
            C[i,j] = 0.
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]

@guvectorize([(float64[:,:,:,:],float64[:,:],float64[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec42(A,B,C):
    m,_,_,_=A.shape
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    C[i,j,k,q] = 0.
                    for l in range(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 


@guvectorize([(float64[:,:,:,:],complex128[:,:],complex128[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec42_comp(A,B,C):
    m,_,_,_=A.shape
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    C[i,j,k,q] = 0.
                    for l in range(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
 
@guvectorize([(complex128[:,:,:,:],float64[:,:],complex128[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec42_comp2(A,B,C):
    m,_,_,_=A.shape
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    C[i,j,k,q] = 0.
                    for l in range(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 

@guvectorize([(complex128[:,:,:,:],complex128[:,:],complex128[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec42_comp3(A,B,C):
    m,_,_,_=A.shape
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    C[i,j,k,q] = 0.
                    for l in range(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 

@guvectorize([(float64[:,:,:,:],float64[:,:],float64[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec42_firstpair(A,B,C):
    m,_,_,_=A.shape
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    C[i,j,k,q] = 0.
                    for l in range(m):
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 

@guvectorize([(float64[:,:,:,:],float64[:,:],float64[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec42_secondpair(A,B,C):
    m,_,_,_=A.shape
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for q in range(m):
                    C[i,j,k,q] = 0.
                    for l in range(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]


@guvectorize([(float64[:,:,:,:],float64[:,:],float64[:],float64[:,:])],'(n,n,n,n),(n,n),(n)->(n,n)',target='cpu',nopython=True)
def con_vec42_NO(A,B,state,C):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    m,_=B.shape
    C = np.zeros(B.shape,dtype=np.float64)
    for i in range(m):
        for j in range(m):
            # C[i,j] = 0.
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])
                        C[i,j] += A[k,q,i,j]*B[q,k]*(state[k]-state[q])
                        C[i,j] += -A[k,j,i,q]*B[q,k]*(state[k]-state[q])
                        C[i,j] += A[i,q,k,j]*B[q,k]*(state[k]-state[q])

@guvectorize([(float64[:,:,:,:],float64[:,:],float64[:],float64[:,:])],'(n,n,n,n),(n,n),(n)->(n,n)',target='cpu',nopython=True)
def con_vec42_NO_secondpair(A,B,state,C):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    m,_=B.shape
    for i in range(m):
        for j in range(m):
            C[i,j] = 0.
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])

@guvectorize([(float64[:,:,:,:],float64[:,:],float64[:],float64[:,:])],'(n,n,n,n),(n,n),(n)->(n,n)',target='cpu',nopython=True)
def con_vec42_NO_firstpair(A,B,state,C):
    """ 2-point contractions of a rank-4 tensor with a square matrix. Computes upper half only and then symmetrises. """
    m,_=B.shape
    for i in range(m):
        for j in range(m):
            C[i,j] = 0.
            for k in range(m):
                for q in range(m):
                    if state[k] != state[q]:
                        C[i,j] += A[i,j,k,q]*B[q,k]*(state[k]-state[q])

@guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec44_NO(A,B,state,C):
    m0,_,_,_=A.shape
    for i in range(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        C[i,j,k,q] = 0.
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += A[i,j,l,m]*(B[m,l,k,q]+B[k,q,m,l]-B[m,q,k,l]+B[k,l,m,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += A[l,m,i,j]*(B[m,l,k,q]+B[k,q,m,l]-B[m,q,k,l]+B[k,l,m,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -A[l,j,i,m]*(B[m,l,k,q]+B[k,l,m,q]-B[m,q,k,l]+B[k,q,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -A[i,l,m,j]*(B[k,m,l,q]+B[k,q,l,m]+B[l,m,k,q]-B[l,q,k,m])*(state[l]-state[m]) #-
                                C[i,j,k,q] +=  A[l,j,m,q]*(B[i,m,k,l]+B[i,l,k,m])*(state[l]+state[m]) #--
                                C[i,j,k,q] +=  A[i,l,k,m]*(B[m,j,l,q]+B[l,j,m,q])*(state[l]+state[m]) #--

@guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec44_NO_up_mixed(A,B,state,C):
    m0,_,_,_=A.shape
    for i in range(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += A[i,j,l,m]*(B[m,l,k,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += A[l,m,i,j]*(B[m,l,k,q])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -A[l,j,i,m]*(B[m,l,k,q])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -A[i,l,m,j]*(B[l,m,k,q])*(state[l]-state[m]) #-

@guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec44_NO_down_mixed(A,B,state,C):
    m0,_,_,_=A.shape
    for i in range(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += A[i,j,l,m]*(B[k,q,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += A[l,m,i,j]*(B[k,q,m,l])*(state[l]-state[m]) #+
                                    C[i,j,k,q] += -A[l,j,i,m]*(B[k,q,m,l])*(state[l]-state[m]) #-
                                    C[i,j,k,q] += -A[i,l,m,j]*(B[k,q,l,m])*(state[l]-state[m]) #-

@guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec44_NO_mixed(A,B,upstate,downstate,C):
    m0,_,_,_=A.shape
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

@guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec44_NO_mixed_mixed_up(A,B,state,C):
    m0,_,_,_=A.shape
    for i in range(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += A[i,j,l,m]*(B[k,q,m,l])*(state[l]-state[m]) #+               

@guvectorize([(float64[:,:,:,:],float64[:,:,:,:],float64[:],float64[:,:,:,:])],'(n,n,n,n),(n,n,n,n),(n)->(n,n,n,n)',target='cpu',nopython=True)
def con_vec44_NO_mixed_mixed_down(A,B,state,C):
    m0,_,_,_=A.shape
    for i in range(m0):
        for j in range(m0):
            for k in range(m0):
                for q in range(m0):
                        # Indices to be summed over
                        for l in range(m0):
                            for m in range(m0):
                                if state[l] != state[m]:
                                    C[i,j,k,q] += A[l,m,i,j]*(B[m,l,k,q])*(state[l]-state[m]) #+