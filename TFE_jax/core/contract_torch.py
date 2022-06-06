import os
from multiprocessing import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count())) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count())) # set number of MKL threads to run in parallel

import numpy as np
# import psutil
#import cupy as np
import torch
from numba import jit,prange,guvectorize,float64,complex64, complex128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#------------------------------------------------------------------------------
# Tensor contraction subroutines

# p=psutil.Process()
        
# General contraction function
def contract(A,B,method='jit',comp=False,eta=False):

            if A.ndim == B.ndim == 2:
                con = con22(A,B,method,comp,eta)
            if A.ndim != B.ndim:
                if A.ndim == 4:
                    if B.ndim == 2:
                        con = con42(A,B,method,comp)
                if A.ndim == 2:
                    if B.ndim == 4:
                        con = con24(A,B,method,comp)
            return con

# Contract square matrices (matrix multiplication)
def con22(A,B,method='jit',comp=False,eta=False):
    # print(p.cpu_percent())
    # print('con22',A.dtype,B.dtype,comp)
    if method == 'einsum':
        # print('einsum')
        return torch.einsum('ij,jk->ik',A,B) - torch.einsum('ki,ij->kj',B,A)
    elif method == 'tensordot':
        # print('tensordot')
        return torch.tensordot(A,B,axes=1) - torch.tensordot(B,A,axes=1)
    elif method == 'jit' and comp==False:
        # print('jit')
        con = torch.zeros(A.shape,dtype=torch.float32,device=device)
        if eta==False:
            return con_jit(A,B,con)
        elif eta==True:
            return con_jit_anti(A,B,con)
        #con_jit(A,B,con)
        #return con
    elif method == 'jit' and comp==True:
        if eta == False:
            return con_jit_comp(A,B)
        else:
            return con_jit_anti_comp(A,B)
    # elif method == 'cython':
    #     return con_cython(A,B)
    
# Contract rank-4 tensor with square matrix
def con42(A,B,method='jit',comp=False,eta=False):
    # print(p.cpu_percent())
    # print('con42',A.dtype,B.dtype)
    if method == 'einsum':
        # print('einsum')
        con = torch.einsum('abcd,df->abcf',A,B) 
        con += -torch.einsum('abcd,ec->abed',A,B)
        con += torch.einsum('abcd,bf->afcd',A,B)
        con += -torch.einsum('abcd,ea->ebcd',A,B)
    elif method == 'tensordot':
        # print('tensordot')
        con = - torch.moveaxis(torch.tensordot(A,B,axes=[0,1]),[0,1,2,3],[1,2,3,0])
        con += - torch.moveaxis(torch.tensordot(A,B,axes=[2,1]),[0,1,2,3],[0,1,3,2])
        con += torch.moveaxis(torch.tensordot(A,B,axes=[1,0]),[0,1,2,3],[0,2,3,1])
        con += torch.tensordot(A,B,axes=[3,0])
    elif method == 'jit' and comp == False:
        # print('jit')
        # if eta == False:
        con = con_jit42(A,B)
        # elif eta==True:
        #     con = con_jit42_anti(A,B)
    elif method == 'vec' and comp == False:
        # print('jit')
        # print(A.dtype,B.dtype)
        con = np.zeros(A.shape,dtype=np.float32)
        con_vec42(A,B,con)
        return con
    elif method == 'vec' and comp == True:
        # print('jit')
        con = np.zeros(A.shape,dtype=np.complex128)
        if A.dtype == np.float64 and B.dtype==np.complex128:
            con_vec42_comp(A,B,con)
        elif A.dtype == np.complex128 and B.dtype==np.float64:
            con_vec42_comp2(A,B,con)
        elif A.dtype == np.complex128 and B.dtype==np.complex128:
            con_vec42_comp3(A,B,con)
        return con
    elif method == 'jit' and comp == True:
        if eta == False:
            con = con_jit42_comp(A,B)
        else:
            con = con_jit42_anti_comp(A,B)
    # elif method == 'cython':
    #     con = con_cython42(A,B)
    
    return con

# Contract square matrix with rank-4 tensor
def con24(A,B,method='jit',comp=False,eta=False):
    return -con42(B,A,method,comp,eta)

#------------------------------------------------------------------------------
# jit functions
    
# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
#@guvectorize([(float64[:,:],float64[:,:],float64[:,:])],'(n,n),(n,n)->(n,n)',target='parallel',nopython=True)
#@cuda.jit
@torch.jit.script
def con_jit(A,B,C):
    
    m,_=A.shape
    for i in range(m):
        for j in range(i,m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                # C[j,i] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                # C[j,i] += A[j,k]*B[k,i] - B[j,k]*A[k,i]
            C[j,i] = C[i,j]

    return C


# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
@torch.jit.script
def con_jit_anti(A,B,C):
    
    m,_=A.shape
    for i in range(m):
        for j in range(i,m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
            C[j,i] = -C[i,j]
    return C

# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
# @torch.jit.script
def con_jit_comp(A,B):
    C = torch.zeros(A.shape,dtype=torch.complex64)
    m,_=A.shape
    for i in range(m):
        for j in range(i,m):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                # C[j,i] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                # C[j,i] += A[j,k]*B[k,i] - B[j,k]*A[k,i]
            C[j,i] = np.conj(C[i,j])

    return C

# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
# @torch.jit.script
def con_jit_anti_comp(A,B):
    C = torch.zeros(A.shape,dtype=torch.complex64)
    m,_=A.shape
    for i in prange(m):
        for j in prange(i,m):
            for k in prange(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                # C[j,i] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                # C[j,i] += A[j,k]*B[k,i] - B[j,k]*A[k,i]
            C[j,i] = -np.conj(C[i,j])

    return C

# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
@torch.jit.script
def con_jit42(A,B):
    C = torch.zeros(A.shape,dtype=torch.float32)
    D = torch.zeros(A.shape,dtype=torch.int8)
    m,_,_,_=A.shape
    for i in range(m):
        for j in range(m):
            # if abs(i-j)<m//2:
            for k in range(m):
                for q in range(m):
                    # if abs(k-q)<m//2:
                    # if D[i,j,k,q] == 0:
                        for l in range(m):
                            C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                            C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                            C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                            C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                    
                        # # C[k,q,i,j] = C[i,j,k,q]
                        # C[q,k,j,i] = C[i,j,k,q]
                        # C[k,j,i,q] = -C[i,j,k,q]
                        # C[i,q,k,j] = -C[i,j,k,q]
                        # D[i,j,k,q] = 1
                        # D[q,k,j,i] = 1
                        # D[k,j,i,q] = 1
                        # D[i,q,k,j] = 1
                        
    return C

# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
@torch.jit.script
def con_jit42_anti(A,B):
    C = torch.zeros(A.shape,dtype=torch.float32)
    D = torch.zeros(A.shape,dtype=torch.int8)
    m,_,_,_=A.shape
    for i in range(m):
        for j in range(i,m):
            # if abs(i-j)<m//2:
            for k in range(j,m):
                for q in range(k,m):
                    # if abs(k-q)<m//2:
                    if D[i,j,k,q] == 0:
                        for l in range(m):
                            C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                            C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                            C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                            C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                    
                        # C[k,q,i,j] = C[i,j,k,q]
                        C[q,k,j,i] = -C[i,j,k,q]
                        D[i,j,k,q] = 1
                        D[q,k,j,i] = 1
    return C

@jit(nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit42_comp(A,B):
    C = np.zeros(A.shape,dtype=np.complex64)
    # D = np.zeros(A.shape,dtype=np.int8)
    m,_,_,_=A.shape
    for i in prange(m):
        for j in prange(m):
            # if abs(i-j)<m//2:
            for k in prange(m):
                for q in prange(m):
                    # if abs(k-q)<m//2:
                    # if D[i,j,k,q] == 0:
                        for l in prange(m):
                            C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                            C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                            C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                            C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                    
                        # # C[k,q,i,j] = C[i,j,k,q]
                        # C[q,k,j,i] = np.conj(C[i,j,k,q])
                        # C[k,j,i,q] = -C[i,j,k,q]
                        # C[i,q,k,j] = -C[i,j,k,q]
                        # D[i,j,k,q] = 1
                        # D[q,k,j,i] = 1
                        # D[k,j,i,q] = 1
                        # D[i,q,k,j] = 1
    return C

@jit(nopython=True,parallel=True,fastmath=True,cache=True)
def con_jit42_anti_comp(A,B):
    C = np.zeros(A.shape,dtype=np.complex64)
    D = np.zeros(A.shape,dtype=np.int8)
    m,_,_,_=A.shape
    for i in prange(m):
        for j in prange(i,m):
            # if abs(i-j)<m//2:
            for k in prange(j,m):
                for q in prange(k,m):
                    # if abs(k-q)<m//2:
                    if D[i,j,k,q] == 0:
                        for l in prange(m):
                            C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                            C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                            C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                            C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                    
                        # C[k,q,i,j] = C[i,j,k,q]
                        C[q,k,j,i] = -np.conj(C[i,j,k,q])
                        D[i,j,k,q] = 1
                        D[q,k,j,i] = 1
    return C



# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
# def con_jit42(A,B):
#     C = np.zeros(A.shape,dtype=np.float32)
#     m,_,_,_=A.shape
#     for i in prange(m):
#         for j in prange(m):
#             # if abs(i-j)<m//2:
#             for k in prange(m):
#                 for q in prange(m):
#                     # if abs(k-q)<m//2:
#                     for l in prange(m):
#                         C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
#                         C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
#                         C[i,j,k,q] += A[i,l,k,q]*B[l,j]
#                         C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                
                        
#     return C

# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
# def con_jit42_test(A,B):
#     C = np.zeros(A.shape,dtype=np.float32)
#     m,_,_,_=A.shape
#     for i in prange(m):
#         for j in prange(m,i):
#             if abs(i-j)<m//2:
#                 for k in prange(m):
#                     for q in prange(m,k):
#                         if abs(k-q)<m//2:
#                             for l in prange(m):
#                                 C[i,j,k,q] += A[i,j,k,l]*B[l,q]
#                                 C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
#                                 C[i,j,k,q] += A[i,l,k,q]*B[l,j]
#                                 C[i,j,k,q] += -A[l,j,k,q]*B[i,l]

#                                 C[j,i,k,q] += A[j,i,k,l]*B[l,q]
#                                 C[j,i,k,q] += -A[j,i,l,q]*B[k,l]
#                                 C[j,i,k,q] += A[j,l,k,q]*B[l,i]
#                                 C[j,i,k,q] += -A[l,i,k,q]*B[j,l]
    
#                                 C[i,j,q,k] += A[i,j,q,l]*B[l,k]
#                                 C[i,j,q,k] += -A[i,j,l,k]*B[q,l]
#                                 C[i,j,q,k] += A[i,l,q,k]*B[l,j]
#                                 C[i,j,q,k] += -A[l,j,q,k]*B[i,l]

#                                 C[j,i,q,k] += A[j,i,q,l]*B[l,k]
#                                 C[j,i,q,k] += -A[j,i,l,k]*B[q,l]
#                                 C[j,i,q,k] += A[j,l,q,k]*B[l,i]
#                                 C[j,i,q,k] += -A[l,i,q,k]*B[j,l]

#     return C

# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
# def con_jit42_comp(A,B):
#     C = np.zeros(A.shape,dtype=np.complex64)
#     m,_,_,_=A.shape
#     for i in prange(m):
#         for j in prange(m):
#             if abs(i-j)<m//2:
#                 for k in prange(m):
#                     for q in prange(m):
#                         if abs(k-q)<m//2:
#                             for l in prange(m):
#                                 C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
#                                 C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
#                                 C[i,j,k,q] += A[i,l,k,q]*B[l,j]
#                                 C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                        
#     return C

#------------------------------------------------------------------------------
# guvectorize functions
# Note the different functions for different combinations of Re/Im inputs
# needed because of the explicit type declarations, not needed for jit (above)

# @jit(nopython=True,parallel=True,fastmath=True)
@guvectorize([(float64[:,:],float64[:,:],float64[:,:])],'(n,n),(n,n)->(n,n)',target='parallel',nopython=True)
#@cuda.jit
def con_vec(A,B,C):
    
    m,_=A.shape
    for i in prange(m):
        for j in prange(m):
            for k in prange(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                
    # return C
    
# @jit(nopython=True,parallel=True,fastmath=True)
@guvectorize([(float64[:,:],complex128[:,:],complex128[:,:])],'(n,n),(n,n)->(n,n)',target='parallel',nopython=True)
#@cuda.jit
def con_vec_comp(A,B,C):
    
    m,_=A.shape
    for i in prange(m):
        for j in prange(m):
            for k in prange(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                
    # return C
    
@guvectorize([(complex128[:,:],complex128[:,:],complex128[:,:])],'(n,n),(n,n)->(n,n)',target='parallel',nopython=True)
#@cuda.jit
def con_vec_comp2(A,B,C):
    
    m,_=A.shape
    for i in prange(m):
        for j in prange(m):
            for k in prange(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                
    # return C
    
@guvectorize([(float64[:,:],complex128[:,:],complex128[:,:])],'(n,n),(n,n)->(n,n)',target='parallel',nopython=True)
#@cuda.jit
def con_vec_comp3(A,B,C):
    
    m,_=A.shape
    for i in prange(m):
        for j in prange(m):
            for k in prange(m):
                C[i,j] += A[i,k]*B[k,j] - B[i,k]*A[k,j]
                
    # return C



@guvectorize([(float64[:,:,:,:],float64[:,:],float64[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='parallel',nopython=True)
# @jit(nopython=True,parallel=True,fastmath=True)
def con_vec42(A,B,C):
    
    m,_,_,_=A.shape
    for i in prange(m):
        for j in prange(m):
            for k in prange(m):
                for q in prange(m):
                    for l in prange(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                        
    # return C
    
@guvectorize([(float64[:,:,:,:],complex128[:,:],complex128[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='parallel',nopython=True)
# @jit(nopython=True,parallel=True,fastmath=True)
def con_vec42_comp(A,B,C):
    
    m,_,_,_=A.shape
    for i in prange(m):
        for j in prange(m):
            for k in prange(m):
                for q in prange(m):
                    for l in prange(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                        
    # return C
    
        
@guvectorize([(complex128[:,:,:,:],float64[:,:],complex128[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='parallel',nopython=True)
# @jit(nopython=True,parallel=True,fastmath=True)
def con_vec42_comp2(A,B,C):
    
    m,_,_,_=A.shape
    for i in prange(m):
        for j in prange(m):
            for k in prange(m):
                for q in prange(m):
                    for l in prange(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                        
    # return 
    
@guvectorize([(complex128[:,:,:,:],complex128[:,:],complex128[:,:,:,:])],'(n,n,n,n),(n,n)->(n,n,n,n)',target='parallel',nopython=True)
# @jit(nopython=True,parallel=True,fastmath=True)
def con_vec42_comp3(A,B,C):
    
    m,_,_,_=A.shape
    for i in prange(m):
        for j in prange(m):
            for k in prange(m):
                for q in prange(m):
                    for l in prange(m):
                        C[i,j,k,q] += A[i,j,k,l]*B[l,q] 
                        C[i,j,k,q] += -A[i,j,l,q]*B[k,l]
                        C[i,j,k,q] += A[i,l,k,q]*B[l,j]
                        C[i,j,k,q] += -A[l,j,k,q]*B[i,l] 
                        
    # return 
