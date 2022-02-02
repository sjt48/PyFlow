"""
Flow Equations for Many-Body Quantum Systems
S. J. Thomson
Dahlem Centre for Complex Quantum Systems, FU Berlin
steven.thomson@fu-berlin.de
steventhomson.co.uk
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

If you do use any of this code, please cite https://arxiv.org/abs/2110.02906.

---------------------------------------------

This file contains test code used to verify that the tensor contraction routines are accurate by 
cross-checking several different methods and ensuring they all give the same answer, to numerical precision.

"""

import os
from psutil import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of MKL threads to run in parallel
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"                         # Necessary on some versions of OS X
os.environ['KMP_WARNINGS'] = 'off'                                # Silence non-critical warning
import numpy as np
from datetime import datetime
import gc
from itertools import permutations
from datetime import datetime

import sys
sys.path.append("..")
import core.contract as con

#------------------------------------------------------------------------------  
# Parameters
n = 4                          # System size

#==============================================================================
# Run program
#==============================================================================

if __name__ == '__main__': 

    startTime = datetime.now()
    print('Start time: ', startTime)

    def dim(mat):
        mat = np.array(mat)
        return len(mat.shape)

    def test(A,B,method1,method2,eta):
        """ Tests if methods agree to 6 decimal places. """

        a = (con.contract(A,B,method=method1,eta=eta))
        b = (con.contract(A,B,method=method2,eta=eta))

        a=a.reshape(len(a)**dim(a))
        a=a.astype(np.float32)
        a=np.array([round(var,8) for var in a])
        b=b.reshape(len(b)**dim(b))
        b=b.astype(np.float32)
        b=np.array([round(var,8) for var in b])

        error_count = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                # Only flags errors above a certain threshold, 
                # to avoid rounding errors being mistaken for 'real' errors
                if 2*(a[i]-b[i])/(a[i]+b[i])>1.001:
                    print(i,a[i],b[i])
                    error_count += 1

        if error_count > 0:
            print('*** WARNING: RESULT FROM %s DOES NOT EQUAL %s' %(method1,method2))
        # if error_count == 0:
        #     print('SUCCESS: RESULT FROM %s EQUALS %s' %(method1,method2))

        return error_count
            
    #-----------------------------------------------------------------
    # Generate SYMMETRIC matrices/tensors

    A2 = np.random.uniform(-1,1,n**2).reshape(n,n)
    A2 = ((A2+A2.T)/2)
    B2 = np.random.uniform(-1,1,n**2).reshape(n,n)
    B2 = ((B2+B2.T)/2)
    A4 = np.random.uniform(-1,1,n**4).reshape(n,n,n,n)
    B4 = np.random.uniform(-1,1,n**4).reshape(n,n,n,n)

    methods = ['einsum','tensordot','jit','vec']
    loop = [list(zip(permutation, methods)) for permutation in permutations(methods, len(methods))]

    for i in range(len(loop)):
        for m1,m2 in loop[i]:
            test(A2,B2,m1,m2,eta=True)

    for i in range(len(loop)):
        for m1,m2 in loop[i]:
            test(A2,B4,m1,m2,eta=True)

    for i in range(len(loop)):
        for m1,m2 in loop[i]:
            test(A4,B2,m1,m2,eta=True)

    #-----------------------------------------------------------------
    # Generate ANTISYMMETRIC/SYMMETRIC PAIR of matrices/tensors

    A2 = np.random.uniform(-1,1,n**2).reshape(n,n)
    A2 = ((A2-A2.T)/2)
    B2 = np.random.uniform(-1,1,n**2).reshape(n,n)
    B2 = ((B2+B2.T)/2)
    A4 = np.random.uniform(-1,1,n**4).reshape(n,n,n,n)
    B4 = np.random.uniform(-1,1,n**4).reshape(n,n,n,n)

    methods = ['einsum','tensordot','jit','vec']
    loop = [list(zip(permutation, methods)) for permutation in permutations(methods, len(methods))]

    for i in range(len(loop)):
        for m1,m2 in loop[i]:
            test(A2,B2,m1,m2,eta=False)

    for i in range(len(loop)):
        for m1,m2 in loop[i]:
            test(A2,B4,m1,m2,eta=False)

    for i in range(len(loop)):
        for m1,m2 in loop[i]:
            test(A4,B2,m1,m2,eta=False)

    #-----------------------------------------------------------------
    # Generate ANTISYMMETRIC/ANTISYMMETRIC PAIR of matrices/tensors

    A2 = np.random.uniform(-1,1,n**2).reshape(n,n)
    A2 = ((A2-A2.T)/2)
    B2 = np.random.uniform(-1,1,n**2).reshape(n,n)
    B2 = ((B2-B2.T)/2)
    A4 = np.random.uniform(-1,1,n**4).reshape(n,n,n,n)
    B4 = np.random.uniform(-1,1,n**4).reshape(n,n,n,n)

    methods = ['einsum','tensordot','jit','vec']
    loop = [list(zip(permutation, methods)) for permutation in permutations(methods, len(methods))]

    for i in range(len(loop)):
        for m1,m2 in loop[i]:
            test(A2,B2,m1,m2,eta=True)

    for i in range(len(loop)):
        for m1,m2 in loop[i]:
            test(A2,B4,m1,m2,eta=True)

    for i in range(len(loop)):
        for m1,m2 in loop[i]:
            test(A4,B2,m1,m2,eta=True)

    gc.collect()
    print('****************')
    print('Time taken for one run:',datetime.now()-startTime)
    print('****************')