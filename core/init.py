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

---------------------------------------------

This file contains the code used to initialise the Hamiltonian for a variety of choices of disorder.

"""

import os
from psutil import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of MKL threads to run in parallel
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
import numpy as np
from sympy import prime

def Hinit(n,d,J,dis_type,x=0):
    """ Generate the non-interacting part of the Hamiltonian with the specified on-site potential. """

    np.random.seed()
    print('Choice of potential = %s' %dis_type)

    #-----------------------------------------------------------------
    # Non-interacting matrices
    H0 = np.zeros((n,n),dtype=np.float32)
    if dis_type == 'random':
        for i in range(n):
            # Initialise Hamiltonian with random on-site terms
            H0[i,i] = np.random.uniform(-d,d)
    elif dis_type == 'test':
        # For testing purposes: fixed list of randomly generated numbers for method comparisons
        randomlist = [2.4979359120556666, -4.477621238657079, -4.448810326437316, -3.6115452666436543, -1.2802110535766298, -3.336862075297363, -0.3370611440832194, -3.8232260796601523, -0.5134617674857918, 1.32895294857477]
        for i in range(n):
            H0[i,i] = randomlist[i]
    elif dis_type == 'linear':
        for i in range(n):
            # Initialise Hamiltonian with linearly increasing on-site terms
            H0[i,i] = d
    elif dis_type == 'curved':
        for i in range(n):
            # Initialise Hamiltonian with linearly increasing on-site terms, plus some small curvature
            H0[i,i] = d*i+x*(i/n)**2
    elif dis_type == 'prime':
        for i in range(n):
            # Initialise Hamiltonian with square root prime numbers as on-site terms
            H0[i,i] = d*np.sqrt(prime(i+1))
    elif dis_type == 'QPgolden':
        phase = np.random.uniform(-np.pi,np.pi)
        print('phase = ', phase)
        phi = (1.+np.sqrt(5.))/2.
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
    elif dis_type == 'QPsilver':
        phase = np.random.uniform(-np.pi,np.pi)
        print('phase = ', phase)
        phi = 1.+np.sqrt(2.)
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
    elif dis_type == 'QPbronze':
        phase = np.random.uniform(-np.pi,np.pi)
        print('phase = ', phase)
        phi = (3.+np.sqrt(13.))/2.
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
    elif dis_type == 'QPrandom':
        phi = (1.+np.sqrt(5.))/2.
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + np.random.uniform(-np.pi,np.pi))
    elif dis_type == 'QPtest':
        print('**** FIXED PHASE FOR TESTING - DISABLE FOR REAL DATA ****')
        phase=0.
        phi = (1.+np.sqrt(5.))/2.
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
    # Initialise V0 with nearest-neighbour hopping
    V0 = np.diag(J*np.ones(n-1,dtype=np.float32),1) + np.diag(J*np.ones(n-1,dtype=np.float32),-1)
            
    return H0,V0

def Hint_init(n,delta):
    # Interaction tensors
    Hint = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            if abs(i-j)==1:
                # Initialise nearest-neighbour interactions
                Hint[i,i,j,j] = 0.5*delta
    
    # Initialise off-diagonal quartic tensor (empty)
    Vint = np.zeros((n,n,n,n),dtype=np.float32)

    return Hint,Vint
 
def namevar(dis_type,dyn,norm,n):
    if norm == True:
        nm = 'NO'
    else:
        nm = 'PT'
     # Make directory to store data
    if dyn == False:
        if not os.path.exists('%s/%s/%s/dataN%s' %(dis_type,nm,'static',n)):
            os.makedirs('%s/%s/%s/dataN%s' %(dis_type,nm,'static',n))
        namevar = '%s/%s/%s/dataN%s' %(dis_type,nm,'static',n) 
    elif dyn == True:
        if not os.path.exists('%s/%s/%s/dataN%s' %(dis_type,nm,'dyn',n)):
            os.makedirs('%s/%s/%s/dataN%s' %(dis_type,nm,'dyn',n))
        namevar = '%s/%s/%s/dataN%s' %(dis_type,nm,'dyn',n)
    
    return namevar
