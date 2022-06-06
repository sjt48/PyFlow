#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for FU Berlin HPC Cluster

"""

import numpy as np
import h5py
from scipy.integrate import odeint

dis_type = 'QP'             # Specify potential: `random' or `QP'
n = 64                     # System size
d = 15.                      # Quasi-disorder strength
J = 1.                      # Nearest-neighbour hopping strength

np.random.seed()            # Re-seed random number generator

# Generate list of flow time steps to store
dl_list = np.linspace(0,50,300,endpoint=True)

# Non-interacting matrices
H0 = np.zeros((n,n),dtype=np.float64)
if dis_type == 'random':
    for i in range(n):
    # Initialise Hamiltonian with random on-site terms
        H0[i,i] = np.random.uniform(-d,d)
elif dis_type == 'QP':
    phase = np.random.uniform(-np.pi,np.pi) # Random phase
    phi = (1.+np.sqrt(5.))/2.               # Golden ratio
    for i in range(n):
    # Initialise Hamiltonian with quasiperiodic on-site terms
        H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
        
# Initialise V0 with nearest-neighbour hopping along leading diagonals
V0 = np.diag(J*np.ones(n-1,dtype=np.float64),1) 
V0 += np.diag(J*np.ones(n-1,dtype=np.float64),-1)

# Function to contract square matrices (matrix multiplication)
def comm(A,B):
    return np.einsum('ik,kj->ij',A,B) - np.einsum('ik,kj->ij',B,A)

def nonint_ode(H,l):
    n= int(np.sqrt(len(H)))
    H = H.reshape(n,n)
    H0 = np.diag(np.diag(H))
    V0 = H - H0
    eta = comm(H0,V0)
    sol = comm(eta,H)

    return sol.reshape(n**2)

sol = odeint(nonint_ode,(H0+V0).reshape(n**2),dl_list)

eig=np.sort(np.diag(sol[-1].reshape(n,n)))
print('Flow eigenvalues', eig)
print('NumPy eigenvalues', np.sort(np.linalg.eigvalsh(H0+V0)))

sol=sol.reshape(len(sol),n,n)

# with h5py.File('/scratch/st1607fu/test.h5','w') as hf:
#     hf.create_dataset('sol',data=sol[-1])