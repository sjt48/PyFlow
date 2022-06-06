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

This file contains exact diagonalisation code making use of the QuSpin package, which is used to 
benchmark the results of the flow equation approach.

"""

import os
from psutil import cpu_count
# Import ED code from QuSpin
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.tools.measurements import ED_state_vs_time
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spin basis
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count(logical=False))) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count(logical=False))) # set number of MKL threads to run in parallel
import numpy as np


def ED(n,H0,J0,delta,times,dyn,imbalance):
    """ Exact diagonalisation function, using the QuSpin package.
    
        See the QuSpin documentation for further details on the algorithms and notation used.

        Parameters
        ----------
        n : int
            Linear system size.
        H0 : array
            Diagonal part of Hamiltonian, with disorder along the diagonal.
        J0 : float
            Hopping amplitude.
        delta : float
            Interaction strength.
        times : array
            List of times for the dynamical evolution.
        dyn : bool
            Whether or not to compute the dynamics.
        imbalance : bool
            If dynamics are true, whether or not to compute imbalance or single-site dynamics.
    
     """

    hlist = np.diag(H0)
    J = [[J0,i,i+1] for i in range(n-1)]
    J2 = [[-J0,i,i+1] for i in range(n-1)]
    Delta = [[delta,i,i+1] for i in range(n-1)]
    h = [[hlist[i],i] for i in range(n)]
    static = [["n",h],["+-",J],["-+",J2],["nn",Delta]]

    dynamic=[]
    no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}

    basis = spinless_fermion_basis_1d(n)
    H = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)
    E1,V1 = H.eigh()

    if dyn == True:
        st = "".join("10" for i in range(n//2))
        iDH = basis.index(st)
        psi1 = np.zeros(basis.Ns)
        psi1[iDH] = 1.0

        psi1_t = ED_state_vs_time(psi1,E1,V1,times,iterate=False)
        
        if imbalance == False:
            # Time evolution of observables
            n_list = [hamiltonian([["n",[[1.0,n//2]]]],[],basis=basis,dtype=np.complex64,**no_checks)]
            n_t = np.vstack([n.expt_value(psi1_t).real for n in n_list]).T
        elif imbalance == True:
            imblist = np.zeros((n,len(times)))
            for site in range(n):
                 # Time evolution of observables
                n_list = [hamiltonian([["n",[[1.0,site]]]],[],basis=basis,dtype=np.complex64,**no_checks)]
                n_t = np.vstack([n.expt_value(psi1_t).real for n in n_list]).T
                imblist[site] = ((-1)**site)*n_t[::,0]/n
            n_t = 2*np.sum(imblist,axis=0)

        return [E1,n_t]
    
    else:
        return [E1,0]
