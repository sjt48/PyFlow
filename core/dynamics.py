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

This file contains all of the functions used to compute the time evolution of observables 
in the diagonal (l -> infinity) basis.

"""

import os
from multiprocessing import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count())) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count())) # set number of MKL threads to run in parallel
import numpy as np
from .contract import contract
import gc, copy
from numba import jit,complex128,prange
from scipy.integrate import ode

# =============================================================================

# Weights for numerical integration
def w(q,steps):
    """ Weight function for numerical integration - deprecated, not currently in use. """
    nsteps = steps+1
    # Trapzoidal rule
    if nsteps == 2:
        return 0.5
    elif nsteps == 3:
        if q==0 or q ==2:
            return 1./3.
        elif q == 1:
            return 4./3.
    elif nsteps == 4:
        if q==0 or q==3:
            return 3./8.
        elif q ==1 or q==2:
            return 9./8.
    elif nsteps == 5:
        if q==0 or q==4:
            return 1./3.
        elif q==1 or q==3:
            return 4./3.
        elif q==2:
            return 2./3.
    elif nsteps > 5:
        if q==0 or q==nsteps:
            return 3./8.
        elif q==1 or q==nsteps-1:
            return 7./6.
        elif q ==2 or q==nsteps-2:
            return 23./24.
        else:
            return 1.

def n_evo(t,nlist,y,n,method='jit'):
        """ Function to compute the RHS of the Heisenberg equation, dn/dt = i[H,n]
        
            Parameters
            ----------
            t : float
                Time (not explicitly used, but needed for integrator).
            nlist : array
                Operator to be time-evolved.
            y : array
                Diagonal Hamiltonian used to generate the dynamics.
            n : int
                Linear system size.
            method : string, optional
                Specify which method to use to generate the RHS of the Heisenberg equation.
                Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
                The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        """
        
        # Extract quadratic part of Hamiltonian from input array y
        H = y[0:n**2]
        H = H.reshape(n,n)
        
        # Extract quadratic part of density operator from input array nlist
        nlist = nlist[0:len(y)]+1j*nlist[len(y):]
        n2 = nlist[:n**2]
        n2 = n2.reshape(n,n)

        # print('*****')
        # print(H)
        # print(n2)
        # print('*****')

        # If the system is interacting, extract interacting components of H and n
        if len(y) > n**2:
            Hint = y[n**2::]
            Hint = Hint.reshape(n,n,n,n)
            nint = nlist[n**2::]
            nint = nint.reshape(n,n,n,n)

        # Perform time evolution of non-interacting terms
        sol = 1j*contract(H,n2,eta=True,method=method,comp=True)
        # sol = 1j*(H@n2-n2@H)
        sol0 = np.zeros(len(y),dtype=np.complex128)
        sol0[:n**2] = sol.reshape(n**2)
        
        # If interacting, perform time evolution of interacting terms
        if len(y) > n**2:
            sol2 = 1j*contract(Hint,n2,method=method,comp=True) + 1j*contract(H,nint,method=method,comp=True)
            sol0[n**2:] = sol2.reshape(n**4)

        sol_complex = np.zeros(2*len(y),dtype=np.float64)
        sol_complex[0:len(y)] = sol0.real
        sol_complex[len(y)::] = sol0.imag

        return sol_complex

def dyn_con(n,num,y,tlist,method='jit'):
    """ Function to compute the RHS of the Heisenberg equation, dn/dt = i[H,n], using matrix/tensor contractions.
    
        Parameters
        ----------
        n : int
            Linear system size.
        num : array
            Operator to be time-evolved.
        y : array
            Diagonal Hamiltonian used to generate the dynamics.
        tlist : array
            List of timesteps for the dynamical evolution.
        method : string, optional
            Specify which method to use to generate the RHS of the Heisenberg equation.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

    """

    num=num.astype(np.float64)
    y = y.astype(np.float64)

    # Initialise list for time-evolved operator
    num_t_list = np.zeros((len(tlist),2*len(y)),dtype=np.float64)
    # Prepare first element of list with the initial operator (t=0)
    num_t_list[0,:len(y)] = num

    # Define the integrator used to evolve the operator
    # Note: we use `zvode' here as it allows complex numbers, contrary to many other SciPy integrators
    n_int = ode(n_evo).set_integrator('dopri5', nsteps=100)
    n_int.set_f_params(y,n,method)
    n_int.set_initial_value(num_t_list[0],tlist[0])

    # Run the integrator for all times in tlist and return time-evolved operator
    t0=1
    while n_int.successful() and t0 < len(tlist):
        n_int.integrate(tlist[t0])
        num_t_list[t0] = (n_int.y)
        # print(((n_int.y).imag).reshape(n,n))
        # num_t_list[t0] = num_t_list[t0-1] + 1j*(tlist[t0]-tlist[t0-1])*(H2@(num_t_list[t0-1].reshape(n,n))-(num_t_list[t0-1]).reshape(n,n)@H2).reshape(n**2)
        # print(num_t_list[t0].reshape(n,n))
        t0 += 1

    num_t_list2 = np.zeros((len(tlist),len(y)))
    for t in range(len(tlist)):
        num_t_list2[t] = num_t_list[t,:len(y)] + 1j*num_t_list[t,len(y)::]

    return num_t_list2


# @jit(nopython=True,parallel=True,fastmath=True)
# Change all complex to complex128 if running with jit
def dyn_exact(n,num,y,tlist):
    """ Function to compute the 'exact' time evolution, using an analytical solution of the Heisenberg equation.
    
        Parameters
        ----------
        n : int
            Linear system size.
        num : array
            Operator to be time-evolved.
        y : array
            Diagonal Hamiltonian used to generate the dynamics.
        tlist : array
            List of timesteps for the dynamical evolution.
        method : string, optional
            Specify which method to use to generate the RHS of the Heisenberg equation.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

    """

    # Initialise list for time-evolved operator
    num_t_list = np.zeros((len(tlist),len(y)),dtype=np.complex64)
    # Prepare first element of list with the initial operator (t=0)
    num_t_list[0] = num
    
    # Extract quadratic part of Hamiltonian from input array y
    H = y[0:n**2]
    H = H.reshape(n,n)

    # If the system is interacting, extract interacting components of H 
    if len(y) > n**2:
        Hint = y[n**2::]
        Hint = Hint.reshape(n,n,n,n)

    t0 = 1
    # Compute the time-evolved operator for every time t in input array tlist
    for t in tlist[1::]:
        n2 = (num.copy())[:n**2]
        n2 = n2.reshape(n,n)
    
        # Initialise array to hold solution
        sol0=np.zeros(n**2+n**4,dtype=np.complex64)
        
        # Initialise various arrays depending on whether the pr
        if len(y) > n**2:
            n4 = np.zeros(n**4,dtype=np.complex64)
            n4[:] = (num.copy())[n**2::]
            n4 = n4.reshape(n,n,n,n)

        # If non-interacting, time evolve matrix:
        if len(y)==n**2:
            for i in range(n):
                for j in range(i,n):  
                    n2[i,j] *= np.cos((H[i,i]-H[j,j])*t)
                    n2[j,i] *= np.cos((H[i,i]-H[j,j])*t)
            sol0[:n**2] = n2.reshape(n**2)

        # If interacting, time evolve matrix and tensor:   
        else:
            matb = num[:n**2].reshape(n,n)
            for i in prange(n):
                for j in prange(n):
                    phase = -(H[i,i]-H[j,j])
                    n2[i,j] *= np.cos(phase*t)
                    
                    for k in prange(n):
                        for q in prange(n):
                            n4[i,j,k,q] *= np.exp(-1j*(H[i,i]+H[k,k]-H[j,j]-H[q,q])*t)
                            if k == q and i != j:
                                n4[i,j,k,q] += -1j*(Hint[i,i,k,k]-Hint[j,j,k,k])*matb[i,j]*(1-np.exp(-1j*(H[i,i]-H[j,j])*t))/(H[i,i]-H[j,j])/2.
                            if i == j and k != q:
                                n4[i,j,k,q] += -1j*(Hint[k,k,i,i]-Hint[q,q,i,i])*matb[k,q]*(1-np.exp(-1j*(H[k,k]-H[q,q])*t))/(H[k,k]-H[q,q])/2.

            # Load time-evolved operator into solution array
            sol0[:n**2] = n2.reshape(n**2)
            sol0[n**2:] = n4.reshape(n**4)
    
        # Load time-evolved operator at time t into output array num_t_list
        num_t_list[t0] = sol0
        t0 += 1
                
    return num_t_list

# @jit(nopython=True,parallel=True,fastmath=True)
def dyn_mf(n,num,y,tlist,state=[],method='jit'):
    """ Function to compute the mean-field time evolution, using a mean-field decoupling of the interaction term.

        In principle this state could be time-dependent, however that is not yet implemented here.
    
        Parameters
        ----------
        n : int
            Linear system size.
        num : array
            Operator to be time-evolved.
        y : array
            Diagonal Hamiltonian used to generate the dynamics.
        tlist : array
            List of timesteps for the dynamical evolution.
        state : array
            State used to compute the mean-field expectation values.

    """
    
    # Initialise list for time-evolved operator
    num_t_list = np.zeros((len(tlist),len(y)),dtype=np.complex64)
    # Prepare first element of list with the initial operator (t=0)
    num_t_list[0] = num
    
    # Extract quadratic part of Hamiltonian from input array y
    H = y[0:n**2]
    H = H.reshape(n,n)

    # If the system is interacting, extract interacting components of H 
    if len(y) > n**2:
        Hint = y[n**2::]
        Hint = Hint.reshape(n,n,n,n)

    t0 = 0
    # Compute the time-evolved operator for every time t in input array tlist
    for t in tlist:
        n2 = (num.copy())[:n**2]
        n2 = n2.reshape(n,n)
    
        # Initialise array to hold solution
        sol0=np.zeros(n**2+n**4,dtype=np.complex64)
        
        # Initialise various arrays depending on whether the pr
        if len(y) > n**2:
            n4 = np.zeros(n**4,dtype=np.complex64)
            n4[:] = (num.copy())[n**2::]
            n4 = n4.reshape(n,n,n,n)
     
        # If non-interacting, time evolve matrix:
        if len(y)==n**2:
            for i in range(n):
                for j in range(i,n):
                    n2[i,j] *= np.cos((H[i,i]-H[j,j])*t)
                    n2[j,i] *= np.cos((H[i,i]-H[j,j])*t)
            sol0[:n**2] = n2.reshape(n**2)
                
        # If interacting, make a mean-field decoupling of the interaction term and time evolve matrix:
        else:
            for i in range(n):
                for j in range(i,n):
                    phase = -(H[i,i]-H[j,j])
                    if len(y)>n**2:
                        for k in range(n):
                            phase += (Hint[i,i,k,k]-Hint[j,j,k,k])*state[k]
                    n2[i,j] *= np.cos(phase*t)
                    n2[j,i] *= np.cos(phase*t)
            # Load time-evolved operator into solution array
            sol0[:n**2] = n2.reshape(n**2)
    
        # Load time-evolved operator at time t into output array num_t_list
        num_t_list[t0] = sol0
        t0 += 1

    return num_t_list

def hberg(n,y,num,tlist,state):
    """ Test function used to brute-force integrate the Heisenberg equation in the initial basis. 

    Note that this *should not* be expected to be accurate beyond short times, and is implemented 
    here only as a test comparison case for systems too large to use exact diagonalization on.

    Parameters
    ----------
    n : int
        Linear system size.
    y : array
        Array containing elements of the Hamiltonian, here in the initial basis.
    num: array
        Initial number operator which is to undergo time evolution.
    tlist : array
            List of timesteps for the dynamical evolution.
    state : array
        State used to compute the mean-field expectation values.
    
    """
    #-----------------------------------------------------------------
    #-----------------------------------------------------------------

    # Get time-evolved operator at all times t in tlist
    ntlist=dyn_con(n,num,y,tlist)

    # Define two lists to store time-evolved data
    nlist = np.zeros(len(tlist))
    nqlist = np.zeros(len(tlist))

    # Define quadratic and quartic pieces of the time-evolved operator
    n2list = ntlist[::,:n**2]
    n4list = ntlist[::,n**2:]

    # Compute expectation values at all times t in tlist
    # Note that nlist includes quartic interaction contributions
    # while nqlist only includes quadratic components (and will not diverge)
    for t0 in range(len(tlist)):
        mat = n2list[t0].reshape(n,n)
        mat4 = n4list[t0].reshape(n,n,n,n)
        for i in range(n):
            nlist[t0] += (mat[i,i]*state[i]**2).real
            nqlist[t0] += (mat[i,i]*state[i]**2).real
            for j in range(n):
                if i != j:
                    nlist[t0] += (mat4[i,i,j,j]*state[i]*state[j]).real
                    nlist[t0] += -(mat4[i,j,j,i]*state[i]*state[j]).real

    return [nlist,nqlist]