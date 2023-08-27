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
# from math import factorial
from quspin.basis import spinless_fermion_basis_1d
from numba import jit,prange,int16
from scipy.integrate import ode
from .utility import nstate
from .dyn_cython import *

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

# @jit(nopython=False)
def tstep(n,t,H,Hint,sol0,n2,matb,n4,n6=0.,O6=False,int=True):

    # If non-interacting, time evolve matrix:
    if int == False:
        for i in range(n):
            for j in range(i,n):  
                n2[i,j] *= np.cos((H[i,i]-H[j,j])*t)
                n2[j,i] *= np.cos((H[i,i]-H[j,j])*t)
        sol0[:n**2] = n2.reshape(n**2)

    # If interacting, time evolve matrix and tensor(s):   
    else:
        test = np.zeros((n,n,n,n),dtype=np.int32)
        test2 = np.zeros((n,n,n,n,n,n),dtype=np.int32)
        # matb = copy.deepcopy(n2)
        for i in range(n):
            for j in range(n):
                phase = (H[i,i]-H[j,j])
                n2[i,j] *= np.exp(1j*phase*t)
                # n2[i,j] *= np.cos(phase*t)

        for i in range(n):
            for j in range(n):       
                for k in range(n):
                    for q in range(n):
                        if test[i,j,k,q] == 0:
                            phase = (H[i,i]+H[k,k]-H[j,j]-H[q,q])
                            n4[i,j,k,q] *= np.exp(1j*phase*t)
                            n4[q,k,j,i] *= np.exp(-1j*phase*t)
                            test[q,k,j,i] = 1
                            test[i,j,k,q] = 1

        for i in range(n):
            for j in range(n):       
                for k in range(n):
                    if np.round(H[j,j],4) != np.round(H[k,k],4):
                        n4[i,i,j,k] += 1*(Hint[i,i,j,j])*matb[j,k]*(np.exp(1j*(H[j,j]-H[k,k])*t))/(H[j,j]-H[k,k])
                        n4[i,i,k,j] += -1*(Hint[i,i,j,j])*matb[k,j]*(np.exp(1j*(H[k,k]-H[j,j])*t))/(H[k,k]-H[j,j])
                    elif np.round(H[j,j],4) == np.round(H[k,k],4):
                        n4[i,i,j,k] +=   1j*(Hint[i,i,j,j])*matb[j,k]*t
                        n4[i,i,k,j] +=  -1j*(Hint[i,i,j,j])*matb[k,j]*t

                    if np.round(H[i,i],4) != np.round(H[j,j],4):
                        n4[i,j,k,k] +=  1*(Hint[i,i,k,k])*matb[i,j]*(np.exp(1j*(H[i,i]-H[j,j])*t))/(H[i,i]-H[j,j])
                        n4[j,i,k,k] += -1*(Hint[i,i,k,k])*matb[j,i]*(np.exp(1j*(H[j,j]-H[i,i])*t))/(H[j,j]-H[i,i])
                    elif np.round(H[i,i],4) != np.round(H[j,j],4):
                        n4[i,j,k,k] +=   1j*(Hint[i,i,k,k])*matb[i,j]*t
                        n4[j,i,k,k] +=  -1j*(Hint[i,i,k,k])*matb[j,i]*t

        # Leading-order term in dynamics of 6th order term
        if O6 == True:
            for i in range(n):
                for j in range(n):       
                    for k in range(n):
                        for q in range(n):
                            for m in range(n):
                                for l in range(n):
                                    if test2[i,j,k,q,m,l] == 0:
                                        phase = (H[i,i]+H[k,k]+H[m,m]-H[j,j]-H[q,q]-H[l,l])
                                        n6[i,j,k,q,m,l] *= np.exp(1j*phase*t)
                                        n6[l,m,q,k,j,i] *= np.exp(-1j*phase*t)
                                        test2[i,j,k,q,m,l] = 1
                                        test2[l,m,q,k,j,i] = 1

        # HERMITIAN TEST
        if n < 10:
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for q in range(n):
                            if np.round(n4[i,j,k,q],4) != np.round(np.conjugate(n4[q,k,j,i]),4):
                                print('HERMITIAN ERROR',np.round(n4[i,j,k,q],6),np.round(np.conjugate(n4[q,k,j,i]),6))

    # Load time-evolved operator into solution array
    sol0[:n**2] = n2.reshape(n**2)
    sol0[n**2:n**4+n**2] = n4.reshape(n**4)
    if O6 == True:
        sol0[n**4+n**2:] = n6.reshape(n**6)

    return sol0

def dyn_itc(n,tlist,num2,H,Hint=[0],num4=[0],num6=[0],int=True):
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
    # print(num2.shape,H.shape,Hint.shape,num4.shape)
    # Initialise list for time-evolved operator
    # if int == True:
    #     num_t_list = np.zeros((len(tlist),n**2+n**4+n**6),dtype=np.complex128)
    # elif int == False:
    #     num_t_list = np.zeros((len(tlist),n**2),dtype=np.complex128)
    # Prepare first element of list with the initial operator (t=0)
    # num_t_list[0,0:n**2] = num2.reshape(n**2)
    # if int == True:
    #     num_t_list[0,n**2:n**4+n**2] = num4.reshape(n**4)
    #     if np.max(np.abs(num6)) != 0:
    #         num_t_list[0,n**4+n**2::] = num6.reshape(n**6)

    t0 = 0
    # Compute the time-evolved operator for every time t in input array tlist
    cython = True
    # for t in tlist:
    #     if cython == False:
    #         if np.max(np.abs(num6)) == 0:
    #             sol0 = tstep(n,t,H,Hint,np.zeros(n**2+n**4+n**6,dtype=np.complex128),np.array(num2,dtype=np.complex128),np.array(num2,dtype=np.complex128),np.array(num4,dtype=np.complex128))
    #         else:
    #             sol0 = tstep(n,t,H,Hint,np.zeros(n**2+n**4+n**6,dtype=np.complex128),np.array(num2,dtype=np.complex128),np.array(num2,dtype=np.complex128),np.array(num4,dtype=np.complex128),n6=np.array(num6,dtype=np.complex128),O6=True)
    #     elif cython == True:
    #         sol0 = np.zeros((n**2+n**4+n**6),dtype=np.complex64)
    #         n2t,n4t,n6t = cy_tstep(n,t,H,Hint,np.array(num2,dtype=np.complex128),np.array(num2,dtype=np.complex128), np.array(num4,dtype=np.complex128), np.array(num6,dtype=np.complex128), np.zeros((n,n,n,n),dtype=np.int32), np.zeros((n,n,n,n,n,n),dtype=np.int32))
    #         n2t,n4t,n6t = np.array(n2t),np.array(n4t),np.array(n6t)
    #         if t0 == 0:
    #             print(np.sum(np.diag(n2t)))

    #         # Normal-ordering requires that no creation or annihilation operators are repeated
    #         # We can set them safely to zero here, as when we un-normal-order these terms, the
    #         # terms containing repeated operators will be precisely cancelled out by the n/o corrections
    #         for i in range(n):
    #             n4t[i,:,i,:] = np.zeros((n,n))
    #             n4t[:,i,:,i] = np.zeros((n,n))

    #             n6t[i,:,i,:,:,:] = np.zeros((n,n,n,n))
    #             n6t[i,:,:,:,i,:] = np.zeros((n,n,n,n))
    #             n6t[:,:,i,:,i,:] = np.zeros((n,n,n,n))
    #             n6t[:,i,:,i,:,:] = np.zeros((n,n,n,n))
    #             n6t[:,i,:,:,:,i] = np.zeros((n,n,n,n))
    #             n6t[:,:,:,i,:,i] = np.zeros((n,n,n,n))

    #         sol0[:n**2] = n2t.reshape(n**2)
    #         sol0[n**2:n**4+n**2] = n4t.reshape(n**4)
    #         sol0[n**4+n**2:] = n6t.reshape(n**6)
    #         # print(sol0[0:n**4])
    #     # Load time-evolved operator at time t into output array num_t_list
    #     num_t_list[t0] = sol0
    #     t0 += 1

    if n <= 12:
        basis = spinless_fermion_basis_1d(n,Nf=n//2)
        no_states = min(basis.Ns,256)
    elif n <=64:
        no_states = 256
    else:
        no_states = 64
    # for i in range(n):
    #     for j in range(n):
    #         num4[i,i,j,j] += -num4[i,j,j,i]
    #         num4[i,j,j,i] = 0.

    print('No. states: ',no_states)
    statelist = np.zeros((no_states,n))
    itc = np.zeros(len(tlist),dtype=complex)
    itc2 = np.zeros((no_states,len(tlist)),dtype=complex)
    avg_test = 0.
    for ns in range(no_states):
        flag = False
        while flag == False:
            state = np.array(nstate(n,'random_half'))
            if not any((state == x).all() for x in np.array(statelist)):
                statelist[ns] = state
                avg_test += state[n//2]
                flag = True

    n2_0 = np.array(num2,dtype=np.complex128)
    # matb = np.deepcopy(n2_0,order='C')
    n4_0 = np.array(num4,dtype=np.complex128)
    if n <= 36:
        n6_0 = np.array(num6,dtype=np.complex128)
    for time in range(len(tlist)):
        # In Python (with Numba JIT)
        # itc[ns,time] += trace(n,num_t_list[time],num_t_list[0],state)
        # if cython == False:
        #     if np.max(np.abs(num6)) == 0:
        #         sol0 = tstep(n,tlist[time],H,Hint,np.zeros(n**2+n**4+n**6,dtype=np.complex128),np.array(num2,dtype=np.complex128),np.array(num2,dtype=np.complex128),np.array(num4,dtype=np.complex128))
        #     else:
        #         sol0 = tstep(n,tlist[time],H,Hint,np.zeros(n**2+n**4+n**6,dtype=np.complex128),np.array(num2,dtype=np.complex128),np.array(num2,dtype=np.complex128),np.array(num4,dtype=np.complex128),n6=np.array(num6,dtype=np.complex128),O6=True)
        # elif cython == True:
        #     sol0 = np.zeros((n**2+n**4+n**6),dtype=np.complex64)
        #     n2t,n4t,n6t = cy_tstep(n,tlist[time],H,Hint,np.array(num2,dtype=np.complex128),np.array(num2,dtype=np.complex128), np.array(num4,dtype=np.complex128), np.array(num6,dtype=np.complex128), np.zeros((n,n,n,n),dtype=np.int32), np.zeros((n,n,n,n,n,n),dtype=np.int32))
        #     n2t,n4t,n6t = np.array(n2t),np.array(n4t),np.array(n6t)

        # Normal-ordering imposes that terms with repeated creation or annihilation operators are zero
        # We can set them safely to zero here, as when we un-normal-order these terms, the
        # terms containing repeated operators will be precisely cancelled out by the n/o corrections
        # for i in range(n):
        #     n4t[i,:,i,:] = np.zeros((n,n))
        #     n4t[:,i,:,i] = np.zeros((n,n))
        #     n6t[i,:,i,:,:,:] = np.zeros((n,n,n,n))
        #     n6t[i,:,:,:,i,:] = np.zeros((n,n,n,n))
        #     n6t[:,:,i,:,i,:] = np.zeros((n,n,n,n))
        #     n6t[:,i,:,i,:,:] = np.zeros((n,n,n,n))
        #     n6t[:,i,:,:,:,i] = np.zeros((n,n,n,n))
        #     n6t[:,:,:,i,:,i] = np.zeros((n,n,n,n))

        # Zero each of the arrays for testing purposes (helps to isolate bugs)
        # Make sure the below three lines are commented for any 'real' simulations
        # num2 = np.zeros((n,n))
        # num4 = np.zeros((n,n,n,n))
        # num6 = np.zeros((n,n,n,n,n,n))
        
        # n2t = np.copy(n2_0,order='C')
        # n4t = np.copy(n4_0,order='C')
        # n6t = np.copy(n6_0,order='C')

        # Using Cython module (dyn_cython.pyx: must be compiled before first use)
        if n <= 36:
            #itc[:,time] = cytrace(n,tlist[time],H,Hint,n2_0,n4_0,n6_0,statelist,np.zeros(no_states,dtype=np.complex128))
            itc[time] = cytrace2(n,tlist[time],H,Hint,n2_0,n4_0,n6_0,statelist)
        else:
            itc[time] = cytrace3(n,tlist[time],H,Hint,n2_0,n4_0,statelist)
        itc2[:,time] = cytrace_nonint(n,tlist[time],H,n2_0,statelist,np.zeros(no_states,dtype=np.complex128))
        # itc[time] *= 1.0/no_states

    # print('itc',itc[0:10])
    
    print('itc mean',itc[0:5])
    #print('itc var',itc[0:5])
    #print('itc med',itc[0:5])
    itc = itc.real
    # import matplotlib.pyplot as plt
    # if np.max(np.abs(num6)) == 0:
    #     plt.plot(tlist,np.mean(itc,axis=0),'x--',label=r'Old')
    # else:
    #     plt.plot(tlist,np.mean(itc,axis=0),'o--',linewidth=2,label=r'Avg')
    #     # plt.plot(tlist,np.mean(itc,axis=0)+(0.25-np.mean(itc,axis=0)[0]),'o-.',linewidth=2,label=r'Avg2')
    #     plt.plot(tlist,0.25*np.mean(itc,axis=0)/np.mean(itc,axis=0)[0],'r--',linewidth=2,label=r'Norm')
        # plt.plot(tlist,np.median(itc,axis=0),'x--',linewidth=2,label=r'Med')
    # for ns in range(no_states):
    #     plt.plot(tlist,itc[ns],'k-',alpha=0.01)


    return itc.real,itc2.real

@jit(nopython=True,parallel=True,fastmath=True)
def trace(n,sol_a,sol_b,state,int=True):
    n2a = sol_a[:n**2].reshape(n,n)
    n2b = sol_b[:n**2].reshape(n,n)

    if int == True:
        n4a = sol_a[n**2:].reshape(n,n,n,n)
        n4b = sol_b[n**2:].reshape(n,n,n,n)

    corr = 0

    # Product of quadratic terms
    for i in range(n):
        for j in range(n):
            corr += n2a[i,i]*n2b[j,j]*state[i]*state[j]
            if i != j:
                corr += n2a[i,j]*n2b[j,i]*state[i]*(1-state[j])

    if int == True:
        # Product of quadaratic and quartic terms
        for i in prange(n):
            for j in range(n):
                for k in range(n):
                    # Colour codes refer to highlighted notes
                    # Contributions should come in pairs, 2x4 and 4x2

                    # BLUE CONTRIBUTIONS
                    corr += (n2a[i,i]*n4b[j,j,k,k] + n4a[i,i,j,j]*n2b[k,k])*state[i]*state[j]*state[k]

                    # GREEN CONTRIBUTIONS
                    # First one
                    if j != k:
                        corr += (n2a[i,i]*n4b[j,k,k,j])*state[i]*state[j]*(1-state[k])
                    if i != j:
                        corr += (n4a[i,j,j,i]*n2b[k,k])*state[i]*(1-state[j])*state[k]

                    # Second one
                    if i != j:
                        corr += (n2a[i,j]*n4b[j,i,k,k])*state[i]*(1-state[j])*state[k]


                        corr += (n4a[i,j,k,k]*n2b[j,i])*state[i]*(1-state[j])*state[k]
                        if j == k:
                            corr += (n4a[i,j,k,k]*n2b[j,i])*state[i]*(1-state[j])
                        if i == k:
                            corr += -1*(n4a[i,j,k,k]*n2b[j,i])*state[i]*(1-state[j])

                    # Third one
                    if i != j:
                        corr += n2a[i,j]*n4b[k,k,j,i]*state[i]*(1-state[j])*state[k]
                        if j == k:
                            corr += n2a[i,j]*n4b[k,k,j,i]*state[i]*(1-state[j])
                        if i == k:
                            corr += -1*n2a[i,j]*n4b[k,k,j,i]*state[i]*(1-state[j])

                    if j != k:
                        corr += n4a[i,i,j,k]*n2b[k,j]*state[i]*state[j]*(1-state[k])

                    # YELLOW CONTRIBUTIONS
                    if i != j != k:
                        corr += n2a[i,j]*n4b[k,i,j,k]*state[i]*(1-state[j])*state[k]
                        corr += n2a[i,j]*n4b[j,k,k,i]*state[i]*(1-state[j])*(1-state[k])

                        corr += n4a[i,j,k,i]*n2b[k,j]*state[i]*(1-state[j])*state[k]
                        corr += n4a[i,j,j,k]*n2b[k,i]*state[i]*(1-state[j])*(1-state[k])

    return corr.real

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
