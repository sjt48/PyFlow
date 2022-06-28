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

This file contains all of the code used to construct the RHS of the flow equations using matrix/tensor contractions 
and numerically integrate the flow equation to obtain a diagonal Hamiltonian.

"""

import os
from psutil import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count(logical=False)))       # Set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count(logical=False)))       # Set number of MKL threads to run in parallel
os.environ['NUMBA_NUM_THREADS'] = str(int(cpu_count(logical=False)))    # Set number of Numba threads
import jax.numpy as jnp
from jax import jit
import numpy as np
from diffrax import diffeqsolve, ODETerm, Dopri5
from jax.experimental.host_callback import id_print
from jax.lax import dynamic_slice as slice
# from jax.config import config
# config.update("jax_enable_x64", True)
from datetime import datetime
from ..dynamics import dyn_con,dyn_exact
# from numba import jit,prange
import gc,copy
from ..contract import contract,contractNO
from ..utility import nstate, state_spinless, indices
from  jax.experimental.ode import odeint as ode

#------------------------------------------------------------------------------ 

# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
def cut(y,n,cutoff,indices):
    """ Checks if ALL quadratic off-diagonal parts have decayed below cutoff*10e-3 and TYPICAL (median) off-diag quartic term have decayed below cutoff. """
    mat2 = y[:n**2].reshape(n,n)
    mat2_od = mat2-jnp.diag(jnp.diag(mat2))

    if jnp.max(jnp.abs(mat2_od)) < cutoff*10**(-3):
        mat4 = y[n**2:n**2+n**4]
        mat4_od = jnp.zeros(n**4)
        for i in indices:               
            mat4_od = mat4_od.at[i].set(mat4[i])
        mat4_od = mat4_od[mat4_od != 0]
        if jnp.median(jnp.abs(mat4_od)) < cutoff:
            return 0 
        else:
            return 1
    else:
        return 1

def nonint_ode(H,l,method='einsum'):
    """ Generate the flow equation for non-interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.

            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        sol : RHS of the flow equation

    """

    H0 = jnp.diag(jnp.diag(H))
    V0 = H - H0
    eta = contract(H0,V0,method=method,eta=True)
    sol = contract(eta,H,method=method,eta=False)

    return sol

#------------------------------------------------------------------------------
# Build the generator eta at each flow time step
def eta_con(y,n,method='jit',norm=False):
    """ Generates the generator at each flow time step. 
    
        Parameters
        ----------

        y : array
            Running Hamiltonian used to build generator.
        n : integer
            Linear system size
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.

    """

    # Extract quadratic parts of Hamiltonian from array y
    H = y[0:n**2]
    H = H.reshape(n,n)
    H0 = jnp.diag(jnp.diag(H))
    V0 = H - H0

    # Extract quartic parts of Hamiltonian from array y
    Hint = y[n**2::]
    Hint = Hint.reshape(n,n,n,n)
    Hint0 = jnp.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                Hint0[i,i,j,j] = Hint[i,i,j,j]
                Hint0[i,j,j,i] = Hint[i,j,j,i]
    Vint = Hint-Hint0

    # Compute quadratic part of generator
    eta2 = contract(H0,V0,method=method,eta=True)

    # Compute quartic part of generator
    eta4 = contract(Hint0,V0,method=method,eta=True) + contract(H0,Vint,method=method,eta=True)

    # Add normal-ordering corrections into generator eta, if norm == True
    if norm == True:
        state=nstate(n,0.5)

        eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
        eta2 += eta_no2
        eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
        eta4 += eta_no4

    # Combine into array
    eta = jnp.zeros(n**2+n**4)
    eta[:n**2] = eta2.reshape(n**2)
    eta[n**2:] = eta4.reshape(n**4)

    return eta

def extract_diag(H2,Hint):

    n,_ = H2.shape
    H2_0 = jnp.diag(jnp.diag(H2))       # Define diagonal quadratic part H0
    V0 = H2 - H2_0                      # Define off-diagonal quadratic part

    Hint0 = jnp.zeros((n,n,n,n))        # Define diagonal quartic part 

    # This loop structure looks weird but it's to avoid a mysterious segfault on some systems
    # Including both updates in the same pair of loops can lead to crashes

    for i in range(n):                  # Load Hint0 with values
        for j in range(n):
                Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
    for i in range(n):
        for j in range(n):
                Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
    Vint = Hint-Hint0

    return H2_0,V0,Hint0,Vint


#------------------------------------------------------------------------------

# def extract_diag(A,norm=False):
#     B = jnp.zeros(A.shape)
#     n,_,_,_ = A.shape
#     for i in range(n): 
#         for j in range(n):
#             if i != j:
#                 if norm == True:
#                     # Symmetrise (for normal-ordering wrt inhomogeneous states)
#                     A[i,i,j,j] += -A[i,j,j,i]
#                     A[i,j,j,i] = 0.
#             if i != j:
#                 if norm == True:
#                     # Symmetrise (for normal-ordering wrt inhomogeneous states)
#                     A[i,i,j,j] += A[j,j,i,i]
#                     A[i,i,j,j] *= 0.5
#                 # Load new array with diagonal values
#                 B[i,i,j,j] = A[i,i,j,j]
#     return B,A

def int_ode(y,l,eta=[],method='einsum',norm=False,Hflow=True):
        """ Generate the flow equation for the interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the ijnput array eta will be used to specify the generator at this flow time step. The latter option will result 
        in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
        integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
        steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
        interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
        these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
        the benefits from the speed increase likely outweigh the decrease in accuracy.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for interacting system.

        """
        # print('y shape', y.shape)
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        # id_print(y)
        # id_print(l)
        H2 = y[0]                           # Define quadratic part of Hamiltonian
        n,_ = H2.shape
        # H2_0 = jnp.diag(jnp.diag(H2))       # Define diagonal quadratic part H0
        # V0 = H2 - H2_0                      # Define off-diagonal quadratic part

        Hint = y[1]                         # Define quartic part of Hamiltonian
        # Hint0 = jnp.zeros((n,n,n,n))        # Define diagonal quartic part 
        # for i in range(n):                  # Load Hint0 with values
        #     for j in range(n):
        #             Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
        #             Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
        # Vint = Hint-Hint0
        # id_print(H2)

        H2_0,V0,Hint0,Vint = extract_diag(H2,Hint)

        if norm == True:
            state = state_spinless(H2)

        if Hflow == True:
            # Compute the generator eta
            eta0 = contract(H2_0,V0,method=method,eta=True)
            eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H2_0,Vint,method=method,eta=True)

            # Add normal-ordering corrections into generator eta, if norm == True
            if norm == True:

                eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H2_0,Vint,method=method,eta=True,state=state)
                eta0 += eta_no2

                eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
                eta_int += eta_no4
        else:
            eta0 = (eta[:n**2]).reshape(n,n)
            eta_int = (eta[n**2:]).reshape(n,n,n,n)

        # id_print(eta0)
   
        # Compute the RHS of the flow equation dH/dl = [\eta,H]
        sol = contract(eta0,H2,method=method)
        sol2 = contract(eta_int,H2,method=method) + contract(eta0,Hint,method=method)

        # Add normal-ordering corrections into flow equation, if norm == True
        if norm == True:
            sol_no = contractNO(eta_int,H2,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
            sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
            sol+=sol_no
            sol2 += sol4_no

        # id_print([sol,sol2])
        return [sol,sol2]

# @jit
def update(n2,n4,H2,Hint,steps,method='einsum'):

    n,_ = H2.shape
    H0,V0,Hint0,Vint = extract_diag(H2,Hint)

    eta2 = contract(H0,V0,method=method,comp=False,eta=True)
    eta4 = contract(Hint0,V0,method=method,comp=False,eta=True) + contract(H0,Vint,method=method,comp=False,eta=True)

    dl = steps[-1]-steps[0]
    # id_print(dl)
    dn2 = contract(eta2,n2,method=method,comp=False)
    dn4 = contract(eta4,n2,method=method,comp=False) + contract(eta2,n4,method=method,comp=False)
    n2 += dl*dn2
    n4 += dl*dn4

    return n2,n4

def liom_ode(y,l,n,array,method='tensordot',comp=False,Hflow=True,norm=False,bck=True):
    """ Generate the flow equation for density operators of the interacting systems.

        e.g. compute the RHS of dn/dl = [\eta,n] which will be used later to integrate n(l) -> n(l + dl)

        Note that this can be used to integrate density operators either 'forward' (from l=0 to l -> infinity) or
        also 'backward' (from l -> infinity to l=0), as the flow equations are the same either way. The only changes
        are the initial condition and the sign of the timestep dl.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running density operator at flow time l.
        H : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        comp : bool, optional
            Specify whether the density operator is complex, e.g. for use in time evolution.
            Triggers the 'contract' function with 'jit' method to use efficient complex conjugation routine.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for the density operator of the interacting system.


    """

    if Hflow == True:
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        H2 = array[0]                       # Define quadratic part of Hamiltonian
        print('h2shape',H2.shape)
        # H0 = jnp.diag(jnp.diag(H2))           # Define diagonal quadratic part H0
        # V0 = H2 - H0                        # Define off-diagonal quadratic part B
        
        if len(array)>1:
            Hint = array[1]            # Define quartic part of Hamiltonian
            # Hint0 = jnp.zeros((n,n,n,n))     # Define diagonal quartic part 
            # for i in range(n):              # Load Hint0 with values
            #     for j in range(n):
            #         # if i != j:
            #             # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
            #             Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
            #             Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
            # Vint = Hint-Hint0

            H0,V0,Hint0,Vint = extract_diag(H2,Hint)

        # Compute the quadratic generator eta2
        eta2 = contract(H0,V0,method=method,comp=False,eta=True)
        # id_print(eta2)

        if len(array) > 1:
            eta4 = contract(Hint0,V0,method=method,comp=comp,eta=True) + contract(H0,Vint,method=method,comp=comp,eta=True)

        # Add normal-ordering corrections into generator eta, if norm == True
        # if norm == True:
        #     state=state_spinless(H2)
        #     eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
        #     eta2 += eta_no2

        #     eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
        #     eta4 += eta_no4

    # else:
    #     eta2 = (array[0:]).reshape(n,n)
    #     eta4 = (array[n**2::]).reshape(n,n,n,n)

    # Extract components of the density operator from ijnput array 'y'
    n2 = y[0]                           # Define quadratic part of density operator
    if len(y)>1:                     # If interacting system...
        n4 = y[1]                  #...then define quartic part of density operator
                    
    # Compute the quadratic terms in the RHS of the flow equation
    sol2 = contract(eta2,n2,method=method,comp=comp)

    # Compute quartic terms, if interacting system
    if len(y) > 1:
        sol4 = contract(eta4,n2,method=method,comp=comp) + contract(eta2,n4,method=method,comp=comp)

    # Add normal-ordering corrections into flow equation, if norm == True
    # if norm == True:
    #     sol_no = contractNO(eta4,n2,method=method,eta=False,state=state) + contractNO(eta2,n4,method=method,eta=False,state=state)
    #     sol+=sol_no
    #     if len(y) > n**2:
    #         sol4_no = contractNO(eta4,n4,method=method,eta=False,state=state)
    #         sol2 += sol4_no

    if bck == True:
        return [-1*sol2,-1*sol4]
    elif bck == False:
        return [sol2,sol4]

def liom_ode_int(y,l,n,array,bck=True,method='tensordot',comp=False,Hflow=True,norm=False):
    """ Generate the flow equation for density operators of the interacting systems.

        e.g. compute the RHS of dn/dl = [\eta,n] which will be used later to integrate n(l) -> n(l + dl)

        Note that this can be used to integrate density operators either 'forward' (from l=0 to l -> infinity) or
        also 'backward' (from l -> infinity to l=0), as the flow equations are the same either way. The only changes
        are the initial condition and the sign of the timestep dl.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running density operator at flow time l.
        H : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        comp : bool, optional
            Specify whether the density operator is complex, e.g. for use in time evolution.
            Triggers the 'contract' function with 'jit' method to use efficient complex conjugation routine.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for the density operator of the interacting system.


    """

    if Hflow == True:
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        H2 = array[0]                       # Define quadratic part of Hamiltonian
        # H0 = jnp.diag(jnp.diag(H2))           # Define diagonal quadratic part H0
        # V0 = H2 - H0                        # Define off-diagonal quadratic part B

        # if len(array)>1:
        m,_ = y[0].shape
        Hint = array[1]                         # Define quartic part of Hamiltonian
        # Hint0 = jnp.zeros((m,m,m,m))        # Define diagonal quartic part 
        # for i in range(m):                  # Load Hint0 with values
        #     for j in range(m):
        #             Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
        #             Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
        # Vint = Hint-Hint0

        H0,V0,Hint0,Vint = extract_diag(H2,Hint)
        # id_print(H2)
        # Compute the quadratic generator eta2
        eta2 = contract(H0,V0,method=method,comp=False,eta=True)
        eta4 = contract(Hint0,V0,method=method,comp=comp,eta=True) + contract(H0,Vint,method=method,comp=comp,eta=True)
        # id_print(eta2)
        # Add normal-ordering corrections into generator eta, if norm == True
        # if norm == True:
        #     state=state_spinless(H2)
        #     eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
        #     eta2 += eta_no2

        #     eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
        #     eta4 += eta_no4

    # Extract components of the density operator from ijnput array 'y'
    n2 = y[0]                           # Define quadratic part of density operator
    n4 = y[1]                           #...then define quartic part of density operator
                    
    # Compute the quadratic terms in the RHS of the flow equation
    sol2 = -1*(contract(eta2,n2,method=method,comp=comp))

    # Compute quartic terms, if interacting system
    sol4 = -1*(contract(eta4,n2,method=method,comp=comp) + contract(eta2,n4,method=method,comp=comp))

    # Add normal-ordering corrections into flow equation, if norm == True
    # if norm == True:
    #     sol_no = contractNO(eta4,n2,method=method,eta=False,state=state) + contractNO(eta2,n4,method=method,eta=False,state=state)
    #     sol+=sol_no
    #     if len(y) > n**2:
    #         sol4_no = contractNO(eta4,n4,method=method,eta=False,state=state)
    #         sol2 += sol4_no

    return [sol2,sol4]

def liom_ode_int_fwd(y,l,n,array,bck=False,method='einsum',comp=False,Hflow=True,norm=False):
    """ Generate the flow equation for density operators of the interacting systems.

        e.g. compute the RHS of dn/dl = [\eta,n] which will be used later to integrate n(l) -> n(l + dl)

        Note that this can be used to integrate density operators either 'forward' (from l=0 to l -> infinity) or
        also 'backward' (from l -> infinity to l=0), as the flow equations are the same either way. The only changes
        are the initial condition and the sign of the timestep dl.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running density operator at flow time l.
        H : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        comp : bool, optional
            Specify whether the density operator is complex, e.g. for use in time evolution.
            Triggers the 'contract' function with 'jit' method to use efficient complex conjugation routine.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for the density operator of the interacting system.


    """

    if Hflow == True:
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        H2 = array[0]                       # Define quadratic part of Hamiltonian
        # H0 = jnp.diag(jnp.diag(H2))           # Define diagonal quadratic part H0
        # V0 = H2 - H0                        # Define off-diagonal quadratic part B

        # if len(array)>1:
        m,_ = array[0].shape
        Hint = array[1]                         # Define quartic part of Hamiltonian
        # Hint0 = jnp.zeros((m,m,m,m))        # Define diagonal quartic part 
        # for i in range(m):                  # Load Hint0 with values
        #     for j in range(m):
        #             Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
        #             Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
        # Vint = Hint-Hint0

        H0,V0,Hint0,Vint = extract_diag(H2,Hint)

        # Compute the quadratic generator eta2
        eta2 = contract(H0,V0,method=method,comp=False,eta=True)
        eta4 = contract(Hint0,V0,method=method,comp=comp,eta=True) + contract(H0,Vint,method=method,comp=comp,eta=True)

        # Add normal-ordering corrections into generator eta, if norm == True
        # if norm == True:
        #     state=state_spinless(H2)
        #     eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
        #     eta2 += eta_no2

        #     eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
        #     eta4 += eta_no4

    # Extract components of the density operator from ijnput array 'y'
    n2 = y[0]                           # Define quadratic part of density operator
    n4 = y[1]                           #...then define quartic part of density operator
                    
    # Compute the quadratic terms in the RHS of the flow equation
    sol2 = contract(eta2,n2,method=method,comp=comp)

    # Compute quartic terms, if interacting system
    sol4 = contract(eta4,n2,method=method,comp=comp) + contract(eta2,n4,method=method,comp=comp)

    # Add normal-ordering corrections into flow equation, if norm == True
    # if norm == True:
    #     sol_no = contractNO(eta4,n2,method=method,eta=False,state=state) + contractNO(eta2,n4,method=method,eta=False,state=state)
    #     sol+=sol_no
    #     if len(y) > n**2:
    #         sol4_no = contractNO(eta4,n4,method=method,eta=False,state=state)
    #         sol2 += sol4_no

    if bck == True:
        return [-1*sol2,-1*sol4]
    elif bck == False:
        return [sol2,sol4]


def int_ode_fwd(l,y0,n,eta=[],method='jit',norm=False,Hflow=False,comp=False):
        """ Generate the flow equation for the interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the ijnput array eta will be used to specify the generator at this flow time step. The latter option will result 
        in a huge speed increase, at the potential cost of accuracy. This is because the SciPi routine used to 
        integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
        steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
        interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
        these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
        the benefits from the speed increase likely outweigh the decrease in accuracy.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for interacting system.

        """
        y = y0[:n**2+n**4]
        nlist = y0[n**2+n**4::]

        # Extract various components of the Hamiltonian from the ijnput array 'y'
        H = y[0:n**2]                   # Define quadratic part of Hamiltonian
        H = H.reshape(n,n)              # Reshape into matrix
        H0 = jnp.diag(jnp.diag(H))        # Define diagonal quadratic part H0
        V0 = H - H0                     # Define off-diagonal quadratic part B

        Hint = y[n**2:]                 # Define quartic part of Hamiltonian
        Hint = Hint.reshape(n,n,n,n)    # Reshape into rank-4 tensor
        Hint0 = jnp.zeros((n,n,n,n))     # Define diagonal quartic part 
        for i in range(n):              # Load Hint0 with values
            for j in range(n):
                # if i != j:
                    # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                    Hint0[i,i,j,j] = Hint[i,i,j,j]
                    Hint0[i,j,j,i] = Hint[i,j,j,i]
        Vint = Hint-Hint0

        # Extract components of the density operator from ijnput array 'y'
        n2 = nlist[0:n**2]                  # Define quadratic part of density operator
        n2 = n2.reshape(n,n)            # Reshape into matrix
        if len(nlist)>n**2:                 # If interacting system...
            n4 = nlist[n**2::]              #...then define quartic part of density operator
            n4 = n4.reshape(n,n,n,n)    # Reshape into tensor
        
        if norm == True:
            state=state_spinless(H)

        if Hflow == True:
            # Compute the generator eta
            eta0 = contract(H0,V0,method=method,eta=True)
            eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H0,Vint,method=method,eta=True)

            # Add normal-ordering corrections into generator eta, if norm == True
            if norm == True:

                eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
                eta0 += eta_no2

                eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
                eta_int += eta_no4
        else:
            eta0 = (eta[:n**2]).reshape(n,n)
            eta_int = (eta[n**2:]).reshape(n,n,n,n)
   
        # Compute the RHS of the flow equation dH/dl = [\eta,H]
        sol = contract(eta0,H0+V0,method=method)
        sol2 = contract(eta_int,H0+V0,method=method) + contract(eta0,Hint,method=method)

        nsol = contract(eta0,n2,method=method,comp=comp)
        if len(y) > n**2:
            nsol2 = contract(eta_int,n2,method=method,comp=comp) + contract(eta0,n4,method=method,comp=comp)


        # Add normal-ordering corrections into flow equation, if norm == True
        if norm == True:
            sol_no = contractNO(eta_int,H0+V0,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
            sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
            sol+= sol_no
            sol2 += sol4_no
        
        # Define and load output list sol0
        sol0 = jnp.zeros(2*(n**2+n**4))
        sol0[:n**2] = sol.reshape(n**2)
        sol0[n**2:n**2+n**4] = sol2.reshape(n**4)
        sol0[n**2+n**4:2*n**2+n**4] = nsol.reshape(n**2)
        sol0[2*n**2+n**4:] = nsol2.reshape(n**4)

        return sol0
#------------------------------------------------------------------------------  

def flow_static(n,hamiltonian,dl_list,qmax,cutoff,method='jit',store_flow=True):
    """
    Diagonalise an initial non-interacting Hamiltonian and compute the integrals of motion.

    Note that this function does not use the trick of fixing eta: as non-interacting systems are
    quadratic in terms of fermion operators, there are no high-order tensor contractions and so 
    fixing eta is not necessary here as the performance gain is not expected to be significant.

        Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        output : dict
            Dictionary containing diagonal Hamiltonian ("H0_diag") and LIOM on central site ("LIOM").
    
    """
    H2 = hamiltonian.H2_spinless
    H2 = H2.astype(jnp.float64)

    # Define integrator
    sol = ode(nonint_ode,(H2),dl_list)
    print(jnp.sort(jnp.diag(sol[-1])))

    # Initialise a density operator in the diagonal basis on the central site
    init_liom = jnp.zeros((n,n))
    init_liom = init_liom.at[n//2,n//2].set(1.0)

    # Reverse list of flow times in order to conduct backwards integration
    dl_list = -1*dl_list[::-1]
    sol = sol[::-1]

    # Do backwards integration
    k0=0
    for k0 in range(len(dl_list)-1):
        liom = ode(liom_ode,init_liom,dl_list[k0:k0+2],n,sol[k0])
        init_liom = liom[-1]

    # Take final value for the transformed density operator and reshape to a matrix
    central = (liom[-1,:n**2]).reshape(n,n)

    # Build output dictionary
    output = {"H0_diag":sol[0].reshape(n,n),"LIOM":central}
    if store_flow == True:
        output["flow"] = sol[::-1]
        output["dl_list"] = dl_list[::-1]

    return output
    
# @jit(nopython=True,parallel=True,fastmath=True)
def proc(mat,cutoff):
    """ Test function to zero all matrix elements below a cutoff. """
    for i in prange(len(mat)):
        if mat[i] < cutoff:
            mat[i] = 0.
    return mat

def flow_static_int(n,hamiltonian,dl_list,qmax,cutoff,method='jit',norm=True,Hflow=False,store_flow=False):
    """
    Diagonalise an initial interacting Hamiltonian and compute the integrals of motion.

    Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        output : dict
            Dictionary containing diagonal Hamiltonian ("H0_diag","Hint"), LIOM interaction coefficient ("LIOM Interactions"),
            the LIOM on central site ("LIOM") and the value of the second invariant of the flow ("Invariant").
    
    """
    H2,Hint = hamiltonian.H2_spinless,hamiltonian.H4_spinless

    # Number of intermediate setps specified for the solver to use
    # It will in any case insert others as needed
    increment = 2

    print('dl_list',len(dl_list),dl_list[0],dl_list[-1])

    # Define integrator
    # sol = ode(int_ode,[H2,Hint],dl_list)
    mem_tot = (len(dl_list)*(n**2+n**4)*4)/1e9
    chunk = int(np.ceil(mem_tot/6))
    #print('MANUALLY SET TO 20 CHUNKS FOR DEBUG PURPOSES')
    # chunk =int(2)
    if chunk > 1:
        chunk_size = len(dl_list)//chunk
        print('Memory',mem_tot,chunk,chunk_size)
    # chunk =int(2)

    if chunk <= 1:
        # Integration with hard-coded event handling
        k=1
        sol2 = jnp.zeros((len(dl_list),n,n))
        sol4 = jnp.zeros((len(dl_list),n,n,n,n))
        sol2 = sol2.at[0].set(H2)
        sol4 = sol4.at[0].set(Hint)
        J0 = 1

        # term = ODETerm(int_ode)
        # solver = Dopri5()

        while k <len(dl_list) and J0 > cutoff:
            # print(k)
            steps = np.linspace(dl_list[k-1],dl_list[k],num=increment,endpoint=True)
            # print('steps',steps)
            # soln = ode(int_ode,[sol2[k-1],sol4[k-1]],dl_list[k-1:k+1],rtol=1e-8,atol=1e-8)
            soln = ode(int_ode,[sol2[k-1],sol4[k-1]],steps,rtol=1e-8,atol=1e-8)
            # sol = diffeqsolve(term,solver,t0=dl_list[k-1],t1=dl_list[k],dt0=dl_list[k]-dl_list[k-1],y0=[sol2[k-1],sol4[k-1]])
            # soln = sol.ys
            # print(k,sol.ts,sol.ys)
            sol2 = sol2.at[k].set(soln[0][-1])
            # print(soln[0][-1])
            sol4 = sol4.at[k].set(soln[1][-1])
            J0 = jnp.max(jnp.abs(soln[0][-1] - jnp.diag(jnp.diag(soln[0][-1]))))
            k += 1

    else:

        # Initialise arrays
        # Note: the memory required for these arrays is *not* pre-allocated. The reason is twofold: partly, it
        # is likely that the integration will finish before the max value of dl_list is encountered, therefore 
        # it's a waste to allocate all the memory. Secondly, in a later step we create a shortened copy of the array
        # of the form sol2[0:k], which returns a new array of length k that requires separate memory allocation, so if we 
        # max out the memory allocation here, there's no space left for to allocate the shortened array later.
        # (Modifying the array in-place would be better, but I don't know how to do that...)

        k=1
        sol2 = np.zeros((len(dl_list),n,n),dtype=np.float32)
        # sol2.fill(0.)
        sol4 = np.zeros((len(dl_list),n,n,n,n),dtype=np.float32)
        # sol4.fill(0.)
        sol2_gpu = jnp.zeros((chunk_size,n,n))
        sol4_gpu = jnp.zeros((chunk_size,n,n,n,n))
        sol2_gpu = sol2_gpu.at[0].set(H2)
        sol4_gpu = sol4_gpu.at[0].set(Hint)
        J0 = 1
    
        # Integration with hard-coded event handling
        while k <len(dl_list) and J0 > cutoff:
            # print(k)
            steps = np.linspace(dl_list[k-1],dl_list[k],num=increment,endpoint=True)
            soln = ode(int_ode,[sol2_gpu[k%chunk_size-1],sol4_gpu[k%chunk_size-1]],steps,rtol=1e-8,atol=1e-8)
            sol2_gpu = sol2_gpu.at[k%chunk_size].set(soln[0][-1])
            sol4_gpu = sol4_gpu.at[k%chunk_size].set(soln[1][-1])
            J0 = jnp.max(jnp.abs(soln[0][-1] - jnp.diag(jnp.diag(soln[0][-1]))))

            if k%chunk_size==0:
                count = int(k/chunk_size)
                # print('****')
                # print(count,k)
                # print((count-1)*chunk_size,(count)*chunk_size)
                # print(sol2[(count-1)*chunk_size:(count)*chunk_size].shape)
                # print(np.array(sol2_gpu).shape)
                if (sol2[(count-1)*chunk_size:(count)*chunk_size]).shape==np.array(sol2_gpu).shape:
                    sol2[(count-1)*chunk_size:(count)*chunk_size] = np.array(sol2_gpu)
                    sol4[(count-1)*chunk_size:(count)*chunk_size] = np.array(sol4_gpu)
                    # print(sol2[(count-1)*chunk_size:(count)*chunk_size])
                # else:
                #     remainder = len(sol2[count*chunk_size::])
                #     sol2[count*chunk_size::] = np.array(sol2_gpu[0:remainder])
                #     sol4[count*chunk_size::] = np.array(sol4_gpu[0:remainder])
            elif k == len(dl_list)-1 or J0 <= cutoff:
                remainder = len(sol2_gpu[0:k%chunk_size])
                # print('rem',remainder)
                sol2[(count)*chunk_size:k] = np.array(sol2_gpu[0:remainder])
                sol4[(count)*chunk_size:k] = np.array(sol4_gpu[0:remainder])
            k += 1
        

    print('dl_list',len(dl_list),dl_list[0],dl_list[-1])
    print(k,J0,dl_list[k-1])
    if k != len(dl_list):
        dl_list = dl_list[0:k]
        sol2 = sol2[0:k]
        sol4 = sol4[0:k]
        

    steps = np.zeros(len(dl_list)-1)
    for i in range(len(dl_list)-1):
        steps[i] = dl_list[i+1]-dl_list[i]
    print(np.max(steps),np.min(steps))

    # Resize chunks
    if chunk > 1:
        mem_tot = (len(dl_list)*(n**2+n**4)*4)/1e9
        chunk = int(np.ceil(mem_tot/6))
        # chunk = 2
        chunk_size = len(dl_list)//chunk
        print('NEW CHUNK SIZE',chunk_size)
        if int(chunk_size*chunk) != int(len(dl_list)):
            print('Chunk size error - LIOMs may not be reliable.')

        del sol2_gpu
        del sol4_gpu 

        sol2_gpu = jnp.zeros((chunk_size,n,n))
        sol4_gpu = jnp.zeros((chunk_size,n,n,n,n))

    # Store initial interaction value and trace of initial H^2 for later error estimation
    delta = jnp.max(Hint)
    e1 = jnp.trace(jnp.dot(H2,H2))
    
    # Define final diagonal quadratic Hamiltonian
    H0_diag = soln[0][-1].reshape(n,n)
    print(jnp.sort(jnp.diag(H0_diag)))
    print('Max |V|: ',jnp.max(jnp.abs(H0_diag-jnp.diag(jnp.diag(H0_diag)))))
    # Define final diagonal quartic Hamiltonian
    Hint2 = soln[1][-1].reshape(n,n,n,n)   
    # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint = HFint.at[i,j].set(Hint2[i,i,j,j])
            HFint = HFint.at[i,j].set(-Hint2[i,j,j,i])

    # Compute the difference in the second invariant of the flow at start and end
    # This acts as a measure of the unitarity of the transform
    Hflat = HFint.reshape(n**2)
    inv = 0.
    for i in range(n**2):
        inv += 2*Hflat[i]**2
    e2 = jnp.trace(jnp.dot(H0_diag,H0_diag))
    inv2 = jnp.abs(e1 - e2 + ((2*delta)**2)*(n-1) - inv)/jnp.abs(e2+((2*delta)**2)*(n-1))

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits = lbits.at[q-1].set(jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.)))

    # Initialise a density operator in the microscopic basis on the central site
    init_liom2 = jnp.zeros((n,n))
    init_liom4 = jnp.zeros((n,n,n,n))
    init_liom2 = init_liom2.at[n//2,n//2].set(1.0)

    # Do forwards integration
    k0=0
    jit_update = jit(update)
    if chunk <= 1:
        for k0 in range(len(dl_list)-1):
            steps = np.linspace(dl_list[k0],dl_list[k0+1],num=increment,endpoint=True)
            init_liom2,init_liom4  = jit_update(init_liom2,init_liom4,sol2[k0],sol4[k0],steps)

    else:
        for k0 in range(len(dl_list)-1):
            # print(k0,int(k0/chunk_size),k0%chunk_size)
            if k0%chunk_size==0:
                # print('load mem')
                count = int(k0/chunk_size)
                if jnp.array(sol2[count*chunk_size:(count+1)*chunk_size]).shape == sol2_gpu.shape:
                    # print('load1')
                    sol2_gpu = sol2_gpu.at[:,:].set(jnp.array(sol2[count*chunk_size:(count+1)*chunk_size]))
                    sol4_gpu = sol4_gpu.at[:,:].set(jnp.array(sol4[count*chunk_size:(count+1)*chunk_size]))
                else:
                    # print('load2')
                    sol2_gpu = jnp.array(sol2[count*chunk_size:(count+1)*chunk_size])
                    sol4_gpu = jnp.array(sol4[count*chunk_size:(count+1)*chunk_size])

            # print('****')
            # print(k0)
            # print(sol2_gpu[k0%chunk_size])
            # print(sol2[k0])
            steps = np.linspace(dl_list[k0],dl_list[k0+1],num=increment,endpoint=True)

            init_liom2,init_liom4  = jit_update(init_liom2,init_liom4,sol2_gpu[k0%chunk_size],sol4_gpu[k0%chunk_size],steps)

    liom_fwd2 = np.array(init_liom2)
    liom_fwd4 = np.array(init_liom4)
    

    if chunk > 1:
        del sol2_gpu
        del sol4_gpu
        sol2_gpu = jnp.zeros((chunk_size,n,n))
        sol4_gpu = jnp.zeros((chunk_size,n,n,n,n))

    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]
    # sol2=sol2[::-1]
    # sol4 = sol4[::-1]

    # Initialise a density operator in the diagonal basis on the central site
    init_liom2 = jnp.zeros((n,n))
    init_liom4 = jnp.zeros((n,n,n,n))
    init_liom2 = init_liom2.at[n//2,n//2].set(1.0)

    # Do backwards integration
    k0=0
    if chunk <= 1:
        for k0 in range(len(dl_list)-1):
            steps = np.linspace(dl_list[k0],dl_list[k0+1],num=increment,endpoint=True)
            init_liom2,init_liom4  = jit_update(init_liom2,init_liom4,sol2[-k0+1],sol4[-k0+1],steps)
    else:
        for k0 in range(len(dl_list)-1):
            if k0%chunk_size==0:
                count = int(k0/chunk_size)

                if count == 0 and ((sol2[-1*((count+1)*chunk_size)::]).shape == sol2_gpu.shape):
                    sol2_gpu = sol2_gpu.at[:,:].set(jnp.array(sol2[-1*((count+1)*chunk_size)::]))
                    sol4_gpu = sol4_gpu.at[:,:].set(jnp.array(sol4[-1*((count+1)*chunk_size)::]))
                elif count > 0 and (sol2[-1*((count+1)*chunk_size):-((count)*chunk_size)]).shape == sol2_gpu.shape:
                    sol2_gpu = sol2_gpu.at[:,:].set(jnp.array(sol2[-1*((count+1)*chunk_size):-((count)*chunk_size)]))
                    sol4_gpu = sol4_gpu.at[:,:].set(jnp.array(sol4[-1*((count+1)*chunk_size):-((count)*chunk_size)]))
                else:
                    sol2_gpu = jnp.array(sol2[0:-1*((count)*chunk_size)])
                    sol4_gpu = jnp.array(sol4[0:-1*((count)*chunk_size)])

            steps = np.linspace(dl_list[k0],dl_list[k0+1],num=increment,endpoint=True)
            # print('****')
            # print(k0)
            # print(sol2_gpu[-(k0)%chunk_size-1])
            # print(sol2[-k0-1])
            init_liom2,init_liom4  = jit_update(init_liom2,init_liom4,sol2_gpu[-(k0)%chunk_size-1],sol4_gpu[-(k0)%chunk_size-1],steps)
    
    # Reverse again to get these lists the right way around
    # dl_list = -1*dl_list[::-1]
    # sol2=sol2[::-1]
    # sol4 = sol4[::-1]

    # import matplotlib.pyplot as plt
    # plt.plot(jnp.log10(jnp.abs(jnp.diag(init_liom2.reshape(n,n)))))
    # plt.plot(jnp.log10(jnp.abs(jnp.diag(liom_fwd2.reshape(n,n)))),'--')

    output = {"H0_diag":np.array(H0_diag), "Hint":np.array(Hint2),"LIOM Interactions":lbits,"LIOM2":init_liom2,"LIOM4":init_liom4,"LIOM2_FWD":liom_fwd2,"LIOM4_FWD":liom_fwd4,"Invariant":inv2}
    if store_flow == True:
        output["flow2"] = np.array(sol2)
        output["flow4"] = np.array(sol4)
        output["dl_list"] = dl_list

    # Free up some memory
    # del sol2,sol4
    # gc.collect()

    return output
    
def flow_static_int_fwd(n,hamiltonian,dl_list,qmax,cutoff,method='jit',norm=True,Hflow=False,store_flow=False):
    """
    Diagonalise an initial interacting Hamiltonian and compute the integrals of motion.

    Note: this function does not compute the LIOMs in the conventional way. Rather, it starts with a local 
    operator in the initial basis and transforms it into the diagonal basis, essentially the inverse of 
    the process used to produce LIOMs conventionally. This bypasses the requirement to store the full 
    unitary transform in memory, meaning that only a single tensor of order O(L^4) needs to be stored at 
    each flow time step, dramatically increasing the accessible system sizes. However, use this with care
    as it is *not* a conventional LIOM, despite displaying essentially the same features, and should be 
    understood as such.

    Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        output : dict
            Dictionary containing diagonal Hamiltonian ("H0_diag","Hint"), LIOM interaction coefficient ("LIOM Interactions"),
            the LIOM on central site ("LIOM") and the value of the second invariant of the flow ("Invariant").
    
    """

    H2,Hint = hamiltonian.H2_spinless,hamiltonian.H4_spinless

    if store_flow == True:
        # Initialise array to hold solution at all flow times
        flow_list = jnp.zeros((qmax,2*(n**2+n**4)))

    # print('Memory64 required: MB', sol_int.nbytes/10**6)

    # Store initial interaction value and trace of initial H^2 for later error estimation
    delta = jnp.max(Hint)
    e1 = jnp.trace(jnp.dot(H2,H2))
    
    # Define integrator
    r_int = ode(int_ode_fwd).set_integrator('dopri5',nsteps=100,rtol=10**(-6),atol=10**(-6))
    
    # Set initial conditions
    init = jnp.zeros(2*(n**2+n**4),dtype=jnp.float32)
    init[:n**2] = ((H2)).reshape(n**2)
    init[n**2:n**2+n**4] = (Hint).reshape(n**4)

    # Initialise a density operator in the diagonal basis on the central site
    init_liom = jnp.zeros(n**2+n**4)
    init_liom2 = jnp.zeros((n,n))
    init_liom2[n//2,n//2] = 1.0
    init_liom[:n**2] = init_liom2.reshape(n**2)
    init[n**2+n**4:] = init_liom
    if store_flow == True:
        flow_list[0] = init

    r_int.set_initial_value(init,dl_list[0])
    r_int.set_f_params(n,[],method,norm,Hflow)

        
    # Numerically integrate the flow equations
    k = 1                       # Flow timestep index
    J0 = 10.                    # Seed value for largest off-diagonal term
    decay = 1
    index_list = indices(n)
    # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
    while r_int.successful() and k < qmax-1 and decay == 1:
        if Hflow == True:
            r_int.integrate(dl_list[k])
            step = r_int.y
            if store_flow == True:
               flow_list[k] = step

        # Commented out: code to zero all off-diagonal variables below some cutoff
        # sim = proc(r_int.y,n,cutoff)
        # sol_int[k] = sim

        # jnp.set_printoptions(suppress=True)
        # print((r_int.y)[:n**2])

        decay = cut(step,n,cutoff,index_list)

        mat = step[:n**2].reshape(n,n)
        off_diag = mat-jnp.diag(jnp.diag(mat))
        J0 = max(off_diag.reshape(n**2))
        k += 1 
    print(k,J0)

    # Truncate solution list and flow time list to max timestep reached
    dl_list = dl_list[:k-1]
    if store_flow == True:
        flow_list = flow_list[:k-1]
    
    liom = step[n**2+n**4::]
    step = step[:n**2+n**4]

    # Define final diagonal quadratic Hamiltonian
    H0_diag = step[:n**2].reshape(n,n)
    # Define final diagonal quartic Hamiltonian
    Hint2 = step[n**2::].reshape(n,n,n,n)   
    # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint2[i,i,j,j]
            HFint[i,j] += -Hint2[i,j,j,i]

    # Compute the difference in the second invariant of the flow at start and end
    # This acts as a measure of the unitarity of the transform
    Hflat = HFint.reshape(n**2)
    inv = 2*jnp.sum([d**2 for d in Hflat])
    e2 = jnp.trace(jnp.dot(H0_diag,H0_diag))
    inv2 = jnp.abs(e1 - e2 + ((2*delta)**2)*(n-1) - inv)/jnp.abs(e2+((2*delta)**2)*(n-1))

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.))

    # liom_all = jnp.sum([j**2 for j in liom])
    f2 = jnp.sum([j**2 for j in liom[0:n**2]])
    f4 = jnp.sum([j**2 for j in liom[n**2::]])
    print('LIOM',f2,f4)
    print('Hint max',jnp.max(jnp.abs(Hint2)))

    output = {"H0_diag":H0_diag, "Hint":Hint2,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":inv2}
    if store_flow == True:
        output["flow"] = flow_list
        output["dl_list"] = dl_list

        # Free up some memory
        del flow_list
        gc.collect()

    return output

def flow_dyn(n,hamiltonian,num,dl_list,qmax,cutoff,tlist,method='jit',store_flow=False):
    """
    Diagonalise an initial non-interacting Hamiltonian and compute the quench dynamics.

        Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        num : array, float
            Density operator n_i(t=0) to be time-evolved.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        tlist : array
            List of timesteps to return time-evolved operator n_i(t).
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        sol : array, float
            Final (diagonal) Hamiltonian
        central : array, float
            Local integral of motion (LIOM) computed on the central lattice site of the chain
    
    """
    H2 = hamiltonian.H2_spinless

    # Initialise array to hold solution at all flow times
    sol = jnp.zeros((qmax,n**2))
    sol[0] = (H2).reshape(n**2)

    # Define integrator
    r = ode(nonint_ode).set_integrator('dopri5', nsteps=1000)
    r.set_initial_value((H2).reshape(n**2),dl_list[0])
    r.set_f_params(n,method)
    
    # Numerically integrate the flow equations
    k = 1                       # Flow timestep index
    J0 = 10.                    # Seed value for largest off-diagonal term
    # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
    while r.successful() and k < qmax-1 and J0 > cutoff:
        r.integrate(dl_list[k])
        sol[k] = r.y
        mat = sol[k].reshape(n,n)
        off_diag = mat-jnp.diag(jnp.diag(mat))
        J0 = max(jnp.abs(off_diag.reshape(n**2)))
        k += 1
    print(k,J0)
    sol=sol[0:k-1]
    dl_list= dl_list[0:k-1]

    # Initialise a density operator in the diagonal basis on the central site
    # liom = jnp.zeros((qmax,n**2))
    init_liom = jnp.zeros((n,n))
    init_liom[n//2,n//2] = 1.0
    # liom[0,:n**2] = init_liom.reshape(n**2)
    
    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]

    # Define integrator for density operator
    n_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
    n_int.set_initial_value(init_liom.reshape(n**2),dl_list[0])

    # Numerically integrate the flow equations for the density operator 
    # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
    k0=1
    while n_int.successful() and k0 < len(dl_list[:k]):
        n_int.set_f_params(n,sol[-k0],method)
        n_int.integrate(dl_list[k0])
        liom = n_int.y
        k0 += 1
    
    # Take final value for the transformed density operator and reshape to a matrix
    # central = (liom.reshape(n,n))
    
    # Invert dl again back to original
    dl_list = dl_list[::-1] 

    # Define integrator for density operator again
    # This time we integrate from l=0 to l -> infinity
    num_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
    num_int.set_initial_value(num.reshape(n**2),dl_list[0])
    k0=1
    num=jnp.zeros((k,n**2))
    while num_int.successful() and k0 < k-1:
        num_int.set_f_params(n,sol[k0],method)
        num_int.integrate(dl_list[k0])
        num[k0] = num_int.y
        k0 += 1
    num = num[:k0-1]

    # Run non-equilibrium dynamics following a quench from CDW state
    # Returns answer *** in LIOM basis ***
    evolist = dyn_con(n,num[-1],sol[-1],tlist,method=method)
    print(evolist)

    # For each timestep, integrate back from l -> infinity to l=0
    # i.e. from LIOM basis back to original microscopic basis
    num_t_list = jnp.zeros((len(tlist),n**2))
    dl_list = dl_list[::-1] # Reverse dl for backwards flow
    for t0 in range(len(tlist)):
        num_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
        num_int.set_initial_value(evolist[t0],dl_list[0])

        k0=1
        while num_int.successful() and k0 < k-1:
            num_int.set_f_params(n,sol[-k0],method)
            num_int.integrate(dl_list[k0])
            k0 += 1
        num_t_list[t0] = num_int.y
        
    # Initialise a list to store the expectation value of time-evolved density operator at each timestep
    nlist = jnp.zeros(len(tlist))

    # Set up initial state as a CDW
    list1 = jnp.array([1. for i in range(n//2)])
    list2 = jnp.array([0. for i in range(n//2)])
    state = jnp.array([val for pair in zip(list1,list2) for val in pair])
    
    # Compute the expectation value <n_i(t)> for each timestep t
    n2list = num_t_list[::,:n**2]
    for t0 in range(len(tlist)):
        mat = n2list[t0].reshape(n,n)
        for i in range(n):
            nlist[t0] += (mat[i,i]*state[i]**2).real

    output = {"H0_diag":sol[-1].reshape(n,n),"LIOM":liom,"Invariant":0,"Density Dynamics":nlist}
    if store_flow == True:
        output["flow"] = sol
        output["dl_list"] = dl_list[::-1]

    return output

     
def flow_dyn_int_singlesite(n,hamiltonian,num,num_int,dl_list,qmax,cutoff,tlist,method='jit',store_flow=False):
    """
    Diagonalise an initial interacting Hamiltonian and compute the quench dynamics.

    This function will return a time-evolved number operator.

        Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        num : array, float
            Density operator n_i(t=0) to be time-evolved.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        tlist : array
            List of timesteps to return time-evolved operator n_i(t).
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        sol : array, float
            Final (diagonal) Hamiltonian
        central : array, float
            Local integral of motion (LIOM) computed on the central lattice site of the chain
    
    """
    H2 = hamiltonian.H2_spinless
    H4=hamiltonian.H4_spinless

    # Initialise array to hold solution at all flow times
    sol_int = jnp.zeros((qmax,n**2+n**4),dtype=jnp.float32)
    # print('Memory64 required: MB', sol_int.nbytes/10**6)
    
    # Initialise the first flow timestep
    init = jnp.zeros(n**2+n**4,dtype=jnp.float32)
    init[:n**2] = (H2).reshape(n**2)
    init[n**2:] = (H4).reshape(n**4)
    sol_int[0] = init

    # Define integrator
    r_int = ode(int_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
    r_int.set_initial_value(init,dl_list[0])
    r_int.set_f_params(n,[],method)
    
    
    # Numerically integrate the flow equations
    k = 1                       # Flow timestep index
    J0 = 10.                    # Seed value for largest off-diagonal term
    # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
    while r_int.successful() and k < qmax-1 and J0 > cutoff:
        r_int.integrate(dl_list[k])
        sol_int[k] = r_int.y
        mat = sol_int[k,0:n**2].reshape(n,n)
        off_diag = mat-jnp.diag(jnp.diag(mat))
        J0 = max(off_diag.reshape(n**2))
        k += 1

    # Truncate solution list and flow time list to max timestep reached
    sol_int=sol_int[:k-1]
    dl_list=dl_list[:k-1]

    # Define final Hamiltonian, for function return
    H0final,Hintfinal = sol_int[-1,:n**2].reshape(n,n),sol_int[-1,n**2::].reshape(n,n,n,n)
    
    # Define final diagonal quadratic Hamiltonian
    H0_diag = sol_int[-1,:n**2].reshape(n,n)
    # Define final diagonal quartic Hamiltonian
    Hint = sol_int[-1,n**2::].reshape(n,n,n,n)   
    # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint[i,i,j,j]
            HFint[i,j] += -Hint[i,j,j,i]

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.))

    # Initialise a density operator in the diagonal basis on the central site
    liom = jnp.zeros((k,n**2+n**4),dtype=jnp.float32)
    init_liom = jnp.zeros((n,n))
    init_liom[n//2,n//2] = 1.0
    liom[0,:n**2] = init_liom.reshape(n**2)
    
    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]

    # Define integrator for density operator
    n_int = ode(liom_ode).set_integrator('dopri5', nsteps=50)
    n_int.set_initial_value(liom[0],dl_list[0])

    # Numerically integrate the flow equations for the density operator 
    # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
    k0=1
    while n_int.successful() and k0 < k-1:
        n_int.set_f_params(n,sol_int[-k0],method)
        n_int.integrate(dl_list[k0])
        liom[k0] = n_int.y
        k0 += 1

    # Take final value for the transformed density operator and reshape quadratic part to a matrix
    central = (liom[k0-1,:n**2]).reshape(n,n)

    # Invert dl again back to original
    dl_list = dl_list[::-1] 

    # Define integrator for density operator again
    # This time we integrate from l=0 to l -> infinity
    num = jnp.zeros((k,n**2+n**4),dtype=jnp.float32)
    num_init = jnp.zeros((n,n))
    num_init[n//2,n//2] = 1.0
    num[0,0:n**2] = num_init.reshape(n**2)
    num_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
    num_int.set_initial_value(num[0],dl_list[0])

    # Integrate the density operator
    k0=1
    while num_int.successful() and k0 < k-1:
        num_int.set_f_params(n,sol_int[k0],method)
        num_int.integrate(dl_list[k0])
        k0 += 1
    num = num_int.y

    # Run non-equilibrium dynamics following a quench from CDW state
    # Returns answer *** in LIOM basis ***
    evolist2 = dyn_exact(n,num,sol_int[-1],tlist)
    # evolist2 = dyn_con(n,num,sol_int[-1],tlist)
    
    # For each timestep, integrate back from l -> infinity to l=0
    # i.e. from LIOM basis back to original microscopic basis
    num_t_list2 = jnp.zeros((len(tlist),n**2+n**4),dtype=jnp.complex128)
    dl_list = dl_list[::-1] # Reverse dl for backwards flow
    for t0 in range(len(tlist)):
        
        num_int = ode(liom_ode).set_integrator('dopri5',nsteps=100,atol=10**(-8),rtol=10**(-8))
        num_int.set_initial_value(evolist2[t0],dl_list[0])
        k0=1
        while num_int.successful() and k0 < k-1:
            num_int.set_f_params(n,sol_int[-k0],method,True)
            num_int.integrate(dl_list[k0])
            k0 += 1
        num_t_list2[t0] = (num_int.y).real

    # Initialise a list to store the expectation value of time-evolved density operator at each timestep
    nlist2 = jnp.zeros(len(tlist))

    # Set up initial state as a CDW
    list1 = jnp.array([1. for i in range(n//2)])
    list2 = jnp.array([0. for i in range(n//2)])
    state = jnp.array([val for pair in zip(list1,list2) for val in pair])
    
    # Compute the expectation value <n_i(t)> for each timestep t
    n2list = num_t_list2[::,:n**2]
    n4list = num_t_list2[::,n**2:]
    for t0 in range(len(tlist)):
        mat = n2list[t0].reshape(n,n)
        mat4 = n4list[t0].reshape(n,n,n,n)
        for i in range(n):
            nlist2[t0] += (mat[i,i]*state[i]).real
            for j in range(n):
                if i != j:
                    nlist2[t0] += (mat4[i,i,j,j]*state[i]*state[j]).real
                    nlist2[t0] += -(mat4[i,j,j,i]*state[i]*state[j]).real
    print(nlist2)

    output = {"H0_diag":H0_diag,"Hint":Hintfinal,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":0,"Density Dynamics":nlist2}
    if store_flow == True:
        output["flow"] = sol_int
        output["dl_list"] = dl_list[::-1]

    return output
 
    
def flow_dyn_int_imb(n,hamiltonian,num,num_int,dl_list,qmax,cutoff,tlist,method='jit',store_flow=False):
    """
    Diagonalise an initial interacting Hamiltonian and compute the quench dynamics.

    This function will return the imbalance following a quench, which involves computing the 
    non-equilibrium dynamics of the densiy operator on every single lattice site.

        Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        num : array, float
            Density operator n_i(t=0) to be time-evolved.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        tlist : array
            List of timesteps to return time-evolved operator n_i(t).
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        sol : array, float
            Final (diagonal) Hamiltonian
        central : array, float
            Local integral of motion (LIOM) computed on the central lattice site of the chain
    
    """

    H2 = hamiltonian.H2_spinless
    H4 = hamiltonian.H4_spinless

    # Initialise array to hold solution at all flow times
    sol_int = jnp.zeros((qmax,n**2+n**4),dtype=jnp.float32)
    # print('Memory64 required: MB', sol_int.nbytes/10**6)
    
    # Initialise the first flow timestep
    init = jnp.zeros(n**2+n**4,dtype=jnp.float32)
    init[:n**2] = (H2).reshape(n**2)
    init[n**2:] = (H4).reshape(n**4)
    sol_int[0] = init

    # Define integrator
    r_int = ode(int_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
    r_int.set_initial_value(init,dl_list[0])
    r_int.set_f_params(n,[],method)
    
    
    # Numerically integrate the flow equations
    k = 1                       # Flow timestep index
    J0 = 10.                    # Seed value for largest off-diagonal term
    # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
    while r_int.successful() and k < qmax-1 and J0 > cutoff:
        r_int.integrate(dl_list[k])
        sol_int[k] = r_int.y
        mat = sol_int[k,0:n**2].reshape(n,n)
        off_diag = mat-jnp.diag(jnp.diag(mat))
        J0 = max(off_diag.reshape(n**2))
        k += 1

    # Truncate solution list and flow time list to max timestep reached
    sol_int=sol_int[:k-1]
    dl_list=dl_list[:k-1]

    # Define final Hamiltonian, for function return
    H0final,Hintfinal = sol_int[-1,:n**2].reshape(n,n),sol_int[-1,n**2::].reshape(n,n,n,n)
    
    # Define final diagonal quadratic Hamiltonian
    H0_diag = sol_int[-1,:n**2].reshape(n,n)
    # Define final diagonal quartic Hamiltonian
    Hint = sol_int[-1,n**2::].reshape(n,n,n,n)   
    # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint[i,i,j,j]
            HFint[i,j] += -Hint[i,j,j,i]

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.))

    # Initialise a density operator in the diagonal basis on the central site
    liom = jnp.zeros((k,n**2+n**4),dtype=jnp.float32)
    init_liom = jnp.zeros((n,n))
    init_liom[n//2,n//2] = 1.0
    liom[0,:n**2] = init_liom.reshape(n**2)
    
    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]
    
    # Set up initial state as a CDW
    list1 = jnp.array([1. for i in range(n//2)])
    list2 = jnp.array([0. for i in range(n//2)])
    state = jnp.array([val for pair in zip(list1,list2) for val in pair])
    
    # Define lists to store the time-evolved density operators on each lattice site
    # 'imblist' will include interaction effects
    # 'imblist2' includes only single-particle effects
    # Both are kept to check for diverging interaction terms
    imblist = jnp.zeros((n,len(tlist)))
    imblist2 = jnp.zeros((n,len(tlist)))

    # Compute the time-evolution of the number operator on every site
    for site in range(n):
        # Initialise operator to be time-evolved
        num = jnp.zeros((k,n**2+n**4))
        num_init = jnp.zeros((n,n),dtype=jnp.float32)
        num_init[site,site] = 1.0

        num[0,0:n**2] = num_init.reshape(n**2)
        
            # Invert dl again back to original
        dl_list = dl_list[::-1]

        # Define integrator for density operator again
        # This time we integrate from l=0 to l -> infinity
        num_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        num_int.set_initial_value(num[0],dl_list[0])
        k0=1
        while num_int.successful() and k0 < k-1:
            num_int.set_f_params(n,sol_int[k0],method)
            num_int.integrate(dl_list[k0])
            # liom[k0] = num_int.y
            k0 += 1
        num = num_int.y
        
        # Run non-equilibrium dynamics following a quench from CDW state
        # Returns answer *** in LIOM basis ***
        evolist2 = dyn_exact(n,num,sol_int[-1],tlist)
        dl_list = dl_list[::-1] # Reverse the flow
        
        num_t_list2 = jnp.zeros((len(tlist),n**2+n**4))
        # For each timestep, integrate back from l -> infinity to l=0
        # i.e. from LIOM basis back to original microscopic basis
        for t0 in range(len(tlist)):
            
            num_int = ode(liom_ode).set_integrator('dopri5',nsteps=50,atol=10**(-8),rtol=10**(-8))
            num_int.set_initial_value(evolist2[t0],dl_list[0])
            k0=1
            while num_int.successful() and k0 < k-1:
                num_int.set_f_params(n,sol_int[-k0],method,True)
                num_int.integrate(dl_list[k0])
                k0 += 1
            num_t_list2[t0] = num_int.y
        
        # Initialise lists to store the expectation value of time-evolved density operator at each timestep
        nlist = jnp.zeros(len(tlist))
        nlist2 = jnp.zeros(len(tlist))
        
        # Compute the expectation value <n_i(t)> for each timestep t
        n2list = num_t_list2[::,:n**2]
        n4list = num_t_list2[::,n**2:]
        for t0 in range(len(tlist)):
            mat = n2list[t0].reshape(n,n)
            mat4 = n4list[t0].reshape(n,n,n,n)
            # phaseMF = 0.
            for i in range(n):
                # nlist[t0] += (mat[i,i]*state[i]**2).real
                nlist[t0] += (mat[i,i]*state[i]).real
                nlist2[t0] += (mat[i,i]*state[i]).real
                for j in range(n):
                    if i != j:
                        nlist[t0] += (mat4[i,i,j,j]*state[i]*state[j]).real
                        nlist[t0] += -(mat4[i,j,j,i]*state[i]*state[j]).real
                        
        imblist[site] = ((-1)**site)*nlist/n
        imblist2[site] = ((-1)**site)*nlist2/n

    # Compute the imbalance over the entire system
    # Note that the (-1)^i factors are included in imblist already
    imblist = 2*jnp.sum(imblist,axis=0)
    imblist2 = 2*jnp.sum(imblist2,axis=0)

    output = {"H0_diag":H0_diag,"Hint":Hintfinal,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":0,"Imbalance":imblist}
    if store_flow == True:
        output["flow"] = sol_int
        output["dl_list"] = dl_list[::-1]

    return output

#------------------------------------------------------------------------------
# Function for benchmarking the non-interacting system using 'einsum'
def flow_einsum_nonint(H0,V0,dl):
    """ Benchmarking function to diagonalise H for a non-interacting system using NumPy's einsum function.

        This function is used to test the routines included in contract.py by explicitly calling 
        the 'einsum' function, which is a slow but very transparent way to do the matrix/tensor contractions.

        Parameters
        ----------
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        dl : float
            Size of step in flow time (dl << 1)
    
    """
      
    startTime = datetime.now()
    q = 0
    while jnp.max(jnp.abs(V0))>10**(-2):
    
        # Non-interacting generator
        eta0 = jnp.einsum('ij,jk->ik',H0,V0) - jnp.einsum('ki,ij->kj',V0,H0,optimize=True)
        
        # Flow of non-interacting terms
        dH0 = jnp.einsum('ij,jk->ik',eta0,(H0+V0)) - jnp.einsum('ki,ij->kj',(H0+V0),eta0,optimize=True)
    
        # Update non-interacting terms
        H0 = H0+dl*jnp.diag(jnp.diag(dH0))
        V0 = V0 + dl*(dH0-jnp.diag(jnp.diag(dH0)))

        q += 1

    print('***********')
    print('FE time - einsum',datetime.now()-startTime)
    print('Max off diagonal element: ', jnp.max(jnp.abs(V0)))
    print(jnp.sort(jnp.diag(H0)))

#------------------------------------------------------------------------------  
# Function for benchmarking the non-interacting system using 'tensordot'
def flow_tensordot_nonint(H0,V0,dl):  

    """ Benchmarking function to diagonalise H for a non-interacting system using NumPy's tensordot function.

        This function is used to test the routines included in contract.py by explicitly calling 
        the 'tensordot' function, which is slightly faster than einsum but also less transparent.

        Parameters
        ----------
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        dl : float
            Size of step in flow time (dl << 1)
    
    """   

    startTime = datetime.now()
    q = 0
    while jnp.max(jnp.abs(V0))>10**(-3):
    
        # Non-interacting generator
        eta = jnp.tensordot(H0,V0,axes=1) - jnp.tensordot(V0,H0,axes=1)
        
        # Flow of non-interacting terms
        dH0 = jnp.tensordot(eta,H0+V0,axes=1) - jnp.tensordot(H0+V0,eta,axes=1)
    
        # Update non-interacting terms
        H0 = H0+dl*jnp.diag(jnp.diag(dH0))
        V0 = V0 + dl*(dH0-jnp.diag(jnp.diag(dH0)))

        q += 1
        
    print('***********')
    print('FE time - Tensordot',datetime.now()-startTime)
    print('Max off diagonal element: ', jnp.max(jnp.abs(V0)))
    print(jnp.sort(jnp.diag(H0)))




