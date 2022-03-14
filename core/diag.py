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
import numpy as np
from datetime import datetime
from .dynamics import dyn_con,dyn_exact,dyn_mf
from numba import jit,prange
import gc
from .contract import contract,contractNO,contractNO2
from .utility import nstate
from scipy.integrate import ode
# import matplotlib.pyplot as plt

#------------------------------------------------------------------------------ 
def CUT(params,hamiltonian,num,num_int):
    """ Function to take input Hamiltonian and other variables, then call the appropriate flow method subroutine. """

    n = params["n"]
    logflow = params["logflow"]
    lmax = params["lmax"]
    qmax = params["qmax"]
    dyn = params["dyn"]
    intr = params["intr"]
    imbalance = params["imbalance"]
    cutoff = params["cutoff"]
    method = params["method"]
    tlist = params["tlist"]
    precision = params["precision"]
    norm = params["norm"]
    Hflow = params["Hflow"]
    LIOM = params["LIOM"]
    store_flow = params["store_flow"]

    if logflow == False:
            dl = np.linspace(0,lmax,qmax,endpoint=True)
    elif logflow == True:
        print('Warning: careful choices of qmax and lmax required for log flow.')
        dl = np.logspace(np.log10(0.01), np.log10(lmax),qmax,endpoint=True,base=10)

    if hamiltonian.species == 'spinless fermion':
        H2 = hamiltonian.H2_spinless

        # Fix this later:
        H0 = np.diag(np.diag(H2))
        V0 = H2 - H0
        Vint = np.zeros((n,n,n,n))
        
        if dyn == True:
            if intr == True:
                # Hint = hamiltonian.H4_spinless
                if imbalance == True:
                    flow = flow_dyn_int_imb(n,hamiltonian,num,num_int,dl,qmax,cutoff,tlist,method=method,store_flow=store_flow)
                else:
                    flow = flow_dyn_int_singlesite(n,hamiltonian,num,num_int,dl,qmax,cutoff,tlist,method=method,store_flow=store_flow)
            elif intr == False:
                flow = flow_dyn(n,hamiltonian,num,dl,qmax,cutoff,tlist,method=method,store_flow=store_flow)
        elif dyn == False:
            if intr == True:
                if LIOM == 'bck':
                    flow = flow_static_int(n,hamiltonian,dl,qmax,cutoff,method=method,precision=precision,norm=norm,Hflow=Hflow,store_flow=store_flow)
                elif LIOM == 'fwd':
                    flow = flow_static_int_fwd(n,hamiltonian,dl,qmax,cutoff,method=method,precision=precision,norm=norm,Hflow=Hflow,store_flow=store_flow)
            elif intr == False:
                flow = flow_static(n,hamiltonian,dl,qmax,cutoff,method=method,store_flow=store_flow)

        return flow

    elif hamiltonian.species == 'spinful fermion':

        flow = flow_static_int_spin(n,hamiltonian,dl,qmax,cutoff,method=method,store_flow=store_flow,norm=norm)

        return flow

    else:
        print('ERROR: Unknown type of particle.')


def indices(n):
    """ Gets indices of off-diagonal elements of quartic tensor when shaped as a list of length n**4. """
    mat = np.ones((n,n,n,n),dtype=np.int8)    
    for i in range(n):              
        for j in range(n):
            if i != j:
                # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                mat[i,i,j,j] = 0
                mat[i,j,j,i] = 0
    mat = mat.reshape(n**4)
    indices = np.nonzero(mat)

    return indices

def cut(y,n,cutoff,indices):
    """ Checks if ALL quadratic off-diagonal parts have decayed below cutoff*10e-3 and TYPICAL (median) off-diag quartic term have decayed below cutoff. """
    mat2 = y[:n**2].reshape(n,n)
    mat2_od = mat2-np.diag(np.diag(mat2))

    if np.max(np.abs(mat2_od)) < cutoff*10**(-3):
        mat4 = y[n**2:n**2+n**4]
        mat4_od = np.zeros(n**4)            # Define diagonal quartic part 
        for i in indices:                   # Load Hint0 with values
            mat4_od[i] = mat4[i]
        mat4_od = mat4_od[mat4_od != 0]
        if np.median(np.abs(mat4_od)) < cutoff:
            return 0 
        else:
            return 1
    else:
        return 1

def nonint_ode(l,y,n,method='einsum'):
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
    H = y.reshape(n,n)
    H0 = np.diag(np.diag(H))
    V0 = H - H0
    eta = contract(H0,V0,method=method,eta=True)
    sol = contract(eta,H,method=method,eta=False)
    sol = sol.reshape(n**2)

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
    H0 = np.diag(np.diag(H))
    V0 = H - H0

    # Extract quartic parts of Hamiltonian from array y
    Hint = y[n**2::]
    Hint = Hint.reshape(n,n,n,n)
    Hint0 = np.zeros((n,n,n,n))
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
        # print(eta_no2)
        eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
        eta4 += eta_no4

    # Combine into array
    eta = np.zeros(n**2+n**4)
    eta[:n**2] = eta2.reshape(n**2)
    eta[n**2:] = eta4.reshape(n**4)

    return eta

#------------------------------------------------------------------------------

def int_ode(l,y,n,eta=[],method='jit',norm=False,Hflow=True):
        """ Generate the flow equation for the interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the input array eta will be used to specify the generator at this flow time step. The latter option will result 
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
        
        # Extract various components of the Hamiltonian from the input array 'y'
        H = y[0:n**2]                   # Define quadratic part of Hamiltonian
        H = H.reshape(n,n)              # Reshape into matrix
        H0 = np.diag(np.diag(H))        # Define diagonal quadratic part H0
        V0 = H - H0                     # Define off-diagonal quadratic part B

        Hint = y[n**2:]                 # Define quartic part of Hamiltonian
        Hint = Hint.reshape(n,n,n,n)    # Reshape into rank-4 tensor
        Hint0 = np.zeros((n,n,n,n))     # Define diagonal quartic part 
        for i in range(n):              # Load Hint0 with values
            for j in range(n):
                if i != j:
                    # if norm == True:
                        # # Symmetrise (for normal-ordering)
                        # Hint[i,i,j,j] += Hint[j,j,i,i]
                        # Hint[i,i,j,j] *= 0.5
                        # Hint[i,i,j,j] += -Hint[i,j,j,i]
                        # # Hint[i,i,j,j] += -Hint[j,i,i,j]
                        # Hint[i,j,j,i] = 0.
                    # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                    Hint0[i,i,j,j] = Hint[i,i,j,j]
                    
        Vint = Hint-Hint0

        if norm == True:
            state=nstate(n,'CDW')

            ## The below is an attempt at scale-dependent normal-ordering: not yet working reliably.
            # _,V1 = np.linalg.eigh(H)
            # state = np.zeros(n)
            # sites = np.array([i for i in range(n)])
            # random = np.random.choice(sites,n//2)
            # for i in random:
            #     psi = V1[:,i]
            #     state += np.array([v**2 for v in psi])
            # state = np.round(state,decimals=3)
            # # print(np.dot(state,state))

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

        # Add normal-ordering corrections into flow equation, if norm == True
        if norm == True:
            sol_no = contractNO(eta_int,H0+V0,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
            sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
            sol+=sol_no
            sol2 += sol4_no
        
        # Define and load output list sol0
        sol0 = np.zeros(n**2+n**4)
        sol0[:n**2] = sol.reshape(n**2)
        sol0[n**2:] = sol2.reshape(n**4)

        return sol0

def int_ode_spin(l,y,n,method='jit',norm=True):
        """ Generate the flow equation for an interacting system of SPINFUL fermions.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the input array eta will be used to specify the generator at this flow time step. The latter option will result 
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
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.


        Returns
        -------
        sol0 : RHS of the flow equation for interacting system.

        """

        # Extract various components of the Hamiltonian from the input array 'y'
        # Start with the quadratic part of the spin-up fermions
        Hup = y[0:n**2]
        Hup = Hup.reshape(n,n)
        if norm == True:
            # Symmetrise
            Hup += Hup.T
            Hup *= 0.5
        H0up = np.diag(np.diag(Hup))
        V0up = Hup - H0up
        
        # Now the quadratic part of the spin-down fermions
        Hdown = y[n**2:2*n**2]
        Hdown = Hdown.reshape(n,n)
        if norm == True:
            # Symmetrise
            Hdown += Hdown.T
            Hdown *= 0.5
        H0down = np.diag(np.diag(Hdown))
        V0down = Hdown - H0down

        # Now we define the quartic (interaction) terms for the spin-up fermions
        Hintup = y[2*n**2:2*n**2+n**4]
        Hintup = Hintup.reshape(n,n,n,n)
        Hint0up = np.zeros((n,n,n,n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    if norm == True:
                        # Re-order interaction terms
                        Hintup[i,i,j,j] += Hintup[j,j,i,i]
                        Hintup[i,i,j,j] *= 0.5
                        Hintup[i,i,j,j] += -Hintup[i,j,j,i]
                        Hintup[i,j,j,i] = 0.

                    # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                    Hint0up[i,i,j,j] = Hintup[i,i,j,j]
                    # Hint0up[i,j,j,i] = Hintup[i,j,j,i]
        Vintup = Hintup-Hint0up
        
        # The same for spin-down fermions
        Hintdown = y[2*n**2+n**4:2*n**2+2*n**4]
        Hintdown = Hintdown.reshape(n,n,n,n)
        Hint0down = np.zeros((n,n,n,n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    if norm == True:
                        # Re-order interaction terms
                        Hintdown[i,i,j,j] += Hintdown[j,j,i,i]
                        Hintdown[i,i,j,j] *= 0.5
                        Hintdown[i,i,j,j] += -Hintdown[i,j,j,i]
                        Hintdown[i,j,j,i] = 0.

                    # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                    Hint0down[i,i,j,j] = Hintdown[i,i,j,j]
                    # Hint0down[i,j,j,i] = Hintdown[i,j,j,i]
        Vintdown = Hintdown-Hint0down

        # And the same for the mixed quartic term, with 2 spin-up fermion operators and 2 spin-down fermion operators
        Hintupdown = y[2*n**2+2*n**4:]
        Hintupdown = Hintupdown.reshape(n,n,n,n)
        Hint0updown = np.zeros((n,n,n,n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    if norm == True:
                        # Re-order interaction terms
                        Hintupdown[i,i,j,j] += Hintupdown[j,j,i,i]
                        Hintupdown[i,i,j,j] *= 0.5

                    # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                Hint0updown[i,i,j,j] = Hintupdown[i,i,j,j]
                # Hint0updown[i,j,j,i] = Hintupdown[i,j,j,i]
        Vintupdown = Hintupdown-Hint0updown                 
        
        upstate=nstate(n,'CDW')
        downstate=np.array([np.abs(1-i) for i in upstate])
        # downstate = upstate

        # Compute all relevant generators
        eta0up = contract(H0up,V0up,method=method,eta=True)
        eta0down = contract(H0down,V0down,method=method,eta=True)
        eta_int_up = contract(Hint0up,V0up,method=method,eta=True) + contract(H0up,Vintup,method=method,eta=True)
        eta_int_down = contract(Hint0down,V0down,method=method,eta=True) + contract(H0down,Vintdown,method=method,eta=True)
        eta_int_updown = -contract(Vintupdown,H0up,method=method,eta=True,pair='first') - contract(Vintupdown,H0down,method=method,eta=True,pair='second')
        eta_int_updown += contract(Hint0updown,V0up,method=method,eta=True,pair='first') + contract(Hint0updown,V0down,method=method,eta=True,pair='second')
   
        if norm == True:
            eta0up += contractNO(Hint0up,V0up,method=method,eta=True,state=upstate)
            eta0up += contractNO(H0up,Vintup,method=method,eta=True,state=upstate)
            eta0down += contractNO(Hint0down,V0down,method=method,eta=True,state=downstate)
            eta0down += contractNO(H0down,Vintdown,method=method,eta=True,state=downstate)
            eta0up += contractNO(Hint0updown,V0down,method=method,eta=True,state=downstate,pair='second')
            eta0up += contractNO(H0down,Vintupdown,method=method,eta=True,state=downstate,pair='second')
            eta0down += contractNO(Hint0updown,V0up,method=method,eta=True,state=upstate,pair='first')
            eta0down += contractNO(H0up,Vintupdown,method=method,eta=True,state=upstate,pair='first')

            eta_int_up += contractNO(Hint0up,Vintup,method=method,eta=True,state=upstate)
            eta_int_down += contractNO(Hint0down,Vintdown,method=method,eta=True,state=downstate)

            eta_int_updown += contractNO(Hint0up,Vintupdown,method=method,eta=True,pair='up-mixed',state=upstate)
            eta_int_up += contractNO(Hint0updown,Vintupdown,method=method,eta=True,pair='mixed-mixed-up',state=downstate)
            eta_int_updown += contractNO(Hint0down,Vintupdown,method=method,eta=True,pair='down-mixed',state=downstate)
            eta_int_down += contractNO(Hint0updown,Vintupdown,method=method,eta=True,pair='mixed-mixed-down',state=upstate)
            eta_int_updown += contractNO(Hint0updown,Vintup,method=method,eta=True,pair='mixed-up',state=upstate)
            eta_int_updown += contractNO(Hint0updown,Vintdown,method=method,eta=True,pair='mixed-down',state=downstate)

            eta_int_updown += contractNO(Hint0updown,Vintupdown,method=method,eta=True,pair='mixed',upstate=upstate,downstate=downstate)


        # Then compute the RHS of the flow equations
        sol_up = contract(eta0up,H0up+V0up,method=method)
        sol_down = contract(eta0down,H0down+V0down,method=method)
        sol_int_up = contract(eta_int_up,H0up+V0up,method=method) + contract(eta0up,Hintup,method=method)
        sol_int_down = contract(eta_int_down,H0down+V0down,method=method) + contract(eta0down,Hintdown,method=method)
        sol_int_updown = contract(eta_int_updown,H0down+V0down,method=method,pair='second') + contract(eta0down,Hintupdown,method=method,pair='second')
        sol_int_updown += contract(eta_int_updown,H0up+V0up,method=method,pair='first') + contract(eta0up,Hintupdown,method=method,pair='first')

        if norm == True:
            sol_up += contractNO(eta_int_up,Hup,method=method,state=upstate)
            sol_up += contractNO(eta0up,Hintup,method=method,state=upstate)
            sol_down += contractNO(eta_int_down,Hdown,method=method,state=downstate)
            sol_down += contractNO(eta0down,Hintdown,method=method,state=downstate)
            sol_up += contractNO(eta_int_updown,Hdown,method=method,state=downstate,pair='second')
            sol_up += contractNO(eta0down,Hintupdown,method=method,state=downstate,pair='second')
            sol_down += contractNO(eta_int_updown,Hup,method=method,state=upstate,pair='first')
            sol_down += contractNO(eta0up,Hintupdown,method=method,state=upstate,pair='first')

            sol_int_up += contractNO(eta_int_up,Hintup,method=method,state=upstate)
            sol_int_down += contractNO(eta_int_down,Hintdown,method=method,state=downstate)

            sol_int_updown += contractNO(eta_int_up,Hintupdown,method=method,pair='up-mixed',state=upstate)
            sol_int_up += contractNO(eta_int_updown,Hintupdown,method=method,pair='mixed-mixed-up',state=downstate)
            sol_int_updown += contractNO(eta_int_down,Hintupdown,method=method,pair='down-mixed',state=downstate)
            sol_int_down += contractNO(eta_int_updown,Hintupdown,method=method,pair='mixed-mixed-down',state=upstate)
            sol_int_updown += contractNO(eta_int_updown,Hintup,method=method,pair='mixed-up',state=upstate)
            sol_int_updown += contractNO(eta_int_updown,Hintdown,method=method,pair='mixed-down',state=downstate)

            sol_int_updown += contractNO(eta_int_updown,Hintupdown,method=method,pair='mixed',upstate=upstate,downstate=downstate)

        # Assemble output array
        sol0 = np.zeros(2*n**2+3*n**4)
        sol0[:n**2] = sol_up.reshape(n**2)
        sol0[n**2:2*n**2] = sol_down.reshape(n**2)
        sol0[2*n**2:2*n**2+n**4] = sol_int_up.reshape(n**4)
        sol0[2*n**2+n**4:2*n**2+2*n**4] = sol_int_down.reshape(n**4)
        sol0[2*n**2+2*n**4:] = sol_int_updown.reshape(n**4)

        return sol0

def liom_ode(l,y,n,array,method='jit',comp=False,Hflow=True,norm=False):
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

    array=array.astype(np.float64)
    norm = False
    state=nstate(n,'CDW')
    if Hflow == True:
        # Extract various components of the Hamiltonian from the input array 'y'
        H2 = array[0:n**2]                  # Define quadratic part of Hamiltonian
        H2 = H2.reshape(n,n)                # Reshape into matrix
        H0 = np.diag(np.diag(H2))            # Define diagonal quadratic part H0
        V0 = H2 - H0                        # Define off-diagonal quadratic part B

        if len(array)>n**2:
            Hint = array[n**2::]            # Define quartic part of Hamiltonian
            Hint = Hint.reshape(n,n,n,n)    # Reshape into rank-4 tensor
            Hint0 = np.zeros((n,n,n,n))     # Define diagonal quartic part 
            for i in range(n):              # Load Hint0 with values
                for j in range(n):
                    if i != j:
                        # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                        Hint0[i,i,j,j] = Hint[i,i,j,j]
                        Hint0[i,j,j,i] = Hint[i,j,j,i]
            Vint = Hint-Hint0

        # Compute the quadratic generator eta2
        eta2 = contract(H0,V0,method=method,comp=False,eta=True)

        if len(array) > n**2:
            eta4 = contract(Hint0,V0,method=method,comp=comp,eta=True) + contract(H0,Vint,method=method,comp=comp,eta=True)

        # Add normal-ordering corrections into generator eta, if norm == True
        if norm == True:

            eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
            eta2 += eta_no2

            eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
            eta4 += eta_no4

    else:
        eta2 = (array[0:n**2]).reshape(n,n)
        eta4 = (array[n**2::]).reshape(n,n,n,n)

    # Extract components of the density operator from input array 'y'
    n2 = y[0:n**2]                  # Define quadratic part of density operator
    n2 = n2.reshape(n,n)            # Reshape into matrix
    if len(y)>n**2:                 # If interacting system...
        n4 = y[n**2::]              #...then define quartic part of density operator
        n4 = n4.reshape(n,n,n,n)    # Reshape into tensor
                    
    # Compute the quadratic terms in the RHS of the flow equation
    sol = contract(eta2,n2,method=method,comp=comp)

    # Define output array as either real or complex, as required
    if comp == False:
        sol0 = np.zeros(len(y))
    elif comp == True:
        sol0 = np.zeros(len(y),dtype=complex)

    # Compute quartic terms, if interacting system
    if len(y) > n**2:
        sol2 = contract(eta4,n2,method=method,comp=comp) + contract(eta2,n4,method=method,comp=comp)

    # Add normal-ordering corrections into flow equation, if norm == True
    if norm == True:
        sol_no = contractNO(eta4,n2,method=method,eta=False,state=state) + contractNO(eta2,n4,method=method,eta=False,state=state)
        sol+=sol_no
        if len(y) > n**2:
            sol4_no = contractNO(eta4,n4,method=method,eta=False,state=state)
            sol2 += sol4_no

    # Load solution into output array
    sol0[:n**2] = sol.reshape(n**2)
    if len(y)> n**2:
        sol0[n**2:] = sol2.reshape(n**4)

    return sol0

def liom_spin(l,nlist,y,n,method='jit',comp=False,norm=False):

    Hup = y[0:n**2]
    Hup = Hup.reshape(n,n)
    if norm == True:
        # Symmetrise
        Hup += Hup.T
        Hup *= 0.5
    H0up = np.diag(np.diag(Hup))
    V0up = Hup - H0up
    
    Hdown = y[n**2:2*n**2]
    Hdown = Hdown.reshape(n,n)
    if norm == True:
        # Symmetrise
        Hdown += Hdown.T
        Hdown *= 0.5
    H0down = np.diag(np.diag(Hdown))
    V0down = Hdown - H0down

    Hintup = y[2*n**2:2*n**2+n**4]
    Hintup = Hintup.reshape(n,n,n,n)
    Hint0up = np.zeros((n,n,n,n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if norm == True:
                        # Re-order interaction terms
                        Hintup[i,i,j,j] += Hintup[j,j,i,i]
                        Hintup[i,i,j,j] *= 0.5
                        Hintup[i,i,j,j] += -Hintup[i,j,j,i]
                        Hintup[i,j,j,i] = 0.

                # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                Hint0up[i,i,j,j] = Hintup[i,i,j,j]
                Hint0up[i,j,j,i] = Hintup[i,j,j,i]
    Vintup = Hintup-Hint0up
    
    Hintdown = y[2*n**2+n**4:2*n**2+2*n**4]
    Hintdown = Hintdown.reshape(n,n,n,n)
    Hint0down = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if norm == True:
                        # Re-order interaction terms
                        Hintdown[i,i,j,j] += Hintdown[j,j,i,i]
                        Hintdown[i,i,j,j] *= 0.5
                        Hintdown[i,i,j,j] += -Hintdown[i,j,j,i]
                        Hintdown[i,j,j,i] = 0.
                # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                Hint0down[i,i,j,j] = Hintdown[i,i,j,j]
                Hint0down[i,j,j,i] = Hintdown[i,j,j,i]
    Vintdown = Hintdown-Hint0down
    # print(Vintdown)
    
    Hintupdown = y[2*n**2+2*n**4:]
    Hintupdown = Hintupdown.reshape(n,n,n,n)
    Hint0updown = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if norm == True:
                        # Re-order interaction terms
                        Hintupdown[i,i,j,j] += Hintupdown[j,j,i,i]
                        Hintupdown[i,i,j,j] *= 0.5

                # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
            Hint0updown[i,i,j,j] = Hintupdown[i,i,j,j]
            # Hint0updown[i,j,j,i] = Hintupdown[i,j,j,i]
    Vintupdown = Hintupdown-Hint0updown               
    
    eta0up = contract(H0up,V0up,method=method,eta=True)
    eta0down = contract(H0down,V0down,method=method,eta=True)
    eta_int_up = contract(Hint0up,V0up,method=method,eta=True) + contract(H0up,Vintup,method=method,eta=True)
    eta_int_down = contract(Hint0down,V0down,method=method,eta=True) + contract(H0down,Vintdown,method=method,eta=True)
    eta_int_updown = -contract(Vintupdown,H0up,method=method,eta=True,pair='first') - contract(Vintupdown,H0down,method=method,eta=True,pair='second')
    eta_int_updown += contract(Hint0updown,V0up,method=method,eta=True,pair='first') + contract(Hint0updown,V0down,method=method,eta=True,pair='second')

    if norm == True:
        upstate=nstate(n,'CDW')
        downstate=np.array([np.abs(1-i) for i in upstate])

        eta0up += contractNO2(Hint0up,V0up,method=method,eta=True,state=upstate)
        eta0down += contractNO2(Hint0down,V0down,method=method,eta=True,state=downstate)
        eta0up += contractNO2(Hint0updown,V0down,method=method,eta=True,state=downstate,pair='second')
        eta0down += contractNO2(Hint0updown,V0up,method=method,eta=True,state=upstate,pair='first')
        eta_int_updown += contractNO(Hint0up,Vintupdown,method=method,eta=True,pair='up-mixed',state=upstate)
        eta_int_up += contractNO(Hint0updown,Vintupdown,method=method,eta=True,pair='mixed-mixed-up',state=downstate)
        eta_int_updown += contractNO(Hint0down,Vintupdown,method=method,eta=True,pair='down-mixed',state=upstate)
        eta_int_down += contractNO(Hint0updown,Vintupdown,method=method,eta=True,pair='mixed-mixed-down',state=upstate)
        eta_int_updown += contractNO(Hint0updown,Vintup,method=method,eta=True,pair='mixed-up',state=upstate)
        eta_int_updown += contractNO(Hint0updown,Vintdown,method=method,eta=True,pair='mixed-down',state=downstate)

    # print(eta_int_updown)
                
    n2_up = nlist[0:n**2]
    n2_up = n2_up.reshape(n,n)
    
    n2_down = nlist[n**2:2*n**2]
    n2_down = n2_down.reshape(n,n)

    n4_up = nlist[2*n**2:2*n**2+n**4]
    n4_up = n4_up.reshape(n,n,n,n)

    n4_down = nlist[2*n**2+n**4:2*n**2+2*n**4]
    n4_down = n4_down.reshape(n,n,n,n)
    n4_down = np.zeros((n,n,n,n))

    n4_updown = nlist[2*n**2+2*n**4:]
    n4_updown = n4_updown.reshape(n,n,n,n)

    sol_up = contract(eta0up,n2_up,method=method)
    sol_down = contract(eta0down,n2_down,method=method)
    sol_int_up = contract(eta_int_up,n2_up,method=method) + contract(eta0up,n4_up,method=method)
    sol_int_down = contract(eta_int_down,n2_down,method=method) + contract(eta0down,n4_down,method=method)
    sol_int_updown = contract(eta_int_updown,n2_down,method=method,pair='second') + contract(eta0down,n4_updown,method=method,pair='second')
    sol_int_updown += contract(eta_int_updown,n2_up,method=method,pair='first') + contract(eta0up,n4_updown,method=method,pair='first')
    
    if norm == True:
            sol_up += contractNO2(eta_int_up,n2_up,method=method,state=upstate)
            sol_down += contractNO2(eta_int_down,n2_down,method=method,state=downstate)
            sol_up += contractNO2(eta_int_updown,n2_down,method=method,state=downstate,pair='second')
            sol_down += contractNO2(eta_int_updown,n2_up,method=method,state=upstate,pair='first')
            sol_int_up += contractNO(eta_int_up,n4_updown,method=method,pair='up-mixed',state=upstate)
            sol_int_up += contractNO(eta_int_updown,n4_updown,method=method,pair='mixed-mixed-up',state=downstate)
            sol_int_down += contractNO(eta_int_down,n4_updown,method=method,pair='down-mixed',state=upstate)
            sol_int_down += contractNO(eta_int_updown,n4_updown,method=method,pair='mixed-mixed-down',state=upstate)
            sol_int_updown += contractNO(eta_int_updown,n4_up,method=method,pair='mixed-up',state=upstate)
            sol_int_updown += contractNO(eta_int_updown,n4_down,method=method,pair='mixed-down',state=downstate)

    sol0 = np.zeros(2*n**2+3*n**4)
    sol0[:n**2] = sol_up.reshape(n**2)
    sol0[n**2:2*n**2] = sol_down.reshape(n**2)
    sol0[2*n**2:2*n**2+n**4] = sol_int_up.reshape(n**4)
    sol0[2*n**2+n**4:2*n**2+2*n**4] = sol_int_down.reshape(n**4)
    sol0[2*n**2+2*n**4:] = sol_int_updown.reshape(n**4)

    return sol0

def int_ode_fwd(l,y0,n,eta=[],method='jit',norm=False,Hflow=False,comp=False):
        """ Generate the flow equation for the interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the input array eta will be used to specify the generator at this flow time step. The latter option will result 
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

        # Extract various components of the Hamiltonian from the input array 'y'
        H = y[0:n**2]                   # Define quadratic part of Hamiltonian
        H = H.reshape(n,n)              # Reshape into matrix
        H0 = np.diag(np.diag(H))        # Define diagonal quadratic part H0
        V0 = H - H0                     # Define off-diagonal quadratic part B

        Hint = y[n**2:]                 # Define quartic part of Hamiltonian
        Hint = Hint.reshape(n,n,n,n)    # Reshape into rank-4 tensor
        Hint0 = np.zeros((n,n,n,n))     # Define diagonal quartic part 
        for i in range(n):              # Load Hint0 with values
            for j in range(n):
                if i != j:
                    # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                    Hint0[i,i,j,j] = Hint[i,i,j,j]
                    Hint0[i,j,j,i] = Hint[i,j,j,i]
        Vint = Hint-Hint0

        # Extract components of the density operator from input array 'y'
        n2 = nlist[0:n**2]                  # Define quadratic part of density operator
        n2 = n2.reshape(n,n)            # Reshape into matrix
        if len(nlist)>n**2:                 # If interacting system...
            n4 = nlist[n**2::]              #...then define quartic part of density operator
            n4 = n4.reshape(n,n,n,n)    # Reshape into tensor
        
        if norm == True:
            state=nstate(n,0.5)
            # _,V1 = np.linalg.eigh(H)
            # state = np.zeros(n)
            # sites = np.array([i for i in range(n)])
            # random = np.random.choice(sites,n//2)
            # for i in random:
            #     psi = V1[:,i]
            #     state += np.array([v**2 for v in psi])
            # state = np.round(state,decimals=3)
            # # print(np.dot(state,state))

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
        sol0 = np.zeros(2*(n**2+n**4))
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

    # Initialise array to hold solution at all flow times
    sol = np.zeros((qmax,n**2),dtype=np.float64)
    sol[0] = (H2).reshape(n**2)

    # Define integrator
    r = ode(nonint_ode).set_integrator('dopri5', nsteps=100)
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
        off_diag = mat-np.diag(np.diag(mat))
        J0 = max(np.abs(off_diag.reshape(n**2)))
        k += 1
    print(k,J0)
    sol = sol[0:k-1]
    dl_list = dl_list[0:k-1]

    # Initialise a density operator in the diagonal basis on the central site
    liom = np.zeros((qmax,n**2))
    init_liom = np.zeros((n,n))
    init_liom[n//2,n//2] = 1.0
    liom[0,:n**2] = init_liom.reshape(n**2)
    
    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]
    sol = sol[::-1]

    # Degine integrator for density operator
    n_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
    n_int.set_initial_value(liom[0],dl_list[0])

    # Numerically integrate the flow equations for the density operator 
    # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
    k0=1
    while n_int.successful() and k0 < k-1:
        n_int.set_f_params(n,sol[k0])
        n_int.integrate(dl_list[k0])
        liom[k0] = n_int.y
        k0 += 1
    
    # Take final value for the transformed density operator and reshape to a matrix
    central = (liom[k0-1,:n**2]).reshape(n,n)

    # Build output dictionary
    output = {"H0_diag":sol[0].reshape(n,n),"LIOM":central}
    if store_flow == True:
        output["flow"] = sol[::-1]
        output["dl_list"] = dl_list[::-1]

    return output
    
@jit(nopython=True,parallel=True,fastmath=False)
def proc(mat,cutoff):
    """ Test function to zero all matrix elements below a cutoff. """
    for i in prange(len(mat)):
        if mat[i] < cutoff:
            mat[i] = 0.
    return mat

def flow_static_int(n,hamiltonian,dl_list,qmax,cutoff,method='jit',precision=np.float32,norm=True,Hflow=False,store_flow=False):
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
        precision : dtype
            Specify what precision to store the running Hamiltonian/generator. Can be any of the following
            values: np.float16, np.float32, np.float64. Using half precision enables access to the largest system 
            sizes, but naturally will result in loss of precision of any later steps. It is recommended to first 
            test with single or double precision before using half precision to ensure results are accurate.
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

    # Initialise array to hold solution at all flow times
    flow_list = np.zeros((qmax,n**2+n**4),dtype=precision)
    # print('Memory64 required: MB', sol_int.nbytes/10**6)

    # Store initial interaction value and trace of initial H^2 for later error estimation
    delta = np.max(Hint)
    e1 = np.trace(np.dot(H2,H2))
    
    # Define integrator
    r_int = ode(int_ode).set_integrator('dopri5',nsteps=100,rtol=10**(-6),atol=10**(-12))
    
    # Set initial conditions
    init = np.zeros(n**2+n**4,dtype=np.float32)
    init[:n**2] = ((H2)).reshape(n**2)
    init[n**2:] = (Hint).reshape(n**4)
    r_int.set_initial_value(init,dl_list[0])
    r_int.set_f_params(n,[],method,norm,Hflow)
    flow_list[0] = init
    
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
            flow_list[k] = step.astype(dtype=precision)
        else:
            if k == 1:
                eta = eta_con(init,n)
            else:
                eta = eta_con(step,n)
            flow_list[k] = eta.astype(dtype=precision)
            r_int.set_f_params(n,eta,method,norm,Hflow)
            r_int.integrate(dl_list[k])
            step = r_int.y
        
        decay = cut(step,n,cutoff,index_list)

        # Commented out: code to zero all off-diagonal variables below some cutoff
        # sim = proc(step,cutoff)
        # sol_int[k] = sim

        # np.set_printoptions(suppress=True)
        # print((r_int.y)[:n**2])
        mat = step[:n**2].reshape(n,n)
        off_diag = mat-np.diag(np.diag(mat))
        J0 = max(off_diag.reshape(n**2))
        k += 1 
    print(k,J0)

    # Truncate solution list and flow time list to max timestep reached
    flow_list=flow_list[:k-1]
    dl_list = dl_list[:k-1]
    
    # Define final diagonal quadratic Hamiltonian
    H0_diag = step[:n**2].reshape(n,n)
    # Define final diagonal quartic Hamiltonian
    Hint2 = step[n**2::].reshape(n,n,n,n)   
    # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
    HFint = np.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint2[i,i,j,j]
            HFint[i,j] += -Hint2[i,j,j,i]

    # Compute the difference in the second invariant of the flow at start and end
    # This acts as a measure of the unitarity of the transform
    Hflat = HFint.reshape(n**2)
    print(HFint)
    inv = 2*np.sum([d**2 for d in Hflat])
    e2 = np.trace(np.dot(H0_diag,H0_diag))
    inv2 = np.abs(e1 - e2 + ((2*delta)**2)*(n-1) - inv)/np.abs(e2+((2*delta)**2)*(n-1))

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = np.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))

    # Initialise a density operator in the diagonal basis on the central site
    init_liom = np.zeros(n**2+n**4)
    init_liom2 = np.zeros((n,n))
    init_liom2[n//2,n//2] = 1.0
    init_liom[:n**2] = init_liom2.reshape(n**2)

    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]
    flow_list=flow_list[::-1]

    # Define integrator for density operator
    n_int = ode(liom_ode).set_integrator('dopri5',nsteps=100,rtol=10**(-6),atol=10**(-6))
    n_int.set_initial_value(init_liom,dl_list[0])

    if store_flow == True:
        liom_list = np.zeros((k-1,n**2+n**4))
        liom_list[0] = init_liom 

    # Numerically integrate the flow equations for the density operator 
    # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
    k0=1
    # norm = True
    # print('*** SETTING LIOM NORMAL ORDERING TO TRUE ***')
    if Hflow == True:
        while n_int.successful() and k0 < k-1:
            n_int.set_f_params(n,flow_list[k0],method,False,Hflow,norm)
            n_int.integrate(dl_list[k0])
            liom = n_int.y
            if store_flow == True:
                liom_list[k0] = liom
            k0 += 1
    else:
        while k0 < k-1:
            # Note: the .successful() test is not used here as it causes errors
            # due to SciPy being unable to add interpolation steps and the 
            # generator being essentially zero at the 'start' of this reverse flow
            if n_int.successful() == True:
                n_int.set_f_params(n,flow_list[k0],method,False,Hflow,norm)
                n_int.integrate(dl_list[k0])
            else:
                n_int.set_initial_value(init_liom,dl_list[k0])
            liom = n_int.y
            if store_flow == True:
                liom_list[k0] = liom
            k0 += 1

    # liom_all = np.sum([j**2 for j in liom])
    f2 = np.sum([j**2 for j in liom[0:n**2]])
    f4 = np.sum([j**2 for j in liom[n**2::]])
    print('LIOM',f2,f4)
    print('Hint max',np.max(np.abs(Hint2)))

    output = {"H0_diag":H0_diag, "Hint":Hint2,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":inv2}
    if store_flow == True:
        flow_list2 = np.zeros((k-1,2*n**2+2*n**4))
        flow_list2[::,0:n**2+n**4] = flow_list
        flow_list2[::,n**2+n**4:] = liom_list
        output["flow"] = flow_list2
        output["dl_list"] = dl_list

    # Free up some memory
    del flow_list
    gc.collect()

    return output
    
def flow_static_int_fwd(n,hamiltonian,dl_list,qmax,cutoff,method='jit',precision=np.float32,norm=True,Hflow=False,store_flow=False):
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
        precision : dtype
            Specify what precision to store the running Hamiltonian/generator. Can be any of the following
            values: np.float16, np.float32, np.float64. Using half precision enables access to the largest system 
            sizes, but naturally will result in loss of precision of any later steps. It is recommended to first 
            test with single or double precision before using half precision to ensure results are accurate.
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
        flow_list = np.zeros((qmax,2*(n**2+n**4)),dtype=precision)

    # print('Memory64 required: MB', sol_int.nbytes/10**6)

    # Store initial interaction value and trace of initial H^2 for later error estimation
    delta = np.max(Hint)
    e1 = np.trace(np.dot(H2,H2))
    
    # Define integrator
    r_int = ode(int_ode_fwd).set_integrator('dopri5',nsteps=100,rtol=10**(-6),atol=10**(-6))
    
    # Set initial conditions
    init = np.zeros(2*(n**2+n**4),dtype=np.float32)
    init[:n**2] = ((H2)).reshape(n**2)
    init[n**2:n**2+n**4] = (Hint).reshape(n**4)

    # Initialise a density operator in the diagonal basis on the central site
    init_liom = np.zeros(n**2+n**4)
    init_liom2 = np.zeros((n,n))
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
               flow_list[k] = step.astype(precision)

        # Commented out: code to zero all off-diagonal variables below some cutoff
        # sim = proc(r_int.y,n,cutoff)
        # sol_int[k] = sim

        # np.set_printoptions(suppress=True)
        # print((r_int.y)[:n**2])

        decay = cut(step,n,cutoff,index_list)

        mat = step[:n**2].reshape(n,n)
        off_diag = mat-np.diag(np.diag(mat))
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
    HFint = np.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint2[i,i,j,j]
            HFint[i,j] += -Hint2[i,j,j,i]

    # Compute the difference in the second invariant of the flow at start and end
    # This acts as a measure of the unitarity of the transform
    Hflat = HFint.reshape(n**2)
    inv = 2*np.sum([d**2 for d in Hflat])
    e2 = np.trace(np.dot(H0_diag,H0_diag))
    inv2 = np.abs(e1 - e2 + ((2*delta)**2)*(n-1) - inv)/np.abs(e2+((2*delta)**2)*(n-1))

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = np.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))

    # liom_all = np.sum([j**2 for j in liom])
    f2 = np.sum([j**2 for j in liom[0:n**2]])
    f4 = np.sum([j**2 for j in liom[n**2::]])
    print('LIOM',f2,f4)
    print('Hint max',np.max(np.abs(Hint2)))

    output = {"H0_diag":H0_diag, "Hint":Hint2,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":inv2}
    if store_flow == True:
        output["flow"] = flow_list
        output["dl_list"] = dl_list

        # Free up some memory
        del flow_list
        gc.collect()

    return output

def flow_static_int_spin(n,hamiltonian,dl_list,qmax,cutoff,method='jit',store_flow=False,norm=False):
      
        H0_up,H0_down,Hint_up,Hint_down,Hint_updown = hamiltonian.H2_spinup,hamiltonian.H2_spindown,hamiltonian.H4_spinup,hamiltonian.H4_spindown,hamiltonian.H4_mixed

        sol_int = np.zeros((qmax,2*n**2+3*n**4),dtype=np.float64)
        # print('Memory64 required: MB', sol_int.nbytes/10**6)
        # sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
        # print('Memory32 required: MB', sol_int.nbytes/10**6)
        
        r_int = ode(int_ode_spin).set_integrator('dopri5', nsteps=150,atol=10**(-6),rtol=10**(-3))
        r_int.set_f_params(n,method,norm)

        init = np.zeros(2*n**2+3*n**4,dtype=np.float64)
        init[:n**2] = (H0_up).reshape(n**2)
        init[n**2:2*n**2] = (H0_down).reshape(n**2)
        init[2*n**2:2*n**2+n**4] = (Hint_up).reshape(n**4)
        init[2*n**2+n**4:2*n**2+2*n**4] = (Hint_down).reshape(n**4)
        init[2*n**2+2*n**4:] = (Hint_updown).reshape(n**4)
        
        r_int.set_initial_value(init,dl_list[0])
        sol_int[0] = init
   
        k = 1
        J0 = 10.
        while r_int.successful() and k < qmax-1 and J0 > cutoff:
            r_int.integrate(dl_list[k])
            # sim = proc(r_int.y,n,cutoff)
            # sol_int[k] = sim
            sol_int[k] = r_int.y
            mat_up = sol_int[k,0:n**2].reshape(n,n)
            mat_down = sol_int[k,n**2:2*n**2].reshape(n,n)
            off_diag_up = mat_up-np.diag(np.diag(mat_up))
            off_diag_down = mat_down-np.diag(np.diag(mat_down))
            J0_up = max(np.abs(off_diag_up).reshape(n**2))
            J0_down = max(np.abs(off_diag_down).reshape(n**2))
            J0=max(J0_up,J0_down)
            # print(mat_up)

            k += 1
        print(k,J0)   
        sol_int=sol_int[:k-1]
        dl_list = dl_list[:k-1]

        print('eigenvalues',np.sort(np.diag(sol_int[-1,0:n**2].reshape(n,n))))
  
        H0_diag_up = sol_int[-1,:n**2].reshape(n,n)
        H0_diag_down = sol_int[-1,n**2:2*n**2].reshape(n,n)
        
        Hint_up = sol_int[-1,2*n**2:2*n**2+n**4].reshape(n,n,n,n)
        Hint_down = sol_int[-1,2*n**2+n**4:2*n**2+2*n**4].reshape(n,n,n,n) 
        Hint_updown = sol_int[-1,2*n**2+2*n**4:].reshape(n,n,n,n)  
              
        HFint_up = np.zeros(n**2).reshape(n,n)
        HFint_down = np.zeros(n**2).reshape(n,n)
        HFint_updown = np.zeros(n**2).reshape(n,n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    HFint_up[i,j] = Hint_up[i,i,j,j]
                    HFint_up[i,j] += -Hint_up[i,j,j,i]

                    HFint_down[i,j] = Hint_down[i,i,j,j]
                    HFint_down[i,j] += -Hint_down[i,j,j,i]
                    
                HFint_updown[i,j] = Hint_updown[i,i,j,j]

        charge = HFint_up+HFint_down+HFint_updown
        spin = HFint_up+HFint_down-HFint_updown

        lbits_up = np.zeros(n-1)
        lbits_down = np.zeros(n-1)
        lbits_updown = np.zeros(n-1)
        lbits_charge = np.zeros(n-1)
        lbits_spin = np.zeros(n-1)

        for q in range(1,n):

            lbits_up[q-1] = np.median(np.log10(np.abs(np.diag(HFint_up,q)+np.diag(HFint_up,-q))/2.))
            lbits_down[q-1] = np.median(np.log10(np.abs(np.diag(HFint_down,q)+np.diag(HFint_down,-q))/2.))
            lbits_updown[q-1] = np.median(np.log10(np.abs(np.diag(HFint_updown,q)+np.diag(HFint_updown,-q))/2.))

            lbits_charge[q-1] = np.median(np.log10(np.abs(np.diag(charge,q)+np.diag(charge,-q))/2.))
            lbits_spin[q-1] = np.median(np.log10(np.abs(np.diag(spin,q)+np.diag(spin,-q))/2.))

        # plt.plot(lbits_charge)
        # plt.plot(lbits_spin,'--')
        # plt.show()
        # plt.close()

        r_int.set_initial_value(init,dl_list[0])
        init = np.zeros(2*n**2+3*n**4,dtype=np.float64)
        temp = np.zeros((n,n))
        temp[n//2,n//2] = 1.0
        init[:n**2] = temp.reshape(n**2)
        init[n**2:2*n**2] = temp.reshape(n**2)


        dl_list = dl_list[::-1]

        r_int = ode(liom_spin).set_integrator('dopri5', nsteps=150,atol=10**(-6),rtol=10**(-3))
        r_int.set_initial_value(init,dl_list[0])

        k0 = 1
        while r_int.successful() and k0 < k-1:
            r_int.set_f_params(sol_int[-k0],n,method)
            r_int.integrate(dl_list[k0])
            # sim = proc(r_int.y,n,cutoff)
            # sol_int[k] = sim
            liom = r_int.y
 
            k0 += 1

        output = {"H0_diag":[H0_diag_up,H0_diag_down],"Hint":[Hint_up,Hint_down,Hint_updown],
                    "LIOM":liom,"LIOM Interactions":[lbits_up,lbits_down,lbits_updown,lbits_charge,lbits_spin],"Invariant":0}
        if store_flow == True:
            output["flow"] = sol_int
            output["dl_list"] = dl_list

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
    sol = np.zeros((qmax,n**2),dtype=np.float64)
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
        off_diag = mat-np.diag(np.diag(mat))
        J0 = max(np.abs(off_diag.reshape(n**2)))
        k += 1
    print(k,J0)
    sol=sol[0:k-1]
    dl_list= dl_list[0:k-1]

    # Initialise a density operator in the diagonal basis on the central site
    # liom = np.zeros((qmax,n**2))
    init_liom = np.zeros((n,n))
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
    num=np.zeros((k,n**2))
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
    num_t_list = np.zeros((len(tlist),n**2))
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
    nlist = np.zeros(len(tlist))

    # Set up initial state as a CDW
    list1 = np.array([1. for i in range(n//2)])
    list2 = np.array([0. for i in range(n//2)])
    state = np.array([val for pair in zip(list1,list2) for val in pair])
    
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
    sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
    # print('Memory64 required: MB', sol_int.nbytes/10**6)
    
    # Initialise the first flow timestep
    init = np.zeros(n**2+n**4,dtype=np.float32)
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
        off_diag = mat-np.diag(np.diag(mat))
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
    HFint = np.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint[i,i,j,j]
            HFint[i,j] += -Hint[i,j,j,i]

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = np.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))

    # Initialise a density operator in the diagonal basis on the central site
    liom = np.zeros((k,n**2+n**4),dtype=np.float32)
    init_liom = np.zeros((n,n))
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
    num = np.zeros((k,n**2+n**4),dtype=np.float32)
    num_init = np.zeros((n,n))
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
    num_t_list2 = np.zeros((len(tlist),n**2+n**4),dtype=np.complex128)
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
    nlist2 = np.zeros(len(tlist))

    # Set up initial state as a CDW
    list1 = np.array([1. for i in range(n//2)])
    list2 = np.array([0. for i in range(n//2)])
    state = np.array([val for pair in zip(list1,list2) for val in pair])
    
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
    sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
    # print('Memory64 required: MB', sol_int.nbytes/10**6)
    
    # Initialise the first flow timestep
    init = np.zeros(n**2+n**4,dtype=np.float32)
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
        off_diag = mat-np.diag(np.diag(mat))
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
    HFint = np.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint[i,i,j,j]
            HFint[i,j] += -Hint[i,j,j,i]

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = np.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))

    # Initialise a density operator in the diagonal basis on the central site
    liom = np.zeros((k,n**2+n**4),dtype=np.float32)
    init_liom = np.zeros((n,n))
    init_liom[n//2,n//2] = 1.0
    liom[0,:n**2] = init_liom.reshape(n**2)
    
    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]
    
    # Set up initial state as a CDW
    list1 = np.array([1. for i in range(n//2)])
    list2 = np.array([0. for i in range(n//2)])
    state = np.array([val for pair in zip(list1,list2) for val in pair])
    
    # Define lists to store the time-evolved density operators on each lattice site
    # 'imblist' will include interaction effects
    # 'imblist2' includes only single-particle effects
    # Both are kept to check for diverging interaction terms
    imblist = np.zeros((n,len(tlist)))
    imblist2 = np.zeros((n,len(tlist)))

    # Compute the time-evolution of the number operator on every site
    for site in range(n):
        # Initialise operator to be time-evolved
        num = np.zeros((k,n**2+n**4))
        num_init = np.zeros((n,n),dtype=np.float32)
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
        
        num_t_list2 = np.zeros((len(tlist),n**2+n**4))
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
        nlist = np.zeros(len(tlist))
        nlist2 = np.zeros(len(tlist))
        
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
    imblist = 2*np.sum(imblist,axis=0)
    imblist2 = 2*np.sum(imblist2,axis=0)

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
    while np.max(np.abs(V0))>10**(-2):
    
        # Non-interacting generator
        eta0 = np.einsum('ij,jk->ik',H0,V0) - np.einsum('ki,ij->kj',V0,H0,optimize=True)
        
        # Flow of non-interacting terms
        dH0 = np.einsum('ij,jk->ik',eta0,(H0+V0)) - np.einsum('ki,ij->kj',(H0+V0),eta0,optimize=True)
    
        # Update non-interacting terms
        H0 = H0+dl*np.diag(np.diag(dH0))
        V0 = V0 + dl*(dH0-np.diag(np.diag(dH0)))

        q += 1

    print('***********')
    print('FE time - einsum',datetime.now()-startTime)
    print('Max off diagonal element: ', np.max(np.abs(V0)))
    print(np.sort(np.diag(H0)))

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
    while np.max(np.abs(V0))>10**(-3):
    
        # Non-interacting generator
        eta = np.tensordot(H0,V0,axes=1) - np.tensordot(V0,H0,axes=1)
        
        # Flow of non-interacting terms
        dH0 = np.tensordot(eta,H0+V0,axes=1) - np.tensordot(H0+V0,eta,axes=1)
    
        # Update non-interacting terms
        H0 = H0+dl*np.diag(np.diag(dH0))
        V0 = V0 + dl*(dH0-np.diag(np.diag(dH0)))

        q += 1
        
    print('***********')
    print('FE time - Tensordot',datetime.now()-startTime)
    print('Max off diagonal element: ', np.max(np.abs(V0)))
    print(np.sort(np.diag(H0)))

#------------------------------------------------------------------------------
        
def flow_levels(n,array,intr):
    """ Function to compute the many-body eigenvalues from the Hamiltonian returned by the TFE method. """
    H0 = array["H0_diag"]
    if intr == True:
        Hint = array["Hint"]
    flevels = np.zeros(2**n)

    for i in range(2**n):
        lev0 = bin(i)[2::].rjust(n,'0') # Generate the many-body states
        # Compute the energies of each state from the fixed point Hamiltonian
        # if lev0.count('1')==n//2:
        for j in range(n):
            flevels[i] += H0[j,j]*int(lev0[j])
            if intr == True:
                for q in range(n):
                    if q !=j:
                        # flevels[i] += Hint[j,j,q,q]*int(lev0[j])
                        flevels[i] += Hint[j,j,q,q]*int(lev0[j])*int(lev0[q]) 
                        flevels[i] += -Hint[j,q,q,j]*int(lev0[j])*int(lev0[q]) 

    # flevels=flevels[flevels != 0]
    return np.sort(flevels)

def flow_levels_spin(n,flow,intr=True):

    H0 = flow["H0_diag"]
    Hint = flow["Hint"]
    H0_up = H0[0]
    H0_down = H0[1]
    Hint_up_full = Hint[0]
    Hint_down_full = Hint[1]
    Hint_updown_full = Hint[2]

    Hint_up = np.zeros((n,n))
    Hint_down = np.zeros((n,n))
    Hint_updown = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Hint_up[i,j] += Hint_up_full[i,i,j,j]
            Hint_up[i,j] += -Hint_up_full[i,j,j,i]
            Hint_down[i,j] += Hint_down_full[i,i,j,j]
            Hint_down[i,j] += -Hint_down_full[i,j,j,i]
            Hint_updown[i,j] += Hint_updown_full[i,i,j,j]
            Hint_updown[i,j] += -Hint_updown_full[i,j,j,i]
    
    flevels = np.zeros(4**n)
    
    count = 0
    for i in range(2**n):
        lev0 = bin(i)[2::].rjust(n,'0') # Generate the many-body states
        for i1 in range(2**n):
            lev1 = bin(i1)[2::].rjust(n,'0') # Generate the many-body states
            
            # Compute the energies of each state from the fixed point Hamiltonian
            for j in range(n):
                flevels[count] += H0_up[j,j]*int(lev0[j])+H0_down[j,j]*int(lev1[j])
                
                if intr == True:
                    for q in range(n):
                        if q !=j:
                            # flevels[i] += Hint[j,j,q,q]*int(lev0[j])
                            flevels[count] += Hint_up[j,q]*int(lev0[j])*int(lev0[q]) 
                            # flevels[i] += -Hint_up[j,q,q,j]*int(lev0[j])*int(lev0[q]) 
                            
                            flevels[count] += Hint_down[j,q]*int(lev1[j])*int(lev1[q]) 
                            # flevels[i] += -Hint_down[j,q,q,j]*int(lev1[j])*int(lev1[q]) 
                            
                        flevels[count] += Hint_updown[j,q]*int(lev0[j])*int(lev1[q]) 
            count += 1

    
    return np.sort(flevels)

#------------------------------------------------------------------------------
# Compute averaged level spacing ratio
def level_stat(levels):
    """ Function to compute the level spacing statistics."""
    list1 = np.zeros(len(levels))
    lsr = 0.
    for i in range(1,len(levels)):
        list1[i-1] = levels[i] - levels[i-1]
    for j in range(len(levels)-2):
        lsr += min(list1[j],list1[j+1])/max(list1[j],list1[j+1])
    lsr *= 1/(len(levels)-2)
    
    return lsr
