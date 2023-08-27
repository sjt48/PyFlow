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
import jax.numpy as np
from .diag_routines.spinful_fermion import *
from .diag_routines.spinless_fermion import *

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
    # precision = params["precision"]
    norm = params["norm"]
    Hflow = params["Hflow"]
    LIOM = params["LIOM"]
    store_flow = params["store_flow"]
    no_state = params["NO_state"]
    ITC = params["ITC"]
    ladder = params["ladder"]
    order = params["order"]
    dim = params["dim"]

    if logflow == False:
            dl = np.linspace(0,lmax,qmax,endpoint=True)
    elif logflow == True:
        print('Warning: careful choices of qmax and lmax required for log flow.')
        dl = np.logspace(np.log10(0.001), np.log10(lmax),qmax,endpoint=True,base=10)
    if hamiltonian.species == 'spinless fermion':
        if ITC == True:
            flow = flow_int_ITC(n,hamiltonian,dl,qmax,cutoff,tlist,method=method,norm=norm,Hflow=Hflow,store_flow=store_flow)
        elif ladder == True:
            flow = flow_int_fl(n,hamiltonian,dl,qmax,cutoff,tlist,method=method,norm=norm,Hflow=Hflow,store_flow=store_flow,order=order,dim=dim)
        elif dyn == True:
            if intr == True:
                if imbalance == True:
                    flow = flow_dyn_int_imb(n,hamiltonian,num,num_int,dl,qmax,cutoff,tlist,method=method,store_flow=store_flow)
                else:
                    flow = flow_dyn_int_singlesite(n,hamiltonian,num,num_int,dl,qmax,cutoff,tlist,method=method,store_flow=store_flow)
            elif intr == False:
                flow = flow_dyn(n,hamiltonian,num,dl,qmax,cutoff,tlist,method=method,store_flow=store_flow)
        elif dyn == False:
            if intr == True:
                if LIOM == 'bck':
                    flow = flow_static_int(n,hamiltonian,dl,qmax,cutoff,method=method,norm=norm,Hflow=Hflow,store_flow=store_flow)
                elif LIOM == 'fwd':
                    flow = flow_static_int_fwd(n,hamiltonian,dl,qmax,cutoff,method=method,norm=norm,Hflow=Hflow,store_flow=store_flow)
            elif intr == False:
                flow = flow_static(n,hamiltonian,dl,qmax,cutoff,method=method,store_flow=store_flow)
        return flow
    elif hamiltonian.species == 'spinful fermion':
        flow = flow_static_int_spin(n,hamiltonian,dl,qmax,cutoff,method=method,store_flow=store_flow,norm=norm,no_state=no_state)
        return flow
    else:
        print('ERROR: Unknown type of particle.')


# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
# def cut(y,n,cutoff,indices):
#     """ Checks if ALL quadratic off-diagonal parts have decayed below cutoff*10e-3 and TYPICAL (median) off-diag quartic term have decayed below cutoff. """
#     mat2 = y[:n**2].reshape(n,n)
#     mat2_od = mat2-np.diag(np.diag(mat2))

#     if np.max(np.abs(mat2_od)) < cutoff*10**(-3):
#         mat4 = y[n**2:n**2+n**4]
#         mat4_od = np.zeros(n**4)
#         for i in indices:               
#             mat4_od[i] = mat4[i]
#         mat4_od = mat4_od[mat4_od != 0]
#         if np.median(np.abs(mat4_od)) < cutoff:
#             return 0 
#         else:
#             return 1
#     else:
#         return 1

# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
# def cut_spin(y,n,cutoff,indices):
#     """ Checks if ALL quadratic off-diagonal parts have decayed below cutoff*10e-3 and TYPICAL (median) off-diag quartic term have decayed below cutoff. """
#     mat2 = y[:n**2].reshape(n,n)
#     mat3 = y[n**2:2*n**2].reshape(n,n)
#     mat2_od = mat2-np.diag(np.diag(mat2))
#     mat3_od = mat2-np.diag(np.diag(mat3))
#     mat_od = np.zeros(2*n**2)
#     mat_od[:n**2] = mat2_od.reshape(n**2)
#     mat_od[n**2:] = mat3_od.reshape(n**2)

#     if np.max(np.abs(mat_od)) < cutoff*10**(-3):
#         mat4 = y[2*n**2:2*n**2+n**4]
#         mat5 = y[2*n**2+n**4:2*n**2+2*n**4]
#         mat6 = y[2*n**2+2*n**2:2*n**2+3*n**4]
#         mat4_od = np.zeros(3*n**4)     
#         for i in indices:         
#             mat4_od[i] = mat4[i]
#             mat4_od[i+n**4] = mat5[i]
#             mat4_od[i+2*n**4] = mat6[i]
#         mat4_od = mat4_od[mat4_od != 0]
#         if np.median(np.abs(mat4_od)) < cutoff:
#             return 0 
#         else:
#             return 1
#     else:
#         return 1

# def nonint_ode(l,y,n,method='einsum'):
#     """ Generate the flow equation for non-interacting systems.

#         e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

#         Parameters
#         ----------
#         l : float
#             The (fictitious) flow time l which parameterises the unitary transform.
#         y : array
#             Array of size n**2 containing all coefficients of the running Hamiltonian at flow time l.
#         n : integer
#             Linear system size.
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.

#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.

#         Returns
#         -------
#         sol : RHS of the flow equation

#     """
#     H = y.reshape(n,n)
#     H0 = np.diag(np.diag(H))
#     V0 = H - H0
#     eta = contract(H0,V0,method=method,eta=True)
#     sol = contract(eta,H,method=method,eta=False)
#     sol = sol.reshape(n**2)

#     return sol

# #------------------------------------------------------------------------------
# # Build the generator eta at each flow time step
# def eta_con(y,n,method='jit',norm=False):
#     """ Generates the generator at each flow time step. 
    
#         Parameters
#         ----------

#         y : array
#             Running Hamiltonian used to build generator.
#         n : integer
#             Linear system size
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.
#         norm : bool, optional
#             Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
#             This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
#             ensure that use of normal-ordering is warranted and that the contractions are computed with 
#             respect to an appropriate state.

#     """

#     # Extract quadratic parts of Hamiltonian from array y
#     H = y[0:n**2]
#     H = H.reshape(n,n)
#     H0 = np.diag(np.diag(H))
#     V0 = H - H0

#     # Extract quartic parts of Hamiltonian from array y
#     Hint = y[n**2::]
#     Hint = Hint.reshape(n,n,n,n)
#     Hint0 = np.zeros((n,n,n,n))
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
#                 Hint0[i,i,j,j] = Hint[i,i,j,j]
#                 Hint0[i,j,j,i] = Hint[i,j,j,i]
#     Vint = Hint-Hint0

#     # Compute quadratic part of generator
#     eta2 = contract(H0,V0,method=method,eta=True)

#     # Compute quartic part of generator
#     eta4 = contract(Hint0,V0,method=method,eta=True) + contract(H0,Vint,method=method,eta=True)

#     # Add normal-ordering corrections into generator eta, if norm == True
#     if norm == True:
#         state=nstate(n,0.5)

#         eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
#         eta2 += eta_no2
#         eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
#         eta4 += eta_no4

#     # Combine into array
#     eta = np.zeros(n**2+n**4)
#     eta[:n**2] = eta2.reshape(n**2)
#     eta[n**2:] = eta4.reshape(n**4)

#     return eta

# #------------------------------------------------------------------------------

# # @jit('UniTuple(float64[:,:,:,:],2)(float64[:,:,:,:],boolean)',nopython=True,parallel=True,fastmath=True,cache=True)
# def extract_diag(A,norm=False):
#     B = np.zeros(A.shape,dtype=np.float64)
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

# def int_ode(l,y,n,eta=[],method='jit',norm=False,Hflow=True):
#         """ Generate the flow equation for the interacting systems.

#         e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

#         Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
#         the input array eta will be used to specify the generator at this flow time step. The latter option will result 
#         in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
#         integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
#         steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
#         interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
#         these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
#         the benefits from the speed increase likely outweigh the decrease in accuracy.

#         Parameters
#         ----------
#         l : float
#             The (fictitious) flow time l which parameterises the unitary transform.
#         y : array
#             Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
#         n : integer
#             Linear system size.
#         eta : array, optional
#             Provide a pre-computed generator, if desired.
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.
#         norm : bool, optional
#             Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
#             This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
#             ensure that use of normal-ordering is warranted and that the contractions are computed with 
#             respect to an appropriate state.
#         Hflow : bool, optional
#             Choose whether to use pre-computed generator or re-compute eta on the fly.

#         Returns
#         -------
#         sol0 : RHS of the flow equation for interacting system.

#         """
 
#         # Extract various components of the Hamiltonian from the input array 'y'
#         H2 = y[0:n**2]                   # Define quadratic part of Hamiltonian
#         H2 = H2.reshape(n,n)              # Reshape into matrix
#         H2_0 = np.diag(np.diag(H2))        # Define diagonal quadratic part H0
#         V0 = H2 - H2_0                     # Define off-diagonal quadratic part B

#         Hint = y[n**2:]                 # Define quartic part of Hamiltonian
#         Hint = Hint.reshape(n,n,n,n)    # Reshape into rank-4 tensor
#         Hint0 = np.zeros((n,n,n,n))     # Define diagonal quartic part 
#         for i in range(n):              # Load Hint0 with values
#             for j in range(n):
#                     # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
#                     Hint0[i,i,j,j] = Hint[i,i,j,j]
#         Vint = Hint-Hint0

#         if norm == True:
#             state = state_spinless(H2)

#         if Hflow == True:
#             # Compute the generator eta
#             eta0 = contract(H2_0,V0,method=method,eta=True)
#             eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H2_0,Vint,method=method,eta=True)

#             # Add normal-ordering corrections into generator eta, if norm == True
#             if norm == True:

#                 eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H2_0,Vint,method=method,eta=True,state=state)
#                 eta0 += eta_no2

#                 eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
#                 eta_int += eta_no4
#         else:
#             eta0 = (eta[:n**2]).reshape(n,n)
#             eta_int = (eta[n**2:]).reshape(n,n,n,n)
   
#         # Compute the RHS of the flow equation dH/dl = [\eta,H]
#         sol = contract(eta0,H0+V0,method=method)
#         sol2 = contract(eta_int,H0+V0,method=method) + contract(eta0,Hint,method=method)

#         # Add normal-ordering corrections into flow equation, if norm == True
#         if norm == True:
#             sol_no = contractNO(eta_int,H0+V0,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
#             sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
#             sol+=sol_no
#             sol2 += sol4_no
        
#         # Define and load output list sol0
#         sol0 = np.zeros(n**2+n**4)
#         sol0[:n**2] = sol.reshape(n**2)
#         sol0[n**2:] = sol2.reshape(n**4)

#         return sol0

# def int_ode_spin(l,y,n,method='jit',norm=True,no_state='CDW'):
#         """ Generate the flow equation for an interacting system of SPINFUL fermions.

#         e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

#         Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
#         the input array eta will be used to specify the generator at this flow time step. The latter option will result 
#         in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
#         integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
#         steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
#         interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
#         these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
#         the benefits from the speed increase likely outweigh the decrease in accuracy.

#         Parameters
#         ----------
#         l : float
#             The (fictitious) flow time l which parameterises the unitary transform.
#         y : array
#             Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
#         n : integer
#             Linear system size.
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.


#         Returns
#         -------
#         sol0 : RHS of the flow equation for interacting system.

#         """

#         ham = unpack_spin_hamiltonian(y,n)
#         eta0up,eta0down,eta_int_up,eta_int_down,eta_int_updown,upstate,downstate = eta_spin(ham,method=method,norm=norm,no_state=no_state)

#         H2up = ham["H2up"]
#         H2dn = ham["H2dn"]
#         H4up = ham["H4up"]
#         H4dn = ham["H4dn"]
#         H4updn = ham["H4updn"]

#         # Then compute the RHS of the flow equations
#         sol_up = contract(eta0up,H2up,method=method)
#         sol_down = contract(eta0down,H2dn,method=method)
#         sol_int_up = contract(eta_int_up,H2up,method=method) + contract(eta0up,H4up,method=method)
#         sol_int_down = contract(eta_int_down,H2dn,method=method) + contract(eta0down,H4dn,method=method)
#         sol_int_updown = contract(eta_int_updown,H2dn,method=method,pair='second') + contract(eta0down,H4updn,method=method,pair='second')
#         sol_int_updown += contract(eta_int_updown,H2up,method=method,pair='first') + contract(eta0up,H4updn,method=method,pair='first')

#         if norm == True:
#             sol_up += contractNO(eta_int_up,H2up,method=method,state=upstate)
#             sol_up += contractNO(eta0up,H4up,method=method,state=upstate)
#             sol_down += contractNO(eta_int_down,H2dn,method=method,state=downstate)
#             sol_down += contractNO(eta0down,H4dn,method=method,state=downstate)
#             sol_up += contractNO(eta_int_updown,H2dn,method=method,state=downstate,pair='second')
#             sol_up += contractNO(eta0down,H4updn,method=method,state=downstate,pair='second')
#             sol_down += contractNO(eta_int_updown,H2up,method=method,state=upstate,pair='first')
#             sol_down += contractNO(eta0up,H4updn,method=method,state=upstate,pair='first')

#             sol_int_up += contractNO(eta_int_up,H4up,method=method,state=upstate)
#             sol_int_down += contractNO(eta_int_down,H4dn,method=method,state=downstate)

#             sol_int_updown += contractNO(eta_int_up,H4updn,method=method,pair='up-mixed',state=upstate)
#             sol_int_up += contractNO(eta_int_updown,H4updn,method=method,pair='mixed-mixed-up',state=downstate)
#             sol_int_updown += contractNO(eta_int_down,H4updn,method=method,pair='down-mixed',state=downstate)
#             sol_int_down += contractNO(eta_int_updown,H4updn,method=method,pair='mixed-mixed-down',state=upstate)
#             sol_int_updown += contractNO(eta_int_updown,H4up,method=method,pair='mixed-up',state=upstate)
#             sol_int_updown += contractNO(eta_int_updown,H4dn,method=method,pair='mixed-down',state=downstate)

#             sol_int_updown += contractNO(eta_int_updown,H4updn,method=method,pair='mixed',upstate=upstate,downstate=downstate)

#         # Assemble output array
#         sol0 = np.zeros(2*n**2+3*n**4)
#         sol0[:n**2] = sol_up.reshape(n**2)
#         sol0[n**2:2*n**2] = sol_down.reshape(n**2)
#         sol0[2*n**2:2*n**2+n**4] = sol_int_up.reshape(n**4)
#         sol0[2*n**2+n**4:2*n**2+2*n**4] = sol_int_down.reshape(n**4)
#         sol0[2*n**2+2*n**4:] = sol_int_updown.reshape(n**4)

#         return sol0

# def liom_ode(l,y,n,array,method='jit',comp=False,Hflow=True,norm=False):
#     """ Generate the flow equation for density operators of the interacting systems.

#         e.g. compute the RHS of dn/dl = [\eta,n] which will be used later to integrate n(l) -> n(l + dl)

#         Note that this can be used to integrate density operators either 'forward' (from l=0 to l -> infinity) or
#         also 'backward' (from l -> infinity to l=0), as the flow equations are the same either way. The only changes
#         are the initial condition and the sign of the timestep dl.

#         Parameters
#         ----------
#         l : float
#             The (fictitious) flow time l which parameterises the unitary transform.
#         y : array
#             Array of size n**2 + n**4 containing all coefficients of the running density operator at flow time l.
#         H : array
#             Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
#         n : integer
#             Linear system size.
#         eta : array, optional
#             Provide a pre-computed generator, if desired.
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.
#         norm : bool, optional
#             Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
#             This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
#             ensure that use of normal-ordering is warranted and that the contractions are computed with 
#             respect to an appropriate state.
#         comp : bool, optional
#             Specify whether the density operator is complex, e.g. for use in time evolution.
#             Triggers the 'contract' function with 'jit' method to use efficient complex conjugation routine.
#         Hflow : bool, optional
#             Choose whether to use pre-computed generator or re-compute eta on the fly.

#         Returns
#         -------
#         sol0 : RHS of the flow equation for the density operator of the interacting system.


#     """

#     if Hflow == True:
#         # Extract various components of the Hamiltonian from the input array 'y'
#         H2 = array[0:n**2]                  # Define quadratic part of Hamiltonian
#         H2 = H2.reshape(n,n)                # Reshape into matrix
#         H0 = np.diag(np.diag(H2))            # Define diagonal quadratic part H0
#         V0 = H2 - H0                        # Define off-diagonal quadratic part B

#         if len(array)>n**2:
#             Hint = array[n**2::]            # Define quartic part of Hamiltonian
#             Hint = Hint.reshape(n,n,n,n)    # Reshape into rank-4 tensor
#             Hint0 = np.zeros((n,n,n,n))     # Define diagonal quartic part 
#             for i in range(n):              # Load Hint0 with values
#                 for j in range(n):
#                     if i != j:
#                         # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
#                         Hint0[i,i,j,j] = Hint[i,i,j,j]
#                         Hint0[i,j,j,i] = Hint[i,j,j,i]
#             Vint = Hint-Hint0

#         # Compute the quadratic generator eta2
#         eta2 = contract(H0,V0,method=method,comp=False,eta=True)

#         if len(array) > n**2:
#             eta4 = contract(Hint0,V0,method=method,comp=comp,eta=True) + contract(H0,Vint,method=method,comp=comp,eta=True)

#         # Add normal-ordering corrections into generator eta, if norm == True
#         if norm == True:
#             state=state_spinless(H2)
#             eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
#             eta2 += eta_no2

#             eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
#             eta4 += eta_no4

#     else:
#         eta2 = (array[0:n**2]).reshape(n,n)
#         eta4 = (array[n**2::]).reshape(n,n,n,n)

#     # Extract components of the density operator from input array 'y'
#     n2 = y[0:n**2]                  # Define quadratic part of density operator
#     n2 = n2.reshape(n,n)            # Reshape into matrix
#     if len(y)>n**2:                 # If interacting system...
#         n4 = y[n**2::]              #...then define quartic part of density operator
#         n4 = n4.reshape(n,n,n,n)    # Reshape into tensor
                    
#     # Compute the quadratic terms in the RHS of the flow equation
#     sol = contract(eta2,n2,method=method,comp=comp)

#     # Define output array as either real or complex, as required
#     if comp == False:
#         sol0 = np.zeros(len(y))
#     elif comp == True:
#         sol0 = np.zeros(len(y),dtype=complex)

#     # Compute quartic terms, if interacting system
#     if len(y) > n**2:
#         sol2 = contract(eta4,n2,method=method,comp=comp) + contract(eta2,n4,method=method,comp=comp)

#     # Add normal-ordering corrections into flow equation, if norm == True
#     if norm == True:
#         sol_no = contractNO(eta4,n2,method=method,eta=False,state=state) + contractNO(eta2,n4,method=method,eta=False,state=state)
#         sol+=sol_no
#         if len(y) > n**2:
#             sol4_no = contractNO(eta4,n4,method=method,eta=False,state=state)
#             sol2 += sol4_no

#     # Load solution into output array
#     sol0[:n**2] = sol.reshape(n**2)
#     if len(y)> n**2:
#         sol0[n**2:] = sol2.reshape(n**4)

#     return sol0

# def liom_spin(l,nlist,y,n,method='jit',comp=False,norm=False):

#     ham = unpack_spin_hamiltonian(y,n)
#     eta0up,eta0down,eta_int_up,eta_int_down,eta_int_updown,upstate,downstate = eta_spin(ham,method=method,norm=norm)
                
#     n2_up = nlist[0:n**2].reshape(n,n)
#     n2_down = nlist[n**2:2*n**2].reshape(n,n)
#     n4_up = nlist[2*n**2:2*n**2+n**4].reshape(n,n,n,n)
#     n4_down = nlist[2*n**2+n**4:2*n**2+2*n**4].reshape(n,n,n,n)
#     n4_updown = nlist[2*n**2+2*n**4:].reshape(n,n,n,n)

#     sol_up = contract(eta0up,n2_up,method=method)
#     sol_down = contract(eta0down,n2_down,method=method)
#     sol_int_up = contract(eta_int_up,n2_up,method=method) + contract(eta0up,n4_up,method=method)
#     sol_int_down = contract(eta_int_down,n2_down,method=method) + contract(eta0down,n4_down,method=method)
#     sol_int_updown = contract(eta_int_updown,n2_down,method=method,pair='second') + contract(eta0down,n4_updown,method=method,pair='second')
#     sol_int_updown += contract(eta_int_updown,n2_up,method=method,pair='first') + contract(eta0up,n4_updown,method=method,pair='first')
    
#     if norm == True:
#         sol_up += contractNO2(eta_int_up,n2_up,method=method,state=upstate)
#         sol_down += contractNO2(eta_int_down,n2_down,method=method,state=downstate)
#         sol_up += contractNO2(eta_int_updown,n2_down,method=method,state=downstate,pair='second')
#         sol_down += contractNO2(eta_int_updown,n2_up,method=method,state=upstate,pair='first')
#         sol_int_up += contractNO(eta_int_up,n4_updown,method=method,pair='up-mixed',state=upstate)
#         sol_int_up += contractNO(eta_int_updown,n4_updown,method=method,pair='mixed-mixed-up',state=downstate)
#         sol_int_down += contractNO(eta_int_down,n4_updown,method=method,pair='down-mixed',state=upstate)
#         sol_int_down += contractNO(eta_int_updown,n4_updown,method=method,pair='mixed-mixed-down',state=upstate)
#         sol_int_updown += contractNO(eta_int_updown,n4_up,method=method,pair='mixed-up',state=upstate)
#         sol_int_updown += contractNO(eta_int_updown,n4_down,method=method,pair='mixed-down',state=downstate)

#     sol0 = np.zeros(2*n**2+3*n**4)
#     sol0[:n**2] = sol_up.reshape(n**2)
#     sol0[n**2:2*n**2] = sol_down.reshape(n**2)
#     sol0[2*n**2:2*n**2+n**4] = sol_int_up.reshape(n**4)
#     sol0[2*n**2+n**4:2*n**2+2*n**4] = sol_int_down.reshape(n**4)
#     sol0[2*n**2+2*n**4:] = sol_int_updown.reshape(n**4)

#     return sol0

# def int_ode_fwd(l,y0,n,eta=[],method='jit',norm=False,Hflow=False,comp=False):
#         """ Generate the flow equation for the interacting systems.

#         e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

#         Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
#         the input array eta will be used to specify the generator at this flow time step. The latter option will result 
#         in a huge speed increase, at the potential cost of accuracy. This is because the SciPi routine used to 
#         integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
#         steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
#         interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
#         these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
#         the benefits from the speed increase likely outweigh the decrease in accuracy.

#         Parameters
#         ----------
#         l : float
#             The (fictitious) flow time l which parameterises the unitary transform.
#         y : array
#             Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
#         n : integer
#             Linear system size.
#         eta : array, optional
#             Provide a pre-computed generator, if desired.
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.
#         norm : bool, optional
#             Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
#             This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
#             ensure that use of normal-ordering is warranted and that the contractions are computed with 
#             respect to an appropriate state.
#         Hflow : bool, optional
#             Choose whether to use pre-computed generator or re-compute eta on the fly.

#         Returns
#         -------
#         sol0 : RHS of the flow equation for interacting system.

#         """
#         y = y0[:n**2+n**4]
#         nlist = y0[n**2+n**4::]

#         # Extract various components of the Hamiltonian from the input array 'y'
#         H = y[0:n**2]                   # Define quadratic part of Hamiltonian
#         H = H.reshape(n,n)              # Reshape into matrix
#         H0 = np.diag(np.diag(H))        # Define diagonal quadratic part H0
#         V0 = H - H0                     # Define off-diagonal quadratic part B

#         Hint = y[n**2:]                 # Define quartic part of Hamiltonian
#         Hint = Hint.reshape(n,n,n,n)    # Reshape into rank-4 tensor
#         Hint0 = np.zeros((n,n,n,n))     # Define diagonal quartic part 
#         for i in range(n):              # Load Hint0 with values
#             for j in range(n):
#                 if i != j:
#                     # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
#                     Hint0[i,i,j,j] = Hint[i,i,j,j]
#                     Hint0[i,j,j,i] = Hint[i,j,j,i]
#         Vint = Hint-Hint0

#         # Extract components of the density operator from input array 'y'
#         n2 = nlist[0:n**2]                  # Define quadratic part of density operator
#         n2 = n2.reshape(n,n)            # Reshape into matrix
#         if len(nlist)>n**2:                 # If interacting system...
#             n4 = nlist[n**2::]              #...then define quartic part of density operator
#             n4 = n4.reshape(n,n,n,n)    # Reshape into tensor
        
#         if norm == True:
#             state=state_spinless(H)

#         if Hflow == True:
#             # Compute the generator eta
#             eta0 = contract(H0,V0,method=method,eta=True)
#             eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H0,Vint,method=method,eta=True)

#             # Add normal-ordering corrections into generator eta, if norm == True
#             if norm == True:

#                 eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
#                 eta0 += eta_no2

#                 eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
#                 eta_int += eta_no4
#         else:
#             eta0 = (eta[:n**2]).reshape(n,n)
#             eta_int = (eta[n**2:]).reshape(n,n,n,n)
   
#         # Compute the RHS of the flow equation dH/dl = [\eta,H]
#         sol = contract(eta0,H0+V0,method=method)
#         sol2 = contract(eta_int,H0+V0,method=method) + contract(eta0,Hint,method=method)

#         nsol = contract(eta0,n2,method=method,comp=comp)
#         if len(y) > n**2:
#             nsol2 = contract(eta_int,n2,method=method,comp=comp) + contract(eta0,n4,method=method,comp=comp)


#         # Add normal-ordering corrections into flow equation, if norm == True
#         if norm == True:
#             sol_no = contractNO(eta_int,H0+V0,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
#             sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
#             sol+= sol_no
#             sol2 += sol4_no
        
#         # Define and load output list sol0
#         sol0 = np.zeros(2*(n**2+n**4))
#         sol0[:n**2] = sol.reshape(n**2)
#         sol0[n**2:n**2+n**4] = sol2.reshape(n**4)
#         sol0[n**2+n**4:2*n**2+n**4] = nsol.reshape(n**2)
#         sol0[2*n**2+n**4:] = nsol2.reshape(n**4)

#         return sol0
# #------------------------------------------------------------------------------  

# def flow_static(n,hamiltonian,dl_list,qmax,cutoff,method='jit',store_flow=True):
#     """
#     Diagonalise an initial non-interacting Hamiltonian and compute the integrals of motion.

#     Note that this function does not use the trick of fixing eta: as non-interacting systems are
#     quadratic in terms of fermion operators, there are no high-order tensor contractions and so 
#     fixing eta is not necessary here as the performance gain is not expected to be significant.

#         Parameters
#         ----------
#         n : integer
#             Linear system size.
#         H0 : array, float
#             Diagonal component of Hamiltonian
#         V0 : array, float
#             Off-diagonal component of Hamiltonian.
#         dl_list : array, float
#             List of flow times to use for the numerical integration.
#         qmax : integer
#             Maximum number of flow time steps.
#         cutoff : float
#             Threshold value below which off-diagonal elements are set to zero.
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.

#         Returns
#         -------
#         output : dict
#             Dictionary containing diagonal Hamiltonian ("H0_diag") and LIOM on central site ("LIOM").
    
#     """
#     H2 = hamiltonian.H2_spinless

#     # Initialise array to hold solution at all flow times
#     sol = np.zeros((qmax,n**2),dtype=np.float64)
#     sol[0] = (H2).reshape(n**2)

#     # Define integrator
#     r = ode(nonint_ode).set_integrator('dopri5', nsteps=100)
#     r.set_initial_value((H2).reshape(n**2),dl_list[0])
#     r.set_f_params(n,method)
    
#     # Numerically integrate the flow equations
#     k = 1                       # Flow timestep index
#     J0 = 10.                    # Seed value for largest off-diagonal term
#     # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
#     while r.successful() and k < qmax-1 and J0 > cutoff:
#         r.integrate(dl_list[k])
#         sol[k] = r.y
#         mat = sol[k].reshape(n,n)
#         off_diag = mat-np.diag(np.diag(mat))
#         J0 = max(np.abs(off_diag.reshape(n**2)))
#         k += 1
#     print(k,J0)
#     sol = sol[0:k-1]
#     dl_list = dl_list[0:k-1]

#     # Initialise a density operator in the diagonal basis on the central site
#     liom = np.zeros((qmax,n**2))
#     init_liom = np.zeros((n,n))
#     init_liom[n//2,n//2] = 1.0
#     liom[0,:n**2] = init_liom.reshape(n**2)
    
#     # Reverse list of flow times in order to conduct backwards integration
#     dl_list = dl_list[::-1]
#     sol = sol[::-1]

#     # Define integrator for density operator
#     n_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
#     n_int.set_initial_value(liom[0],dl_list[0])

#     # Numerically integrate the flow equations for the density operator 
#     # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
#     k0=1
#     while n_int.successful() and k0 < k-1:
#         n_int.set_f_params(n,sol[k0])
#         n_int.integrate(dl_list[k0])
#         liom[k0] = n_int.y
#         k0 += 1
    
#     # Take final value for the transformed density operator and reshape to a matrix
#     central = (liom[k0-1,:n**2]).reshape(n,n)

#     # Build output dictionary
#     output = {"H0_diag":sol[0].reshape(n,n),"LIOM":central}
#     if store_flow == True:
#         output["flow"] = sol[::-1]
#         output["dl_list"] = dl_list[::-1]

#     return output
    
# @jit(nopython=True,parallel=True,fastmath=True)
# def proc(mat,cutoff):
#     """ Test function to zero all matrix elements below a cutoff. """
#     for i in prange(len(mat)):
#         if mat[i] < cutoff:
#             mat[i] = 0.
#     return mat

# def flow_static_int(n,hamiltonian,dl_list,qmax,cutoff,method='jit',precision=np.float64,norm=True,Hflow=False,store_flow=False):
#     """
#     Diagonalise an initial interacting Hamiltonian and compute the integrals of motion.

#     Parameters
#         ----------
#         n : integer
#             Linear system size.
#         H0 : array, float
#             Diagonal component of Hamiltonian
#         V0 : array, float
#             Off-diagonal component of Hamiltonian.
#         Hint : array, float
#             Diagonal component of Hamiltonian
#         Vint : array, float
#             Off-diagonal component of Hamiltonian.
#         dl_list : array, float
#             List of flow times to use for the numerical integration.
#         qmax : integer
#             Maximum number of flow time steps.
#         cutoff : float
#             Threshold value below which off-diagonal elements are set to zero.
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.
#         precision : dtype
#             Specify what precision to store the running Hamiltonian/generator. Can be any of the following
#             values: np.float16, np.float32, np.float64. Using half precision enables access to the largest system 
#             sizes, but naturally will result in loss of precision of any later steps. It is recommended to first 
#             test with single or double precision before using half precision to ensure results are accurate.
#         norm : bool, optional
#             Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
#             This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
#             ensure that use of normal-ordering is warranted and that the contractions are computed with 
#             respect to an appropriate state.
#         Hflow : bool, optional
#             Choose whether to use pre-computed generator or re-compute eta on the fly.

#         Returns
#         -------
#         output : dict
#             Dictionary containing diagonal Hamiltonian ("H0_diag","Hint"), LIOM interaction coefficient ("LIOM Interactions"),
#             the LIOM on central site ("LIOM") and the value of the second invariant of the flow ("Invariant").
    
#     """
#     H2,Hint = hamiltonian.H2_spinless,hamiltonian.H4_spinless

#     # Initialise array to hold solution at all flow times
#     flow_list = np.zeros((qmax,n**2+n**4),dtype=precision)
#     # print('Memory64 required: MB', sol_int.nbytes/10**6)

#     # Store initial interaction value and trace of initial H^2 for later error estimation
#     delta = np.max(Hint)
#     e1 = np.trace(np.dot(H2,H2))
    
#     # Define integrator
#     r_int = ode(int_ode).set_integrator('dopri5',nsteps=100,rtol=10**(-6),atol=10**(-12))
    
#     # Set initial conditions
#     init = np.zeros(n**2+n**4,dtype=np.float64)
#     init[:n**2] = ((H2)).reshape(n**2)
#     init[n**2:] = (Hint).reshape(n**4)
#     r_int.set_initial_value(init,dl_list[0])
#     r_int.set_f_params(n,[],method,norm,Hflow)
#     flow_list[0] = init
    
#     # Numerically integrate the flow equations
#     k = 1                       # Flow timestep index
#     J0 = 10.                    # Seed value for largest off-diagonal term
#     decay = 1
#     index_list = indices(n)
#     # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
#     while r_int.successful() and k < qmax-1 and decay == 1:
#         if Hflow == True:
#             r_int.integrate(dl_list[k])
#             step = r_int.y
#             flow_list[k] = step.astype(dtype=precision)
#         else:
#             if k == 1:
#                 eta = eta_con(init,n)
#             else:
#                 eta = eta_con(step,n)
#             flow_list[k] = eta.astype(dtype=precision)
#             r_int.set_f_params(n,eta,method,norm,Hflow)
#             r_int.integrate(dl_list[k])
#             step = r_int.y
        
#         decay = cut(step,n,cutoff,index_list)

#         # Commented out: code to zero all off-diagonal variables below some cutoff
#         # sim = proc(step,cutoff)
#         # sol_int[k] = sim

#         # np.set_printoptions(suppress=True)
#         # print((r_int.y)[:n**2])
#         mat = step[:n**2].reshape(n,n)
#         off_diag = mat-np.diag(np.diag(mat))
#         J0 = max(off_diag.reshape(n**2))
#         k += 1 
#     print(k,J0)

#     # Truncate solution list and flow time list to max timestep reached
#     flow_list=flow_list[:k-1]
#     dl_list = dl_list[:k-1]
    
#     # Define final diagonal quadratic Hamiltonian
#     H0_diag = step[:n**2].reshape(n,n)
#     # Define final diagonal quartic Hamiltonian
#     Hint2 = step[n**2::].reshape(n,n,n,n)   
#     # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
#     HFint = np.zeros(n**2).reshape(n,n)
#     for i in range(n):
#         for j in range(n):
#             HFint[i,j] = Hint2[i,i,j,j]
#             HFint[i,j] += -Hint2[i,j,j,i]

#     # Compute the difference in the second invariant of the flow at start and end
#     # This acts as a measure of the unitarity of the transform
#     Hflat = HFint.reshape(n**2)
#     # print(H0_diag)
#     # print(HFint)
#     inv = 2*np.sum([d**2 for d in Hflat])
#     e2 = np.trace(np.dot(H0_diag,H0_diag))
#     inv2 = np.abs(e1 - e2 + ((2*delta)**2)*(n-1) - inv)/np.abs(e2+((2*delta)**2)*(n-1))

#     # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
#     lbits = np.zeros(n-1)
#     for q in range(1,n):
#         lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))

#     # Initialise a density operator in the diagonal basis on the central site
#     init_liom = np.zeros(n**2+n**4)
#     init_liom2 = np.zeros((n,n))
#     init_liom2[n//2,n//2] = 1.0
#     init_liom[:n**2] = init_liom2.reshape(n**2)

#     # Reverse list of flow times in order to conduct backwards integration
#     dl_list = dl_list[::-1]
#     flow_list=flow_list[::-1]

#     # Define integrator for density operator
#     n_int = ode(liom_ode).set_integrator('dopri5',nsteps=100,rtol=10**(-6),atol=10**(-6))
#     n_int.set_initial_value(init_liom,dl_list[0])

#     if store_flow == True:
#         liom_list = np.zeros((k-1,n**2+n**4))
#         liom_list[0] = init_liom 

#     # Numerically integrate the flow equations for the density operator 
#     # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
#     k0=1
#     # norm = True
#     # print('*** SETTING LIOM NORMAL ORDERING TO TRUE ***')
#     if Hflow == True:
#         while n_int.successful() and k0 < k-1:
#             n_int.set_f_params(n,flow_list[k0],method,False,Hflow,norm)
#             n_int.integrate(dl_list[k0])
#             liom = n_int.y
#             if store_flow == True:
#                 liom_list[k0] = liom
#             k0 += 1
#     else:
#         while k0 < k-1:
#             # Note: the .successful() test is not used here as it causes errors
#             # due to SciPy being unable to add interpolation steps and the 
#             # generator being essentially zero at the 'start' of this reverse flow
#             if n_int.successful() == True:
#                 n_int.set_f_params(n,flow_list[k0],method,False,Hflow,norm)
#                 n_int.integrate(dl_list[k0])
#             else:
#                 n_int.set_initial_value(init_liom,dl_list[k0])
#             liom = n_int.y
#             if store_flow == True:
#                 liom_list[k0] = liom
#             k0 += 1

#     # liom_all = np.sum([j**2 for j in liom])
#     f2 = np.sum([j**2 for j in liom[0:n**2]])
#     f4 = np.sum([j**2 for j in liom[n**2::]])
#     print('LIOM',f2,f4)
#     print('Hint max',np.max(np.abs(Hint2)))

#     output = {"H0_diag":H0_diag, "Hint":Hint2,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":inv2}
#     if store_flow == True:
#         flow_list2 = np.zeros((k-1,2*n**2+2*n**4))
#         flow_list2[::,0:n**2+n**4] = flow_list
#         flow_list2[::,n**2+n**4:] = liom_list
#         output["flow"] = flow_list2
#         output["dl_list"] = dl_list

#     # Free up some memory
#     del flow_list
#     gc.collect()

#     return output
    
# def flow_static_int_fwd(n,hamiltonian,dl_list,qmax,cutoff,method='jit',precision=np.float32,norm=True,Hflow=False,store_flow=False):
#     """
#     Diagonalise an initial interacting Hamiltonian and compute the integrals of motion.

#     Note: this function does not compute the LIOMs in the conventional way. Rather, it starts with a local 
#     operator in the initial basis and transforms it into the diagonal basis, essentially the inverse of 
#     the process used to produce LIOMs conventionally. This bypasses the requirement to store the full 
#     unitary transform in memory, meaning that only a single tensor of order O(L^4) needs to be stored at 
#     each flow time step, dramatically increasing the accessible system sizes. However, use this with care
#     as it is *not* a conventional LIOM, despite displaying essentially the same features, and should be 
#     understood as such.

#     Parameters
#         ----------
#         n : integer
#             Linear system size.
#         H0 : array, float
#             Diagonal component of Hamiltonian
#         V0 : array, float
#             Off-diagonal component of Hamiltonian.
#         Hint : array, float
#             Diagonal component of Hamiltonian
#         Vint : array, float
#             Off-diagonal component of Hamiltonian.
#         dl_list : array, float
#             List of flow times to use for the numerical integration.
#         qmax : integer
#             Maximum number of flow time steps.
#         cutoff : float
#             Threshold value below which off-diagonal elements are set to zero.
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.
#         precision : dtype
#             Specify what precision to store the running Hamiltonian/generator. Can be any of the following
#             values: np.float16, np.float32, np.float64. Using half precision enables access to the largest system 
#             sizes, but naturally will result in loss of precision of any later steps. It is recommended to first 
#             test with single or double precision before using half precision to ensure results are accurate.
#         norm : bool, optional
#             Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
#             This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
#             ensure that use of normal-ordering is warranted and that the contractions are computed with 
#             respect to an appropriate state.
#         Hflow : bool, optional
#             Choose whether to use pre-computed generator or re-compute eta on the fly.

#         Returns
#         -------
#         output : dict
#             Dictionary containing diagonal Hamiltonian ("H0_diag","Hint"), LIOM interaction coefficient ("LIOM Interactions"),
#             the LIOM on central site ("LIOM") and the value of the second invariant of the flow ("Invariant").
    
#     """

#     H2,Hint = hamiltonian.H2_spinless,hamiltonian.H4_spinless

#     if store_flow == True:
#         # Initialise array to hold solution at all flow times
#         flow_list = np.zeros((qmax,2*(n**2+n**4)),dtype=precision)

#     # print('Memory64 required: MB', sol_int.nbytes/10**6)

#     # Store initial interaction value and trace of initial H^2 for later error estimation
#     delta = np.max(Hint)
#     e1 = np.trace(np.dot(H2,H2))
    
#     # Define integrator
#     r_int = ode(int_ode_fwd).set_integrator('dopri5',nsteps=100,rtol=10**(-6),atol=10**(-6))
    
#     # Set initial conditions
#     init = np.zeros(2*(n**2+n**4),dtype=np.float32)
#     init[:n**2] = ((H2)).reshape(n**2)
#     init[n**2:n**2+n**4] = (Hint).reshape(n**4)

#     # Initialise a density operator in the diagonal basis on the central site
#     init_liom = np.zeros(n**2+n**4)
#     init_liom2 = np.zeros((n,n))
#     init_liom2[n//2,n//2] = 1.0
#     init_liom[:n**2] = init_liom2.reshape(n**2)
#     init[n**2+n**4:] = init_liom
#     if store_flow == True:
#         flow_list[0] = init

#     r_int.set_initial_value(init,dl_list[0])
#     r_int.set_f_params(n,[],method,norm,Hflow)

        
#     # Numerically integrate the flow equations
#     k = 1                       # Flow timestep index
#     J0 = 10.                    # Seed value for largest off-diagonal term
#     decay = 1
#     index_list = indices(n)
#     # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
#     while r_int.successful() and k < qmax-1 and decay == 1:
#         if Hflow == True:
#             r_int.integrate(dl_list[k])
#             step = r_int.y
#             if store_flow == True:
#                flow_list[k] = step.astype(precision)

#         # Commented out: code to zero all off-diagonal variables below some cutoff
#         # sim = proc(r_int.y,n,cutoff)
#         # sol_int[k] = sim

#         # np.set_printoptions(suppress=True)
#         # print((r_int.y)[:n**2])

#         decay = cut(step,n,cutoff,index_list)

#         mat = step[:n**2].reshape(n,n)
#         off_diag = mat-np.diag(np.diag(mat))
#         J0 = max(off_diag.reshape(n**2))
#         k += 1 
#     print(k,J0)

#     # Truncate solution list and flow time list to max timestep reached
#     dl_list = dl_list[:k-1]
#     if store_flow == True:
#         flow_list = flow_list[:k-1]
    
#     liom = step[n**2+n**4::]
#     step = step[:n**2+n**4]

#     # Define final diagonal quadratic Hamiltonian
#     H0_diag = step[:n**2].reshape(n,n)
#     # Define final diagonal quartic Hamiltonian
#     Hint2 = step[n**2::].reshape(n,n,n,n)   
#     # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
#     HFint = np.zeros(n**2).reshape(n,n)
#     for i in range(n):
#         for j in range(n):
#             HFint[i,j] = Hint2[i,i,j,j]
#             HFint[i,j] += -Hint2[i,j,j,i]

#     # Compute the difference in the second invariant of the flow at start and end
#     # This acts as a measure of the unitarity of the transform
#     Hflat = HFint.reshape(n**2)
#     inv = 2*np.sum([d**2 for d in Hflat])
#     e2 = np.trace(np.dot(H0_diag,H0_diag))
#     inv2 = np.abs(e1 - e2 + ((2*delta)**2)*(n-1) - inv)/np.abs(e2+((2*delta)**2)*(n-1))

#     # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
#     lbits = np.zeros(n-1)
#     for q in range(1,n):
#         lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))

#     # liom_all = np.sum([j**2 for j in liom])
#     f2 = np.sum([j**2 for j in liom[0:n**2]])
#     f4 = np.sum([j**2 for j in liom[n**2::]])
#     print('LIOM',f2,f4)
#     print('Hint max',np.max(np.abs(Hint2)))

#     output = {"H0_diag":H0_diag, "Hint":Hint2,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":inv2}
#     if store_flow == True:
#         output["flow"] = flow_list
#         output["dl_list"] = dl_list

#         # Free up some memory
#         del flow_list
#         gc.collect()

#     return output

# def flow_static_int_spin(n,hamiltonian,dl_list,qmax,cutoff,method='jit',store_flow=False,norm=False,no_state='CDW'):
      
#         H0_up,H0_down,Hint_up,Hint_down,Hint_updown = hamiltonian.H2_spinup,hamiltonian.H2_spindown,hamiltonian.H4_spinup,hamiltonian.H4_spindown,hamiltonian.H4_mixed

#         sol_int = np.zeros((qmax,2*n**2+3*n**4),dtype=np.float64)
#         # print('Memory64 required: MB', sol_int.nbytes/10**6)
#         # sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
#         # print('Memory32 required: MB', sol_int.nbytes/10**6)
        
#         r_int = ode(int_ode_spin).set_integrator('dopri5', nsteps=150,atol=10**(-6),rtol=10**(-3))
#         r_int.set_f_params(n,method,norm,no_state)

#         init = np.zeros(2*n**2+3*n**4,dtype=np.float64)
#         init[:n**2] = (H0_up).reshape(n**2)
#         init[n**2:2*n**2] = (H0_down).reshape(n**2)
#         init[2*n**2:2*n**2+n**4] = (Hint_up).reshape(n**4)
#         init[2*n**2+n**4:2*n**2+2*n**4] = (Hint_down).reshape(n**4)
#         init[2*n**2+2*n**4:] = (Hint_updown).reshape(n**4)
        
#         r_int.set_initial_value(init,dl_list[0])
#         sol_int[0] = init
   
#         k = 1
#         J0 = 10.
#         decay = 1
#         index_list = indices(n)
#         while r_int.successful() and k < qmax-1 and J0 > cutoff and decay == 1:
#             r_int.integrate(dl_list[k])
#             # sim = proc(r_int.y,n,cutoff)
#             # sol_int[k] = sim
#             sol_int[k] = r_int.y
#             mat_up = (r_int.y)[0:n**2].reshape(n,n)
#             mat_down = (r_int.y)[n**2:2*n**2].reshape(n,n)
#             off_diag_up = mat_up-np.diag(np.diag(mat_up))
#             off_diag_down = mat_down-np.diag(np.diag(mat_down))
#             J0_up = max(np.abs(off_diag_up).reshape(n**2))
#             J0_down = max(np.abs(off_diag_down).reshape(n**2))
#             J0=max(J0_up,J0_down)

#             decay = cut_spin(r_int.y,n,cutoff,index_list)

#             k += 1
#         print(k,J0)   
#         sol_int=sol_int[:k-1]
#         dl_list = dl_list[:k-1]

#         print('eigenvalues',np.sort(np.diag(sol_int[-1,0:n**2].reshape(n,n))))
  
#         H0_diag_up = sol_int[-1,:n**2].reshape(n,n)
#         H0_diag_down = sol_int[-1,n**2:2*n**2].reshape(n,n)
        
#         Hint_up = sol_int[-1,2*n**2:2*n**2+n**4].reshape(n,n,n,n)
#         Hint_down = sol_int[-1,2*n**2+n**4:2*n**2+2*n**4].reshape(n,n,n,n) 
#         Hint_updown = sol_int[-1,2*n**2+2*n**4:].reshape(n,n,n,n)  
              
#         HFint_up = np.zeros(n**2).reshape(n,n)
#         HFint_down = np.zeros(n**2).reshape(n,n)
#         HFint_updown = np.zeros(n**2).reshape(n,n)
#         for i in range(n):
#             for j in range(n):
#                 if i != j:
#                     HFint_up[i,j] = Hint_up[i,i,j,j]
#                     HFint_up[i,j] += -Hint_up[i,j,j,i]

#                     HFint_down[i,j] = Hint_down[i,i,j,j]
#                     HFint_down[i,j] += -Hint_down[i,j,j,i]
                    
#                 HFint_updown[i,j] = Hint_updown[i,i,j,j]

#         charge = HFint_up+HFint_down+HFint_updown
#         spin = HFint_up+HFint_down-HFint_updown

#         lbits_up = np.zeros(n-1)
#         lbits_down = np.zeros(n-1)
#         lbits_updown = np.zeros(n-1)
#         lbits_charge = np.zeros(n-1)
#         lbits_spin = np.zeros(n-1)

#         for q in range(1,n):

#             lbits_up[q-1] = np.median(np.log10(np.abs(np.diag(HFint_up,q)+np.diag(HFint_up,-q))/2.))
#             lbits_down[q-1] = np.median(np.log10(np.abs(np.diag(HFint_down,q)+np.diag(HFint_down,-q))/2.))
#             lbits_updown[q-1] = np.median(np.log10(np.abs(np.diag(HFint_updown,q)+np.diag(HFint_updown,-q))/2.))

#             lbits_charge[q-1] = np.median(np.log10(np.abs(np.diag(charge,q)+np.diag(charge,-q))/2.))
#             lbits_spin[q-1] = np.median(np.log10(np.abs(np.diag(spin,q)+np.diag(spin,-q))/2.))

#         r_int.set_initial_value(init,dl_list[0])
#         init_up = np.zeros(2*n**2+3*n**4,dtype=np.float64)
#         init_dn = np.zeros(2*n**2+3*n**4,dtype=np.float64)
#         temp = np.zeros((n,n))
#         temp[n//2,n//2] = 1.0
#         init_up[:n**2] = temp.reshape(n**2)
#         init_dn[n**2:2*n**2] = temp.reshape(n**2)

#         dl_list = dl_list[::-1]

#         r_int = ode(liom_spin).set_integrator('dopri5', nsteps=150,atol=10**(-6),rtol=10**(-3))
#         r_int.set_initial_value(init_up,dl_list[0])

#         k0 = 1
#         while r_int.successful() and k0 < k-1:
#             r_int.set_f_params(sol_int[-k0],n,method,no_state)
#             r_int.integrate(dl_list[k0])
#             # sim = proc(r_int.y,n,cutoff)
#             # sol_int[k] = sim
#             liom_up = r_int.y
 
#             k0 += 1

#         r_int = ode(liom_spin).set_integrator('dopri5', nsteps=150,atol=10**(-6),rtol=10**(-3))
#         r_int.set_initial_value(init_dn,dl_list[0])

#         k0 = 1
#         while r_int.successful() and k0 < k-1:
#             r_int.set_f_params(sol_int[-k0],n,method,no_state)
#             r_int.integrate(dl_list[k0])
#             # sim = proc(r_int.y,n,cutoff)
#             # sol_int[k] = sim
#             liom_dn = r_int.y
 
#             k0 += 1
        
#         output = {"H0_diag":[H0_diag_up,H0_diag_down],"Hint":[Hint_up,Hint_down,Hint_updown],
#                     "LIOM":[liom_up,liom_dn],"LIOM Interactions":[lbits_up,lbits_down,lbits_updown,lbits_charge,lbits_spin],"Invariant":0}
#         if store_flow == True:
#             output["flow"] = sol_int
#             output["dl_list"] = dl_list

#         return output
    
# def flow_dyn(n,hamiltonian,num,dl_list,qmax,cutoff,tlist,method='jit',store_flow=False):
#     """
#     Diagonalise an initial non-interacting Hamiltonian and compute the quench dynamics.

#         Parameters
#         ----------
#         n : integer
#             Linear system size.
#         H0 : array, float
#             Diagonal component of Hamiltonian
#         V0 : array, float
#             Off-diagonal component of Hamiltonian.
#         Hint : array, float
#             Diagonal component of Hamiltonian
#         Vint : array, float
#             Off-diagonal component of Hamiltonian.
#         num : array, float
#             Density operator n_i(t=0) to be time-evolved.
#         dl_list : array, float
#             List of flow times to use for the numerical integration.
#         qmax : integer
#             Maximum number of flow time steps.
#         cutoff : float
#             Threshold value below which off-diagonal elements are set to zero.
#         tlist : array
#             List of timesteps to return time-evolved operator n_i(t).
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.

#         Returns
#         -------
#         sol : array, float
#             Final (diagonal) Hamiltonian
#         central : array, float
#             Local integral of motion (LIOM) computed on the central lattice site of the chain
    
#     """
#     H2 = hamiltonian.H2_spinless

#     # Initialise array to hold solution at all flow times
#     sol = np.zeros((qmax,n**2),dtype=np.float64)
#     sol[0] = (H2).reshape(n**2)

#     # Define integrator
#     r = ode(nonint_ode).set_integrator('dopri5', nsteps=1000)
#     r.set_initial_value((H2).reshape(n**2),dl_list[0])
#     r.set_f_params(n,method)
    
#     # Numerically integrate the flow equations
#     k = 1                       # Flow timestep index
#     J0 = 10.                    # Seed value for largest off-diagonal term
#     # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
#     while r.successful() and k < qmax-1 and J0 > cutoff:
#         r.integrate(dl_list[k])
#         sol[k] = r.y
#         mat = sol[k].reshape(n,n)
#         off_diag = mat-np.diag(np.diag(mat))
#         J0 = max(np.abs(off_diag.reshape(n**2)))
#         k += 1
#     print(k,J0)
#     sol=sol[0:k-1]
#     dl_list= dl_list[0:k-1]

#     # Initialise a density operator in the diagonal basis on the central site
#     # liom = np.zeros((qmax,n**2))
#     init_liom = np.zeros((n,n))
#     init_liom[n//2,n//2] = 1.0
#     # liom[0,:n**2] = init_liom.reshape(n**2)
    
#     # Reverse list of flow times in order to conduct backwards integration
#     dl_list = dl_list[::-1]

#     # Define integrator for density operator
#     n_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
#     n_int.set_initial_value(init_liom.reshape(n**2),dl_list[0])

#     # Numerically integrate the flow equations for the density operator 
#     # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
#     k0=1
#     while n_int.successful() and k0 < len(dl_list[:k]):
#         n_int.set_f_params(n,sol[-k0],method)
#         n_int.integrate(dl_list[k0])
#         liom = n_int.y
#         k0 += 1
    
#     # Take final value for the transformed density operator and reshape to a matrix
#     # central = (liom.reshape(n,n))
    
#     # Invert dl again back to original
#     dl_list = dl_list[::-1] 

#     # Define integrator for density operator again
#     # This time we integrate from l=0 to l -> infinity
#     num_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
#     num_int.set_initial_value(num.reshape(n**2),dl_list[0])
#     k0=1
#     num=np.zeros((k,n**2))
#     while num_int.successful() and k0 < k-1:
#         num_int.set_f_params(n,sol[k0],method)
#         num_int.integrate(dl_list[k0])
#         num[k0] = num_int.y
#         k0 += 1
#     num = num[:k0-1]

#     # Run non-equilibrium dynamics following a quench from CDW state
#     # Returns answer *** in LIOM basis ***
#     evolist = dyn_con(n,num[-1],sol[-1],tlist,method=method)
#     print(evolist)

#     # For each timestep, integrate back from l -> infinity to l=0
#     # i.e. from LIOM basis back to original microscopic basis
#     num_t_list = np.zeros((len(tlist),n**2))
#     dl_list = dl_list[::-1] # Reverse dl for backwards flow
#     for t0 in range(len(tlist)):
#         num_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
#         num_int.set_initial_value(evolist[t0],dl_list[0])

#         k0=1
#         while num_int.successful() and k0 < k-1:
#             num_int.set_f_params(n,sol[-k0],method)
#             num_int.integrate(dl_list[k0])
#             k0 += 1
#         num_t_list[t0] = num_int.y
        
#     # Initialise a list to store the expectation value of time-evolved density operator at each timestep
#     nlist = np.zeros(len(tlist))

#     # Set up initial state as a CDW
#     list1 = np.array([1. for i in range(n//2)])
#     list2 = np.array([0. for i in range(n//2)])
#     state = np.array([val for pair in zip(list1,list2) for val in pair])
    
#     # Compute the expectation value <n_i(t)> for each timestep t
#     n2list = num_t_list[::,:n**2]
#     for t0 in range(len(tlist)):
#         mat = n2list[t0].reshape(n,n)
#         for i in range(n):
#             nlist[t0] += (mat[i,i]*state[i]**2).real

#     output = {"H0_diag":sol[-1].reshape(n,n),"LIOM":liom,"Invariant":0,"Density Dynamics":nlist}
#     if store_flow == True:
#         output["flow"] = sol
#         output["dl_list"] = dl_list[::-1]

#     return output

     
# def flow_dyn_int_singlesite(n,hamiltonian,num,num_int,dl_list,qmax,cutoff,tlist,method='jit',store_flow=False):
#     """
#     Diagonalise an initial interacting Hamiltonian and compute the quench dynamics.

#     This function will return a time-evolved number operator.

#         Parameters
#         ----------
#         n : integer
#             Linear system size.
#         H0 : array, float
#             Diagonal component of Hamiltonian
#         V0 : array, float
#             Off-diagonal component of Hamiltonian.
#         Hint : array, float
#             Diagonal component of Hamiltonian
#         Vint : array, float
#             Off-diagonal component of Hamiltonian.
#         num : array, float
#             Density operator n_i(t=0) to be time-evolved.
#         dl_list : array, float
#             List of flow times to use for the numerical integration.
#         qmax : integer
#             Maximum number of flow time steps.
#         cutoff : float
#             Threshold value below which off-diagonal elements are set to zero.
#         tlist : array
#             List of timesteps to return time-evolved operator n_i(t).
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.

#         Returns
#         -------
#         sol : array, float
#             Final (diagonal) Hamiltonian
#         central : array, float
#             Local integral of motion (LIOM) computed on the central lattice site of the chain
    
#     """
#     H2 = hamiltonian.H2_spinless
#     H4=hamiltonian.H4_spinless

#     # Initialise array to hold solution at all flow times
#     sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
#     # print('Memory64 required: MB', sol_int.nbytes/10**6)
    
#     # Initialise the first flow timestep
#     init = np.zeros(n**2+n**4,dtype=np.float32)
#     init[:n**2] = (H2).reshape(n**2)
#     init[n**2:] = (H4).reshape(n**4)
#     sol_int[0] = init

#     # Define integrator
#     r_int = ode(int_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
#     r_int.set_initial_value(init,dl_list[0])
#     r_int.set_f_params(n,[],method)
    
    
#     # Numerically integrate the flow equations
#     k = 1                       # Flow timestep index
#     J0 = 10.                    # Seed value for largest off-diagonal term
#     # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
#     while r_int.successful() and k < qmax-1 and J0 > cutoff:
#         r_int.integrate(dl_list[k])
#         sol_int[k] = r_int.y
#         mat = sol_int[k,0:n**2].reshape(n,n)
#         off_diag = mat-np.diag(np.diag(mat))
#         J0 = max(off_diag.reshape(n**2))
#         k += 1

#     # Truncate solution list and flow time list to max timestep reached
#     sol_int=sol_int[:k-1]
#     dl_list=dl_list[:k-1]

#     # Define final Hamiltonian, for function return
#     H0final,Hintfinal = sol_int[-1,:n**2].reshape(n,n),sol_int[-1,n**2::].reshape(n,n,n,n)
    
#     # Define final diagonal quadratic Hamiltonian
#     H0_diag = sol_int[-1,:n**2].reshape(n,n)
#     # Define final diagonal quartic Hamiltonian
#     Hint = sol_int[-1,n**2::].reshape(n,n,n,n)   
#     # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
#     HFint = np.zeros(n**2).reshape(n,n)
#     for i in range(n):
#         for j in range(n):
#             HFint[i,j] = Hint[i,i,j,j]
#             HFint[i,j] += -Hint[i,j,j,i]

#     # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
#     lbits = np.zeros(n-1)
#     for q in range(1,n):
#         lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))

#     # Initialise a density operator in the diagonal basis on the central site
#     liom = np.zeros((k,n**2+n**4),dtype=np.float32)
#     init_liom = np.zeros((n,n))
#     init_liom[n//2,n//2] = 1.0
#     liom[0,:n**2] = init_liom.reshape(n**2)
    
#     # Reverse list of flow times in order to conduct backwards integration
#     dl_list = dl_list[::-1]

#     # Define integrator for density operator
#     n_int = ode(liom_ode).set_integrator('dopri5', nsteps=50)
#     n_int.set_initial_value(liom[0],dl_list[0])

#     # Numerically integrate the flow equations for the density operator 
#     # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
#     k0=1
#     while n_int.successful() and k0 < k-1:
#         n_int.set_f_params(n,sol_int[-k0],method)
#         n_int.integrate(dl_list[k0])
#         liom[k0] = n_int.y
#         k0 += 1

#     # Take final value for the transformed density operator and reshape quadratic part to a matrix
#     central = (liom[k0-1,:n**2]).reshape(n,n)

#     # Invert dl again back to original
#     dl_list = dl_list[::-1] 

#     # Define integrator for density operator again
#     # This time we integrate from l=0 to l -> infinity
#     num = np.zeros((k,n**2+n**4),dtype=np.float32)
#     num_init = np.zeros((n,n))
#     num_init[n//2,n//2] = 1.0
#     num[0,0:n**2] = num_init.reshape(n**2)
#     num_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
#     num_int.set_initial_value(num[0],dl_list[0])

#     # Integrate the density operator
#     k0=1
#     while num_int.successful() and k0 < k-1:
#         num_int.set_f_params(n,sol_int[k0],method)
#         num_int.integrate(dl_list[k0])
#         k0 += 1
#     num = num_int.y

#     # Run non-equilibrium dynamics following a quench from CDW state
#     # Returns answer *** in LIOM basis ***
#     evolist2 = dyn_exact(n,num,sol_int[-1],tlist)
#     # evolist2 = dyn_con(n,num,sol_int[-1],tlist)
    
#     # For each timestep, integrate back from l -> infinity to l=0
#     # i.e. from LIOM basis back to original microscopic basis
#     num_t_list2 = np.zeros((len(tlist),n**2+n**4),dtype=np.complex128)
#     dl_list = dl_list[::-1] # Reverse dl for backwards flow
#     for t0 in range(len(tlist)):
        
#         num_int = ode(liom_ode).set_integrator('dopri5',nsteps=100,atol=10**(-8),rtol=10**(-8))
#         num_int.set_initial_value(evolist2[t0],dl_list[0])
#         k0=1
#         while num_int.successful() and k0 < k-1:
#             num_int.set_f_params(n,sol_int[-k0],method,True)
#             num_int.integrate(dl_list[k0])
#             k0 += 1
#         num_t_list2[t0] = (num_int.y).real

#     # Initialise a list to store the expectation value of time-evolved density operator at each timestep
#     nlist2 = np.zeros(len(tlist))

#     # Set up initial state as a CDW
#     list1 = np.array([1. for i in range(n//2)])
#     list2 = np.array([0. for i in range(n//2)])
#     state = np.array([val for pair in zip(list1,list2) for val in pair])
    
#     # Compute the expectation value <n_i(t)> for each timestep t
#     n2list = num_t_list2[::,:n**2]
#     n4list = num_t_list2[::,n**2:]
#     for t0 in range(len(tlist)):
#         mat = n2list[t0].reshape(n,n)
#         mat4 = n4list[t0].reshape(n,n,n,n)
#         for i in range(n):
#             nlist2[t0] += (mat[i,i]*state[i]).real
#             for j in range(n):
#                 if i != j:
#                     nlist2[t0] += (mat4[i,i,j,j]*state[i]*state[j]).real
#                     nlist2[t0] += -(mat4[i,j,j,i]*state[i]*state[j]).real
#     print(nlist2)

#     output = {"H0_diag":H0_diag,"Hint":Hintfinal,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":0,"Density Dynamics":nlist2}
#     if store_flow == True:
#         output["flow"] = sol_int
#         output["dl_list"] = dl_list[::-1]

#     return output
 
    
# def flow_dyn_int_imb(n,hamiltonian,num,num_int,dl_list,qmax,cutoff,tlist,method='jit',store_flow=False):
#     """
#     Diagonalise an initial interacting Hamiltonian and compute the quench dynamics.

#     This function will return the imbalance following a quench, which involves computing the 
#     non-equilibrium dynamics of the densiy operator on every single lattice site.

#         Parameters
#         ----------
#         n : integer
#             Linear system size.
#         H0 : array, float
#             Diagonal component of Hamiltonian
#         V0 : array, float
#             Off-diagonal component of Hamiltonian.
#         Hint : array, float
#             Diagonal component of Hamiltonian
#         Vint : array, float
#             Off-diagonal component of Hamiltonian.
#         num : array, float
#             Density operator n_i(t=0) to be time-evolved.
#         dl_list : array, float
#             List of flow times to use for the numerical integration.
#         qmax : integer
#             Maximum number of flow time steps.
#         cutoff : float
#             Threshold value below which off-diagonal elements are set to zero.
#         tlist : array
#             List of timesteps to return time-evolved operator n_i(t).
#         method : string, optional
#             Specify which method to use to generate the RHS of the flow equations.
#             Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
#             The first two are built-in NumPy methods, while the latter two are custom coded for speed.

#         Returns
#         -------
#         sol : array, float
#             Final (diagonal) Hamiltonian
#         central : array, float
#             Local integral of motion (LIOM) computed on the central lattice site of the chain
    
#     """

#     H2 = hamiltonian.H2_spinless
#     H4 = hamiltonian.H4_spinless

#     # Initialise array to hold solution at all flow times
#     sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
#     # print('Memory64 required: MB', sol_int.nbytes/10**6)
    
#     # Initialise the first flow timestep
#     init = np.zeros(n**2+n**4,dtype=np.float32)
#     init[:n**2] = (H2).reshape(n**2)
#     init[n**2:] = (H4).reshape(n**4)
#     sol_int[0] = init

#     # Define integrator
#     r_int = ode(int_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
#     r_int.set_initial_value(init,dl_list[0])
#     r_int.set_f_params(n,[],method)
    
    
#     # Numerically integrate the flow equations
#     k = 1                       # Flow timestep index
#     J0 = 10.                    # Seed value for largest off-diagonal term
#     # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
#     while r_int.successful() and k < qmax-1 and J0 > cutoff:
#         r_int.integrate(dl_list[k])
#         sol_int[k] = r_int.y
#         mat = sol_int[k,0:n**2].reshape(n,n)
#         off_diag = mat-np.diag(np.diag(mat))
#         J0 = max(off_diag.reshape(n**2))
#         k += 1

#     # Truncate solution list and flow time list to max timestep reached
#     sol_int=sol_int[:k-1]
#     dl_list=dl_list[:k-1]

#     # Define final Hamiltonian, for function return
#     H0final,Hintfinal = sol_int[-1,:n**2].reshape(n,n),sol_int[-1,n**2::].reshape(n,n,n,n)
    
#     # Define final diagonal quadratic Hamiltonian
#     H0_diag = sol_int[-1,:n**2].reshape(n,n)
#     # Define final diagonal quartic Hamiltonian
#     Hint = sol_int[-1,n**2::].reshape(n,n,n,n)   
#     # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
#     HFint = np.zeros(n**2).reshape(n,n)
#     for i in range(n):
#         for j in range(n):
#             HFint[i,j] = Hint[i,i,j,j]
#             HFint[i,j] += -Hint[i,j,j,i]

#     # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
#     lbits = np.zeros(n-1)
#     for q in range(1,n):
#         lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))

#     # Initialise a density operator in the diagonal basis on the central site
#     liom = np.zeros((k,n**2+n**4),dtype=np.float32)
#     init_liom = np.zeros((n,n))
#     init_liom[n//2,n//2] = 1.0
#     liom[0,:n**2] = init_liom.reshape(n**2)
    
#     # Reverse list of flow times in order to conduct backwards integration
#     dl_list = dl_list[::-1]
    
#     # Set up initial state as a CDW
#     list1 = np.array([1. for i in range(n//2)])
#     list2 = np.array([0. for i in range(n//2)])
#     state = np.array([val for pair in zip(list1,list2) for val in pair])
    
#     # Define lists to store the time-evolved density operators on each lattice site
#     # 'imblist' will include interaction effects
#     # 'imblist2' includes only single-particle effects
#     # Both are kept to check for diverging interaction terms
#     imblist = np.zeros((n,len(tlist)))
#     imblist2 = np.zeros((n,len(tlist)))

#     # Compute the time-evolution of the number operator on every site
#     for site in range(n):
#         # Initialise operator to be time-evolved
#         num = np.zeros((k,n**2+n**4))
#         num_init = np.zeros((n,n),dtype=np.float32)
#         num_init[site,site] = 1.0

#         num[0,0:n**2] = num_init.reshape(n**2)
        
#             # Invert dl again back to original
#         dl_list = dl_list[::-1]

#         # Define integrator for density operator again
#         # This time we integrate from l=0 to l -> infinity
#         num_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
#         num_int.set_initial_value(num[0],dl_list[0])
#         k0=1
#         while num_int.successful() and k0 < k-1:
#             num_int.set_f_params(n,sol_int[k0],method)
#             num_int.integrate(dl_list[k0])
#             # liom[k0] = num_int.y
#             k0 += 1
#         num = num_int.y
        
#         # Run non-equilibrium dynamics following a quench from CDW state
#         # Returns answer *** in LIOM basis ***
#         evolist2 = dyn_exact(n,num,sol_int[-1],tlist)
#         dl_list = dl_list[::-1] # Reverse the flow
        
#         num_t_list2 = np.zeros((len(tlist),n**2+n**4))
#         # For each timestep, integrate back from l -> infinity to l=0
#         # i.e. from LIOM basis back to original microscopic basis
#         for t0 in range(len(tlist)):
            
#             num_int = ode(liom_ode).set_integrator('dopri5',nsteps=50,atol=10**(-8),rtol=10**(-8))
#             num_int.set_initial_value(evolist2[t0],dl_list[0])
#             k0=1
#             while num_int.successful() and k0 < k-1:
#                 num_int.set_f_params(n,sol_int[-k0],method,True)
#                 num_int.integrate(dl_list[k0])
#                 k0 += 1
#             num_t_list2[t0] = num_int.y
        
#         # Initialise lists to store the expectation value of time-evolved density operator at each timestep
#         nlist = np.zeros(len(tlist))
#         nlist2 = np.zeros(len(tlist))
        
#         # Compute the expectation value <n_i(t)> for each timestep t
#         n2list = num_t_list2[::,:n**2]
#         n4list = num_t_list2[::,n**2:]
#         for t0 in range(len(tlist)):
#             mat = n2list[t0].reshape(n,n)
#             mat4 = n4list[t0].reshape(n,n,n,n)
#             # phaseMF = 0.
#             for i in range(n):
#                 # nlist[t0] += (mat[i,i]*state[i]**2).real
#                 nlist[t0] += (mat[i,i]*state[i]).real
#                 nlist2[t0] += (mat[i,i]*state[i]).real
#                 for j in range(n):
#                     if i != j:
#                         nlist[t0] += (mat4[i,i,j,j]*state[i]*state[j]).real
#                         nlist[t0] += -(mat4[i,j,j,i]*state[i]*state[j]).real
                        
#         imblist[site] = ((-1)**site)*nlist/n
#         imblist2[site] = ((-1)**site)*nlist2/n

#     # Compute the imbalance over the entire system
#     # Note that the (-1)^i factors are included in imblist already
#     imblist = 2*np.sum(imblist,axis=0)
#     imblist2 = 2*np.sum(imblist2,axis=0)

#     output = {"H0_diag":H0_diag,"Hint":Hintfinal,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":0,"Imbalance":imblist}
#     if store_flow == True:
#         output["flow"] = sol_int
#         output["dl_list"] = dl_list[::-1]

#     return output

# #------------------------------------------------------------------------------
# # Function for benchmarking the non-interacting system using 'einsum'
# def flow_einsum_nonint(H0,V0,dl):
#     """ Benchmarking function to diagonalise H for a non-interacting system using NumPy's einsum function.

#         This function is used to test the routines included in contract.py by explicitly calling 
#         the 'einsum' function, which is a slow but very transparent way to do the matrix/tensor contractions.

#         Parameters
#         ----------
#         H0 : array, float
#             Diagonal component of Hamiltonian
#         V0 : array, float
#             Off-diagonal component of Hamiltonian.
#         dl : float
#             Size of step in flow time (dl << 1)
    
#     """
      
#     startTime = datetime.now()
#     q = 0
#     while np.max(np.abs(V0))>10**(-2):
    
#         # Non-interacting generator
#         eta0 = np.einsum('ij,jk->ik',H0,V0) - np.einsum('ki,ij->kj',V0,H0,optimize=True)
        
#         # Flow of non-interacting terms
#         dH0 = np.einsum('ij,jk->ik',eta0,(H0+V0)) - np.einsum('ki,ij->kj',(H0+V0),eta0,optimize=True)
    
#         # Update non-interacting terms
#         H0 = H0+dl*np.diag(np.diag(dH0))
#         V0 = V0 + dl*(dH0-np.diag(np.diag(dH0)))

#         q += 1

#     print('***********')
#     print('FE time - einsum',datetime.now()-startTime)
#     print('Max off diagonal element: ', np.max(np.abs(V0)))
#     print(np.sort(np.diag(H0)))

# #------------------------------------------------------------------------------  
# # Function for benchmarking the non-interacting system using 'tensordot'
# def flow_tensordot_nonint(H0,V0,dl):  

#     """ Benchmarking function to diagonalise H for a non-interacting system using NumPy's tensordot function.

#         This function is used to test the routines included in contract.py by explicitly calling 
#         the 'tensordot' function, which is slightly faster than einsum but also less transparent.

#         Parameters
#         ----------
#         H0 : array, float
#             Diagonal component of Hamiltonian
#         V0 : array, float
#             Off-diagonal component of Hamiltonian.
#         dl : float
#             Size of step in flow time (dl << 1)
    
#     """   

#     startTime = datetime.now()
#     q = 0
#     while np.max(np.abs(V0))>10**(-3):
    
#         # Non-interacting generator
#         eta = np.tensordot(H0,V0,axes=1) - np.tensordot(V0,H0,axes=1)
        
#         # Flow of non-interacting terms
#         dH0 = np.tensordot(eta,H0+V0,axes=1) - np.tensordot(H0+V0,eta,axes=1)
    
#         # Update non-interacting terms
#         H0 = H0+dl*np.diag(np.diag(dH0))
#         V0 = V0 + dl*(dH0-np.diag(np.diag(dH0)))

#         q += 1
        
#     print('***********')
#     print('FE time - Tensordot',datetime.now()-startTime)
#     print('Max off diagonal element: ', np.max(np.abs(V0)))
#     print(np.sort(np.diag(H0)))




