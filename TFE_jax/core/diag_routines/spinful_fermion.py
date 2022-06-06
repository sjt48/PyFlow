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
from numba import jit
import gc
from ..contract import contract,contractNO
from ..utility import unpack_spin_hamiltonian, eta_spin, indices
from scipy.integrate import ode

#------------------------------------------------------------------------------ 

@jit(nopython=True,parallel=True,fastmath=True,cache=True)
def cut_spin(y,n,cutoff,indices):
    """ Checks if ALL quadratic off-diagonal parts have decayed below cutoff*10e-3 and TYPICAL (median) off-diag quartic term have decayed below cutoff. """
    mat2 = y[:n**2].reshape(n,n)
    mat3 = y[n**2:2*n**2].reshape(n,n)
    mat2_od = mat2-np.diag(np.diag(mat2))
    mat3_od = mat2-np.diag(np.diag(mat3))
    mat_od = np.zeros(2*n**2)
    mat_od[:n**2] = mat2_od.reshape(n**2)
    mat_od[n**2:] = mat3_od.reshape(n**2)

    if np.max(np.abs(mat_od)) < cutoff*10**(-3):
        mat4 = y[2*n**2:2*n**2+n**4]
        mat5 = y[2*n**2+n**4:2*n**2+2*n**4]
        mat6 = y[2*n**2+2*n**2:2*n**2+3*n**4]
        mat4_od = np.zeros(3*n**4)     
        for i in indices:         
            mat4_od[i] = mat4[i]
            mat4_od[i+n**4] = mat5[i]
            mat4_od[i+2*n**4] = mat6[i]
        mat4_od = mat4_od[mat4_od != 0]
        if np.median(np.abs(mat4_od)) < cutoff:
            return 0 
        else:
            return 1
    else:
        return 1

#------------------------------------------------------------------------------

# @jit('UniTuple(float64[:,:,:,:],2)(float64[:,:,:,:],boolean)',nopython=True,parallel=True,fastmath=True,cache=True)
def extract_diag(A,norm=False):
    B = np.zeros(A.shape,dtype=np.float64)
    n,_,_,_ = A.shape
    for i in range(n): 
        for j in range(n):
            if i != j:
                if norm == True:
                    # Symmetrise (for normal-ordering wrt inhomogeneous states)
                    A[i,i,j,j] += -A[i,j,j,i]
                    A[i,j,j,i] = 0.
            if i != j:
                if norm == True:
                    # Symmetrise (for normal-ordering wrt inhomogeneous states)
                    A[i,i,j,j] += A[j,j,i,i]
                    A[i,i,j,j] *= 0.5
                # Load new array with diagonal values
                B[i,i,j,j] = A[i,i,j,j]
    return B,A

def int_ode_spin(l,y,n,method='jit',norm=True,no_state='CDW'):
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

        ham = unpack_spin_hamiltonian(y,n)
        eta0up,eta0down,eta_int_up,eta_int_down,eta_int_updown,upstate,downstate = eta_spin(ham,method=method,norm=norm,no_state=no_state)

        H2up = ham["H2up"]
        H2dn = ham["H2dn"]
        H4up = ham["H4up"]
        H4dn = ham["H4dn"]
        H4updn = ham["H4updn"]

        # Then compute the RHS of the flow equations
        sol_up = contract(eta0up,H2up,method=method)
        sol_down = contract(eta0down,H2dn,method=method)
        sol_int_up = contract(eta_int_up,H2up,method=method) + contract(eta0up,H4up,method=method)
        sol_int_down = contract(eta_int_down,H2dn,method=method) + contract(eta0down,H4dn,method=method)
        sol_int_updown = contract(eta_int_updown,H2dn,method=method,pair='second') + contract(eta0down,H4updn,method=method,pair='second')
        sol_int_updown += contract(eta_int_updown,H2up,method=method,pair='first') + contract(eta0up,H4updn,method=method,pair='first')

        if norm == True:
            sol_up += contractNO(eta_int_up,H2up,method=method,state=upstate)
            sol_up += contractNO(eta0up,H4up,method=method,state=upstate)
            sol_down += contractNO(eta_int_down,H2dn,method=method,state=downstate)
            sol_down += contractNO(eta0down,H4dn,method=method,state=downstate)
            sol_up += contractNO(eta_int_updown,H2dn,method=method,state=downstate,pair='second')
            sol_up += contractNO(eta0down,H4updn,method=method,state=downstate,pair='second')
            sol_down += contractNO(eta_int_updown,H2up,method=method,state=upstate,pair='first')
            sol_down += contractNO(eta0up,H4updn,method=method,state=upstate,pair='first')

            sol_int_up += contractNO(eta_int_up,H4up,method=method,state=upstate)
            sol_int_down += contractNO(eta_int_down,H4dn,method=method,state=downstate)

            sol_int_updown += contractNO(eta_int_up,H4updn,method=method,pair='up-mixed',state=upstate)
            sol_int_up += contractNO(eta_int_updown,H4updn,method=method,pair='mixed-mixed-up',state=downstate)
            sol_int_updown += contractNO(eta_int_down,H4updn,method=method,pair='down-mixed',state=downstate)
            sol_int_down += contractNO(eta_int_updown,H4updn,method=method,pair='mixed-mixed-down',state=upstate)
            sol_int_updown += contractNO(eta_int_updown,H4up,method=method,pair='mixed-up',state=upstate)
            sol_int_updown += contractNO(eta_int_updown,H4dn,method=method,pair='mixed-down',state=downstate)

            sol_int_updown += contractNO(eta_int_updown,H4updn,method=method,pair='mixed',upstate=upstate,downstate=downstate)

        # Assemble output array
        sol0 = np.zeros(2*n**2+3*n**4)
        sol0[:n**2] = sol_up.reshape(n**2)
        sol0[n**2:2*n**2] = sol_down.reshape(n**2)
        sol0[2*n**2:2*n**2+n**4] = sol_int_up.reshape(n**4)
        sol0[2*n**2+n**4:2*n**2+2*n**4] = sol_int_down.reshape(n**4)
        sol0[2*n**2+2*n**4:] = sol_int_updown.reshape(n**4)

        return sol0

def liom_spin(l,nlist,y,n,method='jit',comp=False,norm=False):

    ham = unpack_spin_hamiltonian(y,n)
    eta0up,eta0down,eta_int_up,eta_int_down,eta_int_updown,upstate,downstate = eta_spin(ham,method=method,norm=norm)
                
    n2_up = nlist[0:n**2].reshape(n,n)
    n2_down = nlist[n**2:2*n**2].reshape(n,n)
    n4_up = nlist[2*n**2:2*n**2+n**4].reshape(n,n,n,n)
    n4_down = nlist[2*n**2+n**4:2*n**2+2*n**4].reshape(n,n,n,n)
    n4_updown = nlist[2*n**2+2*n**4:].reshape(n,n,n,n)

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
#-----------------------------------------------  

def flow_static_int_spin(n,hamiltonian,dl_list,qmax,cutoff,method='jit',store_flow=False,norm=False,no_state='CDW'):
      
        H0_up,H0_down,Hint_up,Hint_down,Hint_updown = hamiltonian.H2_spinup,hamiltonian.H2_spindown,hamiltonian.H4_spinup,hamiltonian.H4_spindown,hamiltonian.H4_mixed

        sol_int = np.zeros((qmax,2*n**2+3*n**4),dtype=np.float64)
        # print('Memory64 required: MB', sol_int.nbytes/10**6)
        # sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
        # print('Memory32 required: MB', sol_int.nbytes/10**6)
        
        r_int = ode(int_ode_spin).set_integrator('dopri5', nsteps=150,atol=10**(-6),rtol=10**(-3))
        r_int.set_f_params(n,method,norm,no_state)

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
        decay = 1
        index_list = indices(n)
        while r_int.successful() and k < qmax-1 and J0 > cutoff and decay == 1:
            r_int.integrate(dl_list[k])
            # sim = proc(r_int.y,n,cutoff)
            # sol_int[k] = sim
            sol_int[k] = r_int.y
            mat_up = (r_int.y)[0:n**2].reshape(n,n)
            mat_down = (r_int.y)[n**2:2*n**2].reshape(n,n)
            off_diag_up = mat_up-np.diag(np.diag(mat_up))
            off_diag_down = mat_down-np.diag(np.diag(mat_down))
            J0_up = max(np.abs(off_diag_up).reshape(n**2))
            J0_down = max(np.abs(off_diag_down).reshape(n**2))
            J0=max(J0_up,J0_down)

            decay = cut_spin(r_int.y,n,cutoff,index_list)

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

        print(H0_diag_up)
        print(H0_diag_down)
        print(HFint_up)
        print(HFint_down)
        print(HFint_up - HFint_down)

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

        r_int.set_initial_value(init,dl_list[0])
        init_up = np.zeros(2*n**2+3*n**4,dtype=np.float64)
        init_dn = np.zeros(2*n**2+3*n**4,dtype=np.float64)
        temp = np.zeros((n,n))
        temp[n//2,n//2] = 1.0
        init_up[:n**2] = temp.reshape(n**2)
        init_dn[n**2:2*n**2] = temp.reshape(n**2)

        dl_list = dl_list[::-1]

        r_int = ode(liom_spin).set_integrator('dopri5', nsteps=150,atol=10**(-6),rtol=10**(-3))
        r_int.set_initial_value(init_up,dl_list[0])

        k0 = 1
        while r_int.successful() and k0 < k-1:
            r_int.set_f_params(sol_int[-k0],n,method,no_state)
            r_int.integrate(dl_list[k0])
            # sim = proc(r_int.y,n,cutoff)
            # sol_int[k] = sim
            liom_up = r_int.y
 
            k0 += 1

        r_int = ode(liom_spin).set_integrator('dopri5', nsteps=150,atol=10**(-6),rtol=10**(-3))
        r_int.set_initial_value(init_dn,dl_list[0])

        k0 = 1
        while r_int.successful() and k0 < k-1:
            r_int.set_f_params(sol_int[-k0],n,method,no_state)
            r_int.integrate(dl_list[k0])
            # sim = proc(r_int.y,n,cutoff)
            # sol_int[k] = sim
            liom_dn = r_int.y
 
            k0 += 1
        
        output = {"H0_diag":[H0_diag_up,H0_diag_down],"Hint":[Hint_up,Hint_down,Hint_updown],
                    "LIOM":[liom_up,liom_dn],"LIOM Interactions":[lbits_up,lbits_down,lbits_updown,lbits_charge,lbits_spin],"Invariant":0}
        if store_flow == True:
            output["flow"] = sol_int
            output["dl_list"] = dl_list

        return output