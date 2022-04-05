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

This file contains various helper functions.

"""

import os
import numpy as np
from .contract import contract,contractNO 

def namevar(dis_type,dsymm,dyn,norm,n,LIOM,species):
    if norm == True:
        nm = 'NO'
    else:
        nm = 'PT'
    if species == 'spinful fermion':
        spec = 'spin_fermion/%s' %(dsymm)
    elif species == 'spinless fermion':
        spec = 'fermion'
     # Make directory to store data
    if dyn == False:
        if not os.path.exists('%s/data/%s/%s/%s/%s/dataN%s' %(spec,dis_type,nm,LIOM,'static',n)):
            os.makedirs('%s/data/%s/%s/%s/%s/dataN%s' %(spec,dis_type,nm,LIOM,'static',n))
        namevar = '%s/data/%s/%s/%s/%s/dataN%s' %(spec,dis_type,nm,LIOM,'static',n) 
    elif dyn == True:
        if not os.path.exists('%s/data/%s/%s/%s/%s/dataN%s' %(spec,dis_type,nm,LIOM,'dyn',n)):
            os.makedirs('%s/data/%s/%s/%s/%s/dataN%s' %(spec,dis_type,nm,LIOM,'dyn',n))
        namevar = '%s/data/%s/%s/%s/%s/dataN%s' %(spec,dis_type,nm,LIOM,'dyn',n)
    
    return namevar


    #------------------------------------------------------------------------------
# Generate the state vector used for normal-ordering
def nstate(n,a):
    """ Generates a NumPy array that represents the state used to compute normal-ordering corrections.

        Parameters
        ----------

        n : integer
            Linear system size
        a : string, float
            Specify the state used: can be a string specifying a charge density wave (010101..., 'CDW'), 
            a 'step' (000....111), a randomly filled state of 0s and 1s ('random'), a randomly filled state 
            of 0s and 1s with a total of n//2 particles ('random_half') or else a homogeneous product state
            with occupancy on every site given by the float 'a' (i.e. <n_i> = a for all sites i).

    """
    if a == 'CDW':
        # Neel state, e.g. 101010...
        list1 = np.array([1. for i in range(n//2)])
        list2 = np.array([0. for i in range(n//2)])
        state0 = np.array([val for pair in zip(list1,list2) for val in pair])
    elif a == 'step':
        list1 = np.array([1. for i in range(n//2)])
        list2 = np.array([0. for i in range(n//2)])
        state0 = np.join(list1,list2)
    elif a == 'random':
        state0 = np.random.choice([0.,1.0],n)
    elif a == 'random_half':
        state0 = np.array([1. for i in range(n//2)]+[0.0 for i in range(n//2)])
        np.random.shuffle(state0)
    else:
        state0 = np.array([a for i in range(n)])
        
    return state0

def state_spinless(H2,state='CDW'):

    n,_ = H2.shape
    _,V1 = np.linalg.eigh(H2)
    state = np.zeros(n)
    sites = np.array([i for i in range(n)])
    random = np.random.choice(sites,n//2)
    count = 0
    random = range(0,n,2)
    for site in random:
        for i in range(n):
            if np.argmax(np.abs(V1[:,i])) == site:
                psi = V1[:,i]
                state += np.array([v**2 for v in psi])
                count += 1
    state *= 1/count
    state = np.round(state,decimals=6)
    if np.round(np.sum(state),3) != 1.0:
        print('NORMALISATION ERROR - CHECK N/O STATE')
        print(state)
    return state

def states_spin(H2up,H2dn,state='CDW'):

    n,_ = H2up.shape
    # Scale-dependent normal ordering wrt an excited state of the free Hamiltonian(s)
    _,V1 = np.linalg.eigh(H2up)
    upstate = np.zeros(n)
    random = range(0,n,2)
    count = 0
    for site in random:
        for i in range(n):
            if np.argmax(np.abs(V1[:,i]))==site:
                psi = V1[:,i]
                upstate += np.array([v**2 for v in psi])
                count += 1
    upstate *= 1/(2*count)
    upstate = np.round(upstate,decimals=6)
    _,V1 = np.linalg.eigh(H2dn)
    downstate = np.zeros(n)
    if state == 'SDW':
        random = range(1,n,2)
    count = 0
    for site in random:
        for i in range(n):
            if np.argmax(np.abs(V1[:,i]))==site:
                psi = V1[:,i]
                downstate += np.array([v**2 for v in psi])
                count += 1
    downstate *= 1/(2*count)
    downstate = np.round(downstate,decimals=6)
    if np.round(np.sum(upstate)+np.sum(downstate),3) != 1.0:
        print('NORMALISATION ERROR - CHECK N/O STATE')
        print(np.round(np.sum(upstate)+np.sum(downstate),3))
        print(upstate,downstate)

    return upstate,downstate

def unpack_spin_hamiltonian(y,n):

    # Extract various components of the Hamiltonian from the input array 'y'
    # Start with the quadratic part of the spin-up fermions
    H2up = y[0:n**2].reshape(n,n)
    H2up_0 = np.diag(np.diag(H2up))
    V2up = H2up - H2up_0
    
    # Now the quadratic part of the spin-down fermions
    H2dn = y[n**2:2*n**2].reshape(n,n)
    H2dn_0 = np.diag(np.diag(H2dn))
    V2dn = H2dn - H2dn_0

    # Now we define the quartic (interaction) terms for the spin-up fermions
    H4up = y[2*n**2:2*n**2+n**4].reshape(n,n,n,n)
    H4up_0 = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
                # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                H4up_0[i,i,j,j] = H4up[i,i,j,j]
                H4up_0[i,j,j,i] = H4up[i,j,j,i]
    V4up = H4up-H4up_0
    
    # The same for spin-down fermions
    H4dn = y[2*n**2+n**4:2*n**2+2*n**4].reshape(n,n,n,n)
    H4dn_0 = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
                # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                H4dn_0[i,i,j,j] = H4dn[i,i,j,j]
                H4dn_0[i,j,j,i] = H4dn[i,j,j,i]
    V4dn = H4dn-H4dn_0

    # And the same for the mixed quartic term, with 2 spin-up fermion operators and 2 spin-down fermion operators
    H4updn = y[2*n**2+2*n**4:].reshape(n,n,n,n)
    H4updn_0 = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            # Load dHint_diag with diagonal values (n_i n_j)
            H4updn_0[i,i,j,j] = H4updn[i,i,j,j]
    V4updn = H4updn-H4updn_0 

    return {"H2up":H2up,"H2dn":H2dn,"H2up_0":H2up_0,"H2dn_0":H2dn_0,"V2up":V2up,"V2dn":V2dn,
            "H4up":H4up,"H4dn":H4dn,"H4updn":H4updn,"H4up_0":H4up_0,"H4dn_0":H4dn_0,"H4updn_0":H4updn_0,
                "V4up":V4up,"V4dn":V4dn,"V4updn":V4updn}

def eta_spin(y,state='CDW',norm=False,method='vec'):

    H2up_0 = y["H2up_0"]
    H2dn_0 = y["H2dn_0"]
    V2up = y["V2up"]
    V2dn = y["V2dn"]
    H4up_0 = y["H4up_0"]
    H4dn_0 = y["H4dn_0"]
    H4updn_0 = y["H4updn_0"]
    V4up = y["V4up"]
    V4dn = y["V4dn"]
    V4updn = y["V4updn"]

    upstate,downstate = states_spin(y["H2up"],y["H2dn"],state=state)

    # Compute all relevant generators
    eta2up = contract(H2up_0,V2up,method=method,eta=True)
    eta2dn = contract(H2dn_0,V2dn,method=method,eta=True)
    eta4up = contract(H4up_0,V2up,method=method,eta=True) + contract(H2up_0,V4up,method=method,eta=True)
    eta4dn = contract(H4dn_0,V2dn,method=method,eta=True) + contract(H2dn_0,V4dn,method=method,eta=True)
    eta4updn = -contract(V4updn,H2up_0,method=method,eta=True,pair='first') - contract(V4updn,H2dn_0,method=method,eta=True,pair='second')
    eta4updn += contract(H4updn_0,V2up,method=method,eta=True,pair='first') + contract(H4updn_0,V2dn,method=method,eta=True,pair='second')

    if norm == True:
        eta2up += contractNO(H4up_0,V2up,method=method,eta=True,state=upstate)
        eta2up += contractNO(H2up_0,V4up,method=method,eta=True,state=upstate)
        eta2dn += contractNO(H4dn_0,V2dn,method=method,eta=True,state=downstate)
        eta2dn += contractNO(H2dn_0,V4dn,method=method,eta=True,state=downstate)
        eta2up += contractNO(H4updn_0,V2dn,method=method,eta=True,state=downstate,pair='second')
        eta2up += contractNO(H2dn_0,V4updn,method=method,eta=True,state=downstate,pair='second')
        eta2dn += contractNO(H4updn_0,V2up,method=method,eta=True,state=upstate,pair='first')
        eta2dn += contractNO(H2up_0,V4updn,method=method,eta=True,state=upstate,pair='first')

        eta4up += contractNO(H4up_0,V4up,method=method,eta=True,state=upstate)
        eta4dn += contractNO(H4dn_0,V4dn,method=method,eta=True,state=downstate)

        eta4updn += contractNO(H4up_0,V4updn,method=method,eta=True,pair='up-mixed',state=upstate)
        eta4up += contractNO(H4updn_0,V4updn,method=method,eta=True,pair='mixed-mixed-up',state=downstate)
        eta4updn += contractNO(H4dn_0,V4updn,method=method,eta=True,pair='down-mixed',state=downstate)
        eta4dn += contractNO(H4updn_0,V4updn,method=method,eta=True,pair='mixed-mixed-down',state=upstate)
        eta4updn += contractNO(H4updn_0,V4up,method=method,eta=True,pair='mixed-up',state=upstate)
        eta4updn += contractNO(H4updn_0,V4dn,method=method,eta=True,pair='mixed-down',state=downstate)

        eta4updn += contractNO(H4updn_0,V4updn,method=method,eta=True,pair='mixed',upstate=upstate,downstate=downstate)

    return eta2up,eta2dn,eta4up,eta4dn,eta4updn,upstate,downstate


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
                            flevels[count] += Hint_up[j,q]*int(lev0[j])*int(lev0[q]) 
                            flevels[count] += Hint_down[j,q]*int(lev1[j])*int(lev1[q])                             
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