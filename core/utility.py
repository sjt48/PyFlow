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