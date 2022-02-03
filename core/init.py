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

This file contains the code used to initialise the Hamiltonian for a variety of choices of disorder.

"""

import os
from psutil import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of MKL threads to run in parallel
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
import numpy as np
from sympy import prime

def Hinit(n,d,J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False):
    """ Generate the non-interacting part of the Hamiltonian with the specified on-site potential. """

    np.random.seed()
    print('Choice of potential = %s' %dis_type)

    #-----------------------------------------------------------------
    # Non-interacting matrices
    H0 = np.zeros((n,n),dtype=np.float32)
    if dis_type == 'random':
        for i in range(n):
            # Initialise Hamiltonian with random on-site terms
            H0[i,i] = np.random.uniform(-d,d)
    elif dis_type == 'test':
        # For testing purposes: fixed list of randomly generated numbers for method comparisons
        print('**** FIXED DISORDER REALISATION FOR TESTING - DISABLE FOR REAL DATA ****')
        randomlist = [2.4979359120556666, -4.477621238657079, -4.448810326437316, -3.6115452666436543, -1.2802110535766298, -3.336862075297363, -0.3370611440832194, -3.8232260796601523, -0.5134617674857918, 1.32895294857477]
        for i in range(n):
            H0[i,i] = randomlist[i]
    elif dis_type == 'linear':
        for i in range(n):
            # Initialise Hamiltonian with linearly increasing on-site terms
            H0[i,i] = d*i
    elif dis_type == 'curved':
        for i in range(n):
            # Initialise Hamiltonian with linearly increasing on-site terms, plus some small curvature
            H0[i,i] = d*i+x*(i/n)**2
    elif dis_type == 'prime':
        for i in range(n):
            # Initialise Hamiltonian with square root prime numbers as on-site terms
            H0[i,i] = d*np.sqrt(prime(i+1))
    elif dis_type == 'QPgolden':
        phase = np.random.uniform(-np.pi,np.pi)
        print('phase = ', phase)
        phi = (1.+np.sqrt(5.))/2.
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
    elif dis_type == 'QPsilver':
        phase = np.random.uniform(-np.pi,np.pi)
        print('phase = ', phase)
        phi = 1.+np.sqrt(2.)
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
    elif dis_type == 'QPbronze':
        phase = np.random.uniform(-np.pi,np.pi)
        print('phase = ', phase)
        phi = (3.+np.sqrt(13.))/2.
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
    elif dis_type == 'QPrandom':
        phi = (1.+np.sqrt(5.))/2.
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + np.random.uniform(-np.pi,np.pi))
    elif dis_type == 'QPtest':
        print('**** FIXED PHASE FOR TESTING - DISABLE FOR REAL DATA ****')
        phase=0.
        phi = (1.+np.sqrt(5.))/2.
        for i in range(n):
            # Initialise Hamiltonian with quasiperiodic on-site terms
            H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)

    # Initialise V0 with nearest-neighbour hopping
    if pwrhop == False:
        V0 = np.diag(J*np.ones(n-1,dtype=np.float32),1) + np.diag(J*np.ones(n-1,dtype=np.float32),-1)
    else:
        V0 = np.zeros((n,n))
        for k in range(1,n):
            templist = [0.]*(n-k)
            for q in range(n-k):
                if pwrhop == 'random':
                    # Pick hopping randomly from a distribution with standard deviation decaying like a power-law with distance
                    templist[q] = J*np.random.normal(0,k**(-alpha))
                else:
                    # Use a hopping that decays uniformly like a power-law with distance
                    templist[q] = J*k**(-alpha)
            V0 += np.diag(templist,k)
            V0 += np.diag(templist,-k)

    #--------------------------------------------------------------------------
    # elif Fourier == True:
    #     # Initialise momentum-space Hamiltonian

    #     # Diagonal terms
    #     y0 = [-2*Jxx*np.cos(2*np.pi*i/n) for i in range(-n//2,n//2)]

    #     # Fourier-transform the on-site disorder
    #     hlist = [np.random.uniform(-d,d) for i in range(n)]
    #     H0 += np.diag(hlist)
    #     H0 += np.diag([Jxx for i in range(n-1)],1)
    #     H0 += np.diag([Jxx for i in range(n-1)],-1)
    #     H0[0,n-1] += Jxx
    #     H0[n-1,0] += Jxx

    #     hmat = np.zeros(n**2,dtype=complex).reshape(n,n)
    #     for i in range(-n//2,n//2):
    #         for j in range(-n//2,n//2):
    #             hmat[i+n//2,j+n//2] = (1/float(n))*np.sum([hlist[k]*np.exp(1j*2*np.pi*(i-j)*k/L) for k in range(n)])
    #     for k in range(1,n+1):
    #         jlist += [np.diag(hmat,k)]
    #     jlist = np.concatenate(jlist)
    #     diag = y0+np.diag(hmat).real
    #     dlist = [Jz/np.sqrt(n)]*(n*(n-1)//2) #Diagonal part only

    #     #----------------------------------------------------------------------
    #     # Set up final Hamiltonian

    #     if intr == False:
    #         y00 = np.concatenate((diag, jlist.real,jlist.imag),axis=None)
    #     elif intr == True:
    #         y00 = np.concatenate((diag, jlist.real,jlist.imag,dlist),axis=None)

         # Compare eigenvalues of free Hamiltonian in momentum- and real-space
#        H1 = np.zeros(n**2,dtype=complex).reshape(n,n)
#        H1 += hmat + np.diag(y0)
#        print(sorted(np.linalg.eigvalsh(H0)))
#        print(sorted(np.linalg.eigvalsh(H1)))
#        print('****************************')

    # elif pwrhop == False and dim == 2:
    #     jmat = np.diagflat(np.concatenate([[Jxx for i in range(L-1)]+[0] for j in range(L)])[0:-1], 1)+np.diagflat([Jxx for i in range(n-L)], L)
    #     jlist = np.concatenate(list(map(lambda x: np.diag(jmat, k = x), range(1,L**2))))
    # elif pwrhop == False and dim == 3:
    #     jmat = np.diagflat(np.concatenate([[Jxx for i in range(L-1)]+[0] for j in range(L**2)])[0:-1], 1)+np.diagflat([Jxx for i in range(n-L)], L)+np.diagflat([Jxx for i in range(n-L**2)], L**2)
    #     jlist = np.concatenate(list(map(lambda x: np.diag(jmat, k = x), range(1,L**3))))


    return H0+V0

def H2_spin_init(n,d,J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False,dsymm='charge'):
    H2_spin_up = Hinit(n,d,J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False)

    if dsymm == 'charge':
        H2_spin_down = H2_spin_up
    elif dsymm == 'spin':
        H2_spin_down = H2_spin_up
        for i in range(len(H2_spin_down)):
            H2_spin_down[i,i] = -H2_spin_up[i,i]
    elif dsymm == 'random':
        H2_spin_up = Hinit(n,d,J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False)

    return H2_spin_up,H2_spin_down
    

def Hint_init(n,delta,pwrint=False,beta=0):
    # Interaction tensors
    Hint = np.zeros((n,n,n,n),dtype=np.float32)
    for i in range(n):
        for j in range(i,n):
            if pwrint == False:
                if abs(i-j)==1:
                    # Initialise nearest-neighbour interactions
                    Hint[i,i,j,j] = 0.5*delta
            else:
                k = np.abs(i-j)
                if pwrint == 'random' and i != j:
                    # Pick interactions randomly from a distribution with standard deviation decaying like a power-law with distance
                    Hint[i,i,j,j] = 0.5*delta*np.random.normal(0,k**(-beta))
                elif i != j:
                    # Use an interaction strength that decays uniformly like a power-law with distance
                    Hint[i,i,j,j] = 0.5*delta*k**(-beta)
            Hint[j,j,i,i] = Hint[i,i,j,j]
    
    # Initialise off-diagonal quartic tensor (empty)
    Vint = np.zeros((n,n,n,n),dtype=np.float32)

    # elif intr == True and dim ==2:
    #     Dmat = np.diagflat(np.concatenate([[Jz for i in range(L-1)]+[0] for j in range(L)])[0:-1], 1)+np.diagflat([Jz for i in range(n-L)], L)
    #     dlist = np.concatenate(list(map(lambda x: np.diag(Dmat, k = x), range(1,L**2))))
    # elif intr == True and dim ==3:
    #     Dmat = np.diagflat(np.concatenate([[Jz for i in range(L-1)]+[0] for j in range(L**2)])[0:-1], 1)+np.diagflat([Jz for i in range(n-L)], L)+np.diagflat([Jz for i in range(n-L**2)], L**2)
    #     dlist = np.concatenate(list(map(lambda x: np.diag(Dmat, k = x), range(1,L**3))))

    return Hint+Vint

def H4_spin_init(n,delta_up=0,delta_down=0,delta_updown=0,delta_onsite=0,delta_mixed=0):

    # Interaction tensors
    Hint_up = np.zeros((n,n,n,n),dtype=np.float32)
    Hint_down = np.zeros((n,n,n,n),dtype=np.float32)
    Hint_updown = np.zeros((n,n,n,n),dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if abs(i-j)==1:
                # Initialise nearest-neighbour interactions
                Hint_up[i,i,j,j] = 0.5*delta_up
                Hint_down[i,i,j,j] = 0.5*delta_down
                Hint_updown[i,i,j,j] = 0.5*delta_updown
        Hint_updown [i,i,i,i] = delta_onsite #*np.random.uniform(0,1)
    
    # Initialise off-diagonal quartic tensor (empty)
    # Vint_up = np.zeros((n,n,n,n),dtype=np.float32)
    # Vint_down = np.zeros((n,n,n,n),dtype=np.float32)
    # Vint_updown = np.zeros((n,n,n,n),dtype=np.float32)

    return Hint_up,Hint_down,Hint_updown
 
def namevar(dis_type,dyn,norm,n,LIOM,species):
    if norm == True:
        nm = 'NO'
    else:
        nm = 'PT'
    if species == 'spinful fermion':
        spec = 'spin_fermion'
    elif species == 'spinless fermion':
        spec = 'fermion'
     # Make directory to store data
    if dyn == False:
        if not os.path.exists('spec/data/%s/%s/%s/%s/dataN%s' %(dis_type,nm,LIOM,'static',n)):
            os.makedirs('spec/data/%s/%s/%s/%s/dataN%s' %(dis_type,nm,LIOM,'static',n))
        namevar = 'spec/data/%s/%s/%s/%s/dataN%s' %(dis_type,nm,LIOM,'static',n) 
    elif dyn == True:
        if not os.path.exists('spec/data/%s/%s/%s/%s/dataN%s' %(dis_type,nm,LIOM,'dyn',n)):
            os.makedirs('spec/data/%s/%s/%s/%s/dataN%s' %(dis_type,nm,LIOM,'dyn',n))
        namevar = 'spec/data/%s/%s/%s/%s/dataN%s' %(dis_type,nm,LIOM,'dyn',n)
    
    return namevar


class hamiltonian:
    def __init__(self,species,dis_type,intr):
        self.species = species
        self.dis_type = dis_type
        # self.n = []
        # self.dim = []

        # self.H2_spinless = []
        # self.H4_spinless = []

        # self.H2_spinup = []
        # self.H2_spindown = []
        # self.H4_spinup = []
        # self.H4_spindown = []
        # self.H4_mixed = []
        self.intr = intr

    def build(self,n,dim,d,J,dis_type,delta=0,delta_up=0,delta_down=0,delta_mixed=0,delta_onsite=0,alpha=0,beta=0,dsymm='charge'):
        self.n = n
        self.dim = dim

        if self.species == 'spinless fermion':
            self.d = d
            self.J = J
            self.H2_spinless = Hinit(n,d,J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False)
            if self.intr == True:
                self.delta = delta
                self.H4_spinless = Hint_init(n,delta,pwrint=False,beta=0)

        elif self.species == 'spinful fermion':
            self.d = d
            self.J = J
            self.dsymm = dsymm
            H2up,H2dn = H2_spin_init(n,d,J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False,dsymm='charge')
            self.H2_spinup = H2up
            self.H2_spindown = H2dn
            if self.intr == True:

                self.delta_onsite = delta_onsite
                self.delta_up = delta_up 
                self.delta_down = delta_down 
                self.delta_mixed = delta_mixed
                H4up,H4dn,H4updn = H4_spin_init(n,delta_onsite=delta_onsite,delta_up=delta_up,delta_down=delta_down,delta_mixed=delta_mixed)
                self.H4_spinup = H4up
                self.H4_spindown = H4dn
                self.H4_mixed = H4updn

        elif self.species == 'boson':
            self.d = d
            self.J = J
        elif self.species =='hard core boson':
            self.d  = d

    
