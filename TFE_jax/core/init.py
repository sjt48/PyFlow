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
import copy

def Hinit(n,d,J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False,dim=1):
    """ Generate the non-interacting part of the Hamiltonian with the specified on-site potential. """

    np.random.seed()
    print('Choice of potential = %s' %dis_type)

    #-----------------------------------------------------------------
    # Non-interacting matrices
    H0 = np.zeros((n,n),dtype=np.float64)

    if isinstance(d,float) == False:
        H0 = np.diag(d)
    elif dis_type == 'random':
        for i in range(n):
            # Initialise Hamiltonian with random on-site terms
            H0[i,i] = np.random.uniform(-d,d)
    elif dis_type == 'test':
        # For testing purposes: fixed list of randomly generated numbers for method comparisons
        print('**** FIXED DISORDER REALISATION FOR TESTING - DISABLE FOR REAL DATA ****')
        randomlist = [2.4979359120556666, -4.477621238657079, -4.448810326437316, -3.6115452666436543, -1.2802110535766298, -3.336862075297363, -0.3370611440832194, -3.8232260796601523, -0.5134617674857918, 1.32895294857477]
        for i in range(n):
            H0[i,i] = randomlist[i%n]
    elif dis_type == 'test2':
        # H0 = np.diag([0.63305823,0.49472645,1.02878404,-0.1982988,-2.38908197,2.40333365,-1.705861,-0.67025341,0.72327997,-0.36466963,-1.48150466,0.95687038])
        # H0 = np.diag([ 0.00220783,0.17725875,-0.17469319,-0.34102253,-0.22480943,0.21774356,0.0138178,0.09863276,0.28596514,0.37551004,-0.3836448,0.47296222])
        # H0 = np.diag([0.53947236,-1.87679578,2.96251111,1.26385442,2.87516597,1.81604499,-2.5698877,2.68828105,-0.14110541,2.52083621,-2.47714679,2.79505097])
        # H0 = np.diag([0.20794954,-0.34591258,0.25465915,-0.37131532,0.47769051,0.26771804,-0.22449367,-0.17061238,0.05344497,0.04503425,0.47542292,-0.48663382])
        H0 = np.diag([-0.20509672,-0.48258169,0.07354666,0.4035225,-0.38249293,-0.02020861,0.29574013,-0.32896991,-0.05816845,-0.30851041,-0.35800805,-0.03625906])
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
            H0[i,i] = np.sqrt(prime(i+1))
        bw = np.max(np.diag(H0))-np.min(np.diag(H0))
        H0 *= 1/bw
        H0 *= d
        
    elif dis_type == 'QPgolden':
        phase = np.random.uniform(-np.pi,np.pi)
        phase2 = np.random.uniform(-np.pi,np.pi)
        print('phase = ', phase)
        phi = (1.+np.sqrt(5.))/2.
        if dim == 1:
            for i in range(n):
                # Initialise Hamiltonian with quasiperiodic on-site terms
                H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
        elif dim == 2:
            phi2 = 1+np.sqrt(2)
            L = int(np.sqrt(n))
            temp = np.zeros((L,L))
            for i in range(L):
                for j in range(L):
                    #temp[i,j] = d*(np.cos(2*np.pi*(i+j)/phi + phase) + np.cos(2*np.pi*(i-j)/phi + phase))
                    temp[i,j] = d*(np.cos(2*np.pi*(i)/phi + phase) + np.cos(2*np.pi*(j)/phi2 + phase2))
            H0 = np.diag(temp.reshape(n))

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
    if pwrhop == False and dim ==1:
        V0 = np.diag(J*np.ones(n-1,dtype=np.float32),1) + np.diag(J*np.ones(n-1,dtype=np.float32),-1)
    elif pwrhop == False and dim ==2:
        L = np.int(np.sqrt(n))
        jmat = np.diagflat(np.concatenate([[J for i in range(L-1)]+[0] for j in range(L)])[0:-1], 1)+np.diagflat([J for i in range(n-L)], L)
        V0 = jmat + jmat.T

    elif pwrhop == False and dim == 3:
        L = int(n**(1/3))
        jmat = np.diagflat(np.concatenate([[J for i in range(L-1)]+[0] for j in range(L**2)])[0:-1], 1)+np.diagflat([J for i in range(n-L)], L)+np.diagflat([J for i in range(n-L**2)], L**2)
        V0 = jmat + jmat.T
    elif pwrhop == True:
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

    print('str',isinstance(dsymm,str))
    if isinstance(dsymm,str) == False:
        hlist_up = dsymm[0]
        hlist_down = dsymm[1]
        print('hup',hlist_up)
        print('hdn',hlist_down)
        H2_spin_up = Hinit(n,hlist_up,-1*J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False)
        H2_spin_down = Hinit(n,hlist_down,-1*J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False)
    else:
        H2_spin_up = Hinit(n,d,-1*J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False)
        if dsymm == 'charge':
            H2_spin_down = H2_spin_up
        elif dsymm == 'spin':
            H2_spin_down = copy.deepcopy(H2_spin_up)
            for i in range(len(H2_spin_down)):
                H2_spin_down[i,i] = -H2_spin_up[i,i]
        elif dsymm == 'random':
            H2_spin_down = Hinit(n,d,-1*J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False)

    return H2_spin_up,H2_spin_down
    
def Hint_init(n,delta,pwrint=False,beta=0,dim=1,U=0):
    # Interaction tensors
    Hint = np.zeros((n,n,n,n),dtype=np.float32)

    if dim == 2:
        L = int(np.sqrt(n))
        Dmat = np.diagflat(np.concatenate([[delta for i in range(L-1)]+[0] for j in range(L)])[0:-1], 1)+np.diagflat([delta for i in range(n-L)], L)
        Dmat += Dmat.T
        Dmat *= 0.5
    if dim == 3:
        L = int(n**(1/3))
        Dmat = np.diagflat(np.concatenate([[delta for i in range(L-1)]+[0] for j in range(L**2)])[0:-1], 1)+np.diagflat([delta for i in range(n-L)], L)+np.diagflat([delta for i in range(n-L**2)], L**2)
        Dmat += Dmat.T
        Dmat *= 0.5
    for i in range(n):
        Hint[i,i,i,i] = U
        for j in range(i,n):
            if pwrint == False and dim == 1:
                if abs(i-j)==1:
                    # Initialise nearest-neighbour interactions
                    Hint[i,i,j,j] = 0.5*delta
            elif pwrint == False and (dim ==2 or dim ==3):
                Hint[i,i,j,j] = Dmat[i,j]
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
        Hint_updown [i,i,i,i] = delta_onsite

    return Hint_up,Hint_down,Hint_updown
