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

If you do use any of this code, please cite https://arxiv.org/abs/2110.02906.

---------------------------------------------

This file contains the main code used to set up and run the flow equation method,
and save the output as an HDF5 file containing various different datasets.

"""

import os, sys
from psutil import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of MKL threads to run in parallel
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"                         # Necessary on some versions of OS X
os.environ['KMP_WARNINGS'] = 'off'                                # Silence non-critical warning

# JAX options - must be set BEFORE importing the JAX library
# os.environ['CUDA_VISIBLE_DEVIES'] = '2'                         # Set which device to use ('' is CPU)
os.environ['JAX_ENABLE_X64'] = 'true'                           # Enable 64-bit floats in JAX
# from jax.config import config
# config.update('jax_disable_jit', True)                          # Disable JIT compilation in JAX for debugging
# config.update("jax_enable_x64", True)                           # Alternate way to enable float64 in JAX

import jax.numpy as jnp 
import numpy as np
from scipy.special import jv as jv
from datetime import datetime
import h5py,gc
import core.diag as diag
import models.models as models
import core.utility as utility
from ED.ed import ED

import matplotlib.pyplot as plt
# Part to change plotting system
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,6)
plt.rc('font',family='serif')
plt.rcParams.update({'font.size': 24})
plt.rc('text', usetex=True)
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
#------------------------------------------------------------------------------  
# Parameters
L = int(sys.argv[1])            # Linear system size
dim = 2                         # Spatial dimension
n = L**dim                      # Total number of sites
species = 'spinless fermion'    # Type of particle
dsymm = 'spin'                  # Type of disorder (spinful fermions only)
Ulist = [0.1]
# List of interaction strengths
J = 1.0                         # Nearest-neighbour hopping amplitude
cutoff = J*10**(-3)             # Cutoff for the off-diagonal elements to be considered zero
dis = [0.7+0.02*i for i in range(26)]    
dis = [5.]                
# List of disorder strengths
lmax = 75                       # Flow time max
qmax = 750                      # Max number of flow time steps
reps = 1                        # Number of disorder realisations
norm = False                    # Normal-ordering, can be true or false
no_state = 'SDW'                # State to use for normal-ordering, can be CDW or SDW
                                # For vacuum normal-ordering, just set norm=False
ladder = False                  # TEST FEATURE: compute LIOMs using creation/annihilation operators
ITC = False                     # Infinite temp correlation function (TEST PARAMETER)
Hflow = True                    # Whether to store the flowing Hamiltonian (true) or generator (false)
                                # Storing H(l) allows SciPy ODE integration to add extra flow time steps
                                # Storing eta(l) reduces number of tensor contractions, at cost of accuracy
                                # NB: if the flow step dl is too large, this can lead to Vint diverging!
# precision = np.float64        # Precision with which to store running Hamiltonian/generator
                                # Default throughout is double precision (np.float64)
                                # Using np.float16 will half the memory cost, at loss of precision
                                # Only affects the backwards transform, not the forward transform
method = str(sys.argv[3])       # Method for computing tensor contractions
                                # Options are 'einsum', 'tensordot','jit' or 'vec'
                                # In general 'tensordot' is fastest for small systems, 'jit' for large systems
                                # (Note that 'jit' requires compilation on the first run, increasing run time.)
print('Norm = %s' %norm)
intr = True                     # Turn on/off interactions
dyn = False                     # Run the dynamics
imbalance = True                # Sets whether to compute global imbalance or single-site dynamics
LIOM = 'bck'                    # Compute LIOMs with forward ('fwd') or backward ('bck') flow
                                # Forward uses less memory by a factor of qmax, and transforms a local operator
                                # in the initial basis into the diagonal basis; backward does the reverse
dyn_MF = True                   # Mean-field decoupling for dynamics (used only if dyn=True)
logflow = True                  # Use logarithmically spaced steps in flow time
store_flow = True               # Store the full flow of the Hamiltonian and LIOMs
dis_type = str(sys.argv[2])     # Options: 'random', 'QPgolden', 'QPsilver', 'QPbronze', 'QPrandom', 'linear', 'curved', 'prime'
                                # Also contains 'test' and 'QPtest', potentials that do not change from run to run
xlist = [1.]
# For 'dis_type = curved', controls the gradient of the curvature
if intr == False:               # Zero the interactions if set to False (for ED comparison and filename)
    delta = 0
if dis_type != 'curved':
    xlist = [0.0]
if (species == 'spinless fermion' and n > 12) or (species == 'spinful fermion' and n > 6) or qmax > 2000:
    print('SETTING store_flow = False DUE TO TOO MANY VARIABLES AND/OR FLOW TIME STEPS')
    store_flow = False

# Define list of timesteps for non-equilibrium dynamics
# Only used if 'dyn = True'
tlist = [0.01*i for i in range(31)]

# Make directory to store data
nvar = utility.namevar(dis_type,dsymm,no_state,dyn,norm,n,LIOM,species)

if Hflow == False:
    print('*** Warning: Setting Hflow=False requires small flow time steps in order for backwards transform to be accurate. ***')
if intr == False and norm == True:
    print('Normal ordering is only for interacting systems.')
    norm = False
if norm == True and n%2 != 0:
    print('Normal ordering is only for even system sizes')
    norm = False
# if species == 'spinful fermion' and norm == True:
#     print('Normal ordering not implemented for spinful fermions.')
#     norm = False
#==============================================================================
# Run program
#==============================================================================

if __name__ == '__main__': 

    startTime = datetime.now()
    print('Start time: ', startTime)

    for p in range(reps):
        for x in xlist:
            for d in dis:
                # lmax *= 1/d
                print(d)
                print(lmax)
                for delta in Ulist:

                    # Create dictionary of parameters to pass to functions; avoids having to have too many function args
                    params = {"n":n,"delta":delta,"J":J,"cutoff":cutoff,"dis":dis,"dsymm":dsymm,"NO_state":no_state,"lmax":lmax,"qmax":qmax,
                                "reps":reps,"norm":norm,"Hflow":Hflow,"method":method, "intr":intr,"dyn":dyn,"imbalance":imbalance,"species":species,
                                    "LIOM":LIOM, "dyn_MF":dyn_MF,"logflow":logflow,"dis_type":dis_type,"x":x,"tlist":tlist,"store_flow":store_flow,"ITC":ITC,"ladder":ladder}

                    #-----------------------------------------------------------------
                    # Initialise Hamiltonian
                    ham = models.hamiltonian(species,dis_type,intr=intr)
                    if species == 'spinless fermion':
                        ham.build(n,dim,d,J,x,delta=delta)
                    elif species == 'spinful fermion':
                        ham.build(n,dim,d,J,x,delta_onsite=delta,delta_up=0.,delta_down=0.,dsymm=dsymm)

                    print(ham.H2_spinless)
                    print((ham.H2_spinless).shape)
                    # Initialise the number operator on the central lattice site
                    num = jnp.zeros((n,n))
                    num = num.at[n//2,n//2].set(1.0)
                    
                    # Initialise higher-order parts of number operator (empty)
                    num_int=jnp.zeros((n,n,n,n),dtype=jnp.float64)
                    
                    #-----------------------------------------------------------------

                    # Diag non-interacting system w/NumPy
                    # print(ham.H2_spinless)
                    startTime = datetime.now()
                    print(jnp.sort(jnp.linalg.eigvalsh(ham.H2_spinless)))
                    # print('NumPy diag time',datetime.now()-startTime)

                    #-----------------------------------------------------------------

                    # Diagonalise with flow equations
                    flow = diag.CUT(params,ham,num,num_int)

                    bessel = jnp.zeros(n)
                    for i in range(n):
                        bessel = bessel.at[i].set(jnp.log10(jnp.abs(jv(jnp.abs(i-n//2),2/dis[0])**2)))
                    plt.plot(bessel,'x--')
                    plt.show()
                    plt.close()

                    print('Time after flow finishes: ',datetime.now()-startTime)

                    if species == 'spinless fermion':
                        ncut = 12
                    elif species == 'spinful fermion':
                        ncut = 6

                    # Diagonalise with ED
                    if n <= ncut and dyn == True:
                        ed=ED(n,ham,tlist,dyn,imbalance)
                        ed_dyn=ed[1]
                    elif n <= ncut and dyn == False:
                        ed=ED(n,ham,np.ones(2),dyn,imbalance)
                    else:
                        ed = np.zeros(n)
                    print('Time after ED: ',datetime.now()-startTime)

                    if intr == False or n <= ncut:
                        if species == 'spinless fermion':
                            flevels = utility.flow_levels(n,flow,intr)
                        elif species == 'spinful fermion':
                            
                            flevels = utility.flow_levels_spin(n,flow,intr)
                        flevels = flevels-np.median(flevels)
                        ed = ed[0] - np.median(ed[0])
                    
                    else:
                        flevels=np.zeros(n)
                        ed=np.zeros(n)

                    if intr == False or n <= ncut:
                        lsr = utility.level_stat(flevels)
                        lsr2 = utility.level_stat(ed)

                        errlist = np.zeros(len(ed))
                        for i in range(len(ed)):
                            if np.round(ed[i],10)!=0.:
                                errlist[i] = np.abs((ed[i]-flevels[i])/ed[i])

                        print('***** ERROR *****: ', np.median(errlist))  

                    if dyn == True:
                        plt.plot(tlist,ed_dyn,label=r'ED')
                        if imbalance == True:
                            plt.plot(tlist,flow["Imbalance"],'o')
                            plt.ylabel(r'$\mathcal{I}(t)$')
                        else:
                            plt.plot(tlist,flow["Density Dynamics"],'o',label='Flow')
                            plt.ylabel(r'$\langle n_i(t) \rangle$')
                        plt.xlabel(r'$t$')
                        plt.legend()
                        plt.show()
                        plt.close()

                    print(flow["LIOM2"])
                    print(flow["LIOM2_FWD"])

                    #==============================================================
                    # Export data   
                    with h5py.File('%s/tflow-d%.2f-x%.2f-Jz%.2f-p%s.h5' %(nvar,d,x,delta,p),'w') as hf:
                        hf.create_dataset('params',data=str(params))

                        hf.create_dataset('H2_diag',data=flow["H0_diag"])
                        if species == 'spinless fermion':
                            hf.create_dataset('H2_initial',data=ham.H2_spinless)
                        elif species == 'spinful fermion':
                            hf.create_dataset('H2_up',data=ham.H2_spinup)
                            hf.create_dataset('H2_dn',data=ham.H2_spindown)

                        if n <= ncut:
                            hf.create_dataset('flevels', data = flevels,compression='gzip', compression_opts=9)
                            hf.create_dataset('ed', data = ed, compression='gzip', compression_opts=9)
                            hf.create_dataset('lsr', data = [lsr,lsr2])
                            hf.create_dataset('err',data = errlist)
                            if store_flow == True:
                                hf.create_dataset('flow2',data=flow["flow2"])
                                hf.create_dataset('flow4',data=flow["flow4"])
                                hf.create_dataset('dl_list',data=flow["dl_list"])
                        if intr == True:
                                hf.create_dataset('lbits', data = flow["LIOM Interactions"])
                                hf.create_dataset('Hint', data = flow["Hint"], compression='gzip', compression_opts=9)
                                hf.create_dataset('liom2', data = flow["LIOM2"], compression='gzip', compression_opts=9)
                                hf.create_dataset('liom4', data = flow["LIOM4"], compression='gzip', compression_opts=9)
                                hf.create_dataset('liom2_fwd', data = flow["LIOM2_FWD"], compression='gzip', compression_opts=9)
                                hf.create_dataset('liom4_fwd', data = flow["LIOM4_FWD"], compression='gzip', compression_opts=9)
                                hf.create_dataset('inv',data=flow["Invariant"])
                        if dyn == True:
                            hf.create_dataset('tlist', data = tlist)
                            if imbalance == True:
                                hf.create_dataset('imbalance',data=flow["Imbalance"])
                            else:
                                hf.create_dataset('flow_dyn', data = flow["Density Dynamics"])
                            if n <= ncut:
                                hf.create_dataset('ed_dyn', data = ed_dyn)
                                    
                gc.collect()
                print('****************')
                print('Time taken for one run:',datetime.now()-startTime)
                print('****************')
