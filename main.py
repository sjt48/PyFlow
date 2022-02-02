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
import numpy as np 
from datetime import datetime
import h5py,gc
import core.diag as diag
import core.init as init
from ED.ed import ED
from datetime import datetime

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
n = int(sys.argv[1])            # System size
delta = 0.1                     # Nearest-neighbour interaction strength
J = 1.0                         # Nearest-neighbour hopping amplitude
cutoff = J*10**(-6)             # Cutoff for the off-diagonal elements to be considered zero
dis = [1.0]                    
# List of disorder strengths
lmax = 1500                     # Flow time max
qmax = 1000                     # Max number of flow time steps
reps = 1                        # Number of disorder realisations
norm = False                    # Normal-ordering, can be true or false
Hflow = True                    # Whether to store the flowing Hamiltonian (true) or generator (false)
                                # Storing H(l) allows SciPy ODE integration to add extra flow time steps
                                # Storing eta(l) reduces number of tensor contractions, at cost of accuracy
                                # NB: if the flow step dl is too large, this can lead to Vint diverging!
precision = np.float32          # Precision with which to store running Hamiltonian/generator
                                # Default throughout is single precision (np.float32)
                                # Using np.float16 will half the memory cost, at loss of precision
                                # Only affects the backwards transform, not the forward transform
method = 'tensordot'            # Method for computing tensor contractions
                                # Options are 'einsum', 'tensordot','jit' or 'vec'
                                # In general 'tensordot' is fastest for small systems, 'jit' for large systems
                                # (Note that 'jit' requires compilation on the first run, increasing run time.)
print('Norm = %s' %norm)
intr = True                     # Turn on/off interactions
dyn = False                     # Run the dynamics
imbalance = False               # Sets whether to compute global imbalance or single-site dynamics
LIOM = 'bck'                    # Compute LIOMs with forward ('fwd') or backward ('bck') flow
                                # Forward uses less memory by a factor of qmax, and transforms a local operator
                                # in the initial basis into the diagonal basis; backward does the reverse
dyn_MF = True                   # Mean-field decoupling for dynamics (used only if dyn=True)
logflow = True                  # Use logarithmically spaced steps in flow time
store_flow = True               # Store the full flow of the Hamiltonian and LIOMs
dis_type = str(sys.argv[2])     # Options: 'random', 'QPgolden', 'QPsilver', 'QPbronze', 'QPrandom', 'linear', 'curved', 'prime'
                                # Also contains 'test' and 'QPtest', potentials that do not change from run to run
x = 0.1                         # For 'dis_type = curved', controls the gradient of the curvature
if intr == False:               # Zero the interactions if set to False (for ED comparison and filename)
    delta = 0
if dis_type != 'curved':
    x = 0.0
if n > 12 or qmax > 2000:
    store_flow = False

# Define list of timesteps for non-equilibrium dynamics
# Only used if 'dyn = True'
tlist = [0.01*i for i in range(51)]

# Create dictionary of parameters to pass to functions; avoids having to have too many function args
params = {"n":n,"delta":delta,"J":J,"cutoff":cutoff,"dis":dis,"lmax":lmax,"qmax":qmax,"reps":reps,"norm":norm,
            "Hflow":Hflow,"precision":precision,"method":method, "intr":intr,"dyn":dyn,"imbalance":imbalance,
                "LIOM":LIOM, "dyn_MF":dyn_MF,"logflow":logflow,"dis_type":dis_type,"x":x,"tlist":tlist,"store_flow":store_flow}

# Make directory to store data
nvar = init.namevar(dis_type,dyn,norm,n,LIOM)

if Hflow == False:
    print('*** Warning: Setting Hflow=False requires small flow time steps in order for backwards transform to be accurate. ***')
if intr == False and norm == True:
    print('Normal ordering is only for interacting systems.')
    norm = False
#==============================================================================
# Run program
#==============================================================================

if __name__ == '__main__': 

    startTime = datetime.now()
    print('Start time: ', startTime)

    for p in range(reps):
        for d in dis:
            
            #-----------------------------------------------------------------
            # Non-interacting matrices
            H0,V0 = init.Hinit(n,d,J,dis_type,x)
            
            # Initialise the number operator on the central lattice site
            num = np.zeros((n,n))
            num[n//2,n//2] = 1.0

            #-----------------------------------------------------------------
            # Interaction tensors
            Hint,Vint = init.Hint_init(n,delta)
            
            # Initialise higher-order parts of number operator (empty)
            num_int=np.zeros((n,n,n,n),dtype=np.float32)
            
            #-----------------------------------------------------------------
            
            # Diag non-interacting system w/NumPy
            startTime = datetime.now()
            print(np.sort(np.linalg.eigvalsh(H0+V0)))
            print('NumPy diag time',datetime.now()-startTime)

            #-----------------------------------------------------------------

            # Diagonalise with flow equations
            flow = diag.CUT(params,H0,V0,Hint,Vint,num,num_int)
            
            print('Time after flow finishes: ',datetime.now()-startTime)
            print(np.sort(np.diag(flow["H0_diag"])))

            # Diagonalise with ED
            if n <= 12 and dyn == True:
                ed=ED(n,H0,J,delta,tlist,dyn,imbalance)
                ed_dyn=ed[1]
            elif n <= 12 and dyn == False:
                ed=ED(n,H0,J,delta,np.ones(2),dyn,imbalance)
                print('ED',ed[0])
            else:
                ed = np.zeros(n)
            print('Time after ED: ',datetime.now()-startTime)

            if n <= 12:
                flevels = diag.flow_levels(n,flow,intr)
                flevels = flevels-np.median(flevels)
                ed = ed[0] - np.median(ed[0])
            else:
                flevels=np.zeros(n)
                ed=np.zeros(n)

            if n <= 12:
                lsr = diag.level_stat(flevels)
                lsr2 = diag.level_stat(ed)

                errlist = np.zeros(len(ed))
                for i in range(len(ed)):
                    errlist[i] = np.abs((ed[i]-flevels[i])/ed[i])
                print('***** ERROR *****: ', np.mean(errlist))   

            if dyn == True:
                plt.plot(tlist,ed_dyn)
                if imbalance == True:
                    plt.plot(tlist,flow["Imbalance"],'o')
                else:
                    plt.plot(tlist,flow["Density Dynamics"],'o')
                plt.show()
                plt.close()

            #==============================================================
            # Export data   
            with h5py.File('%s/tflow-d%.2f-x%.2f-Jz%.2f-p%s.h5' %(nvar,d,x,delta,p),'w') as hf:
                hf.create_dataset('params',data=str(params))
                hf.create_dataset('H2_diag',data=flow["H0_diag"])
                hf.create_dataset('H2_initial',data=H0+V0)

                if n <= 12:
                    hf.create_dataset('flevels', data = flevels,compression='gzip', compression_opts=9)
                    hf.create_dataset('ed', data = ed, compression='gzip', compression_opts=9)
                    hf.create_dataset('lsr', data = [lsr,lsr2])
                    hf.create_dataset('err',data = errlist)
                    if store_flow == True:
                        hf.create_dataset('flow',data=flow["flow"])
                        hf.create_dataset('dl_list',data=flow["dl_list"])
                if intr == True:
                        hf.create_dataset('lbits', data = flow["LIOM Interactions"])
                        hf.create_dataset('Hint', data = flow["Hint"], compression='gzip', compression_opts=9)
                        hf.create_dataset('liom', data = flow["LIOM"], compression='gzip', compression_opts=9)
                        hf.create_dataset('inv',data=flow["Invariant"])
                if dyn == True:
                    hf.create_dataset('tlist', data = tlist)
                    if imbalance == True:
                        hf.create_dataset('imbalance',data=flow["Imbalance"])
                    else:
                        hf.create_dataset('flow_dyn', data = flow["Density Dynamics"])
                    if n <= 12:
                        hf.create_dataset('ed_dyn', data = ed_dyn)
                                
        gc.collect()
        print('****************')
        print('Time taken for one run:',datetime.now()-startTime)
        print('****************')
