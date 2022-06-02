import os,sys
from multiprocessing import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count())) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count())) # set number of MKL threads to run in parallel
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import h5py,gc
import torch
import core.diag_gpu as diag
from ED.ed2 import ED
from scipy.special import jv as jv

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
n = int(sys.argv[1])                          # System size
delta = 0.1                     # Nearest-neighbour interaction strength
J = 1.0                         # Nearest-neighbour hopping
cutoff = J*10**(-3)             # Cutoff for the off-diagonal elements to be considered zero
dis = [0.7+0.02*i for i in range(26)]
W = 1.0
                                # List of disorder strengths
reps = 1                      # Number of disorder realisations
intr = True                     # Turn on/off interactions

lmax = 25                       # Flow time max
qmax = 500                      # Max number of flow time steps

logflow = False                  # Use logarithmically spaced steps in flow time
                                # PROTOTYPE FEATURE
dis_type = 'linear'                 # Options: 'random' or 'QP' (quasiperiodic)


# params=[n,delta,J,cutoff,dis,lmax,qmax,intr,dyn,LIOM,logflow,dis_type,tlist]
#==============================================================================
# Run program
#==============================================================================

if __name__ == '__main__': 
#     #with np.cuda.Device(0):
        print('Start time: ', datetime.now())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = 'cpu'
        # device = torch.device('cuda:1')
        #device = torch.device('cpu')
        print(device)
        
        # Make directory to store data
        if not os.path.exists('%s/%s/dataN%s' %(device,dis_type,n)):
            os.makedirs('%s/%s/dataN%s' %(device,dis_type,n))
        
        if device =='cuda':
            del model
            gc.collect()
            torch.cuda.empty_cache()
# def main():  
        for p in range(reps):
            for d in dis:
                np.random.seed()               
                startTime = datetime.now()
                
                if logflow == False:
                    dl = np.linspace(0,lmax,qmax,endpoint=True)
                elif logflow == True:
                    print('Warning: careful choices of qmax and lmax required for log flow.')
                    dl = np.logspace(np.log10(0.01), np.log10(lmax),qmax,endpoint=True,base=10)
 
                #-----------------------------------------------------------------
                # Non-interacting matrices
                H0 = np.zeros((n,n),dtype=np.float32)
                if dis_type == 'random':
                    for i in range(n):
                        # Initialise Hamiltonian with random on-site terms
                        H0[i,i] = np.random.uniform(-d,d)
                elif dis_type == 'linear':
                    for i in range(n):
                        H0[i,i] = i*d
                elif dis_type == 'linear_dis':
                    for i in range(n):
                        H0[i,i] = i*d+np.random.uniform(-W,W)
                elif dis_type == 'linear_dis_inv_symm':
                    dlist = np.random.uniform(-W,W,n//2)
                    dlist2 = -1*dlist[::-1]
                    dlist3 = np.concatenate(dlist,[0],dlist2)
                    for i in range(n):
                        H0[i,i] = i*d
                elif dis_type == 'QP':
                    phase = np.random.uniform(-np.pi,np.pi)
                    print('phase = ', phase)
                    #print('**** FIXED PHASE FOR TESTING - DISABLE FOR REAL DATA ****')
                    #phase=0.
                    #phase=p*2*np.pi/10
                    print('phase = %.2f' %phase)
                    phi = (1.+np.sqrt(5.))/2.
                    for i in range(n):
                        # Initialise Hamiltonian with quasiperiodic on-site terms
                        H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
                elif dis_type == 'QPsilver':
                    phase = np.random.uniform(-np.pi,np.pi)
                    phi = (1+np.sqrt(2.))
                    for i in range(n):
                        H0[i,i] = d*np.cos(2*np.pi*(1./phi)*i + phase)
                # Initialise V0 with nearest-neighbour hopping
                V0 = np.diag(J*np.ones(n-1,dtype=np.float32),1) + np.diag(J*np.ones(n-1,dtype=np.float32),-1)
                
                # Initialise the number operator on the central lattice site
                num = torch.zeros((n,n),device=device)
                num[n//2,n//2] = 1.0

                #-----------------------------------------------------------------
                
                # Interaction tensors
                Hint = torch.zeros((n,n,n,n),dtype=torch.float64,device=device)

                for i in range(n):
                    for j in range(n):
                        if abs(i-j)==1:
                            # Initialise nearest-neighbour interactions
                            Hint[i,i,j,j] = 0.5*delta
                
                # Initialise off-diagonal quartic tensor (empty)
                Vint = torch.zeros((n,n,n,n),dtype=torch.float64,device=device)
                
                # Initialise higher-order parts of number operator (empty)
                num_int=torch.zeros((n,n,n,n),dtype=torch.float64,device=device)
                
                #-----------------------------------------------------------------
                
                # Diag non-interacting system w/NumPy
                startTime = datetime.now()
                print(np.sort(np.linalg.eigvalsh(H0+V0)))
                print('NumPy diag time',datetime.now()-startTime)

                # Torch all tensors and move to GPU if available
                H0 = torch.from_numpy(H0)
                H0 = H0.to(device)
                V0 = torch.from_numpy(V0)
                V0 = V0.to(device)
                # Hint = torch.from_numpy(Hint,device=device)
                # Hint = Hint.to(device)
                # Vint = torch.from_numpy(Vint,device=device)
                # Vint = Vint.to(device)
                
                #-----------------------------------------------------------------
                # Interacting systems
                
                # Diagonalise with flow equations
                # Possible methods are: einsum, tensordot, jit, vec
                if intr == True:
                    flow = diag.flow_static_int_torch(n,J,H0,V0,Hint,Vint,dl,qmax,cutoff,method='einsum')
                elif intr == False:
                    flow = diag.flow_static(n,J,H0,V0,dl,qmax,cutoff,method='einsum')

                # bessel = np.zeros(n)
                # for i in range(n):
                #     bessel[i] = np.log10(np.abs(jv(np.abs(i-n//2),2)**2))
                # plt.plot(bessel,'x--')
                # plt.show()
                # plt.close()


                print('Time after flow finishes: ',datetime.now()-startTime)
                runtime = datetime.now()-startTime
                # Diagonalise with ED
                H0=torch.Tensor.numpy(torch.Tensor.cpu(H0))
                if n <= 12:
                    ed=ED(n,H0,J,delta,np.ones(2),False,False)
                else:
                    ed = np.zeros(n)
                print('Time after ED: ',datetime.now()-startTime)

                #-----------------------------------------------------------------
                # Plots

                if n <= 12:
                    flevels = diag.flow_levels(n,flow,intr,False)
                    flevels = flevels-np.median(flevels)
                    ed = ed[0] - np.median(ed[0])
                else:
                    flevels=np.zeros(n)
                    ed=np.zeros(n)
                
                if n <= 12:
                    lsr = diag.level_stat(flevels)
                    lsr2 = diag.level_stat(ed)
                    print(lsr,lsr2)

                    errlist = np.zeros(2**n)
                    for i in range(2**n):
                        errlist[i] = np.abs((ed[i]-flevels[i])/ed[i])
                    print('***** ERROR *****: ', np.median(errlist))     
                    
                #==============================================================
                # Export data
                
                with h5py.File('%s/%s/dataN%s/tflow-d%.2f-Jz%.2f-p%s-cutoff%.2f.h5' %(device,dis_type,n,d,delta,p,-np.log10(cutoff)),'w') as hf:
                     print('%s/%s/dataN%s/tflow-d%.2f-Jz%.2f-p%s-cutoff%.2f.h5' %(device,dis_type,n,d,delta,p,-np.log10(cutoff)))
                         # hf.create_dataset('params',data=params)
                     hf.create_dataset('H0',data=flow[0])
                     hf.create_dataset('time',data=[runtime.total_seconds()])
                     if n <= 12:
                         hf.create_dataset('flevels', data = flevels,compression='gzip', compression_opts=9)
                         hf.create_dataset('ed', data = ed, compression='gzip', compression_opts=9)
                         hf.create_dataset('lsr', data = [lsr,lsr2])
                         hf.create_dataset('err',data=errlist)
                     if intr == True:
                         hf.create_dataset('lbits', data = flow[2])
                         hf.create_dataset('Hint', data = flow[1], compression='gzip', compression_opts=9)
                         hf.create_dataset('liom', data = flow[3])
                         hf.create_dataset('liom_all_fwd', data = flow[4], compression='gzip', compression_opts=9)
                         hf.create_dataset('liom_all_bck', data = flow[5], compression='gzip', compression_opts=9)
                    
      
            gc.collect()
            print('****************')
            print('Time taken for one run:',datetime.now()-startTime)
            print('****************')
