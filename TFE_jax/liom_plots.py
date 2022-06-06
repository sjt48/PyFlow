import os
from multiprocessing import cpu_count
# Set up threading options
os.environ['OMP_NUM_THREADS']= str(int(cpu_count())) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count())) # set number of MKL threads to run in parallel
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

import numpy as np
import matplotlib.pyplot as plt
import h5py,cycler
from datetime import datetime

# Part to change plotting system
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (16,8)
plt.rc('font',family='serif')
plt.rcParams.update({'font.size': 24})
plt.rc('text', usetex=True)
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
nc = 21
color = plt.cm.viridis(np.linspace(0, 1,nc))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
#------------------------------------------------------------------------------  
# Parameters
n = 12                          # System size
delta = 0.1                     # Nearest-neighbour interaction strength
J = 1.0                         # Nearest-neighbour hopping
cutoff = J*10**(-6)             # Cutoff for the off-diagonal elements to be considered zero
dis = [0.7+0.05*i for i in range(1,11)]                # List of disorder strengths
reps = 1                        # Number of disorder realisations
dis_type = 'linear'             # Options: 'random' or 'QP' (quasiperiodic)

if dis_type == 'curved':
    x=0.1
else:
    x=0.0

# Make directory to store data
if not os.path.exists('%s/PT/static/dataN%s' %(dis_type,n)):
        os.makedirs('%s/PT/static/dataN%s' %(dis_type,n))
        
#==============================================================================
# Run program
#==============================================================================

if __name__ == '__main__': 

    # Initialize figure axes
    fig, axes = plt.subplots(2,2)
    ax1,ax2 = axes[0]
    ax3,ax4 = axes[1]

    # Initialize lists for the f2 and f4 coefficients
    f2list=np.zeros(len(dis))
    f4list=np.zeros(len(dis))

    # Set up counter and run loop over 'disorder' strength
    dcount = 0
    for d in dis:

        # Set up arrays for fixed-point interactions (lbit_list) and 
        lbit_list = np.zeros((reps,n-1))
        liom_list = np.zeros((reps,n))
        l4_list = np.zeros((reps,n-1))

        for p in range(reps):

            with h5py.File('%s/PT/static/dataN%s/tflow-d%.2f-x%.2f-Jz%.2f-p%s.h5' %(dis_type,n,d,x,delta,p),'r') as hf:
                # Get datasets from the HDF5 file
                lbits=np.array(hf.get('lbits'))             # Fixed-point interactions
                liom=np.array(hf.get('liom'))               # Quadratic part of LIOM (same as non-interacting system)
                # liom_all=np.array(hf.get('liom_all'))       # Entire LIOM including quartic terms

            # Initialize lists for the quadratic and quartic parts of the LIOM (l2 and l4 respectively)
            l2=liom[:n**2].reshape(n,n)
            l4=(liom[n**2::]).reshape(n,n,n,n)

            # Create matrix 'mat' for diagonal part of l4 (the Gamma_ij term in my notation)
            mat = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    mat[i,j] += l4[i,i,j,j]
                    mat[i,j] += -l4[i,j,j,i]

            # Average the n_j*n_j coefficient over distance |i-j|
            # As we discussed, this might not be a very insightful thing to do...!
            vlist=np.zeros(n-1)
            for i in range(1,n):
                vlist[i-1] = np.mean(np.abs(np.diag(mat,i)))

            # Add the computed quantities into the arrays for averaging
            lbit_list[p] = lbits
            liom_list[p] = np.diag(liom[:n**2].reshape(n,n))
            l4_list[p] = vlist

        # Plot various quantities
        ax1.plot(np.mean(lbit_list,axis=0))
        ax2.plot(np.log10(np.abs(np.mean(liom_list,axis=0))))
        ax3.plot(np.log10(np.abs(np.mean(l4_list,axis=0))))


        dcount += 1

    # Computes f2 and f4, for a loop over system sizes L
    for L in [8,10,12]:
        dcount = 0
        dis = [0.7+0.01*i for i in range(1,51)] 

        f2list=np.zeros(len(dis))
        f4list=np.zeros(len(dis))
        for d in dis:
            with h5py.File('%s/PT/static/dataN%s/tflow-d%.2f-x%.2f-Jz%.2f-p%s.h5' %(dis_type,L,d,x,delta,p),'r') as hf:
                liom_all=np.array(hf.get('liom'))
            squared = [i**2 for i in liom_all]
            norm = np.sum(squared)

            # f2 is the sum of the square of the quadratic terms
            f2 = np.sum(squared[:L**2])/norm
            # f2 is the sum of the square of the quartic terms
            f4 = np.sum(squared[L**2::])/norm

            f2list[dcount]=f2
            f4list[dcount]=f4
            dcount += 1

        ax4.plot(dis,f2list,'k-',alpha=L/12)
        ax4.plot(dis,f4list,'r--',alpha=L/12)

    ax1.set(xlabel=r'$r=|i-j|$',ylabel=r'$\log_{10} \overline{\Delta_{ij}}$')
    ax2.set(xlabel=r'$j$',ylabel=r'$\log_{10} \overline{\alpha_j{(L/2)}}$')
    ax3.set(xlabel=r'$r=|i-j|$',ylabel=r'$\log_{10} \overline{\Gamma_{ij}}$')
    ax4.set(xlabel=r'$W/J$',ylabel=r'$f_n$')
    ax4.text(0.75,0.85,r'$f_4$',color='red')
    ax4.text(0.75,0.2,r'$f_2$',color='black')
    plt.subplots_adjust(hspace=0.5,wspace=0.25)
    plt.savefig('WS_Linear.pdf',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()
