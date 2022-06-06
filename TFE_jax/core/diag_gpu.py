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

This file contains all of the GPU code used to construct the RHS of the flow equations using matrix/tensor contractions 
and numerically integrate the flow equation to obtain a diagonal Hamiltonian.

This GPU implementation uses PyTorch to handle the tensors on the GPU.

"""

import os
from multiprocessing import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count())) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count())) # set number of MKL threads to run in parallel

import numpy as np
#import cupy as np
import matplotlib.pyplot as plt
from datetime import datetime
# from dynamics import dyn_con2,dyn_mf,dyn_exact
from numba import jit,prange
import copy, gc
import torch
from .contract_torch import contract
# from scipy.integrate import ode
from torchdiffeq import odeint as ode
from torchdiffeq import odeint_event
# Import ED code from QuSpin
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.tools.measurements import ED_state_vs_time
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spin basis

#------------------------------------------------------------------------------  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'

def nonint_ode(l,H,method='einsum'):
        # H = y.reshape(n,n)
        # n = H[1]
        H0 = torch.diag(torch.diag(H))
        V0 = H - H0
        eta = contract(H0,V0,method=method)
        sol = contract(eta,H,method=method)

        return sol
        
def nonint_ode_nloc(l,y,method='tensordot'):
        
        # H = y.reshape(n,n)
        # n = H[1]
        H = y[0]
        n = y[1]
        
        H0 = torch.diag(torch.diag(H))
        V0 = H - H0
        eta = contract(H0,V0,method=method,eta=True)
        sol = contract(eta,H,method=method)
        nsol  = contract(eta,n,method=method)

        return torch.stack([sol,nsol])
    
    
def int_ode_nloc(l,y,method='einsum'):

        n = list(y.size())[-1]
        H = y[0,0]
        Hint=y[n:2*n]
        n2 = y[2*n,0]
        nint = y[3*n:]
        
        H0 = torch.diag(torch.diag(H))
        V0 = H - H0
        
        Hint0 = torch.zeros(Hint.shape,dtype=torch.float32,device=device)
        # Hint0 = Hint0.to(device)
        # n=len(H)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                    Hint0[i,i,j,j] = Hint[i,i,j,j]
                    Hint0[i,j,j,i] = Hint[i,j,j,i]
        Vint = Hint-Hint0
        Vint = Vint.to(device)
        
        eta0 = contract(H0,V0,method=method)
        eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H0,Vint,method=method,eta=True)
        
        sol = contract(eta0,H,method=method)
        sol2 = contract(eta_int,H0+V0,method=method) + contract(eta0,Hint,method=method)
        
        nsol  = contract(eta0,n2,method=method)
        nsol2 = contract(eta_int,n2,method=method) + contract(eta0,nint,method=method)
        
        soln = torch.zeros((n,n,n,n),dtype=torch.float32,device=device)
        soln[0,0] = sol
        nsoln = torch.zeros((n,n,n,n),dtype=torch.float32,device=device)
        nsol2 = torch.zeros((n,n,n,n),dtype=torch.float32,device=device)
        nsoln[0,0] = nsol
        
        # print((torch.cat([torch.cat([soln,sol2]),torch.cat([nsoln,nsol2])],axis=0)).shape)
        del(eta0,eta_int,Hint0,Vint)
        
        return torch.cat([torch.cat([soln,sol2]),torch.cat([nsoln,nsol2])],axis=0)


def int_ode(l,y,n,method='jit'):

        H = y[0:n**2]
        H = H.reshape(n,n)
        H0 = np.diag(np.diag(H))
        V0 = H - H0

        Hint = y[n**2::]
        Hint = Hint.reshape(n,n,n,n)
        Hint0 = np.zeros((n,n,n,n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                    Hint0[i,i,j,j] = Hint[i,i,j,j]
                    Hint0[i,j,j,i] = Hint[i,j,j,i]
        Vint = Hint-Hint0
                     
        
        eta0 = contract(H0,V0,method=method,eta=True)
        eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H0,Vint,method=method,eta=True)
   
        sol = contract(eta0,H0+V0,method=method)
        sol2 = contract(eta_int,H0+V0,method=method) + contract(eta0,Hint,method=method)
        
        sol0 = np.zeros(n**2+n**4)
        sol0[:n**2] = sol.reshape(n**2)
        sol0[n**2:] = sol2.reshape(n**4)

        # return np.concatenate((sol.reshape(n**2), sol2.reshape(n**4)))
        return sol0


def liom_ode(l,nlist,y,n,method='jit',comp=False):

            H = y[0:n**2]
            H = H.reshape(n,n)
            H0 = np.diag(np.diag(H))
            V0 = H - H0
            
            if len(y) > n**2:
                Hint = y[n**2::]
                Hint = Hint.reshape(n,n,n,n)
                Hint0 = np.zeros((n,n,n,n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                            Hint0[i,i,j,j] = Hint[i,i,j,j]
                            Hint0[i,j,j,i] = Hint[i,j,j,i]
                Vint = Hint-Hint0
                       
            n2 = nlist[0:n**2]
            n2 = n2.reshape(n,n)

            if len(nlist)>n**2:
                nint = nlist[n**2::]
                nint = nint.reshape(n,n,n,n)
                         
            eta0 = contract(H0,V0,method=method,comp=False,eta=True)
            sol = contract(eta0,n2,method=method,comp=comp)
            if comp == False:
                sol0 = np.zeros(len(y))
            elif comp == True:
                sol0 = np.zeros(len(y),dtype=complex)

            sol0[:n**2] = sol.reshape(n**2)
            
            if len(y) > n**2:
                eta_int = contract(Hint0,V0,method=method,comp=comp,eta=True) + contract(H0,Vint,method=method,comp=comp,eta=True)
                sol2 = contract(eta_int,n2,method=method,comp=comp) + contract(eta0,nint,method=method,comp=comp)
                sol0[n**2:] = sol2.reshape(n**4)

            # return np.concatenate((sol.reshape(n**2), sol2.reshape(n**4)))
            return sol0
#------------------------------------------------------------------------------  

def flow_static(n,J,H0,V0,dl_list,qmax,cutoff,method='jit'):
      
    # FE Diag (CPU)
        print('***********')
        startTime = datetime.now()

        dl_list = torch.tensor(dl_list,device=device)
        # dl_list=dl_list.to(device)
        
        init_liom = torch.zeros((n,n),dtype=torch.float32,device=device)
        init_liom[n//2,n//2] = 1.0
        # init_liom = init_liom.to(device)

        # sol = ode(nonint_ode_nloc,torch.stack([(H0+V0),init_liom]),dl_list)
        def event(t,y):
            H = y[0]
            # print(H)
            V = H-torch.diag(torch.diag(H))
            # V2 = torch.round(V,decimals=10)
            # V2 = V2[V2 != 0.]
            # print(torch.max(torch.abs(V)))
            if torch.max(torch.abs(V))<cutoff:
                return torch.tensor(0,device=device)
            else:
                return torch.tensor(1,device=device)
            
        # def event2(t,y):
        #     H = y[0]
        #     # print(H)
            
        #     V = H-torch.diag(torch.diag(H))
        #     # print(torch.max(torch.abs(V)))
        #     Vflat=V.flatten()
        #     var = 0
        #     zc = 0
        #     for v in Vflat:
        #         if v < cutoff:
        #             var += 1
        #         if v != 0:
        #             zc += 1
        #     if t>1 and var/zc > 0.99:
        #     # if torch.max(torch.abs(V))<cutoff:
        #         return torch.tensor(0,device=device)
        #     else:
        #         return torch.tensor(1,device=device)
            
        _,sol = odeint_event(nonint_ode_nloc,torch.stack([(H0+V0),init_liom]),dl_list[0], event_fn=event, reverse_time=False, odeint_interface=ode)

        print(sol[-1])
 
        # print('Time for flow diag: ', datetime.now()-startTime)

        return [torch.Tensor.numpy(torch.Tensor.cpu(sol[-1,0])),torch.Tensor.numpy(torch.Tensor.cpu(sol[-1,1]))]

def flow_static_int_torch(n,J,H0,V0,Hint,Vint,dl_list,qmax,cutoff,method='jit'):
      
    # FE Diag (CPU)
        print('***********')
        startTime = datetime.now()

        dl_list = torch.tensor(dl_list)
        dl_list=dl_list.to(device)
        
        # print(dl_list)

        init_liom = torch.zeros((n,n),dtype=torch.float32,device=device)
        init_liom4 = torch.zeros((n,n,n,n),dtype=torch.float32,device=device)
        init_liom[n//2,n//2] = 1.0
        # init_liom = init_liom.to(device)

        H = torch.zeros((n,n,n,n),dtype=torch.float32,device=device)
        H[0,0] = H0+V0
        init_l = torch.zeros((n,n,n,n),dtype=torch.float32,device=device)
        init_l[0,0] = init_liom
        
        # print(torch.cat([H,Hint]))
        # print(torch.cat([H,Hint]).shape)
        # print(torch.cat([H,Hint])[0,0])
        # print((torch.cat([torch.cat([H,Hint]),torch.cat([init_l,init_liom4])])).shape)

        # sol = ode(int_ode_nloc,torch.cat([torch.cat([H,Hint]),torch.cat([init_l,init_liom4])],axis=0),dl_list)
        # sol = ode(int_ode_nloc,[(H0+V0),(Hint+Vint),init_liom,init_liom4],dl_list)
        # print(sol[-1])
        
        def event(t,y):
            H = y[0,0]
            # print(H)
            V = H-torch.diag(torch.diag(H))
            # V2 = torch.round(V,decimals=10)
            # V2 = V2[V2 != 0.]
            # print(torch.max(torch.abs(V)))
            if torch.max(torch.abs(V))<cutoff:
                return torch.tensor(0,device=device)
            else:
                return torch.tensor(1,device=device)
            
        et,sol = odeint_event(int_ode_nloc,torch.cat([torch.cat([H,Hint]),torch.cat([init_l,init_liom4])],axis=0),dl_list[0], event_fn=event, reverse_time=False, odeint_interface=ode)

        print('et',et)
        
        
        # dl_list = np.logspace(np.log10(0.01), np.log10(et),qmax,endpoint=True,base=10)
        # dl_list = torch.tensor(dl_list)
        # dl_list=dl_list.to(device)
        
        # sol = ode(int_ode_nloc,torch.cat([torch.cat([H,Hint]),torch.cat([init_l,init_liom4])],axis=0),dl_list)
        
        #print((torch.Tensor.numpy(torch.Tensor.cpu(sol))).shape)
        
        # for i in range(n):
        #     for j in range(n):
        #         plt.plot(dl_list,torch.Tensor.numpy(torch.Tensor.cpu(sol[:,0,0,i,j])),linewidth='1',marker='x')
        # # plt.plot()
        # plt.show()
        # plt.close()
 
        # print('Time for flow diag: ', datetime.now()-startTime)
        
        H0_diag = torch.Tensor.numpy(torch.Tensor.cpu(sol[-1,0,0]))
        print(H0_diag)
        Hint = torch.Tensor.numpy(torch.Tensor.cpu(sol[-1,n:2*n]))
        n2 = torch.Tensor.numpy(torch.Tensor.cpu(sol[-1,2*n,0]))
        nint = torch.Tensor.numpy(torch.Tensor.cpu(sol[-1,3*n:]))
        
        liom=np.concatenate((n2.reshape(n**2),nint.reshape(n**4)))
        
        HFint = np.zeros(n**2).reshape(n,n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    HFint[i,j] = Hint[i,i,j,j]
                    HFint[i,j] += -Hint[i,j,j,i]
        lbits = np.zeros(n-1)
        for q in range(1,n):
            lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))

        #plt.plot(lbits,marker='o',linewidth='2')
        #plt.show()
        #plt.close()
        
        #plt.plot(np.diag(n2),color='r',marker='o',linewidth='2')
        #plt.show()
        #plt.close()
        
        # return [torch.Tensor.numpy(torch.Tensor.cpu(sol[-1,0])),torch.Tensor.numpy(torch.Tensor.cpu(sol[-1,1]))]    
    
        return([H0_diag,Hint,lbits,np.diag(n2),liom])

def flow_static_int(n,J,H0,V0,Hint,Vint,dl_list,qmax,cutoff,method='jit'):
      
    # FE Diag (CPU)
        # print('***********')
        startTime = datetime.now()
     
        sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
        # print('Memory64 required: MB', sol_int.nbytes/10**6)
        # sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
        # print('Memory32 required: MB', sol_int.nbytes/10**6)
        
        r_int = ode(int_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        r_int.set_f_params(n,method)
        
        init = np.zeros(n**2+n**4,dtype=np.float32)
        init[:n**2] = ((H0+V0)).reshape(n**2)
        init[n**2:] = (Hint+Vint).reshape(n**4)
        
        r_int.set_initial_value(init,dl_list[0])
        sol_int[0] = init
        
        k = 1
        J0 = 10.
        while r_int.successful() and k < qmax-1 and J0 > cutoff:
            r_int.integrate(dl_list[k])
            # sim = proc(r_int.y,n,cutoff)
            # sol_int[k] = sim
            sol_int[k] = r_int.y
            mat = sol_int[k,0:n**2].reshape(n,n)
            off_diag = mat-np.diag(np.diag(mat))
            J0 = max(off_diag.reshape(n**2))
            # print(k,J0)
 
            k += 1
        print(k,J0)   
        sol_int=sol_int[:k-1]
        dl_list = dl_list[:k-1]
        # print(k,J0)
        # print('eigenvalues',np.sort(np.diag(sol_int[k-1,0:n**2].reshape(n,n))))
        
        # print('Time for flow diag: ', datetime.now()-startTime)
        
        #for i in range(n**2):
        #    plt.plot(dl_list,sol_int[::,i])
        #    plt.plot(dl_list,sol_int[::,i],'o')
        #plt.show()
        #plt.close()
        
        #for i in range(n**2,n**4):
        #    plt.plot(dl_list,sol_int[::,i])
        #    plt.plot(dl_list,sol_int[::,i],'x')
        #plt.show()
        #plt.close()
        
        H0_diag = sol_int[-1,:n**2].reshape(n,n)
        
        Hint = sol_int[-1,n**2::].reshape(n,n,n,n)                        
        HFint = np.zeros(n**2).reshape(n,n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    HFint[i,j] = Hint[i,i,j,j]
                    HFint[i,j] += -Hint[i,j,j,i]
        # HFint=np.asnumpy(HFint)
        
        print(HFint)
 
        lbits = np.zeros(n-1)
        for q in range(1,n):
            lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))
         
        print(lbits)
        plt.plot(lbits)
        plt.plot(lbits,'o')
        plt.show()
        plt.close()
        
        startTime2 = datetime.now()

        liom = np.zeros((k,n**2+n**4),dtype=np.float32)
        init_liom = np.zeros((n,n))
        init_liom[n//2,n//2] = 1.0
        liom[0,:n**2] = init_liom.reshape(n**2)
        
        dl_list = dl_list[::-1]
        n_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        n_int.set_initial_value(liom[0],dl_list[0])
        k0=1
        while n_int.successful() and k0 < k-1:
            n_int.set_f_params(sol_int[-k0],n,method)
            n_int.integrate(dl_list[k0])
            liom[k0] = n_int.y
            k0 += 1

        
        # print('LIOM flow time: ', datetime.now()-startTime2)
        
        central = (liom[k0-1,:n**2]).reshape(n,n)
        
        # plt.plot(np.diag(central))
        # plt.yscale('log')
        # plt.show()
        # plt.close()
        del sol_int
        gc.collect()
        return([H0_diag,Hint,lbits,central,liom[k0-1]])
    
    
def flow_dyn(n,J,H0,V0,Hint,Vint,num,num_int,dl_list,qmax,cutoff,tlist,LIOM,method='jit'):
      
    # FE Diag (CPU)
        print('***********')
        startTime = datetime.now()
 
        sol = np.zeros((qmax,n**2),dtype=np.float64)
        r = ode(nonint_ode).set_integrator('dopri5', nsteps=100)
        r.set_initial_value((H0+V0).reshape(n**2),dl_list[0])
        r.set_f_params(n,method)
        sol[0] = (H0+V0).reshape(n**2)

        k = 1
        J0 = 10.
        while r.successful() and k < qmax-1 and J0 > cutoff:
            r.integrate(dl_list[k])
            sol[k] = r.y
            mat = sol[k,0:n**2].reshape(n,n)
            off_diag = mat-np.diag(np.diag(mat))
            J0 = max(off_diag.reshape(n**2))
            print(k,J0)
            k += 1
        print('eigenvalues',np.sort(np.diag(sol[k-1].reshape(n,n))))
        dl_list=dl_list[:k-1]
        
        H0final = sol[k-1,:n**2].reshape(n,n)
        print('Time for flow diag: ', datetime.now()-startTime)
        
        startTime2 = datetime.now()
        
        #liom = np.zeros((qmax,n**2))
        init_liom = np.zeros((n,n))
        init_liom[n//2,n//2] = 1.0
        #liom[0,:n**2] = init_liom.reshape(n**2)
        
        dl_list = dl_list[::-1]
        n_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
        n_int.set_initial_value(init_liom.reshape(n**2),dl_list[0])
        k0=1
        while n_int.successful() and k0 < k:
            # print(k0,k)
            n_int.set_f_params(sol[-k0],n,method)
            n_int.integrate(dl_list[k0])
            #liom[k0] = n_int.y
            k0 += 1
            
        liom = n_int.y
        print('LIOM flow time: ', datetime.now()-startTime2)
        
        central = liom.reshape(n,n)
        
        num_init = np.zeros((n,n))
        num_init[n//2,n//2] = 1.0
        
        dl_list = dl_list[::-1] # Invert dl again back to original
        num_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
        num_int.set_initial_value(init_liom.reshape(n**2),dl_list[0])
        k0=1
        num=np.zeros((k,n**2+n**4))
        while num_int.successful() and k0 < len(dl_list[:k]):
            num_int.set_f_params(sol[-k0],n,method)
            num_int.integrate(dl_list[k0])
            num[k0] = num_int.y
            k0 += 1
        # num = num_int.y
        
        for i in range(n**2):
            plt.plot(dl_list,liom[:k0,i])
            plt.plot(dl_list,liom[:k0,i],'o')
        plt.show()
        plt.close()
        
        for i in range(n**2,n**4):
            plt.plot(dl_list,liom[:k0,i])
            plt.plot(dl_list,liom[:k0,i],'x')
        plt.show()
        plt.close()
        
        num = num_int.y
        
        # Run non-equilibrium dynamics following a quench from CDW state
        # Returns answer *** in LIOM basis ***
        evolist = dyn_con2(n,num,sol[k-1],tlist,method=method)
        
        num_t_list = np.zeros((len(tlist),n**2))
        dl_list = dl_list[::-1] # Reverse dl for backwards flow
        for t0 in range(len(tlist)):
            
            num_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
            num_int.set_initial_value(evolist[t0],dl_list[0])
            k0=1
            
            while num_int.successful() and k0 < k:
                num_int.set_f_params(sol[len(dl_list[:k])-k0-1],n,method)
                num_int.integrate(dl_list[k0])
                # num_t[k0]=num_int.y
                k0 += 1
            num_t_list[t0] = num_int.y
            

            
        nlist = np.zeros(len(tlist))
#        nlist2 = np.zeros(len(tlist))
        # Set up initial state as a CDW
        list1 = np.array([1. for i in range(n//2)])
        list2 = np.array([0. for i in range(n//2)])
        state = np.array([val for pair in zip(list1,list2) for val in pair])
        
        n2list = num_t_list[::,:n**2]
        for t0 in range(len(tlist)):
            mat = n2list[t0].reshape(n,n)
            # phaseMF = 0.
            for i in range(n):
                nlist[t0] += (mat[i,i]*state[i]**2).real
#                nlist2[t0] += (mat[i,i]*state[i]**2).real
        
        return([H0final,central,liom[-1],nlist,tlist])
        # return([H0,Hint,num,nlist,tlist,n2list,lbits])
    
        
def flow_dyn_int(n,J,H0,V0,Hint,Vint,num,num_int,dl_list,qmax,cutoff,tlist,LIOM,method='jit'):
      
    # FE Diag (CPU)
        # print('***********')
        startTime = datetime.now()
  
        sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
        r_int = ode(int_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        # print('Memory64 required: MB', sol_int.nbytes/10**6)
        
        init = np.zeros(n**2+n**4,dtype=np.float32)
        init[:n**2] = ((H0+V0)).reshape(n**2)
        init[n**2:] = (Hint+Vint).reshape(n**4)
        
        r_int.set_initial_value(init,dl_list[0])
        r_int.set_f_params(n,method)
        sol_int[0] = init
        
        k = 1
        J0 = 10.
        while r_int.successful() and k < qmax-1 and J0 > cutoff:
            r_int.integrate(dl_list[k])
            sol_int[k] = r_int.y
            mat = sol_int[k,0:n**2].reshape(n,n)
            off_diag = mat-np.diag(np.diag(mat))
            J0 = max(off_diag.reshape(n**2))
            k += 1
        sol_int=sol_int[:k-1]
        print(k,J0)
        dl_list=dl_list[:k-1]
        # print('eigenvalues',np.sort(np.diag(sol_int[-1,0:n**2].reshape(n,n))))
        
        # for i in range(n**2):
        #     plt.plot(dl_list,sol_int[::,i])
        #     plt.plot(dl_list,sol_int[::,i],'o')
        # plt.show()
        # plt.close()
        
        # for i in range(n**2,n**4):
        #     plt.plot(dl_list,sol_int[::,i])
        #     plt.plot(dl_list,sol_int[::,i],'x')
        # plt.show()
        # plt.close()
        
        print('Time for flow diag: ', datetime.now()-startTime)
        H0final,Hintfinal = sol_int[-1,:n**2].reshape(n,n),sol_int[-1,n**2::].reshape(n,n,n,n)
        
        Hint = sol_int[-1,n**2::].reshape(n,n,n,n)                        
        HFint = np.zeros(n**2,dtype=np.float32).reshape(n,n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    HFint[i,j] = Hint[i,i,j,j]
                    HFint[i,j] += -Hint[i,j,j,i]
        # HFint=np.asnumpy(HFint)
 
        lbits = np.zeros(n-1)
        for q in range(1,n):
            lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))
        
        startTime2 = datetime.now()
        
        
        liom = np.zeros((k,n**2+n**4),dtype=np.float32)
        init_liom = np.zeros((n,n),dtype=np.float32)
        init_liom[n//2,n//2] = 1.0
        liom[0,:n**2] = init_liom.reshape(n**2)
        
        dl_list = dl_list[::-1]
        n_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        n_int.set_initial_value(liom[0],dl_list[0])
        k0=1
        while n_int.successful() and k0 < k-1:
            n_int.set_f_params(sol_int[-k0],n,method)
            n_int.integrate(dl_list[k0])
            liom[k0] = n_int.y
            k0 += 1
            
        print('LIOM flow time: ', datetime.now()-startTime2)
        
        central = (liom[k0-1,:n**2]).reshape(n,n)
        
        # plt.plot(np.diag(central),'ro')
        # plt.plot(np.diag(central),'r-')
        
        # startTime3= datetime.now()
        
        num = np.zeros((k,n**2+n**4),dtype=np.float32)
        num_init = np.zeros((n,n))
        num_init[n//2,n//2] = 1.0
        num[0,0:n**2] = num_init.reshape(n**2)
        
        dl_list = dl_list[::-1] # Invert dl again back to original
        num_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        num_int.set_initial_value(liom[0],dl_list[0])
        k0=1
        # num_t=np.zeros((k,n**2+n**4))
        while num_int.successful() and k0 < k-1:
            num_int.set_f_params(sol_int[k0],n,method)
            num_int.integrate(dl_list[k0])
            # num_t[k0]=num_int.y
            # liom[k0] = num_int.y
            k0 += 1
        num = num_int.y
        
        # for i in range(n**2):
        #     plt.plot(dl_list[:k0-1],num_t[:k0-1,i])
        #     plt.plot(dl_list[:k0-1],num_t[:k0-1,i],'o')
        # plt.title(r'fwd n2')
        # plt.show()
        # plt.close()
        
        # for i in range(n**2,n**4):
        #     plt.plot(dl_list[:k0-1],num_t[:k0-1,i])
        #     plt.plot(dl_list[:k0-1],num_t[:k0-1,i],'x')
        # plt.title(r'fwd n4')
        # plt.show()
        # plt.close()
        
        # print('n flow forward time',datetime.now()-startTime3)
        
        # central2 = (num[:n**2]).reshape(n,n)
        # plt.plot(np.diag(central2),'bx')
        # plt.plot(np.diag(central2),'b--')
        # plt.yscale('log')
        # plt.show()
        # plt.close()
        
        # Set up initial state as a CDW
        list1 = np.array([1. for i in range(n//2)])
        list2 = np.array([0. for i in range(n//2)])
        state = np.array([val for pair in zip(list1,list2) for val in pair])
        
        dl_list = dl_list[::-1] # Reverse dl for backwards flow
        
        # Run non-equilibrium dynamics following a quench from CDW state
        # Returns answer *** in LIOM basis ***
        # num_t_list = np.zeros((len(tlist),n**2))
        # evolist = dyn_mf(n,num,sol_int[-1],tlist,state,method=method)
        # print(evolist.shape)

        # for t0 in range(len(tlist)):
            
        #     num_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        #     num_int.set_initial_value(evolist[t0,:n**2],dl_list[0])
        #     k0=1
        #     while num_int.successful() and k0 < len(dl_list[:k]):
        #         num_int.set_f_params(sol_int[-k0,:n**2],n,method)
        #         num_int.integrate(dl_list[k0])
        #         k0 += 1
        #     num_t_list[t0] = (num_int.y)[:n**2]
            
        nlist = np.zeros(len(tlist),dtype=np.float32)
        # n2list = num_t_list[::,:n**2]
        # for t0 in range(len(tlist)):
        #     mat = n2list[t0].reshape(n,n)
        #     for i in range(n):
        #         nlist[t0] += (mat[i,i]*state[i]).real
        
        evolist2 = dyn_exact(n,num,sol_int[-1],tlist,method=method)
        # evolist2=evolist2.real
        
        startTime4 = datetime.now()
        
        num_t_list2 = np.zeros((len(tlist),n**2+n**4),dtype=np.complex64)
        # num_t_list2 = np.zeros((len(tlist),n**2+n**4))
        for t0 in range(len(tlist)):
            
            num_int = ode(liom_ode).set_integrator('zvode',nsteps=50,atol=10**(-8),rtol=10**(-8))
            num_int.set_initial_value(evolist2[t0],dl_list[0])
            k0=1
            # num_t=np.zeros((k,n**2+n**4))
            while num_int.successful() and k0 < k-1:
                num_int.set_f_params(sol_int[-k0],n,method,True)
                num_int.integrate(dl_list[k0])
                # print(k0)
                # num_t[k0]=num_int.y
                k0 += 1
            num_t_list2[t0] = num_int.y
            
            # for i in range(n**2):
            #     plt.plot(dl_list,num_t[:k0,i])
            #     plt.plot(dl_list,num_t[:k0,i],'o')
            # plt.title(r'bck n2')
            # plt.show()
            # plt.close()
            
            # for i in range(n**2,n**4):
            #     plt.plot(dl_list,num_t[:k0,i])
            #     plt.plot(dl_list,num_t[:k0,i],'x')
            # plt.title(r'bck n4')
            # plt.show()
            # plt.close()
            
        # nlist = np.zeros(len(tlist))
        nlist2 = np.zeros(len(tlist))
        print('n flow reverse time',datetime.now()-startTime4)
        
        n2list = num_t_list2[::,:n**2]
        n4list = num_t_list2[::,n**2:]
        for t0 in range(len(tlist)):
            mat = n2list[t0].reshape(n,n)
            mat4 = n4list[t0].reshape(n,n,n,n)
            # phaseMF = 0.
            for i in range(n):
                # nlist[t0] += (mat[i,i]*state[i]).real
                nlist2[t0] += (mat[i,i]*state[i]).real
                for j in range(n):
                    if i != j:
                        nlist2[t0] += (mat4[i,i,j,j]*state[i]*state[j]).real
                        nlist2[t0] += -(mat4[i,j,j,i]*state[i]*state[j]).real
      
        # del sol_int,r_int,evolist2,num,num_int
        # gc.collect()
        
        return([H0final,central,liom[-1],Hintfinal,lbits,nlist2,nlist,tlist])
        # return([H0,Hint,num,nlist,tlist,n2list,lbits])
 
    
def flow_dyn_int2(n,J,H0,V0,Hint,Vint,num,num_int,dl_list,qmax,cutoff,tlist,LIOM,method='jit'):
      
    # FE Diag (CPU)
        print('***********')
        startTime = datetime.now()
  
        sol_int = np.zeros((qmax,n**2+n**4),dtype=np.float32)
        print('Memory required: MB', sol_int.nbytes/10**6)
        r_int = ode(int_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        
        init = np.zeros(n**2+n**4,dtype=np.float32)
        init[:n**2] = ((H0+V0)).reshape(n**2)
        init[n**2:] = (Hint+Vint).reshape(n**4)
        
        r_int.set_initial_value(init,dl_list[0])
        r_int.set_f_params(n,method)
        sol_int[0] = init
        
        k = 1
        J0 = 10.
        while r_int.successful() and k < qmax-1 and J0 > cutoff:
            r_int.integrate(dl_list[k])
            sol_int[k] = r_int.y
            mat = sol_int[k,0:n**2].reshape(n,n)
            off_diag = mat-np.diag(np.diag(mat))
            J0 = max(off_diag.reshape(n**2))
            # print(k,J0)
            k += 1
        sol_int=sol_int[:k]
        dl_list=dl_list[:k]
        print('Max step: ',k, 'Max J: ', J0)
            
        print('eigenvalues',np.sort(np.diag(sol_int[k-1,0:n**2].reshape(n,n))))
        
        print('Time for flow diag: ', datetime.now()-startTime)
        H0final,Hintfinal = sol_int[k-1,:n**2].reshape(n,n),sol_int[k-1,n**2::].reshape(n,n,n,n)
        
        Hint = sol_int[k-1,n**2::].reshape(n,n,n,n)                        
        HFint = np.zeros(n**2).reshape(n,n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    HFint[i,j] = Hint[i,i,j,j]
                    HFint[i,j] += -Hint[i,j,j,i]
        # HFint=np.asnumpy(HFint)
 
        lbits = np.zeros(n-1)
        for q in range(1,n):
            lbits[q-1] = np.median(np.log10(np.abs(np.diag(HFint,q)+np.diag(HFint,-q))/2.))
  
        startTime2 = datetime.now()
        
        
        liom = np.zeros((k,n**2+n**4),dtype=np.float32)
        init_liom = np.zeros((n,n),dtype=np.float32)
        init_liom[n//2,n//2] = 1.0
        liom[0,:n**2] = init_liom.reshape(n**2)
        
        dl_list = dl_list[::-1]
        n_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        n_int.set_initial_value(liom[0],dl_list[0])
        k0=1
        while n_int.successful() and k0 < k-1:
            n_int.set_f_params(sol_int[-k0],n,method)
            n_int.integrate(dl_list[k0])
            liom[k0] = n_int.y
            k0 += 1
            
        print('LIOM flow time: ', datetime.now()-startTime2)
        
        central = (liom[k0-1,:n**2]).reshape(n,n)
        # plt.plot(np.diag(central))
        # plt.yscale('log')
        # plt.show()
        # plt.close()
        
        # Set up initial state as a CDW
        list1 = np.array([1. for i in range(n//2)])
        list2 = np.array([0. for i in range(n//2)])
        state = np.array([val for pair in zip(list1,list2) for val in pair])
        
        imblist = np.zeros((n,len(tlist)))
        imblist2 = np.zeros((n,len(tlist)))
        for site in range(n):
            num = np.zeros((k,n**2+n**4))
            num_init = np.zeros((n,n),dtype=np.float32)
            num_init[site,site] = 1.0

            num[0,0:n**2] = num_init.reshape(n**2)
            
            dl_list = dl_list[::-1] # Invert dl again back to original
            num_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
            num_int.set_initial_value(num[0],dl_list[0])
            k0=1
            while num_int.successful() and k0 < k-1:
                num_int.set_f_params(sol_int[k0],n,method)
                num_int.integrate(dl_list[k0])
                # liom[k0] = num_int.y
                k0 += 1
            num = num_int.y
            
            # Run non-equilibrium dynamics following a quench from CDW state
            # Returns answer *** in LIOM basis ***
            evolist2 = dyn_exact(n,num,sol_int[-1],tlist,method=method)
            # evolist2=evolist2.real
            dl_list = dl_list[::-1] # Reverse the flow
            
            num_t_list2 = np.zeros((len(tlist),n**2+n**4),dtype=np.complex64)
            # num_t_list2 = np.zeros((len(tlist),n**2+n**4))
            for t0 in range(len(tlist)):
                
                num_int = ode(liom_ode).set_integrator('zvode',nsteps=50,atol=10**(-8),rtol=10**(-8))
                num_int.set_initial_value(evolist2[t0],dl_list[0])
                k0=1
                while num_int.successful() and k0 < k-1:
                    num_int.set_f_params(sol_int[-k0],n,method,True)
                    num_int.integrate(dl_list[k0])
                    # print(k0)
                    k0 += 1
                num_t_list2[t0] = num_int.y
                
            nlist = np.zeros(len(tlist))
            nlist2 = np.zeros(len(tlist))
            
            
            n2list = num_t_list2[::,:n**2]
            n4list = num_t_list2[::,n**2:]
            for t0 in range(len(tlist)):
                mat = n2list[t0].reshape(n,n)
                mat4 = n4list[t0].reshape(n,n,n,n)
                # phaseMF = 0.
                for i in range(n):
                    # nlist[t0] += (mat[i,i]*state[i]**2).real
                    nlist[t0] += (mat[i,i]*state[i]).real
                    nlist2[t0] += (mat[i,i]*state[i]).real
                    for j in range(n):
                        if i != j:
                            nlist[t0] += (mat4[i,i,j,j]*state[i]*state[j]).real
                            nlist[t0] += -(mat4[i,j,j,i]*state[i]*state[j]).real
                            
            imblist[site] = ((-1)**site)*nlist/n
            imblist2[site] = ((-1)**site)*nlist2/n

        imblist = 2*np.sum(imblist,axis=0)
        imblist2 = 2*np.sum(imblist2,axis=0)
    
        return([H0final,central,liom[k0-1],Hintfinal,lbits,imblist2,imblist,tlist])
        # return([H0,Hint,num,nlist,tlist,n2list,lbits])
 
#------------------------------------------------------------------------------
# Function for benchmarking the non-interacting system using 'einsum'
def flow_einsum_nonint(H0,V0,dl):
      
        startTime = datetime.now()
        q = 0
        while np.max(np.abs(V0))>10**(-2):
        
            # Non-interacting generator
            eta0 = np.einsum('ij,jk->ik',H0,V0) - np.einsum('ki,ij->kj',V0,H0,optimize=True)
            
            # Flow of non-interacting terms
            dH0 = np.einsum('ij,jk->ik',eta0,(H0+V0)) - np.einsum('ki,ij->kj',(H0+V0),eta0,optimize=True)
        
            # Update non-interacting terms
            H0 = H0+dl*np.diag(np.diag(dH0))
            V0 = V0 + dl*(dH0-np.diag(np.diag(dH0)))

            q += 1

        print('***********')
        print('FE time - CPU',datetime.now()-startTime)
        print('Max off diagonal element: ', np.max(np.abs(V0)))
        print(np.sort(np.diag(H0)))

#------------------------------------------------------------------------------  
# Function for benchmarking the non-interacting system using 'tensordot'
def flow_tensordot_nonint(H0,V0,dl):     

    startTime = datetime.now()
    q = 0
    while np.max(np.abs(V0))>10**(-3):
    
        # Non-interacting generator
        eta = np.tensordot(H0,V0,axes=1) - np.tensordot(V0,H0,axes=1)
        
        # Flow of non-interacting terms
        dH0 = np.tensordot(eta,H0+V0,axes=1) - np.tensordot(H0+V0,eta,axes=1)
    
        # Update non-interacting terms
        H0 = H0+dl*np.diag(np.diag(dH0))
        V0 = V0 + dl*(dH0-np.diag(np.diag(dH0)))

        q += 1
        
    print('***********')
    print('FE time - Tensordot',datetime.now()-startTime)
    print('Max off diagonal element: ', np.max(np.abs(V0)))
    print(np.sort(np.diag(H0)))

#------------------------------------------------------------------------------
        
def flow_levels(n,array,intr,dyn):
    
    # H0 = array[0]
    H0 = array[0]
    H0 = np.array(H0)
    Hint=array[1]
    Hint = np.array(Hint)
    #     n2 = y[2*n,0]
    #     nint = y[3*n:]
    # if intr == True and dyn ==True:
    #     Hint = array[3]
    # elif intr ==True and dyn == False:
    #     Hint=array[1]
    flevels = np.zeros(2**n)

    for i in range(2**n):
        lev0 = bin(i)[2::].rjust(n,'0') # Generate the many-body states
        # Compute the energies of each state from the fixed point Hamiltonian
        for j in range(n):
            flevels[i] += H0[j,j]*int(lev0[j])
            if intr == True:
                for q in range(n):
                    if q !=j:
                        # flevels[i] += Hint[j,j,q,q]*int(lev0[j])
                        flevels[i] += Hint[j,j,q,q]*int(lev0[j])*int(lev0[q]) 
                        flevels[i] += -Hint[j,q,q,j]*int(lev0[j])*int(lev0[q]) 
    
    return np.sort(flevels)

#------------------------------------------------------------------------------

# Compute averaged level spacing ratio
def level_stat(levels):
    
    list1 = np.zeros(len(levels))
    lsr = 0.
    for i in range(1,len(levels)):
        list1[i-1] = levels[i] - levels[i-1]
    for j in range(len(levels)-2):
        lsr += min(list1[j],list1[j+1])/max(list1[j],list1[j+1])
    lsr *= 1/(len(levels)-2)
    
    return lsr
