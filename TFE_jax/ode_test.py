from  jax.experimental.ode import odeint
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
import os

#==============================================================================
# Define functions
#==============================================================================

# General contraction function
def contract(A,B):
    """ General contract function: gets shape and calls appropriate contraction function. """
    if A.ndim == B.ndim == 2:
        con = con2(A,B)
    if A.ndim != B.ndim:
        if A.ndim == 4:
            if B.ndim == 2:
                con = con4(A,B)
        if A.ndim == 2:
            if B.ndim == 4:
                con = -1*con4(B,A)

    return con

# Contract square matrices (matrix multiplication)
def con2(A,B):
    return jnp.einsum('ij,jk->ik',A,B) - jnp.einsum('ki,ij->kj',B,A)

# Contract rank-4 tensor with square matrix
def con4(A,B):

    con = jnp.einsum('abcd,df->abcf',A,B) 
    con += -jnp.einsum('abcd,ec->abed',A,B)
    con += jnp.einsum('abcd,bf->afcd',A,B)
    con += -jnp.einsum('abcd,ea->ebcd',A,B)

    return con

def int_ode(y,l,index_list):
        
    # Extract various components of the Hamiltonian from the ijnput array 'y'
    H2 = y[0]                           # Define quadratic part of Hamiltonian
    n,_ = H2.shape
    H2_0 = jnp.diag(jnp.diag(H2))       # Define diagonal quadratic part H0
    V0 = H2 - H2_0                      # Define off-diagonal quadratic part

    Hint = y[1]                         # Define quartic part of Hamiltonian
    Hint0 = jnp.zeros((n,n,n,n))        # Define diagonal quartic part 
    for index in [[0,0,0,0]]:                  # Load Hint0 with values
        id_print(index)
        id_print(Hint[jnp.array(index)])
        Hint0 = Hint0.at[jnp.array(index)].set(Hint[jnp.array(index)])
    # id_print(Hint0)
    Vint = Hint-Hint0                   # Define off-diagonal quartic part

    # Compute the generator eta
    eta0 = contract(H2_0,V0)
    eta_int = contract(Hint0,V0) + contract(H2_0,Vint)

    # Compute the RHS of the flow equation dH/dl = [\eta,H]
    sol = contract(eta0,H2)
    sol2 = contract(eta_int,H2) + contract(eta0,Hint)

    return [sol,sol2]

def indices(n):
    """ Gets indices of diagonal elements of quartic tensor. """
    mat = jnp.zeros((n,n,n,n),dtype=jnp.int8)    
    for i in range(n):              
        for j in range(n):
            if i != j:
                # Zero the diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                mat = mat.at[i,i,j,j].set(1)
                mat = mat.at[i,j,j,i].set(1)
    # mat = mat.reshape(n**4)
    indices = jnp.argwhere(mat)
    return indices

#==============================================================================
# Run program
#==============================================================================

if __name__ == '__main__': 

    # Define variables
    dl_list = jnp.linspace(0,10,100)
    cutoff = 1e-3
    n = int(os.sys.argv[1])

    # Set up Hamiltonian (H2 = quadratic part, Hint = quartic part)
    H2 = jnp.zeros((n,n))
    for i in range(n):
        # Initialise Hamiltonian with linearly increasing on-site terms (Wannier-Stark)
        H2 = H2.at[i,i].set(i)
    H2 += jnp.diag(jnp.ones(n-1),1) + jnp.diag(jnp.ones(n-1),-1)

    Hint = jnp.zeros((n,n,n,n))
    for i in range(n):
        for j in range(i,n):
            if abs(i-j)==1:
                # Initialise nearest-neighbour interactions
                Hint = Hint.at[i,i,j,j].set(0.5)

    index_list = jnp.array(indices(n))
    print(index_list)

    # Set up integration variables
    k=1
    sol2 = jnp.zeros((len(dl_list),n,n))
    sol4 = jnp.zeros((len(dl_list),n,n,n,n))
    sol2 = sol2.at[0].set(H2)
    sol4 = sol4.at[0].set(Hint)
    J0 = 1

    # Do integration step by step until the max off-diagonal term is smaller than the cutoff, then exit
    while k <len(dl_list) and J0 > cutoff:
        print(k)
        soln = odeint(int_ode,[sol2[k-1],sol4[k-1]],dl_list[k:k+2],index_list)
        sol2 = sol2.at[k].set(soln[0][-1])
        sol4 = sol4.at[k].set(soln[1][-1])
        J0 = jnp.max(jnp.abs(soln[0][-1] - jnp.diag(jnp.diag(soln[0][-1]))))
        k += 1