from  jax.experimental.ode import odeint as ode
import jax.numpy as jnp


def int_ode(y,l,method='einsum'):
        """ Generate the flow equation for the interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the ijnput array eta will be used to specify the generator at this flow time step. The latter option will result 
        in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
        integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
        steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
        interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
        these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
        the benefits from the speed increase likely outweigh the decrease in accuracy.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for interacting system.

        """
        
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        H2 = y[0]                           # Define quadratic part of Hamiltonian
        n,_ = H2.shape
        H2_0 = jnp.diag(jnp.diag(H2))       # Define diagonal quadratic part H0
        V0 = H2 - H2_0                      # Define off-diagonal quadratic part

        Hint = y[1]                         # Define quartic part of Hamiltonian
        Hint0 = jnp.zeros((n,n,n,n))        # Define diagonal quartic part 
        for i in range(n):                  # Load Hint0 with values
            for j in range(n):
                    Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
                    Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
        Vint = Hint-Hint0

        # Compute the generator eta
        eta0 = contract(H2_0,V0,method=method,eta=True)
        eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H2_0,Vint,method=method,eta=True)

   
        # Compute the RHS of the flow equation dH/dl = [\eta,H]
        sol = contract(eta0,H2,method=method)
        sol2 = contract(eta_int,H2,method=method) + contract(eta0,Hint,method=method)

        return [sol,sol2]

# General contraction function
def contract(A,B,method='jit',comp=False,eta=False,pair=None):
    """ General contract function: gets shape and calls appropriate contraction function. """
    # A = A.astype(np.float64)
    # B = B.astype(np.float64)
    if A.ndim == B.ndim == 2:
        con = con22(A,B,method,comp,eta)
    if A.ndim != B.ndim:
        if A.ndim == 4:
            if B.ndim == 2:
                if pair == None:
                    con = con42(A,B,method,comp)
        if A.ndim == 2:
            if B.ndim == 4:
                if pair == None:
                    con = con24(A,B,method,comp)

    # print(get_num_threads())
    # print("Threading layer: %s" % threading_layer())

    return con

# Contract square matrices (matrix multiplication)
def con22(A,B):
    return jnp.einsum('ij,jk->ik',A,B) - jnp.einsum('ki,ij->kj',B,A)

# Contract rank-4 tensor with square matrix
def con42(A,B,method='jit',comp=False):

    if method == 'einsum':
        con = jnp.einsum('abcd,df->abcf',A,B) 
        con += -jnp.einsum('abcd,ec->abed',A,B,optimize=True)
        con += jnp.einsum('abcd,bf->afcd',A,B,optimize=True)
        con += -jnp.einsum('abcd,ea->ebcd',A,B,optimize=True)

    return con


if __name__ == '__main__': 

    # Define integrator
    dl_list = jnp.linspace(0,10,100)
    cutoff = 1e-3
    n = 4


    H2 = jnp.zeros((n,n))

    if isinstance(d,float) == False:
        H0 = jnp.diag(d)
    for i in range(n):
        # Initialise Hamiltonian with random on-site terms
        H0[i,i] = i
    H0 += jnp.diag(jnp.ones(n-1),1) + jnp.diag(jnp.ones(n-1),-1)

    Hint = jnp.zeros((n,n,n,n))
    for i in range(n):
        for j in range(i,n):
            if abs(i-j)==1:
                # Initialise nearest-neighbour interactions
                Hint[i,i,j,j] = 0.5

    # Integration with hard-coded event handling
    k=1
    sol2 = jnp.zeros((len(dl_list),n,n))
    sol4 = jnp.zeros((len(dl_list),n,n,n,n))
    sol2 = sol2.at[0].set(H2)
    sol4 = sol4.at[0].set(Hint)
    J0 = 1
    print('test')
    # ode_jit = jax.jit(int_ode)
    # ode(ode_jit,[sol2[0],sol4[0]],dl_list[0:2])

    while k <len(dl_list) and J0 > cutoff:
        print(k)
        soln = ode(int_ode,[sol2[k-1],sol4[k-1]],dl_list[k:k+2])
        sol2 = sol2.at[k].set(soln[0][-1])
        sol4 = sol4.at[k].set(soln[1][-1])
        J0 = jnp.max(jnp.abs(soln[0][-1] - jnp.diag(jnp.diag(soln[0][-1]))))
        k += 1