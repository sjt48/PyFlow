# PyFlow

Codebase for the ~~Tensor Flow Equation~~ **PyFlow** library, used for computing local integrals of motion of a variety of disordered quantum systems and computing their non-equilibrium dynamics. The method is described in https://arxiv.org/abs/2110.02906.

*By Dr S. J. Thomson (steven.thomson@fu-berlin.de)*  
*steventhomson.co.uk / @PhysicsSteve*  
*https://orcid.org/0000-0001-9065-9842*  

The Tensor Flow Equation (TFE) method was first proposed in arXiv:2110.02906 by Marco Schiró and I as a way to take the method of continuous unitary transforms (also known as 'flow equations') and turn it into a robust numerical tool that did not require the algebraic complexity of previous analytical approaches. This was based on our previous work using flow equation techniques which, while powerful, were extremely complicated to work with, and was inspired by my time working in the group of Laurent Sanchez-Palencia where I learned a lot about tensor network methods and realised I could use a similar approach to turn flow equations into a powerful numerical tool.

# How to run

The only file you need to edit is `main.py`. It takes three command line arguments, the first to specify the system size, the second the disorder potential, and the third the method used for computing the tensor contractions. For example:

```
python main.py 4 linear tensordot
```

will run the simulation for a system size `L=4` and a linear potential, using NumPy's `tensordot` method. To change other parameters (interaction strength, disorder strength, etc) you will need to edit the file `main.py`. All editable parameters are at the top of the file before the line `# Run Program` - in general, nothing below this line needs to be edited.

In general, `tensordot` is the best method to use for small systems if you don't require normal ordering or other advanced features. I find `vec` to be faster, particularly when running on a single thread, but as it's explicitly typed it can throw errors if you pass a dtype it's not expecting. For large systems and/or normal-ordering, use `jit`, but beware the compilation overhead on the first call to the contraction functions.

The other files are as follows:

* `flow_plots.ipnyb` : a Jupyter notebook for plotting the flow and testing convergence, for small systems only.
* `ED/ed.py` : the exact diagonalization script, calls QuSpin and returns result.
* `core/diag.py` : main body of the flow equation code, handles integration and computes RHS of differential equations.
* `core/contract.py` : tensor contraction routines, using several different methods ('einsum', 'tendordot', 'jit', 'vec').
* `core/init.py` : utility functions, e.g. generating Hamiltonian matrices/tensors and making directory to save data in.
* `core/utility.py` : various utility functions. **To do: move other helper fns from diag.py to here.**
* `tests/con_method_test.py` : test to check different contraction methods return same answer, to within numerical precision.
* `models/models.py` : contains model classes and initialisation functions. **To be cleaned up.**
* `tutorials/` : a folder for tutorial files illustrating and explaining aspects of the method.
* `examples/` : empty folder which will contain example scripts at a later date.
* `PyFlow_cuda/` : prototype GPU code using PyTorch. (Contains duplicate files - **to be cleaned up**.)
* `TFE_jax/` : prototype GPU code using JAX. (Faster than PyTorch but requires compilation - **to be cleaned up**.)


**GPU files have been added but are not yet fully tested.**
**Note that the GPU codes may not (yet) work for anything other than static propeties of spinless fermions.**

# Required Packages

* NumPy
* SciPy
* h5py
* [QuSpin](https://weinbe58.github.io/QuSpin/)
* Numba
* SymPy
* matplotlib
* os 
* sys
* gc 
* psutil (can replace with `multiprocessing` if necessary)
* datetime 

For the GPU implementation, the following additional packages (and their dependencies) are required:

* PyTorch
* torchdiffeq

OR

* JAX

# To do 

* Clean up GPU codes to avoid file duplication; remove debug print commands.
* Fix comments and docstrings throughout.
* Add possibility to specify more general Hamiltonians via sparse input matrix, like QuSpin? Then the user only has to specify the Hamiltonian and call the contraction routines, without using the built-in model classes at all.
* ~~Update non-equilibrium dynamics code to match syntax used for static functions.~~ (**Done, can be tidied up.**)
* Write more tests for non-interacting systems.
* ~~Normal-ordering for Hubbard models can introduce small deviations from Hermitian matrices. Why?~~ (**Fixed!**)
* For non-interacting systems, compute LIOMs with ED as a comparison/test? 
* Add creation/annihilation operators (factor of N cheaper than number operators to compute).
* ~~Add long-range couplings from earlier code versions.~~ **Done, to be tested thoroughly.**
* ~~Add 2D and 3D code.~~ **Done, passed tests in 2D.**
* Figure out better solution for filenames for various models.
* ~~Implement model classes?~~ **Done, to be tested thoroughly.**
* ~~Merge with spinful fermion (Hubbard model) code.~~ **Done, works for static properties.**
* Add bosons: comes down to changing a few minus signs in contract.py. **In progress.**
* Merge GPU code (via CuPy/PyTorch) into main.
* Benchmark against ED and tensor networks for finding ground state energy, non-equilibrium dynamics?
* Test code for normal ordering? **In progress, bugs fixed.**
* Add Floquet functionality? (Following https://www.scipost.org/SciPostPhys.11.2.028.)
* Possibility to study dissipative systems by using non-Hermitian Hamiltonians?
* Add Majorana functionality. (Done in different codebase...)
* Timing tests for einsum/tensordot/jit and MKL/BLAS?
* Add Cython as a method? Not good for GPUs, but useful for CPUs.
* Add old version of writing differential equations explicitly? Useful for systems not easily encoded as tensors.
* Add momentum space flow, for translationally invariant systems or systems deep in delocalized phase. (Added, commented out until tested.)
* Try to add spins? Their commutation relations are not friendly, but there must be some way to automate it.
* Open quantum systems: Lindblad operators possible by adapting spinful fermion code (2x2 block structure of L).
