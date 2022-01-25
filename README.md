# TensorFlowEquations

Codebase for the Tensor Flow Equation method, used for computing local integrals of motion of a variety of disordered quantum systems and computing their non-equilibrium dynamics. The method is described in https://arxiv.org/abs/2110.02906.

# Main Required Packages

* NumPy
* SciPy
* h5py
* [QuSpin](https://weinbe58.github.io/QuSpin/)
* Numba
* SymPy
* matplotlib

# To do 

* Update non-equilibrium dynamics code to match syntax used for static functions. (**Current version will throw an error!**)
* Write more tests for non-interacting systems.
* For non-interacting systems, compute LIOMs with ED as a comparison/test? 
* Add long-range couplings from earlier code versions.
* Figure out better solution for filenames for various models.
* Implement model classes?
* Merge with spinful fermion (Hubbard model) code.
* Add bosons: comes down to changing a few minus signs in contract.py.
* Merge GPU code (via CuPy/PyTorch) into main.
* Benchmark against ED and tensor networks for finding ground state energy, non-equilibrium dynamics?
* Test code for normal ordering?
* Add Floquet functionality?
* Possibility to study dissipative systems by using non-Hermitian Hamiltonians?
* Add Majorana functionality. (Done in different codebase...)
* Timing tests for einsum/tensordot/jit and MKL/BLAS?
* Add Cython as a method? Not good for GPUs, but useful for CPUs.
* Add old version of writing differential equations explicitly? Useful for systems not easily encoded as tensors.
* Add momentum space flow, for translationally invariant systems or systems deep in delocalized phase.
* Try to add spins? Their commutation relations are not friendly, but there must be some way to automate it.
