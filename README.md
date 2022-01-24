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

* Update non-equilibrium dynamics code to match syntax used for static functions. ((**Current version will throw an error!**))
* Write more tests for non-interacting systems.
* Add long-range couplings from earlier code versions.
* Figure out better solution for filenames for various models.
* Implement model classes?
* Merge with spinful fermion (Hubbard model) code.
* Add bosons: comes down to changing a few minus signs in contract.py.
* Merge GPU code (via CuPy/PyTorch) into main.
