# -*- coding: utf-8 -*-
"""
---------------------------------------------
Flow Equations for Many-Body Quantum Systems
S. J. Thomson
Institue de Physique Theorique, CEA Paris-Saclay
steven.thomson@ipht.fr
steventhomson.co.uk
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

This work makes use of the QuSpin package (http://weinbe58.github.io/QuSpin/),
as well as conventional scientific Python libraries such as NumPy, SciPy and matplotlib.

---------------------------------------------

This file compiles the Cython script "contract_cython.pyx" into the module "contract_cython" which may 
be called from Python. The compile code may need editing depending on your 
particular system configuration.

"""

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
	
import numpy
from Cython.Distutils import build_ext

setup(
  name = 'contract_cython',
  ext_modules=[
    Extension('contract_cython', ['contract_cython.pyx'],include_dirs=[numpy.get_include()],extra_compile_args = ["-ffast-math"])
    ],
  cmdclass = {'build_ext': build_ext}
)
