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

This file contains the code used to initialise the Hamiltonian classes for a variety of particle types.

"""

import core.init as init
import numpy as np

class hamiltonian:
    def __init__(self,species,dis_type,intr,pwrhop=False,pwrint=False):
        self.species = species
        self.dis_type = dis_type
        self.intr = intr
        self.pwrhop = pwrhop
        self.pwrint = pwrint

    def build(self,n,dim,d,J,x,delta=0,delta_up=0,delta_down=0,delta_mixed=0,delta_onsite=0,alpha=0,beta=0,U=0,dsymm='charge'):
        self.n = n
        self.dim = dim
        self.x = x

        if dim == 1:
            self.L = n
        elif dim == 2:
            self.L = int(np.sqrt(n))
        elif dim == 3:
            self.L = int(n**(1/3))

        if self.species == 'spinless fermion':
            self.d = d
            self.J = J
            self.H2_spinless = init.Hinit(n,d,J,self.dis_type,x,pwrhop=False,alpha=0,Fourier=False,dim=dim)
            if self.intr == True:
                self.delta = delta
                self.H4_spinless = init.Hint_init(n,delta,pwrint=False,beta=0,dim=dim)

        elif self.species == 'spinful fermion':
            self.d = d
            self.J = J
            self.dsymm = dsymm
            H2up,H2dn = init.H2_spin_init(n,d,J,self.dis_type,x,pwrhop=False,alpha=0,Fourier=False,dsymm=dsymm)
            self.H2_spinup = H2up
            self.H2_spindown = H2dn
            if self.intr == True:

                self.delta_onsite = delta_onsite
                self.delta_up = delta_up 
                self.delta_down = delta_down 
                self.delta_mixed = delta_mixed
                H4up,H4dn,H4updn = init.H4_spin_init(n,delta_onsite=delta_onsite,delta_up=delta_up,delta_down=delta_down,delta_mixed=delta_mixed)
                self.H4_spinup = H4up
                self.H4_spindown = H4dn
                self.H4_mixed = H4updn

        elif self.species == 'boson':
            self.d = d
            self.J = J
            self.U = U
            self.boson = init.Hinit(n,d,J,self.dis_type,x=0,pwrhop=False,alpha=0,Fourier=False,dim=dim)
            if self.intr == True:
                self.H4_boson = init.Hint_init(n,0,pwrint=False,beta=0,dim=dim,U=U)
        elif self.species =='hard core boson':
            self.d  = d

class fermion:
    def __init__(self,species,dis_type,intr,pwrhop=False,pwrint=False):
        self.species = species
        self.dis_type = dis_type
        self.intr = intr
        self.pwrhop = pwrhop
        self.pwrint = pwrint

    def build(self,n,dim,d,J,dis_type,delta=0,delta_up=0,delta_down=0,delta_mixed=0,delta_onsite=0,alpha=0,beta=0,U=0,dsymm='charge'):
        self.n = n
        self.dim = dim

        if dim == 1:
            self.L = n
        elif dim == 2:
            self.L = int(np.sqrt(n))
        elif dim == 3:
            self.L = int(n**(1/3))

        self.d = d
        self.J = J
        self.H2_spinless = init.Hinit(n,d,J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False,dim=dim)
        if self.intr == True:
            self.delta = delta
            self.H4_spinless = init.Hint_init(n,delta,pwrint=False,beta=0,dim=dim)


class hubbard:
    def __init__(self,species,dis_type,intr,pwrhop=False,pwrint=False):
        self.species = species
        self.dis_type = dis_type
        self.intr = intr
        self.pwrhop = pwrhop
        self.pwrint = pwrint

    def build(self,n,dim,d,J,dis_type,delta=0,delta_up=0,delta_down=0,delta_mixed=0,delta_onsite=0,alpha=0,beta=0,U=0,dsymm='charge'):
        self.n = n
        self.dim = dim

        if dim == 1:
            self.L = n
        elif dim == 2:
            self.L = int(np.sqrt(n))
        elif dim == 3:
            self.L = int(n**(1/3))


        self.d = d
        self.J = J
        self.dsymm = dsymm
        H2up,H2dn = init.H2_spin_init(n,d,J,dis_type,x=0,pwrhop=False,alpha=0,Fourier=False,dsymm=dsymm)
        self.H2_spinup = H2up
        self.H2_spindown = H2dn
        if self.intr == True:

            self.delta_onsite = delta_onsite
            self.delta_up = delta_up 
            self.delta_down = delta_down 
            self.delta_mixed = delta_mixed
            H4up,H4dn,H4updn = init.H4_spin_init(n,delta_onsite=delta_onsite,delta_up=delta_up,delta_down=delta_down,delta_mixed=delta_mixed)
            self.H4_spinup = H4up
            self.H4_spindown = H4dn
            self.H4_mixed = H4updn