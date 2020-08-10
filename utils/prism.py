######
##
##  Prism - 
##  A file/information interface
##  Crystal, LSDalton, Libint, XDEC
##
#######

import os
import sys
import argparse
import subprocess as sp
import copy
import re

try:    
    import numba
except:
    numba = False
    


import sympy as syp
import numpy as np
import subprocess as sp

try:
    import crystal_interface as ci
    import sympy_basis as sb
except:
    import utils.crystal_interface as ci
    import utils.sympy_basis as sb


from ast import literal_eval


class mo_function():
    """
    Numeric representation of MO-function
    """
    def __init__(self, mo_basis):
        self.mo_basis = mo_basis
    
    #@numba.autojit(parallel=True)
    #@numba.njit
    def at(self, x,y,z):
        #ret = np.zeros(x.shape, dtype = float)
        ret = 0*x*y*z
        for chi in self.mo_basis:
            #print(chi)
            ret += chi(x,y,z)
        return ret
        
        
        


class prism():
    def __init__(self, filename, theta = None):
        self.theta = theta
        

        self.filename = filename
        if filename.split(".")[-1] == "npy":
            self.load(filename)
        if filename.split(".")[-1] == "d12":
            #print("READ:Crystal not yet implemented")
            
            #Locate executables for crystal
            try:
                self.crystal_exe_path = os.environ['CRYSTAL_EXE_PATH'] 
            except:
                try:
                    self.crystal_exe_path = os.environ['CRY14_EXEDIR'] + "/" + os.environ['VERSION']
                except:
                    #cryscor_exe = "/Crystal/bin/cryscor.x"
                    self.crystal_exe_path = "/Users/audunhansen/Crystal/bin/MacOsX-ifort-64/v.1.0.3"
                    print("\033[93m" + "WARNING"+ "\033[0m"+": Crystal executable folder not found.")
                    print("\033[93m" + "       "+ "\033[0m"+": Environment variable CRY14_EXEDIR needed.")
                    print("\033[93m" + "       "+ "\033[0m"+"  Environment variable VERSION needed.")
                    print("\033[93m" + "       "+ "\033[0m"+"  (These are default variables for Crystal 14.)")
                    print("         Instead using:", self.crystal_exe_path)   
                
            
            f = open(filename).read()
            #geom = ""
            self.shrink = [literal_eval(i) for i in f.split("SHRINK")[1].split("\n")[1].split()]
            testgeom = f.lower().split("endg")[0] + "TESTGEOM\nEND\n"
            filename_tg = filename.split(".")[0] + "_tg.d12"
            f_testgeom = open(filename_tg, "w")
            f_testgeom.write(testgeom)
            f_testgeom.close()
            
            
            # Get periodicity
            self.cperiodicity = f.split("\n")[1]
            
            if self.cperiodicity=="POLYMER":
                self.lsperiodicity = ["active", "inactive", "inactive"]
            if self.cperiodicity=="SLAB":
                self.lsperiodicity = ["active", "active", "inactive"]
            if self.cperiodicity=="CRYSTAL":
                self.lsperiodicity = ["active", "active", "active"]
                
            
            
            # Extract lattice vectors
            geom = sp.check_output([self.crystal_exe_path + "/crystal"], stdin = open(filename_tg))

            lattice_str = str(geom).split("DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)")[1]
            
            a1 = [literal_eval(i) for i in lattice_str.split("""\\n""")[2].split()]
            a2 = [literal_eval(i) for i in lattice_str.split("""\\n""")[3].split()]
            a3 = [literal_eval(i) for i in lattice_str.split("""\\n""")[4].split()]

            
            
            # Extract atom positions in refcell
            
            #pos_str = str(geom).split("CARTESIAN COORDINATES - PRIMITIVE CELL")[1].split("WARNING")[0]
            #print(pos_str)
            #pos_str_s = pos_str.split("""\\n""")
            
            pos_str_s = str(geom).split("CARTESIAN COORDINATES - PRIMITIVE CELL")[1].split("WARNING")[0].split("ATOM")[1].split("**\\n")[1].split("\\n\\n")[0].split("\\n") #.split("\n\n")
            #print(pos_str_s)
            self.atoms = []
            self.charges = []
            
            for i in np.arange(len(pos_str_s)):
                if len(pos_str_s[i])>1:
                    self.atoms.append([literal_eval(j) for j in pos_str_s[i].split()[3:]])    
                    self.charges.append(literal_eval(pos_str_s[i].split()[1]))
            
            #for i in np.arange(4,len(pos_str_s)-1):
            #    pos_i = [literal_eval(j) for j in pos_str_s[i].split()[3:]]
            #    charge_i = literal_eval(pos_str_s[i].split()[1])
            #    self.atoms.append(pos_i)
            #    self.charges.append(charge_i)
            
            
            #convert to bohr, store to self
            #0.5291772083
            self.lattice = np.array([a1,a2,a3])/0.52917720830000000000000000000
            self.atoms = np.array(self.atoms)/0.52917720830000000000000000000
            self.charges = np.array(self.charges)
            
            
            # Get basis
            basis_s = f.lower().split("endg")[1].split("endbs")[0]
            
            self.atomic_numbers, self.basis_set = parse_crystal_basis(basis_s)
            self._sort_basis()

            self.n_core = 0 #set number of core orbitals manually
            self.set_nocc()
            
            
            
            
                
        if filename.split(".")[-1] == "INP":
            #Read LSDalton INPUT
            self._setup_blattice() #lattice + geometry
            self._sort_basis()
            self.n_core = 0 #set number of core orbitals manually
            self.set_nocc()
            
            
        if filename.split(".")[-1] == "g94":
            print("READ:Libint not yet implemented")
        if filename.split(".")[-1] == "xyz":
            pass

        # set number of virtuals
        self.set_nvirt()
    
    def ndim_layer(self, N):
        if self.cperiodicity=="POLYMER":
            return np.array([N,0,0])
        if self.cperiodicity=="SLAB":
            return np.array([N,N,0])
        if self.cperiodicity=="CRYSTAL":
            return np.array([N, N, N])
        
            

            
    def save(self, filename):
        data = np.array([self.lattice,
                         self.atoms,
                         self.charges,
                         self.atomic_numbers,
                         self.basis_set])
        np.save(filename, data)
    
    def load(self, filename):
        self.lattice, self.atoms, self.charges, self.atomic_numbers, self.basis_set = np.load(filename)
        
    def _sort_basis(self):
        #sort contracted functions per atom in order of increasing angular momentum
        basis_set = []
        for atom in np.arange(len(self.basis_set)):
            basis_set.append([])
            for l in np.arange(6):
                basis_set[-1].append([])
                for contracted in np.arange(len(self.basis_set[atom])):
                    c_l = self.basis_set[atom][contracted][0][2]
                    if c_l == l:
                        basis_set[-1][-1].append([])
                        for primitive in np.arange(len(self.basis_set[atom][contracted])):
                            basis_set[-1][-1][-1].append(self.basis_set[atom][contracted][primitive])
        #print(basis_set)
        self.basis_set = basis_set
    
    def show_basis(self):
        for atom in np.arange(len(self.basis_set)):
            for shell in np.arange(len(self.basis_set[atom])):
                for contracted in np.arange(len(self.basis_set[atom][shell])):
                    for primitive in np.arange(len(self.basis_set[atom][shell][contracted])):
                        print(atom, shell, contracted, primitive, self.basis_set[atom][shell][contracted][primitive])
    
    def get_shell_permutation(self, l):
        """
        Returns the shell permutation vector
        for a given angular momentum l
        Author: Audun Skau Hansen
        """
        if l==0:
            return np.array([0])
        if l==1:
            return np.array([0,1,2])
            #return np.array([1,2,0])
            #return np.array([2,0,1]) #z,x,y <- goes for ethylene for some reason, inconsistent with manuals
        if l==2:
            return np.array([4, 2, 0, 1, 3])
            #return np.array([3, 1, 0, 2, 4])
        if l==3:
            return np.array([6, 4,2,0,1,3,5])            
        if l==4:
            return np.array([8,6, 4,2,0,1,3,5,7])
        if l==5:
            return np.array([10, 8,6, 4,2,0,1,3,5,7, 9])
        if l>5:
            print("Error: Angular momentum permutation not defined.")
    
    def get_permutation_vector(self):
        """
        Set up basis permutation from Crystal to XDEC
        Author: Audun Skau Hansen
        """
        permutation_vector = []
        Ltot = 0
        for charge in self.charges:
            for j in np.arange(len(self.atomic_numbers)):
                if charge==self.atomic_numbers[j]:
                    break
            for shell in np.arange(len(self.basis_set[j])):

                l = 2*shell + 1 #number of functions
                for contracted in np.arange(len(self.basis_set[j][shell])):
                    
                    permutation_vector.append(self.get_shell_permutation(shell)+Ltot)
                    Ltot += l
        return np.concatenate(permutation_vector)
    
    def get_inverse_permutation(self):
        """
        Set up basis permutation from XDEC to Crystal
        Author: Audun Skau Hansen
        """
        p = self.get_permutation_vector()
        return np.arange(len(p))[p][p]       
            
            
    def get_lsdalton_input(self):
        # Write MOLECULE.INP for use with LSDalton
        s = "BASIS\nUSERDEFINED\n\n\n"
        s += self.get_lsdalton_geom()
        s += self.get_lsdalton_basis()
        return s
    
    def get_multicell_lsdalton_input(self, cells = np.array([0,0,0])):
        s = "BASIS\nUSERDEFINED\n\n\n"
        s += self.get_lsdalton_multicell_geom_sorting(cells)
        s += self.get_lsdalton_basis()
        return s
    
    def get_lsdalton_geom(self):
        unique_charges = np.sort(np.unique(self.charges))
        #print(unique_charges)
        s = "Atomtypes=%i Nosymmetry periodic\n" % unique_charges.shape[0]
        for i in np.arange(unique_charges.shape[0]):
            s += "Charge=%i. Atoms=%i \n" % (unique_charges[i], np.sum(self.charges==unique_charges[i]))
            for j in np.arange(self.charges.shape[0]):
                if self.charges[j] == unique_charges[i]:
                    s += "n  %.10e  %.10e  %.10e \n" % (self.atoms[j][0], self.atoms[j][1],self.atoms[j][2])
        for i in np.arange(3):
            s += "a%i = %.10e  %.10e  %.10e %s\n" % (i+1, self.lattice[i][0],self.lattice[i][1],self.lattice[i][2], self.lsperiodicity[i])
        return s
    
    #def get_nearest_neighbour(self, coords = np.array([np.array([0,0,0])]), r0 = np.array([0,0,0])):
    #    atoms = self.get_atoms(coords)
        
        
    
    def get_doublecell_basis_order(self):
        #count number of contracteds per atom
        nc_a = np.zeros(len(self.basis_set)*2, dtype = int)
        for atom in np.arange(len(self.basis_set)):
            for shell in np.arange(len(self.basis_set[atom])):
                nc_a[atom] += len(self.basis_set[atom][shell])*(2*shell + 1)
        nc_a[int(len(nc_a)/2):] = nc_a[:int(len(nc_a)/2)]*1
        return nc_a
            
    
    def get_lsdalton_multicell_geom(self, cells = np.array([[0,0,0]])):
        unique_charges = np.sort(np.unique(self.charges))
        #print(unique_charges)
        s = "Atomtypes=%i Nosymmetry periodic\n" % unique_charges.shape[0]
        for i in np.arange(unique_charges.shape[0]):
            s += "Charge=%i. Atoms=%i \n" % (unique_charges[i], np.sum(self.charges==unique_charges[i])*cells.shape[0])
            for cell in cells:
                atoms, charges = self.get_atoms(m = [cell])
                #print(charges)
                #print(atoms)
                #print("--")
                for j in np.arange(len(charges)):
                    if charges[j] == unique_charges[i]:
                        s += "n  %.10e  %.10e  %.10e \n" % (atoms[j][0], atoms[j][1],atoms[j][2])
        for i in np.arange(3):
            s += "a%i = %.10e  %.10e  %.10e %s\n" % (i+1, self.lattice[i][0],self.lattice[i][1],self.lattice[i][2], self.lsperiodicity[i])
        return s
    
    def get_lsdalton_multicell_geom_sorting(self, cells = np.array([[0,0,0]])):
        #atoms, charges = self.get_atoms(m = [cell])
        unique_charges = np.sort(np.unique(self.charges))
        #print(unique_charges)
        s = "Atomtypes=%i Nosymmetry periodic\n" % (unique_charges.shape[0]*2)
        for cell in cells:
            atoms, charges = self.get_atoms(m = [cell])
            unique_charges = np.sort(np.unique(charges))
            for i in np.arange(unique_charges.shape[0]):
                #s += "Charge=%i. Atoms=%i \n" % (unique_charges[i], np.sum(charges==unique_charges[i]))
                s += "Charge=%i. Atoms=%i \n" % (unique_charges[i], np.sum(charges==unique_charges[i]))
                for j in np.arange(len(charges)):
                    if charges[j] == unique_charges[i]:
                        s += "n  %.10e  %.10e  %.10e \n" % (atoms[j][0], atoms[j][1],atoms[j][2])
        for i in np.arange(3):
            s += "a%i = %.10e  %.10e  %.10e %s\n" % (i+1, self.lattice[i][0],self.lattice[i][1],self.lattice[i][2], self.lsperiodicity[i])
        return s        
        
    
    def get_lsdalton_basis(self):
        s = "USERDEFINED BASIS \n"
        for atom in np.arange(len(self.atomic_numbers)):
            s += "a %i \n"  % self.atomic_numbers[atom]
            
            for shell in np.arange(len(self.basis_set[atom])):
                nb = 0
                nc = len(self.basis_set[atom][shell]) #number of contracted in shell
                
                
                for contracted in np.arange(len(self.basis_set[atom][shell])):
                    nb += len(self.basis_set[atom][shell][contracted])
                
                tab = np.zeros((nb,nc+1), dtype = float)
                nb = 0
                for contracted in np.arange(len(self.basis_set[atom][shell])):
                    for primitive in np.arange(len(self.basis_set[atom][shell][contracted])):
                        tab[nb,0] = self.basis_set[atom][shell][contracted][primitive][0]
                        tab[nb,contracted+1] = self.basis_set[atom][shell][contracted][primitive][1]
                        nb += 1

                if tab.shape[0]!=0:
                    s += "    %i    %i    0\n" % (tab.shape[0], tab.shape[1]-1)
                    for primitive in np.arange(tab.shape[0]):
                        tab_p = tab[primitive]
                        s += "        "
                        for elm in tab_p:
                            s += "%.12f  " % elm
                        s += "\n"

                    
                    
                
                
                
            
        return s
    def get_lambda_basis(self):
        """
        generate a list of lambda functions 
        corresponding to the AO-basis 
        (same ordering and so on)
        """
        lambda_basis = []
        for charge in np.arange(self.charges.shape[0]):
            
            
            atomic_number = self.charges[charge]
            atom = np.argwhere(self.atomic_numbers==atomic_number)[0,0] #index of basis function
            #atom = self.charges[]
            
            pos = self.atoms[charge]
            
            #ao_basis = 
            
        
            
            for shell in np.arange(len(self.basis_set[atom])):
                for contracted in np.arange(len(self.basis_set[atom][shell])):
                    W = np.array(self.basis_set[atom][shell][contracted])
                    w = W[:,1]
                    a = W[:,0]
                    #print("a:", a)
                    #print("w:", w)
                    #print(w)
                    if shell == 1:
                        for m in np.array([1,-1,0]):
                            #print("shell l/m:", shell, m)
                            lambda_basis.append([sb.get_contracted(a,w,shell,m), pos])
                    else:
                        for m in np.arange(-shell, shell+1):
                            #print("shell l/m:", shell, m)
                            lambda_basis.append([sb.get_contracted(a,w,shell,m), pos])
                    
                    
                    #for primitive in np.arange(len(self.basis_set[atom][shell][contracted])):
                    #    print(atom, shell, contracted, primitive, self.basis_set[atom][shell][contracted][primitive])
        return lambda_basis
    
    #@numba.autojit()
    def evaluate_orbital(self, x,y,z, p, C, thresh = 10e-6):
        
        lattice = self.lattice
        
        lbasis = self.get_lambda_basis()
        
        Z = 0*x*y*z #np.zeros(x.shape[0], dtype = float)
        
        for c in np.arange(C.coords.shape[0]):
            
            Cb = C.cget(C.coords[c]).T
            
            T = np.dot(C.coords[c],lattice)
            
            for i in np.arange(Cb[p].shape[0]):
                #print(i)
                ta = lbasis[i][1]
                Z += Cb[p,i]*lbasis[i][0](x-T[0]-ta[0], y-T[1]-ta[1], z-T[2]-ta[2])
        return Z
        
    def get_basis_index(self, charge):
        for i in np.arange(len(self.atomic_numbers)):
            if self.atomic_numbers[i] == charge:
                return i
    def get_product_center(self, mu, nu, coord):
        """
        return the cartesian center of the gaussian product x_mu*x_nu
        """
        pass
        
    def get_ao_angmom(self):
        """
        return a list containing the angular moments 
        of each contracted AO-function
        """
        angmoms = []
        
        for atom in np.arange(len(self.charges)):
                
            atom_basis = self.get_basis_index(self.charges[atom])
            
            
            for shell in np.arange(len(self.basis_set[atom_basis])):
                
                for contracted in np.arange(len(self.basis_set[atom_basis][shell])):
                    for i in np.arange(2*shell + 1):
                        angmoms.append(shell)
        return angmoms
    
    
    #@numba.autojit(nopython= True, parallell = True)
    #@numba.jit(nopython=True, parallel=True)
    
    def evaluate_doi_at(self, Ca, Cb, x,y,z, coord = None):
        """
        Evaluate 
            \int_{R^3} \varphi_p^2 \varphi_q^2 dr
        in 
            r = xi + yj + zk
        """
        lbasis = self.get_lambda_basis()
        lattice = self.lattice
        
        #Tc = np.dot(coord, lattice)
        Z = np.zeros((x.shape[0],Ca.cget([0,0,0]).shape[1]), dtype = float)
        if coord is not None:
            Z_ = np.zeros((x.shape[0],Ca.cget([0,0,0]).shape[1]), dtype = float)
            #print(Z.shape, Z_.shape)
        for c in np.arange(len(Ca.coords)):
            Ta = -np.dot(Ca.coords[c], lattice)
            Cmu = Ca.cget(Ca.coords[c])
            if coord is not None:
                Cmu_ = Cb.cget(Ca.coords[c] + coord)
            for mu in np.arange(len(lbasis)):
                ta = lbasis[mu][1]
                #print(ta, Ta)
                #print(lbasis[mu][0](x,y,z))
                #@numba.autojit(parallel = True)
                lb = lbasis[mu][0](x-ta[0]-Ta[0], y-ta[1]-Ta[1], z-ta[2]-Ta[2])
                
                Cm = Cmu[mu,:]
                
                #print(lb.shape)
                #print(Cm.shape)
                
                Z += Cm*lb[:, None]
                if coord is not None:
                    #print(Z_.shape)
                    Cm_ = Cmu_[mu,:]
                    Z_ += Cm_*lb[:, None]
        #print("1")
        if coord is None:
            Z_ = Z*1.0
        #Z = Z**2
        Z_ = Z[:,None,:]*Z_[:,:,None]
        #Z = np.mean(Z_, axis = 0)
        Zabs = np.mean(np.abs(Z_), axis = 0)
        Z2 = np.mean((Z_)**2, axis = 0)
        Z = np.mean(Z_, axis = 0)
        return Z,Z2, Zabs #, Z2

    #@numba.jit(nopython=True, parallel=True)
    def evaluate_weighted_doi_at(self, Ca, Cb, x,y,z, weights, coord = None):
        """
        Evaluate 
            \int_{R^3} \varphi_p^2 \varphi_q^2 dr
        in 
            r = xi + yj + zk
        """
        lbasis = self.get_lambda_basis()
        lattice = self.lattice
        
        #Tc = np.dot(coord, lattice)
        Z = np.zeros((x.shape[0],Ca.cget([0,0,0]).shape[1]), dtype = float)
        if coord is not None:
            Z_ = np.zeros((x.shape[0],Ca.cget([0,0,0]).shape[1]), dtype = float)
            #print(Z.shape, Z_.shape)
        for c in np.arange(len(Ca.coords)):
            Ta = -np.dot(Ca.coords[c], lattice)
            Cmu = Ca.cget(Ca.coords[c])
            if coord is not None:
                Cmu_ = Cb.cget(Ca.coords[c] + coord)
            for mu in np.arange(len(lbasis)):
                ta = lbasis[mu][1]
                #print(ta, Ta)
                #print(lbasis[mu][0](x,y,z))
                #@numba.autojit(parallel = True)
                lb = lbasis[mu][0](x-ta[0]-Ta[0], y-ta[1]-Ta[1], z-ta[2]-Ta[2])
                
                Cm = Cmu[mu,:]
                
                #print(lb.shape)
                #print(Cm.shape)
                
                Z += Cm*lb[:, None]
                if coord is not None:
                    #print(Z_.shape)
                    Cm_ = Cmu_[mu,:]
                    Z_ += Cm_*lb[:, None]
        #print("1")
        if coord is None:
            Z_ = Z*1.0
        #Z = Z**2
        Z_ = Z[:,None,:]*Z_[:,:,None]
        #Z = np.mean(Z_, axis = 0)
        Zabs = np.sum(np.abs(Z_)*weights[:,None,None], axis = 0)
        Z2 = np.sum((Z_)**2*weights[:,None,None], axis = 0)
        Z = np.sum(Z_*weights[:,None,None], axis = 0)
        return Z,Z2, Zabs #, Z2

    def evaluate_overlap(self, mu, nu, coord, x,y,z):
        """
        Evaluate the overlap matrix element:
        returns
            <0 mu ! coord nu>(x,y,z)
        
        where x,y,z are arrays 
        mu, nu integer indices
        coord is three component integer vector

        """
        lattice = self.lattice
        
        lbasis = self.get_lambda_basis()
        
        #Z = 0*x*y*z #np.zeros(x.shape[0], dtype = float)
        
        T = np.dot(coord,lattice)
        
        ta = lbasis[mu][1]
        tb = lbasis[nu][1]
        
        Z = lbasis[mu][0](x-ta[0], y-ta[1], z-ta[2])* \
            lbasis[nu][0](x-T[0]-tb[0], y-T[1]-tb[1], z-T[2]-tb[2])
        return Z
        
        
        
    def get_mo_function(self, p, C, thresh = 10e-15, offset = None):
        #Cp = C.blocks[:,:,p]
        #print(Cp.shape)
        
        
        
        
        # construct reference basis
        
        ref_basis = []
        x,y,z = syp.symbols("x y z")
        for atom in np.arange(len(self.charges)):
                
            atom_basis = self.get_basis_index(self.charges[atom])
            
            
            for shell in np.arange(len(self.basis_set[atom_basis])):
                for contracted in np.arange(len(self.basis_set[atom_basis][shell])):
                    W = np.array(self.basis_set[atom_basis][shell][contracted])
                    w = W[:,1]
                    a = W[:,0]

                    pos = self.atoms[atom]  #+ np.dot(C.coords[c],lattice)
                    if shell == 1:
                        for m in np.array([1,-1,0]):                            
                            #print(pos)
                            #print( " " )
                            ref_basis.append(syp.lambdify([x,y,z], sb.get_contracted_at(pos, a,w,shell,m), "numpy"))

                    else:
                        for m in np.arange(-shell, shell+1):
                            #print(pos)
                            #print( " " )
 
                            #ref_basis.append(sb.get_contracted_at(pos, a,w,shell,m))
                            ref_basis.append(syp.lambdify([x,y,z], sb.get_contracted_at(pos, a,w,shell,m), "numpy"))
        ref_basis = np.array(ref_basis)
        #print(ref_basis)
        #print(len(ref_basis))
        
        #construct mo
        
        mo_basis = []
        

        
        for c in np.arange(C.coords.shape[0]):
            
            Cb = C.cget(C.coords[c])[:,p]
            #Cb = C.cget(C.coords[c])[p,:]
            
            Ct = Cb[np.abs(Cb)>=thresh]
            Aot = ref_basis[np.abs(Cb)>=thresh]
            
            Tc = -np.dot(C.coords[c],self.lattice) #cell translation
            if offset is not None:
                Tc -= offset #translation for cell
            #print(c, Tc)
            for i in np.arange(Ct.shape[0]):
                #ao_t = Aot[i]
                
                #ao_t = ao_t.subs(x, x-Tc[0])
                #ao_t = ao_t.subs(y, y-Tc[1])
                #ao_t = ao_t.subs(z, z-Tc[2])
                C_pa = copy.deepcopy(Ct[i])
                #print(C_pa)
                chi_a = copy.deepcopy(Aot[i])
                X,Y,Z = copy.deepcopy(Tc)
                #X = copy.deepcopy(X)
                #Y 
                fc = lambda x1,x2,x3, X=X,Y=Y,Z=Z,C_pa=C_pa, chi_a = chi_a : C_pa*chi_a(x1-X, x2-Y, x3-Z)
                mo_basis.append(copy.deepcopy(fc))
                #mo_basis.append(lambda x1,x2,x3 : C_pa*x1*x2*x3)
        
        #print(C.coords[c], c/C.coords.shape[0])
        #print(len(mo_basis))
        #print(mo_basis[0])
        return mo_function(np.array(mo_basis))

    

        
    def get_lambda_orbital(self, p, C, thresh = 10e-6):
        """
        Returns orbital p as lambda function
        (numpy)
        
        p      = orbital (integer)
        C      = orbital coefficients (toeplitz matrix)
        tresh  = coefficient screening
        """
        

        
        orbital_p = 0
        
        lattice = self.lattice
        

        for c in np.arange(C.coords.shape[0]):
            
            Cb = C.cget(C.coords[c]).T
            #print(C.coords[c], c/C.coords.shape[0])
            alpha = 0
            
            
            
            for atom in np.arange(len(self.charges)):
                
                atom_basis = self.get_basis_index(self.charges[atom])
                
                
                for shell in np.arange(len(self.basis_set[atom_basis])):
                    for contracted in np.arange(len(self.basis_set[atom_basis][shell])):
                        W = np.array(self.basis_set[atom_basis][shell][contracted])
                        w = W[:,1]
                        a = W[:,0]
                        #print("Contracted:", contracted, Cb[p, alpha])
                        #print(w,a)
                        #w = W[:,0]
                        #a = W[:,1]
                        #print("a:", a)
                        #print("w:", w)
                        #print(w)
                        pos = self.atoms[atom] + np.dot(C.coords[c],lattice)
                        if shell == 1:
                            for m in np.array([1,-1,0]):
                                #print("shell l/m:", shell, m)
                                # ReMEMBER: ORDERING OF Px Py Pz
                                
                                if np.abs(Cb[p,alpha])>=thresh:
                                    chi = Cb[p,alpha]*sb.get_contracted_at(pos, a,w,shell,m)
                                    #print(c, shell, m, chi)
                                    #print( " " )
                                    orbital_p += chi
                                alpha += 1
                        else:
                            for m in np.arange(-shell, shell+1):
                                #print("shell l/m:", shell, m)
                                # ReMEMBER: ORDERING OF Px Py Pz
                                
                                if np.abs(Cb[p,alpha])>=thresh:
                                    chi = Cb[p,alpha]*sb.get_contracted_at(pos, a,w,shell,m)
                                    #print(c, shell, m, chi)
                                    #print(" " )
                                    orbital_p += chi
                                alpha += 1
                            
                            #lambda_basis.append([sb.get_contracted(a,w,shell,m), self.atoms[atom]])
                        
                        
                        #for primitive in np.arange(len(self.basis_set[atom][shell][contracted])):
                        #    print(atom, shell, contracted, primitive, self.basis_set[atom][shell][contracted][primitive])
                
        x,y,z = syp.symbols("x y z")
        
        return syp.lambdify([x,y,z], orbital_p, "numpy"), orbital_p
                
        
        
        
        
        
    def get_crystal_input(self):
        # Write outfile.d12 for use with Crystal
        pass
    
    def get_libint_input(self):
        # Write basis.g94 for use with libint
        pass
    
    def get_libint_basis(self):
        crystal_basis, basis_info, libint_basis = ci.d2c_basis(self.get_lsdalton_basis())
        return libint_basis

    def rotation_matrix(self, i):
        ti = self.theta*i
        return np.array([[ 1, 0, 0],
                         [0,  np.cos(ti), np.sin(ti)],
                         [0, -np.sin(ti), np.cos(ti)]])


        
            
    def get_atoms(self, m = [[0,0,0]]):
        ######################
        #  Atomic lattice    #
        ######################
        #Returns atoms in cells m
        
        lattice = self.lattice
        
        Positi = []
        Charges = []
        
        for i in m:
            
            for c in range(len(self.charges)):
                #print(lattice, i, np.dot(lattice,i))
                charge, pos = self.charges[c],self.atoms[c] #+np.dot(i,lattice)
                if self.theta is not None:
                    pos = np.dot(self.rotation_matrix(i[0]), pos) + np.dot(i,lattice)
                else:

                    pos = pos + np.dot(i,lattice)

                #charge, pos = self.charges[c],self.atoms[c]-np.dot(lattice, i)

                Positi.append(pos)
                
                Charges.append(charge)
                

        return np.array(Positi), np.array(Charges)
    
    def get_charges(self):
        #return number of charges
        return(self.charges)

    def set_nocc(self, n_occ = None):
        if n_occ is None:
            self.n_occ = int(np.sum(self.charges)/2) - self.n_core
        else:
            self.n_occ = n_occ

    def get_nocc_all(self):
        # include core orbitals
        return int(np.sum(self.charges)/2)

    
    def get_nocc(self):
        #returns n occupied, rhf
        return self.n_occ 

    def set_nvirt(self, n_virt = None):
        if n_virt is None:
            self.n_virt = self.get_n_ao()-self.get_nocc() -self.n_core
        else:
            self.n_virt = n_virt
        
    def get_nvirt(self):
        #returns n occupied, rhf
        return self.n_virt
        
    def get_n_ao(self):
        N_ao = 0
        for atom in np.arange(len(self.charges)):
            atom_basis = self.get_basis_index(self.charges[atom])
            for shell in np.arange(len(self.basis_set[atom_basis])):
                for contracted in np.arange(len(self.basis_set[atom_basis][shell])):
                    N_ao += 2*shell + 1
        return N_ao
    
    def match_ao_to_atom(self):
        
        #find number of atoms
        pass

    

    
    def coor2vec(self, c):
        return np.dot(c,self.lattice)
        
        
    def _setup_blattice(self):
        basis, geom = ci.dalton2lambdas(self.filename)
        atoms, charges, ltc = np.array(geom[0])/0.52917720830000000000000000000, geom[1], geom[2]
        #atoms, charges, ltc = np.array(geom[0]), geom[1], geom[2]
        lattice = [] #periodic vectors
        self.lsperiodicity = []
        dim =0
        for i in ltc:
            l = i.split()
            self.lsperiodicity.append(l[-1])
            if l[-1] == "active":
                dim += 1
            lattice.append(np.array([literal_eval(v) for v in l[2:-1]]))
        
        if dim==1:
            self.cperiodicity = "POLYMER"
        if dim==2:
            self.cperiodicity = "SLAB"
        if dim==3:
            self.cperiodicity = "CRYSTAL"
        self.lattice = np.array(lattice)
        self.charges = charges
        self.atoms = atoms
        
        bset, self.atomic_numbers = basis
        #basis_set = []

        self.bf_per_atom = {}

        self.basis_set = []
        for natom in np.arange(len(bset)):
            self.basis_set.append([])
            #self.bf_per_atom[self.charges[natom]] = len(bset[natom])
            self.bf_per_atom[self.charges[natom]] = 0
            for ntype in np.arange(len(bset[natom])):
                self.bf_per_atom[self.charges[natom]] += len(bset[natom][ntype])*(2*ntype + 1)
             
                for ncontracted in np.arange(len(bset[natom][ntype])):
                    self.basis_set[-1].append([]) #add list for a contracted
                    for nb in np.arange(len(bset[natom][ntype][ncontracted])):
                        prim = bset[natom][ntype][ncontracted][nb]
                        prim.append(ntype)
                        self.basis_set[-1][-1].append(prim)
    def get_bf_per_atom(self, charge):
        return self.bf_per_atom[charge]
    

        
def parse_crystal_basis(basis_s):
    #print(basis_s)
    basis_n = []
    for line in basis_s.split("\n")[1:-1]:
        basis_n.append([literal_eval(i.replace("d", "e")) for i in line.split()])
    
    c = 0
    basis_set = []
    atomic_numbers = []
    while c<=len(basis_n)-1:
        atom, ncontracted = basis_n[c]
        basis_set.append([])
        atomic_numbers.append(atom)
        for nc in np.arange(ncontracted):
            basis_set[-1].append([])
            nbasis = basis_n[c+1][2]
            btype  = basis_n[c+1][1]

            for nb in np.arange(nbasis):
                if btype == 0:
                    basis_n[c+2+nb].append(btype)
                    basis_set[-1][-1].append(basis_n[c+2+nb])
                if btype == 1:
                    basis_n[c+2+nb].append(0)
                    basis_set[-1][-1].append(basis_n[c+2+nb])
                    
                    basis_n[c+2+nb].append(1)
                    basis_set[-1][-1].append(basis_n[c+2+nb])
                    # Crystal reserves btype 1 for sp-functions, halt conversion
                    print("WARNING: SP-FUNCTION IN BASIS, NOT TESTED")
                    #convert to s and p functions
                    #basis_set[-1][-1].append(basis_n[c+2+nb])
                    
                if btype > 1:
                    basis_n[c+2+nb].append(btype-1)
                    #s = 0, p= 1, d = 2 and so on....
                    basis_set[-1][-1].append(basis_n[c+2+nb])
                
                
                #print(nb, basis_n[c+2+nb])
                
            
            c += nbasis + 1
        c += 1
        
    atomic_numbers = atomic_numbers[:-1]
    basis_set = basis_set[:-1]
    
    return atomic_numbers, basis_set

if __name__ == "__main__":
    print("""#####################################################
##  ,--.   ,--.      ,------.  ,------. ,-----.    ##
##   \  `.'  /,-----.|  .-.  \ |  .---''  .--./    ##
##    .'    \ '-----'|  |  \  :|  `--, |  |        ## 
##   /  .'.  \       |  '--'  /|  `---.'  '--'\    ##
##  '--'   '--'      `------PRISM-----''--- --'    ##
#####################################################""")
    parser = argparse.ArgumentParser(prog = "orb_refinery",
                                        description = "Convert input-files between Crystal-LSDalton",
                                        epilog = "Author: Audun Skau Hansen (e-mail: a.s.hansen@kjemi.uio.no)")
    
    #parser.add_argument("project_folder", type = str, help ="Project folder containing input files.")
    parser.add_argument("-c2lsd", action="store_true", default=False, help="Convert from Crystal to LSDalton.")
    parser.add_argument("-c2g94", action="store_true", default=False, help="Convert from Crystal to LSDalton.")
    parser.add_argument("-infile", type=str, default=False, help="Input file")
    parser.add_argument("-outfile", type=str, default=False, help="Output file")
    
    args = parser.parse_args()
    
    #project_folder = args.project_folder + "/" #folder name only
    
    # Make sure folder exists
    
    #project_folder_f = os.getcwd() + "/" + project_folder #full path
    
    if args.c2lsd:
        print("Converting %s to LSDalton input file." % args.infile)
        #print("")
        P = prism(args.infile)
        
        lsinput = P.get_lsdalton_input()
        
        #sp.call(["mv", project_folder_f + "/LSDalton/MOLECULE.INP", project_folder_f + "/LSDalton/MOLECULE_OLD.INP"])
        
        f = open(args.outfile, "w")
        f.write(lsinput)
        f.close()
    
    if args.c2g94:
        P = prism(args.infile)
        
        lsinput = P.get_libint_basis()
        
        f = open(args.outfile, "w")
        f.write(lsinput)
        f.close()
        print("Basis saved to file %s ." % args.outfile)
        
        
        
