#!/usr/bin/env python

import numpy as np

import numba

import os

import subprocess as sp

from ast import literal_eval

import argparse

import utils.toeplitz as tp

import lwrap.lwrap as li

import utils.prism as pr

import time



#os.environ["LIBINT_DATA_PATH"] = os.getcwd() #"/usr/local/libint/2.5.0-beta.2/share/libint/2.5.0-beta.2/basis/"
#os.environ["CRYSTAL_EXE_PATH"] = "/Users/audunhansen/PeriodicDEC/utils/crystal_bin/"

def basis_trimmer(p, auxbasis, alphacut = 0.5):
    """
    # make basis more suited for periodic calculations
    # this solution is rather ad-hoc, should not be the ultimate way of solving this problem of linear dependencies
    """
    f = open(auxbasis, "r")
    basis = f.readlines()
    trimmed_basis_list = []
    for line in basis:
        try: 
            # We only retain basis functions with exponent > alphacut
            exponent = literal_eval(line.split()[0])
            if  exponent >= alphacut:
                
                trimmed_basis_list.append(line)
            else:
                trimmed_basis_list = trimmed_basis_list[:-1]
                pass

        except:
            trimmed_basis_list.append(line)
    #print(trimmed_basis)
    f.close()
    
    trimmed_basis = ""
    for l in trimmed_basis_list:
        trimmed_basis += l

    return trimmed_basis




def occ_virt_split(c,p):
    """
    Split matrix c(tmat object) in occupied and virtual columns depending on information contained in p (prism object)
    Returns two tmat objects c_occ and c_virt (typically coefficients)
    """
    print("virt0:", p.get_nocc()+p.n_core ) 
    c_virt = tp.tmat()
    c_virt.load_nparray(c.blocks[:-1,:,p.get_nocc()+p.n_core:], c.coords[:], screening = False)

    print("occ:", p.n_core,p.get_nocc()+p.n_core)
    c_occ = tp.tmat()
    c_occ.load_nparray(c.blocks[:-1,:,p.n_core:p.get_nocc()+p.n_core], c.coords[:], screening = False)
    return c_occ, c_virt

def get_xyz(p, t = np.array([[0,0,0]])):
    """
    Generate xyz input file for libint
    Note that Libint automatically converts from angstrom, so we have to preconvert to angstrom
    with on of (unknown which) these factors (from https://github.com/evaleev/libint/blob/master/include/libint2/atom.h)
    0.52917721067
    0.52917721092
    """
    
    pos, charge = p.get_atoms(t)
    ptable = [None, "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
    sret = "%i\n\n" % len(charge)
    #conversion_factor = 0.5291972108
    conversion_factor = 0.5291772109200000 #<.-. this one
    #conversion_factor = 0.52917721067 #**-1
    for i in range(len(charge)):
        sret += "%s %.15f %.15f %.15f\n" % (ptable[int(charge[i])],conversion_factor*pos[i][0], conversion_factor*pos[i][1], conversion_factor*pos[i][2])
    sret = sret[:-2]
    return sret

def compute_pqrs(p, t = np.array([[0,0,0]])):
    """
    Computes integrals of the type ( 0 p 0 q | T r T s)
    for all coordinates T provided in t
    """

    bname = "libint_basis" #libint basis (will be generated)
    #auxname = "cc-pvdz"    #auxiliary basis (must be present)
    
    
    basis = os.environ["LIBINT_DATA_PATH"] + "/%s.g94" % bname
    
    # Save basis
    f = open(basis, "w")
    f.write(p.get_libint_basis())
    f.close()
    
    
    atomsJ = "atoms_J.xyz"
    atomsK = "atoms_K.xyz"
    
    f = open(atomsJ, "w")
    f.write(get_xyz(p))
    f.close()
    
    f = open(atomsK, "w")
    f.write(get_xyz(p, t))
    f.close()    
    
    lint = li.engine()
    
    #if not coulomb:
    #    lint.set_operator_erfc()
    #if coulomb:
    #    lint.set_operator_coulomb()
    
    
    # compute one cell to get dimensions
    
    lint.setup_pqrs(atomsJ, bname, atomsJ, bname, atomsK, bname, atomsK, bname, 0)
    integrals = lint.get_pqrs(atomsJ, bname, atomsJ, bname, atomsK, bname, atomsK, bname)
    
    sp.call(["rm", "-rf", atomsJ])
    sp.call(["rm", "-rf", atomsK])
    sp.call(["rm", "-rf", basis])
    
    return np.array(integrals)

def compute_pqrs_(p, tq = np.array([[0,0,0]]),tr = np.array([[0,0,0]]),ts = np.array([[0,0,0]])):
    """
    Computes integrals of the type ( 0 p Tq q | Tr r Ts s)
    for all coordinates Tn provided in tn
    """

    bname = "libint_basis" #libint basis (will be generated)
    #auxname = "cc-pvdz"    #auxiliary basis (must be present)
    
    
    basis = os.environ["LIBINT_DATA_PATH"] + "/%s.g94" % bname
    
    # Save basis
    f = open(basis, "w")
    f.write(p.get_libint_basis())
    f.close()
    
    
    atomsJ = "atoms_J.xyz"
    atomsK = "atoms_K.xyz"
    atomsL = "atoms_L.xyz"
    atomsM = "atoms_M.xyz"


    
    f = open(atomsJ, "w")
    f.write(get_xyz(p))
    f.close()
    
    f = open(atomsK, "w")
    f.write(get_xyz(p, tq))
    f.close()    

    f = open(atomsL, "w")
    f.write(get_xyz(p, tr))
    f.close()    

    f = open(atomsM, "w")
    f.write(get_xyz(p, ts))
    f.close()    
    
    lint = li.engine()
    
    #if not coulomb:
    #    lint.set_operator_erfc()
    #if coulomb:
    #    lint.set_operator_coulomb()
    
    
    # compute one cell to get dimensions
    
    lint.setup_pqrs(atomsJ, bname, atomsK, bname, atomsL, bname, atomsM, bname, 0)
    integrals = lint.get_pqrs(atomsJ, bname, atomsK, bname, atomsL, bname, atomsM, bname)
    
    sp.call(["rm", "-rf", atomsJ])
    sp.call(["rm", "-rf", atomsK])
    sp.call(["rm", "-rf", atomsL])
    sp.call(["rm", "-rf", atomsM])
    sp.call(["rm", "-rf", basis])
    
    return np.array(integrals)

def compute_pqpq(p, t = np.array([[0,0,0]])):
    """
    Computes integrals of the type ( 0 p 0 q | T p T q)
    for all coordinates T provided in t
    """

    bname = "libint_basis" #libint basis (will be generated)
    #auxname = "cc-pvdz"    #auxiliary basis (must be present)
    
    
    basis = os.environ["LIBINT_DATA_PATH"] + "/%s.g94" % bname
    
    # Save basis
    f = open(basis, "w")
    f.write(p.get_libint_basis())
    f.close()
    
    
    atomsJ = "atoms_J.xyz"
    atomsK = "atoms_K.xyz"
    
    f = open(atomsJ, "w")
    f.write(get_xyz(p))
    f.close()
    
    f = open(atomsK, "w")
    f.write(get_xyz(p, t))
    f.close()    
    
    lint = li.engine()
    
    #if not coulomb:
    #    lint.set_operator_erfc()
    #if coulomb:
    #    lint.set_operator_coulomb()
    
    
    # compute one cell to get dimensions
    
    lint.setup_pqpq(atomsJ, bname, atomsK, bname)
    
    integrals = lint.get_pqpq(atomsJ, bname, atomsK, bname)
    
    sp.call(["rm", "-rf", atomsJ])
    sp.call(["rm", "-rf", atomsK])
    
    return np.array(integrals)





def compute_Jmm(p, s, attenuation = 0.0, auxname = "cc-pvdz", coulomb = False, nshift = np.array([[0,0,0]])):
    """
    Originally intended for a Cauchy-Schwartz-like screening, never implemented
    """
    pass


def compute_Jmn(p, s, attenuation = 0.0, auxname = "cc-pvdz", coulomb = False, nshift = np.array([[0,0,0]])):
    """
    Computes integrals of the type ( T J |0 mu nu)
    for all coordinates T provided in s.coords (tmat)
    """
    
    
    bname = "libint_basis" #libint basis (will be generated)
    #auxname = "cc-pvdz"    #auxiliary basis (must be present)
    
    
    basis = os.environ["LIBINT_DATA_PATH"] + "/%s.g94" % bname

    # Save basis
    f = open(basis, "w")
    f.write(p.get_libint_basis())
    f.close()
    
    
    
    atomsJ = "atoms_J.xyz"
    atomsm = "atoms_m.xyz"
    atomsn = "atoms_n.xyz"
    
    f = open(atomsJ, "w")
    f.write(get_xyz(p))
    f.close()

    
    f = open(atomsm, "w")
    f.write(get_xyz(p))
    f.close()

    #print(nshift)
    #print(get_xyz(p, nshift))
    f = open(atomsn, "w")
    f.write(get_xyz(p, nshift))
    f.close()
    
    
    
    lint = li.engine()
    
    if not coulomb:
        lint.set_operator_erfc()
    if coulomb:
        lint.set_operator_coulomb()
    
    
    # compute one cell to get dimensions
    
    lint.setup_pqr(atomsJ, auxname, atomsm, bname, atomsn, bname, 0)
    if not coulomb:
        lint.set_integrator_params(attenuation)
    
    vint = np.array(lint.get_pqr(atomsJ, auxname, atomsm, bname, atomsn, bname))
    #print(vint.shape)
    
    blockshape = (vint.shape[0], vint.shape[1]*vint.shape[2])
    
    Jmn = tp.tmat()
    #Jmn.load_nparray(np.ones((s.coords.shape[0], blockshape[0], blockshape[1]),dtype = float),  s.coords)
    #Jmn.blocks *= 0.0
    
    
    
    #for coord in s.coords:
    #    #f.write(get_xyz(p, [p.coor2vec(coord)]))
    #    #print(np.array([coord]))
        
        
    f = open(atomsJ, "w")
    
    f.write(get_xyz(p, s.coords))
    f.close()
    
    lint.setup_pqr(atomsJ, auxname, atomsm, bname, atomsn, bname, 0)

    #lint.set_integrator_params(0.2)
    if not coulomb:
        lint.set_integrator_params(attenuation)
    
    vint = np.array(lint.get_pqr(atomsJ, auxname, atomsm, bname, atomsn, bname))
    #print("Jmn computed")

    Jmn.load_nparray(vint.reshape((s.coords.shape[0], blockshape[0], blockshape[1])), s.coords)

    #print("Size: ", Jmn.blockshape, Jmn.blocks.shape)
    #sp.call(["rm", "-rf", atomsJ])
        #print(coord, np.abs(vint).max())
    
    sp.call(["rm", "-rf", atomsn])
    sp.call(["rm", "-rf", atomsm])
    
    sp.call(["rm", "-rf", atomsJ])
    
    return Jmn


def compute_onebody(p,s, T = np.array([[0,0,0]]), operator = "overlap"):
    """
    Computes integrals of the type 
        ( 0 p | O^ | T q)
    for all coordinates T provided in t. 
    The operator O^ is provided as a string, see available operators below
    (More are easily available from libint, needs some small extra functions in lwrap)

    """
    atomsJ = "atoms_J.xyz"
    atomsK = "atoms_K.xyz"
    
    bname = "temp_basis"

    basis = os.environ["LIBINT_DATA_PATH"] + "/%s.g94" % bname
    
    f = open(basis, "w")
    f.write(p.get_libint_basis())
    f.close()
    
    f = open(atomsJ, "w")
    f.write(get_xyz(p))
    f.close()
    
    f = open(atomsK, "w")
    f.write(get_xyz(p, T))
    f.close()    
    
    lint = li.engine()
    
    if operator == "overlap":
        lint.set_operator_overlap()
    if operator == "kinetic":
        lint.set_operator_kinetic()
    if operator == "nuclear":
        lint.set_operator_nuclear()

        
    
    lint.setup_pq(atomsJ, bname, atomsK, bname)
    vint = np.array(lint.get_pq(atomsJ, bname, atomsK, bname))
    return vint
        



def compute_JK(p, s, attenuation = 1, auxname = "cc-pvdz", coulomb = False):
    """
    Computes integrals of the type ( 0 J | T K )
    for all coordinates T provided in t
    attenuation set to 0 is equivalent to the coulomb operator, approaching 
    infinity it tends to the Dirac delta-function.
    
    """
    #auxname = "cc-pvdz"    #auxiliary basis (must be present)

    atomsJ = "atoms_J.xyz"
    atomsK = "atoms_K.xyz"
    
    f = open(atomsJ, "w")
    f.write(get_xyz(p))
    f.close()
    
    f = open(atomsK, "w")
    f.write(get_xyz(p, np.array([[0,0,0]])))
    f.close()    
    
    lint = li.engine()
    
    if not coulomb:
        lint.set_operator_erfc()
    if coulomb:
        lint.set_operator_coulomb()
    
    
    # compute one cell to get dimensions
    
    lint.setup_pq(atomsJ, auxname, atomsK, auxname)
    lint.set_braket_xsxs()
    lint.set_integrator_params(attenuation)
    
    vint = np.array(lint.get_pq(atomsJ, auxname, atomsK, auxname))
    
    blockshape = (vint.shape[0], vint.shape[1])
    
    
    JK = tp.tmat()


    atomsK = "atoms_K.xyz"
    f = open(atomsK, "w")
    f.write(get_xyz(p, s.coords))
    f.close()
    
    lint.setup_pq(atomsJ, auxname, atomsK, auxname)
    lint.set_braket_xsxs()
    if not coulomb:
        lint.set_operator_erfc()
        lint.set_integrator_params(attenuation)
    
    vint = np.array(lint.get_pq(atomsJ, auxname, atomsK, auxname))

    vint = vint.reshape((blockshape[0], s.coords.shape[0], blockshape[1]))
    vint = vint.swapaxes(0,1)

    JK.load_nparray(vint, s.coords)


    return JK

def verify_pqpq(attenuation = 0, coulomb = False):
    bname = "libint_basis" #libint basis (will be generated)
    
    basis = os.environ["LIBINT_DATA_PATH"] + "/%s.g94" % bname
    
    # Save basis
    f = open(basis, "w")
    f.write(p.get_libint_basis())
    f.close()
    
    atomsm = "atoms_m.xyz"
    atomsn = "atoms_n.xyz"
    

    
    f = open(atomsm, "w")
    f.write(get_xyz(p))
    f.close()
    
    f = open(atomsn, "w")
    f.write(get_xyz(p))
    f.close()
    
    
    
    lint = li.engine()
    
    if not coulomb:
        lint.set_operator_erfc()
    if coulomb:
        lint.set_operator_coulomb()
    
    
    # compute one cell to get dimensions
    
    lint.setup_pqpq(atomsm, bname, atomsn, bname) #, bname, atomsn, bname)
    
    lint.set_integrator_params(attenuation)
    
    vint = np.array(lint.get_pqpq(atomsm, bname, atomsn, bname)) #, atomsm, bname, atomsn, bname))
    return vint




def invert_JK_test(JK):
    """
    Obsolete debugging function
    """
    #n_points = 2*np.array(tp.n_lattice(JK))
    JKk = tp.dfft(JK)*1
    JKk_inv = JKk*0
    norm_error = 0
    for k in JKk.coords:
        JKk_k = np.linalg.inv(JKk.cget(k))
        JKk_inv.cset(k, JKk_k)
        # Test each inversioon
        norm_error += np.sqrt(np.sum((np.eye(JKk_k.shape[0]) - np.dot(JKk_k, JKk.cget(k)))**2))
    print(norm_error)
    JK_inv = tp.idfft(JKk_inv)
    return JK_inv

def invert_JK(JK):
    """
    Matrix inversion by means of a Fourier transform
    Note that this is also implemented as a method internally to the tmat class
    in the following way:

       M^-1 = M.inv()

    
    """
    n_points = np.array(tp.n_lattice(JK))
    #print(n_points)
    JKk = tp.transform(JK, np.fft.fftn, n_points = n_points)
    JKk_inv = JKk*1.0
    JKk_inv.blocks[:-1] = np.linalg.inv(JKk.blocks[:-1])



    JK_inv_direct = tp.transform(JKk_inv, np.fft.ifftn, n_points = n_points, complx = False)

    return JK_inv_direct 

#def kspace_contraction(JK_att, Jmn):
#    Jmn_ret = Jmn*0.0

def estimate_attenuation_distance(p, attenuation = 0.1, c2 = [0,0,0], thresh = 10e-12, auxname = "cc-pvdz-ri"):
    """
    For a given attenuation parameter, basis and lattice geometry, estimate
    which blocks contain elements above the provided treshold.
    Returns a tmat object with blocks initialized accordingly

    # THIS IS NOT WELL BEHAVED, CHANGE this part of the code

    """
    for i in np.arange(1,100):
        cube = tp.lattice_coords([i,0,0]) #assumed max twobody AO-extent (subst. C-S Screening)
        
        cube = np.zeros((4,3), dtype = int)
        cube[1] = np.array([i,0,0])
        cube[2] = np.array([0,i,0])
        cube[3] = np.array([0,0,i])
        
        
        big_tmat = tp.tmat()
        #big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)
        big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)
        #print(big_tmat.coords)
        Jmnc = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = [c2])
        #cmax = big_tmat.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))]
        #print(np.sqrt(np.sum(p.coor2vec(c2)**2)), c2, attenuation, cmax, "Largest element in %i th layer:" % i, np.max(np.abs(Jmnc.cget(cmax))))
        if (np.max(np.abs(Jmnc.cget(cube[1:]))))<thresh:
            #
            #i = int(np.sqrt(3*i**2))
            #r
            cmax = np.argmax(np.sqrt(np.sum(np.dot(cube,p.lattice)**2, axis = 1)))
            rmax = np.sqrt(np.sum(np.dot(cube,p.lattice)**2, axis = 1))[cmax]
            #print("Converged to %i layers for shifted coordinate:" % i, c2, cmax, big_tmat.coords[cmax], rmax)
            break
    #print("Converged to %i layers for shifted coordinate:" % i, c2)
    cube = tp.lattice_coords([2*i+1,2*i+1,2*i+1]) #assumed max twobody AO-extent (subst. C-S Screening)
    
    cmax = np.argmax(np.sqrt(np.sum(np.dot(cube,p.lattice)**2, axis = 1)))
    cube = cube[np.sqrt(np.sum(np.dot(cube,p.lattice)**2, axis = 1))<=rmax+.1] #this is not entirely correct, should be in rvec-measure
    #print(cube.shape)
    big_tmat = tp.tmat()
    big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)
    return big_tmat #return expansion region in form of toeplitz matrix
    
def estimate_attenuation_distance_(p, attenuation = 0.1, c2 = [0,0,0], thresh = 10e-12, auxname = "cc-pvdz-ri"):
    """
    For a given attenuation parameter, basis and lattice geometry, estimate
    which blocks contain elements above the provided treshold.
    Returns a tmat object with blocks initialized accordingly


    """
    
    cube = tp.lattice_coords([5,5,5]) #assumed max twobody AO-extent (subst. C-S Screening)
    #cube = cube[np.sqrt(np.sum(np.dot(cube, p.lattice)**2, axis = 1))<=40.0] #this is not entirely correct, should be in rvec-measure
    #print(cube.shape)Coulomb
    big_tmat = tp.tmat()
    big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)
    return big_tmat #return expansion region in form of toeplitz matrix

class coefficient_fitter_static():
    """
    Coefficient fitting for the integrator, with integrals stored in memory
    """

    def __init__(self, c, p, attenuation, auxname, JK, JKInv, screening_thresh = 1e-12, robust = False, circulant = True, jpm_screening = 1e-10):
        cube = tp.lattice_coords([3,3,3]) #assumed max twobody AO-extent (subst. C-S Screening)
        print("WARNING: MINIMAL FITTING DOMAIN")
        cube = tp.lattice_coords([1,1,1])
        #cube = np.array([[-1,0,0],[1,0,0],[0,0,0],[0,1,0],[0,-1,0],[0,0,1], [0,0,-1]])
        self.robust = robust
        self.coords = []
        self.Jmn = []
        self.attenuation = attenuation
        self.screening_thresh = screening_thresh
        self.p = p
        self.c = c
        self.JK = JK
        self.JKInv = JKInv
        self.circulant = circulant
        self.c_occ, self.c_virt = occ_virt_split(self.c,p)
        if self.robust:
            self.Jmnc = []
        #big_tmat = estimate_attenuation_distance_(p, attenuation = attenuation, auxname = auxname, thresh = 1e-5)
                
        for c2 in cube:
            # Compute JMN with nsep =  c2
            
            big_tmat = estimate_attenuation_distance(p, attenuation = attenuation, c2 = c2, auxname = auxname, thresh = screening_thresh)
            
            
            
            #Jmnc2 = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()
            Jmnc2 = compute_Jmn(p,self.JK, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()
            
            if robust:
                Jmnc2_c = compute_Jmn(p,big_tmat, attenuation = 0.0, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()


                
            if np.max(np.abs(Jmnc2.blocks))>screening_thresh:
                print("Jmn fit:", c2, np.max(np.abs(Jmnc2.blocks)), Jmnc2.coords.shape[0])
                self.coords.append(c2)
                self.Jmn.append(Jmnc2) #make sure it is uppercase-transposed (due to how it is computed efficiently in libint)
                if self.robust:
                    self.Jmnc.append(Jmnc2_c)
        self.coords = np.array(self.coords)
        self.c0 = np.argwhere(np.all(self.coords==np.array([0,0,0]), axis = 1))[0][0]
        
        self.Jmnc_tensors = []
        self.Jmnc_screening = []
        

        t0 = time.time()
        #jpm_screening = 1e-10
        for i in np.arange(self.coords.shape[0]):

            Jmnc2 = self.Jmn[i]


            NL = Jmnc2.coords.shape[0]
            NJ = Jmnc2.blockshape[0] # Number of aux functions
            Np = self.c_occ.blockshape[1] # Number of occupieds
            Nq = self.c_virt.blockshape[1] # Number of virtuals
            Nn = self.c_occ.blockshape[0] # Number of AO functions



            self.Jmnc_tensors.append([])
            self.Jmnc_screening.append([])

            for coord in np.arange(self.c.coords.shape[0]):
                #tj = np.einsum("LJmn,Lmp->JpLn", Jmnc2.blocks[:-1].reshape(NL, NJ,Nn,Nn ), self.c_occ.cget(-Jmnc2.coords + self.c.coords[coord]), optimize = True).reshape(NJ*Np, NL*Nn)
                
                tj = np.einsum("LJmn,Lmp->JpLn", Jmnc2.blocks[:-1].reshape(NL, NJ,Nn,Nn ), self.c_occ.cget(-Jmnc2.coords - self.c.coords[coord]), optimize = True).reshape(NJ*Np, NL*Nn)
                
                
                screen = np.max(np.abs(tj), axis = 0)>jpm_screening
                
                self.Jmnc_tensors[-1].append(tj[:, screen])


                #print("tj sparsity  :", np.sum(np.abs(tj)>1e-10), tj.size)
                #print("tj b-sparsity:", np.sum(np.max(np.abs(tj)>1e-10, axis = 0)), tj.shape)
                self.Jmnc_screening[-1].append(screen)
        print("Contraction of occupieds:", time.time()-t0)
            
            



    def get(self, coords_q):
        c_occ, c_virt = self.c_occ, self.c_virt
        c = self.c
        JK = self.JK
        Jpq_c = []
        Jpq_c_coulomb = []

        Jpq_coords = self.Jmn[self.c0].coords
 
        
        for i in np.arange(coords_q.shape[0]):

            Jpq_c.append(tp.tmat())
            Jpq_c[-1].load_nparray(np.ones((Jpq_coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  Jpq_coords)
            Jpq_c[-1].blocks *= 0
            #print(np.max(np.abs(Jpq_c[-1].blocks)))
            if self.robust:
                Jpq_c_coulomb.append(tp.tmat())
                Jpq_c_coulomb[-1].load_nparray(np.ones((c.coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  c.coords)
                Jpq_c_coulomb[-1].blocks *= 0

        
        for i in np.arange(self.coords.shape[0]):
            c2 = self.coords[i]

            Jmnc2 = self.Jmn[i]

            NL = Jmnc2.coords.shape[0]
            NJ = Jmnc2.blockshape[0] # Number of aux functions
            Np = c_occ.blockshape[1] # Number of occupieds
            Nq = c_virt.blockshape[1] # Number of virtuals
            Nn = c_occ.blockshape[0] # Number of AO functions
            

            #Jp_Ln = tp.tmat()
            #Jp_Ln.load_nparray(np.zeros((NL, NJ*Np, Nn*NL)))

            #tensors = []

            #for coord in np.arange(self.c.coords.shape[0]):
            #    tensors.append(np.einsum("LJmn,Lmp->JpLn", Jmnc2.blocks[:-1].reshape(NL, NJ,Nn,Nn ), c_occ.cget(-Jmnc2.coords + c.coords[coord]), optimize = True))
            tensors = self.Jmnc_tensors[i]
            screening = self.Jmnc_screening[i]
            #t0 = time.time()
            for j in np.arange(coords_q.shape[0]):
                for coord in np.arange(c.coords.shape[0]):
                    #block_j = np.einsum("JpLn,Lnq->Jpq", tensors[coord], c_virt.cget(-Jmnc2.coords + c.coords[coord] + coords_q[j]), optimize = True)
                    screen = screening[coord]
                    block_j = np.dot(tensors[coord], c_virt.cget(-Jmnc2.coords - c.coords[coord] + coords_q[j] - c2).reshape(NL*Nn, Nq)[screen,:]).reshape(NJ, Np, Nq)
                    Jpq_c[j].blocks[Jpq_c[j].mapping[ Jpq_c[j]._c2i(c.coords[coord]) ]] +=  block_j.reshape(NJ, Np*Nq)
            #print("Contraction of virtuals:", time.time()-t0)
        X = []
        for i in np.arange(len(Jpq_c)):
            t0 = time.time()
            if self.circulant:
                #X.append(self.JKInv.circulantdot(Jpq_c[i]))
                #X.append(self.JK.kspace_linear_solve(Jpq_c[i]))
                #X.append(self.JK.kspace_cholesky_solve(Jpq_c[i]))
                X.append(self.JK.kspace_svd_solve(Jpq_c[i]))
            
            else:
                X.append(self.JKInv.cdot(Jpq_c[i]))
            print("Solving for Jpqc:", time.time()-t0)

        return X

    def get__(self, coord_q):
        # Earlier attempt
        c_occ, c_virt = self.c_occ, self.c_virt
        #print(self.c.blocks.shape)
        #print(c_occ.blocks.shape)
        #print(c_virt.blocks.shape)
        c = self.c
        JK = self.JK
        Jpq_c = []
        Jpq_c_coulomb = []

        Jpq_coords = self.Jmn[self.c0].coords
        #print("Jpq_0     :", self.coords[self.c0])
        #print("Jpq_length:", Jpq_coords.shape[0], c_virt.coords.shape[0], c_occ.coords.shape[0])

        
        for i in np.arange(coord_q.shape[0]):

            Jpq_c.append(tp.tmat())
            Jpq_c[-1].load_nparray(np.ones((Jpq_coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  Jpq_coords)
            Jpq_c[-1].blocks *= 0
            #print(np.max(np.abs(Jpq_c[-1].blocks)))
            if self.robust:
                Jpq_c_coulomb.append(tp.tmat())
                Jpq_c_coulomb[-1].load_nparray(np.ones((c.coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  c.coords)
                Jpq_c_coulomb[-1].blocks *= 0

        
        for i in np.arange(self.coords.shape[0]):
            c2 = self.coords[i]

            Jmnc2 = self.Jmn[i] .T()

            # Transform (RJ|mn) -> (RJ|pq)

            NL = Jmnc2.coords.shape[0]
            NJ = Jmnc2.blockshape[0] # Number of aux functions
            Np = c_occ.blockshape[1] # Number of occupieds
            Nq = c_virt.blockshape[1] # Number of virtuals
            Nn = c_occ.blockshape[0] # Number of AO functions
            

            # This can be done once and for all when setting up integrals

            t0 = time.time()
             
            LJn_m = tp.tmat()

            LJn_m_blocks = Jmnc2.blocks[:-1].reshape(NL, NJ, Nn, Nn)

            #assert(np.linalg.norm(LJn_m[1,1,:,:].T - LJn_m[1,1,:,:])<10e-10), "break"
            print(np.linalg.norm(LJn_m_blocks[1,1,:,:].T - LJn_m_blocks[1,1,:,:]))
            LJn_m_blocks = LJn_m_blocks.swapaxes(2,3)

            LJn_m_blocks = LJn_m_blocks.reshape(NL, NJ*Nn, Nn)

            LJn_m.load_nparray(LJn_m_blocks, Jmnc2.coords)

            LJn_p = LJn_m.circulantdot(c_occ).T() #transpose or not?

            t1 = time.time()

            NL = LJn_p.coords.shape[0]

            # Reorganize, prepare for virtuals
            LJp_n = tp.tmat()

            LJp_n_blocks = LJn_p.blocks[:-1].reshape(NL, NJ, Nn, Np)

            LJp_n_blocks = LJp_n_blocks.swapaxes(2,3)

            LJp_n_blocks = LJp_n_blocks.reshape(NL, NJ*Np, Nn)

            LJp_n.load_nparray(LJp_n_blocks, LJn_p.coords)
            
            #LJp_m = LJp_m.T()

            for j in np.arange(coord_q.shape[0]):

                c_virt_trans = c_virt*1.0

                c_virt_trans.blocks[:-1] = c_virt.cget(c_virt.coords + c2 - coord_q[j])

                LJp_q = LJp_n.circulantdot(c_virt_trans) # transpose or not?
                
                t2 = time.time()

                #reorganize, finish it

                LJ_pq = tp.tmat()

                LJ_pq_blocks = LJp_q.blocks[:-1]
                LJ_pq_blocks = LJ_pq_blocks.reshape(NL, NJ, Np*Nq)

                LJ_pq.load_nparray(LJ_pq_blocks, LJp_q.coords)

                #LJ_pq = LJ_pq.T()

                Jpq_c[j] = Jpq_c[j] + LJ_pq.T()
                print("Jpq_c.blocks.shape:", Jpq_c[j].blocks.shape)

                t3 = time.time()

                print("Time spent on contracting occupieds:", t1-t0)
                print("Time spent on contracting virtuals :", t2-t1)
                print("Time spent on rearrangement        :", t3-t2)



        X = []
        for i in np.arange(len(Jpq_c)):
            if self.circulant:
                #X.append(self.JKInv.circulantdot(Jpq_c[i]))
                #X.append(self.JK.kspace_linear_solve(Jpq_c[i]))
                #X.append(self.JK.kspace_cholesky_solve(Jpq_c[i]))
                X.append(self.JK.kspace_svd_solve(Jpq_c[i]))
                
            else:
                X.append(self.JKInv.cdot(Jpq_c[i]))
        
        if self.robust:
            X_c = []
            for i in np.arange(len(Jpq_c_coulomb)):
                X_c.append(Jpq_c_coulomb[i])
            X = [X, X_c]
        return X

    def get_(self, coord_q):
        c_occ, c_virt = self.c_occ, self.c_virt
        #print(self.c.blocks.shape)
        #print(c_occ.blocks.shape)
        #print(c_virt.blocks.shape)
        c = self.c
        JK = self.JK
        Jpq_c = []
        Jpq_c_coulomb = []

        Jpq_coords = self.Jmn[self.c0].coords
        #print("Jpq_0     :", self.coords[self.c0])
        #print("Jpq_length:", Jpq_coords.shape[0], c_virt.coords.shape[0], c_occ.coords.shape[0])

        
        for i in np.arange(coord_q.shape[0]):

            Jpq_c.append(tp.tmat())
            Jpq_c[-1].load_nparray(np.ones((Jpq_coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  Jpq_coords)
            Jpq_c[-1].blocks *= 0
            #print(np.max(np.abs(Jpq_c[-1].blocks)))
            if self.robust:
                Jpq_c_coulomb.append(tp.tmat())
                Jpq_c_coulomb[-1].load_nparray(np.ones((c.coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  c.coords)
                Jpq_c_coulomb[-1].blocks *= 0

        
        for i in np.arange(self.coords.shape[0]):
            c2 = self.coords[i]
            #print(i, c2)

            Jmnc2 = self.Jmn[i]
            if self.robust:
                Jmnc2_c = self.Jmnc[i]

            #print("Norm ")

        
            # We go for maximum vectorization here, for some reason it changes the results ever so slightly - it should not, but it is still far below the FOT.
            
            #if i==self.c0:


            # Transform (RJ|mn) -> (RJn|m)


            """
            Jpnc2 = Jmnc2*1.0
            print(Jpnc2.blockshape, Jpnc2.blocks.shape)
            Jpnc2.blocks = Jpnc2.blocks.reshape(Jpnc2.coords.shape[0]+1, self.JK.blockshape[0],self.c_occ.blockshape[0], self.c_occ.blockshape[0])
            Jpnc2.blocks = np.swapaxes(Jpnc2.blocks, 2,3) #R,J,n,m
            Jpnc2.blocks = Jpnc2.blocks.reshape(Jpnc2.coords.shape[0]+1, self.JK.blockshape[0]*self.c_occ.blockshape[0], self.c_occ.blockshape[0]) #R, Jn, m

            Jpnc2 = Jpnc2.circulantdot(c_occ) #R,Jn,p

            Jpnc2.blocks = Jpnc2.blocks.reshape(Jpnc2.coords.shape[0]+1, self.JK.blockshape[0],self.c_occ.blockshape[0], self.c_occ.blockshape[1]) #R,J,n,p
            Jpnc2.blocks = np.swapaxes(Jpnc2.blocks, 2,3) #R,J,p,n

            Jpnc2.blocks = Jpnc2.blocks.reshape(Jpnc2.coords.shape[0]+1, self.JK.blockshape[0]*self.c_occ.blockshape[1], self.c_occ.blockshape[0]) #R,jp,n
            """
            #Contract virtuals
            




            #Jmnc2_.blocks = np.einsum("RJmn", Jmnc2_.blocks)




            for j in np.arange(coord_q.shape[0]):
                #t = time.time()
                C2bo = c_occ.cget(c_virt.coords) # Kmp
                C2vi = c_virt.cget(c_virt.coords - c2 + coord_q[j]) #Knq
                #C2bo = c_occ.cget(-c_virt.coords) # Kmp
                #C2vi = c_virt.cget(-c_virt.coords + c2 - coord_q[j]) #Knq
                 

                #print(C2bo.max())
                #print(C2vi.max())
                
                C2b = np.einsum("Kmp,Knq->Kmnpq", C2bo, C2vi, optimize = True).reshape(c_virt.coords.shape[0], c_occ.blockshape[0]**2, c_occ.blockshape[1]*c_virt.blockshape[1])
                
            
                # build 
                #t0 = time.time()
                #for k in np.arange(Jmnc2.coords.shape[0]):
                #    bl = np.tensordot(Jmnc2.cget(c_virt.coords + Jmnc2.coords[k]), C2b, axes = ([0,2], [0,1]))
                #    Jpq_c[j].cset(Jmnc2.coords[k], Jpq_c[j].cget(Jmnc2.coords[k]) + bl)
                #print(time.time() - t0)

                #print("CCX", C2b.shape)
                
                if np.abs(C2b).max()>10e-10:
                    
                    #print(C2b.shape, np.abs(C2b).max())
                    C2 = tp.tmat()
                    C2.load_nparray(C2b, c_virt.coords, screening = False)
                    #print("C2max", np.max(np.abs(C2.coords), axis = 0))
                    #print("Jmnc2", np.max(np.abs(Jmnc2.coords), axis = 0))

                    #print("mx:", C2.cget([0,0,0]).max())
                    #print("mx:", C2.cget([1,0,0]).max())
                    #print("mx:", C2.blocks.max())
                    



                    #print("t:", time.time()-t)

                    #Cmnpq000 = C2.cget([0,0,0]).reshape(c_occ.blockshape[0],c_occ.blockshape[0],c_occ.blockshape[1],c_virt.blockshape[1])
                    #print(c2, Cmnpq000.max(),c_occ.cget([0,0,0]).max(), c_virt.cget([0,0,0]).max())
                    #for ii in np.arange(2):
                    #    for aa in np.arange(9):
                    #        print("verif:", ii, aa, Cmnpq000[0,0,ii,aa], c_occ.cget([0,0,0])[0,ii]*c_virt.cget([0,0,0])[0,aa])
                        
                

                    
                    #print(c_occ.)
                    #print(C2.blocks.shape)
                    #print(c2, Jmnc2.blocks.shape)
                    #t = time.time()
                    #print(Jmnc2.blocks.shape, C2.blocks.shape)
                    #print(Jmnc2.coords.shape, C2.coords.shape)
                    #print(np.max(np.abs(Jmnc2.coords), axis = 0))
                    #print(np.max(np.abs(C2.coords), axis = 0))
                    #n_points = np.max(np.array([tp.n_lattice(Jmnc2), tp.n_lattice(C2)]), axis = 0) + 1
                    #print(n_points)

                    #self_k = tp.transform(Jmnc2, np.fft.fftn) #, n_points = n_points)
                    #other_k = tp.transform(C2, np.fft.fftn) #, n_points = n_points)


                    #Jpq_c[j] = Jpq_c[j] + Jmnc2%C2


                    
                    try:
                        #print("circ")
                        #t = time.time()

                        # Original

                        contrib = Jmnc2.circulantdot(C2)

                        #contrib = Jpnc2.circulantdot(c_virt.T())

                        #Jpc2.blocks = Jpc2.blocks.reshape(Jpc2.coords.shape[0]+1, self.JK.blockshape[0], self.c_occ.blockshape[1]*self.c_virt.blockshape[1]) #R,jp,n
            



                        #print(np.abs((contrib-Jpc2).blocks).max())

                        # Test

                        #contrib = Jmnc2.T().circulantdot(C2)


                        #contrib = Jmnc2%C2
                        #print("Max contrib", np.max(np.abs(contrib.blocks)))
                        #print(c2, coord_q[j])
                        Jpq_c[j] = Jpq_c[j] + contrib
                        #print("Timing:", time.time()-t)

                        # testfunc
                        #t = time.time()
                        #JJJ = Jpnc2.circulantdot(c_virt) #R,jp,q
                        #JJJ.blocks = JJJ.blocks.reshape(Jpnc2.coords.shape[0]+1, self.JK.blockshape[0],self.c_occ.blockshape[1]*self.c_virt.blockshape[1]) #R,j,pq
                        #print("Timing:", time.time()-t)
                        


                    except:
                        print("direct")
                        Jpq_c[j] = Jpq_c[j] + Jmnc2.cdot(C2) #Why doesn't circulantdot work here?
                    #print("Norm  :", np.linalg.norm(JJJ.cget([0,0,0]) - Jpq_c[j].cget([0,0,0])))
                    #print(JJJ.cget([0,0,0])[0])
                    #print( Jpq_c[j].cget([0,0,0])[0] )
                    #print(Jpq_c[j].cget([0,0,0]))
                    #print("t:", time.time()-t)
                    assert(np.max(np.abs(Jpq_c[j].blocks[-1]))==0.0),"error: %.2f" % np.max(np.abs(Jpq_c[j].blocks[-1]))

                    """
                    for R in Jmnc2.coords:
                        #print(c2, Jpq_c[j].blocks.shape, np.linalg.norm(Jmnc2.blocks))
                        #print(np.linalg.norm(Jmnc2.cget([1000,1000,1000])))
                        #print(np.linalg.norm(c_occ.cget([1000,1000,1000])))
                        #print(np.linalg.norm(c_virt.cget([1000,1000,1000])))


                        contrib = np.einsum("Kjmn,Kmk,Knl->Kjkl", \
                                Jmnc2.cget(-Jpq_coords + R).reshape(Jpq_coords.shape[0], \
                                    JK.blockshape[0],c_occ.blockshape[0],c_virt.blockshape[0]),\
                                        c_occ.cget(Jpq_coords), c_virt.cget(Jpq_coords + c2 + coord_q[j]), \
                                            optimize = True).reshape(\
                                                (Jpq_coords.shape[0],JK.blockshape[0],\
                                                    c_occ.blockshape[1]*c_virt.blockshape[1]))
                        #print(np.max(np.abs(contrib)))
                        #print(np.max(np.abs(c_occ.cget(Jpq_coords))))
                        #print(np.max(np.abs(c_virt.cget(Jpq_coords + c2 + coord_q[j]))))
                        #print(np.max(np.abs(Jmnc2.cget(Jpq_coords))))

                        #print("---")
                        Jpq_c[j].blocks[:-1] = Jpq_c[j].cget(-Jpq_coords) + contrib
                    """

                    """
                    Jpq_c[j].blocks[:-1] = Jpq_c[j].cget(Jpq_coords) + \
                        np.einsum("Kjmn,Kmk,Knl->Kjkl", \
                            Jmnc2.cget(Jpq_coords).reshape(Jpq_coords.shape[0], \
                                JK.blockshape[0],c_occ.blockshape[0],c_virt.blockshape[0]),\
                                    c_occ.cget(Jpq_coords), c_virt.cget(Jpq_coords + c2 + coord_q[j]), \
                                        optimize = True).reshape(\
                                            (Jpq_coords.shape[0],JK.blockshape[0],\
                                                c_occ.blockshape[1]*c_virt.blockshape[1]))
                    """
                    if self.robust:
                        Jpq_c_coulomb[j].blocks[:-1] = Jpq_c_coulomb[j].cget(c.coords) + np.einsum("Kjmn,Kmk,Knl->Kjkl", Jmnc2_c.cget(c.coords).reshape(c.coords.shape[0], JK.blockshape[0],c_occ.blockshape[0],c_virt.blockshape[0]), c_occ.cget(c.coords), c_virt.cget(c.coords + c2 + coord_q[j]), optimize = True).reshape((c.coords.shape[0],JK.blockshape[0],c_occ.blockshape[1]*c_virt.blockshape[1]))
        
        X = []
        for i in np.arange(len(Jpq_c)):
            if self.circulant:
                #X.append(self.JKInv.circulantdot(Jpq_c[i]))
                #X.append(self.JK.kspace_linear_solve(Jpq_c[i]))
                #X.append(self.JK.kspace_cholesky_solve(Jpq_c[i]))
                X.append(self.JK.kspace_svd_solve(Jpq_c[i]))
                
            else:
                X.append(self.JKInv.cdot(Jpq_c[i]))
        
        if self.robust:
            X_c = []
            for i in np.arange(len(Jpq_c_coulomb)):
                X_c.append(Jpq_c_coulomb[i])
            X = [X, X_c]
        return X

    def get_old(self, coord_q):
        c_occ, c_virt = self.c_occ, self.c_virt
        c = self.c
        JK = self.JK
        Jpq_c = []
        Jpq_c_coulomb = []

        
        for i in np.arange(coord_q.shape[0]):

            Jpq_c.append(tp.tmat())
            Jpq_c[-1].load_nparray(np.ones((c.coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  c.coords)
            Jpq_c[-1].blocks *= 0
            if self.robust:
                Jpq_c_coulomb.append(tp.tmat())
                Jpq_c_coulomb[-1].load_nparray(np.ones((c.coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  c.coords)
                Jpq_c_coulomb[-1].blocks *= 0
        for i in np.arange(self.coords.shape[0]):
            c2 = self.coords[i]

            Jmnc2 = self.Jmn[i]
            if self.robust:
                Jmnc2_c = self.Jmnc[i]

        
            # We go for maximum vectorization here, for some reason it changes the results ever so slightly - it should not, but it is still far below the FOT.
            for j in np.arange(coord_q.shape[0]):
                
                Jpq_c[j].blocks[:-1] = Jpq_c[j].cget(c.coords) + \
                    np.einsum("Kjmn,Kmk,Knl->Kjkl", \
                         Jmnc2.cget(c.coords).reshape(c.coords.shape[0], \
                             JK.blockshape[0],c_occ.blockshape[0],c_virt.blockshape[0]),\
                                  c_occ.cget(c.coords), c_virt.cget(c.coords + c2 + coord_q[j]), \
                                      optimize = True).reshape(\
                                          (c.coords.shape[0],JK.blockshape[0],\
                                              c_occ.blockshape[1]*c_virt.blockshape[1]))
                   
                if self.robust:
                    Jpq_c_coulomb[j].blocks[:-1] = Jpq_c_coulomb[j].cget(c.coords) + np.einsum("Kjmn,Kmk,Knl->Kjkl", Jmnc2_c.cget(c.coords).reshape(c.coords.shape[0], JK.blockshape[0],c_occ.blockshape[0],c_virt.blockshape[0]), c_occ.cget(c.coords), c_virt.cget(c.coords + c2 + coord_q[j]), optimize = True).reshape((c.coords.shape[0],JK.blockshape[0],c_occ.blockshape[1]*c_virt.blockshape[1]))
            

        X = []
        for i in np.arange(len(Jpq_c)):
            if self.circulant:
                #X.append(self.JKInv.circulantdot(Jpq_c[i]))
                #X.append(self.JK.kspace_linear_solve(Jpq_c[i]))
                X.append(self.JK.kspace_cholesky_solve(Jpq_c[i]))
                #X.append(self.JK.kspace_svd_solve(Jpq_c[i]))

            else:
                X.append(self.JKInv.cdot(Jpq_c[i]))
        
        if self.robust:
            X_c = []
            for i in np.arange(len(Jpq_c_coulomb)):
                X_c.append(Jpq_c_coulomb[i])
            X = [X, X_c]
        return X

        


            





def compute_fitting_coeffs(c,p,coord_q = np.array([[0,0,0]]), attenuation = 0.1, auxname = "cc-pvdz-ri", JKmats = None,robust = False, circulant = False):
    """
    Perform a least-squares type fit of products of functions expanded in gaussians
    """
    cube = tp.lattice_coords([2,2,2]) #assumed max twobody AO-extent (subst. C-S Screening)
    #print("C-S like screening set to extent [ 2,2,2]")
    if JKmats is None:
        # build JK and inverse
        big_tmat = estimate_attenuation_distance(p, attenuation = .5*attenuation, thresh = 10e-14, auxname = auxname)
        cmax = big_tmat.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))]
        print("JK outer max (should be small):", cmax, np.max(np.abs(JK.cget(cmax))))

        #print(cmax,c.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))])
        #print(big_tmat.coords)
        #JK = compute_JK(p,big_tmat, attenuation = attenuation, auxname = auxname)
        JK = compute_JK(p,big_tmat, attenuation = attenuation, auxname = auxname)

        
        
        
        JKinv = invert_JK(JK)
        print("Condition:", np.abs(JK.blocks).max(), np.abs(JKinv.blocks).max())
        # test inversion
        tcoords = np.zeros((np.max(JK.coords),3), dtype = int)
        tcoords[:,0] = np.arange(JK.coords.max(), dtype = int)
        I = JKinv.cdot(JK, coords = tcoords )
        #I = JKinv.circulantdot(JK)
        print("Direct space inversion test (0,0,0):", np.max(np.abs(I.cget([0,0,0])-np.eye(I.blockshape[0]))))
        for cc in tcoords[1:]:
            print("Direct space inversion test (%i,0,0):" % cc[0], np.max(np.abs(I.cget(cc))))
        print("---")
        
        I = JKinv.circulantdot(JK)
        print("Circulant inversion test (0,0,0):", np.max(np.abs(I.cget([0,0,0])-np.eye(I.blockshape[0]))))
        for cc in tcoords[1:]:
            print("Circulant inversion test (%i,0,0):" % cc[0], np.max(np.abs(I.cget(cc))))
        print("---")
    else:
        JK, JKinv = JKmats
    ## -> Perform inversion product in reciprocal space OR oversample the attenuated matrix

    
    
    c_occ, c_virt = occ_virt_split(c,p)
    #print(c_occ.blockshape, c_virt.blockshape)

    # Remove core orbitals
    #p.n_core = 1
    #c_occ = tp.tmat()
    #c_occ.load_nparray(c_occ_full.blocks[:-1, :, 1:], c_occ_full.coords)

    Jpq_c = []
    Jpq_c_coulomb = []

    
    for i in np.arange(coord_q.shape[0]):

        Jpq_c.append(tp.tmat())
        Jpq_c[-1].load_nparray(np.ones((c.coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  c.coords)
        Jpq_c[-1].blocks *= 0
        if robust:
            Jpq_c_coulomb.append(tp.tmat())
            Jpq_c_coulomb[-1].load_nparray(np.ones((c.coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  c.coords)
            Jpq_c_coulomb[-1].blocks *= 0

    
    for c2 in cube:
        # Compute JMN with nsep =  c2
        
        big_tmat = estimate_attenuation_distance(p, attenuation = attenuation, c2 = c2, auxname = auxname)
        
        
        Jmnc2 = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()
        if robust:
            Jmnc2_c = compute_Jmn(p,big_tmat, attenuation = 0.0, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()

        cmax = big_tmat.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))]
               
        
        if np.max(np.abs(Jmnc2.blocks))>1e-12:
            # We go for maximum vectorization here, for some reason it changes the results ever so slightly - it should not, but it is still far below the FOT.
            
            
            for i in np.arange(coord_q.shape[0]):
                Jpq_c[i].blocks[:-1] = Jpq_c[i].cget(c.coords) + np.einsum("Kjmn,Kmk,Knl->Kjkl", Jmnc2.cget(c.coords).reshape(c.coords.shape[0], JK.blockshape[0],c_occ.blockshape[0],c_virt.blockshape[0]), c_occ.cget(c.coords), c_virt.cget(c.coords + c2 + coord_q[i]), optimize = True).reshape((c.coords.shape[0],JK.blockshape[0],c_occ.blockshape[1]*c_virt.blockshape[1]))
                if robust:
                    Jpq_c_coulomb[i].blocks[:-1] = Jpq_c_coulomb[i].cget(c.coords) + np.einsum("Kjmn,Kmk,Knl->Kjkl", Jmnc2_c.cget(c.coords).reshape(c.coords.shape[0], JK.blockshape[0],c_occ.blockshape[0],c_virt.blockshape[0]), c_occ.cget(c.coords), c_virt.cget(c.coords + c2 + coord_q[i]), optimize = True).reshape((c.coords.shape[0],JK.blockshape[0],c_occ.blockshape[1]*c_virt.blockshape[1]))
                #print(Jpqc.shape, Jpq_c[i].blocks.shape)
            

            
            """
            for c1 in c.coords:
                #@numba.autojit(nopython= True)
                for i in np.arange(coord_q.shape[0]):
                    
                    #Jpqc = Jpq.cget(c1) + np.einsum("jmn,mk,nl->jkl", Jmnc2.cget(c1).reshape((70,11,11)), c.cget(c1), c.cget(c2), optimize = True).reshape((70,11**2))
                    Jpqc = Jpq_c[i].cget(c1) + np.einsum("jmn,mk,nl->jkl", Jmnc2.cget(c1).reshape((JK.blockshape[0],c_occ.blockshape[0],c_virt.blockshape[0])), c_occ.cget(c1), c_virt.cget(c1 + c2 + coord_q[i]), optimize = True).reshape((JK.blockshape[0],c_occ.blockshape[1]*c_virt.blockshape[1]))

                    Jpq_c[i].cset(c1, Jpqc)
                    #nrm = np.linalg.norm(Jpq_c[i].cget(c1))
                    #if nrm>10e-10:
                    #    print(c1 - c2 + coord_q[i], c1, c2, coord_q[i])
                    #    print(np.linalg.norm(Jpqc))
                    #    print("Max element in shifted matrix ", c2, " is ", np.max(np.abs(Jmnc2.blocks)))
            """
        #print(c2, " done.")
    

    
    

    #print(np.sum(np.linalg.norm(JKinv.blocks, axis = 0)>10e-10))
    JKinv = tp.screen_tmat(JKinv)
    #print("Compute coeffs", JKinv.blocks.shape, JK.blocks.shape, Jpq.blocks.shape)

    X = []
    for i in np.arange(len(Jpq_c)):
        #print("Compute coeffs for shifted coordinate", coord_q[i])
        if circulant:
            X.append(JKinv.circulantdot(Jpq_c[i]))
        else:
            X.append(JKinv.cdot(Jpq_c[i]))
    
    if robust:
        X_c = []
        for i in np.arange(len(Jpq_c_coulomb)):
            #print("Compute coeffs for shifted coordinate", coord_q[i])
            X_c.append(Jpq_c_coulomb[i])
        X = [X, X_c]
    #X = JKinv.cdot(Jpq, coords = Jpq.coords) #compute coefficients
    #print("done")
    return X

def test_matrix_kspace_condition(M, n_fourier):
    Mk = tp.dfft(M, n_fourier)*1
    #Mk_inv = M*0
    for k in Mk.coords:
        Mk_e, Mk_v = np.linalg.eig(Mk.cget(k))
        print(k, np.abs(Mk_e).max()/np.abs(Mk_e).min(),np.linalg.det(Mk.cget(k)), np.max(np.abs(Mk.cget(k).real)))
        print()


class integral_builder_ao():
    """
    This class i used when selecting the attenuation parameter only
    For direct access to integrals use either lwrap directly, or some
    of the functions defined above (like compute_pqpq)
    """
    def __init__(self, c, p, attenuation = 0.1, auxname = "cc-pvdz-ri", circulant = False, robust  = False):
        self.c = c
        self.p = p
        self.attenuation = attenuation
        self.auxname = auxname
        self.circulant = circulant
        self.robust = robust
    def getcell(self, dL, M, dM):
        p = self.p
        pqMrs = compute_pqrs(p, t = np.array([M])).reshape(p.get_n_ao(), p.get_n_ao(),p.get_n_ao(),p.get_n_ao())
        return pqMrs[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():]
    def nbytes(self):
        return 0




class integral_builder_static():
    """
    RI-integral builder with stored AO-integrals
    For high performance (but high memory demand) 
    """
    def __init__(self, c,p, attenuation = 0.1, auxname = "cc-pvdz-ri", initial_virtual_dom = [1,1,1], circulant = False, extent_thresh = 1e-14, robust  = False, ao_screening = 1e-12, inverse_test = True, coulomb_extent = None, JKa_extent = None, jpm_screening = 1e-10):
        self.c = c
        self.p = p
        self.attenuation = attenuation
        self.auxname = auxname
        self.circulant = circulant
        self.robust = robust

        # Oneshot calculations:
        # build attenuated JK matrix and inverse
        #big_tmat = estimate_attenuation_distance(p, attenuation = .5*self.attenuation, thresh = 1e-14, auxname = auxname)
        big_tmat = estimate_attenuation_distance(p, attenuation = self.attenuation, thresh = extent_thresh, auxname = auxname)
        
        cmax = big_tmat.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))]
        cmax = np.argmax(np.sqrt(np.sum(np.dot(big_tmat.coords,p.lattice)**2, axis = 1)))
        if JKa_extent is not None:
            big_tmat = tp.tmat()
            scoords = tp.lattice_coords(JKa_extent)
            big_tmat.load_nparray(np.ones((scoords.shape[0],2,2), dtype = float), scoords)

        #self.JKa = compute_JK(self.p,self.c, attenuation = attenuation, auxname = auxname)
        self.JKa = compute_JK(self.p,big_tmat, attenuation = attenuation, auxname = auxname)
        
        print("")
        print("Attenuated coulomb matrix (JKa) computed.")
        print("JKa outer coordinate (should be smaller than %.2e):" % extent_thresh, cmax, big_tmat.coords[cmax], np.max(np.abs(self.JKa.cget(self.JKa.coords[cmax]))))
        
        
        #assert(np.max(np.abs(self.JKa.cget(self.JKa.coords[cmax])))<=extent_thresh), "JKa outer coordinate (should be smaller than %.2e):" % extent_thresh
        
        #print(np.max(self.JKa.coords, axis = 0))
        #print(np.min(self.JKa.coords, axis = 0))
        #print(self.JKa.coords)



        print("Number of auxiliary basis functions (in supercell):", self.JKa.blocks[:-1].shape[0]*self.JKa.blocks[:-1].shape[1])
        #print("JKa block shape:", self.JKa.blocks[:-1].shape)
        print("")
    
        self.JKinv = invert_JK(self.JKa)
        #inverse_test = True
        if inverse_test:
            print("JKa inverse computed, checking max deviation from 0 = JKa^-1 JKa - I within extent")


            tcoords = np.zeros((np.max(self.JKa.coords),3), dtype = int)
            tcoords[:,0] = np.arange(self.JKa.coords.max(), dtype = int)
            I = self.JKinv.cdot(self.JKa, coords = tcoords )

            print("Direct space inversion (0,0,0): %.3e" % np.max(np.abs(I.cget([0,0,0])-np.eye(I.blockshape[0]))))
            for cc in tcoords[1:]:
                print("Direct space inversion (%i,0,0): %.3e" % (cc[0], np.max(np.abs(I.cget(cc)))))


            I = self.JKinv.circulantdot(self.JKa)
            print("Circulant inversion    (0,0,0): %.3e" % np.max(np.abs(I.cget([0,0,0])-np.eye(I.blockshape[0]))))
            for cc in tcoords[1:]:
                print("Circulant inversion    (%i,0,0): %.3e" % (cc[0], np.max(np.abs(I.cget(cc)))))
            print(" ")

        
        self.XregT = np.zeros((15,15,15), dtype = tp.tmat)  # RI - coefficient matrices ^T
        self.VXreg = np.zeros((15,15,15), dtype = tp.tmat) # RI - matrices with V contracted
        if robust:
            self.JpqXreg = np.zeros((15,15,15), dtype = tp.tmat)

        # initial coeffs computed in single layer around center cell
        coord_q =  tp.lattice_coords(initial_virtual_dom) #initial virtual domain
        print("Computing fitting coefficients for dL = ")
        print(coord_q)
        #if robust:
        #    t0 = time.time()
        #    Xreg, Jpq = compute_fitting_coeffs(self.c,self.p,coord_q = coord_q, attenuation = self.attenuation, auxname = self.auxname, JKmats = [self.JKa, self.JKinv], robust = True, circulant = self.circulant)
        #    t1 = time.time() - t0
        #else:
        #   t0 = time.time()
        #    Xreg = compute_fitting_coeffs(self.c,self.p,coord_q = coord_q, attenuation = self.attenuation, auxname = self.auxname, JKmats = [self.JKa, self.JKinv])
        #    t1 = time.time() - t0
        #print("Time spent on fitting %i cells: %.2f (s)" % (len(coord_q), t1))
        #print("Number of auxiliary functions in use:", Xreg[0].blocks[:-1].shape[0]*Xreg[0].blocks[:-1].shape[1])
        
        print("Coeff fitter static tresh set to 1e-8")
        t0 = time.time()
        self.cfit = coefficient_fitter_static(c, p, attenuation, auxname, self.JKa, self.JKinv, screening_thresh = ao_screening, robust = robust, circulant = circulant, jpm_screening=jpm_screening)
        t1 = time.time()
        print("Spent %.1f s preparing fitting (three index) integrals." % (t1-t0))
        if self.robust:
            Xreg, Jpq = self.cfit.get(coord_q)
        else:
             Xreg = self.cfit.get(coord_q)
        t2 = time.time()
        print("Spent %.1f s computing fitting coefficient for %i coords" % (t2-t1, len(coord_q)))


        for i in np.arange(coord_q.shape[0]):
            #print("Transpose of coordinate", coord_q[i])
            self.XregT[coord_q[i][0], coord_q[i][1],coord_q[i][2]] = Xreg[i].tT()
        if robust:
            for i in np.arange(coord_q.shape[0]):
                #print("Transpose of coordinate", coord_q[i])
                self.JpqXreg[coord_q[i][0], coord_q[i][1],coord_q[i][2]] = Jpq[i]


        if coulomb_extent is None:
            # Compute JK_coulomb
            coulomb_extent = np.max(np.abs(self.XregT[0,0,0].coords), axis = 0) 
            #coulomb_extent = np.max(np.abs(self.JKa[0,0,0].coords), axis = 0)  
            print("Extent of Xreg          :", coulomb_extent)

            #coulomb_extent = (10,10,10)
            print("Extent of Coulomb matrix:", coulomb_extent)
            
        s = tp.tmat()
        scoords = tp.lattice_coords(coulomb_extent)
        s.load_nparray(np.ones((scoords.shape[0],2,2), dtype = float), scoords)
            
        self.JK = compute_JK(p,s, coulomb=True, auxname = self.auxname)
        #self.JK = compute_JK(p,big_tmat, coulomb=True, auxname = self.auxname)

        print("Coulomb matrix (JK) computed.")


        for i in np.arange(coord_q.shape[0]):
            #print("JK dot X for cell ", coord_q[i])

            if circulant:
                self.VXreg[coord_q[i][0], coord_q[i][1],coord_q[i][2]]= self.JK.circulantdot(Xreg[i])
            else:
                self.VXreg[coord_q[i][0], coord_q[i][1],coord_q[i][2]]= self.JK.cdot(Xreg[i])

    def getorientation(self, dL, dM):
        for d in [dL, dM]:
            #print(d)
            if self.XregT[d[0], d[1], d[2]] is 0:
                
                #Xreg, Jpq = self.cfit.get(np.array([d]))
                Xreg = self.cfit.get(np.array([d]))
                self.XregT[d[0], d[1], d[2]] = Xreg[0].tT()
                #self.JpqXreg[d[0], d[1], d[2]] = Jpq[0]
                if self.circulant:
                    self.VXreg[d[0], d[1], d[2]] =  self.JK.circulantdot(Xreg[0])
                else:
                    self.VXreg[d[0], d[1], d[2]] =  self.JK.cdot(Xreg[0])
                
                print("        On-demand calculation:", d)
        if self.robust:
            print("Robust orientation not yet implemented")
            return None
        else:
            #print("Return:")
            if self.circulant:
                return self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]]), \
                    (self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())
            else:
                return self.XregT[dL[0], dL[1], dL[2]].dot(self.VXreg[dM[0], dM[1], dM[2]]), \
                    (self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())

    def getcell(self, dL, M, dM):
        if self.robust:
            circulant = self.circulant
            for d in [dL, dM]:
                if self.XregT[d[0], d[1], d[2]] is 0:

                    #coords_q = tp.lattice_coords()


                    # Should compute all within range
                    #Xreg, Jpq = compute_fitting_coeffs(self.c,self.p,coord_q = np.array([d]), attenuation = self.attenuation, auxname = self.auxname, JKmats = [self.JKa, self.JKinv], robust = True)
                    Xreg, Jpq = self.cfit.get(np.array([d]))
                    self.XregT[d[0], d[1], d[2]] = Xreg[0].tT()
                    self.JpqXreg[d[0], d[1], d[2]] = Jpq[0]
                    if circulant:
                        self.VXreg[d[0], d[1], d[2]] =  self.JK.circulantdot(Xreg[0])
                    else:
                        self.VXreg[d[0], d[1], d[2]] =  self.JK.cdot(Xreg[0])
                    
                    print("        On-demand calculation:", d)
                    #self.Xreg[d[0], d[1], d[2]] =  self.XregT[d[0], d[1], d[2]].tT() #transpose matrix

            if circulant:
                return (self.JpqXreg[dL[0], dL[1], dL[2]].tT().circulantdot(self.XregT[dM[0], dM[1], dM[2]].tT()) + \
                    self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.JpqXreg[dM[0], dM[1], dM[2]]) - \
                    self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]])).cget(M).reshape(self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())

            else:
                #return self.XregT[dL[0], dL[1], dL[2]].cdot(self.VXreg[dM[0], dM[1], dM[2]], coords = [M]).cget(M).reshape(self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())
                return (self.JpqXreg[dL[0], dL[1], dL[2]].tT().cdot(self.XregT[dM[0], dM[1], dM[2]].tT(), coords = [M]) + \
                    self.XregT[dL[0], dL[1], dL[2]].cdot(self.JpqXreg[dM[0], dM[1], dM[2]], coords = [M]) - \
                    self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]])).cget(M).reshape(self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())

        
        else:
            circulant = self.circulant
            for d in [dL, dM]:
                if self.XregT[d[0], d[1], d[2]] is 0:

                    #coords_q = tp.lattice_coords()
                    t0 = time.time() 
                    Xreg = self.cfit.get(np.array([d]))
                    t1 = time.time() 


                    # Should compute all within range
                    #Xreg = compute_fitting_coeffs(self.c,self.p,coord_q = np.array([d]), attenuation = self.attenuation, auxname = self.auxname, JKmats = [self.JKa, self.JKinv])
                    self.XregT[d[0], d[1], d[2]] = Xreg[0].tT()
                    if circulant:
                        self.VXreg[d[0], d[1], d[2]] =  self.JK.circulantdot(Xreg[0])
                    else:
                        self.VXreg[d[0], d[1], d[2]] =  self.JK.cdot(Xreg[0])
                    t2 = time.time()
                    print("        On-demand calculation:", d, "(%.1f + %.1f s)" % (t1-t0, t2-t1))
                    #self.Xreg[d[0], d[1], d[2]] =  self.XregT[d[0], d[1], d[2]].tT() #transpose matrix

            if circulant:
                return self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]]).cget(M).reshape(self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())

            else:
                return self.XregT[dL[0], dL[1], dL[2]].cdot(self.VXreg[dM[0], dM[1], dM[2]], coords = [M]).cget(M).reshape(self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())

    def nbytes(self):
        # Return memory usage of all arrays in instance
        total_mem_usage = 0.0
        for i in np.arange(self.VXreg.shape[0]):
            for j in np.arange(self.VXreg.shape[1]):
                for k in np.arange(self.VXreg.shape[2]):
                    if type(self.XregT[i,j,k]) is tp.tmat:
                        total_mem_usage += self.XregT[i,j,k].blocks.nbytes
                    if type(self.VXreg[i,j,k]) is tp.tmat:
                        total_mem_usage += self.VXreg[i,j,k].blocks.nbytes
        return total_mem_usage*1e-6 #return in MB









if __name__ == "__main__":
    os.environ["LIBINT_DATA_PATH"] = os.getcwd() 
    print("""#####################################################
##  ,--.   ,--.      ,------.  ,------. ,-----.    ##
##   \  `.'  /,-----.|  .-.  \ |  .---''  .--./    ##
##    .'    \ '-----'|  |  \  :|  `--, |  |        ## 
##   /  .'.  \       |  '--'  /|  `---.'  '--'\    ##
##  '--'   '--'      PRI : Periodic RI framework   ##
##                Author : Audun Skau Hansen       ##
##                                                 ##
##  Use keyword "--help" for more info             ## 
#####################################################""")

        
    # Parse input
    parser = argparse.ArgumentParser(prog = "Periodic Resolution of Identity framework",
                                     description = "Fitting of periodic two-body interaction matrix using Block Toeplitz matrices.",
                                     epilog = "Author: Audun Skau Hansen (e-mail: a.s.hansen@kjemi.uio.no)")
    parser.add_argument("project_file", type = str, help ="input file for project (.d12 file)")
    parser.add_argument("coefficient_matrix", type= str,help="Block Toeplitz matrix with coefficients")
    parser.add_argument("auxbasis", type = str, help="Auxiliary fitting basis.")
    parser.add_argument("-attenuation", type = float, default = 0.2)
    parser.add_argument("-test_ao", default = False, action = "store_true", help="Run test for AO-basis")
    parser.add_argument("-test_ibuild", default = False, action = "store_true", help="Test integral builder")
    parser.add_argument("-basis_truncation", type = float, default = 0.5, help = "Truncate AO-basis function below this threshold." )
    parser.add_argument("-attenuated_truncation", type = float, default = 1e-14, help = "Truncate blocks in the attenuated matrix where (max) elements are below this threshold." )
    args = parser.parse_args()

    p = pr.prism(args.project_file)
    
    auxbasis = basis_trimmer(p, args.auxbasis, alphacut = args.basis_truncation)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    
    c = tp.tmat()
    c.load(args.coefficient_matrix)

    #print(p.get_libint_basis())
    #print(c.cget([0,0,0]))
    virtual_region = tp.lattice_coords([1,1,1])
    if args.test_ao:
        #c.blocks *= 0
        #c.cset([0,0,0], np.eye(11))
        #c.cset([1,0,0], .5*np.eye(11))

        virtual_region = tp.lattice_coords([0,0,0])


    if args.test_ibuild:
        print("Testing integral builder")
        ib = integral_builder(c,p,attenuation = args.attenuation, auxname="ri-fitbasis")
        # generating integrals
        print(ib.getcell([0,0,0],[0,0,0],[0,0,0]).shape)
        print(ib.getcell([0,0,0],[0,0,1],[0,0,0]).shape)
        print(ib.getcell([0,0,0],[0,0,2],[0,0,0]).shape)
        print(ib.getcell([0,0,0],[0,0,3],[0,0,0]).shape)


    X = compute_fitting_coeffs(c,p, attenuation = args.attenuation, auxname = "ri-fitbasis", coord_q=virtual_region)
    print(X)
    #print(compute_onebody(p,c))

    Xreg = np.zeros((3,3,3), dtype = object)
    for cc in np.arange(virtual_region.shape[0]):
        i,j,k = virtual_region[cc]
        Xreg[i,j,k] = X[cc]
        
    if args.test_ao:
        # calculate some regions and compare to ao-integrals
        coulomb_extent = np.max(np.abs(Xreg[0,0,0].coords), axis = 0)
        #print(coulomb_extent)

        s = tp.tmat()
        scoords = tp.lattice_coords(coulomb_extent)
        s.load_nparray(np.ones((scoords.shape[0],2,2), dtype = float), scoords)

        JK = compute_JK(p,s, coulomb=True, auxname = "ri-fitbasis") 
        JK.tolerance = 10e-12

        JKX = np.zeros_like(Xreg)

        #print(JK.circulantdot(Xreg[0,0,0]))
        for i in np.arange(len(Xreg)):
            for j in np.arange(len(Xreg[i])):
                for k in np.arange(len(Xreg[i,j])):
                    try:
                        JKX[i,j,k] = JK.cdot(Xreg[i,j,k])
                        #JKX[i,j,k] = JK.circulantdot(Xreg[i,j,k])
                        print(i,j,k,"partially computed.")
                    except:
                        print(i,j,k,"skipped.")
                        pass
        for i in np.arange(10):
            print(p.get_nocc(), p.get_nvirt())

            #try:
            xjkx = Xreg[0,0,0].tT().cdot(JKX[0,0,0], coords = [[i,0,0]]).cget([i,0,0]).reshape((p.get_nocc(),p.get_nvirt(),p.get_nocc(),p.get_nvirt()))
            #xjkx = Xreg[0,0,0].tT().circulantdot(JKX[0,0,0]).cget([i,0,0]).reshape((p.get_nocc(),p.get_nvirt(),p.get_nocc(),p.get_nvirt()))

            #pqrs = compute_pqrs(p, t = [[i,0,0]]).reshape([p.get_nocc(),p.get_nvirt(),p.get_nocc(),p.get_nvirt()])
            pqrs = compute_pqrs(p, t = [[i,0,0]]).reshape((p.get_n_ao(), p.get_n_ao(),p.get_n_ao(),p.get_n_ao()))[:p.get_nocc(),p.get_nocc():,:p.get_nocc(),p.get_nocc():]

            for pp in np.arange(p.get_nocc()):
                for qq in np.arange(p.get_nvirt()):
                    print("  " , pp,qq,pp,qq, "   %.6e   %.6e   %.6e" % (xjkx[pp,qq,pp,qq], pqrs[pp,qq,pp,qq], np.abs(xjkx[pp,qq,pp,qq] - pqrs[pp,qq,pp,qq])))
                    #for rr in np.arange(p.get_nocc()):
                    #    for ss in np.arange(p.get_nvirt()):
                    #        print("  " , pp,qq,rr,ss, "   %.4e   %.4e   %.4e" % (xjkx[pp,qq,rr,ss], pqrs[pp,qq,rr,ss], np.abs(xjkx[pp,qq,rr,ss] - pqrs[pp,qq,rr,ss])))
            print("Max deviation:" , np.abs(xjkx - pqrs).max())



                #print(xjkx[:2,np.arange(2, 11),:2, np.arange(2,11)])  # = (000i000a|100j100b)
                #print(pqrs[:2,np.arange(2, 11),:2, np.arange(2,11)])
                #print(np.linalg.norm(xjkx - pqrs))

            print("   ", )
            #except:
            #    print(i,0,0,"outside region (try except).")


    else:
        np.save("fitted_coeffs", Xreg)
        print("Fitting complete, stored to file.")
