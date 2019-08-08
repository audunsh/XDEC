#!/usr/bin/env python

import numpy as np

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

def basis_trimmer(p, auxbasis):
    # make basis more suited for periodic calculations
    # this solution is quite ad-hoc, should not be the ultimate way of solving this problem of linear dependencies
    f = open(auxbasis, "r")
    basis = f.readlines()
    trimmed_basis_list = []
    for line in basis:
        try: 
            if literal_eval(line.split()[0]) >= 0.1:
                trimmed_basis_list.append(line)
            else:
                print("Basis trimmer removed function from basis:")
                print(line)
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
    # returns two matrices with occupied and virtual space separately

    c_virt = tp.tmat()
    c_virt.load_nparray(c.blocks[:,:,p.get_nocc():], c.coords[:])

    c_occ = tp.tmat()
    c_occ.load_nparray(c.blocks[:,:,:p.get_nocc()], c.coords[:])
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
    conversion_factor = 0.52917721092**-1
    for i in range(len(charge)):
        sret += "%s %.15f %.15f %.15f\n" % (ptable[int(charge[i])],conversion_factor*pos[i][0], conversion_factor*pos[i][1], conversion_factor*pos[i][2])
    sret = sret[:-2]
    return sret

def compute_pqrs(p, t = np.array([[0,0,0]])):

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

def compute_pqpq(p, t = np.array([[0,0,0]])):

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
    """
    lint.set_braket_xsxs()
    lint.set_integrator_params(attenuation)
    
    vint = np.array(lint.get_pq(atomsJ, auxname, atomsK, auxname))
    #print(vint.shape)
    #print(vint)
    
    blockshape = (vint.shape[0], vint.shape[1])
    
    
    JK = tp.tmat()
    JK.load_nparray(np.ones((s.coords.shape[0], blockshape[0], blockshape[1]),dtype = float),  s.coords)
    
    
    count = 0
    for coord in s.coords:
        atomsK = "atoms_K%i.xyz" % count
        f = open(atomsK, "w")
        f.write(get_xyz(p, [p.coor2vec(coord)]))
        f.close()
        
        lint.setup_pq(atomsJ, auxname, atomsK, auxname)
        lint.set_braket_xsxs()
        if not coulomb:
            lint.set_integrator_params(attenuation)
        
        vint = np.array(lint.get_pq(atomsJ, auxname, atomsK, auxname))
        
        
        JK.cset(coord, vint.reshape((blockshape[0], blockshape[1])))
        count += 1
        #print(coord, np.abs(vint).max())
        sp.call(["rm", "-rf", "atoms_K%i.xyz" % count])
    return JK
    """




def compute_Jmm(p, s, attenuation = 0.0, auxname = "cc-pvdz", coulomb = False, nshift = np.array([[0,0,0]])):
    pass

def compute_Jmn_(p, s, attenuation = 0.0, auxname = "cc-pvdz", coulomb = False, nshift = np.array([[0,0,0]])):
    
    
    
    
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
    print(vint.shape)
    
    blockshape = (vint.shape[0], vint.shape[1]*vint.shape[2])
    
    Jmn = tp.tmat()
    Jmn.load_nparray(np.ones((s.coords.shape[0], blockshape[0], blockshape[1]),dtype = float),  s.coords)
    Jmn.blocks *= 0.0
    
    
    for coord in s.coords:
        #f.write(get_xyz(p, [p.coor2vec(coord)]))
        #print(np.array([coord]))
        
        
        f = open(atomsJ, "w")
        
        f.write(get_xyz(p, np.array([coord])))
        f.close()
        
        lint.setup_pqr(atomsJ, auxname, atomsm, bname, atomsn, bname, 0)
    
        #lint.set_integrator_params(0.2)
        if not coulomb:
            lint.set_integrator_params(attenuation)
        
        vint = np.array(lint.get_pqr(atomsJ, auxname, atomsm, bname, atomsn, bname))
        
        Jmn.cset(coord, vint.reshape((blockshape[0], blockshape[1])))
        sp.call(["rm", "-rf", atomsJ])
        #print(coord, np.abs(vint).max())
    
    sp.call(["rm", "-rf", atomsn])
    sp.call(["rm", "-rf", atomsm])
    
    sp.call(["rm", "-rf", atomsJ])
    
    return Jmn

def compute_Jmn(p, s, attenuation = 0.0, auxname = "cc-pvdz", coulomb = False, nshift = np.array([[0,0,0]])):
    
    
    
    
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
    #print(vint.shape)
    #print(vint)
    
    blockshape = (vint.shape[0], vint.shape[1])
    
    
    JK = tp.tmat()
    #JK.load_nparray(np.ones((s.coords.shape[0], blockshape[0], blockshape[1]),dtype = float),  s.coords)
    #JK.blocks *= 0
    #JK.blocks *= 0
    #print("Shape (blocks/coords):", JK.blocks.shape, JK.coords.shape)
    #count = 0

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


    """
    for coord in s.coords:
        atomsK = "atoms_K%i.xyz" % count
        f = open(atomsK, "w")
        f.write(get_xyz(p, [coord]))
        f.close()
        
        lint.setup_pq(atomsJ, auxname, atomsK, auxname)
        lint.set_braket_xsxs()
        if not coulomb:
            lint.set_operator_erfc()
            lint.set_integrator_params(attenuation)
        
        vint = np.array(lint.get_pq(atomsJ, auxname, atomsK, auxname))
        
        
        JK.cset(coord, vint.reshape((blockshape[0], blockshape[1])))
        
        #print(coord, np.abs(vint).max())
        sp.call(["rm", "-rf", "atoms_K%i.xyz" % count])
        count += 1
    """
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

def invert_JK_(JK):
    n_points = 2*np.array(tp.n_lattice(JK))  #+ 1
    JKk = tp.transform(JK, np.fft.fftn, n_points = n_points)
    #JKk = tp.dfft(JK)*1
    #JKk = tp.transform(JK, )
    JKk_inv = JKk*1
    for k in JKk.coords:
        JKk_k = np.linalg.inv(JKk.cget(k))
        JKk_inv.cset(k, JKk_k)
        # Test each inversioon
        #print(k, np.abs(np.sum((np.eye(JKk_k.shape[0]) - np.dot(JKk_k, JKk.cget(k)))**2)))
    #JK_inv = tp.idfft(JKk_inv)
    JK_inv = tp.transform(JK, np.fft.ifftn, complx = False, n_points = n_points)
    return JK_inv


def invert_JK_test(JK):
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
    n_points = np.array(tp.n_lattice(JK))
    print(n_points)
    JKk = tp.transform(JK, np.fft.fftn, n_points = n_points)
    JKk_inv = JKk*1.0
    JKk_inv.blocks[:-1] = np.linalg.inv(JKk.blocks[:-1])

    #for k in JKk.coords:
    #    JKk_k = np.linalg.inv(JKk.cget(k))
    #    
    #    JKk_inv.cset(k, JKk_k)

    JK_inv_direct = tp.transform(JKk_inv, np.fft.ifftn, n_points = n_points, complx = False)

    return JK_inv_direct 

#def kspace_contraction(JK_att, Jmn):
#    Jmn_ret = Jmn*0.0

def estimate_attenuation_distance(p, attenuation = 0.1, c2 = [0,0,0], thresh = 10e-10, auxname = "cc-pvdz-ri"):
    
    for i in np.arange(1,100):
        cube = tp.lattice_coords([i,0,0]) #assumed max twobody AO-extent (subst. C-S Screening)
        big_tmat = tp.tmat()
        big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)

        Jmnc = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = [c2])
        cmax = big_tmat.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))]
        #print("Largest element in %i th layer:" % i, np.max(np.abs(JK.cget(cmax))))
        if (np.max(np.abs(Jmnc.cget(cmax))))<thresh:
            print("Converged to %i layers for shifted coordinate:" % i, c2)
            break
    cube = tp.lattice_coords([i,i,i]) #assumed max twobody AO-extent (subst. C-S Screening)
    cube = cube[np.sqrt(np.sum(cube**2, axis = 1))<=i]
    big_tmat = tp.tmat()
    big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)
    return big_tmat #return expansion region in form of toeplitz matrix
    


#s2 = s.cdot(s)
def compute_fitting_coeffs(c,p,coord_q = np.array([[0,0,0]]), attenuation = 0.1, auxname = "cc-pvdz-ri"):
    cube = tp.lattice_coords([1,1,1]) #assumed max twobody AO-extent (subst. C-S Screening)

    big_tmat = estimate_attenuation_distance(p, attenuation = attenuation, thresh = 10e-12, auxname = auxname)
    cmax = big_tmat.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))]


    JK = compute_JK(p,big_tmat, attenuation = attenuation, auxname = auxname)

    
    
    print("JK outer max (should be small):", cmax, np.max(np.abs(JK.cget(cmax))))
    JKinv = invert_JK(JK)
    print("Condition:", np.abs(JK.blocks).max(), np.abs(JKinv.blocks).max())
    # test inversion
    tcoords = np.zeros((np.max(JK.coords),3), dtype = int)
    tcoords[:,0] = np.arange(JK.coords.max(), dtype = int)
    I = JKinv.cdot(JK, coords = tcoords )
    #I = JKinv.circulantdot(JK)
    print("Inversion test (0,0,0):", np.max(np.abs(I.cget([0,0,0])-np.eye(I.blockshape[0]))))
    for cc in tcoords[1:]:
        print("Inversion test (%i,0,0):" % cc[0], np.max(np.abs(I.cget(cc))))
    print("---")
    
    I = JKinv.circulantdot(JK)
    print("Circulant inversion test (0,0,0):", np.max(np.abs(I.cget([0,0,0])-np.eye(I.blockshape[0]))))
    for cc in tcoords[1:]:
        print("Circulant inversion test (%i,0,0):" % cc[0], np.max(np.abs(I.cget(cc))))
    print("---")

    
    
    
    

    #print(JK.cget([s.coords]))
    #Jmn = compute_Jmn(p,s2, attenuation = attenuation, auxname = "cc-pvdz-ri", coulomb = False)
    #print(Jmn.cget([0,0,0]).shape)
    
    c_occ, c_virt = occ_virt_split(c,p)
    c_occ = c
    c_virt = c

    Jpq_c = []
    for i in np.arange(coord_q.shape[0]):

        Jpq_c.append(tp.tmat())
        Jpq_c[-1].load_nparray(np.ones((c.coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  c.coords)
        Jpq_c[-1].blocks *= 0
    #print(Jpq.blocks.shape)

    for c2 in cube:
        # Compute JMN with nsep =  c2
        #print(c2)
        #print(get_xyz(p,[c2]))

        # Use s2 here?
        
        big_tmat = estimate_attenuation_distance(p, attenuation = attenuation, c2 = c2, auxname = auxname)
        
        
        Jmnc2 = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()
        #print("Jmnc2")
        
        cmax = big_tmat.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))]
        #print("Max element in shifted matrix ", c2, " is ", np.max(np.abs(Jmnc2.blocks)))
        #print("Jmnc2 outer max (should be small):", cmax, np.max(np.abs(Jmnc2.cget(cmax))))
        
        #print(get_xyz(p, [c2]))
        # C-S like screening
        #Jmnc2_est = 
        
        if np.max(np.abs(Jmnc2.blocks))>1e-12:
            
            for c1 in c.coords:
                for i in np.arange(coord_q.shape[0]):
                    #Jpqc = Jpq.cget(c1) + np.einsum("jmn,mk,nl->jkl", Jmnc2.cget(c1).reshape((70,11,11)), c.cget(c1), c.cget(c2), optimize = True).reshape((70,11**2))
                    Jpqc = Jpq_c[i].cget(c1) + np.einsum("jmn,mk,nl->jkl", Jmnc2.cget(c1).reshape((JK.blockshape[0],c_occ.blockshape[0],c_virt.blockshape[0])), c_occ.cget(c1), c_virt.cget(c1 - c2 - coord_q[i]), optimize = True).reshape((JK.blockshape[0],c_occ.blockshape[1]*c_virt.blockshape[1]))

                    Jpq_c[i].cset(c1, Jpqc)
        #print(c2, " done.")
    

    
    

    #print(np.sum(np.linalg.norm(JKinv.blocks, axis = 0)>10e-10))
    JKinv = tp.screen_tmat(JKinv)
    #print("Compute coeffs", JKinv.blocks.shape, JK.blocks.shape, Jpq.blocks.shape)

    X = []
    for i in np.arange(len(Jpq_c)):
        print("Compute coeffs for shifted coordinate", coord_q[i])
        X.append(JKinv.cdot(Jpq_c[i]))
    #X = JKinv.cdot(Jpq, coords = Jpq.coords) #compute coefficients
    print("done")
    return X

def test_matrix_kspace_condition(M, n_fourier):
    Mk = tp.dfft(M, n_fourier)*1
    #Mk_inv = M*0
    for k in Mk.coords:
        Mk_e, Mk_v = np.linalg.eig(Mk.cget(k))
        print(k, np.abs(Mk_e).max()/np.abs(Mk_e).min(),np.linalg.det(Mk.cget(k)), np.max(np.abs(Mk.cget(k).real)))
        print()

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
    
    args = parser.parse_args()

    p = pr.prism(args.project_file)
    
    auxbasis = basis_trimmer(p, args.auxbasis)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    
    c = tp.tmat()
    c.load(args.coefficient_matrix)

    #print(p.get_libint_basis())
    #print(c.cget([0,0,0]))
    #c.blocks *= 0
    #c.cset([0,0,0], np.eye(11))

    virtual_region = tp.lattice_coords([1,1,1])

    X = compute_fitting_coeffs(c,p, attenuation = args.attenuation, auxname = "ri-fitbasis", coord_q=virtual_region)
    print(X)
    #print(compute_onebody(p,c))

    Xreg = np.zeros((3,3,3), dtype = object)
    for cc in np.arange(virtual_region.shape[0]):
        i,j,k = virtual_region[cc]
        Xreg[i,j,k] = X[cc]

    np.save("fitted_coeffs", Xreg)
    print("Fitting complete, stored to file.")