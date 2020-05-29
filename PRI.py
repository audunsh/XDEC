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

def basis_trimmer(p, auxbasis, alphacut = 0.1):
    """
    # Trim basis by removal of primitives with exponent < 0.1
    # Based on "rule-of-thumb" proposed in Crystal-manual
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
    f.close()

    trimmed_basis = ""
    for l in trimmed_basis_list:
        trimmed_basis += l

    return trimmed_basis






def occ_virt_split(c,p, n = None):
    """
    Split matrix c(tmat object) in occupied and virtual columns depending on information contained in p (prism object)
    Returns two tmat objects c_occ and c_virt (typically coefficients)
    if n is not None, orb space is split at index n (first virtual)
    """
    if n is None:
        c_virt = tp.tmat()
        c_virt.load_nparray(c.cget(c.coords)[:,:,p.get_nocc_all():], c.coords, screening = False)

        c_occ = tp.tmat()
        c_occ.load_nparray(c.cget(c.coords)[:,:,p.n_core:p.get_nocc_all()], c.coords, screening = False)
    else:
        c_virt = tp.tmat()
        c_virt.load_nparray(c.cget(c.coords)[:,:,n:], c.coords, screening = False)

        c_occ = tp.tmat()
        c_occ.load_nparray(c.cget(c.coords)[:,:,:n], c.coords, screening = False)




    return c_occ, c_virt

def get_xyz(p, t = np.array([[0,0,0]]), conversion_factor = 0.5291772109200000):
    """
    Generate xyz input file for libint
    Note that Libint automatically converts from angstrom, so we have to preconvert to angstrom
    with one of (unknown which) these factors (from https://github.com/evaleev/libint/blob/master/include/libint2/atom.h)
    0.52917721067
    0.52917721092 <- this one, confirmed in Libint source
    """

    pos, charge = p.get_atoms(t)
    ptable = [None, "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
    sret = "%i\n\n" % len(charge)

    for i in range(len(charge)):
        sret += "%s %.15f %.15f %.15f\n" % (ptable[int(charge[i])],conversion_factor*pos[i][0], conversion_factor*pos[i][1], conversion_factor*pos[i][2])
    sret = sret[:-2]
    return sret

"""
Mapping functions
"""
def mapgen(vex1, vex2):
    """
    returns a matrix where vex1[ ret_indx[i,j] ] = vex1[i] - vex2[j]
    """
    D = vex1[:, None] - vex2[None, :]
    ret_indx = -1*np.ones((vex1.shape[0], vex2.shape[0]), dtype = int)
    for i in np.arange(vex1.shape[0]):
        for j in np.arange(vex2.shape[0]):
            try:
                ret_indx[i,j] = np.argwhere(np.all(np.equal(vex1, D[i,j]), axis = 1))
            except:
                pass
    return ret_indx

def intersection3s(coords1, coords2):
    """
    Returns the intersection of 3-vectors between two sets of (N,3) arrays
    """
    distance = np.sum((coords1[:,None] - coords2[None,:])**2, axis = 2)

    coords1_in_coords2 = np.any(distance==0, axis = 0)
    return coords2[coords1_in_coords2]

def c2_not_in_c1(c1, c2, sort = False):
    """
    Returns the 3-vectors in c2 which are not present in c1
    """
    distance = np.sum((c1[:,None] - c2[None,:])**2, axis = 2)

    c1_in_c2 = np.any(distance==0, axis = 0)
    if not sort:
        return c2[c1_in_c2==False]
    if sort:
        c2_ret = c2[c1_in_c2==False]
        c2_ret = c2_ret[np.argsort(np.sum(c2_ret**2, axis = 1))]
        return c2_ret

"""
Integral calculations
The functions in this section computes the integrals required for the Periodic Resolution-of-Identity approximation

Overview:
compute_pqrs        - Computes the two-body (AO) integrals ( 0 p 0 q | T r T s) for a range of cells in T
compute_pqrs_       - Computes the two-body (AO) integrals ( 0 p Tq q | Tr r Ts s) for a range of cells in (0,Tq,Tr,Ts)
compute_pqpq        - Computes the two-body (AO) integrals ( 0 p 0 q | T p T q) for a range of cells in T
compute_Jmn         - Computes the three-center (AO) integrals ( T J |0 mu nu) for a range of cells T
compute_onebody     - Computes onebody integrals ( 0 p | O^ | T q ) where O^ is an optional operator
compute_JK          - Computes two center coulomb ( 0 J | O^ | T K ) where O^ is the coulomb operator
verify_pqpq


"""

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
    f.write(get_xyz(p, nshift))
    f.close()

    #print(nshift)
    #print(get_xyz(p, nshift))
    f = open(atomsn, "w")
    f.write(get_xyz(p))
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
    #print(vint.shape, nshift)

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

    Jmn.load_nparray(vint.reshape((s.coords.shape[0], blockshape[0], blockshape[1])), s.coords, screening = False)

    #print("Size: ", Jmn.blockshape, Jmn.blocks.shape)
    #sp.call(["rm", "-rf", atomsJ])
        #print(coord, np.abs(vint).max())

    sp.call(["rm", "-rf", atomsn])
    sp.call(["rm", "-rf", atomsm])

    sp.call(["rm", "-rf", atomsJ])

    return Jmn

def compute_onebody(p,s, T = np.array([[0,0,0]]), operator = "overlap", conversion_factor = .5291772109200000):
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
    f.write(get_xyz(p, conversion_factor=conversion_factor))
    f.close()

    f = open(atomsK, "w")
    f.write(get_xyz(p, T, conversion_factor=conversion_factor))
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

def compute_overlap_matrix(p, T = np.array([[0,0,0]]), conversion_factor = .5291772109200000):
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
    f.write(get_xyz(p, conversion_factor=conversion_factor))
    f.close()

    f = open(atomsK, "w")
    f.write(get_xyz(p, T, conversion_factor=conversion_factor))
    f.close()

    lint = li.engine()

    lint.set_operator_overlap()

    lint.setup_pq(atomsJ, bname, atomsK, bname)
    vint = np.array(lint.get_pq(atomsJ, bname, atomsK, bname))
    vint = vint.reshape((p.get_n_ao(), T.shape[0], p.get_n_ao())).swapaxes(0,1)
    print(vint.shape)
    s = tp.tmat()
    s.load_nparray(vint, T)
    return s

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

def compute_JK_auto(p, s, attenuation = 1, auxname = "cc-pvdz", coulomb = False):
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


"""
Functions
"""

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
    JKk_inv.blocks[:-1] = np.linalg.pinv(JKk.blocks[:-1])



    JK_inv_direct = tp.transform(JKk_inv, np.fft.ifftn, n_points = n_points, complx = False)

    return JK_inv_direct


def estimate_attenuation_distance(p, attenuation = 0.1, c2 = [0,0,0], thresh = 10e-12, auxname = "cc-pvdz-ri"):
    """
    For a given attenuation parameter, basis and lattice geometry, estimate
    which blocks contain elements above the provided treshold.
    Returns a tmat object with blocks initialized accordingly

    # NOTE: this procedure may be obsolete, in the latest version we rather compute a large chunk whereby we only keep non-zero blocks

    """
    cube = tp.lattice_coords([0,0,0])
    big_tmat = tp.tmat()
    big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)

    Jmnc = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = [c2])

    if np.max(np.abs(Jmnc.blocks[:-1]))<thresh:
        return big_tmat


    for i in np.arange(1,100):
        cube = tp.lattice_coords([i,0,0]) #assumed max twobody AO-extent (subst. C-S Screening)

        cube = np.zeros((8,3), dtype = int)
        cube[1] = np.array([i,0,0])
        cube[2] = np.array([0,i,0])
        cube[3] = np.array([0,0,i])

        cube[4] = np.array([i,0,i])
        cube[5] = np.array([0,i,i])
        cube[6] = np.array([i,i,0])
        cube[7] = np.array([i,i,i])


        big_tmat = tp.tmat()
        big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)
        Jmnc = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = [c2])
        if (np.max(np.abs(Jmnc.cget(cube[1:]))))<thresh:

            cmax = np.argmax(np.sqrt(np.sum(np.dot(cube,p.lattice)**2, axis = 1)))
            rmax = np.sqrt(np.sum(p.coor2vec(cube)**2, axis = 1))[cmax]
            #print("Converged to %i layers for shifted coordinate:" % i, c2, cmax, big_tmat.coords[cmax], rmax)
            break

    cube = tp.lattice_coords([i,i,i]) #assumed max twobody AO-extent (subst. C-S Screening)


    big_tmat = tp.tmat()
    big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)
    return big_tmat #return expansion region in form of toeplitz matrix

def estimate_attenuation_domain(p, attenuation = 0.1, xi0 = 1e-8,  auxname = "cc-pvdz-ri"):
    """
    Estimate the attenuation domains as described in algorithm 3 in PRI-notes

    """
    # Generate a grid of cells
    coords = tp.lattice_coords(n_points_p(p, 6)) #assumed max extent
    #coords = n_points_p(p, 12)

    #print(coords)

    # Compute distance^2 of cells
    d2 = np.sum(p.coor2vec(coords)**2, axis = 1)

    # Sort in increasing distance
    coords = coords[ np.argsort(d2) ]

    # Sort the distances as well
    d2 = d2[np.argsort(d2)]

    # sort into discrete bins
    d2_unique = np.unique(d2) #unique distances
    indices = [] # indices/bins, containing the outermost index at a given distance
    for i in d2_unique:
        indices.append(np.sum(d2<=i))

    indices = np.array(indices)


    # Determine the cutoff between m-n

    for m in indices:
        cube = np.zeros((2,3), dtype = int)
        cube[1,0] = 1.0
        big_tmat = tp.tmat()

        big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)

        Jmnc = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = [coords[m]])
        if np.max(np.abs(Jmnc.blocks[:-1]))<xi0:
            #print("Break at :", m)
            #print(coords[:m])
            break

    xi_0_domain = coords[:m]*1
    xi_domains = []

    # Determine cutoff between J and mn

    for m in xi_0_domain:
        xi_domains.append([m, tp.tmat()])

    # The following is not used, so can safely be skipped

    """
    xi_domains = []

    # Determine cutoff between J and mn

    for m in xi_0_domain:


        for n in indices:
            cube = np.zeros((2,3), dtype = int)
            cube[1] = coords[n]

            big_tmat = tp.tmat()

            big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)
            # coords[m]
            Jmnc = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = [m])
            #print("J-mn-screen:", m, coords[n], np.sqrt(np.sum(p.coor2vec(coords[n])**2)), np.max(np.abs(Jmnc.cget(coords[n]))))

            if np.max(np.abs(Jmnc.cget(coords[n])))<xi0:
                xi_domains.append([m, tp.tmat()])
                xi_domains[-1][1].load_nparray(np.ones((coords[:n].shape[0], 2,2),dtype = float),  coords[:n])
                break
    """
    return xi_domains

def estimate_center_domain(p, attenuation = 0.1, xi0 = 1e-8,  auxname = "cc-pvdz-ri"):
    """
    Estimate the attenuation domains as described in algorithm 3 in PRI-notes

    """
    # Generate a grid of cells
    coords = tp.lattice_coords([12,12,12]) #assumed max extent

    #print(coords)

    # Compute distance^2 of cells
    d2 = np.sum(p.coor2vec(coords)**2, axis = 1)

    # Sort in increasing distance
    coords = coords[ np.argsort(d2) ]

    # Sort the distances as well
    d2 = d2[np.argsort(d2)]

    # sort into discrete bins
    d2_unique = np.unique(d2) #unique distances
    indices = [] # indices/bins, containing the outermost index at a given distance
    for i in d2_unique:
        indices.append(np.sum(d2<=i))

    indices = np.array(indices)

    xi_domains =  tp.tmat()

    for n in indices:

        cube = np.zeros((2,3), dtype = int)
        cube[1] = coords[n]

        big_tmat = tp.tmat()

        big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)

        Jmnc = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False)

        if np.max(np.abs(Jmnc.cget(coords[n])))<xi0:

            xi_domains.load_nparray(np.ones((coords[:n].shape[0], 2,2),dtype = float),  coords[:n])
            break

    return xi_domains


def estimate_attenuation_distance_(p, attenuation = 0.1, c2 = [0,0,0], thresh = 10e-12, auxname = "cc-pvdz-ri"):
    """
    For a given attenuation parameter, basis and lattice geometry, estimate
    which blocks contain elements above the provided treshold.
    Returns a tmat object with blocks initialized accordingly
    """



    cube = tp.lattice_coords([12,12,12]) #assumed max twobody AO-extent (subst. C-S Screening)

    # sort in increasing distance
    cube = cube[np.argsort(np.sum(p.coor2vec(cube)**2, axis = 1))]
    #print(cube[:10])

    for i in np.arange(1,cube.shape[0]):

        #cc = cube[i]
        big_tmat = tp.tmat()
        big_tmat.load_nparray(np.ones((2, 2,2),dtype = float),  cube[i-1:i+1])

        Jmnc = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = [c2])

        if np.max(np.abs(Jmnc.blocks[:-1]))<thresh:
            print("Converged to %i blocks for shifted coordinate:" % i, cube[i])
            break


    big_tmat = tp.tmat()
    big_tmat.load_nparray(np.ones((cube[:i+1].shape[0], 2,2),dtype = float),  cube[:i+1])

    return big_tmat #return expansion region in form of toeplitz matrix


class estimate_coordinate_domain():
    def __init__(self, p, basis, N_c, attenuation):
        self.p = p
        self.coords = tp.lattice_coords(n_points_p(p, N_c))
        self.R = np.sqrt(np.sum(p.coor2vec(self.coords)**2, axis = 1))
        self.coords = self.coords[ np.argsort(self.R) ]
        self.R = self.R[np.argsort(self.R)]
        self.basis = basis
        self.attenuation = attenuation
        
        # set up unique list of distances
        self.Ru = np.unique(self.R)
        #print(self.Ru.max(), len(self.Ru))
            
        
    
    def get_shell(self, r0, r1):
        #print(self.R)
        return self.coords[np.all(np.array([self.R<r1, self.R>=r0]), axis = 0)]
        #return self.coords[self.R<]
        

        
    def get_shell_maxelement(self,i, c2 = np.array([0,0,0])):
        try:
            r0 = self.Ru[i]
            r1 = self.Ru[i+1] +0.1
        except:
            # Expand list
            print("WARNING: maximum fitting domain too small")
            r0 = self.Ru[-2]
            r1 = self.Ru[-1]
            
        #print(r0, r1)
        
        
        
        s = tp.tmat()
        scoords = self.get_shell(r0,r1)
        
        s.load_nparray(np.ones((len(scoords), 2,2), dtype = float), scoords)
        
        Jmn = compute_Jmn(self.p, s, attenuation = self.attenuation, auxname =self.basis, nshift = np.array([c2]))
        return np.abs(Jmn.blocks).max()
    
    def estimate_attenuation_domain(self, c2 = np.array([0,0,0]), thresh = 1e-8):
        
        for i in np.arange(self.Ru.shape[0]):
            cmR, cmM = self.Ru[i+1]+0.1,self.get_shell_maxelement(i)
            if cmM<thresh:
                break
        return np.max(np.abs(self.coords[self.R<cmR]), axis = 0), cmR
    
    def compute_Jmn(self, c2, thresh = 1e-8):
        
        Jmn_full = []
        N_blocks = 0
        
        
        # compute first block
        #PRI.compute_Jmn(self.p, s, attenuation = self.attenuation, auxname =self.basis, nshift = np.array([c2]))
        
        for i in np.arange(self.Ru.shape[0]-1):
            r0 = self.Ru[i]
            r1 = self.Ru[i+1] +0.1
            
            
            s = tp.tmat()
            scoords = self.get_shell(r0,r1)
            #print(scoords)
            s.load_nparray(np.ones((len(scoords), 2,2), dtype = float), scoords)
            #print(s.coords)
            
            Jmn_shell = compute_Jmn(self.p, s, attenuation = self.attenuation, auxname =self.basis, nshift = np.array([c2]))
            
            
            cmM = np.abs(Jmn_shell.blocks).max()
            #print(cmM)
            if cmM<thresh:
                break
            else:
                Jmn_full.append(Jmn_shell)
                #print(Jmn_shell.coords)
                N_blocks += Jmn_shell.coords.shape[0]
        
        ret = None
        if len(Jmn_full)>=1:
            # gather results, return tmat
            rcoords = np.zeros((N_blocks, 3), dtype = int)
            rblocks = np.zeros((N_blocks, Jmn_full[0].blocks.shape[1],Jmn_full[0].blocks.shape[2]), dtype = float)
            
            ni = 0
            for i in np.arange(len(Jmn_full)):
                dn = Jmn_full[i].coords.shape[0]
                #print(i)
                
                rcoords[ni:ni+dn] = Jmn_full[i].coords
                rblocks[ni:ni+dn] = Jmn_full[i].cget(Jmn_full[i].coords)
                ni += dn
            
            ret = tp.tmat()
            ret.load_nparray(rblocks, rcoords)
        else:
            s = tp.tmat()
            scoords = np.array([[0,0,0], [1,0,0]])
            s.load_nparray(np.ones((len(scoords), 2,2), dtype = float), scoords)
            ret = compute_Jmn(self.p, s, attenuation = self.attenuation, auxname =self.basis, nshift = np.array([c2]))

        
        return ret

    def compute_JK(self, thresh = 1e-8):
        
        Jmn_full = []
        N_blocks = 0
        
        
        # compute first block
        #PRI.compute_Jmn(self.p, s, attenuation = self.attenuation, auxname =self.basis, nshift = np.array([c2]))
        
        for i in np.arange(self.Ru.shape[0]-1):
            r0 = self.Ru[i]
            r1 = self.Ru[i+1] +0.1
            
            
            s = tp.tmat()
            scoords = self.get_shell(r0,r1)
            #print(scoords)
            s.load_nparray(np.ones((len(scoords), 2,2), dtype = float), scoords)
            #print(s.coords)
            
            Jmn_shell = compute_JK(self.p, s, attenuation = self.attenuation, auxname =self.basis)
            
            
            
            cmM = np.abs(Jmn_shell.blocks).max()
            #print(cmM)
            if cmM<thresh:
                break
            else:
                Jmn_full.append(Jmn_shell)
                #print(Jmn_shell.coords)
                N_blocks += Jmn_shell.coords.shape[0]
        
        ret = None
        if len(Jmn_full)>=1:
            # gather results, return tmat
            rcoords = np.zeros((N_blocks, 3), dtype = int)
            rblocks = np.zeros((N_blocks, Jmn_full[0].blocks.shape[1],Jmn_full[0].blocks.shape[2]), dtype = float)
            
            ni = 0
            for i in np.arange(len(Jmn_full)):
                dn = Jmn_full[i].coords.shape[0]
                #print(i)
                
                rcoords[ni:ni+dn] = Jmn_full[i].coords
                rblocks[ni:ni+dn] = Jmn_full[i].cget(Jmn_full[i].coords)
                ni += dn
            
            ret = tp.tmat()
            ret.load_nparray(rblocks, rcoords)
        else:
            s = tp.tmat()
            scoords = np.array([[0,0,0], [1,0,0]])
            s.load_nparray(np.ones((len(scoords), 2,2), dtype = float), scoords)
            ret = Jmn_shell = compute_JK(self.p, s, attenuation = self.attenuation, auxname =self.basis)

        
        return ret
            


class coefficient_fitter_static():
    """
    Coefficient fitting for the integrator, with integrals stored in memory


    """

    def __init__(self, c_occ,c_virt, p, attenuation, auxname, JK, JKInv, robust = False, circulant = True, xi0 = 1e-10, xi1 = 1e-10, float_precision = np.float64, printing = False, N_c = 7):
        self.robust = robust
        self.coords = []
        self.Jmn = []
        self.attenuation = attenuation
        #self.screening_thresh = screening_thresh
        self.p = p
        #self.c = c
        self.N_c = N_c
        self.JK = JK
        self.JKInv = JKInv
        self.circulant = circulant
        #self.c_occ, self.c_virt = occ_virt_split(self.c,p)
        self.c_occ = c_occ
        self.c_virt = c_virt
        self.float_precision = float_precision
        self.printing = printing
        self.xi0 = xi0
        self.xi1 = xi1
        if self.robust:
            self.Jmnc = []


        xi_domain = estimate_attenuation_domain(p, attenuation = attenuation, xi0 = xi0,  auxname = auxname)

        #ddM = tp.lattice_coords([20,0,0])
        #ddM = ddM[np.argsort(np.sum(p.coor2vec(ddM)**2, axis = 1))]
        
        #xi_domain = []
        #for i in np.arange(20):
        #    xi_domain.append([dM[i], 1])
        N_c_max_layers = 20
        cm = estimate_coordinate_domain(p, auxname, N_c_max_layers, attenuation = attenuation)

        C2 = tp.lattice_coords(n_points_p(p, N_c_max_layers))

        R = np.sqrt(np.sum(p.coor2vec(C2)**2, axis = 1))

        C2 = C2[np.argsort(R)]
        R = R[np.argsort(R)]

        
        #for i in np.arange(len(xi_domain)):

        for ci in np.arange(len(C2)):
            c2, R2 = C2[ci], R[ci] 
            # Compute JMN with nsep =  c2

            #c2, big_tmat = xi_domain[i]

            




        
            #for i in np.arange(ddM.shape[0]):
            #c2 = ddM[i]


            if False:
                """
                Alternative screening approach, somewhat inefficient but avoids premature truncation
                It basically computes a large chunk, presumably larger than required
                TODO: check boundaries for significant integrals, break if present
                """

                




                Nc = self.N_c
                if i == 0:
                    cellcut = 65 # bohr
                    c_x = np.sum(p.coor2vec([Nc,0,0])**2)
                    c_y = np.sum(p.coor2vec([0,Nc,0])**2)
                    c_z = np.sum(p.coor2vec([0,0,Nc])**2)
                    #print()

                    
                if p.cperiodicity == "POLYMER":
                    #cellcut = 165 # bohr
                    if i== 0:
                        cellcut = 2*np.sqrt(c_x)
                        #cellcut = 165
                    Rc = np.sqrt(np.sum(self.p.coor2vec(tp.lattice_coords([2*Nc,0,0]))**2, axis = 1))
                    bc = tp.lattice_coords([2*Nc,0,0])[Rc<=cellcut]
                    
                elif p.cperiodicity == "SLAB":
                    if i == 0:
                        cellcut = np.sqrt(np.max([c_x,c_y]))
                    Rc = np.sqrt(np.sum(self.p.coor2vec(tp.lattice_coords([Nc,Nc,0]))**2, axis = 1))
                    bc = tp.lattice_coords([Nc,Nc,0])[Rc<=cellcut]
                else:
                    if i == 0:
                        cellcut = np.sqrt(np.max([c_x,c_y, c_z]))
                        print(" Cellcut0:", cellcut)
                    Rc = np.sqrt(np.sum(self.p.coor2vec(tp.lattice_coords([Nc,Nc,Nc]))**2, axis = 1))
                    bc = tp.lattice_coords([Nc,Nc,Nc])[Rc<=cellcut] #Huge domain, will consume some memory
                # if i == 0:
                # print(bc)
                cellmax = np.max(np.sqrt(np.sum(self.p.coor2vec(bc)**2, axis = 1)))

                #print(c2)
                """
                Nc, cellmax = cm.estimate_attenuation_domain(thresh = xi0, c2 = c2)
                bcoords = tp.lattice_coords(Nc)
                bR = np.sqrt(np.sum(p.coor2vec(bcoords)**2, axis = 1))
                bc = bcoords[bR<cellmax]
                """
                
                print("fitting:", c2, bc.shape)
                



                big_tmat = tp.tmat()
                big_tmat.load_nparray(np.ones((bc.shape[0], 2,2), dtype = float), bc)

                # print(" Cellmax.", cellmax)
                # print(bc.shape)



                Jmnc2_temp = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = np.array([c2]))

                





                Jmnc2_temp.set_precision(self.float_precision)

                screen = np.max(np.abs(Jmnc2_temp.cget(Jmnc2_temp.coords)), axis = (1,2))>=xi0
                screen[np.sum(Jmnc2_temp.coords**2, axis = 1)==0] = True
                
                

                # Screened distances
                distances = np.sqrt(np.sum(self.p.coor2vec(Jmnc2_temp.coords)**2, axis = 1))

                Jmnc2_max =  np.max(distances[screen])

                
                if np.sum(screen)>1:
                    max_outer_5pcnt = np.max(Jmnc2_temp.cget(Jmnc2_temp.coords[distances>Jmnc2_max*0.95]))
                    
                    if self.printing:
                        print("Attenuation screening induced sparsity is %i of a total of %i blocks." %( np.sum(screen), len(screen)))
                        print("         Maximum value in outer 5 percentage of block (rim) :", max_outer_5pcnt)
                        




                    
                    
                    #r_curr, r_prev = np.sum(p.coor2vec(c2)**2), np.sum(p.coor2vec(xi_domain[i-1][0])**2)
                    if cellmax<=Jmnc2_max:
                        #if r_curr-r-prev
                        print("Warning: Jmnc2 fitting integrals for c = ", c2, " exceeds predefined matrix extents.")
                        print("         Jmnc2_max = %.2e,    truncation_threshold = %.2e" % (Jmnc2_max, cellcut))
                        #print("         Max value at boundary: ", Jmnc2_temp)
                        #print("         Truncation threshold (cellcut) should be increased.") 
                        print("         Maximum value in outer 5 percentage of block (rim) :", max_outer_5pcnt)
                        #print("->", c2, xi_domain[i-1][0])
                        #print("->", np.sum(p.coor2vec(c2)**2), np.sum(p.coor2vec(xi_domain[i-1][0])**2))
                    cellcut =  Jmnc2_max*1.2#update truncation threshold

                Jmnc2 = tp.tmat()

                Jmnc2.load_nparray(Jmnc2_temp.cget(Jmnc2_temp.coords[screen]), Jmnc2_temp.coords[screen])


            else:

                #Jmnc2 = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()
                Jmnc2_temp = cm.compute_Jmn(c2, thresh = xi0)
                Jmnc2_temp.set_precision(self.float_precision)
                #print("Number of blocks in ")

                screen = np.max(np.abs(Jmnc2_temp.cget(Jmnc2_temp.coords)), axis = (1,2))>=xi0
                screen[np.sum(Jmnc2_temp.coords**2, axis = 1)==0] = True
                
                

                # Screened distances
                distances = np.sqrt(np.sum(self.p.coor2vec(Jmnc2_temp.coords)**2, axis = 1))

                Jmnc2_max =  np.max(distances[screen])

                
                if np.sum(screen)>1:
                    max_outer_5pcnt = np.max(Jmnc2_temp.cget(Jmnc2_temp.coords[distances>Jmnc2_max*0.95]))
                    
                    if True: #self.printing:
                        
                        print("Attenuation screening induced sparsity is %i of a total of %i blocks." %( np.sum(screen), len(screen)))
                        print("         Maximum value in outer 5 percentage of block (rim) :", max_outer_5pcnt)
                        print("         Maximum value overall                              :", np.max(np.abs(Jmnc2_temp.blocks)))
                        print("         c2 = ", c2)
                        print("         R  = ", R2)




                    
                    
                    #r_curr, r_prev = np.sum(p.coor2vec(c2)**2), np.sum(p.coor2vec(xi_domain[i-1][0])**2)
                    """
                    if cellmax<=Jmnc2_max:
                        #if r_curr-r-prev
                        print("Warning: Jmnc2 fitting integrals for c = ", c2, " exceeds predefined matrix extents.")
                        print("         Jmnc2_max = %.2e,    truncation_threshold = %.2e" % (Jmnc2_max, cellcut))
                        #print("         Max value at boundary: ", Jmnc2_temp)
                        #print("         Truncation threshold (cellcut) should be increased.") 
                        print("         Maximum value in outer 5 percentage of block (rim) :", max_outer_5pcnt)
                        #print("->", c2, xi_domain[i-1][0])
                        #print("->", np.sum(p.coor2vec(c2)**2), np.sum(p.coor2vec(xi_domain[i-1][0])**2))
                    cellcut =  Jmnc2_max*1.2#update truncation threshold
                    """

                Jmnc2 = tp.tmat()

                Jmnc2.load_nparray(Jmnc2_temp.cget(Jmnc2_temp.coords[screen]), Jmnc2_temp.coords[screen])


            self.coords.append(c2)
            self.Jmn.append( Jmnc2 ) #New formulation without uppercase-transpose


            if self.printing:
                print("Intermediate overlaps (LJ|0mNn) with N =", c2, " included with %i blocks and maximum absolute %.2e" % (Jmnc2.blocks.shape[0],np.max(np.abs(Jmnc2.blocks)) ))
            if np.max(np.abs(Jmnc2.blocks))<xi0 and np.abs(R[ci]-R[ci+1])>1e-10:
                break

        self.coords = np.array(self.coords)
        self.c0 = np.argwhere(np.all(self.coords==np.array([0,0,0]), axis = 1))[0][0]

        self.Jmnc_tensors = []
        self.Jmnc_screening = []

        self.Jmnc_sparse_tensors = []
        self.Jmnc_sparse_screening = []
        total = 0
        compr = 0



        t0 = time.time()


        self.virtual_coords = []
        #print("Will screen away:", xi1, xi0)


        t0 = time.time()

        

        Nc = 15
        cellcut = 35
        if p.cperiodicity == "POLYMER":
            cellcut = 180
            Rc = np.sqrt(np.sum(self.p.coor2vec(tp.lattice_coords([Nc,0,0]))**2, axis = 1))
            pq_region = tp.lattice_coords([Nc,0,0])[Rc<=cellcut]
            
        elif p.cperiodicity == "SLAB":
            Rc = np.sqrt(np.sum(self.p.coor2vec(tp.lattice_coords([Nc,Nc,0]))**2, axis = 1))
            pq_region = tp.lattice_coords([Nc,Nc,0])[Rc<=cellcut]
        else:
            Rc = np.sqrt(np.sum(self.p.coor2vec(tp.lattice_coords([Nc,Nc,Nc]))**2, axis = 1))
            pq_region = tp.lattice_coords([Nc,Nc,Nc])[Rc<=cellcut]
        
        self.pq_region = pq_region[np.argsort(np.sum(p.coor2vec(pq_region)**2, axis = 1))]

        # Contracting occupieds
        self.OC_L_np, self.c_virt_coords_L, self.c_virt_screen, self.NJ, self.Np = contract_occupieds(self.p, self.Jmn, self.coords, self.pq_region, self.c_occ, self.xi1)
        
        # Can delete self.Jmn now, no longer required and takes up a lot of memory
        del(self.Jmn)
        

        if self.printing:

            #print("\r>> Contracted occupieds (LJ|0p Mn) for L = ", self.c_occ.coords[coord], "(%.2f percent complete, compression rate: %.2e)" % (100.0*coord/self.c_occ.coords.shape[0], 100.0*compr/total), end='')
            print("\r>> Contracted occupieds (LJ|0p Mn) ")
        
        
        
        
        if self.printing:
            print(time.time()-t0)
            #print("Screening-induced sparsity is at %.2e percent." % (100.0*compr/total))

    def get(self, coords_q, robust = False):
        pq_c = []


        

        #print(self.JK.cutoffs(), Jpq_c[i].cutoffs())
        

        for dM in np.arange(coords_q.shape[0]):
            if robust:
                J_pq_c = contract_virtuals(self.OC_L_np, self.c_virt_coords_L, self.c_virt_screen, self.c_virt, self.NJ, self.Np, self.pq_region, dM = coords_q[dM])
                
                n_points = n_points_p(self.p, np.max(np.abs(J_pq_c.coords)))
                
                pq_c.append([self.JK.kspace_svd_solve(J_pq_c, n_points = n_points), J_pq_c ])
            else:
                J_pq_c = contract_virtuals(self.OC_L_np, self.c_virt_coords_L, self.c_virt_screen, self.c_virt, self.NJ, self.Np, self.pq_region, dM = coords_q[dM])
                
                n_points = n_points_p(self.p, np.max(np.abs(J_pq_c.coords)))
                #n_points = n_points_p(self.p, self.N_c)
                
                pq_c.append(
                    self.JK.kspace_svd_solve(
                        J_pq_c, 
                        n_points = n_points))
        
        return pq_c
        



def n_points_p(p, Nc):

    if p.cperiodicity == "POLYMER":
        return np.array([Nc,0,0])
        
    elif p.cperiodicity == "SLAB":
        return np.array([Nc,Nc,0])
    else:
        return np.array([Nc,Nc,Nc])

def contract_virtuals(OC_L_np, c_virt_coords_L, c_virt_screen, c_virt, NJ, Np, pq_region, dM = np.array([0,0,0])):
    """
    On-demand contraction of virtual block dM
    See section 3.2, algorithm 1 and 2 in the notes for details
    Author: Audun
    """

    # Tensor dimensions
    Nn = c_virt.blocks.shape[1]    
    Nq = c_virt.blocks.shape[2]
    NL = len(c_virt_coords_L)

    Jpq_blocks = np.zeros((NL,NJ,Np*Nq), dtype = float)
    for i in np.arange(len(c_virt_coords_L)):
        NN = len(c_virt_coords_L[i])
        sc = c_virt_screen[i]
        
        
        Jpq_blocks[i] = np.dot(OC_L_np[i],c_virt.cget(c_virt_coords_L[i]+dM).reshape(NN*Nn,Nq)[sc,:]).reshape(NJ, Np*Nq)

        

    Jpq_blocks = Jpq_blocks.reshape(NL, NJ, Np*Nq)

    

    Jpq = tp.tmat()
    Jpq.load_nparray(Jpq_blocks[:-1], -pq_region[:NL-1])    
    #for coord in pq_region[:NL]:
    #    
    #    print("Normcheck: (jpq):", coord, np.linalg.norm(Jpq.cget([coord])))
    #    #print(J)

    return Jpq



def contract_occupieds(p, Jmn_dm, dM_region, pq_region, c_occ, xi2 = 1e-10):
    """
    Intermediate contraction of the occupieds.
    See section 3.2, algorithm 1 and 2 in the notes for details
    Author: Audun
    """



    


    O = []

    # dimensions to match the equations
    
    NN = Jmn_dm[0].coords.shape[0] # Number of blocks in coefficients
    
    Np = c_occ.blocks.shape[2]  # number of occupieds
    Nn = c_occ.blocks.shape[1]  # number of ao functions
    Nm = c_occ.blocks.shape[1]  # number of ao functions


    # Set max possible extent

    #n_points = np.max(np.abs(dM_region), axis = 0) + np.max(np.abs(Jmn_dm[0].coords), axis = 0)
    s = tp.get_zero_tmat(n_points_p(p, 10), [2,2])
    #s = tp.get_zero_tmat(n_points, [2,2])
    NN = s.coords.shape[0]      # number of 
    #print("domain setup in contract_occupieds may not be optimal")
    #print(np.max(s.coords, axis = 0))
    #print("xi2:", xi2)
    

    #xi2 = 1e-7

    c_virt_coords_L = []
    c_virt_screen = []
    OC_L_np = []



    NJ = Jmn_dm[0].blocks.shape[1]

    elms_retained = 0
    elms_total    = 0

    # optimize
    
    optimized = True
    if optimized:
        n_points = np.max(np.abs(dM_region), axis = 0) + np.max(np.abs(c_occ.coords), axis = 0)
        
        N_coords = tp.lattice_coords(n_points)
        
        NN = N_coords.shape[0]

        print("Shape of N_coords:", N_coords.shape)

    for Li in np.arange(pq_region.shape[0]):
        L = pq_region[Li]
        O_LN_np = np.zeros((NN, NJ,Nn, Np), dtype = float)

        #N_coords = -Jmn_dm[0].coords 
        
        

        # J = -N - dM - L, L in all
        # O = N + dM = -(J) - L 
        # 
        # N_coords = tp.lattice_coords([np.max(np.abs(Jmnc.coords), axis = 0) + np.abs()])
        

        for dMi in np.arange(dM_region.shape[0]):
            dM = dM_region[dMi]
            

            Jmn = Jmn_dm[dMi]
            NJ = Jmn.blocks.shape[1]

            #if optimized:
            #    # optimize
            #    c_occ_blocks = c_occ.cget(N_coords + dM)
            #    Jmn_blocks = Jmn.cget(-N_coords-dM-L).reshape(NN,NJ,Nm,Nn)
            #else:
            #Jmn_blocks = Jmn.cget(-N_coords-dM-L).reshape(NN,NJ,Nm,Nn) #screen on these coordinates, use as "zero", all other offsets
            t0 = time.time()
            Jmn_blocks = Jmn.cget(-N_coords+L).reshape(NN,NJ,Nm,Nn) #screen on these coordinates, use as "zero", all other offsets
            
            #print("Jmn.cget:", time.time()-t0)
            #print("")
            #t0 = time.time()
            
            c_occ_blocks = c_occ.cget(-N_coords - dM)  #+ here (used to be)

            #print("c_occ.cget:", time.time()-t0)
            #print("")
            #t0 = time.time()
            



            # Screen out zero blocks here 
            #cs = np.max(np.abs(c_occ_blocks), axis = (1,2))>xi2
            cs = np.any(np.greater(np.abs(c_occ_blocks), xi2), axis = (1,2))


            #bs = np.max(np.abs(Jmn_blocks), axis = (1,2,3))>xi2
            bs = np.any(np.greater(np.abs(Jmn_blocks), xi2), axis = (1,2,3))



            #print(np.argmax(np.abs(Jmn_blocks), axis = (1,2,3)))
            sc = np.logical_and(cs, bs)

            #print("screening:", time.time()-t0)
            #print("")
            #t0 = time.time()
            
            #print("number of sc:", np.sum(sc))
            if np.sum(sc)>0:
                # If any block non-zero : contract occupieds
                #dO_LN_np = np.einsum("NJmn,Nmp->NJnp", Jmn_blocks[sc], c_occ_blocks[sc], optimize = True) #change

                Jsc, csc = Jmn_blocks[sc], c_occ_blocks[sc]
                dO_LN_np = np.zeros((np.sum(sc), NJ, Nn, Np), dtype = float)
                for k in np.arange(np.sum(sc)):
                    dO_LN_np[k] =  np.dot(Jsc[k].reshape(NJ,Nm,Nn).swapaxes(1,2).reshape(NJ*Nn,Nm), csc[k]).reshape(NJ, Nn, Np)
                
                #print("for k in sc:", time.time()-t0)
                #print("")
                #t0 = time.time()

                if np.abs(dO_LN_np).max()<xi2:
                    break
                O_LN_np[sc] += dO_LN_np

            #print("if np.sumsc:", time.time()-t0)
            #print("")
            #t0 = time.time()

        

        c_virt_coords_L.append(-N_coords)

        # optimize
        # c_virt_coords_L.append(c_occ.coords)


        #print("full np.sumsc:", time.time()-t0)
        #print("")
        #t0 = time.time()
        
        # Prepare for screening + sparse storage
        O_LN_np = np.einsum("NJnp->JpNn", O_LN_np).reshape(NJ*Np, NN*Nn)

        #print("transpose:", time.time()-t0)
        #print("")
        #t0 = time.time()
        
        sc = np.max(np.abs(O_LN_np), axis = 0)>xi2
        #print(sc.shape)
        c_virt_screen.append(sc)
        
        #print
        elms_retained += np.sum(sc)*O_LN_np.shape[0]
        elms_total    += NJ*Np*NN*Nn
        print("Interm.contr. at L =", L, " (|R_L| = %.2e bohr). Abs.max value: %.2e. Compression rate: %.2e (%i retained columns)." % (np.sqrt(np.sum(p.coor2vec(pq_region[Li])**2)), np.abs(O_LN_np).max(), elms_retained/elms_total, np.sum(sc)))
        
        
        OC_L_np.append(O_LN_np[:,sc])

        #print("add to list:", time.time()-t0)
        #print("")
        #t0 = time.time()
        #screen_L.append(np.max(np.abs(O_LN_np.reshape(NN*NJ,Nn*Np)), axis = 1)) #->NJnp-Jp,Nn
        
        if np.abs(O_LN_np).max()<xi2:
            dR = np.abs(np.sum(p.coor2vec(pq_region[Li-1])**2) - np.sum(p.coor2vec(pq_region[Li])**2))

            if dR > 1e-12:
                print("    ->   Truncation of intermediate tensor O at L = ", L, " with dR = %.2e  <-" % dR)
                print("         Final compression ratio of intermediate tensor is %.2e" % (elms_retained/elms_total))
                print("                              (%i out of %i elements retained)" % (elms_retained, elms_total))
                print("         Max value : %.2e" % np.abs(O_LN_np).max())
                break


        # Screen out negligible 
    return OC_L_np, c_virt_coords_L, c_virt_screen, NJ, Np




def test_matrix_kspace_condition(M, n_fourier):
    Mk = tp.dfft(M, n_fourier)*1
    #Mk_inv = M*0
    for k in Mk.coords:
        Mk_e, Mk_v = np.linalg.eig(Mk.cget(k))
        print(k, np.abs(Mk_e).max()/np.abs(Mk_e).min(),np.linalg.det(Mk.cget(k)), np.max(np.abs(Mk.cget(k).real)))
        print()




class integral_builder_static():
    """
    RI-integral builder with stored AO-integrals
    For high performance (but high memory demand)
    """
    def __init__(self, c_occ, c_virt,p, attenuation = 0.1, auxname = "cc-pvdz-ri", initial_virtual_dom = [1,1,1], circulant = True, robust  = False,  inverse_test = True, coulomb_extent = None, JKa_extent = None, xi0 = 1e-10, xi1 = 1e-10, float_precision = np.float64, printing = True, N_c = 10):
        self.c_occ = c_occ
        self.c_virt = c_virt
        self.p = p
        self.attenuation = attenuation
        self.auxname = auxname
        self.circulant = circulant
        self.robust = robust
        self.float_precision = float_precision
        self.N_c = N_c # (2*N_c + 1)  = k-space resolution

        self.n_occ = c_occ.blocks.shape[2]
        self.n_virt = c_virt.blocks.shape[2]

        # Oneshot calculations:
        # build attenuated JK matrix and inverse
        #big_tmat = estimate_attenuation_distance(p, attenuation = .5*self.attenuation, thresh = 1e-14, auxname = auxname)
        #big_tmat = estimate_attenuation_distance(p, attenuation = self.attenuation, thresh = extent_thresh, auxname = auxname)


        #big_tmat = estimate_center_domain(p, attenuation = attenuation, xi0 = xi0, auxname=auxname)
        if p.cperiodicity=="POLYMER":
            bc = tp.lattice_coords([N_c,0,0])
            #coulomb_extent = np.array([N_c, 0,0])
        if p.cperiodicity=="SLAB":
            bc = tp.lattice_coords([N_c,N_c,0])
            #coulomb_extent = np.array([N_c, N_c,0])
        if p.cperiodicity=="CRYSTAL":
            bc = tp.lattice_coords([N_c,N_c,N_c])
            #coulomb_extent = np.array([N_c, n_c,N_c])

        
        
        big_tmat = tp.tmat()
        big_tmat.load_nparray(np.ones((bc.shape[0], 2,2)), bc)

        cmax = big_tmat.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))]
        cmax = np.argmax(np.sqrt(np.sum(p.coor2vec(big_tmat.coords)**2, axis = 1)))

        """
        if JKa_extent is not None:
            big_tmat = tp.tmat()
            scoords = tp.lattice_coords(JKa_extent)
            big_tmat.load_nparray(np.ones((scoords.shape[0],2,2), dtype = float), scoords)
        """

        #self.JKa = compute_JK(self.p,self.c, attenuation = attenuation, auxname = auxname)
        # How large should actually this one be?
        #self.JKa = compute_JK(self.p,big_tmat, attenuation = attenuation, auxname = auxname)
        #self.JKa.set_precision(self.float_precision)



        N_c_max_layers = 20
        cm = estimate_coordinate_domain(p, auxname, N_c_max_layers, attenuation = attenuation)
        self.JKa = cm.compute_JK(thresh = xi0)
        self.JKa.set_precision(self.float_precision)



        if inverse_test:
            print(" Testing conditions")
            self.JKa.check_condition()
        #print(self.JKa.blocks.shape)@
        if printing:
            print("")
            print("Attenuated coulomb matrix (JKa) computed.")
            print("JKa outer coordinate (should be small):", cmax, big_tmat.coords[cmax], np.max(np.abs(self.JKa.cget(self.JKa.coords[cmax]))))


        #assert(np.max(np.abs(self.JKa.cget(self.JKa.coords[cmax])))<=extent_thresh), "JKa outer coordinate (should be smaller than %.2e):" % extent_thresh

        #print(np.max(self.JKa.coords, axis = 0))
        #print(np.min(self.JKa.coords, axis = 0))
        #print(self.JKa.coords)


        if printing:
            print("Number of auxiliary basis functions (in supercell):", self.JKa.blocks[:-1].shape[0]*self.JKa.blocks[:-1].shape[1])
            #print("JKa block shape:", self.JKa.blocks[:-1].shape)
            print("")

        self.JKinv = invert_JK(self.JKa)
        self.JKinv.set_precision(self.float_precision)
        #inverse_test = True
        if inverse_test:
            if printing:
                print("JKa inverse computed, checking max deviation from 0 = JKa^-1 JKa - I within extent")

            


            tcoords = np.zeros((np.max(self.JKa.coords),3), dtype = int)
            tcoords[:,0] = np.arange(self.JKa.coords.max(), dtype = int)
            I = self.JKinv.cdot(self.JKa, coords = tcoords )
            
            if printing:
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
        if initial_virtual_dom is not None:
            coord_q =  tp.lattice_coords(initial_virtual_dom) #initial virtual domain
        else:
            coords_q = []
        if printing:
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

        #print("Coeff fitter static tresh set to 1e-8")
        t0 = time.time()
        self.cfit = coefficient_fitter_static(self.c_occ, self.c_virt, p, attenuation, auxname, self.JKa, self.JKinv, robust = robust, circulant = circulant, xi0=xi0, xi1=xi1, float_precision = self.float_precision, printing = printing, N_c = self.N_c)
        t1 = time.time()
        if printing:
            print("Spent %.1f s preparing fitting (three index) integrals." % (t1-t0))
        if initial_virtual_dom is not None:
            if self.robust:
                Xreg = self.cfit.get(coord_q, robust = True)
            else:
                Xreg = self.cfit.get(coord_q, robust = False)
            t2 = time.time()
            if printing:
                print("Spent %.1f s computing fitting coefficient for %i coords" % (t2-t1, len(coord_q)))

        if initial_virtual_dom is not None:
            for i in np.arange(coord_q.shape[0]):
                #print("Transpose of coordinate", coord_q[i])
                if self.robust:
                    self.JpqXreg[coord_q[i][0], coord_q[i][1],coord_q[i][2]] = Xreg[i][1]
                    self.XregT[coord_q[i][0], coord_q[i][1],coord_q[i][2]] = Xreg[i][0].tT()

                else:
                    print("domain info:", coord_q[i], np.max(np.abs(Xreg[i].coords), axis = 0))
                    self.XregT[coord_q[i][0], coord_q[i][1],coord_q[i][2]] = Xreg[i].tT()
        #if robust:
        #    for i in np.arange(coord_q.shape[0]):
        #        #print("Transpose of coordinate", coord_q[i])
        #        self.JpqXreg[coord_q[i][0], coord_q[i][1],coord_q[i][2]] = Jpq[i]


        if coulomb_extent is None:
            # Compute JK_coulomb
            coulomb_extent = np.max(np.abs(self.XregT[0,0,0].coords), axis = 0)
            #coulomb_extent = np.max(np.abs(self.JKa[0,0,0].coords), axis = 0)
            #if printing:
            print("Extent of Xreg          :", coulomb_extent)

            #coulomb_extent = (10,10,10)
            #print("Extent of Coulomb matrix:", coulomb_extent)

        s = tp.tmat()
        scoords = tp.lattice_coords(coulomb_extent)
        s.load_nparray(np.ones((scoords.shape[0],2,2), dtype = float), scoords)

        #self.JK = compute_JK(p,self.JKa, coulomb=True, auxname = self.auxname)
        #self.JK = compute_JK(p,big_tmat, coulomb=True, auxname = self.auxname)
        self.JK = compute_JK(p,s, coulomb=True, auxname = self.auxname)
        self.JK.set_precision(self.float_precision)

        if printing:

            print("Coulomb matrix (JK) computed.")

        if initial_virtual_dom is not None:
            for i in np.arange(coord_q.shape[0]):
                #print("JK dot X for cell ", coord_q[i])
                if robust:

                    if circulant:
                        self.VXreg[coord_q[i][0], coord_q[i][1],coord_q[i][2]]= self.JK.circulantdot(Xreg[i][0])
                    else:
                        self.VXreg[coord_q[i][0], coord_q[i][1],coord_q[i][2]]= self.JK.cdot(Xreg[i][0])
                else:
                    if circulant:
                        self.VXreg[coord_q[i][0], coord_q[i][1],coord_q[i][2]]= self.JK.circulantdot(Xreg[i])
                    else:
                        self.VXreg[coord_q[i][0], coord_q[i][1],coord_q[i][2]]= self.JK.cdot(Xreg[i])

    def get_adaptive(self, dL, dM, M, keep = True):
        """
        Fit integrals, compute required blocks in the Coulomb matrix on the fly
        Returns a toeplitz matrix where the blocks in M are calculated and shape of (pq|rs)
        """
        dels = []
        for d in [dL, dM]:
            # Make sure Jpq are available in fitter, if not calculate them
            if self.XregT[d[0], d[1], d[2]] is 0:
                Xreg = self.cfit.get(np.array([d]))
                self.XregT[d[0], d[1], d[2]] = Xreg[0].tT()
                dels.append(d)
        self.JK, v_pqrs = tdot(self.p, self.XregT[dL[0], dL[1], dL[2]],self.JK,self.XregT[dM[0], dM[1], dM[2]].tT(), auxname = self.auxname, coords = M)
        
        if not keep:
            for d in dels:
                self.XregT[d[0], d[1], d[2]] = 0
        return v_pqrs, (self.n_occ, self.n_virt, self.n_occ, self.n_virt)






    def getorientation(self, dL, dM, adaptive_cdot = False, M=None):
        if adaptive_cdot:
            # not yet implementet



            return None
        else:
            # use circulant fomulation
            for d in [dL, dM]:
                #print(d)
                if self.XregT[d[0], d[1], d[2]] is 0:
                    if self.robust:
                        Xreg = self.cfit.get(np.array([d]), robust = True)
                        self.XregT[d[0], d[1], d[2]] = Xreg[0][0].tT()

                        self.JpqXreg[d[0], d[1], d[2]] = Xreg[0][1]

                        if self.circulant:
                            self.VXreg[d[0], d[1], d[2]] =  self.JK.circulantdot(Xreg[0][0])
                        else:
                            self.VXreg[d[0], d[1], d[2]] =  self.JK.cdot(Xreg[0][0])



                    else:

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
            print("Warning: Robust orientation not tested")
            if self.circulant:
                return (self.JpqXreg[dL[0], dL[1], dL[2]].tT().circulantdot(self.XregT[dM[0], dM[1], dM[2]].tT()) + \
                    self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.JpqXreg[dM[0], dM[1], dM[2]]) - \
                    self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]])), \
                    (self.n_occ, self.n_virt, self.n_occ, self.n_virt) #.cget(M).reshape(self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())

            else:
                #return self.XregT[dL[0], dL[1], dL[2]].cdot(self.VXreg[dM[0], dM[1], dM[2]], coords = [M]).cget(M).reshape(self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())
                return (self.JpqXreg[dL[0], dL[1], dL[2]].tT().cdot(self.XregT[dM[0], dM[1], dM[2]].tT(), coords = [M]) + \
                    self.XregT[dL[0], dL[1], dL[2]].cdot(self.JpqXreg[dM[0], dM[1], dM[2]], coords = [M]) - \
                    self.XregT[dL[0], dL[1], dL[2]].cdot(self.VXreg[dM[0], dM[1], dM[2]], coords = [M])), \
                    (self.n_occ, self.n_virt, self.n_occ, self.n_virt) #.cget(M).reshape(self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())

        
        
        else:
            #print("Return:")
            if self.circulant:
                return self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]]), \
                    (self.n_occ, self.n_virt, self.n_occ, self.n_virt)
            else:
                return self.XregT[dL[0], dL[1], dL[2]].dot(self.VXreg[dM[0], dM[1], dM[2]]), \
                    (self.n_occ, self.n_virt, self.n_occ, self.n_virt)



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
                    self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]])).cget(M).reshape(self.n_occ, self.n_virt, self.n_occ, self.n_virt)

            else:
                #return self.XregT[dL[0], dL[1], dL[2]].cdot(self.VXreg[dM[0], dM[1], dM[2]], coords = [M]).cget(M).reshape(self.p.get_nocc(), self.p.get_nvirt(), self.p.get_nocc(), self.p.get_nvirt())
                return (self.JpqXreg[dL[0], dL[1], dL[2]].tT().cdot(self.XregT[dM[0], dM[1], dM[2]].tT(), coords = [M]) + \
                    self.XregT[dL[0], dL[1], dL[2]].cdot(self.JpqXreg[dM[0], dM[1], dM[2]], coords = [M]) - \
                    self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]])).cget(M).reshape(self.n_occ, self.n_virt, self.n_occ, self.n_virt)


        else:
            circulant = self.circulant
            for d in [dL, dM]:
                if self.XregT[d[0], d[1], d[2]] is 0:

                    #coords_q = tp.lattice_coords()
                    t0 = time.time()
                    Xreg = self.cfit.get(np.array([d]))
                    t1 = time.time()


                    # Should compute all within range
                    self.XregT[d[0], d[1], d[2]] = Xreg[0].tT()
                    if circulant:
                        self.VXreg[d[0], d[1], d[2]] =  self.JK.circulantdot(Xreg[0])
                    else:
                        self.VXreg[d[0], d[1], d[2]] =  self.JK.cdot(Xreg[0])
                    t2 = time.time()
                    print("        On-demand calculation:", d, "(%.1f + %.1f s)" % (t1-t0, t2-t1))
                    #self.Xreg[d[0], d[1], d[2]] =  self.XregT[d[0], d[1], d[2]].tT() #transpose matrix

            if circulant:
                return self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]]).cget(M).reshape(self.n_occ, self.n_virt, self.n_occ, self.n_virt)

            else:
                return self.XregT[dL[0], dL[1], dL[2]].cdot(self.VXreg[dM[0], dM[1], dM[2]], coords = [M]).cget(M).reshape(self.n_occ, self.n_virt, self.n_occ, self.n_virt)

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




def tdot(p, c1,v,c2, auxname, coords = np.array([[0,0,0]])):
    """
    'Minimal-route' calculation of triple product c1*v*c2

    v = the two index Coulomb matrix
    auxname = name of auxiliary basis file
    
    Since not all blocks may be required in the end-product c1*(sc)
    it is possible to compute a minimal number of intermediate products.
    """
    # Find required coordinates in s
    vcoords = tcoords_metric(c1,c2, coords)
     
    # Compute potential extra coordinates in s
    vcoords_extra = c2_not_in_c1(vcoords, v.coords)
    if vcoords_extra.shape[0]>0:
        vn_coords = np.zeros((v.coords.shape[0]+vcoords_extra.shape[0], 3), dtype = int)
        vn_coords[:v.coords.shape[0]] = v.coords
        vn_coords[v.coords.shape[0]:] = vcoords_extra


        vn_blocks = np.zeros((v.coords.shape[0]+vcoords_extra.shape[0], v.blocks.shape[1], v.blocks.shape[2]), dtype = float)
        vn_blocks[:v.coords.shape[0]] = v.cget(v.coords)
        

        block_matrix = tp.tmat()
        block_matrix.load_nparray(np.ones((vcoords_extra.shape[0], 2,2)), vcoords_extra)
        #self.JK = 
        vn_blocks[v.coords.shape[0]:] = compute_JK(p,block_matrix, coulomb=True, auxname = auxname).cget(vcoords_extra)

        v = tp.tmat()
        v.load_nparray(vn_blocks,vn_coords, screening = False)
    
    
    return v, c1.cdot(v.cdot(c2, coords = tcoords(c1, coords)), coords = coords)
    #return v, c1.circulantdot(v.cdot(c2, coords = tcoords(c1, coords))) # POssibly do circulantdot here

def tcoords(c1,coords = np.array([[0,0,0]])):
    """
    identify unique intermediate coordinates for triple-product, metric/middle toeplitz matrix
    """
    nsep = 200

    
    
    nc = (-c1.coords[None,:] + coords[:,None]).reshape(c1.coords.shape[0]*coords.shape[0], 3)
    return nc[np.unique(np.dot(nc-nc.min(), np.array([nsep**0,nsep**1,nsep**2])), return_index = True)[1]]

def tcoords_metric(c1,c2, coords):
    """
    Identify required indices to compute in middle matrix when contracting 
    
    c1 * middle * c2
    
    
    """
    nsep = 200

    #print(coords)

    #print(c1.coords.shape, coords.shape)

    nc = (-c1.coords[None,:] + coords[:,None]).reshape(c1.coords.shape[0]*coords.shape[0], 3)
    c1coords = nc[np.unique(np.dot(nc-nc.min(), np.array([nsep**0,nsep**1,nsep**2])), return_index = True)[1]]
    
    nc = (c1coords[None,:] - c2.coords[:,None]).reshape(c1coords.shape[0]*c2.coords.shape[0], 3)
    
    return nc[np.unique(np.dot(nc-nc.min(), np.array([nsep**0,nsep**1,nsep**2])), return_index = True)[1]]




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
    parser = argparse.ArgumentParser(prog = "Periodic Resolution of Identity framework - Debug tests",
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
