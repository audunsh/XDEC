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




def occ_virt_split(c,p):
    """
    Split matrix c(tmat object) in occupied and virtual columns depending on information contained in p (prism object)
    Returns two tmat objects c_occ and c_virt (typically coefficients)
    """

    c_virt = tp.tmat()
    c_virt.load_nparray(c.blocks[:-1,:,p.get_nocc()+p.n_core:], c.coords[:], screening = False)

    c_occ = tp.tmat()
    c_occ.load_nparray(c.blocks[:-1,:,p.n_core:p.get_nocc()+p.n_core], c.coords[:], screening = True)

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

    Jmn.load_nparray(vint.reshape((s.coords.shape[0], blockshape[0], blockshape[1])), s.coords)

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
            rmax = np.sqrt(np.sum(np.dot(cube,p.lattice)**2, axis = 1))[cmax]
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
    coords = tp.lattice_coords([12,12,12]) #assumed max extent

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
        big_tmat = tp.tmat()

        big_tmat.load_nparray(np.ones((cube.shape[0], 2,2),dtype = float),  cube)

        Jmnc = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = [coords[m]])
        #print("mn-screen:", m, coords[m], np.sqrt(d2)[m], np.max(np.abs(Jmnc.blocks[:-1])))
        if np.max(np.abs(Jmnc.blocks[:-1]))<xi0:
            #print(m)
            break

    xi_0_domain = coords[:m]*1
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
    cube = cube[np.argsort(np.sum(np.dot(cube, p.lattice)**2, axis = 1))]
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


def contract_occupied(vals):
    coords, Jmn, NL, NJ, Nn, Np, Nq, coord, c_occ = vals

    tj = np.zeros((NJ, Np, NL, Nn), dtype = float) # LJpn

    for i in np.arange(coords.shape[0]):
        # For all dN offsets in (LJ|0 m dN n)

        Jmnc2 = Jmn[i]
        Jmnc_coords = -c_occ.coords - coords[i] # HERE
        Jmnc2blocks = Jmnc2.cget(Jmnc_coords).reshape(NL, NJ,Nn,Nn )



        occupied_coords = -Jmnc_coords - c_occ.coords[coord]

        cb = c_occ.cget(occupied_coords)
        if True:
            dotk(tj, NL, Jmnc2blocks, NJ, Nn, cb, Np)
        else:
            b1 = Jmnc2blocks.swapaxes(2,3).reshape(NL, NJ*Nn, Nn)
            screening = np.max(np.abs(cb), axis = (1,2))>1e-12
            NLs = np.sum(screening)

            tjk = np.einsum("LJn,Lnp->LJp",b1[screening], cb[screening], optimize = True).reshape(NLs, NJ, Nn, Np).swapaxes(2,3) #reshape(NLs, NJ*Np, Nn)
            #print(tjk.shape, tj.shape)
            indxs = np.arange(NL)[screening]
            for k in np.arange(NLs):
                #print(tj[:,:, k, :].shape, k)
                tj[:,:, indxs[k], :]= tjk[k]


            #tj[:,:, np.arange(NL)[screening], :]= tjk
            #tj[:,:, screening, :] = t



        """
        for k in np.arange(NL):
            b1 = Jmnc2blocks[k].swapaxes(1,2).reshape(NJ*Nn, Nn) #Jn,m
            b2 = cb[k] #m,p
            tj[:,:,k,:] += np.dot(b1,b2).reshape(NJ, Nn, Np).swapaxes(1,2) #Jn,p->J,p,k,n
        """

    tj = tj.reshape(NJ*Np, NL*Nn) #L J p n -> p J L n -> J p L n

    return tj




def dotk(tj, NL, Jmnc2blocks, NJ, Nn, cb, Np):
    screening = np.max(np.abs(cb), axis = (1,2))>1e-12
    for k in np.arange(NL)[screening]:
        b1 = Jmnc2blocks[k].swapaxes(1,2).reshape(NJ*Nn, Nn) #Jn,m
        b2 = cb[k] #m,p

        tj[:,:,k,:] += np.dot(b1,b2).reshape(NJ, Nn, Np).swapaxes(1,2) #Jn,p->J,p,k,n






class coefficient_fitter_static():
    """
    Coefficient fitting for the integrator, with integrals stored in memory


    """

    def __init__(self, c_occ,c_virt, p, attenuation, auxname, JK, JKInv, screening_thresh = 1e-12, robust = False, circulant = True, xi0 = 1e-10, xi1 = 1e-10, float_precision = np.float64):
        self.robust = robust
        self.coords = []
        self.Jmn = []
        self.attenuation = attenuation
        self.screening_thresh = screening_thresh
        self.p = p
        #self.c = c
        self.JK = JK
        self.JKInv = JKInv
        self.circulant = circulant
        #self.c_occ, self.c_virt = occ_virt_split(self.c,p)
        self.c_occ = c_occ
        self.c_virt = c_virt
        self.float_precision = float_precision
        if self.robust:
            self.Jmnc = []


        xi_domain = estimate_attenuation_domain(p, attenuation = attenuation, xi0 = xi0,  auxname = auxname)



        for i in np.arange(len(xi_domain)):
            # Compute JMN with nsep =  c2

            c2, big_tmat = xi_domain[i]


            if True:
                """
                Alternative screening approach, somewhat inefficient but avoids premature truncation
                It basically computes a large chunk, presumably larger than required
                TODO: check boundaries for significant integrals, break if present
                """
                Nc = 10
                if i == 0:
                    cellcut = 30 # bohr
                if p.cperiodicity == "POLYMER":
                    Rc = np.sqrt(np.sum(self.p.coor2vec(tp.lattice_coords([Nc,0,0]))**2, axis = 1))
                    bc = tp.lattice_coords([Nc,0,0])[Rc<=cellcut]
                    
                elif p.cperiodicity == "SLAB":
                    Rc = np.sqrt(np.sum(self.p.coor2vec(tp.lattice_coords([Nc,Nc,0]))**2, axis = 1))
                    bc = tp.lattice_coords([Nc,Nc,0])[Rc<=cellcut]
                else:
                    Rc = np.sqrt(np.sum(self.p.coor2vec(tp.lattice_coords([Nc,Nc,Nc]))**2, axis = 1))
                    bc = tp.lattice_coords([Nc,Nc,Nc])[Rc<=cellcut] #Huge domain, will consume some memory
                #if i == 0:
                cellmax = np.max(np.sqrt(np.sum(self.p.coor2vec(bc)**2, axis = 1)))

                big_tmat = tp.tmat()
                big_tmat.load_nparray(np.ones((bc.shape[0], 2,2), dtype = float), bc)

                Jmnc2_temp = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()
                Jmnc2_temp.set_precision(self.float_precision)

                screen = np.max(np.abs(Jmnc2_temp.blocks[:-1]), axis = (1,2))>=xi0
                screen[np.sum(bc**2, axis = 1)==0] = True
                print("Attenuation screening induced sparsity is %i of a total of %i blocks." %( np.sum(screen), len(screen)))

                Jmnc2_max =  np.max(np.sqrt(np.sum(self.p.coor2vec(Jmnc2_temp.coords[screen])**2, axis = 1)))
                if cellmax-Jmnc2_max <= 1e-9:
                    print("Warning: Jmnc2 fit for c = ", c2, " extends beyond truncation threshold.")
                    print("         Jmnc2_max = %.2e,    truncation_threshold = %.2e" % (Jmnc2_max, cellcut))
                    print("         Truncation threshold (cellcut) should be increased.") 
                cellcut =  Jmnc2_max+1.0 #update truncation threshold

                Jmnc2 = tp.tmat()

                Jmnc2.load_nparray(Jmnc2_temp.blocks[:-1][screen], Jmnc2_temp.coords[screen])


            else:

                Jmnc2 = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()
                Jmnc2.set_precision(self.float_precision)


            self.coords.append(c2)
            self.Jmn.append( Jmnc2 ) #New formulation without uppercase-transpose



            print("Intermediate overlaps (LJ|0mNn) with N =", c2, " included with %i blocks and maximum absolute %.2e" % (Jmnc2.blocks.shape[0],np.max(np.abs(Jmnc2.blocks)) ))


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


        t0 = time.time()
        for coord in np.arange(self.c_occ.coords.shape[0]):
            # For all offsets in the coefficient matrix



            NL = self.c_occ.coords.shape[0] # Number of cells
            NJ = self.Jmn[0].blockshape[0] # Number of aux functions
            self.NJ = self.Jmn[0].blockshape[0]
            Np = self.c_occ.blockshape[1] # Number of occupieds
            Nq = self.c_virt.blockshape[1] # Number of virtuals
            Nn = self.c_occ.blockshape[0] # Number of AO functions




            vals = [self.coords, self.Jmn, NL, NJ, Nn, Np, Nq, coord, self.c_occ]

            tj = contract_occupied(vals)


            # further delimination
            screening = []
            vectors   = []

            indx = np.arange(tj.shape[1])

            for k in np.arange(tj.shape[0]):
                vindx = np.abs(tj[k,:])>xi1 # screening step, these elements are retained
                screening.append(indx[vindx])
                vectors.append( tj[k,vindx] )
                total += tj.size
                compr += vectors[-1].size


            self.Jmnc_sparse_tensors.append(vectors)
            self.Jmnc_sparse_screening.append(screening)

            print("\r>> Contracted occupieds (LJ|0p Mn) for L = ", self.c_occ.coords[coord], "(%.2f percent complete, compression rate: %.2e)" % (100.0*coord/self.c_occ.coords.shape[0], 100.0*compr/total), end='')
        print(time.time()-t0)
        print("Screening-induced sparsity is at %.2e percent." % (100.0*compr/total))



    def get(self, coords_q):
        c_occ, c_virt = self.c_occ, self.c_virt
        #c = self.c
        JK = self.JK
        Jpq_c = []
        Jpq_c_coulomb = []


        for i in np.arange(coords_q.shape[0]):
            # Initialize empty coefficient matrices
            Jpq_c.append(tp.tmat())
            Jpq_c[-1].load_nparray(np.ones((c_occ.coords.shape[0], JK.blockshape[0], c_occ.blockshape[1]*c_virt.blockshape[1]),dtype = float),  c_occ.coords)
            Jpq_c[-1].blocks *= 0


        NL = self.c_occ.coords.shape[0]
        NJ = self.NJ # Number of aux functions
        Np = c_occ.blockshape[1] # Number of occupieds
        Nq = c_virt.blockshape[1] # Number of virtuals
        Nn = c_occ.blockshape[0] # Number of AO functions





        for j in np.arange(coords_q.shape[0]):
            for coord in np.arange(c_occ.coords.shape[0]):

                block_j = np.zeros((NJ*Np, Nq), dtype = float)

                # ALTERATION II

                #cvirt_n = c_virt.cget(self.c.coords + coords_q[j]).reshape(NL*Nn, Nq)
                cvirt_n = c_virt.cget(self.c_occ.coords - self.c_occ.coords[coord] + coords_q[j]).reshape(NL*Nn, Nq)


                sparse_tensors = self.Jmnc_sparse_tensors[coord]
                sparse_screening = self.Jmnc_sparse_screening[coord]

                for k in np.arange(NJ*Np):
                    screen = np.array(sparse_screening[k], dtype = int)

                    if(len(screen)>0):
                        #print(screen)
                        #print(sparse_tensors[coord][k])
                        #print(cvirt_n[screen, :])

                        block_j[k, :] = np.dot(sparse_tensors[k], cvirt_n[screen, :])


                block_j.reshape(NJ, Np, Nq)


                Jpq_c[j].blocks[Jpq_c[j].mapping[ Jpq_c[j]._c2i(c_occ.coords[coord]) ]] =  block_j.reshape(NJ, Np*Nq)



                Jpq_c[j].blocks[-1] *= 0
                    #print(Jpq_c[j].blocks[-1])
            #print("Contraction of virtuals:", time.time()-t0)

        X = []
        for i in np.arange(len(Jpq_c)):
            t0 = time.time()
            if self.circulant:
                """
                A range of methods has been tested, the svd_solve seems to be fastest and stable
                """
                #X.append(self.JKInv.circulantdot(Jpq_c[i]))
                #X.append(self.JK.kspace_linear_solve(Jpq_c[i]))
                #X.append(self.JK.kspace_cholesky_solve(Jpq_c[i]))
                X.append(self.JK.kspace_svd_solve(Jpq_c[i])) #, complex_precision = np.complex128))
                #print(X[-1].blocks.dtype)

            else:
                X.append(self.JKInv.cdot(Jpq_c[i]))
            print("Solving for Jpqc:", time.time()-t0)

        return X










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
    def __init__(self, c_occ, c_virt,p, attenuation = 0.1, auxname = "cc-pvdz-ri", initial_virtual_dom = [1,1,1], circulant = True, extent_thresh = 1e-14, robust  = False, ao_screening = 1e-12, inverse_test = True, coulomb_extent = None, JKa_extent = None, xi0 = 1e-10, xi1 = 1e-10, float_precision = np.float64):
        self.c_occ = c_occ
        self.c_virt = c_virt
        self.p = p
        self.attenuation = attenuation
        self.auxname = auxname
        self.circulant = circulant
        self.robust = robust
        self.float_precision = float_precision

        # Oneshot calculations:
        # build attenuated JK matrix and inverse
        #big_tmat = estimate_attenuation_distance(p, attenuation = .5*self.attenuation, thresh = 1e-14, auxname = auxname)
        #big_tmat = estimate_attenuation_distance(p, attenuation = self.attenuation, thresh = extent_thresh, auxname = auxname)


        #big_tmat = estimate_center_domain(p, attenuation = attenuation, xi0 = xi0, auxname=auxname)
        if p.cperiodicity=="POLYMER":
            bc = tp.lattice_coords([5,0,0])
        if p.cperiodicity=="SLAB":
            bc = tp.lattice_coords([5,5,0])
        if p.cperiodicity=="CRYSTAL":
            bc = tp.lattice_coords([5,5,5])
        
        big_tmat = tp.tmat()
        big_tmat.load_nparray(np.ones((bc.shape[0], 2,2)), bc)

        cmax = big_tmat.coords[np.argmax(np.sum(big_tmat.coords**2, axis = 1))]
        cmax = np.argmax(np.sqrt(np.sum(np.dot(big_tmat.coords,p.lattice)**2, axis = 1)))

        """
        if JKa_extent is not None:
            big_tmat = tp.tmat()
            scoords = tp.lattice_coords(JKa_extent)
            big_tmat.load_nparray(np.ones((scoords.shape[0],2,2), dtype = float), scoords)
        """

        #self.JKa = compute_JK(self.p,self.c, attenuation = attenuation, auxname = auxname)
        self.JKa = compute_JK(self.p,big_tmat, attenuation = attenuation, auxname = auxname)
        self.JKa.set_precision(self.float_precision)
        #print(self.JKa.blocks.shape)
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
        self.JKinv.set_precision(self.float_precision)
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

        #print("Coeff fitter static tresh set to 1e-8")
        t0 = time.time()
        self.cfit = coefficient_fitter_static(self.c_occ, self.c_virt, p, attenuation, auxname, self.JKa, self.JKinv, screening_thresh = ao_screening, robust = robust, circulant = circulant, xi0=xi0, xi1=xi1, float_precision = self.float_precision)
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

        #s = tp.tmat()
        #scoords = tp.lattice_coords(coulomb_extent)
        #s.load_nparray(np.ones((scoords.shape[0],2,2), dtype = float), scoords)

        #self.JK = compute_JK(p,self.JKa, coulomb=True, auxname = self.auxname)
        self.JK = compute_JK(p,big_tmat, coulomb=True, auxname = self.auxname)
        self.JK.set_precision(self.float_precision)

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
