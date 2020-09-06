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

import gc

import multiprocessing as mp
        

#from memory_profiler import profile

#from pympler import muppy, summary

#from guppy import hpy

#os.environ["LIBINT_DATA_PATH"] = os.getcwd() #"/usr/local/libint/2.5.0-beta.2/share/libint/2.5.0-beta.2/basis/"
#os.environ["CRYSTAL_EXE_PATH"] = "/Users/audunhansen/PeriodicDEC/utils/crystal_bin/"

# Auxiliary basis set manipulations

def basis_trimmer(p, auxbasis, alphacut = 0.1, trimlist = None):
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
    for li in range(len(trimmed_basis_list)):
        l = trimmed_basis_list[li]
        if trimlist is None:
            trimmed_basis += l
        else:
            if trimlist[li]:
                trimmed_basis += l

    return trimmed_basis

def basis_scaler(p, auxbasis, alphascale = 1.0, trimlist = None):
    """
    # SCALE basis 
    """
    f = open(auxbasis, "r")
    basis = f.readlines()
    trimmed_basis_list = []
    for line in basis:
        try:
            # We only retain basis functions with exponent > alphacut
            exponent = literal_eval(line.split()[0])
            weight   = literal_eval(line.split()[1])
            
            trimmed_basis_list.append("    %.8f   %.8f \n" % (np.log(alphascale*np.exp(exponent)), weight))
            #trimmed_basis_list.append("    %.8f   %.8f" (np.log(alphascale*np.exp(exponent)), weight))



        except:
            trimmed_basis_list.append(line)
    f.close()

    trimmed_basis = ""
    for li in range(len(trimmed_basis_list)):
        l = trimmed_basis_list[li]
        if trimlist is None:
            trimmed_basis += l
        else:
            if trimlist[li]:
                trimmed_basis += l

    return trimmed_basis



def extract_atom_from_basis( atom, bname = "ri-fitbasis.g94", header = True):

    f = open("ri-fitbasis.g94", "r")
    atom_ = atom + f.read().split(atom)[1].split("****")[0]
    header = atom_.split("\n")[0]
    body   = atom_.split("\n")[1:]
    #body   = f.read().split(atom)[1].split("\n")[1].split("****")[0]
    #print(body)
    #print(header)
    if header:
        return atom_
    else:
        return body


    
def get_basis(atoms, bname = "ri-fitbasis.g94"):
    basis = ""
    for a in atoms:
        basis += extract_atom_from_basis(a, bname)
    return basis

def get_basis_list(atoms, bname ="ri-fitbasis.g94", prescreen = None):
    angmom = {"S":0, "P":1, "D":2, "F":3, "G":4, "H":5}
    basis = []
    
    bstring = "****"
    
    c = 0
    for a in atoms:
        
        bstring += "\n"
        bset = extract_atom_from_basis(a, bname, header = False).split("\n")[1:]
        
        bset_string = extract_atom_from_basis(a, bname, header = False).split("\n")[0] + "\n"
        #print(bset_string)
        
        for i in range(int(len(bset)/2)):
            #print("func:", bset[2*i].split()[0])
            #print(bset[2*i +1])
            
            basis.append([angmom[bset[2*i].split()[0]], literal_eval(bset[2*i +1].split()[0]) ])
            
            
            if prescreen is not None:
                if prescreen[c]:
                    # remove function in question
                    basis = basis[:-1]
                else:
                    # add function to string
                    bset_string += bset[2*i] + "\n" + bset[2*i + 1] + "\n"
                    
            else:
                bset_string += bset[2*i] + "\n" + bset[2*i + 1] + "\n"
                
            c += 1
        bstring += bset_string + "****"
        
    #print(bstring)
    basis = np.array(basis)
    
    # generate a map from basis index to input file order
    
    #mp = np.zeros(int(np.sum(2*basis[:,0]+1)), dtype = int)
    mc = []
    for i in np.arange(len(basis)):
        l = 2*basis[i,0] + 1
        for j in range(int(l)):
            mc.append(i)
        #mp[nc:l+1] = i
    
    return np.array(mc), bstring, basis.shape[0], basis



def remove_redundancies(p, N_c, basis_input, plotting = False, analysis = True, tolerance = 1e-6, attenuation = 0.0):
    """
    Removes redundancies in auxiliary basis sets by systematic reduction of the condition by means of a SVD
    """
    
    atoms_table = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
    
    atoms = [atoms_table[i-1] for i in p.charges]
    
    ### Fitting basis
    auxbasis = basis_trimmer(p, basis_input, alphacut = 0.0)
    #auxbasis = s
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()


    prescreen = None
    
    n_points = n_points_p(p, N_c)

    while True:

        mc, bs, bl, basis_list = get_basis_list(atoms, "ri-fitbasis.g94", prescreen = prescreen)





        f = open("ri-fitbasis.g94", "w")
        f.write(bs)
        f.close()

        #f = open("ri-fitbasis.g94", "r")
        #print(f.read())
        #f.close()
        newk = 1
        Nb = N_c*1



        sc = tp.get_random_tmat(n_points, [2,2])

        JKa  =  compute_JK(p, sc, attenuation = attenuation, auxname = "ri-fitbasis")
        JKs = JKa.fft(n_layers = n_points)

        if analysis:
            newk = 1
            bands = JKa.get_kspace_eigenvalues(n_points = n_points, sort = True)
            bands = np.roll(bands, -np.int(Nb*newk)-1, axis = 0)
            print(bands.shape)

            bands = bands.reshape(np.prod(bands.shape[:3]), JKa.blocks.shape[1]).real

            kp = np.arange(-Nb*newk, newk*Nb+1)
            if plotting:
                plt.figure(1, figsize = (10,4))
                for i in range(-Nb, Nb+1):
                    plt.axvline(i, linewidth = 0.5, color = (0,0,0))
                plt.plot(kp/newk, bands, "-", alpha = 1, linewidth = 1)
                plt.yscale("symlog")
                plt.xlabel("$|| \mathbf{m} ||^{-1}$ (reciprocal coordinate)")
                #plt.ylim(1e-12,1000)
                #plt.xscale("symlog")
                plt.axhline(0)


                plt.show()

            """
            plt.figure(3, figsize = (14,3))
            for i in np.arange(-Nb, Nb+1):
                plt.axvline(i, linewidth = 0.5, color = (0,0,0), alpha = .3)
            plt.plot(kp/newk, bands, alpha = .9)
            plt.yscale("symlog")
            plt.ylim(-1e-2,1e-2)
            plt.axhline(0)



            plt.show()"""


            #print(bands[:,0,0][:,0 ].real.min())
            #print(bands[:,0,0][:,0 ].real.max())

            cond = np.max(bands,axis  = 0)/np.min(bands, axis = 0)
            cond2 = np.max(np.abs(bands),axis  = 0)/np.min(np.abs(bands), axis = 0)

            cond3 = np.max(bands,axis  = 1)/np.min(bands, axis = 1)
            cond4 = np.max(np.abs(bands),axis  = 1)/np.min(np.abs(bands), axis = 1)


            # check close eigenvalues / degeneracy at every kpoints


            #print(cond.real)
            #plt.plot(cond.real, ".")
            #plt.yscale("log")
            #plt.show()
            
            if plotting:


                if np.any(cond<0):
                    plt.figure(1, figsize = (10,4))
                    for i in np.arange(-Nb, Nb+1):
                        plt.axvline(i, linewidth = 0.5, color = (0,0,0), alpha = .3)
                    plt.plot(kp/newk, bands.real[:,cond<0], alpha = .9)
                    plt.yscale("symlog")
                    #plt.ylim(-0.0001,0.0001)
                    plt.axhline(0)


                    plt.show()

            print("BASIS EIGENVALUE SPECTRUM ANALYSIS")
            print("sign change occurs within band:", np.any(cond<0)) # imminent catastrophy
            print("    max and min within band   : %.3e %.3e" % (np.max(cond2), np.min(cond2)))
            print("abs max and min within band   : %.3e %.3e" % (np.max(cond2), np.min(cond2)))
            print("    max and min overall       : %.3e %.3e" % (np.max(bands), np.min(bands)))
            print("abs max and min overall       : %.3e %.3e" % (np.max(np.abs(bands)), np.min(np.abs(bands))))
            print("    max condition at kpoint   : %.3e" % np.max(cond3))
            print("abs max condition at kpoint   : %.3e" % np.abs(np.max(cond4)))
            JKainv = JKa.inv()

            # inversion test
            I = JKainv.circulantdot(JKa)-tp.get_identity_tmat(JKa.blocks.shape[1])
            #print(np.mean(JKa_2inv.blocks.real), np.abs(JKa_2inv.blocks.real).max())
            print("Dev. from unity in inverse    : %.3e" % np.abs(I.blocks).max())




        done = True

        vf = 0
        for coord in JKs.coords:
            u,d,vh = np.linalg.svd(JKs.cget(coord))
            

            # remove relatively small eigenvals
            vi = d<d.max()*tolerance




            #print(d.max())
            if np.any(vi==True):
                dm = d**-1
                #dm[vi==False] = 0
                #vf += vh.T.dot(np.diag(dm).dot(u.T)).real
                vf += u.dot(np.diag(dm).dot(vh)).real
                vf += u[:, vi].dot(np.diag(dm[vi]).dot(vh[vi, :])).real
                done = False


        if not done:
            vfx = np.sum(np.abs(vf), axis = 0)
            vfy = np.sum(np.abs(vf), axis = 1)
            
            bs_to_remove = np.argsort(vfx)[::-1][0]

            prescreen = np.zeros(bl, dtype = np.bool)
            prescreen[mc[bs_to_remove]] = True

            print("Removing shell from aux.basis:", mc[bs_to_remove])
            print("                              ", basis_list[mc[bs_to_remove]])

        else:
            print("===================================================================================")
            print("Basis is determined to be sufficiently independent for fitting at given resolution.")
            print("")
            print(bs)
            print("")
            print("Stored to file: ri-fitbasis.g94")
            print("===================================================================================")
            break
    return bs


## Toeplitz related


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
    #print(sret)
    #
    #print(" ----- ")
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

def compute_onebody(p,s, T = np.array([[0,0,0]]), nearfield = np.array([[0,0,0]]), operator = "overlap", conversion_factor = .5291772109200000):
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
        atomsK0 = "atoms_K0.xyz"
        lint.set_operator_nuclear()
        



    lint.setup_pq(atomsJ, bname, atomsK, bname)
    if operator == "nuclear":
        f = open(atomsK0, "w")
        f.write(get_xyz(p, nearfield, conversion_factor=conversion_factor))
        f.close()
        
        lint.set_charges(atomsJ,atomsK0)

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
    #print(vint.shape)
    s = tp.tmat()
    s.load_nparray(vint, T)
    return s

def compute_JK(p, s, attenuation = 0, auxname = "cc-pvdz", coulomb = False):
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

def compute_JK_auto(p, s, attenuation = 0, auxname = "cc-pvdz", coulomb = False):
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
    return JK.inv()
    """
    n_points = np.array(tp.n_lattice(JK))
    #print(n_points)
    JKk = tp.transform(JK, np.fft.fftn, n_points = n_points)
    JKk_inv = JKk*1.0
    JKk_inv.blocks[:-1] = np.linalg.pinv(JKk.blocks[:-1])



    JK_inv_direct = tp.transform(JKk_inv, np.fft.ifftn, n_points = n_points, complx = False)

    return JK_inv_direct
    """


def estimate_attenuation_distance(p, attenuation = 0.0, c2 = [0,0,0], thresh = 10e-12, auxname = "cc-pvdz-ri"):
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

def estimate_attenuation_domain(p, attenuation = 0.0, xi0 = 1e-8,  auxname = "cc-pvdz-ri"):
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

def estimate_center_domain(p, attenuation = 0.0, xi0 = 1e-8,  auxname = "cc-pvdz-ri"):
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


def estimate_attenuation_distance_(p, attenuation = 0.0, c2 = [0,0,0], thresh = 10e-12, auxname = "cc-pvdz-ri"):
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
    
    def compute_Jmn(self, c2, thresh = 1e-10):
        
        Jmn_full = []
        N_blocks = 0
        
        
        # compute first block
        #PRI.compute_Jmn(self.p, s, attenuation = self.attenuation, auxname =self.basis, nshift = np.array([c2]))
        
        for i in np.arange(self.Ru.shape[0]-1):
            r0 = self.Ru[i]
            r1 = self.Ru[i+1] # +0.1
            
            
            s = tp.tmat()
            scoords = self.get_shell(r0,r1)
            
            s.load_nparray(np.ones((len(scoords), 2,2), dtype = float), scoords)
            #print(s.coords)
            
            Jmn_shell = compute_Jmn(self.p, s, attenuation = self.attenuation, auxname =self.basis, nshift = np.array([c2]))
            
            
            cmM = np.abs(Jmn_shell.blocks).max()
            #print("cmM:", cmM)
            #print("estimate_coordinate_domain",c2,": computing shell ", r0, "to", r1," max value:", cmM) #+0.1)
            #print(scoords)

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
            r1 = self.Ru[i+1] # +0.1
            
            
            s = tp.tmat()
            scoords = self.get_shell(r0,r1)
            #print("shell:, r0, r1", r0, r1,scoords)
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
        Jmn = []
        self.attenuation = attenuation
        #self.screening_thresh = screening_thresh
        print("att:", attenuation)
        self.p = p
        #self.c = c
        self.N_c = N_c
        self.JK = JK
        self.primed = False

        

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

        N_c_max_layers = 30
        cm = estimate_coordinate_domain(p, auxname, N_c_max_layers, attenuation = attenuation)

        C2 = tp.lattice_coords(n_points_p(p, N_c_max_layers))

        R = np.sqrt(np.sum(p.coor2vec(C2)**2, axis = 1))

        C2 = C2[np.argsort(R)]
        R = R[np.argsort(R)]

        #self.printing = True

        #print("C2:", C2, len(C2))

        

        
        #for i in np.arange(len(xi_domain)):

        for ci in range(len(C2)):
            c2, R2 = C2[ci], R[ci] 
            # Compute JMN with nsep =  c2
            #print("ci, c2, R2:", ci, c2, R2)

            #c2, big_tmat = xi_domain[i]

            




        
            #for i in np.arange(ddM.shape[0]):
            #c2 = ddM[i]
            #print(cm.attenuation)


        

            #Jmnc2 = compute_Jmn(p,big_tmat, attenuation = attenuation, auxname = auxname, coulomb = False, nshift = np.array([c2])) #.T()
            Jmnc2_temp = cm.compute_Jmn(c2, thresh = xi0)
            #print(Jmnc2_temp.coords) 


            Jmnc2_temp.set_precision(self.float_precision)
            #print("Number of blocks in ")

            screen = np.max(np.abs(Jmnc2_temp.cget(Jmnc2_temp.coords)), axis = (1,2))>=xi0
            screen[np.sum(Jmnc2_temp.coords**2, axis = 1)==0] = True
            
            

            # Screened distances
            distances = np.sqrt(np.sum(self.p.coor2vec(Jmnc2_temp.coords)**2, axis = 1))

            Jmnc2_max =  np.max(distances[screen])

            
            if np.sum(screen)>1:
                fc = 0.95
                outer_coords = []
                #print(Jmnc2_temp.coords)
                """
                while len(outer_coords) == 0:
                    fc *= 0.95
                    outer_coords = Jmnc2_temp.coords[distances>Jmnc2_max*fc]
                    
                    #if len(outer_coords)>0:
                    #    break
                    #else:
                    #    fc *= 0.8
                """
                outer_coords = Jmnc2_temp.coords[distances>Jmnc2_max*fc]


                
                max_outer_5pcnt = np.max(Jmnc2_temp.cget(outer_coords))

                max_coord = np.max(np.abs(Jmnc2_temp.coords), axis = 0)
                
                if self.printing:
                    
                    print("Attenuation screening induced sparsity is %i of a total of %i blocks." %( np.sum(screen), len(screen)))
                    print("         Maximum value in outer %.2f percentage of block (rim) :" % (100*fc,  max_outer_5pcnt))
                    print("         Maximum value overall                              :", np.max(np.abs(Jmnc2_temp.blocks)))
                    print("         c2 = ", c2)
                    print("         R  = ", R2, "(", max_coord, ")")




                
                
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

            #if np.linalg.norm(Jmnc2.cget([0,0,0]))>1e-15:
            self.coords.append(c2)
            Jmn.append( Jmnc2 ) #New formulation without uppercase-transpose

            print("max | (LJ | 0 m [%i %i %i] n) |  = " % tuple(c2), "%.4e" % np.abs(np.max(Jmnc2.blocks)), " max (L) <=",  np.max(np.abs(Jmnc2.coords), axis = 0))

            #print("Min overall:", np.max(np.abs(Jmnc2.blocks)))


            if self.printing:
                print("Intermediate overlaps (LJ|0mNn) with N =", c2, " included with %i blocks and maximum absolute %.2e" % (Jmnc2.blocks.shape[0],np.max(np.abs(Jmnc2.blocks)) ))
                print("Max interm.:", np.abs(Jmnc2.blocks).max())
            if np.max(np.abs(Jmnc2.blocks))<xi0 and np.abs(R[ci]-R[ci+1])>1e-10:
                #print("Min overall:", np.max(np.abs(Jmnc2.blocks)))
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

        

        
        
        #pq_region = self.JK.coords

        #self.pq_region = pq_region[np.argsort(np.sum(p.coor2vec(pq_region)**2, axis = 1))]

        #if self.N_c>0:
        #    pq_region = tp.lattice_coords(n_points_p(self.p, self.N_c))
        #else:

        


        Nc = 50
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

        #self.pq_region = self.JK.coords[np.argsort(np.sum(p.coor2vec(self.JK.coords)**2, axis = 1))]*1

        #print(self.JK.coords)
        #print(self.pq_region)
        #print(self.c_occ.coords)

        

        # Contracting occupieds

        

        self.OC_L_np, self.c_virt_coords_L, self.c_virt_screen, self.NJ, self.Np = contract_occupieds(self.p, Jmn, self.coords, self.pq_region, self.c_occ, self.xi1)



        
        # Can delete self.Jmn now, no longer required and takes up a lot of memory
        del(Jmn)
        #import gc 
        #print(gc.garbage)
        #gc.collect()
        

        if self.printing:

            #print("\r>> Contracted occupieds (LJ|0p Mn) for L = ", self.c_occ.coords[coord], "(%.2f percent complete, compression rate: %.2e)" % (100.0*coord/self.c_occ.coords.shape[0], 100.0*compr/total), end='')
            print("\r>> Contracted occupieds (LJ|0p Mn) ")
        
        
        
        
        if self.printing:
            print(time.time()-t0)
            #print("Screening-induced sparsity is at %.2e percent." % (100.0*compr/total))
        

    def set_n_layers(self, n_layers, rcond, inv = "lpinv"):
        self.JKa = self.JK.get_prepared_circulant_prod(n_layers = n_layers, inv = True, rcond = rcond, inv_mod = inv)
        #self.JKa = self.JK.get_prepared_circulant_prod(n_layers = n_layers, inv = False, rcond = rcond)
        self.primed = True

    def get(self, coords_q, robust = False):
        pq_c = []


        

        #print(self.JK.cutoffs(), Jpq_c[i].cutoffs())
        

        for dM in np.arange(coords_q.shape[0]):
            if robust:
                J_pq_c = contract_virtuals(self.OC_L_np, self.c_virt_coords_L, self.c_virt_screen, self.c_virt, self.NJ, self.Np, self.pq_region, dM = coords_q[dM])

                
                
                #n_points = n_points_p(self.p, np.max(np.abs(J_pq_c.coords)))
                
                #
                #
                if self.primed:
                    pq_c.append([
                            self.JKa.circulantdot(
                                J_pq_c, complx = False), J_pq_c])
                        # test solution
                    #print(" Solution satisfied:", np.abs((self.JK.circulantdot(pq_c[-1][0]) - J_pq_c).cget(tp.lattice_coords(n_points))).max())
                    #print(" ", pq_c[-1].blocks.max(), np.abs(self.JKa.M1).max())
                else:
                    if self.N_c>0:
                        #print("Userdefined coulomb extent:", self.N_c)
                        #print(self.JK.coords)
                        n_points = n_points_p(self.p, self.N_c)
                    pq_c.append([self.JK.kspace_svd_solve(J_pq_c, n_points = n_points), J_pq_c ])
            else:
                J_pq_c = contract_virtuals(self.OC_L_np, self.c_virt_coords_L, self.c_virt_screen, self.c_virt, self.NJ, self.Np, self.pq_region, dM = coords_q[dM])

                #print("J_pq_c.coords:", np.max(np.abs(J_pq_c.coords), axis = 0))


                
                
                #n_points = n_points_p(self.p, np.max(np.abs(J_pq_c.coords)))
                n_points = n_points_p(self.p, np.max(np.abs(self.JK.coords)))
                #n_points = np.max()
                if self.N_c>0:
                    #print("Userdefined coulomb extent:", self.N_c)
                    #print(self.JK.coords)
                    n_points = n_points_p(self.p, self.N_c)
                #print("coefficient fitter statig, n_points:", n_points)
                #print("C                      J_pq_c.shape:", J_pq_c.blocks.shape)

                if self.primed:
                    pq_c.append(
                        self.JKa.circulantdot(
                            J_pq_c, complx = False))

                    #pq_c.append(
                    #    self.JKa.linear_solve(
                    #        J_pq_c, complx = False))
                    if True:
                        # test solution - remove if stability problems are solved
                        d = pq_c[-1]
                        ssat = np.abs((self.JK.circulantdot(d) - J_pq_c).blocks).max()
                        #ssat= np.abs(self.JK.cget(self.JK.coords) - d.tT().circulantdot(self.JK.circulantdot(d)).cget(self.JK.coords)).max()
                        print("   Solution satisfied:", ssat ) #np.abs((self.JK.circulantdot(pq_c[-1]) - J_pq_c).cget(tp.lattice_coords(n_points))).max())
                        print("   Inversion ok      :", np.max(np.abs((self.JKa.circulantdot(self.JK).cget(self.JK.coords) - tp.get_identity_tmat(self.JK.blocks.shape[1]).cget(self.JK.coords))), axis = (0,1,2)))
                    
                    print("   Max DF-coefficient:", np.abs(pq_c[-1].blocks).max()) #, np.abs(self.JKa.M1).max())
                    #print("J_pq_c.blocks active:", np.max(np.abs(J_pq_c.cget(self.JK.coords)), axis = (1,2)))




                else:
                    pq_c.append(
                        self.JK.kspace_svd_solve(
                            J_pq_c,
                            n_points = n_points))
                
                
                '''
                pq_c.append(
                    self.JK.gamma_inv().circulantdot(
                        J_pq_c, n_layers = n_points*0))
                '''
                    




                #print("                   pq_c.blocks.shape:", pq_c[-1].blocks.shape)

        
        #all_objects = muppy.get_objects()
        #sum1 = summary.summarize(all_objects)
        # Prints out a summary of the large objects
        #print(" --- cfit")
        #summary.print_(sum1)
        #print(" ---")

        
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
    for i in range(len(c_virt_coords_L)):
        NN = len(c_virt_coords_L[i])
        sc = c_virt_screen[i]
        
        
        Jpq_blocks[i] = np.dot(OC_L_np[i],c_virt.cget(c_virt_coords_L[i]+dM).reshape(NN*Nn,Nq)[sc,:]).reshape(NJ, Np*Nq)

        

    Jpq_blocks = Jpq_blocks.reshape(NL, NJ, Np*Nq)

    

    Jpq = tp.tmat()
    Jpq.load_nparray(Jpq_blocks[:-1], -1*pq_region[:NL-1])    
    #for coord in pq_region[:NL]:
    #    
    #    print("Normcheck: (jpq):", coord, np.linalg.norm(Jpq.cget([coord])))
    #    #print(J)

    return Jpq

def pcontr(pooldata):
    Jsc, NJ, Nm, Nn, Np, csc, k = pooldata
    return [np.dot(Jsc[k].reshape(NJ,Nm,Nn).swapaxes(1,2).reshape(NJ*Nn,Nm), csc[k]).reshape(NJ, Nn, Np), k]


#@profile
def contract_occupieds(p, Jmn_dm, dM_region, pq_region, c_occ, xi1 = 1e-10, mpi = False):
    """
    Intermediate contraction of the occupieds.
    See section 3.2, algorithm 1 and 2 in the notes for details
    Author: Audun
    """

    #print("pq_region:", pq_region)



    


    O = []

    # dimensions to match the equations
    
    NN = Jmn_dm[0].coords.shape[0] # Number of blocks in coefficients
    
    Np = c_occ.blocks.shape[2]  # number of occupieds
    Nn = c_occ.blocks.shape[1]  # number of ao functions
    Nm = c_occ.blocks.shape[1]  # number of ao functions


    # Set max possible extent

    #n_points = np.max(np.abs(dM_region), axis = 0) + np.max(np.abs(Jmn_dm[0].coords), axis = 0)
    #s = tp.get_zero_tmat(n_points_p(p, 10), [2,2])
    #s = tp.get_zero_tmat(n_points, [2,2])
    #NN = s.coords.shape[0]      # number of 
    #print("domain setup in contract_occupieds may not be optimal")

    


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

    #gc.disable()
    if mpi:
        # Multiprocessing
        
        comm = MPI.COMM_WORLD 
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        rank = 0


    for Li in range(pq_region.shape[0]):
        L = pq_region[Li]
        #print("Contracting occupieds for :", L)
        O_LN_np = np.zeros((NN, NJ,Nn, Np), dtype = float)

        #N_coords = -Jmn_dm[0].coords 
        
        

        # J = -N - dM - L, L in all
        # O = N + dM = -(J) - L 
        # 
        # N_coords = tp.lattice_coords([np.max(np.abs(Jmnc.coords), axis = 0) + np.abs()])
        norm_prev = 0.0
        

        for dMi in range(dM_region.shape[0]):
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
            

            #scale = 0.0001

            # Screen out zero blocks here 
            # cs = np.max(np.abs(c_occ_blocks), axis = (1,2))>xi1
            """
            cs = np.any(np.greater(np.abs(c_occ_blocks),xi1), axis = (1,2))


            # bs = np.max(np.abs(Jmn_blocks), axis = (1,2,3))>xi1
            bs = np.any(np.greater(np.abs(Jmn_blocks), xi1), axis = (1,2,3))



            #print(np.argmax(np.abs(Jmn_blocks), axis = (1,2,3)))
            sc = np.logical_and(cs, bs)


            sc = np.logical_and(np.max(np.abs(c_occ_blocks), axis = (1,2))>xi1,  \
                                np.max(np.abs(Jmn_blocks), axis = (1,2,3))>xi1)



            """
            


            sc = np.max(np.abs(c_occ_blocks), axis = (1,2))>xi1
            sc[np.max(np.abs(Jmn_blocks), axis = (1,2,3))<=xi1] = False

            #print(sc.shape)


            #import sys
            #sys.exit()

            #print("mp:Number of processors: ", mp.cpu_count())

            
            

            #print("screening:", time.time()-t0)
            #print("")
            #t0 = time.time()
            
            #print("number of sc:", np.sum(sc))
            if np.any(sc):
                # If any block non-zero : contract occupieds
                #dO_LN_np = np.einsum("NJmn,Nmp->NJnp", Jmn_blocks[sc], c_occ_blocks[sc], optimize = True) #change

                Jsc, csc = Jmn_blocks[sc], c_occ_blocks[sc]
                scs = np.sum(sc)
                dO_LN_np = np.zeros((scs, NJ, Nn, Np), dtype = float)

                if mpi:
                    if rank==0:
                        k, dk = comm.recv(source=0, tag=11)
                        dO_LN_np[k] = dk

                    else:

                        
                        for k in range(scs):
                            #print(pres[k][1], pres[k][0].shape)
                            if (rank-1)==f%(size-1):
                                data = [k, np.dot(Jsc[k].reshape(NJ,Nm,Nn).swapaxes(1,2).reshape(NJ*Nn,Nm), csc[k]).reshape(NJ, Nn, Np)]


                                comm.send(data, dest=0, tag=11)


                            #dO_LN_np[ pres[k][1] ] = pres[k][0]



                        pooldata = []
                        for k in range(scs):
                            pooldata.append([ Jsc, NJ, Nm, Nn, Np, csc, k])

                        pres = pool.map(pcontr, pooldata)

                        #print(pres)

                        



                else:
                    #print("summing over ", scs, "elements.")
                    for k in range(scs):
                        dO_LN_np[k] =  np.dot(Jsc[k].reshape(NJ,Nm,Nn).swapaxes(1,2).reshape(NJ*Nn,Nm), csc[k]).reshape(NJ, Nn, Np)
                    
                
                #print("for k in sc:", time.time()-t0)
                #print("")
                #t0 = time.time()

                
                




                

                #if np.abs(dO_LN_np).max()<scale*xi1:
                #    print("Break at dM=", dM)
                #    break

                O_LN_np[sc] += dO_LN_np

                norm_new = np.linalg.norm(O_LN_np)

                ndiff = np.abs(norm_prev-norm_new)

                if ndiff/norm_new<xi1:
                    #print("Break at ", dM)
                    break
                

                
                #print("O_LN_np_norm:", ndiff, ndiff/norm_new)
                norm_prev = norm_new*1


            #print("if np.sumsc:", time.time()-t0)
            #print("")
            #t0 = time.time()
        #gc.collect()

        

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
        
        sc = np.max(np.abs(O_LN_np), axis = 0)>xi1
        #sc[:] = True
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

        
        
        if np.abs(O_LN_np).max()<xi1:
            dR = np.abs(np.sum(p.coor2vec(pq_region[Li-1])**2) - np.sum(p.coor2vec(pq_region[Li])**2))

            if dR > 1e-12:
                print("    ->   Truncation of intermediate tensor O at L = ", L, " with dR = %.2e  <-" % dR)
                print("         Final compression ratio of intermediate tensor is %.2e" % (elms_retained/elms_total))
                print("                              (%i out of %i elements retained)" % (elms_retained, elms_total))
                print("         Max value : %.2e" % np.abs(O_LN_np).max())
                break
        

        # Screen out negligible 
    #gc.enable()
    return OC_L_np, c_virt_coords_L, c_virt_screen, NJ, Np

def contract_occupieds_(p, Jmn_dm, dM_region, pq_region, c_occ, xi1 = 1e-10):
    """
    Intermediate contraction of the occupieds.
    See section 3.2, algorithm 1 and 2 in the notes for details
    Author: Audun
    """

    #print("pq_region:", pq_region)



    


    O = []

    # dimensions to match the equations
    
    NN = Jmn_dm[0].coords.shape[0] # Number of blocks in coefficients
    
    Np = c_occ.blocks.shape[2]  # number of occupieds
    Nn = c_occ.blocks.shape[1]  # number of ao functions
    Nm = c_occ.blocks.shape[1]  # number of ao functions


    # Set max possible extent

    #n_points = np.max(np.abs(dM_region), axis = 0) + np.max(np.abs(Jmn_dm[0].coords), axis = 0)
    #s = tp.get_zero_tmat(n_points_p(p, 10), [2,2])
    #s = tp.get_zero_tmat(n_points, [2,2])
    #NN = s.coords.shape[0]      # number of 
    #print("domain setup in contract_occupieds may not be optimal")

    


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
        norm_prev = 0.0
        

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
            

            #scale = 0.0001

            # Screen out zero blocks here 
            #cs = np.max(np.abs(c_occ_blocks), axis = (1,2))>xi1
            cs = np.any(np.greater(np.abs(c_occ_blocks),xi1), axis = (1,2))


            #bs = np.max(np.abs(Jmn_blocks), axis = (1,2,3))>xi1
            bs = np.any(np.greater(np.abs(Jmn_blocks), xi1), axis = (1,2,3))



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

                
                




                

                #if np.abs(dO_LN_np).max()<scale*xi1:
                #    print("Break at dM=", dM)
                #    break

                O_LN_np[sc] += dO_LN_np

                norm_new = np.linalg.norm(O_LN_np)

                ndiff = np.abs(norm_prev-norm_new)

                if ndiff/norm_new<xi1:
                    print("Break at ", dM)
                    break
                

                
                print("O_LN_np_norm:", ndiff, ndiff/norm_new)
                norm_prev = norm_new*1


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
        
        sc = np.max(np.abs(O_LN_np), axis = 0)>xi1
        #sc[:] = True
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

        
        
        if np.abs(O_LN_np).max()<xi1:
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
    
    def __init__(self, c_occ, c_virt,p, attenuation = 0.0, auxname = "cc-pvdz-ri", initial_virtual_dom = [1,1,1], circulant = True, robust  = False,  inverse_test = True, coulomb_extent = None, JKa_extent = None, xi0 = 1e-10, xi1 = 1e-10, float_precision = np.float64, printing = True, N_c = 10, rcond = 1e-10, inv = "lpinv"):
        self.c_occ = c_occ
        self.c_virt = c_virt
        self.p = p
        self.attenuation = attenuation
        self.auxname = auxname
        self.circulant = circulant
        self.robust = robust
        self.float_precision = float_precision
        self.N_c = N_c # (2*N_c + 1)  = k-space resolution

        self.tprecision = np.complex128 #precision in final d.T V d matrix product, set to complex64 for reduced memory usage

        self.screening_thresh = 1e-10
        self.screen_trigger = 0
        self.activation_count = 60 # activate global screening when this many cells has been screened

        self.inv = inv

        self.n_occ = c_occ.blocks.shape[2]
        self.n_virt = c_virt.blocks.shape[2]

        self.d_forget = [] #a list of orientations to forget (memory handling)

        # Oneshot calculations:
        # build attenuated JK matrix and inverse
        #big_tmat = estimate_attenuation_distance(p, attenuation = .5*self.attenuation, thresh = 1e-14, auxname = auxname)
        #big_tmat = estimate_attenuation_distance(p, attenuation = self.attenuation, thresh = extent_thresh, auxname = auxname)


        #print("Self.N_c:", self.N_c)


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
        #self.JKa = cm.compute_JK(thresh = xi0)
        #self.JKa.set_precision(self.float_precision)
        
        #self.JKa = compute_JK(p, tp.get_random_tmat(np.max(np.abs(self.JKa.coords), axis = 0), [2,2]), attenuation = attenuation, auxname = auxname)
        #print(tp.lattice_coords(n_points_p(p, self.N_c)))
        if self.N_c >0:
            self.JKa = compute_JK(p, tp.get_random_tmat(n_points_p(p, self.N_c), [2,2]), attenuation = attenuation, auxname = auxname)
        else:
            N_c_max_layers = 20
            cm = estimate_coordinate_domain(p, auxname, N_c_max_layers, attenuation = attenuation)
            self.JKa = cm.compute_JK(thresh = xi1)
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

        self.JKinv = 0 #invert_JK(self.JKa)
        #self.JKinv.set_precision(self.float_precision)
        #inverse_test = True
        if inverse_test:
            self.JKinv = invert_JK(self.JKa)
            self.JKinv.set_precision(self.float_precision)
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


        mx_l = 35 # maximum number of 


        self.XregT = np.zeros((mx_l,mx_l,mx_l), dtype = tp.tmat)  # RI - coefficient matrices ^T
        self.VXreg = np.zeros((mx_l,mx_l,mx_l), dtype = tp.tmat) # RI - matrices with V contracted


        self.Xscreen = np.ones((mx_l,mx_l,mx_l), dtype = np.float) #for integral screening
        self.Xdist = np.zeros((mx_l,mx_l,mx_l), dtype = np.float) #for integral screening extrapolation
        



        if robust:
            self.JpqXreg = np.zeros((mx_l,mx_l,mx_l), dtype = tp.tmat)

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
        if self.N_c >0:
            self.cfit.set_n_layers(n_points_p(p, self.N_c), rcond = rcond, inv = self.inv)
            self.n_layers = n_points_p(p, self.N_c)
        else:
            #J_pq_c = contract_virtuals(self.OC_L_np, self.c_virt_coords_L, self.c_virt_screen, self.c_virt, self.NJ, self.Np, self.pq_region, dM = coords_q[dM])
            #print(np.max(np.abs(self.JKa.coords), axis = 0))
            self.n_layers = np.max(np.abs(-1*self.cfit.pq_region[:len(self.cfit.c_virt_coords_L)]), axis = 0)
            #self.cfit.set_n_layers(np.max(np.abs(self.JKa.coords), axis = 0), rcond = rcond)
            self.cfit.set_n_layers(self.n_layers, rcond = rcond, inv = self.inv)
        
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
                    #print("domain info:", coord_q[i], np.max(np.abs(Xreg[i].coords), axis = 0))
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
            

            #coulomb_extent = (10,10,10)
            #print("Extent of Coulomb matrix:", coulomb_extent)
        #coulomb_extent = (3,0,0)
        #if self.N_c>0:
        #    coulomb_extent = n_points_p(p, 2*self.N_c)
        print("Extent of Xreg          :", coulomb_extent)
        
        #coulomb_extent = [3,0,0]
        #print("Extent of Xreg          :", coulomb_extent)

        s_ = tp.tmat()
        s_coords = tp.lattice_coords(self.n_layers*2)
        s_.load_nparray(np.ones((s_coords.shape[0],2,2), dtype = float), s_coords)

        #self.JK = compute_JK(p,self.JKa, coulomb=True, auxname = self.auxname)
        #self.JK = compute_JK(p,big_tmat, coulomb=True, auxname = self.auxname)
        #self.JK = compute_JK(p,self.JKa, coulomb=True, auxname = self.auxname)
        self.JK = compute_JK(p,s_, coulomb=True, auxname = self.auxname)
        self.JK.set_precision(self.float_precision)
        print("COULOMB SET TO DOUBLE SIZE!")

        print("JKa domain:", np.max(np.abs(self.JKa.coords), axis = 0), self.JKa.coords.shape[0])

        print("JK  domain:", np.max(np.abs(self.JK.coords), axis = 0), self.JK.coords.shape[0])
        print("Number of AUX-functions:", self.JK.blocks.shape[1])

       # dm, de = self.JKa.absmax_decay(self.p)
       # print("absmax decay", dm, de)




        #self.cfit.JKa = compute_JK(p,s, attenuation = attenuation, auxname = self.auxname)
        
        
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
        #print("Coulomb (JK) shape:", self.JK.blocks.shape)
        #print(self.XregT.nbytes)
        
        if not keep:
            for d in dels:
                self.XregT[d[0], d[1], d[2]] = 0
        return v_pqrs, (self.n_occ, self.n_virt, self.n_occ, self.n_virt)





    def getorientation(self, dL, dM, adaptive_cdot = False, M=None):
        
        
        D = [] #to be forgotten
        if adaptive_cdot:
            # not yet implementet



            return None
        

        else:
            # use circulant fomulation
            for d in [dL, dM]:
                #print(d)
                if self.XregT[d[0], d[1], d[2]] is 0 and self.Xscreen[d[0], d[1], d[2]]>=self.screening_thresh:
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
                            #self.VXreg[d[0], d[1], d[2]] =  self.JK.cdot(Xreg[0])
                        else:
                            self.VXreg[d[0], d[1], d[2]] =  self.JK.cdot(Xreg[0])

                        print("        On-demand calculation:", d)
                        #import sys
                        #sys.exit()
                        

                    #D.append(d)
                    self.d_forget.append(d)
                    
                    #del(Xreg)

        #print(" ... ibuild --- ")
        
        #all_objects = muppy.get_objects()
        #sum1 = summary.summarize(all_objects)
        # Prints out a summary of the large objects
        #summary.print_(sum1)
            
        
        if self.robust:
            print("Warning: Robust orientation not tested")
            if self.circulant:
                mmm = self.JpqXreg[dL[0], dL[1], dL[2]].tT().circulantdot(self.XregT[dM[0], dM[1], dM[2]].tT()) + self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]])
                #self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.JpqXreg[dM[0], dM[1], dM[2]])
               
               
                print("maxdevv:", np.abs(mmm.blocks).max())
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
            #print("Return:", dL, dM)
            if self.circulant:

                #for e in self.XregT.ravel():
                #    print(e.blocks.nbytes)
                #print("Tensor mem usage:", self.nbytes(), " bytes.")
                
                
                #return self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]]), \
                #    (self.n_occ, self.n_virt, self.n_occ, self.n_virt)

                
                if self.Xscreen[dL[0], dL[1], dL[2]] >= self.screening_thresh and self.Xscreen[dM[0], dM[1], dM[2]] >= self.screening_thresh:
                    ret = self.XregT[dL[0], dL[1], dL[2]].circulantdot(self.VXreg[dM[0], dM[1], dM[2]], precision = self.tprecision)

                    if np.sum((dL-dM)**2)==0 and self.Xdist[dL[0], dL[1], dL[2]] <=0.1:
                        self.Xscreen[dL[0], dL[1], dL[2]] = np.abs(ret.cget([0,0,0])).max()
                        self.Xdist[dL[0], dL[1], dL[2]] = np.sqrt(np.sum(self.p.coor2vec(dL)**2))
                        self.screen_trigger += 1
                        if np.sum(self.Xscreen<self.screening_thresh)>self.activation_count:
                            #if self.screen_trigger == self.activation_count:
                            print("---------------------------")
                            print("Activating global screening at ", self.activation_count, " n. screens performed.")

                            self.screening_cutoff = self.get_screening_cutoff()
                            print("Cutoff at ", self.screening_cutoff, " bohr.")


                            for i in np.arange(-17,18):
                                for j in np.arange(-17,18):
                                    for k in np.arange(-17,18):
                                        self.Xdist[i,j,k] = np.sqrt(np.sum(self.p.coor2vec([i,j,k])**2))
                                        if self.Xdist[i,j,k] >= self.screening_cutoff:
                                            self.Xscreen[i,j,k] = 0.0


                else:
                    #print("Integrals omitted due to screening:", dL, dM)
                    ret = tp.get_zero_tmat([1,1,0], [self.n_occ*self.n_virt, self.n_occ*self.n_virt])
                    #self.screen_trigger += 1
                
                return ret, (self.n_occ, self.n_virt, self.n_occ, self.n_virt)



            else:
                #print("Kept tensors")
                return self.XregT[dL[0], dL[1], dL[2]].cdot(self.VXreg[dM[0], dM[1], dM[2]]), \
                    (self.n_occ, self.n_virt, self.n_occ, self.n_virt)

    def get_screening_cutoff(self):
        e = np.log(self.Xscreen.ravel())
        d = self.Xdist.ravel()

        e = e[d!=0]
        d = d[d!=0]

        di = np.argsort(d)


        d = d[di]
        e = e[di]
        #nbins = 20
        #print(d)

        e_remain = e*1
        d_remain = d*1
        E = []
        D = []
        while len(d_remain) >0:
            ei = np.argmax(e_remain)
            E.append(e_remain[ei])
            D.append(d_remain[ei])
            
            
            
            d_remain = d_remain[ei+1:]*1
            e_remain = e_remain[ei+1:]*1







        x = np.polyfit(D,E, 1)

        np.save("screen_vals.npy", np.exp(e))
        np.save("screen_dist.npy", d)


        return (np.log(self.screening_thresh) - x[1]) / x[0] #cutoff distance



    def forget(self, memwall = None, retain = None):
        if retain is None:
            if memwall is None:
                for d in self.d_forget:

                    self.VXreg[d[0], d[1], d[2]] = 0
                    self.XregT[d[0], d[1], d[2]] = 0

                    #print("Fitting: Removed tensor: ", d)
                self.d_forget = []
            else:
                dynbuff = 0
                for d in self.d_forget:
                    
                    dynbuff += self.VXreg[d[0], d[1], d[2]].nbytes()
                    dynbuff += self.XregT[d[0], d[1], d[2]].nbytes()

                tot = self.nbytes()





                

                print("---------------------------------------")
                print("Integrator freeing up memory")
                print("Current total usage :", tot, "bytes.")
                print("Dynamic buffer usage:", dynbuff, "bytes / ", len(self.d_forget), "cells.")
                print("Freeing up ", memwall, "cells.")
                print("---------------------------------------")
                for d in self.d_forget[:memwall]:

                    self.VXreg[d[0], d[1], d[2]] = 0
                    self.XregT[d[0], d[1], d[2]] = 0

                self.d_forget = self.d_forget[memwall:]
        else:
            #print("---------------------------------------")
            #print("Integrator freeing up memory")
            keep = np.zeros(self.XregT.shape, dtype = bool)
            for d in retain:
                keep[d[0], d[1], d[2]] = True
            d_forget_new = []
            for d in self.d_forget:
                if not keep[d[0], d[1], d[2]]:
                    #print("Freeing up cell", d)
                    self.VXreg[d[0], d[1], d[2]] = 0
                    self.XregT[d[0], d[1], d[2]] = 0
                else:
                    d_forget_new.append(d)
            self.d_forget = d_forget_new
            #print("Retained integrals")
            #print(self.d_forget)
            


                


            
            
        
    def getcell_conventional(self, A, J, B):
        return self.getcell(A, J, B-J)


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
        total_mem_usage =  self.JK.blocks.nbytes
        for i in np.arange(self.VXreg.shape[0]):
            for j in np.arange(self.VXreg.shape[1]):
                for k in np.arange(self.VXreg.shape[2]):
                    if type(self.XregT[i,j,k]) is tp.tmat:
                        total_mem_usage += self.XregT[i,j,k].blocks.nbytes
                    if type(self.VXreg[i,j,k]) is tp.tmat:
                        total_mem_usage += self.VXreg[i,j,k].blocks.nbytes
        return total_mem_usage #*1e-6 #return in MB




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
