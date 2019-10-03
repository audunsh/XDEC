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

import PRI

import time


def test_ao_refcell(attenuation = .2):
    p = pr.prism("inputs/neon_3d.d12")

    # Compute overlap matrix
    #s = PRI.compute_onebody(p, s)

    # temporary libint data path
    # datapath = os.environ["LIBINT_DATA_PATH"]

    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    

    # build ao coeff matrix so that (i,a|i,a)_mo = (p,q,p,q)_ao
    # (0,0|0,0)_mo =  (2,2|2,2)_ao
    # (1,1|2,2)_mo  = (6,6|T_1|2,3)_ao #translated
    

    coords = tp.lattice_coords([1,1,1])
    c = tp.tmat()
    c.load_nparray(np.ones((coords.shape[0], p.get_n_ao(),p.get_n_ao()), dtype = float), coords)
    c.blocks *= 0

    c0 = np.zeros((p.get_n_ao(),p.get_n_ao()), dtype = float) #coefficients in refcell
    c0[2,0] = 1.0              # first occupied, ao # 2 is a diffuse s-type gaussian
    c0[6,1] = 1.0              # second occupied, ao # 2 is a diffuse s-type gaussian
    c0[2,p.get_nocc()] = 1.0   # first virtual
    c0[6,p.get_nocc()+1] = 1.0 # second virtual 
    c0[:,p.get_nocc()+3] = 1.0 # fourth virtual

    c.blocks[ c.mapping[ c._c2i([0,0,0])]] = c0

    c1 = np.zeros((p.get_n_ao(),p.get_n_ao()), dtype = float) #coefficients in refcell
    c1[2,2] = 1.0              #third occupied, ao # 2 is a diffuse s-type gaussian
    c1[3,p.get_nocc()+2] = 1.0 #third virtual
    

    c.blocks[ c.mapping[ c._c2i([1,0,0])]] = c1

    print(c.cget([0,0,0]))

    # generate fit-basis

    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvdz-ri.g94", alphacut = .9)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    # test integrals in refcell 

    ib = PRI.integral_builder_static(c,p,attenuation = attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=True, extent_thresh=1e-10, robust = False)
    
    i0, ishape = ib.getorientation([0,0,0],[0,0,0])
    pqrs_ex = PRI.compute_pqrs(p, np.array([[0,0,0]]))

    # Test max deviation
    #print("Maxdev:", np.max(np.abs(pqrs_ex[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():] - i0.cget([0,0,0]).reshape(ishape))))
     
    #import matplotlib.pyplot as plt
    #np.save("pqrs_ex", pqrs_ex[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():])
    #np.save("fitted", i0.cget([0,0,0]).reshape(ishape))

    


    # Test that a single AO in refcell is properly fitted
    print("Fit   :", i0.cget([0,0,0]).reshape(ishape)[0,0,0,0])
    print("Exact :", pqrs_ex[2,2,2,2])
    print(np.abs(pqrs_ex[2,2,2,2]-i0.cget([0,0,0]).reshape(ishape)[0,0,0,0])) #, "err"
    
    # Test that all a linear combination of all AOs in refcell is properly fitted
    print("Fit   :", i0.cget([0,0,0]).reshape(ishape)[0,0,0,3])
    print("Exact :", np.sum(pqrs_ex[2,2,2,:]))
    print(np.abs( i0.cget([0,0,0]).reshape(ishape)[0,0,0,3] - np.sum(pqrs_ex[2,2,2,:])))

    pqrs_ex = PRI.compute_pqrs(p, np.array([[-1,0,0]]))#[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():]

    #print(np.max(np.abs(pqrs_ex - i0.cget([0,0,0]).reshape(ishape))))
    # Test that AOs outside the refcell are properly fitted
    # NOTE: the convention we use is |Lp> := sum_{Mm} C^-M_{mp} |Mm>
    #       this must be taken into account in the indexing
    print("Fit   :", i0.cget([0,0,0]).reshape(ishape)[1,1,2,2])
    print("Exact :", pqrs_ex[6,6,2,3])
    print(np.abs(pqrs_ex[6,6,2,3]-i0.cget([0,0,0]).reshape(ishape)[1,1,2,2])) #, "err"

    # Test that translated product AOs are properly fitted
    print("Fit   :", i0.cget([1,0,0]).reshape(ishape)[0,0,0,0])
    print("Exact :", pqrs_ex[2,2,2,2])
    print(np.abs(pqrs_ex[2,2,2,2]-i0.cget([1,0,0]).reshape(ishape)[0,0,0,0])) #, "err"

    
    # test that translated AOs outside the refcell are properly fitted
    pqrs_ex = PRI.compute_pqrs(p, np.array([[-2,0,0]]))#[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():]
    print("Fit   :", i0.cget([-2,0,0]).reshape(ishape)[1,1,2,2])
    print("Fit-> :", i0.cget([-1,0,0]).reshape(ishape)[1,1,2,2]) # <-
    print("Fit   :", i0.cget([0,0,0]).reshape(ishape)[1,1,2,2])
    print("Fit   :", i0.cget([1,0,0]).reshape(ishape)[1,1,2,2])
    print("Fit   :", i0.cget([2,0,0]).reshape(ishape)[1,1,2,2])
    print("Exact :", pqrs_ex[6,6,2,3])
    print(np.abs(pqrs_ex[6,6,2,3]- i0.cget([-1,0,0]).reshape(ishape)[1,1,2,2]))

def test_ao_all():
    p = pr.prism("inputs/neon_3d.d12")

    # Compute overlap matrix
    #s = PRI.compute_onebody(p, s)

    # temporary libint data path
    # datapath = os.environ["LIBINT_DATA_PATH"]

    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    coords = tp.lattice_coords([1,1,1])
    c = tp.tmat()
    c.load_nparray(np.ones((coords.shape[0], p.get_n_ao(),p.get_n_ao()), dtype = float), coords)
    c.blocks *= 0

    c0 = np.eye(p.get_n_ao(), dtype = float) #coefficients in refcell
    
    c.blocks[ c.mapping[ c._c2i([0,0,0])]] = c0

    # generate fit-basis

    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvdz-ri.g94", alphacut = .9)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    # test integrals in refcell 

    ib = PRI.integral_builder_static(c,p,attenuation = .2, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=True, extent_thresh=1e-10, robust = False)
    
    i0, ishape = ib.getorientation([0,0,0],[0,0,0])
    pqrs_ex = PRI.compute_pqrs(p, np.array([[0,0,0]]))
    
    print("Maxdev:", np.max(np.abs(pqrs_ex[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():] - i0.cget([0,0,0]).reshape(ishape))))
    

def test_lih_ao():
    """
    Test that MO-integrals from previous XDEC code is reproduced in the case of LiH
    """

    p = pr.prism("/Users/audunhansen/papers/pao-paper/results/LiH_122018/Crystal/LiH_111218.d12")
    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    c = tp.tmat()
    c.load("/Users/audunhansen/papers/pao-paper/results/LiH_122018/crystal_reference.npy")
    c.blocks *= 0
    c.blocks[c.mapping[c._c2i([0,0,0])]] = np.eye(p.get_n_ao(), dtype = float)

    
    s = tp.tmat()
    s.load("/Users/audunhansen/papers/pao-paper/results/LiH_122018/S_crystal.npy")
    
    S = PRI.compute_onebody(p,s,s.coords)
    S = S.reshape(p.get_n_ao(), s.coords.shape[0], p.get_n_ao())
    S = np.swapaxes(S, 0,1)
    S_libint = tp.tmat()
    S_libint.load_nparray(S.reshape(s.coords.shape[0], p.get_n_ao(), p.get_n_ao()), s.coords)
    #S_libint = S_libint
    #print(S_libint.cget([1,0,0])[0])
    #print(s.cget([1,0,0])[:])
    #S_libint.save("new_overlap2.npy")
    print("Max devitation in overlap:", np.max(np.abs(s.cget(s.coords) - S_libint.cget(s.coords))))

    c000 = np.array([0.263028,0.122179,0.122179,0.122179,0,9.6662e-15,1.42742e-13,1.42742e-13,1.42742e-13])
    
    c100 = np.array([0.0497651,0.000956966,-0.000476781,-0.000476781,0,1.82921e-15,2.69096e-14,2.68543e-14,2.68543e-14])
    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvdz-ri.g94", alphacut = .5)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    ib = PRI.integral_builder_static(c,p,attenuation = .2, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=True, extent_thresh=1e-10, robust = False,ao_screening = 1e-7)
    
    i0, ishape = ib.getorientation([0,0,0],[0,0,0])

    fit = i0.cget([0,0,0]).reshape(ishape)
    v = np.arange(9)
    print(c000)
    print(fit[0,v,0,v])
    print("Abs dev :", np.max(np.abs(c000-fit[0,v,0,v])))
    #print(fit[0,0,v,v])

    fit = i0.cget([1,0,0]).reshape(ishape)
    print(c100)
    print(fit[0,v,0,v])
    print("Abs dev :", np.max(np.abs(c100-fit[0,v,0,v])))

def test_lih_mo(attenuation = .3):
    """
    Test that MO-integrals from previous XDEC code is reproduced in the case of LiH
    """

    p = pr.prism("/Users/audunhansen/papers/pao-paper/results/LiH_122018/Crystal/LiH_111218.d12")
    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    c = tp.tmat()
    c.load("/Users/audunhansen/papers/pao-paper/results/LiH_122018/crystal_reference.npy")
    #c.blocks *= 0
    #c.blocks[c.mapping[c._c2i([0,0,0])]] = np.eye(p.get_n_ao(), dtype = float)

    
    s = tp.tmat()
    s.load("/Users/audunhansen/papers/pao-paper/results/LiH_122018/S_crystal.npy")

    # test orthonormality

    smo = c.tT().cdot(s.cdot(c), coords = c.coords)
    for i in np.arange(5):
        print(i, np.max(np.abs(smo.cget([i,i,i]))))
    
    S = PRI.compute_onebody(p,s,s.coords)
    S = S.reshape(p.get_n_ao(), s.coords.shape[0], p.get_n_ao())
    S = np.swapaxes(S, 0,1)
    S_libint = tp.tmat()
    S_libint.load_nparray(S.reshape(s.coords.shape[0], p.get_n_ao(), p.get_n_ao()), s.coords)
    #S_libint = S_libint
    #print(S_libint.cget([1,0,0])[0])
    #print(s.cget([1,0,0])[:])
    #S_libint.save("new_overlap2.npy")
    print("Max devitation in overlap:", np.max(np.abs(s.cget(s.coords) - S_libint.cget(s.coords))))

    #c000 = np.array([0.263028,0.122179,0.122179,0.122179,0,9.6662e-15,1.42742e-13,1.42742e-13,1.42742e-13])
    
    #c100 = np.array([0.0497651,0.000956966,-0.000476781,-0.000476781,0,1.82921e-15,2.69096e-14,2.68543e-14,2.68543e-14])
    c000 = np.array([0.00013888,0.107661,0.107663,0.107664,4.39405e-05,0.000173713,0.000246441,0.107597,7.89759e-05])

    c100 = np.array([-1.72341e-05,-0.000273289,-0.00119725,-0.000304476,-5.46021e-06,-2.58003e-05,-3.38905e-05,-0.000398317,-9.37923e-06])
    
    
    
    
    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvdz-ri.g94", alphacut = .4)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    ib = PRI.integral_builder_static(c,p,attenuation = attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=True, extent_thresh=1e-10, robust = False,ao_screening = 1e-8)
    
    i0, ishape = ib.getorientation([0,0,0],[0,0,0])

    fit = i0.cget([0,0,0]).reshape(ishape)
    np.save("fit000.npy", fit)
    v = np.arange(9)
    print(c000)
    print(fit[0,v,0,v])
    print("========================")
    print(" ")
    print("Abs dev :", np.max(np.abs(c000-fit[0,v,0,v])))
    print(" ")
    print("========================")
    #print(fit[0,0,v,v])
    #print([7.46984355e-06, 1.08231020e-01, 1.08233234e-01, 1.08233838e-01, 4.11946015e-06, 9.70397972e-06, 1.39753261e-05, 1.08155724e-01, 4.57334127e-06])

    """
    fit = i0.cget([1,0,0]).reshape(ishape)
    print(c100)
    print(fit[0,v,0,v])
    np.save("fit100.npy", fit)
    print("Abs dev :", np.max(np.abs(c100-fit[0,v,0,v])))
    """


def test_Jmn():
    """
    Test consistency of three-index integrals
    """

    p = pr.prism("/Users/audunhansen/papers/pao-paper/results/LiH_122018/Crystal/LiH_111218.d12")
    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    s = tp.tmat()
    s.load("/Users/audunhansen/papers/pao-paper/results/LiH_122018/S_crystal.npy")
    
    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvdz-ri.g94", alphacut = .9)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    Jmnc = PRI.compute_Jmn(p,s, attenuation = 0.1, auxname = "ri-fitbasis", coulomb = False, nshift = [[0,0,0]])


    onecoord = tp.tmat()
    onecoord.load_nparray(np.ones((1,2,2), dtype = float), np.array([[0,0,0]]))
    

    Jmnc2 = PRI.compute_Jmn(p,onecoord, attenuation = 0.1, auxname = "ri-fitbasis", coulomb = False, nshift = [[0,0,0]])

    print(np.max(np.abs(Jmnc.cget([0,0,0])-Jmnc2.cget([0,0,0]))))


def test_exact_mo():
    """
    compare against exact mos in refcell
    """

    """
    p = pr.prism("/Users/audunhansen/papers/pao-paper/results/LiH_122018/Crystal/LiH_111218.d12")
    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    c = tp.tmat()
    c.load("/Users/audunhansen/papers/pao-paper/results/LiH_122018/crystal_reference.npy")
    """
    p = pr.prism("inputs/neon_3d.d12")
    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    c = tp.tmat()

    c.load("/Users/audunhansen/papers/XDEC-RI-MP2/results/neon_3d/psm1.npy")

    # computing p0q0r0s0
    c2 = tp.lattice_coords([1,1,1])
    c2 = c2[np.argsort(np.sum(c2**2, axis = 1))]
    print(c2)

    c_occ, c_virt = PRI.occ_virt_split(c,p)
    i = 0
    v = np.arange(p.get_nvirt())
    pqrs = np.zeros((p.get_nocc(), p.get_nvirt(), p.get_nocc(), p.get_nvirt()), dtype = float)
    for dL in c2:
        for M in c2:
            for dM in c2:
                for L in c_occ.coords:
                    # compute ao 
                    lL, ldL, lM, ldM =L,  dL+L, M + L, dM+L
                    mnkl = PRI.compute_pqrs_(p, [lL], [ldL], [ldM])#[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():]
                    
                    #mnkl = PRI.compute_pqrs_(p, [-1*(dL-L)], [-1*(M-L)], [-1*(dM-L)])#[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():]
                    pqrs += np.einsum("mnjk,mp,nq,jr,ks->pqrs", mnkl, c_occ.cget(lL), c_virt.cget(ldL), c_occ.cget(lM), c_virt.cget(ldM))
                i += 1
                print(i/c_occ.coords.shape[0], pqrs[0,v,0,v])
                np.save("neon_pqrs.npy", pqrs)

def test_exact_mo_neo():
    """
    compare against exact mos in refcell
    """




    p = pr.prism("inputs/neon_3d.d12")
    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    c = tp.tmat()

    c.load("/Users/audunhansen/papers/XDEC-RI-MP2/results/neon_3d/psm1.npy")

    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvdz-ri.g94", alphacut = .1)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    ib = PRI.integral_builder_static(c,p,attenuation = 1.0, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=True, extent_thresh=1e-9, robust = False,ao_screening = 1e-8)
    
    i0, ishape = ib.getorientation([0,0,0],[0,0,0])
    fit = i0.cget([0,0,0]).reshape(ishape)[0,np.arange(p.get_nvirt()),0,np.arange(p.get_nvirt()) ]
    cex = np.array([0.06547297,0.06542333,0.06544593,0.06547479])
    print("fit:", fit)
    print("ex :", cex)
    print("(estimated)")
    print(" -- ")
    print("Maxdiff:", np.max(np.abs(cex - fit)))
    print(" --")
    print(" ")

def test_x(attenuation = .4):
    
    p = pr.prism("/Users/audunhansen/papers/pao-paper/results/LiH_122018/Crystal/LiH_111218.d12")
    os.environ["LIBINT_DATA_PATH"] = os.getcwd()


    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    
    coords = tp.lattice_coords([2,2,2])
    c = tp.tmat()


    c.load_nparray(np.ones((coords.shape[0], p.get_n_ao(),p.get_n_ao()), dtype = float), coords)
    c.blocks *= 0

    c0 = np.zeros((p.get_n_ao(),p.get_n_ao()), dtype = float) #coefficients in refcell
    c0 = 0*np.eye(p.get_n_ao(), dtype = float)
    c0[0,0] = 1.0
    #c0[:,1] = 0.0
    c0[2,2] = 1.0 #first virtual

    c0[:,4] = 1.0 #third virtual
    c0[:,5] = 1.0 #third virtual

    c.blocks[ c.mapping[ c._c2i([0,0,0])]] = c0 #np.eye(p.get_n_ao(), dtype = float)

    c1 = np.zeros((p.get_n_ao(),p.get_n_ao()), dtype = float) #coefficients in refcell
    c1[0,1] = 1.0 #second occupied
    c1[:,3] = 1.0 #second virtual
    c1[:,4] = 1.0 #third virtual
    
    c.blocks[ c.mapping[ c._c2i([-1,0,0])]] = c1 #np.eye(p.get_n_ao(), dtype = float)


    # generate fit-basis

    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvdz-ri.g94", alphacut = .4)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    print(c.cget([0,0,0]))
    print(c.cget([-1,0,0]))

    # test integrals in refcell 

    ib = PRI.integral_builder_static(c,p,attenuation = attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=True, extent_thresh=1e-10, robust = False)
    
    i0, ishape = ib.getorientation([0,0,0],[0,0,0])

    pqrs_ex  = PRI.compute_pqrs(p, np.array([[0,0,0]]))
    pqTrs_ex = PRI.compute_pqrs(p, np.array([[1,0,0]]))

    # Test max deviation
    #print("Maxdev:", np.max(np.abs(pqrs_ex[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():] - i0.cget([0,0,0]).reshape(ishape))))
     
    #import matplotlib.pyplot as plt
    #np.save("pqrs_ex", pqrs_ex[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():])
    #np.save("fitted", i0.cget([0,0,0]).reshape(ishape))

    


    # Test that a single AO in refcell is properly fitted



    v = np.arange(p.get_n_ao())
    v_ = np.arange(p.get_nvirt())
    print("Fit       :", i0.cget([0,0,0]).reshape(ishape)[0,0,1,1]) # = (0,2,0,:)_AO
    print("Exact (3):",  np.sum(pqTrs_ex[0,2,0,v]))
    print("Fit       :", i0.cget([0,0,0]).reshape(ishape)[0,0,1,2]) # = (0,2,0,:)_AO
    print("Fit       :", i0.cget([0,0,0]).reshape(ishape)[0,0,1,3]) # = (0,2,0,:)_AO
    print("Fit       :", i0.cget([0,0,0]).reshape(ishape)[0,0,1,4]) # = (0,2,0,:)_AO
    print("Exact (2):",  np.sum(pqrs_ex[0,2,0,v])+ np.sum(pqTrs_ex[0,2,0,v]))
    
    #print("Fit       :", i0.cget([0,0,0]).reshape(ishape)[0,0,1,4]) # = (0,2,0,:)_AO

    #print("Exact (1) :", np.sum(pqrs_ex[0,2,0,v]))
    
    print(" ")
    print("Diff  :", np.abs(i0.cget([0,0,0]).reshape(ishape)[0,0,1,1] - np.sum(pqrs_ex[0,2,0,v]))) #, "err"


def test_x_(attenuation = .4):
    
    p = pr.prism("/Users/audunhansen/papers/pao-paper/results/LiH_122018/Crystal/LiH_111218.d12")
    os.environ["LIBINT_DATA_PATH"] = os.getcwd()


    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    
    coords = tp.lattice_coords([2,2,2])
    c = tp.tmat()


    c.load_nparray(np.ones((coords.shape[0], p.get_n_ao(),p.get_n_ao()), dtype = float), coords)
    c.blocks *= 0

    c0 = np.zeros((p.get_n_ao(),p.get_n_ao()), dtype = float) #coefficients in refcell
    c0 = 0*np.eye(p.get_n_ao(), dtype = float)
    c0[0,0] = 1.0
    #c0[:,1] = 0.0
    c0[2,2] = 1.0 #first virtual

    c0[:,4] = 1.0 #third virtual
    c0[:,5] = 1.0 #third virtual

    c.blocks[ c.mapping[ c._c2i([0,0,0])]] = c0 #np.eye(p.get_n_ao(), dtype = float)

    c1 = np.zeros((p.get_n_ao(),p.get_n_ao()), dtype = float) #coefficients in refcell
    c1[0,1] = 1.0 #second occupied
    c1[:,3] = 1.0 #second virtual
    c1[:,4] = 1.0 #third virtual
    
    c.blocks[ c.mapping[ c._c2i([-1,0,0])]] = c1 #np.eye(p.get_n_ao(), dtype = float)


    # generate fit-basis

    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvdz-ri.g94", alphacut = .4)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    print(c.cget([0,0,0]))
    print(c.cget([-1,0,0]))

    # test integrals in refcell 

    ib = PRI.integral_builder_static(c,p,attenuation = attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=True, extent_thresh=1e-10, robust = False)
    
    i0, ishape = ib.getorientation([0,0,0],[0,0,0])

    pqrs_ex  = PRI.compute_pqrs_(p, np.array([[0,0,0]]),np.array([[0,0,0]]),np.array([[0,0,0]]))
    pqTrs_ex = PRI.compute_pqrs_(p, np.array([[0,0,0]]),np.array([[1,0,0]]),np.array([[1,0,0]]))
    pqRrs_ex = PRI.compute_pqrs_(p, np.array([[0,0,0]]),np.array([[0,0,0]]),np.array([[1,0,0]]))

    # Test max deviation
    #print("Maxdev:", np.max(np.abs(pqrs_ex[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():] - i0.cget([0,0,0]).reshape(ishape))))
     
    #import matplotlib.pyplot as plt
    #np.save("pqrs_ex", pqrs_ex[:p.get_nocc(), p.get_nocc():, :p.get_nocc(), p.get_nocc():])
    #np.save("fitted", i0.cget([0,0,0]).reshape(ishape))

    


    # Test that a single AO in refcell is properly fitted



    v = np.arange(p.get_n_ao())
    print("Integral(ao):  ( (000)0, (000)2 | (100) 0 , (100)(0+1+2+3+4+...+N_ao) )")
    print("Fit         :", i0.cget([0,0,0]).reshape(ishape)[0,0,1,1]) # = (0,2,0,:)_fit
    print("Exact (1)   :",  np.sum(pqTrs_ex[0,2,0,v])) # = (0,2|0,:) libint
    print("Diff        :", np.abs( i0.cget([0,0,0]).reshape(ishape)[0,0,1,1]) - np.sum(pqTrs_ex[0,2,0,v]))
    #print("Fit       :", i0.cget([0,0,0]).reshape(ishape)[0,0,1,2]) # = (0,2,0,:)_AO
    #print("Fit       :", i0.cget([0,0,0]).reshape(ishape)[0,0,1,3]) # = (0,2,0,:)_AO
    print("Integral(ao):  ( (000)0, (000)2 | (000) 0 , (000 & 100)(0+1+2+3+4+...+N_ao) )")
    #print("Fit         :", i0.cget([0,0,0]).reshape(ishape)[0,0,0,1]) # = (0,2,0,:)_AO
    print("Fit         :", i0.cget([0,0,0]).reshape(ishape)[0,0,0,2]) # = (0,2,0,:)_AO
    #print("Fit         :", i0.cget([0,0,0]).reshape(ishape)[0,0,0,3]) # = (0,2,0,:)_AO
    print("Exact (2)   :",  np.sum(pqrs_ex[0,2,0,v])+ np.sum(pqRrs_ex[0,2,0,v]))
    print("Diff  :", np.abs(i0.cget([0,0,0]).reshape(ishape)[0,0,0,2]) - np.sum(pqrs_ex[0,2,0,v]) -np.sum(pqRrs_ex[0,2,0,v])) #, "err"

    print("Integral(ao):  ( (000)0, (000)2 | (000) 0 , (000)(0+1+2+3+4+...+N_ao) )")
    #print("Fit         :", i0.cget([0,0,0]).reshape(ishape)[0,0,0,1]) # = (0,2,0,:)_AO
    print("Fit         :", i0.cget([0,0,0]).reshape(ishape)[0,0,0,3]) # = (0,2,0,:)_AO
    #print("Fit         :", i0.cget([0,0,0]).reshape(ishape)[0,0,0,3]) # = (0,2,0,:)_AO
    print("Exact (2)   :",  np.sum(pqrs_ex[0,2,0,v]))
    print("Diff        :", np.abs(i0.cget([0,0,0]).reshape(ishape)[0,0,0,3]) - np.sum(pqrs_ex[0,2,0,v]) ) #, "err"

    i0, ishape = ib.getorientation([0,0,0],[1,0,0])


    print("Integral(ao):  ( (000)0, (000)2 | (000) 0 , (100)(0+1+2+3+4+...+N_ao) )")
    print("Fit         :", i0.cget([0,0,0]).reshape(ishape)[0,0,0,3]) # = (0,2,0,:)_fit
    print("Exact (1)   :",  np.sum(pqRrs_ex[0,2,0,v])) # = (0,2|0,:) libint
    pqRrs_ex = PRI.compute_pqrs_(p, np.array([[0,0,0]]),np.array([[0,0,0]]),np.array([[-1,0,0]]))
    print("Exact (1)   :",  np.sum(pqRrs_ex[0,2,0,v])) # = (0,2|0,:) libint
    #print("Diff        :", np.abs( i0.cget([0,0,0]).reshape(ishape)[0,0,1,1]) - np.sum(pqTrs_ex[0,2,0,v]))
    


    
    #print("Fit       :", i0.cget([0,0,0]).reshape(ishape)[0,0,1,4]) # = (0,2,0,:)_AO

    #print("Exact (1) :", np.sum(pqrs_ex[0,2,0,v]))
    
    print(" ")
    

    
    
def test_ao_refcell_(attenuation = .2):
    p = pr.prism("inputs/neon_3d.d12")

    # Compute overlap matrix
    #s = PRI.compute_onebody(p, s)

    # temporary libint data path
    # datapath = os.environ["LIBINT_DATA_PATH"]

    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    

    # build ao coeff matrix so that (i,a|i,a)_mo = (p,q,p,q)_ao
    # (0,0|0,0)_mo =  (2,2|2,2)_ao
    # (1,1|2,2)_mo  = (6,6|T_1|2,3)_ao #translated
    

    coords = tp.lattice_coords([1,1,1])
    c = tp.tmat()
    c.load_nparray(np.ones((coords.shape[0], p.get_n_ao(),p.get_n_ao()), dtype = float), coords)
    c.blocks *= 0

    c0 = np.eye(p.get_n_ao(), dtype = float)

    c.blocks[ c.mapping[ c._c2i([0,0,0])]] = c0



    # generate fit-basis

    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvdz-ri.g94", alphacut = .3)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    # test integrals in refcell 

    ib = PRI.integral_builder_static(c,p,attenuation = attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=True, extent_thresh=1e-10, robust = False)
    
    i0, ishape = ib.getorientation([0,0,0],[0,0,0])
    pqrs_ex = PRI.compute_pqrs_(p, np.array([[0,0,0]]),np.array([[0,0,0]]),np.array([[0,0,0]]))[:p.get_nocc(),p.get_nocc():,:p.get_nocc(),p.get_nocc(): ]
    return pqrs_ex, i0.cget([0,0,0]).reshape(ishape)





#test_lih_mo(0.1)
#test_lih_ao()
#test_Jmn()

#test_ao_refcell()  
#test_exact_mo()


#for i in [0.1,0.2,0.3,0.4,0.5,0.75,1.0,2.0,3.0,4.0,5.0]:

"""
for i in [.9,.8,.7,.6,.5,.4,.3,.2]:
    print("================***")
    print(i)
    pqrs, i0 = test_ao_refcell_(i)
    print(i, np.max(np.abs(pqrs - i0)))
    #test_x_(i) #test lih ao integrals
    print("================***")
"""

test_x_(attenuation = .3)