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

def test_prism():
    """
    Test that PRISM is available and funcitoning
    """
    m = True
    try:
        p = pr.prism("inputs/neon_3d.d12")
    except:
        m = False
    assert(m), "Prism is not working."



def test_toeplitz():
    B = tp.tmat()

    b = np.array([[[6., 8.],
        [9., 9.]],
       [[7., 0.],
        [5., 2.]],
       [[4., 4.],
        [9., 6.]]])
    
    B.load_nparray(b, tp.lattice_coords([1,0,0]))

    B2b = np.array([[[108., 120.],
        [135., 153.]],
       [[124.,  72.],
        [156.,  76.]],
       [[205., 140.],
        [270., 220.]],
       [[ 76.,  36.],
        [131.,  44.]],
       [[ 52.,  40.],
        [ 90.,  72.]]])

    # test toeplitz matrix product
    B2 = B.cdot(B)
    assert(np.linalg.norm(B2.cget(B2.coords)-B2b)<=1e-14)

    # test circulant matrix product

    B2c = np.array([[[205., 140.],
        [270., 220.]],
       [[184., 156.],
        [266., 197.]],
       [[176., 112.],
        [246., 148.]]])

    B2 = B.circulantdot(B)
    assert(np.linalg.norm(B2.cget(B2.coords)-B2c)<=1e-14)

# test libint / PRI

def test_libint():
    """
    compute neon overlap matrix
    """
    s_ = np.array([[[ 1.00000000e+00,  2.49018009e-01,  1.71913109e-01,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 2.49018009e-01,  1.00000000e+00,  7.59030906e-01,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 1.71913109e-01,  7.59030906e-01,  1.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          4.92981905e-01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  4.92981905e-01,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  4.92981905e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          4.92981905e-01,  0.00000000e+00,  0.00000000e+00,
          1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  4.92981905e-01,  0.00000000e+00,
          0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  4.92981905e-01,
          0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
       [[ 2.87582673e-45,  4.79752351e-15,  1.26552027e-05,
         -3.16746686e-14,  0.00000000e+00,  0.00000000e+00,
         -7.66553692e-05,  0.00000000e+00,  0.00000000e+00],
        [ 4.79752351e-15,  9.28530035e-09,  3.32313526e-04,
         -3.62844038e-08,  0.00000000e+00,  0.00000000e+00,
         -1.64585151e-03,  0.00000000e+00,  0.00000000e+00],
        [ 1.26552027e-05,  3.32313526e-04,  7.26950163e-03,
         -5.71586618e-04,  0.00000000e+00,  0.00000000e+00,
         -2.28129599e-02,  0.00000000e+00,  0.00000000e+00],
        [ 3.16746686e-14,  3.62844038e-08,  5.71586618e-04,
         -1.38007071e-07,  0.00000000e+00,  0.00000000e+00,
         -2.67719848e-03,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  3.78402630e-09,  0.00000000e+00,
          0.00000000e+00,  1.82139883e-04,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  3.78402630e-09,
          0.00000000e+00,  0.00000000e+00,  1.82139883e-04],
        [ 7.66553692e-05,  1.64585151e-03,  2.28129599e-02,
         -2.67719848e-03,  0.00000000e+00,  0.00000000e+00,
         -6.43215324e-02,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  1.82139883e-04,  0.00000000e+00,
          0.00000000e+00,  7.26950163e-03,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  1.82139883e-04,
          0.00000000e+00,  0.00000000e+00,  7.26950163e-03]]])
    os.environ["LIBINT_DATA_PATH"] = os.getcwd() 

    p = pr.prism("inputs/neon_3d.d12")

    # Compute overlap matrix
    s = PRI.compute_overlap_matrix(p, tp.lattice_coords([1,0,0])[1:])
    
    assert(np.linalg.norm(s.cget(s.coords)-s_)<1e-6)












"""
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
"""
    
