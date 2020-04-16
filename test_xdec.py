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

import XDEC

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

def test_toeplitz_products():
    """
    Test that Toeplitz- and Circulant products are consistently defined
    """

    B = tp.get_random_tmat([1,1,1], [2,2])

    B2t = B.cdot(B)

    B2c = B.circulantdot(B, n_layers = np.array([2,2,2]))

    m = np.linalg.norm(B2t.cget(B2t.coords) - B2c.cget(B2t.coords))
    assert(m<1e-13)




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


def test_pri():
    """
    test fitting framework, use AO basis 
    """
    os.environ["LIBINT_DATA_PATH"] = os.getcwd() 
    #p = pr.prism("inputs/LiH_2D.d12")
    p = pr.prism("inputs/LiH_1D.d12")

    I = tp.get_identity_tmat(p.get_n_ao())

    auxbasis = PRI.basis_trimmer(p, "inputs/cc-pvtz-ri.g94", alphacut = .0)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()

    ib = PRI.integral_builder_static(I,I,p,attenuation = 0.06, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=True, robust = False, xi0=1e-10, xi1 = 1e-10,  N_c = 27)

    dL = np.array([0,0,0])
    I, Ishape = ib.getorientation(dL, dL)
    I0 = I.cget([0,0,0]).reshape((11,11,11,11))

   

    pqrs_ex = PRI.compute_pqrs(p, np.array([[0,0,0]]))
    Im = np.argmax(np.abs(pqrs_ex.ravel()))

    m = np.abs(pqrs_ex.ravel()[Im]-I0.ravel()[Im])/pqrs_ex.ravel()[Im] #relative error in 
    assert(m<1e-4)








