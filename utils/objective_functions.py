#!/usr/bin/env python

import numpy as np
import os 

from scipy.linalg import expm

import sys

#import lwrap.lwrap as lwrap
#sys.path.append(os.path.realpath('../lwrap/'))
sys.path.insert(0,'../lwrap')
import lwrap.lwrap as lwrap

import matplotlib.pyplot as plt


import argparse


import utils.toeplitz as tp
import utils.prism as pr

import utils.annealing_tools as at

#import lwrap_interface as lint

import subprocess as sp

import PRI

def psm_m(c, p,coords, m= 1):



    lint = lwrap.engine()

    lint.set_operator_emultipole()



    xyzname = "temp_geom.xyz"
    xyzfile = open(xyzname, "w")
    xyzfile.write(PRI.get_xyz(p, conversion_factor = 0.5291772109200000))
    xyzfile.close()

    xyzname_ = "temp_geom_0.xyz"
    xyzfile_ = open(xyzname_, "w")
    xyzfile_.write(PRI.get_xyz(p, coords, conversion_factor = 0.5291772109200000))
    xyzfile_.close()





    basis = p.get_libint_basis()
    bname = "temp_basis"
    bfile = open(bname + ".g94", "w")
    bfile.write(basis)
    bfile.close()

    lint.setup_pq(xyzname, bname, 
                    xyzname_, bname)
    vint = np.array(lint.get_pq_multipole(xyzname, bname, 
                                            xyzname_, bname))
    
    s0b = PRI.compute_onebody(p,c,coords)
    #s0 = tp.tmat()
    #s0.load_nparray(s0b.reshape(coords.shape[0], p.get_n_ao(), p.get_n_ao()), coords)
    #s0.load("/Users/audunhansen/papers/globopt-paper/results/ethylene/S_crystal.npy")

    #C = np.fromfile("/Users/audunhansen/papers/globopt-paper/results/aa/lcm_orbitals.u")[2:].reshape((508,508)).T[:, 22+66:] # (fortran is column major)
    #C = np.fromfile("/Users/audunhansen/papers/globopt-paper/results/aa/cmo_orbitals.u")[2:].reshape((508,508)).T[:, 22:66] # (fortran is column major)
    #print(vint[:,:,0].shape)
    ts = (p.get_n_ao(), coords.shape[0], p.get_n_ao() ) #shape
    sb = vint[:,:,0].reshape(ts).swapaxes(0,1)
    xb, yb, zb = vint[:,:,1].reshape(ts).swapaxes(0,1), vint[:,:,2].reshape(ts).swapaxes(0,1), vint[:,:,3].reshape(ts).swapaxes(0,1)
    xxb, yyb,zzb = vint[:,:,4].reshape(ts).swapaxes(0,1), vint[:,:,7].reshape(ts).swapaxes(0,1), vint[:,:,9].reshape(ts).swapaxes(0,1)
    
    s = tp.tmat()
    s.load_nparray(sb, coords)
    

    #print(np.max(s.blocks))
    #for i in np.arange(coords.shape[0]):
    #    print(coords[i], np.linalg.norm(s.cget(coords[i])-s0.cget(coords[i])))
    #print(s.cget([0,0,0])[0])
    #print(s0.cget([0,0,0])[0])
    x = tp.tmat()
    x.load_nparray(xb, coords)
    y = tp.tmat()
    y.load_nparray(yb, coords)
    z = tp.tmat()
    z.load_nparray(zb, coords)

    xx = tp.tmat()
    xx.load_nparray(xxb, coords)
    yy = tp.tmat()
    yy.load_nparray(yyb, coords)
    zz = tp.tmat()
    zz.load_nparray(zzb, coords)

    #print(s.blocks.shape)
    #print(c.blocks.shape)

    #x2 = tp.tmat()
    #x2.load("/Users/audunhansen/papers/pao-paper/results/X2ao.npy")

    #for i in np.arange(x2.coords.shape[0]):
    #    print(x2.coords[i], np.linalg.norm( s0.cget(x2.coords[i]) - s.cget(x2.coords[i])) )

    #print(s0.cget([0,0,0])[:4,:4], np.max(s0.blocks))
    #print(s.cget([0,0,0])[:4,:4], np.max(s.blocks))

    
    
    #for i in np.arange(s0.coords.shape[0]):
    #    print(s0.coords[i], i, np.linalg.norm(s0[i]- s.cget(-sc[i])))
    #print(s0[0,0])
    #print(" " )
    #print(s.cget(sc[0])[0])
    #print(sc[0])

    
    SC = s.circulantdot(c)

    
   
            
    lattice = p.lattice

    xSC = SC.cscale(0, lattice)
    ySC = SC.cscale(1, lattice)
    zSC = SC.cscale(2, lattice)
    
    xxSC = SC.cscale(0, lattice, exponent = 2)
    yySC = SC.cscale(1, lattice, exponent = 2)
    zzSC = SC.cscale(2, lattice, exponent = 2)
    
    XC = x.circulantdot(c)
    YC = y.circulantdot(c)
    ZC = z.circulantdot(c)
    
    xXC = XC.cscale(0, lattice)
    yYC = YC.cscale(1, lattice)
    zZC = ZC.cscale(2, lattice)

    
    Xmo = c.tT().circulantdot(XC) - c.tT().circulantdot(xSC)
    Ymo = c.tT().circulantdot(YC) - c.tT().circulantdot(ySC)
    Zmo = c.tT().circulantdot(ZC) - c.tT().circulantdot(zSC)
    #print(np.diag(Xmo.cget([0,0,0])))
    
    #wcenters_x = np.diag(Xmo.cget([0,0,0]))
    #wcenters_y = np.diag(Ymo.cget([0,0,0]))
    #wcenters_z = np.diag(Zmo.cget([0,0,0]))
    
    #wcenters = np.array([wcenters_x, wcenters_y, wcenters_z])
    
    
    
    
    
    #print(wcenters.T)

    #X2mo = C.tT()*X2*C -  C.tT()*xXC*2 +  C.tT()*xxSC
    #Y2mo = C.tT()*Y2*C -  C.tT()*yYC*2 +  C.tT()*yySC
    #Z2mo = C.tT()*Z2*C -  C.tT()*zZC*2 +  C.tT()*zzSC


    
    X2mo = c.tT().circulantdot(xx.circulantdot(c)) -  c.tT().circulantdot(xXC*2) +  c.tT().circulantdot(xxSC)
    Y2mo = c.tT().circulantdot(yy.circulantdot(c)) -  c.tT().circulantdot(yYC*2) +  c.tT().circulantdot(yySC)
    Z2mo = c.tT().circulantdot(zz.circulantdot(c)) -  c.tT().circulantdot(zZC*2) +  c.tT().circulantdot(zzSC)
    R2mo = X2mo + Y2mo + Z2mo

    #X2mo = c.tT().cdot(xx*c, coords = np.array([[0,0,0]])) -  c.tT().cdot(xXC*2, coords = np.array([[0,0,0]])) +  c.tT().cdot(xxSC, coords = np.array([[0,0,0]]))
    #Y2mo = c.tT().cdot(yy*c, coords = np.array([[0,0,0]])) -  c.tT().cdot(yYC*2, coords = np.array([[0,0,0]])) +  c.tT().cdot(yySC, coords = np.array([[0,0,0]]))
    #Z2mo = c.tT().cdot(zz*c, coords = np.array([[0,0,0]])) -  c.tT().cdot(zZC*2, coords = np.array([[0,0,0]])) +  c.tT().cdot(zzSC, coords = np.array([[0,0,0]]))
    
    #rr = xx+yy+zz
    #print(np.diag(rr.cget([0,0,0])))
    #print(np.diag(Y2mo.cget([0,0,0])))
    #print(np.diag(Z2mo.cget([0,0,0])))
    
    
    #PSM_objective_function = X2mo.cget([0,0,0])  + Y2mo.cget([0,0,0]) + Z2mo.cget([0,0,0]) - Xmo.cget([0,0,0])**2 - Ymo.cget([0,0,0])**2 - Zmo.cget([0,0,0])**2
    #print(np.diag(PSM_objective_function))
    #                                         - 
    f_psm1 = lambda tens : np.sum(np.diag(tens[0].cget([0,0,0]) - tens[1].cget([0,0,0])**2 - tens[2].cget([0,0,0])**2 - tens[3].cget([0,0,0])**2)**m) # PSM-m objective function
    tensors = [R2mo, Xmo, Ymo, Zmo]
    #X = tensors
    #print(np.diag(X[0].cget([0,0,0]) - X[1].cget([0,0,0])**2 - X[2].cget([0,0,0])**2 - X[3].cget([0,0,0])**2))
    #print(f_psm1(tensors))
    return tensors, f_psm1 #X2mo + Y2mo + Z2mo, Xmo, Ymo, Zmo, wcenters


def centers_spreads(c, p,coords, m= 1):



    lint = lwrap.engine()

    lint.set_operator_emultipole()



    xyzname = "temp_geom.xyz"
    xyzfile = open(xyzname, "w")
    xyzfile.write(PRI.get_xyz(p, conversion_factor = 0.5291772109200000))
    xyzfile.close()

    xyzname_ = "temp_geom_0.xyz"
    xyzfile_ = open(xyzname_, "w")
    xyzfile_.write(PRI.get_xyz(p, coords, conversion_factor = 0.5291772109200000))
    xyzfile_.close()





    basis = p.get_libint_basis()
    bname = "temp_basis"
    bfile = open(bname + ".g94", "w")
    bfile.write(basis)
    bfile.close()

    lint.setup_pq(xyzname, bname, 
                    xyzname_, bname)
    vint = np.array(lint.get_pq_multipole(xyzname, bname, 
                                            xyzname_, bname))
    
    s0b = PRI.compute_onebody(p,c,coords)
    #s0 = tp.tmat()
    #s0.load_nparray(s0b.reshape(coords.shape[0], p.get_n_ao(), p.get_n_ao()), coords)
    #s0.load("/Users/audunhansen/papers/globopt-paper/results/ethylene/S_crystal.npy")

    #C = np.fromfile("/Users/audunhansen/papers/globopt-paper/results/aa/lcm_orbitals.u")[2:].reshape((508,508)).T[:, 22+66:] # (fortran is column major)
    #C = np.fromfile("/Users/audunhansen/papers/globopt-paper/results/aa/cmo_orbitals.u")[2:].reshape((508,508)).T[:, 22:66] # (fortran is column major)
    #print(vint[:,:,0].shape)
    ts = (p.get_n_ao(), coords.shape[0], p.get_n_ao() ) #shape
    sb = vint[:,:,0].reshape(ts).swapaxes(0,1)
    xb, yb, zb = vint[:,:,1].reshape(ts).swapaxes(0,1), vint[:,:,2].reshape(ts).swapaxes(0,1), vint[:,:,3].reshape(ts).swapaxes(0,1)
    xxb, yyb,zzb = vint[:,:,4].reshape(ts).swapaxes(0,1), vint[:,:,7].reshape(ts).swapaxes(0,1), vint[:,:,9].reshape(ts).swapaxes(0,1)
    
    s = tp.tmat()
    s.load_nparray(sb, coords)
    

    #print(np.max(s.blocks))
    #for i in np.arange(coords.shape[0]):
    #    print(coords[i], np.linalg.norm(s.cget(coords[i])-s0.cget(coords[i])))
    #print(s.cget([0,0,0])[0])
    #print(s0.cget([0,0,0])[0])
    x = tp.tmat()
    x.load_nparray(xb, coords)
    y = tp.tmat()
    y.load_nparray(yb, coords)
    z = tp.tmat()
    z.load_nparray(zb, coords)

    xx = tp.tmat()
    xx.load_nparray(xxb, coords)
    yy = tp.tmat()
    yy.load_nparray(yyb, coords)
    zz = tp.tmat()
    zz.load_nparray(zzb, coords)

    #print(s.blocks.shape)
    #print(c.blocks.shape)

    #x2 = tp.tmat()
    #x2.load("/Users/audunhansen/papers/pao-paper/results/X2ao.npy")

    #for i in np.arange(x2.coords.shape[0]):
    #    print(x2.coords[i], np.linalg.norm( s0.cget(x2.coords[i]) - s.cget(x2.coords[i])) )

    #print(s0.cget([0,0,0])[:4,:4], np.max(s0.blocks))
    #print(s.cget([0,0,0])[:4,:4], np.max(s.blocks))

    
    
    #for i in np.arange(s0.coords.shape[0]):
    #    print(s0.coords[i], i, np.linalg.norm(s0[i]- s.cget(-sc[i])))
    #print(s0[0,0])
    #print(" " )
    #print(s.cget(sc[0])[0])
    #print(sc[0])

    
    SC = s.circulantdot(c)

    
   
            
    lattice = p.lattice

    xSC = SC.cscale(0, lattice)
    ySC = SC.cscale(1, lattice)
    zSC = SC.cscale(2, lattice)
    
    xxSC = SC.cscale(0, lattice, exponent = 2)
    yySC = SC.cscale(1, lattice, exponent = 2)
    zzSC = SC.cscale(2, lattice, exponent = 2)
    
    XC = x.circulantdot(c)
    YC = y.circulantdot(c)
    ZC = z.circulantdot(c)
    
    xXC = XC.cscale(0, lattice)
    yYC = YC.cscale(1, lattice)
    zZC = ZC.cscale(2, lattice)

    
    Xmo = c.tT().circulantdot(XC) - c.tT().circulantdot(xSC)
    Ymo = c.tT().circulantdot(YC) - c.tT().circulantdot(ySC)
    Zmo = c.tT().circulantdot(ZC) - c.tT().circulantdot(zSC)
    #print(np.diag(Xmo.cget([0,0,0])))
    
    wcenters_x = np.diag(Xmo.cget([0,0,0]))
    wcenters_y = np.diag(Ymo.cget([0,0,0]))
    wcenters_z = np.diag(Zmo.cget([0,0,0]))
    
    wcenters = np.array([wcenters_x, wcenters_y, wcenters_z]).T
    
    
    
    
    
    #print(wcenters.T)

    #X2mo = C.tT()*X2*C -  C.tT()*xXC*2 +  C.tT()*xxSC
    #Y2mo = C.tT()*Y2*C -  C.tT()*yYC*2 +  C.tT()*yySC
    #Z2mo = C.tT()*Z2*C -  C.tT()*zZC*2 +  C.tT()*zzSC


    
    X2mo = c.tT().circulantdot(xx.circulantdot(c)) -  c.tT().circulantdot(xXC*2) +  c.tT().circulantdot(xxSC)
    Y2mo = c.tT().circulantdot(yy.circulantdot(c)) -  c.tT().circulantdot(yYC*2) +  c.tT().circulantdot(yySC)
    Z2mo = c.tT().circulantdot(zz.circulantdot(c)) -  c.tT().circulantdot(zZC*2) +  c.tT().circulantdot(zzSC)
    R2mo = X2mo + Y2mo + Z2mo

    #X2mo = c.tT().cdot(xx*c, coords = np.array([[0,0,0]])) -  c.tT().cdot(xXC*2, coords = np.array([[0,0,0]])) +  c.tT().cdot(xxSC, coords = np.array([[0,0,0]]))
    #Y2mo = c.tT().cdot(yy*c, coords = np.array([[0,0,0]])) -  c.tT().cdot(yYC*2, coords = np.array([[0,0,0]])) +  c.tT().cdot(yySC, coords = np.array([[0,0,0]]))
    #Z2mo = c.tT().cdot(zz*c, coords = np.array([[0,0,0]])) -  c.tT().cdot(zZC*2, coords = np.array([[0,0,0]])) +  c.tT().cdot(zzSC, coords = np.array([[0,0,0]]))
    
    #rr = xx+yy+zz
    #print(np.diag(rr.cget([0,0,0])))
    #print(np.diag(Y2mo.cget([0,0,0])))
    #print(np.diag(Z2mo.cget([0,0,0])))
    
    
    #PSM_objective_function = X2mo.cget([0,0,0])  + Y2mo.cget([0,0,0]) + Z2mo.cget([0,0,0]) - Xmo.cget([0,0,0])**2 - Ymo.cget([0,0,0])**2 - Zmo.cget([0,0,0])**2
    #print(np.diag(PSM_objective_function))
    #                                         - 
    f_psm1 = lambda tens : np.sum(np.diag(tens[0].cget([0,0,0]) - tens[1].cget([0,0,0])**2 - tens[2].cget([0,0,0])**2 - tens[3].cget([0,0,0])**2)**m) # PSM-m objective function

    spreads_ = lambda tens : np.diag(tens[0].cget([0,0,0]) - tens[1].cget([0,0,0])**2 - tens[2].cget([0,0,0])**2 - tens[3].cget([0,0,0])**2)
    spreads = spreads_(tensors)


    tensors = [R2mo, Xmo, Ymo, Zmo]
    #X = tensors
    #print(np.diag(X[0].cget([0,0,0]) - X[1].cget([0,0,0])**2 - X[2].cget([0,0,0])**2 - X[3].cget([0,0,0])**2))
    #print(f_psm1(tensors))
    return wcenters, spreads #X2mo + Y2mo + Z2mo, Xmo, Ymo, Zmo, wcenters






    



