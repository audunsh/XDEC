#!/usr/bin/env python

import numpy as np
import os 

from scipy.linalg import expm

import sys

#import lwrap.lwrap as lwrap
#sys.path.append(os.path.realpath('../lwrap/'))
#sys.path.insert(0,'../lwrap')
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
    


    tensors = [R2mo, Xmo, Ymo, Zmo]
    spreads = spreads_(tensors)
    #X = tensors
    #print(np.diag(X[0].cget([0,0,0]) - X[1].cget([0,0,0])**2 - X[2].cget([0,0,0])**2 - X[3].cget([0,0,0])**2))
    #print(f_psm1(tensors))
    return wcenters, spreads #X2mo + Y2mo + Z2mo, Xmo, Ymo, Zmo, wcenters




def overlap_matrix(p, coords = None, thresh = 1e-10):
    # Compute overlap matrix to some given thresh

    if coords is None:
        #coords = tp.lattice_coords([10,10,10]) # Compute first, screen afterwards
        coords = tp.lattice_coords(p.ndim_layer(10))

    xyzname = "temp_geom.xyz"
    xyzfile = open(xyzname, "w")
    xyzfile.write(PRI.get_xyz(p, conversion_factor = 0.5291772109200000))
    xyzfile.close()

    xyzname_ = "temp_geom_0.xyz"
    xyzfile_ = open(xyzname_, "w")
    xyzfile_.write(PRI.get_xyz(p, coords, conversion_factor = 0.5291772109200000))
    xyzfile_.close()

    ts = (p.get_n_ao(), coords.shape[0], p.get_n_ao() ) #shape

    lint = lwrap.engine()

    lint.set_operator_overlap()


    basis = p.get_libint_basis()
    bname = "temp_basis"
    bfile = open(bname + ".g94", "w")
    bfile.write(basis)
    bfile.close()

    lint.setup_pq(xyzname, bname, 
                    xyzname_, bname)
    sb = np.array(lint.get_pq(xyzname, bname, xyzname_, bname)).reshape(ts).swapaxes(0,1)
 

    # Screen out elements below thresh, warn if edge elements
    screen = np.max(np.abs(sb), axis = (1,2))>thresh

    ret = tp.tmat()
    ret.load_nparray(sb[screen], coords[screen])

    return ret




    
def conventional_paos(c,p, s = None, orthonormalize = False, thresh = 1e-2, fock_metric = None):
    """
    Concstruct the PAOs according to 
    """
    ncore = p.n_core 
    p.n_core = 0 #temporarily include core orbital
    c_occ, c_virt = PRI.occ_virt_split( c, p)

    c_virt = tp.tmat()
    c_virt.load_nparray(c.cget(c.coords)[:,:,p.get_nocc()+p.n_core:], c.coords, screening = False)

    c_occ = tp.tmat()
    c_occ.load_nparray(c.cget(c.coords)[:,:,p.n_core:p.get_nocc()+p.n_core], c.coords, screening = False)


    
    p.n_core = ncore

    

    coords = c.coords

    if s is not None:
        coords = s.coords

    #coords = tp.lattice_coords([8,8,8])

    ts = (p.get_n_ao(), coords.shape[0], p.get_n_ao() ) #shape

    lint = lwrap.engine()

    lint.set_operator_emultipole()



    xyzname = "temp_geom.xyz"
    xyzfile = open(xyzname, "w")
    xyzfile.write(PRI.get_xyz(p))
    xyzfile.close()

    xyzname_ = "temp_geom_0.xyz"
    xyzfile_ = open(xyzname_, "w")
    xyzfile_.write(PRI.get_xyz(p, coords))
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
    
    #sb = PRI.compute_onebody(p,c,coords)



    sb = vint[:,:,0].reshape(ts).swapaxes(0,1)
    xb, yb, zb = vint[:,:,1].reshape(ts).swapaxes(0,1), vint[:,:,2].reshape(ts).swapaxes(0,1), vint[:,:,3].reshape(ts).swapaxes(0,1)
    xxb, yyb,zzb = vint[:,:,4].reshape(ts).swapaxes(0,1), vint[:,:,7].reshape(ts).swapaxes(0,1), vint[:,:,9].reshape(ts).swapaxes(0,1)
    
    s = tp.tmat()
    s.load_nparray(sb, coords)
    

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


    


    



    d = c_occ.circulantdot(c_occ.tT())
    c_pao = d.circulantdot(s)
    if fock_metric is not None:
        #s = fock_metric*1
        c_pao = d.circulantdot(fock_metric)
        
    c_pao.blocks*=-1 #.5


    #d = c_occ*c_occ.tT()*2
    #c_pao = d*s
    #c_pao = c_pao * (-.5) #.5



    
    
    c_pao.blocks[ c_pao.mapping[ c_pao._c2i([0,0,0]) ] ] += np.eye(c_pao.blockshape[0], dtype = float)

    
    
    #c_pao = unfolded_pao(c,p)
    
    
    smo_ov = c_occ.tT().circulantdot(s.circulantdot(c_pao))
    print("Circulant PAO test:", np.max(np.abs(smo_ov.blocks)))


    norms = np.diag(c_pao.tT().circulantdot(s.circulantdot(c_pao)).cget([0,0,0]))


    c_pao_screened_blocks = c_pao.cget(c_pao.coords)[:, :, norms>thresh]
    c_pao_screened_coords = c_pao.coords

    c_pao = tp.tmat()
    c_pao.load_nparray(c_pao_screened_blocks, c_pao_screened_coords)

    norms = np.diag(c_pao.tT().circulantdot(s.circulantdot(c_pao)).cget([0,0,0]))

    #normalize
    
    for i in np.arange(len(norms)):
        c_pao.blocks[:,:,i] = c_pao.blocks[:,:,i]/np.sqrt(norms[i])
    


    # Compute centers

    SC = s.circulantdot(c_pao)

    # Check span
    w = c_virt.tT().circulantdot(SC)
    print("Span conserved?")
    print(np.sum(w.blocks**2, axis = (0,1)))
   
            
    lattice = p.lattice

    xSC = SC.cscale(0, lattice)
    ySC = SC.cscale(1, lattice)
    zSC = SC.cscale(2, lattice)

    XC = x.circulantdot(c_pao)
    YC = y.circulantdot(c_pao)
    ZC = z.circulantdot(c_pao)

    Xmo = c_pao.tT().circulantdot(XC) - c_pao.tT().circulantdot(xSC)
    Ymo = c_pao.tT().circulantdot(YC) - c_pao.tT().circulantdot(ySC)
    Zmo = c_pao.tT().circulantdot(ZC) - c_pao.tT().circulantdot(zSC)
    
    wcenters_x = np.diag(Xmo.cget([0,0,0]))
    wcenters_y = np.diag(Ymo.cget([0,0,0]))
    wcenters_z = np.diag(Zmo.cget([0,0,0]))
    
    wcenters = np.array([wcenters_x, wcenters_y, wcenters_z]).T


    return s, c_pao, wcenters


def orthogonalize_tmat_svd(c,p, thresh = 0.01, coords = None):
    """
    pseudo-inverse based orthogonalization
    """

    if coords is not None:
        c_ = tp.tmat()
        c_.load_nparray(c.cget(coords), coords)
        c = c_*1

    
    s = overlap_matrix(p)

    smo = c.tT().circulantdot(s.cdot(c))

    #delta_ = smo.inv().kspace_cholesky()

    u,s,vh = smo.kspace_svd()

    s_diag = np.diag(s.cget([0,0,0]))
    
    # screen elements

    sc = s_diag>thresh
    #print(sc)

    #print(s.coords, s.cget(s.coords))

    S = tp.tmat()
    S.load_nparray(s.cget(s.coords)[:, sc][:, :, sc], s.coords, safemode = False)

    VH = tp.tmat()
    VH.load_nparray(vh.cget(vh.coords)[:, sc, :], vh.coords)

    U = tp.tmat()
    U.load_nparray(u.cget(u.coords)[:, :, sc], u.coords)



    cd = VH.tT().circulantdot(S.circulantdot( U.tT() ) )

    #cd = c.circulantdot(delta_)

    return cd



def orthogonalize_tmat_cholesky(c,p, thresh = 0.01, coords = None):

    if coords is not None:
        c_ = tp.tmat()
        c_.load_nparray(c.cget(coords), coords)

        c = c_*1

    
    s = overlap_matrix(p)

    smo = c.tT().circulantdot(s.circulantdot(c))

    delta_ = smo.inv().kspace_cholesky() # S^-1 = delta_.T * delta_ 

    cd = c.circulantdot(delta_)

    return cd

    #smo = cd.tT().circulantdot(s.circulantdot(cd))


    """
    # Compute "metric-matrix" / overlap
    delta = c.tT().circulantdot(s.circulantdot(c))

    u,s,vh = delta.kspace_svd()
    print(np.diag(s.cget([0,0,0])))

    #s0_diag = np.diag(s.blocks[ s.mapping[ s._c2i([0,0,0])] ])
    #s.blocks[ s.mapping[ s._c2i([0,0,0])], np.arange(s.blocks.shape[1]), np.arange(s.blocks.shape[1]) ] = s0_diag**-.5
    s.blocks[:-1, np.arange(s.blocks.shape[1]), np.arange(s.blocks.shape[1])] = s.blocks[:-1, np.arange(s.blocks.shape[1]), np.arange(s.blocks.shape[1])]**-.5

    print(np.diag(s.cget([0,0,0])))
    #s_blocks = s.cget(s.coords)[:,np.abs(np.diag(s.cget([0,0,0])))>thresh, np.abs(np.diag(s.cget([0,0,0])))>thresh ]

    #s_ = tp.tmat()
    #s_.load_nparray(s_blocks, s.coords)

    u_ = vh.tT().circulantdot( s.circulantdot( u.tT() ) )

    print(u_.blocks.shape)

    return c.circulantdot(u_)
    """


    """
    # Determine s^-1/2
    d_ = delta.kspace_svd_lowdin()

    # Eigenvectors and eigenvalues
    mu, u = d_.kspace_eig()

    mu_sum = np.sqrt(np.diag(np.sum(mu.blocks**2, axis = 0)))


    mu_ = tp.tmat()
    mu_.load_nparray(mu.blocks[:-1, mu_sum>thresh,:], mu.coords)

    u_ = tp.tmat()
    u_.load_nparray(u.blocks[:-1, :, mu_sum>thresh], mu.coords)


    c_orthonormal = c.circulantdot(u_.circulantdot(mu_))

    c_ortho = tp.tmat()
    c_ortho.load_nparray(c_orthonormal.blocks[:-1, :, mu_sum>thresh], c_orthonormal.coords)

    return c_ortho
    """

def orthogonalize_tmat_unfold(c,p, coords = None, mx  = None, thresh = 1e-5):

    s = overlap_matrix(p)

    if coords is None:
        mx = np.max( np.array([np.max(np.abs(c.coords), axis = 0), 
                               np.max(np.abs(s.coords), axis = 0)]), axis = 0)

        coords = tp.lattice_coords(mx)
    else:
        if mx is not None:
            coords = tp.lattice_coords(mx)
    #print(coords, coords[int(len(coords)/2)])
    C = c.tofull(c,coords,coords)
    S = s.tofull(s,coords,coords)

    print(C.shape)

    SMO = np.dot(C.T, np.dot(S,C))

    u_,s_,vh_ = np.linalg.svd(SMO)

    U_ = np.dot(vh_.T.conj(), np.dot(np.diag(s_**-.5), u_.T.conj()))

    C_ = np.dot(C, U_)

    rnx = c.blocks.shape[1]
    rny = c.blocks.shape[2]
    rc =  int(len(coords)/2)
    retblocks = C_[rnx*rc:rnx*(rc+1),:].reshape(rnx, coords.shape[0], rny).swapaxes(0,1)
    
    ret = tp.tmat()
    ret.load_nparray(retblocks, coords)

    return ret


    
def orthogonal_paos_gs(c,p, N_paos, thresh = 1e-1):
    # Construct max N_paos number of PAOs from the occupied orbitals
    # S = overlap matrix
    # C_occ = coefficients of occupied orbitals ( C_{mu, i} )
    # returns a matrix with occupied and virtual orbitals 
    #co, cv = PRI.occ_virt_split(c,p)
    
    #native functions
    def eyemin(D,S):
        ret = D.circulantdot(S)*-1
        ret.blocks[ ret.mapping[ ret._c2i([0,0,0])]] += np.eye(S.blocks.shape[1])
        return ret

    def extend(C, C_):
        newshape = np.array(C.cget(C.coords).shape)
        newshape[2] += C_.blocks.shape[2]
        ret_blocks = np.zeros(newshape, dtype = float)
        
        ret_blocks[:, :, :C.blocks.shape[2]]= C.cget(C.coords)
        ret_blocks[:, :,  C.blocks.shape[2]:]= C_.cget(C.coords)
        
        ret = tp.tmat()
        ret.load_nparray(ret_blocks, C.coords)
        return ret



    S = overlap_matrix(p)

    # temporarily include core orbitals
    n_core = p.n_core*1
    p.n_core = 0
    co, cv = PRI.occ_virt_split(c,p)
    p.n_core = n_core
    
    C_full = co*1 #initial set to return
    
    for i in np.arange(N_paos):
        smo = C_full.tT().circulantdot(S.circulantdot(C_full))
        #print("Max dev.:", np.max(   np.abs((smo - tp.get_identity_tmat(smo.blocks.shape[1])).blocks)    ) )
        D = C_full.cdot(C_full.tT()) #a "mock" density matrix

        #C_pao = np.eye(S.shape[0]) - np.dot(D, S) # construct PAOs 
        
        C_pao = eyemin(D,S)
        
        #print("PAO-occupied overlap:", np.abs(C_pao.tT().circulantdot(S.circulantdot(co)).blocks).max())
        #print("PAO-fullspace overlap:", np.abs(C_pao.tT().circulantdot(S.circulantdot(C_full)).blocks).max())
        #print(np.sqrt(np.sum(C_pao.tT().circulantdot(S.circulantdot(C_full)).blocks**2, axis = (0,2))))
        
        #Normalize the pao-set
        norm = np.diag(C_pao.tT().circulantdot(S.circulantdot(C_pao)).cget([0,0,0]))
        for j in np.arange(norm.shape[0]):
            C_pao.blocks[:, :, j] = C_pao.blocks[:, :, j]/np.sqrt(norm[j])

        # re-compute the norms
        norm = np.diag(C_pao.tT().circulantdot(S.circulantdot(C_pao)).cget([0,0,0]))
        #print(norm)

        
        #if norm.max()<thresh:
        #    break

        
        # Compute overlap to existing space

        residuals = np.max(np.abs(C_pao.tT().circulantdot(S.circulantdot(C_full)).blocks), axis = (0,2))
        
        smo = C_pao.tT().circulantdot(S.circulantdot(C_pao))

        res2 = np.max(   np.abs((smo - tp.get_identity_tmat(smo.blocks.shape[1])).blocks), axis = (0,1)    )
        #np.max(np.abs(smo).blocks), axis = (0,2))
        #print("res2:", res2)

        residuals = np.max(np.array([residuals, res2]), axis = 0)





        #print(residuals, len(residuals), C_full.blocks.shape)

        residuals = np.max(np.abs(C_full.tT().circulantdot(S.circulantdot(C_pao)).blocks), axis = (0,1))
        #print(residuals)
        

        #print(norm.argmax(), norm.max())
        #print(norm[residuals.argmin()])

        #print("PAO-occupied overlap:",  np.abs(C_pao.tT().circulantdot(S.circulantdot(co)).blocks).max())
        #print("PAO-fullspace overlap:", np.abs(C_pao.tT().circulantdot(S.circulantdot(C_full)).blocks).max())
        #print(np.sqrt(np.sum(C_pao.tT().circulantdot(S.circulantdot(C_full)).blocks**2, axis = (0,2))))

        

        

        #pick the pao with least overlap to existing space
        pao_i = residuals.argmin()
        if residuals[pao_i]>thresh:
            print(" Final ", residuals[pao_i])
            break

        print("Span of excisting space?")
        smo = C_full.tT().circulantdot(S.circulantdot(C_pao))
        #print(np.sum(smo.blocks**2, axis = (0,1)))
        print(np.sum(smo.blocks[:,:,pao_i]**2, axis = (0)))


        norm_i = norm[pao_i]
        overlap_occ = np.max(np.abs(C_pao.tT().circulantdot(S.circulantdot(co)).blocks[:, pao_i, :]))
        overlap_full = np.max(np.abs(C_pao.tT().circulantdot(S.circulantdot(C_full)).blocks[:, pao_i, :]))


        print("picked PAO %i, norm %.2e , overlap %.2e" % (pao_i, norm_i, overlap_full))
        #print(norm_i**.5)
        new_blocks = np.zeros(C_full.blocks.shape + np.array([0,0,1]), dtype = float)
        new_blocks[:-1, :, :-1] = C_full.cget(C_full.coords)
        new_blocks[:-1, :, -1]  = C_pao.cget(C_full.coords)[:, :, pao_i]

        coords = C_full.coords

        C_full = tp.tmat()
        C_full.load_nparray(new_blocks, coords)

        #print(C_full.blocks.shape)
        smo = C_full.tT().circulantdot(S.circulantdot(C_full))
        print("Max dev.:", np.max(   np.abs((smo - tp.get_identity_tmat(smo.blocks.shape[1])).blocks)    ) )

        #print(np.abs((smo - tp.get_identity_tmat(smo.blocks.shape[1])).blocks[:, -1, -1]) )

        print("Conserved span of virtual space?")

        smo = cv.tT().circulantdot(S.circulantdot(C_full))
        #print(np.sum(smo.blocks**2, axis = (0,1)))
        print(np.sum(smo.blocks**2, axis = (0,2)))

        


        

        

        #C_pao_max_blocks = C_pao.cget(C_pao.coords)[:,:, pao_i]# /np.sqrt(norm[pao_i]) # extract PAO with max norm + normalization
        #print(C_pao_max_blocks.shape)
        
        #C_new = tp.tmat()
        #C_new.load_nparray(C_pao_max_blocks.reshape(C_pao.coords.shape[0], C_pao_max_blocks.shape[1], 1), C_pao.coords)
        #C_full = extend(C_full, C_new)
        
        
        
        #print(i, norm, np.sum(np.diag(np.dot(C_pao.T, np.dot(S, C_pao)))))
        
        # print(np.diag(np.dot(C_pao.T, np.dot(S, C_pao))).max()) # uncomment to print norm of appended PAO
        
        #C_full_new = np.zeros((C_full.shape[0],C_full.shape[1]+1), dtype = float) #extend array with one more column
        ##C_full_new[:, -1] = C_pao_max # append new PAO at final column
        #C_full_new[:,:-1] = C_full # Insert the already accumulated space
        #C_full = C_full_new*1 # Update matrix
    print(" Extracted %i paos" % i)
    
    return C_full # when done, return complete set with both occupied and virtual space


def orthogonal_paos_rep(c,p, N_paos, thresh = 1e-1, orthogonalize = True):
    # Construct max N_paos number of PAOs from the occupied orbitals
    # S = overlap matrix
    # C_occ = coefficients of occupied orbitals ( C_{mu, i} )
    # returns a matrix with occupied and virtual orbitals 
    #co, cv = PRI.occ_virt_split(c,p)
    
    #native functions
    def eyemin(D,S):
        ret = D.circulantdot(S)*-1
        ret.blocks[ ret.mapping[ ret._c2i([0,0,0])]] += np.eye(S.blocks.shape[1])
        return ret

    def extend(C, C_, i):
        # add orbital i from C_ into C
        new_blocks = np.zeros(C.blocks.shape + np.array([0,0,1]), dtype = float)
        new_blocks[:-1, :, :-1] = C.cget(C.coords)
        new_blocks[:-1, :, -1]  = C_.cget(C.coords)[:, :, i]

        

        coords = C.coords

        ret = tp.tmat()
        ret.load_nparray(new_blocks[:-1], coords)
        
        return ret



    S = overlap_matrix(p)

    # temporarily include core orbitals
    n_core = p.n_core*1
    p.n_core = 0
    co, cv = PRI.occ_virt_split(c,p)
    p.n_core = n_core
    
    C_full = co*1 #initial set to return
    
    #D = C_full.cdot(C_full.tT()) #a "mock" density matrix
    
    print(co.blocks.shape)

    #C_pao = eyemin(D,S) #initial PAOs
    
    I = tp.get_identity_tmat(co.blocks.shape[1])
    
    for i in np.arange(N_paos):
        # 0. a) construct PAOs
        
        D = C_full.circulantdot(C_full.tT()) #a "mock" density matrix        
        C_pao = eyemin(D,S)
        #C_pao = of.orthogonalize_tmat_svd(C_pao, p)
        # 0. b ) Normalize the pao-set
        
        norm = np.diag(C_pao.tT().circulantdot(S.circulantdot(C_pao)).cget([0,0,0]))
        
        for j in np.arange(norm.shape[0]):
            C_pao.blocks[:, :, j] = C_pao.blocks[:, :, j]/np.sqrt(norm[j])
            
            
        if i==0:
            # Special pick, least overlap to occupied space?
            
            smo = C_pao.tT().circulantdot(S.circulantdot(C_pao))
            
            smo_full = np.sum((smo - I).blocks**2, axis = (0,2))
            
            occ_span  = np.sum(
                C_pao.tT().circulantdot(S.circulantdot(co)).blocks**2, 
                axis = (0,1))
            
            virt_span  = np.sum(
                cv.tT().circulantdot(S.circulantdot(C_pao)).blocks**2, 
                axis = 0)
            
            virt_span = np.max(virt_span, axis = 1)
            
            
            C_full = extend(C_full, C_pao, np.argmin(occ_span))
            
            virtblocks = C_full.cget(C_full.coords)[:, :, np.argmin(occ_span)].reshape(C_full.blocks.shape[0]-1, 
                                                                                       C_full.blocks.shape[1],
                                                                                       1)
            C_virt = tp.tmat()
            C_virt.load_nparray(virtblocks, C_full.coords)
            
            
            
        else:
        
            # 1. Measure properties of every PAO

            # 1.a ) Overlap to occupied space

            occ_span = np.sum(np.abs(co.tT().circulantdot(S.circulantdot(C_pao)).blocks**2), axis = 0)
            occ_span = np.max(occ_span, axis = 0)
                              
            
            virt_span  = np.sum(
                C_virt.tT().circulantdot(S.circulantdot(C_pao)).blocks**2, 
                axis = 0)
            
            virt_span = np.max(virt_span, axis = 0)
            
            
            # norm should be 1
            #norm = np.diag(C_pao.tT().circulantdot(S.circulantdot(C_pao)).cget([0,0,0]))
            
            #np.arange(len(norm))
            
            #print("Norm")
            #print(norm)
            #print("Occ span")
            #print(occ_span)
            #print("Virt span")
            #print(virt_span)
            
            
            pao_i = np.argmin(virt_span)
            #print(" ")
            print("Selected orbital", pao_i, )
            print("Representation of orbital in occupied space:", virt_span[pao_i])
            print("Representation of orbital in virtual space :", occ_span[pao_i])
            # "representation" = largest expansion coefficient in wannier space
            
            #print(" ")
            if virt_span[pao_i]>thresh or occ_span[pao_i]>thresh:
                break
            
            
            
            
            
            
            
            
            
            #virt_span 
            C_full = extend(C_full, C_pao, pao_i)
            C_virt = extend(C_virt, C_pao, pao_i)
            

        
        #print("Span comparison of virtual spaces")

        smo = cv.tT().circulantdot(S.circulantdot(C_full))
        #print(np.sum(smo.blocks**2, axis = (0,1)))
        #print(np.sum(smo.blocks**2, axis = (0,2)))
        #print(" ")

        
    print(" Extracted %i paos" % i)
    
    if orthogonalize:
        C_full = orthogonalize_tmat_cholesky(C_full, p)
    
    return C_full # when done, return complete set with both occupied and virtual space

def orthogonal_paos(c,p):
    # Generate linearly dependent and non-orthogonal PAOs
    s, c_pao, wcenters = conventional_paos(c,p)

    #c_pao = tp.get_zero_tmat(np.max(np.abs(c_pao_.coords), axis = 0), (c_pao_.blocks.shape[1], c_pao_.blocks.shape[2]))

    #c_pao.blocks[ c_pao.mapping[ c_pao._c2i(c_pao_.coords)]] = c_pao_.cget(c_pao_.coords)

    c_occ, c_virt = PRI.occ_virt_split(c,p)




    #s = compute_smat(p)

    # Compute "metric-matrix" / overlap
    delta = c_pao.tT().circulantdot(s.circulantdot(c_pao))

    # Determine s^-1/2
    d_ = delta.kspace_svd_lowdin()

    # Eigenvectors and eigenvalues
    mu, u = d_.kspace_eig()

    mu_sum = np.sqrt(np.diag(np.sum(mu.blocks**2, axis = 0)))

    thresh = 0.01


    mu_ = tp.tmat()
    mu_.load_nparray(mu.blocks[:-1, mu_sum>thresh,:], mu.coords)

    u_ = tp.tmat()
    u_.load_nparray(u.blocks[:-1, :, mu_sum>thresh], mu.coords)

    c_pao_orthonormal = c_pao.circulantdot(u_.circulantdot(mu_))

    c_pao_ortho = tp.tmat()
    c_pao_ortho.load_nparray(c_pao_orthonormal.blocks[:-1, :, mu_sum>thresh], c_pao_orthonormal.coords)

    # normalize
    norms = np.diag(c_pao_ortho.tT().circulantdot(s.circulantdot(c_pao_ortho)).cget([0,0,0]))

    #normalize
    
    for i in np.arange(len(norms)):
        c_pao_ortho.blocks[:,:,i] = c_pao_ortho.blocks[:,:,i]/np.sqrt(norms[i])

    
    # span conserved?

    s_virt_pao = c_virt.tT().circulantdot(s.circulantdot(c_pao_ortho))
    print("Virtual span (ortho):", np.sum(s_virt_pao.blocks**2, axis = (0,1)))
    print("Virtual span (ortho):", np.sum(s_virt_pao.blocks**2, axis = (0,2)))



    wcenters, spreads = centers_spreads(c_pao_ortho, p,s.coords, m= 1)

    return s, c_pao_ortho, wcenters

def unfolded_pao(c,p, mx = None):
    """
    Computes PAOs using unfolded matrix products
    """
    co,cv = PRI.occ_virt_split(c,p)
    
    d = tp.unfolded_product(co, co.tT(), mx = mx)
    
    s = overlap_matrix(p)
    
    cpao = tp.unfolded_product(d,s, mx = mx)*-1
    
    cpao.blocks[cpao.mapping[cpao._c2i([0,0,0])]]+= np.eye(p.get_n_ao(), dtype = float)
    
    # Test orthogonality 

    print(np.max(np.abs(cpao.tT().circulantdot(s.circulantdot(co)).blocks)))
    
    return cpao