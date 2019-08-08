import numpy as np

import os

import subprocess as sp

from ast import literal_eval

import argparse

import utils.toeplitz as tp

import lwrap.lwrap as li

import utils.prism as pr

import PRI 

import time


if __name__ == "__main__":
    os.environ["LIBINT_DATA_PATH"] = os.getcwd() 
    print("""#####################################################
##  ,--.   ,--.      ,------.  ,------. ,-----.    ##
##   \  `.'  /,-----.|  .-.  \ |  .---''  .--./    ##
##    .'    \ '-----'|  |  \  :|  `--, |  |        ## 
##   /  .'.  \       |  '--'  /|  `---.'  '--'\    ##
##  '--'   '--'      eXtended local correlation    ##
##                   Author : Audun Skau Hansen    ##
##                                                 ##
##  Use keyword "--help" for more info             ## 
#####################################################""")

        
    # Parse input
    parser = argparse.ArgumentParser(prog = "X-DEC: eXtended Divide-Expand-Consolidate scheme",
                                     description = "Local correlation for periodic systems.",
                                     epilog = "Author: Audun Skau Hansen (e-mail: a.s.hansen@kjemi.uio.no)")
    parser.add_argument("project_file", type = str, help ="input file for project (.d12 file)")
    parser.add_argument("coefficients", type= str,help = "Coefficient matrix from Crystal")
    parser.add_argument("fock_matrix", type= str,help = "AO-Fock matrix from Crystal")
    parser.add_argument("fitted_coeffs", type= str,help="Array of coefficient matrices from RI-fitting")
    parser.add_argument("auxbasis", type = str, help="Auxiliary fitting basis.")
    
    args = parser.parse_args()

    p = pr.prism(args.project_file)
    auxbasis = PRI.basis_trimmer(p, args.auxbasis)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()
    

    Xreg = np.load(args.fitted_coeffs)

    coulomb_extent = np.max(np.abs(Xreg[0,0,0].coords), axis = 0)
    print(coulomb_extent)

    s = tp.tmat()
    scoords = tp.lattice_coords(coulomb_extent)
    s.load_nparray(np.ones((scoords.shape[0],2,2), dtype = float), scoords)

    C = tp.tmat()
    C.load(args.coefficients)

    F = tp.tmat()
    F.load(args.fock_matrix)
    
    print(coulomb_extent)
    # Compute coulomb matrix
    #JK = PRI.compute_JK(p, s, auxname = "ri-fitbasis")

    JK = PRI.compute_JK(p,s, coulomb=True, auxname = "ri-fitbasis") 

    print("JK.max:", JK.blocks.max())
    print("JK(1,0,0).max:", np.abs(JK.cget([1,0,0])).max())
    print(JK.cget([1,0,0])[:3,:3])
    print(JK.coords)
    print(scoords)
    #construct EOS and AOS for cell fragment
    JKX = np.zeros_like(Xreg)

    for i in np.arange(len(Xreg)):
        for j in np.arange(len(Xreg[i])):
            for k in np.arange(len(Xreg[i,j])):
                try:
                    JKX[i,j,k] = JK.cdot(Xreg[i,j,k])
                except:
                    pass
                print(i,j,k,"partially computed")
    

    # for instance
    print(Xreg[0,0,0].blocks.max())
    t0 = time.process_time()

    print(Xreg[0,0,0].cget([0,0,0])[0])

    x000 = Xreg[0,0,0].tT().cdot(JKX[0,0,0], coords = [[0,0,0]])  # = (000i000a|000j000b)

    Xreg[0,0,0].tT().cdot(JKX[0,0,0], coords = [[1,0,0]])  # = (000i000a|100j100b)

    Xreg[0,0,0].tT().cdot(JKX[1,0,0], coords = [[0,0,0]])  # = (000i000a|000j100b)

    t1 = time.process_time()

    print("timing:", t1-t0)

    Fmo = C.tT().cdot(F.cdot(C), coords = C.coords)

    print(x000.cget([0,0,0])[0])




    Ni = 11
    Na = 11
    virtual_extent = tp.lattice_coords((1,1,1))

    pair_extent    = tp.lattice_coords((0,0,0))

    Nv = len(virtual_extent)
    Np = len(pair_extent)

    t2 = np.zeros((Ni, Nv, Na, Np, Ni, Nv, Na), dtype = float)
    G = np.zeros((Ni, Nv, Na, Np, Ni, Nv, Na), dtype = float)
    
    print("Computing the g-tensor")

    # fock matrix elements
    #f_aa = f[np.arange(2,11), np.arange(2,11)]
    #f_ii = f[np.arange(2), np.arange(2)]
    
    fij = Fmo.cget([0,0,0])[:2,:2]
    fab = Fmo.cget([0,0,0])[2:,2:]
    
    #fij[np.arange(2), np.arange(2)] *= 0
    #fab[np.arange(9), np.arange(9)] *= 0
    
    f_aa = Fmo.cget([0,0,0])[np.arange(2,11), np.arange(2,11)]
    f_ii = Fmo.cget([0,0,0])[np.arange(2), np.arange(2)]
    
    #f_iajb = np.einsum("i,a,j,b->iajb", f_ii,-1*f_aa, f_ii, -1*f_aa)
    e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]

    # two body interaction elements
    iterations_total = len(pair_extent)*len(virtual_extent)**2
    i_count = 0

    e0 = 0
    for M in np.arange(len(pair_extent)):
        cM = pair_extent[M]
        for dL in np.arange(len(virtual_extent)):
            cdL = virtual_extent[dL]
            for dM in np.arange(len(virtual_extent)):
                cdM = virtual_extent[dM]
                v_ = Xreg[cdL[0],cdL[1],cdL[2]].tT().cdot(JKX[cdM[0],cdM[1],cdM[2]], coords = [[cM[0],cM[1],cM[2]]]).cget([cM[0],cM[1],cM[2]])
                g_ = v_.reshape((Ni,Na,Ni,Na))[:2,2:,:2,2:]
                print(g_.max())
                #v_anti = Xreg[cdL[0],cdL[1],cdL[2]].tT().cdot(JKX[cdM[0],cdM[1],cdM[2]], coords = [[cM[0],cM[1],cM[2]]]).cget([cM[0],cM[1],cM[2]])
                #e_ = 
                #print(t2[:2,:,:,:,:,:,:].shape)
                t2[:2, dL, 2:, M, :2, dM, 2:] = g_*e_iajb**-1
                #print((g_*e_iajb**-1).shape, t2[:2, dL, 2:, M, :2, dM, 2:].shape)
                #print(t2.shape)
                i_count += 1

                print(cdL, cM, cdM, " computed (%.2f  complete)." % (i_count/iterations_total))
                print()

                e0 += 2*np.einsum("iajb,iajb",t2[:2, dL, 2:, M, :2, dM, 2:],g_, optimize = True)  - np.einsum("iajb,ibja",t2[:2, dL, 2:, M, :2, dM, 2:],g_, optimize = True)

    # Solving the MP2-equations for the non-canonical reference state

    print("Initial energy:", e0)
    # Calculate energy
    # 2*np.einsum("iajb,iajb",,g, optimize = True) - np.einsum("iajb,ibja",t,g, optimize = True)






    """
    # Hardcoding first run
    Ni = 2
    Na = 9
    Nj = 2
    Nb = 9
    NdLx = 3
    NdLy = 3
    NdLz = 3
    NMx = 3
    NMy = 3
    NMz = 3
    NdMx = 3
    NdMy = 3
    NdMz = 3




    t2 = np.zeros((Ni, NdLx, NdLy, NdLz,Na, NMx, NMy, NMz, Nj, NdMx, NdMy, Ndmz, Nb), dtype = float) #periodic t2 amplitudes
    for c_pair_M in pair_extent:
        Mx, My, Mz = c_pair_M
        for c_virt_L in virtual_extent:
            dLx, dLy, dLz = c_virt_L
            for c_virt_M in virtual_extent:
                dMx, dMy, dMz = c_virt_M
                t2[:,dLx,dLy,dLz,:,Mx,My,Mz,:,dMx,dMy,dMz] = Xreg[dLx,dLy,dLz].tT().cdot(JKX[dMx, dMy, dMz], coords = [[Mx,My,Mz]])
    """

