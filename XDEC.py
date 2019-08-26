import numpy as np

import os

import subprocess as sp

from ast import literal_eval

import argparse

import utils.toeplitz as tp

import lwrap.lwrap as li

import utils.prism as pr

import domdef as dd

import PRI 

import time

def mapgen(vex1, vex2):
    # vex1 - vex2
    # returns a matrix where vex1[ ret_indx[i,j] ] = vex1[i] - vex2[j]
    D = vex1[:, None] - vex2[None, :]
    ret_indx = -1*np.ones((vex1.shape[0], vex2.shape[0]), dtype = int)
    for i in np.arange(vex1.shape[0]):
        for j in np.arange(vex2.shape[0]):
            try:
                ret_indx[i,j] = np.argwhere(np.all(np.equal(vex1, D[i,j]), axis = 1))
            except:
                pass
    return ret_indx

def advance_amplitudes(pair_extent, virtual_extent, t2,G,E):
    t2_new = np.zeros_like(t2)
    for M in np.arange(len(pair_extent)):
        for dL in np.arange(len(virtual_extent)):
            for dM in np.arange(len(virtual_extent)):
                #for L in np.arange(len(pair_extent)):
                #cL = pair_extent[L]
                #t2[:, dL, :, M, :, dM, :]


                # t2[i, K, a, L, j, M, b]
                # Contract each term using einsum
                # + left(0i \Delta L a |Mj \Delta M b \right) 
                tnew  = -G[:,dL,:,M,:,dM,:] 
                
                # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{0i,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                Fac = Fmo_aa.cget(virtual_extent - virtual_extent[dL])
                tnew -= np.einsum("iKcjb,Kac->iajb", t2[:, :-1, :, M, :, dM, :], Fac)

                # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                Fbc = Fmo_aa.cget(virtual_extent - virtual_extent[dM])
                tnew -= np.einsum("iajKb,Kbc->iajb", t2[:, dL, :, M, :, :-1, :], Fbc)
                
                # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                Fki = Fmo_ii.cget(-1*pair_extent)
                tnew += np.einsum("Kkajb,Kki->iajb",t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)
                
                # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                Fkj = Fmo_ii.cget(-1*pair_extent + pair_extent[M])
                tnew += np.einsum("iaKkb,Kkj->iajb",t2[:, dL, :, :, :, dM, :], Fkj)

                #tnew *= E[:,dL,:,M,:,dM,:]


                # + \left(t^{\Delta L a, \Delta Mb}_{L'k,Mj}\right)_{n}\varepsilon^{\Delta L a, \Delta Mb}_{0i,Mj},
                #tnew += t2[:, dL, :, M, :, dM, :] #*E[:,dL,:,M,:,dM,:]**-1

                t2_new[:, dL, :, M, :, dM, :] = tnew #*E[:,dL,:,M,:,dM,:]
    #t2_new[:,:-1,:,:-1,:,:-1,:] *= E
    t2_new[:,:-1,:,:-1,:,:-1,:] = t2_new[:,:-1,:,:-1,:,:-1,:] + t2[:,:-1,:,:-1,:,:-1,:]
    return t2_new

def advance_amplitudes_(pair_extent, virtual_extent, t2,G,E):
    t2_new = np.zeros_like(t2)
    for M in np.arange(len(pair_extent)):
        for dL in np.arange(len(virtual_extent)):
            for dM in np.arange(len(virtual_extent)):
                #for L in np.arange(len(pair_extent)):
                #cL = pair_extent[L]
                #t2[:, dL, :, M, :, dM, :]


                # t2[i, K, a, L, j, M, b]
                # Contract each term using einsum
                # + left(0i \Delta L a |Mj \Delta M b \right) 
                tnew  = -G[:,dL,:,M,:,dM,:] 
                
                # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{0i,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                Fac = Fmo_aa.cget(virtual_extent - virtual_extent[dL])
                tnew = np.einsum("iKcjb,Kac->iajb", t2[:, :-1, :, M, :, dM, :], Fac)

                # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                Fbc = Fmo_aa.cget(virtual_extent - virtual_extent[dM])
                tnew += np.einsum("iajKb,Kbc->iajb", t2[:, dL, :, M, :, :-1, :], Fbc)
                
                # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                Fki = Fmo_ii.cget(-1*pair_extent)
                tnew -= np.einsum("Kkajb,Kki->iajb",t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)
                
                # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                Fkj = Fmo_ii.cget(-1*pair_extent + pair_extent[M])
                tnew -= np.einsum("iaKkb,Kkj->iajb",t2[:, dL, :, :, :, dM, :], Fkj)

                #tnew *= E[:,dL,:,M,:,dM,:]


                # + \left(t^{\Delta L a, \Delta Mb}_{L'k,Mj}\right)_{n}\varepsilon^{\Delta L a, \Delta Mb}_{0i,Mj},
                #tnew += t2[:, dL, :, M, :, dM, :] #*E[:,dL,:,M,:,dM,:]**-1

                t2_new[:, dL, :, M, :, dM, :] = tnew #*E[:,dL,:,M,:,dM,:]
    #t2_new[:,:-1,:,:-1,:,:-1,:] *= E
    t2_new[:,:-1,:,:-1,:,:-1,:] = t2_new[:,:-1,:,:-1,:,:-1,:]*E + t2[:,:-1,:,:-1,:,:-1,:]
    return t2_new

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
    parser.add_argument("wcenters", type = str, help="Wannier centers")
    parser.add_argument("attenuation", type = float, default = 1.2, help = "Attenuation paramter for RI")
    args = parser.parse_args()


    # Load system

    p = pr.prism(args.project_file)



    # Fitting basis
    auxbasis = PRI.basis_trimmer(p, args.auxbasis)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()
    

    # Wannier coefficients
    c = tp.tmat()
    c.load(args.coefficients)
    
    # AO Fock matrix
    f_ao = tp.tmat()
    f_ao.load(args.fock_matrix)

    # Compute MO Fock matrix
    f_mo = c.tT().cdot(f_ao*c, coords = c.coords)

    # Compute energy denominator
    f_aa = f_mo.cget([0,0,0])[np.arange(p.get_nocc(),p.get_n_ao()), np.arange(p.get_nocc(),p.get_n_ao())]
    f_ii = f_mo.cget([0,0,0])[np.arange(p.get_nocc()),np.arange(p.get_nocc()) ]
    
    e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]


    # Wannier centers
    wcenters = np.load(args.wcenters)

    # Initialize integrals 
    ib = PRI.integral_builder(c,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[1,0,0])

    # Initialize domain definitions



    d = dd.build_distance_matrix(p, c.coords, wcenters, wcenters)
    center_fragments = dd.atomic_fragmentation(p, d, 3.0)

    

    # Converge atomic fragment energies

    for fragment in center_fragments:
        print("Fragment:", fragment)
        
        # Expand virtual space
        di = dd.build_local_domain_index_matrix(fragment, d, 5.0)

        # Set up initial guess for amplitudes
        t = np.zeros((15,15,15), dtype = tp.tmat) #cluster amplitudes

        params = []

        for ddL in np.arange(di.coords.shape[0]):
            for ddM in np.arange(di.coords.shape[0]):
                dL, dM = di.coords[ddL], di.coords[ddM]

                
                g_direct = ib.getcell(dL, [0,0,0], dM)
                g_exchange = ib.getcell(dM, [0,0,0], dL)
                t = g_direct*e_iajb**-1
                params.append([dL, dM, t, g_direct, g_exchange])




        e0 = 0
        # Perform (initial guess) energy calculation for fragment
        for dL in di.coords:
            for dM in di.coords:
                
                g_direct = ib.getcell(dL, [0,0,0], dM)
                g_exchange = ib.getcell(dM, [0,0,0], dL)

                t = g_direct*e_iajb**-1




                e0 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
                print(dL, dM, e0)
        print("e0", e0)









    #ib.getcell()




    """
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

    Fmo = C.tT().cdot(F.cdot(C), coords = C.coords)

    Ni = 11
    Na = 11

    Ni = p.get_nocc()
    Na = p.get_nvirt()
    Nmo = p.get_n_ao()

    virtual_extent = tp.lattice_coords((0,0,0))

    pair_extent    = tp.lattice_coords((0,0,0))

    

    Fmo.cget( F.coords - np.array([0,0,0]))

    
    print(coulomb_extent)
    # Compute coulomb matrix
    #JK = PRI.compute_JK(p, s, auxname = "ri-fitbasis")

    JK = PRI.compute_JK(p,s, coulomb=True, auxname = "ri-fitbasis") 
    JK.tolerance = 10e-12
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
                    print(i,j,k,"partially computed.")
                except:
                    print(i,j,k,"skipped.")
                    pass
    
    

    # for instance
    """



    """
    print(Xreg[0,0,0].blocks.max())
    t0 = time.process_time()

    print(Xreg[0,0,0].cget([0,0,0])[0])

    x000 = Xreg[0,0,0].tT().cdot(JKX[0,0,0], coords = [[0,0,0]])  # = (000i000a|000j000b)

    Xreg[0,0,0].tT().cdot(JKX[0,0,0], coords = [[1,0,0]])  # = (000i000a|100j100b)

    Xreg[0,0,0].tT().cdot(JKX[1,0,0], coords = [[0,0,0]])  # = (000i000a|000j100b)

    t1 = time.process_time()

    print("timing:", t1-t0)

    

    print(x000.cget([0,0,0])[0])
    """

    


    """

    Nv = len(virtual_extent)
    Np = len(pair_extent)

    t2 = np.zeros((Ni, Nv+1, Na, Np+1, Ni, Nv+1, Na), dtype = float) #pad one extra blocks of zeros in each coordinate direction for negative indexing to zero block
    G = np.zeros((Ni, Nv, Na, Np, Ni, Nv, Na), dtype = float)
    E = np.zeros((Ni, Nv, Na, Np, Ni, Nv, Na), dtype = float)
    
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

    # split Fmo in Fmo_aa and Fmo_ii (virtual + occupied)
    Fmo_aa = tp.tmat()
    Fmo_aa.load_nparray(Fmo.blocks[:,p.get_nocc():,p.get_nocc():], Fmo.coords[:])

    Fmo_ii = tp.tmat()
    Fmo_ii.load_nparray(Fmo.blocks[:,:p.get_nocc(),:p.get_nocc()], Fmo.coords[:])

    
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
                g_ = v_.reshape((Nmo,Nmo,Nmo,Nmo))[:Ni,Ni:,:Ni,Ni:]
                print(g_.max())

                t2[:, dL, :, M, :, dM, :] = g_*e_iajb**-1
                G[:, dL, :, M, :, dM, :]=g_
                E[:, dL, :, M, :, dM, :]=e_iajb**-1


                i_count += 1

                print(cdL, cM, cdM, " computed (%.2f  complete)." % (i_count/iterations_total))
                print()

                #e0 += 2*np.einsum("iajb,iajb",t2[:2, dL, 2:, M, :2, dM, 2:],g_, optimize = True)  - np.einsum("iajb,ibja",t2[:2, dL, 2:, M, :2, dM, 2:],g_, optimize = True)
                e0 += 2*np.einsum("iajb,iajb",t2[:, dL, :, M, :, dM, :],g_, optimize = True)  - np.einsum("iajb,ibja",t2[:, dL, :, M, :, dM, :],g_, optimize = True)
    

    print("Initial (guess) energy:", e0)
    
    



    print("Energy:", 2*np.einsum("iKaLjMb,iKaLjMb",t2[:,:-1,:,:-1,:,:-1,:],G, optimize = True)  - np.einsum("iKaLjMb,iMbLjKa",t2[:,:-1,:,:-1,:,:-1,:],G, optimize = True))
    
    # debug option

    #np.save("t2amplitudes.npy", t2)
    #np.save("gtens.npy",         G)
    #np.save("energy_denom.npy",  E)











    #t2 = np.load("t2amplitudes.npy")
    #G = np.load("gtens.npy")
    #E = np.load("energy_denom.npy")
    print("Energy:", 2*np.einsum("iKaLjMb,iKaLjMb",t2[:,:-1,:,:-1,:,:-1,:],G, optimize = True)  - np.einsum("iKaLjMb,iMbLjKa",t2[:,:-1,:,:-1,:,:-1,:],G, optimize = True))

    # map indices

    vp_indx = mapgen(virtual_extent, pair_extent)
    pp_indx = mapgen(pair_extent, pair_extent)
 
    # Solving the MP2-equations for the non-canonical reference state
    for i in np.arange(200):
        t2_new = advance_amplitudes(pair_extent, virtual_extent, t2,G,E)
        print(np.linalg.norm(t2_new[:,:-1,:,:-1,:,:-1,:]-t2[:,:-1,:,:-1,:,:-1,:]))
        t2 = t2_new

        print("Energy:", 2*np.einsum("iKaLjMb,iKaLjMb",t2[:,:-1,:,:-1,:,:-1,:],G, optimize = True)  - np.einsum("iKaLjMb,iMbLjKa",t2[:,:-1,:,:-1,:,:-1,:],G, optimize = True))


    #print("Gradient norm:", np.linalg.norm(t2-t2_new))
    #return t2_new
    """

                








    

#def advance_amplitudes(g0, t,fab,fij, f_iajb):
#    # Advance MP2 amplitudes one iteration
#    d1 =   np.einsum("icjb,ac->iajb", t,fab, optimize = True)
#    d2 =   np.einsum("iajc,bc->iajb", t,fab, optimize = True)
#    d3 =   np.einsum("kajb,ki->iajb", t,fij, optimize = True)
#    d4 =   np.einsum("iakb,kj->iajb", t,fij, optimize = True)
#    return (g0 + d1 + d2 -d3-d4)*f_iajb**-1 + t #*f_iajb/f_iajb