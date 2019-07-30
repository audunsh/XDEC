import os
import sys
import argparse

import numpy as np
import prism as pr
import toeplitz as tp
import post_process as pp

def sphere_distribution(N_samples, R, R0 = None):
    """
    Generate N random coordinates within a 
    spherical shell with radius R0-R
    """
    
    if R0 is None:
        V0 = 0
        R0 = 0 #-R
    else:
        V0 = (4*np.pi*R0**3)/3
    r = np.random.uniform(R0**3, R**3, N_samples)**(1/3.0)
    
    theta = np.random.uniform(0,2*np.pi, N_samples)
    phi = np.arccos(np.random.uniform(-1,1, N_samples))
    

    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    
    
    V = (4*np.pi*R**3)/3 - V0 #volume
    
    return x,y,z, np.abs(V)

def converge_shell(p, Ca, Cb,S, coord, opf = "outfile", Nmax= 121, nchunks = 20, chunksize = 2000):
    """
    Monte Carlo integration with discrete 
    importance sampling:
    Converge 
    """
    Sc = S.cget(coord)
    r0 = np.dot(coord+np.ones(3)*.5,p.lattice) #+np.dot(np.ones(3), p.lattice)*.5
    print("intergation center:", r0)
    r = np.linspace(0,50,Nmax)
    Z, Zabs, Z2 = 0,0,0
    Z_err, Zabs_err, Z2_err = 0,0,0
    Z_tot, Zabs_tot, Z2_tot = 0,0,0

    #nchunks = 20
    #chunksize = 2000
    chunkshape = (nchunks, Ca.blockshape[1], Cb.blockshape[1])
    
    SL2 = tp.L2norm(S.cget(-coord))
    
    for i in np.arange(Nmax-1):
        Zc = np.zeros(chunkshape, dtype = float)
        Zabsc = np.zeros(chunkshape, dtype = float)
        Z2c = np.zeros(chunkshape, dtype = float)
        
        for j in np.arange(nchunks):
            x,y,z,V = sphere_distribution(chunksize,r[i+1], r[i])
            
            x += r0[0]
            y += r0[1]
            z += r0[2]
            
            Zc[j], Z2c[j], Zabsc[j] = p.evaluate_doi_at(Ca,Cb,x,y,z, coord)
            
        Z += np.mean(Zc*V, axis = 0)
        Z2+= np.mean(Z2c*V, axis = 0)
        Zabs += np.mean(Zabsc*V, axis = 0)
        
        Z_tot += V*Zc
        Zabs_tot += V*Zabsc
        Z2_tot += V*Z2c
        
        Z_tot_err = np.std(Z_tot, axis = 0)/np.sqrt(nchunks)
        Zabs_tot_err = np.std(Z_tot, axis = 0)/np.sqrt(nchunks)
        Z2_tot_err = np.std(Z_tot, axis = 0)/np.sqrt(nchunks)
        
        
        Z_err += np.std(Zc*V, axis = 0)/np.sqrt(nchunks)
        Z2_err+= np.std(Z2c*V, axis = 0)/np.sqrt(nchunks)
        Zabs_err += np.std(Zabsc*V, axis = 0)/np.sqrt(nchunks)

        #d = np.arange(4)
        print("Block  ", -coord)
        print("Interval (%.2e, %.2e)" % (r[i], r[i+1]))
        print("Overlap comparison:")
        print(Z[-1])
        print(S.cget(-coord)[-1])
        print("S2")
        print(Z2[0])
        print("!S!")
        print(Zabs[0])
        
        print("Elements within error of overlap matrix:")
        print(np.sum(np.abs(Z-S.cget(-coord))<Z_err), "/", Z.size)
        print("Norm difference    :", tp.L2norm(Z-S.cget(-coord)))
        print(np.max(np.abs(S.cget(-coord)-Z)/Z_tot_err)/Z.size)
        #print("Relative difference:", tp.L2norm((Z-S.cget(-coord))/S.cget(-coord)))
    results = np.array([-coord, Z_tot, Zabs_tot, Z2_tot, nchunks, chunksize, chunkshape])
    np.save(opf+"block_%i_%i_%i.npy" %(-coord[0], -coord[1], -coord[2]), results)
    print("Results stored in block_%i_%i_%i.npy" %(-coord[0], -coord[1], -coord[2]))
    print(" ")
    return Z, Zabs, Z2
    
def batch_iterate(p, Ca, Cb,S, coords, N, process = 0, opf = "outfile"):
    """(a, n, N)"""
    n_iter = len(coords)/N
    for i in np.arange(n_iter):
        I = int(i*N + process)
        if  I<len(coords):
            c = coords[I]
            print(c)
            #converge_shell(p, Ca, Cb,S, c, opf)


def batch_integrate(p, Ca, Cb,S, N, process = 0, opf = "outfile"):
    """
    Integrate all blocks in <Ca!Cb>
    """
    dN = int(len(Cb.coords)/N)
    
    #set up all batches
    batch = []
    for i in np.arange(N-2):
        batch.append(np.arange(dN*i, dN*(i+1)))
    batch.append(np.arange(dN*(N-1), len(Cb.coords))) #last one is special
    #print(len(batch))
    for i in batch[process]:
        converge_shell(p, Ca, Cb,S, Cb.coords[i], opf)
    
if __name__ == "__main__":
    print("""#####################################################
##  ,--.   ,--.      ,------.  ,------. ,-----.    ##
##   \  `.'  /,-----.|  .-.  \ |  .---''  .--./    ##
##    .'    \ '-----'|  |  \  :|  `--, |  |        ## 
##   /  .'.  \       |  '--'  /|  `---.'  '--'\    ##
##  '--'   '--'      `------DOI ------''--- --'    ##
#####################################################""")
    parser = argparse.ArgumentParser(prog = "Differential overlap integrator",
                                        description = "",
                                        epilog = "Author: Audun Skau Hansen (e-mail: a.s.hansen@kjemi.uio.no)")
    
    parser.add_argument("-Ca", type=str, default="Ca.npy", help="tmat file")
    parser.add_argument("-Cb", type=str, default="Cb.npy", help="tmat file")
    parser.add_argument("-S", type=str, default="S.npy", help="AO overlap file")
    parser.add_argument("-Np", type=int, default=20, help="Number of processes")
    parser.add_argument("-n", type=int, default=0, help="This process")
    parser.add_argument("-pfile", type=str, default="input.d12", help="crystal input file")
    parser.add_argument("-opf", type=str, default="results", help="out prefix")
    parser.add_argument("-coords", type=str, default=None, help="coords")
    
    args = parser.parse_args()
    
    
    p = pr.prism(args.pfile)
    
    Ca = tp.tmat()
    Ca.load(args.Ca)
    
    Cb = tp.tmat()
    Cb.load(args.Cb)
    
    Cb = tp.tmat()
    Cb.load(args.Cb)
    
    S = tp.tmat()
    S.load(args.S)
    
    Smo = Ca.tT().cdot(S*Cb, coords = Ca.coords)
    if args.coords is None:
        batch_integrate(p, Ca, Cb, Smo, args.Np, args.n, args.opf)
    else:
        coor = np.load(args.coords)
        batch_iterate(p, Ca, Cb, Smo, coor, args.Np, args.n, args.opf)
