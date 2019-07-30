# -*- coding: utf-8 -*-
###
#
#  Post-process/
#  Unitary annealing module
#
#####

import crystal_interface as ci

import os
import sys
import subprocess as sp
from ast import literal_eval

import numpy as np

#sys.path.insert(0, '/Users/audunhansen/PeriodicDEC/august2016/src/libint_interface/onepartop')  
#sys.path.insert(0, os.getcwd() + \
#                "/build/src/libint_interface/onepartop")
this_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_directory + \
                "/../build/src/libint_interface/onepartop")

import prism as pr
from toeplitz import tmat
from toeplitz_latmat import latmat



import py_oneptoper
#import libint_interface.onepartop.py_oneptoper



class Atom:
    def __init__(self,x,y,z,n):
        self.pos = [x,y,z]
        self.atomic_number = n
    def x(self):
        return self.pos[0]
    def y(self):
        return self.pos[1]
    def z(self):
        return self.pos[2]
    def __mul__(self, other):
        #return self
        #print(other)
        x,y,z = self.pos+other
        return Atom(x,y,z,self.atomic_number)

def l2atoms(positions, charges):
    atoms = []
    for i in np.arange(len(charges)):
        x,y,z = positions[i]
        n = charges[i]
        atoms.append(Atom(x,y,z,n))
    return atoms
    
def get_uncontracted(fname):
    f = open(fname, "r")
    #print(f.read())
    F = f.read().split("****")[1:]
    for i in F:
        print(i)
    contracteds = []
    
    orb_types = "spdfghi"
    
    #print(F)
    l = "s"
    sret = "****\nH 0\n"
    for i in range(len(F)):
        Fs = F[i].split("\n")[2:-1]


        for j in range(len(Fs)):
            Fsj = Fs[j].split()
            #print(Fsj)
            if len(Fsj)==3:
                l = Fsj[0]
                for k in np.arange(2*orb_types.index(l)  + 1):
                    contracteds.append(literal_eval(Fsj[1]))
            else:
                sret += "%s 1 1.0000\n" % l
                sret += "%s \n" % Fs[j]


            #print(i,j, Fs[j])
            #print([k for k in Fs[j].split()])
    sret += "****\n"
    print(sret)
    return sret, contracteds
    

    

def compute_contraction_coeffs(g94file, project_folder):
    #Compute normalization coefficients for contracted gaussians
    
    uncont, contracteds = get_uncontracted(g94file)
    
    print(contracteds)
    opath = os.environ["LIBINT_DATA_PATH"]
    
    
    os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC/"
    
    f = open(project_folder + "XDEC/basisfile_uncontract.g94", "w")
    f.write(uncont)
    f.close()
    
    p = py_oneptoper.PyOnePtOper()
    p.set_basis("basisfile_uncontract")
    
    atoms_0 = [Atom(0,0,0,1), Atom(0,0,0,1)]

    #p.set_atoms(atoms_0, atoms_0)
    p.set_atoms(atoms_0, atoms_0)
    
    p.init_engine("emultipole2", p.max_nprim(), p.max_l())
    
    

    mpole2 = p.calc_mpole()

    mpole2 = np.array(mpole2[0]).reshape(mpole2[1])

    sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()
    
    os.environ["LIBINT_DATA_PATH"] = opath #restore original path
    
    W = np.zeros(len(contracteds), dtype = float)
    for i in range(len(contracteds)):
        i0 = np.sum(contracteds[:i])
        #print(np.arange(i0,i0+contracteds[i]))
        #W[i] = 1.0/np.sqrt(np.sum(sm[i0:i0+contracteds[i],i0:i0+contracteds[i]]))
        W[i] = np.sum(sm[i0:i0+contracteds[i],i0:i0+contracteds[i]])
    return W


def shifttest(project_folder):
    geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
    cc = np.load(project_folder + "crystal_reference_coords.npy")
    C  = np.load(project_folder + "crystal_reference_state.npy")
    
    p0, c0 = geometry.get_atoms([[0,0,0]])
    p1, c1 = geometry.get_atoms([[-1,0,0]])
    p2, c2 = geometry.get_atoms([[1,0,0]])
    #print(p0, c0)
    atoms_0 = l2atoms(p0, c0)
    atoms_1 = l2atoms(p1, c1)
    atoms_2 = l2atoms(p2, c2)
    
    opath = os.environ["LIBINT_DATA_PATH"]
    os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC/"
    
    p = py_oneptoper.PyOnePtOper()
    p.set_basis("basis")
    
        
    #atoms_0 = [Atom(0,0,0,1), Atom(2,0,0,1)]

    p.set_atoms(atoms_0, atoms_0)
    p.init_engine("emultipole2", p.max_nprim(), p.max_l())

    p.set_atoms(atoms_0, atoms_0)

    mpole0 = p.calc_mpole()

    mpole0 = np.array(mpole0[0]).reshape(mpole0[1])

    sm0,xm0,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole0 #p.calc_mpole()
    
    p.set_atoms(atoms_1, atoms_1)

    mpole1 = p.calc_mpole()

    mpole1 = np.array(mpole1[0]).reshape(mpole1[1])

    sm1,xm1,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole1 #p.calc_mpole()
    
    p.set_atoms(atoms_2, atoms_2)

    mpole2 = p.calc_mpole()

    mpole2 = np.array(mpole2[0]).reshape(mpole2[1])

    sm2,xm2,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()

    s0 = xm0[0,:]
    s1 = xm1[0,:]
    s2 = xm2[0,:]
    print("Integral overlap, shifted")
    print(s0)
    print(s1)
    print(s2)
    
def brute_force_X0(project_folder):
    ## return <0 x^2 m>, <0 L_x x m> 
    
    
    geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
    cc = np.load(project_folder + "crystal_reference_coords.npy")
    C  = np.load(project_folder + "crystal_reference_state.npy")
    
    
    #cc = np.load(project_folder + "lsdalton_reference_coords.npy")
    #C  = np.load(project_folder + "lsdalton_reference_state.npy")
    
    opath = os.environ["LIBINT_DATA_PATH"]
    os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC/"

    p = py_oneptoper.PyOnePtOper()
    p.set_basis("basis")
    
    p0, c0 = geometry.get_atoms([[0,0,0]])
    #print(p0, c0)
    atoms_0 = l2atoms(p0, c0)
    
    p.set_atoms(atoms_0, atoms_0)
    p.init_engine("emultipole2", p.max_nprim(), p.max_l())

    mpole = p.calc_mpole()

    mpole = np.array(mpole[0]).reshape(mpole[1])



    sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole #p.calc_mpole()
    
    Lmo1 = np.zeros(sm.shape, dtype = float)
    Lmo2 = np.zeros(sm.shape, dtype = float)
    Smo0 = np.zeros(sm.shape, dtype = float)
    
    X2mo = np.zeros(sm.shape, dtype = float)
    xXmo = np.zeros(sm.shape, dtype = float)
    xxSmo = np.zeros(sm.shape, dtype = float)
    
    
    
    M = np.zeros(sm.shape, dtype = float)
    m = cc[0]
    for l in np.arange(len(cc)):
        #M = np.dot(cc[m]
        #L = np.dot(cc[l]
        pm, cm = geometry.get_atoms([cc[m]])
        pl, cl = geometry.get_atoms([cc[l]])
        #print(pm)
        atoms_1 = l2atoms(pm,cm)
        atoms_2 = l2atoms(pl,cl)
        
        #print(atoms_1)
        #print(atoms_2)

        p.set_atoms(atoms_1, atoms_2) # <m,l>
        
        mpole2 = p.calc_mpole()

        mpole2 = np.array(mpole2[0]).reshape(mpole2[1])


        sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()
        
        
        #if m==0:
        Xmo += np.dot(C[m].T, np.dot(xm, C[l]))
        Ymo += np.dot(C[m].T, np.dot(ym, C[l]))
        Zmo += np.dot(C[m].T, np.dot(zm, C[l]))
        
        #if np.all(cc[l]-cc[m]==0) :
        #if l==0 and m==0:
        
        #if l == 0: # and m== 0:
        X2mo += np.dot(C[m].T, np.dot(x2m, C[l]))
        Y2mo += np.dot(C[m].T, np.dot(y2m, C[l]))
        Z2mo += np.dot(C[m].T, np.dot(z2m, C[l]))
        
        #if m==0:
        XYmo += np.dot(C[m].T, np.dot(np.dot(xm,ym), C[l]))
        XZmo += np.dot(C[m].T, np.dot(np.dot(xm,zm), C[l]))
        YZmo += np.dot(C[m].T, np.dot(np.dot(ym,zm), C[l]))
        
        
        
        
        
        Lao1 = x2m + y2m + z2m #+2*xym + 2*xzm + 2*yzm
        Lao2 = xm + ym + zm
        
        #X = xm
        #if m==0:
        Lmo1 += np.dot(C[m].T, np.dot(Lao1, C[l]))
        
        Lmo2 += np.dot(C[m].T, np.dot(Lao2, C[l]))
        
        Smo0 += np.dot(C[m].T, np.dot(sm, C[l]))
        
        
        #Xmo += np.dot(C[m].T, np.dot(X, C[l]))
        
        #M += Lmo1 - Lmo2**2
        
        #if np.any(Lmo1 - Lmo2**2 < 0):
        #    pass
        #else:
        #M[np.arange(len(M)), np.arange(len(M))] += np.diag(Lmo1) - np.diag(Lmo2)**2
        
        #xC = np.dot(xm, C[l])
        #M += np.dot(C[m].T, xC )
            
    #M =  Xmo # Lmo1 #+ (Smo0-2)*Lmo2**2
    #M = Lmo1 + (Smo0-2)*Lmo2**2
    #M = Lmo1 - Lmo2**2
    M = Xmo  #+ Y2mo + Z2mo - (Xmo**2 + Ymo**2 + Zmo**2) # +2*(XYmo + XZmo + YZmo)) 
    #M = Xmo**2
    
    #M = np.dot(Xmo,Ymo) + np.dot(Xmo,Zmo) + np.dot(Ymo, Zmo)
    os.environ["LIBINT_DATA_PATH"] = opath #restore original path
    #np.save("L_bf_ethylene_column.npy", M)
    return M
    
    

def toeplitz_integrals(project_folder, tpe = "nuclear"):
    geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
    cc = np.load(project_folder + "crystal_reference_coords.npy")
    C  = np.load(project_folder + "crystal_reference_state.npy")
    
    #self.cc = np.array([np.arange(-30,31), np.zeros(61), np.zeros(61)]).T

    
    I = latmat()
    I.cc = cc
    
    p0, c0 = geometry.get_atoms([[0,0,0]])
    #print(p0, c0)
    atoms_0 = l2atoms(p0, c0)
    
    
    #set/unset libint data path, read correct basis
    opath = os.environ["LIBINT_DATA_PATH"]
    
    os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC"

    p = py_oneptoper.PyOnePtOper()
    p.set_basis("basis")
    
        
    #atoms_0 = [Atom(0,0,0,1), Atom(2,0,0,1)]

    p.set_atoms(atoms_0, atoms_0)
    p.init_engine(tpe, p.max_nprim(), p.max_l())
    #print(p.calc_mpole())
    
    
    
    for m in cc:
        pm, cm = geometry.get_atoms([m])
        #print(pm)
        atoms_1 = l2atoms(pm,cm)

        p.set_atoms(atoms_0, atoms_1)
        block = p.calc_elms()
        
        #print(sm)
        
        I.set(m, block)

    
    os.environ["LIBINT_DATA_PATH"] = opath #restore original path
    
    return I

def compute_overlap_cell(project_folder, cell = [[0,0,0]], tpe = "overlap"):
    geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
    
    #self.cc = np.array([np.arange(-30,31), np.zeros(61), np.zeros(61)]).T

    
    #I = latmat()
    #I.cc = cc
    
    p0, c0 = geometry.get_atoms([[0,0,0]])
    #print(p0, c0)
    atoms_0 = l2atoms(p0, c0)
    
    p1, c1 = geometry.get_atoms(cell)
    atoms_1 = l2atoms(p1, c1)
    
    #set/unset libint data path, read correct basis
    opath = os.environ["LIBINT_DATA_PATH"]
    
    os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC"

    p = py_oneptoper.PyOnePtOper()
    p.set_basis("basis")
    
        
    #atoms_0 = [Atom(0,0,0,1), Atom(2,0,0,1)]

    p.set_atoms(atoms_0, atoms_1)
    p.init_engine(tpe, p.max_nprim(), p.max_l())
    #print(p.calc_mpole()
    p.init_engine("emultipole2", p.max_nprim(), p.max_l())
    
    mpole = p.calc_mpole()

    mpole = np.array(mpole[0]).reshape(mpole[1])

 
    sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole #p.calc_mpole()
    
    os.environ["LIBINT_DATA_PATH"] = opath #restore original path
    return sm
    
def generate_basis(basis, a1, ep, Ns,Np,Nd,Nf, Ng, enum = "Au"):
    #bs = basis
    bs = basis  + " %s 0\n" % enum
    

    for k in np.arange(1,Ns+1):
        ak = a1[0]*ep[0]**(k-1)
        bs += "s 1 0.000000\n"
        bs += "    %.10f 1.0\n" % ak


    for k in np.arange(1,Np+1):
        ak = a1[1]*ep[1]**(k-1)
        bs += "p 1 0.000000\n"
        bs += "    %.10f 1.0\n" % ak  


    for k in np.arange(1,Nd+1):
        ak = a1[2]*ep[2]**(k-1)
        bs += "d 1 0.000000\n"
        bs += "    %.10f 1.0\n" % ak  

    for k in np.arange(1,Nf+1):
        ak = a1[3]*ep[3]**(k-1)
        bs += "f 1 0.000000\n"
        bs += "    %.10f 1.0\n" % ak  
        
    for k in np.arange(1,Ng+1):
        ak = a1[4]*ep[4]**(k-1)
        bs += "g 1 0.000000\n"
        bs += "    %.10f 1.0\n" % ak  
            
    #for k in np.arange(1,Nh+1):
    #    ak = a1[5]*ep[5]**(k-1)
    #    bs += "h 1 0.000000\n"
    #    bs += "    %.10f 1.0\n" % ak      

    #computing number of functions
    
    N_ao = Ns + Np*3 + Nd*5 + Nf*7 + Ng*9
    
    """
        840.000000000000000 0.002640000000000
    """

    bs+= """****
"""



    #f = open("/usr/local/libint/2.3.0-beta.3/share/libint/2.3.0-beta.3/basis/proj.g94", "w")
    #f.write(bs)
    #f.close()
    return bs, N_ao


    
def compute_intermediate_overlap_cell(project_folder, new_basis, cell = [[0,0,0]], tpe = "overlap", center = np.array([0,0,0]), s_1 = False, grid = None):
    geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
    x0,y0,z0 = center
    #self.cc = np.array([np.arange(-30,31), np.zeros(61), np.zeros(61)]).T

    
    #I = latmat()
    #I.cc = cc
    
    p0, c0 = geometry.get_atoms([[0,0,0]])
    #print(p0, c0)
    atoms_0 = l2atoms(p0, c0)
    if grid is None:
        atoms_1 = [Atom(x0,y0,z0,79)]
    else:
        atoms_1 = []
        for i in np.arange(grid[0]):
            for j in np.arange(grid[1]):
                for k in np.arange(grid[2]):
                    r = np.array([float(i)/grid[0],float(j)/grid[1],float(k)/grid[2]])
                    x,y,z = np.dot(r, geometry.lattice)
                    atoms_1.append(Atom(x,y,z,79))
            
    #for i in atoms_0:
    #    i.atomic_number = 79
    
    p1, c1 = geometry.get_atoms(cell)
    atoms_0 = l2atoms(p1, c1)
    #print(atoms_0[0].number)
    #set/unset libint data path, read correct basis
    opath = os.environ["LIBINT_DATA_PATH"]
    
    os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC"
    
    f = open(project_folder + "XDEC/inter_basis.g94", "w")
    f.write(new_basis)
    f.close()
    
    p = py_oneptoper.PyOnePtOper()
    p.set_basis("inter_basis")
    
        
    #atoms_0 = [Atom(0,0,0,1), Atom(2,0,0,1)]

    p.set_atoms(atoms_0, atoms_1)
    p.init_engine(tpe, p.max_nprim(), p.max_l())
    #print(p.calc_mpole()
    p.init_engine("emultipole2", p.max_nprim(), p.max_l())
    
    mpole = p.calc_mpole()

    mpole = np.array(mpole[0]).reshape(mpole[1])

 
    sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole #p.calc_mpole()
    
    #COMPUTE <a0,a0>
    s1 = 0
    if s_1:
    
        p.set_atoms(atoms_1, atoms_1)
        p.init_engine(tpe, p.max_nprim(), p.max_l())
        #print(p.calc_mpole()
        p.init_engine("emultipole2", p.max_nprim(), p.max_l())
        
        mpole = p.calc_mpole()
    
        mpole = np.array(mpole[0]).reshape(mpole[1])
    
    
        s1,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole #p.calc_mpole()
    
    os.environ["LIBINT_DATA_PATH"] = opath #restore original path
    return sm, s1
    
def compute_orbspread(project_folder):
    pass
    
    

def brute_force_orbspread(project_folder):
    geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
    cc = np.load(project_folder + "crystal_reference_coords.npy")
    C  = np.load(project_folder + "crystal_reference_state.npy")
    
    #cc*= -1
    #cc = np.load(project_folder + "lsdalton_reference_coords.npy")
    #C  = np.load(project_folder + "lsdalton_reference_state.npy")
    
    opath = os.environ["LIBINT_DATA_PATH"]
    os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC/"

    p = py_oneptoper.PyOnePtOper()
    p.set_basis("basis")
    
    p0, c0 = geometry.get_atoms([[0,0,0]])
    #print(p0, c0)
    atoms_0 = l2atoms(p0, c0)
    
    p.set_atoms(atoms_0, atoms_0)
    p.init_engine("emultipole2", p.max_nprim(), p.max_l())
    
    mpole = p.calc_mpole()

    mpole = np.array(mpole[0]).reshape(mpole[1])

 
    sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole #p.calc_mpole()
    
    Lmo1 = np.zeros(sm.shape, dtype = float)
    Lmo2 = np.zeros(sm.shape, dtype = float)
    Smo0 = np.zeros(sm.shape, dtype = float)
    
    
    X2mo = np.zeros(sm.shape, dtype = float)
    Y2mo = np.zeros(sm.shape, dtype = float)
    Z2mo = np.zeros(sm.shape, dtype = float)
    
    Xmo = np.zeros(sm.shape, dtype = float)
    Ymo = np.zeros(sm.shape, dtype = float)
    Zmo = np.zeros(sm.shape, dtype = float)
    
    #XYmo = np.zeros(sm.shape, dtype = float)
    #XZmo = np.zeros(sm.shape, dtype = float)
    #YZmo = np.zeros(sm.shape, dtype = float)
    
    
    
    M = np.zeros(sm.shape, dtype = float)
    st = ""
    #lim = np.abs(cc).max()
    
    for m in np.arange(len(cc)):
        #print(
        for l in np.arange(len(cc)):
            #if np.abs(cc[m]-cc[l]).max()<10*lim:
            #M = np.dot(cc[m]
            #L = np.dot(cc[l]
            pm, cm = geometry.get_atoms([cc[m]])
            pl, cl = geometry.get_atoms([cc[l]])
            
            #print(pm)
            atoms_1 = l2atoms(pm,cm)
            atoms_2 = l2atoms(pl,cl)
            
            #print(atoms_1)
            #print(atoms_2)

            p.set_atoms(atoms_1, atoms_2) # <m,l>
            
            mpole2 = p.calc_mpole()

            mpole2 = np.array(mpole2[0]).reshape(mpole2[1])



            sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()
            
            
            #if m==2:
            #print(pm)
            #print(pl)
            """
            cml = cc[m]-cc[l]
            if m==0:
                st += "Computing: (%i, %i, %i)*(%i, %i, %i)*(%i, %i, %i)\n" % (cc[l][0],cc[l][1],cc[l][2],
                                                                             cml[0],cml[1],cml[2],
                                                                             cc[l][0],cc[l][1],cc[l][2]) 
                Xmo += np.dot(C[l].T, np.dot(xm, C[l]))
            """
            #    Xmo += np.dot(xm, C[l])
            Xmo += np.dot(C[m].T, np.dot(xm, C[l]))
            Ymo += np.dot(C[m].T, np.dot(ym, C[l]))
            Zmo += np.dot(C[m].T, np.dot(zm, C[l]))
            
            #if np.all(cc[l]-cc[m]==0) :
            #if l==0 and m==0:
            
            #if m == 0: # and m== 0:
            #    #print(cc[m], cc[l])
            X2mo += np.dot(C[m].T, np.dot(x2m, C[l]))
            #    #X2mo += np.dot(x2m, C[l])
            Y2mo += np.dot(C[m].T, np.dot(y2m, C[l]))
            Z2mo += np.dot(C[m].T, np.dot(z2m, C[l]))
            
            #if m==0:
            #XYmo += np.dot(C[m].T, np.dot(np.dot(xm,ym), C[l]))
            #XZmo += np.dot(C[m].T, np.dot(np.dot(xm,zm), C[l]))
            #YZmo += np.dot(C[m].T, np.dot(np.dot(ym,zm), C[l]))
            
            ###
            ###
            
            #(cxc)0 = c(0-i)
            
            ###
            ###
            
            
            
            
            
            #Lao1 = x2m + y2m + z2m #+2*xym + 2*xzm + 2*yzm
            #Lao2 = xm + ym + zm
            
            #X = xm
            #if m==0:
            #Lmo1 += np.dot(C[m].T, np.dot(Lao1, C[l]))
            
            #Lmo2 += np.dot(C[m].T, np.dot(Lao2, C[l]))
            
            #Smo0 += np.dot(C[m].T, np.dot(sm, C[l]))
            
            
            #Xmo += np.dot(C[m].T, np.dot(X, C[l]))
            
            #M += Lmo1 - Lmo2**2
            
            #if np.any(Lmo1 - Lmo2**2 < 0):
            #    pass
            #else:
            #M[np.arange(len(M)), np.arange(len(M))] += np.diag(Lmo1) - np.diag(Lmo2)**2
            
            #xC = np.dot(xm, C[l])
            #M += np.dot(C[m].T, xC )
            
    #M =  Xmo # Lmo1 #+ (Smo0-2)*Lmo2**2
    #M = Lmo1 + (Smo0-2)*Lmo2**2
    #M = Lmo1 - Lmo2**2
    #print(st)
    M = X2mo  + Y2mo + Z2mo - (Xmo**2 + Ymo**2 + Zmo**2) # +2*(XYmo + XZmo + YZmo)) 
    #M = Xmo
    #M = Xmo**2
    
    #M = np.dot(Xmo,Ymo) + np.dot(Xmo,Zmo) + np.dot(Ymo, Zmo)
    os.environ["LIBINT_DATA_PATH"] = opath #restore original path
    #np.save("L_bf_ethylene_column.npy", M)
    return M
    
    
def debug_orbspread(project_folder, Xl):
    geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
    cc = np.load(project_folder + "crystal_reference_coords.npy")
    C  = np.load(project_folder + "crystal_reference_state.npy")
    
    sc = np.load(project_folder + "crystal_overlap_coords.npy")
    S  = np.load(project_folder + "crystal_overlap_matrix.npy")
    
    
    C_ = latmat()
    C_.npload(project_folder + "/crystal_reference_state.npy",
        project_folder + "/crystal_reference_coords.npy")
        
    #C_.cc = -1*C_.cc #why does this work?
    
    Sl = latmat()
    Sl.npload(project_folder + "/crystal_overlap_matrix.npy",
        project_folder + "/crystal_overlap_coords.npy")

    
    
    
    #cc*= -1
    #cc = np.load(project_folder + "lsdalton_reference_coords.npy")
    #C  = np.load(project_folder + "lsdalton_reference_state.npy")
    
    opath = os.environ["LIBINT_DATA_PATH"]
    os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC/"

    p = py_oneptoper.PyOnePtOper()
    p.set_basis("basis")
    
    p0, c0 = geometry.get_atoms([[0,0,0]])
    #print(p0, c0)
    atoms_0 = l2atoms(p0, c0)
    
    p.set_atoms(atoms_0, atoms_0)
    p.init_engine("emultipole2", p.max_nprim(), p.max_l())
    
    mpole = p.calc_mpole()

    mpole = np.array(mpole[0]).reshape(mpole[1])



    sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole #p.calc_mpole()

    
    Xmo = np.zeros(sm.shape, dtype = float)
    Ymo = np.zeros(sm.shape, dtype = float)
    Zmo = np.zeros(sm.shape, dtype = float)
    
    #XYmo = np.zeros(sm.shape, dtype = float)
    #XZmo = np.zeros(sm.shape, dtype = float)
    #YZmo = np.zeros(sm.shape, dtype = float)
    
    
    
    M = np.zeros(sm.shape, dtype = float)
    
    #lim = np.abs(cc).max()
    
    for m in np.arange(len(cc)):
        pm, cm = geometry.get_atoms([cc[m]])
        pl, cl = geometry.get_atoms([cc[0]])
        
        atoms_1 = l2atoms(pm,cm)
        atoms_2 = l2atoms(pl,cl)
        
        #print(atoms_1)
        #print(atoms_2)

        p.set_atoms(atoms_1, atoms_2) # <m,l>
        mpole2 = p.calc_mpole()

        mpole2 = np.array(mpole2[0]).reshape(mpole2[1])



        sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()
        
        Xmo += np.dot(C[m].T, np.dot(sm, C[0]))
        
        #print(np.sum(np.abs(xm-Xl.get(cc[m]))))
        #print(xm[0])
        #print(Xl.get(cc[m])[0])
        
        #print(np.sum(np.abs(C[m]-C_.get(cc[m]))))
        #print(C[m][0])
        #print(C_.get(cc[m])[0])
        
        print(np.sum(np.abs(C[0]-C_.get(cc[0]))))
        print(C[0][0])
        print(C_.get(cc[0])[0])
        
        
            

    M = Xmo # + Y2mo + Z2mo + (Xmo**2 + Ymo**2 + Zmo**2) # +2*(XYmo + XZmo + YZmo)) 

 
    os.environ["LIBINT_DATA_PATH"] = opath #restore original path
    #np.save("L_bf_ethylene_column.npy", M)
    return M
    
class carmom_lsdalton():
    def __init__(self, geometry, s_matrix):
        self.geometry = geometry
        #self.geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
        self.cc = s_matrix.coords #np.load(project_folder + "crystal_overlap_coords.npy")
        self.C  = s_matrix.blocks #np.load(project_folder + "crystal_overlap_matrix.npy" )
        
        
        #lsdalton_exe_path = os.environ["LSDALTON_EXE_PATH"]
        
        lsdalton_exe_path = "/Users/audunhansen/Dalton/PFM_integrals/lsdalton/build/lsdalton.x"
        
        #Nc = 30
        #self.cc = np.array([np.arange(-Nc,Nc+1), np.zeros(2*Nc+1), np.zeros(2*Nc+1)]).T
        #need a temp folder here
        project_folder = os.getcwd()
        tempfolder = project_folder + "/temp_pfm_integrals/"
        sp.call(["mkdir", project_folder + "/temp_pfm_integrals"])
        
        
        cshape = np.array([self.C.shape[0], self.C.shape[1]*2, self.C.shape[2]*2])
        #self.C.shape*1
        #cshape[1:]*=2
        
        bshape = np.array([self.C.shape[1], self.C.shape[2]])
        
        S = np.zeros(self.C.shape, dtype = float)
        
        S  = np.zeros(self.C.shape, dtype = float)
        X  = np.zeros(self.C.shape, dtype = float)
        Y  = np.zeros(self.C.shape, dtype = float)
        Z  = np.zeros(self.C.shape, dtype = float)
        X2 = np.zeros(self.C.shape, dtype = float)
        Y2 = np.zeros(self.C.shape, dtype = float)
        Z2 = np.zeros(self.C.shape, dtype = float)
    

        #need a temp folder here
        
        X3 = np.zeros(self.C.shape, dtype = float)
        Y3 = np.zeros(self.C.shape, dtype = float)
        Z3 = np.zeros(self.C.shape, dtype = float)
        
        X3 = np.zeros(self.C.shape, dtype = float)
        Y3 = np.zeros(self.C.shape, dtype = float)
        Z3 = np.zeros(self.C.shape, dtype = float)        
        
        X4 = np.zeros(self.C.shape, dtype = float)
        Y4 = np.zeros(self.C.shape, dtype = float)
        Z4 = np.zeros(self.C.shape, dtype = float)
        
        # Mixed terms
        
        XY = np.zeros(self.C.shape, dtype = float)
        XZ = np.zeros(self.C.shape, dtype = float)
        YZ = np.zeros(self.C.shape, dtype = float)

        X2Y = np.zeros(self.C.shape, dtype = float)
        X2Z = np.zeros(self.C.shape, dtype = float)
        Y2Z = np.zeros(self.C.shape, dtype = float)
        
        XY2 = np.zeros(self.C.shape, dtype = float)
        XZ2 = np.zeros(self.C.shape, dtype = float)
        YZ2 = np.zeros(self.C.shape, dtype = float)
        
        X2Y2 = np.zeros(self.C.shape, dtype = float)
        X2Z2 = np.zeros(self.C.shape, dtype = float)
        Y2Z2 = np.zeros(self.C.shape, dtype = float)
        
        
        YX = np.zeros(self.C.shape, dtype = float)
        Y2X= np.zeros(self.C.shape, dtype = float)
        ZX = np.zeros(self.C.shape, dtype = float)
        Z2X= np.zeros(self.C.shape, dtype = float)
        YX = np.zeros(self.C.shape, dtype = float)
        YX2= np.zeros(self.C.shape, dtype = float)
        ZY=  np.zeros(self.C.shape, dtype = float)
        ZX2= np.zeros(self.C.shape, dtype = float)
        ZY2= np.zeros(self.C.shape, dtype = float)
        ZY = np.zeros(self.C.shape, dtype = float)
        

        lsdalton_inp = """**GENERAL
.CM_IO
.NOGCBASIS
**INTEGRALS
*FMM
.SCREEN
 1e-10
.LMAX
 5
**INFO
.DEBUG_MPI_MEM
*END OF INPUT
"""
        f = open(tempfolder + "LSDALTON.INP", "w")
        f.write(lsdalton_inp)
        f.close()
        
        sorting = self.geometry.get_doublecell_basis_order()
        
        sorts = np.array([2,3,13,14,15,16,17,18,19,20,21])
        sorts = np.array([0,1,4,5,6,7,8,9,10,11,12])
        for i in np.arange(self.cc.shape[0]):
            m = self.cc[i]
            if np.all(m==np.array([0,0,0])):
                # Special case, avoid initializing LSDalton with overlapping atoms
                molecule_inp = self.geometry.get_multicell_lsdalton_input(cells = np.array([m, [1,0,0]]))
                
                f = open(tempfolder + "MOLECULE.INP", "w")
                f.write(molecule_inp)
                f.close()
                
                sp.call([lsdalton_exe_path], cwd = tempfolder, shell = False)
                
                X[i]  = self.read_fortranfile(tempfolder + "X_ao.npy")[:bshape[0],:bshape[1]].T #[sorts][:,sorts]
                Y[i]  = self.read_fortranfile(tempfolder + "Y_ao.npy")[:bshape[0],:bshape[1]].T
                Z[i]  = self.read_fortranfile(tempfolder + "Z_ao.npy")[:bshape[0],:bshape[1]].T
                
                X2[i]  = self.read_fortranfile(tempfolder + "X2_ao.npy")[:bshape[0],:bshape[1]].T
                Y2[i]  = self.read_fortranfile(tempfolder + "Y2_ao.npy")[:bshape[0],:bshape[1]].T
                Z2[i]  = self.read_fortranfile(tempfolder + "Z2_ao.npy")[:bshape[0],:bshape[1]].T
                
                X3[i]  = self.read_fortranfile(tempfolder + "X3_ao.npy")[:bshape[0],:bshape[1]].T
                Y3[i]  = self.read_fortranfile(tempfolder + "Y3_ao.npy")[:bshape[0],:bshape[1]].T
                Z3[i]  = self.read_fortranfile(tempfolder + "Z3_ao.npy")[:bshape[0],:bshape[1]].T
                
                X4[i]  = self.read_fortranfile(tempfolder + "X4_ao.npy")[:bshape[0],:bshape[1]].T
                Y4[i]  = self.read_fortranfile(tempfolder + "Y4_ao.npy")[:bshape[0],:bshape[1]].T
                Z4[i]  = self.read_fortranfile(tempfolder + "Z4_ao.npy")[:bshape[0],:bshape[1]].T
                
                XY[i]  = self.read_fortranfile(tempfolder + "XY_ao.npy")[:bshape[0],:bshape[1]].T
                XZ[i]  = self.read_fortranfile(tempfolder + "XZ_ao.npy")[:bshape[0],:bshape[1]].T
                YZ[i]  = self.read_fortranfile(tempfolder + "YZ_ao.npy")[:bshape[0],:bshape[1]].T            
                
                X2Y[i]  = self.read_fortranfile(tempfolder + "X2Y_ao.npy")[:bshape[0],:bshape[1]].T
                X2Z[i]  = self.read_fortranfile(tempfolder + "X2Z_ao.npy")[:bshape[0],:bshape[1]].T
                Y2Z[i]  = self.read_fortranfile(tempfolder + "Y2Z_ao.npy")[:bshape[0],:bshape[1]].T            
                
                
                XY2[i]  = self.read_fortranfile(tempfolder + "XY2_ao.npy")[:bshape[0],:bshape[1]].T
                XZ2[i]  = self.read_fortranfile(tempfolder + "XZ2_ao.npy")[:bshape[0],:bshape[1]].T
                YZ2[i]  = self.read_fortranfile(tempfolder + "YZ2_ao.npy")[:bshape[0],:bshape[1]].T            
                
                
                X2Y2[i]  = self.read_fortranfile(tempfolder + "X2Y2_ao.npy")[:bshape[0],:bshape[1]].T
                X2Z2[i]  = self.read_fortranfile(tempfolder + "X2Z2_ao.npy")[:bshape[0],:bshape[1]].T
                Y2Z2[i]  = self.read_fortranfile(tempfolder + "Y2Z2_ao.npy")[:bshape[0],:bshape[1]].T
                
                YX[i] =self.read_fortranfile(tempfolder + "XY_ao.npy")[:bshape[0],:bshape[1]].T
                Y2X[i]=self.read_fortranfile(tempfolder + "XY2_ao.npy")[:bshape[0],:bshape[1]].T
                ZX[i] =self.read_fortranfile(tempfolder + "XZ_ao.npy")[:bshape[0],:bshape[1]].T
                Z2X[i]=self.read_fortranfile(tempfolder + "XZ2_ao.npy")[:bshape[0],:bshape[1]].T
                YX[i] =self.read_fortranfile(tempfolder + "XY_ao.npy")[:bshape[0],:bshape[1]].T
                YX2[i]=self.read_fortranfile(tempfolder + "X2Y_ao.npy")[:bshape[0],:bshape[1]].T
                ZY[i]= self.read_fortranfile(tempfolder + "YZ_ao.npy")[:bshape[0],:bshape[1]].T
                ZX2[i]=self.read_fortranfile(tempfolder + "X2Z_ao.npy")[:bshape[0],:bshape[1]].T
                ZY2[i]=self.read_fortranfile(tempfolder + "Y2Z_ao.npy")[:bshape[0],:bshape[1]].T
                ZY[i] =self.read_fortranfile(tempfolder + "YZ_ao.npy")[:bshape[0],:bshape[1]].T
                
                
            else:
                molecule_inp = self.geometry.get_multicell_lsdalton_input(cells = np.array([m, [0,0,0]]))
                #print(molecule_inp)
                
                f = open(tempfolder + "MOLECULE.INP", "w")
                f.write(molecule_inp)
                f.close()
                
                sp.call([lsdalton_exe_path], cwd = tempfolder, shell = False)
                
                X[i]  = self.read_fortranfile(tempfolder + "X_ao.npy")[:bshape[0],bshape[1]:].T #[sorts][:,sorts]
                Y[i]  = self.read_fortranfile(tempfolder + "Y_ao.npy")[:bshape[0],bshape[1]:].T
                Z[i]  = self.read_fortranfile(tempfolder + "Z_ao.npy")[:bshape[0],bshape[1]:].T
                
                X2[i]  = self.read_fortranfile(tempfolder + "X2_ao.npy")[:bshape[0],bshape[1]:].T
                Y2[i]  = self.read_fortranfile(tempfolder + "Y2_ao.npy")[:bshape[0],bshape[1]:].T
                Z2[i]  = self.read_fortranfile(tempfolder + "Z2_ao.npy")[:bshape[0],bshape[1]:].T
                
                X3[i]  = self.read_fortranfile(tempfolder + "X3_ao.npy")[:bshape[0],bshape[1]:].T
                Y3[i]  = self.read_fortranfile(tempfolder + "Y3_ao.npy")[:bshape[0],bshape[1]:].T
                Z3[i]  = self.read_fortranfile(tempfolder + "Z3_ao.npy")[:bshape[0],bshape[1]:].T
                
                X4[i]  = self.read_fortranfile(tempfolder + "X4_ao.npy")[:bshape[0],bshape[1]:].T
                Y4[i]  = self.read_fortranfile(tempfolder + "Y4_ao.npy")[:bshape[0],bshape[1]:].T
                Z4[i]  = self.read_fortranfile(tempfolder + "Z4_ao.npy")[:bshape[0],bshape[1]:].T
                
                XY[i]  = self.read_fortranfile(tempfolder + "XY_ao.npy")[:bshape[0],bshape[1]:].T
                XZ[i]  = self.read_fortranfile(tempfolder + "XZ_ao.npy")[:bshape[0],bshape[1]:].T
                YZ[i]  = self.read_fortranfile(tempfolder + "YZ_ao.npy")[:bshape[0],bshape[1]:].T            
                
                X2Y[i]  = self.read_fortranfile(tempfolder + "X2Y_ao.npy")[:bshape[0],bshape[1]:].T
                X2Z[i]  = self.read_fortranfile(tempfolder + "X2Z_ao.npy")[:bshape[0],bshape[1]:].T
                Y2Z[i]  = self.read_fortranfile(tempfolder + "Y2Z_ao.npy")[:bshape[0],bshape[1]:].T            
                
                
                XY2[i]  = self.read_fortranfile(tempfolder + "XY2_ao.npy")[:bshape[0],bshape[1]:].T
                XZ2[i]  = self.read_fortranfile(tempfolder + "XZ2_ao.npy")[:bshape[0],bshape[1]:].T
                YZ2[i]  = self.read_fortranfile(tempfolder + "YZ2_ao.npy")[:bshape[0],bshape[1]:].T            
                
                
                X2Y2[i]  = self.read_fortranfile(tempfolder + "X2Y2_ao.npy")[:bshape[0],bshape[1]:].T
                X2Z2[i]  = self.read_fortranfile(tempfolder + "X2Z2_ao.npy")[:bshape[0],bshape[1]:].T
                Y2Z2[i]  = self.read_fortranfile(tempfolder + "Y2Z2_ao.npy")[:bshape[0],bshape[1]:].T            
                
                YX[i] =self.read_fortranfile(tempfolder + "XY_ao.npy")[:bshape[0],:bshape[1]].T
                Y2X[i]=self.read_fortranfile(tempfolder + "XY2_ao.npy")[:bshape[0],:bshape[1]].T
                ZX[i] =self.read_fortranfile(tempfolder + "XZ_ao.npy")[:bshape[0],:bshape[1]].T
                Z2X[i]=self.read_fortranfile(tempfolder + "XZ2_ao.npy")[:bshape[0],:bshape[1]].T
                YX[i] =self.read_fortranfile(tempfolder + "XY_ao.npy")[:bshape[0],:bshape[1]].T
                YX2[i]=self.read_fortranfile(tempfolder + "X2Y_ao.npy")[:bshape[0],:bshape[1]].T
                ZY[i]= self.read_fortranfile(tempfolder + "YZ_ao.npy")[:bshape[0],:bshape[1]].T
                ZX2[i]=self.read_fortranfile(tempfolder + "X2Z_ao.npy")[:bshape[0],:bshape[1]].T
                ZY2[i]=self.read_fortranfile(tempfolder + "Y2Z_ao.npy")[:bshape[0],:bshape[1]].T
                ZY[i] =self.read_fortranfile(tempfolder + "YZ_ao.npy")[:bshape[0],:bshape[1]].T
                
                
                #self.X.set(m, X)
                #print(x.shape)
                
                #pm, cm = self.geometry.get_atoms([m])
    
                #atoms_1 = l2atoms(pm,cm)
    
                #print(atoms_1)
    
    
                #p.set_atoms(atoms_1, atoms_0)
    
                #mpole2 = p.calc_mpole()
    
                #mpole2 = np.array(mpole2[0]).reshape(mpole2[1])
                
                """
    
                sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()
                
                
                
                S[i] = sm
                X[i] = xm
                Y[i] = ym
                Z[i] = zm
                X2[i]= x2m
                Y2[i]= y2m
                Z2[i]= z2m
                XZ[i] =xzm
                XY[i] =xym
                YZ[i] =yzm
                
                
                #self.X.set(m, xm)
                #self.Y.set(m, ym)
                #self.Z.set(m, zm)
                
                #self.X2.set(m, x2m)
                #self.Y2.set(m, y2m)
                #self.Z2.set(m, z2m)
                
                #self.XY.set(m, xym)
                #self.XZ.set(m, xzm)
                #self.YZ.set(m, yzm)
                """
        
        self.S = tmat(coords=self.cc, blocks=S, blockshape=S[0].shape)
        self.X = tmat(coords=self.cc, blocks=X, blockshape=X[0].shape)
        self.Y = tmat(coords=self.cc, blocks=Y, blockshape=Y[0].shape)
        self.Z = tmat(coords=self.cc, blocks=Z, blockshape=Z[0].shape)
        
        self.X2 = tmat(coords=self.cc, blocks=X2, blockshape=X[0].shape)
        self.Y2 = tmat(coords=self.cc, blocks=Y2, blockshape=X[0].shape)
        self.Z2 = tmat(coords=self.cc, blocks=Z2, blockshape=X[0].shape)
        
        self.X3 = tmat(coords=self.cc, blocks=X3, blockshape=X[0].shape)
        self.Y3 = tmat(coords=self.cc, blocks=Y3, blockshape=X[0].shape)
        self.Z3 = tmat(coords=self.cc, blocks=Z3, blockshape=X[0].shape)
        
        self.X4 = tmat(coords=self.cc, blocks=X4, blockshape=X[0].shape)
        self.Y4 = tmat(coords=self.cc, blocks=Y4, blockshape=X[0].shape)
        self.Z4 = tmat(coords=self.cc, blocks=Z4, blockshape=X[0].shape)
        
        self.XY = tmat(coords=self.cc, blocks=XY, blockshape=X[0].shape)
        self.XZ = tmat(coords=self.cc, blocks=XZ, blockshape=X[0].shape)
        self.YZ = tmat(coords=self.cc, blocks=YZ, blockshape=X[0].shape)
        
        self.X2Y = tmat(coords=self.cc, blocks=X2Y, blockshape=X[0].shape)
        self.X2Z = tmat(coords=self.cc, blocks=X2Z, blockshape=X[0].shape)
        self.Y2Z = tmat(coords=self.cc, blocks=Y2Z, blockshape=X[0].shape)
        
        self.XY2 = tmat(coords=self.cc, blocks=XY2, blockshape=X[0].shape)
        self.XZ2 = tmat(coords=self.cc, blocks=XZ2, blockshape=X[0].shape)
        self.YZ2 = tmat(coords=self.cc, blocks=YZ2, blockshape=X[0].shape)
        
        self.X2Y2 = tmat(coords=self.cc, blocks=X2Y2, blockshape=X[0].shape)
        self.X2Z2 = tmat(coords=self.cc, blocks=X2Z2, blockshape=X[0].shape)
        self.Y2Z2 = tmat(coords=self.cc, blocks=Y2Z2, blockshape=X[0].shape)
        
        self.YX = tmat(coords=self.cc, blocks=YX , blockshape=X[0].shape)
        self.Y2X= tmat(coords=self.cc, blocks=Y2X, blockshape=X[0].shape)
        self.ZX = tmat(coords=self.cc, blocks=ZX , blockshape=X[0].shape)
        self.Z2X= tmat(coords=self.cc, blocks=Z2X, blockshape=X[0].shape)
        self.YX = tmat(coords=self.cc, blocks=YX , blockshape=X[0].shape)
        self.YX2= tmat(coords=self.cc, blocks=YX2, blockshape=X[0].shape)
        self.ZY=  tmat(coords=self.cc, blocks=ZY , blockshape=X[0].shape)
        self.ZX2= tmat(coords=self.cc, blocks=ZX2, blockshape=X[0].shape)
        self.ZY2= tmat(coords=self.cc, blocks=ZY2, blockshape=X[0].shape)
        self.ZY = tmat(coords=self.cc, blocks=ZY , blockshape=X[0].shape)
        
        
        #if opath != None:
        #    os.environ["LIBINT_DATA_PATH"] = opath #restore original path
    def read_fortranfile(self,filename):
        #reads output from lsdalton, for the integrals
        f = open(filename, "r").read()
        return np.array([[literal_eval(i) for i in e.split()] for e in f.split("\n")[:-1]])
        

class carmom():
    def __init__(self, project_folder):
        self.geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
        self.cc = np.load(project_folder + "crystal_reference_coords.npy")
        self.C  = np.load(project_folder + "crystal_reference_state.npy")
        
        
        #Nc = 30
        #self.cc = np.array([np.arange(-Nc,Nc+1), np.zeros(2*Nc+1), np.zeros(2*Nc+1)]).T
        
        
        self.S = latmat()
        self.S.cc = self.cc
        
        self.X = latmat()
        self.X.cc = self.cc
        
        
        self.Y = latmat()
        self.Y.cc = self.cc
        
        self.Z = latmat()
        self.Z.cc = self.cc
        
        self.X2 = latmat()
        self.X2.cc = self.cc
        
        self.Y2 = latmat()
        self.Y2.cc = self.cc
        
        self.Z2 = latmat()
        self.Z2.cc = self.cc
        
        self.XY = latmat()
        self.XY.cc = self.cc
        
        self.XZ = latmat()
        self.XZ.cc = self.cc
        
        self.YZ = latmat()
        self.YZ.cc = self.cc
        
        p0, c0 = self.geometry.get_atoms([[0,0,0]])
        #print(p0, c0)
        atoms_0 = l2atoms(p0, c0)
        
        
        #set/unset libint data path, read correct basis
        try:
            opath = os.environ["LIBINT_DATA_PATH"]
        except:
            opath = None
        os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC/"

        p = py_oneptoper.PyOnePtOper()
        p.set_basis("basis")
        
            
        #atoms_0 = [Atom(0,0,0,1), Atom(2,0,0,1)]

        p.set_atoms(atoms_0, atoms_0)
        p.init_engine("emultipole2", p.max_nprim(), p.max_l())
        #print(p.calc_mpole())
        
        
        
        for m in self.cc:
            pm, cm = self.geometry.get_atoms([m])

            atoms_1 = l2atoms(pm,cm)

            p.set_atoms(atoms_0, atoms_1)

            mpole2 = p.calc_mpole()

            mpole2 = np.array(mpole2[0]).reshape(mpole2[1])

            sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()
            
            
            
            self.S.set(m, sm)
            
            self.X.set(m, xm)
            self.Y.set(m, ym)
            self.Z.set(m, zm)
            
            self.X2.set(m, x2m)
            self.Y2.set(m, y2m)
            self.Z2.set(m, z2m)
            
            self.XY.set(m, xym)
            self.XZ.set(m, xzm)
            self.YZ.set(m, yzm)
            
        
        if opath != None:
            os.environ["LIBINT_DATA_PATH"] = opath #restore original path

class carmom_tmat():
    def __init__(self, geometry, s_matrix):
        self.geometry = geometry #pr.prism(project_folder + "XDEC/MOLECULE.INP")
        
        
        #s_matrix = np.load(project_folder + "S_crystal.npy")
        #self.cc = np.load(project_folder + "crystal_overlap_coords.npy")
        #self.C  = np.load(project_folder + "crystal_overlap_matrix.npy" )
        #[self.cc, self.C] = [s_matrix[0], s_matrix[1]]
        
        self.cc = s_matrix.coords
        self.C = s_matrix.blocks
        
        #Nc = 30
        #self.cc = np.array([np.arange(-Nc,Nc+1), np.zeros(2*Nc+1), np.zeros(2*Nc+1)]).T
        
        
        S = np.zeros(self.C.shape, dtype = float)
        
        S  = np.zeros(self.C.shape, dtype = float)
        X  = np.zeros(self.C.shape, dtype = float)
        Y  = np.zeros(self.C.shape, dtype = float)
        Z  = np.zeros(self.C.shape, dtype = float)
        X2 = np.zeros(self.C.shape, dtype = float)
        Y2 = np.zeros(self.C.shape, dtype = float)
        Z2 = np.zeros(self.C.shape, dtype = float)
        XY = np.zeros(self.C.shape, dtype = float)
        XZ = np.zeros(self.C.shape, dtype = float)
        YZ = np.zeros(self.C.shape, dtype = float)
        
        
        p0, c0 = self.geometry.get_atoms([[0,0,0]])
        #print(p0, c0)
        atoms_0 = l2atoms(p0, c0)
        
        
        #set/unset libint data path, read correct basis
        try:
            opath = os.environ["LIBINT_DATA_PATH"]
        except:
            opath = None
            
        # temp storage of basis
        basisfile = open("post_process_basis.g94", "w")
        
        basisfile.write(geometry.get_libint_basis())
        
        basisfile.close()
        
            
        os.environ["LIBINT_DATA_PATH"] = os.getcwd() # + "/post_process_basis.g94"

        p = py_oneptoper.PyOnePtOper()
        p.set_basis("post_process_basis")
        
            
        #atoms_0 = [Atom(0,0,0,1), Atom(2,0,0,1)]

        p.set_atoms(atoms_0, atoms_0)
        p.init_engine("emultipole2", p.max_nprim(), p.max_l())
        #print(p.calc_mpole())
        
        
        
        for i in np.arange(self.cc.shape[0]):
            m = self.cc[i]
            pm, cm = self.geometry.get_atoms([m])

            atoms_1 = l2atoms(pm,cm)

            p.set_atoms(atoms_0, atoms_1)

            mpole2 = p.calc_mpole()

            mpole2 = np.array(mpole2[0]).reshape(mpole2[1])

            sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()
            
            
            
            S[i] = sm
            X[i] = xm
            Y[i] = ym
            Z[i] = zm
            X2[i]= x2m
            Y2[i]= y2m
            Z2[i]= z2m
            XZ[i] =xzm
            XY[i] =xym
            YZ[i] =yzm
            
            
            #self.X.set(m, xm)
            #self.Y.set(m, ym)
            #self.Z.set(m, zm)
            
            #self.X2.set(m, x2m)
            #self.Y2.set(m, y2m)
            #self.Z2.set(m, z2m)
            
            #self.XY.set(m, xym)
            #self.XZ.set(m, xzm)
            #self.YZ.set(m, yzm)
        
        self.S = tmat(coords=self.cc, blocks=S, blockshape=X[0].shape)
        self.X = tmat(coords=self.cc, blocks=X, blockshape=X[0].shape)
        self.Y = tmat(coords=self.cc, blocks=Y, blockshape=X[0].shape)
        self.Z = tmat(coords=self.cc, blocks=Z, blockshape=X[0].shape)
        
        self.X2 = tmat(coords=self.cc, blocks=X2, blockshape=X[0].shape)
        self.Y2 = tmat(coords=self.cc, blocks=Y2, blockshape=X[0].shape)
        self.Z2 = tmat(coords=self.cc, blocks=Z2, blockshape=X[0].shape)
        
        self.XY = tmat(coords=self.cc, blocks=XY, blockshape=X[0].shape)
        self.XZ = tmat(coords=self.cc, blocks=XZ, blockshape=X[0].shape)
        self.YZ = tmat(coords=self.cc, blocks=YZ, blockshape=X[0].shape)
        
        
        
        
        if opath != None:
            os.environ["LIBINT_DATA_PATH"] = opath #restore original path

class fock_tmat():
    # build periodic hamiltonian matrix for a given input
    def __init__(self, project_folder):
        self.geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
        self.cc = np.load(project_folder + "crystal_overlap_coords.npy")
        self.C  = np.load(project_folder + "crystal_overlap_matrix.npy" )
        
        T  = np.zeros(self.C.shape, dtype = float) #kinetic
        Z  = np.zeros(self.C.shape, dtype = float) #e-n
        C  = np.zeros(self.C.shape, dtype = float) #direct
        X  = np.zeros(self.C.shape, dtype = float) #exchange

        
        
        p0, c0 = self.geometry.get_atoms([[0,0,0]])
        #print(p0, c0)
        atoms_0 = l2atoms(p0, c0)
        
        
        #set/unset libint data path, read correct basis
        try:
            opath = os.environ["LIBINT_DATA_PATH"]
        except:
            opath = None
        os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC/"

        
        
            
        #atoms_0 = [Atom(0,0,0,1), Atom(2,0,0,1)]
        p_t = py_oneptoper.PyOnePtOper()
        p_t.set_basis("basis")
        p_t.set_atoms(atoms_0, atoms_0)
        p_t.init_engine("kinetic", p_t.max_nprim(), p_t.max_l())

        p_z = py_oneptoper.PyOnePtOper()
        p_z.set_basis("basis")
        p_z.set_atoms(atoms_0, atoms_0)
        p_z.init_engine("nuclear", p_z.max_nprim(), p_z.max_l())
        
        
        #print(p.calc_mpole())
        
        
        
        for i in np.arange(self.cc.shape[0]):
            m = self.cc[i]
            pm, cm = self.geometry.get_atoms([m])

            atoms_1 = l2atoms(pm,cm)

            p_t.set_atoms(atoms_0, atoms_1)

            p_z.set_atoms(atoms_0, atoms_1)

            tm = p_t.calc()
            zm = p_z.calc()
            #mpole2 = np.array(mpole2[0]).reshape(mpole2[1])

            #sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()
            
            
            
            T[i] = tm
            Z[i] = zm

        self.T = tmat(coords=self.cc, blocks=T, blockshape=T[0].shape)
        self.Z = tmat(coords=self.cc, blocks=Z, blockshape=Z[0].shape)

        
        
        
        
        if opath != None:
            os.environ["LIBINT_DATA_PATH"] = opath #restore original path

def brute_force_wcenter(project_folder):
    geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
    cc = np.load(project_folder + "crystal_reference_coords.npy")
    C  = np.load(project_folder + "crystal_reference_state.npy")
    
    cc*= -1
    #cc = np.load(project_folder + "lsdalton_reference_coords.npy")
    #C  = np.load(project_folder + "lsdalton_reference_state.npy")
    
    
    
    opath = os.environ["LIBINT_DATA_PATH"]
    os.environ["LIBINT_DATA_PATH"] = project_folder + "XDEC/"

    p = py_oneptoper.PyOnePtOper()
    p.set_basis("basis")
    
    p0, c0 = geometry.get_atoms([[0,0,0]])
    #print(p0, c0)
    atoms_0 = l2atoms(p0, c0)
    
    p.set_atoms(atoms_0, atoms_0)
    p.init_engine("emultipole2", p.max_nprim(), p.max_l())
    
    mpole = p.calc_mpole()

    mpole = np.array(mpole[0]).reshape(mpole[1])

 
    sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole #p.calc_mpole()
    
    Lmo1 = np.zeros(sm.shape, dtype = float)
    Lmo2 = np.zeros(sm.shape, dtype = float)
    Smo0 = np.zeros(sm.shape, dtype = float)
    
    
    X2mo = np.zeros(sm.shape, dtype = float)
    Y2mo = np.zeros(sm.shape, dtype = float)
    Z2mo = np.zeros(sm.shape, dtype = float)
    
    Xmo = np.zeros(sm.shape, dtype = float)
    Ymo = np.zeros(sm.shape, dtype = float)
    Zmo = np.zeros(sm.shape, dtype = float)
    
    #XYmo = np.zeros(sm.shape, dtype = float)
    #XZmo = np.zeros(sm.shape, dtype = float)
    #YZmo = np.zeros(sm.shape, dtype = float)
    
    
    
    M = np.zeros(sm.shape, dtype = float)
    st = ""
    #lim = np.abs(cc).max()
    
    for m in np.arange(len(cc)):
        #print(
        for l in np.arange(len(cc)):
            #if np.abs(cc[m]-cc[l]).max()<10*lim:
            #M = np.dot(cc[m]
            #L = np.dot(cc[l]
            pm, cm = geometry.get_atoms([cc[m]])
            pl, cl = geometry.get_atoms([cc[l]])
            
            #print(pm)
            atoms_1 = l2atoms(pm,cm)
            atoms_2 = l2atoms(pl,cl)
            
            #print(atoms_1)
            #print(atoms_2)

            p.set_atoms(atoms_1, atoms_2) # <m,l>
            
            mpole2 = p.calc_mpole()

            mpole2 = np.array(mpole2[0]).reshape(mpole2[1])



            sm,xm,ym,zm,x2m,xym,xzm,y2m,yzm,z2m = mpole2 #p.calc_mpole()
            
            
            #if m==2:
            #print(pm)
            #print(pl)
            """
            cml = cc[m]-cc[l]
            if m==0:
                st += "Computing: (%i, %i, %i)*(%i, %i, %i)*(%i, %i, %i)\n" % (cc[l][0],cc[l][1],cc[l][2],
                                                                             cml[0],cml[1],cml[2],
                                                                             cc[l][0],cc[l][1],cc[l][2]) 
                Xmo += np.dot(C[l].T, np.dot(xm, C[l]))
            """
            #    Xmo += np.dot(xm, C[l])
            #if np.all(cc[m] == np.array([0,0,0])):
            Xmo += np.dot(C[m].T, np.dot(xm, C[l]))
            Ymo += np.dot(C[m].T, np.dot(ym, C[l]))
            Zmo += np.dot(C[m].T, np.dot(zm, C[l]))

    #M = np.dot(Xmo,Ymo) + np.dot(Xmo,Zmo) + np.dot(Ymo, Zmo)
    os.environ["LIBINT_DATA_PATH"] = opath #restore original path
    #np.save("L_bf_ethylene_column.npy", M)
    return np.diag(Xmo), np.diag(Ymo), np.diag(Zmo)
