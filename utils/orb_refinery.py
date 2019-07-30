#!/usr/bin/env python

"""
#####################################################
##  ,--.   ,--.      ,------.  ,------. ,-----.    ##
##   \  `.'  /,-----.|  .-.  \ |  .---''  .--./    ##
##    .'    \ '-----'|  |  \  :|  `--, |  |        ## 
##   /  .'.  \       |  '--'  /|  `---.'  '--'\    ##
##  '--'   '--'      `------Wannier Refinery--'    ##
#####################################################
"""


import os
import sys
import argparse
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt

from ast import literal_eval

from time import time
from copy import deepcopy
from scipy.linalg import expm

import toeplitz
import prism as pr
import post_process as pp
import wannier_parse as wp
import annealing_tools as at
import visuals as vis
#import modified_pm as mpm
import orb_functions as of

from toeplitz import tmat, L2norm




def summarize_orbitals(spreads_occ, spreads_virt, wcenters):
    """
    Print a summary of spreads and centers
    """
    print("Orb.nr.   <x>          <y>           <z>              Spread  ")
    
    for i in np.arange(spreads_occ.shape[0]):
        print("%.2i (o)   %+.5e  %+.5e  %+.5e     %.5e" % (i, wcenters[0,i], wcenters[1,i], wcenters[2,i], spreads_occ[i]))
    for i in np.arange(spreads_virt.shape[0]):
        print("%.2i (v)   %+.5e  %+.5e  %+.5e     %.5e" % (i+Nocc, wcenters[0,i+Nocc], wcenters[1,i+Nocc], wcenters[2,i+Nocc], spreads_virt[i]))  

    print("-----------------------------------------------------------------")
    print("Least local occupied:", spreads_occ.max())
    print("Least local virtual :", spreads_virt.max())
    print("-----------------------------------------------------------------")
    print("\n"*3)

def sphereshell(r0, r1, N):
    """
    # returns N random cartesian 3D coordinates within the interval radius r0 - r1
    # asset to the shell monte carlo integration
    """
    phi   = np.random.uniform(-np.pi,np.pi,N)
    theta = np.random.uniform(-np.pi/2.0, np.pi/2.0, N)
    r     = np.random.uniform(r0,r1,N)
    # Transform to cartesian coordinates
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def radplot_orb(pf, orb_number, blocks, coords, wf_x, title = "Radial distribution of orbital", r_max = 40, ylim = None):
    # Compute a plot of the radial distribution of orbital 
    N = 5000 #10 000number of MC samples per bin
    Nbins = 30 #200
    p = orb_number
    x0,y0,z0 = wf_x[p] #*10000
    
    v = vis.lambda_basis(pf + "/XDEC/MOLECULE.INP", 
                 C = blocks,
                 Cc= coords)

    plt.figure(1, figsize = (10,8))

    Nr = 3
    for j in np.arange(Nr):
        r = np.linspace(0,r_max,Nbins)
        R1 = np.zeros(r.shape[0]-1)
        Z1 = np.zeros(r.shape[0]-1)
        Z2 = np.zeros(r.shape[0]-1)
        dZ1 = np.zeros(r.shape[0]-1)
        dZ2 = np.zeros(r.shape[0]-1)
        for i in np.arange(r.shape[0]-1):
            x,y,z = sphereshell(r[i], r[i+1], N)

            V  = 1 #(r[i+1]**3 - r[i]**3)*4*np.pi/3.0

            #print(np.sqrt(x**2 + y**2 + z**2).min(), np.sqrt(x**2 + y**2 + z**2).max())
            #print(V)

            #zz = orbat(x,y,z) #**2
            zz = v.orb_1d_at(x-x0,y-y0,z-z0, p = p, thresh = 10e-8) #**2

            #print(zz.shape)
            Z1[i] = zz.mean()*V    
            Z2[i] = np.abs(zz).mean()*V
            dZ1[i] = zz.std()*V    
            dZ2[i] = np.abs(zz).std()*V
            R1[i] = np.sqrt(x**2 + y**2 + z**2).mean()



        plt.plot(R1,Z1, "o-", color = (0.5*(1+j/Nr), .1,.1), alpha = 0.4)
        plt.plot(R1,Z2, "*-", color = (.1,.1,0.5*(1+j/Nr)), alpha = 0.4)

        plt.plot(R1,np.abs(dZ1), ".-", color = (0.5*j/Nr, .1,.1), alpha = 0.4)
        plt.plot(R1,np.abs(dZ2), "v-", color = (.1,.1,0.5*j/Nr), alpha = 0.4)
        #print("Sum:", np.sum(Z1))





        #plt.plot(r,Z, ".--")
    plt.legend(["$\\phi$", "$\\vert \\phi \\vert$", "$\\sigma_{\\phi}$", "$\\sigma_{\\vert \\phi \\vert}$"])
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Orbital content")
    if ylim!=None:
        plt.ylim = ylim
    plt.savefig(title, bbox_inches='tight')

def setup_PAO(project_folder, n_fourier):
    '''
    Given a project folder, compute coefficents of projected 
    atomic orbitals (PAOs).
    
    project_folder     : Name of the project folder.
    n_fourier          : A tuplet that defines the number
                         of Fourier points in each of the
                         periodic dimensions.    
    '''
    
    D = tmat()
    D.load_old(project_folder + "/crystal_density_matrix.npy",
               project_folder + "/crystal_density_coords.npy") #CHANGE TO LSDALTON
    print('WARNING! The density matrix provided by ' + \
          'Crystal may have quite large elements ' + \
          'set to zero. It is therefore better to ' + \
          'compute the density matrix using the occupied ' + \
          'Wannier coefficients.')
    
    S = tmat()
    S.load_old(project_folder + "/crystal_overlap_matrix.npy",
               project_folder + "/crystal_overlap_coords.npy")
    
    return compute_PAO(D, S, n_fourier)


def compute_PAO(ao_overlaps, density_matrix, n_fourier):
    '''
    Using a given density matrix and a corresponding 
    AO overlap matrix, compute coefficents of projected 
    atomic orbitals (PAOs).
    
    WARNING! It is important that the given Fourier 
    parameter 'n_fourier' contains the number of Fourier
    points used when obtaining the density matrix.
    
    Author             : Audun Skau Hansen. 
                         Modified by Gustav Baardsen.
    
    Input:
    
    density_matrix     : Density matrix given as a 
                         toeplitz.tmat object. 
    ao_overlaps        : Atomic orbital overlap matrix,
                         given as a toeplitz.tmat
                         object.
    n_fourier          : A tuplet that defines the number
                         of Fourier points in each of the
                         periodic dimensions.
    
    Output:
    
    The PAO elements are given as C_{AO, PAO}^{L}, 
    where a row corresponds to an AO and a column to a PAO.
    '''
    m = 'Error. A tuplet "n_fourier" containing one ' + \
        'Fourier grid number for each of the periodic ' + \
        'dimensions must be given.'
    assert type(n_fourier) is tuple and \
        len(n_fourier) == ao_overlaps.n_periodicdims(), m
    
    D = density_matrix
    S = ao_overlaps
    
    # Discrete Fourier transform
    time1 = time()
    #print('S.dft():')
    Sk = toeplitz.dfft(S, n_fourier)
    
    time2 = time()
    #print('Time for S.dft():', time2 - time1)
    
    Dk = toeplitz.dfft(D, n_fourier)
    time3 = time()
    #print('Time for D.dft():', time3 - time2)
    
    pao = tmat()
    pao.coords = Dk.coords
    
    # Compute the PAO coefficients in Fourier space
    for k in Sk.coords:
        
        d = Dk.get(k)
        s = Sk.get(k)
        
        if (type(d) != int) and (type(s) != int):
            
            I = np.eye(len(d))
            paok = I - 0.5 * np.dot(d, s)
            pao.cset(k, paok)
        else:
            pao.cset(k, 0)
    # Return real-space coefficients
    #print('pao.idft()')
    time8 = time()
    pao_idft = toeplitz.idfft(pao, n_fourier)
    time9 = time()
    #print('Time for pao.idft():', time9 - time8)
    return pao_idft


def compute_pao_direct(ao_overlaps, density_matrix,
                       coords_out = None):
    '''
    Using a given density matrix and a corresponding 
    AO overlap matrix, compute coefficents of projected 
    atomic orbitals (PAOs).

    Input:
    
    density_matrix     : Density matrix given as a 
                         toeplitz.tmat object. 
    ao_overlaps        : Atomic orbital overlap matrix,
                         given as a toeplitz.tmat
                         object.
    coords_out         : Output cell coordinate domain.
    
    Output:
    
    The PAO elements are given as C_{AO, PAO}^{L}, 
    where a row corresponds to an AO and a column to a PAO.
    '''
    # \tilde{C}_{\mu, \tilde{\mu}}^{M} =
    #     \delta_{\mu, \tilde{\mu}}\delta_{M, 0} -
    #     0.5 * \sum_{\nu, N} D_{\mu, \nu}^{N+M} *
    #           S_{\nu, \tilde{mu}}^{M - (N+M)} 
    if coords_out is None:
        pao = density_matrix * ao_overlaps
    else:
        pao = density_matrix.cdot(ao_overlaps,
                                  coords = coords_out)
    pao = pao * (-0.5)
    
    elems_0 = pao.get([0, 0, 0])
    elems_0 += np.eye(elems_0.shape[0], dtype=float)
    pao.cset([0, 0, 0], elems_0)
    
    return pao


def norms(ao_overlaps, coeffs):
    '''
    Given AO overlaps and MO or PAO coefficients, compute the 
    norms <p|p> of the PAOs or MOs.
    
    ao_overlaps   : AO overlaps, given as a 'tmat' object.
    pao_coeffs    : PAO coefficients, given as a 'tmat' 
                    object. The elements are stored as 
                    C_{\mu, p}, where a row corresponds to
                    an AO and a column to an MO or a PAO.
    '''
    coords0 = np.zeros((1, 3), dtype=int)    
    
    time1 = time()
    c_tT = coeffs.tT()
    s_tT = ao_overlaps.tT()
    
    # C_tT * S_tT
    cs = c_tT.cdot(s_tT, coords = coeffs.coords)
    time2 = time()
    print('Time for S*C:', time2 - time1)
    
    # (C_tT * S_tT) * C 
    time3 = time()
    orb_overlaps = cs.cdot(coeffs, coords = coords0)
    time4 = time()
    print('Time for C * SC:', time4 - time3)
    
    # Return orbital norms, in the same order as the MO or
    # PAO orbitals given in 'coeffs'.
    return np.diagonal(orb_overlaps.get(coords0[0]))
    
def compute_density_matrix(C, nocc):
    """
    Compute C*C.T where C is the full coefficient Toeplitz matrix
    Author: Audun
    """
    
    C_occ = C.blockslice(range_y=(0,nocc), range_x=(0,C.blockshape[1])) 
    return C_occ*C_occ.tT()*.5

def pao_norms(ao_overlaps, mo_coeffs, n_occupied):
    '''
    Given AO overlaps, MO coefficients, and the number of
    occupied orbitals, compute the norms <\mu |\mu > of
    the corresponding PAOs. 
    
    This function should give the same output as 'norms()'.
    
    ao_overlaps   : AO overlaps, given as a 'tmat' object.
    mo_coeffs     : MO coefficients, given as a 'tmat'
                    object. NOTE: These should be the original
                    WHF coefficients, and not those further
                    localized (by LSDalton).
    n_occupied    : Number of occupied orbitals per cell.
    '''
    
    (n_bf, n_mo) = mo_coeffs.get([0, 0, 0]).shape
    occupied_coeffs = mo_coeffs.get_slice([0, n_bf],
                                          [0, n_occupied]) 
    
    D =  ao_overlaps * occupied_coeffs
    D_tT = D.tT()
    #
    # DD = \sum_{i, L} |<\mu |i L>|^{2},
    #
    # where \mu, i and L correspond to an AO, an occupied
    # orbital, and a cell coordinate vector, respectively.
    coords0 = np.zeros(3, dtype=int)
    DD = toeplitz.multiply(D, D_tT, coords0)
    
    # <\mu |\mu >, where |\mu > is an AO 
    ao_norms = np.diagonal(ao_overlaps.get([0, 0, 0]))
    
    # Return PAO norms, in the same order as the corresponding
    # AOs in 'ao_overlaps'.
    return ao_norms - np.diagonal(DD)


def compute_normalized_pao(ao_overlaps, density_matrix,
                           n_fourier = None,
                           coords_out = None):
    '''
    Compute normalized PAO coefficients.
    
    ao_overlaps        : AO overlaps, given as a 'tmat' 
                         object.
    density_matrix     : Density matrix, given as a 'tmat'
                         object.
    n_fourier          : Fourier parameter (shrink factor).
    '''
    print('Computing PAO coefficients...')
    time1 = time()
    
    #paos = compute_PAO(ao_overlaps, density_matrix, n_fourier)
    paos = compute_pao_direct(ao_overlaps, density_matrix,
                              coords_out)
    
    #print('same:', paos_d.equal_to(paos, tolerance=1e-14,
    #count_blocks=False))
    time2 = time()
    print('Time for computing PAO coefficients:', time2 - time1)
    
    pao_norms = of.norms(ao_overlaps, paos)
    time3 = time()
    print('PAO norms:', pao_norms)
    print('Time for computing PAO norms:', time3 - time2)
    normalized_paos = of.normalize(paos, pao_norms) 
    time4 = time()
    print('Time for normalizing PAO coefficients:',
          time4 - time3)
    # Return the normalized PAO coefficients and the norms
    return normalized_paos, pao_norms 


def compute_mulliken_charges(project_folder, thresh = None):
    print("""
#####################################################
##  Computing Mulliken Charges                     ##
#####################################################""")
    C = tmat()
    C.load_old(project_folder + "/lsdalton_reference_state.npy",
               project_folder + "/lsdalton_reference_coords.npy") #CHANGE TO LSDALTON
    
    #Correct for storage convention in LSDalton
    # C = C.T()
    
    if thresh != None:
        C.truncate(thresh)
    
    S = tmat()
    S.load_old(project_folder + "/crystal_overlap_matrix.npy",
               project_folder + "/crystal_overlap_coords.npy")
    S = S.t()
    
    system = pr.prism(project_folder + "/XDEC/MOLECULE.INP")
    
    ao_sorting = np.load(project_folder + "/atom_regions.npy")
    
    SC = S*C
    SC_coor = []
    SC_coef = []
    ## Store SC as np.arrays
    for L in SC.coords:
        SC_coor.append(L)
        SC_coef.append(SC.get(L))
    

        
    
    
    
    Ct = C.tT()
    

    
    q = []
    q_coor = []
    
    for L in C.coords:
        Cl = Ct.get(L)
        #for atoms in L:
        q.append([])
        q_coor.append(L)
        d_prev = 0
        for da in ao_sorting:
            q[-1].append(np.sum( np.dot(Cl[d_prev:da], SC.get(-L)), axis = 0 ) )
            d_prev = da
    return q, q_coor, SC_coef, SC_coor
    
    #self.reference_state.npload(self.project_folder + "/lsdalton_reference_state.npy", self.project_folder + "/lsdalton_reference_coords.npy")




#############################
##                         ##
## Objective Functions     ##
##                         ##
#############################

class objective_function():
    def __init__(self, geometry):
        self.geometry = geometry
        
    #def foster_boys(self, coords = None, C = None, compute_integrals = True):
    def foster_boys(self, C, S, compute_integrals = True, coords = None):
        ##################################
        ##                              ##
        ## Compute Boys-Foster function ##
        ##                              ##
        ##################################       
        
        #if C is None:
        #    C= tmat()
        #    C.load_old(self.pf + "/crystal_reference_state.npy", self.pf+"/crystal_reference_coords.npy")
        
        geometry = self.geometry
        
        if compute_integrals:
            self.ints = pp.carmom_tmat(geometry, S)
            print("Integrals computed, stored in memory.")
        #print(ints.geometry.lattice)
        lattice = self.geometry.lattice
        
        #print("Lattice:")
        #print(lattice)
        
        if coords==None:
            print("Computing PSM objective function")
            
            S = self.ints.S
        
            X = self.ints.X
            Y = self.ints.Y
            Z = self.ints.Z
            
            X2 = self.ints.X2
            Y2 = self.ints.Y2 
            Z2 = self.ints.Z2 
            
            
            #testcalc
            
            
            
            
            SC = S*C
            
            #Smo = C.tT().cdot(SC,coords = np.array([[0,0,0]]))
            
            xSC = SC.cscale(0, lattice)
            ySC = SC.cscale(1, lattice)
            zSC = SC.cscale(2, lattice)
            
            xxSC = SC.cscale(0, lattice, exponent = 2)
            yySC = SC.cscale(1, lattice, exponent = 2)
            zzSC = SC.cscale(2, lattice, exponent = 2)
            
            XC = X*C
            YC = Y*C
            ZC = Z*C
            
            
            xXC = XC.cscale(0, lattice)
            yYC = YC.cscale(1, lattice)
            zZC = ZC.cscale(2, lattice)
            
            
            
            Xmo = C.tT().cdot(XC, coords = np.array([[0,0,0]])) - C.tT().cdot(xSC, coords = np.array([[0,0,0]]))
            Ymo = C.tT().cdot(YC, coords = np.array([[0,0,0]])) - C.tT().cdot(ySC, coords = np.array([[0,0,0]]))
            Zmo = C.tT().cdot(ZC, coords = np.array([[0,0,0]])) - C.tT().cdot(zSC, coords = np.array([[0,0,0]]))
            
            
            wcenters_x = np.diag(Xmo.cget([0,0,0]))
            wcenters_y = np.diag(Ymo.cget([0,0,0]))
            wcenters_z = np.diag(Zmo.cget([0,0,0]))
            
            wcenters = np.array([wcenters_x, wcenters_y, wcenters_z])
        
            #X2mo = C.tT()*X2*C -  C.tT()*xXC*2 +  C.tT()*xxSC
            #Y2mo = C.tT()*Y2*C -  C.tT()*yYC*2 +  C.tT()*yySC
            #Z2mo = C.tT()*Z2*C -  C.tT()*zZC*2 +  C.tT()*zzSC
            
            X2mo = C.tT().cdot(X2*C, coords = np.array([[0,0,0]])) -  C.tT().cdot(xXC*2, coords = np.array([[0,0,0]])) +  C.tT().cdot(xxSC, coords = np.array([[0,0,0]]))
            Y2mo = C.tT().cdot(Y2*C, coords = np.array([[0,0,0]])) -  C.tT().cdot(yYC*2, coords = np.array([[0,0,0]])) +  C.tT().cdot(yySC, coords = np.array([[0,0,0]]))
            Z2mo = C.tT().cdot(Z2*C, coords = np.array([[0,0,0]])) -  C.tT().cdot(zZC*2, coords = np.array([[0,0,0]])) +  C.tT().cdot(zzSC, coords = np.array([[0,0,0]]))
            
            #PSM_objective_function = X2mo  + Y2mo + Z2mo #- Xmo**2 - Ymo**2 - Zmo**2
            
            #(non orthogonal basis)
            #PSM_objective_function = X2mo  + Y2mo + Z2mo- (Xmo**2)*2 - (Ymo**2)*2 - (Zmo**2)*2 + Xmo%Smo + Ymo%Smo + Zmo%Smo
            return X2mo + Y2mo + Z2mo, Xmo, Ymo, Zmo, wcenters
            #return PSM_objective_function, wcenters
        else:
            
            S = ints.S
            
            X = ints.X
            Y = ints.Y
            Z = ints.Z
            
            X2 = ints.X2
            Y2 = ints.Y2 
            Z2 = ints.Z2 
            
            SC = S*C
            
            xSC = SC.cscale(0, lattice)
            ySC = SC.cscale(1, lattice)
            zSC = SC.cscale(2, lattice)
            
            xxSC = SC.cscale(0, lattice, exponent = 2)
            yySC = SC.cscale(1, lattice, exponent = 2)
            zzSC = SC.cscale(2, lattice, exponent = 2)
            
            XC = X*C
            YC = Y*C
            ZC = Z*C
            
            
            xXC = XC.cscale(0, lattice)
            yYC = YC.cscale(1, lattice)
            zZC = ZC.cscale(2, lattice)
            
            
            Xmo = C.tT().cdot(XC, coords = coords) + C.tT().cdot(xSC, coords = coords)
            Ymo = C.tT().cdot(YC, coords = coords) + C.tT().cdot(ySC, coords = coords)
            Zmo = C.tT().cdot(ZC, coords = coords) + C.tT().cdot(zSC, coords = coords)
            
            
            wcenters_x = np.diag(Xmo.cget([0,0,0]))
            wcenters_y = np.diag(Ymo.cget([0,0,0]))
            wcenters_z = np.diag(Zmo.cget([0,0,0]))
            
            wcenters = np.array([wcenters_x, wcenters_y, wcenters_z])
        
            X2mo = C.tT().cdot(X2*C, coords = coords) +  C.tT().cdot(xXC*2, coords = coords) +  C.tT().cdot(xxSC, coords = coords)
            Y2mo = C.tT().cdot(Y2*C, coords = coords) +  C.tT().cdot(yYC*2, coords = coords) +  C.tT().cdot(yySC, coords = coords)
            Z2mo = C.tT().cdot(Z2*C, coords = coords) +  C.tT().cdot(zZC*2, coords = coords) +  C.tT().cdot(zzSC, coords = coords)
            
            PSM_objective_function = X2mo  + Y2mo + Z2mo- Xmo**2 - Ymo**2 - Zmo**2
            #print(np.diag(PSM_objective_function.cget([0,0,0])))
            
            return PSM_objective_function, wcenters
    def pfm(self, C, S, coords = None):
        ##################################
        ##                              ##
        ## Compute PFM function         ##
        ##                              ##
        ##################################       
        
        #if C is None:
        #    C= tmat()
        #    C.load_old(self.pf + "/crystal_reference_state.npy", self.pf+"/crystal_reference_coords.npy")
        
        #S = tmat()
        #S.load_old(self.pf + "/crystal_overlap_matrix.npy", self.pf+"/crystal_overlap_coords.npy")
        
        ints = pp.carmom_lsdalton(self.geometry, S)
        
        #print(ints.geometry.lattice)
        lattice = self.geometry.lattice
        
        #print("Lattice:")
        #print(lattice)
        
        #if coords==None:
        print("Computing localization objective function")
        
        SC = S*C
    
        XC = ints.X*C
        YC = ints.Y*C
        ZC = ints.Z*C
        
        X2C = ints.X2*C
        Y2C = ints.Y2*C 
        Z2C = ints.Z2*C 
        
        X3C = ints.X3*C
        Y3C = ints.Y3*C 
        Z3C = ints.Z3*C 
        
        X4C = ints.X4*C
        Y4C = ints.Y4*C 
        Z4C = ints.Z4*C 
        
        XYC = ints.XY*C
        XZC = ints.XZ*C
        YZC = ints.YZ*C
        
        X2YC = ints.X2Y*C
        X2ZC = ints.X2Z*C
        Y2ZC = ints.Y2Z*C
        
        XY2C = ints.XY2*C
        XZ2C = ints.XZ2*C
        YZ2C = ints.YZ2*C
        
        X2Y2C = ints.X2Y2*C
        X2Z2C = ints.X2Z2*C
        Y2Z2C = ints.Y2Z2*C
        
        """
        #NOTE: UNVERIFIED SYMMETRIES 
        YXC =ints.YX*C
        Y2XC=ints.Y2X*C
        ZXC =ints.ZX*C
        Z2XC=ints.YX*C
        YXC =ints.YX*C
        YX2C=ints.YX2*C
        ZYC =ints.ZY*C
        ZX2C=ints.ZX2*C
        ZY2C=ints.ZY2*C
        ZYC =ints.ZY*C
        """
        coords = S.coords
        
        Smo =   C.tT().cdot(SC,coords=coords)
        xSC = SC.cscale(0,lattice)
        ySC = SC.cscale(1,lattice)
        zSC = SC.cscale(2,lattice)
        
        # + \hat{x}            -  L_x
        Xmo =   C.tT().cdot(XC,coords=coords)  - C.tT().cdot(SC.cscale(0,lattice),coords=coords)  
        #Xmo =   C.tT()*XC  -  C.tT()*xSC  
        
        # \hat{y}              -  L_y 
        Ymo =   C.tT().cdot(YC,coords=coords) - C.tT().cdot(SC.cscale(1,lattice),coords=coords)
        #Ymo =   C.tT()*YC  -  C.tT()*ySC
        
        #\hat{z}               -  L_z 
        Zmo =   C.tT().cdot(ZC,coords=coords) - C.tT().cdot(SC.cscale(2,lattice),coords=coords)
        #Zmo =   C.tT()*ZC  -  C.tT()*zSC
        print("Wannier centers (X):")
        print(np.diag(Xmo.cget([0,0,0])))
        
        
        
        #                    L_x**2                         - 2*L_x*\hat{x}          + \hat{x}**2
        #X2mo =  C.tT().cdot(SC.cscale(0,lattice,exponent=2) - XC.cscale(0,lattice)*2 + X2C, coords = coords)
        X2mo =  C.tT().cdot(SC.cscale(0,lattice,exponent=2), coords = coords) -  C.tT().cdot(XC.cscale(0,lattice)*2, coords = coords) +  C.tT().cdot(X2C, coords = coords)
        #X2mo =  C.tT()*SC.cscale(0,lattice,exponent=2) -  C.tT()*XC.cscale(0,lattice)*2 +  C.tT()*X2C
        
        """
        ## Error estimate/correction for non-ortogonality
        
        Sa = S*1
        Sa.blocks[ Sa.mapping[ Sa._c2i(np.array([0,0,0])) ] ] -= np.diag(np.ones(Sa.blockshape))
        Sa = Sa*10.0 #multiply by shift along x-axis
        print(Sa.cget([0,0,0]))
        Qmo = Sa.cget([0,0,0])**2*Smo.cget([0,0,0]) + Sa.cget([0,0,0])%Xmo.cget([0,0,0])*2 + Sa.cget([0,0,0])%Smo.cget([0,0,0])*2
        print("Error estimate:")
        print(np.diag(Qmo))
        """
        
        #                    L_y**2                         - 2*L_y*\hat{y}          + \hat{y}**2
        #Y2mo =  C.tT().cdot(SC.cscale(1,lattice,exponent=2) - YC.cscale(1,lattice)*2 + Y2C, coords = coords)
        Y2mo =  C.tT().cdot(SC.cscale(1,lattice,exponent=2), coords = coords) - C.tT().cdot(YC.cscale(1,lattice)*2, coords = coords) + C.tT().cdot(Y2C, coords = coords)
        #Y2mo =  C.tT()*SC.cscale(1,lattice,exponent=2) - C.tT()*YC.cscale(1,lattice)*2 + C.tT()*Y2C
        
        
        #                    L_z**2                         - 2*L_z*\hat{z}          + \hat{z}**2
        #Z2mo =  C.tT().cdot(SC.cscale(2,lattice,exponent=2) - ZC.cscale(2,lattice)*2 + Z2C, coords = coords)
        Z2mo =  C.tT().cdot(SC.cscale(2,lattice,exponent=2), coords = coords) - C.tT().cdot(ZC.cscale(2,lattice)*2, coords = coords) + C.tT().cdot(Z2C, coords = coords)
        #Z2mo =  C.tT()*SC.cscale(2,lattice,exponent=2) - C.tT()*ZC.cscale(2,lattice)*2 + C.tT()*Z2C
        
        
        
        PSM_objective_function = X2mo.cget([0,0,0])  + Y2mo.cget([0,0,0]) + Z2mo.cget([0,0,0])- Xmo.cget([0,0,0])**2 - Ymo.cget([0,0,0])**2 - Zmo.cget([0,0,0])**2 #+ Qmo
        print("PSM objective function:")
        print(np.diag(PSM_objective_function))
        print(np.sqrt(np.diag(PSM_objective_function)))
        
        
        
        #                    - L_x**3                           + 3*L_x**2*\hat{x}                  - 3*L_x*\hat{x}**2         + \hat{x}**3
        X3mo =   C.tT().cdot(SC.cscale(0,lattice,exponent=3)*-1 + XC.cscale(0,lattice,exponent=2)*3 - X2C.cscale(0,lattice)*3  + X3C, coords = coords)
        
        #                    -L_y**3                            + 3*L_y**2*\hat{y}                  - 3*L_y*\hat{y}**2         + \hat{y}**3
        Y3mo =   C.tT().cdot(SC.cscale(1,lattice,exponent=3)*-1 + YC.cscale(1,lattice,exponent=2)*3 - Y2C.cscale(1,lattice)*3  + Y3C, coords = coords)
        
        #                    -L_z**3                            + 3*L_z**2*\hat{z}                  - 3*L_z*\hat{z}**2         + \hat{z}**3
        Z3mo =   C.tT().cdot(SC.cscale(2,lattice,exponent=3)*-1 + ZC.cscale(2,lattice,exponent=2)*3 - Z2C.cscale(2,lattice)*3  + Z3C, coords = coords)
        
        
        
        
        #                   L_x**4                          - 4*L_x**3*\hat{x}                  + 6*L_x**2*\hat{x}**2                - 4*L_x*\hat{x}**3         + \hat{x}**4
        X4mo =  C.tT().cdot(SC.cscale(0,lattice,exponent=4) - XC.cscale(0,lattice,exponent=3)*4 + X2C.cscale(0,lattice,exponent=2)*6 - X3C.cscale(0,lattice)*4  + X4C, coords = coords)
        
        #                   L_y**4                          - 4*L_y**3*\hat{y}                  + 6*L_y**2*\hat{y}**2                - 4*L_y*\hat{y}**3         + \hat{y}**4
        Y4mo =  C.tT().cdot(SC.cscale(1,lattice,exponent=4) - YC.cscale(1,lattice,exponent=3)*4 + Y2C.cscale(1,lattice,exponent=2)*6 - Y3C.cscale(1,lattice)*4  + Y4C, coords = coords)
        
        #                   L_z**4                          - 4*L_z**3*\hat{z}                  + 6*L_z**2*\hat{z}**2                - 4*L_z*\hat{z}**3         + \hat{z}**4
        Z4mo =  C.tT().cdot(SC.cscale(2,lattice,exponent=4) - ZC.cscale(2,lattice,exponent=3)*4 + Z2C.cscale(2,lattice,exponent=2)*6 - Z3C.cscale(2,lattice)*4  + Z4C, coords = coords)
        
        
        
        
        #                    L_x*L_y                               - L_x*\hat{y}          - L_y*\hat{x}          + \hat{x}*\hat{y}
        XYmo =  C.tT().cdot(SC.cscale(0,lattice).cscale(1,lattice) - YC.cscale(0,lattice) - XC.cscale(1,lattice) + XYC, coords = coords)
        
        #                   L_x*L_z                                - L_x*\hat{z}          - L_z*\hat{x}          + \hat{x}*\hat{z}
        XZmo =  C.tT().cdot(SC.cscale(0,lattice).cscale(2,lattice) - ZC.cscale(0,lattice) - XC.cscale(2,lattice) + XZC, coords = coords)
        
        #                    L_y*L_z                               - L_y*\hat{z}          - L_z*\hat{y}          + \hat{y}*\hat{z}
        YZmo =  C.tT().cdot(SC.cscale(1,lattice).cscale(2,lattice) - ZC.cscale(1,lattice) - YC.cscale(2,lattice) + YZC, coords = coords)
        
        
        
        
        
        #                    - L_y**2*L_z                                          + L_y**2*\hat{z}                  + 2*L_y*L_z*\hat{y}                        - 2*L_y*\hat{y}*\hat{z}   - L_z*\hat{y}**2        + \hat{y}**2*\hat{z}
        Y2Zmo =  C.tT().cdot(SC.cscale(1,lattice,exponent=2).cscale(2,lattice)*-1  + ZC.cscale(1,lattice,exponent=2) + YC.cscale(1,lattice).cscale(2,lattice)*2 - YZC.cscale(1,lattice)*2 - Y2C.cscale(2,lattice) + Y2ZC, coords = coords)
        
        #                     - L_x**2*L_y                                         + L_x**2*\hat{y}                  + 2*L_x*L_y*\hat{x}                        - 2*L_x*\hat{x}*\hat{y}   - L_y*\hat{x}**2        + \hat{x}**2*\hat{y}
        X2Ymo =   C.tT().cdot(SC.cscale(0,lattice,exponent=2).cscale(1,lattice)*-1 + YC.cscale(0,lattice,exponent=2) + XC.cscale(0,lattice).cscale(1,lattice)*2 - XYC.cscale(0,lattice)*2 - X2C.cscale(1,lattice) + X2YC, coords = coords)
        
        #                     -L_x**2*L_z                                          + L_x**2*\hat{z}                  + 2*L_x*L_z*\hat{x}                        - 2*L_x*\hat{x}*\hat{z}   - L_z*\hat{x}**2        + \hat{x}**2*\hat{z}
        X2Zmo =   C.tT().cdot(SC.cscale(0,lattice,exponent=2).cscale(2,lattice)*-1 + ZC.cscale(0,lattice,exponent=2) + XC.cscale(0,lattice).cscale(2,lattice)*2 - XZC.cscale(0,lattice)*2 - X2C.cscale(2,lattice) + X2ZC, coords = coords)
        
        
        
        
        #                     - L_x*L_y**2                                         + 2*L_x*L_y*\hat{y}                        - L_x*\hat{y}**2        + L_y**2*\hat{x}                  - 2*L_y*\hat{x}*\hat{y}   + \hat{x}*\hat{y}**2
        XY2mo =   C.tT().cdot(SC.cscale(1,lattice,exponent=2).cscale(0,lattice)*-1 + YC.cscale(0,lattice).cscale(1,lattice)*2 - Y2C.cscale(0,lattice) + XC.cscale(1,lattice,exponent=2) - XYC.cscale(1,lattice)*2 + XY2C, coords = coords)
        
        #                     - L_x*L_z**2                                         + 2*L_x*L_z*\hat{z}                        - L_x*\hat{z}**2        + L_z**2*\hat{x}                  - 2*L_z*\hat{x}*\hat{z}   + \hat{x}*\hat{z}**2
        XZ2mo =   C.tT().cdot(SC.cscale(2,lattice,exponent=2).cscale(0,lattice)*-1 + ZC.cscale(0,lattice).cscale(2,lattice)*2 - Z2C.cscale(0,lattice) + XC.cscale(2,lattice,exponent=2) - XZC.cscale(2,lattice)*2 + XZ2C, coords = coords)
        
        #                     - L_y*L_z**2                                         + 2*L_y*L_z*\hat{z}                        - L_y*\hat{z}**2        + L_z**2*\hat{y}                  - 2*L_z*\hat{y}*\hat{z}   + \hat{y}*\hat{z}**2
        YZ2mo =   C.tT().cdot(SC.cscale(2,lattice,exponent=2).cscale(1,lattice)*-1 + ZC.cscale(1,lattice).cscale(2,lattice)*2 - Z2C.cscale(1,lattice) + YC.cscale(2,lattice,exponent=2) - YZC.cscale(2,lattice)*2 + YZ2C, coords = coords)
        
        
        
        
        #                       L_x**2*L_z**2                                              - 2*L_x**2*L_z*\hat{z}                                + L_x**2*\hat{z}**2                - 2*L_x*L_z**2*\hat{x}                                + 4*L_x*L_z*\hat{x}*\hat{z}                   - 2*L_x*\hat{x}*\hat{z}**2    + L_z**2*\hat{x}**2                   - 2*L_z*\hat{x}**2*\hat{z}     + \hat{x}**2*\hat{z}**2
        X2Z2mo =  C.tT().cdot(SC.cscale(0,lattice,exponent=2).cscale(2,lattice,exponent=2) - ZC.cscale(0,lattice,exponent=2).cscale(2,lattice)*2 + Z2C.cscale(0,lattice,exponent=2) - XC.cscale(2,lattice,exponent=2).cscale(0,lattice)*2 + XZC.cscale(0,lattice).cscale(2,lattice)*4   - XZ2C.cscale(0,lattice)*2    + X2C.cscale(2,lattice,exponent=2)    - X2ZC.cscale(2,lattice)*2     + X2Z2C, coords = coords)
        
        #                     L_x**2*L_y**2                                                - 2*L_x**2*L_y*\hat{y}                                + L_x**2*\hat{y}**2                - 2*L_x*L_y**2*\hat{x}                                + 4*L_x*L_y*\hat{x}*\hat{y}                   - 2*L_x*\hat{x}*\hat{y}**2    + L_y**2*\hat{x}**2                   - 2*L_y*\hat{x}**2*\hat{y}     + \hat{x}**2*\hat{y}**2
        X2Y2mo=  C.tT().cdot(SC.cscale(0,lattice,exponent=2).cscale(1,lattice,exponent=2)  - YC.cscale(0,lattice,exponent=2).cscale(1,lattice)*2 + Y2C.cscale(0,lattice,exponent=2) - XC.cscale(1,lattice,exponent=2).cscale(0,lattice)*2 + XYC.cscale(0,lattice).cscale(1,lattice)*4   - XY2C.cscale(0,lattice)*2    + X2C.cscale(1,lattice,exponent=2)    - X2YC.cscale(1,lattice)*2     + X2Y2C, coords = coords)
        
        #                      L_y**2*L_z**2                                               - 2*L_y**2*L_z*\hat{z}                                + L_y**2*\hat{z}**2               - 2*L_y*L_z**2*\hat{y}                                 + 4*L_y*L_z*\hat{y}*\hat{z}                   - 2*L_y*\hat{y}*\hat{z}**2    + L_z**2*\hat{y}**2                   - 2*L_z*\hat{y}**2*\hat{z}     + \hat{y}**2*\hat{z}**2
        Y2Z2mo =  C.tT().cdot(SC.cscale(1,lattice,exponent=2).cscale(2,lattice,exponent=2) - ZC.cscale(1,lattice,exponent=2).cscale(2,lattice)*2 + Z2C.cscale(1,lattice,exponent=2) - YC.cscale(2,lattice,exponent=2).cscale(1,lattice)*2 + YZC.cscale(1,lattice).cscale(2,lattice)*4   - YZ2C.cscale(1,lattice)*2    + Y2C.cscale(2,lattice,exponent=2)    - Y2ZC.cscale(2,lattice)*2     + Y2Z2C    , coords = coords)        


        """

        #      L_x*L_y                                            - L_x*\hat{y}          - L_y*\hat{x}          + \hat{y}*\hat{x}
        YXmo = C.tT().cdot(SC.cscale(0,lattice).cscale(1,lattice) - YC.cscale(0,lattice) - XC.cscale(1,lattice) + YXC , coords = coords)
        
        #        -L_x*L_y**2                                                      + 2*L_x*L_y*\hat{y}                        - L_x*\hat{y}**2        + L_y**2*\hat{x}                  - 2*L_y*\hat{y}*\hat{x}   + \hat{y}**2*\hat{x}
        Y2Xmo =  C.tT().cdot(SC.cscale(1,lattice,exponent=2).cscale(0,lattice)*-1 + YC.cscale(0,lattice).cscale(1,lattice)*2 - Y2C.cscale(0,lattice) + XC.cscale(1,lattice,exponent=2) - YXC.cscale(1,lattice)*2 + Y2XC , coords = coords)
        
        #      L_x*L_z                                            - L_x*\hat{z}          - L_z*\hat{x}          + \hat{z}*\hat{x}
        ZXmo = C.tT().cdot(SC.cscale(0,lattice).cscale(2,lattice) - ZC.cscale(0,lattice) - XC.cscale(2,lattice) + ZXC , coords = coords)
        
        #        -L_x*L_z**2                                                      + 2*L_x*L_z*\hat{z}                        - L_x*\hat{z}**2        + L_z**2*\hat{x}                  - 2*L_z*\hat{z}*\hat{x}   + \hat{z}**2*\hat{x}
        Z2Xmo =  C.tT().cdot(SC.cscale(2,lattice,exponent=2).cscale(0,lattice)*-1 + ZC.cscale(0,lattice).cscale(2,lattice)*2 - Z2C.cscale(0,lattice) + XC.cscale(2,lattice,exponent=2) - ZXC.cscale(2,lattice)*2 + Z2XC , coords = coords)
        
        #        -L_x**2*L_y                                                      + L_x**2*\hat{y}                  + 2*L_x*L_y*\hat{x}                        - 2*L_x*\hat{y}*\hat{x}   - L_y*\hat{x}**2        + \hat{y}*\hat{x}**2
        YX2mo =  C.tT().cdot(SC.cscale(0,lattice,exponent=2).cscale(1,lattice)*-1 + YC.cscale(0,lattice,exponent=2) + XC.cscale(0,lattice).cscale(1,lattice)*2 - YXC.cscale(0,lattice)*2 - X2C.cscale(1,lattice) + YX2C , coords = coords)
        
        #       L_y*L_z                                           - L_y*\hat{z}          - L_z*\hat{y}          + \hat{z}*\hat{y}
        ZYmo = C.tT().cdot(SC.cscale(1,lattice).cscale(2,lattice) - ZC.cscale(1,lattice) - YC.cscale(2,lattice) + ZYC , coords = coords)

        #        -L_x**2*L_z                                                       + L_x**2*\hat{z}                  + 2*L_x*L_z*\hat{x}                        - 2*L_x*\hat{z}*\hat{x}   - L_z*\hat{x}**2        + \hat{z}*\hat{x}**2
        ZX2mo =  C.tT().cdot(SC.cscale(0,lattice,exponent=2).cscale(2, lattice)*-1 + ZC.cscale(0,lattice,exponent=2) + XC.cscale(0,lattice).cscale(2,lattice)*2 - ZXC.cscale(0,lattice)*2 - X2C.cscale(2,lattice) + ZX2C , coords = coords)
        
        #        -L_y**2*L_z                                                       + L_y**2*\hat{z}                  + 2*L_y*L_z*\hat{y}                        - 2*L_y*\hat{z}*\hat{y}   - L_z*\hat{y}**2        + \hat{z}*\hat{y}**2
        ZY2mo =  C.tT().cdot(SC.cscale(1,lattice,exponent=2).cscale(1, lattice)*-1 + ZC.cscale(1,lattice,exponent=2) + YC.cscale(1,lattice).cscale(2,lattice)*2 - ZYC.cscale(1,lattice)*2 - Y2C.cscale(2,lattice) + ZY2C  , coords = coords)          
                                
        #        -L_y*L_z**2                                                      + 2*L_y*L_z*\hat{z}                        - L_y*\hat{z}**2        + L_z**2*\hat{y}                  - 2*L_z*\hat{z}*\hat{y}   + \hat{z}**2*\hat{y}
        #Z2Ymo =  C.tT().cdot(SC.cscale(2,lattice,exponent=2).cscale(1,lattice)*-1 + ZC.cscale(1,lattice).cscale(2,lattice)*2 - Z2C.cscale(1,lattice) + YC.cscale(2,lattice,exponent=2) - ZYC.cscale(2,lattice)*2 + Z2YC , coords = coords)
        Z2Ymo =  C.tT().cdot(SC.cscale(2,lattice,exponent=2).cscale(1,lattice)*-1 + ZC.cscale(1,lattice).cscale(2,lattice)*2 - Z2C.cscale(1,lattice) + YC.cscale(2,lattice,exponent=2) - ZYC.cscale(2,lattice)*2 + YZ2C , coords = coords)
        
        #        L_x**2*L_y**2                                                            - 2*L_x**2*L_y*\hat{y}                                + L_x**2*\hat{y}**2                - 2*L_x*L_y**2*\hat{x}                                 + 4*L_x*L_y*\hat{y}*\hat{x}                 - 2*L_x*\hat{y}**2*\hat{x} + L_y**2*\hat{x}**2                - 2*L_y*\hat{y}*\hat{x}**2 + \hat{y}**2*\hat{x}**2
        Y2X2mo = C.tT().cdot(SC.cscale(0,lattice,exponent=2).cscale(1,lattice,exponent=2) - YC.cscale(0,lattice,exponent=2).cscale(1,lattice)*2 + Y2C.cscale(0,lattice,exponent=2) - XC.cscale(1,lattice,exponent=2).cscale(0,lattice)*2  + YXC.cscale(0,lattice).cscale(1,lattice)*4 - Y2C.cscale(0,lattice)*2  + X2C.cscale(1,lattice,exponent=2) - X2C.cscale(1,lattice)*2   + X2Y2C , coords = coords)
        
        #        L_x**2*L_z**2                                                            - 2*L_x**2*L_z*\hat{z}                                + L_x**2*\hat{z}**2                - 2*L_x*L_z**2*\hat{x}                                 + 4*L_x*L_z*\hat{z}*\hat{x}                 - 2*L_x*\hat{z}**2*\hat{x} + L_z**2*\hat{x}**2                - 2*L_z*\hat{z}*\hat{x}**2   + \hat{z}**2*\hat{x}**2
        #Z2X2mo = C.tT().cdot(SC.cscale(0,lattice,exponent=2).cscale(2,lattice,exponent=2) - ZC.cscale(0,lattice,exponent=2).cscale(2,lattice)*2 + Z2C.cscale(0,lattice,exponent=2) - XC.cscale(2,lattice,exponent=2).cscale(0,lattice)*2  + XZC.cscale(0,lattice).cscale(2,lattice)*4 - XZC.cscale(0,lattice)*2  + X2C.cscale(2,lattice,exponent=2) - ZX2C.cscale(2,lattice)*2   + Z2X2C , coords = coords)
        Z2X2mo = C.tT().cdot(SC.cscale(0,lattice,exponent=2).cscale(2,lattice,exponent=2) - ZC.cscale(0,lattice,exponent=2).cscale(2,lattice)*2 + Z2C.cscale(0,lattice,exponent=2) - XC.cscale(2,lattice,exponent=2).cscale(0,lattice)*2  + XZC.cscale(0,lattice).cscale(2,lattice)*4 - XZC.cscale(0,lattice)*2  + X2C.cscale(2,lattice,exponent=2) - ZX2C.cscale(2,lattice)*2   + X2Z2C , coords = coords)
        
        #         L_y**2*L_z**2                                                           - 2*L_y**2*L_z*\hat{z}                                + L_y**2*\hat{z}**2                - 2*L_y*L_z**2*\hat{y}                                + 4*L_y*L_z*\hat{z}*\hat{y}                 - 2*L_y*\hat{z}**2*\hat{y}  + L_z**2*\hat{y}**2                - 2*L_z*\hat{z}*\hat{y}**2   + \hat{z}**2*\hat{y}**2
        Z2Y2mo = C.tT().cdot(SC.cscale(2,lattice,exponent=2).cscale(1,lattice,exponent=2) - ZC.cscale(1,lattice,exponent=2).cscale(2,lattice)*2 + Z2C.cscale(1,lattice,exponent=2) - YC.cscale(2,lattice,exponent=2).cscale(1,lattice)*2 + ZYC.cscale(1,lattice).cscale(2,lattice)*4 - Z2YC.cscale(1,lattice)*2  + Y2C.cscale(2,lattice,exponent=2) - ZY2C.cscale(2,lattice)*2   + Z2Y2C , coords = coords)
        #Z2Y2mo = C.tT().cdot(SC.cscale(2,lattice,exponent=2).cscale(1,lattice,exponent=2) - ZC.cscale(1,lattice,exponent=2).cscale(2,lattice)*2 + Z2C.cscale(1,lattice,exponent=2) - YC.cscale(2,lattice,exponent=2).cscale(1,lattice)*2 + ZYC.cscale(1,lattice).cscale(2,lattice)*4 - YZ2C.cscale(1,lattice)*2  + Y2C.cscale(2,lattice,exponent=2) - ZY2C.cscale(2,lattice)*2   + Y2Z2C , coords = coords)
        """
        
                
        """
        PFM_objective_function = Smo%Xmo**4                   #  < s > <x>^4
        PFM_objective_function+= Smo%Xmo**2%Ymo**2*2          #  2 < s > <x>^2 <y>^2
        PFM_objective_function+= Smo%Xmo**2%Zmo**2*2          #  2 < s > <x>^2 <z>^2
        PFM_objective_function+= Smo%Ymo**4                   #  < s > <y>^4
        PFM_objective_function+= Smo%Ymo**2%Zmo**2*2          #  2 < s > <y>^2 <z>^2
        PFM_objective_function+= Smo%Zmo**4                   #  < s > <z>^4
        PFM_objective_function+= XYmo%Xmo%Ymo*4               #  4 <xy> <x> <y>
        PFM_objective_function+= XY2mo%Xmo*(-2)               #  - 2 <xy^2> <x>
        PFM_objective_function+= XZmo%Xmo%Zmo*4               #  4 <xz> <x> <z>
        PFM_objective_function+= XZ2mo%Xmo*(-2)               #  - 2 <xz^2> <x>
        PFM_objective_function+= Xmo**4*(- 4)                 #  - 4 <x>^4
        PFM_objective_function+= Xmo**2%X2mo*6                #  6 <x>^2 <x^2>
        PFM_objective_function+= Xmo**2%Ymo**2*(- 8)          #  - 8 <x>^2 <y>^2
        PFM_objective_function+= Xmo**2%Y2mo*2                #  2 <x>^2 <y^2>
        PFM_objective_function+= Xmo**2%Zmo**2*(- 8)          #  - 8 <x>^2 <z>^2
        PFM_objective_function+= Xmo**2%Z2mo*2                #  2 <x>^2 <z^2>
        PFM_objective_function+= Xmo%X3mo*(- 4)               #  - 4 <x> <x^3>
        PFM_objective_function+= Xmo%YXmo%Ymo*4               #  4 <x> <yx> <y>
        PFM_objective_function+= Xmo%Y2Xmo*(-2)               #  - 2 <x> <y^2x>
        PFM_objective_function+= Xmo%ZXmo%Zmo*4               #  4 <x> <zx> <z>
        PFM_objective_function+= Xmo%Z2Xmo*(-2)               #  - 2 <x> <z^2x>
        PFM_objective_function+= X2Ymo%Ymo*(- 2)              #  - 2 <x^2y> <y>
        PFM_objective_function+= X2Y2mo                       #  <x^2y^2>
        PFM_objective_function+= X2Zmo%Zmo*(- 2)              #  - 2 <x^2z> <z>
        PFM_objective_function+= X2Z2mo                       #  <x^2z^2>
        PFM_objective_function+= X2mo%Ymo**2*2                #  2 <x^2> <y>^2
        PFM_objective_function+= X2mo%Zmo**2*2                #  2 <x^2> <z>^2
        PFM_objective_function+= X4mo                         #  <x^4>
        PFM_objective_function+= YX2mo%Ymo*(- 2)              #  - 2 <yx^2> <y>
        PFM_objective_function+= YZmo%Ymo%Zmo*4               #  4 <yz> <y> <z>
        PFM_objective_function+= YZ2mo%Ymo*(- 2)              #  - 2 <yz^2> <y>
        PFM_objective_function+= Ymo**4*(- 4)                 #  - 4 <y>^4
        PFM_objective_function+= Ymo**2%Y2mo*6                #  6 <y>^2 <y^2>
        PFM_objective_function+= Ymo**2%Zmo**2*(- 8)          #  - 8 <y>^2 <z>^2
        PFM_objective_function+= Ymo**2%Z2mo*2                #  2 <y>^2 <z^2>
        PFM_objective_function+= Ymo%Y3mo*(-4)                #  - 4 <y> <y^3>
        PFM_objective_function+= Ymo%ZYmo%Zmo*4               #  4 <y> <zy> <z>
        PFM_objective_function+= Ymo%Z2Ymo*(- 2)              #  - 2 <y> <z^2y>
        PFM_objective_function+= Y2X2mo                       #  <y^2x^2>
        PFM_objective_function+= Y2Zmo%Zmo*(- 2)              #  - 2 <y^2z> <z>
        PFM_objective_function+= Y2Z2mo                       #  <y^2z^2>
        PFM_objective_function+= Y2mo%Zmo**2                  #  2 <y^2> <z>^2
        PFM_objective_function+= Y4mo                         #  <y^4>
        PFM_objective_function+= ZX2mo%Zmo*(- 2)              #  - 2 <zx^2> <z>
        PFM_objective_function+= ZY2mo%Zmo*(- 2)              #  - 2 <zy^2> <z>
        PFM_objective_function+= Zmo**4*(- 4)                 #  - 4 <z>^4
        PFM_objective_function+= Zmo**2%Z2mo*6                #  6 <z>^2 <z^2>
        PFM_objective_function+= Zmo%Z3mo*(- 4)               #  - 4 <z> <z^3>
        PFM_objective_function+= Z2X2mo                       #  <z^2x^2>
        PFM_objective_function+= Z2Y2mo                       #  <z^2y^2>
        PFM_objective_function+= Z4mo                         #  <z^4>
        """
        
        #The following objective function assumes commuting cartesian operators
        PFM_tensors = [Smo, Xmo, Ymo, Zmo, XYmo, XZmo, XY2mo, X2mo, Y2mo, Z2mo, X3mo, Y3mo, Z3mo, X2Ymo, X2Y2mo, X2Zmo, XZ2mo, X2Z2mo, X4mo, YZmo, YZ2mo, Y2Zmo, Y2Z2mo, Y4mo, Z4mo]
        
        
        PFM_objective_function = assemble_pfm_function(PFM_tensors)
        """
        PFM_objective_function = Smo%Xmo**2 + \
        Smo%Xmo**2%Ymo**2*2 + \
        Smo%Xmo**2%Zmo**2*2 + \
        Smo%Ymo**4+ \
        Smo%Ymo**2%Zmo**2*2 +\
        Smo%Zmo**4+\
        XYmo%Xmo%Ymo*8-\
        XY2mo%Xmo*4+\
        XZmo%Xmo%Zmo*8-\
        XZ2mo%Xmo*4-\
        Xmo**4*4+\
        Xmo**2%X2mo*6-\
        Xmo**2%Ymo**2*8+\
        Xmo**2%Y2mo*2-\
        Xmo**2%Zmo**2*8+ \
        Xmo**2%Z2mo*2 -\
        Xmo%X3mo*4-\
        X2Ymo%Ymo*4+\
        X2Y2mo*2-\
        X2Zmo%Zmo*4+ \
        X2Z2mo*2+\
        X2mo%Ymo**2*2+\
        X2mo%Zmo**2*2+\
        X4mo+\
        YZmo%Ymo%Zmo*8-\
        YZ2mo%Ymo*4-\
        Ymo**4*4+\
        Ymo**2%Y2mo*6-\
        Ymo**2%Zmo**2*8+\
        Ymo**2%Z2mo*2-\
        Ymo%Y3mo*4-\
        Y2Zmo%Zmo*4+\
        Y2Z2mo*2+\
        Y2mo%Zmo**2*2+\
        Y4mo-\
        Zmo**4*4+\
        Zmo**2%Z2mo*6-\
        Zmo%Z3mo*4+\
        Z4mo
        """
        
        
        
        wcenters_x = np.diag(Xmo.cget([0,0,0]))
        wcenters_y = np.diag(Ymo.cget([0,0,0]))
        wcenters_z = np.diag(Zmo.cget([0,0,0]))
        
        wcenters = np.array([wcenters_x, wcenters_y, wcenters_z])
    

        #PSM_objective_function = X2mo  + Y2mo + Z2mo- Xmo**2 - Ymo**2 - Zmo**2
        #print(np.diag(PSM_objective_function.cget([0,0,0])))
        
        return PFM_tensors, PFM_objective_function,PSM_objective_function, wcenters

def assemble_pfm_function(PFM_tensors):
    Smo, Xmo, Ymo, Zmo, XYmo, XZmo, XY2mo, X2mo, Y2mo, Z2mo, X3mo, Y3mo, Z3mo, X2Ymo, X2Y2mo, X2Zmo, XZ2mo, X2Z2mo, X4mo, YZmo, YZ2mo, Y2Zmo, Y2Z2mo, Y4mo, Z4mo = PFM_tensors

            
    PFM_objective_function = X4mo + Y4mo + Z4mo \
    - (X3mo%Xmo + Y3mo%Ymo + Z3mo%Zmo)*4 \
    + (X2mo%Xmo**2 + Y2mo%Ymo**2 + Z2mo%Zmo**2)*6 \
    - (Xmo%Xmo%Xmo%Xmo + Ymo%Ymo%Ymo%Ymo + Zmo%Zmo%Zmo%Zmo)*3 \
    + (X2Y2mo + X2Z2mo + Y2Z2mo)*2 \
    + (X2mo%Ymo**2 + X2mo%Zmo**2 +Y2mo%Xmo**2 + Y2mo%Zmo**2 + Z2mo%Xmo**2 + Z2mo%Ymo**2)*2 \
    - (X2Ymo%Ymo + X2Zmo%Zmo + XY2mo%Xmo + XZ2mo%Xmo + Y2Zmo%Zmo + YZ2mo%Ymo)*4 \
    - (Xmo**2%Ymo**2 + Xmo**2%Zmo**2 + Ymo**2%Zmo**2)*6 \
    + (XZmo%Xmo%Zmo + XYmo%Xmo%Ymo + YZmo%Ymo%Zmo)*8 
    
    
    return PFM_objective_function

def assemble_pfm_function_(PFM_tensors):
    #original
    Smo, Xmo, Ymo, Zmo, XYmo, XZmo, XY2mo, X2mo, Y2mo, Z2mo, X3mo, Y3mo, Z3mo, X2Ymo, X2Y2mo, X2Zmo, XZ2mo, X2Z2mo, X4mo, YZmo, YZ2mo, Y2Zmo, Y2Z2mo, Y4mo, Z4mo = PFM_tensors
            
    PFM_objective_function = Smo%Xmo**2 + \
    Smo%Xmo**2%Ymo**2*2 + \
    Smo%Xmo**2%Zmo**2*2 + \
    Smo%Ymo**4+ \
    Smo%Ymo**2%Zmo**2*2 +\
    Smo%Zmo**4+\
    XYmo%Xmo%Ymo*8-\
    XY2mo%Xmo*4+\
    XZmo%Xmo%Zmo*8-\
    XZ2mo%Xmo*4-\
    Xmo**4*4+\
    Xmo**2%X2mo*6-\
    Xmo**2%Ymo**2*8+\
    Xmo**2%Y2mo*2-\
    Xmo**2%Zmo**2*8+ \
    Xmo**2%Z2mo*2 -\
    Xmo%X3mo*4-\
    X2Ymo%Ymo*4+\
    X2Y2mo*2-\
    X2Zmo%Zmo*4+ \
    X2Z2mo*2+\
    X2mo%Ymo**2*2+\
    X2mo%Zmo**2*2+\
    X4mo+\
    YZmo%Ymo%Zmo*8-\
    YZ2mo%Ymo*4-\
    Ymo**4*4+\
    Ymo**2%Y2mo*6-\
    Ymo**2%Zmo**2*8+\
    Ymo**2%Z2mo*2-\
    Ymo%Y3mo*4-\
    Y2Zmo%Zmo*4+\
    Y2Z2mo*2+\
    Y2mo%Zmo**2*2+\
    Y4mo-\
    Zmo**4*4+\
    Zmo**2%Z2mo*6-\
    Zmo%Z3mo*4+\
    Z4mo
    
    return PFM_objective_function


    
def compute_mulliken_charge(project_folder_f):
    q, q_coords = compute_mulliken_charges(project_folder_f)
    Q = np.zeros(len(q[0][0]), dtype = float)
    for i in np.arange(len(q_coords)):
        for j in np.arange(len(q[i])):
            Q += np.array(q[i][j])
    for i in Q:
        #for e in i:
        print("%.16f" % i)
        print("")
    #print(Q)
    q = np.array(q)
    q_coords = np.array(q_coords)
    
    #print results to screen
    for i in range(len(q_coords)):
        #print("=====")
        
        #print("---")
        for e in np.arange(len(q[i])):
            if np.sum(np.abs(q[i][e]))>=10e-9 and np.sum(np.abs(q_coords[i]))==0:
                print(q_coords[i])
                Q = np.array(q[i][e])
                print(Q)
    
    
    np.save(project_folder_f + "/mulliken_charges_blocks.npy", q)
    np.save(project_folder_f + "/mulliken_charges_coords.npy", q_coords)
    print("""Done. Results stored in: 
%s/mulliken_charges*.npy
    """ % project_folder_f)
    
    #return q, q_coords

#def kspace_rotate(U_rot)


def get_wannier_data(fname):
    f = open(fname, "r")
    F = f.read()
    
    Fs = F.split("CENTROID'S COORDINATES R0:")[1:]
    
    centers = []
    
    for i in Fs:
        centers.append([literal_eval(j) for j in i.split("\n")[0].split()])
    
    Fs2 = F.split("EXPECTATION VALUE OF (R-R0)**2:")[1:]
    
    spreads = []
    
    for i in Fs2:
        #print(i.split("\n")[0])
        spreads.append(literal_eval(i.split("\n")[0].replace(" ", "")))
        
    return np.array(centers), np.array(spreads)
    
def save_rotation(C, U_rot, project_folder, filename, kspace = False, printing = False):
    if kspace:
        C_rot = at.rotate_k(C, U_rot)
    else:
        C_rot = at.rotate(C, U_rot)
    

    
    C_rot.save(project_folder + "/" + filename)
            
    np.save(project_folder + "/" + filename + "rotation_matrix.npy", U_rot)
    
    if printing:
        print("Results stored to disk:")
        print(project_folder + "/" + filename + ".npy")
        print(project_folder + "/" + filename + "rotation_matrix.npy")
    
    


if __name__ == "__main__":
    print("""#####################################################
##  ,--.   ,--.      ,------.  ,------. ,-----.    ##
##   \  `.'  /,-----.|  .-.  \ |  .---''  .--./    ##
##    .'    \ '-----'|  |  \  :|  `--, |  |        ## 
##   /  .'.  \       |  '--'  /|  `---.'  '--'\    ##
##  '--'   '--'      `------Wannier Refinery--'    ##
#####################################################""")
    parser = argparse.ArgumentParser(prog = "orb_refinery",
                                        description = "Post-process orbitals from Crystal/LSDalton",
                                        epilog = "Author: Audun Skau Hansen (e-mail: a.s.hansen@kjemi.uio.no)")
    
    parser.add_argument("-project_folder", type = str, default = None, help ="Project folder containing input files.")
    parser.add_argument("-compute_overlap", action="store_true", default=False, help="Compute Wannier overlap matrix.")
    parser.add_argument("-compute_mcharge", action="store_true", default=False, help="Compute Mulliken Charge.")
    parser.add_argument("-compute_overlap_ao", action="store_true", default=False, help="Compute AO overlap matrix, store to disk.")
    parser.add_argument("-localize", type=str, default=None, help="Localize using ")
    parser.add_argument("-m", type=int, default=2, help="Power of localization (i.e. PSM-m). Default 2.")
    parser.add_argument("-integral_file", type=str, default=None, help="Integrals provided in file")
    parser.add_argument("-stepsize", type=float, default=0.01, help="Stepsize for unitary rotations in localization.")
    parser.add_argument("-nsteps", type=int, default=15000, help="Number of steps for unitary rotations in localization.")
    parser.add_argument("-nwalkers", type=int, default=1, help="Number of walkers for unitary rotations in localization with ensemble simulated annealing.")
    parser.add_argument("-compute_pao", action = "store_true", default=False, help="Localize using ")
    parser.add_argument("-compute_orbspread", action = "store_true", default=False, help="Compute orbital spread (psm-1)" )
    parser.add_argument("-compute_pfm", action = "store_true", default=False, help="Compute orbital kurtosis (pfm-1)" )
    parser.add_argument("-skip_occupied", action = "store_true", default=False, help="Perform localization on virtuals only." )
    
    parser.add_argument("-crystal", type = str, help ="Input file for prism (.d12 or .INP)")
    
    
    parser.add_argument("-compute_pao_orbspread", action = "store_true", default=False, help="Compute orbital spread (psm-1 for paos)" )
    parser.add_argument("-wannierization", action = "store_true", default=False, help="Perform a Wannierization + Localization" )
    parser.add_argument("-compute_lsint", action = "store_true", default=False, help="Compute and store cartesian multipoles for (0,0,0)." )
    parser.add_argument("-brute_force_wcenter", action = "store_true", default=False, help="Brute force wannier centers" )
    parser.add_argument("-brute_force_orbspread", action = "store_true", default=False, help="Brute force wannier centers" )
    parser.add_argument("-convert_cryscor", action = "store_true", default=False, help="conversion of PAOs to cryscor format" )
    parser.add_argument("-kspace_overlap", action = "store_true", default=False, help="copmute mo overlap in kspace" )
    parser.add_argument("-consistency_check", action = "store_true", default=False, help="Check consistency of rotated orbital space" )
    parser.add_argument("-build_fock", action = "store_true", default=False, help="Construct AO Fock matrix" )
    parser.add_argument("-orbital_analysis", action = "store_true", default=False, help="Analyse orbitals indicated by bfile, cfile (mandatory)" )
    parser.add_argument("-plot_orbitals", action = "store_true", default=False, help="Make radial plots (during orbital analysis)" )
    parser.add_argument("-infile", type = str, default = None, help ="Optional file if other than assumed by algorithm.")
    parser.add_argument("-outfile", type = str, default = None, help ="Optional storage file if other than assumed by algorithm.")
    parser.add_argument("-cfile", type = str, default = None, help ="Optional coord file if other than assumed by algorithm.")
    parser.add_argument("-bfile", type = str, default = None, help ="Optional block file if other than assumed by algorithm.")
    parser.add_argument("-vspace_analysis", action = "store_true", default=False, help="Virtual space analysis" )
    parser.add_argument("-new2old", action = "store_true", default=False, help="Convert infile to old toeplitz format" )
    parser.add_argument("-old2new", action = "store_true", default=False, help="Convert bfile, cfile to new toeplitz format" )
    
    
    
    
    args = parser.parse_args()
    
    # Conversion 
    if args.old2new:
        M= tmat()
        M.load_old(args.bfile, args.cfile)
        M.save(args.outfile)
        sys.exit("Conversion done.")
    if args.new2old:
        M= tmat()
        M.load(args.infile)
        M.save_old(args.bfile, args.cfile)
        sys.exit("Conversion done.")
    
    
    if args.project_folder is None:
        project_folder = os.getcwd()
        project_folder_f = os.getcwd() + "/"   
    else:
        project_folder = args.project_folder + "/" #folder name only
        # Make sure folder exists
        project_folder_f = os.getcwd() + "/" + project_folder #full path
        
    
    # Make sure we know where to find the crystal/prism input
    if args.crystal is None:
        try:
            # Read crystal information
            geometry = pr.prism(project_folder + "XDEC/MOLECULE.INP")
        except:
            print("\033[91m   Error \033[0m : Failed to locate MOLECULE.INP")
            sys.exit("            Please provide prism input (.d12- or .INP-file) with '-crystal' keyword.") 
    else:
        # Read crystal information
        try:
            geometry = pr.prism(args.crystal)
        except:
            print("\033[91m   Error \033[0m : Failed to locate prism input.")
            sys.exit("            Please provide prism input (.d12- or .INP-file) with '-crystal' keyword.") 

    
    if args.vspace_analysis:
        Nocc = geometry.get_nocc()
                
        # Load crystal coefficients
        C= tmat()
        C.load_old(project_folder_f + "/crystal_reference_state.npy", project_folder_f+"/crystal_reference_coords.npy")
        
        wcenters = np.load(project_folder_f + "/crystal_wannier_centers.npy")
        
        
        Nvirt = C.blockshape[0]-Nocc
        
        lim1 = [0,Nocc]
        lim2 = [Nocc,Nocc+Nvirt]
        
        
        C_occ  = C.blockslice([0,Nvirt],lim1)
        C_virt = C.blockslice([0,Nvirt],lim2)
        
        C_occ.blocks = np.abs(C_occ.blocks)
        C_virt.blocks= np.abs(C_virt.blocks)
        
        D_ij = C_occ.tT().cdot(C_occ)
        D_ia = C_occ.tT().cdot(C_virt)
        
        for c in D_ij.coords:
            if np.sum(c**2)<=0:
                print(c)
                print(D_ia.cget(c))
                print(D_ia.cget(c).shape)
                print(" ")
        #construct R**-1 matrix
        R1 = D_ia*1
        for c in R1.coords:
            w_R = wcenters[Nocc:] - geometry.coor2vec(c)
            
            R1.cset(c, np.sum((wcenters[:Nocc,None] - w_R[None,:])**2, axis = 2)**-1)
        
        print("Distances in refcell")
        print(R1.cget([0,0,0]))
        print("Distances in nextcell")
        print(R1.cget([1,0,0]))
        
        print("Done")
        tensors = [C_occ, C_virt]
        #objective_function = lambda x : (x.tT().cdot(x)).cget([0,0,0])
        c_occ, y_occ, u_occ = at.bi_anneal(objective_function, C_occ,C_virt, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize)
        #at.bi_anneal([D_ia])
        
        
    if args.localize is not None:
        """
        #######################################################################
        ##                                                                   ##
        ##  Stochastic localization algorithms:                              ##
        ##  - PSM-1 (Boys-Foster)                                            ##
        ##  - PSM-N                                                          ##
        ##  - PFM-N                                                          ##
        ##  - Pipek-Mezey                                                    ##
        ##  - Modified Pipek-Mezey                                           ##
        ##                                                                   ##
        ##                                                                   ##
        #######################################################################
        """
        
        #Experimental psm-2 in kspace (full variational freedom)
        safemode = True
        
        # Load crystal overlap matrix, used for coordinate truncation
        try:
            S = tmat()
            S.load(project_folder_f + "/S_crystal.npy")
        except:
            try:
                S = tmat()
                S.load("S_crystal.npy")
            except:
                print("\033[91m   Error \033[0m : Failed to load Crystal overlap matrix")
                sys.exit("            Please provide path to .npy file with '-crystal_overlap' keyword.") 
                
                
        Nocc = geometry.get_nocc()
        
        # Load crystal coefficients
        if args.infile is None:
            C = tmat()
            C.load_old(project_folder_f + "/crystal_reference_state.npy", project_folder_f+"/crystal_reference_coords.npy")
        else:
            C = tmat()
            C.load(args.infile)
            
        # FOR VERIFICATION PURPOSES, LIH specific
        #C.blocks[1:] *= 0 #REMOVE THIS LINE
        #print("Only nonzero:", C.coords[0])
        
        Nvirt = C.blockshape[0] - Nocc
        
        if args.localize == "of1":
            F_of1 = lambda C : np.sum(np.sum(C.coords**2, axis = 1)*np.abs(C.blocks))
            
            
        if args.localize == "ao":
            print("Performing AO-%i localization" % args.m)
            
            PFM_objective_function = lambda C : np.sum((np.diag(np.diag(C[0].cget([0,0,0])))-C[0].cget([0,0,0]))**args.m) - np.sum(np.diag(C[0].cget([0,0,0]))**args.m)
            
            PFM_tensors = [C]
            
            print("Prior to optim:", PFM_objective_function(PFM_tensors))
            
            PFM_tensors_occ = []
            PFM_tensors_virt =[]
            
            lim1 = [0,Nocc]
            lim2 = [Nocc,Nocc+Nvirt]
            
            for i in np.arange(len(PFM_tensors)):
                PFM_tensors_occ.append( PFM_tensors[i].blockslice(lim1,lim1) )
                PFM_tensors_virt.append(PFM_tensors[i].blockslice(lim2,lim2) )
             
            u_occ = np.eye(Nocc, dtype = float)
            u_virt= np.eye(Nvirt, dtype = float)
             
            # optimize occupied orbitals
            if not args.skip_occupied:
                print("Localizing occupied space.")
                c_occ, y_occ, u_occ = at.center_anneal(PFM_objective_function, PFM_tensors_occ, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize, uphill_trigger = 3, order = 1)
            
            print("Localizing virtual space.")
            c_virt, y_virt, u_virt = at.center_anneal(PFM_objective_function, PFM_tensors_virt, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize, uphill_trigger = 3, order = 1)
            
            # assemble unitary matrix and rotated coeffs, store to disk
            
            U_rot = np.zeros(C.blockshape, dtype = float)
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt
            
            
            
            f = open(project_folder_f + "/DEC/basis.g94")
            basis = f.read()
            f.close()
            
            print(pp.generate_basis(basis, [.01, .009, .005, .005, 0.005], [1.5, 1.5, 1.5, 1.5, 1.5], 2,2,1,0,0, enum = "Au") )
            #print("Prior to optim:", PFM_objective_function(PFM_tensors))
            
            
        if args.localize == "occupied-shaping":
            print("Performing occupied-shaped-%i localization" % args.m)
            
            #PFM_objective_function = lambda C : np.sum((np.diag(np.diag(C[0].cget([0,0,0])))-C[0].cget([0,0,0]))**args.m) - np.sum(np.diag(C[0].cget([0,0,0]))**args.m)
            
            
            
            lim1 = [0,Nocc]
            lim2 = [Nocc,Nocc+Nvirt]
            
            #PFM_tensors = [C]
            PFM_tensors = [C.blockslice([0,Nvirt+Nocc],lim2)]
            
            cocc = C.blockslice([0,Nvirt+Nocc],lim1).cget([0,0,0])
            
            PFM_objective_function = lambda C : np.sum((C[0].cget([0,0,0])[:,:Nocc]-cocc)**args.m)
            
            print("Prior to optim:", PFM_objective_function(PFM_tensors))
            

             
            u_occ = np.eye(Nocc, dtype = float)
            u_virt= np.eye(Nvirt, dtype = float)
             
           
            print("Localizing virtual space.")
            c_virt, y_virt, u_virt = at.center_anneal(PFM_objective_function,PFM_tensors , T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize, uphill_trigger = 3, order = 1)
            
            # assemble unitary matrix and rotated coeffs, store to disk
            
            U_rot = np.zeros(C.blockshape, dtype = float)
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt
            
            
            
            f = open(project_folder_f + "/DEC/basis.g94")
            basis = f.read()
            f.close()
            
            print(pp.generate_basis(basis, [.01, .009, .005, .005, 0.005], [1.5, 1.5, 1.5, 1.5, 1.5], 2,2,1,0,0, enum = "Au") )
        
        if args.localize == "pfm":
            print("Performing PFM-%s localization." % args.m)
            if args.integral_file is None:
                of = objective_function(geometry)
            
                PFM_tensors, PFM, PSM, wcenters = of.pfm(C, S)
                
                np.save(project_folder_f + "/PFM_tensors.npy",
                        np.array([PFM_tensors, PFM, PSM, wcenters]))
            else:
                PFM_tensors, PFM, PSM, wcenters = np.load(args.integral_file)
            
            PFM_objective_function = lambda X : np.sum(np.diag(assemble_pfm_function(X).cget([0,0,0]))**args.m)
            
            PFM      = assemble_pfm_function(PFM_tensors)
            pfm_vals = np.diag(PFM.cget([0,0,0]))
            
            #pfm_vals = np.diag(PFM.cget([0,0,0]))
            #print(pfm_vals, pfm_vals**.5, pfm_vals**2)
            #print(PFM.cget([0,0,0]))
            print(pfm_vals)
            print("###############################")
            print("# Initial Wannier functions   #")
            print("###############################")
            spreads_occ = np.sqrt(np.diag(PSM[:Nocc,:Nocc]))
            spreads_virt= np.sqrt(np.diag(PSM[Nocc:,Nocc:]))
            summarize_orbitals(spreads_occ, spreads_virt, wcenters)
            
            print("PFM prior to optimization:", np.sum(pfm_vals))
            
            PFM_tensors_occ = []
            PFM_tensors_virt =[]
            
            lim1 = [0,Nocc]
            lim2 = [Nocc,Nocc+Nvirt]
            
            for i in np.arange(len(PFM_tensors)):
                PFM_tensors_occ.append( PFM_tensors[i].blockslice(lim1,lim1) )
                PFM_tensors_virt.append(PFM_tensors[i].blockslice(lim2,lim2) )
             
            u_occ = np.eye(Nocc, dtype = float)
            u_virt= np.eye(Nvirt, dtype = float)
             
            # optimize occupied orbitals
            if not args.skip_occupied:
                print("Localizing occupied space.")
                c_occ, y_occ, u_occ = at.center_anneal(PFM_objective_function, PFM_tensors_occ, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize, uphill_trigger = 3)
            
            print("Localizing virtual space.")
            c_virt, y_virt, u_virt = at.center_anneal(PFM_objective_function, PFM_tensors_virt, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize, uphill_trigger = 3)
            
            # assemble unitary matrix and rotated coeffs, store to disk
            
            U_rot = np.zeros(C.blockshape, dtype = float)
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt
            
            
            
            
            

            
            
        
        if args.localize == "psm":
            print("Performing PSM-%s localization." % args.m)
            safemode = True
            # Compute localization data
            
            
            if args.integral_file is None:
                of = objective_function(geometry)
            

                XYZ2mo, Xmo, Ymo, Zmo, wcenters = of.foster_boys(C, S)
                #for n in PFM_tensor:
                #    print(n)
                np.save(project_folder_f + "/PSM_tensors.npy", np.array([XYZ2mo, Xmo, Ymo, Zmo, wcenters]))
                #np.save(project_folder_f + "/PFM_tensors.npy", np.array([PFM_tensors, PFM, PSM, wcenters]))
            else:
                #PFM_tensors, PFM, PSM, wcenters = np.load(args.integral_file)
                XYZ2mo, Xmo, Ymo, Zmo, wcenters = np.load(args.integral_file)
            #of = objective_function(project_folder_f)
            
            #XYZ2mo, Xmo, Ymo, Zmo, wcenters = of.foster_boys(C = C)
            
            
            # Get reference cell only
            L0_psm = (XYZ2mo - Xmo**2 - Ymo**2 - Zmo**2).cget([0,0,0])     
            Nvirt = L0_psm.shape[0]-Nocc
            # Occupied region
            Locc = L0_psm[:Nocc,:Nocc]
    
            # Virtual region
            Lvirt = L0_psm[Nocc:, Nocc:]
            
            # A unitary matrix (initially the identity matrix)
            U_rot = np.eye(len(L0_psm))
            
            print("###############################")
            print("# Initial Wannier functions   #")
            print("###############################")
            spreads_occ = np.sqrt(np.diag(Locc))
            spreads_virt= np.sqrt(np.diag(Lvirt))
    

            
            summarize_orbitals(spreads_occ, spreads_virt, wcenters)
            
            # The objective function
            F_psm = lambda X : np.sum(np.diag(X[0].cget([0,0,0]) - X[1].cget([0,0,0])**2 - X[2].cget([0,0,0])**2 - X[3].cget([0,0,0])**2)**args.m) # PSM-2 objective function
        
            #F_psm = lambda L_psm : np.max(np.diag(L_psm)) # PSM-2 objective function
            
            #F_psm = lambda L_psm : np.std((L_psm-np.diag(np.diag(L_psm)))**2) #PSM functional
            
            #F_psm = lambda L_psm : np.sum(np.diag(L_psm))  # PSM-1 objective function
            
            print("###############################")
            print("# Stochastic optimization     #")
            print("###############################")

            
            # Optimize occupied and virtual separately
            lim1 = [0,Nocc]
            XYZ2mo_occ = XYZ2mo.blockslice(lim1,lim1)   
            Xmo_occ = Xmo.blockslice(lim1,lim1)
            Ymo_occ = Ymo.blockslice(lim1,lim1)
            Zmo_occ = Zmo.blockslice(lim1,lim1)
            
            lim2 = [Nocc, Nvirt+Nocc]
            XYZ2mo_virt = XYZ2mo.blockslice(lim2,lim2)
            Xmo_virt = Xmo.blockslice(lim2,lim2)
            Ymo_virt = Ymo.blockslice(lim2,lim2)
            Zmo_virt = Zmo.blockslice(lim2,lim2)
            
            print("Optimizing occupied space")
            
            
            
            #c_occ, y_occ, u_occ = at.matrix_anneal(F_psm, XYZ2mo_occ, Xmo_occ, Ymo_occ, Zmo_occ, Locc, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize)
            #c_occ, y_occ, u_occ = at.center_anneal(F_psm, [XYZ2mo_occ, Xmo_occ, Ymo_occ, Zmo_occ], T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize)
            c_occ, y_occ, u_occ = at.ensemble_annealing(objective_function = F_psm,
                                                        tensors = [XYZ2mo_occ,
                                                                   Xmo_occ,
                                                                   Ymo_occ,
                                                                   Zmo_occ],
                                                        initial_temperature = 20.0,
                                                        N_steps_total = args.nsteps,
                                                        N_walkers = args.nwalkers,
                                                        T_decay = 0.975,
                                                        stepsize = args.stepsize,
                                                        order = 2,
                                                        uphill_trigger = 0)
            
            print("\n")
            print("Optimizing virtual space")
            #c_virt, y_virt, u_virt = at.matrix_anneal(F_psm, XYZ2mo_virt, Xmo_virt, Ymo_virt, Zmo_virt, Lvirt, T=20.0, N_steps = args.nsteps, T_decay =0.975,stepsize = args.stepsize)
            #c_virt, y_virt, u_virt = at.center_anneal(F_psm, [XYZ2mo_virt, Xmo_virt, Ymo_virt, Zmo_virt],  T=20.0, N_steps = args.nsteps, T_decay =0.975,stepsize = args.stepsize)
            c_virt, y_virt, u_virt = at.ensemble_annealing(objective_function = F_psm,
                                                           tensors = [XYZ2mo_virt,
                                                                      Xmo_virt,
                                                                      Ymo_virt,
                                                                      Zmo_virt],
                                                           initial_temperature = 20.0,
                                                           N_steps_total = args.nsteps,
                                                           N_walkers = args.nwalkers,
                                                           T_decay = 0.975,
                                                           stepsize = args.stepsize,
                                                           order = 2,
                                                           uphill_trigger = 0)
            
            print("\n"*3)
            
            print("###############################")
            print("# Optimized Wannier functions #")
            print("###############################")
            
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt        
            
           
            X = [XYZ2mo, Xmo, Ymo, Zmo]
            X2 = []
            for i in np.arange(len(X)):
                X2.append(X[i]*1)
                X2[-1].cset(np.array([0,0,0]), np.dot(U_rot.T, np.dot(X2[-1].cget([0,0,0]), U_rot)))
            spreads = np.sqrt( np.diag(X2[0].cget([0,0,0]) - X2[1].cget([0,0,0])**2 - X2[2].cget([0,0,0])**2 - X2[3].cget([0,0,0])**args.m ))
            
            wcenters[0] = np.diag(X2[1].cget([0,0,0]))
            wcenters[1] = np.diag(X2[2].cget([0,0,0]))
            wcenters[2] = np.diag(X2[3].cget([0,0,0]))
            
            
            summarize_orbitals(spreads[:Nocc], spreads[Nocc:], wcenters)
    
            
            print("Objective function before optimization: %.5e" % F_psm(X))
            print("Objective function after optimization : %.5e" % F_psm(X2))
            print("Deviation from unity (rotation matrix): %.5e"  % toeplitz.L2norm(np.dot(U_rot.T,U_rot)-np.diag(np.ones(U_rot.shape[0]))))
    
            
        if args.localize == "mute-orbital":
            
            p_mute = np.sum(np.abs(C.cget([0,0,0])), axis = 0).argmin() #target orbital
            
            norbs = 1
            p_mutes = np.sum(np.abs(C.cget([0,0,0]))[:,Nocc:], axis = 0).argsort()[:norbs]
            
            print(p_mutes)
            
            
            
            PFM_objective_function = lambda C : np.sum(np.abs(C[0].cget([0,0,0]))[:,p_mutes])
            
            
            
            PFM_tensors = [C]
            
            print("Prior to optim:", PFM_objective_function(PFM_tensors))
            
            PFM_tensors_occ = []
            PFM_tensors_virt =[]
            
            lim1 = [0,Nocc]
            lim2 = [Nocc,Nocc+Nvirt]
            
             
            for i in np.arange(len(PFM_tensors)):
                PFM_tensors_occ.append( PFM_tensors[i].blockslice([0,Nvirt+Nocc],lim1) )
                PFM_tensors_virt.append(PFM_tensors[i].blockslice([0,Nvirt+Nocc],lim2) )
             
            u_occ = np.eye(Nocc, dtype = float)
            u_virt= np.eye(Nvirt, dtype = float)
             
            
            print("Localizing virtual space.")
            c_virt, y_virt, u_virt = at.center_anneal(PFM_objective_function, PFM_tensors_virt, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize, uphill_trigger = 3, order = 1)
            
            # assemble unitary matrix and rotated coeffs, store to disk
            
            U_rot = np.zeros(C.blockshape, dtype = float)
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt
            
        if args.localize == "same-center":
            safemode = True
            # Compute localization data
            of = objective_function(geometry)
            
            XYZ2mo, Xmo, Ymo, Zmo, wcenters = of.foster_boys( C, S)
            
            
            # Get reference cell only
            L0_psm = (XYZ2mo - Xmo**2 - Ymo**2 - Zmo**2).cget([0,0,0])     
            Nvirt = L0_psm.shape[0]-Nocc
            # Occupied region
            Locc = L0_psm[:Nocc,:Nocc]
    
            # Virtual region
            Lvirt = L0_psm[Nocc:, Nocc:]
            
            # A unitary matrix (initially the identity matrix)
            U_rot = np.eye(len(L0_psm))
            
            print("###############################")
            print("# Initial Wannier functions   #")
            print("###############################")
            spreads_occ = np.sqrt(np.diag(Locc))
            spreads_virt= np.sqrt(np.diag(Lvirt))
    
    
            #wannier data for comparison
            
            ccenters_occ, cspreads_occ = get_wannier_data(project_folder_f + "/wan_log_occ.txt")
            ccenters_virt, cspreads_virt = get_wannier_data(project_folder_f + "/wan_log_virt.txt")
            
            cspreads_occ = np.sqrt(cspreads_occ)
            cspreads_virt = np.sqrt(cspreads_virt)
            
            summarize_orbitals(spreads_occ, spreads_virt, wcenters)
            
            # The objective function
            F_psm = lambda X : np.std(np.diag(X[1].cget([0,0,0]))) + np.std(np.diag(X[2].cget([0,0,0]))) + np.std(np.diag(X[3].cget([0,0,0])))
        
            #F_psm = lambda L_psm : np.max(np.diag(L_psm)) # PSM-2 objective function
            
            #F_psm = lambda L_psm : np.std((L_psm-np.diag(np.diag(L_psm)))**2) #PSM functional
            
            #F_psm = lambda L_psm : np.sum(np.diag(L_psm))  # PSM-1 objective function
            
            print("###############################")
            print("# Stochastic optimization     #")
            print("###############################")
            
            """
            XYZ2mo_occ = XYZ2mo.cget([0,0,0])[:Nocc,:Nocc]
            Xmo_occ = Xmo.cget([0,0,0])[:Nocc,:Nocc]
            Ymo_occ = Ymo.cget([0,0,0])[:Nocc,:Nocc]
            Zmo_occ = Zmo.cget([0,0,0])[:Nocc,:Nocc]
            
            XYZ2mo_virt = XYZ2mo.cget([0,0,0])[Nocc:,Nocc:]
            Xmo_virt = Xmo.cget([0,0,0])[Nocc:,Nocc:]
            Ymo_virt = Ymo.cget([0,0,0])[Nocc:,Nocc:]
            Zmo_virt = Zmo.cget([0,0,0])[Nocc:,Nocc:]
            """
            
            # Optimize occupied and virtual separately
            lim1 = [0,Nocc]
            XYZ2mo_occ = XYZ2mo.blockslice(lim1,lim1)   
            Xmo_occ = Xmo.blockslice(lim1,lim1)
            Ymo_occ = Ymo.blockslice(lim1,lim1)
            Zmo_occ = Zmo.blockslice(lim1,lim1)
            
            lim2 = [Nocc, Nvirt+Nocc]
            XYZ2mo_virt = XYZ2mo.blockslice(lim2,lim2)
            Xmo_virt = Xmo.blockslice(lim2,lim2)
            Ymo_virt = Ymo.blockslice(lim2,lim2)
            Zmo_virt = Zmo.blockslice(lim2,lim2)
            
            print("Optimizing occupied space")
            
            
            
            #c_occ, y_occ, u_occ = at.matrix_anneal(F_psm, XYZ2mo_occ, Xmo_occ, Ymo_occ, Zmo_occ, Locc, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize)
            c_occ, y_occ, u_occ = at.center_anneal(F_psm, [XYZ2mo_occ, Xmo_occ, Ymo_occ, Zmo_occ], T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize)
            
            print("\n")
            print("Optimizing virtual space")
            #c_virt, y_virt, u_virt = at.matrix_anneal(F_psm, XYZ2mo_virt, Xmo_virt, Ymo_virt, Zmo_virt, Lvirt, T=20.0, N_steps = args.nsteps, T_decay =0.975,stepsize = args.stepsize)
            c_virt, y_virt, u_virt = at.center_anneal(F_psm, [XYZ2mo_virt, Xmo_virt, Ymo_virt, Zmo_virt],  T=20.0, N_steps = args.nsteps, T_decay =0.975,stepsize = args.stepsize)
            
            print("\n"*3)
            
            print("###############################")
            print("# Optimized Wannier functions #")
            print("###############################")
            
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt        
            
            
            X = [XYZ2mo, Xmo, Ymo, Zmo]
            X2 = []
            for i in np.arange(len(X)):
                X2.append(X[i]*1)
                X2[-1].cset(np.array([0,0,0]), np.dot(U_rot.T, np.dot(X2[-1].cget([0,0,0]), U_rot)))
            spreads = np.sqrt( np.diag(X2[0].cget([0,0,0]) - X2[1].cget([0,0,0])**2 - X2[2].cget([0,0,0])**2 - X2[3].cget([0,0,0])**2 ))
            
            wcenters[0] = np.diag(X2[1].cget([0,0,0]))
            wcenters[1] = np.diag(X2[2].cget([0,0,0]))
            wcenters[2] = np.diag(X2[3].cget([0,0,0]))
            
            
            summarize_orbitals(spreads[:Nocc], spreads[Nocc:], wcenters)
    
            
            print("Objective function before optimization: %.5e" % F_psm(X))
            print("Objective function after optimization : %.5e" % F_psm(X2))
            print("Deviation from unity (rotation matrix): %.5e"  %
                  toeplitz.L2norm(np.dot(U_rot.T, U_rot)-np.diag(np.ones(U_rot.shape[0]))))
    


            x0 = wcenters[0,Nocc:].mean()
            y0 = wcenters[1,Nocc:].mean()
            z0 = wcenters[2,Nocc:].mean()
            
            new_center = np.array([x0,y0,z0])
            
            print("WARNING: USING CRAPPY OVERLAP MATRIX")
            S= tmat()
            S.load_old(project_folder_f + "/crystal_overlap_matrix.npy", project_folder_f+"/crystal_overlap_coords.npy")
            
            
            basis = open(project_folder_f + "/DEC/basis.g94", "r").read()
            #new_basis, N_ao = pp.generate_basis(basis, [.5, .9, .5, .005, 0.005], [1.5, 1.5, 1.5, 1.5, 1.5], 10,1,0,0,0)
            new_basis, N_ao = pp.generate_basis(basis, [.5, .5, .5, .005, 0.005], [2,2,2,2,2], 20,10,1,0,0)
            
            #print(pp.compute_intermediate_overlap_cell(project_folder, new_basis, cell = [[0,0,0]], tpe = "overlap")[0].shape)
            
            O = toeplitz.get_zero_tmat(np.max(np.abs(S.coords), axis = 0), (S.blockshape[0], N_ao))
            S_1t = toeplitz.get_zero_tmat(np.max(np.abs(S.coords), axis = 0), (N_ao, N_ao))
            for c in S.coords:
                S_1c, Oc = pp.compute_intermediate_overlap_cell(project_folder, new_basis, cell = [c], tpe = "overlap", s_1 = False, center = new_center)
                O.cset(c, S_1c)
                
            S_1c, Oc = pp.compute_intermediate_overlap_cell(project_folder, new_basis, cell = [[0,0,0]], tpe = "overlap", s_1 = True, center = new_center)

            OtC = O.tT().cdot(C, coords = np.array([[0,0,0]]))
            
            CtO = OtC.tT()
            
            #S_1t = S*1
            S_1t = toeplitz.get_zero_tmat(np.max(np.abs(S.coords), axis = 0), (N_ao, N_ao))
            #S_1t.blocks[1:] *= 0
            #print(Oc)
            
            invOc = np.linalg.inv(Oc).T
            #print(invOc)
            S_1t.cset(np.array([0,0,0]), invOc)
            #
            S_1t = S_1t.tT()
            
            #print(S_1t.cget([0,0,0]))
            
            F_ = CtO.cdot(S_1t.cdot(OtC, coords = OtC.coords), coords = OtC.coords)
            


            print("Residual norm")
            print(np.diag(F_.cget([0,0,0])))
            print("basis")
            print(new_basis)
            

            
            
            lim1 = [0,Nocc]
            O_occ = [F_.blockslice(lim1,lim1)]

            
            lim2 = [Nocc, Nvirt+Nocc]
            O_virt = [F_.blockslice(lim2,lim2)]

            
            F = lambda X : -np.sum(np.diag(X[0].cget([0,0,0]))[:-1])
            #F = lambda X : np.std(np.diag(X[0].cget([0,0,0])))
            
            c_occ, y_occ, u_occ = at.center_anneal(F, O_occ, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize)

            c_virt, y_virt, u_virt = at.center_anneal(F, O_virt, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize)
            U_rot = np.zeros(C.blockshape, dtype = float)
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt
            
            #print(np.dot(U_rot.T, np.dot(F_.cget([0,0,0]), U_rot)))
            #print(np.diag(F_.cget([0,0,0])))
            print(np.diag(np.dot(U_rot.T, np.dot(F_.cget([0,0,0]), U_rot))))
            
            
            
            
            
        if args.localize == "refcell-fitting":
            # Load crystal overlap matrix
            print("WARNING: USING CRAPPY OVERLAP MATRIX")
            S= tmat()
            S.load_old(project_folder_f + "/crystal_overlap_matrix.npy", project_folder_f+"/crystal_overlap_coords.npy")
            
            
            #intermediate basis
            basis = open(project_folder_f + "/DEC/basis.g94", "r").read()
            #new_basis, N_ao = pp.generate_basis(basis, [.5, .9, .5, .005, 0.005], [1.5, 1.5, 1.5, 1.5, 1.5], 10,1,0,0,0)
            new_basis, N_ao = pp.generate_basis(basis, [.5, .5, .5, .005, 0.005], [2,2,2,2,2], 20,10,1,0,0)
            #new_basis, N_ao = pp.generate_basis_grid(basis, [3,3,3], [2,2,2,2,2], 20,10,1,0,0)
            
            
            #print(pp.compute_intermediate_overlap_cell(project_folder, new_basis, cell = [[0,0,0]], tpe = "overlap")[0].shape)
            
            O = toeplitz.get_zero_tmat(np.max(np.abs(S.coords), axis = 0), (S.blockshape[0], N_ao))
            S_1t = toeplitz.get_zero_tmat(np.max(np.abs(S.coords), axis = 0), (N_ao, N_ao))
            for c in S.coords:
                S_1c, Oc = pp.compute_intermediate_overlap_cell(project_folder, new_basis, cell = [c], tpe = "overlap", s_1 = False)
                O.cset(c, S_1c)
                
            S_1c, Oc = pp.compute_intermediate_overlap_cell(project_folder, new_basis, cell = [[0,0,0]], tpe = "overlap", s_1 = True)

            OtC = O.tT().cdot(C, coords = np.array([[0,0,0]]))
            
            CtO = OtC.tT()
            
            #S_1t = S*1
            S_1t = toeplitz.get_zero_tmat(np.max(np.abs(S.coords), axis = 0), (N_ao, N_ao))
            #S_1t.blocks[1:] *= 0
            #print(Oc)
            
            invOc = np.linalg.inv(Oc).T
            #print(invOc)
            S_1t.cset(np.array([0,0,0]), invOc)
            #
            S_1t = S_1t.tT()
            
            #print(S_1t.cget([0,0,0]))
            
            F_ = CtO.cdot(S_1t.cdot(OtC, coords = OtC.coords), coords = OtC.coords)
            


            print("Residual norm")
            print(np.diag(F_.cget([0,0,0])))
            print("basis")
            print(new_basis)
            
            
            
            lim1 = [0,Nocc]
            O_occ = [F_.blockslice(lim1,lim1)]

            
            lim2 = [Nocc, Nvirt+Nocc]
            O_virt = [F_.blockslice(lim2,lim2)]

            
            F = lambda X : -np.sum(np.diag(X[0].cget([0,0,0])))
            #F = lambda X : np.std(np.diag(X[0].cget([0,0,0])))
            
            c_occ, y_occ, u_occ = at.center_anneal(F, O_occ, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize)

            c_virt, y_virt, u_virt = at.center_anneal(F, O_virt, T=20.0, T_decay = 0.975, N_steps = args.nsteps, stepsize = args.stepsize)
            U_rot = np.zeros(C.blockshape, dtype = float)
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt
            
            #print(np.dot(U_rot.T, np.dot(F_.cget([0,0,0]), U_rot)))
            #print(np.diag(F_.cget([0,0,0])))
            print(np.diag(np.dot(U_rot.T, np.dot(F_.cget([0,0,0]), U_rot))))
            
        if args.localize == "grid-fitting":
            # Load crystal overlap matrix
            print("WARNING: USING CRAPPY OVERLAP MATRIX")
            S= tmat()
            S.load_old(project_folder_f + "/crystal_overlap_matrix.npy", project_folder_f+"/crystal_overlap_coords.npy")
            
            
            #intermediate basis
            basis = open(project_folder_f + "/DEC/basis.g94", "r").read()
            #new_basis, N_ao = pp.generate_basis(basis, [.5, .9, .5, .005, 0.005], [1.5, 1.5, 1.5, 1.5, 1.5], 10,1,0,0,0)
            new_basis, N_ao = pp.generate_basis(basis, [.1, .5, .5, .005, 0.005], [2,0,0,0,0], 1,0,0,0,0)
            #new_basis, N_ao = pp.generate_basis_grid(basis, [3,3,3], [2,2,2,2,2], 20,10,1,0,0)
            
            grid = np.array([6,6,6])
            #print(pp.compute_intermediate_overlap_cell(project_folder, new_basis, cell = [[0,0,0]], tpe = "overlap")[0].shape)
            
            N_ao *= np.prod(grid)
            
            O = toeplitz.get_zero_tmat(np.max(np.abs(S.coords), axis = 0), (S.blockshape[0], N_ao))
            S_1t = toeplitz.get_zero_tmat(np.max(np.abs(S.coords), axis = 0), (N_ao, N_ao))
            for c in S.coords:
                S_1c, Oc = pp.compute_intermediate_overlap_cell(project_folder, new_basis, cell = [c], tpe = "overlap", s_1 = False, grid = grid)
                O.cset(c, S_1c)
                
            S_1c, Oc = pp.compute_intermediate_overlap_cell(project_folder, new_basis, cell = [[0,0,0]], tpe = "overlap", s_1 = True, grid = grid)

            OtC = O.tT().cdot(C, coords = np.array([[0,0,0]]))
            
            CtO = OtC.tT()
            
            #S_1t = S*1
            S_1t = toeplitz.get_zero_tmat(np.max(np.abs(S.coords), axis = 0), (N_ao, N_ao))
            #S_1t.blocks[1:] *= 0
            #print(Oc)
            
            invOc = np.linalg.inv(Oc).T
            #print(invOc)
            S_1t.cset(np.array([0,0,0]), invOc)
            #
            S_1t = S_1t.tT()
            
            #print(S_1t.cget([0,0,0]))
            
            F_ = CtO.cdot(S_1t.cdot(OtC, coords = OtC.coords), coords = OtC.coords)
            


            print("Residual norm")
            print(np.diag(F_.cget([0,0,0])))
            
              

        if args.localize == "modified-pm" or args.localize == "pipek-mezey":
            
            
            
            # # ONLY FOR TESTING!
            # C = tmat()
            # C.load_old(project_folder_f + "/stochastic_reference_state.npy",
            #            project_folder_f + "/stochastic_reference_coords.npy")
            # # C = tmat()
            # # C.load_old(project_folder_f + "/lsdalton_reference_state.npy",
            # #            project_folder_f + "/lsdalton_reference_coords.npy")
            # # END ONLY FOR TESTING
            
            # Load crystal coefficients
            ao_overlaps = tmat()
            ao_overlaps.load_old(project_folder_f + "/crystal_overlap_matrix.npy",
                                 project_folder_f + "/crystal_overlap_coords.npy")
            
            # Normalize the MO coefficients
            mo_coeffs_n = of.normalize_orbitals(ao_overlaps, C)
            
            # Compute the S * C matrix
            bfs_per_atom, sc_matrix = mpm.setup_scmatrix(project_folder_f,
                                                         mo_coeffs_n,
                                                         ao_overlaps)
            
            # MO coefficients and SC matrices for occupied and virtuals
            # separately
            occ_coeffs = mo_coeffs_n.blockslice((0, C.blockshape[0]),
                                                (0, Nocc))
            virt_coeffs = mo_coeffs_n.blockslice((0, C.blockshape[0]),
                                                 (Nocc, C.blockshape[1]))
            sc_occ = sc_matrix.blockslice((0, C.blockshape[0]),
                                          (0, Nocc))
            sc_virt = sc_matrix.blockslice((0, C.blockshape[0]),
                                           (Nocc, C.blockshape[1]))
            
            
            # The Pipek-Mezey objective function 
            if args.localize == "modified-pm":
                
                def pm_functional_occ(rotation):
                    
                    U = rotation[0].cget([0, 0, 0])
                    
                    occ_coeffs_r = at.rotate(occ_coeffs, U)
                    sc_occ_r = at.rotate(sc_occ, U)
                    
                    return mpm.modified_pm_functional(occ_coeffs_r,
                                                      sc_occ_r,
                                                      bfs_per_atom,
                                                      epsilon = 0.0005)
                
                def pm_functional_virt(rotation):
                    
                    U = rotation[0].cget([0, 0, 0])
                    
                    virt_coeffs_r = at.rotate(virt_coeffs, U)
                    sc_virt_r = at.rotate(sc_virt, U)
                    
                    return mpm.modified_pm_functional(virt_coeffs_r,
                                                      sc_virt_r,
                                                      bfs_per_atom,
                                                      epsilon = 0.0005)
                
            elif args.localize == "pipek-mezey":
                
                def pm_functional_occ(rotation):
                    
                    U = rotation[0].cget([0, 0, 0])
                    
                    occ_coeffs_r = at.rotate(occ_coeffs, U)
                    sc_occ_r = at.rotate(sc_occ, U)
                    
                    return mpm.original_pm_functional(occ_coeffs_r,
                                                      sc_occ_r,
                                                      bfs_per_atom)
                
                def pm_functional_virt(rotation):
                    
                    U = rotation[0].cget([0, 0, 0])
                    
                    virt_coeffs_r = at.rotate(virt_coeffs, U)
                    sc_virt_r = at.rotate(sc_virt, U)
                    
                    return mpm.original_pm_functional(virt_coeffs_r,
                                                      sc_virt_r,
                                                      bfs_per_atom)
                
                
            rotation_occ  = tmat(coords = np.array([[0, 0, 0]], dtype = int),
                                 blocks = np.array([np.eye(Nocc).tolist()],
                                                   dtype = float)) 
            rotation_virt = tmat(coords = np.array([[0, 0, 0]], dtype = int),
                                 blocks = np.array([np.eye(Nvirt).tolist()],
                                                   dtype = float)) 
            
            
            
            print("###############################")
            print("# Stochastic optimization     #")
            print("###############################")
            
            # Recommendation as a starting point: -stepsize=0.01

            n_trigger = 10 #int(0.0 * args.nsteps / args.nwalkers)
            
            print("Optimizing occupied space")
            
            c_occ, y_occ, u_occ = at.ensemble_annealing(objective_function = pm_functional_occ,
                                                        tensors = [rotation_occ],
                                                        initial_temperature = 1.0,
                                                        N_steps_total = args.nsteps,
                                                        N_walkers = args.nwalkers,
                                                        T_decay = 0.997,
                                                        stepsize = args.stepsize,
                                                        order = 1,
                                                        uphill_trigger = n_trigger)
            #u_occ = rotation_occ.get([0, 0, 0])
            
            print("\n")
            print("Optimizing virtual space")
            
            c_virt, y_virt, u_virt = at.ensemble_annealing(objective_function = pm_functional_virt,
                                                           tensors = [rotation_virt],
                                                           initial_temperature = 1.0,
                                                           N_steps_total = args.nsteps,
                                                           N_walkers = args.nwalkers,
                                                           T_decay = 0.997,
                                                           stepsize = args.stepsize,
                                                           order = 1,
                                                           uphill_trigger = n_trigger)
            
            print("\n"*3)
            
            print("###############################")
            print("# Optimized Wannier functions #")
            print("###############################")

            # Rotation matrix 
            U_rot = np.zeros(C.blockshape, dtype=float)
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt 
            

            
            
        if args.localize == "kspace-psm-2":
            # Compute localization data
            of = objective_function(geometry)
            
            XYZ2mo, Xmo, Ymo, Zmo, wcenters = of.foster_boys( C, S)
            
            
            # Get reference cell only
            L0_psm = (XYZ2mo - Xmo**2 - Ymo**2 - Zmo**2).cget([0,0,0])     
            
            Nvirt = L0_psm.shape[0]-Nocc
            
            # Occupied region
            Locc = L0_psm[:Nocc,:Nocc]
    
            # Virtual region
            Lvirt = L0_psm[Nocc:, Nocc:]
            
            # A unitary matrix (initially the identity matrix)
            U_rot = np.eye(len(L0_psm), dtype = complex)
            
            print("###############################")
            print("# Initial Wannier functions   #")
            print("###############################")
            spreads_occ = np.sqrt(np.diag(Locc))
            spreads_virt= np.sqrt(np.diag(Lvirt))
    
    
            #wannier data for comparison
            
            ccenters_occ, cspreads_occ = get_wannier_data(project_folder_f + "/wan_log_occ.txt")
            ccenters_virt, cspreads_virt = get_wannier_data(project_folder_f + "/wan_log_virt.txt")
            
            cspreads_occ = np.sqrt(cspreads_occ)
            cspreads_virt = np.sqrt(cspreads_virt)
            
            summarize_orbitals(spreads_occ, spreads_virt, wcenters)
    
            
            # The objective function
            #F_psm = lambda L_psm : np.sum(np.diag(L_psm.cget([0,0,0]))**2) # PSM-2 objective function
            
            print("Number of occupied orbitals:", Nocc)
            print("Number of virtual orbitals :", Nvirt)
            print("Blockshape                 :", Xmo.blockshape, Xmo.blocks[0].shape)
            
            #F_psm = lambda M : np.sum(np.sum(M.coords**2,axis=1)[:,None,None]*M.blocks[:-1]) #inward rotations
    
            #F_psm = lambda L_psm : np.max(np.diag(L_psm)) # PSM-2 objective function
            
            #F_psm = lambda L_psm : np.std((L_psm-np.diag(np.diag(L_psm)))**2) #PSM functional
            
            #F_psm = lambda L_psm : np.sum(np.diag(L_psm))  # PSM-1 objective function
            
            # X = [XYZ2mo, Xmo, Ymo, Zmo]
            F_psm = lambda X : np.sum(np.diag((X[0] - X[1]**2 - X[2]**2 - X[3]**2).cget([0,0,0]))**2)
            
            print("###############################")
            print("# Stochastic optimization     #")
            print("###############################")
            
            
            # Optimize occupied and virtual separately
            lim1 = [0,Nocc]
            XYZ2mo_occ = XYZ2mo.blockslice(lim1,lim1)   
            Xmo_occ = Xmo.blockslice(lim1,lim1)
            Ymo_occ = Ymo.blockslice(lim1,lim1)
            Zmo_occ = Zmo.blockslice(lim1,lim1)
            
            lim2 = [Nocc, Nvirt+Nocc]
            XYZ2mo_virt = XYZ2mo.blockslice(lim2,lim2)
            Xmo_virt = Xmo.blockslice(lim2,lim2)
            Ymo_virt = Ymo.blockslice(lim2,lim2)
            Zmo_virt = Zmo.blockslice(lim2,lim2)
            
            print("Optimizing occupied space")
            
            c_occ, y_occ, u_occ = at.kspace_anneal(F_psm, [XYZ2mo_occ, Xmo_occ, Ymo_occ, Zmo_occ],
                                                   T = 20.0, T_decay = 0.975, N_steps = 15000,
                                                   stepsize = args.stepsize)

            
            
            print("\n")
            print("Optimizing virtual space")

            c_virt, y_virt, u_virt = at.kspace_anneal(F_psm, [XYZ2mo_virt, Xmo_virt, Ymo_virt, Zmo_virt],
                                                      T = 20.0, N_steps = 15000, T_decay = 0.975,
                                                      stepsize = args.stepsize)

            
            print("\n"*3)
            
            print("optimization done.")
            
            print("###############################")
            print("# Optimized Wannier functions #")
            print("###############################")
            
            U_rot[:Nocc, :Nocc] = u_occ
            U_rot[Nocc:, Nocc:] = u_virt        
            
            X2 = [XYZ2mo*1, Xmo*1, Ymo*1, Zmo*1]
            
            for i in np.arange(len(X2)):
                xnew =  np.fft.fft(X2[i].blocks, axis = 0)
                for j in np.arange(xnew.shape[0]):
                    xnew[j] = np.dot(U_rot.T.conj(), np.dot(xnew[j], U_rot))
                X2[i].blocks = np.fft.ifft(xnew, axis = 0).real
                
            L0 = np.sqrt(np.diag((X2[0] - X2[1]**2 - X2[2]**2 - X2[3]**2).cget([0,0,0])))
            
            spreads_occ = L0[:Nocc]
            spreads_virt = L0[Nocc:]
            
            wcenters[0] = np.diag(X2[1].cget([0,0,0]))
            wcenters[1] = np.diag(X2[2].cget([0,0,0]))
            wcenters[2] = np.diag(X2[3].cget([0,0,0]))
            
            summarize_orbitals(spreads_occ, spreads_virt, wcenters)
                        
            #C.save(project_folder_f + "/rotated_coeffs")
            
            #np.save(project_folder_f + "/rotation_matrix.npy", U_rot)
            
        # Save results
        
        if args.outfile!=None:
            save_rotation(C, U_rot, project_folder_f, args.outfile, printing = True)
        else:
            save_rotation(C, U_rot, os.getcwd(), "c_rot", printing = True)
        print("Deviation from unity (rotation matrix): %.5e"  %
              toeplitz.L2norm(np.dot(U_rot.T, U_rot) - np.diag(np.ones(U_rot.shape[0]))))
        
        
    if args.build_fock:
        # build AO fock matrix
        pp.fock_tmat(project_folder_f)
        pp.Z.cget([0,0,0])
    
    if args.consistency_check:
        
        
        C= tmat()
        C.load_old(project_folder_f + "/crystal_reference_state.npy",
                   project_folder_f + "/crystal_reference_coords.npy")
        
        UC=tmat()
        UC.load_old(project_folder_f + "/crystal_reference_state.npy",
                    project_folder_f + "/crystal_reference_coords.npy")
        
        U_rot = np.load(project_folder_f + "/rotation_matrix.npy")
        for i in np.arange(C.blocks.shape[0]):
            UC.blocks[i] = np.dot(C.blocks[i], U_rot)
        
        #print( " ")
        #print(np.dot(U_rot.T, U_rot))
        print( " ")
        print(np.dot(U_rot, U_rot.T))
        print( " ")
        
        #UC.load_old(project_folder_f + "/crystal_reference_state.npy", project_folder_f+"/crystal_reference_coords.npy")
        
        S = tmat()
        S.load_old(project_folder_f + "/crystal_overlap_matrix.npy",
                   project_folder_f + "/crystal_overlap_coords.npy")
        
        S1 = tmat()
        S1.load_old(project_folder_f + "/crystal_overlap_matrix.npy",
                    project_folder_f + "/crystal_overlap_coords.npy")
        
        Smo0 = C.tT().dot(C, coords = C.coords)
        
        Smo1 = UC.tT().dot(UC, coords = UC.coords)
        
        for c in Smo0.coords:
            if np.sum(np.abs(c))<=2:
                print("---")
                print(c)
                print(Smo0.cget(c)[:2,:2])
                print(Smo1.cget(c)[:2,:2])
            
            if not toeplitz.matrices_are_equal(Smo0.cget(c), Smo1.cget(c)):
                print("---")
                print(c)
                print(Smo0.cget(c)[:2,:2])
                print(Smo1.cget(c)[:2,:2])
        
    
        
    if args.wannierization:
        #np.fft.fftn or np.fft.ifftn
        
        S = tmat()
        S.load_old(project_folder_f + "/crystal_overlap_matrix.npy",
                   project_folder_f + "/crystal_overlap_coords.npy")
        
        F = tmat()
        F.load_old(project_folder_f + "/crystal_fock_matrix.npy",
                   project_folder_f + "/crystal_fock_coords.npy")
        
        Fk = toeplitz.transform(F, np.fft.fftn)
        
        Sk = toeplitz.transform(S, np.fft.fftn)
        
       
        for c in Fk.coords:
            print("coords:", c)
            Fk_ = Fk.cget(c)
            
            Sk_ = Sk.cget(c)
            
            Sk_e, Sk_v = np.linalg.eig(Sk_)
            
            Sk_vH = 1.0*Sk_v.T
            Sk_vH.imag = -1*Sk_vH.imag
            
            S12 = np.dot(Sk_v, np.dot(np.diag(Sk_e**-.5), Sk_vH))
            
            
            S12H = 1.0*S12.T
            S12H.imag = -1*S12H.imag
            
            #print(np.sum(np.abs(S12H-S12)))
            
            Fk_sim = np.dot(S12H, np.dot(Fk_, S12))
            
            ek_, Ck_ = np.linalg.eig(Fk_sim)
            
            
            Ck_0 = np.dot(S12, Ck_)
            #Ck_ = ek_*Ck_
            #print(Ck_.shape)
            
            Fk.cset(c, Ck_0)
            
        C = toeplitz.transform(Fk, np.fft.ifftn, complx = False) #wannier functions
        #print(C.cget([0,0,0]))
        
        SC = S*C
        Smo = C.tT()*SC
        
        for c in Smo.coords:
            print("Max element at ", c)
            print(np.max(np.abs(Smo.cget(c))))
        
        print(Smo.cget([0,0,0]))
        import matplotlib.pyplot as plt
        #plt.imshow(Smo.blocks.reshape(Smo.blocks.shape[0]*Smo.blocks.shape[1], Smo.blocks.shape[2]))
        plt.imshow(Smo.cget([0,0,0]))
        plt.colorbar()
        plt.show()
    
    if args.orbital_analysis:
        #Perform extensive analysis of orbitals provided in blockfile (bfile) and coordfile (cfile)

        
        Nocc = geometry.get_nocc()
        
        C = tmat()
        if args.infile is None:
            C.load_old(args.bfile, args.cfile)
        else:
            C.load(args.infile)
        
        #if args.integral_file is None:
        of = objective_function(geometry)
        #    
        PFM_tensors, PFM, PSM, wcenters = of.pfm(C, S)
        #else:
        #    PFM_tensors, PFM, PSM, wcenters = np.load(args.integral_file)
            
        PFM_objective_function = lambda X : np.sum(np.diag(assemble_pfm_function(X).cget([0,0,0])))
        
        pfm_vals = np.diag(PFM.cget([0,0,0])) 
        psm_vals = np.diag(PSM) #sigma^2
        
        print("psm, pfm")
        print(np.array([psm_vals, pfm_vals]).T)
        
        np.save(project_folder + args.outfile + "_centers.npy", wcenters.T)
        np.save(project_folder + args.outfile + "_pfm.npy", pfm_vals)
        np.save(project_folder + args.outfile + "_psm.npy", psm_vals)
        
        if args.plot_orbitals:
            for i in np.arange(wcenters.shape[0]):
                radplot_orb(project_folder, i, C.blocks, C.coords, wcenters.T, title = "Radial distribution of orbital %i, %s" % (i, args.infile[:-3]), r_max = 40, ylim = None)
        
        
        
        
        
    
    if args.compute_pfm:
        #ref = refinery(project_folder_f)
        
        Nocc = geometry.get_nocc()
        
        if args.bfile is None:
            C= tmat()
            C.load_old(project_folder_f + "/crystal_reference_state.npy",
                       project_folder_f + "/crystal_reference_coords.npy")
        else:
            C = tmat()
            C.load_old(args.bfile, args.cfile)
        
                # Compute localization data
        of = objective_function(geometry)
        
        PFM_tensors, PFM, PSM, wcenters = of.pfm(C, S)
            
        PFM_objective_function = lambda X : np.sum(np.diag(assemble_pfm_function(X).cget([0,0,0])))
        
        #PFM_tensors, PFM, PSM, wcenters = of.pfm(C = C)
        
        #PFM_objective_function = lambda X : np.sum(np.diag(assemble_pfm_function(X).cget([0,0,0])))
        
        
        
        
        pfm_vals1 = np.diag(assemble_pfm_function(PFM_tensors).cget([0,0,0]))
        
        pfm_vals2 = np.diag(PFM.cget([0,0,0]))
        
        print(PFM_objective_function(PFM_tensors))
        
        print(pfm_vals1)
        print(pfm_vals2)
        
        
    if args.compute_orbspread:
        #ref = refinery(project_folder_f)
        
        
        Nocc = geometry.get_nocc()
        

        if args.infile is None:
            C = tmat()
            C.load_old(project_folder_f + "/crystal_reference_state.npy",
                       project_folder_f + "/crystal_reference_coords.npy")
        else:
            C = tmat()
            C.load(args.infile)
            
        # Compute localization data
        
        of = objective_function(geometry)

        if not 'S' in vars():
            S = tmat()
            S.load_old(project_folder_f + "/crystal_overlap_matrix.npy",
                       project_folder_f + "/crystal_overlap_coords.npy")
        
        XYZ2mo, Xmo, Ymo, Zmo, wcenters = of.foster_boys(C, S)
        
        
        # Get reference cell only
        L0_psm = (XYZ2mo - Xmo**2 - Ymo**2 - Zmo**2).cget([0,0,0])     
        
        # Occupied region
        Locc = L0_psm[:Nocc,:Nocc]
        
        # Virtual region
        Lvirt = L0_psm[Nocc:, Nocc:]
        
        # A unitary matrix (initially the identity matrix)
        U_rot = np.eye(len(L0_psm))
        
        spreads_occ = np.sqrt(np.diag(Locc))
        spreads_virt= np.sqrt(np.diag(Lvirt))
        
        # Write to file spreads and centers of the given
        # reference state 
        np.save("spreads_" + args.infile,
                np.append(spreads_occ, spreads_virt))
        np.save("centers_" + args.infile,
                wcenters.T)
        print("")
        print("----------------------------------------------------")
        print("Spreads and centers of the given reference")
        print("state were written to the files")
        print("'psm_spreads.npy' and 'centers.npy'.")
        print("----------------------------------------------------")
        print("")
        
        
        print("###############################")
        print("# Initial Wannier functions   #")
        print("###############################")
        
        #wannier data for comparison
        
        ccenters_occ, cspreads_occ = get_wannier_data(project_folder_f + "/wan_log_occ.txt")
        ccenters_virt, cspreads_virt = get_wannier_data(project_folder_f + "/wan_log_virt.txt")
        
        cspreads_occ = np.sqrt(cspreads_occ)
        cspreads_virt = np.sqrt(cspreads_virt)
        
        print("Orb.nr.  <x>          <y>          <z>             Spread  ")
        
        for i in np.arange(spreads_occ.shape[0]):
            print("o   :%.2i  %.5e  %.5e  %.5e     %.5e" %
                  (i, wcenters[0,i], wcenters[1,i], wcenters[2,i], spreads_occ[i]))
            #print("o(c):%.2i  %.5e  %.5e  %.5e     %.5e" % (i, ccenters_occ[i,0], ccenters_occ[i,1], ccenters_occ[i,2], cspreads_occ[i]))
        for i in np.arange(spreads_virt.shape[0]):
            print("v   :%.2i  %.5e  %.5e  %.5e     %.5e" %
                  (i+Nocc, wcenters[0,i+Nocc], wcenters[1,i+Nocc],
                   wcenters[2,i+Nocc], spreads_virt[i]))  
            #print("v(c):%.2i  %.5e  %.5e  %.5e     %.5e" % (i+Nocc, ccenters_virt[i,0], ccenters_virt[i,1], ccenters_virt[i,2], cspreads_virt[i]))      
        #print("o = occupied, v = virtual, (c) = crystal")
        print("------------------------------------")
        print("Least local occupied:", spreads_occ.max())
        print("Least local virtual :", spreads_virt.max())
        print("------------------------------------")
        print("\n"*3)
    
    if args.compute_pao_orbspread:
        #ref = refinery(project_folder_f)
    
    
        C = tmat()
        C.load_old(project_folder_f + "/c_pao_blocks.npy",
                   project_folder_f + "/c_pao_coords.npy")

        
        #ref.compute_orbspread(m=1)
        
        #wcenters, spread = ref.compute_orbspread(m=1)
        
        
        of = objective_function(geometry)
        

        XYZ2mo, Xmo, Ymo, Zmo, wcenters = of.foster_boys(C, S)
        
        
        #L0_psm = ofunction.cget([0,0,0])*1
        
        L0_psm_2 = (XYZ2mo - Xmo**2 - Ymo**2 - Zmo**2).cget([0,0,0]) 
        
        spreads = np.sqrt(np.diag(L0_psm_2))
        #objective_function, wcenters = of.foster_boys(C = C)
        
        
        print("Orb.nr.  <x>          <y>          <z>             Spread  ")
        
        for i in np.arange(spreads.shape[0]):
            print("    :%.2i  %.5e  %.5e  %.5e     %.5e" % (i, wcenters[0,i], wcenters[1,i], wcenters[2,i], spreads[i]))
            #print("o(c):%.2i  %.5e  %.5e  %.5e     %.5e" % (i, ccenters_occ[i,0], ccenters_occ[i,1], ccenters_occ[i,2], cspreads_occ[i]))
        #for i in np.arange(spreads_virt.shape[0]):
        #    print("    :%.2i  %.5e  %.5e  %.5e     %.5e" % (i+Nocc, wcenters[0,i+Nocc], wcenters[1,i+Nocc], wcenters[2,i+Nocc], spreads_virt[i]))  
        #    #print("v(c):%.2i  %.5e  %.5e  %.5e     %.5e" % (i+Nocc, ccenters_virt[i,0], ccenters_virt[i,1], ccenters_virt[i,2], cspreads_virt[i]))      
        #print("o = occupied, v = virtual, (c) = crystal")
        print("------------------------------------")
        print("Least local orbital:", spreads.max())
        #print("Least local virtual :", spreads_virt.max())
        print("------------------------------------")
        print("\n"*3)
        

        
        print("Wannier Centers")
        print(wcenters.T)
        np.save(project_folder_f + "/crystal_wannier_centers.npy",
                wcenters.T)
        
        print("Orbital spreads:")
        print(np.sqrt(np.diag(L0_psm_2)))
        print("Orbital spreads squared :")
        print(np.diag(L0_psm_2))
        

    if args.kspace_overlap:
        S= tmat()
        S.load_old(project_folder_f + "/crystal_overlap_matrix.npy",
                   project_folder_f + "/crystal_overlap_coords.npy")
        
        C= tmat()
        C.load_old(project_folder_f + "/crystal_reference_state.npy",
                   project_folder_f + "/crystal_reference_coords.npy")
        
        
        Ck = toeplitz.transform(C, np.fft.fftn)
        
        Sk = toeplitz.transform(S, np.fft.fftn)
        
        CtSCk = Ck*0
        SCk = Ck*0
        
        for c in Ck.coords:
            #print("coords:", c)
            Ck_ = Ck.cget(c)
            
            Sk_ = Sk.cget(c)
            
            CtSCk.cset(c, np.dot(Ck_.T, np.dot(Sk_, Ck_)))
        
        #for c in Ck.coords:
        #    #print("coords:", c)
        #    Ck_ = Ck.cget(c)
        #    
        #    SCk_ = SCk.cget(c)
            
            
        
        CtSC = toeplitz.transform(CtSCk, np.fft.ifftn, complx = False) #wannier functions
        
        print(CtSC.cget(np.array([0,0,0])))
        #import matplotlib.pyplot as plt
        #plt.imshow(CtSC.cget([0,0,0]))
        #plt.show()
        
    
        
    if args.compute_overlap:
        
        if args.infile is None:
            C= tmat()
            C.load_old(project_folder_f + "/crystal_reference_state.npy", project_folder_f+"/crystal_reference_coords.npy")
        else:
            C= tmat()
            C.load(args.infile)

            
        S = tmat()
        S.load_old(project_folder_f + "/crystal_overlap_matrix.npy",
                   project_folder_f + "/crystal_overlap_coords.npy")
        
        SC = S * C
        
        smo = C.tT().cdot(SC, coords = C.coords)
        
        print(smo.cget([0,0,0]))
        
        print("coordinate/L2norm per elm. / max.abs.elm.")
        for c in smo.coords:
            cb = smo.cget(c)    
            print(c, L2norm(cb)/cb.size, np.abs(cb).max())
        
        #print("Deviation from identity: %.15e"  % toeplitz.L2norm(np.dot(U_rot.T,U_rot)-np.diag(np.ones(U_rot.shape[0]))))

        
        
        
    if args.compute_overlap_ao:
        #Sl =  localization_integrals(project_folder)
        #Sl.save("libint_overlap_matrix")
        
        ints = pp.carmom_tmat(project_folder_f)
        
        Sl = ints.S
        

        Sl.save(project_folder_f + "/libint_overlap_matrix.npy")
        


        
    if args.compute_pao:

        S = tmat()
        S.load_old(project_folder + "/crystal_overlap_matrix.npy",
                   project_folder_f + "/crystal_overlap_coords.npy")
        

        if args.infile==None:
            D = tmat()
            D.load_old(project_folder + "/crystal_density_matrix.npy", project_folder_f + "/crystal_density_coords.npy")
        else:
            C = tmat()
            C.load(args.infile)
            D = compute_density_matrix(C, geometry.get_nocc())
        
        C_pao, norms = compute_normalized_pao(S,D, coords_out = S.coords)
        print("norms")
        print(norms)

        #C_pao.save_old(project_folder_f + "c_pao_blocks.npy", project_folder_f + "c_pao_coords.npy")

        
        C_pao.save(args.outfile)
        #c_pao = toeplitz.tmat()
        #c_pao.load(C_pao)
        #c_pao.save("c_pao_orbref_3")
        
        
        #C_pao_2 = 
    
    
    if args.brute_force_wcenter:
        
        
        
        x,y,z = pp.brute_force_wcenter(project_folder)
        

        
        # compare with ref implementation
        C = tmat()
        C.load_old(project_folder_f + "/crystal_reference_state.npy",
                   project_folder_f + "/crystal_reference_coords.npy")
        of = objective_function(geometry)
        objective_function, wcenters = of.foster_boys( C, S)
        print("Lattice")
        print(geometry.lattice)
        print("C:1,2,3")
        print(np.dot(np.array([1,2,3]), geometry.lattice))
        print("Wannier Centers (brute force):")
        R = np.array([x,y,z])
        print(R.T)
        print("Wannier Centers (symmetries)")
        print(wcenters.T)
    if args.convert_cryscor:
        
        
        perm = geometry.get_inverse_permutation()
        #perm = np.array([0,1,2,3,4,5,6,7,8,   11,12,10,13,9,   14,15,16,17,    18,19,20,21,22,       25, 26, 24, 27, 23]) #diamond
        #perm = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 17, 18,16, 19, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 30, 33, 29]) #HCN
        #print(perm)
        print("Warning: Inverse permutation not tested / indicated fails")
        
        #blocks = np.load(project_folder + "/to_lorenzo/pao_coeffs.npy")
        
        #coords = np.load(project_folder + "/to_lorenzo/pao_coords.npy")
        #print(blocks.shape)
        #print(coords.shape)
        
        #nocc = geometry.get_nocc()
        #wp.xdec2cryscor(nocc,blocks, coords, outfile = "pao_oslo_lih.txt")
        
        if args.infile is None:
            if args.bfile is None:
                blocks1 = np.load(project_folder + "/crystal_reference_state.npy")
        
                coords1 = np.load(project_folder + "/crystal_reference_coords.npy")
            else:
                blocks1 = np.load(args.bfile)
        
                coords1 = np.load(args.cfile)
                
            
            #blocks1 = blocks1[:, np.array([0,1,2,3,4,5,6,7,8,   11,12,10,13,9,   14,15,16,17,    18,19,20,21,22,       25, 26, 24, 27, 23]),:] #diamond
                
            #blocks1 = blocks1[:, np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 17, 18,16, 19, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 30, 33, 29]), :] #HCN
            
            blocks1 = blocks1[:, perm, :]
            
                
            #sorting is a problem?
            ret = tmat()
            #ret = tp.get_zero_tmat(cutoffs = self.domain-1 + other.domain-1, blockshape = (self.blockshape[0], other.blockshape[1]))
            ret.load_nparray(blocks1, coords1)
            
            blocks = [] #np.zeros(blocks1.shape)
            coords = [] #np.zeros(coords1.shape)
            
            for i in np.arange(-np.amax(coords1[:,0]),np.amax(coords1[:,0])):
                 for j in np.arange(-np.amax(coords1[:,1]),np.amax(coords1[:,1])):
                     for k in np.arange(-np.amax(coords1[:,2]),np.amax(coords1[:,2])):
                         blocks.append(ret.cget([k,j,i]))
                         coords.append(np.array([k,j,i]))
                         #print([k,j,i])
            
            #ret.load_nparray(blocks, coords)
            blocks = np.array(blocks)
            coords = np.array(coords)
            
            
        else:
            tm = tmat()
            tm.load(args.infile)
            blocks = tm.blocks[:-1]
            coords = tm.coords
            
            #blocks = blocks[:, np.array([0,1,2,3,4,5,6,7,8,   11,12,10,13,9,   14,15,16,17,    18,19,20,21,22,       25, 26, 24, 27, 23]),:] #diamond
            blocks = blocks[:, perm, :]
            
            #blocks = blocks[:, np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 17, 18,16, 19, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 30, 33, 29]), :] #HCN
        
            #coords = np.load(project_folder + args.file.split(",")[1])
        #print("Warn HCN")
        #coords = coords*-1
        
        # sort on increasing first index
        
        
        
        #0,1,2,3,4
        #4,2,0,1,3
        #->  2,3,1,4,0
        
        #blocks = np.swapaxes(blocks, 1,2)
        if args.outfile is None:
            args.outfile = "oslo."
        wp.xdec2cryscor(geometry.get_nocc(),blocks, -1*coords, outfile = args.outfile)
        
        
    
    if args.brute_force_orbspread:
        
        
        
        l_psm = pp.brute_force_orbspread(project_folder)
        print(np.diag(l_psm))
        print(np.sqrt(np.diag(l_psm)))
        
    
    if args.compute_lsint:
        ints = pp.carmom(project_folder)
    
        np.save(project_folder + "/S_ao.npy", ints.S.get([0,0,0]))
        np.save(project_folder + "/X_ao.npy", ints.X.get([0,0,0]))
        np.save(project_folder + "/Y_ao.npy", ints.Y.get([0,0,0]))
        np.save(project_folder + "/Z_ao.npy", ints.Z.get([0,0,0]))
        np.save(project_folder + "/X2_ao.npy", ints.X2.get([0,0,0]))
        np.save(project_folder + "/Y2_ao.npy", ints.Y2.get([0,0,0]))
        np.save(project_folder + "/Z2_ao.npy", ints.Z2.get([0,0,0]))
        np.save(project_folder + "/XY_ao.npy", ints.XY.get([0,0,0]))
        np.save(project_folder + "/XZ_ao.npy", ints.XZ.get([0,0,0]))
        np.save(project_folder + "/YZ_ao.npy", ints.YZ.get([0,0,0]))
         
        
    
        
    
    if args.compute_mcharge:
        q, q_coords, SC, SC_coords = compute_mulliken_charges(project_folder_f)
        Q = np.zeros(len(q[0][0]), dtype = float)
        for i in np.arange(len(q_coords)):
            for j in np.arange(len(q[i])):
                Q += np.array(q[i][j])

        print(Q)
        q = np.array(q)
        q_coords = np.array(q_coords)
        np.save(project_folder + "/mulliken_charges_blocks.npy", q)
        np.save(project_folder + "/mulliken_charges_coords.npy", q_coords)
        np.save(project_folder + "/sc_blocks.npy", SC)
        np.save(project_folder + "/sc_coords.npy", SC_coords)
        print("""Done. Results stored in: 
%s/mulliken_charges*.npy
        """ % project_folder)
        
        
        
        
