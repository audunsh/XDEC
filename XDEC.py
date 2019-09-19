#!/usr/bin/env python

import numpy as np

import os

import subprocess as sp

from ast import literal_eval

import argparse

import utils.toeplitz as tp

import lwrap.lwrap as li # Libint wrapper

import utils.prism as pr
import domdef as dd
import PRI 
import time

def mapgen(vex1, vex2):
    """
    returns a matrix where vex1[ ret_indx[i,j] ] = vex1[i] - vex2[j]
    """
    D = vex1[:, None] - vex2[None, :]
    ret_indx = -1*np.ones((vex1.shape[0], vex2.shape[0]), dtype = int)
    for i in np.arange(vex1.shape[0]):
        for j in np.arange(vex2.shape[0]):
            try:
                ret_indx[i,j] = np.argwhere(np.all(np.equal(vex1, D[i,j]), axis = 1))
            except:
                pass
    return ret_indx

def intersection3s(coords1, coords2):
    """
    Returns the intersection 3-vectors between two sets of (N,3) arrays
    """
    distance = np.sum((coords1[:,None] - coords2[None,:])**2, axis = 2)
    
    coords1_in_coords2 = np.any(distance==0, axis = 0)
    return coords2[coords1_in_coords2]

def c2_not_in_c1(c1, c2, sort = False):
    """
    Returns the 3-vectors in c2 not present in c1
    """
    distance = np.sum((c1[:,None] - c2[None,:])**2, axis = 2)
    
    c1_in_c2 = np.any(distance==0, axis = 0)
    if not sort:
        return c2[c1_in_c2==False]
    if sort:
        c2_ret = c2[c1_in_c2==False]
        c2_ret = c2_ret[np.argsort(np.sum(c2_ret**2, axis = 1))]
        return c2_ret

class pair_fragment_amplitudes():
    # Setup and compute all pair fragments between 0 and M
    def __init__(self, fragment_amplitudes_1, fragment_amplitudes_2, M):
        self.M = M
        self.f1 = fragment_amplitudes_1
        self.f2 = fragment_amplitudes_2

        self.f1_occupied_cells = self.f1.d_ii.coords[:self.f1.n_occupied_cells]
        self.f1_virtual_cells = self.f1.d_ia.coords[:self.f1.n_virtual_cells]

        self.f2_occupied_cells = self.f2.d_ii.coords[:self.f2.n_occupied_cells] + M
        self.f2_virtual_cells = self.f2.d_ia.coords[:self.f2.n_virtual_cells] #+ M
        
        
        # Set up unified occupied domain
        coords_extra = c2_not_in_c1(self.f1_occupied_cells, self.f2_occupied_cells, sort = True)
        self.coords_occupied = np.zeros((self.f1_occupied_cells.shape[0]+coords_extra.shape[0], 3), dtype = int)
        self.coords_occupied[:self.f1_occupied_cells.shape[0]] = self.f1_occupied_cells
        self.coords_occupied[self.f1_occupied_cells.shape[0]:] = coords_extra

        self.n_occupied_cells = self.coords_occupied.shape[0]
        
        # Set up unified amplitude masking matrices for the occupied domain
        d_ii_blocks = []
        for c in self.coords_occupied:
            #print(c)
            #print(self.f1.d_ii.cget(c))
            #print(self.f1.d_ii.cget(c)<=self.f1.occupied_cutoff)

            d_ii_blocks.append(np.any(np.array([self.f1.d_ii.cget(c)<=self.f1.occupied_cutoff, \
                                         self.f2.d_ii.cget(c+M)<=self.f2.occupied_cutoff, \
                                         self.f2.d_ii.cget(c-M)])<=self.f2.occupied_cutoff, axis = 0))
        
        self.d_ii = tp.tmat()
        self.d_ii.load_nparray(np.array(d_ii_blocks, dtype = bool), self.coords_occupied, safemode = False, screening = False)


        # Set up unified virtual domain
        coords_extra = c2_not_in_c1(self.f1_virtual_cells, self.f2_virtual_cells, sort = True)
        self.coords_virtual = np.zeros((self.f1_virtual_cells.shape[0]+coords_extra.shape[0], 3), dtype = int)
        self.coords_virtual[:self.f1_virtual_cells.shape[0]] = self.f1_virtual_cells
        self.coords_virtual[self.f1_virtual_cells.shape[0]:] = coords_extra

        self.n_virtual_cells = self.coords_virtual.shape[0]

        # Set up unified amplitude masking matrices for the virtual domain
        d_ia_blocks = []
        for c in self.coords_virtual:
            d_ia_blocks.append(np.any(np.array([self.f1.d_ia.cget(c)<=self.f1.virtual_cutoff, \
                                         self.f2.d_ia.cget(c+M)<=self.f2.virtual_cutoff, \
                                         self.f2.d_ia.cget(c-M)])<=self.f2.virtual_cutoff, axis = 0))
        
        print("Virtual pair space")
        print(self.coords_virtual, len(self.coords_virtual))
        print("Occupied pair space")
        print(self.coords_occupied, self.coords_occupied.shape[0])
        self.d_ia = tp.tmat()
        self.d_ia.load_nparray(np.array(d_ia_blocks, dtype = bool), self.coords_virtual, safemode=False, screening = False)

        # Then initialize tensors and solve equations as for the fragment-case
        
        self.mM = np.arange(self.coords_occupied.shape[0])[np.sum((self.coords_occupied-self.M)**2, axis = 1)==0][0]
        
        

        self.vv_indx = mapgen(self.coords_virtual, self.coords_virtual)


        nocc = self.f1.p.get_nocc()
        nvirt = self.f1.p.get_nvirt()
        N_occ = self.d_ii.coords.shape[0]
        N_virt = self.d_ia.coords.shape[0]
        
        self.t2 = np.zeros((nocc, N_virt, nvirt, N_occ, nocc, N_virt, nvirt), dtype = float)
        self.g_d = np.zeros((nocc, N_virt, nvirt, N_occ, nocc, N_virt, nvirt), dtype = float)
        self.g_x = np.zeros((nocc, N_virt, nvirt, N_occ, nocc, N_virt, nvirt), dtype = float)

        # Use amplitudes from the two fragments as initial guess, fill in known tensors
        """
        self.t2[self.f1.fragment, :self.f1.n_virtual_cells, :, :self.f1.n_occupied_cells, :, :self.f1.n_virtual_cells, :] = self.f1.t2[self.f1.fragment]
        self.g_d[:, :self.f1.n_virtual_cells, :, :self.f1.n_occupied_cells, :, :self.f1.n_virtual_cells, :] = self.f1.g_d
        self.g_x[:, :self.f1.n_virtual_cells, :, :self.f1.n_occupied_cells, :, :self.f1.n_virtual_cells, :] = self.f1.g_x
        
        
        self.t2[self.f2.fragment, :self.f2.n_virtual_cells, :, :self.f2.n_occupied_cells, :, :self.f2.n_virtual_cells, :] = self.f2.t2[self.f2.fragment]
        self.g_d[:, :self.f2.n_virtual_cells, :, :self.f2.n_occupied_cells, :, :self.f2.n_virtual_cells, :] = self.f2.g_d
        self.g_x[:, :self.f2.n_virtual_cells, :, :self.f2.n_occupied_cells, :, :self.f2.n_virtual_cells, :] = self.f2.g_x
        """


        self.cmap = tp.tmat() #coordinate mapping
        cgrid = tp.lattice_coords([8,8,8])
        self.cmap.load_nparray(np.ones((cgrid.shape[0], 2,2), dtype = int), cgrid, safemode=False)

        sequence = []
        for ddL in np.arange(N_virt):
            for ddM in np.arange(ddL, N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]

                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]
                    if np.linalg.norm(self.g_d[:, ddL, :, mM, :, ddM, :])<10e-10:
                        """
                        # Get exchange block coordinates
                        M_cmap    = self.cmap.mapping[self.cmap._c2i(M) ]
                        ddL_cmap  = self.cmap.mapping[self.cmap._c2i(dL) ]
                        ddM_cmap  = self.cmap.mapping[self.cmap._c2i(dM) ]
                        ddL_M = self.cmap.mapping[self.cmap._c2i(dM + M) ]
                        ddM_M = self.cmap.mapping[self.cmap._c2i(dL - M) ]
                        
                        sequence.append([ddL_cmap  , M_cmap, ddM_cmap  , 0, ddL, mM, ddM])
                        sequence.append([ddL_M, M_cmap, ddM_M, 1, ddL, mM, ddM])
                        """


                        ## Instead

                        M = self.d_ii.coords[mM]

                        # Get exchange block coordinates
                        # Note: these point out of the fragment domain,
                        # use cfit carefully
                        ddL_M = self.cmap.mapping[self.cmap._c2i(dM - M) ]
                        ddM_M = self.cmap.mapping[self.cmap._c2i(dL + M) ]

                        print("Queueing ")
                        #print(ddL)

                        


                        print(dL, M, dM)
                        print(ddL, mM, ddM)
                        print(" ==== ")
                        print(dM -M, M, dL + M)
                        ## Should test here if mM_ is in set
                        print(" ==== ")
                        print(ddL_M, mM,ddM_M)
                        print(" ")
                        # Negative M 
                        mM_ = self.d_ii.mapping[ self.d_ii._c2i(-M) ]
                        
                        # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                        #                ^                 ^           ^          ^
                        #            Calculate these    ex/direct    store here   1=transpose
                        
                        sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # direct
                        sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,  1]) # direct, transposed


                        if mM == self.mM:
                            # block inside EOS
                            sequence.append([ddL_M, mM, ddM_M, 1, ddL, mM , ddM,0])  # exchange
                            sequence.append([ddL_M, mM, ddM_M, 1, ddM, mM_, ddL,1]) # exchange, transposed

        
        
        self.initialize_blocks(sequence)
    def initialize_blocks(self, sequence):
        #print("Initialization sequence:")
        sequence = np.array(sequence)




        # Sort blocks by dL:
        a = np.argsort(sequence[:,0])
        sequence = sequence[a]
        #print(sequence.shape)
        sequence = np.append(sequence, [ [-1000,0,0,0, 0, 0, 0,0] ], axis = 0) #-100 Just to make sure :-)
        #print(sequence.shape)
        j = 0
        for i in np.arange(len(sequence)):
            if sequence[i,0] != sequence[j,0]:
                a = np.argsort(sequence[j:i, 2])
                sq_i = sequence[j:i][a]

                sq_i = np.append(sq_i, [ [-1000,0,-1000,0, 0, 0, 0,0] ], axis = 0) #-100 Just to make sure :-)




                #print(sq_i)

                dL = self.cmap.coords[sq_i[0,0]]
                


                #print(sq_i)
                k = 0
                for l in np.arange(len(sq_i)):
                    

                    if sq_i[k,2] != sq_i[l,2]:
                        print(sq_i[k,2])
                        dM = self.cmap.coords[sq_i[k,2]]


                        #print(sq_i[k:l])
                        # Integrate here, loop over M
                        print(dL, dM)
                        I, Ishape = self.f1.ib.getorientation(dL, dM)

                        for m in sq_i[k:l]:
                            M = self.d_ii.coords[m[1]]
                            ddL, mM, ddM = m[0], m[1], m[2]
                            #print(self.g_x.shape, ddL, mM, ddM)
                            #print(dL, M, dM)
                            #print(I.cget(M).shape, Ishape)
                            if m[7] == 0:
                                if m[3] == 0:
                                    # Direct contribution
                                    self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)
                                    self.t2[:,  ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)*self.f1.e_iajb**-1
                                if m[3] == 1:
                                    # Exchange contribution
                                    ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                    self.g_x[:, ddL_, :, mM_, :, ddM_, :] = I.cget(M).reshape(Ishape)
                            if m[7] == 1:
                                if m[3] == 0:
                                    # Direct contribution
                                    self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)
                                    self.t2[:,  ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)*self.f1.e_iajb**-1
                                if m[3] == 1:
                                    # Exchange contribution
                                    ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                    self.g_x[:, ddL_, :, mM_, :, ddM_, :] = I.cget(M).T.reshape(Ishape)


                        k = l*1
                        
                j = i*1

    def initialize_blocks_(self, sequence):
        sequence = np.array(sequence)


        print("Initializing %i blocks of %i integrals." % (sequence.shape[0], self.f1.p.get_nocc()**2*self.f1.p.get_nvirt()**2))

        # Sort blocks by dL:
        a = np.argsort(sequence[:,0])
        sequence = sequence[a]
        
        sequence = np.append(sequence, [ [-1000,0,0,0, 0, 0, 0] ], axis = 0) #-100 Just to make sure :-)
        
        j = 0
        for i in np.arange(len(sequence)):
            if sequence[i,0] != sequence[j,0]:
                a = np.argsort(sequence[j:i, 2])
                sq_i = sequence[j:i][a]

                sq_i = np.append(sq_i, [ [-1000,0,-1000,0, 0, 0, 0] ], axis = 0) #-100 Just to make sure :-)

                dL = self.cmap.coords[sq_i[0,0]]

                k = 0
                for l in np.arange(len(sq_i)):
                    if sq_i[k,2] != sq_i[l,2]:
                        #print(sq_i[k,2])
                        dM = self.cmap.coords[sq_i[k,2]]
                        #print(dL, dM)

                        # Integrate here, loop over M
                        I, Ishape = self.f1.ib.getorientation(dL, dM)

                        for m in sq_i[k:l]:
                            M = self.cmap.coords[m[1]]
                            ddL, mM, ddM = m[4], m[5], m[6] #this is a little akward, debugging going on

                            if m[3] == 0:
                                # Direct contribution
                                self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)
                                self.t2[:,  ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)*self.f1.e_iajb**-1
                            if m[3] == 1:
                                # Exchange contribution
                                ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                self.g_x[:, ddL_, :, mM_, :, ddM_, :] = I.cget(M).reshape(Ishape)


                        k = l*1

                j = i*1

        print("Size/shape of fragment amplitude tensor:", self.t2.size, self.t2.shape )

    def compute_pair_fragment_energy(self):
        """
        Compute fragment energy
        """
        
        e_mp2 = 0
        
        N_virt = self.n_virtual_cells
        
        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = np.array(self.d_ia.cget(dL)[self.f1.fragment[0],:], dtype = bool) # dL index mask

            
            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM]
                dM_i = np.array(self.d_ia.cget(dM)[self.f2.fragment[0],:], dtype = bool) # dM index mask
                
                
                g_direct = self.g_d[:,ddL,:,self.mM, :, ddM, :][self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]
                g_exchange = self.g_x[:,ddL,:,self.mM, :, ddM, :][self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i]

                
                
                t = self.t2[:,ddL,:,self.mM, :, ddM, :][self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]

                e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
                
        print(e_mp2, 2*e_mp2)
        return e_mp2

    def solve(self, norm_thresh = 1e-10):
        """
        Converge fragment (AOS) amplitudes within occupied and virtual extents 
        """
        nocc = self.f1.p.get_nocc()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        

        vp_indx = mapgen(virtual_extent, pair_extent)
        pp_indx = mapgen(pair_extent, pair_extent)

        for ti in np.arange(100):
            t2_new = np.zeros_like(self.t2)
            for dL in np.arange(self.n_virtual_cells):
                dLv = self.d_ia.coords[dL]
                dL_i = np.array(self.d_ia.cget(dLv)[self.f1.fragment[0], :], dtype = bool) # dL index mask

                for dM in np.arange(self.n_virtual_cells): 
                    dMv = self.d_ia.coords[dM]
                    dM_i = np.array(self.d_ia.cget(dMv)[self.f2.fragment[0], :], dtype = bool)# dM index mask
                    for M in np.arange(self.n_occupied_cells):
                        Mv = self.d_ii.coords[M]
                        M_i = np.array(self.d_ii.cget(Mv)[self.f1.fragment[0], :], dtype = bool)  # M index mask

                        tnew = -self.g_d[:, dL, :, M, :, dM, :]

                        # generate index mapping of non-zero amplitudes in cell
                        cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.f1.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                        # Perform contractions

                        # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                        Fac = self.f1.f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                        tnew -= np.einsum("iKcjb,Kac->iajb", self.t2[:, :, :, M, :, dM, :], Fac)

                        # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                        Fbc = self.f1.f_mo_aa.cget(virtual_extent - virtual_extent[dM])
                        tnew -= np.einsum("iajKc,Kbc->iajb", self.t2[:, dL, :, M, :, :, :], Fbc)






                        """
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        #vpdL = mapgen(virtual_extent-Mv, pair_extent)
                        #Fki = self.f1.f_mo_ii.cget(-1*pair_extent)
                        #tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)
                        
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        # revised summation, include all non-zero blocks
                        #vpdL = mapgen(virtual_extent-Mv, pair_extent)
                        #Fki = self.f1.f_mo_ii.cget(Mv-1*pair_extent)
                        #tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vpdL[dL], :, :, :, vpdL[dM], :], Fki)
                        
                        vpdL = mapgen(virtual_extent-Mv, pair_extent)

                        non_zeros = (vpdL[dL]>=0) * (vpdL[dM]>=0)

                        Fki = self.f1.f_mo_ii.cget(Mv-1*pair_extent)
                        #print(self.t2[:, vpdL[dL], :, :, :, vpdL[dM], :].shape)
                        tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vpdL[dL][non_zeros], :, np.arange(self.n_occupied_cells)[non_zeros], :, vpdL[dM][non_zeros], :], Fki[non_zeros])
                        """

                        

                        # Conceptually simpler approach
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        """
                        Fki = self.f1.f_mo_ii.cget(-1*pair_extent)
                        
                        M_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[M] - pair_extent ) ]
                        dL_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dL] - pair_extent) ]
                        dM_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dM] - pair_extent) ]
                        #make sure indices is in correct domain

                        tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, dL_M, :, M_range, :, dM_M, :], Fki)
                        """

                        Fki = self.f1.f_mo_ii.cget(-1*pair_extent)
                        
                        M_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[M] - pair_extent ) ]
                        dL_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dL] - pair_extent) ]
                        dM_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dM] - pair_extent) ]
                        #make sure indices is in correct domain (not negative or beyond extent)
                        nz = (M_range<self.n_occupied_cells)*(dL_M<self.n_virtual_cells)*(dM_M<self.n_virtual_cells)*\
                             (M_range>=0)*(dL_M>=0)*(dM_M>=0)

                        tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, dL_M[nz], :, M_range[nz], :, dM_M[nz], :], Fki[nz])
                        



                        # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                        Fkj = self.f1.f_mo_ii.cget(-1*pair_extent + pair_extent[M])
                        tnew += np.einsum("iaKkb,Kkj->iajb",self.t2[:, dL, :, :, :, dM, :], Fkj)

                        t2_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*self.f1.e_iajb**-1).ravel()[cell_map]
                        
                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)

            self.t2 -= t2_new
            rnorm = np.linalg.norm(t2_new)
            #print(ti, np.linalg.norm(self.t2))
            if rnorm<norm_thresh:

                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti, np.linalg.norm(t2_new)))
                break


class pair_fragment_amplitudes_():
    def __init__(self, fragment_amplitudes_1, fragment_amplitudes_2, M):
        self.M = M
        self.f1 = fragment_amplitudes_1
        self.f2 = fragment_amplitudes_2

        self.f1_occupied_cells = self.f1.d_ii.coords[:self.f1.n_occupied_cells]
        self.f1_virtual_cells = self.f1.d_ia.coords[:self.f1.n_virtual_cells]

        self.f2_occupied_cells = self.f2.d_ii.coords[:self.f2.n_occupied_cells] + M
        self.f2_virtual_cells = self.f2.d_ia.coords[:self.f2.n_virtual_cells] + M

        # Extend the pair domain with negative M's 
        # (See notes on how the pair fragment energy is expressed for periodic systems)
        coords_extra = c2_not_in_c1(self.f2_occupied_cells, self.f2.d_ii.coords[:self.f2.n_occupied_cells] - M, sort = True)
        coords = np.zeros((self.f2_occupied_cells.shape[0]+coords_extra.shape[0], 3))
        coords[:self.f2_occupied_cells.shape[0]] = self.f2_occupied_cells
        coords[self.f2_occupied_cells.shape[0]:] = coords_extra

        self.f2_occupied_cells = coords*1 #a copy, not a view (presumtively)

        coords_extra = c2_not_in_c1(self.f2_virtual_cells, self.f2.d_ia.coords[:self.f2.n_virtual_cells] - M, sort = True)
        coords = np.zeros((self.f2_virtual_cells.shape[0]+coords_extra.shape[0], 3))
        coords[:self.f2_virtual_cells.shape[0]] = self.f2_virtual_cells
        coords[self.f2_virtual_cells.shape[0]:] = coords_extra

        self.f2_virtual_cells = coords*1 #a copy, not a view




        
        
        # Set up unified occupied domain
        coords_extra = c2_not_in_c1(self.f1_occupied_cells, self.f2_occupied_cells, sort = True)
        self.coords_occupied = np.zeros((self.f1_occupied_cells.shape[0]+coords_extra.shape[0], 3), dtype = int)
        self.coords_occupied[:self.f1_occupied_cells.shape[0]] = self.f1_occupied_cells
        self.coords_occupied[self.f1_occupied_cells.shape[0]:] = coords_extra

        self.n_occupied_cells = self.coords_occupied.shape[0]
        
        # Set up unified amplitude masking matrices for the occupied domain
        d_ii_blocks = []
        for c in self.coords_occupied:
            #print(c)
            #print(self.f1.d_ii.cget(c))
            #print(self.f1.d_ii.cget(c)<=self.f1.occupied_cutoff)

            d_ii_blocks.append(np.any(np.array([self.f1.d_ii.cget(c)<=self.f1.occupied_cutoff, \
                                         self.f2.d_ii.cget(c+M)<=self.f2.occupied_cutoff, \
                                         self.f2.d_ii.cget(c-M)])<=self.f2.occupied_cutoff, axis = 0))
        
        self.d_ii = tp.tmat()
        self.d_ii.load_nparray(np.array(d_ii_blocks, dtype = bool), self.coords_occupied, safemode = False, screening = False)


        # Set up unified virtual domain
        coords_extra = c2_not_in_c1(self.f1_virtual_cells, self.f2_virtual_cells, sort = True)
        self.coords_virtual = np.zeros((self.f1_virtual_cells.shape[0]+coords_extra.shape[0], 3), dtype = int)
        self.coords_virtual[:self.f1_virtual_cells.shape[0]] = self.f1_virtual_cells
        self.coords_virtual[self.f1_virtual_cells.shape[0]:] = coords_extra

        self.n_virtual_cells = self.coords_virtual.shape[0]

        # Set up unified amplitude masking matrices for the virtual domain
        d_ia_blocks = []
        for c in self.coords_virtual:
            d_ia_blocks.append(np.any(np.array([self.f1.d_ia.cget(c)<=self.f1.virtual_cutoff, \
                                         self.f2.d_ia.cget(c+M)<=self.f2.virtual_cutoff, \
                                         self.f2.d_ia.cget(c-M)])<=self.f2.virtual_cutoff, axis = 0))
        
        self.d_ia = tp.tmat()
        self.d_ia.load_nparray(np.array(d_ia_blocks, dtype = bool), self.coords_virtual, safemode=False, screening = False)

        # Then initialize amplitudes and solve equations as for the fragment-case
        # Energy evaluation will be kind of different

        #print(np.sum((self.coords_occupied-self.M)**2, axis = 1)==0)
        self.mM = np.arange(self.coords_occupied.shape[0])[np.sum((self.coords_occupied-self.M)**2, axis = 1)==0][0]
        self.mM_ = np.arange(self.coords_occupied.shape[0])[np.sum((self.coords_occupied+self.M)**2, axis = 1)==0][0]
        
        #print(self.M, self.coords_occupied[self.mM], self.mM)
        #print(-self.M, self.coords_occupied[self.mM_], self.mM_)

        #print("Pair fragment occupied coordinates:")
        #print(self.coords_occupied)
        #print(self.coords_virtual)

        self.vv_indx = mapgen(self.coords_virtual, self.coords_virtual)

        self.init_amplitudes()
            




    def init_amplitudes(self):
        """
        Initialize the amplitudes using the MP2-like starting guess
        """
        
        n_occ = self.f1.p.get_nocc()     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells    # Number of occupied cells
        n_virt = self.f1.p.get_nvirt()   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells    # Number of virtual cells
        

        self.t2 = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = float)

        self.g_d = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = float)

        self.g_x = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = float)

        # Fill in tensors, initial guess, calculate initial energy

        f_aa = np.diag(self.f1.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f1.f_mo_ii.cget([0,0,0]))
    
        self.e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]

        # Initialize cells from the separate fragments
        self.t2[self.f1.fragment, :self.f1.n_virtual_cells, :, :self.f1.n_occupied_cells, :, :self.f1.n_virtual_cells, :] = self.f1.t2[self.f1.fragment]
        self.g_d[:, :self.f1.n_virtual_cells, :, :self.f1.n_occupied_cells, :, :self.f1.n_virtual_cells, :] = self.f1.g_d
        self.g_x[:, :self.f1.n_virtual_cells, :, :self.f1.n_occupied_cells, :, :self.f1.n_virtual_cells, :] = self.f1.g_x
        
        
        self.t2[self.f2.fragment, :self.f2.n_virtual_cells, :, :self.f2.n_occupied_cells, :, :self.f2.n_virtual_cells, :] = self.f2.t2[self.f2.fragment]
        self.g_d[:, :self.f2.n_virtual_cells, :, :self.f2.n_occupied_cells, :, :self.f2.n_virtual_cells, :] = self.f2.g_d
        self.g_x[:, :self.f2.n_virtual_cells, :, :self.f2.n_occupied_cells, :, :self.f2.n_virtual_cells, :] = self.f2.g_x
        

        # Initialize the remaining cells
        for ddL in np.arange(self.f1.n_virtual_cells, N_virt):
            for ddM in np.arange(self.f1.n_virtual_cells, N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]
                for mM in np.arange(self.f1.n_occupied_cells, N_occ):
                    M = self.d_ii.coords[mM]
                    #print(M, mM)

                
                    g_direct = self.f1.ib.getcell(dL, M, dM)
                    g_exchange = self.f1.ib.getcell(dM-M, M, dL+M) 
                    #t = g_direct*self.e_iajb**-1
                    #print(np.linalg.norm(g_direct*self.e_iajb**-1))
                    self.t2[:, ddL, :, mM, :, ddM, :] = g_direct*self.e_iajb**-1
                    self.g_d[:, ddL, :, mM, :, ddM, :] = g_direct
                    self.g_x[:, ddL, :, mM, :, ddM, :] = g_exchange
                    #self.e0 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        #print("Initial energy from dynamic amplitudes:", self.e0)
        print("Size/shape of amplitude tensor:", self.t2.size, self.t2.shape )

    def compute_pair_fragment_energy(self):
        """
        Compute fragment energy
        """
        #print(self.mM, type(self.mM))
        e_mp2 = 0
        e_mp2_ = 0
        N_virt = self.n_virtual_cells
        #print(self.f1.fragment, self.f2.fragment)
        
        #mM = self.d_ii.coords[] #occupied index only runs over fragment
        print(self.vv_indx)
        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = np.array(self.d_ia.cget(dL)[self.f1.fragment[0],:], dtype = bool) # dL index mask

            dL_i_ = np.array(self.d_ia.cget(dL - self.M)[self.f1.fragment[0],:], dtype = bool) # dL index mask
            
            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM]
                dM_i = np.array(self.d_ia.cget(dM)[self.f2.fragment[0],:], dtype = bool) # dM index mask
                dM_i_ = np.array(self.d_ia.cget(dM - self.M)[self.f2.fragment[0],:], dtype = bool) # dM index mask
                
                #print(dM_i, dL_i)
                # Using multiple levels of masking, probably some other syntax makes more sense
                
                g_direct = self.g_d[:,ddL,:,self.mM, :, ddM, :] #[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]
                g_exchange = self.g_x[:,ddL,:,self.mM, :, ddM, :] #[self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i]

                
                
                t = self.t2[:,ddL,:,self.mM, :, ddM, :] #[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]
                #print(np.linalg.norm(t), np.linalg.norm(g_direct), np.linalg.norm(g_exchange))
                #print(t.shape)
                e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
                #print("e_mp2:", e_mp2)
                # Then the other contribution (see notes, expression for pair fragment energy)
                
                #print(self.mM, self.mM_, self.t2[:,ddL,:,self.mM, :, ddM, :].shape)
                #print("g_d", self.t2[:,ddL,:,self.mM, :, ddM, :].shape)
                #print(dM_i_)
                #print(self.vv_indx[ddL, self.mM], self.vv_indx[ddM])
                
                
                g_direct = self.g_d[:,self.vv_indx[ddL, self.mM],:,self.mM_, :, self.vv_indx[ddM, self.mM], :] #[self.f2.fragment][:, dL_i_][:, :, self.f1.fragment][:,:,:,dM_i_]
                g_exchange = self.g_x[:,self.vv_indx[ddL, self.mM],:,self.mM_, :, self.vv_indx[ddM, self.mM], :] #[self.f2.fragment][:, dM_i_][:, :, self.f1.fragment][:,:,:,dL_i_]
                
                # BUT a,b and cell indices are just dummy-indices; we are iterating over the same space

                g_direct = self.g_d[:,ddL,:,self.mM_, :, ddM, :] #[self.f2.fragment][:, dL_i_][:, :, self.f1.fragment][:,:,:,dM_i_]
                g_exchange = self.g_x[:,ddL,:,self.mM_, :, ddM, :] #[self.f2.fragment][:, dM_i_][:, :, self.f1.fragment][:,:,:,dL_i_]
                



                #g_direct = self.g_d[:,ddL,:,self.mM_, :, ddM, :] #[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]
                #g_exchange = self.g_x[:,ddL,:,self.mM_, :, ddM, :] #[self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i]


                #print(self.vv_indx[ddL, self.mM], dL - self.M, self.d_ia.coords[self.vv_indx[ddL, self.mM]])
                #print(ddL, self.mM, ddM, self.vv_indx[ddL, self.mM])
                #print("i", dL-self.M, "a", -self.M, "j", dM-self.M, "b")
                #print("i", self.d_ia.coords[ self.vv_indx[ddL, self.mM]], "a", - self.M, "j", self.d_ia.coords[self.vv_indx[ddM, self.mM]], "b")
                
                #print(self.vv_indx[ddL, self.mM], self.vv_indx[ddM, self.mM])
                if not np.all(dL - self.M == self.d_ia.coords[self.vv_indx[ddL, self.mM]]):
                    break
                
                t = self.t2[:,ddL,:,self.mM_, :, ddM, :] #[self.f2.fragment][:, dL_i_][:, :, self.f1.fragment][:,:,:,dM_i_]
                
                #print(np.linalg.norm(g_direct), np.linalg.norm(g_exchange), np.linalg.norm(t))
                #print(np.linalg.norm(t), np.linalg.norm(g_direct), np.linalg.norm(g_exchange))
                #print(t.shape, g_direct.shape, g_exchange.shape)
                e_mp2_ += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
                #print("e_mp2:", e_mp2)
                #print(" ")
        print(e_mp2, e_mp2_)
        return e_mp2

    def solve(self, norm_thresh = 1e-10):
        """
        Converge fragment (AOS) amplitudes within occupied and virtual extents 
        """
        nocc = self.f1.p.get_nocc()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        

        vp_indx = mapgen(virtual_extent, pair_extent)
        pp_indx = mapgen(pair_extent, pair_extent)

        for ti in np.arange(100):
            t2_new = np.zeros_like(self.t2)
            for dL in np.arange(self.n_virtual_cells):
                dLv = self.d_ia.coords[dL]
                dL_i = np.array(self.d_ia.cget(dLv)[self.f1.fragment[0], :], dtype = bool) # dL index mask

                for dM in np.arange(self.n_virtual_cells): 
                    dMv = self.d_ia.coords[dM]
                    dM_i = np.array(self.d_ia.cget(dMv)[self.f2.fragment[0], :], dtype = bool)# dM index mask
                    for M in np.arange(self.n_occupied_cells):
                        Mv = self.d_ii.coords[M]
                        M_i = np.array(self.d_ii.cget(Mv)[self.f1.fragment[0], :], dtype = bool)  # M index mask

                        tnew = -self.g_d[:, dL, :, M, :, dM, :]

                        # generate index mapping of non-zero amplitudes in cell
                        cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.f1.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                        # Perform contractions

                        # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                        Fac = self.f1.f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                        tnew -= np.einsum("iKcjb,Kac->iajb", self.t2[:, :, :, M, :, dM, :], Fac)

                        # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                        Fbc = self.f1.f_mo_aa.cget(virtual_extent - virtual_extent[dM])
                        tnew -= np.einsum("iajKb,Kbc->iajb", self.t2[:, dL, :, M, :, :, :], Fbc)
                        
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        #vpdL = mapgen(virtual_extent-Mv, pair_extent)
                        #Fki = self.f1.f_mo_ii.cget(-1*pair_extent)
                        #tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)
                        
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        # revised summation, include all non-zero blocks
                        #vpdL = mapgen(virtual_extent-Mv, pair_extent)
                        #Fki = self.f1.f_mo_ii.cget(Mv-1*pair_extent)
                        #tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vpdL[dL], :, :, :, vpdL[dM], :], Fki)
                        
                        vpdL = mapgen(virtual_extent-Mv, pair_extent)

                        non_zeros = (vpdL[dL]>=0) * (vpdL[dM]>=0)

                        Fki = self.f1.f_mo_ii.cget(Mv-1*pair_extent)
                        #print(self.t2[:, vpdL[dL], :, :, :, vpdL[dM], :].shape)
                        tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vpdL[dL][non_zeros], :, np.arange(self.n_occupied_cells)[non_zeros], :, vpdL[dM][non_zeros], :], Fki[non_zeros])
                        



                        # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                        Fkj = self.f1.f_mo_ii.cget(-1*pair_extent + pair_extent[M])
                        tnew += np.einsum("iaKkb,Kkj->iajb",self.t2[:, dL, :, :, :, dM, :], Fkj)

                        t2_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*self.e_iajb**-1).ravel()[cell_map]
                        
                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)

            self.t2 -= t2_new
            rnorm = np.linalg.norm(t2_new)
            #print(ti, np.linalg.norm(self.t2))
            if rnorm<norm_thresh:

                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti, np.linalg.norm(t2_new)))
                break


class fragment_amplitudes():
    """
    Class that handles t2 amplitudes with dynamically increasing size
    """
    def __init__(self, p, wannier_centers, coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 3.0, occupied_cutoff = 1.0):
        self.p = p #prism object
        self.d = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers) # distance matrix
        
        self.d_ii = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[:p.get_nocc()])
        self.d_ia = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[p.get_nocc():])
        self.fragment = fragment

        self.ib = ib #integral builder
        self.virtual_cutoff = virtual_cutoff
        self.occupied_cutoff = occupied_cutoff
        self.min_elm = np.min(self.d.blocks[:-1], axis = (1,2)) #array for matrix-size calc
        
        self.min_elm_ii = np.min(self.d_ii.blocks[:-1], axis = (1,2))
        self.min_elm_ia = np.min(self.d_ia.blocks[:-1], axis = (1,2))
        
        self.f_mo_ii = f_mo_ii # Occupied MO-Fock matrix elements
        self.f_mo_aa = f_mo_aa # Virtual MO-Fock matrix elements

        self.init_amplitudes()
    def init_amplitudes(self):
        """
        Initialize the amplitudes using the MP2-like starting guess
        """
        self.n_virtual_cells = np.sum(self.min_elm_ia<=self.virtual_cutoff)
        self.n_occupied_cells = np.sum(self.min_elm_ii<=self.occupied_cutoff)

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1,self.fragment[0]]<=self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<=self.occupied_cutoff)
    
        
        n_occ = self.p.get_nocc()     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells # Number of occupied cells
        n_virt = self.p.get_nvirt()   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells # Number of virtual cells
        

        self.t2  = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = float)

        self.g_d = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = float)

        self.g_x = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = float)

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
    
        self.e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]


        self.e0 = 0

        sequence = []


        for ddL in np.arange(N_virt):
            for ddM in np.arange(ddL, N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]

                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]

                    # Get exchange block coordinates
                    ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM - M) ]
                    ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL + M) ]

                    



                    # Negative M 
                    mM_ = self.d_ia.mapping[ self.d_ia._c2i(-M) ]
                    
                    # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                    #                ^                 ^           ^          ^
                    #            Calculate these    ex/direct    store here   1=transpose

                    sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # direct
                    sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,  1]) # direct, transposed


                    # For fragments, exchange only required for only M = (0,0,0) 
                    # EOS always has the two occupied indices in the fragment, ie the refcell
                    if np.sum(M**2) == 0:
                        #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                        #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed

                        sequence.append([ddL_M, mM, ddM_M, 1, ddL, mM , ddM,0])  # exchange
                        sequence.append([ddL_M, mM, ddM_M, 1, ddM, mM_, ddL,1]) # exchange, transposed


        self.initialize_blocks(sequence)






    def initialize_blocks(self, sequence):
        #print("Initialization sequence:")
        sequence = np.array(sequence)




        # Sort blocks by dL:
        a = np.argsort(sequence[:,0])
        sequence = sequence[a]
        #print(sequence.shape)
        sequence = np.append(sequence, [ [-1000,0,0,0, 0, 0, 0,0] ], axis = 0) #-100 Just to make sure :-)
        #print(sequence.shape)
        j = 0
        for i in np.arange(len(sequence)):
            if sequence[i,0] != sequence[j,0]:
                a = np.argsort(sequence[j:i, 2])
                sq_i = sequence[j:i][a]

                sq_i = np.append(sq_i, [ [-1000,0,-1000,0, 0, 0, 0,0] ], axis = 0) #-100 Just to make sure :-)




                #print(sq_i)

                dL = self.d_ia.coords[sq_i[0,0]]
                


                #print(sq_i)
                k = 0
                for l in np.arange(len(sq_i)):
                    

                    if sq_i[k,2] != sq_i[l,2]:
                        dM = self.d_ia.coords[sq_i[k,2]]


                        #print(sq_i[k:l])
                        # Integrate here, loop over M
                        I, Ishape = self.ib.getorientation(dL, dM)

                        for m in sq_i[k:l]:
                            M = self.d_ii.coords[m[1]]
                            ddL, mM, ddM = m[0], m[1], m[2]
                            #print(self.g_x.shape, ddL, mM, ddM)
                            #print(dL, M, dM)
                            #print(I.cget(M).shape, Ishape)
                            if m[7] == 0:
                                if m[3] == 0:
                                    # Direct contribution
                                    self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)
                                    self.t2[:,  ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)*self.e_iajb**-1
                                if m[3] == 1:
                                    # Exchange contribution
                                    ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                    self.g_x[:, ddL_, :, mM_, :, ddM_, :] = I.cget(M).reshape(Ishape)
                            if m[7] == 1:
                                if m[3] == 0:
                                    # Direct contribution
                                    self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)
                                    self.t2[:,  ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)*self.e_iajb**-1
                                if m[3] == 1:
                                    # Exchange contribution
                                    ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                    self.g_x[:, ddL_, :, mM_, :, ddM_, :] = I.cget(M).T.reshape(Ishape)


                        k = l*1
                        
                j = i*1

  


    def compute_energy(self):
        """
        Computes the energy of entire AOS 
        """
        e_mp2 = 0
        N_virt = self.n_virtual_cells
        N_occ = self.n_occupied_cells

        for ddL in np.arange(N_virt):
            for ddM in np.arange(N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]
                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]

                
                    g_direct = self.g_d[:,ddL,:,mM, :, ddM, :]
                    g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :]
                    t = self.t2[:,ddL,:,mM, :, ddM, :]
                    e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        return e_mp2
    
    def compute_fragment_energy(self):
        """
        Compute fragment energy
        """
        e_mp2 = 0
        N_virt = self.n_virtual_cells
        
        mM = 0 #occupied index only runs over fragment

        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[0,:]<self.virtual_cutoff # dL index mask
            
            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM]
                dM_i = self.d_ia.cget(dM)[0,:]<self.virtual_cutoff # dM index mask

               # Using multiple levels of masking, probably some other syntax makes more sense
                
                g_direct = self.g_d[:,ddL,:,mM, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]
                g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :][self.fragment][:, dM_i][:, :, self.fragment][:,:,:,dL_i]

                

                t = self.t2[:,ddL,:,mM, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]
                e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        return e_mp2
    
    

    def init_cell_batch(self, coords):
        """
        Could perhaps save some seconds when initializing new cells
        """
        pass



    def init_cell(self, ddL, mmM, ddM):
        """
        Initialize tensors in cell ( 0 i , dL a | M j , dM b )
        ddL, mmM and ddM are integer numbers 
        """
        dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]
        M = self.d_ii.coords[mmM]

        g_direct = self.ib.getcell(dL, M, dM)
        g_exchange = self.ib.getcell(dM+M, M, dL-M)

        self.t2[:, ddL, :, mmM, :, ddM, :]  = g_direct*self.e_iajb**-1
        self.g_d[:, ddL, :, mmM, :, ddM, :] = g_direct
        self.g_x[:, ddL, :, mmM, :, ddM, :] = g_exchange

    def autoexpand_virtual_space(self, n_orbs = 10):
        """
        Include n_orbs more virtual orbitals in the virtual extent
        """
        new_cut = np.sort(self.d_ia.blocks[:-1, self.fragment[0]][self.d_ia.blocks[:-1, self.fragment[0]]>self.virtual_cutoff])[n_orbs -1] 
        #print("Increasing virtual cutoff:", self.virtual_cutoff, "->", new_cut)
        self.set_extent(new_cut, self.occupied_cutoff)
    

    def autoexpand_occupied_space(self, n_orbs = 10):
        """
        Include n_orbs more orbitals in the occupied extent
        """
        new_cut = np.sort(self.d_ii.blocks[:-1, self.fragment[0]][self.d_ii.blocks[:-1, self.fragment[0]]>self.occupied_cutoff])[n_orbs-1] 
        #print("Increasing occupied cutoff:", self.occupied_cutoff, "->", new_cut)
        self.set_extent(self.virtual_cutoff, new_cut)
        

    
    def set_extent(self, virtual_cutoff, occupied_cutoff):
        """
        Set extent of local domain
        Cutoffs are given in bohr, all orbitals within the specified cutoffs are included
        """
        self.virtual_cutoff = virtual_cutoff
        self.occupied_cutoff = occupied_cutoff

        Nv = np.sum(self.min_elm_ia<=self.virtual_cutoff)
        No = np.sum(self.min_elm_ii<=self.occupied_cutoff)

        n_occ = self.p.get_nocc()
        n_virt = self.p.get_nvirt()

        # Note: forking here is due to intended future implementation of block-specific initialization
        if Nv > self.n_virtual_cells:
            if No > self.n_occupied_cells:
                print("Extending both occupied and virtuals")
                # Extend tensors in both occupied and virtual direction
                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = float)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = float)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = float)
                g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                self.g_x = g_x_new

                # Initialize empty blocks
                sequence = []
                for ddL in np.arange(Nv):
                    dL = self.d_ia.coords[ddL]
                    for ddM in np.arange(ddL, Nv):
                        dM = self.d_ia.coords[ddM]
                        for mmM in np.arange(No):
                            M = self.d_ii.coords[mmM]
                            #t0 = time.time()
                            if np.abs(self.t2[:,ddL, :, mmM, :, ddM, :]).max()<=10e-12:


                                M = self.d_ii.coords[mmM]

                                # Get exchange block coordinates
                                ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM - M) ]
                                ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL + M) ]

                                



                                # Negative M 
                                mmM_ = self.d_ia.mapping[ self.d_ia._c2i(-M) ]
                                
                                # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                                #                ^                 ^           ^          ^
                                #            Calculate these    ex/direct    store here   1=transpose
                                
                                sequence.append([ddL, mmM, ddM,   0, ddL, mmM, ddM,   0]) # direct
                                sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed



                                #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                                #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed
                                # For fragments, exchange only required for only M = (0,0,0) 
                                # EOS always has the two occupied indices in the fragment, ie the refcell
                                if np.sum(M**2) == 0:
                                    sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                                    sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed


                                #self.init_cell(ddL, mmM, ddM)
                            #print(ddL, mmM, ddM, time.time()-t0)
                #print(time.time()-t0, " s spent on cell init.")
                self.initialize_blocks(sequence)
                            






            else:
                print("Extending virtuals.")
                # Extend tensors in the virtual direction
                
                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = float)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = float)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = float)
                g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                self.g_x = g_x_new

                # Initialize empty blocks
                sequence = []
                for ddL in np.arange(Nv):
                    dL = self.d_ia.coords[ddL]
                    for ddM in np.arange(ddL, Nv):
                        dM = self.d_ia.coords[ddM]
                        for mmM in np.arange(No):
                            M = self.d_ii.coords[mmM]
                            #t0 = time.time()
                            if np.abs(self.t2[:,ddL, :, mmM, :, ddM, :]).max()<=10e-12:


                                M = self.d_ii.coords[mmM]

                                # Get exchange block coordinates
                                ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM - M) ]
                                ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL + M) ]

                                



                                # Negative M 
                                mmM_ = self.d_ia.mapping[ self.d_ia._c2i(-M) ]
                                
                                # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                                #                ^                 ^           ^          ^
                                #            Calculate these    ex/direct    store here   1=transpose
                                
                                sequence.append([ddL, mmM, ddM,   0, ddL, mmM, ddM,   0]) # direct
                                sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed



                                #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                                #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed

                                # For fragments, exchange only required for only M = (0,0,0) 
                                # EOS always has the two occupied indices in the fragment, ie the refcell
                                if np.sum(M**2) == 0:
                                    sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                                    sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed


                                #self.init_cell(ddL, mmM, ddM)
                            #print(ddL, mmM, ddM, time.time()-t0)
                #print(time.time()-t0, " s spent on cell init.")
                self.initialize_blocks(sequence)

        else:
            if No > self.n_occupied_cells:
                print("extending occupied")
                # Extend tensors in the occupied dimension
                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = float)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = float)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = float)
                g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                self.g_x = g_x_new

                # Initialize empty blocks
                sequence = []
                for ddL in np.arange(Nv):
                    dL = self.d_ia.coords[ddL]
                    for ddM in np.arange(ddL,Nv):
                        dM = self.d_ia.coords[ddM]
                        for mmM in np.arange(No):
                            M = self.d_ii.coords[mmM]
                            #t0 = time.time()
                            if np.abs(self.t2[:,ddL, :, mmM, :, ddM, :]).max()<=10e-12:


                                M = self.d_ii.coords[mmM]

                                # Get exchange block coordinates
                                ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM - M) ]
                                ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL + M) ]

                                



                                # Negative M 
                                mmM_ = self.d_ia.mapping[ self.d_ia._c2i(-M) ]
                                
                                # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                                #                ^                 ^           ^          ^
                                #            Calculate these    ex/direct    store here   1=transpose
                                
                                sequence.append([ddL, mmM, ddM,   0, ddL, mmM, ddM,   0]) # direct
                                sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed

                                # For fragments, exchange only required for only M = (0,0,0) 
                                # EOS always has the two occupied indices in the fragment, ie the refcell
                                if np.sum(M**2) == 0:
                                    sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                                    sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed



                                #self.init_cell(ddL, mmM, ddM)
                            #print(ddL, mmM, ddM, time.time()-t0)
                #print(time.time()-t0, " s spent on cell init.")
                self.initialize_blocks(sequence)

            else:
                

                
                self.t2 = self.t2[:, :Nv, :, :No, :, :Nv, :]
                self.g_d = self.g_d[:, :Nv, :, :No, :, :Nv, :]
                self.g_x = self.g_x[:, :Nv, :, :No, :, :Nv, :]
        
        # Update domain measures
        self.n_virtual_cells = Nv
        self.n_occupied_cells = No

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1,self.fragment[0]]<=self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<=self.occupied_cutoff)
    
    def print_configuration_space_data(self):
        """
        Prints extent sizes in terms of number of orbitals
        """
        print("%i virtual orbitals included in fragment." % self.n_virtual_tot)
        print("%i occupied orbitals included in fragment." % self.n_occupied_tot)

    def solve(self, norm_thresh = 1e-10):
        """
        Converge fragment (AOS) amplitudes within occupied and virtual extents 
        """
        nocc = self.p.get_nocc()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        vp_indx = mapgen(virtual_extent, pair_extent)
        pp_indx = mapgen(pair_extent, pair_extent)

        #print(vp_indx)
        #print(pp_indx)

        for ti in np.arange(100):
            t2_new = np.zeros_like(self.t2)
            for dL in np.arange(self.n_virtual_cells):
                dLv = self.d_ia.coords[dL]
                dL_i = self.d_ia.cget(dLv)[0,:]<self.virtual_cutoff # dL index mask

                for dM in np.arange(self.n_virtual_cells): 
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[0,:]<self.virtual_cutoff # dM index mask
                    for M in np.arange(self.n_occupied_cells):
                        Mv = self.d_ii.coords[M]
                        M_i = self.d_ii.cget(Mv)[0,:]<self.occupied_cutoff # M index mask

                        tnew = -self.g_d[:, dL, :, M, :, dM, :]

                        # generate index mapping of non-zero amplitudes in cell
                        cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                        # Perform contractions

                        # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                        Fac = self.f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                        tnew -= np.einsum("iKcjb,Kac->iajb", self.t2[:, :, :, M, :, dM, :], Fac)

                        # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                        Fbc = self.f_mo_aa.cget(virtual_extent - virtual_extent[dM])
                        tnew -= np.einsum("iajKc,Kbc->iajb", self.t2[:, dL, :, M, :, :, :], Fbc)
                        
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        #Fki = self.f_mo_ii.cget(-1*pair_extent)
                        #tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)
                        


                        """
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        vpdL = mapgen(virtual_extent-Mv, pair_extent)
                        Fki = self.f_mo_ii.cget(Mv-1*pair_extent)
                        #print(self.t2[:, vpdL[dL], :, :, :, vpdL[dM], :].shape)
                        #print(vpdL)
                        #assert(np.all(vpdL>0)), "Negative indexing in vpdL"

                        non_zeros = (vpdL[dL]>=0) *  (vpdL[dM]>=0) #screening matrix to avoid references to amplitudes outside extent
                        #print(non_zeros)
                        #print(self.t2[:, vpdL[dL], :, np.arange(self.n_occupied_cells), :, vpdL[dM], :].shape, Fki.shape)


                        tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vpdL[dL][non_zeros], :, np.arange(self.n_occupied_cells)[non_zeros], :, vpdL[dM][non_zeros], :], Fki[non_zeros])
                        """

                        # Conceptually simpler approach
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        Fki = self.f_mo_ii.cget(-1*pair_extent)
                        
                        M_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[M] - pair_extent ) ]
                        dL_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dL] - pair_extent) ]
                        dM_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dM] - pair_extent) ]
                        #make sure indices is in correct domain (not negative or beyond extent)
                        nz = (M_range<self.n_occupied_cells)*(dL_M<self.n_virtual_cells)*(dM_M<self.n_virtual_cells)*\
                             (M_range>=0)*(dL_M>=0)*(dM_M>=0)

                        tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, dL_M[nz], :, M_range[nz], :, dM_M[nz], :], Fki[nz])
                        


                        # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                        Fkj = self.f_mo_ii.cget(-1*pair_extent + pair_extent[M])
                        tnew += np.einsum("iaKkb,Kkj->iajb",self.t2[:, dL, :, :, :, dM, :], Fkj)

                        t2_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*self.e_iajb**-1).ravel()[cell_map]
                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)

            self.t2 -= t2_new
            rnorm = np.linalg.norm(t2_new)
            if rnorm<norm_thresh:

                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti, np.linalg.norm(t2_new)))
                break


class attenuation_tuner():
    def __init__(self, p, args):
        
        # Generate BT - identity matrix, small cutoff
        c = tp.tmat()
        c.load_nparray(np.ones((3,p.get_n_ao(), p.get_n_ao()), dtype =float),tp.lattice_coords([1,0,0]))
        c.blocks[:-1] = 0.0
        c.cset([0,0,0], np.eye(p.get_n_ao()))

        c_occ, c_virt = PRI.occ_virt_split(c,p)
        
        ib_ao = PRI.integral_builder_ao(c,p,attenuation = 1.2, auxname="ri-fitbasis", circulant=args.circulant, robust = args.robust)

        # AO Fock matrix
        f_ao = tp.tmat()
        f_ao.load(args.fock_matrix)

        # Compute MO Fock matrix
        f_mo = c.tT().cdot(f_ao*c, coords = c.coords)

        f_mo_aa = c_virt.tT().cdot(f_ao*c_virt, coords = c.coords)
        f_mo_ii = c_occ.tT().cdot(f_ao*c_occ, coords = c.coords)



        # Compute energy denominator
        f_aa = f_mo.cget([0,0,0])[np.arange(p.get_nocc(),p.get_n_ao()), np.arange(p.get_nocc(),p.get_n_ao())]
        f_ii = f_mo.cget([0,0,0])[np.arange(p.get_nocc()),np.arange(p.get_nocc()) ]
        
        e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]


        # Wannier centers
        wcenters = np.load(args.wcenters)*0


        d = dd.build_distance_matrix(p, tp.lattice_coords([6,6,6]), wcenters, wcenters)
        for cc in d.coords:
            d.cset(cc, np.zeros_like(d.cget(cc))+np.sqrt(np.sum(cc**2)))
        


        



        center_fragment = dd.atomic_fragmentation(p, d, 3.0)[0]
        omegas = np.exp(np.linspace(np.log(0.15),np.log(10),10))

        errors = []
        for i in np.arange(10):
            omega = omegas[-(i+1)]
            #print(omega)
            ib_ri = PRI.integral_builder(c,p,attenuation =omega, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust)
            

            #print(ib_ri.getcell([0,0,0], [0,0,0], [0,0,0]))
            #print(ib_ao.getcell([0,0,0], [0,0,0], [0,0,0]))

            print(omega, np.linalg.norm(ib_ri.getcell([0,0,0], [0,0,0], [0,0,0]) - ib_ao.getcell([0,0,0], [0,0,0], [0,0,0])), np.max(np.abs(ib_ri.getcell([0,0,0], [0,0,0], [0,0,0]) - ib_ao.getcell([0,0,0], [0,0,0], [0,0,0]))))

            print(omega, np.linalg.norm(ib_ri.getcell([0,0,0], [1,0,0], [0,0,0]) - ib_ao.getcell([0,0,0], [1,0,0], [0,0,0])), np.max(np.abs(ib_ri.getcell([0,0,0], [1,0,0], [0,0,0]) - ib_ao.getcell([0,0,0], [1,0,0], [0,0,0]))))
            
            #print( (ib_ri.getcell([0,0,0], [0,0,0], [0,0,0]) - ib_ao.getcell([0,0,0], [0,0,0], [0,0,0]) )[0,:,0,:] )
            #print( (ib_ri.getcell([0,0,0], [1,0,0], [0,0,0]) - ib_ao.getcell([0,0,0], [1,0,0], [0,0,0]) )[0,:,0,:] )

            print( ib_ri.getcell([0,0,0], [0,0,0], [0,0,0])[0,:,0,:] )
            print( ib_ao.getcell([0,0,0], [0,0,0], [0,0,0])[0,:,0,:] )
            
            print( ib_ri.getcell([0,0,0], [1,0,0], [0,0,0])[0,:,0,:] )
            print( ib_ao.getcell([0,0,0], [1,0,0], [0,0,0])[0,:,0,:] )
            err = []
            err.append(ib_ri.getcell([0,0,0], [0,0,0], [0,0,0])[0,:,0,:])
            err.append(ib_ao.getcell([0,0,0], [0,0,0], [0,0,0])[0,:,0,:])
            err.append(ib_ri.getcell([0,0,0], [1,0,0], [0,0,0])[0,:,0,:])
            err.append(ib_ao.getcell([0,0,0], [1,0,0], [0,0,0])[0,:,0,:])

            errors.append(err)
            #a_frag_ao=fragment_amplitudes(p, wcenters, c.coords, center_fragment, ib_ao, f_mo_ii, f_mo_aa, virtual_cutoff = 10.0, occupied_cutoff = 1.0)
            
            #a_frag_ri=fragment_amplitudes(p, wcenters, c.coords, center_fragment, ib_ri, f_mo_ii, f_mo_aa, virtual_cutoff = 10.0, occupied_cutoff = 1.0)

            #a_frag_ao.solve()
            #a_frag_ri.solve()

            #ao_energy = a_frag_ao.compute_fragment_energy()
            #ri_energy = a_frag_ri.compute_fragment_energy()

            #print(omega, ao_energy, ri_energy)
            np.save("errors_per_orb_cc_pvtz.npy", np.array(errors))
        












def converge_fragment_amplitudes(t2, G_direct, f_mo_ii, f_mo_aa, di_virt, di_occ, fragment,p):
    """
    Solve MP2 equations for the given fragment, return amplitudes
    
    Input parameters

     t2     - array containing initial guess amplitudes

    This function is obsolete, just used for debugging purposes

    """
    nocc = p.get_nocc()


    virtual_extent = di_virt.coords
    pair_extent = di_occ.coords

    vp_indx = mapgen(virtual_extent, pair_extent)
    pp_indx = mapgen(pair_extent, pair_extent)

    for ti in np.arange(100):
        #t2_new = np.zeros((nocc, ndom,  nvirt, pdom, nocc, ndom, nvirt), dtype = float)
        t2_new = np.zeros_like(t2)
        for dL in np.arange(di_virt.coords.shape[0]):
            for dM in np.arange(di_virt.coords.shape[0]): 
                for M in np.arange(di_occ.coords.shape[0]):
                    tnew = -G_direct[:, dL, :, M, :, dM, :]

                  

                    # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                    Fac = f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                    #tb = t2[:, :, :, M, :, dM, :][di_occ.cget(di_occ.coords[M])[:,0]]
                    tnew -= np.einsum("iKcjb,Kac->iajb", t2[:, :, :, M, :, dM, :], Fac)

                    # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                    Fbc = f_mo_aa.cget(virtual_extent - virtual_extent[dM])
                    tnew -= np.einsum("iajKb,Kbc->iajb", t2[:, dL, :, M, :, :, :], Fbc)
                    
                    # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                    
                    Fki = f_mo_ii.cget(-1*pair_extent)
                    tnew += np.einsum("Kkajb,Kki->iajb",t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)
                    
                    # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                    
                    Fkj = f_mo_ii.cget(-1*pair_extent + pair_extent[M])
                    tnew += np.einsum("iaKkb,Kkj->iajb",t2[:, dL, :, :, :, dM, :], Fkj)

                    
                    # + \left(t^{\Delta L a, \Delta Mb}_{L'k,Mj}\right)_{n}\varepsilon^{\Delta L a, \Delta Mb}_{0i,Mj},
                    #tnew += t2[:, dL, :, M, :, dM, :] #*E[:,dL,:,M,:,dM,:]**-1

                    t2_new[:, dL, :, M, :, dM, :] = tnew*e_iajb**-1
        #print("Residual norm: ")
        t2 -= t2_new
        rnorm = np.linalg.norm(t2_new)
        if rnorm<1e-10:

            print("Converged in %i itertions with amplitude gradient norm %.2e." % (ti, np.linalg.norm(t2_new)))
            break
    return t2
    



if __name__ == "__main__":
    os.environ["LIBINT_DATA_PATH"] = os.getcwd() 
    print("""#########################################################
##    ,--.   ,--.      ,------.  ,------. ,-----.      ##
##     \  `.'  /,-----.|  .-.  \ |  .---''  .--./      ##
##      .'    \ '-----'|  |  \  :|  `--, |  |          ## 
##     /  .'.  \       |  '--'  /|  `---.'  '--'\      ##
##    '--'   '--'      eXtended local correlation      ##
##                                                     ##
##  Use keyword "--help" for more info                 ## 
#########################################################""")

        
    # Parse input
    parser = argparse.ArgumentParser(prog = "X-DEC: eXtended Divide-Expand-Consolidate scheme",
                                     description = "Local correlation for periodic systems.",
                                     epilog = "Author: Audun Skau Hansen (e-mail: a.s.hansen@kjemi.uio.no)")
    parser.add_argument("project_file", type = str, help ="input file for project (.d12 file)")
    parser.add_argument("coefficients", type= str,help = "Coefficient matrix from Crystal")
    parser.add_argument("fock_matrix", type= str,help = "AO-Fock matrix from Crystal")
    parser.add_argument("-fitted_coeffs", type= str,help="Array of coefficient matrices from RI-fitting")
    parser.add_argument("auxbasis", type = str, help="Auxiliary fitting basis.")
    parser.add_argument("wcenters", type = str, help="Wannier centers")
    parser.add_argument("-attenuation", type = float, default = 1.2, help = "Attenuation paramter for RI")
    parser.add_argument("-basis_truncation", type = float, default = 0.5, help = "Truncate AO-basis function below this threshold." )
    parser.add_argument("-fot", type = float, default = 0.001, help = "fragment optimization treshold")
    parser.add_argument("-circulant",default = False, action = "store_true", help = "fragment optimization treshold")
    parser.add_argument("-attenuated_truncation", type = float, default = 1e-14, help = "Truncate blocks in the attenuated matrix where (max) elements are below this threshold." )
    parser.add_argument("-robust", default = False, action = "store_true", help = "Enable Dunlap robust fit for improved integral accuracy.")
    parser.add_argument("-disable_static_mem", default = False, action = "store_true", help = "Recompute AO integrals for new fitting sets.")
    parser.add_argument("-n_core", type = int, default = 0, help = "Number of core orbitals (the first n_core orbitals will not be correlated).")
    parser.add_argument("-skip_fragment_optimization", default = False, action = "store_true", help = "Skip fragment optimization (for debugging, will run faster but no error estimate.)")
    
    args = parser.parse_args()

    # Print run-info to screen
    print("Author : Audun Skau Hansen, audunsh4@gmail.com, 2019")

    import sys
    print("Git rev:", sp.check_output(['git', 'rev-parse', 'HEAD'], cwd=sys.path[0]))
    print("_________________________________________________________")
    print("System data")
    print("_________________________________________________________")
    print("Geometry + AO basis :", args.project_file)
    print("Wannier basis       :", args.coefficients)
    print("Number of core orbs.:", args.n_core)
    print("Auxiliary basis     :", args.auxbasis)
    print("Aux. basis cutoff   :", args.basis_truncation)
    print(" ")
    print("Attenuation         :", args.attenuation)
    print("Att. truncation     :", args.attenuated_truncation)
    print(" ")
    print("Dot-product         :", ["Block-Toeplitz", "Circulant"][int(args.circulant)])
    print(" ")
    print("RI fitting          :", ["Non-robust", "Robust"][int(args.robust)])
    print("FOT                 :", args.fot)
    print("_________________________________________________________")


    # Load system
    p = pr.prism(args.project_file)
    p.n_core = args.n_core


    # Fitting basis
    auxbasis = PRI.basis_trimmer(p, args.auxbasis, alphacut = args.basis_truncation)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()


    #attenuation_tuner(p, args)
    

    # Wannier coefficients
    c = tp.tmat()
    c.load(args.coefficients)

    c_occ, c_virt = PRI.occ_virt_split(c,p)

    # Remove core orbitals
    #p.n_core = 1
    #c_occ = tp.tmat()
    #c_occ.load_nparray(c_occ_full.blocks[:-1, :, 1:], c_occ_full.coords)


    
    # AO Fock matrix
    f_ao = tp.tmat()
    f_ao.load(args.fock_matrix)

    # Compute MO Fock matrix
    f_mo = c.tT().cdot(f_ao*c, coords = c.coords)

    f_mo_aa = c_virt.tT().cdot(f_ao*c_virt, coords = c.coords)
    f_mo_ii = c_occ.tT().cdot(f_ao*c_occ, coords = c.coords)



    # Compute energy denominator
    #f_aa = f_mo.cget([0,0,0])[np.arange(p.get_nocc(),p.get_n_ao()), np.arange(p.get_nocc(),p.get_n_ao())]
    #f_ii = f_mo.cget([0,0,0])[np.arange(p.get_nocc()),np.arange(p.get_nocc()) ]
    
    f_aa = np.diag(f_mo_aa.cget([0,0,0]))
    f_ii = np.diag(f_mo_ii.cget([0,0,0]))

    e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]


    # Wannier centers
    wcenters = np.load(args.wcenters)[p.n_core:]

    # Initialize integrals 
    if args.disable_static_mem:
        ib = PRI.integral_builder(c,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[1,0,0], circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust)
    else:
        ib = PRI.integral_builder_static(c,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[1,0,0], circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust)
    
    
    # Test symmetries in the g-tensor
    d = dd.build_distance_matrix(p, c.coords, wcenters, wcenters)

    """
    for ddL in np.arange(d.coords.shape[0]):
        for ddM in np.arange(d.coords.shape[0]):
            dL = d.coords[ddL+1]
            dM = d.coords[ddM+2]
            I_direct, shape = ib.getorientation(dL, dM)
            I_exchange, shape = ib.getorientation(dM, dL)
            for mmM in np.arange(d.coords.shape[0]):
                
                M  = d.coords[mmM]
                
                
                # Shifted indices
                ddL_M = d.mapping[d._c2i(dL - M)]
                ddM_M = d.mapping[d._c2i(dM - M)]
                mmM_  = d.mapping[d._c2i(-M)    ]

                I_explicit, shape = ib.getorientation(dM,  dL)

                print("Allclose:", np.linalg.norm(I_explicit.cget(M).T - I_direct.cget(M)))
                print("dL:", dL)
                print("dM:", dM)
                print("M :", M)
                print(I_explicit.cget(-M).T[:4,:4]) 
                print(I_direct.cget(M)[:4,:4])
                print(I_exchange.cget(M)[:4,:4])
    """










    # Initialize domain definitions



    d = dd.build_distance_matrix(p, c.coords, wcenters, wcenters)




    center_fragments = dd.atomic_fragmentation(p, d, 3.0)
    print(" ")
    print("Fragmentation of orbital space")
    print(center_fragments)
    print(" ")

    print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
    print(" ")
    

    # Converge atomic fragment energies
    
    # Initial fragment extents
    virt_cut = 3.0
    occ_cut = 6.0

    for fragment in center_fragments:
        

        #ib.fragment = fragment
        t0 = time.time()
        a_frag = fragment_amplitudes(p, wcenters, c.coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 2.0, occupied_cutoff = 2.0)
        print("Frag init:", time.time()-t0)
        
        
        print(" ")
        a_frag.solve()
        
        # Converge to fot
        E_prev_outer = a_frag.compute_fragment_energy()
        E_prev = E_prev_outer*1.0
        dE_outer = 10

        print("Initial fragment energy: %.8f" % E_prev)
        print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
        print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))        
            
        if not args.skip_fragment_optimization:
            print("Running fragment optimization for:")
            print(fragment)
            #print("Initial cutoffs:")
            
            while dE_outer>args.fot:
                dE = 10
                e_virt = []
                #
                while dE>args.fot:
                    #for i in np.arange(30):
                    print("e_prev:", E_prev)

                    #print("--- virtual")
                    t_0 = time.time()
                    a_frag.autoexpand_virtual_space(n_orbs=6)
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                    
                    t_1 = time.time()
                    a_frag.solve()
                    t_2 = time.time()
                    E_new = a_frag.compute_fragment_energy()
                    t_3 = time.time()
                    print("E_new:", E_new)
                    #a_frag.print_configuration_space_data()
                    dE = np.abs(E_prev - E_new)
                    
                    print("_________________________________________________________")
                    print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new, dE))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
                    print("Time (expand/solve/energy) (s) : %.1f / %.1f / %.1f" % (t_1-t_0, t_2-t_1, t_3-t_2))
                    print(" ")
                    e_virt.append(E_prev)
                    E_prev = E_new
                    
                    #print("---")
                print("Converged virtual space, expanding occupied space")
                print(e_virt)
                #dE = 10
                #print("--- occupied")
                a_frag.autoexpand_occupied_space(n_orbs=6)
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                
                a_frag.solve()
                E_new = a_frag.compute_fragment_energy()
                
                #a_frag.print_configuration_space_data()
                dE = np.abs(E_prev - E_new) 
                
                print("_________________________________________________________")
                print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new, dE))
                print("_________________________________________________________")
                print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
                print(" ")
                E_prev = E_new
                #print("---")

                while dE>args.fot:

                    #print("--- occupied")
                    a_frag.autoexpand_occupied_space(n_orbs=6)
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                
                    a_frag.solve()
                    E_new = a_frag.compute_fragment_energy()
                    
                    #a_frag.print_configuration_space_data()
                    dE = np.abs(E_prev - E_new)
                    
                    print("_________________________________________________________")
                    print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new, dE))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
                    print(" ")
                    E_prev = E_new
                    #print("---")
                dE_outer = np.abs(E_prev_outer - E_prev)
                print("dE_outer:", dE_outer)
                E_prev_outer = E_prev
            #print("Current memory usage of integrals (in MB):", ib.nbytes())
            print("_________________________________________________________")
            print("Final fragment containing occupied orbitals:", a_frag.fragment)
            print("Converged fragment energy: %.12f" % E_new)
            print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
            print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
            print("_________________________________________________________")
            print(" ")
            print(" ")

        pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([1,0,0]))
        print(pair.compute_pair_fragment_energy())
        pair.solve()
        print("Pair fragment energy for (1,0,0):", pair.compute_pair_fragment_energy())
        """
        


        pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([0,1,0]))
        print(pair.compute_pair_fragment_energy())
        pair.solve()
        print("Pair fragment energy for (0,1,0):",pair.compute_pair_fragment_energy())
        
        pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([0,0,1]))
        print(pair.compute_pair_fragment_energy())
        pair.solve()
        print("Pair fragment energy for (0,0,1):", pair.compute_pair_fragment_energy())

        pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([-1,0,0]))
        print(pair.compute_pair_fragment_energy())
        pair.solve()
        print("Pair fragment energy for (1,0,0):", pair.compute_pair_fragment_energy())


        pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([0,-1,0]))
        print(pair.compute_pair_fragment_energy())
        pair.solve()
        print("Pair fragment energy for (0,1,0):",pair.compute_pair_fragment_energy())
        
        pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([0,0,-1]))
        print(pair.compute_pair_fragment_energy())
        pair.solve()
        print("Pair fragment energy for (0,0,1):", pair.compute_pair_fragment_energy())

        #for pair in np.arange(1,10):
        #    print(a_frag.d_ii.coords[pair], "Pair fragment energy:", a_frag.compute_pair_fragment_energy(pair))
        #    print("-0.0000611450091260 (Gustav, 0,1)")
        #print(-0.114393980708, "(3D Neon, fot 0.0001 (Gustav))")
        """