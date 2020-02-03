#!/usr/bin/env python


import numpy as np

import ad

import os

import subprocess as sp

from ast import literal_eval

import argparse

import utils.toeplitz as tp

import lwrap.lwrap as li # Libint wrapper

import utils.objective_functions as of

import utils.prism as pr
import domdef as dd
import PRI
import time

"""
Functions to aid the mapping of blocks in tensors
"""

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
    Returns the intersection of 3-vectors between two sets of (N,3) arrays
    """
    distance = np.sum((coords1[:,None] - coords2[None,:])**2, axis = 2)

    coords1_in_coords2 = np.any(distance==0, axis = 0)
    return coords2[coords1_in_coords2]

def c2_not_in_c1(c1, c2, sort = False):
    """
    Returns the 3-vectors in c2 which are not present in c1
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
    def __init__(self, fragment_amplitudes_1, fragment_amplitudes_2, M, float_precision = np.float64):
        self.M = M # Translation of fragment 2
        self.f1 = fragment_amplitudes_1
        self.f2 = fragment_amplitudes_2

        # Option for low precision (memory saving)
        self.float_precision = float_precision

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

        self.t2 = np.zeros((nocc, N_virt, nvirt, N_occ, nocc, N_virt, nvirt), dtype = self.float_precision)
        self.g_d = np.zeros((nocc, N_virt, nvirt, N_occ, nocc, N_virt, nvirt), dtype = self.float_precision)
        self.g_x = np.zeros((nocc, N_virt, nvirt, N_occ, nocc, N_virt, nvirt), dtype = self.float_precision)

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
            for ddM in np.arange(N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]

                for mM in np.arange(N_occ):
                    # Further room for improvement here!! (only positive indidces needed) :-)
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

                        #ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM - M) ]
                        #ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL + M) ]

                        # Indices to calculate

                        ddL_index = self.cmap._c2i(dL)
                        ddM_index = self.cmap._c2i(dM)
                        ddL_M_index = self.cmap._c2i(dM + M)
                        ddM_M_index = self.cmap._c2i(dL - M)
                        mM_index    = self.cmap._c2i(M)
                        mM__index   = self.cmap._c2i(-M)







                        #print("Queueing ")
                        #print(ddL)



                        """
                        print(dL, M, dM)
                        print(ddL, mM, ddM)
                        print(" ==== ")
                        print(dM -M, M, dL + M)
                        ## Should test here if mM_ is in set
                        print(" ==== ")
                        print(ddL_M, mM,ddM_M)
                        print(" ")
                        """
                        #assert(ddM!=7)
                        # Negative M - where to store it
                        mM_ = self.d_ii.mapping[ self.d_ii._c2i(-M) ]

                        # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                        #                ^                 ^           ^          ^
                        #            Calculate these    ex/direct    store here   1=transpose

                        sequence.append([ddL_index, mM_index, ddM_index,   0, ddL, mM, ddM,   0]) # direct

                        # make sure - M in domain
                        if mM_<self.d_ii.coords.shape[0]:
                            sequence.append([ddL_index, mM_index, ddM_index,   0, ddM, mM_, ddL,  1]) # direct, transposed


                        if mM == self.mM:
                            # block inside EOS
                            sequence.append([ddL_M_index, mM_index, ddM_M_index, 1, ddL, mM , ddM,0])  # exchange
                            if mM_<self.d_ii.coords.shape[0]:
                                sequence.append([ddL_M_index, mM_index, ddM_M_index, 1, ddM, mM_, ddL,1]) # exchange, transposed



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
            #print(sequence[i])
            if sequence[i,0] != sequence[j,0]:
                a = np.argsort(sequence[j:i, 2])
                sq_i = sequence[j:i][a]

                sq_i = np.append(sq_i, [ [-1000,0,-1000,0, 0, 0, 0,0] ], axis = 0) #-100 Just to make sure :-)




                #print(sq_i)

                #dL = self.d_ia.coords[sq_i[0,0]]
                dL = self.cmap._i2c(np.array([sq_i[0,0]]))[0]


                #print(sq_i)
                k = 0
                for l in np.arange(len(sq_i)):


                    if sq_i[k,2] != sq_i[l,2]:
                        #print(sq_i[k,2])
                        #dM = self.d_ia.coords[sq_i[k,2]]
                        dM = self.cmap._i2c(np.array([sq_i[k,2]]))[0]


                        #print(sq_i[k:l])
                        # Integrate here, loop over M
                        #print(dL, dM)
                        I, Ishape = self.f1.ib.getorientation(dL, dM)

                        for m in sq_i[k:l]:
                            #M = self.d_ii.coords[m[1]]
                            M = self.cmap._i2c(np.array([m[1]]))[0]

                            ddL, mM, ddM = m[4], m[5], m[6]
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
            #dL_i = np.array(self.d_ia.cget(dL)[self.f1.fragment[0],:], dtype = bool) # dL index mask
            dL_i = np.array(self.d_ia.cget([0,0,0])[self.f1.fragment[0],:], dtype = bool) # dL index mask


            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM]
                #dM_i = np.array(self.d_ia.cget(dM)[self.f2.fragment[0],:], dtype = bool) # dM index mask
                dM_i = np.array(self.d_ia.cget([0,0,0])[self.f2.fragment[0],:], dtype = bool) # dM index mask


                g_direct = self.g_d[:,ddL,:,self.mM, :, ddM, :] #[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]
                g_exchange = self.g_x[:,ddL,:,self.mM, :, ddM, :] #[self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i]



                t = self.t2[:,ddL,:,self.mM, :, ddM, :] #[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]

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


        self.t2 = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_d = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_x = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

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
    Input parameters:
        p               = prism object
        wannier_centers
        coords
        fragment        = an index array of the relevant occupied orbitals in the refcell
        ib              = PRI integral builder instance
        f_mo_pp         = Diagonal elements in the fock matrix
        virtual_cutoff  = distance cutoff of virtual space
        occupied_cutoff = distance cutoff of occupied space
        float_precision = bit precision to use in arrays

    Methods
        init_amplitudes()   - make initial guess of amplitudes and define arrays to contain them
        initialize_blocks() - helper method for init_amplitudes()  that computes the relevant blocks in an optimized way
        compute_energy()    - compute energy in entire Amplitude Orbital Space (AOS)
        compute_fragmentt_energy() - compute energy where both occupied indices are located on the fragment.
        init_cell_batch()   -
        init_cell()         -  Initialize tensors in cell ( 0 i , dL a | M j , dM b )
        autoexpand_virtual_space() - includes n more of the closest virtuals into the excitation domain
        autoexpand_occupied_space() - includfes n more of the closest occupied into the excitation domain
        set_extent()        - grows tensors and computes the required elements
        print_configuration_space_data() - prints to screen the current exitation domains
        solve()             - Converges the amplitudes using MP2


    """
    def __init__(self, p, wannier_centers, coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 3.0, occupied_cutoff = 1.0, float_precision = np.float64):
        self.p = p #prism object
        self.d = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers) # distance matrix

        self.float_precision = float_precision

        #self.d_ii = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[:p.get_nocc()])
        #self.d_ia = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[p.get_nocc():])
        self.d_ii = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers[:p.get_nocc()])
        self.d_ia = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers[p.get_nocc():])

        self.fragment = fragment

        self.ib = ib #integral builder
        self.virtual_cutoff = virtual_cutoff
        self.occupied_cutoff = occupied_cutoff
        self.min_elm = np.min(self.d.blocks[:-1], axis = (1,2)) #array for matrix-size calc

        self.min_elm_ii = np.min(self.d_ii.blocks[:-1], axis = (1,2))
        self.min_elm_ia = np.min(self.d_ia.blocks[:-1], axis = (1,2))

        self.f_mo_ii = f_mo_ii # Occupied MO-Fock matrix elements
        self.f_mo_aa = f_mo_aa # Virtual MO-Fock matrix elements

        #print("Fragment extent")
        #print(self.d_ii.coords)
        #print(self.d_ia.coords)

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


        self.t2  = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_d = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_x = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))

        self.e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]


        self.e0 = 0

        sequence = []


        for ddL in np.arange(N_virt):
            # In the following, we use symmetry
            # G(0, :, dL, :, M, :, dM, : ) = G(0, dM, :, -M, :, dL, :).T
            # Saves some seconds here and there :-)
            # Still, have to make sure -M is inside the truncated domain -> if-test inside loop
            for ddM in np.arange(ddL, N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]

                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]



                    if True: #M[0]>=0: #?

                        # Get exchange block coordinates
                        ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM + M) ]
                        #ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL - M) ]
                        ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL) ]





                        # Negative M
                        mM_ = self.d_ia.mapping[ self.d_ia._c2i(-M) ]

                        # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                        #                ^                 ^           ^          ^
                        #            Calculate these    ex/direct    store here   1=transpose

                        sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # direct
                        if mM_<=N_occ: # if the transpose is in the domain
                            sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,  1]) # direct, transposed


                        # For fragments, exchange only required for only M = (0,0,0)
                        # EOS always has the two occupied indices in the fragment, ie the refcell
                        if np.sum(M**2) == 0:
                            #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                            #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed

                            #sequence.append([ddL_M, mM, ddM_M, 1, ddL, mM , ddM,0])  # exchange
                            #sequence.append([ddL_M, mM, ddM_M, 1, ddM, mM_, ddL,1]) # exchange, transposed

                            sequence.append([ddL, mM, ddM,   1, ddL, mM, ddM,   1]) # direct
                            sequence.append([ddL, mM, ddM,   1, ddM, mM, ddL,   0]) # direct, transposed


        self.initialize_blocks(sequence)

    def initialize_blocks(self, sequence):
        #print("Initialization sequence:")
        sequence = np.array(sequence)

        #print(sequence)

        #print("shape sequence:", sequence.shape)


        n_computed_di = 0
        n_computed_ex = 0

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



                #print(sq_i)

                dL = self.d_ia.coords[sq_i[0,0]]



                #print(sq_i)
                k = 0
                for l in np.arange(len(sq_i)):


                    if sq_i[k,2] != sq_i[l,2]:
                        dM = self.d_ia.coords[sq_i[k,2]]
                        #print("-------")


                        #print(sq_i[k:l])
                        # Integrate here, loop over M
                        t0 = time.time()
                        I, Ishape = self.ib.getorientation(dL, dM)

                        for m in sq_i[k:l]:
                            M = self.d_ii.coords[m[1]]
                            ddL, mM, ddM = m[4], m[5], m[6]
                            #print(self.g_x.shape, ddL, mM, ddM)
                            #rint("Computing:", dL, M, dM)
                            #print(I.cget(M).shape, Ishape)
                            if m[7] == 0:
                                if m[3] == 0:
                                    # Direct contribution
                                    self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)
                                    self.t2[:,  ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)*self.e_iajb**-1
                                    n_computed_di += 1
                                if m[3] == 1:
                                    # Exchange contribution
                                    #ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                    self.g_x[:, ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)
                                    n_computed_ex += 1
                            if m[7] == 1:
                                if m[3] == 0:
                                    # Direct contribution
                                    #print(ddL, mM, ddM, self.g_d.shape)
                                    self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)
                                    self.t2[:,  ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)*self.e_iajb**-1
                                    n_computed_di += 1
                                if m[3] == 1:
                                    # Exchange contribution
                                    #ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                    self.g_x[:, ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)
                                    n_computed_ex += 1
                        print("Computed RI-integrals ", dL, dM, " in %.2f seconds." % (time.time()-t0))
                        #print(sq_i)


                        k = l*1

                j = i*1
        print(n_computed_di)
        print(n_computed_ex)

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
        e_mp2_direct = 0
        e_mp2_exchange = 0
        N_virt = self.n_virtual_cells

        mM = 0 #occupied index only runs over fragment

        print("Self.fragment;:", self.fragment)

        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[self.fragment[0],:]<self.virtual_cutoff # dL index mask
            # Doublecount? dL == dM



            for ddM in np.arange(N_virt):

                    dM =  self.d_ia.coords[ddM]
                    dM_i = self.d_ia.cget(dM)[self.fragment[0],:]<self.virtual_cutoff # dM index mask

                    # Using multiple levels of masking, probably some other syntax could make more sense

                    g_direct = self.g_d[:,ddL,:,0, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]
                    #g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :][self.fragment][:, dM_i][:, :, self.fragment][:,:,:,dL_i]
                    #g_direct = self.g_d[:,ddL,:,mM, :, ddM, :][self.fragment][:, :][:, :, self.fragment][:,:,:,:]
                    #g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :][self.fragment][:, :][:, :, self.fragment][:,:,:,:]

                    g_exchange = self.g_d[:,ddM,:,0, :, ddL, :][self.fragment][:, dM_i][:, :, self.fragment][:,:,:,dL_i]
                    #g_exchange = self.g_d[:,ddM,:,mM, :, ddL, :][self.fragment][:, :][:, :, self.fragment][:,:,:,:]

                    #print(g_direct.shape)
                    #print(g_exchange.shape)
                    #print(ddL, mM, ddM, np.max(np.abs(g_direct)), np.max(np.abs(g_exchange)))
                    t = self.t2[:,ddL,:,0, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]
                    #t = self.t2[:,ddL,:,mM, :, ddM, :][self.fragment][:, :][:, :, self.fragment][:,:,:,:]
                    #gd2, ex  =np.einsum("iajb,iajb",t,g_direct, optimize = True) , np.einsum("iajb,ibja",t,g_exchange, optimize = True)
                    #e_mp2_direct += 2*gd2
                    #e_mp2_exchange += ex

                    #print(np.linalg.norm(self.g_d[:,ddL,:,mM, :, ddM, :].reshape(2*9,2*9).T.reshape(2,9,2,9) - self.g_x[:,ddL,:,mM, :, ddM, :]))
                    #print("direct, exchange at:", dL, dM, " = ", ddL, ddM, " = ", gd2, ex)# #, np.linalg.norm(self.t2[:,ddL,:,mM, :, ddM, :]),np.linalg.norm(self.g_d[:,ddL,:,mM, :, ddM, :]),np.linalg.norm(self.g_x[:,ddL,:,mM, :, ddM, :]) )
                    #print(np.max(np.abs(t)), np.max(np.abs(g_direct)), np.max(np.abs(g_exchange)))
                    e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        #print("E TOt", e_mp2_direct, e_mp2_exchange)
        return e_mp2


    def compute_fragment_energy_modt2(self,t2):
        """
        Compute fragment energy
        """
        e_mp2 = 0
        e_mp2_direct = 0
        e_mp2_exchange = 0
        N_virt = self.n_virtual_cells

        mM = 0 #occupied index only runs over fragment

        print("Self.fragment;:", self.fragment)

        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[self.fragment[0],:]<self.virtual_cutoff # dL index mask
            # Doublecount? dL == dM



            for ddM in np.arange(N_virt):

                    dM =  self.d_ia.coords[ddM]
                    dM_i = self.d_ia.cget(dM)[self.fragment[0],:]<self.virtual_cutoff # dM index mask

                    # Using multiple levels of masking, probably some other syntax could make more sense

                    g_direct = self.g_d[:,ddL,:,0, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]
                    #g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :][self.fragment][:, dM_i][:, :, self.fragment][:,:,:,dL_i]
                    #g_direct = self.g_d[:,ddL,:,mM, :, ddM, :][self.fragment][:, :][:, :, self.fragment][:,:,:,:]
                    #g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :][self.fragment][:, :][:, :, self.fragment][:,:,:,:]

                    g_exchange = self.g_d[:,ddM,:,0, :, ddL, :][self.fragment][:, dM_i][:, :, self.fragment][:,:,:,dL_i]
                    #g_exchange = self.g_d[:,ddM,:,mM, :, ddL, :][self.fragment][:, :][:, :, self.fragment][:,:,:,:]

                    #print(g_direct.shape)
                    #print(g_exchange.shape)
                    #print(ddL, mM, ddM, np.max(np.abs(g_direct)), np.max(np.abs(g_exchange)))
                    t = t2[:,ddL,:,0, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]
                    #t = self.t2[:,ddL,:,mM, :, ddM, :][self.fragment][:, :][:, :, self.fragment][:,:,:,:]
                    #gd2, ex  =np.einsum("iajb,iajb",t,g_direct, optimize = True) , np.einsum("iajb,ibja",t,g_exchange, optimize = True)
                    #e_mp2_direct += 2*gd2
                    #e_mp2_exchange += ex

                    #print(np.linalg.norm(self.g_d[:,ddL,:,mM, :, ddM, :].reshape(2*9,2*9).T.reshape(2,9,2,9) - self.g_x[:,ddL,:,mM, :, ddM, :]))
                    #print("direct, exchange at:", dL, dM, " = ", ddL, ddM, " = ", gd2, ex)# #, np.linalg.norm(self.t2[:,ddL,:,mM, :, ddM, :]),np.linalg.norm(self.g_d[:,ddL,:,mM, :, ddM, :]),np.linalg.norm(self.g_x[:,ddL,:,mM, :, ddM, :]) )
                    #print(np.max(np.abs(t)), np.max(np.abs(g_direct)), np.max(np.abs(g_exchange)))
                    e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        #print("E TOt", e_mp2_direct, e_mp2_exchange)
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
                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
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
                            if np.abs(self.t2[:,ddL, :, mmM, :, ddM, :]).max()<=10e-12: # and M[0]>=0:


                                M = self.d_ii.coords[mmM]

                                # Get exchange block coordinates
                                ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM - M) ] # dL to calculate
                                ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL + M) ]





                                # Negative M
                                mmM_ = self.d_ia.mapping[ self.d_ia._c2i(-M) ]

                                # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                                #                ^                 ^           ^          ^
                                #            Calculate these    ex/direct    store here   1=transpose

                                sequence.append([ddL, mmM, ddM,   0, ddL, mmM, ddM,   0]) # direct
                                if mmM_<=No: # if the transpose is in the domain
                                    sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed



                                #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                                #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed
                                # For fragments, exchange only required for only M = (0,0,0)
                                # EOS always has the two occupied indices in the fragment, i.e. inside the refcell
                                if np.sum(M**2) == 0:
                                    #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                                    #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed
                                    sequence.append([ddL, mmM, ddM,   1, ddL, mmM, ddM,   1]) # direct
                                    sequence.append([ddL, mmM, ddM,   1, ddM, mmM_, ddL,  0]) # direct, transposed



                                #self.init_cell(ddL, mmM, ddM)
                            #print(ddL, mmM, ddM, time.time()-t0)
                #print(time.time()-t0, " s spent on cell init.")
                self.initialize_blocks(sequence)







            else:
                print("Extending virtuals.")
                # Extend tensors in the virtual direction

                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
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
                            if np.abs(self.t2[:,ddL, :, mmM, :, ddM, :]).max()<=10e-12 and M[0]>=0:

                                M = self.d_ii.coords[mmM]

                                # Get exchange block coordinates
                                ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM + M) ]
                                ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL - M) ]





                                # Negative M
                                mmM_ = self.d_ia.mapping[ self.d_ia._c2i(-M) ]

                                # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                                #                ^                 ^           ^          ^
                                #            Calculate these    ex/direct    store here   1=transpose

                                sequence.append([ddL, mmM, ddM,   0, ddL, mmM , ddM,  0]) # direct
                                if mmM_<=No: # if the transpose is in the domain
                                    sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed



                                #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                                #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed

                                # For fragments, exchange only required for only M = (0,0,0)
                                # EOS always has the two occupied indices in the fragment, ie the refcell
                                if np.abs(mmM) < 1:
                                    #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0]) # exchange
                                    #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed
                                    sequence.append([ddL, 0, ddM,   1, ddL, 0, ddM,   1]) # exchange
                                    sequence.append([ddL, 0, ddM,   1, ddM, 0, ddL,   0]) # exchange, transposed



                                #self.init_cell(ddL, mmM, ddM)
                            #print(ddL, mmM, ddM, time.time()-t0)
                #print(time.time()-t0, " s spent on cell init.")
                self.initialize_blocks(sequence)

        else:
            if No > self.n_occupied_cells:
                print("extending occupied")
                # Extend tensors in the occupied dimension
                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
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
                            if np.abs(self.t2[:,ddL, :, mmM, :, ddM, :]).max()<=10e-12 and M[0]>=0:


                                M = self.d_ii.coords[mmM]

                                # Get exchange block coordinates
                                ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM + M) ]
                                ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL - M) ]





                                # Negative M
                                mmM_ = self.d_ia.mapping[ self.d_ia._c2i(-M) ]

                                # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                                #                ^                 ^           ^          ^
                                #            Calculate these    ex/direct    store here   1=transpose

                                sequence.append([ddL, mmM, ddM,   0, ddL, mmM, ddM,   0]) # direct
                                if mmM_<=No: # if the transpose is in the domain
                                    sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed

                                # For fragments, exchange only required for only M = (0,0,0)
                                # EOS always has the two occupied indices in the fragment, ie the refcell
                                if np.sum(M**2) == 0:
                                    #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                                    #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed
                                    sequence.append([ddL, mmM, ddM,   1, ddL, mmM, ddM,   1]) # direct
                                    sequence.append([ddL, mmM, ddM,   1, ddM, mmM_, ddL,  0]) # direct, transposed




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

    def solve(self, norm_thresh = 1e-10, eqtype = "mp2", s_virt = None):
        if eqtype == "mp2_nonorth":
            return self.solve_MP2PAO(norm_thresh, s_virt = s_virt)
        elif eqtype == "steepdesc":
            return self.solve_MP2PAO_steepdesc(norm_thresh, s_virt = s_virt)
        elif eqtype == "ls":
            return self.solve_MP2PAO_ls(norm_thresh, s_virt = s_virt)
        else:
            return self.solve_MP2(norm_thresh)

    def omega(self, t2):

        t2_new = np.zeros_like(t2)
        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask
            for dM in np.arange(self.n_virtual_cells):
                dMv = self.d_ia.coords[dM]
                dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                for M in np.arange(self.n_occupied_cells):
                    Mv = self.d_ii.coords[M]
                    M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                    tnew = -self.g_d[:, dL, :, M, :, dM, :]

                    # generate index mapping of non-zero amplitudes in cell
                    cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                    # Perform contractions

                    # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                    Fac = self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dL])
                    tnew -= np.einsum("iKcjb,Kac->iajb", t2[:, :, :, M, :, dM, :], Fac)

                    # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                    Fbc = self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dM])
                    tnew -= np.einsum("iajKc,Kbc->iajb", t2[:, dL, :, M, :, :, :], Fbc)

                    # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                    # Fki = self.f_mo_ii.cget(-1*pair_extent)
                    # tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)



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
                    Fki = self.f_mo_ii.cget(-1*self.pair_extent)

                    M_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[M] - self.pair_extent ) ]
                    dL_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dL] - self.pair_extent) ]
                    dM_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dM] - self.pair_extent) ]
                    #make sure indices is in correct domain (not negative or beyond extent)
                    nz = (M_range<self.n_occupied_cells)*(dL_M<self.n_virtual_cells)*(dM_M<self.n_virtual_cells)*\
                            (M_range>=0)*(dL_M>=0)*(dM_M>=0)

                    tnew += np.einsum("Kkajb,Kki->iajb",t2[:, dL_M[nz], :, M_range[nz], :, dM_M[nz], :], Fki[nz])



                    # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                    Fkj = self.f_mo_ii.cget(-1*self.pair_extent + self.pair_extent[M])
                    tnew += np.einsum("iaKkb,Kkj->iajb",t2[:, dL, :, :, :, dM, :], Fkj)

                    t2_mapped = np.zeros_like(tnew).ravel()
                    t2_mapped[cell_map] = (tnew*self.e_iajb**-1).ravel()[cell_map]
                    #t2_mapped = (tnew*self.e_iajb**-1).ravel()
                    t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
        return t2_new

    def omega_agrad(self, t2):

        t2_new = np.zeros_like(t2)
        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

            for dM in np.arange(self.n_virtual_cells):
                dMv = self.d_ia.coords[dM]
                dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                for M in np.arange(self.n_occupied_cells):
                    Mv = self.d_ii.coords[M]
                    M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                    tnew = -self.g_d[:, dL, :, M, :, dM, :]

                    # generate index mapping of non-zero amplitudes in cell
                    #cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                    # Perform contractions

                    # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                    Fac = np.array(self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dL]))
                    tnew -= np.einsum("iKcjb,Kac->iajb", t2[:, :, :, M, :, dM, :], Fac)

                    # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                    Fbc = np.array(self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dM]))
                    tnew -= np.einsum("iajKc,Kbc->iajb", t2[:, dL, :, M, :, :, :], Fbc)

                    # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                    # Fki = self.f_mo_ii.cget(-1*pair_extent)
                    # tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)



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
                    Fki = np.array(self.f_mo_ii.cget(-1*self.pair_extent))

                    M_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[M] - self.pair_extent ) ]
                    dL_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dL] - self.pair_extent) ]
                    dM_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dM] - self.pair_extent) ]
                    #make sure indices is in correct domain (not negative or beyond extent)
                    nz = (M_range<self.n_occupied_cells)*(dL_M<self.n_virtual_cells)*(dM_M<self.n_virtual_cells)*\
                            (M_range>=0)*(dL_M>=0)*(dM_M>=0)

                    tnew += np.einsum("Kkajb,Kki->iajb",t2[:, dL_M[nz], :, M_range[nz], :, dM_M[nz], :], Fki[nz])



                    # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                    Fkj = np.array(self.f_mo_ii.cget(-1*self.pair_extent + self.pair_extent[M]))
                    tnew += np.einsum("iaKkb,Kkj->iajb",t2[:, dL, :, :, :, dM, :], Fkj)

                    #t2_mapped = np.zeros_like(tnew).ravel()
                    #t2_mapped[cell_map] = (tnew*self.e_iajb**-1).ravel()[cell_map]
                    #t2_mapped = (tnew*self.e_iajb**-1).ravel()

                    #print(dL, M, dM)
                    #print(t2_new[:, dL, :, M, :, dM, :].shape,(tnew*self.e_iajb**-1).shape )
                    t2_new[:, dL, :, M, :, dM, :] = (tnew*self.e_iajb**-1) #t2_mapped.reshape(tnew.shape)
        return t2_new



    def solve_MP2(self, norm_thresh = 1e-10):
        """
        Converge fragment (AOS) amplitudes within occupied and virtual extents
        """
        #from autograd import grad, elementwise_grad,jacobian

        nocc = self.p.get_nocc()

        self.virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        self.pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.vp_indx = mapgen(self.virtual_extent, self.pair_extent)
        self.pp_indx = mapgen(self.pair_extent, self.pair_extent)

        #print(vp_indx)
        #print(pp_indx)

        #dt2 = ad.gh(self.omega_agrad)
        #dt_new = dt2[1](self.t2)
        for ti in np.arange(100):



            dt2_new = self.omega(self.t2)
            #t2_prev =

            # do a line search
            if False:

                n = 1e10 #np.linalg.norm(dt2)
                for j in np.arange(20):
                    dt2 = self.omega(self.t2 - j*0.1*dt2_new)
                    n_ = np.linalg.norm(dt2)
                    if n_>=n:
                        break
                    n = n_
                    print("line search", j, ":", n_)

                self.t2 -= j*0.1*dt2_new
                rnorm = n #np.linalg.norm(n)
            else:
                self.t2 -= dt2_new
                rnorm = np.linalg.norm(dt2_new)


            if rnorm<norm_thresh:

                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti,rnorm))
                break

    def solve_MP2PAO_steepdesc(self, norm_thresh = 1e-10, s_virt = None):
        """
        Solving the MP2 equations for a non-orthogonal virtual space
        (see section 5.6 "The periodic MP2 equations for non-orthogonal virtual space (PAO)" in the notes)
        """
        print ('########### PAO_SOLVER_STEEPDESC ##############')
        print ('########### PAO_SOLVER_STEEPDESC ##############')
        print ('########### PAO_SOLVER_STEEPDESC ##############')
        print ('########### PAO_SOLVER_STEEPDESC ##############')
        print ('########### PAO_SOLVER_STEEPDESC ##############')
        print ('########### PAO_SOLVER_STEEPDESC ##############')
        print ('########### PAO_SOLVER_STEEPDESC ##############')
        print ('AMPLITUDE NORM: ',np.linalg.norm(self.t2))
        alpha = 0.1

        nocc = self.p.get_nocc()
        ener_old = self.compute_fragment_energy()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        self.s_pao1 = tp.get_identity_tmat(self.p.get_nvirt())

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
        s_aa = np.diag(self.s_pao.cget([0,0,0]))
        f_ij = f_ii[:,None] + f_ii[None,:]
        sfs = np.einsum("a,ij,b->iajb",s_aa,f_ij,s_aa)
        fs_ab = np.einsum("a,b,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        fs_ba = np.einsum("b,a,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        f_iajb = sfs - fs_ab - fs_ba
        counter = 0

        for ti in np.arange(1000):
            counter += 1
            print ('Iteration no.: ', ti)
            t2_new = np.zeros_like(self.t2)
            R_new = np.zeros_like(self.t2)

            beta1 = np.zeros_like(self.t2)
            beta2 = np.zeros_like(self.t2)

            for C in np.arange(self.n_virtual_cells):
                for D in np.arange(self.n_virtual_cells):
                    for J in np.arange(self.n_occupied_cells):
                        Fkj = self.f_mo_ii.cget(-pair_extent + pair_extent[J])
                        beta1[:,C,:,J,:,D,:] = np.einsum("icKkd,Kkj->icjd",self.t2[:,C,:,:,:,D,:],Fkj)

                        Fik = self.f_mo_ii.cget(pair_extent)

                        J_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[J] - pair_extent ) ]
                        C_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[C] - pair_extent) ]
                        D_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[D] - pair_extent) ]

                        nz = (J_range<self.n_occupied_cells)*(C_J<self.n_virtual_cells)*(D_J<self.n_virtual_cells)*\
                             (J_range>=0)*(C_J>=0)*(D_J>=0)

                        beta2[:,C,:,J,:,D,:] = np.einsum("Kkcjd,Kik->icjd",self.t2[:,C_J[nz],:,J_range[nz],:,D_J[nz],:],Fik[nz])


            for dL in np.arange(self.n_virtual_cells):
                dLv = self.d_ia.coords[dL]
                dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                    for M in np.arange(self.n_occupied_cells):
                        Mv = self.d_ii.coords[M]
                        M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                        tnew = -self.g_d[:, dL, :, M, :, dM, :]

                        # generate index mapping of non-zero amplitudes in cell
                        cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                        # Perform contractions
                        Fac = self.f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                        Fdb = self.f_mo_aa.cget(-virtual_extent + virtual_extent[dM])
                        Sac = self.s_pao.cget(virtual_extent - virtual_extent[dL])
                        Sdb = self.s_pao.cget(-virtual_extent + virtual_extent[dM])

                        #print ('dL: ', dL)
                        #print ('dM: ', dM)
                        #print ('M: ', M)


                        # \sum_{CcDd} f_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Fac, self.t2[:, :, :, M, :, :, :])
                        tnew -= np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDd} s_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n f_{db}^{B-D}
                        t_int = np.einsum("Ddb,iCcjDd->iCcjb", Fdb, self.t2[:, :, :, M, :, :, :])
                        tnew -= np.einsum("iCcjb,Cac->iajb", t_int, Sac)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{jk}^{J-K} \left(t_{0i,Kk}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta1[:, :, :, M, :, :, :])
                        tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{ik}^{K} \left(t_{0k,J-Kk}^{C-Kc,D-Kd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta2[:, :, :, M, :, :, :])
                        tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                        #print (np.linalg.norm(tnew))


                        t2_mapped = np.zeros_like(tnew).ravel()
                        R_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*f_iajb**-1).ravel()[cell_map]
                        R_mapped[cell_map] = (tnew).ravel()[cell_map]

                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
                        R_new[:, dL, :, M, :, dM, :] = R_mapped.reshape(tnew.shape)

            RNORM_new = np.linalg.norm(R_new)
            self.t2_old = np.copy(self.t2)
            self.t2 -= 0.1*t2_new
            if counter >= 15:
                d_omega = RNORM_new-RNORM_old
                dt = self.t2-self.t2_old
                self.t2 = -dt*np.linalg.norm(self.g_d)/d_omega
                counter = 0
            RNORM_old = RNORM_new
            t2_old = np.copy(t2_new)
            R_old = np.copy(R_new)
            rnorm = np.linalg.norm(t2_new)
            ener = self.compute_fragment_energy()
            print ('R norm: ',rnorm)
            print ('Energy: ',ener)
            print ('dE: ',ener-ener_old)
            ener_old = ener
            if rnorm<norm_thresh:
                print ()
                print ('##############')
                print ('##############')
                print ('##############')
                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti, np.linalg.norm(t2_new)))
                print ()
                break


    def comp_R(self, t2, s_virt=None):
        nocc = self.p.get_nocc()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        self.s_pao1 = tp.get_identity_tmat(self.p.get_nvirt())

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
        s_aa = np.diag(self.s_pao.cget([0,0,0]))
        f_ij = f_ii[:,None] + f_ii[None,:]
        sfs = np.einsum("a,ij,b->iajb",s_aa,f_ij,s_aa)
        fs_ab = np.einsum("a,b,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        fs_ba = np.einsum("b,a,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        f_iajb = sfs - fs_ab - fs_ba
        t2_old = np.zeros_like(t2)

        R_new = np.zeros_like(t2)

        beta1 = np.zeros_like(t2)
        beta2 = np.zeros_like(t2)

        for C in np.arange(self.n_virtual_cells):
            for D in np.arange(self.n_virtual_cells):
                for J in np.arange(self.n_occupied_cells):
                    Fkj = self.f_mo_ii.cget(-pair_extent + pair_extent[J])
                    beta1[:,C,:,J,:,D,:] = np.einsum("icKkd,Kkj->icjd",t2[:,C,:,:,:,D,:],Fkj)

                    Fik = self.f_mo_ii.cget(pair_extent)

                    J_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[J] - pair_extent ) ]
                    C_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[C] - pair_extent) ]
                    D_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[D] - pair_extent) ]

                    nz = (J_range<self.n_occupied_cells)*(C_J<self.n_virtual_cells)*(D_J<self.n_virtual_cells)*\
                         (J_range>=0)*(C_J>=0)*(D_J>=0)

                    beta2[:,C,:,J,:,D,:] = np.einsum("Kkcjd,Kik->icjd",t2[:,C_J[nz],:,J_range[nz],:,D_J[nz],:],Fik[nz])


        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

            for dM in np.arange(self.n_virtual_cells):
                dMv = self.d_ia.coords[dM]
                dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                for M in np.arange(self.n_occupied_cells):
                    Mv = self.d_ii.coords[M]
                    M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                    tnew = -self.g_d[:, dL, :, M, :, dM, :]

                    # generate index mapping of non-zero amplitudes in cell
                    cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                    # Perform contractions
                    Fac = self.f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                    Fdb = self.f_mo_aa.cget(-virtual_extent + virtual_extent[dM])
                    Sac = self.s_pao.cget(virtual_extent - virtual_extent[dL])
                    Sdb = self.s_pao.cget(-virtual_extent + virtual_extent[dM])

                    #print ('dL: ', dL)
                    #print ('dM: ', dM)
                    #print ('M: ', M)


                    # \sum_{CcDd} f_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n s_{db}^{B-D}
                    t_int = np.einsum("Cac,iCcjDd->aijDd", Fac, t2[:, :, :, M, :, :, :])
                    tnew -= np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                    #print (np.linalg.norm(tnew))

                    # \sum_{CcDd} s_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n f_{db}^{B-D}
                    t_int = np.einsum("Ddb,iCcjDd->iCcjb", Fdb, t2[:, :, :, M, :, :, :])
                    tnew -= np.einsum("iCcjb,Cac->iajb", t_int, Sac)
                    #print (np.linalg.norm(tnew))

                    # \sum_{CcDdKk} s_{ac}^{C-A} f_{jk}^{J-K} \left(t_{0i,Kk}^{Cc,Dd}\right)_n s_{db}^{B-D}
                    t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta1[:, :, :, M, :, :, :])
                    tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                    #print (np.linalg.norm(tnew))

                    # \sum_{CcDdKk} s_{ac}^{C-A} f_{ik}^{K} \left(t_{0k,J-Kk}^{C-Kc,D-Kd}\right)_n s_{db}^{B-D}
                    t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta2[:, :, :, M, :, :, :])
                    tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                    #print (np.linalg.norm(tnew))


                    R_mapped = np.zeros_like(tnew).ravel()
                    R_mapped[cell_map] = (tnew*f_iajb**-1).ravel()[cell_map]

                    R_new[:, dL, :, M, :, dM, :] = R_mapped.reshape(tnew.shape)

        return R_new


    def solve_MP2PAO_ls(self, norm_thresh = 1e-10, s_virt = None):
        """
        Solving the MP2 equations for a non-orthogonal virtual space
        (see section 5.6 "The periodic MP2 equations for non-orthogonal virtual space (PAO)" in the notes)
        """
        print ('########### PAO_SOLVER_LS ##############')
        print ('########### PAO_SOLVER_LS ##############')
        print ('########### PAO_SOLVER_LS ##############')
        print ('########### PAO_SOLVER_LS ##############')
        print ('########### PAO_SOLVER_LS ##############')
        print ('########### PAO_SOLVER_LS ##############')
        print ('########### PAO_SOLVER_LS ##############')
        print ('AMPLITUDE NORM: ',np.linalg.norm(self.t2))

        alpha = 0.3
        ener_old = 0

        nocc = self.p.get_nocc()
        ener_old = self.compute_fragment_energy()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        self.s_pao1 = tp.get_identity_tmat(self.p.get_nvirt())

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
        s_aa = np.diag(self.s_pao.cget([0,0,0]))
        f_ij = f_ii[:,None] + f_ii[None,:]
        sfs = np.einsum("a,ij,b->iajb",s_aa,f_ij,s_aa)
        fs_ab = np.einsum("a,b,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        fs_ba = np.einsum("b,a,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        f_iajb = sfs - fs_ab - fs_ba
        t2_old = np.zeros_like(self.t2)

        for ti in np.arange(1000):
            print ('Iteration no.: ', ti)
            t2_new = np.zeros_like(self.t2)
            R_new = np.zeros_like(self.t2)

            beta1 = np.zeros_like(self.t2)
            beta2 = np.zeros_like(self.t2)

            for C in np.arange(self.n_virtual_cells):
                for D in np.arange(self.n_virtual_cells):
                    for J in np.arange(self.n_occupied_cells):
                        Fkj = self.f_mo_ii.cget(-pair_extent + pair_extent[J])
                        beta1[:,C,:,J,:,D,:] = np.einsum("icKkd,Kkj->icjd",self.t2[:,C,:,:,:,D,:],Fkj)

                        Fik = self.f_mo_ii.cget(pair_extent)

                        J_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[J] - pair_extent ) ]
                        C_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[C] - pair_extent) ]
                        D_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[D] - pair_extent) ]

                        nz = (J_range<self.n_occupied_cells)*(C_J<self.n_virtual_cells)*(D_J<self.n_virtual_cells)*\
                             (J_range>=0)*(C_J>=0)*(D_J>=0)

                        beta2[:,C,:,J,:,D,:] = np.einsum("Kkcjd,Kik->icjd",self.t2[:,C_J[nz],:,J_range[nz],:,D_J[nz],:],Fik[nz])


            for dL in np.arange(self.n_virtual_cells):
                dLv = self.d_ia.coords[dL]
                dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                    for M in np.arange(self.n_occupied_cells):
                        Mv = self.d_ii.coords[M]
                        M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                        tnew = -self.g_d[:, dL, :, M, :, dM, :]

                        # generate index mapping of non-zero amplitudes in cell
                        cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                        # Perform contractions
                        Fac = self.f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                        Fdb = self.f_mo_aa.cget(-virtual_extent + virtual_extent[dM])
                        Sac = self.s_pao.cget(virtual_extent - virtual_extent[dL])
                        Sdb = self.s_pao.cget(-virtual_extent + virtual_extent[dM])

                        #print ('dL: ', dL)
                        #print ('dM: ', dM)
                        #print ('M: ', M)


                        # \sum_{CcDd} f_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Fac, self.t2[:, :, :, M, :, :, :])
                        tnew -= np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDd} s_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n f_{db}^{B-D}
                        t_int = np.einsum("Ddb,iCcjDd->iCcjb", Fdb, self.t2[:, :, :, M, :, :, :])
                        tnew -= np.einsum("iCcjb,Cac->iajb", t_int, Sac)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{jk}^{J-K} \left(t_{0i,Kk}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta1[:, :, :, M, :, :, :])
                        tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{ik}^{K} \left(t_{0k,J-Kk}^{C-Kc,D-Kd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta2[:, :, :, M, :, :, :])
                        tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                        #print (np.linalg.norm(tnew))


                        t2_mapped = np.zeros_like(tnew).ravel()
                        R_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*f_iajb**-1).ravel()[cell_map]
                        R_mapped[cell_map] = (tnew).ravel()[cell_map]

                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
                        R_new[:, dL, :, M, :, dM, :] = R_mapped.reshape(tnew.shape)



            norm_R_old = np.linalg.norm(self.comp_R(self.t2,self.s_pao))



            print ('ALPHA: ',alpha)
            #self.t2 =  alpha*(self.t2 - (t2_new)) + (1-alpha)*self.t2
            norm_old = np.linalg.norm(self.comp_R(self.t2,self.s_pao))
            print ()
            print ('Starting microiterations')
            print ('StartNORM: ', norm_old)
            """
            for i in np.linspace(0,1,10):
                norm_new = (np.linalg.norm(self.comp_R(self.t2-i*t2_new,self.s_pao)))
                print ('microit',i)
                print ('RNORM: ',norm_new)
                if norm_new > norm_old:
                    self.t2 = self.t2-(i_old)*t2_new
                    break
                norm_old = norm_new
                i_old = i
            """
            E_prev = self.compute_fragment_energy_modt2(self.t2)
            dE_prev = 0
            dt = t2_new
            for i in np.linspace(0,1,10):
                E_new = self.compute_fragment_energy_modt2(self.t2-i*dt)
                print ('microit',i)
                #print ('RNORM: ',norm_new)
                if abs(E_new-E_prev) > dE_prev:
                    self.t2 = self.t2-(i)*dt
                    print ('### FOUND alpha: ',i)
                    break
                E_prev = E_new
                dE_prev = np.abs(E_new - E_prev)
            print ()

            rnorm = np.linalg.norm(t2_new)
            ener = self.compute_fragment_energy()
            print ('R norm: ',rnorm)
            print ('Energy: ',ener)
            print ('dE: ',ener-ener_old)
            ener_old = ener
            if rnorm<norm_thresh:
                print ()
                print ('##############')
                print ('##############')
                print ('##############')
                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti, np.linalg.norm(t2_new)))
                print ()
                break


    def solve_MP2PAO(self, norm_thresh = 1e-10, s_virt = None):
        """
        Solving the MP2 equations for a non-orthogonal virtual space
        (see section 5.6 "The periodic MP2 equations for non-orthogonal virtual space (PAO)" in the notes)
        """
        print ('########### PAO_SOLVER_I ##############')
        print ('########### PAO_SOLVER_I ##############')
        print ('########### PAO_SOLVER_I ##############')
        print ('########### PAO_SOLVER_I ##############')
        print ('########### PAO_SOLVER_I ##############')
        print ('########### PAO_SOLVER_I ##############')
        print ('########### PAO_SOLVER_I ##############')
        print ('AMPLITUDE NORM: ',np.linalg.norm(self.t2))

        alpha = 0.1
        ener_old = 0

        nocc = self.p.get_nocc()
        ener_old = self.compute_fragment_energy()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        self.s_pao1 = tp.get_identity_tmat(self.p.get_nvirt())

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
        s_aa = np.diag(self.s_pao.cget([0,0,0]))
        f_ij = f_ii[:,None] + f_ii[None,:]
        sfs = np.einsum("a,ij,b->iajb",s_aa,f_ij,s_aa)
        fs_ab = np.einsum("a,b,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        fs_ba = np.einsum("b,a,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        f_iajb = sfs - fs_ab - fs_ba
        t2_old = np.zeros_like(self.t2)

        for ti in np.arange(1000):
            print ('Iteration no.: ', ti)
            t2_new = np.zeros_like(self.t2)

            beta1 = np.zeros_like(self.t2)
            beta2 = np.zeros_like(self.t2)

            for C in np.arange(self.n_virtual_cells):
                for D in np.arange(self.n_virtual_cells):
                    for J in np.arange(self.n_occupied_cells):
                        Fkj = self.f_mo_ii.cget(-pair_extent + pair_extent[J])
                        beta1[:,C,:,J,:,D,:] = np.einsum("icKkd,Kkj->icjd",self.t2[:,C,:,:,:,D,:],Fkj)

                        Fik = self.f_mo_ii.cget(pair_extent)

                        J_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[J] - pair_extent ) ]
                        C_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[C] - pair_extent) ]
                        D_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[D] - pair_extent) ]

                        nz = (J_range<self.n_occupied_cells)*(C_J<self.n_virtual_cells)*(D_J<self.n_virtual_cells)*\
                             (J_range>=0)*(C_J>=0)*(D_J>=0)

                        beta2[:,C,:,J,:,D,:] = np.einsum("Kkcjd,Kik->icjd",self.t2[:,C_J[nz],:,J_range[nz],:,D_J[nz],:],Fik[nz])


            for dL in np.arange(self.n_virtual_cells):
                dLv = self.d_ia.coords[dL]
                dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                    for M in np.arange(self.n_occupied_cells):
                        Mv = self.d_ii.coords[M]
                        M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                        tnew = -self.g_d[:, dL, :, M, :, dM, :]

                        # generate index mapping of non-zero amplitudes in cell
                        cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                        # Perform contractions
                        Fac = self.f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                        Fdb = self.f_mo_aa.cget(-virtual_extent + virtual_extent[dM])
                        Sac = self.s_pao.cget(virtual_extent - virtual_extent[dL])
                        Sdb = self.s_pao.cget(-virtual_extent + virtual_extent[dM])

                        #print ('dL: ', dL)
                        #print ('dM: ', dM)
                        #print ('M: ', M)


                        # \sum_{CcDd} f_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Fac, self.t2[:, :, :, M, :, :, :])
                        tnew -= np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDd} s_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n f_{db}^{B-D}
                        t_int = np.einsum("Ddb,iCcjDd->iCcjb", Fdb, self.t2[:, :, :, M, :, :, :])
                        tnew -= np.einsum("iCcjb,Cac->iajb", t_int, Sac)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{jk}^{J-K} \left(t_{0i,Kk}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta1[:, :, :, M, :, :, :])
                        tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{ik}^{K} \left(t_{0k,J-Kk}^{C-Kc,D-Kd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta2[:, :, :, M, :, :, :])
                        tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb)
                        #print (np.linalg.norm(tnew))


                        t2_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*f_iajb**-1).ravel()[cell_map]

                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)



            norm_R_old = np.linalg.norm(self.comp_R(self.t2,self.s_pao))
            alpha = 0.01
            for i in np.linspace(0.01,1.0,20):
                t2 =  i*(self.t2 - (t2_new)) + (1-i)*self.t2 ##+ 0.2*t2_old)
                norm_R = np.linalg.norm(self.comp_R(t2,self.s_pao))
                if norm_R < norm_R_old:
                    alpha = i
                norm_R_old = norm_R

            print ('ALPHA: ',alpha)
            self.t2 =  alpha*(self.t2 - (t2_new)) + (1-alpha)*self.t2
            rnorm = np.linalg.norm(t2_new)
            ener = self.compute_fragment_energy()
            print ('R norm: ',rnorm)
            print ('Energy: ',ener)
            print ('dE: ',ener-ener_old)
            ener_old = ener
            if rnorm<norm_thresh:
                print ()
                print ('##############')
                print ('##############')
                print ('##############')
                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti, np.linalg.norm(t2_new)))
                print ()
                break


class diis():
    def __init__(self, N):
        self.N = N
        self.i = 0
        self.t = np.zeros(N, dtype = object)
        self.err = np.zeros(N, dtype = object)

    def advance(self, t_i, err_i):
        self.t[self.i % self.N] = t_i
        self.err[self.i % self.N] = err_i

        if self.i<self.N:
            self.i += 1
            return t_i + err_i #remember add damping



    def build_b(self):
        pass










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
    parser.add_argument("-fot", type = float, default = 0.001, help = "fragment optimization treshold")
    parser.add_argument("-circulant", type = bool, default = True, help = "Use circulant dot-product.")
    parser.add_argument("-robust", default = False, action = "store_true", help = "Enable Dunlap robust fit for improved integral accuracy.")
    parser.add_argument("-disable_static_mem", default = False, action = "store_true", help = "Recompute AO integrals for new fitting sets.")
    parser.add_argument("-n_core", type = int, default = 0, help = "Number of core orbitals (the first n_core orbitals will not be correlated).")
    parser.add_argument("-skip_fragment_optimization", default = False, action = "store_true", help = "Skip fragment optimization (for debugging, will run faster but no error estimate.)")
    parser.add_argument("-basis_truncation", type = float, default = 0.1, help = "Truncate AO-basis function below this threshold." )
    parser.add_argument("-ao_screening", type = float, default = 1e-12, help = "Screening of the (J|mn) (three index) integrals.")
    parser.add_argument("-xi0", type = float, default = 1e-10, help = "Screening of the (J|mn) (three index) integrals.")
    parser.add_argument("-xi1", type = float, default = 1e-10, help = "Screening of the (J|pn) (three index) integral transform.")
    parser.add_argument("-float_precision", type = str, default = "np.float64", help = "Floating point precision.")
    parser.add_argument("-attenuated_truncation", type = float, default = 1e-14, help = "Truncate blocks in the attenuated matrix where (max) elements are below this threshold." )
    parser.add_argument("-virtual_space", type = str, default = None, help = "Alternative representation of virtual space, provided as tmat file." )
    parser.add_argument("-solver", type = str, default = "mp2", help = "Solver model." )

    args = parser.parse_args()

    args.float_precision = eval(args.float_precision)

    # Print run-info to screen
    print("Author : Audun Skau Hansen, audunsh4@gmail.com, 2019")

    import sys
    #print("Git rev:", sp.check_output(['git', 'rev-parse', 'HEAD'], cwd=sys.path[0]))
    print("_________________________________________________________")
    print("Input configuration")
    print("_________________________________________________________")
    print("Input files:")
    print("Geometry + AO basis    :", args.project_file)
    print("Wannier basis          :", args.coefficients)
    print("Auxiliary basis        :", args.auxbasis)
    print(" ")
    print("Screening and approximations:")
    print("FOT                    :", args.fot)
    print("Number of core orbs.   :", args.n_core, "(frozen)")
    print("Aux. basis cutoff      :", args.basis_truncation)
    print("Attenuation            :", args.attenuation)
    print("Float precision        :", args.float_precision)

    #print("Att. truncation        :", args.attenuated_truncation)
    #print("AO basis screening     :", args.ao_screening)
    print("(LJ|0mNn)screening(xi0):", args.xi0)
    print("(LJ|0pNq)screening(xi1):", args.xi1)
    print(" ")
    print("General settings:")
    print("Dot-product            :", ["Block-Toeplitz", "Circulant"][int(args.circulant)])
    print("RI fitting             :", ["Non-robust", "Robust"][int(args.robust)])

    print("_________________________________________________________")


    # Load system
    p = pr.prism(args.project_file)
    p.n_core = args.n_core

    # Wannier centers
    wcenters = np.load(args.wcenters)[p.n_core:]


    # Fitting basis
    auxbasis = PRI.basis_trimmer(p, args.auxbasis, alphacut = args.basis_truncation)
    f = open("ri-fitbasis.g94", "w")
    f.write(auxbasis)
    f.close()


    #attenuation_tuner(p, args)


    # Wannier coefficients
    c = tp.tmat()
    c.load(args.coefficients)
    c.set_precision(args.float_precision)



    #c.blocks = np.array(c.blocks, dtype = args.float_precision)


    c_occ, c_virt = PRI.occ_virt_split(c,p)

    if args.virtual_space is not None:
        if args.virtual_space == "pao":
            s, c_virt, wcenters_virt = of.conventional_paos(c,p)
            p.n_core = args.n_core
            p.set_nvirt(c_virt.blocks.shape[2])
            s_virt = c_virt.tT().circulantdot( s.circulantdot( c_virt ))
            args.solver = "mp2_nonorth"

            # Append virtual centers to the list of centers
            wcenters = np.append(wcenters[:p.get_nocc()-p.n_core], wcenters_virt, axis = 0)
            print(wcenters)
        elif args.virtual_space == "ls":
            s, c_virt, wcenters_virt = of.conventional_paos(c,p)
            p.n_core = args.n_core
            p.set_nvirt(c_virt.blocks.shape[2])
            s_virt = c_virt.tT().circulantdot( s.circulantdot( c_virt ))
            args.solver = "ls"

            # Append virtual centers to the list of centers
            wcenters = np.append(wcenters[:p.get_nocc()-p.n_core], wcenters_virt, axis = 0)
            print(wcenters)

        else:
            c_virt = tp.tmat()
            c_virt.load(args.virtual_space)
            p.set_nvirt(c_virt.blocks.shape[2])


    #if args.solver != "mp2":
    s = PRI.compute_overlap_matrix(p, tp.lattice_coords([10,10,10]))
    s_virt = c_virt.tT().circulantdot(s.circulantdot(c_virt))





    # AO Fock matrix
    f_ao = tp.tmat()
    f_ao.load(args.fock_matrix)
    f_ao.set_precision(args.float_precision)



    # Compute MO Fock matrix
    #f_mo = c.tT().cdot(f_ao*c, coords = c.coords)

    f_mo_aa = c_virt.tT().cdot(f_ao*c_virt, coords = c_virt.coords)
    f_mo_ii = c_occ.tT().cdot(f_ao*c_occ, coords = c_occ.coords)



    # Compute energy denominator
    #f_aa = f_mo.cget([0,0,0])[np.arange(p.get_nocc(),p.get_n_ao()), np.arange(p.get_nocc(),p.get_n_ao())]
    #f_ii = f_mo.cget([0,0,0])[np.arange(p.get_nocc()),np.arange(p.get_nocc()) ]

    f_aa = np.diag(f_mo_aa.cget([0,0,0]))
    f_ii = np.diag(f_mo_ii.cget([0,0,0]))

    e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]








    # Initialize integrals
    if args.disable_static_mem:
        ib = PRI.integral_builder(c,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust)
    else:
        ib = PRI.integral_builder_static(c_occ,c_virt,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust, ao_screening = args.ao_screening, xi0=args.xi0, JKa_extent= [6,6,6], xi1 = args.xi1, float_precision = args.float_precision)

    print("Number of (J|mn) tensors:", len(ib.cfit.Jmn))





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
    center_fragments = np.array(center_fragments)[::-1]
    #center_fragments = [[1], [0]]

    print(" ")
    print("Fragmentation of orbital space")
    print(center_fragments)
    print(" ")




    # Converge atomic fragment energies

    # Initial fragment extents
    virt_cut = 3.0
    occ_cut = 6.0

    for fragment in center_fragments:


        #ib.fragment = fragment
        t0 = time.time()
        a_frag = fragment_amplitudes(p, wcenters, c.coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 3.0, occupied_cutoff = 2.0, float_precision = args.float_precision)
        print("Frag init:", time.time()-t0)

        print ('NORM g_d',np.linalg.norm(a_frag.g_d))
        print ('NORM 2*g_d',np.linalg.norm(2*a_frag.g_d))
        print (a_frag.g_d.shape)
        #np.save('g_d_mp2',a_frag.g_d)
        print(" ")
        s_virt_init = tp.get_identity_tmat(a_frag.p.get_nvirt())

        #a_frag.solve(eqtype = args.solver, s_virt = s_virt_init)

        a_frag.t2 *=0.1

        print(a_frag.compute_fragment_energy())
        ###a_frag.solve(eqtype = args.solver, s_virt = s_virt_init)
        #a_frag.solve(eqtype = args.solver, s_virt = s_virt_init)
        a_frag.solve(eqtype = args.solver, s_virt = s_virt)

        # Converge to fot
        E_prev_outer = a_frag.compute_fragment_energy()
        E_prev = E_prev_outer*1.0
        dE_outer = 10

        print("Initial fragment energy: %.8f" % E_prev)
        #print("Target value  (3D Neon): -0.1131957501902151 (single cell)")
        #print("Target value  (3D Neon): -0.113287565289")

        print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
        print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))

        virtual_cutoff_prev = a_frag.virtual_cutoff
        occupied_cutoff_prev = a_frag.occupied_cutoff

        if not args.skip_fragment_optimization:
            print("Running fragment optimization for:")
            print(fragment)
            #print("Initial cutoffs:")

            n_virtuals_ = []
            virtu_cut_  = []



            while dE_outer>args.fot:
                dE = 10
                e_virt = []
                #
                while dE>args.fot:
                    #for i in np.arange(30):
                    print("e_prev:", E_prev)

                    #print("--- virtual")
                    virtual_cutoff_prev = a_frag.virtual_cutoff
                    occupied_cutoff_prev = a_frag.occupied_cutoff

                    t_0 = time.time()
                    a_frag.autoexpand_virtual_space(n_orbs=4)
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))



                    t_1 = time.time()
                    a_frag.solve(eqtype = args.solver, s_virt = s_virt)
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

                    n_virtuals_.append(a_frag.n_virtual_tot)
                    virtu_cut_.append(a_frag.virtual_cutoff)

                    #print("---")
                    print("Energy")

                    print(e_virt)
                    print("Number of virtuals")
                    print(n_virtuals_)
                    print("Virtual cutoff distance")
                    print(virtu_cut_)
                a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)



                print("Converged virtual space, expanding occupied space")
                print(e_virt)
                #dE = 10
                #print("--- occupied")
                a_frag.autoexpand_occupied_space(n_orbs=6)
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))

                a_frag.solve(eqtype = args.solver, s_virt = s_virt)
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
                    virtual_cutoff_prev = a_frag.virtual_cutoff
                    occupied_cutoff_prev = a_frag.occupied_cutoff

                    #print("--- occupied")
                    a_frag.autoexpand_occupied_space(n_orbs=6)
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))

                    a_frag.solve(eqtype = args.solver)
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
                a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)
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

        # Outline of pair fragment calcs

        pair_coords = tp.lattice_coords([10,10,10])
        pair_coords = pair_coords[np.argsort(np.sum(pair_coords**2, axis = 1))[1:]] #Sort in increasing order
        pair_total = 0
        for c in pair_coords:

            pair = pair_fragment_amplitudes(a_frag, a_frag, M = c)
            #print(pair.compute_pair_fragment_energy())
            pair.solve()
            p_energy = pair.compute_pair_fragment_energy()
            pair_total += 2*p_energy
            print("Pair fragment energy for ",c," is ", p_energy, " (total: ", pair_total + E_new, " )")



        #for n in np.arange(10):
        #    pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([0,0,n]))
        #    print(pair.compute_pair_fragment_energy())
        #    pair.solve()
        #    print("Pair fragment energy for (0,0,%i):" %n, pair.compute_pair_fragment_energy())
