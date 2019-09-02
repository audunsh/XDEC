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

def advance_amplitudes__(pair_extent, virtual_extent, t2,G,E):
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

#def advance_amplitudes(t,g,f):

class fragment_amplitudes():
    """
    Class that handles t2 amplitudes with dynamically increasing size
    """
    def __init__(self, p, wannier_centers, coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 3.0, occupied_cutoff = 1.0):
        self.p = p #prism object
        self.d = dd.build_distance_matrix(p, coords, wcenters, wcenters) # distance matrix
        
        self.d_ii = dd.build_distance_matrix(p, coords, wcenters[fragment], wcenters[:p.get_nocc()])
        self.d_ia = dd.build_distance_matrix(p, coords, wcenters[fragment], wcenters[p.get_nocc():])
        self.fragment = fragment

        self.ib = ib #integral builder
        self.virtual_cutoff = virtual_cutoff
        self.occupied_cutoff = occupied_cutoff
        self.min_elm = np.min(self.d.blocks[:-1], axis = (1,2)) #array for matrix-size calc
        
        self.min_elm_ii = np.min(self.d_ii.blocks[:-1], axis = (1,2))
        self.min_elm_ia = np.min(self.d_ia.blocks[:-1], axis = (1,2))
        
        self.f_mo_ii = f_mo_ii
        self.f_mo_aa = f_mo_aa

        #d_order1 = np.argsort(np.min(self.d.blocks[:-1, :p.get_nocc(), :p.get_nocc()],axis=(1,2)))
        #d_order2 = np.argsort(np.min(self.d.blocks[:-1, :p.get_nocc(), p.get_nocc():],axis=(1,2)))
        #for i in np.arange(len(d_order1)):
        #    if d_order1[i]!=d_order2[i]:
        #        print("Inconsistent sorting:", i, d_order1[i], d_order2[i])




        self.init_amplitudes()


    def init_amplitudes(self):
        self.n_virtual_cells = np.sum(self.min_elm_ia<=self.virtual_cutoff)
        self.n_occupied_cells = np.sum(self.min_elm_ii<=self.occupied_cutoff)

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1]<=self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1]<=self.occupied_cutoff)
        
        n_occ = self.p.get_nocc()     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells # Number of occupied cells
        n_virt = self.p.get_nvirt()   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells # Number of virtual cells
        

        self.t2 = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = float)

        self.g_d = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = float)

        self.g_x = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = float)

        # Fill in tensors, initial guess, calculate initial energy

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
    
        self.e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]


        self.e0 = 0

        for ddL in np.arange(N_virt):
            for ddM in np.arange(N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]
                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]

                
                    g_direct = self.ib.getcell(dL, M, dM)
                    g_exchange = self.ib.getcell(dM+M, M, dL-M) 
                    t = g_direct*self.e_iajb**-1
                    self.t2[:, ddL, :, mM, :, ddM, :] = g_direct*self.e_iajb**-1
                    self.g_d[:, ddL, :, mM, :, ddM, :] = g_direct
                    self.g_x[:, ddL, :, mM, :, ddM, :] = g_exchange
                    self.e0 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        print("Initial energy from dynamic amplitudes:", self.e0)

    def compute_energy(self):
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
        e_mp2 = 0
        N_virt = self.n_virtual_cells
        
        mM = 0 #occupied index only runs over fragment

        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[0,:]<self.virtual_cutoff # dL index mask
            
            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM]
                dM_i = self.d_ia.cget(dM)[0,:]<self.virtual_cutoff # dM index mask

                


               # Usung multiple levels of masking, probably some other syntax makes more sense
                
                g_direct = self.g_d[:,ddL,:,mM, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]
                g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :][self.fragment][:, dM_i][:, :, self.fragment][:,:,:,dL_i]

                

                #g_direct =   self.g_d[:,ddL,:,mM, :, ddM, :][self.fragment, :, self.fragment, :]
                #print(g_direct.shape)
                
                #g_exchange = self.g_x[self.fragment,ddL,dM_i,mM, self.fragment, ddM, dL_i]



                t = self.t2[:,ddL,:,mM, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]
                e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        return e_mp2



    def init_cell(self, ddL, mmM, ddM):
        dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]
        M = self.d_ii.coords[mmM]

        g_direct = self.ib.getcell(dL, M, dM)
        g_exchange = self.ib.getcell(dM+M, M, dL-M)

        self.t2[:, ddL, :, mmM, :, ddM, :]  = g_direct*self.e_iajb**-1
        self.g_d[:, ddL, :, mmM, :, ddM, :] = g_direct
        self.g_x[:, ddL, :, mmM, :, ddM, :] = g_exchange

    def autoexpand_virtual_space(self, shellwidth = .2):
        new_cut = np.min(self.d_ia.blocks[:-1][self.d_ia.blocks[:-1]>self.virtual_cutoff] + shellwidth)
        print("Increasing virtual cutoff:", self.virtual_cutoff, "->", new_cut)
        self.set_extent(new_cut, self.occupied_cutoff)
    

    def autoexpand_occupied_space(self, shellwidth = .2):
        new_cut = np.min(self.d_ii.blocks[:-1][self.d_ii.blocks[:-1]>self.occupied_cutoff] + shellwidth)
        print("Increasing occupied cutoff:", self.occupied_cutoff, "->", new_cut)
        self.set_extent(self.virtual_cutoff, new_cut)
        

    
    def set_extent(self, virtual_cutoff, occupied_cutoff):
        self.virtual_cutoff = virtual_cutoff
        self.occupied_cutoff = occupied_cutoff

        Nv = np.sum(self.min_elm_ia<=self.virtual_cutoff)
        No = np.sum(self.min_elm_ii<=self.occupied_cutoff)

        n_occ = self.p.get_nocc()
        n_virt = self.p.get_nvirt()

        # Note: forking is due to future implementation of block-specialized initialization
        if Nv > self.n_virtual_cells:
            if No > self.n_occupied_cells:
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
                for ddL in np.arange(Nv):
                    for ddM in np.arange(Nv):
                        for mmM in np.arange(No):
                            if np.abs(self.t2[:,ddL, :, mmM, :, ddM, :]).max()<=10e-12:
                                self.init_cell(ddL, mmM, ddM)
                            






            else:
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
                for ddL in np.arange(Nv):
                    for ddM in np.arange(Nv):
                        for mmM in np.arange(No):
                            if np.abs(self.t2[:,ddL, :, mmM, :, ddM, :]).max()<=10e-12:
                                self.init_cell(ddL, mmM, ddM)

        else:
            if No > self.n_occupied_cells:
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
                for ddL in np.arange(Nv):
                    for ddM in np.arange(Nv):
                        for mmM in np.arange(No):
                            if np.abs(self.t2[:,ddL, :, mmM, :, ddM, :]).max()<=10e-12:
                                self.init_cell(ddL, mmM, ddM)
            else:
                

                
                self.t2 = self.t2[:, :Nv, :, :No, :, :Nv, :]
                self.g_d = self.g_d[:, :Nv, :, :No, :, :Nv, :]
                self.g_x = self.g_x[:, :Nv, :, :No, :, :Nv, :]
        
        # Update domain measures
        self.n_virtual_cells = Nv
        self.n_occupied_cells = No

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1]<=self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1]<=self.occupied_cutoff)
    
    def print_configuration_space_data(self):
        print("%i virtual orbitals included in fragment." % self.n_virtual_tot)
        print("%i occupied orbitals included in fragment." % self.n_occupied_tot)

    def solve_(self):
        # Converge
        #pass
        nocc = self.p.get_nocc()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        vp_indx = mapgen(virtual_extent, pair_extent)
        pp_indx = mapgen(pair_extent, pair_extent)

        for ti in np.arange(100):
            #t2_new = np.zeros((nocc, ndom,  nvirt, pdom, nocc, ndom, nvirt), dtype = float)
            t2_new = np.zeros_like(self.t2)
            for dL in np.arange(self.n_virtual_cells):
                dLv = self.d_ia.coords[dL]
                dL_i = self.d_ia.cget(dLv)[0,:]<self.virtual_cutoff # dL index mask

                for dM in np.arange(self.n_virtual_cells): 
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[0,:]<self.virtual_cutoff # dL index mask
                    for M in np.arange(self.n_occupied_cells):
                        Mv = self.d_ii.coords[M]
                        M_i = self.d_ii.cget(Mv)[0,:]<self.occupied_cutoff # dL index mask



                        tnew = -self.g_d[:, dL, :, M, :, dM, :] #[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i]

                        


                        # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                        Fac = self.f_mo_aa.cget(virtual_extent - virtual_extent[dL])

                        dL_ac = (self.d_ia.cget(virtual_extent - virtual_extent[dL])<=self.virtual_cutoff)[:, self.fragment[0], :]

                        print(Fac.shape, dL_ac.shape)


                        for K in np.arange(self.n_virtual_cells):
                            dK_i = self.d_ia.cget(self.d_ia.coords[K])[0,:]<self.virtual_cutoff
                            tnew -= np.einsum("icjb,ac->iajb", self.t2[:, K, :, M, :, dM, :][self.fragment][:, :][:, :, dK_i][:, :,:,  M_i][:,:,:,:,dM_i] \
                             , Fac[K][:][:, dK_i])


                        #tb = t2[:, :, :, M, :, dM, :][di_occ.cget(di_occ.coords[M])[:,0]]
                        #tnew -= np.einsum("iKcjb,Kac->iajb", self.t2[:, :, :, M, :, dM, :][self.fragment][:, :][:, :, dL_i][:, :,:,  M_i][:,:,:,:,dM_i] \
                        #     , Fac[dL_ac[:, None, :]])




                        # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                        Fbc = self.f_mo_aa.cget(virtual_extent - virtual_extent[dM])
                        tnew -= np.einsum("iajKb,Kbc->iajb", self.t2[:, dL, :, M, :, :, :], Fbc)
                        
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        
                        Fki = self.f_mo_ii.cget(-1*pair_extent)
                        tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)
                        
                        # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                        
                        Fkj = self.f_mo_ii.cget(-1*pair_extent + pair_extent[M])
                        tnew += np.einsum("iaKkb,Kkj->iajb",self.t2[:, dL, :, :, :, dM, :], Fkj)

                        
                        # + \left(t^{\Delta L a, \Delta Mb}_{L'k,Mj}\right)_{n}\varepsilon^{\Delta L a, \Delta Mb}_{0i,Mj},
                        #tnew += t2[:, dL, :, M, :, dM, :] #*E[:,dL,:,M,:,dM,:]**-1

                        t2_new[:, dL, :, M, :, dM, :][self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i] = (tnew*self.e_iajb**-1)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i]
            #print("Residual norm: ")
            self.t2 -= t2_new
            rnorm = np.linalg.norm(t2_new)
            if rnorm<1e-10:

                print("Iteration:",ti, "Amplitude gradient norm:",  np.linalg.norm(t2_new))
                break
    


    def solve(self):
        # Converge
        #pass
        nocc = self.p.get_nocc()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        vp_indx = mapgen(virtual_extent, pair_extent)
        pp_indx = mapgen(pair_extent, pair_extent)

        for ti in np.arange(100):
            #t2_new = np.zeros((nocc, ndom,  nvirt, pdom, nocc, ndom, nvirt), dtype = float)
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
                        #print("Cell map shape:", cell_map.shape)
                        #print(cell_map)

                        #Solve equations

                        # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                        Fac = self.f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                        tnew -= np.einsum("iKcjb,Kac->iajb", self.t2[:, :, :, M, :, dM, :], Fac)

                        # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                        Fbc = self.f_mo_aa.cget(virtual_extent - virtual_extent[dM])
                        tnew -= np.einsum("iajKb,Kbc->iajb", self.t2[:, dL, :, M, :, :, :], Fbc)
                        
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        
                        Fki = self.f_mo_ii.cget(-1*pair_extent)
                        tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)
                        
                        # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}
                        
                        Fkj = self.f_mo_ii.cget(-1*pair_extent + pair_extent[M])
                        tnew += np.einsum("iaKkb,Kkj->iajb",self.t2[:, dL, :, :, :, dM, :], Fkj)

                        
                        # + \left(t^{\Delta L a, \Delta Mb}_{L'k,Mj}\right)_{n}\varepsilon^{\Delta L a, \Delta Mb}_{0i,Mj},
                        #tnew += t2[:, dL, :, M, :, dM, :] #*E[:,dL,:,M,:,dM,:]**-1
                        #print(np.linalg.norm(t2_new[:, dL, :, M, :, dM, :][self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i]), np.linalg.norm(tnew))
                        
                        t2_mapped = np.zeros_like(tnew).ravel()
                        #print(t2_mapped.shape, tnew.shape, self.e_iajb.shape)
                        t2_mapped[cell_map] = (tnew*self.e_iajb**-1).ravel()[cell_map]
                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
                        # t2_new[:, dL, :, M, :, dM, :][self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i] = (tnew*self.e_iajb**-1)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i]
                        
                        #print(np.linalg.norm(t2_new[:, dL, :, M, :, dM, :][self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i]), np.linalg.norm(tnew))
                        #t2_new[:, dL, :, M, :, dM, :]  = tnew*self.e_iajb**-1
            #print("Residual norm: ")
            self.t2 -= t2_new
            rnorm = np.linalg.norm(t2_new)
            if rnorm<1e-10:

                print("Iteration:",ti, "Amplitude gradient norm:",  np.linalg.norm(t2_new))
                break






def converge_fragment_amplitudes(t2, G_direct, f_mo_ii, f_mo_aa, di_virt, di_occ, fragment,p):
    """
    Solve MP2 equations for the given fragment, return amplitudes
    
    Input parameters

     t2     - array containing initial guess amplitudes

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

                    #L_indx = np.array(di_occ.cget([0,0,0]])[fragment[0],:], dtype = bool)
                    #M_indx = np.array(di_occ.cget(di_occ.coords[M])[fragment[0],:], dtype = bool)
                    #dL_indx = np.array(di_occ.cget([0,0,0]])[fragment[0],:], dtype = bool)

                    
                    
                    #print(np.array(di_occ.cget(di_occ.coords[M]), dtype = bool))



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

            print("Iteration:",ti, "Amplitude gradient norm:",  np.linalg.norm(t2_new))
            break
    return t2
    



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
    parser.add_argument("-fitted_coeffs", type= str,help="Array of coefficient matrices from RI-fitting")
    parser.add_argument("auxbasis", type = str, help="Auxiliary fitting basis.")
    parser.add_argument("wcenters", type = str, help="Wannier centers")
    parser.add_argument("-attenuation", type = float, default = 1.2, help = "Attenuation paramter for RI")
    parser.add_argument("-fot", type = float, default = 0.0001, help = "fragment optimization treshold")
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

    c_occ, c_virt = PRI.occ_virt_split(c,p)
    
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
    wcenters = np.load(args.wcenters)

    # Initialize integrals 
    ib = PRI.integral_builder(c,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[1,0,0])

    # Initialize domain definitions



    d = dd.build_distance_matrix(p, c.coords, wcenters, wcenters)



    center_fragments = dd.atomic_fragmentation(p, d, 3.0)
    


    

    # Converge atomic fragment energies
    
    # Initial fragment extents
    virt_cut = 3.0
    occ_cut = 1.0

    for fragment in center_fragments:
        print("Fragment:", fragment)
        
        a_frag = fragment_amplitudes(p, wcenters, c.coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 2.0, occupied_cutoff = 1.0)
       
        a_frag.solve()
        
        # Converge to fot
        E_prev_outer = a_frag.compute_fragment_energy()
        E_prev = E_prev_outer*1.0
        dE_outer = 10
        print("Initial fragment energy:", E_prev)
        while dE_outer>args.fot:
            dE = 10
            while dE>args.fot:

                print("--- virtual")
                a_frag.autoexpand_virtual_space()
                a_frag.solve()
                E_new = a_frag.compute_fragment_energy()
                
                a_frag.print_configuration_space_data()
                dE = np.abs(E_prev - E_new)
                print("E(fragment):", E_new, " DE(fragment):", dE)
                E_prev = E_new
                print("---")
            dE = 10
            while dE>args.fot:

                print("--- virtual")
                a_frag.autoexpand_occupied_space()
                a_frag.solve()
                E_new = a_frag.compute_fragment_energy()
                
                a_frag.print_configuration_space_data()
                dE = np.abs(E_prev - E_new)
                print("E(fragment):", E_new, " DE(fragment):", dE)
                E_prev = E_new
                print("---")
            dE_outer = np.abs(E_prev_outer - E_prev)
            E_prev_outer = E_prev

            
            


        """
        print("Fragment energy:", a_frag.compute_fragment_energy())
        a_frag.solve()
        print("Fragment energy:", a_frag.compute_fragment_energy())
        print(" Testiteration")

        a_frag.autoexpand_virtual_space()
        a_frag.print_configuration_space_data()
        a_frag.solve()
        print("Fragment energy:", a_frag.compute_fragment_energy())
        print(" Testiteration")

        a_frag.autoexpand_virtual_space()
        a_frag.print_configuration_space_data()
        a_frag.solve()
        print("Fragment energy:", a_frag.compute_fragment_energy())
        """

        
        #print("Recalc energy:", a_frag.compute_energy())





        """
        a_frag.print_configuration_space_data()
        a_frag.set_extent(6.0, 1.0)
        a_frag.print_configuration_space_data()
        print("Recalc energy:", a_frag.compute_energy())
        print("Fragment energy:", a_frag.compute_fragment_energy())
        a_frag.solve()
        print("Recalc energy:", a_frag.compute_energy())
        print("Fragment energy:", a_frag.compute_fragment_energy())

        a_frag.print_configuration_space_data()
        """

        # Screen virtual space
        di_v = dd.build_local_domain_index_matrix(fragment, d, virt_cut)
        #di_v.blocks = di_v_blocks[:, :, p.get_nocc():]

        di_o = dd.build_local_domain_index_matrix(fragment, d, occ_cut)
        #di_o.blocks = di_o_blocks[:, :, :p.get_nocc()]

        # Set up initial guess for amplitudes, integrals and domains

        nocc = p.get_nocc()
        nvirt = p.get_nvirt()

        ndom = di_v.coords.shape[0]

        virtual_extent = di_v.coords
        pair_extent = tp.lattice_coords([1,0,0])
        pair_extent = di_o.coords

        pair_center = di_o.mapping[di_o._c2i([0,0,0])]


        #print("p:", pair_extent)

        pdom = pair_extent.shape[0]



        vp_indx = mapgen(virtual_extent, pair_extent)
        pp_indx = mapgen(pair_extent, pair_extent)

        t2 = np.zeros((nocc, ndom,  nvirt, pdom, nocc, ndom, nvirt), dtype = float)
        G_direct = np.zeros((nocc, ndom,  nvirt, pdom, nocc, ndom, nvirt), dtype = float)
        G_exchange = np.zeros((nocc, ndom,  nvirt, pdom, nocc, ndom, nvirt), dtype = float)
        #T = np.zeros((p.get_nocc()))




        for ddL in np.arange(di_v.coords.shape[0]):
            for ddM in np.arange(di_v.coords.shape[0]):
                dL, dM = di_v.coords[ddL], di_v.coords[ddM]
                for mM in np.arange(pair_extent.shape[0]):
                    M = pair_extent[mM]

                
                    g_direct = ib.getcell(dL, M, dM)
                    g_exchange = ib.getcell(dM+M, M, dL-M) #this one ::-/ is it correct?
                    t = g_direct*e_iajb**-1
                    t2[:, ddL, :, mM, :, ddM, :] = g_direct*e_iajb**-1
                    G_direct[:, ddL, :, mM, :, ddM, :] = g_direct
                    G_exchange[:, ddL, :, mM, :, ddM, :] = g_exchange



                #lattice.append([dL, dM])
                #tensors.append([t, g_direct, g_exchange])
                #domains.append([di.cget(dL)[fragment, :], np.arange(2), di.cget(dM)[fragment, :]])
        e0 = 0
        # Perform (initial guess) energy calculation for fragment

        


        for ddL in np.arange(di_v.coords.shape[0]):
            for ddM in np.arange(di_v.coords.shape[0]):
                for mM in [pair_center]: #np.arange(pdom):
                    dM, dL = di_v.coords[ddL], di_v.coords[ddM]
                    M = pair_extent[mM]
                    
                    #g_direct = ib.getcell(dL, [0,0,0], dM)
                    #g_exchange = ib.getcell(dM, [0,0,0], dL)

                    g_direct   =  G_direct[:, ddL, :,  mM, :, ddM, :]
                    g_exchange =  G_exchange[:, ddL, :, mM, :, ddM, :]
                    

                    t = g_direct*e_iajb**-1

                    #di.blocks[:-1][:, fragment[0], None,:]
                    #di.blocks[:-1][:, fragment[0],:, None]





                    e0 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
                    #print(dL, dM, e0)
        print("pre-optimized energy:", e0)




        # solve MP2 equations

        t2 = converge_fragment_amplitudes(t2, G_direct, f_mo_ii, f_mo_aa, di_v, di_o, fragment,p)
        
        """
        # This part is put in external function, will be called twice later
        for ti in np.arange(20):
            t2_new = np.zeros((nocc, ndom,  nvirt, pdom, nocc, ndom, nvirt), dtype = float)
            M = 0
            for dL in np.arange(di_v.coords.shape[0]):
                for dM in np.arange(di_v.coords.shape[0]): 
                    for M in np.arange(pair_extent.shape[0]):
                        tnew = -G_direct[:, dL, :, M, :, dM, :]


                        Fac = f_mo_aa.cget(virtual_extent - virtual_extent[dL])
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
            print("Iteration:",ti, "Amplitude gradient norm:",  np.linalg.norm(t2_new))
            #print("Residual norm: ", )




            #for i in np.arange(len(domains)):
                #dL, dM = lattice[i]
                #t, g_direct, g_exchange = tensors[i]
                #dLi, Mi, dMi = domains[i]
        """


        e0 = 0
        for ddL in np.arange(di_v.coords.shape[0]):
            for ddM in np.arange(di_v.coords.shape[0]):
                for mM in [pair_center]: #np.arange(pdom):
                    dM, dL = di_v.coords[ddL], di_v.coords[ddM]
                    M = pair_extent[mM]
                    
                    #g_direct = ib.getcell(dL, [0,0,0], dM)
                    #g_exchange = ib.getcell(dM, [0,0,0], dL)

                    g_direct   =  G_direct[:, ddL, :,  mM, :, ddM, :]
                    g_exchange =  G_exchange[:, ddL, :, mM, :, ddM, :]
                    

                    t = t2[:, ddL, :,  mM, :, ddM, :] #g_direct*e_iajb**-1




                    e0 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
                    #print(dL, dM, e0)
        print("optimized fragment energy:", e0)
        










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