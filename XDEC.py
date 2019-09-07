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

        

        #self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1]<=self.virtual_cutoff)
        #self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1]<=self.occupied_cutoff)
        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1,self.fragment[0]]<=self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<=self.occupied_cutoff)
    
        
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
        #print("Initial energy from dynamic amplitudes:", self.e0)

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
                        tnew -= np.einsum("iajKb,Kbc->iajb", self.t2[:, dL, :, M, :, :, :], Fbc)
                        
                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        Fki = self.f_mo_ii.cget(-1*pair_extent)
                        tnew += np.einsum("Kkajb,Kki->iajb",self.t2[:, vp_indx[dL], :, pp_indx[M], :, vp_indx[dM], :], Fki)
                        
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
    parser.add_argument("-basis_truncation", type = float, default = 0.5, help = "Truncate AO-basis function below this threshold." )
    parser.add_argument("-fot", type = float, default = 0.001, help = "fragment optimization treshold")
    parser.add_argument("-circulant",default = False, action = "store_true", help = "fragment optimization treshold")
    parser.add_argument("-attenuated_truncation", type = float, default = 1e-14, help = "Truncate blocks in the attenuated matrix where (max) elements are below this threshold." )
    parser.add_argument("-robust", default = False, action = "store_true", help = "Enable Dunlap robust fit for improved integral accuracy.")
    parser.add_argument("-n_core", type = int, default = 0, help = "Number of core orbitals (the first n_core orbitals will not be correlated).")
    args = parser.parse_args()


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
    
    ib = PRI.integral_builder(c,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[1,0,0], circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust)

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
    occ_cut = 1.0

    for fragment in center_fragments:
        

        ib.fragment = fragment
        
        a_frag = fragment_amplitudes(p, wcenters, c.coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 2.0, occupied_cutoff = 1.0)
        
        
        print("Running fragment optimization for:")
        print(fragment)
        #print("Initial cutoffs:")
        print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
        print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))        
        
        print(" ")
        a_frag.solve()
        
        # Converge to fot
        E_prev_outer = a_frag.compute_fragment_energy()
        E_prev = E_prev_outer*1.0
        dE_outer = 10
        #print("Initial fragment energy: %.5e" % E_prev)
        while dE_outer>args.fot:
            dE = 10
            while dE>args.fot:

                #print("--- virtual")
                a_frag.autoexpand_virtual_space(n_orbs=4)
                a_frag.solve()
                E_new = a_frag.compute_fragment_energy()
                
                #a_frag.print_configuration_space_data()
                dE = np.abs(E_prev - E_new)
                print("_________________________________________________________")
                print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new, dE))
                print("_________________________________________________________")
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
                print(" ")
                E_prev = E_new
                #print("---")
            #dE = 10
            #print("--- occupied")
            a_frag.autoexpand_occupied_space(n_orbs=2)
            a_frag.solve()
            E_new = a_frag.compute_fragment_energy()
            
            #a_frag.print_configuration_space_data()
            dE = np.abs(E_prev - E_new) 

            print("_________________________________________________________")
            print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new, dE))
            print("_________________________________________________________")
            print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
            print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
            print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
            print(" ")
            E_prev = E_new
            #print("---")

            while dE>args.fot:

                #print("--- occupied")
                a_frag.autoexpand_occupied_space(n_orbs=2)
                a_frag.solve()
                E_new = a_frag.compute_fragment_energy()
                
                #a_frag.print_configuration_space_data()
                dE = np.abs(E_prev - E_new)
                print("_________________________________________________________")
                print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new, dE))
                print("_________________________________________________________")
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
                print(" ")
                E_prev = E_new
                #print("---")
            dE_outer = np.abs(E_prev_outer - E_prev)
            E_prev_outer = E_prev
        #print("Current memory usage of integrals (in MB):", ib.nbytes())
        print("Converged fragment energy: %.12f" % E_new, "(Periodic RI")
        print(" ")
        print(" ")
        #print(-0.114393980708, "(3D Neon, fot 0.0001 (Gustav))")