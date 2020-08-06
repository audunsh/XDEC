#!/usr/bin/env python


import numpy as np

#import ad

import os

import subprocess as sp

from ast import literal_eval

from scipy import optimize

import argparse

import utils.toeplitz as tp

import lwrap.lwrap as li # Libint wrapper

import utils.objective_functions as of

import utils.prism as pr
import domdef as dd
import PRI
import time

#from pympler import muppy, summary
#import gc

"""
Cosmetics
"""

def get_progress_bar(progress, total, length = 60):
    """
    returns a string indicating the progress
    """
    s = """|"""
    s += ">"*np.int(length*progress/total)
    s += " "*np.int(length*((total - progress)/total))
    s += "|"
    return s


"""
Functions that aids mapping of blocks in tensors
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


def get_bool_index_where(coords, coord):
    """
    return index i, such that coords[i] == coord
    """
    mask = np.all(coords[:, None] == coord[None, :], axis = 2)
    #items= np.outer( np.arange(coords.shape[0]), np.ones(coord.shape[0], dtype = int))
    return np.any(mask, axis = 1)

def get_bvec_where(coords, coord):
    """
    return bool index vector i, such that coords[i] == coord
    """

    """
    mask = np.all(coords[:, None] == coord[None, :], axis = 2)
    items= np.outer( np.arange(coords.shape[0]),np.ones(coord.shape[0], dtype = int))
    bvec = np.any(mask, axis = 0)
    """

    mask = np.all(coords[None,:] == coord[:, None], axis = 2)
    items= np.outer(np.ones(coord.shape[0], dtype = int), np.arange(coords.shape[0]))
    bvec = np.any(mask, axis = 1)
    
    #print("items[mask]")
    #print(items[mask])
    #print("bvec")
    #print(bvec)
    
    ret = -1*np.ones(coord.shape[0], dtype = int)
    
    ret[bvec] = items[mask]
    return ret

def get_index_where(coords, coord):
    """
    return index i, such that coords[i] == coord
    """
    if coord.size == 3:
        # Special case when only one coordinate requested
        mask = np.all(coords[:, None] == coord[None, :], axis = 2)
        items= np.outer( np.arange(coords.shape[0]), np.ones(1, dtype = int))
        elm = items[mask]
        if elm.size >0:
            return items[mask][0]
        else:
            return None
    else:
        mask = np.all(coords[:, None] == coord[None, :], axis = 2)
        items= np.outer( np.arange(coords.shape[0]), np.ones(coord.shape[0], dtype = int))
        return items[mask] 


def get_index_where__(coords, coord):
    """
    return index i, such that coords[i] == coord
    """
    if coord.size == 3:
        # Special case when only one coordinate requested
        mask = np.all(coords[:, None] == coord[None, :], axis = 2)
        items= np.outer( np.arange(coords.shape[0]), np.ones(1, dtype = int))
        return items[mask][0]
    else:
        mask = np.all(coords[:, None] == coord[None, :], axis = 2)
        items= np.outer( np.arange(coords.shape[0]), np.ones(coord.shape[0], dtype = int))
        return items[mask] 

def get_index_where_old(coords, coord):
    """
    return index i, such that coords[i] == coord
    """
    return np.argwhere(np.sum((coords-coord)**2, axis = 1)==0)[0,0]


"""
Solver class with various residuals and optimization schemes
"""


class amplitude_solver():
    def print_configuration_space_data(self):
        """
        Prints extent sizes in terms of number of orbitals
        """
        print("%i virtual orbitals included in fragment." % self.n_virtual_tot)
        print("%i occupied orbitals included in fragment." % self.n_occupied_tot)

    def solve(self, norm_thresh = 1e-10, eqtype = "mp2", s_virt = None, damping = 1.0, ndiis = 8, energy = None, pairwise = False):
        #print("NDIIS = ", ndiis)
        if eqtype == "mp2_nonorth":
            #return self.solve_MP2PAO(norm_thresh, s_virt = s_virt, damping = damping)
            return self.solve_unfolded_pao(norm_thresh = norm_thresh, maxiter = 100, damping = damping, energy = energy, compute_missing_exchange = False, s_virt = s_virt)

        elif eqtype == "paodot":
            return self.solve_MP2PAO_DOT(norm_thresh, s_virt = s_virt)
        elif eqtype == "ls":
            return self.solve_MP2PAO_ls(norm_thresh, s_virt = s_virt)

        
        
        else:
            #print("ENERGU:", energy)
            #self.t2 *= 0
            #print("NORM_THRESH = ", norm_thresh)
            return self.solve_unfolded(norm_thresh = norm_thresh, maxiter = 100, damping = damping, energy = energy, compute_missing_exchange = False, pairwise = pairwise)
            
            #return self.solve_unfolded_pao(norm_thresh = norm_thresh, maxiter = 100, damping = damping, energy = energy, compute_missing_exchange = False, s_virt = tp.get_identity_tmat(ib.n_virt))
            
            
            
            
            #return self.solve_completely_unfolded(norm_thresh = norm_thresh, maxiter = 100, damping = damping, energy = energy, compute_missing_exchange = False)
            #return self.solve_unfolded(norm_thresh = norm_thresh, maxiter = 100, damping = damping, energy = energy, compute_missing_exchange = False)
            if False:
                dt, it = self.solve_MP2(norm_thresh, ndiis = ndiis, damping = damping)
                if energy == "fragment":
                    return dt, it, self.compute_fragment_energy()
                if energy == "pair":
                    return dt, it, self.compute_pair_fragment_energy()


    def solve_unfolded(self, norm_thresh = 1e-7, maxiter = 100, damping = 1.0, energy = None, compute_missing_exchange = True, pairwise = False):
        # Standard solver for orthogonal virtual space
        #self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i([0,0,0]) ] ] *= 0

        
        t0 = time.time()

        #For quick reference
        t2 = self.t2               # amplitudes
        no = self.ib.n_occ         # number of occupieds per cell
        No = self.n_occupied_cells # numer of occupied cells
        nv = self.ib.n_virt        # number of virtuals per cell
        Nv = self.n_virtual_cells  # number of virtual cells
        no0 = len(self.fragment) #number of occupied orbitals in refcell
        
        # Get the virtual and occupied cells with orbitals inside domain
        vcoords = self.d_ia.coords[:self.n_virtual_cells]
        ocoords = self.d_ii.coords[:self.n_occupied_cells]
        


        # set up boolean masking arrays
        ia_mask = self.d_ia.cget(vcoords)[:, self.fragment[0], :]
        ia_mask = ia_mask.ravel()<self.virtual_cutoff    # active virtual indices


        ii_mask = self.d_ii.cget(ocoords)[:, self.fragment[0], :]        
        ii_mask = ii_mask.ravel()<self.occupied_cutoff   # active occupied indices
        
        i0_mask = self.d_ii.cget([0,0,0])[self.fragment[0], :]<self.occupied_cutoff #active occupied indices in reference cell
        

        # Unfold Fock-matrix
        Fii = self.f_mo_ii.tofull(self.f_mo_ii, ocoords, ocoords)[ii_mask][:, ii_mask]
        Faa = self.f_mo_aa.tofull(self.f_mo_aa, vcoords,vcoords)[ia_mask][:, ia_mask]

        
        t2_unfolded = np.zeros_like(self.t2)  
        g_unfolded  = np.zeros_like(self.g_d)

        # preparing mapping array for t2-tensor unfolding
        indx_flat = np.arange(t2_unfolded.size).reshape(self.t2.shape)
        indx_flat__ = indx_flat.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask]
        indx_flat *= 0
        indx_flat -= 1
        indx_flat = indx_flat.ravel()
        indx_flat[indx_flat__.ravel()] = np.arange(indx_flat__.size) #remapping elements
        indx_flat = indx_flat.reshape(self.t2.shape)








        
        indx = np.zeros(self.g_d.shape, dtype = int)-1 #use -1 as default index

        indx_full = np.zeros((No, no, Nv, nv, No, no, Nv, nv), dtype = int) -1

        print("Preparations (1):", time.time()-t0)
        t0 = time.time()


        for dL in range(Nv):
            for M in range(No):
                for dM in range(Nv):
                    dM_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM] - self.d_ii.coords[M]) # \tilde{t} -> t, B = M + dM


                    

                    if dM_i is not None:
                        t2_unfolded[:, dL, :, M, :, dM, :] = self.t2[:, dL, :, M, :, dM_i]
                        g_unfolded[:, dL, :, M, :, dM, :] = self.g_d[:, dL, :, M, :, dM_i]
                        
                    else:
                        if compute_missing_exchange:
                            # compute the integrals missing from relative index tensor
                        
                            I, Is = self.ib.getorientation(self.d_ia.coords[dL], self.d_ia.coords[dM])
                            g_unfolded[:, dL, :, M, :, dM, :] = I.cget(self.d_ii.coords[M]).reshape(Is)

                        
                    for N in range(No):

                        
                        M_i = get_index_where(self.d_ii.coords[:self.n_occupied_cells], self.d_ii.coords[M]    - self.d_ii.coords[N])

                        dM_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM]  - self.d_ii.coords[N])
                        
                        dL_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dL]  - self.d_ii.coords[N])


                        if np.any(np.array([dL_i, M_i, dM_i])==None):
                            pass
                        else:
                            indx_full[N, :, dL, :, M, :, dM, :] = indx_flat[:, dL_i, :, M_i, :, dM_i, :]  # 
                            """
                            try:
                                indx_full[N, :, dL, :, M, :, dM, :] = indx_flat[:, dL_i, :, M_i, :, dM_i, :] 
                            except:
                                pass
                            """
            #print(dL, "of", Nv, " complete.")
            print(get_progress_bar(dL,Nv), end="\r")
        print(" ")

                        
        print("Unfolding (1):", time.time()-t0)
        t0 = time.time()


        t2s = t2_unfolded.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask] # Rank 4 tensor / 4 dimensions
        v2s = g_unfolded.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask] # Rank 4 tensor / 4 dimensions


        # Unfolded tensor
        idx = indx.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask] # t2s.ravel()[idx] : t^{AB}_{0J} -> t^{B-J, A-J}_{0J}
        idx_f = indx_full.reshape(No*no, Nv*nv, No*no, Nv*nv)[ii_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask]
        idx_f_mask = (idx_f<0).ravel()



        t2i = np.arange(t2.size).reshape(t2.shape)
        t2i_map = t2i.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask].ravel()

        #Construct fock denominator

        fii0 = np.diag(self.f_mo_ii.cget([0,0,0])).ravel()

        faa0 = np.diag(self.f_mo_aa.cget([0,0,0]))

        fii0_ = np.outer(np.ones(No, dtype = float), fii0).ravel()[ii_mask]

        fii0 = np.diag(self.f_mo_ii.cget([0,0,0])).ravel()[i0_mask]

        faa0_ = np.outer(np.ones(Nv, dtype = float), faa0).ravel()[ia_mask]

        fiajb = fii0[:,None,None,None] - faa0_[None,:,None,None] + fii0_[None,None,:,None] - faa0_[None,None,None,:]




        norm_prev = 1000
        
        

        
        nvt = np.sum(ia_mask) #number of axtive virtuals
        ni = t2s.shape[0]
        nj = t2s.shape[2]


        print("Preparations (2):", time.time()-t0)
        t0 = time.time()

        t0a = 0
        t0b = 0
        t0c = 0
        t0c_2 = 0
        t0d = 0
        print("MP2 iterations")


        for i in range(maxiter):
            t0_ = time.time()

            t2new = -1*v2s

            # D1

            #t2new -= np.einsum("icjb,ac->iajb", t2s, Faa) 
            t2new -= np.dot(t2s.swapaxes(1,3).reshape(ni*nvt*nj, nvt), Faa).reshape(ni, nvt, nj, nvt).swapaxes(1,3)



            t0a += time.time()-t0_
            t0_ = time.time()

            # D2
            #t2new -= np.einsum("iajc,bc->iajb", t2s, Faa)
            t2new -= np.dot(t2s.reshape(ni*nvt*nj, nvt), Faa).reshape(ni, nvt, nj, nvt)


            t0b += time.time()-t0_
            t0_ = time.time()

            # D3 
            # Unfold the "fixed" axis
            t2s_ = t2s.ravel()[idx_f]
            t2s_ = t2s_.ravel()
            t2s_[idx_f_mask] = 0
            t2s_ = t2s_.reshape(idx_f.shape)

            t0c_2 += time.time()-t0_
            t0_ = time.time()
            
            #t2new += np.einsum("ik,kajb->iajb", Fii[:t2s.shape[0], :], t2s_)
            t2new += np.dot(Fii[:t2s.shape[0], :], t2s_.reshape(nj, nvt*nj*nvt)).reshape(ni, nvt, nj, nvt)



            t0c += time.time()-t0_
            t0_ = time.time()




            # D4

            #t2new += np.einsum("iakb,kj->iajb", t2s, Fii)
            t2new += np.dot(t2s.swapaxes(2,3).reshape(ni*nvt*nvt, nj), Fii).reshape(ni,nvt,nvt,nj).swapaxes(2,3)
            
            t2s -= damping*t2new*(fiajb**-1)

            abs_dev = np.max(np.abs(t2new))
            t0d += time.time()-t0_
            t0_ = time.time()


            if abs_dev<norm_thresh: #np.abs(norm_prev - norm_new)<norm_thresh:
                #print("Converged at", i, "iterations.")
                break
            #print("Iteration:", i, "of", maxiter, ".", abs_dev)
            print(get_progress_bar(i,maxiter), end="\r")
        print(" ")

        #t_t = (time.time()-t0)/
        #print("Time per iteration:", t_t)

        print("Solving (1):", time.time()-t0)
        print("         D1:", t0a)
        print("         D2:", t0b)
        print("(unfold) D3:", t0c_2)
        print("         D3:", t0c)
        print("         D4:", t0d)
        t0 = time.time()

        
                



        t2new_full = np.zeros_like(t2).ravel()
        t2new_full[t2i_map] = t2s.ravel()
        t2new_full = t2new_full.reshape(t2.shape)

        miss = 0

        for dL in range(Nv):
            for M in range(No):
                for dM in range(Nv):
                    dM_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM] - self.d_ii.coords[M]) 


                    #dA_J = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dL]-self.d_ii.coords[M]) 
                    #dB_J = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM]-self.d_ii.coords[M])

                    

                    #if dM_i is not None:
                    try:

                        self.t2[:, dL, :, M, :, dM_i, :] = t2new_full[:, dL, :, M, :, dM]
                        
                    except:
                        
                        miss += 1
        
        print("Remapping (1):", time.time()-t0)
        t0 = time.time()


        # Compute the requested energy

        if energy == "fragment":
            # Fragment energy
            d_ii_1 = self.d_ii*1
            d_ii_1.blocks *= 0

            d_ii_1.blocks[d_ii_1.mapping[ d_ii_1._c2i(np.array([0,0,0]))], self.fragment[0], self.fragment] = 1

            i0_full_mask = np.array((d_ii_1.cget(ocoords)[:, self.fragment[0], :].ravel())[ii_mask], dtype = np.bool) # fragment 1 in refcell
            f0_mask = np.array((d_ii_1.cget([0,0,0])[self.fragment[0], :].ravel())[i0_mask], dtype = np.bool) #fragment 1 in refcell, only indexes in refcell




            t20 = t2s[f0_mask][:, :, i0_full_mask]
            v20 = v2s[f0_mask][:, :, i0_full_mask]

            print("abs.max. fragment amplitude:", np.max(np.abs(t20)), np.linalg.norm(t20))
            

            energy = 2*np.einsum("iajb,iajb", t20, v20) - np.einsum("iajb,ibja", t20, v20)
            print("Energy (1):", time.time()-t0)
            t0 = time.time()
            
            return np.max(np.abs(t2new)), i, energy

        if energy == "cim": 
            # Cluster-in-molecule energy
            d_ii_1 = self.d_ii*1
            d_ii_1.blocks *= 0

            d_ii_1.blocks[d_ii_1.mapping[ d_ii_1._c2i(np.array([0,0,0]))], self.fragment[0], self.fragment] = 1

            f0_mask = np.array((d_ii_1.cget([0,0,0])[self.fragment[0], :].ravel())[i0_mask], dtype = np.bool) #fragment 1 in refcell, only indexes in refcell

            #print("f0_mask", f0_mask)
            e_ij =  2*np.einsum("iajb,iajb->j", t2s[f0_mask], v2s[f0_mask]) - np.einsum("iajb,ibja->j", t2s[f0_mask], v2s[f0_mask])
            d_ij = self.d_ii.cget(ocoords)[:, self.fragment[0], :].ravel()[ii_mask]

            s_ij = np.argsort(d_ij)

            #print(e_ij.shape)
            #print(d_ij.shape)
            #print("energy_ij:", e_ij[s_ij])
            #print("d_ij:", d_ij[s_ij])
            
            #print("distance_ij", d_ij)
            if True:
                e_a =  2*np.einsum("iajb,iajb->a", t2s[f0_mask], v2s[f0_mask]) - np.einsum("iajb,ibja->a", t2s[f0_mask], v2s[f0_mask])
                d_a = self.d_ia.cget(vcoords)[:, self.fragment[0], :].ravel()[ia_mask]
                a_s = np.argsort(d_a)

                #print("Energy (1):", time.time()-t0)

                #print(np.max(np.abs(t2new)), i, energy, np.array([e_a[a_s],d_a[a_s]]))

                energy = np.sum(e_a)
                print("Energy (1):", time.time()-t0)
                t0 = time.time()

                return np.max(np.abs(t2new)), i, energy, np.array([e_a[a_s],d_a[a_s]])
                #print()


            else:

                energy = 2*np.einsum("iajb,iajb", t2s[f0_mask], v2s[f0_mask]) - np.einsum("iajb,ibja", t2s[f0_mask], v2s[f0_mask])
                print("Energy (1):", time.time()-t0)
                t0 = time.time()

                return np.max(np.abs(t2new)), i, energy


        if energy == "pair":
            # Pair-fragment energy
            d_ii_1 = self.d_ii*1
            d_ii_1.blocks *= 0

            d_ii_1.blocks[d_ii_1.mapping[ d_ii_1._c2i(np.array([0,0,0]))], self.f1.fragment[0], self.f1.fragment] = 1

            d_ii_2 = self.d_ii*1
            d_ii_2.blocks *= 0

            d_ii_2.blocks[d_ii_2.mapping[ d_ii_2._c2i(np.array([0,0,0]))], self.f2.fragment[0], self.f2.fragment] = 1


            mask_01 = np.array((d_ii_1.cget(ocoords)[:, self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) # fragment 1 in refcell
            mask_11 = np.array((d_ii_1.cget(ocoords-self.M)[:, self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) # fragment 1 in +1
            mask_1_1 = np.array((d_ii_1.cget(ocoords+self.M)[:, self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) #fragment 1 in -1
            mask_10 = np.array((d_ii_1.cget([0,0,0])[self.f1.fragment[0], :].ravel())[i0_mask], dtype = np.bool) #fragment 1 in refcell, only indexes in refcell

            mask_02 = np.array((d_ii_2.cget(ocoords)[:, self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)
            mask_22 = np.array((d_ii_2.cget(ocoords-self.M)[:, self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)
            mask_2_2 = np.array((d_ii_2.cget(ocoords+self.M)[:, self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)
            mask_20 = np.array((d_ii_2.cget([0,0,0])[self.f2.fragment[0], :].ravel())[i0_mask], dtype = np.bool)



            if True:
                energy_jj_11 = 2*np.einsum("iajb,iajb->j", t2s[mask_10], v2s[mask_10]) - np.einsum("iajb,ibja->j", t2s[mask_10], v2s[mask_10]) 
                energy_jj_22 = 2*np.einsum("iajb,iajb->j", t2s[mask_20], v2s[mask_20]) - np.einsum("iajb,ibja->j", t2s[mask_20], v2s[mask_20]) 
                #print("All energies:")
                #print(energy_jj_11)
                #print(energy_jj_22)

            energy_11 = 2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_11], v2s[mask_10][:, :, mask_11]) -  \
                          np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_11], v2s[mask_10][:, :, mask_11]) 
            
            energy_1_1 =2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_1_1], v2s[mask_10][:, :, mask_1_1]) -  \
                          np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_1_1], v2s[mask_10][:, :, mask_1_1])

            

            



            energy_12 = 2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_22], v2s[mask_10][:, :, mask_22]) -  \
                          np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_22], v2s[mask_10][:, :, mask_22]) 
            
            

            energy_2_1= 2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_1_1], v2s[mask_20][:, :, mask_1_1]) -  \
                          np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_1_1], v2s[mask_20][:, :, mask_1_1])

            


            energy_1_2 =2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_2_2], v2s[mask_10][:, :, mask_2_2]) -  \
                          np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_2_2], v2s[mask_10][:, :, mask_2_2])

            
            

            

            energy_21 = 2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_11], v2s[mask_20][:, :, mask_11]) -  \
                          np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_11], v2s[mask_20][:, :, mask_11]) 

            

            

            energy_22 = 2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_22], v2s[mask_20][:, :, mask_22]) -  \
                          np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_22], v2s[mask_20][:, :, mask_22]) 

            energy_2_2 =2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_2_2], v2s[mask_20][:, :, mask_2_2]) -  \
                          np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_2_2], v2s[mask_20][:, :, mask_2_2])


            print("Energies:")
            print(energy_11, energy_1_1) # first one
            print(energy_22, energy_2_2) #first one
            print(energy_12, energy_2_1) # first one ... energy_12 \approx 2*energy_2_1 
            print(energy_1_2, energy_21) # sedcond one

            

            #print("Energies:", energy_11, energy_1_1, energy_22, energy_2_2, energy_12, energy_2_1, energy_21,energy_1_2 )

            energy = np.array([energy_11, energy_22, energy_12, energy_21])
            #energy = np.array([energy_1_1, energy_2_2, energy_2_1, energy_1_2])
            
            
            #energy = np.array([energy_11+energy_1_1, energy_22+energy_2_2, energy_2_1+energy_12, energy_21+energy_1_2])

            # Counter-Poise
            #print(mask_10, mask_20, np.sum(mask_02), np.sum(mask_01))
            #energy_0101 = 2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_01], v2s[mask_10][:, :, mask_01]) -  \
            #                np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_01], v2s[mask_10][:, :, mask_01]) 

            #energy_0202 = 2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_02], v2s[mask_20][:, :, mask_02]) -  \
            #                np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_02], v2s[mask_20][:, :, mask_02]) 

            #print("Counter-Poise energies:", energy_0101, energy_0202)




            #print("Energy:", energy)
            print("Energy (1):", time.time()-t0)
            t0 = time.time()
            return np.max(np.abs(t2new)), i, energy


            #ii_mask = ii_mask.ravel()<self.occupied_cutoff   # active occupied indices





        return np.max(np.abs(t2new)), i

    def solve_unfolded_pao(self, norm_thresh = 1e-7, maxiter = 100, damping = 1.0, energy = None, compute_missing_exchange = True, s_virt = None):
        # Standard solver for non-orthogonal virtual space

        self.s_pao = s_virt

        
        t0 = time.time()

        #For quick reference
        t2 = self.t2               # amplitudes
        no = self.ib.n_occ         # number of occupieds per cell
        No = self.n_occupied_cells # numer of occupied cells
        nv = self.ib.n_virt        # number of virtuals per cell
        Nv = self.n_virtual_cells  # number of virtual cells
        no0 = len(self.fragment) #number of occupied orbitals in refcell
        
        # Get the virtual and occupied cells with orbitals inside domain
        vcoords = self.d_ia.coords[:self.n_virtual_cells]
        ocoords = self.d_ii.coords[:self.n_occupied_cells]
        


        # set up boolean masking arrays
        ia_mask = self.d_ia.cget(vcoords)[:, self.fragment[0], :]
        ia_mask = ia_mask.ravel()<self.virtual_cutoff    # active virtual indices


        ii_mask = self.d_ii.cget(ocoords)[:, self.fragment[0], :]        
        ii_mask = ii_mask.ravel()<self.occupied_cutoff   # active occupied indices
        
        i0_mask = self.d_ii.cget([0,0,0])[self.fragment[0], :]<self.occupied_cutoff #active occupied indices in reference cell


        # Unfold Fock-matrix
        Fii = self.f_mo_ii.tofull(self.f_mo_ii, ocoords, ocoords)[ii_mask][:, ii_mask]
        Faa = self.f_mo_aa.tofull(self.f_mo_aa, vcoords,vcoords)[ia_mask][:, ia_mask]

        Saa = self.s_pao.tofull(self.s_pao, vcoords, vcoords)[ia_mask][:, ia_mask]

        

        
        t2_unfolded = np.zeros_like(self.t2)  
        g_unfolded  = np.zeros_like(self.g_d)

        # preparing mapping array for t2-tensor unfolding
        indx_flat = np.arange(t2_unfolded.size).reshape(self.t2.shape)
        indx_flat__ = indx_flat.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask]
        indx_flat *= 0
        indx_flat -= 1
        indx_flat = indx_flat.ravel()
        indx_flat[indx_flat__.ravel()] = np.arange(indx_flat__.size) #remapping elements
        indx_flat = indx_flat.reshape(self.t2.shape)








        
        indx = np.zeros(self.g_d.shape, dtype = int)-1 #use -1 as default index

        indx_full = np.zeros((No, no, Nv, nv, No, no, Nv, nv), dtype = int) -1

        print("Preparations (1):", time.time()-t0)
        t0 = time.time()


        for dL in np.arange(Nv):
            for M in np.arange(No):
                for dM in np.arange(Nv):
                    dM_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM] - self.d_ii.coords[M]) # \tilde{t} -> t, B = M + dM


                    

                    if dM_i is not None:
                        t2_unfolded[:, dL, :, M, :, dM, :] = self.t2[:, dL, :, M, :, dM_i]
                        g_unfolded[:, dL, :, M, :, dM, :] = self.g_d[:, dL, :, M, :, dM_i]
                        
                    else:
                        if compute_missing_exchange:
                            # compute the integrals missing from relative index tensor
                        
                            I, Is = self.ib.getorientation(self.d_ia.coords[dL], self.d_ia.coords[dM])
                            g_unfolded[:, dL, :, M, :, dM, :] = I.cget(self.d_ii.coords[M]).reshape(Is)

                        
                    for N in np.arange(No):

                        
                        M_i = get_index_where(self.d_ii.coords[:self.n_occupied_cells], self.d_ii.coords[M]    - self.d_ii.coords[N])

                        dM_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM]  - self.d_ii.coords[N])
                        
                        dL_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dL]  - self.d_ii.coords[N])


                        if np.any(np.array([dL_i, M_i, dM_i])==None):
                            pass
                        else:
                            indx_full[N, :, dL, :, M, :, dM, :] = indx_flat[:, dL_i, :, M_i, :, dM_i, :]  # 
                            """
                            try:
                                indx_full[N, :, dL, :, M, :, dM, :] = indx_flat[:, dL_i, :, M_i, :, dM_i, :] 
                            except:
                                pass
                            """

                        
        print("Unfolding (1):", time.time()-t0)
        t0 = time.time()


        t2s = t2_unfolded.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask] # Rank 4 tensor / 4 dimensions
        v2s = g_unfolded.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask] # Rank 4 tensor / 4 dimensions


        # Unfolded tensor
        idx = indx.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask] # t2s.ravel()[idx] : t^{AB}_{0J} -> t^{B-J, A-J}_{0J}
        idx_f = indx_full.reshape(No*no, Nv*nv, No*no, Nv*nv)[ii_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask]
        idx_f_mask = (idx_f<0).ravel()



        t2i = np.arange(t2.size).reshape(t2.shape)
        t2i_map = t2i.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask].ravel()

        #Construct fock denominator
        print(np.diag(self.s_pao.cget([0,0,0])))

        fii0 = np.diag(self.f_mo_ii.cget([0,0,0])).ravel()

        faa0 = np.diag(self.f_mo_aa.cget([0,0,0]))

        fii0_ = np.outer(np.ones(No, dtype = float), fii0).ravel()[ii_mask]

        fii0 = np.diag(self.f_mo_ii.cget([0,0,0])).ravel()[i0_mask]

        faa0_ = np.outer(np.ones(Nv, dtype = float), faa0).ravel()[ia_mask]

        fiajb = fii0[:,None,None,None] - faa0_[None,:,None,None] + fii0_[None,None,:,None] - faa0_[None,None,None,:]

        if True:
            f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
            f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
            s_aa = np.diag(self.s_pao.cget([0,0,0]))
            f_ij = f_ii[:,None] + f_ii[None,:]


            sfs = np.einsum("a,ij,b->iajb",s_aa,f_ij,s_aa)
            fs_ab = np.einsum("a,b,i,j->iajb",f_aa,s_aa,np.ones(no),np.ones(no))
            fs_ba = np.einsum("b,a,i,j->iajb",f_aa,s_aa,np.ones(no),np.ones(no))
            f_iajb = sfs - fs_ab - fs_ba

            f_full = np.zeros_like(self.t2)
            for dL in np.arange(Nv):
                for M in np.arange(No):
                    for dM in np.arange(Nv):

                        f_full[:,dL, :, M, :, dM, : ] = f_iajb
            
            fiajb = f_full.reshape(no, Nv*nv, No*no, Nv*nv)[i0_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask]
        


        norm_prev = 1000
        
        

        
        nvt = np.sum(ia_mask) #number of axtive virtuals

        print("Preparations (2):", time.time()-t0)
        t0 = time.time()

        t0a = 0
        t0b = 0
        t0c = 0
        t0d = 0

        #t2new = -1*v2s

        DIIS = diis(8)


        for i in np.arange(maxiter):
            # construct tbar

            t2bar = np.einsum("ac,icjd,db->iajb", Saa, t2s, Saa)
            #t2bar = np.einsum("iajc, cb->iajb", np.einsum("ac, icjb->iajb", Saa, t2s), Saa)





            t0_ = time.time()

            t2new = -1*v2s

            # D1
            t2new -= np.einsum("ac, icjd, db->iajb", Saa, t2s, Faa)
            #t2new -= np.einsum("ac, icjb->iajb"  , Saa,     np.einsum("iajc,cb->iajb", t2s, Faa)    ) #*0

            t0a += t0_-time.time()
            t0_ = time.time()

            # D2

            t2new -= np.einsum("ac, icjd, db->iajb", Faa, t2s, Saa)
            #t2new -= np.einsum("ac, icjb->iajb" ,    Faa, np.einsum("iajc,cb->iajb", t2s, Saa)     )
             

            t0b += t0_-time.time()
            t0_ = time.time()

            # D3 
            # Unfold the "fixed" axis
            t2s_ = t2bar.ravel()[idx_f]
            t2s_ = t2s_.ravel()
            t2s_[idx_f_mask] = 0
            t2s_ = t2s_.reshape(idx_f.shape)
            t2new += np.einsum("kajb, ik->iajb", t2s_, Fii[:t2s.shape[0], :])
            #t2new += np.einsum("ik, kajb->iajb", Fii[:t2s.shape[0], :], t2s_)

            t0c += t0_-time.time()
            t0_ = time.time()




            # D4
            t2new +=  np.einsum("iakb,kj->iajb", t2bar, Fii)
            
            #t2s -= damping*t2new*(fiajb**-1)

            t2s = DIIS.advance(t2s, damping*t2new)

            abs_dev = np.max(np.abs(t2new))
            t0d += t0_-time.time()
            t0_ = time.time()


            if abs_dev<norm_thresh: #np.abs(norm_prev - norm_new)<norm_thresh:
                #print("Converged at", i, "iterations.")
                break

        t_t = (time.time()-t0)/i
        print("Time per iteration:", t_t)

        print("Solving (1):", time.time()-t0)
        print(t0a)
        print(t0b)
        print(t0c)
        print(t0d)
        t0 = time.time()

        
                



        t2new_full = np.zeros_like(t2).ravel()
        t2new_full[t2i_map] = t2s.ravel()
        t2new_full = t2new_full.reshape(t2.shape)

        miss = 0

        for dL in np.arange(Nv):
            for M in np.arange(No):
                for dM in np.arange(Nv):
                    dM_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM] - self.d_ii.coords[M]) 


                    #dA_J = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dL]-self.d_ii.coords[M]) 
                    #dB_J = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM]-self.d_ii.coords[M])

                    

                    #if dM_i is not None:
                    try:

                        self.t2[:, dL, :, M, :, dM_i, :] = t2new_full[:, dL, :, M, :, dM]
                        
                    except:
                        
                        miss += 1
        
        print("Remapping (1):", time.time()-t0)
        t0 = time.time()


        # Compute the requested energy

        if energy == "fragment":
            # Fragment energy
            d_ii_1 = self.d_ii*1
            d_ii_1.blocks *= 0

            d_ii_1.blocks[d_ii_1.mapping[ d_ii_1._c2i(np.array([0,0,0]))], self.fragment[0], self.fragment] = 1

            i0_full_mask = np.array((d_ii_1.cget(ocoords)[:, self.fragment[0], :].ravel())[ii_mask], dtype = np.bool) # fragment 1 in refcell
            f0_mask = np.array((d_ii_1.cget([0,0,0])[self.fragment[0], :].ravel())[i0_mask], dtype = np.bool) #fragment 1 in refcell, only indexes in refcell




            t20 = t2s[f0_mask][:, :, i0_full_mask]
            v20 = v2s[f0_mask][:, :, i0_full_mask]
            

            energy = 2*np.einsum("iajb,iajb", t20, v20) - np.einsum("iajb,ibja", t20, v20)
            print("Energy (1):", time.time()-t0)
            t0 = time.time()
            
            return np.max(np.abs(t2new)), i, energy

        if energy == "cim": 
            # Cluster-in-molecule energy
            d_ii_1 = self.d_ii*1
            d_ii_1.blocks *= 0

            d_ii_1.blocks[d_ii_1.mapping[ d_ii_1._c2i(np.array([0,0,0]))], self.fragment[0], self.fragment] = 1

            f0_mask = np.array((d_ii_1.cget([0,0,0])[self.fragment[0], :].ravel())[i0_mask], dtype = np.bool) #fragment 1 in refcell, only indexes in refcell

            energy = 2*np.einsum("iajb,iajb", t2s[f0_mask], v2s[f0_mask]) - np.einsum("iajb,ibja", t2s[f0_mask], v2s[f0_mask])
            print(energy)
            print("Energy (1):", time.time()-t0)
            t0 = time.time()

            return np.max(np.abs(t2new)), i, energy


        if energy == "pair":
            # Pair-fragment energy
            d_ii_1 = self.d_ii*1
            d_ii_1.blocks *= 0

            d_ii_1.blocks[d_ii_1.mapping[ d_ii_1._c2i(np.array([0,0,0]))], self.f1.fragment[0], self.f1.fragment] = 1

            d_ii_2 = self.d_ii*1
            d_ii_2.blocks *= 0

            d_ii_2.blocks[d_ii_2.mapping[ d_ii_2._c2i(np.array([0,0,0]))], self.f2.fragment[0], self.f2.fragment] = 1


            mask_01 = np.array((d_ii_1.cget(ocoords)[:, self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) # fragment 1 in refcell
            mask_11 = np.array((d_ii_1.cget(ocoords-self.M)[:, self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) # fragment 1 in +1
            mask_1_1 = np.array((d_ii_1.cget(ocoords+self.M)[:, self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) #fragment 1 in -1
            mask_10 = np.array((d_ii_1.cget([0,0,0])[self.f1.fragment[0], :].ravel())[i0_mask], dtype = np.bool) #fragment 1 in refcell, only indexes in refcell

            mask_02 = np.array((d_ii_2.cget(ocoords)[:, self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)
            mask_22 = np.array((d_ii_2.cget(ocoords-self.M)[:, self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)
            mask_2_2 = np.array((d_ii_2.cget(ocoords+self.M)[:, self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)
            mask_20 = np.array((d_ii_2.cget([0,0,0])[self.f2.fragment[0], :].ravel())[i0_mask], dtype = np.bool)



            if True:
                energy_jj_11 = 2*np.einsum("iajb,iajb->j", t2s[mask_10], v2s[mask_10]) - np.einsum("iajb,ibja->j", t2s[mask_10], v2s[mask_10]) 
                energy_jj_22 = 2*np.einsum("iajb,iajb->j", t2s[mask_20], v2s[mask_20]) - np.einsum("iajb,ibja->j", t2s[mask_20], v2s[mask_20]) 
                #print("All energies:")
                #print(energy_jj_11)
                #print(energy_jj_22)

            energy_11 = 2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_11], v2s[mask_10][:, :, mask_11]) -  \
                          np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_11], v2s[mask_10][:, :, mask_11]) 
            
            energy_1_1 =2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_1_1], v2s[mask_10][:, :, mask_1_1]) -  \
                          np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_1_1], v2s[mask_10][:, :, mask_1_1])

            

            



            energy_12 = 2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_22], v2s[mask_10][:, :, mask_22]) -  \
                          np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_22], v2s[mask_10][:, :, mask_22]) 
            
            

            energy_2_1= 2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_1_1], v2s[mask_20][:, :, mask_1_1]) -  \
                          np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_1_1], v2s[mask_20][:, :, mask_1_1])

            


            energy_1_2 =2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_2_2], v2s[mask_10][:, :, mask_2_2]) -  \
                          np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_2_2], v2s[mask_10][:, :, mask_2_2])

            
            

            

            energy_21 = 2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_11], v2s[mask_20][:, :, mask_11]) -  \
                          np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_11], v2s[mask_20][:, :, mask_11]) 

            

            

            energy_22 = 2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_22], v2s[mask_20][:, :, mask_22]) -  \
                          np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_22], v2s[mask_20][:, :, mask_22]) 

            energy_2_2 =2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_2_2], v2s[mask_20][:, :, mask_2_2]) -  \
                          np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_2_2], v2s[mask_20][:, :, mask_2_2])


            print("Energies:")
            print(energy_11, energy_1_1) # first one
            print(energy_22, energy_2_2) #first one
            print(energy_12, energy_2_1) # first one ... energy_12 \approx 2*energy_2_1 
            print(energy_1_2, energy_21) # sedcond one

            

            #print("Energies:", energy_11, energy_1_1, energy_22, energy_2_2, energy_12, energy_2_1, energy_21,energy_1_2 )

            energy = np.array([energy_11, energy_22, energy_12, energy_21])
            #energy = np.array([energy_1_1, energy_2_2, energy_2_1, energy_1_2])
            
            
            #energy = np.array([energy_11+energy_1_1, energy_22+energy_2_2, energy_2_1+energy_12, energy_21+energy_1_2])

            # Counter-Poise
            #print(mask_10, mask_20, np.sum(mask_02), np.sum(mask_01))
            #energy_0101 = 2*np.einsum("iajb,iajb", t2s[mask_10][:, :, mask_01], v2s[mask_10][:, :, mask_01]) -  \
            #                np.einsum("iajb,ibja", t2s[mask_10][:, :, mask_01], v2s[mask_10][:, :, mask_01]) 

            #energy_0202 = 2*np.einsum("iajb,iajb", t2s[mask_20][:, :, mask_02], v2s[mask_20][:, :, mask_02]) -  \
            #                np.einsum("iajb,ibja", t2s[mask_20][:, :, mask_02], v2s[mask_20][:, :, mask_02]) 

            #print("Counter-Poise energies:", energy_0101, energy_0202)




            #print("Energy:", energy)
            print("Energy (1):", time.time()-t0)
            t0 = time.time()
            return np.max(np.abs(t2new)), i, energy


            #ii_mask = ii_mask.ravel()<self.occupied_cutoff   # active occupied indices





        return np.max(np.abs(t2new)), i

    

    def solve_completely_unfolded(self, norm_thresh = 1e-7, maxiter = 100, damping = 1.0, energy = None, compute_missing_exchange = True):
        # debug solver where tensors are completely unfolded (8 indices)
        t2 = self.t2               # amplitudes
        no = self.ib.n_occ         # number of occupieds per cell
        No = self.n_occupied_cells # numer of occupied cells
        nv = self.ib.n_virt        # number of virtuals per cell
        Nv = self.n_virtual_cells  # number of virtual cells

        t0 = time.time()

        vcoords = self.d_ia.coords[:self.n_virtual_cells]
        ocoords = self.d_ii.coords[:self.n_occupied_cells]

        ia_mask = self.d_ia.cget(vcoords)[:, self.fragment[0], :]
        ia_mask = ia_mask.ravel()<self.virtual_cutoff    # active virtual indices

        ii_mask = self.d_ii.cget(ocoords)[:, self.fragment[0], :]
        ii_mask = ii_mask.ravel()<self.occupied_cutoff    # active occupied indices

        Fii = self.f_mo_ii.tofull(self.f_mo_ii, ocoords, ocoords)[ii_mask][:, ii_mask]

        Faa = self.f_mo_aa.tofull(self.f_mo_aa, vcoords,vcoords)[ia_mask][:, ia_mask]

        t2_unfolded = np.zeros((No, no, Nv, nv, No, no, Nv, nv), dtype = float)
        g_unfolded =  np.zeros((No, no, Nv, nv, No, no, Nv, nv), dtype = float)

        print("Preparations (1):", time.time()-t0)
        t0 = time.time()



        for dL in np.arange(Nv):
            for M in np.arange(No):
                for dM in np.arange(Nv):
                    g_ = self.g_d[:, dL, :, M, :, dM]
                    for N in np.arange(No):
                        M_i = get_index_where(self.d_ii.coords[:self.n_occupied_cells], self.d_ii.coords[M]   + self.d_ii.coords[N])

                        dM_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM]  + self.d_ii.coords[N] +  self.d_ii.coords[M])

                        dL_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dL]  + self.d_ii.coords[N])
                        
                        if np.all(np.array([M_i, dM_i, dL_i])!=None):
                            


                            g_unfolded[N,:, dL_i, :, M_i, :, dM_i, :] = g_
                        else:
                            pass
                            #Iv = self.d_ii.coords[N]

                            #Av, Jv, Bv = self.d_ia.coords[dL]-Iv, self.d_ii.coords[M]-Iv, self.d_ii.coords[M] + self.d_ia.coords[dM] - Iv
                            
                            #g_unfolded[N,:, dL, :, M, :, dM, :] = self.ib.getcell_conventional(Av, Jv, Bv)



        print("Unfolding (1):", time.time()-t0)
        t0 = time.time()


        """
        for N in np.arange(No):
            for dL in np.arange(Nv):
                for M in np.arange(No):
                    for dM in np.arange(Nv):
                        M_i = get_index_where(self.d_ii.coords[:self.n_occupied_cells], self.d_ii.coords[M]   - self.d_ii.coords[N])

                        dM_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dM]  - self.d_ii.coords[N] +  self.d_ii.coords[M])

                        dL_i = get_index_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ia.coords[dL]  - self.d_ii.coords[N])

                        #I, Is = self.ib.getorientation(self.d_ia.coords[dL], self.d_ia.coords[dM] - self.d_ii.coords[M])

                        Iv = self.d_ii.coords[N]
                        Av, Jv, Bv = self.d_ia.coords[dL]-Iv, self.d_ii.coords[M]-Iv, self.d_ii.coords[M] + self.d_ia.coords[dM] - Iv
                        
                        g_unfolded[N,:, dL, :, M, :, dM, :] = self.ib.getcell_conventional(Av, Jv, Bv)



                        

                        #g_unfolded[N, :, dL, :, M, :, dM, : ] = I.#self.g_d[:, dL_i, :, M_i, :, dM_i]

                        
                        if np.any(np.array([M_i, dM_i, dL_i])==None):
                            pass
                        
                        else:
                            print(np.abs(self.g_d[:, dL_i, :, M_i, :, dM_i] - g_unfolded[N,:, dL, :, M, :, dM, :]).max())
                            
                            #g_unfolded[N, :, dL, :, M, :, dM, : ] = I.cget(self.d_ii.coords[M]).reshape(Is) #self.g_d[:, dL_i, :, M_i, :, dM_i]
                        

        """
        g = g_unfolded.reshape(No*no, Nv*nv, No*no, Nv*nv)[ii_mask][:, ia_mask][:, :, ii_mask][:, :, :, ia_mask]
        t2s = np.zeros_like(g)

        fii0 = np.diag(self.f_mo_ii.cget([0,0,0])).ravel()

        faa0 = np.diag(self.f_mo_aa.cget([0,0,0]))

        fii0_ = np.outer(np.ones(No, dtype = float), fii0).ravel()[ii_mask]

        #fii0 = np.diag(self.f_mo_ii.cget([0,0,0])).ravel()[i0_mask]

        faa0_ = np.outer(np.ones(Nv, dtype = float), faa0).ravel()[ia_mask]

        fiajb = fii0_[:,None,None,None] - faa0_[None,:,None,None] + fii0_[None,None,:,None] - faa0_[None,None,None,:]

        print("Preparations (2):", time.time()-t0)
        t0 = time.time()

        #print("diff:", np.abs(g - np.einsum("iajb->jbia", g)).max())



        # solve equations
        for i in np.arange(maxiter):
            t2new = -1*g

            # D1

            t2new -= np.einsum("icjb,ac->iajb", t2s, Faa) #*0

            # D2

            t2new -= np.einsum("iajc,bc->iajb", t2s, Faa)

            # D3 --------------
            t2new += 0*np.einsum("kajb,ki->iajb", t2s, Fii)


            # D4

            t2new += np.einsum("iakb,kj->iajb", t2s, Fii)
            
            t2s -= damping*t2new*(fiajb**-1)

            abs_dev = np.max(np.abs(t2new))
            #print(np.sum(t2new[0]**2))
            #print(t2s.shape, v2s.shape)

            

            

            if abs_dev<norm_thresh: #np.abs(norm_prev - norm_new)<norm_thresh:
                print("Converged at", i, "iterations.")
                break
            #else:
            #    pass
            #    norm_prev = norm_new*1
        
        print("Solution (0):", time.time()-t0)
        t0 = time.time()

        if energy == "fragment":
            d_ii_1 = self.d_ii*1
            d_ii_1.blocks *= 0
            d_ii_1.blocks[ d_ii_1.mapping[ d_ii_1._c2i([0,0,0])], self.fragment[0], self.fragment  ] = 1

            fragment_mask = np.array(d_ii_1.cget(ocoords)[:, self.fragment[0], :].ravel()[ii_mask], dtype = np.bool)

            

            





            #t20 = t2s[f0_mask][:, :, i0_full_mask]
            

            energy = 2*np.einsum("iajb,iajb->j", t2s[fragment_mask][:, :, fragment_mask], g[fragment_mask][:, :, fragment_mask]) - np.einsum("iajb,ibja->j", t2s[fragment_mask][:, :, fragment_mask], g[fragment_mask][:, :, fragment_mask])
            print("energy:", energy)
            print("Energy (1):", time.time()-t0)
            t0 = time.time()
            #return np.max(np.abs(t2new)), i
            return np.max(np.abs(t2new)), i, energy #[fragment_mask]


        if energy == "pair":
            d_ii_1 = self.d_ii*1
            d_ii_1.blocks *= 0

            d_ii_1.blocks[d_ii_1.mapping[ d_ii_1._c2i(np.array([0,0,0]))], self.f1.fragment[0], self.f1.fragment] = 1

            d_ii_2 = self.d_ii*1
            d_ii_2.blocks *= 0

            d_ii_2.blocks[d_ii_2.mapping[ d_ii_2._c2i(np.array([0,0,0]))], self.f2.fragment[0], self.f2.fragment] = 1


            #print(d_ii_1.cget(ocoords-self.M)[:, self.f1.fragment[0], :].ravel(), ii_mask)
            #print(d_ii_1.cget([0,0,0])[self.f1.fragment[0], :].ravel(), i0_mask)
            



            mask_01 = np.array((d_ii_1.cget(ocoords)[:, self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) # fragment 1 in refcell
            mask_11 = np.array((d_ii_1.cget(ocoords-self.M)[:, self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) # fragment 1 in +1
            mask_1_1 = np.array((d_ii_1.cget(ocoords+self.M)[:, self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) #fragment 1 in -1
            
            #mask_10 = np.array((d_ii_1.cget([0,0,0])[self.f1.fragment[0], :].ravel())[ii_mask], dtype = np.bool) #fragment 1 in refcell, only indexes in refcell

            mask_02 = np.array((d_ii_2.cget(ocoords)[:, self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)
            mask_22 = np.array((d_ii_2.cget(ocoords-self.M)[:, self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)
            mask_2_2 = np.array((d_ii_2.cget(ocoords+self.M)[:, self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)
            
            #mask_20 = np.array((d_ii_2.cget([0,0,0])[self.f2.fragment[0], :].ravel())[ii_mask], dtype = np.bool)



            
            energy_jj_11 = 2*np.einsum("iajb,iajb->j", t2s[mask_01], g[mask_01]) - np.einsum("iajb,ibja->j", t2s[mask_01], g[mask_01]) 
            energy_jj_22 = 2*np.einsum("iajb,iajb->j", t2s[mask_02], g[mask_02]) - np.einsum("iajb,ibja->j", t2s[mask_02], g[mask_02]) 
            print("All energies:")
            print(energy_jj_11)
            print(energy_jj_22)

            energy_11 = 2*np.einsum("iajb,iajb", t2s[mask_01][:, :, mask_11], g[mask_01][:, :, mask_11]) -  \
                          np.einsum("iajb,ibja", t2s[mask_01][:, :, mask_11], g[mask_01][:, :, mask_11]) 
            
            energy_1_1 =2*np.einsum("iajb,iajb", t2s[mask_01][:, :, mask_1_1], g[mask_01][:, :, mask_1_1]) -  \
                          np.einsum("iajb,ibja", t2s[mask_01][:, :, mask_1_1], g[mask_01][:, :, mask_1_1])

            

            



            energy_12 = 2*np.einsum("iajb,iajb", t2s[mask_01][:, :, mask_22], g[mask_01][:, :, mask_22]) -  \
                          np.einsum("iajb,ibja", t2s[mask_01][:, :, mask_22], g[mask_01][:, :, mask_22]) 
            
            

            energy_2_1= 2*np.einsum("iajb,iajb", t2s[mask_02][:, :, mask_1_1], g[mask_02][:, :, mask_1_1]) -  \
                          np.einsum("iajb,ibja", t2s[mask_02][:, :, mask_1_1], g[mask_02][:, :, mask_1_1])

            


            energy_1_2 =2*np.einsum("iajb,iajb", t2s[mask_01][:, :, mask_2_2], g[mask_01][:, :, mask_2_2]) -  \
                          np.einsum("iajb,ibja", t2s[mask_01][:, :, mask_2_2], g[mask_01][:, :, mask_2_2])

            
            

            

            energy_21 = 2*np.einsum("iajb,iajb", t2s[mask_02][:, :, mask_11], g[mask_02][:, :, mask_11]) -  \
                          np.einsum("iajb,ibja", t2s[mask_02][:, :, mask_11], g[mask_02][:, :, mask_11]) 

            

            

            energy_22 = 2*np.einsum("iajb,iajb", t2s[mask_02][:, :, mask_22], g[mask_02][:, :, mask_22]) -  \
                          np.einsum("iajb,ibja", t2s[mask_02][:, :, mask_22], g[mask_02][:, :, mask_22]) 

            energy_2_2 =2*np.einsum("iajb,iajb", t2s[mask_02][:, :, mask_2_2], g[mask_02][:, :, mask_2_2]) -  \
                          np.einsum("iajb,ibja", t2s[mask_02][:, :, mask_2_2], g[mask_02][:, :, mask_2_2])


            print("Energies:")
            print(energy_11, energy_1_1) # first one
            print(energy_22, energy_2_2) #first one
            print(energy_12, energy_2_1) # first one ... energy_12 \approx 2*energy_2_1 
            print(energy_1_2, energy_21) # sedcond one

            print("Energy (1):", time.time()-t0)
            t0 = time.time()

            

            energy = 2*np.array([energy_11, energy_22, energy_12, energy_21])
            
            return np.max(np.abs(t2new)), i, energy










    


    def bfgs_solve(self, f, x0, N_alpha = 20, thresh = 1e-10):
        """
        Rough outline of bfgs solver
        f       = objective function
        x0      = initial state
        N_alpha = resolution of line search

        """

        xn = x0                                # Initial state
        df = autograd.grad(f)                  # Gradient of f, could be finite difference based
        df = lambda x : x[1] - x[0]
        B = np.eye(xn.shape[0], dtype = float) # Initial guess for approximate Hessian

        for i in np.arange(50):
            df_ = df(xn)

            dx = np.linalg.solve(B, - df(xn))

            # line search
            fj = f(xn) #initial energy
            for j in np.linspace(0,1,N_alpha)[1:]:
                fn = f(xn + j*dx) #update energy
                if fn>fj:
                    # if energy increase, break
                    break
                fj = fn

            xn = xn + j*dx # update state
            if np.abs(fj)<=thresh:
                # if energy close to zero, break
                break

            # Update approximate Hessian
            y = df(xn) - df_
            Bs = np.dot(B, xn)
            B += np.outer(y,y)/np.outer(xn,y).T  - np.outer(Bs, Bs)/np.outer(xn, Bs).T



        return xn

    def map_omega(self):
        No = self.n_occupied_cells
        Nv = self.n_virtual_cells
        self.t2_map = np.zeros((Nv, No, Nv), dtype = np.object)

        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

            
            
            #dL_i = (self.d_ia.cget(dLv)<self.virtual_cutoff)[:self.ib.n_occ, :]
            #dL_i[M0_i == False, :] = 0

            for M in np.arange(self.n_occupied_cells):
                Mv = self.d_ii.coords[M]
                M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask


                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask

                    #K_a = get_bvec_where(self.d_ii.coords[:self.n_occupied_cells], self.d_ii.coords[:self.n_occupied_cells])
                    #dLv_a = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ii.coords[:self.n_occupied_cells] + dLv - Mv)
                    #dMv_a = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ii.coords[:self.n_occupied_cells] + dMv)

                    K_a = get_bvec_where(self.d_ii.coords[:self.n_occupied_cells], self.d_ii.coords[:self.n_occupied_cells])
                    dLv_a = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ii.coords[:self.n_occupied_cells] + dLv - Mv)
                    dMv_a = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells], self.d_ii.coords[:self.n_occupied_cells] + dMv)






                    #dLv_a = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells], -1*self.d_ii.coords[:self.n_occupied_cells] + dLv )
                    #dMv_a = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells], -1*self.d_ii.coords[:self.n_occupied_cells] + dMv - Mv)


                    indx_a = np.all(np.array([K_a, dLv_a, dMv_a])>=0, axis = 0)

                    D3 = np.array([K_a, dLv_a, dMv_a, indx_a])






                    dMv_b = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells],  -1*self.d_ii.coords[:self.n_occupied_cells] + dMv + Mv)
                    K_b = np.arange(self.n_occupied_cells)
                    indx_b = np.all(np.array([K_b, dMv_b])>=0, axis = 0)

                    D4 = np.array([K_b, dMv_b, indx_b])
                    
                    self.t2_map[dL, M, dM] = np.array([D3, D4])





    
    def omega(self, t2):

        #t0 = time.time()

        t2_new = np.zeros_like(t2)
        #print("SHAPE:", t2_new.shape)

        M0_i = self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff
        #print(M0_i)
        #print("M0_i;", M0_i.shape)

        #M0_i = self.d_ii.cget([0,0,0])<self.occupied_cutoff
        #print("self.n_occupied_cells:", self.n_occupied_cells)
        #print("self.n_virtual_cells:", self.n_virtual_cells)

        #print("d_ii.coords:", self.d_ii.coords[:self.n_occupied_cells])
        #print("d_ia.coords:", self.d_ia.coords[:self.n_virtual_cells])

        tm1 = 0
        tm2 = 0
        tm3 = 0
        tm4 = 0
        tm5 = 0

        Nv = self.n_virtual_cells
        No = self.n_occupied_cells
        nv = self.ib.n_virt
        no = self.ib.n_occ




        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

            
            
            #dL_i = (self.d_ia.cget(dLv)<self.virtual_cutoff)[:self.ib.n_occ, :]
            #dL_i[M0_i == False, :] = 0

            for M in np.arange(self.n_occupied_cells):
                Mv = self.d_ii.coords[M]
                M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask


                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                    
                    #dM_i = (self.d_ia.cget(dMv)<self.virtual_cutoff)[:self.ib.n_occ, :]# dM index mask
                    #dM_i[M_i == False, :] = 0

                    




                    
                    
                    
                    
                    #M_i = self.d_ii.cget(Mv)<self.occupied_cutoff # M index mask

                    tnew = -self.g_d[:, dL, :, M, :, dM, :]

                    #print(self.g_d[:, dL, :, M, :, dM, :][M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i])

                    tt = time.time() #TImE
                    cell_map = np.arange(tnew.size).reshape(tnew.shape)[M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()
                    tm3 += time.time()-tt #TIME



                    # D1
                    tt = time.time() #TImE
                    F_ac = self.f_mo_aa.cget(self.d_ia.coords[:self.n_virtual_cells] - dLv)
                    tm3 += time.time()-tt #TIME

                    tt = time.time() #TImE
                    tnew -= np.einsum("iCcjb,Cac->iajb", t2[:, :, :, M, :, dM, :], F_ac)
                    tm1 += time.time()-tt #TIME


                    # D2

                    tt = time.time() #TImE
                    F_bc = self.f_mo_aa.cget(self.d_ia.coords[:self.n_virtual_cells] - dMv)
                    tm3 += time.time()-tt #TIME
                    #print("F_bc:", F_bc)

                    tt = time.time() #TImE
                    tnew -= np.einsum("iajCc,Cbc->iajb", t2[:, dL, :, M, :, :, :], F_bc)
                    tm1 += time.time()-tt #TIme




                    D3, D4 = self.t2_map[dL, M, dM] 

                    tt = time.time() #TImE

                    K_, dLv_, dMv_, indx = D3
                    indx = np.array(indx, dtype = np.bool)

                    #print("F_ii:", self.f_mo_ii.cget(self.d_ii.coords[:self.n_occupied_cells]-Mv))
                    #print("D3:", D3)
                    #print("D4:", D4)

                    # D3


                    No_ = np.sum(indx)
                    if np.any(indx):
                        tt = time.time() #TImE

                        tnew += np.einsum("Kiakb,Kkj->iajb",t2[:, dLv_[indx], :, K_[indx], :, dMv_[indx], :], self.f_mo_ii.cget(self.d_ii.coords[:self.n_occupied_cells]-Mv)[indx])
                        tm4 += time.time()-tt #TIme

                        # dot-implementation
                        # tt = time.time()
                        # F_kj = self.f_mo_ii.cget(-1*self.d_ii.coords[:self.n_occupied_cells]-Mv)[indx]
                        # t2_ = t2[:, dLv_[indx], :, K_[indx], :, dMv_[indx], :].swapaxes(3,4).swapaxes(0,3) #Kiakb -> biaKk
                        # t2F = np.dot(t2_.reshape([nv*no*nv, No_*no]), F_kj.reshape([No_*no, no])).reshape([nv,no*nv*no]) # -> biaj
                        # tnew += t2F.swapaxes(0,1).reshape((no,nv,no,nv))
                        #tm5 += time.time() - tt









 
                    tt = time.time() #TImE

                    # D4

                    K_, dMv_, indx = D4
                    indx = np.array(indx, dtype = np.bool)

                    tm2 += time.time()-tt #TIme

                    if np.any(indx):
                        tt = time.time() #TImE
                        tnew += np.einsum("Kiakb,Kkj->iajb",t2[:, dL, :, K_[indx], :, dMv_[indx], :], self.f_mo_ii.cget(-1*self.d_ii.coords[:self.n_occupied_cells]+Mv)[indx])
                        tm1 += time.time()-tt #TIme



                    tt = time.time() #TImE
                    t2_mapped = np.zeros_like(tnew).ravel()
                    
                    t2_mapped[cell_map] = (tnew*self.e_iajb**-1).ravel()[cell_map]

                    #print("self.e_iajb:", self.e_iajb[M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i])
                    
                    #t2_mapped[cell_map] = .1*tnew.ravel()[cell_map]
                    #t2_mapped = (tnew*self.e_iajb**-1).ravel()




                    t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
                    tm2 += time.time()-tt #TIme
        #print("Cellmap omitted.")
        #print(tm1, tm2, tm3, tm4, tm5)
        return t2_new

    def omega_working(self, t2):

        #t0 = time.time()

        t2_new = np.zeros_like(t2)
        #print("SHAPE:", t2_new.shape)

        M0_i = self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff
        #print(M0_i)
        #print("M0_i;", M0_i.shape)

        #M0_i = self.d_ii.cget([0,0,0])<self.occupied_cutoff
        #print("self.n_occupied_cells:", self.n_occupied_cells)
        #print("self.n_virtual_cells:", self.n_virtual_cells)

        #print("d_ii.coords:", self.d_ii.coords[:self.n_occupied_cells])
        #print("d_ia.coords:", self.d_ia.coords[:self.n_virtual_cells])





        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

            
            
            #dL_i = (self.d_ia.cget(dLv)<self.virtual_cutoff)[:self.ib.n_occ, :]
            #dL_i[M0_i == False, :] = 0

            for M in np.arange(self.n_occupied_cells):
                Mv = self.d_ii.coords[M]
                M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask


                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                    
                    #dM_i = (self.d_ia.cget(dMv)<self.virtual_cutoff)[:self.ib.n_occ, :]# dM index mask
                    #dM_i[M_i == False, :] = 0




                    
                    
                    
                    
                    #M_i = self.d_ii.cget(Mv)<self.occupied_cutoff # M index mask

                    tnew = -self.g_d[:, dL, :, M, :, dM, :]

                    cell_map = np.arange(tnew.size).reshape(tnew.shape)[M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                    #print("Residual:", dLv,Mv, dMv)



                    for C in np.arange(self.n_virtual_cells):

                        dCv = self.d_ia.coords[C]

                        # (1)

                        # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                        tnew -= np.einsum("icjb,ac->iajb", t2[:, C, :, M, :, dM, :], self.f_mo_aa.cget(dCv - dLv))
                        #tnew -= np.einsum("icjb,ac->iajb", t2[:, C, :, M, :, dM, :], self.f_mo_aa.cget(dCv))


                        #print("DA(1)(", dCv-dLv, ",", Mv, ",", dMv, ") * F(", dCv- dLv -Mv -dMv, ") + ")




                        #Fbc = self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dM]) # -Mv

                        # (2)
                        
                        tnew -= np.einsum("iajc,bc->iajb", t2[:, dL, :, M, :, C, :], self.f_mo_aa.cget(dCv - Mv - dMv))
                        #tnew -= np.einsum("iajc,bc->iajb", t2[:, dL, :, M, :, C, :], self.f_mo_aa.cget(dCv))
                        #print("                                              DA(2)(", dCv-dLv, ",", Mv, ",", dMv, ") * F(", dCv- dLv +Mv -dMv, ") + ")
                    #print(" .. ")


                    
                    
                    for K in np.arange(self.n_occupied_cells):
                        Kv = self.d_ii.coords[K]

                        
                        

                        
                        # (3)

                        # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                        #try:
                        K_ = get_index_where(self.d_ii.coords, -Kv)
                        dLv_ = get_index_where(self.d_ia.coords, dLv-Kv-Mv)
                        dMv_ = get_index_where(self.d_ia.coords, dMv-Kv)


                        #if np.all([K_<self.n_occupied_cells,dLv_<self.n_virtual_cells, dMv_<self.n_virtual_cells ]):
                        #if K_<self.n_occupied_cells:
                        #if np.all([dLv_<self.n_virtual_cells, dMv_<self.n_virtual_cells ]):
                        
                        if np.all([K_<self.n_occupied_cells,dLv_<self.n_virtual_cells, dMv_<self.n_virtual_cells ]):
                            tnew += np.einsum("iakb,kj->iajb",t2[:, dLv_, :, K_, :, dMv_, :], self.f_mo_ii.cget(-Kv-Mv))
                        #except:
                        #    #print("Neglected index:", Kv)
                        #    pass

                        # (4)

                        

                        # - \sum_{L' k} \left(t^{\Delta L a, \Delta Mb}_{0i,L'k}\right)_{n} f_{0 k M-L'j}

                        dM_K = get_index_where(self.d_ia.coords, dMv - Kv + Mv)
                        if dM_K<self.n_virtual_cells:

                            tnew += np.einsum("iakb,kj->iajb",t2[:, dL, :, K, :, dM_K, :], self.f_mo_ii.cget(Mv-Kv))

                    






                        





                    # Conceptually simpler approach
                    """
                    # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                    Fki = self.f_mo_ii.cget(-1*self.pair_extent)

                    M_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[M] - self.pair_extent ) ]
                    dL_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dL] - self.pair_extent) ]
                    #dL_M = self.d_ia.mapping[self.d_ia._c2i(  - self.pair_extent) ]
                    
                    dM_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dM] - self.pair_extent) ]
                    #make sure indices is in correct domain (not negative or beyond extent)
                    nz = (M_range<self.n_occupied_cells)*(dL_M<self.n_virtual_cells)*(dM_M<self.n_virtual_cells)*\
                            (M_range>=0)*(dL_M>=0)*(dM_M>=0)

                    tnew += np.einsum("Kkajb,Kki->iajb",t2[:, dL_M[nz], :, M_range[nz], :, dM_M[nz], :], Fki[nz])*0
                    """



                    

                    t2_mapped = np.zeros_like(tnew).ravel()
                    
                    t2_mapped[cell_map] = (tnew*self.e_iajb**-1).ravel()[cell_map]
                    #t2_mapped[cell_map] = .1*tnew.ravel()[cell_map]
                    #t2_mapped = (tnew*self.e_iajb**-1).ravel()




                    t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
        #print("Cellmap omitted.")
        return t2_new


    def omega_old(self, t2):

        #t0 = time.time()

        t2_new = np.zeros_like(t2)
        #print("SHAPE:", t2_new.shape)

        M0_i = self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff
        #print(M0_i)
        #print("M0_i;", M0_i.shape)

        #M0_i = self.d_ii.cget([0,0,0])<self.occupied_cutoff





        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

            
            
            #dL_i = (self.d_ia.cget(dLv)<self.virtual_cutoff)[:self.ib.n_occ, :]
            #dL_i[M0_i == False, :] = 0

            for M in np.arange(self.n_occupied_cells):
                Mv = self.d_ii.coords[M]
                M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask


                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                    
                    #dM_i = (self.d_ia.cget(dMv)<self.virtual_cutoff)[:self.ib.n_occ, :]# dM index mask
                    #dM_i[M_i == False, :] = 0


                    
                    
                    
                    
                    #M_i = self.d_ii.cget(Mv)<self.occupied_cutoff # M index mask

                    tnew = -self.g_d[:, dL, :, M, :, dM, :]

                    # generate index mapping of non-zero amplitudes in cell
                    cell_map = np.arange(tnew.size).reshape(tnew.shape)[M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()


                    
                    #indx = np.outer(dL_i.ravel(), dM_i.ravel()).reshape((self.ib.n_occ, self.ib.n_virt, self.ib.n_occ, self.ib.n_virt))
                    #cell_map = np.arange(tnew.size)[indx.ravel()] 




                    # Perform contractions

                    # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                    Fac = self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dL]) # + Mv
                    tnew -= np.einsum("iKcjb,Kac->iajb", t2[:, :, :, M, :, dM, :], Fac)

                    # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                    Fbc = self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dM]) # -Mv
                    tnew -= np.einsum("iajKc,Kbc->iajb", t2[:, dL, :, M, :, :, :], Fbc)




                    # Conceptually simpler approach
                    # - \sum_{L' k} \left(t^{\Delta L - L'a, \Delta M-L' b}_{0k,M-L'j}\right)_{n} f_{0 k -L' i}
                    Fki = self.f_mo_ii.cget(-1*self.pair_extent)

                    M_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[M] - self.pair_extent ) ]
                    dL_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dL] - self.pair_extent) ]
                    #dL_M = self.d_ia.mapping[self.d_ia._c2i(  - self.pair_extent) ]
                    
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
                    #t2_mapped[cell_map] = .1*tnew.ravel()[cell_map]
                    #t2_mapped = (tnew*self.e_iajb**-1).ravel()




                    t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
        return t2_new

    def omega_(self, t2):

        t2_new = np.zeros_like(t2)
        #print("SHAPE:", t2_new.shape)

        M0_i = self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff
        #print("M0_i;", M0_i.shape)

        #M0_i = self.d_ii.cget([0,0,0])<self.occupied_cutoff

        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

            #dL_i = self.d_ia.cget(dLv) #<self.virtual_cutoff 
            #dL_i[np.not(M0_i)] = 0

            #d_ia = np.outer(M0_i, dL_i).ravel()


            for dM in np.arange(self.n_virtual_cells):
                dMv = self.d_ia.coords[dM]
                dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                #dM_i = self.d_ia.cget(dMv)<self.virtual_cutoff # dM index mask
                for M in np.arange(self.n_occupied_cells):
                    Mv = self.d_ii.coords[M]
                    M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                    
                    #dM_i = self.d_ia.cget(dMv)[np.arange(self.ib.n_occ)[M_i],:]<self.virtual_cutoff
                    
                    
                    
                    
                    #M_i = self.d_ii.cget(Mv)<self.occupied_cutoff # M index mask

                    tnew = -self.g_d[:, dL, :, M, :, dM, :]

                    # generate index mapping of non-zero amplitudes in cell
                    cell_map = np.arange(tnew.size).reshape(tnew.shape)[M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()


                    #indx = np.outer(d_ia, np.outer(M_i, dM_i).ravel()).reshape((self.ib.n_occ, self.ib.n_virt, self.ib.n_occ, self.ib.n_virt))
                    #cell_map = np.arange(tnew.size)[indx.ravel()] 




                    # Perform contractions

                    # + \sum_{\Delta L' c} \left(t^{\Delta L' c, \Delta Mb}_{Li,Mj}\right)_{n} f_{\Delta L a \Delta L' c}
                    Fac = self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dL])
                    tnew -= np.einsum("iKcjb,Kac->iajb", t2[:, :, :, M, :, dM, :], Fac)

                    # + \sum_{\Delta L' c} \left(t^{\Delta L a, \Delta L'c}_{0i,Mj}\right)_{n} f_{\Delta M b \Delta L' c} \\
                    Fbc = self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dM])
                    tnew -= np.einsum("iajKc,Kbc->iajb", t2[:, dL, :, M, :, :, :], Fbc)




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
                    #t2_mapped[cell_map] = .1*tnew.ravel()[cell_map]
                    #t2_mapped = (tnew*self.e_iajb**-1).ravel()




                    t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
        return t2_new

    def omega_agrad(self, t2):
        """
        Automated differentiation (autograd) residual for non-canonical, ortogonal MP2
        """

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



    def solve_MP2(self, norm_thresh = 1e-7, maxiter = 100, damping = 1.0, ndiis = 8):
        """
        Converge fragment (AOS) amplitudes within occupied and virtual extents
        Note: DIIS removed from this solver
        """
        #from autograd import grad, elementwise_grad,jacobian

        nocc = self.ib.n_occ

        self.virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        self.pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.vp_indx = mapgen(self.virtual_extent, self.pair_extent)
        self.pp_indx = mapgen(self.pair_extent, self.pair_extent)

        #DIIS = diis(ndiis)
        
        t0 = time.time()

        self.map_omega()


        for ti in np.arange(maxiter):
            # compute residual

            dt2_new = self.omega(self.t2)

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
                self.t2 -= damping*dt2_new #/np.abs(dt2_new).max()
                #print("dt2_new.max():", np.abs(dt2_new).max())
                #self.t2 = DIIS.advance(self.t2,damping*dt2_new)



                #rnorm = np.linalg.norm(dt2_new)
                #print("%.10e %.10e" % (self.compute_fragment_energy(),update_max ))
                #print("%.10e %.10e" % (self.compute_fragment_energy(),np.abs(dt2_new).max() ))
                #if np.abs(dt2_new).max()<1e-7:
                #    break
                #print("Max update:", np.abs(dt2_new).max())
                #if np.abs(dt2_new).max() <


            if np.abs(dt2_new).max() <norm_thresh:
                """
                Convergence achieved
                """
                #rnorm = np.linalg.norm(dt2_new)

                #print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti,rnorm))
                #print("(Maximum absolute res. norm: %.4e)" % np.max(np.abs(dt2_new)))
                #print("")
                break
        if ti >= maxiter-1:
            print("\033[93m" + "WARNING"+ "\033[0m"+": Solver did not converge in %i iterations (abs.max.res: %.2e)." % (ti, np.abs(dt2_new).max()))

        t_t = time.time() - t0
        print("Time per iteration:", t_t/ti)

        self.max_abs_residual = np.max(np.abs(dt2_new))
        # print("Max.abs. deviation in residual post optimization is %.5e" % np.max(np.abs(dt2_new)))

        return self.max_abs_residual, ti #, self.compute_fragment_energy()

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

        nocc = self.ib.n_occ
        ener_old = self.compute_fragment_energy()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        self.s_pao1 = tp.get_identity_tmat(self.ib.n_virt)

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
            #print ('Iteration no.: ', ti)
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

                        #cell_map_bool = np.zeros(tnew.size, dtype = np.bool)
                        #cell_map_bool[cell_map_bool] = True
                        #cell_map_bool_full[ :, dL, :, M, :, dM, :] =  cell_map_bool[cell_map_bool].reshape(tnew.shape)

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
        nocc = self.ib.n_occ

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        self.s_pao1 = tp.get_identity_tmat(self.ib.n_virt)

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

        nocc = self.ib.n_occ
        ener_old = self.compute_fragment_energy()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        self.s_pao1 = tp.get_identity_tmat(self.ib.n_virt)

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
            #print ('Iteration no.: ', ti)
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


    def solve_MP2PAO(self, thresh = 1e-10, s_virt = None, n_diis=8, damping = 0.2, maxiter = 100):
        """
        Solving the MP2 equations for a non-orthogonal virtual space
        (see section 5.6 "The periodic MP2 equations for non-orthogonal virtual space (PAO)" in the notes)
        """


        alpha = 0.1
        ener_old = 0

        nocc = self.ib.n_occ
        ener_old = self.compute_fragment_energy()

        self.virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        self.pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        #self.s_pao1 = tp.get_identity_tmat(self.p.get_nvirt())

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
        s_aa = np.diag(self.s_pao.cget([0,0,0]))
        f_ij = f_ii[:,None] + f_ii[None,:]
        sfs    = np.einsum("a,ij,b->iajb",s_aa,f_ij,s_aa)
        fs_ab  = np.einsum("a,b,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        fs_ba  = np.einsum("b,a,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        f_iajb = sfs - fs_ab - fs_ba

        #print(f_iajb.max(), f_iajb.min(), np.abs(f_iajb).min())
        #print(fs_ab.max(), fs_ab.min(), np.abs(fs_ab).min())
        #print(fs_ba.max(), fs_ba.min(), np.abs(fs_ba).min())
        #print(sfs.max(), sfs.min(), np.abs(sfs).min())

        DIIS = diis(n_diis)

        #DDIIS = diis(n_diis)

        opt = True

        ener = [0,0]

        #self.t2 *= 0

        M0_i = self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff ###

        #print( self.d_ia.cget([0,0,0])[self.fragment[0],:]<self.virtual_cutoff)
        #print( self.d_ia.cget([0,0,0])[self.fragment[0],:])
        #print( self.virtual_cutoff)
        #print(M0_i)

        #Dt2_new = np.zeros_like(self.t2) #experiment

        for ti in np.arange(maxiter):

            t2_new = np.zeros_like(self.t2)
            t2_bar = np.zeros_like(self.t2)

            # construct tbar
            for dL in np.arange(self.n_virtual_cells):
                dLv = self.d_ia.coords[dL]
                dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                    for M in np.arange(self.n_occupied_cells):
                        Mv = self.d_ii.coords[M]
                        M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                        Sac = self.s_pao.cget( self.virtual_extent - self.virtual_extent[dL])
                        Sdb = self.s_pao.cget(-self.virtual_extent + self.virtual_extent[dM])





                        t2_bar[:, dL, :, M, :, dM, :] = np.einsum("Cac,iCcjDd,Ddb->iajb", Sac, self.t2[:, :, :, M, :, :, :], Sdb, optimize = True)



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
                        cell_map = np.arange(tnew.size).reshape(tnew.shape)[M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()






                        # Perform contractions

                        Fik = np.array(self.f_mo_ii.cget(-1*self.pair_extent))
                        Fkj = np.array(self.f_mo_ii.cget(self.pair_extent[M] -1*self.pair_extent))


                        Fac = self.f_mo_aa.cget(self.virtual_extent - self.virtual_extent[dL])
                        Fdb = self.f_mo_aa.cget(-self.virtual_extent + self.virtual_extent[dM])
                        Sac = self.s_pao.cget(self.virtual_extent - self.virtual_extent[dL])
                        Sdb = self.s_pao.cget(-self.virtual_extent + self.virtual_extent[dM])

                        # First term
                        tnew -= np.einsum("Cac,iCcjDd,Ddb->iajb", Sac, self.t2[:, :, :, M, :, :, :], Fdb, optimize = True)

                        # Second term
                        tnew -= np.einsum("Cac,iCcjDd,Ddb->iajb", Fac, self.t2[:, :, :, M, :, :, :], Sdb, optimize = True)



                        # Troublesome diagram
                        M_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[M] - self.pair_extent ) ]
                        dL_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dL] - self.pair_extent) ]
                        dM_M = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[dM] - self.pair_extent) ]
                        # Make sure indices is in correct domain (not negative or beyond extent)
                        nz = (M_range<self.n_occupied_cells)*(dL_M<self.n_virtual_cells)*(dM_M<self.n_virtual_cells)*\
                                (M_range>=0)*(dL_M>=0)*(dM_M>=0)



                        tnew += np.einsum("Kkajb,Kik->iajb",t2_bar[:, dL_M[nz], :, M_range[nz], :, dM_M[nz], :], Fik[nz])


                        # Final diagram

                        Fkj = np.array(self.f_mo_ii.cget(-1*self.pair_extent + self.pair_extent[M]))
                        tnew += np.einsum("iaKkb,Kkj->iajb",t2_bar[:, dL, :, :, :, dM, :], Fkj)


                        # Final diagram
                        #Fkj = np.array(self.f_mo_ii.cget(1*self.pair_extent + self.pair_extent[M]))
                        #tnew += np.einsum("iaKkb,Kkj->iajb",t2_bar[:, dL, :, :, :, dM, :], Fkj)




                        t2_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*f_iajb**-1).ravel()[cell_map]
                        #t2_mapped[cell_map] = tnew.ravel()[cell_map]
                        #print(dLv, dMv, np.abs(np.max(tnew)))

                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)

            #t2 = time.time()

            #self.t2 = t2_new


            #Dt2_new = DDIIS.advance(Dt2_new, 0.1*(t2_new-Dt2_new))
            #self.t2 += damping*Dt2_new



            #self.t2 -= damping*t2_new

            self.t2 = DIIS.advance(self.t2, damping*t2_new)


            #t3 = time.time()

            #print ('time(beta):         ',t1-t0)
            #print ('time(resudial):     ',t2-t1)
            #print ('time(diis):         ',t3-t2)

            rnorm = np.linalg.norm(t2_new)
            dt_abs= np.abs(t2_new).max()
            ener = self.compute_fragment_energy()
            #print ('R norm: ',rnorm)
            print (ti, 'Energy: ',ener, dt_abs, rnorm)
            #ener.append(self.compute_fragment_energy())
            #print ('R norm: ',rnorm)
            #print (ti, 'Energy: ',ener[-1], "Diff:",  ener[-1]-ener[-2], "Res.norm:", rnorm, "Abs max norm:", np.abs(t2_new).max())
            #print ('dE: ',ener-ener_old)
            #ener_old = ener
            if dt_abs<thresh:
                print("Converged in %i iterations with abs.max.diff in residual %.2e." % (ti, dt_abs))
                print ()
                break
        if ti >= maxiter:
            print("\033[93m" + "WARNING"+ "\033[0m"+": Solver did not converge in %i iterations (abs.max.res: %.2e)." % (ti, dt_abs))
        return dt_abs, ti


    def solve_MP2PAO_(self, norm_thresh = 1e-10, s_virt = None, n_diis=8):
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

        nocc = self.ib.n_occ
        ener_old = self.compute_fragment_energy()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        self.s_pao1 = tp.get_identity_tmat(self.ib.n_virt)

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
        s_aa = np.diag(self.s_pao.cget([0,0,0]))
        f_ij = f_ii[:,None] + f_ii[None,:]
        sfs    = np.einsum("a,ij,b->iajb",s_aa,f_ij,s_aa)
        fs_ab  = np.einsum("a,b,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        fs_ba  = np.einsum("b,a,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        f_iajb = sfs - fs_ab - fs_ba

        DIIS = diis(n_diis)
        opt = True

        ener = [0,0]

        M0_i = self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff ###


        for ti in np.arange(1000):
            #print ('Iteration no.: ', ti)
            t2_new = np.zeros_like(self.t2)

            beta1 = np.zeros_like(self.t2)
            beta2 = np.zeros_like(self.t2)

            t0 = time.time()
            for C in np.arange(self.n_virtual_cells):
                for D in np.arange(self.n_virtual_cells):
                    for J in np.arange(self.n_occupied_cells):
                        Fkj = self.f_mo_ii.cget(-pair_extent + pair_extent[J])
                        beta1[:,C,:,J,:,D,:] = np.einsum("icKkd,Kkj->icjd",self.t2[:,C,:,:,:,D,:],Fkj,optimize=opt)

                        Fik = self.f_mo_ii.cget(pair_extent)

                        J_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[J] - pair_extent ) ]

                        C_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[C] - pair_extent) ]
                        D_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[D] - pair_extent) ]

                        nz = (J_range<self.n_occupied_cells)*(C_J<self.n_virtual_cells)*(D_J<self.n_virtual_cells)*\
                             (J_range>=0)*(C_J>=0)*(D_J>=0)

                        beta2[:,C,:,J,:,D,:] = np.einsum("Kkcjd,Kik->icjd",self.t2[:,C_J[nz],:,J_range[nz],:,D_J[nz],:],Fik[nz],optimize=opt)

            t1 = time.time()

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
                        cell_map = np.arange(tnew.size).reshape(tnew.shape)[M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                        # Perform contractions
                        Fac = self.f_mo_aa.cget(virtual_extent - virtual_extent[dL])
                        Fdb = self.f_mo_aa.cget(-virtual_extent + virtual_extent[dM])
                        Sac = self.s_pao.cget(virtual_extent - virtual_extent[dL])
                        Sdb = self.s_pao.cget(-virtual_extent + virtual_extent[dM])

                        #print ('dL: ', dL)
                        #print ('dM: ', dM)
                        #print ('M: ', M)


                        # \sum_{CcDd} f_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        tc1 = time.time()
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Fac, self.t2[:, :, :, M, :, :, :],optimize=opt)
                        tnew -= np.einsum("aijDd,Ddb->iajb", t_int, Sdb,optimize=opt)
                        tc2 = time.time()
                        #print ('Contr.time1: ',tc2-tc1)
                        #print ('NORM1: ',np.linalg.norm(np.einsum("aijDd,Ddb->iajb", t_int, Sdb,optimize=opt)))
                        tc3 = time.time()
                        #print (np.linalg.norm(tnew))
                        nC, na, nc = Fac.shape
                        #ts =
                        ni, nC, nc, nj, nD, nd = self.t2[:, :, :, M, :, :, :].shape
                        F_a_Cc = Fac.swapaxes(0,1).reshape((na, nC*nc)) # F(C,a,c) -> F(a,Cc)

                        t_Cc_ijDd = self.t2[:, :, :, M, :, :, :].swapaxes(0,1).swapaxes(1,2).reshape(nC*nc, ni*nj*nD*nd )# t2(i,C,c,j,D,d) -> t2(Cc, ijDd)

                        FT_a_ijDd = np.dot(F_a_Cc, t_Cc_ijDd)
                        FT_aij_Dd = FT_a_ijDd.reshape((na*ni*nj, nD*nd))

                        S_Dd_b = Sdb.reshape(nD*nd, nd)

                        FTS_i_a_j_b = np.dot(FT_aij_Dd, S_Dd_b).reshape((na,ni,nj,nd)).swapaxes(0,1)
                        tc4 = time.time()
                        #print ('Contr.time2: ',tc4-tc3)
                        #print ('NORM2: ',np.linalg.norm(FTS_i_a_j_b))





                        # \sum_{CcDd} s_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n f_{db}^{B-D}
                        t_int = np.einsum("Ddb,iCcjDd->iCcjb", Fdb, self.t2[:, :, :, M, :, :, :],optimize=opt)
                        tnew -= np.einsum("iCcjb,Cac->iajb", t_int, Sac,optimize=opt)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{jk}^{J-K} \left(t_{0i,Kk}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta1[:, :, :, M, :, :, :],optimize=opt)
                        tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb,optimize=opt)
                        #print (np.linalg.norm(tnew))

                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{ik}^{K} \left(t_{0k,J-Kk}^{C-Kc,D-Kd}\right)_n s_{db}^{B-D}
                        t_int = np.einsum("Cac,iCcjDd->aijDd", Sac, beta2[:, :, :, M, :, :, :],optimize=opt)
                        tnew += np.einsum("aijDd,Ddb->iajb", t_int, Sdb,optimize=opt)
                        #print (np.linalg.norm(tnew))


                        t2_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*f_iajb**-1).ravel()[cell_map]

                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)

            t2 = time.time()

            self.t2 -= 0.3*t2_new
            self.t2 = DIIS.advance(self.t2,t2_new)

            t3 = time.time()

            #print ('time(beta):         ',t1-t0)
            #print ('time(resudial):     ',t2-t1)
            #print ('time(diis):         ',t3-t2)

            rnorm = np.linalg.norm(t2_new)
            #ener = self.compute_fragment_energy()
            #print ('R norm: ',rnorm)
            #print ('Energy: ',ener)
            ener.append(self.compute_fragment_energy())
            #print ('R norm: ',rnorm)
            print (ti, 'Energy: ',ener[-1], "Diff:",  ener[-1]-ener[-2], "Res.norm:", rnorm)
            #print ('dE: ',ener-ener_old)
            #ener_old = ener
            if rnorm<norm_thresh:
                print ()
                print ('##############')
                print ('##############')
                print ('##############')
                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti, np.linalg.norm(t2_new)))
                print ()
                break

    def solve_MP2PAO_DOT(self, norm_thresh = 1e-10, s_virt = None, n_diis=8):
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

        nocc = self.ib.n_occ
        ener_old = self.compute_fragment_energy()

        virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.s_pao = s_virt
        self.s_pao1 = tp.get_identity_tmat(self.ib.n_virt)

        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))
        s_aa = np.diag(self.s_pao.cget([0,0,0]))
        f_ij = f_ii[:,None] + f_ii[None,:]
        sfs = np.einsum("a,ij,b->iajb",s_aa,f_ij,s_aa)
        fs_ab = np.einsum("a,b,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        fs_ba = np.einsum("b,a,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        f_iajb = sfs - fs_ab - fs_ba

        print("denom:", np.abs(f_iajb).min())

        DIIS = diis(n_diis)
        opt = True

        ener = [0,0]

        for ti in np.arange(1000):
            #print ('Iteration no.: ', ti)
            t2_new = np.zeros_like(self.t2)

            beta1 = np.zeros_like(self.t2)
            beta2 = np.zeros_like(self.t2)

            t0 = time.time()
            for C in np.arange(self.n_virtual_cells):
                for D in np.arange(self.n_virtual_cells):
                    for J in np.arange(self.n_occupied_cells):
                        Fkj = self.f_mo_ii.cget(-pair_extent + pair_extent[J])
                        Fik = self.f_mo_ii.cget(pair_extent)

                        ni, nc, nJ, nj, nd = self.t2[:,C,:,:,:,D,:].shape

                        T_icd_Kk = self.t2[:,C,:,:,:,D,:].swapaxes(3,4).swapaxes(2,3).reshape((ni*nc*nd,nJ*nj))
                        F_Kk_j = Fkj.reshape((nJ*nj,nj))
                        beta1[:,C,:,J,:,D,:] = np.dot(T_icd_Kk,F_Kk_j).reshape((ni,nc,nd,nj)).swapaxes(2,3)


                        J_range = self.d_ii.mapping[self.d_ii._c2i(  self.d_ii.coords[J] - pair_extent ) ]
                        C_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[C] - pair_extent) ]
                        D_J = self.d_ia.mapping[self.d_ia._c2i( self.d_ia.coords[D] - pair_extent) ]

                        nz = (J_range<self.n_occupied_cells)*(C_J<self.n_virtual_cells)*(D_J<self.n_virtual_cells)*\
                             (J_range>=0)*(C_J>=0)*(D_J>=0)

                        nI,ni,nc,nj,nd = self.t2[:,C_J[nz],:,J_range[nz],:,D_J[nz],:].shape

                        T_Kk_cjd = self.t2[:,C_J[nz],:,J_range[nz],:,D_J[nz],:].reshape((nI*ni,nc*nj*nd))
                        F_i_Kk = Fik[nz].swapaxes(0,1).reshape((ni,nI*ni))
                        beta2[:,C,:,J,:,D,:] = np.dot(F_i_Kk,T_Kk_cjd).reshape((ni,nc,nj,nd))

            t1 = time.time()

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

                        ni, nC, nc, nj, nD, nd = self.t2[:, :, :, M, :, :, :].shape
                        t_Cc_ijDd = self.t2[:, :, :, M, :, :, :].swapaxes(0,1).swapaxes(1,2).reshape(nC*nc, ni*nj*nD*nd )# t2(i,C,c,j,D,d) -> t2(Cc, ijDd)
                        S_a_Cc = Sac.swapaxes(0,1).reshape((nc, nC*nc)) # F(C,a,c) -> F(a,Cc)
                        S_Dd_b = Sdb.reshape(nD*nd, nd)


                        # \sum_{CcDd} f_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        F_a_Cc = Fac.swapaxes(0,1).reshape((nc, nC*nc)) # F(C,a,c) -> F(a,Cc)

                        FT_a_ijDd = np.dot(F_a_Cc, t_Cc_ijDd)
                        FT_aij_Dd = FT_a_ijDd.reshape((nc*ni*nj, nD*nd))

                        tnew -= np.dot(FT_aij_Dd, S_Dd_b).reshape((nc,ni,nj,nd)).swapaxes(0,1)


                        # \sum_{CcDd} s_{ac}^{C-A} \left(t_{0i,Jj}^{Cc,Dd}\right)_n f_{db}^{B-D}
                        F_Dd_b = Fdb.reshape(nD*nd, nd)

                        ST_a_ijDd = np.dot(S_a_Cc, t_Cc_ijDd)
                        ST_aij_Dd = ST_a_ijDd.reshape((nc*ni*nj, nD*nd))

                        tnew -= np.dot(ST_aij_Dd, F_Dd_b).reshape((nc,ni,nj,nd)).swapaxes(0,1)


                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{jk}^{J-K} \left(t_{0i,Kk}^{Cc,Dd}\right)_n s_{db}^{B-D}
                        B1_Cc_ijDd = beta1[:, :, :, M, :, :, :].swapaxes(0,1).swapaxes(1,2).reshape(nC*nc, ni*nj*nD*nd )# t2(i,C,c,j,D,d) -> t2(Cc, ijDd)

                        B1T_a_ijDd = np.dot(S_a_Cc, B1_Cc_ijDd)
                        B1T_aij_Dd = B1T_a_ijDd.reshape((nc*ni*nj, nD*nd))

                        tnew += np.dot(B1T_aij_Dd, S_Dd_b).reshape((nc,ni,nj,nd)).swapaxes(0,1)


                        # \sum_{CcDdKk} s_{ac}^{C-A} f_{ik}^{K} \left(t_{0k,J-Kk}^{C-Kc,D-Kd}\right)_n s_{db}^{B-D}
                        B2_Cc_ijDd = beta2[:, :, :, M, :, :, :].swapaxes(0,1).swapaxes(1,2).reshape(nC*nc, ni*nj*nD*nd )# t2(i,C,c,j,D,d) -> t2(Cc, ijDd)

                        B2T_a_ijDd = np.dot(S_a_Cc, B2_Cc_ijDd)
                        B2T_aij_Dd = B2T_a_ijDd.reshape((nc*ni*nj, nD*nd))

                        tnew += 0*np.dot(B2T_aij_Dd, S_Dd_b).reshape((nc,ni,nj,nd)).swapaxes(0,1)



                        t2_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*f_iajb**-1).ravel()[cell_map]

                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)

            t2 = time.time()

            self.t2 -= 0.3*t2_new
            self.t2 = DIIS.advance(self.t2,t2_new)

            t3 = time.time()

            #print ('time(beta):         ',t1-t0)
            #print ('time(resudial):     ',t2-t1)
            #print ('time(diis):         ',t3-t2)

            rnorm = np.linalg.norm(t2_new)

            ener.append(self.compute_fragment_energy())
            #print ('R norm: ',rnorm)
            print (ti, 'Energy: ',ener[-1], "Diff:",  ener[-1]-ener[-2])
            #print ('dE: ',ener-ener_old)
            #ener_old = ener
            if rnorm<norm_thresh:
                print ()
                print ('##############')
                print ('##############')
                print ('##############')
                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti, np.linalg.norm(t2_new)))
                print ()
                break

    def compute_mp2_density(self, orb_n = 0):
        """
        Compute fragment energy
        """
        N_virt = self.n_virtual_cells

        mM = 0 #occupied index only runs over fragment

        d_full = np.zeros(self.t2.shape, dtype = np.bool)

        occ_ind = np.zeros(self.ib.n_occ, dtype = np.bool)
        occ_ind[orb_n] = True


        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[self.fragment[0],:]<self.virtual_cutoff # dL index mask
            # Doublecount? dL == dM

            for ddM in np.arange(N_virt):

                    dM =  self.d_ia.coords[ddM]
                    dM_i = self.d_ia.cget(dM)[self.fragment[0],:]<self.virtual_cutoff # dM index mask


                    cell_map = np.arange(d_full[:,ddL,:,0, :, ddM, :].size).reshape(d_full[:,ddL,:,0, :, ddM, :].shape)[occ_ind][:, dL_i][:, :, occ_ind][:,:,:,dM_i].ravel()

                    # Using multiple levels of masking, probably some other syntax could make more sense

                    #g_direct = self.g_d[:,ddL,:,0, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]

                    #g_exchange = self.g_d[:,ddM,:,0, :, ddL, :][self.fragment][:, dM_i][:, :, self.fragment][:,:,:,dL_i]

                    #t = self.t2[:,ddL,:,0, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]


                    d_block_bool = np.zeros(d_full[:,ddL,:,0, :, ddM, :].shape, dtype = np.bool).ravel()
                    d_block_bool[cell_map] = True

                    #d_full[:,ddL,:,0, :, ddM, :][orb_n][:, dL_i][:, :, orb_n][:,:,:,dM_i] = True

                    d_full[:,ddL,:,0, :, ddM, :] = d_block_bool.reshape(d_full[:,ddL,:,0, :, ddM, :].shape)

                    #e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)

        return self.t2[d_full].reshape(self.n_virtual_tot, self.n_virtual_tot)


class fragment_amplitudes(amplitude_solver):
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
    def __init__(self, p, wannier_centers, coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 3.0, occupied_cutoff = 1.0, float_precision = np.float64, d_ia = None, store_exchange = False):
        self.p = p #prism object
        #self.d = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers) # distance matrix

        self.float_precision = float_precision
        self.store_exchange = store_exchange




        #self.d_ii = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[:p.get_nocc()])
        #self.d_ia = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[p.get_nocc():])
        #self.d_ii = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers[:ib.n_occ], sort_index = fragment[0])
        self.d_ii = dd.build_distance_matrix(p, coords, wannier_centers[:ib.n_occ], wannier_centers[:ib.n_occ], sort_index = fragment[0])

        if d_ia is None:
            self.d_ia = dd.build_distance_matrix(p, coords, wannier_centers[:ib.n_occ], wannier_centers[ib.n_occ:], sort_index = fragment[0])
        else:
            self.d_ia = d_ia



        self.fragment = fragment

        self.ib = ib #integral builder
        self.virtual_cutoff = virtual_cutoff
        self.occupied_cutoff = occupied_cutoff


        #self.min_elm = np.min(self.d.cget(self.d.coords), axis = (1,2)) #array for matrix-size calc



        self.min_elm_ii = np.min(self.d_ii.cget(self.d_ii.coords)[:,self.fragment[0], :], axis = 1)
        self.min_elm_ia = np.min(self.d_ia.cget(self.d_ia.coords)[:,self.fragment[0], :], axis = 1)


        self.f_mo_ii = f_mo_ii # Occupied MO-Fock matrix elements
        self.f_mo_aa = f_mo_aa # Virtual MO-Fock matrix elements



        self.init_amplitudes()

    def init_amplitudes_experimental(self):
        #experimental
        print("SAFE/CONVENTIONAL (slow) AMPLITUDE SETUP")
        """
        Initialize the amplitudes using the MP2-like starting guess
        """
        self.n_virtual_cells = np.sum(self.min_elm_ia<self.virtual_cutoff)
        self.n_occupied_cells = np.sum(self.min_elm_ii<self.occupied_cutoff)

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1,self.fragment[0]]<self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<self.occupied_cutoff)

        n_occ = self.ib.n_occ     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells # Number of occupied cells
        n_virt = self.ib.n_virt   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells # Number of virtual cells



        self.t2  = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_d = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)
        self.g_x = None
        if self.store_exchange:
            self.g_x = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)



        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))

        self.e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]


        self.e0 = 0

        sequence = []


        for A in np.arange(N_virt):
            for B in np.arange(N_virt):
                for J in np.arange(N_occ):
                    Av, Bv = self.d_ia.coords[A], self.d_ia.coords[B]
                    Jv = self.d_ii.coords[J]
                    self.g_d[:, A, :, J, :, B, :] = self.ib.getcell_conventional(Av, Jv, Bv)



                    
                    """
                    The following symmetry will be used here

                    I(dL, dM).cget(M)(ia,jb) = (0 i dL a |  M j M + dM b) = (-M i dL-M a | 0 j dM b)
                                             = (0 j dM b | -M i dL- M  a) = I(dM, dL).cget(-M)(jb,ia)

                    
                    """
                    #if True: #M[0]>=0: #?
                    # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                    #                ^                 ^           ^          ^
                    #            Calculate these    ex/direct    store here   1=transpose

                    
                    """
                    sequence.append([A, J, B,   0, ddL, mM, ddM,   0]) # direct




                    sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # direct

                    mM_ = get_index_where(self.d_ii.coords, -M)
                    #print(mM_)

                    if mM_<N_occ: # if the transpose is in the domain

                        sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,  1]) # direct, transposed
                    """
                    




        #self.initialize_blocks(sequence)

    def init_amplitudes(self):
        """
        Initialize the amplitudes using the MP2-like starting guess
        """
        self.n_virtual_cells = np.sum(self.min_elm_ia<self.virtual_cutoff)
        self.n_occupied_cells = np.sum(self.min_elm_ii<self.occupied_cutoff)

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1,self.fragment[0]]<self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<self.occupied_cutoff)


        n_occ = self.ib.n_occ     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells # Number of occupied cells
        n_virt = self.ib.n_virt   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells # Number of virtual cells



        self.t2  = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_d = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)
        self.g_x = None
        if self.store_exchange:
            self.g_x = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)



        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))

        self.e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]


        self.e0 = 0

        sequence = []

        v_map = np.ones((N_virt, N_occ, N_virt), dtype = np.bool) #avoid redundant calculations


        for ddL in np.arange(N_virt):
            for ddM in np.arange(ddL, N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]

                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]
                    """
                    The following symmetry will be used here

                    I(dL, dM).cget(M)(ia,jb) = (0 i dL a |  M j M + dM b) = (-M i dL-M a | 0 j dM b)
                                             = (0 j dM b | -M i dL- M  a) = I(dM, dL).cget(-M)(jb,ia)

                    Consequently:
                    (1) V^{dL, dM}_{ M } = I(dL, dM).cget( M ) 
                    (2) V^{dM, dL}_{-M } = I(dL, dM).cget( M ).T
                    (3) V^{dL, dM}_{-M } = I(dL, dM).cget(-M )
                    (4) V^{dM, dL}_{ M } = I(dL, dM).cget(-M ).T

                    Elements are added to the list 'sequence', where the following logic is used for
                    subsequent calculations: 

                    # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                    #                ^                 ^           ^          ^
                    #            Calculate these    ex/direct    store here   1=transpose

                    """

                    mM_ = get_index_where(self.d_ii.coords, -M) 

                    if v_map[ddL, mM, ddM]:
                        sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # (1)
                        v_map[ddL, mM, ddM] = False
                    
                    if v_map[ddM, mM, ddL]:
                        sequence.append([ddL, mM_, ddM,   0, ddM, mM , ddL,   1]) # (4)
                        v_map[ddM, mM, ddL] = False

                    if mM_<N_occ: # if the negative index is inside the domain

                        if v_map[ddM, mM_, ddL]:
                            sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,   1]) # (2)
                            v_map[ddM, mM_, ddL] = False
                        
                        if v_map[ddL, mM_, ddM]:
                            sequence.append([ddL, mM_, ddM,   0, ddL, mM_, ddM,   0]) # (3)
                            v_map[ddL, mM_, ddM] = False

                        

        print("occupied domain:", self.d_ii.coords[:N_occ])
        self.initialize_blocks(sequence)

    def initialize_blocks(self, sequence):
        sequence = np.array(sequence)
        print("sequuence shape:", sequence.shape)

        n_computed_di = 0
        n_computed_ex = 0

        # Sort blocks by dL:
        a = np.argsort(sequence[:,0])
        sequence = sequence[a]

        sequence = np.append(sequence, [ [-1000,0,0,0, 0, 0, 0,0] ], axis = 0) #-100 Just to make sure :-)

        j = 0
        for i in np.arange(len(sequence)):

            if sequence[i,0] != sequence[j,0]:
                a = np.argsort(sequence[j:i, 2])
                sq_i = sequence[j:i][a]


                sq_i = np.append(sq_i, [ [-1000,0,-1000,0, 0, 0, 0,0] ], axis = 0) #-100 Just to make sure :-)


                dL = self.d_ia.coords[sq_i[0,0]]



                k = 0
                for l in np.arange(len(sq_i)):


                    if sq_i[k,2] != sq_i[l,2]:
                        dM = self.d_ia.coords[sq_i[k,2]]

                        # Integrate here, loop over M
                        t0 = time.time()
                        I, Ishape = self.ib.getorientation(dL, dM)


                        for m in sq_i[k:l]:
                            M = self.d_ii.coords[m[1]]
                            ddL, mM, ddM = m[4], m[5], m[6]

                            if m[7] == 0:
                                if m[3] == 0:
                                    # Direct contribution
                                    self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)
                                    self.t2[:,  ddL, :, mM, :, ddM, :] = 0*I.cget(M).reshape(Ishape)*self.e_iajb**-1
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
                                    self.t2[:,  ddL, :, mM, :, ddM, :] = 0*I.cget(M).T.reshape(Ishape)*self.e_iajb**-1
                                    n_computed_di += 1
                                if m[3] == 1:
                                    # Exchange contribution
                                    #ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                    self.g_x[:, ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)
                                    n_computed_ex += 1
                        print("Computed RI-integrals ", dL, dM, " in %.2f seconds." % (time.time()-t0))
                        #, \
                        #        M, \
                        #        np.abs(I.cget(M)).max(), 
                        #        np.abs(self.ib.XregT[dL[0], dL[1], dL[2]].blocks).max() , \
                        #        np.abs(self.ib.XregT[dM[0], dM[1], dM[2]].blocks).max() )
                        #print(sq_i)


                        k = l*1

                j = i*1
        #print("n_computed_di:", n_computed_di)


    #def compute_energy_map(self, exchange = True):


    def compute_energy(self, exchange = True):
        """
        Computes the energy of entire AOS
        Disable exhange if sole purpose is to converge amplitude spaces
        """
        e_mp2 = 0
        N_virt = self.n_virtual_cells
        N_occ = self.n_occupied_cells

        M_0 = self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff # M index mask

        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[self.fragment[0],:]<self.virtual_cutoff # dL index mask
            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM]
                dM_i = self.d_ia.cget(dM)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]
                    #Mv = self.d_ii.coords[mM]
                    M_i = self.d_ii.cget(M)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                    #print(ddL,mM, ddM, self.g_d.shape)


                    g_direct = self.g_d[:,ddL,:,mM, :, ddM, :][M_0][:, dL_i][:, :, M_i][:,:,:,dM_i]
                    #g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :]
                    

                    ddM_M, ddL_M = get_index_where(self.d_ia.coords, dM+M), get_index_where(self.d_ia.coords, dL-M)
                    #g_exchange = self.g_d[:,ddM_M,:,mM, :, ddL_M, :][self.fragment][:, dM_i][:, :, M_i][:,:,:,dL_i]

                    t = self.t2[:,ddL,:,mM, :, ddM, :][M_0][:, dL_i][:, :, M_i][:,:,:,dM_i]

                    e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)

                    if exchange:

                        try:
                            # Get exchange index / np.argwhere
                            #assert(False)
                            

                            ddM_M, ddL_M = get_index_where(self.d_ia.coords, dM+M), get_index_where(self.d_ia.coords, dL-M)
                            g_exchange = self.g_d[:,ddM_M,:,mM, :, ddL_M, :][M_0][:, dM_i][:, :, M_i][:,:,:,dL_i]
                            #print("Reuse integrals for exchange")
                            #assert(False), "no"
                            #reuse += 1
                        except:

                            #print("Exchange not precomputed")
                            I, Ishape = self.ib.getorientation(dM+M, dL-M)
                            g_exchange = I.cget(M).reshape(Ishape)[M_0][:, dM_i][:, :, M_i][:,:,:,dL_i]
                            #computed += 1

                        e_mp2 += - np.einsum("iajb,ibja",t,g_exchange, optimize = True)







                    #e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)   #- np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        return e_mp2

    def compute_cim_energy(self, exchange = True):
        """
        Computes the CIM energy for a cluster defined by self.fragment
        Disable exhange if sole purpose is to converge amplitude spaces
        """
        e_mp2 = 0
        N_virt = self.n_virtual_cells
        N_occ = self.n_occupied_cells

        M_0 = self.fragment #self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff # M index mask

        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[self.fragment[0],:]<self.virtual_cutoff # dL index mask
            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM]
                dM_i = self.d_ia.cget(dM)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]
                    #Mv = self.d_ii.coords[mM]
                    M_i = self.d_ii.cget(M)[self.fragment[0],:]<self.occupied_cutoff # M index mask

                    #print(ddL,mM, ddM, self.g_d.shape)


                    g_direct = self.g_d[:,ddL,:,mM, :, ddM, :][M_0][:, dL_i][:, :, M_i][:,:,:,dM_i]
                    #g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :]

                    ddM_M, ddL_M = get_index_where(self.d_ia.coords, dM+M), get_index_where(self.d_ia.coords, dL-M)
                    #g_exchange = self.g_d[:,ddM_M,:,mM, :, ddL_M, :][self.fragment][:, dM_i][:, :, M_i][:,:,:,dL_i]

                    t = self.t2[:,ddL,:,mM, :, ddM, :][M_0][:, dL_i][:, :, M_i][:,:,:,dM_i]

                    e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)

                    if exchange:

                        try:
                            # Get exchange index / np.argwhere
                            #assert(False)
                            

                            ddM_M, ddL_M = get_index_where(self.d_ia.coords, dM+M), get_index_where(self.d_ia.coords, dL-M)
                            g_exchange = self.g_d[:,ddM_M,:,mM, :, ddL_M, :][M_0][:, dM_i][:, :, M_i][:,:,:,dL_i]
                            #print("Reuse integrals for exchange")
                            #assert(False), "no"
                            #reuse += 1
                        except:
                            if self.store_exchange:
                                if np.abs(self.g_x[:,ddL,:,mM, :, ddM, :]).max()>1e-10:
                                    g_exchange = self.g_x[:,ddL,:,mM, :, ddM, :][M_0][:, dM_i][:, :, M_i][:,:,:,dL_i]



                                else:
                                    I, Ishape = self.ib.getorientation(dM+M, dL-M)
                                    g_exchange = I.cget(M).reshape(Ishape) #
                                    
                                    self.g_x[:,ddL,:,mM, :, ddM, :] = g_exchange
                                    g_exchange = g_exchange[M_0][:, dM_i][:, :, M_i][:,:,:,dL_i]

                            
                            else:

                                #print("Exchange not precomputed")
                                I, Ishape = self.ib.getorientation(dM+M, dL-M)
                                g_exchange = I.cget(M).reshape(Ishape)[M_0][:, dM_i][:, :, M_i][:,:,:,dL_i]
                                #computed += 1

                        e_mp2 += - np.einsum("iajb,ibja",t,g_exchange, optimize = True)







                    #e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)   #- np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        return e_mp2

    def compute_eos_norm(self):
        """
        Compute eos norm
        """
        eos_norm = 0
        N_virt = self.n_virtual_cells

        mM = 0 #occupied index only runs over fragment

        #print("Self.fragment;:", self.fragment)

        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[self.fragment[0],:]<self.virtual_cutoff # dL index mask
            # Doublecount? dL == dM



            for ddM in np.arange(N_virt):

                    dM =  self.d_ia.coords[ddM]
                    dM_i = self.d_ia.cget(dM)[self.fragment[0],:]<self.virtual_cutoff # dM index mask

                    t = self.t2[:,ddL,:,0, :, ddM, :][self.fragment][:, dL_i][:, :, self.fragment][:,:,:,dM_i]

                    eos_norm += np.sum(t**2)
        return np.sqrt(eos_norm)


    def compute_fragment_energy(self):
        """
        Compute fragment energy
        """
        e_mp2 = 0
        e_mp2_direct = 0
        e_mp2_exchange = 0
        N_virt = self.n_virtual_cells

        mM = 0 #occupied index only runs over fragment

        #print("Self.fragment;:", self.fragment)

        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[self.fragment[0],:]<self.virtual_cutoff # dL index mask
            # Doublecount? dL == dM



            for ddM in np.arange(N_virt):

                    dM =  self.d_ia.coords[ddM]
                    dM_i = self.d_ia.cget(dM)[self.fragment[0],:]<self.virtual_cutoff # dM index mask

                    # Using multiple levels of masking, probably some other syntax could make more sense

                    #print(dM_i.shape, dL_i.shape, self.fragment)

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
                    #print(ddL, ddM, np.max(np.abs(t)), np.max(np.abs(g_direct)), np.max(np.abs(g_exchange)))
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
        #g_exchange = self.ib.getcell(dM+M, M, dL-M)

        self.t2[:, ddL, :, mmM, :, ddM, :]  = g_direct*self.e_iajb**-1
        self.g_d[:, ddL, :, mmM, :, ddM, :] = g_direct

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

        Nv = np.sum(self.min_elm_ia<self.virtual_cutoff)
        No = np.sum(self.min_elm_ii<self.occupied_cutoff)

        n_occ = self.ib.n_occ
        n_virt = self.ib.n_virt

        v_map = np.ones((Nv, No, Nv), dtype = np.bool) #avoid redundant calculations
        v_map[:self.n_virtual_cells,:self.n_occupied_cells,:self.n_virtual_cells] = False


        # Note: forking here is due to intended future implementation of block-specific initialization
        if Nv > self.n_virtual_cells:
            if No > self.n_occupied_cells:
                #print("Extending both occupied and virtuals")
                # Extend tensors in both occupied and virtual direction
                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                if self.store_exchange:
                    g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                    g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                    self.g_x = g_x_new

                


                #g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                #g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                #self.g_x = g_x_new

                # Initialize empty blocks
                sequence = []
                for ddL in np.arange(Nv):
                    dL = self.d_ia.coords[ddL]
                    for ddM in np.arange(ddL, Nv):
                        dM = self.d_ia.coords[ddM]
                        for mM in np.arange(No):
                            M = self.d_ii.coords[mM]
                            """
                            The following symmetry will be used here

                            I(dL, dM).cget(M)(ia,jb) = (0 i dL a |  M j M + dM b) = (-M i dL-M a | 0 j dM b)
                                                    = (0 j dM b | -M i dL- M  a) = I(dM, dL).cget(-M)(jb,ia)

                            Consequently:
                            (1) V^{dL, dM}_{ M } = I(dL, dM).cget( M ) 
                            (2) V^{dM, dL}_{-M } = I(dL, dM).cget( M ).T
                            (3) V^{dL, dM}_{-M } = I(dL, dM).cget(-M )
                            (4) V^{dM, dL}_{ M } = I(dL, dM).cget(-M ).T

                            Elements are added to the list 'sequence', where the following logic is used for
                            subsequent calculations: 

                            # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                            #                ^                 ^           ^          ^
                            #            Calculate these    ex/direct    store here   1=transpose

                            """

                            mM_ = get_index_where(self.d_ii.coords, -M) 

                            if v_map[ddL, mM, ddM]:
                                sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # (1)
                                v_map[ddL, mM, ddM] = False
                            
                            if v_map[ddM, mM, ddL]:
                                sequence.append([ddL, mM_, ddM,   0, ddM, mM , ddL,   1]) # (4)
                                v_map[ddM, mM, ddL] = False

                            if mM_<No: # if the negative index is inside the domain

                                if v_map[ddM, mM_, ddL]:
                                    sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,   1]) # (2)
                                    v_map[ddM, mM_, ddL] = False
                                
                                if v_map[ddL, mM_, ddM]:
                                    sequence.append([ddL, mM_, ddM,   0, ddL, mM_, ddM,   0]) # (3)
                                    v_map[ddL, mM_, ddM] = False




                            """

                            # Get exchange block coordinates
                            ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM - M) ] # dL to calculate
                            ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL + M) ]





                            # Negative M
                            mmM_ = get_index_where(self.d_ii.coords, -M)

                            # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                            #                ^                 ^           ^          ^
                            #            Calculate these    ex/direct    store here   1=transpose

                            sequence.append([ddL, mmM, ddM,   0, ddL, mmM, ddM,   0]) # direct

                            if mmM_<No: # if the transpose is in the domain
                                sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed
                            else:
                                if ddM!=ddL:
                                    # explicit calculation
                                    sequence.append([ddM, mmM, ddL,   0, ddM, mmM, ddL,  0])


                            #sequence.append([ddL, mmM, ddM,   0, ddL, mmM, ddM,   0]) # direct
                            #if mmM_<=No: # if the transpose is in the domain
                            #    sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed
                            """





                if len(sequence)>0:
                    self.initialize_blocks(sequence)







            else:
                #print("Extending virtuals.")
                # Extend tensors in the virtual direction

                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2[:, :, :, :No, :, :, :]
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d[:, :, :, :No, :, :, :]
                self.g_d = g_d_new

                if self.store_exchange:
                    g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                    g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x[:, :, :, :No, :, :, :]
                    self.g_x = g_x_new

                #g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                #g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                #self.g_x = g_x_new

                # Initialize empty blocks
                sequence = []
                for ddL in np.arange(Nv):
                    dL = self.d_ia.coords[ddL]
                    for ddM in np.arange(ddL, Nv):
                        dM = self.d_ia.coords[ddM]
                        for mM in np.arange(No):
                            M = self.d_ii.coords[mM]
                            """
                            The following symmetry will be used here

                            I(dL, dM).cget(M)(ia,jb) = (0 i dL a |  M j M + dM b) = (-M i dL-M a | 0 j dM b)
                                                    = (0 j dM b | -M i dL- M  a) = I(dM, dL).cget(-M)(jb,ia)

                            Consequently:
                            (1) V^{dL, dM}_{ M } = I(dL, dM).cget( M ) 
                            (2) V^{dM, dL}_{-M } = I(dL, dM).cget( M ).T
                            (3) V^{dL, dM}_{-M } = I(dL, dM).cget(-M )
                            (4) V^{dM, dL}_{ M } = I(dL, dM).cget(-M ).T

                            Elements are added to the list 'sequence', where the following logic is used for
                            subsequent calculations: 

                            # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                            #                ^                 ^           ^          ^
                            #            Calculate these    ex/direct    store here   1=transpose

                            """

                            mM_ = get_index_where(self.d_ii.coords, -M) 

                            if v_map[ddL, mM, ddM]:
                                sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # (1)
                                v_map[ddL, mM, ddM] = False
                            
                            if v_map[ddM, mM, ddL]:
                                sequence.append([ddL, mM_, ddM,   0, ddM, mM , ddL,   1]) # (4)
                                v_map[ddM, mM, ddL] = False

                            if mM_<No: # if the negative index is inside the domain

                                if v_map[ddM, mM_, ddL]:
                                    sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,   1]) # (2)
                                    v_map[ddM, mM_, ddL] = False
                                
                                if v_map[ddL, mM_, ddM]:
                                    sequence.append([ddL, mM_, ddM,   0, ddL, mM_, ddM,   0]) # (3)
                                    v_map[ddL, mM_, ddM] = False



                self.initialize_blocks(sequence)

        else:
            if No > self.n_occupied_cells:
                #print("extending occupied")
                # Extend tensors in the occupied dimension
                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                if self.store_exchange:
                    g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                    g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                    self.g_x = g_x_new

                #g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                #g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                #self.g_x = g_x_new

                # Initialize empty blocks
                sequence = []
                for ddL in np.arange(Nv):
                    dL = self.d_ia.coords[ddL]
                    for ddM in np.arange(ddL, Nv):
                        dM = self.d_ia.coords[ddM]
                        for mM in np.arange(No):
                            M = self.d_ii.coords[mM]
                            """
                            The following symmetry will be used here

                            I(dL, dM).cget(M)(ia,jb) = (0 i dL a |  M j M + dM b) = (-M i dL-M a | 0 j dM b)
                                                    = (0 j dM b | -M i dL- M  a) = I(dM, dL).cget(-M)(jb,ia)

                            Consequently:
                            (1) V^{dL, dM}_{ M } = I(dL, dM).cget( M ) 
                            (2) V^{dM, dL}_{-M } = I(dL, dM).cget( M ).T
                            (3) V^{dL, dM}_{-M } = I(dL, dM).cget(-M )
                            (4) V^{dM, dL}_{ M } = I(dL, dM).cget(-M ).T

                            Elements are added to the list 'sequence', where the following logic is used for
                            subsequent calculations: 

                            # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                            #                ^                 ^           ^          ^
                            #            Calculate these    ex/direct    store here   1=transpose

                            """

                            mM_ = get_index_where(self.d_ii.coords, -M) 

                            if v_map[ddL, mM, ddM]:
                                sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # (1)
                                v_map[ddL, mM, ddM] = False
                            
                            if v_map[ddM, mM, ddL]:
                                sequence.append([ddL, mM_, ddM,   0, ddM, mM , ddL,   1]) # (4)
                                v_map[ddM, mM, ddL] = False

                            if mM_<No: # if the negative index is inside the domain

                                if v_map[ddM, mM_, ddL]:
                                    sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,   1]) # (2)
                                    v_map[ddM, mM_, ddL] = False
                                
                                if v_map[ddL, mM_, ddM]:
                                    sequence.append([ddL, mM_, ddM,   0, ddL, mM_, ddM,   0]) # (3)
                                    v_map[ddL, mM_, ddM] = False


                if len(sequence)>0:
                    self.initialize_blocks(sequence)

            else:



                self.t2 = self.t2[:, :Nv, :, :No, :, :Nv, :]
                self.g_d = self.g_d[:, :Nv, :, :No, :, :Nv, :]
                if self.store_exchange:
                    self.g_x = self.g_x[:, :Nv, :, :No, :, :Nv, :]






        # Update domain measures
        self.n_virtual_cells = Nv
        self.n_occupied_cells = No

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1,self.fragment[0]]<self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<self.occupied_cutoff)

    def set_extent_(self, virtual_cutoff, occupied_cutoff):
        """
        Set extent of local domain
        Cutoffs are given in bohr, all orbitals within the specified cutoffs are included
        """
        self.virtual_cutoff = virtual_cutoff
        self.occupied_cutoff = occupied_cutoff

        Nv = np.sum(self.min_elm_ia<self.virtual_cutoff)
        No = np.sum(self.min_elm_ii<self.occupied_cutoff)

        n_occ = self.ib.n_occ
        n_virt = self.ib.n_virt

        # Note: forking here is due to intended future implementation of block-specific initialization
        if Nv > self.n_virtual_cells:
            if No > self.n_occupied_cells:
                #print("Extending both occupied and virtuals")
                # Extend tensors in both occupied and virtual direction
                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                if self.store_exchange:
                    g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                    g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                    self.g_x = g_x_new


                #g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                #g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                #self.g_x = g_x_new

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
                                mmM_ = get_index_where(self.d_ii.coords, -M)

                                # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                                #                ^                 ^           ^          ^
                                #            Calculate these    ex/direct    store here   1=transpose

                                sequence.append([ddL, mmM, ddM,   0, ddL, mmM, ddM,   0]) # direct
                                if mmM_<=No: # if the transpose is in the domain
                                    sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed





                if len(sequence)>0:
                    self.initialize_blocks(sequence)







            else:
                #print("Extending virtuals.")
                # Extend tensors in the virtual direction

                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2[:, :, :, :No, :, :, :]
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d[:, :, :, :No, :, :, :]
                self.g_d = g_d_new

                if self.store_exchange:
                    g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                    g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x[:, :, :, :No, :, :, :]
                    self.g_x = g_x_new

                #g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                #g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                #self.g_x = g_x_new

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
                                mmM_ = get_index_where(self.d_ii.coords, -M)

                                # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                                #                ^                 ^           ^          ^
                                #            Calculate these    ex/direct    store here   1=transpose

                                sequence.append([ddL, mmM, ddM,   0, ddL, mmM , ddM,  0]) # direct
                                if mmM_<No: # if the transpose is in the domain
                                    sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed



                self.initialize_blocks(sequence)

        else:
            if No > self.n_occupied_cells:
                #print("extending occupied")
                # Extend tensors in the occupied dimension
                t2new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                t2new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.t2
                self.t2 = t2new

                g_d_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                g_d_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_d
                self.g_d = g_d_new

                if self.store_exchange:
                    g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                    g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                    self.g_x = g_x_new

                #g_x_new = np.zeros((n_occ, Nv, n_virt, No, n_occ, Nv, n_virt), dtype = self.float_precision)
                #g_x_new[:, :self.n_virtual_cells, :, :self.n_occupied_cells, : , :self.n_virtual_cells, :] = self.g_x
                #self.g_x = g_x_new

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
                                mmM_ = get_index_where(self.d_ii.coords, -M)

                                # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                                #                ^                 ^           ^          ^
                                #            Calculate these    ex/direct    store here   1=transpose

                                sequence.append([ddL, mmM, ddM,   0, ddL, mmM, ddM,   0]) # direct
                                if mmM_<No: # if the transpose is in the domain
                                    sequence.append([ddL, mmM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed


                if len(sequence)>0:
                    self.initialize_blocks(sequence)

            else:



                self.t2 = self.t2[:, :Nv, :, :No, :, :Nv, :]
                self.g_d = self.g_d[:, :Nv, :, :No, :, :Nv, :]
                if self.store_exchange:
                    self.g_x = self.g_x[:, :Nv, :, :No, :, :Nv, :]






        # Update domain measures
        self.n_virtual_cells = Nv
        self.n_occupied_cells = No

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1,self.fragment[0]]<self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<self.occupied_cutoff)







class pair_fragment_amplitudes(amplitude_solver):
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
    def __init__(self, fragment_1, fragment_2, M, recycle_integrals = True, adaptive = False, retain_integrals = False, domain_def = 0, old_pair = None):
        import copy

        self.f1 = fragment_1
        self.f2 = fragment_2

        self.adaptive = adaptive

        self.recycle_integrals = recycle_integrals
        self.retain_integrals = retain_integrals # retain (adaptive) fitted integrals when new are computed

        self.p = self.f1.p #prism object
        #self.d = self.f1.d*1 #  dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers) # distance matrix
        self.M = M
        print("M:", M)

        self.float_precision = self.f1.float_precision

        # Set up occupied pair domain

        self.d_ii = copy.deepcopy(self.f1.d_ii) #dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers[:p.get_nocc()])
        
        for coord in self.d_ii.coords:
            #print(coord - M)
            elmn = self.f2.d_ii.cget(coord)[self.f2.fragment[0], :] < self.f2.occupied_cutoff
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord+M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99


            elmn = self.f1.d_ii.cget(coord)[self.f1.fragment[0], :] < self.f1.occupied_cutoff
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord+M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99




            """
            elmn = self.f2.d_ii.cget(coord)[self.f2.fragment[0], :] < self.f2.occupied_cutoff
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord+M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99
            if domain_def == 0:
                self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99


            elmn = self.f1.d_ii.cget(coord)[self.f1.fragment[0], :] < self.f1.occupied_cutoff
            if domain_def == 0:
                self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord+M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99
            """








            #self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99


            #self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99

            #elmn = self.f2.d_ii.cget(coord + M)[self.f2.fragment[0], :] < self.f2.occupied_cutoff

            #self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99

        self.d_ii.blocks[-1] = 2*self.f1.occupied_cutoff
        #sort in increasing order to avoid miscounting
        order = np.argsort( np.min(self.d_ii.cget(self.d_ii.coords)[:, self.f1.fragment[0], :], axis = 1) )
        #print(order)
        d_ii = tp.tmat()
        d_ii.load_nparray(self.d_ii.cget(self.d_ii.coords[order]), self.d_ii.coords[order] )
        #d_ii.load_nparray(self.d_ii.cget(self.d_ii.coords), self.d_ii.coords )
        #print(np.min(d_ii.cget(d_ii.coords)[:, self.f1.fragment[0], :], axis = 1))
        self.d_ii = d_ii
        self.d_ii.blocks[-1] = 2*self.f1.occupied_cutoff




        # Set index of f2 coordinate (for energy summations)
        self.mM = np.argwhere(np.sum((self.d_ii.coords-M)**2, axis = 1)==0)[0,0]
        self.mM_ = np.argwhere(np.sum((self.d_ii.coords+M)**2, axis = 1)==0)[0,0]


        # Set up virtual pair domain
        self.d_ia = copy.deepcopy(self.f1.d_ia)





        # Unite with translated virtual domain (set these distances to zero)
        for coord in self.d_ia.coords:
            """
            elmn = self.f2.d_ia.cget(coord)[self.f2.fragment[0], :] < self.f2.virtual_cutoff
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord+M) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99
            #if domain_def == 0:
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99

            elmn = self.f1.d_ia.cget(coord)[self.f1.fragment[0], :] < self.f1.virtual_cutoff
            #if domain_def == 0:
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord+M) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99
            """
           
            elmn = self.f2.d_ia.cget(coord)[self.f2.fragment[0], :] < self.f2.virtual_cutoff
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord+M) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99 # remove to reproduce cryscor
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99 #remove this one for smaller space

            elmn = self.f1.d_ia.cget(coord)[self.f1.fragment[0], :] < self.f1.virtual_cutoff
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord-M) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99 #remove this one for smaller space
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99 #superfluous?
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord+M) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99 # remove to reproduce cryscor


        self.d_ia.blocks[-1] = 1000
        #sort in increasing order to avoid miscounting
        order = np.argsort( np.min(self.d_ia.cget(self.d_ia.coords)[:, self.f1.fragment[0], :], axis = 1) )
        d_ia = tp.tmat()
        d_ia.load_nparray(self.d_ia.cget(self.d_ia.coords[order]), self.d_ia.coords[order] )

        self.d_ia = d_ia
        self.d_ia.blocks[-1] = 2*self.f1.virtual_cutoff



        self.fragment = self.f1.fragment*1



        self.ib = self.f1.ib #integral builder
        self.virtual_cutoff = self.f1.virtual_cutoff*1.0
        self.occupied_cutoff = self.f1.occupied_cutoff*1.0


        #self.min_elm = np.min(self.d.cget(self.d.coords), axis = (1,2)) #array for matrix-size calc

        self.min_elm_ii = np.min(self.d_ii.cget(self.d_ii.coords)[:,self.fragment[0], :], axis = 1)
        self.min_elm_ia = np.min(self.d_ia.cget(self.d_ia.coords)[:,self.fragment[0], :], axis = 1)


        self.f_mo_ii = self.f1.f_mo_ii # Occupied MO-Fock matrix elements
        self.f_mo_aa = self.f1.f_mo_aa # Virtual MO-Fock matrix elements



        self.init_amplitudes(old_pair) #Reuse old pair integrals if present

        

    def init_amplitudes(self, old_pair):
        """
        Initialize the amplitudes using the MP2-like starting guess
        """
        self.n_virtual_cells = np.sum(self.min_elm_ia<self.virtual_cutoff)
        self.n_occupied_cells = np.sum(self.min_elm_ii<self.occupied_cutoff)

        self.n_virtual_tot =  np.sum(self.d_ia.blocks[:-1, self.fragment[0]]<self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<self.occupied_cutoff)


        n_occ = self.ib.n_occ     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells # Number of occupied cells
        n_virt = self.ib.n_virt   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells # Number of virtual cells





        self.t2  = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_d = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_x = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision) #optional storage for exchange type integrals

        print("t2 shape:", self.t2.shape)

        reuse = 0    # count instances where coulomb integrals are recycled
        compute = 0  # count instances where coulomb integrals are computed

        if self.recycle_integrals:
            """
            Recycle integrals from fragment 1
            """

            for mM in np.arange(self.n_occupied_cells):
                for ddL in np.arange(self.n_virtual_cells):
                    for ddM in np.arange(self.n_virtual_cells):
                        ddLf1 =  get_index_where(self.f1.d_ia.coords, self.d_ia.coords[ddL])
                        ddMf1  =  get_index_where(self.f1.d_ia.coords, self.d_ia.coords[ddM])
                        mMf1   =  get_index_where(self.f1.d_ii.coords, self.d_ii.coords[mM])



                        if np.any(np.array([ddLf1, ddMf1, mMf1]) == None):
                            pass
                        else:
                            try:
                                #assert(False)
                                self.g_d[:,ddL,:,mM,:,ddM,:] = self.f1.g_d[:,ddLf1,:,mMf1,:,ddMf1,:]
                                reuse += 0
                            except:
                                compute += 0
                                #pass
            if old_pair is not None:
                for mM in np.arange(self.n_occupied_cells):
                    for ddL in np.arange(self.n_virtual_cells):
                        for ddM in np.arange(self.n_virtual_cells):
                            ddLf1 =  get_index_where(old_pair.d_ia.coords, self.d_ia.coords[ddL])
                            ddMf1  =  get_index_where(old_pair.d_ia.coords, self.d_ia.coords[ddM])
                            mMf1   =  get_index_where(old_pair.d_ii.coords, self.d_ii.coords[mM])


                            if np.any(np.array([ddLf1, ddMf1, mMf1]) == None):
                                pass
                            else:

                                try:
                                    #assert(False)
                                    self.g_d[:,ddL,:,mM,:,ddM,:] = old_pair.g_d[:,ddLf1,:,mMf1,:,ddMf1,:]
                                    reuse += 1
                                except:
                                    compute += 1
                                    #pass

                                try:
                                    self.g_x[:,ddL,:,mM,:,ddM,:] = old_pair.g_x[:,ddLf1,:,mMf1,:,ddMf1,:]
                                except:
                                    pass

        print("Compute, reuse:", compute, reuse)







        f_aa = np.diag(self.f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(self.f_mo_ii.cget([0,0,0]))

        self.e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]


        self.e0 = 0

        sequence = []


        v_map = np.ones((N_virt, N_occ, N_virt), dtype = np.bool)



        for ddL in np.arange(N_virt):
            """
            The following symmetry will be used here

            I(dL, dM).cget(M)(ia,jb) = (0 i dL a |  M j M + dM b) = (-M i dL-M a | 0 j dM b)
                                        = (0 j dM b | -M i dL- M  a) = I(dM, dL).cget(-M)(jb,ia)
            """
            for ddM in np.arange(ddL, N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]

                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]
                    if np.abs(self.g_d[:, ddL, :, mM, :, ddM, :]).max()<1e-14: #check if present
                        """
                        The following symmetry will be used here

                        I(dL, dM).cget(M)(ia,jb) = (0 i dL a |  M j M + dM b) = (-M i dL-M a | 0 j dM b)
                                                = (0 j dM b | -M i dL- M  a) = I(dM, dL).cget(-M)(jb,ia)

                        Consequently:
                        (1) V^{dL, dM}_{ M } = I(dL, dM).cget( M ) 
                        (2) V^{dM, dL}_{-M } = I(dL, dM).cget( M ).T
                        (3) V^{dL, dM}_{-M } = I(dL, dM).cget(-M )
                        (4) V^{dM, dL}_{ M } = I(dL, dM).cget(-M ).T

                        Elements are added to the list 'sequence', where the following logic is used for
                        subsequent calculations: 

                        # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                        #                ^                 ^           ^          ^
                        #            Calculate these    ex/direct    store here   1=transpose

                        """

                        mM_ = get_index_where(self.d_ii.coords, -M) 

                        if v_map[ddL, mM, ddM]:
                            sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # (1)
                            v_map[ddL, mM, ddM] = False
                        
                        if v_map[ddM, mM, ddL]:
                            sequence.append([ddL, mM_, ddM,   0, ddM, mM , ddL,   1]) # (4)
                            v_map[ddM, mM, ddL] = False

                        if mM_<N_occ: # if the negative index is inside the domain

                            if v_map[ddM, mM_, ddL]:
                                sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,   1]) # (2)
                                v_map[ddM, mM_, ddL] = False
                            
                            if v_map[ddL, mM_, ddM]:
                                sequence.append([ddL, mM_, ddM,   0, ddL, mM_, ddM,   0]) # (3)
                                v_map[ddL, mM_, ddM] = False
                        
                        
                        """
                        sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # direct
                        # Negative M
                        mmM_ = get_index_where(self.d_ii.coords, -M)

                        # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                        #                ^                 ^           ^          ^
                        #            Calculate these    ex/direct    store here   1=transpose


                        if mmM_<N_occ: # if the transpose is in the domain
                            sequence.append([ddL, mM, ddM,   0, ddM, mmM_, ddL,  1]) # direct, transposed
                        else:
                            #if ddM!=ddL:
                            #    # explicit calculation
                            sequence.append([ddM, mM, ddL,   0, ddM, mM, ddL,  0])

                        

                        mM_ = get_index_where(self.d_ii.coords, -M)

                        if mM_<N_occ: # if the transpose is in the domain
                            if np.abs(self.g_d[:, ddM, :, mM_, :, ddL, :]).max()<1e-14: #check if present

                                sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,  1]) # direct, transposed
                        """

        if len(sequence)>0:
            self.initialize_blocks(sequence)

    def initialize_blocks(self, sequence):
        sequence = np.array(sequence)

        n_computed_di = 0
        n_computed_ex = 0

        # Sort blocks by dL:
        try:
            a = np.argsort(sequence[:,0])
            sequence = sequence[a]
        except:
            print("Failed sequence sort:")
            print(sequence)
            print(sequence[:,0])


        sequence = np.append(sequence, [ [-1000,0,0,0, 0, 0, 0,0] ], axis = 0) #-100 Just to make sure :-)

        j = 0
        for i in np.arange(len(sequence)):

            if sequence[i,0] != sequence[j,0]:
                a = np.argsort(sequence[j:i, 2])
                sq_i = sequence[j:i][a]


                sq_i = np.append(sq_i, [ [-1000,0,-1000,0, 0, 0, 0,0] ], axis = 0) #-100 Just to make sure :-)


                dL = self.d_ia.coords[sq_i[0,0]]



                k = 0
                for l in np.arange(len(sq_i)):


                    if sq_i[k,2] != sq_i[l,2]:
                        dM = self.d_ia.coords[sq_i[k,2]]

                        # Integrate here, loop over M
                        t0 = time.time()

                        if self.adaptive:
                            di_indices = np.unique(np.array(sq_i[k:l])[:, 1])
                            I, Ishape = self.ib.get_adaptive(dL, dM,self.d_ii.coords[ di_indices ], keep = self.retain_integrals)
                        else:

                            I, Ishape = self.ib.getorientation(dL, dM)
                            #I, Ishape = self.ib.getorientation(dL*0, dM*0)
                            


                        for m in sq_i[k:l]:
                            M = self.d_ii.coords[m[1]]
                            ddL, mM, ddM = m[4], m[5], m[6]

                            if m[7] == 0:
                                if m[3] == 0:
                                    # Direct contribution
                                    #print(dL, M, dM)
                                    #print(ddL, mM, ddM, self.g_d.shape, "0")
                                    self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)
                                    #self.t2[:,  ddL, :, mM, :, ddM, :] = 0*I.cget(M).reshape(Ishape)*self.e_iajb**-1
                                    n_computed_di += 1
                                if m[3] == 1:
                                    # Exchange contribution
                                    #ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                    self.g_x[:, ddL, :, mM, :, ddM, :] = I.cget(M).reshape(Ishape)
                                    n_computed_ex += 1
                            if m[7] == 1:
                                if m[3] == 0:
                                    # Direct contribution
                                    #print(dL, M, dM)
                                    #print(self.d_ia.coords[ddL], self.d_ii.coords[mM],self.d_ia.coords[ddM])
                                    #print(ddL, mM, ddM, self.g_d.shape, "T")
                                    self.g_d[:, ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)
                                    #print("Transpose", ddL, mM, ddM)
                                    #self.t2[:,  ddL, :, mM, :, ddM, :] = 0*I.cget(M).T.reshape(Ishape)*self.e_iajb**-1
                                    n_computed_di += 1
                                if m[3] == 1:
                                    # Exchange contribution
                                    #ddL_, mM_, ddM_ = m[4], m[5], m[6]
                                    self.g_x[:, ddL, :, mM, :, ddM, :] = I.cget(M).T.reshape(Ishape)
                                    n_computed_ex += 1
                        #print("Computed RI-integrals ", dL, dM, " in %.2f seconds." % (time.time()-t0))
                        #print(sq_i)


                        k = l*1

                j = i*1

        # test g_d tensor


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

        #print("Self.fragment;:", self.fragment)

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
                    #print(ddL, ddM, np.max(np.abs(t)), np.max(np.abs(g_direct)), np.max(np.abs(g_exchange)))
                    e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)
        #print("E TOt", e_mp2_direct, e_mp2_exchange)
        return e_mp2

    def compute_pair_fragment_energy(self, opt = False):
        """
        Compute fragment energy
        """

        n_occ = self.ib.n_occ     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells # Number of occupied cells
        n_virt = self.ib.n_virt   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells # Number of virtual cells



        e_mp2_ab = 0 #_ab = 0
        e_mp2_ba = 0
        e_mp2_aa = 0 #_ab = 0
        e_mp2_bb = 0
        e_mp2_ab_ = 0 #_ab = 0
        e_mp2_ba_ = 0
        e_mp2_aa_ = 0 #_ab = 0
        e_mp2_bb_ = 0

        N_virt = self.n_virtual_cells


        reuse = 0     #count instances where exchange integrals can be recycled
        computed = 0  #count instances where exchange integrals are computed
        #print(self.f1.fragment)
        #print(self.f2.fragment)

        #R_1_accu = []
        #R_2_accu = []
        #EX_accu = []




        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[self.f1.fragment[0],:]<self.virtual_cutoff


            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM] # - self.M
                dM_i = self.d_ia.cget(dM)[self.f1.fragment[0],:]<self.virtual_cutoff
                if np.sum(dM_i)>0 and np.sum(dL_i)>0:
                    g_direct = self.g_d[:,ddL,:,self.mM, :, ddM, :] #[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]

                    if np.max(np.abs(self.g_x[:,ddL,:,self.mM, :, ddM, :]))>1e-14:
                        g_exchange = self.g_x[:,ddL,:,self.mM, :, ddM, :]

                    else:
                        try:
                            
                            # Get exchange index / np.argwhere
                            ddM_M, ddL_M = get_index_where(self.d_ia.coords, dM+self.M)[0], get_index_where(self.d_ia.coords, dL-self.M)[0]
                            #print(ddM_M, ddL_M)
                            g_exchange = self.g_d[:,ddM_M,:,self.mM, :, ddL_M, :] # [self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i]

                            reuse += 1
                        except:

                            
                            if self.adaptive:
                                I, Ishape = self.ib.get_adaptive(dM+self.M, dL-self.M, np.array([self.M]), keep = self.retain_integrals)
                            else:
                                I, Ishape = self.ib.getorientation(dM+self.M, dL-self.M)
                            g_exchange = I.cget(self.M).reshape(Ishape) #[self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i]
                            computed += 1
                        self.g_x[:,ddL,:,self.mM, :, ddM, :] = g_exchange

                    
                    #print(dM+self.M, dL-self.M, self.M, np.linalg.norm(g_exchange))

                    #r0 = dM+self.M
                    #r1 = dL-self.M +self.M
                    #R_ = np.sqrt(np.sum((r1-r0)**2))
                    #print(R_, np.max(np.abs(g_exchange)))

                    # self.d_ia.cget(dM + self.M).ravel()
                    # self.d_ia.cget(dL - self.M).ravel()
                    # self.d_ii.cget(self.M).ravel()
                    #R_1 = np.ones(g_exchange.shape, dtype = float)*(-self.d_ia.cget(dM + self.M))[:2,:, None, None]
                    #R_2 = np.ones(g_exchange.shape, dtype = float)*(-self.d_ia.cget(dL - self.M))[None, None, :2, :] #*self.d_ii.cget(self.M)[:, None]**-3
                    

                    #im = np.argmax(g_exchange.ravel())
                    
                    
                    #R_1_accu.append(R_1.ravel()[im])
                    #R_2_accu.append(R_2.ravel()[im])
                    #EX_accu.append(g_exchange.ravel()[im])


                    #r0 = self.p.coor2vec(dM+self.M)
                    #r1 = self.p.coor2vec(dL-self.M) #-self.M
                    #d0 = np.sqrt(np.sum(r0**2))
                    #d1 = np.sqrt(np.sum(r1**2))
                    #d2 = np.sqrt(np.sum(self.p.coor2vec(self.M)**2))
                    #R_ = np.exp(-d0)*np.exp(-d1)*d2**-3
                    #print(R_, np.max(np.abs(g_exchange)))
                    




                    t = self.t2[:,ddL,:,self.mM, :, ddM, :] #[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]

                    e_mp2_ab += 2*np.einsum("iajb,iajb",t[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        g_direct[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        optimize = opt) \
                                - np.einsum("iajb,ibja",t[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        g_exchange[self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i],
                                                        optimize = opt)


                    e_mp2_ba += 2*np.einsum("iajb,iajb",t[self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        g_direct[self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        optimize = opt) \
                                - np.einsum("iajb,ibja",t[self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        g_exchange[self.f2.fragment][:, dM_i][:, :, self.f1.fragment][:,:,:,dL_i],
                                                        optimize = opt)

                    e_mp2_aa += 2*np.einsum("iajb,iajb",t[self.f1.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        g_direct[self.f1.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        optimize = opt) \
                                - np.einsum("iajb,ibja",t[self.f1.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        g_exchange[self.f1.fragment][:, dM_i][:, :, self.f1.fragment][:,:,:,dL_i],
                                                        optimize = opt)

                    e_mp2_bb += 2*np.einsum("iajb,iajb",t[self.f2.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        g_direct[self.f2.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        optimize = opt) \
                                - np.einsum("iajb,ibja",t[self.f2.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        g_exchange[self.f2.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i],
                                                        optimize = opt)







                    # The opposite case

                    g_direct = self.g_d[:,ddL,:,self.mM_, :, ddM, :] #[self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i]


                    if np.max(np.abs(self.g_x[:,ddL,:,self.mM_, :, ddM, :]))>1e-14:
                        g_exchange = self.g_x[:,ddL,:,self.mM_, :, ddM, :]
                    else:
                        try:
                            # Get exchange index / np.argwhere
                            ddM_M, ddL_M = get_index_where(self.d_ia.coords, dM-self.M)[0], get_index_where(self.d_ia.coords, dL+self.M)[0]
                            g_exchange = self.g_d[:,ddM_M,:,self.mM_, :, ddL_M, :] # [self.f2.fragment][:, dM_i][:, :, self.f1.fragment][:,:,:,dL_i]
                            reuse += 1
                        except:
                            if self.adaptive:
                                I, Ishape = self.ib.get_adaptive( dM-self.M, dL+self.M, np.array([-self.M]), keep = self.retain_integrals)
                            else:
                                I, Ishape = self.ib.getorientation(dM-self.M, dL+self.M)
                            g_exchange = I.cget(-self.M).reshape(Ishape) # [self.f2.fragment][:, dM_i][:, :, self.f1.fragment][:,:,:,dL_i]
                            computed += 1
                        self.g_x[:,ddL,:,self.mM_, :, ddM, :] = g_exchange

                    #r0 = self.p.coor2vec(dM-self.M)
                    #r1 = self.p.coor2vec(dL+self.M) #-self.M
                    #d0 = np.sqrt(np.sum(r0**2))
                    #d1 = np.sqrt(np.sum(r1**2))
                    #d2 = np.sqrt(np.sum(self.p.coor2vec(self.M)**2))
                    #R_ = np.exp(-d0)*np.exp(-d1)*d2**-3

                    #R_ = np.exp(-self.d_ia.cget(dM - self.M).ravel())*np.exp(-self.d_ia.cget(dL + self.M).ravel())*self.d_ii.cget(-self.M).ravel()**-3
                    #R_accu.append(R_)
                    #EX_accu.append(g_exchange)




                    #R_ = np.sqrt(np.sum((r1-r0)**2))
                    #R_ = np.exp(-d0)*np.exp(-d1)/d2**6
                    #print(R_, np.max(np.abs(g_exchange)))

                    t = self.t2[:,ddL,:,self.mM_, :, ddM, :] #[self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i]





                    e_mp2_ab_ += 2*np.einsum("iajb,iajb",t[self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        g_direct[self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        optimize = opt)  \
                                - np.einsum("iajb,ibja",t[self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        g_exchange[self.f2.fragment][:, dM_i][:, :, self.f1.fragment][:,:,:,dL_i],
                                                        optimize = opt)

                    e_mp2_ba_ += 2*np.einsum("iajb,iajb",t[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        g_direct[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        optimize = opt)  \
                                - np.einsum("iajb,ibja",t[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        g_exchange[self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i],
                                                        optimize = opt)

                    e_mp2_aa_ += 2*np.einsum("iajb,iajb",t[self.f1.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        g_direct[self.f1.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        optimize = opt)  \
                                - np.einsum("iajb,ibja",t[self.f1.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i],
                                                        g_exchange[self.f1.fragment][:, dM_i][:, :, self.f1.fragment][:,:,:,dL_i],
                                                        optimize = opt)

                    e_mp2_bb_ += 2*np.einsum("iajb,iajb",t[self.f2.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        g_direct[self.f2.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        optimize = opt)  \
                                - np.einsum("iajb,ibja",t[self.f2.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i],
                                                        g_exchange[self.f2.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i],
                                                        optimize = opt)
        #print("Computed/reused:", computed, reuse)
        print("Pair energies:", e_mp2_aa, e_mp2_ab, e_mp2_ba, e_mp2_bb)
        print("Pair energies:", e_mp2_aa_, e_mp2_ab_, e_mp2_ba_, e_mp2_bb_)
        #return e_mp2_ab
        #EX_accu = np.array(EX_accu).ravel()
        #R_1_accu = np.array(R_1_accu).ravel()
        #R_2_accu = np.array(R_2_accu).ravel()
        #np.save("ex_accu.npy",EX_accu )
        #np.save( "r_1_accu.npy",R_1_accu)
        #np.save( "r_2_accu.npy",R_2_accu)

        #np.save("Integral_screen.npy", self.ib.Xscreen)
        #np.save("Integral_dist.npy", self.ib.Xdist)
        


        return np.array([e_mp2_aa, e_mp2_bb, e_mp2_ab, e_mp2_ba])

    def compute_pair_fragment_energy_(self):
        """
        Compute fragment energy
        """

        n_occ = self.ib.n_occ     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells # Number of occupied cells
        n_virt = self.ib.n_virt   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells # Number of virtual cells




        e_mp2 = 0

        N_virt = self.n_virtual_cells


        reuse = 0     #count instances where exchange integrals can be recycled
        computed = 0  #count instances where exchange integrals are computed
        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = self.d_ia.cget(dL)[self.f1.fragment[0],:]<self.virtual_cutoff


            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM] # - self.M
                dM_i = self.d_ia.cget(dM)[self.f1.fragment[0],:]<self.virtual_cutoff
                if np.sum(dM_i)>0 and np.sum(dL_i)>0:
                    g_direct = self.g_d[:,ddL,:,self.mM, :, ddM, :][self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]


                    try:
                        # Get exchange index / np.argwhere
                        ddM_M, ddL_M = get_index_where(self.d_ia.coords, dM+self.M), get_index_where(self.d_ia.coords, dL-self.M)
                        g_exchange = self.g_d[:,ddM_M,:,self.mM, :, ddL_M, :][self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i]

                        reuse += 1
                    except:

                        #print("Exchange not precomputed")
                        if self.adaptive:
                            I, Ishape = self.ib.get_adaptive( dM+self.M, dL-self.M, np.array([self.M]), keep = self.retain_integrals)
                        else:
                            I, Ishape = self.ib.getorientation(dM+self.M, dL-self.M)
                        g_exchange = I.cget(self.M).reshape(Ishape)[self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i]

                        computed += 1

                    t = self.t2[:,ddL,:,self.mM, :, ddM, :][self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]

                    e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)



                    # The opposite case

                    g_direct = self.g_d[:,ddL,:,self.mM_, :, ddM, :][self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i]



                    try:
                        # Get exchange index / np.argwhere
                        ddM_M, ddL_M = get_index_where(self.d_ia.coords, dM-self.M), get_index_where(self.d_ia.coords, dL+self.M)
                        g_exchange = self.g_d[:,ddM_M,:,self.mM_, :, ddL_M, :][self.f2.fragment][:, dM_i][:, :, self.f1.fragment][:,:,:,dL_i]
                        reuse += 1
                    except:
                        if self.adaptive:
                            I, Ishape = self.ib.get_adaptive(dM-self.M, dL+self.M, np.array([-self.M]), keep = self.retain_integrals)
                        else:
                            I, Ishape = self.ib.getorientation(dM-self.M, dL+self.M)
                        g_exchange = I.cget(-self.M).reshape(Ishape)[self.f2.fragment][:, dM_i][:, :, self.f1.fragment][:,:,:,dL_i]
                        computed += 1

                    t = self.t2[:,ddL,:,self.mM_, :, ddM, :][self.f2.fragment][:, dL_i][:, :, self.f1.fragment][:,:,:,dM_i]

                    e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)

        return np.array([e_mp2, e_mp2, e_mp2, e_mp2])

    def map_omega(self):
        No = self.n_occupied_cells
        Nv = self.n_virtual_cells
        self.t2_map = np.zeros((Nv, No, Nv), dtype = np.object)

        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

            
            
            #dL_i = (self.d_ia.cget(dLv)<self.virtual_cutoff)[:self.ib.n_occ, :]
            #dL_i[M0_i == False, :] = 0

            for M in np.arange(self.n_occupied_cells):
                Mv = self.d_ii.coords[M]
                M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask


                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask

                    K_a = get_bvec_where(self.d_ii.coords[:self.n_occupied_cells], -1*self.d_ii.coords[:self.n_occupied_cells])
                    dLv_a = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells], -1*self.d_ii.coords[:self.n_occupied_cells] + dLv - Mv)
                    dMv_a = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells], -1*self.d_ii.coords[:self.n_occupied_cells] + dMv)
                    indx_a = np.all(np.array([K_a, dLv_a, dMv_a])>=0, axis = 0)

                    D3 = np.array([K_a, dLv_a, dMv_a, indx_a])






                    dMv_b = get_bvec_where(self.d_ia.coords[:self.n_virtual_cells],  -1*self.d_ii.coords[:self.n_occupied_cells] + dMv + Mv)
                    K_b = np.arange(self.n_occupied_cells)
                    indx_b = np.all(np.array([K_b, dMv_b])>=0, axis = 0)

                    D4 = np.array([K_b, dMv_b, indx_b])
                    
                    self.t2_map[dL, M, dM] = np.array([D3, D4])






    def omega(self, t2):

        #t0 = time.time()

        t2_new = np.zeros_like(t2)
        #print("SHAPE:", t2_new.shape)

        M0_i = self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff
        #print(M0_i)
        #print("M0_i;", M0_i.shape)

        #M0_i = self.d_ii.cget([0,0,0])<self.occupied_cutoff
        #print("self.n_occupied_cells:", self.n_occupied_cells)
        #print("self.n_virtual_cells:", self.n_virtual_cells)

        #print("d_ii.coords:", self.d_ii.coords[:self.n_occupied_cells])
        #print("d_ia.coords:", self.d_ia.coords[:self.n_virtual_cells])

        tm1 = 0
        tm2 = 0
        tm3 = 0
        tm4 = 0
        tm5 = 0

        Nv = self.n_virtual_cells
        No = self.n_occupied_cells
        nv = self.ib.n_virt
        no = self.ib.n_occ




        for dL in np.arange(self.n_virtual_cells):
            dLv = self.d_ia.coords[dL]
            dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

            
            
            #dL_i = (self.d_ia.cget(dLv)<self.virtual_cutoff)[:self.ib.n_occ, :]
            #dL_i[M0_i == False, :] = 0

            for M in np.arange(self.n_occupied_cells):
                Mv = self.d_ii.coords[M]
                M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask


                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask
                    
                    #dM_i = (self.d_ia.cget(dMv)<self.virtual_cutoff)[:self.ib.n_occ, :]# dM index mask
                    #dM_i[M_i == False, :] = 0

                    




                    
                    
                    
                    
                    #M_i = self.d_ii.cget(Mv)<self.occupied_cutoff # M index mask

                    tnew = -self.g_d[:, dL, :, M, :, dM, :]

                    tt = time.time() #TImE
                    cell_map = np.arange(tnew.size).reshape(tnew.shape)[M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()
                    tm3 += time.time()-tt #TIME



                    tt = time.time() #TImE
                    F_ac = self.f_mo_aa.cget(self.d_ia.coords[:self.n_virtual_cells] - dLv)
                    tm3 += time.time()-tt #TIME

                    tt = time.time() #TImE
                    tnew -= np.einsum("iCcjb,Cac->iajb", t2[:, :, :, M, :, dM, :], F_ac)
                    tm1 += time.time()-tt #TIME

                    tt = time.time() #TImE
                    F_bc = self.f_mo_aa.cget(self.d_ia.coords[:self.n_virtual_cells] - Mv - dMv)
                    tm3 += time.time()-tt #TIME

                    tt = time.time() #TImE
                    tnew -= np.einsum("iajCc,Cbc->iajb", t2[:, dL, :, M, :, :, :], F_bc)
                    tm1 += time.time()-tt #TIme




                    D3, D4 = self.t2_map[dL, M, dM] 

                    tt = time.time() #TImE

                    K_, dLv_, dMv_, indx = D3
                    indx = np.array(indx, dtype = np.bool)


                    No_ = np.sum(indx)
                    if np.any(indx):
                        tt = time.time() #TImE

                        tnew += np.einsum("Kiakb,Kkj->iajb",t2[:, dLv_[indx], :, K_[indx], :, dMv_[indx], :], self.f_mo_ii.cget(-1*self.d_ii.coords[:self.n_occupied_cells]-Mv)[indx])
                        tm4 += time.time()-tt #TIme

                        # dot-implementation
                        # tt = time.time()
                        # F_kj = self.f_mo_ii.cget(-1*self.d_ii.coords[:self.n_occupied_cells]-Mv)[indx]
                        # t2_ = t2[:, dLv_[indx], :, K_[indx], :, dMv_[indx], :].swapaxes(3,4).swapaxes(0,3) #Kiakb -> biaKk
                        # t2F = np.dot(t2_.reshape([nv*no*nv, No_*no]), F_kj.reshape([No_*no, no])).reshape([nv,no*nv*no]) # -> biaj
                        # tnew += t2F.swapaxes(0,1).reshape((no,nv,no,nv))
                        #tm5 += time.time() - tt









 
                    tt = time.time() #TImE

                    K_, dMv_, indx = D4
                    indx = np.array(indx, dtype = np.bool)

                    tm2 += time.time()-tt #TIme

                    if np.any(indx):
                        tt = time.time() #TImE
                        tnew += np.einsum("Kiakb,Kkj->iajb",t2[:, dL, :, K_[indx], :, dMv_[indx], :], self.f_mo_ii.cget(-1*self.d_ii.coords[:self.n_occupied_cells]+Mv)[indx])
                        tm1 += time.time()-tt #TIme



                    tt = time.time() #TImE
                    t2_mapped = np.zeros_like(tnew).ravel()
                    
                    t2_mapped[cell_map] = (tnew*self.e_iajb**-1).ravel()[cell_map]
                    
                    #t2_mapped[cell_map] = .1*tnew.ravel()[cell_map]
                    #t2_mapped = (tnew*self.e_iajb**-1).ravel()




                    t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
                    tm2 += time.time()-tt #TIme
        #print("Cellmap omitted.")
        #print(tm1, tm2, tm3, tm4, tm5)
        return t2_new


    def omega_old(self, t2):

        t2_new = np.zeros_like(t2)

        M0_i = self.d_ii.cget([0,0,0])[self.fragment[0],:]<self.occupied_cutoff

        for M in np.arange(self.n_occupied_cells):
            Mv = self.d_ii.coords[M]
            M_i = self.d_ii.cget(Mv)[self.fragment[0],:]<self.occupied_cutoff # M index mask

            for dL in np.arange(self.n_virtual_cells):
                dLv = self.d_ia.coords[dL]
                dL_i = self.d_ia.cget(dLv)[self.fragment[0],:]<self.virtual_cutoff # dL index mask

                #dL_i = (self.d_ia.cget(dLv)<self.virtual_cutoff)[:self.ib.n_occ, :]
                #dL_i[M0_i == False, :] = 0



                for dM in np.arange(self.n_virtual_cells):
                    dMv = self.d_ia.coords[dM]  #- self.M
                    dM_i = self.d_ia.cget(dMv)[self.fragment[0],:]<self.virtual_cutoff # dM index mask

                    #dM_i = (self.d_ia.cget(dMv)<self.virtual_cutoff)[:self.ib.n_occ, :]# dM index mask
                    #dM_i[M_i == False, :] = 0



                    tnew = -self.g_d[:, dL, :, M, :, dM, :]

                    # generate index mapping of non-zero amplitudes in cell
                    cell_map = np.arange(tnew.size).reshape(tnew.shape)[M0_i][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()

                    #indx = np.outer(dL_i.ravel(), dM_i.ravel()).reshape((self.ib.n_occ, self.ib.n_virt, self.ib.n_occ, self.ib.n_virt))

                    #cell_map = np.arange(tnew.size)[indx.ravel()] 

                    #t2_mask = np.ones(tnew.shape, dtype = np.bool)
                    #t2_mask[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i] = False

                    #t2[:, :, dL_i==False, M, :, dM, dM_i==False] *= 0



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
                    #t2_mapped[cell_map] = .1*tnew.ravel()[cell_map]
                    #t2_mapped = (tnew*self.e_iajb**-1).ravel()




                    t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)
        return t2_new

    def nbytes(self):
        mem_usage = self.d_ia.blocks.nbytes + self.d_ii.blocks.nbytes
        mem_usage += self.t2.nbytes + self.g_d.nbytes
        return mem_usage

    def memory_profile(self):
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        # Prints out a summary of the large objects
        summary.print_(sum1)
    
    def deallocate(self):
        pass





class converge():
    def __init__(self, t2):
        self.t = [t2]

    def advance(self, dt2):
        self.t.append(self.t[-1] + dt2)


class diis():
    def __init__(self, N):
        self.N = N
        self.i = 0
        self.t = np.zeros(N, dtype = object)
        #self.dir = np.zeros(N, dtype = object)
        self.err = np.zeros(N, dtype = object)


    def advance(self, t_i, err_i):
        if self.i<self.N:
            self.t[self.i] = (t_i-err_i).ravel()
            self.err[self.i] = err_i.ravel()
            #self.dir[self.i] = err_i.ravel()
            self.i += 1
            return t_i-err_i #remember add damping

        self.err = np.roll(self.err,-1)
        self.t = np.roll(self.t,-1)
        #self.dir = np.roll(self.dir,-1)

        self.err[-1] = err_i.ravel()
        self.t[-1] = (t_i-err_i).ravel()
        #self.dir[-1] = err_i.ravel()

        self.build_b()

        w = np.linalg.pinv(self.b)[:, -1]

        ret = np.zeros(t_i.shape, dtype = float)
        err = np.zeros(t_i.shape, dtype = float)

        #print ('SUM COEFFS: ',np.sum(w))
        for i in np.arange(len(w)-1):
            ret += w[i] * self.t[i].reshape(t_i.shape)
            err += w[i] * self.err[i].reshape(err_i.shape)

        #print("sum w:", np.sum(w))
        self.t[-1] = ret.ravel()
        self.err[-1] = err.ravel()



        self.i += 1
        return ret - err



    def build_b(self):
        N = self.N
        b = np.zeros((N+1,N+1))
        b[:-1,N] = np.ones(N)
        b[N,:-1] = np.ones(N)
        for i in np.arange(N):
            for j in np.arange(i,N):
                b[i,j] = np.dot(self.err[i],self.err[j])
                b[j,i] = b[i,j]



        #eigvals = np.linalg.eigvals(b)
        #print ('MAX eigval: ',np.max(eigvals))
        #print ('MIN eigval: ',np.min(eigvals))
        #print ('RATIO eigval: ',np.max(eigvals)/np.min(eigvals))
        """
        if np.any(np.diag(b)[:-1] <= 0):
            pre_condition[:-1] = 1
        else:
            pre_condition[:-1] += np.power(np.diag(b_mat)[:-1], -0.5)

        pre_condition[b_dim] = 1

        for i in range(b_dim + 1):
            for j in range(b_dim + 1):
                b_mat[i, j] *= pre_condition[i] * pre_condition[j]
        """




        self.b = b


class diis__():
    def __init__(self, N):
        self.N = N
        self.i = 0
        self.t = np.zeros(N, dtype = object)
        self.err = np.zeros(N, dtype = object)


    def advance(self, t_i, err_i):
        if self.i<self.N:
            self.t[self.i] = t_i.ravel()
            self.err[self.i] = err_i.ravel()
            self.i += 1
            return t_i #remember add damping

        self.err = np.roll(self.err,-1)
        self.t = np.roll(self.t,-1)

        self.err[-1] = err_i.ravel()
        self.t[-1] = t_i.ravel()

        self.build_b()

        w = np.linalg.pinv(self.b)[:, -1]

        ret = np.zeros(t_i.shape, dtype = float)
        err = np.zeros(t_i.shape, dtype = float)

        #print ('SUM COEFFS: ',np.sum(w))
        for i in np.arange(len(w)-1):
            ret += w[i] * self.t[i].reshape(t_i.shape)
            err += w[i] * self.err[i].reshape(err_i.shape)



        self.i += 1
        return ret




    def build_b(self):
        N = self.N
        b = np.zeros((N+1,N+1))
        b[:N,N] = np.ones(N)
        b[N,:N] = np.ones(N)
        for i in np.arange(N):
            for j in np.arange(i,N):
                b[i,j] = np.dot(self.err[i],self.err[j])
                b[j,i] = b[i,j]

        eigvals = np.linalg.eigvals(b)
        #print ('MAX eigval: ',np.max(eigvals))
        #print ('MIN eigval: ',np.min(eigvals))
        #print ('RATIO eigval: ',np.max(eigvals)/np.min(eigvals))
        self.b = b

class diis_():
    def __init__(self, N):
        self.N = N
        self.i = 0
        self.t = np.zeros(N, dtype = object)
        self.err = np.zeros(N, dtype = object)


    def advance(self, t_i, err_i):
        if self.i<self.N:
            self.t[self.i] = (t_i + err_i).ravel()
            self.err[self.i] = err_i.ravel()
            self.i += 1
            return t_i + err_i #remember add damping

        self.err = np.roll(self.err,-1)
        self.t = np.roll(self.t,-1)

        self.err[-1] = err_i.ravel()
        self.t[-1] = (t_i + err_i).ravel()

        self.build_b()

        w = np.linalg.pinv(self.b)[:, -1]

        ret = np.zeros(t_i.shape, dtype = float)
        err = np.zeros(t_i.shape, dtype = float)

        #print ('SUM COEFFS: ',np.sum(w))
        for i in np.arange(len(w)-1):
            ret += w[i] * self.t[i].reshape(t_i.shape)
            err += w[i] * self.err[i].reshape(err_i.shape)

        self.err[-1] = ret.ravel() - self.t[-2]
        self.t[-1] = ret.ravel()





        self.i += 1
        return ret


    def advance_old(self, t_i, err_i):
        if self.i<self.N:
            self.t[self.i % self.N] = t_i.ravel()
            self.err[self.i % self.N] = err_i.ravel()
            self.i += 1
            return t_i - 0.3*err_i #remember add damping


        self.build_b()

        w = np.linalg.pinv(self.b)[:, -1]

        ret = np.zeros(t_i.shape, dtype = float)
        err = np.zeros(t_i.shape, dtype = float)
        for i in np.arange(len(w)-1):
            ret += w[i] * self.t[i].reshape(t_i.shape)
            err += w[i] * self.err[i].reshape(err_i.shape)
        self.t[self.i % self.N] = ret.ravel()
        self.err[self.i % self.N] = err_i.ravel()
        self.i += 1
        return ret


    def build_b(self):
        N = self.N
        b = np.zeros((N+1,N+1))
        b[:N,N] = np.ones(N)
        b[N,:N] = np.ones(N)
        for i in np.arange(N):
            for j in np.arange(i,N):
                b[i,j] = np.dot(self.err[i],self.err[j])
                b[j,i] = b[i,j]

        eigvals = np.linalg.eigvals(b)
        #print ('MAX eigval: ',np.max(eigvals))
        #print ('MIN eigval: ',np.min(eigvals))
        #print ('RATIO eigval: ',np.max(eigvals)/np.min(eigvals))
        self.b = b





def build_weight_matrix(p, c, coords, fragment = None):
    # Compute PAOs w/ positions
    s, c_pao, wcenters_pao = of.conventional_paos(c, p)

    # Split wannier space
    c_occ, c_virt = PRI.occ_virt_split(c,p)

    # Associate every PAO with a occupied orbital
    # Compute occupied positions
    c_occ_centers, c_occ_spreads = of.centers_spreads(c_occ, p,coords, m= 1)
    #print("nocc :", c_occ.blocks.shape[2])
    #print("nvirt:", c_virt.blocks.shape[2])


    # Compute occupied - pao distance in refcell
    d_im = np.sum((c_occ_centers[:, None] - wcenters_pao[None,:])**2, axis = 2)

    #print(d_im)
    #pao_map = np.argmin(d_im, axis = 0)
    #print(pao_map.shape)
    #print(pao_map)



    w = c_pao.tT().cdot(s.cdot(c_virt), coords = coords)
    #wblocks = np.sum(w.blocks**2, axis = 1)
    #print(wblocks.shape)

    #wblocks = np.zeros((w.blocks.shape[0]-1,c_occ.blocks.shape[2], c_virt.blocks.shape[2]), dtype = float)
    #wblocks[:, :,np.arange(c_virt.blocks.shape[2])] =
    wblocks = np.sum(w.cget(w.coords)**2, axis = 1)[:,:,None]*np.ones(c_occ.blocks.shape[2], dtype = float)[None, None, :]

    wblocks = wblocks.swapaxes(1,2)
    wblocks = wblocks**-.5
    print(wblocks.shape)

    sort = np.argsort(np.min(wblocks, axis = (1,2)))



    d_ia = tp.tmat()
    d_ia.load_nparray(wblocks[sort], w.coords[sort])
    return d_ia

def plot_convergence_fig(e_mp2, de_mp2, n_virt, v_dist, n_occ, v_occ, p_ = None, FOT = 1e-6,t2_norm = None, dt_max = None, i = 0, n_iters = None, de_res = None):
    import matplotlib.pyplot as plt
    N = np.arange(e_mp2.shape[0])

    fitting_function = lambda x,a,b,c : a*np.exp(b*x**-c) #fitting function for error estimate

    plt.figure(i, figsize = (8,10))
    plt.subplot(9,1,1)
    plt.axhline(e_mp2[-1], color = (0,0,0), alpha = .3)
    plt.text(0, e_mp2[-1],"%.4e Ha" % e_mp2[-1])
    plt.plot(N, e_mp2, ".-", label ="e_mp2")

    if p_ is not None:
        plt.plot(N, fitting_function(v_dist, p_[0], p_[1], p_[2]), "o-", linewidth = 5, alpha = .2, label = "e_fit")
        plt.legend()


    plt.xlim(N[0], N[-1])
    plt.ylabel("$E_{mp2}$")

    plt.subplot(9,1,2)
    plt.plot(N, np.abs(de_mp2),".-", label ="$\Delta E$")
    plt.axhline(FOT, alpha = .1, label = "FOT")
    plt.yscale("log")
    plt.legend()
    plt.xlim(N[0], N[-1])
    plt.ylim(FOT*0.1,FOT*100)
    plt.ylabel("$| \Delta E |$")

    if t2_norm is not None:
        plt.subplot(9,1,3)
        plt.plot(N, t2_norm,".-")
        #plt.axhline(FOT, alpha = .1, label = "FOT")
        #plt.yscale("log")
        #plt.legend()
        plt.xlim(N[0], N[-1])
        plt.ylabel("$\\| t_2 \\|$")

        plt.subplot(9,1,4)
        plt.plot(N[1:], np.abs(t2_norm[1:]-t2_norm[:-1]), ".-")
        #plt.axhline(FOT, alpha = .1, label = "FOT")
        plt.yscale("log")
        #plt.legend()
        plt.xlim(N[0], N[-1])
        plt.ylabel("$ \\Delta \\|  t_2 \\|$")


    plt.subplot(9,1,5)
    plt.plot(N, n_virt,".-", label ="Virtual")
    plt.plot(N, n_occ, ".-", label = "Occupied")
    plt.xlim(N[0], N[-1])
    plt.legend()
    plt.ylabel("# orbitals")
    plt.subplot(9,1,6)
    plt.plot(N, v_dist,".-", label ="Virtual")
    plt.plot(N, v_occ, ".-", label = "Occupied")
    plt.xlim(N[0], N[-1])
    plt.legend()
    plt.ylabel("cutoff / a.u.")

    plt.subplot(9,1,7)
    plt.plot(N, de_res,".-")
    plt.xlim(N[0], N[-1])
    plt.ylabel("max. dev. res.")

    plt.subplot(9,1,8)
    plt.plot(N, n_iters,".-")
    plt.xlim(N[0], N[-1])
    plt.ylabel("# cycles")



    plt.savefig("convergence_fragment_%i.pdf" % i)



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
    parser.add_argument("auxbasis", type = str, help="Auxiliary fitting basis.")
    parser.add_argument("-fitted_coeffs", type= str,help="Array of coefficient matrices from RI-fitting")
    #parser.add_argument("wcenters", type = str, help="Wannier centers")
    parser.add_argument("-attenuation", type = float, default = 1.2, help = "Attenuation paramter for RI")
    parser.add_argument("-fot", type = float, default = 0.001, help = "fragment optimization treshold")
    parser.add_argument("-circulant", type = bool, default = True, help = "Use circulant dot-product.")
    parser.add_argument("-robust", default = False, action = "store_true", help = "Enable Dunlap robust fit for improved integral accuracy.")
    parser.add_argument("-ibuild", type = str, default = None, help = "Filename of integral fitting module (will be computed if missing).")
    parser.add_argument("-n_core", type = int, default = 0, help = "Number of core orbitals (the first n_core orbitals will not be correlated).")
    parser.add_argument("-skip_fragment_optimization", default = False, action = "store_true", help = "Skip fragment optimization (for debugging, will run faster but no error estimate.)")
    parser.add_argument("-basis_truncation", type = float, default = 0.1, help = "Truncate fitting basis function below this exponent threshold." )
    #parser.add_argument("-ao_screening", type = float, default = 1e-12, help = "Screening of the (J|mn) (three index) integrals.")
    parser.add_argument("-xi0", type = float, default = 1e-10, help = "Screening of the (J|mn) (three index) integrals.")
    parser.add_argument("-xi1", type = float, default = 1e-10, help = "Screening of the (J|pn) (three index) integral transform.")
    parser.add_argument("-float_precision", type = str, default = "np.float64", help = "Floating point precision.")
    #parser.add_argument("-attenuated_truncation", type = float, default = 1e-14, help = "Truncate blocks in the attenuated matrix where (max) elements are below this threshold." )
    parser.add_argument("-virtual_space", type = str, default = None, help = "Alternative representation of virtual space, provided as tmat file." )
    parser.add_argument("-solver", type = str, default = "mp2", help = "Solver model." )
    parser.add_argument("-N_c", type = int, default = 0, help = "Force number of layers in Coulomb BvK-cell." )
    parser.add_argument("-pairs", type = bool, default = False, help = "Compute pair fragments" )
    parser.add_argument("-pair_setup", type = str, default = "standard", help = "Setup of pair calculations. Choose between standard, alternative and auto." )
    parser.add_argument("-print_level", type = int, default = 0, help = "Print level" )
    parser.add_argument("-orb_increment", type = int, default = 6, help = "Number of orbitals to include at every XDEC-iteration." )
    parser.add_argument("-pao_sorting", type = bool, default = False, help = "Sort LVOs in order of decreasing PAO-coefficient" )
    parser.add_argument("-adaptive_domains", default = False, action = "store_true", help = "Activate adaptive Coulomb matrix calculation. (currently affects only pair calculations).")
    parser.add_argument("-recycle_integrals", type = bool, default = True, help = "Recycle fragment integrals when computing pairs." )
    parser.add_argument("-retain_integrals", type = bool, default = False, help = "Keep new fiitting-coefficients when computing pairs. (More memory intensive)" )
    parser.add_argument("-fragmentation", type = str, default = "dec", help="Fragmentation scheme (dec/cim)")
    parser.add_argument("-afrag", type = float, default = 2.0, help="Atomic fragmentation threshold.")
    parser.add_argument("-virtual_cutoff", type = float, default = 3.0, help="Initial virtual cutoff for DEC optimization.")
    parser.add_argument("-occupied_cutoff", type = float, default = 1.0, help="Initial virtual cutoff for DEC optimization.")
    parser.add_argument("-pao_thresh", type = float, default = 0.1, help="PAO norm truncation cutoff.")
    parser.add_argument("-damping", type = float, default = 1.0, help="PAO norm truncation cutoff.")
    parser.add_argument("-fragment_center", action = "store_true",  default = False, help="Computes the mean position of every fragment")
    parser.add_argument("-atomic_association", action = "store_true",  default = False, help="Associate virtual (LVO) space with atomic centers.")
    parser.add_argument("-orthogonalize", action = "store_true",  default = False, help="Orthogonalize orbitals prior to XDEC optim")
    parser.add_argument("-spacedef", type = str, default = None, help = "Define occupied space and virtual space based on indexing (ex. spacedef 0,4,5,10 <-first occupied, final occupied, first virtual, final virtual")
    parser.add_argument("-ndiis", type = int, default = 4, help = "DIIS for mp2 optim.")
    #parser.add_argument("-inverse_test", type = bool, default = False, help = "Perform inversion and condition tests when initializing integral fitter." )
    parser.add_argument("-inverse_test", action = "store_true",  default = False, help="Perform inversion and condition testing")
    parser.add_argument("-pair_domain_def", type = int, default = 0, help = "Specification of pair domain type")
    parser.add_argument("-coeff_screen", type = float, default = None, help="Screen coefficients blockwise.")
    parser.add_argument("-error_estimate", type = bool, default = False, help = "Perform error estimate on DEC fragment energies." )
    
 

    args = parser.parse_args()

    #print("Invtest:", args.inverse_test)

    args.float_precision = eval(args.float_precision)
    import sys

    


    #git_hh = sp.Popen(['git' , 'rev-parse', 'HEAD'], shell=False, stdout=sp.PIPE, cwd = sys.path[0])

    #git_hh = git_hh.communicate()[0].strip()

    # Print run-info to screen
    #print("Git rev : ", git_hh)
    print("Contributors : Audun Skau Hansen (a.s.hansen@kjemi.uio.no) ")
    print("               Einar Aurbakken")
    print("               Thomas Bondo Pedersen")
    print(" ")
    print("   \u001b[38;5;117m0\u001b[38;5;27mø \033[0mHylleraas Centre for Quantum Molecular Sciences")
    print("                        UiO 2020")


    #print
    #print("Git rev:", sp.check_output(['git', 'rev-parse', 'HEAD'], cwd=sys.path[0]))
    print("_________________________________________________________")
    print("Input configuration")
    print("_________________________________________________________")
    print("command line args:")
    print(parser.parse_args())
    print(" ")
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
    print("Virtual space          :", args.virtual_space)
    print("Coulomb extent (layers):", args.N_c)
    print("Atomic fragmentation   :", args.afrag)
    #print("Dot-product            :", ["Block-Toeplitz", "Circulant"][int(args.circulant)])
    #print("RI fitting             :", ["Non-robust", "Robust"][int(args.robust)])
    print("_________________________________________________________",flush=True)




    # Load system
    p = pr.prism(args.project_file)
    p.n_core = args.n_core
    p.set_nocc()

    # Compute overlap matrix
    s = of.overlap_matrix(p)





    # Fitting basis
    if args.basis_truncation < 0:
        auxbasis = PRI.remove_redundancies(p, 2*args.N_c, args.auxbasis, analysis = True)
        f = open("ri-fitbasis.g94", "w")
        f.write(auxbasis)
        f.close()
    else:
        auxbasis = PRI.basis_trimmer(p, args.auxbasis, alphacut = args.basis_truncation)
        f = open("ri-fitbasis.g94", "w")
        f.write(auxbasis)
        f.close()

    # Load wannier coefficients
    c = tp.tmat()
    c.load(args.coefficients)
    c.set_precision(args.float_precision)

    #cnew = tp.get_zero_tmat([20,0,0], [c.blocks.shape[1], c.blocks.shape[2]])
    #cnew.blocks[ cnew.mapping[ cnew._c2i(c.coords)]] = c.cget(c.coords)
    #c = cnew*1
    #print(c.coords)


    #c = of.orthogonalize_tmat(c, p, coords = tp.lattice_coords([30,0,0]))
    if args.orthogonalize:
        c = of.orthogonalize_tmat_cholesky(c, p)

    if args.virtual_space == "gs":
        # Succesive outprojection of virtual space
        # (see https://colab.research.google.com/drive/1Cvpid-oBrvsSza8YEqh6qhm_VhtR6QgK?usp=sharing)
        c_occ, c_virt = PRI.occ_virt_split(c, p)

        #c = of.orthogonal_paos_gs(c,p, p.get_n_ao(), thresh = args.pao_thresh)
        c = of.orthogonal_paos_rep(c,p, p.get_nvirt(), thresh = args.pao_thresh, orthogonalize = args.orthogonalize)

        c_occ, c_pao = PRI.occ_virt_split(c, p)

        args.virtual_space = None #"pao"
        p.set_nvirt(c.blocks.shape[2] - p.get_nocc())

        #if args.orthogonalize:
        #    c = of.orthogonalize_tmat_cholesky(c, p)

        smo = c.tT().circulantdot(s.circulantdot(c))



        print("Gram-Schmidt like virtual space construted.")
        print("Max dev. from orthogonality:", np.abs((smo - tp.get_identity_tmat(smo.blocks.shape[1])).blocks).max()   )

        print("Conserved span of virtual space?")

        smo = c_virt.tT().circulantdot(s.circulantdot(c_pao))
        print(np.sum(smo.blocks**2, axis = (0,1)))
        print(np.sum(smo.blocks**2, axis = (0,2)),flush=True)



    #print("coeff shape:", c.blocks.shape)

    #if args.orthogonalize:
    #    c = of.orthogonalize_tmat_cholesky(c, p)

    #c = of.orthogonalize_tmat_unfold(c,p, thresh = 0.0001)


    # compute wannier centers

    wcenters, spreads = of.centers_spreads(c, p, s.coords)
    #wcenters = wcenters[p.n_core:] #remove core orbitals
    #print(wcenters.T)
    #print(spreads)

    



    c_occ, c_virt = PRI.occ_virt_split(c,p) #, n = p.get_nocc_all())



    #print("orbspace:", p.n_core, p.get_nocc())

    #print("orbspace:", c_occ.blocks.shape[2], c_virt.blocks.shape[2])
    #print("ncore   :", p.n_core, p.get_nocc(), p.get_nvirt())

    if args.virtual_space is not None:
        if args.virtual_space == "pao":
            s_, c_virt, wcenters_virt = of.conventional_paos(c,p, thresh = args.pao_thresh)
            #s_, c_virt, wcenters_virt = of.orthogonal_paos(c,p)
            #p.n_core = args.n_core
            p.set_nvirt(c_virt.blocks.shape[2])

            args.solver = "mp2_nonorth"

            #c_virt = of.orthogonalize_tmat_unfold(c_virt,p, thresh = 0.0001)
            #c_virt = of.orthogonalize_tmat(c_virt, p)

            #p.n_core = args.n_core
            # Append virtual centers to the list of centers
            #args.virtual_space = None
            wcenters = np.append(wcenters[p.n_core:p.get_nocc()+p.n_core], wcenters_virt, axis = 0)


        elif args.virtual_space == "paodot":
            s_, c_virt, wcenters_virt = of.conventional_paos(c,p)

            p.set_nvirt(c_virt.blocks.shape[2])

            args.solver = "paodot"

            # Append virtual centers to the list of centers

            #p.n_core = args.n_core
            # Append virtual centers to the list of centers
            wcenters = np.append(wcenters[p.n_core:p.get_nocc()+p.n_core], wcenters_virt, axis = 0)

        else:
            c_virt = tp.tmat()
            c_virt.load(args.virtual_space)
            p.set_nvirt(c_virt.blocks.shape[2])
        """
        if args.virtual_space == "pao":
            s_, c_virt, wcenters_virt = of.conventional_paos(c,p)
            #p.n_core = args.n_core

            p.set_nvirt(c_virt.blocks.shape[2])
            print("p.get_nvirt:", p.get_nvirt())

            args.solver = "mp2_nonorth"


            # Append virtual centers to the list of centers
            wcenters = np.append(wcenters[:p.get_nocc()], wcenters_virt, axis = 0)
            p.n_core = args.n_core
        """

    else:
        #p.n_core = args.n_core
        wcenters = wcenters[p.n_core:]

    #args.virtual_space = "pao"


    #c_occ, c_virt_lvo = PRI.occ_virt_split(c,p)
    if args.spacedef is not None:
        from ast import literal_eval
        occI, occF, virtI, virtF = [literal_eval(i) for i in args.spacedef.split(",")]
        print("Subspace definition (occupied0, occupiedF, virtual0, virtualF):", occI, occF, virtI, virtF)

        c_occ = tp.tmat()
        c_occ.load_nparray(c.cget(c.coords)[:, :, occI:occF], c.coords)
        wcenters_occ = wcenters[occI:occF]

        if virtF == -1:
            c_virt = tp.tmat()
            c_virt.load_nparray(c.cget(c.coords)[:, :, virtI:], c.coords)
            wcenters_virt = wcenters[virtI:]
        else:
            c_virt = tp.tmat()
            c_virt.load_nparray(c.cget(c.coords)[:, :, virtI:virtF], c.coords)
            wcenters_virt = wcenters[virtI:virtF]

        wcenters = np.append(wcenters_occ, wcenters_virt, axis = 0)

    
    wcenters_occ, spreads_occ = of.centers_spreads(c_occ, p, s.coords)

    wcenters_virt, spreads_virt = of.centers_spreads(c_virt, p, s.coords)

    #spreads = np.array([spreads_occ, spreads_virt]).ravel()
    #print(spreads)

    wcenters = np.append(wcenters_occ, wcenters_virt, axis = 0)

    #print("Number of coefficients to truncate:", np.sum(np.abs(c_occ.blocks[np.abs(c_occ.blocks)>1e-14])<1e-6))
    #print("Number of coefficients to keep    :", np.sum(np.abs(c_occ.blocks[np.abs(c_occ.blocks)>1e-14])>1e-6))
    #print("Number of coefficients in total   :", np.prod(c_occ.blocks.shape))
    print("Spaces:")
    print(c_occ.blocks.shape, c_virt.blocks.shape, wcenters.shape)

    if args.coeff_screen is not None:
        
        c_occ = tp.screen_tmat(c_occ, tolerance = args.coeff_screen)
        c_virt = tp.screen_tmat(c_virt, tolerance = args.coeff_screen)

        print("Spaces after screening:")
        print(c_occ.blocks.shape, c_virt.blocks.shape, wcenters.shape)

    






    s_virt = c_virt.tT().circulantdot(s.circulantdot(c_virt))
    #print(np.diag(s_virt.cget([0,0,0])))
    #s_virt = c_virt.tT().cdot(s*c_virt, coords = c_virt.coords)



    # AO Fock matrix
    f_ao = tp.tmat()
    f_ao.load(args.fock_matrix)
    f_ao.set_precision(args.float_precision)



    # Compute MO Fock matrix

    f_mo_aa = c_virt.tT().cdot(f_ao*c_virt, coords = c_virt.coords)
    f_mo_ii = c_occ.tT().cdot(f_ao*c_occ, coords = c_occ.coords)
    f_mo_ia = c_occ.tT().cdot(f_ao*c_virt, coords = c_occ.coords)

    print("Maximum f_ia:", np.abs(f_mo_ia.blocks).max(),flush=True)

    #f_mo_aa = c_virt.tT().circulantdot(f_ao.circulantdot(c_virt))
    #f_mo_ii = c_occ.tT().circulantdot(f_ao.circulantdot(c_occ))
    #f_mo_ia = c_occ.tT().circulantdot(f_ao.circulantdot(c_virt))

    #f_mo_ia = c_occ.tT().circulantdot(f_ao.circulantdot(c_virt)) #, coords = c_occ.coords)

    # Compute energy denominator

    f_aa = np.diag(f_mo_aa.cget([0,0,0]))
    f_ii = np.diag(f_mo_ii.cget([0,0,0]))

    e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]





    # Initialize integrals
    if args.ibuild is None:
        ib = PRI.integral_builder_static(c_occ,c_virt,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=args.circulant, robust = args.robust, xi0=args.xi0, xi1 = args.xi1, float_precision = args.float_precision, N_c = args.N_c,printing = args.print_level, inverse_test = args.inverse_test)
        #ib = PRI.integral_builder_static(c_occ,c_virt,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=None, circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust, ao_screening = args.ao_screening, xi0=args.xi0, JKa_extent= [6,6,6], xi1 = args.xi1, float_precision = args.float_precision, N_c = args.N_c,printing = args.print_level)

        np.save("integral_build.npy", np.array([ib]), allow_pickle = True)
    else:
        ib = np.load(args.ibuild, allow_pickle = True)[0]
        #print(ib.getorientation([0,0,0],[0,0,0]))
        print("Integral build read from file:", args.ibuild)
        args.attenuation = ib.attenuation
        print("Attenuation parameter set to %.4e" % args.attenuation)


    """
    for i in np.arange(60):
        Is, Ishape = ib.getorientation(np.array([i,0,0]), np.array([0,0,0]))
        #print("integral test:", i, np.max(np.abs(Is.cget([0,0,0]))))
        for M in np.arange(60):
            print("integral test:", i, M, np.max(np.abs(Is.cget([M,0,0]))))
    """




    # Initialize domain definitions


    d = dd.build_distance_matrix(p, c.coords, wcenters, wcenters)






    d_occ_ref = d.cget([0,0,0])[:ib.n_occ, :ib.n_occ]
    #if args.spacedef is not None:
    #    d_occ_ref = d.cget([0,0,0])[PRI.]

    center_fragments = dd.atomic_fragmentation(ib.n_occ, d_occ_ref, args.afrag) #[::-1]
    #center_fragments = [[0,1,2], [3,4,5]]

    #print(wcenters)


    print(" ")
    print("_________________________________________________________")
    print("Fragmentation of occupied space:")
    for i in np.arange(len(center_fragments)):
        print("  Fragment %i:" %i, center_fragments[i], wcenters[center_fragments[i][0]])
    print("_________________________________________________________")
    print(" ",flush=True)

    if args.fragment_center:
        # use a reduced charge expression to estimate the center of a fragment
        for i in np.arange(len(center_fragments)):
            pos_ = np.sum(wcenters[center_fragments[i]], axis = 0)/len(center_fragments[i])
            print(pos_)
            print(wcenters[center_fragments[i]])
            for j in np.arange(len(center_fragments[i])):
                wcenters[center_fragments[i][j]] = pos_
    if args.atomic_association:
        # associate virtual orbitals to atomic centers
        pos_0, charge_0 = p.get_atoms([[0,0,0]])
        r_atom = np.sqrt(np.sum((wcenters[p.get_nocc():][:, None]  - pos_0[ None,:])**2, axis = 2))
        for i in np.arange(p.get_nocc(), len(wcenters)):
            wcenters[i] = pos_0[np.argmin(r_atom[i-p.get_nocc()])]

    #print()





    # Converge atomic fragment energies

    # Initial fragment extents
    if args.fragmentation == "dec-sweep":
        """
        Map convergence of energy orbital spaces in the virtual direction

        """
        virt_cut = args.virtual_cutoff


        if args.occupied_cutoff < args.afrag:
            occ_cut = args.afrag
        else:
            occ_cut = args.occupied_cutoff

        refcell_fragments = []
        fragment_errors = []
        fragment_energy_total = 0

        # Run DEC-fragment optimization

        for fragment in center_fragments:


            #ib.fragment = fragment
            t0 = time.time()
            #domain_max = tp.lattice_coords(PRI.n_points_p(p, 2*args.N_c))
            domain_max = tp.lattice_coords(PRI.n_points_p(p, 20))

            if args.pao_sorting:
                d_ia = build_weight_matrix(p, c, domain_max)
                a_frag = fragment_amplitudes(p, wcenters,domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision, d_ia = d_ia)


            else:
                a_frag = fragment_amplitudes(p, wcenters, domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision)

            #print("Frag init:", time.time()-t0)

            dt, it, E_prev_outer = a_frag.solve(eqtype = args.solver, s_virt = s_virt, damping = args.damping)

            #print("t2 (max/min/absmin):", np.max(a_frag.t2), np.min(a_frag.t2), np.abs(a_frag.t2).min())
            #print("g_d (max/min/absmin):", np.max(a_frag.g_d), np.min(a_frag.g_d), np.abs(a_frag.g_d).min())

            # Converge to fot
            #E_prev_outer = a_frag.compute_fragment_energy()
            E_prev = E_prev_outer*1.0
            dE_outer = 10

            print("Initial fragment energy: %.8f" % E_prev)

            print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
            print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot),flush=True)

            virtual_cutoff_prev = a_frag.virtual_cutoff
            occupied_cutoff_prev = a_frag.occupied_cutoff

            if not args.skip_fragment_optimization:
                print("Running fragment optimization for:")
                print(fragment)
                #print("Initial cutoffs:")

                n_virtuals_  = []
                virtu_cut_   = []
                n_occupieds_ = []
                occu_cut_    = []
                e_mp2        = []
                de_mp2       = []
                amp_norm     = []
                n_iters      = [] # number of mp2 iterations
                de_res       = [] # max deviation in residual




                dE = 10

                # Update state
                """
                e_mp2.append(E_prev)
                de_mp2.append(dE)
                n_virtuals_.append(a_frag.n_virtual_tot)
                virtu_cut_.append(a_frag.virtual_cutoff)
                n_occupieds_.append(a_frag.n_occupied_tot)
                occu_cut_.append(a_frag.occupied_cutoff)
                amp_norm.append(a_frag.compute_eos_norm())
                n_iters.append(0)
                de_res.append(0)
                """


                def expand_fragment_space(a_frag, args, s_virt, E_prev, occupied = False, virtual = False):
                    # Expand virtual space

                    virtual_cutoff_prev = a_frag.virtual_cutoff
                    occupied_cutoff_prev = a_frag.occupied_cutoff

                    t_0 = time.time()

                    if virtual:
                        a_frag.autoexpand_virtual_space(n_orbs=args.orb_increment)
                    if occupied:
                        a_frag.autoexpand_occupied_space(n_orbs=args.orb_increment)
                    t_1 = time.time()
                    dt, it, E_new = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.01, damping = args.damping)
                    t_2 = time.time()
                    #E_new = a_frag.compute_fragment_energy()
                    t_3 = time.time()
                    dE = np.abs(E_prev - E_new)
                    """
                    print("_________________________________________________________")
                    print("E(fragment): %.8f      DE(fragment): %.8e" % (E_new, dE))
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
                    print("Time (expand/solve/energy) (s) : %.1f / %.1f / %.1f" % (t_1-t_0, t_2-t_1, t_3-t_2))
                    print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                    print(" ",flush=True)
                    """
                    return E_new, dt, it, virtual_cutoff_prev, occupied_cutoff_prev


                """
                dE = 1000

                E_prev = a_frag.compute_fragment_energy()

                #conv_stats = []

                e_mp2.append(E_prev)
                de_mp2.append(0)
                n_virtuals_.append(a_frag.n_virtual_tot)
                virtu_cut_.append(a_frag.virtual_cutoff)
                n_occupieds_.append(a_frag.n_occupied_tot)
                occu_cut_.append(a_frag.occupied_cutoff)
                amp_norm.append(np.linalg.norm(a_frag.t2))
                """

                # Converge occupied space / buffer amplitudes
                dE_o = 1000
                E_prev_o = E_prev
                while dE_o>args.fot:
                    # Expand occupied AOS space

                    E_new_o, dt, it, virtual_cutoff_prev, occupied_cutoff_prev = expand_fragment_space(a_frag, args, s_virt, E_prev, occupied = True)

                    dE_o = np.abs(E_prev_o- E_new_o)

                    E_prev_o = E_new_o

                    print("_________________________________________________________")
                    print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new_o, dE_o))
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in bytes): %.2f" % ib.nbytes())
                    print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                    print(" ",flush=True)

                # When converged, take one step back
                a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)




                fitting_function = lambda x,a,b,c : a*np.exp(b*x**-c) #fitting function for error estimate

                # Expand virtual space to FOT

                dE_v = 1000

                E_prev_v = E_prev_o

                dE_estimate = 1000

                #args.orb_increment = 2 # FIX HARDCODED increment

                estimated_pts = 0

                while a_frag.n_virtual_tot<100:

                    # Expand virtual space

                    E_new_v, dt, it, virtual_cutoff_prev, occupied_cutoff_prev = expand_fragment_space(a_frag, args, s_virt, E_prev, virtual = True)

                    dE_v = np.abs(E_prev_v - E_new_v)


                    E_prev_v = E_new_v









                    #conv_stats.append([dt, it, E_new, dE_v,a_frag.virtual_cutoff, a_frag.n_virtual_tot,a_frag.occupied_cutoff, a_frag.n_occupied_tot])
                    print("_________________________________________________________")
                    print("E(fragment): %.8e        DE(fragment): %.8e" % (E_new_v, dE_v))
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))

                    if args.error_estimate:

                        try:
                            p_, cov = optimize.curve_fit(fitting_function, np.array(virtu_cut_), np.array(e_mp2))
                            if estimated_pts>=2:
                                dE_estimate = np.abs(E_new_v - p_[0])

                            print("Estimated error in energy            : %.6e" % np.abs(E_new_v - p_[0]))
                            estimated_pts += 1

                        except:
                            print("Unable to estimate error.")
                            print(" ")
                        print("estimated_pts", estimated_pts)

                    else:
                        dE_estimate = 0

                    
                    print("Current memory usage of integrals (in bytes): %.2f" % ib.nbytes())
                    print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                    print(" ")
                    print("_________________________________________________________",flush=True)

                    e_mp2.append(E_new_v)
                    de_mp2.append(dE_estimate)
                    n_virtuals_.append(a_frag.n_virtual_tot)
                    virtu_cut_.append(a_frag.virtual_cutoff)
                    n_occupieds_.append(a_frag.n_occupied_tot)
                    occu_cut_.append(a_frag.occupied_cutoff)
                    amp_norm.append(a_frag.compute_eos_norm())
                    n_iters.append(it)
                    de_res.append(dt)





                # When converged, take one step back
                a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)

                #E_prev_o = a_frag.compute_fragment_energy()

                e_mp2.append(E_prev_v)
                de_mp2.append(dE_estimate)
                n_virtuals_.append(a_frag.n_virtual_tot)
                virtu_cut_.append(a_frag.virtual_cutoff)
                n_occupieds_.append(a_frag.n_occupied_tot)
                occu_cut_.append(a_frag.occupied_cutoff)
                amp_norm.append(a_frag.compute_eos_norm())
                n_iters.append(0)
                de_res.append(dt)

                print("e_mp2 = np.array(",e_mp2, ")")
                print("de_mp2 = np.array(",de_mp2, ")")
                print("n_virt = np.array(",n_virtuals_, ")")
                print("v_dist = np.array(",virtu_cut_, ")")
                print("n_occ = np.array(",n_occupieds_, ")")
                print("v_occ = np.array(",occu_cut_, ")")
                print("t2_norm = np.array(",amp_norm, ")",flush=True)

                e_mp2 = np.array(e_mp2)
                de_mp2 = np.array(de_mp2)
                n_virtuals_ = np.array(n_virtuals_)
                virtu_cut_ = np.array(virtu_cut_)
                occu_cut_ = np.array(occu_cut_)
                n_occupieds_ = np.array(n_occupieds_)
                amp_norm = np.array(amp_norm)
                n_iters = np.array(n_iters)
                de_res = np.array(de_res)

                np.save("fragment_optim_data_%i.npy" % fragment[0], np.array([e_mp2, de_mp2, n_virtuals_, virtu_cut_, n_occupieds_, occu_cut_, amp_norm, n_iters, de_res]))


                plot_convergence_fig(e_mp2, de_mp2, n_virtuals_, virtu_cut_, n_occupieds_, occu_cut_, p_, FOT = args.fot,t2_norm = amp_norm, i = fragment[0], n_iters = n_iters, de_res = de_res)




                # experimental error estimate
                # p_, cov = optimize.curve_fit(fitting_function, virtu_cut_**-1, e_mp2)


                print("=========================================================")
                print("Final fragment containing occupied orbitals:", a_frag.fragment)
                print("Converged fragment energy: %.12f" % E_prev_v)
                print("Estimated error in energy: %.12f" % np.abs(E_prev_v - p_[0]))
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                print("=========================================================")
                print(" ")
                print(" ",flush=True)
                refcell_fragments.append(a_frag)
                #fragment_errors.append(dE)
                fragment_errors.append(np.abs(E_prev_v - p_[0]))
                fragment_energy_total += E_prev_v






        print(" ")
        print("Fragment energies")
        for fa in np.arange(len(refcell_fragments)):
            #print(fa, ,)
            print(fa,"%.10e" % refcell_fragments[fa].compute_fragment_energy() ,"+/-",  "%.5e" % fragment_errors[fa],  " Ha / virt %.2f bohr ( n=%i ) / occ %.2f bohr ( n=%i )" %  (refcell_fragments[fa].virtual_cutoff, refcell_fragments[fa].n_virtual_tot,
                                                                                  refcell_fragments[fa].occupied_cutoff, refcell_fragments[fa].n_occupied_tot))




        fragment_errors = np.array(fragment_errors)

        print("Total fragment energy:", fragment_energy_total, "+/-", np.sqrt(np.sum(fragment_errors**2)),flush=True)




    # Initial fragment extents
    if args.fragmentation == "dec-efe":
        virt_cut = args.virtual_cutoff


        if args.occupied_cutoff < args.afrag:
            occ_cut = args.afrag
        else:
            occ_cut = args.occupied_cutoff

        refcell_fragments = []
        fragment_errors = []
        fragment_energy_total = 0

        # Run DEC-fragment optimization

        for fragment in center_fragments:


            #ib.fragment = fragment
            t0 = time.time()
            domain_max = tp.lattice_coords(PRI.n_points_p(p, 20))

            if args.pao_sorting:
                d_ia = build_weight_matrix(p, c, domain_max)
                a_frag = fragment_amplitudes(p, wcenters,domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision, d_ia = d_ia)


            else:
                a_frag = fragment_amplitudes(p, wcenters, domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision)

            #print("Frag init:", time.time()-t0)

            dt, it, E_prev_outer = a_frag.solve(eqtype = args.solver, s_virt = s_virt, damping = args.damping)

            #print("t2 (max/min/absmin):", np.max(a_frag.t2), np.min(a_frag.t2), np.abs(a_frag.t2).min())
            #print("g_d (max/min/absmin):", np.max(a_frag.g_d), np.min(a_frag.g_d), np.abs(a_frag.g_d).min())

            # Converge to fot
            #E_prev_outer = a_frag.compute_fragment_energy()
            E_prev = E_prev_outer*1.0
            dE_outer = 10

            print("Initial fragment energy: %.8f" % E_prev)

            print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
            print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot),flush=True)

            virtual_cutoff_prev = a_frag.virtual_cutoff
            occupied_cutoff_prev = a_frag.occupied_cutoff

            if not args.skip_fragment_optimization:
                print("Running fragment optimization for:")
                print(fragment)
                #print("Initial cutoffs:")

                n_virtuals_  = []
                virtu_cut_   = []
                n_occupieds_ = []
                occu_cut_    = []
                e_mp2        = []
                de_mp2       = []
                amp_norm     = []
                n_iters      = [] # number of mp2 iterations
                de_res       = [] # max deviation in residual




                dE = 10

                # Update state
                """
                e_mp2.append(E_prev)
                de_mp2.append(dE)
                n_virtuals_.append(a_frag.n_virtual_tot)
                virtu_cut_.append(a_frag.virtual_cutoff)
                n_occupieds_.append(a_frag.n_occupied_tot)
                occu_cut_.append(a_frag.occupied_cutoff)
                amp_norm.append(a_frag.compute_eos_norm())
                n_iters.append(0)
                de_res.append(0)
                """


                def expand_fragment_space(a_frag, args, s_virt, E_prev, occupied = False, virtual = False):
                    # Expand virtual space

                    virtual_cutoff_prev = a_frag.virtual_cutoff
                    occupied_cutoff_prev = a_frag.occupied_cutoff

                    t_0 = time.time()

                    if virtual:
                        a_frag.autoexpand_virtual_space(n_orbs=args.orb_increment)
                    if occupied:
                        a_frag.autoexpand_occupied_space(n_orbs=args.orb_increment)
                    t_1 = time.time()
                    dt, it, E_new = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.01, damping = args.damping,energy = "fragment")
                    t_2 = time.time()
                    #E_new = a_frag.compute_fragment_energy()
                    t_3 = time.time()
                    dE = np.abs(E_prev - E_new)
                    """
                    print("_________________________________________________________")
                    print("E(fragment): %.8f      DE(fragment): %.8e" % (E_new, dE))
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
                    print("Time (expand/solve/energy) (s) : %.1f / %.1f / %.1f" % (t_1-t_0, t_2-t_1, t_3-t_2))
                    print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                    print(" ",flush=True)
                    """
                    return E_new, dt, it, virtual_cutoff_prev, occupied_cutoff_prev


                """
                dE = 1000

                E_prev = a_frag.compute_fragment_energy()

                #conv_stats = []

                e_mp2.append(E_prev)
                de_mp2.append(0)
                n_virtuals_.append(a_frag.n_virtual_tot)
                virtu_cut_.append(a_frag.virtual_cutoff)
                n_occupieds_.append(a_frag.n_occupied_tot)
                occu_cut_.append(a_frag.occupied_cutoff)
                amp_norm.append(np.linalg.norm(a_frag.t2))
                """

                # Converge occupied space / buffer amplitudes
                dE_o = 1000
                E_prev_o = E_prev
                while dE_o>args.fot*0.1:
                    # Expand occupied AOS space

                    E_new_o, dt, it, virtual_cutoff_prev, occupied_cutoff_prev = expand_fragment_space(a_frag, args, s_virt, E_prev, occupied = True)

                    dE_o = np.abs(E_prev_o- E_new_o)

                    E_prev_o = E_new_o

                    print("_________________________________________________________")
                    print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new_o, dE_o))
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in bytes): %.2f" % ib.nbytes())
                    print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                    print(" ",flush=True)

                # When converged, take one step back
                a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)




                fitting_function = lambda x,a,b,c : a*np.exp(b*x**-c) #fitting function for error estimate

                # Expand virtual space to FOT

                dE_v = 1000

                E_prev_v = E_prev_o

                dE_estimate = 1000

                #args.orb_increment = 2 # FIX HARDCODED increment

                estimated_pts = 0

                while dE_estimate>args.fot:

                    # Expand virtual space

                    E_new_v, dt, it, virtual_cutoff_prev, occupied_cutoff_prev = expand_fragment_space(a_frag, args, s_virt, E_prev, virtual = True)

                    dE_v = np.abs(E_prev_v - E_new_v)


                    E_prev_v = E_new_v









                    #conv_stats.append([dt, it, E_new, dE_v,a_frag.virtual_cutoff, a_frag.n_virtual_tot,a_frag.occupied_cutoff, a_frag.n_occupied_tot])
                    print("_________________________________________________________")
                    print("E(fragment): %.8e        DE(fragment): %.8e" % (E_new_v, dE_v))
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))

                    if args.error_estimate:

                        try:
                            p_, cov = optimize.curve_fit(fitting_function, np.array(virtu_cut_), np.array(e_mp2))
                            if estimated_pts>=2:
                                dE_estimate = np.abs(E_new_v - p_[0])

                            print("Estimated error in energy            : %.6e" % np.abs(E_new_v - p_[0]))
                            estimated_pts += 1

                        except:
                            print("Unable to estimate error.")
                            print(" ")
                        print("estimated_pts", estimated_pts)


                    print("Current memory usage of integrals (in bytes): %.2f" % ib.nbytes())
                    print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                    print(" ")
                    print("_________________________________________________________")

                    e_mp2.append(E_new_v)
                    de_mp2.append(dE_estimate)
                    n_virtuals_.append(a_frag.n_virtual_tot)
                    virtu_cut_.append(a_frag.virtual_cutoff)
                    n_occupieds_.append(a_frag.n_occupied_tot)
                    occu_cut_.append(a_frag.occupied_cutoff)
                    amp_norm.append(a_frag.compute_eos_norm())
                    n_iters.append(it)
                    de_res.append(dt)





                # When converged, take one step back
                a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)

                #E_prev_o = a_frag.compute_fragment_energy()

                e_mp2.append(E_prev_v)
                de_mp2.append(dE_estimate)
                n_virtuals_.append(a_frag.n_virtual_tot)
                virtu_cut_.append(a_frag.virtual_cutoff)
                n_occupieds_.append(a_frag.n_occupied_tot)
                occu_cut_.append(a_frag.occupied_cutoff)
                amp_norm.append(a_frag.compute_eos_norm())
                n_iters.append(0)
                de_res.append(dt)

                print("e_mp2 = np.array(",e_mp2, ")")
                print("de_mp2 = np.array(",de_mp2, ")")
                print("n_virt = np.array(",n_virtuals_, ")")
                print("v_dist = np.array(",virtu_cut_, ")")
                print("n_occ = np.array(",n_occupieds_, ")")
                print("v_occ = np.array(",occu_cut_, ")")
                print("t2_norm = np.array(",amp_norm, ")")

                e_mp2 = np.array(e_mp2)
                de_mp2 = np.array(de_mp2)
                n_virtuals_ = np.array(n_virtuals_)
                virtu_cut_ = np.array(virtu_cut_)
                occu_cut_ = np.array(occu_cut_)
                n_occupieds_ = np.array(n_occupieds_)
                amp_norm = np.array(amp_norm)
                n_iters = np.array(n_iters)
                de_res = np.array(de_res)

                np.save("fragment_optim_data_%i.npy" % fragment[0], np.array([e_mp2, de_mp2, n_virtuals_, virtu_cut_, n_occupieds_, occu_cut_, amp_norm, n_iters, de_res]))


                plot_convergence_fig(e_mp2, de_mp2, n_virtuals_, virtu_cut_, n_occupieds_, occu_cut_, p_, FOT = args.fot,t2_norm = amp_norm, i = fragment[0], n_iters = n_iters, de_res = de_res)




                # experimental error estimate
                # p_, cov = optimize.curve_fit(fitting_function, virtu_cut_**-1, e_mp2)


                print("=========================================================")
                print("Final fragment containing occupied orbitals:", a_frag.fragment)
                print("Converged fragment energy: %.12f" % E_prev_v)
                print("Estimated error in energy: %.12f" % np.abs(E_prev_v - p_[0]))
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                print("=========================================================")
                print(" ")
                print(" ")
                refcell_fragments.append(a_frag)
                #fragment_errors.append(dE)
                fragment_errors.append(np.abs(E_prev_v - p_[0]))
                fragment_energy_total += E_prev_v




        for fa in np.arange(len(refcell_fragments)):
            print(fa, refcell_fragments[fa].compute_fragment_energy(), fragment_errors[fa])


        fragment_errors = np.array(fragment_errors)

        print("Total fragment energy:", fragment_energy_total, "+/-", np.sqrt(np.sum(fragment_errors**2)))




        if False:

            # LiH_specific run
            pair_energies = []
            pair_distances = []
            pair_coords = tp.lattice_coords([10,10,10])
            pair_coords = pair_coords[np.argsort(np.sum(p.coor2vec(pair_coords)**2, axis = 1))[0:]] #Sort in increasing distance
            pair_total = 0

            #pair_coords = np.array([[1,0,0], [-1,0,0]])

            #domain = tp.lattice_coords([10,0,0])

            #print(domain)
            frag_a = refcell_fragments[0]
            frag_b = refcell_fragments[1]
            pos_a = wcenters[refcell_fragments[0].fragment[0]]
            pos_b = wcenters[refcell_fragments[1].fragment[0]]


            for c in pair_coords:

                pair = pair_fragment_amplitudes(frag_a, frag_b, M = c, recycle_integrals = args.recycle_integrals, domain_def = args.pair_domain_def)
                rn, it = pair.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh=1e-9)
                print("Convergence:", rn, it)
                p_energy = pair.compute_pair_fragment_energy()

                pair_distances.append(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_a)**2)**.5)
                pair_energies.append(p_energy[0])

                pair_distances.append(0.529177*np.sum((pos_b - p.coor2vec(c) - pos_b)**2)**.5)
                pair_energies.append(p_energy[1])

                pair_distances.append(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5)
                pair_energies.append(p_energy[2])

                pair_distances.append(0.529177*np.sum((pos_b - p.coor2vec(c) - pos_a)**2)**.5)
                pair_energies.append(p_energy[3])

                print("_________________________________________________________")
                print("dist_xdec = np.array(", pair_distances, ")")
                print("e_mp2_xdec = np.array(", pair_energies, ")")
                print(" ----- ")










                """
                for fa in np.arange(len(refcell_fragments)):
                    for fb in np.arange(len(refcell_fragments)):
                        frag_a = refcell_fragments[fa]
                        frag_b = refcell_fragments[fb]

                        pos_a = wcenters[refcell_fragments[fa].fragment[0]]
                        pos_b = wcenters[refcell_fragments[fb].fragment[0]]

                        pair = pair_fragment_amplitudes(frag_a, frag_b, M = c, recycle_integrals = args.recycle_integrals)
                        #print(pair.compute_pair_fragment_energy())
                        rn, it = pair.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh=1e-9)
                        print("Convergence:", rn, it)

                        p_energy = pair.compute_pair_fragment_energy()
                        pair_total += p_energy
                        pair_distances.append(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5)
                        pair_energies.append(p_energy)


                        print(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5, c, fa, fb, p_energy,np.sum(pair.d_ii.blocks[:, frag_a.fragment[0], :]<frag_a.occupied_cutoff), "/", np.sum(pair.d_ia.blocks[:, frag_a.fragment[0], :]<frag_a.virtual_cutoff))
                        print("_________________________________________________________")
                        print(pair_distances)
                        print(pair_energies)
                        print(" ----- ")
                """




        if args.pairs:
            alternative_loop = False

            import copy

            # Outline of pair fragment calcs





            if alternative_loop == False:

                pair_energies = []
                pair_distances = []
                pair_coords = tp.lattice_coords([10,10,10])
                pair_coords = pair_coords[np.argsort(np.sum(p.coor2vec(pair_coords)**2, axis = 1))[0:]] #Sort in increasing distance
                pair_total = 0

                #pair_coords = np.array([[1,0,0], [-1,0,0]])

                domain = tp.lattice_coords([10,0,0])
                #print(domain)

                for c in pair_coords:
                    for fa in np.arange(len(refcell_fragments)):
                        for fb in np.arange(len(refcell_fragments)):
                            frag_a = refcell_fragments[fa]
                            frag_b = refcell_fragments[fb]

                            pos_a = wcenters[refcell_fragments[fa].fragment[0]]
                            pos_b = wcenters[refcell_fragments[fb].fragment[0]]

                            pair = pair_fragment_amplitudes(frag_a, frag_b, M = c, recycle_integrals = args.recycle_integrals, adaptive = args.adaptive_domains,  domain_def = args.pair_domain_def)
                            #print(pair.compute_pair_fragment_energy())
                            rn, it = pair.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh=1e-9)
                            print("Convergence:", rn, it)

                            p_energy = pair.compute_pair_fragment_energy()[2]
                            pair_total += p_energy
                            pair_distances.append(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5)
                            pair_energies.append(p_energy)


                            print(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5, c, fa, fb, p_energy,np.sum(pair.d_ii.blocks[:, frag_a.fragment[0], :]<frag_a.occupied_cutoff), "/", np.sum(pair.d_ia.blocks[:, frag_a.fragment[0], :]<frag_a.virtual_cutoff))
                            print("_________________________________________________________")
                            #print(pair_distances)
                            #print(pair_energies)
                            print("dist_xdec = np.array(", pair_distances, ")")
                            print("e_mp2_xdec = np.array(", pair_energies, ")")
                            print(" ----- ")








            else:
                #alternative pair fragment loop
                def fix_pair_coords(pc):
                    """
                    adds reference cell and removes opposing cells
                    """
                    pc = np.concatenate([np.atleast_2d(np.zeros(3,dtype=int)),pc])
                    ind = np.array(np.where((pc[:,0][:,None] == -pc[:,0])&(pc[:,1][:,None] == -pc[:,1])&(pc[:,2][:,None] == -pc[:,2]))).T
                    ind = np.unique(np.sort(ind,axis=1),axis=0)[:,1]
                    return pc[ind]


                if p.cperiodicity == "POLYMER":
                    pair_coords = tp.lattice_coords([10,0,0])
                elif p.cperiodicity == "SLAB":
                    pair_coords = tp.lattice_coords([10,10,0])
                else:
                    pair_coords = tp.lattice_coords([10,10,10])

                pair_coords = pair_coords[np.argsort(np.sum(p.coor2vec(pair_coords)**2, axis = 1))[1:]] #Sort in increasing distance
                pair_coords = fix_pair_coords(pair_coords)

                pair_total = 0


                for c in pair_coords:
                    for fa in np.arange(len(refcell_fragments)):
                        for fb in np.arange(len(refcell_fragments)):

                            if (c == pair_coords[0]).all() and ((fa == fb) or (fa > fb)):
                                continue

                            frag_a = refcell_fragments[fa]
                            frag_b = refcell_fragments[fb]

                            pair = pair_fragment_amplitudes(frag_a, frag_b, M = c,  domain_def = args.pair_domain_def)


                            #print(pair.compute_pair_fragment_energy())
                            #pair.solve()
                            pair.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh=1e-9)
                            p_energy = pair.compute_pair_fragment_energy()
                            pair_total += p_energy
                            print ()
                            print("Pair fragment energy for ",c," is ", 2*p_energy, " (total: ", pair_total + 0*E_new, " )")
                            print("R = ", np.sum(p.coor2vec(c)**2)**.5, fa, fb)
                            print(fa, fb, np.sum(p.coor2vec(c)**2)**.5)








        #for n in np.arange(10):
        #    pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([0,0,n]))
        #    print(pair.compute_pair_fragment_energy())
        #    pair.solve()
        #    print("Pair fragment energy for (0,0,%i):" %n, pair.compute_pair_fragment_energy())



    if args.fragmentation == "dec":
        virt_cut = args.virtual_cutoff


        if args.occupied_cutoff < args.afrag:
            occ_cut = args.afrag
        else:
            occ_cut = args.occupied_cutoff

        refcell_fragments = []
        fragment_errors = []
        fragment_energy_total = 0

        # Run DEC-fragment optimization

        for fragment in center_fragments:


            #ib.fragment = fragment
            t0 = time.time()
            domain_max = tp.lattice_coords(PRI.n_points_p(p, 20))

            if args.pao_sorting:
                d_ia = build_weight_matrix(p, c, domain_max)
                a_frag = fragment_amplitudes(p, wcenters,domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision, d_ia = d_ia)


            else:
                a_frag = fragment_amplitudes(p, wcenters, domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision)

            #print("Frag init:", time.time()-t0)

            dt, it, E_prev_outer = a_frag.solve(eqtype = args.solver, s_virt = s_virt, damping = args.damping, norm_thresh = args.fot*0.01,energy = "fragment")

            #print("t2 (max/min/absmin):", np.max(a_frag.t2), np.min(a_frag.t2), np.abs(a_frag.t2).min())
            #print("g_d (max/min/absmin):", np.max(a_frag.g_d), np.min(a_frag.g_d), np.abs(a_frag.g_d).min())

            # Converge to fot
            #E_prev_outer = a_frag.compute_fragment_energy()
            E_prev = E_prev_outer*1.0
            dE_outer = 10

            print("Initial fragment energy: %.8f" % E_prev)

            print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
            print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot),flush=True)

            virtual_cutoff_prev = a_frag.virtual_cutoff
            occupied_cutoff_prev = a_frag.occupied_cutoff




            if not args.skip_fragment_optimization:
                print("Running fragment optimization for:")
                print(fragment)
                #print("Initial cutoffs:")

                n_virtuals_  = []
                virtu_cut_   = []
                n_occupieds_ = []
                occu_cut_    = []
                e_mp2        = []
                de_mp2       = []
                amp_norm     = []
                n_iters      = [] # number of mp2 iterations
                de_res       = [] # max deviation in residual




                dE = 10

                # Update state
                e_mp2.append(E_prev)
                de_mp2.append(dE)
                n_virtuals_.append(a_frag.n_virtual_tot)
                virtu_cut_.append(a_frag.virtual_cutoff)
                n_occupieds_.append(a_frag.n_occupied_tot)
                occu_cut_.append(a_frag.occupied_cutoff)
                amp_norm.append(a_frag.compute_eos_norm())
                n_iters.append(0)
                de_res.append(0)


                def expand_fragment_space(a_frag, args, s_virt, E_prev, occupied = False, virtual = False):
                    # Expand virtual space

                    virtual_cutoff_prev = a_frag.virtual_cutoff
                    occupied_cutoff_prev = a_frag.occupied_cutoff

                    t_0 = time.time()

                    if virtual:
                        a_frag.autoexpand_virtual_space(n_orbs=args.orb_increment)
                    if occupied:
                        a_frag.autoexpand_occupied_space(n_orbs=args.orb_increment)
                    t_1 = time.time()
                    dt, it, E_new  = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.01, damping = args.damping, energy = "fragment")
                    t_2 = time.time()
                    #E_new = a_frag.compute_fragment_energy()
                    t_3 = time.time()
                    dE = np.abs(E_prev - E_new)
                    print("_________________________________________________________")
                    print("E(fragment): %.8f      DE(fragment): %.8e" % (E_new, dE))
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in bytes): %.2f" % ib.nbytes())
                    print("Time (expand/solve/energy) (s) : %.1f / %.1f / %.1f" % (t_1-t_0, t_2-t_1, t_3-t_2))
                    print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                    print(" ",flush=True)
                    return E_new, dt, it, virtual_cutoff_prev, occupied_cutoff_prev


                """
                dE = 1000

                E_prev = a_frag.compute_fragment_energy()

                #conv_stats = []

                e_mp2.append(E_prev)
                de_mp2.append(0)
                n_virtuals_.append(a_frag.n_virtual_tot)
                virtu_cut_.append(a_frag.virtual_cutoff)
                n_occupieds_.append(a_frag.n_occupied_tot)
                occu_cut_.append(a_frag.occupied_cutoff)
                amp_norm.append(np.linalg.norm(a_frag.t2))
                """
                fitting_function = lambda x,a,b,c : a*np.exp(b*x**-c) #fitting function for error estimate




                while dE>args.fot:

                    #E_new = a_frag.compute_fragment_energy()
                    estimated_pts = 0


                    dE_v = 1000

                    E_prev_v = E_prev

                    virtu_cut_ee = []
                    e_mp2_ee = []

                    e_mp2_ee.append(E_prev)
                    virtu_cut_ee.append(a_frag.virtual_cutoff)

                    while dE_v>args.fot:
                        # Expand virtual space

                        E_new_v, dt, it, virtual_cutoff_prev, occupied_cutoff_prev = expand_fragment_space(a_frag, args, s_virt, E_prev, virtual = True)

                        dE_v = np.abs(E_prev_v - E_new_v)


                        E_prev_v = E_new_v


                        """

                        print("_________________________________________________________")
                        print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new_v, dE_v))
                        print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                        print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                        print("_________________________________________________________")
                        print("Current memory usage of integrals (in bytes): %.2f" % ib.nbytes())
                        print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                        print(" ",flush=True)
                        """

                        e_mp2.append(E_new_v)
                        de_mp2.append(dE_v)
                        n_virtuals_.append(a_frag.n_virtual_tot)
                        virtu_cut_.append(a_frag.virtual_cutoff)
                        n_occupieds_.append(a_frag.n_occupied_tot)
                        occu_cut_.append(a_frag.occupied_cutoff)
                        amp_norm.append(a_frag.compute_eos_norm())
                        n_iters.append(it)
                        de_res.append(dt)

                        e_mp2_ee.append(E_new_v)
                        virtu_cut_ee.append(a_frag.virtual_cutoff)

                        #conv_stats.append([dt, it, E_new, dE_v,a_frag.virtual_cutoff, a_frag.n_virtual_tot,a_frag.occupied_cutoff, a_frag.n_occupied_tot])
                        if args.error_estimate:
                            try:
                                p_, cov = optimize.curve_fit(fitting_function, np.array(virtu_cut_ee), np.array(e_mp2_ee))
                                if estimated_pts>=2:
                                    dE_estimate = np.abs(E_new_v - p_[0])

                                print("Estimated error in energy            : %.6e" % np.abs(E_new_v - p_[0]))
                                estimated_pts += 1

                            except:
                                print("Unable to estimate error.")
                                print(" ")




                    # When converged, take one step back
                    a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)

                    #E_prev_o = a_frag.compute_fragment_energy()
                    dt, it, E_prev_o = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.01, damping = args.damping, energy = "fragment")

                    e_mp2.append(E_prev_o)
                    de_mp2.append(dE_v)
                    n_virtuals_.append(a_frag.n_virtual_tot)
                    virtu_cut_.append(a_frag.virtual_cutoff)
                    n_occupieds_.append(a_frag.n_occupied_tot)
                    occu_cut_.append(a_frag.occupied_cutoff)
                    amp_norm.append(a_frag.compute_eos_norm())
                    n_iters.append(0)
                    de_res.append(0)



                    #conv_stats.append([dt, it, E_new, dE_v,a_frag.virtual_cutoff, a_frag.n_virtual_tot,a_frag.occupied_cutoff, a_frag.n_occupied_tot])


                    dE_o = 1000

                    while dE_o>args.fot:
                        # Expand occupied space

                        E_new_o, dt, it, virtual_cutoff_prev, occupied_cutoff_prev = expand_fragment_space(a_frag, args, s_virt, E_prev, occupied = True)

                        dE_o = np.abs(E_prev_o- E_new_o)

                        E_prev_o = E_new_o

                        """

                        print("_________________________________________________________")
                        print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new_o, dE_o))
                        print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                        print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                        print("_________________________________________________________")
                        print("Current memory usage of integrals (in bytes): %.2f" % ib.nbytes())
                        print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                        print(" ",flush=True)
                        """

                        e_mp2.append(E_new_o)
                        de_mp2.append(dE_o)
                        n_virtuals_.append(a_frag.n_virtual_tot)
                        virtu_cut_.append(a_frag.virtual_cutoff)
                        n_occupieds_.append(a_frag.n_occupied_tot)
                        occu_cut_.append(a_frag.occupied_cutoff)
                        amp_norm.append(a_frag.compute_eos_norm())
                        n_iters.append(it)
                        de_res.append(dt)
                        #conv_stats.append([dt, it, E_new, dE_v,a_frag.virtual_cutoff, a_frag.n_virtual_tot,a_frag.occupied_cutoff, a_frag.n_occupied_tot])

                    # When converged, take one step back
                    a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)

                    dt, it, E_new = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.01, damping = args.damping, energy = "fragment")


                    dE = np.abs(E_prev - E_new)

                    E_prev = E_new

                    # Update state
                    e_mp2.append(E_prev)
                    de_mp2.append(dE)
                    n_virtuals_.append(a_frag.n_virtual_tot)
                    virtu_cut_.append(a_frag.virtual_cutoff)
                    n_occupieds_.append(a_frag.n_occupied_tot)
                    occu_cut_.append(a_frag.occupied_cutoff)
                    amp_norm.append(a_frag.compute_eos_norm())
                    n_iters.append(0)
                    de_res.append(0)





                if args.print_level >2:
                    print("e_mp2 = np.array(",e_mp2, ")")
                    print("de_mp2 = np.array(",de_mp2, ")")
                    print("n_virt = np.array(",n_virtuals_, ")")
                    print("v_dist = np.array(",virtu_cut_, ")")
                    print("n_occ = np.array(",n_occupieds_, ")")
                    print("v_occ = np.array(",occu_cut_, ")")
                    print("t2_norm = np.array(",amp_norm, ")",flush=True)

                e_mp2 = np.array(e_mp2)
                de_mp2 = np.array(de_mp2)
                n_virtuals_ = np.array(n_virtuals_)
                virtu_cut_ = np.array(virtu_cut_)
                occu_cut_ = np.array(occu_cut_)
                n_occupieds_ = np.array(n_occupieds_)
                amp_norm = np.array(amp_norm)
                n_iters = np.array(n_iters)
                de_res = np.array(de_res)

                np.save("fragment_optim_data_%i.npy" % fragment[0], np.array([e_mp2, de_mp2, n_virtuals_, virtu_cut_, n_occupieds_, occu_cut_, amp_norm, n_iters, de_res]))

                try:
                    plot_convergence_fig(e_mp2, de_mp2, n_virtuals_, virtu_cut_, n_occupieds_, occu_cut_, p_, FOT = args.fot,t2_norm = amp_norm, i = fragment[0], n_iters = n_iters, de_res = de_res)
                except:
                    plot_convergence_fig(e_mp2, de_mp2, n_virtuals_, virtu_cut_, n_occupieds_, occu_cut_, FOT = args.fot,t2_norm = amp_norm, i = fragment[0], n_iters = n_iters, de_res = de_res)





                #p_, cov = optimize.curve_fit(fitting_function, virtu_cut_**-1, e_mp2)


                print("=========================================================")
                print("Final fragment containing occupied orbitals:", a_frag.fragment)
                print("Converged fragment energy: %.12f" % E_new)
                #print("Estimated error in energy: %.12f" % np.abs(E_new - p_[0]))
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                print("=========================================================")
                print(" ")
                print(" ",flush=True)
                refcell_fragments.append(a_frag)
                fragment_errors.append(dE)
                #fragment_errors.append(np.abs(E_new - p_[0]))
                fragment_energy_total += E_new

                # experimental error estimate



        print(" ")
        print("Fragment energies")
        for fa in np.arange(len(refcell_fragments)):
            #print(fa, ,)
            print(fa,"%.10e" % refcell_fragments[fa].compute_fragment_energy() ,"+/-",  "%.5e" % fragment_errors[fa],  " Ha / virt %.2f bohr ( n=%i ) / occ %.2f bohr ( n=%i )" %  (refcell_fragments[fa].virtual_cutoff, refcell_fragments[fa].n_virtual_tot,
                                                                                  refcell_fragments[fa].occupied_cutoff, refcell_fragments[fa].n_occupied_tot))




        fragment_errors = np.array(fragment_errors)

        print("Total fragment energy:", fragment_energy_total, "+/-", np.sqrt(np.sum(fragment_errors**2)))
        print(" ")

        ib.d_forget = [] # Ensure that only integrals beyond here are forgotten


        if args.pairs:
            import copy

            # Outline of pair fragment calcs
            print(flush=True)

            if args.pair_setup == 'standard':

                pair_energies = []
                pair_distances = []
                pair_coords = tp.lattice_coords([10,10,10])
                pair_coords = pair_coords[np.argsort(np.sum(p.coor2vec(pair_coords)**2, axis = 1))[0:]] #Sort in increasing distance
                pair_total = 0

                #pair_coords = np.array([[1,0,0], [-1,0,0]])

                domain = tp.lattice_coords([10,0,0])
                #print(domain)

                for c in pair_coords:
                    for fa in np.arange(len(refcell_fragments)):
                        for fb in np.arange(len(refcell_fragments)):
                            #if fa==1:
                            frag_a = refcell_fragments[fa]
                            frag_b = refcell_fragments[fb]

                            pos_a = wcenters[refcell_fragments[fa].fragment[0]]
                            pos_b = wcenters[refcell_fragments[fb].fragment[0]]

                            pair = pair_fragment_amplitudes(frag_a, frag_b, M = c, recycle_integrals = args.recycle_integrals, adaptive = args.adaptive_domains,  domain_def = args.pair_domain_def)
                            #print(pair.compute_pair_fragment_energy())
                            rn, it, p_energies = pair.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh=args.fot*0.01, ndiis = args.ndiis, energy = "pair", damping = args.damping)

                            #rn, it, p_energy = pair.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh=1e-9, ndiis = args.ndiis)
                            print("Convergence:", rn, it)

                            #p_energy = pair.compute_pair_fragment_energy()[2]
                            #pair_total += p_energy
                            #pair_distances.append(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5)
                            print(p_energies)
                            pair_energies.append(p_energies[0])
                            print(np.sum(pair_energies))
                            print("Memory usage (bytes)", pair.nbytes())
                            


                            print(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5, c, fa, fb, p_energies,np.sum(pair.d_ii.blocks[:, frag_a.fragment[0], :]<frag_a.occupied_cutoff), "/", np.sum(pair.d_ia.blocks[:, frag_a.fragment[0], :]<frag_a.virtual_cutoff))
                            print("_________________________________________________________")
                            #print(pair_distances)
                            #print(pair_energies)
                            print("dist_xdec = np.array(", pair_distances, ")")
                            print("e_mp2_xdec = np.array(", pair_energies, ")")
                            print(" ----- ",flush=True)





            else:
                def fix_pair_coords(pc):
                    """
                    adds reference cell and removes opposing cells
                    """
                    pc = np.concatenate([np.atleast_2d(np.zeros(3,dtype=int)),pc])
                    ind = np.array(np.where((pc[:,0][:,None] == -pc[:,0])&(pc[:,1][:,None] == -pc[:,1])&(pc[:,2][:,None] == -pc[:,2]))).T
                    ind = np.unique(np.sort(ind,axis=1),axis=0)[:,1]
                    return pc[ind]


                if p.cperiodicity == "POLYMER":
                    pair_coords = tp.lattice_coords([10,0,0])
                elif p.cperiodicity == "SLAB":
                    pair_coords = tp.lattice_coords([10,10,0])
                else:
                    pair_coords = tp.lattice_coords([10,10,10])

                pair_coords = pair_coords[np.argsort(np.sum(p.coor2vec(pair_coords)**2, axis = 1))[1:]] #Sort in increasing distance
                pair_coords = fix_pair_coords(pair_coords)

                pair_energies = []
                pair_distances = []

                pair_total = 0

                if args.pair_setup == 'alternative':


                    for c in pair_coords:
                        for fa in np.arange(len(refcell_fragments)):
                            for fb in np.arange(len(refcell_fragments)):

                                if (c == pair_coords[0]).all() and ((fa == fb) or (fa > fb)):
                                    continue

                                frag_a = refcell_fragments[fa]
                                frag_b = refcell_fragments[fb]

                                print ('Calculating pair: ',fa,fb,c)

                                pos_a = wcenters[refcell_fragments[fa].fragment[0]]
                                pos_b = wcenters[refcell_fragments[fb].fragment[0]]

                                pair = pair_fragment_amplitudes(frag_a, frag_b, M = c, recycle_integrals = args.recycle_integrals, adaptive = args.adaptive_domains, retain_integrals = args.retain_integrals,  domain_def = args.pair_domain_def)
                                #print(pair.compute_pair_fragment_energy())
                                #print()
                                #print("Memory profile before optim:")
                                #pair.memory_profile()
                                rn, it = pair.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh=1e-9, ndiis = args.ndiis)
                                
                                print("Convergence:", rn, it)

                                p_energy = pair.compute_pair_fragment_energy()[2]
                                pair_total += p_energy
                                pair_distances.append(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5)
                                pair_energies.append(p_energy)


                                


                                #print(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5, c, fa, fb, p_energy,np.sum(pair.d_ii.blocks[:, frag_a.fragment[0], :]<frag_a.occupied_cutoff), "/", np.sum(pair.d_ia.blocks[:, frag_a.fragment[0], :]<frag_a.virtual_cutoff))
                                print("_________________________________________________________")
                                print("Pair distance: ",0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5)
                                print("Pair energy:   ",p_energy,'          (total: ',pair_total,')')
                                #print(pair_distances))
                                #print(pair_energies)
                                #print("dist_xdec = np.array(", pair_distances, ")")
                                #print("e_mp2_xdec = np.array(", pair_energies, ")")
                                print("Memory usage (bytes)", pair.nbytes())
                                p
                                print(" ----- ")
                                print(flush=True)




                elif args.pair_setup == 'auto':
                    import pairs

                    PD = pairs.setup_pairs(p,center_fragments,pair_coords,wcenters,spreads,args.fot)

                    n_pairs = 0

                    cprev = None

                    pair = None #to be updated

                    time_total = []

                    while not PD.conv:
                        fa,fb,c_ind,dist = PD.get_pair()
                        c = pair_coords[c_ind]

                        frag_a = refcell_fragments[fa]
                        frag_b = refcell_fragments[fb]

                        print ('Calculating pair: ',fa,fb,c)

                        t0 = time.time()

                        pair = pair_fragment_amplitudes(frag_a, frag_b, M = c, recycle_integrals = args.recycle_integrals, adaptive = args.adaptive_domains, retain_integrals = args.retain_integrals,  domain_def = args.pair_domain_def, old_pair = pair)
                        
                        t1 = time.time()

                        

                        if cprev is None:
                            cprev = c*1
                        else:
                            if np.sum((c-cprev)**2)!=0:
                                print("Forget pair integrals")

                                ib.forget(retain = pair.d_ia.coords[:pair.n_virtual_cells]) #Clear fitting coeffs not used by pair (from dyn buffer)
                            
                            cprev = c*1
                        


                        t2 = time.time()


                        
                        #print(pair.compute_pair_fragment_energy())
                        #del(pair)
                        #pair = 0
                        #print("Memory profile pre optim:")
                        #pair.memory_profile()
                        
                        
                        #rn, it = 0,0, #pair.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh=1e-9, ndiis = args.ndiis)
                        #p_energies = [0,0,0,0] #pair.compute_pair_fragment_energy()

                        rn, it, p_energies = pair.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh=args.fot*0.01, ndiis = args.ndiis, energy = "pair", damping = args.damping)

                        #rn, it  = pair.solve_MP2(norm_thresh=args.fot*0.01, ndiis = args.ndiis)

                        t3 = time.time()


                        print("Absolute deviation in residual:", rn, "Number of iterations:", it)
                        #print(p_energies)
                        
                        #p_energies = pair.compute_pair_fragment_energy()
                        
                        print(p_energies)
                        #print("Alternative energies:")
                        #print(pair.compute_pair_fragment_energy())
                        #print(pair.compute_pair_fragment_energy_())
                        #print("Done computing energy")

                        t4 = time.time()

                        print("Performance")
                        print("Setup :", t1-t0, "s.")
                        print("Sort  :", t2-t1, "s.")
                        print("Solve :", t3-t2, "s.")
                        #print("Energy:", t4-t3, "s.")
                        #time_total.append(t4-t0)
                        #print("Total :", time_total)

                        #print(" We are here!")
                        print("Screening:", np.sum(ib.Xscreen<ib.screening_thresh))


                        

                        



                        
                        p_energy = p_energies[2]
                        pair_total += p_energy
                        pair_distances.append(dist)
                        pair_energies.append(p_energy)

                        PD.add([fa,fb,c_ind,p_energy],p_energies)
                        PD.estim_remainE()

                        #print(0.529177*np.sum((pos_a - p.coor2vec(c) - pos_b)**2)**.5, c, fa, fb, p_energy,np.sum(pair.d_ii.blocks[:, frag_a.fragment[0], :]<frag_a.occupied_cutoff), "/", np.sum(pair.d_ia.blocks[:, frag_a.fragment[0], :]<frag_a.virtual_cutoff))
                        print("_________________________________________________________")
                        print ('pair_energies: ',p_energies)
                        print("Pair distance: ",dist)
                        print("Pair energy:   ",p_energy,'          (total: ',pair_total,')')
                        #print(pair_distances))
                        #print(pair_energies)
                        #print("dist_xdec = np.array(", pair_distances, ")")
                        #print("e_mp2_xdec = np.array(", pair_energies, ")")
                        print("Integrator memory usage:", ib.nbytes(), "(bytes)")
                        print ()
                        PD.P.print_de()
                        print ()
                        print(" ----- ")
                        print(flush=True)
                        #print("Memory profile post optim:")
                        #pair.memory_profile()
                        #pair.deallocate()

                        #gc.collect()

                        #del(pair.t2)
                        #del(pair.g_d)
                        #del(pair.f1)
                        #del(pair.f2)
                        #del(pair.ib)
                        #del(pair)
                        #del(frag_a)
                        #del(frag_b)
                        
                        

                        

                        n_pairs += 1

                    print ()
                    print ('Distance, pair energy: ')
                    PD.P.print_de()
                    print ()
                    print ('Number of calculated pairs: ',n_pairs)
                    print ('Estimated non-calculated pair energy: ',PD.remainE)
                    print ()
                    print ('Total pair energy:          ',pair_total)
                    print ('Total fragment energy:      ',fragment_energy_total)
                    print ('Total correlation energy:   ',pair_total+fragment_energy_total,flush=True)
                    #PD.P.save_de() #saving distance_energy to file

                else:
                    print ('WARNING: the input for pair_setup is not recognized. Pairs will not be calculated.')








        #for n in np.arange(10):
        #    pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([0,0,n]))
        #    print(pair.compute_pair_fragment_energy())
        #    pair.solve()
        #    print("Pair fragment energy for (0,0,%i):" %n, pair.compute_pair_fragment_energy())

    if args.fragmentation == "cim-adaptive":
        """
        cluster-in-molecule like scheme
        """

        virt_cut = 6.0
        occ_cut = 2.0

        refcell_fragments = []
        energies = []
        fragment_energy_total = 0

        for fragment in center_fragments:


            #ib.fragment = fragment
            t0 = time.time()
            domain_max = tp.lattice_coords(PRI.n_points_p(p, 20))

            if args.pao_sorting:
                d_ia = build_weight_matrix(p, c, domain_max)
                a_frag = fragment_amplitudes(p, wcenters,domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision, d_ia = d_ia)


            else:
                a_frag = fragment_amplitudes(p, wcenters, domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision)

            #print("Frag init:", time.time()-t0)

            dt, it, E_prev_outer = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.1, damping = args.damping, energy = "cim")

            #print("t2 (max/min/absmin):", np.max(a_frag.t2), np.min(a_frag.t2), np.abs(a_frag.t2).min())
            #print("g_d (max/min/absmin):", np.max(a_frag.g_d), np.min(a_frag.g_d), np.abs(a_frag.g_d).min())

            # Converge to fot
            #E_prev_outer = a_frag.compute_energy(exchange = False)
            E_prev = E_prev_outer*1.0
            dE_outer = 10

            virtual_cutoff_prev = a_frag.virtual_cutoff
            occupied_cutoff_prev = a_frag.occupied_cutoff

            print("_________________________________________________________")
            print("Initial cluster domain for fragment:")
            print(fragment)
            print(" ")
            print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
            print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
            print("E(CIM): %.8f      DE(fragment): %.8e" % (E_prev, dE_outer))
            print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
            print("_________________________________________________________")






            #print("Running fragment optimization for:")
            #print(fragment)
            #print("Initial cutoffs:")

            n_virtuals_ = []
            virtu_cut_  = []



            while dE_outer>args.fot:
                dE = 10
                e_virt = []

                while dE>args.fot:
                    virtual_cutoff_prev = a_frag.virtual_cutoff
                    occupied_cutoff_prev = a_frag.occupied_cutoff

                    t_0 = time.time()
                    a_frag.autoexpand_occupied_space(n_orbs=args.orb_increment)

                    print("Expanded occupied space")
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))

                    t_1 = time.time()
                    dt, it, E_new = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.001, damping = args.damping, energy = "cim")
                    t_2 = time.time()
                    #E_new = a_frag.compute_energy(exchange = False)
                    t_3 = time.time()
                    dE = np.abs(E_prev - E_new)

                    print("_________________________________________________________")
                    print("E(CIM): %.8f      DE(fragment): %.8e" % (E_new, dE))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in bytes): %.2f" % ib.nbytes())
                    print("Time (expand/solve/energy) (s) : %.1f / %.1f / %.1f" % (t_1-t_0, t_2-t_1, t_3-t_2))
                    print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                    print(" ")
                    E_prev = E_new


                #a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)



                print("Converged occupied space for this iteration")
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                print("=========================================================")

                dE = 10
                while dE>args.fot:
                    virtual_cutoff_prev = a_frag.virtual_cutoff
                    occupied_cutoff_prev = a_frag.occupied_cutoff

                    t_0 = time.time()
                    a_frag.autoexpand_virtual_space(n_orbs=args.orb_increment)

                    print("Expanded occupied space")
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))

                    t_1 = time.time()
                    dt, it, E_new = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.001, damping = args.damping, energy = "cim")
                    t_2 = time.time()
                    #E_new = a_frag.compute_energy(exchange = False)
                    t_3 = time.time()
                    dE = np.abs(E_prev - E_new)

                    print("_________________________________________________________")
                    print("E(CIM): %.8f      DE(fragment): %.8e" % (E_new, dE))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in bytes): %.2f" % ib.nbytes())
                    print("Time (expand/solve/energy) (s) : %.1f / %.1f / %.1f" % (t_1-t_0, t_2-t_1, t_3-t_2))
                    print("Max.dev. residual: %.2e . Number of iterations: %i" % (dt, it))
                    print(" ")
                    E_prev = E_new

                print("Converged virtual space for this iteration")
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                print("=========================================================")



                #a_frag.set_extent(virtual_cutoff_prev, occupied_cutoff_prev)
                dE_outer = np.abs(E_prev_outer - E_prev)
                print("dE_outer:", dE_outer)
                E_prev_outer = E_prev
            #print("Current memory usage of integrals (in MB):", ib.nbytes())
            #E_final = a_frag.compute_energy(exchange = True)
            print("_________________________________________________________")
            print("Final cluster containing occupied orbitals:", a_frag.fragment)
            print("Converged CIM energy: %.12f" % E_new)
            print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
            print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
            print("_________________________________________________________")
            print(" ")
            print(" ")
            refcell_fragments.append(a_frag)
            energies.append(E_new)
            fragment_energy_total += E_new
        for i in np.arange(len(energies)):
            print("     E_{Fragment %i} :" %i, energies[i], " Hartree")
        print("_________________________________________________________")
        print("Total cluster energy :", fragment_energy_total, " Hartree")
        print("=========================================================")

    if args.fragmentation == "cim":
        """
        cluster-in-molecule scheme
        """

        virt_cut = args.virtual_cutoff
        occ_cut = args.occupied_cutoff

        refcell_fragments = []
        energies = []
        fragment_energy_total = 0

        #complete fragmentation of the occupied space
        #center_fragments = [[i] for i in np.arange(p.get_nocc()+args.n_core)[args.n_core:]]
        #center_fragments = [[i] for i in np.arange(p.get_nocc())]
        #print(center_fragments)

        for fragment in center_fragments:


            #ib.fragment = fragment
            t0 = time.time()
            domain_max = tp.lattice_coords(PRI.n_points_p(p, 20))

            if args.pao_sorting:
                d_ia = build_weight_matrix(p, c, domain_max)
                a_frag = fragment_amplitudes(p, wcenters,domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision, d_ia = d_ia)


            else:
                a_frag = fragment_amplitudes(p, wcenters, domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision)

            #print("Frag init:", time.time()-t0)

            dt, it, E_prev_outer, E_pairwise = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = 1e-10, damping = args.damping, energy = "cim", pairwise = True)


            print(dt, it)
            print("shape:", a_frag.g_d.shape)
            # Converge to fot
            #E_prev_outer = a_frag.compute_energy(exchange = True)
            #E_prev_outer = a_frag.compute_cim_energy(exchange = True)

            #print("fragment_energy:", a_frag.compute_fragment_energy())
            E_prev = E_prev_outer*1.0
            dE_outer = 10

            virtual_cutoff_prev = a_frag.virtual_cutoff
            occupied_cutoff_prev = a_frag.occupied_cutoff

            print("_________________________________________________________")
            print("Initial cluster domain for fragment:")
            print(fragment)
            print(" ")
            print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
            print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
            print("E(CIM): %.8f      DE(fragment): %.8e" % (E_prev, dE_outer))
            print("dt , it:", dt, it)
            print("_________________________________________________________")




            refcell_fragments.append(a_frag)
            fragment_energy_total += E_prev_outer
            energies.append(E_prev_outer)
            np.save("pair_info_%i.npy" % fragment[0], E_pairwise)
        for i in np.arange(len(energies)):
            print("     E_{Fragment %i} :" %i, energies[i], " Hartree")
        print("_________________________________________________________")
        print("Total cluster energy :", fragment_energy_total, " Hartree")
        #print("Total cluster energy : %.4e" % fragment_energy_total, " Hartree")
        
        print("=========================================================")
        print("Pairwise energycontributions stored to disk.")


    if args.fragmentation == "cim-sweep":
        """
        cluster-in-molecule scheme
        """

        dO = 9
        dV = 9

        N = 100

        v_range = np.linspace(args.virtual_cutoff, args.virtual_cutoff + dV, N)
        o_range = np.linspace(args.occupied_cutoff, args.occupied_cutoff + dO, N)

        energies = np.zeros((len(center_fragments),N,N), dtype = np.float)

        domain_max = tp.lattice_coords(PRI.n_points_p(p, 20))

        for f in np.arange(len(center_fragments)):
            fragment = center_fragments[f]
            a_frag = fragment_amplitudes(p, wcenters, domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = v_range[i], occupied_cutoff = o_range[0], float_precision = args.float_precision, store_exchange = True)
            nv = a_frag.n_virtual_tot
            no = a_frag.n_occupied_tot
            eng = a_frag.compute_cim_energy(exchange = True)
            for i in np.arange(len(v_range)):
                
                
                for j in np.arange(len(o_range)):
                    
                    a_frag.set_extent(v_range[i], o_range[j])
                    if nv == a_frag.n_virtual_tot and no == a_frag.n_occupied_tot:
                        energies[f,i,j] = eng
                    else:
                    
                        dt, it = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = 1e-10, damping = args.damping)


                        #print("Convegence (dt, it):", dt, it)
                        #print("shape:", a_frag.g_d.shape)
                        # Converge to fot
                        eng =  a_frag.compute_cim_energy(exchange = True)
                        energies[f,i,j] = eng

                        nv = a_frag.n_virtual_tot*1
                        no = a_frag.n_occupied_tot*1

                        
                        
                        



                        print("_________________________________________________________")
                        print("Fragment:", fragment)
                        print("Sweep   :", i,j)
                        print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                        print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                        print("E(CIM): %.8f      DE(fragment): %.8e" % ( energies[f,i,j], 0.0))
                        print("dt , it:", dt, it)
                        print("_________________________________________________________")

                    


                
                #print("Total fragment energy:", fragment_energy_total)
                np.save("cim_energies.npy", energies)
                np.save("cim_occ_cuts.npy", o_range)
                np.save("cim_vrt_cuts.npy", v_range)


            
    


    #all_objects = muppy.get_objects()
    #sum1 = summary.summarize(all_objects)
    # Prints out a summary of the large objects
    #summary.print_(sum1)
    # Get references to certain types of objects such as dataframe
    """
    dataframes = [ao for ao in all_objects if isinstance(ao, pd.DataFrame)]
    for d in dataframes:
        print(d.columns.values)
        print(len(d))
    """
    if args.fragmentation == "fullspace":
        """
        cluster-in-molecule scheme
        """

        virt_cut = args.virtual_cutoff
        occ_cut = args.occupied_cutoff

        refcell_fragments = []
        fragment_energy_total = 0

        #complete fragmentation of the occupied space
        #center_fragments = [[i] for i in np.arange(p.get_nocc()+args.n_core)[args.n_core:]]
        #center_fragments = [[i] for i in np.arange(p.get_nocc())]
        #print(center_fragments)

        for fragment in center_fragments:


            #ib.fragment = fragment
            t0 = time.time()
            domain_max = tp.lattice_coords(PRI.n_points_p(p, 20))

            if args.pao_sorting:
                d_ia = build_weight_matrix(p, c, domain_max)
                a_frag = fragment_amplitudes(p, wcenters,domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision, d_ia = d_ia)


            else:
                a_frag = fragment_amplitudes(p, wcenters, domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = virt_cut, occupied_cutoff = occ_cut, float_precision = args.float_precision)

            #print("Frag init:", time.time()-t0)

            tt = 100

            convergence = []


            for i in np.arange(100):
                a_frag.autoexpand_occupied_space(n_orbs=args.orb_increment)

                dt, it = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = 1e-10, damping = args.damping)


                print(dt, it)
                print("shape:", a_frag.g_d.shape)
                tt_new = np.sum(np.abs(a_frag.t2))
                tt_new=  a_frag.compute_energy(exchange = False)
                


                

                print("Expanded occupied space")
                print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                #print("Convergence:", dtt)
                #print("Convergence", np.abs(tt_new-tt))
                print("energy (-exhange):", a_frag.compute_energy(exchange = False))
                convergence.append(np.abs(tt_new-tt))
                if np.abs(tt_new-tt)<args.fot:
                    break
                else:
                    tt = tt_new
                #a_frag.t2 *= 0
                print("Convergence")
                print(convergence)
                



            # Converge to fot
            E_prev_outer = a_frag.compute_energy(exchange = True)
            print("fragment_energy:", a_frag.compute_fragment_energy())
            E_prev = E_prev_outer*1.0
            dE_outer = 10

            virtual_cutoff_prev = a_frag.virtual_cutoff
            occupied_cutoff_prev = a_frag.occupied_cutoff

            print("_________________________________________________________")
            print("Initial cluster domain for fragment:")
            print(fragment)
            print(" ")
            print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
            print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
            print("E(CIM): %.8f      DE(fragment): %.8e" % (E_prev, dE_outer))
            print("dt , it:", dt, it)
            print("_________________________________________________________")




            refcell_fragments.append(a_frag)
            fragment_energy_total += E_prev_outer
        print("Total fragment energy:", fragment_energy_total)
