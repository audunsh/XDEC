#!/usr/bin/env python


import numpy as np

#import ad

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

from mpi4py import MPI


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

class amplitude_solver():
    def print_configuration_space_data(self):
        """
        Prints extent sizes in terms of number of orbitals
        """
        print("%i virtual orbitals included in fragment." % self.n_virtual_tot)
        print("%i occupied orbitals included in fragment." % self.n_occupied_tot)

    def solve(self, norm_thresh = 1e-10, eqtype = "mp2", s_virt = None):
        if eqtype == "mp2_nonorth":
            return self.solve_MP2PAO(norm_thresh, s_virt = s_virt)
        elif eqtype == "paodot":
            return self.solve_MP2PAO_DOT(norm_thresh, s_virt = s_virt)
        elif eqtype == "ls":
            return self.solve_MP2PAO_ls(norm_thresh, s_virt = s_virt)
        else:
            return self.solve_MP2(norm_thresh)

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
            #dx = np.dot(np.linalg.pinv(B), -df_) #should avoid inversion

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



    def solve_MP2(self, norm_thresh = 1e-7):
        """
        Converge fragment (AOS) amplitudes within occupied and virtual extents
        """
        #from autograd import grad, elementwise_grad,jacobian

        nocc = self.p.get_nocc()

        self.virtual_extent = self.d_ia.coords[:self.n_virtual_cells]
        self.pair_extent = self.d_ii.coords[:self.n_occupied_cells]

        self.vp_indx = mapgen(self.virtual_extent, self.pair_extent)
        self.pp_indx = mapgen(self.pair_extent, self.pair_extent)

        DIIS = diis(8)

        #print(vp_indx)
        #print(pp_indx)

        #dt2 = ad.gh(self.omega_agrad)
        #dt_new = dt2[1](self.t2)
        for ti in np.arange(500):



            dt2_new = self.omega(self.t2)
            #print("dt (max/min/absmin):", np.max(dt2_new), np.min(dt2_new), np.abs(dt2_new).min())
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
                self.t2 -= dt2_new #/np.abs(dt2_new).max()
                self.t2 = DIIS.advance(self.t2,dt2_new)



                rnorm = np.linalg.norm(dt2_new)
                #print("%.10e %.10e" % (self.compute_fragment_energy(),update_max ))
                #print("%.10e %.10e" % (self.compute_fragment_energy(),np.abs(dt2_new).max() ))
                if np.abs(dt2_new).max()<1e-7:
                    break
                #print("Max update:", np.abs(dt2_new).max())
                #if np.abs(dt2_new).max() <


            if np.abs(dt2_new).max() <norm_thresh:

                print("Converged in %i iterations with amplitude gradient norm %.2e." % (ti,rnorm))
                #print("(Maximum absolute res. norm: %.4e)" % np.max(np.abs(dt2_new)))
                print("")
                break
        self.max_abs_residual = np.max(np.abs(dt2_new))
        print("Max.abs. deviation in residual post optimization is %.5e" % np.max(np.abs(dt2_new)))

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


    def solve_MP2PAO(self, norm_thresh = 1e-10, s_virt = None, n_diis=8):
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

        DIIS = diis(n_diis)
        opt = True

        ener = [0,0]

        #self.t2 *= 0

        for ti in np.arange(100):

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
                        cell_map = np.arange(tnew.size).reshape(tnew.shape)[self.fragment][:, dL_i][:, :, M_i][:,:,:,dM_i].ravel()





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



                        t2_mapped = np.zeros_like(tnew).ravel()
                        t2_mapped[cell_map] = (tnew*f_iajb**-1).ravel()[cell_map]
                        #print(dLv, dMv, np.abs(np.max(tnew)))

                        t2_new[:, dL, :, M, :, dM, :] = t2_mapped.reshape(tnew.shape)

            #t2 = time.time()

            self.t2 -= .1*t2_new
            self.t2 = DIIS.advance(self.t2,t2_new)

            #t3 = time.time()

            #print ('time(beta):         ',t1-t0)
            #print ('time(resudial):     ',t2-t1)
            #print ('time(diis):         ',t3-t2)

            rnorm = np.linalg.norm(t2_new)
            #ener = self.compute_fragment_energy()
            #print ('R norm: ',rnorm)
            #print ('Energy: ',ener)
            ener.append(self.compute_fragment_energy())
            #print ('R norm: ',rnorm)
            print (ti, 'Energy: ',ener[-1], "Diff:",  ener[-1]-ener[-2], "Res.norm:", rnorm, "Abs max norm:", np.abs(t2_new).max())
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
        sfs    = np.einsum("a,ij,b->iajb",s_aa,f_ij,s_aa)
        fs_ab  = np.einsum("a,b,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        fs_ba  = np.einsum("b,a,i,j->iajb",f_aa,s_aa,np.ones(nocc),np.ones(nocc))
        f_iajb = sfs - fs_ab - fs_ba

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

        occ_ind = np.zeros(self.p.get_nocc(), dtype = np.bool)
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
    def __init__(self, p, wannier_centers, coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 3.0, occupied_cutoff = 1.0, float_precision = np.float64, d_ia = None):
        self.p = p #prism object
        self.d = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers) # distance matrix

        self.float_precision = float_precision

        #self.d_ii = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[:p.get_nocc()])
        #self.d_ia = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[p.get_nocc():])
        self.d_ii = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers[:p.get_nocc()])
        if d_ia is None:
            self.d_ia = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers[p.get_nocc():])
        else:
            self.d_ia = d_ia




        self.fragment = fragment

        self.ib = ib #integral builder
        self.virtual_cutoff = virtual_cutoff
        self.occupied_cutoff = occupied_cutoff


        self.min_elm = np.min(self.d.cget(self.d.coords), axis = (1,2)) #array for matrix-size calc

        self.min_elm_ii = np.min(self.d_ii.cget(self.d_ii.coords), axis = (1,2))
        self.min_elm_ia = np.min(self.d_ia.cget(self.d_ia.coords), axis = (1,2))

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
        self.n_virtual_cells = np.sum(self.min_elm_ia<self.virtual_cutoff)
        self.n_occupied_cells = np.sum(self.min_elm_ii<self.occupied_cutoff)

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1,self.fragment[0]]<self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<self.occupied_cutoff)


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
                        if mM_<N_occ: # if the transpose is in the domain
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
        sequence = np.array(sequence)

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


                dL = self.d_ia.coords[sq_i[0,0]]



                k = 0
                for l in np.arange(len(sq_i)):


                    if sq_i[k,2] != sq_i[l,2]:
                        dM = self.d_ia.coords[sq_i[k,2]]

                        # Integrate here, loop over M
                        t0 = time.time()
                        I, Ishape = self.ib.getorientation(dL, dM)

                        # adaptive
                        #di_indices = np.unique(np.array(sq_i[k:l])[:, 1])
                        #I, Ishape = self.ib.get_adaptive(dL, dM, self.d_ii.coords[ di_indices ])

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
                        #print(sq_i)


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

        Nv = np.sum(self.min_elm_ia<self.virtual_cutoff)
        No = np.sum(self.min_elm_ii<self.occupied_cutoff)

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
    def __init__(self, fragment_1, fragment_2, M):

        self.f1 = fragment_1
        self.f2 = fragment_2

        self.p = p #prism object
        self.d = self.f1.d #  dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers) # distance matrix
        self.M = M

        self.float_precision = self.f1.float_precision

        #self.d_ii = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[:p.get_nocc()])
        #self.d_ia = dd.build_distance_matrix(p, coords, wannier_centers[fragment], wannier_centers[p.get_nocc():])
        self.d_ii = self.f1.d_ii*1 #dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers[:p.get_nocc()])
        #print(self.f2.d_ii.blocks[:, self.f1.fragment[0], :])
        # Unite with translated occupied domain (set these distances to zero)
        for coord in self.d_ii.coords:
            elmn = self.f2.d_ii.cget(coord - M)[self.f2.fragment[0], :] < self.f2.occupied_cutoff
            #print(elmn, self.f2.d_ii.cget(coord - M)[self.f2.fragment[0], :] )
            #print(coord, coord-M) #happens outside the region, distance problem
            #print(" ")
            self.d_ii.blocks[ self.d_ii.mapping[ self.d_ii._c2i(coord) ], self.f1.fragment[0],  elmn] = self.f1.occupied_cutoff*0.99


        #sort in increasing order to avoid miscounting
        order = np.argsort( np.min(self.d_ii.cget(self.d_ii.coords)[:, self.f1.fragment[0], :], axis = 1) )
        #print(order)
        d_ii = tp.tmat()
        d_ii.load_nparray(self.d_ii.cget(self.d_ii.coords[order]), self.d_ii.coords[order] )
        #print(np.min(d_ii.cget(d_ii.coords)[:, self.f1.fragment[0], :], axis = 1))
        self.d_ii = d_ii




        # Set index of f2 coordinate (for energy summations)
        self.mM = np.argwhere(np.sum((self.d_ii.coords-M)**2, axis = 1)==0)[0,0]
        #print(self.mM, self.d_ii.coords[self.mM], self.M)
        #print(self.d_ii.coords)
        #print(" ..... ")


        #if d_ia is None:
        #    self.d_ia = dd.build_distance_matrix(p, coords, wannier_centers, wannier_centers[p.get_nocc():])
        #else:
        self.d_ia = self.f1.d_ia*1.0

        # Unite with translated virtual domain (set these distances to zero)
        for coord in self.d_ia.coords:
            elmn = self.f2.d_ia.cget(coord - M)[self.f2.fragment[0], :] < self.f2.virtual_cutoff
            self.d_ia.blocks[ self.d_ia.mapping[ self.d_ia._c2i(coord) ], self.f1.fragment[0],  elmn] = self.f1.virtual_cutoff*0.99

        #sort in increasing order to avoid miscounting
        #order = np.argsort( np.min(self.d_ia.cget(self.d_ia.coords), axis = (1,2)) )
        order = np.argsort( np.min(self.d_ia.cget(self.d_ia.coords)[:, self.f1.fragment[0], :], axis = 1) )
        d_ia = tp.tmat()
        d_ia.load_nparray(self.d_ia.cget(self.d_ia.coords[order]), self.d_ia.coords[order] )
        #print(self.d_ia.coords[order])
        self.d_ia = d_ia
        #print(self.d_ii.coords)
        #print(self.d_ia.coords)
        #print(self.d_ia.coords.shape)



        self.fragment = self.f1.fragment

        self.ib = self.f1.ib #integral builder
        self.virtual_cutoff = self.f1.virtual_cutoff*1.0
        self.occupied_cutoff = self.f1.occupied_cutoff*1.0


        self.min_elm = np.min(self.d.cget(self.d.coords), axis = (1,2)) #array for matrix-size calc

        self.min_elm_ii = np.min(self.d_ii.cget(self.d_ii.coords), axis = (1,2))
        self.min_elm_ia = np.min(self.d_ia.cget(self.d_ia.coords), axis = (1,2))

        self.f_mo_ii = self.f1.f_mo_ii # Occupied MO-Fock matrix elements
        self.f_mo_aa = self.f1.f_mo_aa # Virtual MO-Fock matrix elements

        #print("Fragment extent")
        #print(self.d_ii.coords)
        #print(self.d_ia.coords)

        self.init_amplitudes()

    def init_amplitudes(self):
        """
        Initialize the amplitudes using the MP2-like starting guess
        """
        self.n_virtual_cells = np.sum(self.min_elm_ia<self.virtual_cutoff)
        self.n_occupied_cells = np.sum(self.min_elm_ii<self.occupied_cutoff)

        self.n_virtual_tot = np.sum(self.d_ia.blocks[:-1,self.fragment[0]]<self.virtual_cutoff)
        self.n_occupied_tot = np.sum(self.d_ii.blocks[:-1, self.fragment[0]]<self.occupied_cutoff)


        n_occ = self.p.get_nocc()     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells # Number of occupied cells
        n_virt = self.p.get_nvirt()   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells # Number of virtual cells


        self.t2  = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_d = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        self.g_x = np.zeros((n_occ, N_virt, n_virt, N_occ, n_occ, N_virt, n_virt), dtype = self.float_precision)

        print(self.t2.shape)
        print("Virtual domain:")
        print(self.d_ia.coords[:N_virt])
        print("Occupied domain:")
        print(self.d_ii.coords[:N_occ])



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
            for ddM in np.arange(N_virt):
                dL, dM = self.d_ia.coords[ddL], self.d_ia.coords[ddM]

                for mM in np.arange(N_occ):
                    M = self.d_ii.coords[mM]



                    if True: #M[0]>=0: #?

                        # Get exchange block coordinates
                        ddL_M = self.d_ia.mapping[self.d_ia._c2i(dM + M) ]
                        #ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL - M) ]
                        ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL - M) ]





                        # Negative M
                        mM_ = self.d_ia.mapping[ self.d_ia._c2i(-M) ]
                        #print(mM_)

                        # sequence[ ddL, mM, ddM        ,  0 ,   ddL, mM, ddM,    0]
                        #                ^                 ^           ^          ^
                        #            Calculate these    direct/ex    store here   1=transpose

                        sequence.append([ddL, mM, ddM,   0, ddL, mM, ddM,   0]) # direct

                        # Testing here
                        #sequence.append([ddL_M, mM, ddM_M,   1, ddL, mM, ddM,   0]) # exchange
                        """



                        if mM_<N_occ: # if the transpose is in the domain
                            sequence.append([ddL, mM, ddM,   0, ddM, mM_, ddL,  1]) # direct, transposed

                            #ddL_M_ = self.d_ia.mapping[self.d_ia._c2i(dM - M) ]
                            #ddM_M = self.d_ia.mapping[self.d_ia._c2i(dL - M) ]
                            #ddM_M_ = self.d_ia.mapping[self.d_ia._c2i(dL + M) ]
                            #sequence.append([ddL_M_, mM, ddM_M_,   1, ddM, mM_, ddL,  1]) # exchange, transposed
                        """





                        """
                        # For fragments, exchange only required for only M = (0,0,0)
                        # EOS always has the two occupied indices in the fragment, ie the refcell
                        if np.sum(M**2) == 0:
                            #sequence.append([ddL_M, mmM, ddM_M, 1, ddL, mmM , ddM,0])  # exchange
                            #sequence.append([ddL_M, mmM, ddM_M, 1, ddM, mmM_, ddL,1]) # exchange, transposed

                            #sequence.append([ddL_M, mM, ddM_M, 1, ddL, mM , ddM,0])  # exchange
                            #sequence.append([ddL_M, mM, ddM_M, 1, ddM, mM_, ddL,1]) # exchange, transposed

                            sequence.append([ddL, mM, ddM,   1, ddL, mM, ddM,   1]) # direct
                            sequence.append([ddL, mM, ddM,   1, ddM, mM, ddL,   0]) # direct, transposed
                        """


        #print(sequence)
        self.initialize_blocks(sequence)

    def initialize_blocks(self, sequence):
        sequence = np.array(sequence)

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


                dL = self.d_ia.coords[sq_i[0,0]]



                k = 0
                for l in np.arange(len(sq_i)):


                    if sq_i[k,2] != sq_i[l,2]:
                        dM = self.d_ia.coords[sq_i[k,2]]

                        # Integrate here, loop over M
                        t0 = time.time()
                        I, Ishape = self.ib.getorientation(dL, dM)

                        # adaptive
                        #di_indices = np.unique(np.array(sq_i[k:l])[:, 1])
                        #I, Ishape = self.ib.get_adaptive(dL, dM, self.d_ii.coords[ di_indices ])

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
                        #print(sq_i)


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

    def compute_pair_fragment_energy(self):
        """
        Compute fragment energy
        """

        n_occ = self.p.get_nocc()     # Number of occupied orbitals per cell
        N_occ = self.n_occupied_cells # Number of occupied cells
        n_virt = self.p.get_nvirt()   # Number of virtual orbitals per cell
        N_virt = self.n_virtual_cells # Number of virtual cells
        #print(N_virt, N_occ)
        #print(self.d_ia.coords[:N_virt])
        #print(self.d_ii.coords[:N_occ])




        e_mp2 = 0

        N_virt = self.n_virtual_cells

        #print(self.mM)
        #print(self.g_d.shape)
        print("Compute PAIR FRAGMENT")

        for ddL in np.arange(N_virt):
            dL = self.d_ia.coords[ddL]
            dL_i = np.array(self.d_ia.cget(dL)[self.f1.fragment[0],:], dtype = bool)<self.virtual_cutoff # dL index mask
            #dL_i = np.array(self.d_ia.cget([0,0,0])[self.f1.fragment[0],:], dtype = bool) # dL index mask


            for ddM in np.arange(N_virt):
                dM =  self.d_ia.coords[ddM]
                dM_i = np.array(self.d_ia.cget(dM) [self.f1.fragment[0],:], dtype = bool)<self.virtual_cutoff # dM index mask
                #dM_i = np.array(self.d_ia.cget([0,0,0])[self.f1.fragment[0],:], dtype = bool) # dM index mask


                g_direct = self.g_d[:,ddL,:,self.mM, :, ddM, :][self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]
                #g_exchange = self.g_x[:,ddL,:,self.mM, :, ddM, :] #[self.f1.fragment][:, dM_i][:, :, self.f2.fragment][:,:,:,dL_i]

                #g_exchange = self.cfit.()
                print(" Energy:", dL, dM, dM+self.M, dL-self.M)
                I, Ishape = self.ib.getorientation(dM+self.M, dL-self.M)
                g_exchange = I.cget(self.M).reshape(Ishape)[self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]



                t = self.t2[:,ddL,:,self.mM, :, ddM, :][self.f1.fragment][:, dL_i][:, :, self.f2.fragment][:,:,:,dM_i]

                e_mp2 += 2*np.einsum("iajb,iajb",t,g_direct, optimize = True)  - np.einsum("iajb,ibja",t,g_exchange, optimize = True)

        print(e_mp2, 2*e_mp2)
        return e_mp2



class diis():
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
    print("nocc :", c_occ.blocks.shape[2])
    print("nvirt:", c_virt.blocks.shape[2])
    # This will be removed, provided on input
    #d = dd.build_distance_matrix(p, np.array([[0,0,0]]), c_occ_centers, c_occ_centers)
    #fragment = dd.atomic_fragmentation(p, d, 3.0)[0]
    #print("Fragm:", fragment[0])

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



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    os.environ["LIBINT_DATA_PATH"] = os.getcwd()

    if mpi_rank == 0:
        print("""#########################################################
##    ,--.   ,--.      ,------.  ,------. ,-----.      ##
##     \  `.'  /,-----.|  .-.  \ |  .---''  .--./      ##
##      .'    \ '-----'|  |  \  :|  `--, |  |          ##
##     /  .'.  \       |  '--'  /|  `---.'  '--'\      ##
##    '--'   '--'      eXtended local correlation      ##
##                                                     ##
##  Use keyword "--help" for more info                 ##
#########################################################""",flush=True)
    else:
        pass


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
    parser.add_argument("-ibuild", type = str, default = None, help = "Filename of integral fitting module (will be computed if missing).")
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
    parser.add_argument("-N_c", type = int, default = 6, help = "Number of layers in Coulomb BvK-cell." )
    parser.add_argument("-pairs", type = bool, default = False, help = "Compute pair fragments" )
    parser.add_argument("-print_level", type = int, default = 0, help = "Print level" )
    parser.add_argument("-orb_increment", type = int, default = 6, help = "Number of orbitals to include at every XDEC-iteration." )
    parser.add_argument("-pao_sorting", type = bool, default = False, help = "Sort LVOs in order of decreasing PAO-coefficient" )
    parser.add_argument("-adaptive_domains", default = False, action = "store_true", help = "Activate adaptive Coulomb matrix calculation.")

    args = parser.parse_args()

    args.float_precision = eval(args.float_precision)
    import sys

    #git_hh = sp.Popen(['git' , 'rev-parse', 'HEAD'], shell=False, stdout=sp.PIPE, cwd = sys.path[0])

    #git_hh = git_hh.communicate()[0].strip()

    # Print run-info to screen
    #print("Git rev : ", git_hh)

    if mpi_rank == 0:
        print("Authors : Audun Skau Hansen (a.s.hansen@kjemi.uio.no) ")
        print("          Einar Aurbakken")
        print(" ")
        print("   \u001b[38;5;117m0\u001b[38;5;27mø \033[0mHylleraas Centre for Quantum Molecular Sciences")
        print("                        UiO 2020")


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
        print("Virtual space          :", args.virtual_space)
        print("Coulomb extent (layers):", args.N_c)
        #print("Dot-product            :", ["Block-Toeplitz", "Circulant"][int(args.circulant)])
        #print("RI fitting             :", ["Non-robust", "Robust"][int(args.robust)])
        print("_________________________________________________________",flush=True)
    else:
        pass


    if mpi_rank == 0:
        # Load system
        p = pr.prism(args.project_file)
        p.n_core = args.n_core

        # Compute overlap matrix
        s = of.overlap_matrix(p)


        # Fitting basis
        auxbasis = PRI.basis_trimmer(p, args.auxbasis, alphacut = args.basis_truncation)
        f = open("ri-fitbasis.g94", "w")
        f.write(auxbasis)
        f.close()

        # Load wannier coefficients
        c = tp.tmat()
        c.load(args.coefficients)
        c.set_precision(args.float_precision)


        # Load wannier centers

        wcenters, spreads = of.centers_spreads(c, p, s.coords)
        wcenters = wcenters[p.n_core:] #remove core orbitals



        c_occ, c_virt = PRI.occ_virt_split(c,p)

        if args.virtual_space is not None:
            if args.virtual_space == "pao":
                s_, c_virt, wcenters_virt = of.conventional_paos(c,p)
                p.n_core = args.n_core
                p.set_nvirt(c_virt.blocks.shape[2])

                args.solver = "mp2_nonorth"

                # Append virtual centers to the list of centers
                wcenters = np.append(wcenters[:p.get_nocc()-p.n_core], wcenters_virt, axis = 0)

            elif args.virtual_space == "paodot":
                s_, c_virt, wcenters_virt = of.conventional_paos(c,p)
                p.n_core = args.n_core
                p.set_nvirt(c_virt.blocks.shape[2])

                args.solver = "paodot"

                # Append virtual centers to the list of centers
                wcenters = np.append(wcenters[:p.get_nocc()-p.n_core], wcenters_virt, axis = 0)


            else:
                c_virt = tp.tmat()
                c_virt.load(args.virtual_space)
                p.set_nvirt(c_virt.blocks.shape[2])

        s_virt = c_virt.tT().circulantdot(s.circulantdot(c_virt))



        # AO Fock matrix
        f_ao = tp.tmat()
        f_ao.load(args.fock_matrix)
        f_ao.set_precision(args.float_precision)



        # Compute MO Fock matrix

        f_mo_aa = c_virt.tT().cdot(f_ao*c_virt, coords = c_virt.coords)
        f_mo_ii = c_occ.tT().cdot(f_ao*c_occ, coords = c_occ.coords)
        f_mo_ia = c_occ.tT().cdot(f_ao*c_virt, coords = c_occ.coords)
        f_mo_ia = c_occ.tT().circulantdot(f_ao.circulantdot(c_virt)) #, coords = c_occ.coords)

        # Compute energy denominator

        f_aa = np.diag(f_mo_aa.cget([0,0,0]))
        f_ii = np.diag(f_mo_ii.cget([0,0,0]))

        e_iajb = f_ii[:,None,None,None] - f_aa[None,:,None,None] + f_ii[None,None,:,None] - f_aa[None,None,None,:]








        # Initialize integrals
        if args.ibuild is None:
            ib = PRI.integral_builder_static(c_occ,c_virt,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust, ao_screening = args.ao_screening, xi0=args.xi0, JKa_extent= [6,6,6], xi1 = args.xi1, float_precision = args.float_precision, N_c = args.N_c,printing = args.print_level)
            #ib = PRI.integral_builder_static(c_occ,c_virt,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=None, circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust, ao_screening = args.ao_screening, xi0=args.xi0, JKa_extent= [6,6,6], xi1 = args.xi1, float_precision = args.float_precision, N_c = args.N_c,printing = args.print_level)

            np.save("integral_build.npy", np.array([ib]), allow_pickle = True)
        else:
            ib = np.load(args.ibuild, allow_pickle = True)[0]
            print(ib.getorientation([0,0,0],[0,0,0]))


        # Initialize domain definitions
        d = dd.build_distance_matrix(p, c.coords, wcenters, wcenters)




        center_fragments = dd.atomic_fragmentation(p, d, 2.0)[::-1]

        print(" ")
        print("_________________________________________________________")
        print("Fragmentation of occupied space:")
        for i in np.arange(len(center_fragments)):
            print("  Fragment %i:" %i, center_fragments[i])
        print("_________________________________________________________")
        print(" ",flush=True)

    else:
        pass


    comm.barrier()
    print ('Process no.',mpi_rank,'has passed the first barrier',flush=True)
    comm.barrier()



    # Initial fragment extents
    virt_cut = 3.0
    occ_cut = 6.0

    refcell_fragments = []

    if mpi_rank != 0:
        p = None
        center_fragments = None
        c = None
        f_mo_ii = None
        f_mo_aa = None
        s_virt = None
        ib = None
        wcenters = None
    else:
        print (flush=True)

    p = comm.bcast(p,root=0)
    c = comm.bcast(c,root=0)
    f_mo_ii = comm.bcast(f_mo_ii,root=0)
    f_mo_aa = comm.bcast(f_mo_aa,root=0)
    s_virt = comm.bcast(s_virt,root=0)
    ib = comm.bcast(ib,root=0)
    wcenters = comm.bcast(wcenters,root=0)

    if mpi_rank == 0:
        n_fragms = len(center_fragments)

        for r in np.arange(1,mpi_size):
            ind_i = int(n_fragms*r/mpi_size)
            ind_f = int(n_fragms*(r+1)/mpi_size)
            list_to_send = center_fragments[ind_i:ind_f]
            comm.send(list_to_send,dest=r)

        center_fragments_r = center_fragments[:int(n_fragms/mpi_size)]
    else:
        center_fragments_r = None
        center_fragments_r = comm.recv(center_fragments_r,source=0)



    fragment_total = 0

    T1 = time.time()

    # Converge atomic fragment energies
    for fragment in center_fragments_r:


        #ib.fragment = fragment
        t0 = time.time()
        if args.pao_sorting:
            d_ia = build_weight_matrix(p, c, s.coords)
            a_frag = fragment_amplitudes(p, wcenters, c.coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 3.0, occupied_cutoff = 2.0, float_precision = args.float_precision, d_ia = d_ia)


        else:
            a_frag = fragment_amplitudes(p, wcenters, c.coords, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = 3.0, occupied_cutoff = 2.0, float_precision = args.float_precision)

        #print("Frag init:", time.time()-t0)

        a_frag.solve(eqtype = args.solver, s_virt = s_virt)

        #print("t2 (max/min/absmin):", np.max(a_frag.t2), np.min(a_frag.t2), np.abs(a_frag.t2).min())
        #print("g_d (max/min/absmin):", np.max(a_frag.g_d), np.min(a_frag.g_d), np.abs(a_frag.g_d).min())

        # Converge to fot
        E_prev_outer = a_frag.compute_fragment_energy()
        E_prev = E_prev_outer*1.0
        dE_outer = 10

        print("Initial fragment energy: %.8f" % E_prev)

        print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
        print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))

        virtual_cutoff_prev = a_frag.virtual_cutoff
        occupied_cutoff_prev = a_frag.occupied_cutoff

        if not args.skip_fragment_optimization:
            print("Running fragment optimization for:",fragment,flush=True)
            #print("Initial cutoffs:")

            n_virtuals_ = []
            virtu_cut_  = []



            while dE_outer>args.fot:
                dE = 10
                e_virt = []
                #
                e_virt.append(E_prev)
                n_virtuals_.append(a_frag.n_virtual_tot)
                virtu_cut_.append(a_frag.virtual_cutoff)
                while dE>args.fot:
                    virtual_cutoff_prev = a_frag.virtual_cutoff
                    occupied_cutoff_prev = a_frag.occupied_cutoff

                    t_0 = time.time()
                    a_frag.autoexpand_virtual_space(n_orbs=args.orb_increment)
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot),flush = True)



                    t_1 = time.time()
                    a_frag.solve(eqtype = args.solver, s_virt = s_virt)
                    t_2 = time.time()
                    E_new = a_frag.compute_fragment_energy()
                    t_3 = time.time()
                    dE = np.abs(E_prev - E_new)
                    print("D_ii = ", a_frag.compute_mp2_density(orb_n = 0).shape)

                    print("_________________________________________________________")
                    print("E(fragment): %.8f      DE(fragment): %.8e" % (E_new, dE))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
                    print("Time (expand/solve/energy) (s) : %.1f / %.1f / %.1f" % (t_1-t_0, t_2-t_1, t_3-t_2))
                    print(" ")
                    e_virt.append(E_new)
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



                print("Converged virtual space, expanding occupied space",flush = True)
                print(e_virt)
                #dE = 10
                #print("--- occupied")
                a_frag.autoexpand_occupied_space(n_orbs=args.orb_increment)
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
                print(" ",flush = True)
                E_prev = E_new
                #print("---")

                while dE>args.fot:
                    virtual_cutoff_prev = a_frag.virtual_cutoff
                    occupied_cutoff_prev = a_frag.occupied_cutoff

                    #print("--- occupied")
                    a_frag.autoexpand_occupied_space(n_orbs=args.orb_increment)
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot),flush = True)

                    a_frag.solve(eqtype = args.solver, s_virt = s_virt)
                    E_new = a_frag.compute_fragment_energy()

                    #a_frag.print_configuration_space_data()
                    dE = np.abs(E_prev - E_new)

                    print("_________________________________________________________")
                    print("E(fragment): %.6f        DE(fragment): %.6e" % (E_new, dE))
                    print("_________________________________________________________")
                    print("Current memory usage of integrals (in MB): %.2f" % ib.nbytes())
                    print(" ",flush = True)
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
            print(" ",flush = True)
            refcell_fragments.append(a_frag)
            fragment_total += E_new

    T2 = time.time()

    comm.barrier()
    print ('Time for process',mpi_rank,'was',T2-T1,flush=True)
    comm.barrier()
    T_tot = time.time()
    if mpi_rank == 0:
        print ()
        print ('Total time used to optimize fragments: ',T_tot-T1,flush=True)
    else:
        pass


    fragment_total = comm.reduce(fragment_total,root=0)
    fragment_total = comm.bcast(fragment_total,root=0)


    refcell_fragments_gathered = comm.gather(refcell_fragments,root=0)


    if mpi_rank == 0:
        refcell_fragments = []
        for el in refcell_fragments_gathered:
            refcell_fragments = refcell_fragments + el
    else:
        pass

    refcell_fragments = comm.bcast(refcell_fragments,root=0)


    comm.barrier()
    print ('Rank',mpi_rank,'has passed the second barrier',flush=True)


    if args.pairs:
        if mpi_rank != 0:
            pair_coords = None
            refcell_fragments = None
        else:
            print (flush=True)

        refcell_fragments = comm.bcast(refcell_fragments,root=0)

        loop_setup = 1

        #pair fragments
        if loop_setup == 0:
            #parallilization of the pair_coords loop
            def fix_pair_coords(pc):
                """
                adds reference cell and removes opposing cells
                """
                pc = np.concatenate([np.atleast_2d(np.zeros(3,dtype=int)),pc])
                ind = np.array(np.where((pc[:,0][:,None] == -pc[:,0])&(pc[:,1][:,None] == -pc[:,1])&(pc[:,2][:,None] == -pc[:,2]))).T
                ind = np.unique(np.sort(ind,axis=1),axis=0)[:,1]
                return pc[ind]


            if mpi_rank == 0:
                if p.cperiodicity == "POLYMER":
                    pair_coords = tp.lattice_coords([6,0,0])
                elif p.cperiodicity == "SLAB":
                    pair_coords = tp.lattice_coords([6,6,0])
                else:
                    pair_coords = tp.lattice_coords([6,6,6])

                pair_coords = pair_coords[np.argsort(np.sum(p.coor2vec(pair_coords)**2, axis = 1))[1:]] #Sort in increasing distance
                pair_coords = fix_pair_coords(pair_coords)

                n_cells = len(pair_coords)

                for r in np.arange(1,mpi_size):
                    comm.send(pair_coords[int(n_cells*r/mpi_size):int(n_cells*(r+1)/mpi_size),:],dest=r)

                pair_coords = pair_coords[:int(n_cells/mpi_size),:]
            else:
                pair_coords = None
                pair_coords = comm.recv(pair_coords,source=0)


            T1 = time.time()

            ref_coord = np.zeros(3,dtype=int)
            pair_total = 0
            for c in pair_coords:
                for fa in np.arange(len(refcell_fragments)):
                    for fb in np.arange(len(refcell_fragments)):

                        if (c == ref_coord).all() and ((fa == fb) or (fa > fb)):
                            continue

                        frag_a = refcell_fragments[fa]
                        frag_b = refcell_fragments[fb]

                        pair = pair_fragment_amplitudes(frag_a, frag_b, M = c)
                        #print(pair.compute_pair_fragment_energy())
                        pair.solve()
                        p_energy = pair.compute_pair_fragment_energy()
                        pair_total += 2*p_energy
                        print ('_________________________________________________________')
                        print ('Computed by process: ',mpi_rank)
                        print ('IND_A:',fa,'IND_B:',fb,'CELL_B',c)
                        print ("Pair fragment energy for ",c," is ", 2*p_energy, " (total: ", pair_total , " )")
                        print ("R = ", np.sum(p.coor2vec(c)**2)**.5, fa, fb)
                        print (fa, fb, np.sum(p.coor2vec(c)**2)**.5)
                        print (flush=True)

            T2 = time.time()

            comm.barrier()
            print ('Time for process',mpi_rank,'was',T2-T1)

            T_total = time.time()
            pair_total = comm.reduce(pair_total,root=0)
            if mpi_rank == 0:
                print ()
                print ('Total time used to compute pair energies: ',T_total-T1)
                print ('Total fragment energy: ',fragment_total)
                print ('Total pair energy: ',pair_total)
                print ('Total correlation energy: ',fragment_total + pair_total)
            else:
                pass



        elif loop_setup == 1:
            #setting up complete pair array before parallelizing to share pairs
            #more evenly among the processes
            def fix_pairs(pc,fragms):
                """
                creates a full pair array with columns cell_x,cell_y,cell_z,ind_a,ind_b
                """
                #Adding reference cell
                pc = np.concatenate([np.atleast_2d(np.zeros(3,dtype=int)),pc])

                #Removing opposing cells
                ind = np.array(np.where((pc[:,0][:,None] == -pc[:,0])&(pc[:,1][:,None] == -pc[:,1])&(pc[:,2][:,None] == -pc[:,2]))).T
                ind = np.unique(np.sort(ind,axis=1),axis=0)[:,1]
                pc = pc[ind]

                #Setting up extended pair_coords array
                n_fragms = len(fragms)

                n_cpairs = n_fragms**2
                n_cpairs_ref = int(0.5*n_fragms*(n_fragms-1))

                old_len = len(pc)
                new_len = n_cpairs_ref + n_cpairs*(old_len-1)
                pc_ext = np.zeros([new_len,3],dtype=int)
                pc_ext[n_cpairs_ref:] = np.repeat(pc[1:],n_cpairs,axis=0)

                #Setting up pair indices
                X,Y = np.mgrid[0:n_fragms:1, 0:n_fragms:1]
                X = X.ravel()
                Y = Y.ravel()
                indx_cell = np.zeros([len(X),2],dtype=int)
                indx_cell[:,0] = X
                indx_cell[:,1] = Y
                indx_cell_ref = indx_cell[np.invert(np.equal(X,Y))]
                indx_cell_ref = np.sort(indx_cell_ref,axis=1)
                indx_cell_ref = np.unique(indx_cell_ref,axis=0)

                ind_len = len(pc_ext)
                indx = np.zeros([ind_len,2],dtype=int)
                indx[:n_cpairs_ref] = indx_cell_ref
                indx[n_cpairs_ref:] = np.tile(indx_cell.T,int(len(indx[n_cpairs_ref:])/n_cpairs)).T

                pairs = np.zeros([ind_len,5],dtype=int)
                pairs[:,:3] = pc_ext
                pairs[:,3:5] = indx

                np.random.seed(0)
                np.random.shuffle(pairs)

                return pairs


            if mpi_rank == 0:
                if p.cperiodicity == "POLYMER":
                    pair_coords = tp.lattice_coords([6,0,0])
                elif p.cperiodicity == "SLAB":
                    pair_coords = tp.lattice_coords([6,6,0])
                else:
                    pair_coords = tp.lattice_coords([6,6,6])

                pair_coords = pair_coords[np.argsort(np.sum(p.coor2vec(pair_coords)**2, axis = 1))[1:]] #Sort in increasing distance
                pairs = fix_pairs(pair_coords,center_fragments)

                n_pairs = len(pairs)

                for r in np.arange(1,mpi_size):
                    comm.send(pairs[int(n_pairs*r/mpi_size):int(n_pairs*(r+1)/mpi_size),:],dest=r)

                pairs = pairs[:int(n_pairs/mpi_size),:]
            else:
                pairs = None
                pairs = comm.recv(pairs,source=0)


            T1 = time.time()

            ref_coord = np.zeros(3,dtype=int)
            pair_total = 0
            for i in np.arange(len(pairs)):
                c = pairs[i,:3]
                fa = pairs[i,3]
                fb = pairs[i,4]

                frag_a = refcell_fragments[fa]
                frag_b = refcell_fragments[fb]

                pair = pair_fragment_amplitudes(frag_a, frag_b, M = c)
                #print(pair.compute_pair_fragment_energy())
                pair.solve()
                p_energy = pair.compute_pair_fragment_energy()
                pair_total += 2*p_energy
                print ('_________________________________________________________')
                print ('Computed by process: ',mpi_rank)
                print ('IND_A:',fa,'IND_B:',fb,'CELL_B',c)
                print ("Pair fragment energy for ",c," is ", 2*p_energy, " (total: ", pair_total , " )")
                print ("R = ", np.sum(p.coor2vec(c)**2)**.5, fa, fb)
                print (fa, fb, np.sum(p.coor2vec(c)**2)**.5)
                print (flush=True)



            T2 = time.time()

            comm.barrier()
            print ('Time for process',mpi_rank,'was',T2-T1)

            T_total = time.time()
            pair_total = comm.reduce(pair_total,root=0)
            if mpi_rank == 0:
                print ()
                print ('Total time used to compute pair energies: ',T_total-T1)
                print ('Total fragment energy: ',fragment_total)
                print ('Total pair energy: ',pair_total)
                print ('Total correlation energy: ',fragment_total + pair_total)
            else:
                pass







        #for n in np.arange(10):
        #    pair = pair_fragment_amplitudes(a_frag, a_frag, M = np.array([0,0,n]))
        #    print(pair.compute_pair_fragment_energy())
        #    pair.solve()
        #    print("Pair fragment energy for (0,0,%i):" %n, pair.compute_pair_fragment_energy())
