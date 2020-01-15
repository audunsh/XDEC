#!/usr/bin/env python

import numpy as np
import os 

from scipy.linalg import expm

import sys

import lwrap.lwrap as lwrap

import matplotlib.pyplot as plt


import argparse


import utils.toeplitz as tp
import utils.prism as pr

import utils.annealing_tools as at

import lwrap_interface as lint

import subprocess as sp

import PRI

import utils.objective_functions as objective_functions

import utils.sunmat as sm

import autograd.numpy as agnp

import autograd as ag

def summarize_anneal(N_uphill_steps, N_accept_steps, N_steps, T, yp0, yp):
    """
    Print optimization results to screen
    """
    print("Optimization results")
    print("------------------------------------------")
    print("Number of uphill steps    :", N_uphill_steps)
    print("Number of accepted steps  :", N_accept_steps)
    print("Number of rejected steps  :", N_steps-N_accept_steps)
    print("Final temperature         :", T)
    print("Initial objective function:", yp0)
    print("Final objective function  :", yp)
    print("Difference                :", yp-yp0)
    if yp-yp0>=0:
        print("    Warning: resulting orbitals not improved.")
        print("             Try changing step size or other optimization parameters.")
    print("------------------------------------------")    

## Unitary matrix generators

def birandU(N, nocc, stepsize = 1.0):  
    """
    ############################
    # Random unitary matrix    #
    ############################      
    """
    U = np.eye(N, dtype = float)
    X = np.random.uniform(-stepsize,stepsize,(nocc,nocc))
    U[:nocc,:nocc] = expm((X- X.T)/2.0)
    X = np.random.uniform(-stepsize,stepsize,(N-nocc,N-nocc))
    U[nocc:,nocc:] = expm((X- X.T)/2.0)
    return U

def randU(N, stepsize = 1.0):  
    """
    ############################
    # Random unitary matrix    #
    ############################      
    """
    X = np.random.uniform(-stepsize, stepsize, (N,N))
    return expm((X - X.T) / 2.0)

def zrandU(N, stepsize = 1.0):
    """
    ############################
    # Complex randU            #
    ############################      
    """   
    X = np.complex(1,0)*np.random.uniform(-stepsize,stepsize,(N,N)) + np.complex(0,1)*np.random.uniform(-stepsize,stepsize,(N,N))
    return expm((X- X.T.conj())/2.0)
    
## Rotation functions

def rotate_k(M, Uk):
    # Return rotated M by unitary Uk in reciprocal space
    M_ret = M*1
    #M_ret.blocks = np.array(M_ret.blocks, dtype = complex)
    ret_blocks_k = np.fft.fft(M_ret.blocks[:-1], axis = 0)
    for k in np.arange(ret_blocks_k.shape[0]):
        ret_blocks_k[k] = np.dot(ret_blocks_k[k], Uk)
    M_ret.blocks[:-1] = np.fft.ifft(ret_blocks_k, axis = 0).real
    return M_ret


def rotate(M, U):
    M_ret = M*1
    #M_ret.blocks = np.array(M_ret.blocks, dtype = complex)
    for i in np.arange(M_ret.coords.shape[0]):
        M_ret.cset(M_ret.coords[i], np.dot(M_ret.cget(M_ret.coords[i]), U))
    return M_ret

    
def kspace_anneal(F, X, T, N_steps = 10000, T_decay = 0.997, stepsize = 1.0, order = 2):
    """
    #########################################
    ##                                     ##
    ## Matrix FFT-annealer                 ##
    ## Input:                              ##
    ##  F = objective function             ##
    ##  X                                  ##
    ##          = List of tensors          ##
    ##  T  = initial temperature           ##
    ##  N_steps = Number of iterations     ##
    ##  stepsize = stepsize                ##
    ##  order = 1 -> u' =     x*u          ##
    ##  order = 2 -> u' = u.T*x*u          ##
    ##                                     ##
    ## This function rotates the set of    ##
    ## input-tensors such that             ##
    ## X_fin = np.dot(U.T, np.dot(X_in, U) ##
    ## minimize the function f(X_fin).     ##
    ##                                     ##
    #########################################
    """
        
    yp = F(X) #initial value of functional

    yp0 = 1*yp
    
    N_uphill_steps = 0
    N_accept_steps = 0
    
    #make toeplitz arrays complex
    X_rot = []
    for i in np.arange(len(X)):
        X[i].blocks = np.array(X[i].blocks, dtype = complex)
        X_rot.append(X[i]*1) #store a copy for rotation

    #intermediate storage of fft-blocks
    
    X_k  = []
    X_kn = [] 
    
    for i in np.arange(len(X)):
        X_k.append( np.fft.fft(X[i].blocks, axis = 0))
        X_kn.append(np.fft.fft(X[i].blocks, axis = 0))

    U_prev = np.eye(X[0].blockshape[0], dtype = complex) #initial unitary matrix
    
    for i in np.arange(N_steps):
        #Perform random unitary rotation of Up        
        U_new = np.dot(U_prev, randU(X[0].blockshape[0], stepsize = stepsize)) #
        
        #Rotate every tensor in X
        if order == 2:
            for j in np.arange(X[0].blocks.shape[0]-1):
                for k in np.arange(len(X)):
                    X_kn[k][j] = np.dot(U_new.conj().T,np.dot(X_k[k][j], U_new))
                    
        if order == 1:
            for j in np.arange(X[0].blocks.shape[0]-1):
                for k in np.arange(len(X)):
                    X_kn[k][j] = np.dot(X_k[k][j], U_new)

        #
        for j in np.arange(len(X)):
            X_rot[j].blocks = np.fft.ifft(X_kn[j], axis = 0).real
            #print(X_rot[j].blocks.shape)
        
        yn = F(X_rot)

        if yn<yp:
            #Cp = Cn
            #print(yn)
            yp = yn
            U_prev = U_new
            N_accept_steps += 1
        # if not, compute acceptance probability
        # this basically avoids getting trapped in
        # local minimas
        else:
            #print(np.exp(-np.abs(yp-yn)/T))
            if np.exp(-np.abs(yn-yp)/T)>0.5:
                N_uphill_steps += 1
                N_accept_steps += 1
                #Cp = Cn
                yp = yn
                U_prev = U_new
        #decrease temp
        T *= T_decay
        if(i%100==0):
            print("%.2f percent complete." % (100*i/np.float(N_steps)), "\r", end="")
            #print
    summarize_anneal(N_uphill_steps, N_accept_steps, N_steps,T, yp0, yp)
    return 0, yp, U_prev

def center_annealing(F, X, T, N_steps = 10000, T_decay = 0.997,
                     stepsize = 1.0, order = 2, uphill_trigger = 0):
    """
    #########################################
    ##                                     ##
    ## Center-cell annealer                ##
    ## Input:                              ##
    ##  F = objective function             ##
    ##  X                                  ##
    ##          = List of tensors          ##
    ##  T  = initial temperature           ##
    ##  N_steps = Number of iterations     ##
    ##  stepsize = stepsize                ##
    ##  order = 1 -> u' =     x*u          ##
    ##  order = 2 -> u' = u.T*x*u          ##
    ##  uphill_trigger = t-decay trigger   ##
    ##                                     ##
    ## This function rotates the set of    ##
    ## input-tensors such that             ##
    ## X_fin = np.dot(U.T, np.dot(X_in, U) ##
    ## minimize the function F(X_fin).     ##
    ##                                     ##
    #########################################
    
    See 
    
    P. Salamon, P. Sibani, and R. Frost, 
    Facts, Conjectures, and Improvements for Simulated Annelaing,
    SIAM, (2002)
    
    for a desctiption of simulated annealing.
    """
    
    yp = F(X) #initial value of functional

    yp0 = 1*yp
    
    N_uphill_steps = 0
    N_accept_steps = 0
    
    #make toeplitz arrays complex
    X_rot = []
    for i in np.arange(len(X)):
        #X[i].blocks = np.array(X[i].blocks, dtype = complex)
        X_rot.append(X[i]*1) #store a copy for rotation

    #intermediate storage of fft-blocks
    
    #X_k  = []
    #X_kn = [] 
    
    #for i in np.arange(len(X)):
    #    X_k.append( np.fft.fft(X[i].blocks, axis = 0))
    #    X_kn.append(np.fft.fft(X[i].blocks, axis = 0))
    
    U_prev = np.eye(X[0].blockshape[1], dtype = float) #initial unitary matrix
    c = np.array([0,0,0])
    
    min_value = yp0
    U_optimal = U_prev
    
    for i in np.arange(N_steps):
        #Perform random unitary rotation of Up        
        U_new = np.dot(U_prev, randU(X[0].blockshape[1], stepsize = stepsize)) #
        
        #Rotate every tensor in X
        
        if order == 2:
            #for j in np.arange(X[0].blocks.shape[0]-1):
            for k in np.arange(len(X)):
                X_rot[k].cset(c, np.dot(U_new.conj().T,
                                        np.dot(X[k].cget([0,0,0]),
                                               U_new)))
                    
        if order == 1:
            for k in np.arange(len(X)):
                X_rot[k].cset(c, np.dot(X[k].cget([0,0,0]), U_new))
                
        #
        #for j in np.arange(len(X)):
        #    X_rot[j].blocks = np.fft.ifft(X_kn[j], axis = 0).real
        #    #print(X_rot[j].blocks.shape)
        
        yn = F(X_rot)
        
        if yn<yp:
            #Cp = Cn
            #print(yn)
            yp = yn
            U_prev = U_new
            N_accept_steps += 1
        # if not, compute acceptance probability
        # this basically avoids getting trapped in
        # local minimas
        else:
            #print(np.exp(-np.abs(yp-yn)/T))
            #if np.exp(-np.abs(yn-yp)/T)>0.5:
            random = np.random.uniform()
            
            if np.exp(-(yn-yp)/T) > random:
                N_uphill_steps += 1
                N_accept_steps += 1
                #Cp = Cn
                yp = yn
                U_prev = U_new
        #decrease temp
        if N_uphill_steps>=uphill_trigger:
            T *= T_decay
            
        if yp < min_value:
            min_value = yp
            U_optimal = U_prev
            
        if(i==0):
            print("%5.2f percent complete, acceptance: %.2f, temperature: %8.2e, objective function: %.2f ." \
                  % (100*i/np.float(N_steps), (N_accept_steps/(i+1)), T, yp))
        if(i%100==0):
            print("%5.2f percent complete, acceptance: %.2f, temperature: %8.2e, objective function: %.2f ." \
                  % (100*i/np.float(N_steps), (N_accept_steps/(i+1)), T, yp), "\r", end="")
            #print()
    summarize_anneal(N_uphill_steps, N_accept_steps, N_steps, T, yp0, min_value)
    
    return U_optimal



def ensemble_annealing(objective_function,
                       tensors,
                       initial_temperature,
                       N_steps_total = 10000,
                       N_walkers = 5,
                       T_decay = 0.997,
                       stepsize = 1.0,
                       order = 2,
                       uphill_trigger = 0):
    '''
    Do simulated annealing with an ensamble of 'N_walkers' 
    walkers. The number of steps per walker is 
    'N_steps_total' / 'N_walkers'.
    
    See 
    
    P. Salamon, P. Sibani, and R. Frost, 
    Facts, Conjectures, and Improvements for Simulated Annelaing,
    SIAM, (2002)
    
    for a desctiption of simulated annealing using ensembles.
    '''
    value_min = 10**10
    rotation_matrix = None
    for i in range(N_walkers):

        print('\nStarting optimization for walker',
              i,
              '\n')
        
        N_steps = int(N_steps_total / N_walkers)
        
        coeffs, objective_value, unitary_matrix = \
            center_annealing(objective_function,
                             tensors,
                             initial_temperature,
                             N_steps,
                             T_decay,
                             stepsize,
                             order,
                             uphill_trigger)
        
        print('\nFinal objective-function value of walker',
              '{:>4}'.format(i),
              ':',
              '{0:10.6f}'.format(objective_value))
        
        if objective_value < value_min:
            value_min = objective_value
            rotation_matrix = unitary_matrix
            
    print('The optimal objective-function value in this ensemble:',
          '{0:10.6f}'.format(value_min),
          '\n')
    
    return 0, value_min, rotation_matrix

if __name__== "__main__":
    os.environ["LIBINT_DATA_PATH"] = os.getcwd() 
    print("""#####################################################
##  ,--.   ,--.      ,------.  ,------. ,-----.    ##
##   \  `.'  /,-----.|  .-.  \ |  .---''  .--./    ##
##    .'    \ '-----'|  |  \  :|  `--, |  |        ## 
##   /  .'.  \       |  '--'  /|  `---.'  '--'\    ##
##  '--'   '--'       ==== Annealing Tools ====    ##
##                Author : Audun Skau Hansen       ##
##                                                 ##
##  Use keyword "--help" for more info             ## 
#####################################################""")
    parser = argparse.ArgumentParser(prog = "Annealing tools for periodic functions",
                                     description = "Optimization of Crystal orbital systems",
                                     epilog = "Author: Audun Skau Hansen (e-mail: a.s.hansen@kjemi.uio.no)")
    parser.add_argument("project_file", type = str, help ="input file for project (.d12 file)")
    parser.add_argument("coefficient_matrix", type= str,help="Block Toeplitz coefficient matrix with coefficients")
    parser.add_argument("overlap_matrix", type= str,help="Block Toeplitz overlap matrix with coefficients")
    parser.add_argument("-objective", type = str, default = "psm-1", help="Name of the objective function to optimize")
    parser.add_argument("-ncore", type = int, default = 0, help="Number of core orbitals. (will not be affected)")
    parser.add_argument("-m", type = int, default = 1, help="Power of the moment (PXM-m).")
    parser.add_argument("-T0", type = float, default = 1, help="Initial temperature.")
    parser.add_argument("-nsteps", type = int, default = 40000, help="Number of annealing steps.")
    parser.add_argument("-stepsize", type = float, default = 0.005, help="Stepsize of rotations.")
    parser.add_argument("-save", type = str, default = "c_new", help="Name of the savefile for the optimized set.")


    
    args = parser.parse_args()

    p = pr.prism(args.project_file)
    c = tp.tmat()
    c.load(args.coefficient_matrix)
    s = tp.tmat()
    s.load(args.overlap_matrix)

    st = tp.tmat()
    sc = tp.lattice_coords([1,1,1])
    st.load_nparray(np.ones((sc.shape[0], p.get_n_ao(), p.get_n_ao()), dtype = float), sc)
    st.blocks*=0

    spreads = lambda tens : np.diag(tens[0].cget([0,0,0]) - tens[1].cget([0,0,0])**2 - tens[2].cget([0,0,0])**2 - tens[3].cget([0,0,0])**2)

    if args.objective == "psm-1k":
        #tensors, assembler = 
        c_occ, c_virt = PRI.occ_virt_split(c, p)
        
        tensors, psm1 = objective_functions.psm_m(c_occ, p, s.coords, m=1)


        tensors_k = []
        for i in np.arange(len(tensors)):
            n_points = tp.n_lattice(tensors[i])
            tensors_k.append(tp.transform(tensors[i], np.fft.fftn, n_points = n_points))


        

        f_psm1 = lambda tens : -np.sum(np.diag(tens[1].cget([0,0,0]) + tens[2].cget([0,0,0]) + tens[3].cget([0,0,0]))**m) # PSM-m objective function
    


        print("Initial:", psm1(tensors))
        #print(wcenters.T)
        print(psm1(tensors))
        print(spreads(tensors))
        u = center_annealing(psm1, tensors_k, 1.0, N_steps = 100000, T_decay = 0.997,
                     stepsize = 0.005, order = 2, uphill_trigger = 0)

        for i in np.arange(len(tensors)):
            for j in np.arange(tensors[i].coords.shape[0]):
                tensors[i].blocks[j] = np.dot(u.T, np.dot(tensors[i].blocks[j], u))

        print(psm1(tensors))
        print(spreads(tensors))

        u = center_annealing(psm1, tensors, 1.0, N_steps = 100000, T_decay = 0.997,
                     stepsize = 0.005, order = 2, uphill_trigger = 0)
        
        for i in np.arange(len(tensors)):
            for j in np.arange(tensors[i].coords.shape[0]):
                tensors[i].blocks[j] = np.dot(u.T, np.dot(tensors[i].blocks[j], u))
        
        #print(psm1(tensors))
        #print(spreads(tensors))

    if args.objective == "psm-1":
        #tensors, assembler = 
        c_occ, c_virt = PRI.occ_virt_split(c, p)
        dens_old = c_occ.circulantdot(c_occ.tT())
        utot = np.zeros((p.get_n_ao(), p.get_n_ao()), dtype = float)

        if args.ncore >0:
            #split further
            c_valence = tp.tmat()
            c_valence.load_nparray(c_occ.blocks[:, :, args.ncore:], c_occ.coords)

            c_core = tp.tmat()
            c_core.load_nparray(c_occ.blocks[:, :, :args.ncore], c_occ.coords)

            # optimize core
            tensors, ofunc = objective_functions.psm_m(c_core, p, s.coords, m=args.m)
            u_core = center_annealing(ofunc, tensors, args.T0, N_steps = args.nsteps, T_decay = 0.997,
                     stepsize = args.stepsize, order = 2, uphill_trigger = 0)

            # rotate core
            for i in np.arange(c_core.blocks.shape[0]-1):
                c_core.blocks[i] = np.dot(c_core.blocks[i], u_core)

            utot[:args.ncore, :args.ncore] = u_core
            print("Max deviation from unity:", np.abs(np.dot(u_core.T, u_core)-np.eye(u_core.shape[0])).max())


            # optimize valence
            tensors, ofunc = objective_functions.psm_m(c_valence, p, s.coords, m=args.m)
            u_valence = center_annealing(ofunc, tensors, args.T0, N_steps = args.nsteps, T_decay = 0.997,
                     stepsize = args.stepsize, order = 2, uphill_trigger = 0)

            # rotate core
            for i in np.arange(c_core.blocks.shape[0]-1):
                c_valence.blocks[i] = np.dot(c_valence.blocks[i], u_valence)
            
            utot[args.ncore:p.get_nocc(), args.ncore:p.get_nocc()] = u_valence
            print("Max deviation from unity:", np.abs(np.dot(u_valence.T, u_valence)-np.eye(u_valence.shape[0])).max())





        else:
            # optimize occupied
            tensors, ofunc = objective_functions.psm_m(c_occ, p, s.coords, m=args.m)
            u_occ = center_annealing(ofunc, tensors, args.T0, N_steps = args.nsteps, T_decay = 0.997,
                     stepsize = args.stepsize, order = 2, uphill_trigger = 0)

            # rotate occupied
            for i in np.arange(c_occ.blocks.shape[0]-1):
                c_occ.blocks[i] = np.dot(c_occ.blocks[i], u_occ)
            
            utot[:p.get_nocc(), :p.get_nocc()] = u_occ
            print("Max deviation from unity:", np.abs(np.dot(u_occ.T, u_occ)-np.eye(u_occ.shape[0])).max())

        # optimize virtuala
        tensors, ofunc = objective_functions.psm_m(c_virt, p, s.coords, m=args.m)
        u_virt = center_annealing(ofunc, tensors, args.T0, N_steps = args.nsteps, T_decay = 0.997,
                    stepsize = args.stepsize, order = 2, uphill_trigger = 0)

        utot[p.get_nocc():, p.get_nocc():] = u_virt
        print("Max deviation from unity:", np.abs(np.dot(u_virt.T, u_virt)-np.eye(u_virt.shape[0])).max())
        # rotate virtuals
        for i in np.arange(c_virt.blocks.shape[0]-1):
            c_virt.blocks[i] = np.dot(c_virt.blocks[i], u_virt)

        
        nb = np.zeros_like(c.blocks[:-1])
        if args.ncore!=0:
            nb[:,:,:args.ncore] = c_core.cget(c.coords)
            nb[:,:,args.ncore:p.get_nocc()] = c_valence.cget(c.coords)
        else:
            nb[:,:,:p.get_nocc()] = c_occ.cget(c.coords)
        
        nb[:,:,p.get_nocc():] = c_virt.cget(c.coords)


        U_tot = tp.tmat()
        U_tot.load_nparray(c.blocks[:-1], c.coords)
        U_tot.blocks *= 0
        U_tot.cset([0,0,0], utot)


        c_new = tp.tmat()
        c_new.load_nparray(nb, c.coords)

        c_new = c.circulantdot(U_tot)

        # recompute spreads and so on
        tensors, psm1 = objective_functions.psm_m(c_new, p, s.coords, m=args.m)
        wcenter = np.array([np.diag(tensors[1].cget([0,0,0])), np.diag(tensors[2].cget([0,0,0])),np.diag(tensors[3].cget([0,0,0]))]).T

        print("New centers:")
        print(wcenter)

        np.save("%s_centers.npy" % args.save, wcenter)

        print("New spreads:")
        print(spreads(tensors))

        np.save("%s_spreads.npy" % args.save, spreads(tensors))

        c_new.save("%s.npy" % args.save)

        # Compare density

        c_occ_new, c_virt_new = PRI.occ_virt_split(c_new, p)
        dens_new = c_occ_new.circulantdot(c_occ_new.tT())

        #smo = c_new.tT().circulantdot(s.circulantdot(c_new))
        #print(smo.cget([0,0,0]))
        print("Max deviation in density matrix:", np.abs(dens_new.cget([0,0,0]) - dens_old.cget([0,0,0])).max())
        

        




        
        """

        #f_psm1 = lambda tens : -np.sum(np.diag(tens[1].cget([0,0,0]) + tens[2].cget([0,0,0]) + tens[3].cget([0,0,0]))**m) # PSM-m objective function
    


        print("Initial:", psm1(tensors))
        print(spreads(tensors))
        np.save("psm1_spreads.npy", spreads(tensors))
        u = center_annealing(psm1, tensors, 1.0, N_steps = 100000, T_decay = 0.997,
                     stepsize = 0.005, order = 2, uphill_trigger = 0)

        print("Deviation from unity:", np.linalg.norm(np.eye(u.shape[0]) - np.dot(u.T, u)))

        for i in np.arange(len(tensors)):
            for j in np.arange(tensors[i].coords.shape[0]):
                tensors[i].blocks[j] = np.dot(u.T, np.dot(tensors[i].blocks[j], u))

        d_occ = c_occ.circulantdot(c_occ.tT())

        for i in np.arange(c_occ.blocks.shape[0]-1):
            c_occ.blocks[i] = np.dot(c_occ.blocks[i], u)

        udu_occ = c_occ.circulantdot(c_occ.tT())

        print(psm1(tensors))
        print(spreads(tensors))

        u = center_annealing(psm1, tensors, 1.0, N_steps = 100000, T_decay = 0.997,
                     stepsize = 0.005, order = 2, uphill_trigger = 0)
        
        for i in np.arange(len(tensors)):
            for j in np.arange(tensors[i].coords.shape[0]):
                tensors[i].blocks[j] = np.dot(u.T, np.dot(tensors[i].blocks[j], u))
        """

    if args.objective == "psm_sunmat":
        c_occ, c_virt = PRI.occ_virt_split(c, p)

        c_valence = tp.tmat()
        c_valence.load_nparray(c_occ.blocks[:,:,2:], c_occ.coords)

        tensors, psm1 = objective_functions.psm_m(c_valence, p, s.coords, m=args.m)
         
        n,u = sm.full_suN(p.get_nocc()-2) #generate su(nocc) matrix
        #n,u = sm.suNia(4, [0,1,2,3], [0,1,2,3])
        x2 = tensors[0].cget([0,0,0])
        x = tensors[1].cget([0,0,0])
        y = tensors[2].cget([0,0,0])
        z = tensors[3].cget([0,0,0])

        def objective_(params):
            up = u(params)
            return np.sum(np.diag(  np.dot(up.T, np.dot(x2, up)) 
                                  - np.dot(up.T, np.dot(x , up))**2
                                  - np.dot(up.T, np.dot(y , up))**2
                                  - np.dot(up.T, np.dot(z , up))**2 )**args.m)
    


        dm = ag.grad(objective_)
        #dm = ag.jacobian(objective_)
        print("AGRAD:")
        print(dm(np.zeros(n, dtype = float)))

        params = np.zeros(n, dtype = float)
        print(objective_(params))
        for i in np.arange(40):
            #print(i, objective)

            params -=  args.stepsize*dm(params) #gradient method
            #params -= np.dot(np.linalg.inv(dm(params)), objective_(params)) #newton method
            print(i, objective_(params))

        
        up = u(params)
        for i in np.arange(len(tensors)):
            for j in np.arange(tensors[i].coords.shape[0]):
                tensors[i].blocks[j] = np.dot(up.T, np.dot(tensors[i].blocks[j], up))

        wcenter = np.array([np.diag(tensors[1].cget([0,0,0])), np.diag(tensors[2].cget([0,0,0])),np.diag(tensors[3].cget([0,0,0]))]).T

        print("New centers:")
        print(wcenter)

        #np.save("%s_centers.npy" % args.save, wcenter)

        print("New spreads:")
        print(spreads(tensors))
            


        #f_psm1 = lambda tens : np.sum(np.diag(tens[0].cget([0,0,0]) - tens[1].cget([0,0,0])**2 - tens[2].cget([0,0,0])**2 - tens[3].cget([0,0,0])**2)**m) 

        



    #print("Final results")
    #print(psm1(tensors))
    #print(spreads(tensors))

    # Compute required integrals

    

