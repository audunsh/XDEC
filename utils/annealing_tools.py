import numpy as np
from scipy.linalg import expm

import utils.toeplitz as tp


def summarize_anneal(N_uphill_steps, N_accept_steps, N_steps, T, yp0, yp):
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
    
def center_full_anneal(F, X, T, nocc, N_steps = 10000, T_decay = 0.997, stepsize = 1.0, order = 2, uphill_trigger = 0):
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
    for i in np.arange(N_steps):
        #Perform random unitary rotation of Up        
        U_new = np.dot(U_prev, birandU(X[0].blockshape[1], nocc = nocc, stepsize = stepsize)) #
        
        #Rotate every tensor in X
        
        if order == 2:
            #for j in np.arange(X[0].blocks.shape[0]-1):
            for k in np.arange(len(X)):
                X_rot[k].cset(c, np.dot(U_new.conj().T,np.dot(X[k].cget([0,0,0]), U_new)))
                    
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
            if np.exp(-np.abs(yn-yp)/T)>0.5:
                N_uphill_steps += 1
                N_accept_steps += 1
                #Cp = Cn
                yp = yn
                U_prev = U_new
        #decrease temp
        if N_uphill_steps>=uphill_trigger:
            T *= T_decay
        if(i==0):
            print("%5.2f percent complete, acceptance: %.2f, temperature: %8.2e, objective function: %.2f ." % (100*i/np.float(N_steps), (N_accept_steps/(i+1)), T, yp))
        if(i%100==0):
            print("%5.2f percent complete, acceptance: %.2f, temperature: %8.2e, objective function: %.2f ." % (100*i/np.float(N_steps), (N_accept_steps/(i+1)), T, yp), "\r", end="")
            #print()
    summarize_anneal(N_uphill_steps, N_accept_steps, N_steps,T, yp0, yp)
    
    
    return 0, yp, U_prev

def center_anneal(F, X, T, N_steps = 10000, T_decay = 0.997, stepsize = 1.0, order = 2, uphill_trigger = 0):
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
            if np.exp(-np.abs(yn-yp)/T)>0.5:
                N_uphill_steps += 1
                N_accept_steps += 1
                #Cp = Cn
                yp = yn
                U_prev = U_new
        #decrease temp
        if N_uphill_steps>=uphill_trigger:
            T *= T_decay
        if(i==0):
            print("%5.2f percent complete, acceptance: %.2f, temperature: %8.2e, objective function: %.2f ." \
                  % (100*i/np.float(N_steps), (N_accept_steps/(i+1)), T, yp))
        if(i%100==0):
            print("%5.2f percent complete, acceptance: %.2f, temperature: %8.2e, objective function: %.2f ." \
                  % (100*i/np.float(N_steps), (N_accept_steps/(i+1)), T, yp), "\r", end="")
            #print()
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
    
    return 0, min_value, U_optimal


def bi_anneal(F, X_a, X_b, T, N_steps = 10000, T_decay = 0.997, stepsize = 1.0, order = 2):
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
    ##                                     ##
    ## This function rotates the set of    ##
    ## input-tensors such that             ##
    ## X_fin = np.dot(U.T, np.dot(X_in, U) ##
    ## minimize the function F(X_fin).     ##
    ##                                     ##
    #########################################
    """
        
    #yp = F(X) #initial value of functional

    #yp0 = 1*yp
    
    N_uphill_steps = 0
    N_accept_steps = 0
    
    #make toeplitz arrays complex
    X_rot_a = X_a*1
    X_rot_b = X_b*1
    #for i in np.arange(len(X)):
        #X[i].blocks = np.array(X[i].blocks, dtype = complex)
    #    X_rot_a.append(X_a[i]*1) #store a copy for rotation
    #    X_rot_b.append(X_b[i]*1) #store a copy for rotation

    #intermediate storage of fft-blocks
    
    #X_k  = []
    #X_kn = [] 
    
    #for i in np.arange(len(X)):
    #    X_k.append( np.fft.fft(X[i].blocks, axis = 0))
    #    X_kn.append(np.fft.fft(X[i].blocks, axis = 0))

    U_prev_a = np.eye(X_a.blockshape[0], dtype = float) #initial unitary matrix
    U_prev_b = np.eye(X_b.blockshape[1], dtype = float) #initial unitary matrix
    
    c0 = np.array([0,0,0])
    
    OF = lambda x : -1000*np.sum(x[x<np.mean(x)]) + np.sum(x[x>np.mean(x)])
    
    yp = 100
    
    for i in np.arange(N_steps):
        #Perform random unitary rotation of Up        
        U_new_a = np.dot(U_prev_a, randU(X_a.blockshape[0], stepsize = stepsize)) #
        U_new_b = np.dot(U_prev_b, randU(X_b.blockshape[1], stepsize = stepsize)) #
        
        
        
        
        #Rotate every tensor in X
        
        for c in X_a.coords:
            X_rot_a.cset(c, np.dot(U_prev_a.conj().T,X_a.cget(c)))
            X_rot_b.cset(c, np.dot(X_b.cget(c), U_prev_b))
        
        
        X_rot_a.blocks = np.abs(X_rot_a.blocks)
        
        X_rot_b.blocks = np.abs(X_rot_b.blocks)
        
        D_ia = X_rot_a.tT().cdot(X_rot_b)
        
        #yn = D_ia.cget(c0)
                    
        #if order == 1:
        #    for k in np.arange(len(X)):
        #        X_rot[k].cset(c, np.dot(X[k].cget([0,0,0]), U_new))

        #
        #for j in np.arange(len(X)):
        #    X_rot[j].blocks = np.fft.ifft(X_kn[j], axis = 0).real
        #    #print(X_rot[j].blocks.shape)
        
        yn = OF(D_ia.cget(c0))

        if yn<yp:
            #Cp = Cn
            #print(yn)
            yp = yn
            U_prev_a = U_new_a
            U_prev_b = U_new_b
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
                U_prev_a = U_new_a
                U_prev_b = U_new_b
        #decrease temp
        T *= T_decay
        if(i%100==0):
            print("%.2f percent complete, acceptance: %.2f, temperature: %.2e, objective function: %.2f ." % (100*i/np.float(N_steps), (N_accept_steps/(i+1)), T, yp), "\r", end="")
            #print()
    print(D_ia.cget([0,0,0]))
    summarize_anneal(N_uphill_steps, N_accept_steps, N_steps,T, yp0, yp)
    print('')
    
    return 0, yp, U_prev


def center_anneal_ndarray(F, X, T, N_steps = 10000,
                          T_decay = 0.997, stepsize = 1.0,
                          order = 2):
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
    ##                                     ##
    ## This function rotates the set of    ##
    ## input-tensors such that             ##
    ## X_fin = np.dot(U.T, np.dot(X_in, U) ##
    ## minimize the function F(X_fin).     ##
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
        #X[i].blocks = np.array(X[i].blocks, dtype = complex)
        X_rot.append(X[i]*1) #store a copy for rotation
        
    #intermediate storage of fft-blocks
    
    #X_k  = []
    #X_kn = [] 
    
    #for i in np.arange(len(X)):
    #    X_k.append( np.fft.fft(X[i].blocks, axis = 0))
    #    X_kn.append(np.fft.fft(X[i].blocks, axis = 0))
    
    U_prev = np.eye(X[0].shape[0], dtype = float) #initial unitary matrix
    c = np.array([0,0,0])
    for i in np.arange(N_steps):
        #Perform random unitary rotation of Up        
        U_new = np.dot(U_prev, randU(X[0].shape[0], stepsize = stepsize)) #
        
        #Rotate every tensor in X
        
        if order == 2:
            #for j in np.arange(X[0].blocks.shape[0]-1):
            for k in np.arange(len(X)):
                X_rot[k] = np.dot(U_new.conj().T,
                                  np.dot(X[k],
                                         U_new))
                    
        if order == 1:
            for k in np.arange(len(X)):
                X_rot[k] = np.dot(X[k], U_new)
                
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
            if np.exp(-np.abs(yn-yp)/T)>0.5:
                N_uphill_steps += 1
                N_accept_steps += 1
                #Cp = Cn
                yp = yn
                U_prev = U_new
        #decrease temp
        T *= T_decay
        if(i%100==0):
            print("%.2f percent complete, acceptance: %.2f, temperature: %.2e, objective function: %.2f ." % (100*i/np.float(N_steps), (N_accept_steps/(i+1)), T, yp), "\r", end="")
            #print()
    summarize_anneal(N_uphill_steps, N_accept_steps, N_steps,T, yp0, yp)
    print('')
    
    return 0, yp, U_prev


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
