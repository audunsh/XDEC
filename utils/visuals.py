
import numba
import numpy as np
import sympy as sp

from ast import literal_eval

import solid_harmonics as sh
import crystal_interface as ci



def get_lambda_basis(molfile):
    #b,g = ci.dalton2lambdas("/Users/audunhansen/PeriodicDEC/LiH.INP")
    b,g = ci.dalton2lambdas(molfile)
    
    bs = []
    positions = g[0] #/0.52917720830000000000000000000
    #print("positions:", g)
    #atomic_numbers = g[1]
    #print(g[2])
    lattice = g[2] #needs parsing
    R = []
    for r in lattice:
        #print(r.split("=")[1])
        R.append([literal_eval(i) for i in r.split("=")[1].split()[:3]])
    #print(R)
    
    atoms = []
    
    for a in range(len(b[0])):
        atom = b[0][a]
        pos = positions[a]
        
        atoms.append([pos, atom])
        l = 0
        for shelltype in atom:
            for contracted in shelltype:
                c = np.array(contracted)
                e = c[:,0] #exponent
                w = c[:,1] #weights
                for m in np.arange(-l, l+1):
                    spfunc, lfunc = sh.get_contracted(e,w,l,m)
                    bs.append([lfunc, np.array(pos)/0.52917720830000000000000000000, spfunc])
                    #converting back to bohr

                    
            l += 1

    return bs, np.array(R), atoms

"""
def get_lambda_basis_absdist(molfile):
    #returns a basis of lambda functions where the gaussians are positioned along the x-axis
    #depending on their position norm
    #b,g = ci.dalton2lambdas("/Users/audunhansen/PeriodicDEC/LiH.INP")
    b,g = ci.dalton2lambdas(molfile)
    
    bs = []
    positions = g[0] #/0.52917720830000000000000000000
    #print("positions:", g)
    #atomic_numbers = g[1]
    #print(g[2])
    lattice = g[2] #needs parsing
    R = []
    for r in lattice:
        #print(r.split("=")[1])
        R.append([literal_eval(i) for i in r.split("=")[1].split()[:3]])
    #print(R)
    
    atoms = []
    
    for a in range(len(b[0])):
        atom = b[0][a]
        pos = positions[a]
        pos = np.array([np.sqrt(np.sum(pos**2)),0,0])
        
        atoms.append([pos, atom])
        l = 0
        for shelltype in atom:
            for contracted in shelltype:
                c = np.array(contracted)
                e = c[:,0] #exponent
                w = c[:,1] #weights
                for m in np.arange(-l, l+1):
                    spfunc, lfunc = sh.get_contracted(e,w,l,m)
                    bs.append([lfunc, np.array(pos)/0.52917720830000000000000000000, spfunc])
                    #converting back to bohr

                    
            l += 1

    return bs, np.array(R), atoms
"""

class lambda_basis:
    def __init__(self, molfile, C, Cc):
        self.bs, self.lattice, self.atoms = get_lambda_basis(molfile)
        self.C = C
        self.Cc = Cc
        
    @numba.autojit()
    def orb_at(self, x,y,z, p = 0, thresh = 10e-10):
        #evaluate orbital p at x,y,z
        #print((x+y+z).shape)
        ret = np.zeros((x+y+z).shape, dtype = float)
        #for i in range(len(self.Cc)):
        #    T = np.dot(self.lattice, self.Cc[i])
        #    
        #    for j in range(len(self.bs)):
        #        t = self.bs[j][1]


        sporb = 0
        
        for i in range(len(self.Cc)):
            T = np.dot(self.Cc[i], self.lattice) #translation of coordinates
            #print(self.lattice,self.Cc[i], np.dot(self.lattice,self.Cc[i]))
            #print(T)
            X = x-T[0]
            Y = y-T[1]
            Z = z-T[2]
            for j in range(len(self.bs)):
                t = self.bs[j][1]

                
                if np.abs(self.C[i,j,p])>thresh: 
                    
                    #ret += self.C[i,j,p]*self.bs[j][0](X-t[0],Y-t[1],Z-t[2])
                    #ret += self.C[i,j,p]*np.exp(-.1*((X-t[0])**2 + (Y-t[1])**2 + (Z-t[2])**2))
                    ret += self.C[i,j,p]*self.bs[j][0](X-t[0],Y-t[1],Z-t[2])
                    
                    
        return ret
    
    #@numba.autojit()
    def orb_1d_at(self, x,y,z, p = 0, thresh = 10e-10):
        #evaluate orbital p at r = np.sqrt(x**2 + y **2 + z**2)
        #print(type(x))
        
        ret = np.zeros(x.shape, dtype = float)
        #for i in range(len(self.Cc)):
        #    T = np.dot(self.lattice, self.Cc[i])
        #    
        #    for j in range(len(self.bs)):
        #        t = self.bs[j][1]


        #sporb = 0
        
        for i in range(len(self.Cc)):
            T = np.dot(self.Cc[i],self.lattice) #translation of coordinates
            #print(self.lattice,self.Cc[i], np.dot(self.lattice,self.Cc[i]))
            #print(T)
            X = x-T[0]
            Y = y-T[1]
            Z = z-T[2]
            #print(X.shape, type(X))
            for j in range(len(self.bs)):
                t = self.bs[j][1] #AO center

                
                if np.abs(self.C[i,j,p])>thresh: 
                    
                    #ret += self.C[i,j,p]*self.bs[j][0](X-t[2],Y-t[1],Z-t[0])
                    ret += self.C[i,j,p]*self.bs[j][0](X-t[0],Y-t[1],Z-t[2])
                    
                    
        return ret
    
    
    def AO_overlap(self, L, alpha,M, beta, x,y,z):
        # returns product of orbital phi_{L, alpha} phi_{M, beta} in points x,y,z
        print(self.lattice)
        Ta = np.dot(L, self.lattice)
        Tb = np.dot(M, self.lattice)
        print(Ta, Tb)
        ta = self.bs[alpha][1]
        tb = self.bs[ beta][1]
        
        print(ta, tb)
        return self.bs[alpha][0](x-Ta[0]-ta[0], y-Ta[1]-ta[1],z-Ta[2]-ta[2])*\
               self.bs[ beta][0](x-Tb[0]-tb[0], y-Tb[1]-tb[1],z-Tb[2]-tb[2]), self.bs[alpha][2]*self.bs[beta][2]
    
    def sporb_at(self, p = 0):
        #return symbolic orbital p at x,y,z

        sporb = 0

        for i in range(len(self.Cc)):
            T = np.dot(self.lattice,self.Cc[i]) #translation of coordinates
            for j in range(len(self.bs)):
                t = self.bs[j][1]
                Cjp = self.C[i,j,p]

                if np.abs(Cjp)>10e-8: 
                    
                    fct = Cjp*self.bs[j][2]
                    fct = fct.subs(sp.Symbol("x"), sp.Symbol("x")-t[0]-T[0])
                    fct = fct.subs(sp.Symbol("y"), sp.Symbol("y")-t[1]-T[1])
                    fct = fct.subs(sp.Symbol("z"), sp.Symbol("z")-t[2]-T[2])
                    sporb +=fct
                    
                    
                
        #print(sporb)
        return sporb
        
def vis_orbital(project_folder_f, orbital_index = 1, Nx = 200):
    Cc = np.load(project_folder_f + "/crystal_reference_coords.npy")
    C  = np.load(project_folder_f + "/crystal_reference_state.npy")
    
    Cc = np.load(project_folder_f + "/lsdalton_reference_coords.npy")
    C  = np.load(project_folder_f + "/lsdalton_reference_state.npy")


    b = lambda_basis(project_folder_f + "/LSDalton/MOLECULE.INP", C, Cc)
    
    region = np.max(Cc, axis = 0) +1

    Lx,Ly,Lz = np.dot(region, b.lattice)*5
    Ly *= 2
    Lz *= 2
    Rx = b.lattice[0][0]
    Ry = b.lattice[1][1]
    Rz = b.lattice[2][2]
    
    #load wannier centers (if present)
    
    #wc_occ = np.load(project_folder_f + "/crystal_wannier_centers_occ.npy")
    #wc_vir = np.load(project_folder_f + "/crystal_wannier_centers_vir.npy")
    #wc = np.append(wc_occ, wc_vir, axis = 0) #wannier centers
    
    #print(wc_vir)
    
    
    
    #number of points in each direction: 200 in 0th
    #Nx = 200
    Ny = 2*int(2*Nx*Ly/(2*Lx))
    Nz = 2*int(2*Nx*Lz/(2*Lx))
    print (Nx,Ny,Nz)
    
    
    
    x = np.linspace(-Lx,Lx,Nx)[:,None,None]
    y = np.linspace(-Ly,Ly,Ny)[None,:,None]
    z = np.linspace(-Lz,Lz,Nz)[None,None,:]
    
    
    X = np.linspace(-Lx,Lx,Nx)
    
    Z = b.orb_at(x,y,z,p=orbital_index) #**2
    
    return Z, [Lx,Nx,Ly,Ny,Lz,Nz], b.sporb_at(p=orbital_index)

def visualize2(project_folder_f, orbital_index):
    Cc = np.load(project_folder_f + "/crystal_reference_coords.npy")
    C  = np.load(project_folder_f + "/crystal_reference_state.npy")


    b = lambda_basis(project_folder_f + "/LSDalton/MOLECULE.INP", C, Cc)
    
    region = np.max(Cc, axis = 0) +1

    Lx,Ly,Lz = np.dot(region, b.lattice)*5
    Ly *= 2
    Lz *= 2
    Rx = b.lattice[0][0]
    Ry = b.lattice[1][1]
    Rz = b.lattice[2][2]
    
    #load wannier centers (if present)
    
    #wc_occ = np.load(project_folder_f + "/crystal_wannier_centers_occ.npy")
    #wc_vir = np.load(project_folder_f + "/crystal_wannier_centers_vir.npy")
    #wc = np.append(wc_occ, wc_vir, axis = 0) #wannier centers
    
    #print(wc_vir)
    
    
    
    #number of points in each direction: 200 in 0th
    Nx = 200
    Ny = 2*int(2*Nx*Ly/(2*Lx))
    Nz = 2*int(2*Nx*Lz/(2*Lx))
    print (Nx,Ny,Nz)
    
    
    
    x = np.linspace(-Lx,Lx,Nx)[:,None,None]
    y = np.linspace(-Ly,Ly,Ny)[None,:,None]
    z = np.linspace(-Lz,Lz,Nz)[None,None,:]
    
    
    X = np.linspace(-Lx,Lx,Nx)
    
    Z = b.orb_at(x,y,z,p=orbital_index) #**2
    
    #optional: tag unit cell
    #Z[x[:,None,None]>0 and x[:,None,None]<b.lattice[0][0]] += .4
    R = 7.7
    
    Z2 = Z*0.0
    
    
    


    import matplotlib.pyplot as plt

    fs = 8
    
    
    im1 = np.sum(Z, axis = 1)
    im2 = np.sum(Z, axis = 2)
    
    
    #print(Z.max(), Z.min())
    #print(im1.max(), im2.max())
    mx = np.max([im1.max(), im2.max()])
    mn = np.min([im1.min(), im2.min()])
    #print(mn,mx)

    lvl = np.linspace(mn,mx,100)
    
    lvl2 = np.linspace(mn,mx,20)
    
    
    f1 = plt.figure(1, figsize = (fs*Nz/float(Nx), fs))
    
    

    plt.contourf(np.linspace(-Lz,Lz,Nz), X,im1 , levels = lvl)
    plt.contour(np.linspace(-Lz,Lz,Nz), X,im1 , levels = lvl2, linewidths = 1.0, cmap = "bone")
    
    plt.savefig(project_folder_f+ "/orbital_%i_xz.png" % orbital_index)
    #plt.plot(wc[orbital_index][0], wc[orbital_index][2], "+", color = (0,0,0), alpha = .5)
    
    plt.figure(2, figsize = (fs*Ny/float(Nx), fs))
    
    #im = np.sum(Z, axis = 2)
    
    #lvl = np.linspace(im.min(),im.max(),100)
    
    #lvl2 =  np.linspace(im.min(),im.max(),20)

    plt.contourf(np.linspace(-Ly,Ly,Ny), X,im2, levels = lvl)
    plt.contour(np.linspace(-Ly,Ly,Ny), X,im2, levels = lvl2, linewidths = 1.0, cmap = "bone")
    #plt.plot(wc[orbital_index][0], wc[orbital_index][1], "+", color = (0,0,0), alpha = .5)
    plt.savefig(project_folder_f+ "/orbital_%i_xy.png" % orbital_index)
    plt.show()    
            

def vispylize(project_folder_f, orbital_index):
    Cc = np.load(project_folder_f + "/crystal_reference_coords.npy")
    C  = np.load(project_folder_f + "/crystal_reference_state.npy")

    #bs = get_lambda_basis("LiH.MOL")
    #for j in bs:
    #    print(j)
    #print(len(bs))

    b = lambda_basis(project_folder_f + "/LSDalton/MOLECULE.INP", C, Cc)
    
    region = np.max(Cc, axis = 0) +1
    #print(region)
    #print(b.lattice)
    Lx,Ly,Lz = np.dot(region, b.lattice)*7
    Rx = b.lattice[0][0]
    Ry = b.lattice[1][1]
    Rz = b.lattice[2][2]
    
    
    
    #number of points in each direction: 200 in 0th
    Nx = 600
    Ny = 2*int(2*Nx*Ly/(2*Lx))
    Nz = 2*int(2*Nx*Lz/(2*Lx))
    print (Nx,Ny,Nz)
    
    
    
    x = np.linspace(-Lx,Lx,Nx)[:,None,None]
    y = np.linspace(-Ly,Ly,Ny)[None,:,None]
    z = np.linspace(-Lz,Lz,Nz)[None,None,:]
    
    
    Z = b.orb_at(x,y,z,p=orbital_index) #**2
    
    #optional: tag unit cell
    #Z[x[:,None,None]>0 and x[:,None,None]<b.lattice[0][0]] += .4
    R = 7.7
    
    Z2 = Z*0.0
    
    
    lvl = np.linspace(Z.min(),Z.max(),100)


    #from mayavi import mlab
    #import matplotlib.pyplot as plt
    from vispy.plot import Fig
    fig = Fig()
    ax = fig[0,0]
    ax2 = fig[0,1]
    
    Z = np.abs(Z)
    #Z -= Z.min()
    Z/= Z.max()
    #Z[Z<.1] = 0
    #ax.volume(Z, method="mip", threshold = .4) #cmap = "single_hue", 
    Z2[(np.abs(x)<Rx)*(np.abs(y)<Ry)*(np.abs(z)<Rz)] += 100
    ax.volume(Z, method = "mip", threshold = .15, cmap = "grays") #['translucent', 'additive', 'iso', 'mip'
    ax2.volume(Z2, method = "mip")
    fig.bgcolor = (0,0,0)
    fig.show(run=True)
    #f.show()
    #plt.show()
    
    
    
def get_datapoints(project_folder_f, orbital_index):
    Cc = np.load(project_folder_f + "/crystal_reference_coords.npy")
    C  = np.load(project_folder_f + "/crystal_reference_state.npy")

    #bs = get_lambda_basis("LiH.MOL")
    #for j in bs:
    #    print(j)
    #print(len(bs))

    b = lambda_basis(project_folder_f + "/LSDalton/MOLECULE.INP", C, Cc)
    
    region = np.max(Cc, axis = 0) +1
    #print(region)
    #print(b.lattice)
    Lx,Ly,Lz = np.dot(region, b.lattice)*7
    Rx = b.lattice[0][0]
    Ry = b.lattice[1][1]
    Rz = b.lattice[2][2]
    
    
    
    #number of points in each direction: 200 in 0th
    Nx = 600
    Ny = 2*int(2*Nx*Ly/(2*Lx))
    Nz = 2*int(2*Nx*Lz/(2*Lx))
    print (Nx,Ny,Nz)
    
    
    
    x = np.linspace(-Lx,Lx,Nx)[:,None,None]
    y = np.linspace(-Ly,Ly,Ny)[None,:,None]
    z = np.linspace(-Lz,Lz,Nz)[None,None,:]
    
    
    return b.orb_at(x,y,z,p=orbital_index) #**2
    
    
    
"""
autumn=Colormap([(1., 0., 0., 1.), (1., 1., 0., 1.)]),
    blues=Colormap([(1., 1., 1., 1.), (0., 0., 1., 1.)]),
    cool=Colormap([(0., 1., 1., 1.), (1., 0., 1., 1.)]),
    greens=Colormap([(1., 1., 1., 1.), (0., 1., 0., 1.)]),
    reds=Colormap([(1., 1., 1., 1.), (1., 0., 0., 1.)]),
    spring=Colormap([(1., 0., 1., 1.), (1., 1., 0., 1.)]),
    summer=Colormap([(0., .5, .4, 1.), (1., 1., .4, 1.)]),
    fire=_Fire(),
    grays=_Grays(),
    hot=_Hot(),
    ice=_Ice(),
    winter=_Winter(),
    light_blues=_SingleHue(),
    orange=_SingleHue(hue=35)
"""

def vismplize(project_folder_f, orbital_index):
    Cc = np.load(project_folder_f + "/crystal_reference_coords.npy")
    C  = np.load(project_folder_f + "/crystal_reference_state.npy")

    #bs = get_lambda_basis("LiH.MOL")
    #for j in bs:
    #    print(j)
    #print(len(bs))

    b = lambda_basis(project_folder_f + "/LSDalton/MOLECULE.INP", C, Cc)
    
    #print(b.lattice)
    Lx = b.lattice[0][0]*(2*np.max(Cc[:,0])+1) #1D periodicity only
    
    #print(Lx)
    x = np.linspace(-Lx,Lx,400)
    Z = b.orb_at(x[:,None],0,x[None,:],p=orbital_index)**2
    lvl = np.linspace(Z.min(),Z.max(),100)
    
    

    #from mayavi import mlab
    import matplotlib.pyplot as plt
    plt.figure(1, figsize = (6,6))
    plt.hold("on")
    plt.contourf(x,x,Z, levels = lvl)
    for i in range(len(Cc)):
        #print(Cc[i])
        x0 = np.dot(Cc[i],b.lattice)
        x0 = Cc[i]*b.lattice[0]
        y0 = Cc[i]*b.lattice[1]
        z0 = Cc[i]*b.lattice[2]
        
        #xn = 
        #print(x0)
        #print(" ")
        X = x0 #[:,0] #, x0[:,1], x0[:,2]
        plt.plot([x[0], x[-1]], [x0[0], x0[0]], "-", color = (0,0,0), linewidth = 1)
    
        #plot atoms
        for j in range(len(b.atoms)):
            xn,yn,zn = b.atoms[j][0]
            n = b.atoms[j][1]
            plt.plot([yn],[x0[0]+ xn], "o", color = (0,0,0), alpha = .4)
    
    #plt.imshow(Z)
    plt.show()
    
    

    #f.show()
    #plt.show()

if __name__ == "__main__":
    import argparse
    import os
    import sys
    ########
    ##
    ##  Simple visualization toolkit
    ##  Requires python3.5 + PyQT + VisPy4
    ##
    ########

    parser = argparse.ArgumentParser(prog = "run_project",
                                     description= "Visualize orbital in refstate.",
                                     epilog= "Author: Audun Skau Hansen (e-mail: audunsh@student.matnat.uio.no")
    parser.add_argument("project_folder", type = str, help= "Project folder containing input files.")
    parser.add_argument("-orbital_number", type = int, default = 1, help ="Orbital to visualize.")
    parser.add_argument("-vislib", type = str, default = "matplotlib", help ="Visualization library")

    args = parser.parse_args()

    project_folder_f = os.getcwd() + "/" + args.project_folder
    if args.vislib == "vispy":
        vispylize(project_folder_f, args.orbital_number)
    elif args.vislib == "matplotlib":
        visualize2(project_folder_f, args.orbital_number)
    else:
        print("Unknown plot library: ", args.vislib)
    #vismplize(project_folder_f, args.orbital_number)

