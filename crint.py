#!/usr/bin/env python

import numpy as np
import argparse
import utils.prism as pr
import utils.toeplitz as tp

import utils.objective_functions as of

try:
    import orb_refinery as oref
except:
    oref = None
    print("\033[93m" + "WARNING"+ "\033[0m"+": Crint failed to import orb_refinery.")
    print("\033[93m" + "       "+ "\033[0m"+": Probably caused by dependencies such as Libint.")
    print("\033[93m" + "       "+ "\033[0m"+": Crint will still run with the '-skip_local' keyword.")

    
import subprocess as sp
import sys
import os

from ast import literal_eval

def run_crystal(filename, outputfolder, P):
    """
    Run Crystal for [filename], store output in [outputfolder]
    P = prism object
    """
    
    crystal_log = open(outputfolder + "/crystal_log.txt", "w")
    sp.call([P.crystal_exe_path + "/crystal"], stdin = open(infile), cwd=outputfolder, stdout = crystal_log)
    crystal_log.close()
    
    
    crystal_log = open(outputfolder + "/crystal_log.txt", "r")
    log_read = crystal_log.read()
    crystal_log.close()
    if("SCF ENDED - CONVERGENCE ON ENERGY" in log_read):
        print("##                                                 ##")
        print("##   Reference energy: %.5e                ##" % literal_eval(log_read.split("E(AU)")[1].split()[0]))
        print("##                                                 ##")
    else:
        print("\033[91m   Error \033[0m : Failed to parse valence string.")
        
    
    
    
def read_wann(folder, P, newk=9):
    C = tp.get_zero_tmat([newk, newk, newk],blockshape =(P.get_n_ao(), P.get_n_ao()))
    bands = C.load_gred_dat(folder + "/gred_valence.txt")
    bands = C.load_gred_dat(folder + "/gred_core.txt")
    bands = C.load_gred_dat(folder + "/gred_virtual.txt")
        

    # Permute blocks according to order of basis functions
    C.permute_blocks(P.get_permutation_vector(), axis = 0)


    # Read matrices
    S, F, D = get_matrices(folder + "/gred_virtual.txt", ordering = P.get_permutation_vector())
    
    #remove zero blocks
    C = tp.screen_tmat(C, tolerance = 1e-16)

    return S, F, D, C

def wannierization(folder, P, newk = 9, fullboys = 10, cyctol = 9, run = True, core = [1,2], virt = [3,4]):    
    if not run:
        print("##                                                 ##")
        print("##       Input file generation only                ##")        
        print("##                                                 ##")

        bands = [core[1]+1, virt[0]-1]
    
    # Generate input files

    properties_input = """NEWK
%i 0
1 0
LOCALWF
VALENCE
CYCTOL
%i %i %i
MAXCYCLE
100
BOYSCTRL
8 8 100
WANDM
-2 4 1
END
CRYAPI_OUT
END""" % (newk, cyctol, cyctol, cyctol)

    # Alternative input

    properties_input_ = """NEWK
%i %i %i
1 0
LOCALWF
VALENCE
END
CRYAPI_OUT
END
END""" % (newk, newk, newk)
    

    
    f = open(folder + "/wann_input_valence.d3" , "w")
    f.write(properties_input)
    f.close()

    
    C = tp.get_zero_tmat(P.ndim_layer(newk),blockshape =(P.get_n_ao(), P.get_n_ao()))
    #print(C.coords)
    
    
    if run:
        # Run properties, log to txt
        properties_log = open(folder + "/properties_log_valence.txt", "w")
        sp.call([P.crystal_exe_path + "/properties"], stdin = open(folder + "/wann_input_valence.d3"), cwd=folder, stdout = properties_log)
        properties_log.close()


        # Run Cryapi for valence
        cryapi_out = open(folder + "/gred_valence.txt", "w")
        sp.call([cryapi_exe], cwd = folder, stdout = cryapi_out)
        cryapi_out.close()


        # Initialize coefficient matrix 
        #print(P.get_n_ao())
        #C = tp.get_tmat_from_gred(folder + "/gred_valence.txt", blockshape =(P.get_n_ao(), P.get_n_ao()))

        # Read valence orbitals
        bands = C.load_gred_dat(folder + "/gred_valence.txt")

        """
        if P.get_nocc()-len(bands)<=1:
            core = [0, P.get_nocc()]
            virt = [P.get_nocc()+1, P.get_n_ao()]
            print("##       Core orbitals   : %.2i-%.2i                   ##" % (core[0], core[-1]))
            print("##       Virtual orbitals: %.2i-%.2i                   ##" % (virt[0], virt[-1]))      
            print("##                                                 ##")
        """

        if True:
            core = [0, bands[0]-1]
            virt = [bands[-1]+1,P.get_n_ao()]
        
            print("##       Core orbitals   : %.2i-%.2i                   ##" % (core[0], core[-1]))
            print("##       Valence orbitals: %.2i-%.2i                   ##" % (core[-1]+1, virt[0]-1))
            print("##       Virtual orbitals: %.2i-%.2i                   ##" % (virt[0], virt[-1]))      
            print("##                                                 ##")

    # localize core and virtual orbitals
    properties_core_input = """NEWK
%i 0
1 0
LOCALWF
INIFIBND
%i %i
CYCTOL
%i %i %i
MAXCYCLE
100
BOYSCTRL
8 8 100
WANDM
-2 4 1
END
CRYAPI_OUT
END""" % (newk, core[0]+1, core[1]+1, cyctol, cyctol, cyctol)

   

    properties_core_input_ = """NEWK
%i %i %i
1 0
LOCALWF
INIFIBND
%i %i
END
CRYAPI_OUT
END
END""" % (newk, newk, newk, core[0]+1, core[1]+1)
    
    
    f = open(folder + "/wann_input_core.d3"  , "w")
    f.write(properties_core_input)
    f.close()

    properties_virtual_input = """NEWK
%i 0
1 0
LOCALWF
INIFIBND
%i %i
CYCTOL
%i %i %i
MAXCYCLE
100
BOYSCTRL
8 8 100
WANDM
-2 4 1
END
CRYAPI_OUT
END""" % (newk, virt[0]+1, virt[-1] , cyctol, cyctol, cyctol)

    properties_virtual_input_ = """NEWK
%i %i %i
1 0
LOCALWF
INIFIBND
%i %i
END
CRYAPI_OUT
END
END""" % (newk, newk, newk, virt[0]+1, virt[-1])
    
    
    
    
    f = open(folder + "/wann_input_virtual.d3" , "w")
    f.write(properties_virtual_input)
    f.close()
    
    
    #if bands[0]>1 or len(bands) == 1:
    if run:
        ## Localize core
        properties_log = open(folder + "/properties_log_core.txt", "w")
        sp.call([P.crystal_exe_path + "/properties"], stdin = open(folder + "/wann_input_core.d3"), cwd=folder, stdout = properties_log)
        properties_log.close()


        # Run Cryapi for core
        cryapi_out = open(folder + "/gred_core.txt", "w")
        sp.call([cryapi_exe], cwd = folder, stdout = cryapi_out)
        cryapi_out.close()

        # Read core orbitals
        bands = C.load_gred_dat(folder + "/gred_core.txt")

    
    if run:
        # Localize virtual orbitals
        properties_log = open(folder + "/properties_log_virtual.d3", "w")
        sp.call([P.crystal_exe_path + "/properties"], stdin = open(folder + "/wann_input_virtual.d3"), cwd=folder, stdout = properties_log)
        properties_log.close()

        # Run Cryapi for virtuals
        cryapi_out = open(folder + "/gred_virtual.txt", "w")
        sp.call([cryapi_exe], cwd = folder, stdout = cryapi_out)
        cryapi_out.close()

        # Read virtual orbitals
        bands = C.load_gred_dat(folder + "/gred_virtual.txt")
        

        # Permute blocks according to order of basis functions
        C.permute_blocks(P.get_permutation_vector(), axis = 0)


        # Read matrices
        S, F, D = get_matrices(folder + "/gred_virtual.txt", ordering = P.get_permutation_vector())
        
        #remove zero blocks
        C = tp.screen_tmat(C, tolerance = 1e-16)
        #print(C.coords)

        return S, F, D, C
    else:
        return 0, 0, 0, 0

def get_dense(M):
    """
    parse from text outputted crystal matrix
    """
    ret = []
    for i in M:
        elm = [literal_eval(j) for j in i.split()]
        if len(elm) != 0:
            ret.append(elm)
    sparse = []
    for i in range(len(ret)):
        if ((len(ret[i]) == 1) or (type(ret[i][1]) == int)):
            #column 
            column = ret[i]
            pass
        elif type(ret[i][1]) == float:
            #row
            row = ret[i][0]
            for j in range(len(ret[i])-1):
                sparse.append([row, column[j], ret[i][j+1]])
    
    #build dense
    sparse = np.array(sparse)
    rowmax,colmax = sparse[:,0].max(), sparse[:,1].max()
    mdense = np.zeros((int(rowmax), int(colmax)), dtype = float)
    for i in range(len(sparse)):
        x,y,e = sparse[i]
        mdense[int(x)-1,int(y)-1] = e
    
    return mdense

def parse_matrix(M):
    """
    """
    blocks = M.split("G")
    coords = []
    dblocks = []
    for i in range(1, len(blocks)):
        m = blocks[i].split("\n")
        coords.append( [literal_eval(j) for j in m[0].split()] )
        dblocks.append(get_dense(m[1:]))
    
    return coords, dblocks
    

def get_matrices(fname, ordering):
    """
    #read matrices from crystal output (cryapi gred.dat)
    #return matrices in numpy format with associated coords
    #if ordering == None:
    #    ordering = get_permutation_vector(basis_sequence)
        
    #print(ordering)
    """

    f = open(fname, "r")
    
    #READ DENISTY MATRIX
    Ms = f.read().split("DENSITY MATRIX DIRECT LATTICE - G=0")[1].split("LOCALIZATION")[0]
    PG = Ms.split("HAMILTONIAN MATRIX DIRECT LATTICE - G=0")[0]
    FG = Ms.split("HAMILTONIAN MATRIX DIRECT LATTICE - G=0")[1].split("OVERLAP")[0]
    SG = Ms.split("OVERLAP MATRIX DIRECT LATTICE - G=0")[1]
    
    f.close()
    
    Scoords, Sblocks = parse_matrix(SG)
    Pcoords, Pblocks = parse_matrix(PG)
    Fcoords, Fblocks = parse_matrix(FG)
    Scoords, Sblocks = symmetrize_full(Scoords, Sblocks)
    Fcoords, Fblocks = symmetrize_full(Fcoords, Fblocks)
    
    #convert to numpy arrays
    Scoords = np.array(Scoords)
    Pcoords = np.array(Pcoords)
    Fcoords = np.array(Fcoords)
    
    Sblocks = np.array(Sblocks)
    Fblocks = np.array(Fblocks)
    Pblocks = np.array(Pblocks)
    
    #print(Sblocks.shape)
    #permute blocks 
    for i in np.arange(len(Sblocks)):
        Sblocks[i] = Sblocks[i][ordering,:][:,ordering]
    
    for i in np.arange(len(Fblocks)):
        Fblocks[i] = Fblocks[i][ordering,:][:,ordering]
        
    for i in np.arange(len(Pblocks)):
        Pblocks[i] = Pblocks[i][ordering,:][:,ordering]
    
    
    S = tp.tmat(blocks=Sblocks, coords=Scoords) # Overlap matrix
    F = tp.tmat(blocks=Fblocks, coords=Fcoords) # Fock matrix
    P = tp.tmat(blocks=Pblocks, coords=Pcoords) # Density matrix
    return S, F, P

def symmetrize_full(coords, blocks):
    """
    ###################################
    ##
    ## Restore dense form of triangular blocked lattice matrices
    ##
    ###################################
    """
    coords = np.array(coords)
    sblocks = []
    scoords = []
    
    # make sure the reference cell is included
    # no matter where it occurs
    for c in range(len(coords)):
        if (coords[c] == np.array([0,0,0])).all():
            sblocks.append(symmetrize(blocks[c], blocks[c]))
            scoords.append(coords[c])
        for d in range(c+1, len(coords)):
            if (coords[c] == -1*coords[d]).all():
                sblock = symmetrize(blocks[c], blocks[d])
                sblocks.append(sblock)
                scoords.append(coords[c])
                
                sblocks.append(sblock.T)
                scoords.append(coords[d])


    return scoords, sblocks

def symmetrize(lower_triangular,upper_triangular):
    """
    combine upper and lower triangular matrix
    """
    ret = lower_triangular + upper_triangular.T 
    np.fill_diagonal(ret, np.diag(lower_triangular))
    return ret

if __name__ == "__main__":
    print("""#####################################################
##  ,--.   ,--.      ,------.  ,------. ,-----.    ##
##   \  `.'  /,-----.|  .-.  \ |  .---''  .--./    ##
##    .'    \ '-----'|  |  \  :|  `--, |  |        ## 
##   /  .'.  \       |  '--'  /|  `---.'  '--'\    ##
##  '--'   '--'  Crint: Lighter Crystal Interface  ##
##              Author: Audun Skau Hansen          ##
##                                                 ##
##  Use keyword "--help" for more info             ## 
#####################################################""")

    #Locate executables for crystal and Cryapi
    os.environ["LIBINT_DATA_PATH"] = os.getcwd() 
    try:
        crystal_exe_path = os.environ['CRYSTAL_EXE_PATH']
    except:
        try:
            crystal_exe_path = os.environ['CRY14_EXEDIR'] + "/" + os.environ['VERSION']
        except:
            crystal_exe_path = "/Crystal/MacOsX-ifort-64/v1.0.3"
            print("\033[93m" + "WARNING"+ "\033[0m"+": Crystal executable folder not found.")
            print("\033[93m" + "       "+ "\033[0m"+": Environment variable CRY14_EXEDIR needed.")
            print("\033[93m" + "       "+ "\033[0m"+"  Environment variable VERSION needed.")
            print("\033[93m" + "       "+ "\033[0m"+"  (These are default variables for Crystal 14.)")
            print("         Instead using:", crystal_exe_path)   
    
    try:
        cryapi_exe = os.environ['CRYAPI_EXE_PATH']
    except:
        cryapi_exe = "../../utils/cryapi/cryapi.x"
        print("\033[93m" + "WARNING"+ "\033[0m"+": Environment variable CRYAPI_EXE_PATH not set.")
        print("         Instead using:", cryapi_exe)
        
    # Parse input
    parser = argparse.ArgumentParser(prog = "Crystal Interface",
                                     description = "Generate Wannier functions for XDEC using Crystal",
                                     epilog = "Author: Audun Skau Hansen (e-mail: a.s.hansen@kjemi.uio.no)")
    parser.add_argument("input_file", type = str, help ="input file for project (.d12 file)")
    parser.add_argument("-setup", action= "store_true", default=False, help="Only generate setup files (requires keyword valence).")
    parser.add_argument("-valence", type = str, default = None, help="Specify initial and final valence orbs (ex. -valence 3,6).")
    parser.add_argument("-folder", type = str, default = None, help="Relative project folder (default is working directory).")
    parser.add_argument("-newk", type = int, default = None, help="Optional resampling of k-mesh. Default is same as shrink[0].")
    parser.add_argument("-skip_crystal", action= "store_true", default=False, help="Skip running Crystal.")
    parser.add_argument("-skip_local", action= "store_true", default=False, help="Skip locality dependence.")
    parser.add_argument("-local_only", action= "store_true", default=False, help="Compute spreads and centers only.")
    parser.add_argument("-skip_orthogonality_test", action= "store_true", default=False, help="Skip orthogonality tests (not recommended).")
    parser.add_argument("-cyctol", type = int, default = 9, help="CYCTOL parameter for properties")
    
    
    
    
    args = parser.parse_args()
    
    if args.valence is not None:
        try:
            valence = [literal_eval(i) for i in args.valence.split(",")]
        except:
            print("\033[91m   Error \033[0m : Failed to parse valence string.")
            sys.exit("            Ex.Usage: >crint.py [input_file.d12] -valence 3,5?")      
    
    
    
    if args.setup:
        args.skip_crystal = True
        args.skip_local = True
        args.skip_orthogonality_test = True
        if args.valence is None:
            print("\033[93m" + "WARNING"+ "\033[0m"+": Valence no set. Assuming (3,5).")
            print("\033[93m" + "       "+ "\033[0m"+": Temporary input files generated.")
            print("\033[93m" + "       "+ "\033[0m"+": Crint will generate correct wann-inputs after running valence.")
            args.valence = "3,5"
            #sys.exit("            Ex.Usage: >crint.py [input_file.d12] -setup -valence 3,5?")      
    
    if args.local_only:
        args.skip_crystal = True
        args.skip_orthogonality_test = True
        args.setup = True
        
    if args.folder is None:
        folder = os.getcwd()
    else:
        folder = os.getcwd() + "/" + args.folder
        try:
            sp.call(["mkdir", folder])
        except:
            print("\033[93m" + "WARNING"+ "\033[0m"+": Folder: ", folder, " already exists.")
            print("\033[93m" + "       "+ "\033[0m"+": Content will be overwritten.")
            
    infile = args.input_file
    
    crintfolder = os.path.dirname(os.path.realpath(__file__))
    
    devnull = open(os.devnull, "w")
    
    sp.call(["mkdir", folder + "/Crystal/"], stdout = devnull,stderr=sp.STDOUT)
    sp.call(["mkdir", folder + "/XDEC/"], stdout = devnull,stderr=sp.STDOUT)
    
    devnull.close()
    
    P = pr.prism(infile)
    
    if args.newk is None:
        newk = P.shrink[0]
    else:
        newk = args.newk
    
    if not args.skip_crystal:
        print("##                                                 ##")
        print("##>- Running CRYSTAL                             -<##")
        print("##                                                 ##")
        run_crystal(infile, folder + "/Crystal/", P)
    
    
    
    print("##                                                 ##")
    print("##>- Wannierization                              -<##")
    print("##                                                 ##")
    S,F,D,C = wannierization(folder + "/Crystal/", P, newk = newk, cyctol = args.cyctol, run = not args.setup)
    
    if args.local_only:
        S,F,D,C = read_wann(folder + "/Crystal/", P, newk = newk)
        
    
        
        
    # Compute centers and spreads of psm1 functions
    if not args.skip_local:
        # Compute locality measures
        #of = oref.objective_function(P)
        wcenters, wspreads = of.centers_spreads(C, P, S.coords, m=1)
        #spreads = lambda tens : np.diag(tens[0].cget([0,0,0]) - tens[1].cget([0,0,0])**2 - tens[2].cget([0,0,0])**2 - tens[3].cget([0,0,0])**2)
        #spreads = spreads(tensors)


        #XYZ2mo, Xmo, Ymo, Zmo, wcenters = of.foster_boys(C,S) #spreads, x,y,z, centers
        #spreads = np.sqrt(np.diag(L0_psm))
        np.save(folder + "/spreads_psm1.npy", wspreads)
        np.save(folder + "/centers_psm1.npy", wcenters)
        
    if not args.setup:
        print("##                                                 ##")
        print("##>- Storing results                             -<##")
        print("##                                                 ##")
        F.save(folder + "/F_crystal.npy")
        D.save(folder + "/D_crystal.npy")
        C.save(folder + "/psm1.npy")
        S.save(folder + "/S_crystal.npy")
        print("##       psm1.npy         - Wannier Coefficients   ##")
        if not args.skip_local:
            
            print("##       spreads_psm1.npy - Wannier spreads        ##")
            print("##       tails_psm1.npy   - Wannier tails          ##")        
        print("##       S_crystal.npy    - Crystal Overlap Matrix ##")
        print("##       D_crystal.npy    - Crystal Density Matrix ##")
        print("##       F_crystal.npy    - Crystal Fock Matrix    ##")

        print("##       Crystal/*        - Crystal related        ##")
        print("##       XDEC/*           - XDEC inputs            ##")
        print("##                                                 ##")
        print("##       Usage example:                            ##")
        print("##           S = toeplit.tmat()                    ##")
        print("##           S.load('S_crystal.npy')               ##")
        print("##                                                 ##")
    
    

    f = open(folder + "/XDEC/basis.g94", "w")
    f.write(P.get_libint_basis())
    f.close()
    
    f = open(folder + "/XDEC/MOLECULE.INP", "w")
    f.write(P.get_lsdalton_input())
    f.close()
    
    sp.call(["cp", infile, folder + "/Crystal/" + infile])
    sp.call(["cp", crintfolder + "/scrint_crystal.sh", folder + "/Crystal/scrint_crystal.sh"])
    sp.call(["cp", crintfolder + "/scrint_crystal_abel.sh", folder + "/Crystal/scrint_crystal_abel.sh"])
    sp.call(["cp", crintfolder + "/scrint_crystal_stallo.sh", folder + "/Crystal/scrint_crystal_stallo.sh"])
    #sp.call(["cp", crintfolder + "/scrint_wann.sh", folder + "/scrint_wann.sh"])
    sp.call(["cp", crintfolder + "/scrint_wann_abel.sh", folder + "/scrint_wann_abel.sh"])
    sp.call(["cp", crintfolder + "/scrint_wann_stallo.sh", folder + "/scrint_wann_stallo.sh"])
    
    
    
    
    if not args.skip_orthogonality_test:
        print("##                                                 ##")
        print("##>- Testing orthogonality                       -<##")
        print("##                                                 ##")
        Smo = C.tT().cdot(S*C, coords = C.coords)
        
        print("##                                                 ##")
        print("##   Normality of orbitals:                        ##")
        print("##   #p   (p,p)                                    ##")
        norm = np.diag(Smo.cget([0,0,0]))
        for i in np.arange(len(norm)):
            print("##   %.2i   %.10e                         ##" % (i, norm[i]))
            
        rt2 = tp.L2norm(Smo.cget([0,0,0])-np.eye(Smo.blockshape[0]))
        n_elems = Smo.blockshape[0]**2
        print("##                                                 ##")
        print("##   Max deviation in [0,0,0]                      ##")
        print("##   %.10e                              ##" % np.abs(Smo.cget([0,0,0])-np.eye(Smo.blockshape[0])).max())
        
        maxdev = []
        
        for c in Smo.coords:
            if np.sum(np.abs(c))!=0:
                maxdev.append(np.abs(Smo.cget(c)).max())
            else:
                maxdev.append(np.abs(Smo.cget(c) - np.eye(Smo.blockshape[0])).max())
        
        maxdev = np.array(maxdev)
        m_max = np.argmax(maxdev)
        print("##                                                 ##")
        print("##   Max deviation in supercell                    ##")
        print("##   %.10e                              ##" % maxdev[m_max])
    print("##                                                 ##")
    print("######################################>All done<#####")
    
