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

from mpi4py import MPI

from XDEC import *




if __name__ == "__main__":

    comm = MPI.COMM_WORLD 
    size = comm.Get_size()
    rank = comm.Get_rank()

    

    os.environ["LIBINT_DATA_PATH"] = os.getcwd()
    print("""#########################################################
##    ,--.   ,--.      ,------.  ,------. ,-----.      ##
##     \  `.'  /,-----.|  .-.  \ |  .---''  .--./      ##
##      .'    \ '-----'|  |  \  :|  `--, |  |          ##
##     /  .'.  \       |  '--'  /|  `---.'  '--'\      ##
##    '--'   '--'      eXtended local correlation      ##
##                         > MPI <                     ##
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
    parser.add_argument("-rcond", type = float, default = 1e-12, help = "Default singular value screening threshold for inversion." )
    parser.add_argument("-inv", type = str, default = "lpinv", help = "Pseudo-inverse rotine, options: lpinv, spinv, spinv2, svd" )
    parser.add_argument("-store_ibuild",action = "store_true",  default = False,  help = "Store intermediate contraction object." )
    parser.add_argument("-rprecision", type = bool, default = False, help = "Reduce precision in final circulant product d.T V d in RI to complex64" )
    parser.add_argument("-set_omp_threads", type = int, default = 0)
    parser.add_argument("-single_cluster", type = int, default = None, help = "Compute only one single cluster/fragment")
    
    
    

    args = parser.parse_args()

    #print("Invtest:", args.inverse_test)

    args.float_precision = eval(args.float_precision)
    import sys

    if args.set_omp_threads > 0:
        os.environ['OMP_NUM_THREADS'] = "%i" %args.set_omp_threads 

    if args.fragmentation == "cim-vsweep":
        """
        cluster-in-molecule scheme
        """
            

        if rank==0:
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
            print("$OMP_NUM_THREADS seen by python:", os.environ.get("OMP_NUM_THREADS"))
            #print("MPI rank / size        :", mpi_rank, mpi_size)
            print("_________________________________________________________",flush=True)




            

            # Load system
            p = pr.prism(args.project_file)
            p.n_core = args.n_core
            p.set_nocc()

            # Compute overlap matrix
            s = of.overlap_matrix(p)





            # Fitting basis
            if args.basis_truncation < 0:
                auxbasis = PRI.remove_redundancies(p, args.N_c, args.auxbasis, analysis = True, tolerance = 10**args.basis_truncation)
                f = open("ri-fitbasis.g94", "w")
                f.write(auxbasis)
                f.close()
            else:
                auxbasis = PRI.basis_trimmer(p, args.auxbasis, alphacut = args.basis_truncation)
                #auxbasis = PRI.basis_scaler(p, args.auxbasis, alphascale = args.basis_truncation)
                #print(auxbasis)
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
                ib = PRI.integral_builder_static(c_occ,c_virt,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=args.circulant, robust = args.robust, xi0=args.xi0, xi1 = args.xi1, float_precision = args.float_precision, N_c = args.N_c,printing = args.print_level, inverse_test = args.inverse_test, rcond = args.rcond, inv = args.inv)
                #ib = PRI.integral_builder_static(c_occ,c_virt,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=None, circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust, ao_screening = args.ao_screening, xi0=args.xi0, JKa_extent= [6,6,6], xi1 = args.xi1, float_precision = args.float_precision, N_c = args.N_c,printing = args.print_level)
                if args.store_ibuild:
                    print("args.store_ibuild", args.store_ibuild)
                    np.save("integral_build.npy", np.array([ib]), allow_pickle = True)
            else:
                ib = np.load(args.ibuild, allow_pickle = True)[0]
                #print(ib.getorientation([0,0,0],[0,0,0]))
                print("Integral build read from file:", args.ibuild)
                args.attenuation = ib.attenuation
                print("Attenuation parameter set to %.4e" % args.attenuation)
                #ib.cfit.set_n_layers(PRI.n_points_p(p, args.N_c), args.rcond)

            if args.rprecision or args.float_precision == np.float32:
                # Use reduced precision in d.T V d product
                ib.tprecision = np.complex64


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
            # broadcast all data from rank 0
        else:
            p = None
            ib = None
            s_virt = None
            f_mo_ii, f_mo_aa, wcenters = None, None, None
            center_fragments = None

        
        
        p = comm.bcast(p, root = 0)
        ib = comm.bcast(ib, root = 0)
        s_virt = comm.bcast(s_virt, root = 0)
        f_mo_ii = comm.bcast(f_mo_ii, root = 0)
        f_mo_aa = comm.bcast(f_mo_aa, root = 0)
        wcenters = comm.bcast(wcenters, root = 0)
        center_fragments = comm.bcast(center_fragments, root = 0)


        #comm.Barrier()
        print(rank, p)
        
            
        

        

        print("MPI info:", rank, size)



        
        dV = 9

        N = 100

        v_range = np.linspace(args.virtual_cutoff, args.virtual_cutoff + dV, N)
        

        #energies = np.zeros((len(center_fragments),N,N), dtype = np.float)


        energies = []

        final_energies = []

        domain_max = tp.lattice_coords(PRI.n_points_p(p, 20))


        for f in range(len(center_fragments)):
            if rank==f%size:
                print(rank, " working on fragment ", f)
                #print(rank, f, size, f%size)
                sp.call(["mkdir", os.getcwd()+"/lpath_%i" %rank])
                os.environ["LIBINT_DATA_PATH"] = os.getcwd() + "/lpath_%i" %rank

                




                fragment = center_fragments[f]
                a_frag = fragment_amplitudes(p, wcenters, domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = v_range[0], occupied_cutoff = args.occupied_cutoff, float_precision = args.float_precision)
                nv = a_frag.n_virtual_tot
                no = a_frag.n_occupied_tot
                #eng = a_frag.compute_cim_energy(exchange = False)
                eng = []
                for i in np.arange(len(v_range)):
                    a_frag.set_extent(v_range[i], args.occupied_cutoff)




                    if nv == a_frag.n_virtual_tot and no == a_frag.n_occupied_tot:
                        pass
                    else:

                        dt, it, E_new, E_pairwise = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.001, damping = args.damping, energy = "cim")
                    
                        #dt, it = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = 1e-10, damping = args.damping)


                        #print("Convegence (dt, it):", dt, it)
                        #print("shape:", a_frag.g_d.shape)
                        # Converge to fot
                        #eng =  a_frag.compute_cim_energy(exchange = False)
                        eng.append([nv,v_range[i], E_pairwise])

                        nv = a_frag.n_virtual_tot*1
                        no = a_frag.n_occupied_tot*1

                        
                        
                        



                        print("_________________________________________________________")
                        print("Fragment:", fragment)
                        print("Sweep   :", i)
                        print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                        print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                        print("E(CIM): %.8f      DE(fragment): %.8e" % (E_new, 0.0))
                        print("dt , it:", dt, it)
                        print("Integrator memory usage (estimate):", ib.nbytes()*1e-6, " Mb.")
                        print("Fragment memory usage   (estimate):", a_frag.nbytes()*1e-6, " Mb.")
                        print("_________________________________________________________")
                    #energies.append(eng)
                    #final_energies.append(E_new)

                    


                    
                
                np.save("sweep_energies_%i.npy" % f, np.array(eng))
        #print("Done")

        #np.save("cim_occ_cuts.npy", o_range)
        #np.save("cim_vrt_cuts.npy", v_range)

        """

        for i in np.arange(len(final_energies)):
            print("     E_{Fragment %i} :" %i, final_energies[i], " Hartree")
        print("_________________________________________________________")
        print("Total cluster energy :", np.sum(np.array(final_energies)), " Hartree")
        #print("Total cluster energy : %.4e" % fragment_energy_total, " Hartree")
        """
        print("=========================================================")
        print("Pairwise energycontributions stored to disk.")

    if args.fragmentation == "cim-asweep":
        """
        cluster-in-molecule scheme
        """
        Nw = 10
        w = np.exp(np.linspace(np.log(0.1), np.log(10.0), Nw))

        cwd_root = os.getcwd()

        for fs in range(Nw):
            os.chdir(cwd_root)
            
            if rank==fs%size:
                sp.call(["mkdir", cwd_root+"/mpi_att_%i" %fs])
                sp.call(["mkdir", cwd_root+"/mpi_att_%i/Crystal" %fs])
                sp.call(["cp", cwd_root+"/psm1.npy", "mpi_att_%i/psm1.npy" %fs])
                sp.call(["cp", cwd_root+"/F_crystal.npy", "mpi_att_%i/F_crystal.npy" %fs])

                
                sp.call(["cp", args.project_file, ".", cwd_root+"/mpi_att_%i/Crystal" %fs])

                os.environ["LIBINT_DATA_PATH"] = cwd_root + "/mpi_att_%i" %fs
                
                

                #os.exit()

                #sp.call(["cd", "mpi_att_%i" %f])
                os.chdir(cwd_root + "/mpi_att_%i" %fs)
                print(rank, " working on run ", fs, w[fs])

                #break
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
                print("$OMP_NUM_THREADS seen by python:", os.environ.get("OMP_NUM_THREADS"))
                #print("MPI rank / size        :", mpi_rank, mpi_size)
                print("_________________________________________________________",flush=True)




                

                # Load system
                p = pr.prism(args.project_file)
                p.n_core = args.n_core
                p.set_nocc()

                # Compute overlap matrix
                s = of.overlap_matrix(p)





                # Fitting basis
                if args.basis_truncation < 0:
                    auxbasis = PRI.remove_redundancies(p, args.N_c, args.auxbasis, analysis = True, tolerance = 10**args.basis_truncation)
                    f = open("ri-fitbasis.g94", "w")
                    f.write(auxbasis)
                    f.close()
                else:
                    auxbasis = PRI.basis_trimmer(p, args.auxbasis, alphacut = args.basis_truncation)
                    #auxbasis = PRI.basis_scaler(p, args.auxbasis, alphascale = args.basis_truncation)
                    #print(auxbasis)
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
                    ib = PRI.integral_builder_static(c_occ,c_virt,p,attenuation = w[fs], auxname="ri-fitbasis", initial_virtual_dom=[0,0,0], circulant=args.circulant, robust = args.robust, xi0=args.xi0, xi1 = args.xi1, float_precision = args.float_precision, N_c = args.N_c,printing = args.print_level, inverse_test = args.inverse_test, rcond = args.rcond, inv = args.inv)
                    #ib = PRI.integral_builder_static(c_occ,c_virt,p,attenuation = args.attenuation, auxname="ri-fitbasis", initial_virtual_dom=None, circulant=args.circulant, extent_thresh=args.attenuated_truncation, robust = args.robust, ao_screening = args.ao_screening, xi0=args.xi0, JKa_extent= [6,6,6], xi1 = args.xi1, float_precision = args.float_precision, N_c = args.N_c,printing = args.print_level)
                    if args.store_ibuild:
                        print("args.store_ibuild", args.store_ibuild)
                        np.save("integral_build.npy", np.array([ib]), allow_pickle = True)
                else:
                    ib = np.load(args.ibuild, allow_pickle = True)[0]
                    #print(ib.getorientation([0,0,0],[0,0,0]))
                    print("Integral build read from file:", args.ibuild)
                    args.attenuation = ib.attenuation
                    print("Attenuation parameter set to %.4e" % args.attenuation)
                    #ib.cfit.set_n_layers(PRI.n_points_p(p, args.N_c), args.rcond)

                if args.rprecision or args.float_precision == np.float32:
                    # Use reduced precision in d.T V d product
                    ib.tprecision = np.complex64


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
            

                

                print("MPI info:", rank, size)



                
                #dV = 9

                #N = 100

                #v_range = np.linspace(args.virtual_cutoff, args.virtual_cutoff + dV, N)
                

                #energies = np.zeros((len(center_fragments),N,N), dtype = np.float)


                #energies = []

                #final_energies = []

                domain_max = tp.lattice_coords(PRI.n_points_p(p, 20))


                for n in range(len(center_fragments)):
                
                    #print(rank, f, size, f%size)

                    




                    fragment = center_fragments[n]
                    a_frag = fragment_amplitudes(p, wcenters, domain_max, fragment, ib, f_mo_ii, f_mo_aa, virtual_cutoff = args.virtual_cutoff, occupied_cutoff = args.occupied_cutoff, float_precision = args.float_precision)
                    nv = a_frag.n_virtual_tot
                    no = a_frag.n_occupied_tot
                    dt, it, E_new, E_pairwise = a_frag.solve(eqtype = args.solver, s_virt = s_virt, norm_thresh = args.fot*0.001, damping = args.damping, energy = "cim")
                    


                        

                    print("_________________________________________________________")
                    print("Fragment:", fragment)
                    
                    print("Attenuation   :", w[fs])
                    print("Virtual cutoff  : %.2f bohr (includes %i orbitals)" %  (a_frag.virtual_cutoff, a_frag.n_virtual_tot))
                    print("Occupied cutoff : %.2f bohr (includes %i orbitals)" %  (a_frag.occupied_cutoff, a_frag.n_occupied_tot))
                    print("E(CIM): %.8f      DE(fragment): %.8e" % (E_new, 0.0))
                    print("dt , it:", dt, it)
                    print("Integrator memory usage (estimate):", ib.nbytes()*1e-6, " Mb.")
                    print("Fragment memory usage   (estimate):", a_frag.nbytes()*1e-6, " Mb.")
                    print("_________________________________________________________")
                        
                    
                    np.save("asweep_energies_%i_%f.npy" % (n, w[fs]), np.array(E_pairwise))
                
                print("=========================================================")
                print("Done")
        #print("Pairwise energycontributions stored to disk.")

        
