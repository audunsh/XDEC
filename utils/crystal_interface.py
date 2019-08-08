from ast import literal_eval
import numpy as np
import os


"""
Interface to Crystal

A collection of functions that enables a conversion from LSDALTON-type input files (LSDALTON.INP, MOLECULE.INP) 
with a userdefined basis to be converted into a CRYSTAL.D12 input file. 

Author: Audun Skau Hansen
Date:   November 2016

"""


def readmol(molfile):
    #NB USERDEFINED BASIS
    ##
    
    f = open(molfile, "r")
    F = f.read()
    f.close()
    

    
    header =  F.split("Atomtypes=")[0]
    body = F.split("USERDEFINED BASIS")[0].split("a1 =")[0].split("Charge=")[1:]
    bottom = "USERDEFINED BASIS" + F.split("USERDEFINED BASIS")[1]
    
    lvec = np.array([literal_eval(i) for i in F.split("a1 =")[1].split("active")[0].split()])
    
    lvec2 = F.split("active")[0:-1]
    lattice = []
    for i in lvec2:
        if i.split()[-1] == "in":
            pass
        else:
            lattice.append(np.array([literal_eval(j) for j in i.split()[-3:]]))

    atoms = []
    atomic_numbers = []
    for i in body:

        atomic_numbers.append(literal_eval(i.split()[0]))
        
        atoms.append(np.array([[literal_eval(k) for k in j.split()[1:]] for j in i.split("\n")[1:-1]]))

    atomtypes = literal_eval(F.split("Atomtypes=")[1].split()[0])
    
    return atoms, atomic_numbers, header, bottom, lattice

def gen_cluster(molfile, ncells):
    #######
    ##
    ##  Generate cluster-expansion input files for LSDalton
    ##  molfile - the file to interpred (should be a periodic input file with lattice-vecs)
    ##  ncells - integer, number of cells along each periodic axis
    ##
    ########
    
    atoms, atomic_numbers, header, bottom, lvec, = readmol(molfile)
    
    
    
    
    new_mfile = header
    new_mfile += "Atomtypes=%i\n" % len(atomic_numbers)

    
    for i in range(len(atoms)):
        
        if len(lvec) == 1:
            new_mfile += "Charge=%i. Atoms=%i\n" %(atomic_numbers[i], len(atoms[i])*(ncells**len(lvec)))
            for xyz in atoms[i]:
                for nx in range(ncells):
                    new_mfile += "%i %.8e %.8e %.8e\n" % (atomic_numbers[i], xyz[0]+nx*lvec[0][0], 
                                                                             xyz[1]+nx*lvec[0][1], 
                                                                             xyz[2]+nx*lvec[0][2])
        if len(lvec) == 2:
            new_mfile += "Charge=%i. Atoms=%i\n" %(atomic_numbers[i], len(atoms[i])*(ncells**len(lvec)))
            for xyz in atoms[i]:
                for nx in range(ncells):
                    for ny in range(ncells):
                        new_mfile += "%i %.8e %.8e %.8e\n" % (atomic_numbers[i], 
                                                              xyz[0]+nx*lvec[0][0]+ny*lvec[1][0], 
                                                              xyz[1]+nx*lvec[0][1]+ny*lvec[1][1], 
                                                              xyz[2]+nx*lvec[0][2]+ny*lvec[1][2])
        if len(lvec) == 3:                          
            new_mfile += "Charge=%i. Atoms=%i\n" %(atomic_numbers[i], len(atoms[i]) * (ncells**len(lvec)))
            for xyz in atoms[i]:
                for nx in range(ncells):
                    for ny in range(ncells):
                        for nz in range(ncells):
                            new_mfile += "%i %.8e %.8e %.8e\n" % (atomic_numbers[i], 
                                                        xyz[0]+nx*lvec[0][0]+ny*lvec[1][0]+nz*lvec[2][0], 
                                                        xyz[1]+nx*lvec[0][1]+ny*lvec[1][1]+nz*lvec[2][1], 
                                                        xyz[2]+nx*lvec[0][2]+ny*lvec[1][2]+nz*lvec[2][2])
            
    
    new_mfile += bottom
    
    lsinp = """**GENERAL
.NOGCBASIS
**INTEGRALS
.THRESH
1.0D-18
.NOLINK
.NO PS
.NO CS
**WAVE FUNCTIONS
.HF
*DENSOPT
.START
H1DIAG
.DIIS
**INFO
.DEBUG_MPI_MEM
**DEC
.MP2
.FOT
1.0e-6
.MEMORY
2.0
*END OF INPUT
"""
    
    return new_mfile, lsinp




def check_geometry(molecule_inp):
    # read molecule.inp and verify that 
    # all atomic positions is within [-.5*lvec, .5*lvec) in 
    # periodic directions. 
    # if not : translate atoms in returned string
    
    f = open(molecule_inp, "r")
    F = f.read()
    f.close()
    
    F = F.split("Charge =")
    header, geom = F[0], F[1:]
    print(header)
    print(geom)

def stalloscript(project_name, nodes = 1, tasks_per_node = 8):
    #crystal stallo input
    return """#!/bin/bash

#SBATCH --job-name=%s
#SBATCH --nodes=%s
#SBATCH --ntasks-per-node=%s
#              d-hh:mm:ss
#SBATCH --time=0-15:00:00

# memory per core not cpu! With more than 1.6GB the cputime subtracted from the account will be doubled.
#SBATCH --mem-per-cpu=1600MB

# turn on all mail notification
#SBATCH --mail-type=ALL


# NB: no bash commands before the last SBATCH directive

if [ $# -ne 1 ]
  then
    echo "Error: the script need 1 argument to run."
    echo "(the relative path of the .d12 inputfile)."
    exit 1
fi

# create a unique scratch directory
SCRATCH_DIRECTORY=/global/work/${USER}/PCrystalScratchDir/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# prepare run
INPUTFILE="$1"
cp ${SLURM_SUBMIT_DIR}/${INPUTFILE} ${SCRATCH_DIRECTORY}/INPUT
cp ${SLURM_SUBMIT_DIR}/${INPUTFILE} ${SCRATCH_DIRECTORY}
# cp ${SLURM_SUBMIT_DIR}/fort.9 ${SCRATCH_DIRECTORY}
# cp ${SLURM_SUBMIT_DIR}/fort.80 ${SCRATCH_DIRECTORY}
cp /global/hds/software/cpu/non-eb/CRYSTAL14/v1.0.4/bin/Linux-ifort_XE_openmpi_emt64/v1.0.4/Pcrystal ${SCRATCH_DIRECTORY}/
# module load crystal

# execute job
mpirun -np ${SLURM_NPROCS} Pcrystal < ${INPUTFILE}
# copy output back to a new folder in the ${SLURM_SUBMIT_DIR}
HF=${SLURM_SUBMIT_DIR}/$1"_"${SLURM_JOBID}
mkdir -p ${HF}
cp ${SCRATCH_DIRECTORY}/fort.9 ${HF}/
cp ${SCRATCH_DIRECTORY}/INPUT ${HF}/

# finalize
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}
exit 0""" % (project_name, nodes, tasks_per_node)

def stalloscript_properties(project_name, nodes = 1, tasks_per_node = 8):
    #crystal stallo input
    return """#!/bin/bash

#SBATCH --job-name=%s
#SBATCH --nodes=%s
#SBATCH --ntasks-per-node=%s
#              d-hh:mm:ss
#SBATCH --time=0-15:00:00

# memory per core not cpu! With more than 1.6GB the cputime subtracted from the account will be doubled.
#SBATCH --mem-per-cpu=1600MB

# turn on all mail notification
#SBATCH --mail-type=ALL


# NB: no bash commands before the last SBATCH directive

if [ $# -ne 1 ]
  then
    echo "Error: the script need 1 argument to run."
    echo "(the relative path of the .d12 inputfile)."
    exit 1
fi

# create a unique scratch directory
SCRATCH_DIRECTORY=/global/work/${USER}/PCrystalScratchDir/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# prepare run
INPUTFILE="$1"
cp ${SLURM_SUBMIT_DIR}/${INPUTFILE} ${SCRATCH_DIRECTORY}/INPUT
cp ${SLURM_SUBMIT_DIR}/${INPUTFILE} ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/fort.9 ${SCRATCH_DIRECTORY}
# cp ${SLURM_SUBMIT_DIR}/fort.80 ${SCRATCH_DIRECTORY}
cp /global/hds/software/cpu/non-eb/CRYSTAL14/v1.0.4/bin/Linux-ifort_XE_openmpi_emt64/v1.0.4/Pproperties ${SCRATCH_DIRECTORY}/
# module load crystal

# execute job
mpirun -np 1 Pproperties < ${INPUTFILE}
# copy output back to a new folder in the ${SLURM_SUBMIT_DIR}
HF=${SLURM_SUBMIT_DIR}/$1"_"${SLURM_JOBID}
mkdir -p ${HF}
cp ${SCRATCH_DIRECTORY}/fort.80 ${HF}/
cp ${SCRATCH_DIRECTORY}/INPUT ${HF}/
cp ${SCRATCH_DIRECTORY}/GRED.DAT ${HF}/

# finalize
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}
exit 0""" % (project_name, nodes, tasks_per_node)

def lsdalton_input_setup(n_layers, additional_keywords = ""):
    return """**GENERAL
.NOGCBASIS
**INTEGRALS
.THRESH
1.0D-18
.NOLINK
.NO PS
.NO CS
**WAVE FUNCTIONS
.HF
*DENSOPT
.START
H1DIAG
**WANNIER
.DEBUG_MODE
-1001
.AO_CUTOFF
%i
.CRYSTAL_INIT
.ONLY_LOCALIZE%s
**LOCALIZATION
.PSM
2 2
*END OF INPUT
""" % (n_layers, additional_keywords)



def crystal_wannierization_setup(nocc, norbs, newk = 17, fullboys = 10, cyctol = 9, frozen = True):
    occ = """NEWK
%i 0
1 0
LOCALWF
VALENCE
CYCTOL
%i %i %i
FULLBOYS
%i
END
CRYAPI_OUT
END
""" % (newk, cyctol, cyctol, cyctol, fullboys)
    if frozen == False:
        occ = """NEWK
%i 0
1 0
LOCALWF
INIFIBND
1 %i
CYCTOL
%i %i %i
FULLBOYS
%i
END
CRYAPI_OUT
END
""" % (newk, nocc, cyctol, cyctol, cyctol, fullboys)    
    virt = """NEWK
%i 0
1 0
LOCALWF
INIFIBND
%i %i
CYCTOL
%i %i %i
FULLBOYS
%i
END
CRYAPI_OUT
END
""" %(newk, nocc +1, norbs, cyctol,cyctol,cyctol, fullboys)
    return occ, virt

#Keyword to be included
#WANDM
#-2 4 1

def d2l_basis(basis_string):
    ###########################################
    ##
    ##  Convert basis from Dalton to Crystal
    ##  Author: Audun
    ##
    ###########################################
    
    atoms = basis_string.split("a ")[1:]
    basis = []
    atomic_numbers = []
    shell_types = []
    contracted_numbers = []
    
    # Parse basis from Dalton
    
    #####
    ##
    ## New parsing routine
    ##
    ######

    basis_set = []
    atomic_numbers = []
    for i in atoms:
        basis_set.append([])
        ao_function = i.split("\n")
        
        atomic_numbers.append(int(ao_function[0]))
        
        b_set= [[literal_eval(k) for k in line.split()] for line in ao_function[1:]]

        c = 0 #counter
        shelltype = -1
        while c<len(b_set):
            params = b_set[c]
            if len(params)!=0:
                if type(params[0]) == int:
                    #shell line
                    shelltype += 1
                    basis_set[-1].append([])
                    
                    n_primitives = params[0]
                    n_contracted = params[1]

                    primitives = np.array(b_set[c+1:c+1+n_primitives])
                    
                    for l in range(1,n_contracted+1):
                        basis_set[-1][-1].append([])
                        for k in range(n_primitives):
                            if primitives[k,l] != 0.0:
                                basis_set[-1][-1][-1].append([primitives[k, 0], primitives[k,l]])
                    #c += n_primitives-1 #skip following lines
            c += 1
    return basis_set, atomic_numbers

    
def d2l_geometry(geom_string):
    ###########################################
    ##
    ##  Get geometry from LSDalton input
    ##  Author: Audun
    ##
    ###########################################

    b2a = 0.52917720830000000000000000000 #b2a conversion (value from crystal)
    geom = geom_string.split("\n")[4:]
    n_atomtypes = literal_eval(geom[0].split()[0].split("=")[1])

    atom_types = geom_string.split("Charge=")[1:]
    
    charges = []
    atoms = []
    n_atoms_tot = 0
    
    for i in atom_types:
        atom_geom = i.split("\n")
        
        charge = literal_eval(atom_geom[0].split()[0])
        
        n_atoms = literal_eval(atom_geom[0].split()[1].split("Atoms=")[1])
        
        #charges.append(charge)
        
        for j in range(n_atoms):
            atom_xyz = atom_geom[j+1].split()[1:]

            atom_pos = [literal_eval(n)*b2a for n in atom_xyz]
            
            atoms.append(atom_pos)
            
            charges.append(int(charge))
            
            n_atoms_tot += 1
    
    ## TODO:
    ## -4:-1 last lines in geom are the lattice vectors, this should be generalized
    ## to 3 D 
    lattice_vectors = atom_types[-1].split("\n")[-4:-1]
    #lattice_param = literal_eval(lattice_vectors[0].split("=")[1].split()[0])
    return atoms, charges, lattice_vectors

def dalton2lambdas(molecule):
    ###########################################
    ##
    ##  Read dalton input files
    ##  Convert to a list of lambda contracted
    ##  Author: Audun 
    ##  Date:   January 2017
    ##  Usage:
    ##
    ###########################################
    
    mol = open(molecule, "r")
    MOL = mol.read()
    mol.close()
    basistype = MOL.split("\n")[1]
    if basistype == "USERDEFINED":
    
        geom, basis = MOL.split("USERDEFINED BASIS")
        
        
        
    else:
        try:
            #get all atom types
            atom_types = [literal_eval(i.split(".")[0]) for i in MOL.split("Charge=")[1:]]

            bnew = get_dalton_basis(basistype, atom_types)
            
            #remove possible comment lines 
            basis = "USERDEFINED BASIS\n"
            for i in bnew.split("\n"):
                if len(i) != 0:
                    if i[0] != "$":
                        basis += i + "\n"
            
            #remove final line
            basis = basis[:-1]
            
            
            geom = MOL
            geom = geom.rstrip().replace(basistype, "USERDEFINED") + "\n"
        except:
            print("\033[91m ERROR \033[0m: Failed to parse basis from lsdalton library.")
            
            

    basis = basis.rstrip()
    
    basis_numeric = d2l_basis(basis)
    geometry_numeric = d2l_geometry(geom)
    return basis_numeric, geometry_numeric
    
def cryscor_stallo_sbatch():
    return """#!/bin/bash

#SBATCH --job-name=Ne2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#              d-hh:mm:ss
#SBATCH --time=0-15:00:00

# memory per core not cpu! With more than 1.6GB the cputime subtracted from the account will be doubled.
#SBATCH --mem-per-cpu=1600MB

# turn on all mail notification
#SBATCH --mail-type=ALL


# NB: no bash commands before the last SBATCH directive

if [ $# -ne 1 ]
  then
    echo "Error: the script need 1 argument to run."
    echo "(the relative path of the .d4 inputfile)."
    exit 1
fi

# create a unique scratch directory
SCRATCH_DIRECTORY=/global/work/${USER}/PCrystalScratchDir/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# prepare run
INPUTFILE="$1"
cp ${SLURM_SUBMIT_DIR}/${INPUTFILE} ${SCRATCH_DIRECTORY}/INPUT
cp ${SLURM_SUBMIT_DIR}/${INPUTFILE} ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/fort.80 ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/fort.9 ${SCRATCH_DIRECTORY}
cp /global/hds/software/cpu/non-eb/CRYSTAL14/v1.0.4/bin/Linux-ifort_XE_openmpi_emt64/v1.0.4/cryscor ${SCRATCH_DIRECTORY}/
# module load crystal

# execute job
mpirun -np ${SLURM_NPROCS} cryscor < ${INPUTFILE}
# copy output back to a new folder in the ${SLURM_SUBMIT_DIR}
HF=${SLURM_SUBMIT_DIR}/$1"_"${SLURM_JOBID}
mkdir -p ${HF}
cp ${SCRATCH_DIRECTORY}/* ${HF}/
cp ${SCRATCH_DIRECTORY}/* ${HF}/

# finalize
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}
exit 0"""

def dalton2crystal(molecule, wavefunction = "hf", exchange = "PBE", 
                   correlat = "PBE", itol1 = 16, itol2 = 16, itol3 = 16, 
                   itol4 = 25, itol5 = 50, shrink1 = 8, shrink2 = 8, center_geometry = False, toldee = 12):
    ###########################################
    ##
    ##  Read dalton input files
    ##  Convert to Crystal input files
    ##  Author: Audun 
    ##  Date:   November 2016
    ##  Usage:
    ##
    ###########################################
    
    mol = open(molecule, "r")
    MOL = mol.read()
    mol.close()
    basistype = MOL.split("\n")[1]
    if basistype == "USERDEFINED":
    
        geom, basis = MOL.split("USERDEFINED BASIS")
        basis = "USERDEFINED BASIS"+basis
        
        
    else:
        if True:
            #get all atom types
            atom_types = [literal_eval(i.split(".")[0]) for i in MOL.split("Charge=")[1:]]


            bnew = get_dalton_basis(basistype,np.sort(np.array(atom_types)))

            #remove possible comment lines 
            basis = "USERDEFINED BASIS\n"
            for i in bnew.split("\n"):
                if len(i) != 0:
                    if i[0] != "$":
                        basis += i + "\n"
            
            #remove final line
            basis = basis[:-1]
            
            
            geom = MOL
            geom = geom.rstrip().replace(basistype, "USERDEFINED") + "\n"
        if False:
            print("\033[91m ERROR \033[0m: Failed to parse basis from lsdalton library.")
            
    
    basis = basis.rstrip()
    
    
    
    
    crystal_basis, basis_info, libint_basis = d2c_basis(basis)
    
    crystal_geom, n_electrons, atom_ordering, geometry_string, is_angstrom = d2c_geom(geom, center_geometry)
    
    crystal_d12 = crystal_geom + crystal_basis
    
    

    
    #add setup params
    #crystal_d12 += """SHRINK\n%i %i 1\nNOSYMADA\nTOLINTEG\n%i %i %i %i %i\nEND""" % (shrink1, shrink2, 
    #                                                      itol1, itol2, itol3, 
    #                                                      itol4, itol5)
    
    # Setup for CRYSCOR - MP2 
    cryscor_oneshot = crystal_d12 + "SHRINK\n%i %i\nTOLINTEG\n%i %i %i %i %i\nMP2\nKNET\n%i %i 1\nMEMORY\n5000\nNEWGMAX\n30\nDOMPUL\n.999999\nPAIR\n8 15\nENDMP2\nEND\n" %(shrink1, shrink2,itol1, itol2, itol3, 
                                                            itol4, itol5, shrink1, shrink2)
    
    #cryscor = crystal_d12 + "SHRINK\n%i %i 1\nTOLMP2\nMP2\nKNET\n%i %i 1\nMEMORY\n50000\nNEWGMAX\n30\nDOMPUL\n.999999\nPAIR\n8 15\nENDMP2\nEND\n" %(shrink1, shrink2, shrink1, shrink2)
    
    
    #in case you run crystal-properties-cryscor
    cryscor = "KNET\n%i\nMEMORY\n5000\nDOMPUL\n0.99999999\nDFITTING\nDIRECT\nPG-VTZ\nENDDF\nEND" % shrink1
    

    
    #cryscor = "KNET\n%i\nMEMORY\n50000\nDOMPUL\n0.99999999\nEND" %shrink1
    #use dfitting here
    
    
    if wavefunction == "hf":
        crystal_d12 += """SHRINK\n%i %i 1\nNOBIPOLA\nTOLINTEG\n%i %i %i %i %i\nTOLDEE\n%i\nEND""" % (shrink1, shrink2, 
                                                            itol1, itol2, itol3, 
                                                            itol4, itol5, toldee)
    if wavefunction == "dft":
        crystal_d12 += "DFT\nEXCHANGE\n%s\nCORRELAT\n%s\nEND\n" % (exchange, correlat)
        #crystal_geom = "GENERATED %s\n%s\n1\n%s\n" % (dimensionality, dimensionality, c_lat.bravais_conversion())
        crystal_d12 += """SHRINK\n%i %i 1\nNOBIPOLA\nTOLINTEG\n%i %i %i %i %i\nEND""" % (shrink1, shrink2, 
                                                            itol1, itol2, itol3, 
                                                            itol4, itol5)
                                                            
    
        
    basis_sequence = ""
    
    for i in range(len(atom_ordering)):
        for j in range(len(basis_info)):
            if atom_ordering[i]==basis_info[j][0]:
                basis_sequence += basis_info[j][1]
      
    orb_types = "spdfg"
    
    #Sort AO-atom associations          
    atom_regions = []
    n_ao_on_atom = 0
    for i in range(len(atom_ordering)):
        for j in range(len(basis_info)):
            #count basis functions on atom
            
            if atom_ordering[i]==basis_info[j][0]:
                bsq = basis_info[j][1] #basis sequence for atom
                for n in bsq:
                    n_ao_on_atom += 2*orb_types.index(n)  + 1
                
                atom_regions.append(n_ao_on_atom)

    #count number of orbitals
    n_orbs = 0
    
    for i in basis_sequence:
        n_orbs += 2*orb_types.index(i)  + 1
    
    
    
    new_molecule_inp = geometry_string + basis + "\n"

                                                                                                                                                                                                                            
    return crystal_d12, basis_sequence, atom_regions, n_electrons, libint_basis, n_orbs, cryscor, cryscor_oneshot, new_molecule_inp, is_angstrom
    


class lattice_converter():
    ###############################
    # 
    # A class for handling conversion of Bravais lattice
    # and coordinates between Crystal and LSD.
    #
    ###############################
    
    
    def __init__(self, lattice_vectors =  [np.array([1.0,0.0,0.0]),
                                           np.array([0.0,1.0,0.0]),
                                           np.array([0.0,0.0,1.0])], 
                       periodic_directions = np.array([True,True,True]),
                       conversion_factor = 1.0):
        self.c = conversion_factor
        self.periodicity = periodic_directions
        # Setting up coordinate transform matrix
        
        self.M = np.diag((1.0,1.0,1.0)) 
        
        # Include any periodic axis
        
        for i in range(3):
            if periodic_directions[i] == True:
                self.M[i] = lattice_vectors[i]

        
    def get_periodic_coords(self, r):
        #transform r(bohr) into periodic coords (angstrom)
        
        return np.linalg.solve(self.M.T, r)
    
    def get_lsdalton_coords(self, r):
        #transform into lsdalton coords
        return np.dot(self.M,r/self.c)
        
    def as_triclinic(self):
        # convert three lattice vectors in bohr
        # to triclinic lattice parameters in angstrom for crystal
        # (returns a string)
        # IGR space group 1 or 2
                
        lattice_vectors = self.M
        
        x = lattice_vectors[0]
        y = lattice_vectors[1]
        z = lattice_vectors[2]
        
        a = np.sqrt(np.sum(x**2))
        b = np.sqrt(np.sum(y**2))
        c = np.sqrt(np.sum(z**2))
        alpha = np.arccos(np.sum(y*z)/(b*c))*360./(2*np.pi) #/np.pi
        beta =  np.arccos(np.sum(z*x)/(c*a))*360./(2*np.pi) #/np.pi
        gamma = np.arccos(np.sum(x*y)/(a*b))*360./(2*np.pi) #/np.pi
        
        return "0 0 0\n1\n%.15f %.15f %.15f %.15f %.15f %.15f" % (a,b,c,alpha,beta,gamma)
    def as_oblique(self):
        # Slab
        # IGR space group 1 
        lattice_vectors = self.M
        
        x = lattice_vectors[0]
        y = lattice_vectors[1]
        a = np.sqrt(np.sum(x**2))
        b = np.sqrt(np.sum(y**2))
        gamma = np.arccos(np.sum(x*y)/(a*b))*360./(2*np.pi) #/np.pi
        return "1\n%.15f %.15f %.15f" % (a,b,gamma)
    def as_polymer(self):
        # Polymer
        # IGR space group 1
        return "1\n%.15f" % self.M[0][0] #
        

    
    def bravais_conversion(self):
        if np.sum(self.periodicity) == 3:
            return self.as_triclinic()
        if np.sum(self.periodicity) == 2:
            return self.as_oblique()
        if np.sum(self.periodicity) == 1:
            return self.as_polymer()
        if np.sum(self.periodicity) == 0:
            #molecule
            return None
        
        


    
def d2c_geom(geom_string, center_geometry = False):
    ###########################################
    ##
    ##  Convert geometry from Dalton to Crystal
    ##  Author: Audun
    ##
    ###########################################

    #IMPORTANT: The following conversion factor is specified only here in the
    #           entire code.
    #           Multiple values defined at different stages of running a 
    #           project will cause inconsistensies in the basis.

    b2a = 0.52917720830000000000000000000 #bohr per angstrom conversion (value from crystal)
           
    geom = geom_string.split("\n")[4:]
    angstrom = False
    if geom[0].split()[-1] in ["angstrom" or "Angstrom"]:
        angstrom = True
        print("\033[93m" + "WARNING:"+ "\033[0m Detected angstrom keyword in .INP file.")
        print("         Will convert to bohr.")

        ngeom = geom_string.split("Charge=")
        new_geom = ngeom[0].replace("angstrom", "")
        
        
        for i in range(1,len(ngeom)):
            new_geom += "Charge=" + ngeom[i].split("\n")[0] + "\n"
            positions = ngeom[i].split("a1")[0].split("\n")
            
            for j in range(1,len(positions)):
  
                pos_split = positions[j].split()
                if len(pos_split)>0:
                    #converting positions to bohr
                    pos = [literal_eval(k) for k in pos_split[1:]]
                    pos_bohr = [str(k/b2a) for k in pos]
                    
                    #writine the atom and its position
                    new_geom += pos_split[0] + " "
                    for k in pos_bohr:
                        new_geom += k + " "
                    new_geom += "\n"    
        
        #then convert the lattice
        l1 = [literal_eval(k) for k in geom_string.split("a1 =")[1].split("\n")[0].split()[:-1]]
        new_geom += "a1 = %.15e %.15e %.15e " % (l1[0]/b2a, l1[1]/b2a, l1[2]/b2a)      
        new_geom += geom_string.split("a1 =")[1].split("\n")[0].split()[-1] + "\n"
        
        l2 = [literal_eval(k) for k in geom_string.split("a2 =")[1].split("\n")[0].split()[:-1]]
        new_geom += "a2 = %.15e %.15e %.15e " % (l2[0]/b2a, l2[1]/b2a, l2[2]/b2a)     
        new_geom += geom_string.split("a2 =")[1].split("\n")[0].split()[-1] + "\n"   
        
        l3 = [literal_eval(k) for k in geom_string.split("a3 =")[1].split("\n")[0].split()[:-1]]
        new_geom += "a3 = %.15e %.15e %.15e " % (l3[0]/b2a, l3[1]/b2a, l3[2]/b2a)    
        new_geom += geom_string.split("a3 =")[1].split("\n")[0].split()[-1] + "\n"     
        
        
        

        
        #replace geometry string with converted geometry
        geom = new_geom.split("\n")[4:]
        geom_string = new_geom
    
    
    
    # verify that 
    # all atomic positions is within [-.5*lvec, .5*lvec) in 
    # periodic directions. 
    # if not : translate atoms in returned string
    
    #lattice vectors
    plattice_vectors = []
    if geom_string.split("a1 =")[1].split("\n")[0].split()[-1] == "active":
        plattice_vectors.append(np.array([literal_eval(k) for k in geom_string.split("a1 =")[1].split("\n")[0].split()[:-1]]))
    if geom_string.split("a2 =")[1].split("\n")[0].split()[-1] == "active":  
        plattice_vectors.append(np.array([literal_eval(k) for k in geom_string.split("a2 =")[1].split("\n")[0].split()[:-1]]))
    if geom_string.split("a2 =")[1].split("\n")[0].split()[-1] == "active":
        plattice_vectors.append(np.array([literal_eval(k) for k in geom_string.split("a3 =")[1].split("\n")[0].split()[:-1]]))
    #lattice_vectors = [l1,l2,l3]
    
    geom_header_string =  geom_string.split("Charge")[0]
    
    geom_end_string = "a1 =" + geom_string.split("a1 =")[1] #ending
    
    geom_atoms_string = geom_string.split("a1 =")[0].split("Charge")[1:]
    
    geom_atoms_string_new = ""
    
    #center_geometry = True
    
    if center_geometry:
        
        #find coordinate center (mean coordinate)
        
        atom_coordinates = []
        
        for i in geom_atoms_string:
    
            atoms = i.split("\n")[1:]
            
            for line in atoms:
    
                p = np.array([literal_eval(j) for j in line.split()[1:]])
                
                if len(p) != 0:
                    
                    atom_coordinates.append(p)
                
        center = np.mean(np.array(atom_coordinates), axis = 0)
        
        
        
        for i in geom_atoms_string:
    
            atoms = i.split("\n")[1:]
            
    
            geom_atoms_string_new += "Charge" + i.split("\n")[0] + "\n"
    
            for line in atoms:
    
                p = np.array([literal_eval(j) for j in line.split()[1:]])
    
                if len(p)!=0:
    
                    geom_atoms_string_new += line.split()[0] + " %.8e %.8e %.8e \n" % (p[0]-center[0], p[1]-center[1], p[2]-center[2])    
    else:
        
        for i in geom_atoms_string:
    
            atoms = i.split("\n")[1:]
    
            geom_atoms_string_new += "Charge" + i.split("\n")[0] + "\n"
    
            for line in atoms:
    
                p = np.array([literal_eval(j) for j in line.split()[1:]])
    
                if len(p)!=0:
                    #if center_atoms_in_cell:
                        #center atom
                        
                    
    
                    for lvec in plattice_vectors:
    
                        if np.dot(p,lvec)/np.float(np.dot(lvec,lvec))>.5:
                            #subtract the lattice vector to shift coordinates inside 
                            #crystals "asymmetric unit", so input is consistent in all
                            #codes (crystal would do something similar to this)
                            
                            
                            print("\033[93m" + "WARNING:"+ "\033[0m Periodic coordinates outside primitive reference cell.")
                            print("         Coordinate:", p)
                            print("         Translated:", p-lvec)
                            
                            #p -= lvec
    
                    geom_atoms_string_new += line.split()[0] + " %.8e %.8e %.8e \n" % (p[0], p[1], p[2])
            
    geom_string = geom_header_string + geom_atoms_string_new + geom_end_string
    
    
    #iterate over all atoms
    
    
    n_atomtypes = literal_eval(geom[0].split()[0].split("=")[1])

    atom_types = geom_string.split("Charge=")[1:]
    

    
    charges = []
    atoms = []
    n_atoms_tot = 0
    
    for i in atom_types:
        atom_geom = i.split("\n")
        
        charge = literal_eval(atom_geom[0].split()[0])
        
        n_atoms = literal_eval(atom_geom[0].split()[1].split("Atoms=")[1])
        
        
        for j in range(n_atoms):
            atom_xyz = atom_geom[j+1].split()[1:]

            atom_pos = [literal_eval(n) for n in atom_xyz]
            
            atoms.append(atom_pos)
            
            charges.append(int(charge))
            
            n_atoms_tot += 1

    

    
    ###################################
    ## Determine periodicity         ##
    ###################################
    
    lattice_vectors_string = atom_types[-1].split("\n")[-4:-1]
    
    #read every active lattice vector
    lattice_vectors = []
    periodicity = [0,0,0]
    dim_type = 0
    for i in range(len(lattice_vectors_string)):
        lvec_i = lattice_vectors_string[i].split("=")[1].split()
        status = lvec_i[-1]
        lattice_vectors.append([literal_eval(k) for k in lvec_i[:-1]])
        
        if status == "active":
            periodicity[i] = 1
            dim_type += 1
            

            
            
        #if status == "inactive":
        #    lattice_vectors.append([0,0,0])
        #    lattice_vectors[-1][i] = 1.0
            
    
    dimensionality = ["MOLECULE", "POLYMER", "SLAB", "CRYSTAL"][dim_type]
    
    #lattice_param = float(literal_eval(lattice_vectors_string[0].split("=")[1].split()[0]))
    
    
    
    ################################
    # Conversion of lattice
    ################################
    
    c_lat = lattice_converter(b2a*np.array(lattice_vectors), periodicity)    
    
    if dim_type == 1:
        lattice_param = float(literal_eval(lattice_vectors_string[0].split("=")[1].split()[0]))
    

    
    
    #if crystal, triclinic should cover all possibilities
    lattice_param = 1
    if dim_type == 1:
        lattice_param = float(literal_eval(lattice_vectors_string[0].split("=")[1].split()[0]))
    if dim_type == 2:
        #slab
        pass
    if dim_type == 3:
        #crystal
        periodic_lattice = c_lat.as_triclinic()
        
    ###################################
    ## Writing geometry for crystal  ##
    ###################################
    
    
    
    #crystal_geom = "GENERATED %s\n%s\n1\n%.15f\n" % (dimensionality, dimensionality, b2a*lattice_param)
    
    crystal_geom = "GENERATED %s\n%s\n%s\n" % (dimensionality, dimensionality, c_lat.bravais_conversion())
    
    
    n_electrons = 0
    
    crystal_geom+= "%i\n" %n_atoms_tot

    for i in range(n_atoms_tot):
        #print (np.array(atoms[i])*b2a)

        x,y,z = c_lat.get_periodic_coords(np.array(atoms[i])*b2a) 
        
        crystal_geom += "%i %.15f %.15f %.15f\n" % (charges[i], x,y,z)
                                                
        n_electrons += charges[i]
        
    crystal_geom+="endg\n" #end geometry section
    
    return crystal_geom, n_electrons, charges, geom_string, angstrom

def d2c_properties(wannier_string):
    ###########################################
    ##
    ##  Generate property input files for Crystal
    ##  Author: Audun
    ##
    ###########################################
    pass
    
def get_dalton_basis(basis_type, atoms):
    try:
        basis_folder = os.environ['LSDALTON_BASIS_FOLDER']
    except:
        basis_folder = ""
        print("WARNING: Environment variable LSDALTON_BASIS_FOLDER not set.")
    
    
    f = open(basis_folder +"/" + basis_type, "r")
    print("Reading basis set from:",(basis_folder +"/" + basis_type))
    basis = ""
    F = f.read().lower().split("\na ")
    
    f.close()
    
    #list atomic numbers of available functions
    

    
    for i in atoms:
        for a in range(1,len(F)):
            if literal_eval(F[a].split()[0]) == i:
                basis += "a " + F[a].lstrip() + "\n"
    
    return basis

    


def d2c_basis(basis_string):
    ###########################################
    ##
    ##  Convert basis from Dalton to Crystal
    ##  Author: Audun
    ##
    ###########################################
    
    atoms = basis_string.split("a ")[1:]
    basis = []
    atomic_numbers = []
    shell_types = []
    contracted_numbers = []
    
    # Parse basis from Dalton
    
    #####
    ##
    ## New parsing routine
    ##
    ######

    basis_set = []
    atomic_numbers = []
    for i in atoms:
        basis_set.append([])
        ao_function = i.split("\n")
        
        atomic_numbers.append(int(ao_function[0]))
        
        b_set= [[literal_eval(k) for k in line.split()] for line in ao_function[1:]]

        c = 0 #counter
        shelltype = -1
        while c<len(b_set):
            params = b_set[c]
            if len(params)!=0:
                if type(params[0]) == int:
                    #shell line
                    shelltype += 1
                    basis_set[-1].append([])
                    
                    n_primitives = params[0]
                    n_contracted = params[1]

                    primitives = np.array(b_set[c+1:c+1+n_primitives])
                    
                    for l in range(1,n_contracted+1):
                        basis_set[-1][-1].append([])
                        for k in range(n_primitives):
                            if primitives[k,l] != 0.0:
                                basis_set[-1][-1][-1].append([primitives[k, 0], primitives[k,l]])
                    #c += n_primitives-1 #skip following lines
            c += 1

    #generate basis for crystal and libint

    crystal_basis = ""
    libint_basis = ""

    #Params in Crystal

    input_type = 0 #we will only use userdefined basis functions here
    
    scale_factor = 1.0

    atom_type = ["H", "He", "Li", "Be", "B","C","N","O","F","Ne","Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr" ]
    orb_type = "spdfghij"

    basis_info = []

    for atom in range(len(basis_set)):
        libint_basis  += "****\n" 
        atom_number = atomic_numbers[atom]
        
        basis_info.append([atom_number, ""])
        
        charge_counter = 0+atom_number #number of electrons to distribute
        
        #count number of contracted in atom

        n_contracted = 0
        for shell in range(len(basis_set[atom])):
            n_contracted += len(basis_set[atom][shell])
      
        crystal_basis += "%i %i\n" % (atom_number, n_contracted)
        
        libint_basis  += "%s 0\n" % atom_type[atom_number-1]  
        
        for shell in range(len(basis_set[atom])):
            crystal_shell = shell+0
            
            
            #account for special sp-type shell in crystal
            if shell>=1:
                crystal_shell += 1
                
            n_contracted = len(basis_set[atom][shell]) #number of contracted in atom
            
            
                
            for contracted in range(n_contracted):
                
                basis_info[-1][1] += orb_type[shell]
                
                n_primitives_in_contracted = len(basis_set[atom][shell][contracted])
                
                n_electrons_in_shell = (2*shell+1)*2 #number of electrons accomodated by shell
                
                if n_electrons_in_shell<=charge_counter:
                    #still more charge to distribute than accommodated by shell
                    charge_counter -= n_electrons_in_shell
                else:
                    #if n_electrons_in_shell>charge_counter:
                    #final shell filled with remaining (or zero) electrons
                    n_electrons_in_shell = max(0, charge_counter)
                    charge_counter -= n_electrons_in_shell
                
                crystal_basis += "  %i %i %i %.15f %.15f\n" % (input_type, 
                                                        crystal_shell, 
                                                        n_primitives_in_contracted,
                                                        n_electrons_in_shell,
                                                        scale_factor)
                libint_basis += "%s %i %f\n" % (orb_type[shell], n_primitives_in_contracted, 1.00)
                primitives = basis_set[atom][shell][contracted]
                for p in range(len(primitives)):
                    crystal_basis += "    %.15f %.15f\n" % (primitives[p][0],
                                                            primitives[p][1])
                    libint_basis +=  "    %.15f %.15f\n"  % (primitives[p][0],
                                                            primitives[p][1])

    libint_basis += "****\n"
    crystal_basis += "99 0\nendbs\n"
                                                        
    return crystal_basis, basis_info, libint_basis

    
    
def c2d_basis(basis):
    ###########################################
    ##
    ##  Convert basis from Crystal to Dalton
    ##  Author: Audun
    ##
    ###########################################
    pass
    
