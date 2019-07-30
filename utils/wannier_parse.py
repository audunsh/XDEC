
import numpy as np
import subprocess as sp
import sys
import os
from ast import literal_eval

def get_crystal_localization_info(fname = "CRYSTAL.OUT"):
    #####
    ##
    ##  Returns initial+final wannier centers and spread for each crystal orb.
    ##
    ######
    f = open(fname, "r")
    F = f.read()
    f.close()
    centers = []
    spreads = []
    for i in F.split("CENTROID'S COORDINATES R0:")[1:]:
        centers.append([literal_eval(j) for j in i.split("\n")[0].split()])
        spreads.append(literal_eval(i.split("\n")[1].split()[-1]))
    return centers, spreads
    

def get_localization_info(fname = "LSDALTON.OUT"):
    #####
    ##
    ##  Returns initial+final wannier centers and spread for eacb.
    ##
    ######
    
    f = open(fname, "r")
    F = f.read()
    f.close()
    
    orbinfo = F.split("WF_CENTERS_BEGIN")[1].split("WF_CENTERS_END")[0]
    
    centers = []
    spreads = []
    
    for line in orbinfo.split("\n"):
        #if len(line) > 0:
        elms = line.split()
        if len(elms)>0:
            centers.append([literal_eval(i) for i in elms[4:]])
            spreads.append(literal_eval(elms[2]))
    centers = np.array(centers)
    spread = np.array(spreads)
    
    return centers, spread

def get_wfvis(project_folder_f):
    #generate a visualization of the wannier functions
    
    C = np.load(project_folder_f + "/crystal_reference_state.npy")
    Cc = np.load(project_folder_f + "/crystal_reference_coords.npy")
    
    #sort in order of increasing coords
    cmax = np.max(Cc)
    order = np.argsort(np.sum(Cc*np.array([1,cmax,cmax**2]), axis = 1))
    
    C = C[order]
    Cc = Cc[order]
    
    #C = C.reshape(C.shape[0]*C.shape[1], C.shape[2])
    C = np.einsum("iap->ipa", C)
    C = C.reshape(C.shape[0]*C.shape[1], C.shape[2])
    #indexes in C : C[L, alpha, p] - > want C[p,alpha]
    

    import scipy.misc as sm
    
    sm.imsave(project_folder_f + "/crystal_reference_state.png", C**2)
    
    
    
    
    
    

def get_bandstructure(fname = "wann_occ.outp"):
    f = open(fname)
    F = f.read()
    #print(F)
    f.close()
    
    evals = F.split("KFULL KFULL")[-1].split("EIGENVALUES - K=")
    
    bands = []
    kpoints = []
    
    for i in np.arange(1,len(evals)):
        c = evals[i].split("EIGENVECTORS")[0]
        #print(c)
        k = [literal_eval(i) for i in c.split("(")[1].split(")")[0].split()]
        #print(k)
        e = [literal_eval(i) for i in c.split("(IBZ)")[1].split()]
        #bands.append(e)
        
        
        bands.append(e)
        kpoints.append(k)
    return [bands, kpoints]
    

def get_scf_energies(fname = "crystal.out"):
    #########################################
    ##
    ##  Extract SCF energy from crystal ("TOTAL ENERGY")
    ##  Returns a string with energy output + a float of E_ref
    ##
    ##########################################
    f = open(fname, "r")
    E = f.read().split("+++")[2].split("TTTTTTT")[0] #.split("\n")
    E_ref = float(E.split("TOTAL   ENERGY")[1].split("\n")[0])
    return E, E_ref


#tool to interpret lines in matrix output from crystal
def linetype(l):
  #########################################
  ##
  ##  Evaluate which kind of line is read from crystal output
  ##  returns "Column", "Row" or None
  ##
  ##########################################
  lt = None
  ls = l.split()
  try:
    if "." not in ls[0] and len(ls[0]) > 0:
      #ls[0] probably integer
      if "." not in ls[1] and len(ls[1]) > 0:
        #ls[1] probably integer as well
        #-> ls is column indices
        lt = "column"
      if "." in ls[1]:
        #ls[1] probably double
        lt = "row"
      #else:
      #  lt = "column"


  except:
    lt = None
    if len(ls) == 1 and ls[0]== "1":
        lt = "column"
  return lt
    
     
#read matrix block in crystal output, return numpy matrix and coord array
def parse_matrix_block(s, bsize = None):
  #########################################
  ##
  ##  Read a string matrix from Crystal output
  ##  returns dense np.array (float)
  ##
  ##########################################
  #0 . get lattice index
  line = s.split("""\n""")
  cell_coord = [int(i) for i in line[0].strip(")").split()]

  #1. read matrix line by line, store as sparse matrix
  sparsemat = [[],[],[]] #row/col/elem  

  for l in range(2, len(line)):
    lt = linetype(line[l]) 
    if lt == "column":
      col = [int(i) for i in line[l].split()]
    if lt == "row":
      r = line[l].split()
      row = int(r[0])
      elms = [float(u) for u in r[1:]]
      
      for n in range(len(elms)):
        sparsemat[0].append(row)
        sparsemat[1].append(col[n])
        sparsemat[2].append(elms[n])


  
  #2. determine number of rows and cols
  nrows, ncols = max(sparsemat[0]), max(sparsemat[1]) 
  if bsize != None:
      nrows, ncols = bsize[0], bsize[1]

  #Create dense np.array of floats  
  M = np.zeros((nrows, ncols), dtype = float)

  for i in range(len(sparsemat[0])):
    r,c,e = sparsemat[0][i], sparsemat[1][i], sparsemat[2][i]
    M[r-1,c-1] = e

  return cell_coord, M  
 

#Extract wannier coeffs from crystal output

def extract_c14_output(fname="evecs.txt", bsize = None):
  #########################################
  ##
  ##  Read output from Cryapi 
  ##  returns wannier functions as list of np.arrays in float
  ##  (Should be done for virtuals and occupied separately)
  ##
  ##########################################  
  f = open(fname, "r")
  Fn = f.read()
  
  
  # Get number of bands
  bands = [literal_eval(i) for i in Fn.split("BANDS")[1].split("G")[0].split()]

  n_bands = len(bands)
  
  bsize = [bsize, n_bands]
  
  
  F = Fn.split("G = (")
  
  f.close()

  blocks = []
  
  coords = []
  
  for i in range(len(F)-1):
      
    block = F[i+1].split("WANNIER")[0]
    
    cell_coord, M  = parse_matrix_block(block, bsize)
    
    blocks.append(M)
    
    coords.append(cell_coord)

  return blocks, coords, n_bands


def get_permutation_vector(ordering):
    #########################################
    ##
    ##  Get a index permutation vector based on AO ordering
    ##  Crystal does not use the same order as LSDalton and Libint
    ##  This will sort d,e,f,... orbitals, while s and p are unaffected
    ##  For example:
    ##    WF_sorted = WF[get_permutation_vector(ordering)]
    ##
    ########################################## 
    
    # get full size and every unique basis function (and type)

    bsize = 0
    ao_order = ""
    btypes = "spdfgh"
    #print(ordering)
    for i in ordering:
        if i in btypes:
            #print(":", i)
            #print(i,btypes.index(i))
            bsize += 2*btypes.index(i) +1  
            ao_order +=  (2*btypes.index(i) +1)*i
    
    perm_pattern = np.arange(bsize)
    
    c = 0
    while c<bsize:
        o_type = ao_order[c]
        if o_type == "s":
            c += 1
            pass
        if o_type == "p":
            c += 3
            pass
        if o_type == "d":
            perm_pattern[c:c+5] = c + np.array([4, 2, 0, 1, 3])
            c += 5
            pass
        if o_type == "f":
            perm_pattern[c:c+7] = c + np.array([6, 4,2,0,1,3,5])
            c += 7
            pass
            
        if o_type == "g":
            perm_pattern[c:c+9] = c + np.array([8,6, 4,2,0,1,3,5,7])
            c += 9
            pass
        if o_type not in "spdfg":
            print("WARNING: ", o_type, "-orbitals not yet supported. See get_permutation_vector.")
            c += 1
            pass
        #c += 1
    return perm_pattern
    

            


#save output file for wannier code

def export_c14_output(f_occupied="occ.txt", f_virtual="virt.txt", ordering = None, outfile = "crystal_output.txt", bsize = None, perm_pattern = None, project_folder_f = ""):
    #########################################
    ##
    ##  Read output files from Cryapi
    ##  Write WFs to file crystal_output.txt (reference_state.txt)
    ##
    ##########################################     

    #get number of basis functions
    if bsize == None:
        bsize = 0
        btypes = "spdfgh"
        for i in ordering:
            if i in btypes:
                bsize += 2*btypes.index(i) +1    
            
    if perm_pattern == None:
        perm_pattern = get_permutation_vector(ordering)

    blocks, coords, n_occ = extract_c14_output(f_occupied, bsize)     
    n_virt = 0
    
    if f_virtual!=None:
        blocks_v, coords_v, n_virt = extract_c14_output(f_virtual, bsize)
    

    else:
        #workaround, create empty matrices
        n_virt = bsize-n_occ
        blocks_v = []
        coords_v = []
        for i in range(len(blocks)):
            coords_v.append(coords[i])
            blocks_v.append(np.zeros((bsize, n_virt), dtype = float))
    
    blocks_temp = []
    coords_temp = []
    
    #add orbitals to temp lists
    
    #occupieds
    

    
    for i in range(len(coords)):
        #b = np.zeros((n_occ+n_virt, bsize), dtype = float) #original, pre frozen
        
        b = np.zeros((bsize,n_occ+n_virt), dtype = float) #new, post frozen
        
        

        coords_temp.append(coords[i])
        b[:, :n_occ] = blocks[i] #this one is transposed
        blocks_temp.append(b)
    
    if f_virtual!=None:
        #append virtual orbitals
        for i in range(len(coords_v)):
            #check to see if block exists
            block_found = False
            for j in range(len(coords)):
                if coords_v[i] == coords[j]:

                    blocks_temp[j][:, n_occ:] = blocks_v[i]
                    block_found = True
                    break
            if block_found == False:
                #b = np.zeros((n_occ+n_virt, bsize), dtype = float)
                
                b = np.zeros((bsize, n_occ+n_virt), dtype = float)
                coords_temp.append(coords_v[i])
                b[:,n_occ:] = blocks_v[i]
                blocks_temp.append(b)

    blocks = blocks_temp
    coords = coords_temp
    
    # for some reason, the lattice convention for the coefficients is 
    # not equivalent to the one used for symmetric matrices in Crystal
    # we correct for it here
    # C^{L-M}_{p \mu} := C^{ML}_{p \mu}
    coords = -1*np.array(coords)
    
    print(perm_pattern)

    #permute according to ordering
    for k in range(len(coords)):
        blocks[k] = blocks[k][perm_pattern]

    np.save(project_folder_f + "/crystal_reference_state.npy", blocks)
    
    np.save(project_folder_f + "/crystal_reference_coords.npy", coords)
    
    #print("saving output")
    
    f = open(project_folder_f + "/" + outfile, "w")
    
    f.write(str(len(coords)) + """\n""") #number of blocks
    
    f.write(str(5+len(blocks[0][0])) + """\n""") #number of lines per block  
    
    for i in range(len(blocks)):

        for n in range(3):
            
            f.write(str(coords[i][n]) + """\n""")
        
        f.write(str(len(blocks[i])) + """\n""")
        
        f.write(str(len(blocks[i][0])))
        
        f.write("""\n""")
    
        for row in range(len(blocks[i])):
            
            for col in range(len(blocks[i][0])):
                
                f.write("%.20E" % blocks[i][row, col] + " ")
            
            f.write("""\n""")

    f.close()
    
    print("Done.")



def export_c14_overlaps(outfile, blocks, coords):
    f = open(outfile, "w")
    
    f.write(str(len(coords)) + """\n""") #number of blocks
    
    f.write(str(5+len(blocks[0][0])) + """\n""") #number of lines in files  
    
    for i in range(len(blocks)):
        

        for n in range(3):
            
            f.write(str(coords[i][n]) + """\n""")
        
        f.write(str(len(blocks[i])) + """\n""")
        
        f.write(str(len(blocks[i][0])))
        
        f.write("""\n""")
    
        for row in range(len(blocks[i])):
            
            for col in range(len(blocks[i][0])):
                
                f.write("%.20E" % blocks[i][row, col] + " ")
            
            f.write("""\n""")

    f.close()

if __name__ == "__main__":
    occ, virt = sys.argv[1], sys.argv[2]
    
    if virt=="None":
        virt = None
    
    #read basis function order
    
    f = open(sys.argv[3], "r")
    basis_info = f.read()
    f.close()
    
    
    
    export_c14_output(f_occupied=occ, f_virtual=virt, ordering = basis_info)
    

###########################
##
## Functions to extract S (overlap), P (density) and F (fock) from Crystal
##
############################

def get_dense(M):
    #parse from text outputted crystal matrix
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
    blocks = M.split("G")
    coords = []
    dblocks = []
    for i in range(1, len(blocks)):
        m = blocks[i].split("\n")
        coords.append( [literal_eval(j) for j in m[0].split()] )
        dblocks.append(get_dense(m[1:]))
    
    return coords, dblocks
    
def symmetrize(lower_triangular,upper_triangular):
    #lower = -, upper = +
    
    ret = lower_triangular + upper_triangular.T 
    #HERE: TRANSPOSE  the first one to create latter matrix (+) 
    np.fill_diagonal(ret, np.diag(lower_triangular))
    return ret
    

    
def symmetrize_full_(coords, blocks):
    ###################################
    ##
    ## Restore dense form of triangular blocked lattice matrices
    ##
    ###################################
    
    coords = np.array(coords)
    sblocks = []
    scoords = []
    
    # make sure the reference cell is included
    # no matter where it occurs
    for c in range(len(coords)):
        if (coords[0] == np.array([0,0,0])).all():
            sblocks.append(symmetrize(blocks[0], blocks[0]))
            scoords.append(coords[0])
        
    # assume crystal ordering: +1, -1, +2, -2, ...
    for c in range(len(coords)-1):
        if (coords[c] == -1*coords[c+1]).all():
            sblock = symmetrize(blocks[c], blocks[c+1])
            sblocks.append(sblock)
            scoords.append(coords[c])
            
            sblocks.append(sblock.T)
            scoords.append(coords[c+1])

    return scoords, sblocks
    
def symmetrize_full(coords, blocks):
    ###################################
    ##
    ## Restore dense form of triangular blocked lattice matrices
    ##
    ###################################
    
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
    
def get_matrices(fname, basis_sequence, ordering = None):
    #print(basis_sequence)
    #read matrices from crystal output (cryapi gred.dat)
    #return matrices in numpy format with associated coords
    if ordering == None:
        ordering = get_permutation_vector(basis_sequence)
        
    #print(ordering)
    

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
    
    print(Sblocks.shape)
    #permute blocks 
    for i in np.arange(len(Sblocks)):
        Sblocks[i] = Sblocks[i][ordering,:][:,ordering]
    
    for i in np.arange(len(Fblocks)):
        Fblocks[i] = Fblocks[i][ordering,:][:,ordering]
        
    for i in np.arange(len(Pblocks)):
        Pblocks[i] = Pblocks[i][ordering,:][:,ordering]
    
    
    return Sblocks, Fblocks, Pblocks, Scoords, Fcoords, Pcoords    

def read_wannier_mos(filename = "wannier.mos"):
    blocks = []
    coords = []
    
    f = open(filename, "r")
    F = f.read().split("Cell coor:")
    for j in range(1,len(F)):
        cellblock = F[j].split("\n")
        
        cell = [literal_eval(i) for i in cellblock[0].split()[:-1]]
        
        block = []
        for i in range(1,len(cellblock)):
            #block.append([literal_eval(k) for k in cellblock[i] ])

            try:
                row = [literal_eval(k) for k in cellblock[i].split()]
                if len(row) != 0:
                    block.append(row)
            except:
                pass
        blocks.append(np.array(block))
        coords.append(cell)
    return np.array(blocks), np.array(coords)

def get_coord_index(coords, c):
    #return index n where coords[n] == c
    n = False
    for i in range(len(coords)):
        if (coords[i] == c).all():
            n = i
            break
    return n

def get_crystal_sorting_1d(coords):
    #this should be generalized to any dimension,
    #but then we need to know the crystal convention in detail
    lst = []
    c = 0
    indx = np.array([0,0,0])
    lst.append(get_coord_index(coords, indx))
    for c in range(1,coords[:,0].max()+1):
        indx = np.array([c,0,0])

        lst.append(get_coord_index(coords,-indx))
        lst.append(get_coord_index(coords,indx))
    return lst
    
    
            

def blocks_to_reference_state(blocks,coords, order = None):
    #convert numpy arrays into a string compatible with the crystal_output.txt format
    #this format is not optimal, but so far we use it to communicate with Cryscor 
    #so we still need it
    if order == "crystal":
        indx = get_crystal_sorting_1d(coords)

        blocks = blocks[indx]
        coords = coords[indx]
    
    ret = "%i\n%i\n" % (len(blocks), len(blocks))
    for i in range(len(coords)):
        ret+= "%i\n%i\n%i\n" % (coords[i][0], coords[i][1], coords[i][2]) #cell
        ret+= "%i\n%i\n" % blocks[i].shape
        for j in range(len(blocks[i])):
            line = ["%.20e " %(n) for n in blocks[i][j]]

            for l in line:
                ret+= l.replace("e","E")
            ret += "\n"

    return ret

def convert_wannier_mos(infile = "wannier.mos", outfile = "reference_state.txt"):
    #read wannier.mos, store in reference_state format used
    #by cryscor, lsdalton and PeriodicDEC

    blocks, coords =  read_wannier_mos(infile)
    
    
    ref_state = blocks_to_reference_state(blocks, coords, order = "crystal")
    
    f = open(outfile, "w")
    
    f.write(ref_state)
    
    f.close()
    

def wanniermos2npy(infile = "wannier.mos", project_folder = ""):
    #convert wannier.mos to lsdalton_reference_state.npy
    
    blocks, coords =  read_wannier_mos(infile)
    
    
    
    
    # temporary "hack"-fix, problem with npy-output, 1/11/2017
    B = []
    B_c = []
    for i in np.arange(len(coords)):
        b = blocks[i]
        if b.size>=1:
            B.append(blocks[i])
            B_c.append(coords[i])
    blocks = np.array(B)
    
    
    #blocks = np.array(blocks)
    coords = np.array(B_c)
    np.save(project_folder + "lsdalton_reference_state.npy", blocks)
    np.save(project_folder + "lsdalton_reference_coords.npy", coords)
    return blocks, coords

def xdec2cryscor(nocc, blocks, coords, outfile):
    #blocks, coords = wanniermos2npy(project_folder + "/DEC/wannier.mos", project_folder)
    #blocks = np.load(fname_blocks)
    nbast = blocks.shape[1] #len(blocks[len(blocks)/2])
    #t = "8 48\n" #occupied, basis functions
    t = "%i %i\n" % (nocc, nbast)
    #t = "6 28\n"
    for i in np.arange(len(coords)):
        c = coords[i]
        b = blocks[i]
        if np.abs(blocks[i]).max()>=10e-17:
            for ic in c:
                t += "%i " % ic
            t += "\n"
            for ir in range(len(b)):
                for ic in range(len(b[ir])):
                    t += "%.15e " % (b[ir][ic])
                t += "\n"
                
    t += "1000 1000 1000\n" #finish with something "crazy"
    f = open(outfile, "w")
    f.write(t)
    f.close()

def wmos2cryscor(nocc, project_folder):
    blocks, coords = wanniermos2npy(project_folder + "/DEC/wannier.mos", project_folder)
    #blocks = np.load(project_folder + "/crystal_")
    nbast = len(blocks[len(blocks)/2])
    #t = "8 48\n" #occupied, basis functions
    t = "%i %i\n" % (nocc, nbast)
    for i in np.arange(len(coords)):
        c = coords[i]
        b = blocks[i]
        if np.abs(blocks[i]).max()>=10e-17:
            for ic in c:
                t += "%i " % ic
            t += "\n"
            for ir in range(len(b)):
                for ic in range(len(b[ir])):
                    t += "%.15e " % (b[ir][ic])
                t += "\n"
                
    t += "1000 1000 1000\n" #finish with something "crazy"
    f = open(project_folder + "/cryscor_orbitals.txt", "w")
    f.write(t)
    f.close()
    
def read_reference_state(infile):
    f = open(infile, "r")
    F = f.read().split("\n")
    F[0] = nblocks
    F[5] = nlines_per_block
    coords = []
    blocks = []
    
