# -*- coding: utf-8 -*-

import sys
import os
from ast import literal_eval
#import numba
import numpy as np
import copy

from copy import deepcopy


from scipy.linalg import expm
from scipy.linalg import pinv2, pinv, svd
#from numba import jit

'''
Module for handling (truncated) Toeplitz matrices.

Authors: Audun Skau Hansen (main author) and
         Gustav Baardsen (modifications, additions,
         functions related to fast Fourier transform)
'''

def c2_not_in_c1(c1, c2, sort = False):
    """
    Returns the 3-vectors in c2 not present in c1
    """
    distance = np.sum((c1[:,None] - c2[None,:])**2, axis = 2)
    
    c1_in_c2 = np.any(distance==0, axis = 0)
    if not sort:
        return c2[c1_in_c2==False]
    if sort:
        c2_ret = c2[c1_in_c2==False]
        c2_ret = c2_ret[np.argsort(np.sum(c2_ret**2, axis = 1))]
        return c2_ret

def c2_union_c1(c1,c2, sort = False):
    dc = c2_not_in_c1(c1,c2)
    return np.append(c1, dc, axis = 0)

def test_tmat(t):
    '''
    This function tests an instance of tmat:
        - Data types
        - Coordinate transform
        - get methods
        - Domain definition
        - Consistent matrix product for diagonal block
    
    returns True if all tests pass
    '''
    
    t.test_data_types() # test data types
    
    #test coordinate transform
    coor_assertion = True
    for i in range(len(t.coords)):
        assert(np.all(t.coords[i]==t._i2c(t._c2i(t.coords[i])))), \
            "Inconsistent index transformation"
        
    #test matrix idempotency
    coor_assertion2 = True
    block_assertion = True
    index_assertion = True
    for i in range(len(t.coords)):
        assert(np.all(t.coords[i]==t._i2c(t._c2i(t.coords[i])))), \
            "Inconsistent index2coor transformation (_c2i)"
        
        assert(matrices_are_equal(t.blocks[i], t.cget(t.coords[i]))), \
            "Inconsistent matrix retrieval (cget)"
        
        assert(t.mapping[t._c2i(t.coords[i])] == i), \
            "Incinsistent coor2index transformation (_c2i)"
        
    #Test outside domain
    cmax = np.abs(t.coords.max(axis = 0))
    
    assert(L2norm(t.cget(2*cmax)) ==0), "Nonzero blocks outside truncation domain (+ direction)."
        
    assert(L2norm(t.cget(-2*cmax))==0), "Nonzero blocks outside truncation domain (- direction)."    
    
    #compute zeroth block brute force
    blocks = t.blocks
    coords = t.coords
    
    block0 = np.dot(blocks[0].T, blocks[0])
    
    for i in np.arange(1,blocks.shape[0]):
        block0 += np.dot(blocks[i].T,blocks[i])

    
    
    t2 = t.tT()*t
    
    
    assert(np.abs(L2norm(t2.get([0,0,0]))-L2norm(block0))<=10e-10), "Inconsistent t.tT()*t in 0,0,0"
    
     
    return True
    


        
def matrices_are_equal(m1,m2, precision = 10e-12):
    """
    Asserts that m1 == m2 to precision (per element)
    Author: Audun Skau Hansen
    """
    premise1 = np.abs(L2norm(m1)-L2norm(m2))/m1.size<=precision
    premise2 = np.abs(L2norm(m1-m2))/m1.size<=precision
    return premise1*premise2

def L2norm(m):
    """
    Computes the L2-norm of a matrix
    Author: Audun Skau Hansen
    """
    return np.sqrt(np.sum(m**2))

def max_norm(array):
    return np.amax(np.absolute(array))

def lattice_coords(cutoffs):
    """
    Generates all lattices coordinates for a given cutoff
    cuttofs = array with BvK type cuttofs,
    returns all coordinates within BvK domain inside +/-cuttoffs
    Author: Audun Skau Hansen
    """
    Nx = cutoffs[0]
    Ny = cutoffs[1]
    Nz = cutoffs[2]
    
    return np.array([np.kron(np.arange(-Nx,Nx+1, dtype = int), np.kron(np.ones(2*Ny+1,dtype = int), np.ones(2*Nz+1, dtype = int))  ),
                     np.kron(np.ones(2*Nx+1, dtype = int), np.kron(np.arange(-Ny,Ny+1, dtype = int), np.ones(2*Nz+1, dtype = int))  ),
                     np.kron(np.ones(2*Nx+1, dtype = int), np.kron(np.ones(2*Ny+1, dtype = int), np.arange(-Nz, Nz+1, dtype = int)) )]).T


def get_random_tmat(cutoffs, blockshape):
    """
    return a random tmat with cutoff and blockshape as dimensions
    mtype = 0 -> returns tmat object
    mtype = 1 -> returns latmat object
    Author: Audun Skau Hansen
    """
    coords = np.array(lattice_coords(cutoffs), dtype = int)
    blocks = np.random.uniform(-1,1,(len(coords), blockshape[0], blockshape[1]))
    
    ret = tmat()
    ret.load_nparray(blocks, coords)
    return ret
    
def get_tmat_from_gred(wannier_txt_file, blockshape, index_word = "WANNIER FUNCTIONS - LIST OF ACTIVE BANDS"):
    """
    Read output from Crystal, determine all non-zero blocks in BvK-supercell
    """
    f = open(wannier_txt_file, "r")
    F = f.read()
    f.close()
    F = os.linesep.join([s for s in F.splitlines() if s]) #remove empty lines
    F = F.split(index_word)[1].split("WANNIER")[0].split("G = ")

    bands = np.array([literal_eval(i) for i in F[0].split()])-1 # indexing begins at 0
    #print("Reading vectors ", bands, " into tmat.")
    
    G_vectors = []
    
    for i in np.arange(1,len(F[1:])):
        # Reading block index vector
        
        G = -1*np.array([literal_eval(j) for j in F[i].split(")")[0].split("(")[1].split()])
        G_vectors.append(G)
    #print(G_vectors)
    G_vectors = np.array(G_vectors)
    cutoffs = np.max(np.abs(G_vectors), axis = 0)
    return get_zero_tmat(cutoffs, blockshape)


def get_zero_tmat(cutoffs, blockshape):
    coords = np.array(lattice_coords(cutoffs), dtype = int)
    
    return setup_zero_tmat(coords, blockshape)

def get_identity_tmat(N):
    coords = np.array([[0,0,0], [1,0,0]])
    blocks = np.zeros((coords.shape[0], N,N))
    blocks[0] = np.eye(N)
    I = tmat()
    I.load_nparray(blocks, coords)
    return I
    
def setup_zero_tmat(coords, blockshape):
    blocks = np.zeros((len(coords), blockshape[0], blockshape[1]),
                      dtype = float)
    ret = tmat()
    ret.load_nparray(blocks, coords, screening = False)
    return ret


def screen1(coords, blocks, norm, tolerance):

    si = np.max(np.abs(blocks), axis = (1,2))>tolerance
    si[np.sum(coords**2, axis = 1)==0] = True
    return coords[si], blocks[si]

def screen(coords, blocks, norm, tolerance):
    '''
    Blocks with a norm larger than 'tolerance', except for the
    block [0, 0, 0]. are screened away. 
    
    coords        : Coordinates. A Numpy array of size (N, 3)
    blocks        : Elements corresponding to 'coords'. 
                    A Numpy array of size (N, n1, n2)
    WARNING: This routine assumes coincidence in the coords and blocks arrays
    '''
    n = coords.shape[0]
    screened_coords = []
    screened_blocks = []
    for i in range(n):
        
        if (np.max(np.abs(blocks[i])) > tolerance) or \
           (coords[i] == [0, 0, 0]).all():
            
            screened_coords.append(coords[i].tolist())
            screened_blocks.append(blocks[i].tolist())
            
    return np.array(screened_coords, dtype=int), \
        np.array(screened_blocks, dtype=blocks.dtype)

def screen_(coords, blocks, norm, tolerance = 1e-14):
    screening = np.zeros(blocks.shape[0], dtype = np.bool)
    screening = np.max(np.abs(blocks), axis = (1,2))>tolerance
    z0 = np.argwhere(np.sum(coords**2, axis = 1)==0)
    screening[z0] = True
    screening[-1] = True
    return coords[screening[:-1]],blocks[screening]




def screen__(coords, blocks, norm, tolerance = 1e-14):
    #screening = np.zeros(coords.shape[0], dtype = np.bool)
    screening = np.max(np.abs(blocks[:-1]), axis = (1,2))>tolerance
    z0 = np.argwhere(np.sum(coords**2, axis = 1)==0)
    screening[z0] = True
    #screening[-1] = True
    return coords[screening],blocks[:-1][screening]
        
def screen_tmat(m, tolerance = 1e-30):
    #coords, blocks = screen(m.coords, m.blocks, L2norm, tolerance)
    #ret = tp.tmat()
    mblocks = m.cget(m.coords)
    mx = np.max(np.abs(mblocks), axis = (1,2))>tolerance
    mx[np.sum(m.coords**2)==0] = True
    #print("mx:", np.sum(mx))
    mxi = np.sum(mx) 
    return tmat(m.coords[mx].reshape(mxi, 3), mblocks[mx].reshape(mxi, m.blocks.shape[1], m.blocks.shape[2]), screening = False)

def merge_blocks(tmatrix1, tmatrix2, axis):
    '''
    Merge blocks of two Toeplitz matrices.
    '''
    
    coords = tmatrix1.coords
    blocks_list = []
    
    for c1 in tmatrix1.coords:
        
        block = np.concatenate((tmatrix1.get(c1),
                                tmatrix2.get(c1)),
                               axis=axis)
        blocks_list.append(block.tolist())
        
    blocks = np.array(blocks_list)
    
    return tmat(coords = coords, blocks = blocks)


def convert_files_old2new(blockfile, coordfile, newfile):
    '''
    Convert coordinate and block files to the new format
    with coordinates and blocks in one file.
    '''
    t = tmat()
    t.load_old(blockfile, coordfile)
    t.save(newfile)
    
    
class tmat():
    """
    ###############################
    ##                           ##
    ##  Toeplitz matrix class    ##
    ##                           ##
    ###############################
    
    Optional input parameters
    
    
    
    """
    
    def __init__(self,
                 coords = None, 
                 blocks = None, 
                 blockshape = None, 
                 infile = None,
                 domain = None,
                 delim = None,
                 tolerance = 1e-30,
                 norm_tolerance=max_norm,
                 screening = True):
        
        self.tolerance = tolerance
        self.norm = norm_tolerance
        
        if blocks is not None:
            if blockshape is None:
                blockshape = blocks[0].shape
            
            if screening:
                # Screen out blocks with negligibly small
                # coefficients
                # NOTE: screening on this level assumes coincidental coords and blocks
                coords, blocks = screen(coords, blocks,
                                        self.norm, self.tolerance)
                
        if (coords is not None) and (domain is None):
            try:
                self.domain = np.amax(np.absolute(coords), axis=0) + 1
            except:
                print("Init error")
                print(coords)
                print(np.absolute(coords))
                print("")
                assert(False), "Error"
        else:
            self.domain = domain
            
        if self.domain is not None:
            #set up delimiters from domain
            self.delim = self.dom2delim(self.domain)
            
        self.zero_padded = False
        self.zero_block  = 0
        
        if blockshape is not None:
            self.blockshape = blockshape
            
        if coords is not None:
            self.coords = coords
            self.indices = self._c2i(self.coords)
            
        if blocks is not None:
            self.blocks = blocks
            self.map_blocks()
            
        if infile is not None:
            self.load(infile)
            
        if delim is not None:
            self.delim = delim
            
            
    # Functions to handle domain size and band-truncations

    def expand(self, domain):
        # Increase domain of matrix
        self.domain = domain
        
        delim = self.dom2delim(domain)
        
        self._extend(delim)
    
    def _extend(self, delim):
        # Remap according to delim
        # Will create new mapping array
        
        self.delim = delim
        
        self.blocks = self.blocks[:-1] #remove zero padding at end
        self.zero_padded = False
        
        self.indices = self._c2i(self.coords)
        
        self.map_blocks()
                
    def dom2delim(self, domain):
        # Compute delimiting vector from domain size
        # For index mapping:
        # I = m1 + m2 * self.delim[1] + 3*self.delim[2]
        delim = np.zeros(3, dtype = 'int64')        
        
        delim[0] =  1
        delim[1] =  domain[0] * 2 + 1
        delim[2] = (domain[0] * 2 + 1) * (domain[1] * 2 + 1)
        
        return delim

    def n_periodicdims(self):
        '''
        Get the number of periodic dimensions.
        '''
        sum_xyz = np.sum(np.abs(self.coords), axis=0)
        if np.sum(sum_xyz[0:]) == 0:
            # All coordinates are zero
            return 0
        elif np.sum(sum_xyz[1:]) == 0:
            # All y and z coordinates are zero
            return 1
        elif np.sum(sum_xyz[2:]) == 0:
            # All z coordinates are zero
            return 2
        else:
            return 3 
        
    def cutoffs(self):
        '''
        Get largest absolute values of cell coordinates
        in each Cartesian direction.
        '''
        return np.amax(np.absolute(self.coords), axis = 0)
    
    # Diagnostics
    
    def test_data_types(self):
        # Test input data content
        assert(self.coords.size > 0), "tmat: Warning: empty coords (size: %i)." % self.coords.size 
        assert(self.blocks.size > 0), "tmat: Warning: empty blocks (size: %i)." % self.blocks.size
        assert(self.delim.size  > 0), "tmat: Warning: empty delims (size: %i)." % self.delim.size
        
        # Test consistent dims
        #assert(self.coords.shape[0] == self.blocks.shape[0]-1), "tmat: Warning: conflicting number of blocks and coords."
        assert(self.coords.shape[1] == 3), "tmat: Warning: incorrect shape of coords"
        
        # Test input data types
        assert(self.coords.dtype == 'int64'),   "tmat: Warning: coords not int64 (dtype: %s)." % self.coords.dtype
        assert(self.blocks.dtype == 'float64' or self.blocks.dtype == 'float32'), "tmat: Warning: blocks not float64 (dtype: %s)." % self.blocks.dtype
        assert(self.delim.dtype ==  'int64'),   "tmat: Warning: delim not int64 (dtype: %s)." % self.delim.dtype
        
                                                                                    
    # File handling, output, input ...     
    
    def save_wanniermos(self, nocc, nbast, outfile = "wannier.mos"):
        wmos = "NOCC=%i\nNBAST=%i\n\n*** Molecular Coefficients ***\n" % (nocc, nbast)
        for i in np.arange(len(self.coords)):
            wmos += "--- Cell coor: %i %i %i ---\n" % (self.coords[i][0], self.coords[i][1], self.coords[i][2])
            block = self.cget(self.coords[i])
            if np.max(np.abs(block))>10e-10:
                # PRINT BLOCK   
                for j in np.arange(block.shape[1]):
                    for k in np.arange(block.shape[0]):
                        wmos += "%.15e " % block[j,k]
                    wmos += "\n\n"
                wmos += "\n"
        f = open(outfile, "w")
        f.write(wmos)
        f.close()
        
            
            
            
    
    def save(self, outfile):
        # Store toeplitz matrix to outfile
        
        savearray = np.zeros(3, dtype = object)
        savearray[0] = self.coords
        savearray[1] = self.cget(self.coords) #avoid storing zero blocks, ensure coincidence
        savearray[2] = self.domain 
        #self.delim
        
        np.save(outfile, savearray)

    def save_old(self, file_blocks, file_coords):
        '''
        Save blocks and coordinates in separate files.
        '''
        np.save(file_blocks, self.cget(self.coords))
        np.save(file_coords, self.coords)
    

        
        
    def load_nparray(self, blocks, coords, safemode = False,
                     screening = True):
        '''
        Init matrix from numpy arrays
        '''
        
        # Screen out blocks with negligibly small
        # coefficients
        # NOTE: Screening on this level assumes coincidental coords and blocks
        #print("Size of blocks (tmat):", blocks.shape, blocks.nbytes*1e-6, "Mb")
        if screening:
           coords, blocks = screen1(coords, blocks,
                                   self.norm, self.tolerance)
            
        self.blocks = blocks
        
        self.coords = np.array(coords)
        
        self.domain = np.amax(np.absolute(self.coords), axis=0) + 1 #add zero blocks along edges
        
        self.delim = self.dom2delim(self.domain)
        
        #self.delim = np.array([1,100,10000])
        
        if safemode:
            
            self.test_data_types()
            
        self.blockshape = self.blocks[0].shape
        
        self.indices = self._c2i(self.coords)
        
        self.map_blocks()  
        
    def load_old(self, blockfile, coordfile, safemode = True):
        '''
        Load same format used by old toeplitz class (latmat)
        '''
        blocks = np.load(blockfile)
        
        coords = np.load(coordfile)
        
        # Screen out blocks with negligibly small
        # coefficients
        self.coords, self.blocks = screen(coords, blocks,
                                          self.norm, self.tolerance)
        
        self.domain = \
            np.amax(np.absolute(self.coords), axis=0) + 1
        
        self.delim = self.dom2delim(self.domain)
        
        if safemode:
            
            self.test_data_types()
            
        self.blockshape = self.blocks[0].shape
        
        self.indices = self._c2i(self.coords)
        
        self.map_blocks() 
    
        
    def load_gred_dat(self, wannier_txt_file, index_word = "WANNIER FUNCTIONS - LIST OF ACTIVE BANDS", permutation = None):
        """
        Read output from Crystal directly into toeplitz class
        Author: Audun
        """
        f = open(wannier_txt_file, "r")
        F = f.read()
        f.close()
        F = os.linesep.join([s for s in F.splitlines() if s]) #remove empty lines
        F = F.split(index_word)[1].split("WANNIER")[0].split("G = ")
    
        bands = np.array([literal_eval(i) for i in F[0].split()])-1 # indexing begins at 0

        for i in np.arange(1,len(F[1:])+1):
            # Reading block index vector
            
            
            G = -1*np.array([literal_eval(j) for j in F[i].split(")")[0].split("(")[1].split()])

            gmap = self.mapping[self._c2i(G)]
            
            # parse block
            
            B = F[i].split(")")[1]
            
            # read elements in block

            for line in B.split("\n")[1:]:
                # note : Crystal is column-major (fortran)
                row_list = [literal_eval(j) for j in line.split()]
                if len(row_list)!=0:
                    if len(row_list)==1:
                        # row_list contains index
                        columns = np.array(row_list) -1
                    else:
                        if type(row_list[1]) is int:
                            # line contains indices
                            columns = np.array(row_list) -1
                            
                        else:
                            # line contains elements
                            row = row_list[0] - 1
                            elements = np.array(row_list[1:]) 
                            
                            self.blocks[ gmap ][row, columns + bands[0]] = elements #row and column 
        return bands
                        
    def load_latmat(self, lmat, safemode = True):
        # Load same format used by old toeplitz class (latmat)
        #self.blocks = np.array(blockfile)
        
        self.coords = np.array(lmat.cc)
        self.blocks = []
        for c in self.coords:
            self.blocks.append(lmat.get(c))
        self.blocks = np.array(self.blocks, dtype = float)
        
        
        imax = np.abs(self.coords).max()
        
        self.domain = np.array([imax, imax, imax]) + 1 #add zero blocks along edges
        
        self.delim = self.dom2delim(self.domain)
        
        if safemode:
            
            self.test_data_types()
            
        self.blockshape = self.blocks[0].shape
        
        self.indices = self._c2i(self.coords)
        
        self.map_blocks()      
        
    def load(self, infile, safemode = False):
        # Load toeplitz matrix from infile
        
        coords, blocks, domain = np.load(infile, allow_pickle=True)
        
        # Screen out blocks with negligibly small
        # coefficients
        # NOTE: coincidence assumed
        self.coords, self.blocks = screen(coords, blocks,
                                          self.norm, self.tolerance)
        
        self.domain = \
            np.amax(np.absolute(self.coords), axis=0) + 1
        
        self.delim = self.dom2delim(self.domain)
        
        if safemode:
            
            self.test_data_types()
        
        self.blockshape = self.blocks[0].shape
        
        self.indices = self._c2i(self.coords)
        
        self.map_blocks()                                                                        
        
        
    # Indexing, memory and mapping
    
    def _c2i(self, c, delim = None, domain = None):
        '''
        coordinate to index transformation
        c      = Ncells*3 array containing cell indices
        delim  = optional delimiting array, default internal
        domain = optional domain size, defaults to internal 
        
        returns an Ncells*1 array of indices
        '''
        
        if domain is None:
            domain = self.domain
        if delim is None:
            delim = self.delim
            
        #note: clipping (np.clip) ensures zero blocks outside domain
        return np.dot(domain + np.clip(c, -self.domain, self.domain), delim)
        
    
    def _i2c(self,I, delim = None, domain = None):
        '''
        index to coordinate transformation
        I      = Ncells*1 array containing cell indices
        delim  = optional delimiting array, default internal
        domain = optional domain size, defaults to internal 
        
        returns an Ncells*3 array of coordinates
        '''
        
        if delim==None:
            delim=self.delim
        if domain==None:
            domain = self.domain
            
        m = np.zeros((I.size,3), dtype = 'int64')
        
        m[:,2] = I//delim[2]
        
        m[:,1] = (I - self.delim[2]*m[:,2])//delim[1]
        
        m[:,0] = I - m[:,1]*delim[1] - m[:,2]*delim[2]
        
        return m - domain
        
    def map_blocks(self):
        # Store "sequential address" of block to corresponding "mapped address"
        # So that a lookop in mapping will return address of block in the sequential (and "dense") storage
        if self.zero_padded:
            self.zero_block = self.blocks.shape[0] - 1
        else:
            self.zero_block = self.blocks.shape[0] #a block outside the dense
            
            self.blocks = np.append(self.blocks,
                                    np.zeros(self.blockshape,
                                             dtype = self.blocks.dtype))
            self.zero_padded = True
            
        # new_shape = (self.coords.shape[0] + 1,
        #              self.blockshape[0],
        #              self.blockshape[1])
        self.blocks = self.blocks.reshape(self.coords.shape[0] + 1,
                                          self.blockshape[0],
                                          self.blockshape[1])
        
        mapsize = np.dot(2*self.domain + 1, self.delim)
        
        self.mapping = np.ones(mapsize, dtype = int) * \
                       self.zero_block
                       
        self.mapping[self.indices] = range(len(self.coords))
        
    # Various
    
    def permute_blocks(self, perm_pattern, axis = 0):
        if axis == 0:
            for i in np.arange(self.blocks.shape[0]-1):
                self.blocks[i] = self.blocks[i,perm_pattern,:]
        if axis == 1:
            for i in np.arange(self.blocks.shape[0]-1):
                self.blocks[i] = self.blocks[i,:,perm_pattern]
        if axis == 10:
            for i in np.arange(self.blocks.shape[0]-1):
                #both
                self.blocks[i] = self.blocks[i,:,perm_pattern][:,perm_pattern,:]



    
    # Block access, interfacing 

    def redefine_supercell(self, n_layers, n_s_layers):
        '''
        Unfold/recast/repartition cells onto larger supercells
        for the gamma-point fitting
        better explanation will follow
        '''
        c0 = lattice_coords(n_layers) 

        cm = lattice_coords(n_s_layers)

        newblocks = np.zeros((cm.shape[0], c0.shape[0]*self.blocks.shape[1],c0.shape[0]*self.blocks.shape[1]), dtype = type(self.blocks))

        for c in np.arange(cm.shape[0]):
            ct = np.array(2*n_layers+1) * cm[c]

            newblocks[c] = self.tofull(self, c0 + cm, c0 + cm) #.reshape()
        
        ret = tmat()
        ret.load_nparray(newblocks, cm)
        return ret






    def tofull(self, m, coords_L, coords_M):
        '''
        Unfold the periodic symmetry 
        '''
        ret = np.zeros((m.blockshape[0]*coords_L.shape[0], m.blockshape[1]*coords_M.shape[0]), dtype = float)
        for L in range(coords_L.shape[0]):
            for M in range(coords_M.shape[0]):
                ret[L*m.blockshape[0]:(L+1)*m.blockshape[0], M*m.blockshape[1]:(M+1)*m.blockshape[1]] = m.cget(coords_M[M] - coords_L[L])
        return ret
    
    def get(self, c):
        '''
        Returns a single block corresponding to the cell 
        coordinates or cell index 'c'.
        '''
        if type(c) is list:
            c = np.array(c)
        
        if type(c) is np.ndarray:
            return self.cget(c)
        if type(c) is np.int64:
            return self.iget(c)

    def blockslice(self, range_x, range_y):

        coords = self.coords
        blocks = self.blocks[:-1, range_x[0]:range_x[-1], range_y[0]:range_y[-1]]
        
        ret = tmat()
        ret.load_nparray(blocks, coords)
 
        
        return ret
        
    def get_numpymatrix(self, coords1, coords2, rows, columns):
        '''
        Return a two-dimensional Numpy array. 

        coords1   : Cell coordinates for the first dimension.
        coords2   : Cell coordinates for the second dimension.
        rows      : Rows of the constructed matrix 'mfull' 
                    to be returned.
        columns   : Columns of the constructed matrix 'mfull'
                    to be returned.
        '''
        
        # An array containing
        #
        # |  coords2[0, :] - coords1[0, :]   | coords2[0, :] - coords1[1, :]   | coords2[0, :] - coords1[2, :]   |  ...  |  coords2[0, :] - coords1[-1, :]   |
        # |                                  |                                 |                                 |       |                                   |
        # |  coords2[1, :] - coords1[0, :]   | coords2[1, :] - coords1[1, :]   | coords2[1, :] - coords1[2, :]   |  ...  |  coords2[1, :] - coords1[-1, :]   |
        # |                                  |                                 |                                 |       |                                   |
        # |  coords2[2, :] - coords1[0, :]   | coords2[2, :] - coords1[1, :]   | coords2[2, :] - coords1[2, :]   |  ...  |  coords2[2, :] - coords1[-1, :]   |
        # |                                  |                                 |                                 |       |                                   |
        # |  ...                             | ...                             | ...                             |  ...  |  ...                              |
        # |  coords2[-1, :] - coords1[0, :]  | coords2[-1, :] - coords1[1, :]  | coords2[-1, :] - coords1[2, :]  |  ...  |  coords2[-1, :] - coords1[-1, :]  |
        coords = (coords2[None, :] - coords1[:, None])
        # Get storage indices corresponding to 'coords'
        indx = self.mapping[ self._c2i(coords) ]
        
        # Set up an array constructed of the following blocks:
        #
        # |  self.get(coords2[0, :] - coords1[0, :])   |  self.get(coords2[0, :] - coords1[1, :])   |  self.get(coords2[0, :] - coords1[2, :])   |  ...  |  self.get(coords2[0, :] - coords1[-1, :])   |
        # |                                            |                                            |                                            |                                                     |
        # |  self.get(coords2[1, :] - coords1[0, :])   |  self.get(coords2[1, :] - coords1[1, :])   |  self.get(coords2[1, :] - coords1[2, :])   |  ...  |  self.get(coords2[1, :] - coords1[-1, :])   |
        # |                                            |                                            |                                            |                                                     |
        # |  self.get(coords2[2, :] - coords1[0, :])   |  self.get(coords2[2, :] - coords1[1, :])   |  self.get(coords2[2, :] - coords1[2, :])   |  ...  |  self.get(coords2[2, :] - coords1[-1, :])   |
        # |                                            |                                            |                                            |                                                     |
        # |  ...                                       |  ...                                       |  ...                                       |  ...  |  ...                                        |
        # |                                            |                                            |                                            |                                                     |
        # |  self.get(coords2[-1, :] - coords1[0, :])  |  self.get(coords2[-1, :]) - coords1[1, :]) |  self.get(coords2[-1, :] - coords1[2, :])  |  ...  |  self.get(coords2[-1, :] - coords1[-1, :])  |
        #
        mfull = np.swapaxes(self.blocks[ indx ],
                            1, 2).reshape(coords1.shape[0] *
                                          self.blockshape[0],
                                          coords2.shape[0] *
                                          self.blockshape[1])
        
        # Return only elements corresponding to 'rows' and
        # 'columns' from the 'mfull' matrix
        return mfull[rows, :][:, columns]
    
    def cget(self, c):
        # "Coordinate get"
        # Return block(s) at coordinate c
        # Will work for both c array of coords or single coordinate
        return self.blocks[ self.mapping[self._c2i(c)] ]
    
    def iget(self, i):
        # "Index get"
        # Return block(s) at index i
        return self.blocks[self.mapping[i]]
    
    def cset(self, c, block):
        # Set block(s) in self at coordinate c
        assert(self.zero_block != self.mapping[ self._c2i(c) ]), "Cannot set unallocated blocks"
        self.blocks[ self.mapping[ self._c2i(c) ] ] = block
        
    def iset(self, i, block):
        assert(self.zero_block != self.mapping[ i ]), "Cannot set unallocated blocks"
        self.blocks[ self.mapping[ i ] ] = block

    def cset_element(self, coords, i1, i2, value):
        
        block = deepcopy(self.cget(coords))
        block[i1, i2] = value
        
        self.cset(coords, block)
        
    def get_elements(self, coords, rows, columns):
        '''
        Return the elements corresponding to the cell, row, 
        and columns indices in 'coords', 'rows', and 'columns'. 
        
        Each element 'i' the output array corresponds to
        
                    cell_id = coords[i]
                    row     = rows[i]
                    column  = columns[i]
        '''
        m = 'The parameter "coords" must be a ' + \
            'two-dimensional Numpy array.'
        assert type(coords) == np.ndarray and \
            len(coords.shape) == 2 and \
            coords.shape[1] == 3, m
        m = 'The parameters "rows", and "columns" ' + \
            'must be one-dimensional Numpy arrays'
        assert type(rows) == np.ndarray and \
            len(rows.shape) == 1, m
        assert type(columns) == np.ndarray and \
            len(columns.shape) == 1, m
        assert (coords.shape[0] == rows.shape[0]) and \
            (coords.shape[0] == columns.shape[0]), m

        n = coords.shape[0]
        elems = np.zeros(n)

        for i in range(n):
            block = self.get(coords[i])

            elems[i] = block[rows[i], columns[i]]

        return elems

    
    # Algebraic manipulations and arithmetics
    
    def t(self):
        # Blockwise transpose, currently inefficient
        ret = deepcopy(self)
        ret.blockshape = (self.blockshape[1],
                          self.blockshape[0])
        ret.blocks = np.einsum("abc->acb", ret.blocks)
        
        return ret
    
    def T(self):
        # Toeplitz transpose
        ret = deepcopy(self)
        ret.coords *= -1
        ret.indices = self._c2i(ret.coords)
        ret.mapping[ret.indices] = np.arange(len(ret.coords))
        return ret
    
    def tT(self):
        ret = deepcopy(self)
        ret.blockshape = (self.blockshape[1],
                          self.blockshape[0])
        ret.coords *= -1
        ret.indices = self._c2i(ret.coords)
        ret.blocks = np.einsum("abc->acb", ret.blocks)
        ret.mapping[ret.indices] = np.arange(len(ret.coords))
        return ret
    
    def is_hermitian(self, tolerance=1e-14):
        return self.equal_to(self.tT(), tolerance=tolerance)
    
    def align(self, other):
        if not np.all(self.domain == other.domain):
            # Impose consistent indexing in both tmat objects
            new_imax = np.max(np.array([self.domain, other.domain]))
            new_domain = np.array([new_imax, new_imax, new_imax])
            self.expand(new_domain)
            other.expand(new_domain)
            
        assert(np.all(self.delim == other.delim)), "domain mismatch"

    def fft(self, n_layers = None):
        if n_layers is None:
            n_layers = np.max(np.abs(self.coords), axis = 0)
            

        nx,ny,nz = 2*n_layers + 1
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]
        
        coords = np.roll(lattice_coords(n_layers).reshape(nx,ny,nz, 3), -n_layers, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        
        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        M1 = np.fft.fftn(m1r, axes = (0,1,2))

        
        ret_fft = tmat()
        ret_fft.load_nparray(M1.reshape(coords.shape[0], m1x,m1y), coords, safemode = False)
        return ret_fft

    def inv(self, n_layers = None):
        if n_layers is None:
            n_layers = np.max(np.abs(self.coords), axis = 0)
            

        nx,ny,nz = 2*n_layers + 1
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]
        
        coords = np.roll(lattice_coords(n_layers).reshape(nx,ny,nz, 3), -n_layers, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        
        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        M1 = np.fft.fftn(m1r, axes = (0,1,2))

        cn = np.zeros((coords.shape[0], 6), dtype = np.complex128)


        

        M_inv = np.zeros((nx,ny,nz,m1x, m1y),dtype = np.complex128)

        

        for c in coords:
            M_inv[c[0], c[1], c[2]] = np.linalg.pinv(M1[c[0], c[1], c[2]])
            #mb, me, ms = np.linalg.svd(M1[c[0], c[1], c[2]])

            #cn[ic, np.array([0, 1])] = w.max(),w.min()
            #cn[ic, np.array([2, 3])] = me.max(), me.min()
            #cn[ic, 4] = np.linalg.det(M1[c[0], c[1], c[2]])
            #cn[ic, 5] = np.abs(np.dot(np.linalg.inv(M1[c[0], c[1], c[2]]),M1[c[0], c[1], c[2]])-np.eye(m1x)).max()
            #ic += 1
        ret_inv = tmat()
        ret_inv.load_nparray(np.fft.ifftn(M_inv.reshape(nx,ny,nz,m1x,m1y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m1y), coords, safemode = False)
        return ret_inv
        
        
    
    def inv_(self, n_layers = None):
        # BT inversion through FFT - blockwise inversion - IFFT
        # Mathematical details may be found in notes
        n_points = np.array(n_lattice(self))
        JKk = transform(self, np.fft.fftn, n_points = n_points)
        JKk_inv = JKk*1.0

        #for k in JKk.coords:
        #    JKk_k = np.linalg.inv(JKk.cget(k))
        #    
        #    JKk_inv.cset(k, JKk_k)
        #JK_inv.blocks[:-1] = np.linalg.pinv(JKk.blocks)

        Jkb = np.linalg.pinv(JKk.cget(JKk.coords))

        ret = tmat()
        ret.load_nparray(Jkb, JKk.coords, safemode = False) #avoid complex complaint


        ret = transform(ret, np.fft.ifftn, n_points = n_points, complx = False)

        return ret 

    def check_condition(self, n_layers = None):
        if n_layers is None:
            n_layers = np.max(np.abs(self.coords), axis = 0)
            

        nx,ny,nz = 2*n_layers + 1
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]
        
        coords = np.roll(lattice_coords(n_layers).reshape(nx,ny,nz, 3), -n_layers, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        
        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        M1 = np.fft.fftn(m1r, axes = (0,1,2))

        cn = np.zeros((coords.shape[0], 6), dtype = np.complex128)


        ic = 0

        Mv = np.zeros((nx,ny,nz,m1x, m1y),dtype = np.complex128)

        Md = np.zeros((nx,ny,nz,m1x, m1y),dtype = np.complex128)


        for c in coords:
            w,v = np.linalg.eig(M1[c[0], c[1], c[2]])
            mb, me, ms = np.linalg.svd(M1[c[0], c[1], c[2]])

            cn[ic, np.array([0, 1])] = w.max(),w.min()
            cn[ic, np.array([2, 3])] = me.max(), me.min()
            cn[ic, 4] = np.linalg.det(M1[c[0], c[1], c[2]])
            cn[ic, 5] = np.abs(np.dot(np.linalg.inv(M1[c[0], c[1], c[2]]),M1[c[0], c[1], c[2]])-np.eye(m1x)).max()
            ic += 1

            #sk = np.linalg.abs(np.dot(np.linalg.inv(M1[c[0], c[1], c[2]]),M1[c[0], c[1], c[2]])-np.eye(m1x)).max()
            #print("Condition at k = ",c, ", (max/min) : ", w.max()/w.min(),w.max(),w.min())

            #mb, me, ms = np.linalg.svd(M1[c[0], c[1], c[2]])
            #print("Largest/smallest singular value: ", me.max()/me.min(), me.max(), me.min())
            #print("Determinant:", np.linalg.det(M1[c[0], c[1], c[2]]))
        

        

            Mv[c[0], c[1], c[2]] = np.diag(w) #np.linalg.solve(SVH, Ub)
            Md[c[0], c[1], c[2]] = np.diag(me) #np.linalg.solve(SVH, Ub)


        print("_________________________________________________________")
        print("Testing matrix condition ( imaginary part ignored )")
        print("             eig (min/max) |  singular (min/max) |  determinant   | A^{-1}A - I (abs.max)")
        print("Gamma point: %.4e       %.4e            %.4e      %.4e" % (cn[0,0].real/cn[0,1].real, cn[0,2].real/cn[0,3].real, cn[0,4].real,cn[0,5].real))
        print("Kspace max : %.4e       %.4e            %.4e      %.4e" % ((cn[:,0].real/cn[:,1].real).max(), (cn[:,2].real/cn[:,3].real).max(), cn[:,4].real.max(), cn[:,5].real.max()))
        print("Kspace min : %.4e       %.4e            %.4e      %.4e" % ((cn[:,0].real/cn[:,1].real).min(), (cn[:,2].real/cn[:,3].real).min(), cn[:,4].real.min(), cn[:,5].real.min()))
        #print("Imaginary conditioning report ( eigenvalues / singular values / determinant )")
        #print("Gamma point:", cn[0,0].imag/cn[0,1].imag, "/", cn[0,2].real/cn[0,3].imag, "/", cn[0,4])
        #print("Kspace max :", (cn[:,0].imag/cn[:,1].imag).max(), "/", (cn[:,2].imag/cn[:,3].imag).max(), "/", cn[:,4].max())
        #print("Kspace min :", (cn[:,0].imag/cn[:,1].imag).min(), "/", (cn[:,2].imag/cn[:,3].imag).min(), "/", cn[:,4].min())

        
        #ret_v = tmat()
        #ret_v.load_nparray(np.fft.ifftn(Mv.reshape(nx,ny,nz,m1x,m1y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m1y), coords, safemode = False)
        
        #ret_d = tmat()
        #ret_d.load_nparray(np.fft.ifftn(Md.reshape(nx,ny,nz,m1x,m1y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m1y), coords, safemode = False)

        #print(np.diag(ret_v.cget([0,0,0])))
        #print(np.diag(ret_d.cget([0,0,0])))
        #print("Total condition, eigenvalues    :", ret_v.blocks[:-1, np.arange(m1x),np.arange(m1x)].max()/ret_v.blocks[:-1, np.arange(m1x),np.arange(m1x)].min() )
        #print("Total condition, singular values:", ret_d.blocks[:-1, np.arange(m1x),np.arange(m1x)].max()/ret_d.blocks[:-1, np.arange(m1x),np.arange(m1x)].min() )
        print("_________________________________________________________")
        
        
        




    
    def __pow__(self, other):
        ret = deepcopy(self)
        ret.blocks = ret.blocks**other
        return ret
        
    def __eq__(self, other):
        # Returns the max deviation
        return np.abs(self.cget(self.coords) - other.cget(self.coords)).max()
    
    def __add__(self, other):
        # Add elements in matrix with other
        # If other == tmat -> elementwise
        # else add other to every element of self
        if type(other) == tmat:
            return self._sum( other, scalar1 = 1, scalar2 = 1)
        else:
            #Assume int or float
            ret = deepcopy(self)
            ret.blocks[:-1] = self.blocks[:-1]+other
            return ret
    
    def __sub__(self, other):
        # Subtract elements in matrix with other
        # See "Add" for additional details
        return self._sum(other, scalar1 = 1, scalar2 = -1)

    def _sum(self, other, scalar1 = 1, scalar2 = 1, safemode = True):
        ret = tmat()

        coords_extra = c2_not_in_c1(self.coords, other.coords, sort = True)
        coords = np.zeros((self.coords.shape[0]+coords_extra.shape[0], 3), dtype = int)
        coords[:self.coords.shape[0]] = self.coords
        coords[self.coords.shape[0]:] = coords_extra

        blocks = scalar1*self.cget(coords) + scalar2*other.cget(coords)

        ret.load_nparray(blocks, coords)
        return ret


    
    def __sum(self, other, scalar1 = 1, scalar2 = 1, safemode = True):
        # Summation of Toeplitz matrices
        if safemode:
            self.align(other) #make sure indexing is consistent
        ret_indices = np.unique(np.append(self.indices, other.indices)) #should perhaps be used carefully?
        
        
        #domain remains unchanged
        ret = tmat()
        ret.coords  = self._i2c(ret_indices)
        ret.indices = ret_indices
        ret.domain =  self.domain 
        ret.delim = ret.dom2delim(ret.domain)
        ret.blockshape = (self.blockshape[0], self.blockshape[1])
        
        # Sum is performed here
        
        ret.blocks = scalar1*self.get(ret.coords) + scalar2*other.get(ret.coords)
        
        ret.map_blocks()
        return ret
    
    def __mod__(self, other):
        # Hadamard product for tmat * tmat (elemetwise multiplication)
        assert(type(other)==tmat), "Hadamard product only defined for toeplitz-toeplitz multipication"
        
        self.align(other) #make sure indexing is consistent
        ret_indices = np.unique(np.append(self.indices, other.indices)) #should perhaps be used carefully?
        
        #domain remains unchanged
        ret = tmat()
        ret.coords  = self._i2c(ret_indices)
        ret.indices = ret_indices
        ret.domain =  self.domain 
        ret.delim = ret.dom2delim(ret.domain)
        ret.blockshape = (self.blockshape[0], self.blockshape[1])
        
        # Product is computed here
        
        ret.blocks = self.get(ret.coords)*other.get(ret.coords)
        #ret.blocks = self.blocks[:-1][ self.mapping[ret_indices] ] *other.blocks[:-1][ other.mapping[ret_indices] ]
        
        ret.map_blocks()
        return ret
        
    def __truediv__(self, other):
        # Division (by float)
        return self.__mul__(1.0/other) 
    
    def __mul__(self, other):
        # Multiplication of self by other
        if type(other) == tmat:
            return self.cdot(other)
            
        else:
            #Assume int or float
            ret = copy.copy(self)
            ret.blocks = self.blocks*other
            return ret
    
    def cscale(self, component_n, R = np.array([[1,0,0],[0,1,0],[0,0,1]]), exponent = 1):
        # coordinate scaling
        # for every block c_1,c_2,c_3, scale with component n \in {1,2,3}:
        # A_{c1,c2,c3} <=  A_{c1,c2,c3}* [(c1,c2,c3)*(R1,R2,R3)]_n
        # R = lattice vectors
        
        ret = deepcopy(self)
        
        for c in np.arange(ret.coords.shape[0]):
            ret.blocks[c] = ret.blocks[c]*np.dot(ret.coords[c], R)[component_n]**exponent
            
        return ret
    
    #@numba.autojit()
    def cdot(self, other, coords = None, safemode = True):
        '''
        Toeplitz-toeplitz matrix product
        Compute blocks at "coords" in return-product 
        
        This version should be used for all kinds of Toeplitz 
        matrix multiplications.
        
        The matrix-matrix multiplication is implemented as

          C(L) = \sum_M A^M * B^L - M)
        
        where L and M are cell coordinates. Here, the definition
        
          C^{L2 - L1} \equiv \overline{C}^(L1, L2) 
        
        is used.
        '''
        
        # Ensure consistent domains (to simplify indexing)
        if safemode:
            self.align(other) #make sure indexing is consistent
            
        if coords is None:
            #coords = np.array([[0,0,0]])
            ret = get_zero_tmat(cutoffs = self.domain-1 + other.domain-1,
                                blockshape = (self.blockshape[0],
                                              other.blockshape[1]))    
        else:
            if type(coords) is list:
                coords = np.array(coords, dtype = int)
                
            assert(type(coords) is np.ndarray), \
                "Coords should be provided as array (or list) or Nonetype."
            
            ret = setup_zero_tmat(coords,
                                  blockshape = (self.blockshape[0],
                                                other.blockshape[1]))
            
        ret_indx = ret._c2i(ret.coords)
        
        
        # Determine blocks-coords in product matrix
        
        index_a = self.indices
        index_b = other.indices
        
        
        
        for si in range(self.coords.shape[0]):
            scoords = self.coords[si]
            sblock  = self.blocks[si]
            # determine new 
            prod_coords_flat = scoords + other.coords #.T
            
            prod_indx_flat = ret._c2i(prod_coords_flat) 
            
            block_indices = np.in1d(prod_indx_flat, ret_indx) #pick only blocks allocated in ret
            
            oblocks = other.blocks[:-1][block_indices]
            
            
            
            
            
            sx,sy,sz = oblocks.shape #shape
            
            
            oblocks = oblocks.swapaxes(0,1)
            oblocks = oblocks.reshape(sy, sx*sz)
            
            Sy = self.blockshape[0]
            
            ret_addblocks = np.dot(self.blocks[si], oblocks).reshape(Sy,sx,sz)
            ret_addblocks = ret_addblocks.swapaxes(0,1)
            
            ret.blocks[ret.mapping[prod_indx_flat[block_indices]]] += ret_addblocks
            
        # Screen out negligible blocks
        
        ret = screen_tmat(ret, ret.tolerance)
        #ret.coords, ret.blocks = \
        #    screen(ret.coords, ret.blocks, ret.norm,
        #           ret.tolerance)
        
        
        """
        # Update the object after the screening
        ret.zero_padded = False
        ret.domain = \
            np.amax(np.absolute(ret.coords), axis=0) + 1
        ret.delim = ret.dom2delim(ret.domain)    
        ret.blockshape = ret.blocks[0].shape
        ret.indices = ret._c2i(ret.coords)
        ret.map_blocks()
        """
        
        return ret
    
    #@numba.autojit()   
    def dot(self, other, coords = None, safemode = True):
        '''
        Toeplitz matrix product
        
        NOTE! The function cdot() is fastest for all kinds of
        Toeplitz matrix-matrix multiplications,
        and should therefore be used.
        
        '''
        print('WARNING! The function dot() in the tmat class may have bugs.')
        
        # Ensure consistent domains (to simplify indexing)
        if safemode:
            self.align(other) #make sure indexing is consistent
            
            
        # Determine blocks-coords in product matrix
        
        index_a = self.indices
        index_b = other.indices
        prod_coords = self.coords[:, None] + other.coords[None, :]
        
        # Compute product matrix

        b_blocks = \
            np.swapaxes(other.blocks[:-1], 0, 1).reshape(other.blockshape[0],
                                                         index_b.shape[0] *
                                                         other.blockshape[1])        
        ret = tmat()
        
        ret.domain = self.domain + other.domain
        ret.delim = ret.dom2delim(ret.domain)
        
        ret.indices = np.unique(ret._c2i(prod_coords.reshape(index_a.shape[0]*index_b.shape[0], 3)))
        
        ret.coords = ret._i2c(ret.indices)
        
        ret.blockshape = (self.blockshape[0], other.blockshape[1])
        
        ret.blocks = np.zeros((ret.indices.shape[0], self.blockshape[0], other.blockshape[1]), dtype = float)
        
        ret.map_blocks()
        
        
        #sum blocks to their proper index, this should be further optimized
        nrows, ncols = ret.blockshape
        
        
        prod_blocks = \
            np.dot(self.blocks[:-1].reshape(index_a.shape[0] *
                                            self.blockshape[0],
                                            self.blockshape[1]),
                   b_blocks)
        
        for i in np.arange(index_a.shape[0]):
            for j in np.arange(index_b.shape[0]):
                bb = prod_blocks[nrows*i:nrows*(i+1), ncols*j:ncols*(j+1)]
                c1 = self.coords[i]
                c2 = other.coords[j]
                ret.blocks[ ret.mapping[ ret._c2i(c1+c2) ] ] += bb
        
        return ret
        
    #@numba.autojit()
    def dot2(self, other, coords = None, safemode = True):
        '''
        Toeplitz matrix product 
        
        NOTE! The function cdot() is fastest for all kinds of
        Toeplitz matrix-matrix multiplications,
        and should therefore be used.
        '''
        # ret_  = return toeplitz matrix
        # prod_ = block products (not summed)
        print("Domains  (pre aling):", self.delim, other.delim)
        if safemode:
            self.align(other) #make sure indexing is consistent
        print("Domains (post align):", self.delim, other.delim)
        
        ret_domain = np.max(np.abs(self.coords), axis = 0) + np.max(np.abs(other.coords), axis = 0)
        ret_imax = ret_domain.max() 
        ret_domain = np.array([ret_imax, ret_imax, ret_imax])
        ret_delim = self.dom2delim(ret_domain)
        
        ret_blockshape = (self.blockshape[0], other.blockshape[1])        
        
        index_a = self.indices
        index_b = other.indices
        #print(self.delim, self.indices.min(), self.indices.max())
        prod_coords = self.coords[self.mapping[ index_a ],None] + other.coords[None,other.mapping[ index_b ] ]
        
        #prod_blocks_coords = prod_coords.reshape(index_a.shape[0]*index_b.shape[0], 3)
        
        #print("new coords")
        #print(prod_blocks_coords)
        
        #ret_index  = np.dot(prod_coords.reshape(index_a.shape[0]*index_b.shape[0], 3), ret_delim)
        ret_index = self._c2i(prod_coords.reshape(index_a.shape[0]*index_b.shape[0], 3), delim = ret_delim, domain = ret_domain)
        ret_index_unique = np.unique(ret_index)
        ret_blocks = np.zeros((ret_index_unique.shape[0], ret_blockshape[0], ret_blockshape[1]), dtype = 'float64')
        
        # Determine coordinates from return indices

        ret_coords = self._i2c(ret_index_unique, delim = ret_delim, domain = ret_domain)
        #ret_coords = np.zeros((ret_index_unique.shape[0], 3), dtype = 'int64')
        #ret_coords = self.
        #for c in np.arange(3):
        #    ret_coords[:,c] = ret_index_unique//ret_delim[c]
        #    ret_index_unique -= ret_coords[:,c]*ret_delim[c]
              


        #Remapping not 
        #b_blocks = np.einsum("ipq->piq", other.blocks[ other.mapping[index_b] ]).reshape(other.blockshape[0],index_b.shape[0]*other.blockshape[1])
       
        # Are these the correct blocks? Are they indexed properly?? 
        b_blocks = np.einsum("ipq->piq", other.blocks[:-1]).reshape(other.blockshape[0],index_b.shape[0]*other.blockshape[1])
        
        
        #Sequential elements
        
        prod_blocks = np.dot(self.blocks[:-1].reshape(index_a.shape[0]*self.blockshape[0], other.blockshape[1]), b_blocks)
        
        #prod_seq = prod_blocks.reshape(2, 2*index_a.shape[0]*index_b.shape[0])
        
        
        # setup return matrix
        
        ret = tmat()
        ret.domain = ret_domain
        ret.blockshape = ret_blockshape
        ret.delim = ret.dom2delim(ret_domain)
        ret.blocks= ret_blocks
        ret.coords= ret_coords
        ret.indices = ret._c2i(ret_coords)
        ret.map_blocks()
        
        c1 = ret_coords
        c2 = ret._i2c(ret._c2i(ret_coords))
        for i in np.arange(c1.shape[0]):
            if not np.all(c1[i]==c2[i]):
                print("Error in ",c1[i], c2[i])
                
                
        nrows, ncols = ret_blockshape
        
        for i in np.arange(index_a.shape[0]):
            for j in np.arange(index_b.shape[0]):
                bb = prod_blocks[nrows*i:nrows*(i+1), ncols*j:ncols*(j+1)]
                c1 = self.coords[i]
                c2 = other.coords[j]
                ret.blocks[ ret.mapping[ ret._c2i(c1+c2) ] ]+= bb
                
        return ret
    
    #def fft(self):

    #def kspace_svd_inv()
    def gamma_inv(self, n_points = None):
        if n_points is None:
            n_points = np.max(np.abs(self.coords), axis = 0)

        #print()

        nx,ny,nz = 2*n_points + 1
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]

        coords = np.roll(lattice_coords(n_points).reshape(nx,ny,nz, 3), -n_points, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        #m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        
       
        M3 = np.zeros((nx,ny,nz,m1x, m1y),dtype = np.complex128)
        M3[0,0,0] = np.linalg.pinv(np.sum(self.cget(self.coords)/(nx*ny*nz), axis = 0))

    
        ret = tmat()
        ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m1y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m1y), coords)
        return ret

    


    def kspace_svd(self):
        n_points = np.max(np.array([n_lattice(self)]), axis = 0)
        self_k = transform(self, np.fft.fftn, n_points = n_points)

        s = tmat()
        s.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        s.blocks*=0.0

        vh = tmat()
        vh.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        vh.blocks*=0.0

        u = tmat()
        u.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        u.blocks*=0.0

        for i in np.arange(len(self_k.blocks)-1):
            u_,s_,vh_ = np.linalg.svd(self_k.blocks[i])
            


            s.blocks[i] = np.diag(s_)
            vh.blocks[i] = vh_
            u.blocks[i] = u_


        u = transform(u, np.fft.ifftn, n_points = n_points, complx = False)
        vh = transform(vh, np.fft.ifftn, n_points = n_points, complx = False)
        s = transform(s, np.fft.ifftn, n_points = n_points, complx = False)

        #if thresh is not None:
        #    # remove 



        return u,s,vh

    def kspace_ldl(self):
        import scipy
        n_points = np.max(np.array([n_lattice(self)]), axis = 0)
        self_k = transform(self, np.fft.fftn, n_points = n_points)

        s = tmat()
        s.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        s.blocks*=0.0

        vh = tmat()
        vh.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        vh.blocks*=0.0

        #u = tmat()
        #u.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        #u.blocks*=0.0

        for i in np.arange(len(self_k.blocks)-1):
            l,d,m = scipy.linalg.ldl(self_k.blocks[i])
            


            s.blocks[i] = d
            vh.blocks[i] = l
            #u.blocks[i] = l#


        #u = transform(u, np.fft.ifftn, n_points = n_points, complx = False)
        vh = transform(vh, np.fft.ifftn, n_points = n_points, complx = False)
        s = transform(s, np.fft.ifftn, n_points = n_points, complx = False)

        #if thresh is not None:
        #    # remove 



        return s,vh

    def kspace_project_out(self, other):
        """
        project out other from self
        """

        n_points = np.max(np.array([n_lattice(self), n_lattice(other)]), axis = 0)
        self_k = transform(self, np.fft.fftn, n_points = n_points)
        other_k = transform(other, np.fft.fftn, n_points = n_points)

        for i in np.arange(len(self_k.blocks)-1):
            self_k.blocks[i] -= other_k.blocks[i]

    def kspace_svd_lowdin_(self, screening_threshold = 1e-10):
        # construct self^-.5
        n_points = np.max(np.array([n_lattice(self)]), axis = 0)
        self_k = transform(self, np.fft.fftn, n_points = n_points)
        #print(n_points)

        u = tmat()
        u.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        u.blocks*=0.0

        for i in np.arange(len(self_k.blocks)-1):
            u_,s_,vh_ = np.linalg.svd(self_k.blocks[i])

            screen = s_**-.5>=screening_threshold
            #print(screen)
            


            u.blocks[i] = np.dot(vh_.T.conj()[:, screen], np.dot(np.diag(s_[screen]**-.5), u_.T.conj())[screen,:])

        u = transform(u, np.fft.ifftn, n_points = n_points, complx = False)
        return u
    
    def kspace_svd_lowdin(self, n_points = None, tolerance = 1e-10):
        # construct self^-.5

        if n_points is None:
            n_points = np.max(np.abs(self.coords), axis = 0)

        #print()

        nx,ny,nz = 2*n_points + 1
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]

        coords = np.roll(lattice_coords(n_points).reshape(nx,ny,nz, 3), -n_points, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        
        M1 = np.fft.fftn(m1r, axes = (0,1,2))
        M3 = np.zeros((nx,ny,nz,m1x, m1y),dtype = np.complex128)


        for c in coords:
            u_,s_,vh_ = np.linalg.svd(M1[c[0], c[1], c[2]])
            b = M1[c[0], c[1], c[2]]

            t = s_>tolerance #screening
            if np.any(t == False):
                print("Warning (SVD): poorly conditioned JK matrix (singular values).") #, s_)

            #u.blocks[i] = np.dot(vh_.T.conj()[:, screen], np.dot(np.diag(s_[screen]**-.5), u_.T.conj())[screen,:])
                
            pinv = np.dot(vh_.T.conj()[:, t], np.dot(np.diag(s_[t]**-.5), u_.T.conj())[t,:])
            #pinv = np.dot(vh_[t,:].conj().T, np.dot(np.diag(s_[t]**-.5), u_[:,t].conj().T))
            #x = np.dot(pinv, b)

            M3[c[0], c[1], c[2]] = pinv #x
        

        

        ret = tmat()
        ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m1y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m1y), coords)
        return ret

        


    def kspace_svd_solve(self, other, rcond = 1e-14, complex_precision = np.complex128, n_points = None, complx = False):
        """
        IBT (Infinite Block-Circulant) svd solver
        """
        if n_points is None:
            
            m1n = np.max(np.abs(self.coords), axis = 0)
            m2n = np.max(np.abs(other.coords), axis = 0)
            
            n_points = np.max([m1n,m2n], axis = 0)  

        nx,ny,nz = 2*n_points + 1

        #print("kspace_svd_solve: ", nx,ny,nz)
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]
        m2x,m2y = other.blocks.shape[1], other.blocks.shape[2]
        
        coords = np.roll(lattice_coords(n_points).reshape(nx,ny,nz, 3), -n_points, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        #print(coords)

        
        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        m2r = other.cget(coords).reshape(nx,ny,nz,m2x,m2y)
        M1 = np.fft.fftn(m1r, axes = (0,1,2))
        M2 = np.fft.fftn(m2r, axes = (0,1,2))
        M3 = np.zeros((nx,ny,nz,m1x, m2y),dtype = np.complex128)


        for c in coords:
            #M1[c[0], c[1], c[2]] = np.dot(M1[c[0], c[1], c[2]], M2[c[0], c[1], c[2]])

            #ci = self_k.coords[i]
            u_,s_,vh_ = np.linalg.svd(M1[c[0], c[1], c[2]])
            b = M2[c[0], c[1], c[2]]

            t = np.abs(s_)>(rcond*np.max(np.abs(s_))) #screening

            #if s_.max()/s_.min()>10e6:
            #    print("poor condition in singular values:", c, s_, t)
            
            



            pinv = np.dot(vh_[t,:].conj().T, np.dot(np.diag(s_[t]**-1), u_[:,t].conj().T))
            x = np.dot(pinv, b)

            if np.any(t == False):
                print("Warning (SVD): poorly conditioned JK matrix (singular values).") #, s_)
                print("Eq. :", np.max(np.abs(np.dot(M1[c[0], c[1], c[2]], x)-b)))



            #rhs = np.dot(np.dot(np.diag(s_[t]**-1), u_[:,t].conj().T), b)
            #x = np.linalg.solve(vh_[t,:], rhs)
            
            
            
            #svhx = np.linalg.solve(u_[:,t], b)

            #x = np.linalg.solve(np.dot(np.diag(s_[t]),vh_[t,:]), svhx )

            
            """
            
            U = u_[:,t]
            S = np.diag(s_[t])
            VH = vh_[t,:]

            SVH = np.dot(S, VH)
            Ub  = np.dot(U.conj().T, other_k.blocks[i])
            """

            M3[c[0], c[1], c[2]] = x #np.linalg.solve(SVH, Ub)


            
        
        
        ret = tmat()
        if complx:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m2y), coords)
        else:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m2y), coords)
        return ret

    def kspace_cholesky_solve(self, other, complx = False, n_points = None):
        """
        # Solve linear system in reciprocal space
        #
        #     self \cdot x = other
        # First let self = M M^T, so that
        #    M M^T \cdot x = other
        # Then solve for y and x
        #    (1)   M \cdot y = other 
        #    (2) M^T \cdot x = y
        # return x
        """
        if n_points is None:
            
            m1n = np.max(np.abs(self.coords), axis = 0)
            m2n = np.max(np.abs(other.coords), axis = 0)
            
            n_points = np.max([m1n,m2n], axis = 0)  

        nx,ny,nz = 2*n_points + 1

        #print("kspace_svd_solve: ", nx,ny,nz)
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]
        m2x,m2y = other.blocks.shape[1], other.blocks.shape[2]
        
        coords = np.roll(lattice_coords(n_points).reshape(nx,ny,nz, 3), -n_points, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        #print(coords)

        
        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        m2r = other.cget(coords).reshape(nx,ny,nz,m2x,m2y)
        M1 = np.fft.fftn(m1r, axes = (0,1,2))
        M2 = np.fft.fftn(m2r, axes = (0,1,2))
        M3 = np.zeros((nx,ny,nz,m1x, m2y),dtype = np.complex128)

        for c in coords:
            
            Mk = np.linalg.cholesky(M1[c[0], c[1], c[2]])
            yk = np.linalg.solve(Mk, M2[c[0], c[1], c[2]])



            

            M3[c[0], c[1], c[2]] = np.linalg.solve(Mk.conj().T, yk)



        

        ret = tmat()
        if complx:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m2y), coords)
        else:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m2y), coords)
        return ret


    

    def kspace_cholesky(self):
        """
        # 
        """
        n_points = n_lattice(self)
        self_k = transform(self, np.fft.fftn, n_points = n_points)


        ret = tmat()
        ret.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        #ret = self_k*1.0
        ret.blocks*=0.0

        #ret.blocks[:-1] = np.einsum("ijk,ikl->ijl", self_k.blocks[:-1], other_k.blocks[:-1], optimize = True)

        for i in np.arange(len(self_k.blocks)-1):
            ret.blocks[i] = np.linalg.cholesky(self_k.blocks[i])

        ret = transform(ret, np.fft.ifftn, n_points = n_points, complx = False)
        return ret

    def kspace_cholesky_solve_(self, other):
        """
        # Solve linear system in reciprocal space
        #
        #     self \cdot x = other
        # First let self = M M^T, so that
        #    M M^T \cdot x = other
        # Then solve for y and x
        #    (1)   M \cdot y = other 
        #    (2) M^T \cdot x = y
        # return x
        """
        n_points = np.max(np.array([n_lattice(self), n_lattice(other)]), axis = 0)
        self_k = transform(self, np.fft.fftn, n_points = n_points)
        other_k = transform(other, np.fft.fftn, n_points = n_points)

        ret = tmat()
        ret.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], other_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        #ret = self_k*1.0
        ret.blocks*=0.0

        #ret.blocks[:-1] = np.einsum("ijk,ikl->ijl", self_k.blocks[:-1], other_k.blocks[:-1], optimize = True)

        for i in np.arange(len(self_k.blocks)-1):
            
            #print(np.max(np.abs(self_k.blocks[i].T-self_k.blocks[i])))
            #assert(np.max(np.abs(self_k.blocks[i].T-self_k.blocks[i]))<1e-10), "not symmetric"
            #assert(np.linalg.norm(self_k.blocks[i].T-self_k.blocks[i])<1e-10), "not symmetric"
            Mk = np.linalg.cholesky(self_k.blocks[i])
            yk = np.linalg.solve(Mk, other_k.blocks[i])


            ret.blocks[i] = np.linalg.solve(Mk.conj().T, yk)

        ret = transform(ret, np.fft.ifftn, n_points = n_points, complx = False)
        return ret

    def get_kspace_singular_values(self, n_points = None, tolerance = 1e-10, real = False, sort = True):
        # calculate band structure of matrix

        if n_points is None:
            n_points = np.max(np.abs(self.coords), axis = 0)

        #print()

        nx,ny,nz = 2*n_points + 1
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]

        coords = np.roll(lattice_coords(n_points).reshape(nx,ny,nz, 3), -n_points, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        
        M1 = np.fft.fftn(m1r, axes = (0,1,2))
        M3 = np.zeros((nx,ny,nz,m1x),dtype = np.complex128)
        #M4 = np.zeros((nx,ny,nz,m1x, m1y),dtype = np.complex128)


        for c in coords:
            u, svals, rh = np.linalg.svd(M1[c[0], c[1], c[2]])
            
            
            
            if sort:
                svals = np.sort(svals)

            M3[c[0], c[1], c[2]] = svals #x
            #M4[c[0], c[1], c[2]] = evecs #x

        
        return M3
    
    def get_kspace_eigenvalues(self, n_points = None, tolerance = 1e-10, real = False, sort = True):
        # calculate band structure of matrix

        if n_points is None:
            n_points = np.max(np.abs(self.coords), axis = 0)

        #print()

        nx,ny,nz = 2*n_points + 1
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]

        coords = np.roll(lattice_coords(n_points).reshape(nx,ny,nz, 3), -n_points, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        
        M1 = np.fft.fftn(m1r, axes = (0,1,2))
        M3 = np.zeros((nx,ny,nz,m1x),dtype = np.complex128)
        #M4 = np.zeros((nx,ny,nz,m1x, m1y),dtype = np.complex128)


        for c in coords:
            evals, evecs = np.linalg.eig(M1[c[0], c[1], c[2]])
            
            
            
            if sort:
                evals = np.sort(evals)

            M3[c[0], c[1], c[2]] = evals #x
            #M4[c[0], c[1], c[2]] = evecs #x

        
        return M3


    def kspace_eig(self, n_points = None, tolerance = 1e-10, real = False, sort = True):
        # construct self^-.5

        if n_points is None:
            n_points = np.max(np.abs(self.coords), axis = 0)

        #print()

        nx,ny,nz = 2*n_points + 1
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]

        coords = np.roll(lattice_coords(n_points).reshape(nx,ny,nz, 3), -n_points, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)
        
        M1 = np.fft.fftn(m1r, axes = (0,1,2))
        M3 = np.zeros((nx,ny,nz,m1x, m1y),dtype = np.complex128)
        M4 = np.zeros((nx,ny,nz,m1x, m1y),dtype = np.complex128)


        for c in coords:
            evals, evecs = np.linalg.eig(M1[c[0], c[1], c[2]])

            
            if sort:


                ni = np.argsort(evals)
            
            

                M3[c[0], c[1], c[2]] = np.diag(evals[ni]) #x
                M4[c[0], c[1], c[2]] = evecs[ni] #x
            else:
                M3[c[0], c[1], c[2]] = np.diag(evals) #x
                M4[c[0], c[1], c[2]] = evecs

        

        
        if real:
            ret_evals = tmat()
            ret_evals.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m1y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m1y), coords)


            ret_evecs = tmat()
            ret_evecs.load_nparray(np.fft.ifftn(M4.reshape(nx,ny,nz,m1x,m1y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m1y), coords)

            return ret_evals, ret_evecs
        
        else:
            ret_evals = tmat()
            ret_evals.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m1y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m1y), coords, safemode = False)


            ret_evecs = tmat()
            ret_evecs.load_nparray(np.fft.ifftn(M4.reshape(nx,ny,nz,m1x,m1y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m1y), coords, safemode = False)

            return ret_evals, ret_evecs


    
    def kspace_eig_(self):
        n_points = n_lattice(self)
        self_k = transform(self, np.fft.fftn, n_points = n_points)

        ret_val = tmat()
        ret_val.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        ret_val.blocks*=0.0

        ret_vec = tmat()
        ret_vec.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], self_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        ret_vec.blocks*=0.0

        for i in np.arange(len(self_k.blocks)-1):
            evals, evecs = np.linalg.eig(self_k.blocks[i])
            ret_val.blocks[i] = np.diag(evals)
            ret_vec.blocks[i] = evecs
        
        ret_val = transform(ret_val, np.fft.ifftn, n_points = n_points, complx = False)
        ret_vec = transform(ret_vec, np.fft.ifftn, n_points = n_points, complx = False)
        
        return ret_val, ret_vec





    def kspace_linear_solve(self, other):
        """
        # Solve linear system in reciprocal space
        #     self \cdot x = other
        # returns x
        """
        n_points = np.max(np.array([n_lattice(self), n_lattice(other)]), axis = 0)
        self_k = transform(self, np.fft.fftn, n_points = n_points)
        other_k = transform(other, np.fft.fftn, n_points = n_points)

        ret = tmat()
        ret.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], other_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        #ret = self_k*1.0
        ret.blocks*=0.0

        #ret.blocks[:-1] = np.einsum("ijk,ikl->ijl", self_k.blocks[:-1], other_k.blocks[:-1], optimize = True)

        for i in np.arange(len(self_k.blocks)-1):
            ret.blocks[i] = np.linalg.solve(self_k.blocks[i],other_k.blocks[i])

        ret = transform(ret, np.fft.ifftn, n_points = n_points, complx = False)
        return ret
    def set_precision(self, precision):
        self.blocks = np.array(self.blocks, dtype = precision)

    def circulantdot_(self, other, complx = False, screening = None):
        """
        memory-easy circulant product
        """
        npt = np.max(np.array([n_lattice(self), n_lattice(other)]), axis = 0) 
        print(npt)
        mk = screen_tmat(transform(self, np.fft.fftn, n_points = npt))
        nk = screen_tmat(transform(other, np.fft.fftn, n_points = npt))
        #mk = transform(self, np.fft.fftn, n_points = npt)
        #nk = transform(other, np.fft.fftn, n_points = npt)
        c = c2_union_c1(mk.coords, nk.coords)
        mk = mk.cget(c) #overwrite deliberately to save mem
        nk = nk.cget(c) 
        rb = np.zeros((c.shape[0], mk.shape[1], nk.shape[2]), dtype = complex)
        for i in np.arange(c.shape[0]):
            rb[i] = np.dot(mk[i], nk[i])
        del(mk)
        del(nk)
        return screen_tmat(transform(tmat(c, rb), np.fft.ifftn, n_points = npt, complx = False))

    def kspace_transform(self, n_layers = None):
        if n_layers is None:
            n_layers = np.max(np.abs(self.coords), axis = 0)

        nx,ny,nz = 2*n_layers + 1
        
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]

        coords = np.roll(lattice_coords(n_layers).reshape(nx,ny,nz, 3), -n_layers, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        m1r = self.cget(coords).reshape(nx,ny,nz,m1x,m1y)

        return tmat(coords, np.fft.fftn(m1r, axes = (0,1,2)))

    
    def get_prepared_circulant_prod(self, n_layers = None, inv = False, rcond = 1e-10, inv_mod = "lpinv"):
        """
        Create an object primed for ciruclant matrix-matrix multiplication
        for efficient repeated multiplication with same object
        """
        
        return primed_for_dot(self,n_layers, inv, rcond = rcond, inv_mod = inv_mod)















        



    def circulantdot(self, other, n_layers = None, complx = False, precision = np.complex128):
        """
        IBT (Infinite Block-Circulant) matrix-matrix product
        """
        if n_layers is None:
            
            m1n = np.max(np.abs(self.coords), axis = 0)
            m2n = np.max(np.abs(other.coords), axis = 0)
            
            n_layers = np.max([m1n,m2n], axis = 0)  

        nx,ny,nz = 2*n_layers + 1
        m1x,m1y = self.blocks.shape[1], self.blocks.shape[2]
        m2x,m2y = other.blocks.shape[1], other.blocks.shape[2]
        
        coords = np.roll(lattice_coords(n_layers).reshape(nx,ny,nz, 3), -n_layers, axis = (0,1,2)).reshape(nx*ny*nz, 3)

        
        #m1r = 
        #m2r = 
        M1 = np.fft.fftn(np.array(self.cget(coords).reshape(nx,ny,nz,m1x,m1y), dtype = precision), axes = (0,1,2))
        M2 = np.fft.fftn(np.array(other.cget(coords).reshape(nx,ny,nz,m2x,m2y), dtype = precision), axes = (0,1,2))
        M3 = np.zeros((nx,ny,nz,m1x, m2y),dtype = precision)

        print("Circulant usage:", (M1.nbytes + M2.nbytes + M3.nbytes)*1e-6, "Mb.")
        print(M1.dtype, M2.dtype, M3.dtype)
        for c in coords:
            M3[c[0], c[1], c[2]] = np.dot(M1[c[0], c[1], c[2]], M2[c[0], c[1], c[2]])
        
        
        ret = tmat()
        if complx:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m2y), coords)
        else:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m2y), coords)
        return ret
        

    def circulantdot_(self, other, complx = False, screening = None):
        """
        Computes the dot product assuming a circulant matrix structure
        """

        n_points = np.max(np.array([n_lattice(self), n_lattice(other)]), axis = 0) 
        
        self_k = transform(self, np.fft.fftn, n_points = n_points) 
        #if screening is not None:
        #    self_k = 
        other_k = transform(other, np.fft.fftn, n_points = n_points) 

        
        ret = tmat()
        ret.load_nparray(np.ones((self_k.coords.shape[0],self_k.blockshape[0], other_k.blockshape[1]), dtype = np.complex), self_k.coords, safemode = False)
        #ret = self_k*1.0
        ret.blocks*=0.0

        #ret.blocks[:-1] = np.einsum("ijk,ikl->ijl", self_k.blocks[:-1], other_k.blocks[:-1], optimize = True)

        for i in np.arange(len(self_k.blocks)-1):
            ret.blocks[i] = np.dot(self_k.blocks[i],other_k.blocks[i])

        ret = transform(ret, np.fft.ifftn, n_points = n_points, complx = False)
        
        if screening is not None:
            result = tmat()
            mx = np.max(np.abs(ret.cget(ret.coords)), axis = (1,2))>screening
            result.load_nparray(ret.cget(ret.coords)[mx], ret.coords[mx])
            result.set_precision(self.blocks.dtype)
            #print(np.sum(mx), mx.shape)
            return result
        else:

            return ret



    
    def remove(self, slices, dimension):
        '''
        For each block, remove rows/columns/... specified by 
        the array 'slices' and axis 'dimension'.
        
        slices    : A one-dimensional Numpy array containing 
                    the indices of the rows/columns/etc. to 
                    be removed.
        dimension : Dimension along which elements are removed.
        '''
        m = 'Error. The array "slices" has too many elements.' 
        assert slices.shape[0] < \
            self.get([0, 0, 0]).shape[dimension], m
        
        new_blocks = []
        for c in self.coords:
            
            temp = deepcopy(self.get(c))
            elems = np.delete(temp, slices, dimension)
            
            new_blocks.append(deepcopy(elems.tolist()))
            
        # Return a new 'tmat' object in which rows/columns
        # have been removed
        return tmat(coords = self.coords,
                    blocks = np.array(new_blocks, dtype=float))

    def sort_rows(self, new_order):
        '''
        For each cell, sort the rows according to the
        order given by 'new_order'.
        '''
        m = 'The input variable "new_order" must be a ' + \
            'one-dimensional Numpy array of integers'
        assert (type(new_order) == np.ndarray) and \
            (len(new_order.shape) == 1) and \
            ((new_order.dtype == np.int64) or \
             (new_order.dtype == np.int32)), m
        
        ret = deepcopy(self)
        
        for c in self.coords:
            
            block = self.get(c)
            ret.cset(c, block[new_order, :])
            
        return ret
    
    def sort_columns(self, new_order):
        '''
        For each cell, sort the columns according to the 
        order given by 'new_order'.
        '''
        m = 'The input variable "new_order" must be a ' + \
            'one-dimensional Numpy array of integers'
        assert (type(new_order) == np.ndarray) and \
            (len(new_order.shape) == 1) and \
            ((new_order.dtype == np.int64) or \
             (new_order.dtype == np.int32)), m
        
        ret = deepcopy(self)
        
        for c in self.coords:
            
            block = self.get(c)
            ret.cset(c, block[:, new_order])
            
        return ret
    
    def print_blocks(self, name):
        
        print('\nMatrix ', name, ':')
        for c in self.coords:
            print('\nCoordinates:', c)
            print('Elements:', self.get(c))
        print('')
        
    def equal_to(self, other, tolerance=1e-30, count_blocks=True):
        '''
        Returns True if 'self' and 'other' has the same
        elements to within 'tolerance' in every block and the 
        same cell coordinates in 'self.coords' and 'other.coords'.
         
        '''
        equal = True
        # Loop over cell coordinates
        for c in self.coords:
            block_this = self.get(c)
            try:
                block_other = other.get(c)
            except:
                equal = False
            # Compare the two blocks
            if not np.allclose(block_this, block_other,
                               atol=tolerance):
                equal = False
        if equal and count_blocks:
            # If otherwise equal, check that both have the
            # same number of allocated blocks
            equal = (self.coords.shape == other.coords.shape)
            
        return equal

    def maxval(self):
        '''
        Get the maximum absolute-value element.
        '''
        m = 0.0
        for c in self.coords:
            m = np.amax([m, np.amax(np.absolute(self.get(c)))])
            
        return m
    
    def absmax_decay(self, p):
        R = np.sqrt(np.sum(p.coor2vec(self.coords)**2, axis = 1))
        mx = np.max(np.abs(self.cget(self.coords)), axis = (1,2))
        return R, mx
    
    def print_meminfo(self, name, one_shape=True):
        '''
        For a given 'tmat' object, print information about 
        the size of the matrix.
        
        one_shape     : True if all blocks in 'matrix' have
                        the same shape.
        '''
        print('\nInformation about the size of the '
              + name + ' matrix:\n')
        n_cells = self.coords.shape[0]
        print('Number of allocated cells       :',
              '{:>5}'.format(n_cells))
        
        if one_shape:
            shape = self.get([0, 0, 0]).shape
            # Memory storage in MB
            memory_used = 8.0 * n_cells * get_product(shape) / \
                          (1024**2)
            print('Block shape                 :  ',
                  shape)
            print('Estimated memory usage (MB) :    ',
                  '{:f}'.format(memory_used))
        print('', flush=True)


class primed_for_dot():
    def __init__(self, m, n_layers, inv = False, rcond = 1e-12, inv_mod = "lpinv"):
        
        
        self.n_layers = n_layers
        if n_layers is None:
            self.n_layers = np.max(np.abs(m.coords), axis = 0)
        

        self.nx,self.ny,self.nz = 2*self.n_layers + 1
        
        self.m1x,self.m1y = m.blocks.shape[1], m.blocks.shape[2]

        self.coords = np.roll(lattice_coords(self.n_layers).reshape(self.nx,self.ny,self.nz, 3), -self.n_layers, axis = (0,1,2)).reshape(self.nx*self.ny*self.nz, 3)

        #print("primed coords:", self.coords)
        #print("rcond", rcond)
        self.inv_mod = inv_mod
        

        
        self.M1 = np.fft.fftn(m.cget(self.coords).reshape(self.nx,self.ny,self.nz,self.m1x,self.m1y), axes = (0,1,2))

        #for nx in range(self.nx):
        #    for ny in range(self.ny):
        #        for nz in range(self.nz):
        #            print(nx,ny,nz,np.min(np.sum(np.linalg.qr(self.M1[nx,ny,nz])[1]**2, axis = 1)))

        #self.M1[0,0,0] = np.sum(m.blocks[:-1], axis = 0)
        if inv:
            
            inv_mod = self.inv_mod
            #
            #self.M1 = pinv2(self.M1.reshape(self.nx*self.ny*self.nz, self.m1x, self.m1y), rcond = rcond).reshape(self.nx,self.ny,self.nz,self.m1x,self.m1y)
            
            """
            for c in self.coords:
                if c[0] >=0: #using T-symmetry here
                    p_inv = pinv(self.M1[c[0], c[1], c[2]]) #, rcond = rcond)

                    print(c, "Inverse:", np.allclose(self.M1[c[0], c[1], c[2]], np.dot(self.M1[c[0], c[1], c[2]], np.dot(p_inv, self.M1[c[0], c[1], c[2]]))))
                    self.M1[c[0], c[1], c[2]] = p_inv
                    self.M1[-c[0],-c[1],-c[2]] = p_inv.conj()
            """

            # 1 get singular values of matrix
            s = np.abs(m.get_kspace_singular_values(n_points = n_layers)).reshape(self.nx*self.ny*self.nz, self.m1x) #.ravel())
            #print("singular values:", s.max(), s.min())
            #print(s)
            #smin, smax = s.max(), s.min()
            print(s.shape)
            smax, smin = s.max(), s.min()
            print("Overall condition of Coulomb matrix:", smax/smin, smax, smin)
            print("Conditions at separate K-points     :", np.max(s, axis = 1)/np.min(s, axis = 1))

            
            if inv_mod == "lpinv":

                self.M1 = np.linalg.pinv(self.M1.reshape(self.nx*self.ny*self.nz, self.m1x, self.m1y), rcond = rcond).reshape(self.nx,self.ny,self.nz,self.m1x,self.m1y)
            else:
                for c in self.coords:
                    #if c[0] >=0: #time reversal symmetry does not apply here ?
                    if inv_mod == "spinv":
                        p_inv = pinv(self.M1[c[0], c[1], c[2]], rcond = rcond)
                    if inv_mod == "spinv2":
                        p_inv = pinv2(self.M1[c[0], c[1], c[2]], rcond = rcond)
                    self.M1[c[0], c[1], c[2]] = p_inv #pseudo inverse
                    self.M1[-c[0], -c[1], -c[2]] = p_inv.conj().T



            



        

            




            if False:
                for c in self.coords:
                    if c[0] >=0: #time reversal symmetry
                    
                        #u_,s_,vh_ = np.linalg.svd(self.M1[c[0], c[1], c[2]])
                        u_,s_,vh_ = np.linalg.svd(self.M1[c[0], c[1], c[2]], full_matrices = True)
                        #smax = s_.max()
                        print(c,s_.max()/s_.min() , s_.max(), s_.min(), smax*rcond, s_.max()*rcond)
                        #print(np.sum(u_**2, axis = 1))
                        #print(np.sum(u_**2, axis = 0))
                        t = s_>rcond*smax

                        #print(c, np.linalg.norm(np.dot(u_, np.dot(np.diag(s_), vh_))-self.M1[c[0], c[1], c[2]]))

                        s_i = s_*1
                        s_i[t == False] = 0
                        s_i[t] = s_i[t]**-1

                        #print("inverse:", s_i)
                        


                        p_inv = np.dot(vh_.conj().T.dot(np.diag(s_i)), u_.conj().T)
                        
                        
                        #p_inv = pinv(self.M1[c[0], c[1], c[2]], rcond = rcond)



                        #inv_dev = np.abs(np.eye(p_inv.shape[0], dtype = float) - p_inv.dot(self.M1[c[0], c[1], c[2]])).max()

                        #print(c, "deviation in inverse:", np.abs(self.M1[c[0], c[1], c[2]]- np.dot(self.M1[c[0], c[1], c[2]], np.dot(p_inv, self.M1[c[0], c[1], c[2]]))).max())

                        
                            


                        self.M1[c[0], c[1], c[2]] = p_inv #pseudo inverse
                        self.M1[-c[0], -c[1], -c[2]] = p_inv.conj() 

                        #self.M1[c[0], c[1], c[2]] = np.linalg.pinv(self.M1[c[0], c[1], c[2]], rcond = rcond)


    
    def circulantdot(self, other, complx = False):
        m1x,m1y =  self.m1x, self.m1y
        m2x,m2y =  other.blocks.shape[1], other.blocks.shape[2]
        nx,ny,nz = self.nx,self.ny,self.nz
        coords = self.coords
        M1 = self.M1
        n_layers = self.n_layers
        #m2r = 

        




        M2 = np.fft.fftn(other.cget(coords).reshape(nx,ny,nz,m2x,m2y), axes = (0,1,2))
        #for c in coords:
            #if c[0]>=0:
            #    print(M2[-c[0], -c[1], -c[2]] == M2[c[0], c[1], c[2]].T)



        M3 = np.zeros((nx,ny,nz,m1x, m2y),dtype = np.complex128)


        print("Memory bottleneck in primed circulantdot:", (M3.nbytes + M2.nbytes + M1.nbytes)*1e-6, " Mb." )

        for c in coords:
            if c[0]>=0:
                M3[c[0], c[1], c[2]] = np.dot(M1[c[0], c[1], c[2]], M2[c[0], c[1], c[2]])
                M3[-c[0], -c[1], -c[2]] = M3[c[0], c[1], c[2]].conj()
        #import sys
        #sys.exit()

        #return tmat(blocks = np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m2y), coords = coords, screening = False)
        
        
        ret = tmat()
        
        if complx:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m2y), coords)
        else:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m2y), coords)
        return ret
        
    def linear_solve(self, other, complx = False):
        m1x,m1y =  self.m1x, self.m1y
        m2x,m2y =  other.blocks.shape[1], other.blocks.shape[2]
        nx,ny,nz = self.nx,self.ny,self.nz
        coords = self.coords
        M1 = self.M1
        n_layers = self.n_layers
        m2r = other.cget(coords).reshape(nx,ny,nz,m2x,m2y)

        




        M2 = np.fft.fftn(m2r, axes = (0,1,2))
        #for c in coords:
            #if c[0]>=0:
            #    print(M2[-c[0], -c[1], -c[2]] == M2[c[0], c[1], c[2]].T)



        M3 = np.zeros((nx,ny,nz,m1x, m2y),dtype = np.complex128)

        for c in coords:
            #if c[0]>=0:
            M3[c[0], c[1], c[2]] = np.linalg.solve(M1[c[0], c[1], c[2]], M2[c[0], c[1], c[2]])
            #    M3[-c[0], -c[1], -c[2]] = M3[c[0], c[1], c[2]].T.conj()
        
        
        ret = tmat()
        if complx:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).reshape(coords.shape[0], m1x,m2y), coords)
        else:
            ret.load_nparray(np.fft.ifftn(M3.reshape(nx,ny,nz,m1x,m2y), axes = (0,1,2)).real.reshape(coords.shape[0], m1x,m2y), coords)
        return ret
        
def all_columns(tmatrix, column):
    '''
    Get an array consisting of all column arrays for the
    column 'column'. The column subarrays are placed in
    the same order as the matrices are stored.
    '''
    (n_rows, n_columns) = tmatrix.get([0, 0, 0]).shape
    size = tmatrix.coords.shape[0] * n_rows
    columns_all = np.zeros([size, 1])
    
    coords0 = np.zeros(3, dtype=int)
    zeros = np.zeros((n_rows, n_columns),
                     dtype = tmatrix.get([0, 0, 0]).dtype)
    
    # Loop over lattice vectors
    first = 0
    last = n_rows
    for c in tmatrix.coords:
        
        block = deepcopy(tmatrix.get(c))
        if type(block) is int:
            block = zeros
            
        # Store the columns for this cell
        columns_all[first:last, 0] = \
            deepcopy(block[:, column])
        
        first = last
        last = first + n_rows
        
    return columns_all


def all_blocks(tmatrix):
    '''
    Return all blocks of 'tmatrix' as subsequent blocks 
    of rows in a 2D Numpy array.
    '''
    blocks_all = tmatrix.get(tmatrix.coords[0])
    n_blocks = tmatrix.coords.shape[0]
    
    for i in range(1, n_blocks):
        
        coords = tmatrix.coords[i]
        block = tmatrix.get(coords)
        
        blocks_all = np.append(blocks_all, block, axis=0)
        
    return blocks_all
        
#
# Functions realted to Fourier transform
#

def dfft(tmatrix, n_points=None):
    '''
    Discrete fast Fourier transform.
    
    tmatrix       : A truncated Toeplitz matrix (tmat object)
                    containing Fourier coefficients.
    n_points      : Number of fourier points in each 
                    direction.
    '''
    # Return a Fourier-space Toeplitz matrix
    return transform(tmatrix, np.fft.fftn, n_points, complx=True)


def idfft(tmatrix, n_points=None, complx=False):
    '''
    Inverse discrete fast Fourier transform.
    
    tmatrix       : A truncated Toeplitz matrix (tmat object)
    n_points      : Number of fourier points in each 
                    direction.
    '''
    # Return a direct-space Toeplitz matrix
    return transform(tmatrix, np.fft.ifftn, n_points, complx)
        

def tmatrix2array(tmatrix, n_points):
    '''
    Given a Toeplitz matrix, return a single Numpy array
    containing all elements.
    '''
    if len(n_points) == 1:
        return tmatrix2array_1d(tmatrix, n_points)
    elif len(n_points) == 2:
        return tmatrix2array_2d(tmatrix, n_points)
    elif len(n_points) == 3:
        return tmatrix2array_3d(tmatrix, n_points)
    
    
def get_fftindex(coord, n_coords):
    '''
    Get index corresponding to the coordinate 'coord' 
    and the maximum coordinate 'max_coord', so
    that coordinates are ordered as
    
     0 -> 0
     1 -> 1 
     2 -> 2
     3 -> 3
    -3 -> 4
    -2 -> 5
    -1 -> 6
    
    if max_coord = 3.
    '''
    return int(round(((n_coords * (1 - np.sign(coord)) / 2)
                      + coord) * abs(np.sign(coord))))


def shift_column(tmatrix,
                 translation,
                 column,
                 tolerance = 1.0e-12):
    '''
    Shift the column 'column' of each block in 'tmatrix'
    by the coordinate vector 'translation'.
    '''
    m = "Error. The input parameter 'tmatrix' must be " + \
        "of type 'tmat'."
    assert type(tmatrix) is tmat, m
    m = "Error. The input parameter 'column' has an " + \
        "invalid type or range."
    assert (type(column) is int) and \
        (column >= 0) and \
        (column < tmatrix.get([0, 0, 0]).shape[1]), m
    m = "Error. The input parameter 'translation' must " + \
        "be a 1D Numpy array of shape (3,)."
    assert (type(translation) is np.ndarray) and \
        (translation.shape == (3,)), m
    
    coords_t = deepcopy(tmatrix.coords.tolist())
    coeffs_t = []
    
    for c in tmatrix.coords:
        # Shift 'column' by 'translation'
        block = deepcopy(tmatrix.get(c))
        block[:, column] = \
            deepcopy(tmatrix.get(c - translation)[:, column])
        
        coeffs_t.append(block.tolist())
        
    for c in tmatrix.coords:
        # Shift outside the domain of 'tmatrix.coords'
        if not ((c + translation).tolist() in coords_t):
            
            block = np.zeros(tmatrix.get([0, 0, 0]).shape,
                             dtype = float)
            block[:, column] = \
                deepcopy(tmatrix.get(c)[:, column])
            
            max_element = \
                np.amax(np.absolute(tmatrix.get(c)[:, column]))
            if  max_element > tolerance: 
                coords_t.append((c + translation).tolist())
                coeffs_t.append(block.tolist())
                
    return tmat(coords = np.array(coords_t,
                                  dtype = int),
                blocks = np.array(coeffs_t,
                                  dtype = float))

    
def tmatrix2array_1d(tmatrix, n_points):
    '''
    Given a 'tmat' object, return one array all_elems[c, m, n], 
    where 'c' is an index corresponding to a cell in 
    tmatrix.cc and 'm' and 'n' correspond to matrix elements
    in the block with coordinates tmatrix.cc[c].
    
    This function assumes periodicity in one dimension.
    
    tmatrix       : A Toeplitz matrix (tmat object).
    '''
    # Size of matrix blocks
    (ni, nj) = tmatrix.get([0, 0, 0]).shape
    
    all_elems = np.zeros((n_points[0], ni, nj),
                         dtype=complex)
    
    # Loop over cell coordinates
    for c in tmatrix.coords:
        elems = tmatrix.get(c)
        
        if type(elems) != int:   
            
            # Give the blocks in an order corresponding
            # to the mesh order in numpy.fft, which is
            # 0, 1, 2, 3, -3, -2, -1, for example.
            #
            i = get_fftindex(c[0], n_points[0])
            # Store block in the big array
            all_elems[i, :, :] = elems
            
    return all_elems


def tmatrix2array_2d(tmatrix, n_points):
    '''
    Given a 'tmat' object, return one array 
    all_elems[c1, c2, m, n], where 'c1' and 'c2'
    correspond to cell coordinates, with a mapping defined by
    get_fftindex(), and 'm' and 'n' correspond to matrix 
    elements in the block with coordinates associated
    with c1 and c2.
    '''
    # Size of matrix blocks
    (ni, nj) = tmatrix.get([0, 0, 0]).shape
    
    all_elems = np.zeros((n_points[0], n_points[1], ni, nj),
                         dtype=complex)
    
    # Loop over cell coordinates
    for c in tmatrix.coords:
        elems = tmatrix.get(c)
        
        if type(elems) != int:
            
            # Give the blocks in an order corresponding
            # to the mesh order in numpy.fft, which is
            # 0, 1, 2, 3, -3, -2, -1, for example.
            #
            i = get_fftindex(c[0], n_points[0])
            j = get_fftindex(c[1], n_points[1])
            # Store block in the big array
            all_elems[i, j, :, :] = elems

    return all_elems


def tmatrix2array_3d(tmatrix, n_points):
    '''
    Given a 'tmat' object, return one array 
    all_elems[c1, c2, c3, m, n], where 'c1', 'c2', and 'c3'
    correspond to cell coordinates, with a mapping defined by
    get_fftindex(), and 'm' and 'n' correspond to matrix 
    elements in the block with coordinates associated
    with c1, c2, and c3.
    '''
    # Size of matrix blocks
    (ni, nj) = tmatrix.get([0, 0, 0]).shape
    
    all_elems = np.zeros((n_points[0], n_points[1], n_points[2],
                          ni, nj), dtype=complex)
    
    # Loop over cell coordinates
    for c in tmatrix.coords:
        elems = tmatrix.get(c)
        
        if type(elems) != int:
            
            # Give the blocks in an order corresponding
            # to the mesh order in numpy.fft, which is
            # 0, 1, 2, 3, -3, -2, -1, for example.
            #
            i = get_fftindex(c[0], n_points[0])
            j = get_fftindex(c[1], n_points[1])
            k = get_fftindex(c[2], n_points[2])
            # Store block in the big array
            all_elems[i, j, k, :, :] = elems
            
    return all_elems


def array2tmatrix(array, coords, n_dims, complx):
    '''
    Given a Numpy array containing elements corresponding
    to the lattice coordinates in 'coords', return a 
    Toeplitz matrix.
    '''
    if n_dims == 1:
        return array2tmatrix_1d(array, coords, complx)
    elif n_dims == 2:
        return array2tmatrix_2d(array, coords, complx)
    elif n_dims == 3:
        return array2tmatrix_3d(array, coords, complx)
    
    
def array2tmatrix_1d(all_elems, mesh, complx):
    '''
    Given an array with elements all_elems[c, m, n], where
    'c' is an index corresponding to a cell in 'coords',
    return a 'tmat' object 'tmatrix' where 
    matrix.cc[c] = [mesh[c], 0, 0, ] and the corresponding 
    block is all_elems[c, :, :].
    
    all_elems     : A three-dimensional Numpy array.
    coords        : A one-dimensional Numpy array.
    '''
    coords = []
    blocks = []
    mesh1 = mesh[0]
    # Loop over lattice vectors
    for m in range(mesh1.shape[0]):
        
        l_vector = [mesh1[m], 0, 0]
        if complx:
            elems_l = all_elems[m, :, :]
        else:
            elems_l = all_elems[m, :, :].real
            
        #if np.amax(np.absolute(elems_l)) > 1e-30:
        #    # Store block of Fourier coefficients
        coords.append(l_vector)
        blocks.append(elems_l)

    return tmat(np.array(coords), np.array(blocks),
                blockshape=(all_elems.shape[1],
                            all_elems.shape[2]), screening = False)


def array2tmatrix_2d(all_elems, mesh, complx, screening = True):
    '''
    Given an array with elements all_elems[c1, c2, m, n], where
    'c1' and 'c2' correspond to cell coordinates, 
    return a 'tmat' object 'tmatrix' where 
    matrix.cc[c] = [mesh[0][c1], mesh[1][c2], 0, ] and the corresponding 
    block is all_elems[c1, c2, :, :].
    
    all_elems     : A three-dimensional Numpy array.
    mesh          : A list containing two one-dimensional
                    Numpy arrays with cell coordiantes.
    complx        : True: Return a Toeplitz matrix with 
                    complex blocks.
                    False: Return a Toeplitz matrix with
                    real blocks.
    '''
    coords = []
    blocks = []
    mesh1 = mesh[0]
    mesh2 = mesh[1]
    
    # Loop over lattice vectors
    for m1 in range(mesh1.shape[0]):
        for m2 in range(mesh2.shape[0]):
            
            l_vector = [mesh1[m1], mesh2[m2], 0]
            if complx:
                elems_l = all_elems[m1, m2, :, :]
            else:
                elems_l = all_elems[m1, m2, :, :].real
            #if screening:
            #    if np.amax(np.absolute(elems_l)) > 1e-30:
            #        # Store block of Fourier coefficients
            #        coords.append(l_vector)
            #        blocks.append(elems_l)
            #else:
                # Store block of Fourier coefficients
            coords.append(l_vector)
            blocks.append(elems_l)


    #ret = tmat()
    #ret.load_nparray(np.array(blocks), np.array(coords),)
    
    return tmat(np.array(coords), np.array(blocks),
                blockshape=(all_elems.shape[2],
                            all_elems.shape[3]), screening = False)


def array2tmatrix_3d(all_elems, mesh, complx):
    '''
    Given an array with elements all_elems[c1, c2, c3, m, n], 
    where 'c1' and 'c2' correspond to cell coordinates, 
    return a 'tmat' object 'tmatrix' where 
    matrix.cc[c] = [mesh[0][c1], mesh[1][c2], mesh[2][c3]] 
    and the corresponding block is 
    all_elems[c1, c2, c3, :, :].
    
    all_elems     : A three-dimensional Numpy array.
    mesh          : A list containing two one-dimensional
                    Numpy arrays with cell coordiantes.
    complx        : True: Return a Toeplitz matrix with 
                    complex blocks.
                    False: Return a Toeplitz matrix with
                    real blocks.
    '''
    coords = []
    blocks = []
    mesh1 = mesh[0]
    mesh2 = mesh[1]
    mesh3 = mesh[2]
    # Loop over lattice vectors
    for m1 in range(mesh1.shape[0]):
        for m2 in range(mesh2.shape[0]):
            for m3 in range(mesh3.shape[0]):
                
                l_vector = [mesh1[m1], mesh2[m2], mesh3[m3]]
                if complx:
                    elems_l = all_elems[m1, m2, m3, :, :]
                else:
                    elems_l = all_elems[m1, m2, m3, :, :].real
                    
                #if np.amax(np.absolute(elems_l)) > 1e-30:
                #    # Store block of Fourier coefficients
                coords.append(l_vector)
                blocks.append(elems_l)
                    
    return tmat(np.array(coords), np.array(blocks),
                blockshape=(all_elems.shape[3],
                            all_elems.shape[4]))


def n_lattice(tmatrix):
    '''
    Get the number of lattice points in each Cartesian 
    direction, when the lattice is assumed to be 
    "rectangular" in 3D.
    '''
    # Lattice cutoffs in direct space
    cutoffs_d = tmatrix.cutoffs() 
    # Number of periodic dimensions
    n_dims = tmatrix.n_periodicdims()
    
    n = () 
    for i in range(n_dims):
        n = n + (2 * cutoffs_d[i] + 1, )
    return n


def transform(tmatrix, function, n_points=None, complx=True):
    '''
    Do either a Fourier transform or inverse Fourier 
    transform.
    
    tmatrix       : A truncated Toeplitz matrix (tmat object)
                    containing Fourier coefficients.
    function      : A function for doing discrete FFT or 
                    inverse FFT. For example, it may be
                    np.fft.fftn or np.fft.ifftn.
    n_points      : Number of fourier points in each 
                    direction.
    '''
 
    if n_points is None:
        n_points = n_lattice(tmatrix)
        
    # Number of cell coordinate dimensions
    n_dims = tmatrix.n_periodicdims()
    
    m = 'Error in the function transform(). ' + \
        'A number of lattice points must be given ' + \
        'for all periodic dimensions in the "tmat" object.'
    assert len(n_points) == n_dims, m
    
    # Get an array containing all elements
    all_elems = tmatrix2array(tmatrix, n_points)
    
    grid_axes = ()
    for i in range(n_dims):
        grid_axes += (i, )
        
    # Do the transform
    all_elems = function(all_elems, n_points, axes=grid_axes)

    # Get a lattice vector for each of the periodic dimensions
    l_points = []
    for i in range(n_dims):
        l_points.append((np.round(n_points[i] *
                                  np.fft.fftfreq(n_points[i])) 
        ).astype(int))
    
    # Return a Fourier-space or inverse Fourier-space
    # Toeplitz matrix
    return array2tmatrix(all_elems, l_points, n_dims, complx) 


def get_product(mytuple):
    '''
    Given a tuple 'mytuple', multiply the elements and return
    the product.
    '''
    product = 1
    for e in mytuple:
        product *= e
    return product



class MatrixData():
    '''
    Class for storing data of a 'tmat' object without using 
    Numpy. 

    This kind of storage is needed to convert such data
    to the JSON file format.
    '''
    number_type = None
    coords = []
    blocks = []

    def __init__(self, coords, blocks, number_type):
        '''
        coords      : Coordinates. Type list.
        blocks      : Element blocks. Type list.
        number_type : Type of the elements.
        '''
        m = "The parameter 'coords' should be a list."
        assert type(coords) is list, m
        m = "The parameter 'blocks' should be a list."
        assert type(blocks) is list, m
        m = 'The parameter "number_type" should be ' + \
            'of type str.'
        assert type(number_type) is str, m

        self.coords = coords
        self.blocks = blocks
        self.number_type = number_type
        

def get_data(tmatrix):
    '''
    Given a 'tmat' object, return a corresponding MatrixData 
    object.
    '''
    m = 'The parameter "tmatrix" must be of type "tmat".'
    assert type(tmatrix) is tmat, m
    
    elements_list = []
    for c in tmatrix.coords:
        block = tmatrix.get(c)
        elements_list.append(block.tolist())
        
    coords_list = tmatrix.coords.tolist()
    if tmatrix.get([0, 0, 0]).dtype == 'float64':
        n_type = 'float64'
    else:
        print('ERROR. The number type is not defined in ' + \
              'the function get_data().')
        
    return MatrixData(coords_list, elements_list, n_type)


def get_tmatrix(data):
    '''
    Given a MatrixData object, return a corresponding 'tmat' 
    object.
    '''
    return tmat(coords = np.array(data.coords, dtype=int),
                blocks = np.array(data.blocks, dtype=data.number_type))

def decay_check(m, p, printing = False, distance = 30, threshold = 1e-8):
    # Check that blocks beyond distance is below treshold
    R = np.sqrt(np.sum(p.coor2vec(m.coords))**2, axis =1 )
    Ri = np.argsort(R)
    Jm = np.max(np.abs(m.cget(m.coords)[Ri]), axis = (1,2))
    if printing:
        for l in np.array([R[Ri], Jm]).T:
            print(l)
    mx_d = np.max(np.abs(m.cget(m.coords)[R>distance]))
    if mx_d>threshold:
        print("Warning: found matrix element with abs.value %.6e outside %.6e bohr." % (mx_d, distance))
        if printing:
            for l in np.array([R[R>distance], np.max(np.abs(m.cget(m.coords)[R>distance]), axis = (1,2)) ] ).T:
                print(l)

    
def unfolded_product(m1, m2, coords = None, mx = None):
    """
    Computes the matrix product m1 * m2
    by temporarily unfolding the translational symmetry
    """
    if coords is None:
        mx = np.max( np.array([np.max(np.abs(m1.coords), axis = 0), 
                               np.max(np.abs(m2.coords), axis = 0)]), axis = 0)

        coords = lattice_coords(mx)
    else:
        if mx is not None:
            coords = lattice_coords(mx)
    #print(coords, coords[int(len(coords)/2)])
    M1 = m1.tofull(m1,coords,coords)
    M2 = m2.tofull(m2,coords,coords)
    
    RET = np.dot(M1, M2)
    rnx = m1.blocks.shape[1]
    rny = m2.blocks.shape[2]
    rc =  int(len(coords)/2)
    retblocks = RET[rnx*rc:rnx*(rc+1),:].reshape(rnx, coords.shape[0], rny).swapaxes(0,1)
    
    ret = tmat()
    ret.load_nparray(retblocks, coords)
    return ret