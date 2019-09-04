"""
Domain definitions

This module handles the truncated DEC-domains 

"""

import numpy as np

import utils.toeplitz as tp
import utils.prism as pr


def build_distance_matrix(p, coords, wcenters_a, wcenters_b):
    """
    Given two N*3 arrays with wannier centers, construct the BT-distance matrix with elements

    D^L_{pq} = <0p|r|0p> - <Lq|r|Lq>

    Input parameters

     p       - prism object with crystal geometry
     coords  - tp.tmat.coords array for which lattice cells to compute
     wcenters_a - wannier centers to remain fixed in the reference cell
     wcenters_b - wannier centers that are translated along the crystal lattice

     
    Returns
    d  - tp.tmat() object, BT-distance matrix

    """    
    dblocks = np.zeros((coords.shape[0], wcenters_a.shape[0], wcenters_b.shape[0]), dtype = float)
    for c in np.arange(coords.shape[0]):
        dblocks[c] = np.sqrt(np.sum( (wcenters_a[:, None] - wcenters_b[None, :] - p.coor2vec(coords[c]))**2, axis = 2))
    
    #sort blocks and coords in order of increasing distance

    d_order = np.argsort(np.min(dblocks,axis=(1,2)))

    d = tp.tmat()
    d.load_nparray(dblocks[d_order], coords[d_order])



    return d

def build_local_domain_index_matrix(fragment, d, distance_cutoff = 10.0):
    
    index_blocks = []
    index_coords = []
    for c in d.coords:
        di = d.cget(c)
        if np.any(di[fragment,:]<distance_cutoff):
            index_blocks.append(di[fragment, :]<distance_cutoff)
            index_coords.append(c)
    
    index_matrix = tp.tmat()
    index_matrix.load_nparray(np.array(index_blocks), np.array(index_coords), safemode = False)

    return index_matrix



def atomic_fragmentation(p, d, distance_cutoff= 1.0):
    """
    Split configuration space into atomic/proximal fragments
    """

    d_occ_ref = d.cget([0,0,0])[:p.get_nocc(), :p.get_nocc()] #inter-occupied distances in the refcell
    occ_indices = np.arange(p.get_nocc()) 
    occ_processed = []
    center_fragments = []
    for i in occ_indices:
        #print("Processing orbital ",i)
        if i not in occ_processed:

            
            #print(i, " added to the set:", occ_processed)
            proximal_occupied_indices = np.argwhere(d_occ_ref[ i,:]<distance_cutoff)[:,0]
            new_fragment = []
            for j in proximal_occupied_indices:
                if j not in occ_processed:
                    new_fragment.append(j)
            center_fragments.append(new_fragment)
            occ_processed += [i]
            occ_processed += list(proximal_occupied_indices)
            #print(proximal_occupied_indices, " added to the set:", occ_processed)
            #print("Center fragments:", center_fragments)
    return center_fragments