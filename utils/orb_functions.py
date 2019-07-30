
import numpy as np

from time import time
from toeplitz import tmat



def norms(ao_overlaps, coeffs):
    '''
    Given AO overlaps and MO or PAO coefficients, compute the 
    norms <p|p> of the PAOs or MOs.
    
    ao_overlaps   : AO overlaps, given as a 'tmat' object.
    pao_coeffs    : PAO coefficients, given as a 'tmat' 
                    object. The elements are stored as 
                    C_{\mu, p}, where a row corresponds to
                    an AO and a column to an MO or a PAO.
    '''
    coords0 = np.zeros((1, 3), dtype=int)    
    
    time1 = time()
    c_tT = coeffs.tT()
    s_tT = ao_overlaps.tT()
    
    # C_tT * S_tT
    cs = c_tT.cdot(s_tT, coords = coeffs.coords)
    time2 = time()
    print('Time for S*C:', time2 - time1)
    
    # (C_tT * S_tT) * C 
    time3 = time()
    orb_overlaps = cs.cdot(coeffs, coords = coords0)
    time4 = time()
    print('Time for C * SC:', time4 - time3)
    
    # Return orbital norms, in the same order as the MO or
    # PAO orbitals given in 'coeffs'.
    return np.diagonal(orb_overlaps.get(coords0[0]))


def normalize(coeffs, norms):
    '''
    Given PAO or MO coefficients in the form C_{\mu, p}^{L},
    where \mu is an AO and p is an MO or PAO, and 
    corresponding norms, return normalized coefficients. 
    
    coeffs        : MO or PAO coefficients, given as a 
                    'tmat' object.
    norms         : Norms <p|p>, given as a 1D Numpy array
                    in the same order as the orbitals p
                    in 'coeffs'.
    '''
    blocks = []
    # Loop over cell blocks and normalize
    for c in coeffs.coords:
        block = coeffs.get(c) / np.sqrt(norms)
        blocks.append(block)
        
    return tmat(coords = coeffs.coords,
                blocks = np.array(blocks, dtype=float))


def normalize_orbitals(ao_overlaps, coeffs):
    '''
    Normalize the coefficients 'coeffs'.
    
    ao_overlaps   : AO overlaps, given as a 'tmat' object.
    coeffs        : Coefficients, given as a 'tmat' object.
    '''
    nrms = norms(ao_overlaps, coeffs)
    
    return normalize(coeffs, nrms)
