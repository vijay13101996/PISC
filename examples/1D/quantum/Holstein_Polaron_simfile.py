import numpy as np
import itertools
import scipy
import scipy.sparse
import time
from numba import njit, prange
from functools import partial
from multiprocessing import Pool
from PISC.utils.mptools import chunks, batching
import cProfile
from collections import defaultdict
from itertools import product
from PISC.engine.Holstein_Polaron import HolsteinPolaron



t0 = 1 
w0 = t0/2
g = t0/np.sqrt(2)

L = 6
M = 2

count = 0
def basis_states(L, M):
    """ 
    Generate all basis states for a Holstein polaron model.
    Basis states are product states as follows:
    |i, n1, n2, ..., nL> 
    where 
    i is the electron site (ranging from 0 to L-1)
    n1, n2, ..., nL are phonon occupation numbers (ni <= M)
    """
    states = []
    phonon_basis = itertools.product(range(M + 1), repeat=L)
    phonon_basis = list(phonon_basis)  # Convert to list to reuse it
    for i in range(L):
        for phonon_state in phonon_basis:
            state = (i,) + phonon_state
            states.append(state)
    return states

@njit
def H_hopping(st_n,st_m):
    """
    Electron hopping part of the Hamiltonian for the Holstein polaron model.
    This corresponds to nearest-neighbor hopping on a lattice with 
    periodic boundary conditions.
    """

    ph_n = st_n[1:] # phonon occupation numbers for state i
    ph_m = st_m[1:]

    el_n = st_n[0]+1  # electron site for state i
    el_m = st_m[0]+1  # electron site for state j
    
    if (ph_n != ph_m).any(): # This statement is too expensive!! 
        return 0.0
    elif abs(el_n%L - el_m%L) == 1:
        return -t0
    else:
        return 0.0

@njit
def H_phonon(st_n, st_m):
    """
    Phonon part of the Hamiltonian for the Holstein polaron model.
    This corresponds to the harmonic oscillator Hamiltonian.
    """
    ph = st_n[1:]  # phonon occupation numbers

    if (st_n != st_m).any():  # Only diagonal elements contribute
        return 0.0
    else:
        return w0 * sum(ph)  # Return energy proportional to total phonon occupation

@njit
def H_interaction(st_n, st_m):
    """
    Interaction part of the Hamiltonian for the Holstein polaron model.
    This corresponds to the interaction between the electron and phonons.
    """
    
    if st_n[0] != st_m[0]: # Acts only on states with the same electron site
        return 0.0
    site = st_n[0]  # Electron site
    ph_n = st_n[site]  # phonon occupation numbers for state n
    ph_m = st_m[site]  # phonon occupation numbers for state m
    if ph_m - ph_n == 1:
        return g * (ph_m)**0.5
    elif ph_n - ph_m == 1:
        return g * (ph_n)**0.5
    else:
        return 0.0

@njit
def H_total(st_n, st_m):
    """
    Total Hamiltonian for the Holstein polaron model.
    Combines hopping, phonon, and interaction parts.
    """
    return H_phonon(st_n, st_m) + H_interaction(st_n, st_m) + H_hopping(st_n, st_m) 

@njit(parallel=True)
def init_hamil(H_mat, H_func, basis):
    for i in range(len(basis)):
        for j in range(len(basis)):
            H_mat[i, j] = H_func(basis[i], basis[j])

@njit(parallel=True)
def init_hamil_chunk(H_func, basis, chunkrange):
    """
    Initialize the Hamiltonian matrix in parallel using Numba.
    Each chunk corresponds to a few rows of the Hamiltonian matrix.
    """
    
    chunk = np.zeros((len(chunkrange), len(basis)), dtype=np.float64)

    for i in range(len(chunkrange)):
        n = chunkrange[i]
        for j in range(len(basis)):
            m = j
            #print('Processing chunk:', i, j)
            chunk[i, j] = H_func(basis[n], basis[m])
    return chunk

@njit(parallel=True)
def init_Hint_chunk_pairs(basis, chunkrange, g):
    """
    Same as init_Hint_chunk, but returns only non-zero elements of the interaction Hamiltonian.
    Output is an array of tuples:
    (row_index, column_index, value)
    """
    
    rows = []
    cols = []
    vals = []
    for i in range(len(chunkrange)):
        n = chunkrange[i]
        for j in range(len(basis)):
            m = j
            val = H_interaction(basis[n], basis[m])
            if (val != 0.0):
                rows.append(i)
                cols.append(j)
                vals.append(val)
    return vals, rows, cols

@njit(parallel=True)
def init_hamil_pairs(H_func, basis, pairs):
    """
    Initialize the Hamiltonian matrix for specific pairs of basis states.
    This is used for parallel computation.
    """
    Hlist = np.zeros(len(pairs), dtype=np.float64)
    for n in prange(len(pairs)):
        i, j = pairs[n]
        Hlist[n] = H_func(basis[i], basis[j])  
        
    return Hlist

@njit(parallel=True)
def init_row_hamil(H_func, basis, row_index):
    """
    Initialize a specific row of the Hamiltonian matrix.
    This is used for parallel computation.
    """ 
    H_row = np.zeros(len(basis), dtype=np.float64)
    for j in prange(len(basis)):
        H_row[j] = H_func(basis[row_index], basis[j])
    return H_row

def main():
    start_time = time.time()

    basis = basis_states(L, M)
    basis = np.array(basis, dtype=np.int64)
    print('Total number of basis states:', len(basis), L*(M + 1)**L, len(basis)**2)

    hp = HolsteinPolaron(L, M, t0, w0, g)

    hp.generate_basis_states()
    hp.def_hamil()
    hp.set_H_ph()
    hp.set_H_hop()
    hp.set_H_int(exp=True,chunk_size=100)
    hp.set_H()

    if(0):
        #---------Phonon Hamiltonian------------------------------------------
        H_ph = scipy.sparse.lil_matrix((len(basis), len(basis)))

        for i in range(len(basis)):
            H_ph[i, i] = H_phonon(basis[i], basis[i]) # Diagonal elements only

        print('H_phonon', np.sum(abs(H_ph)))
        assert(np.allclose(H_ph.todense(), hp.H_ph.todense())), "Phonon Hamiltonian does not match!"
        exit() 
    #--------Hopping Hamiltonian------------------------------------------
    if(0):
        H_hop = scipy.sparse.lil_matrix((len(basis), len(basis)))
        
        #Consider pairs where all the phonon occupation numbers are the same
        #pairs_g= [(i, j) for i in range(len(basis)) for j in range(len(basis)) if (basis[i][1:] == basis[j][1:]).all()]

        groups = defaultdict(list)
        for i, b in enumerate(basis): groups[tuple(b[1:])].append(i)
        pairs = np.array([pair for g in groups.values() for pair in product(g, repeat=2)])

        #Check if pair and pairs_g contain the same elements
        #assert(set(pairs) == set(pairs_g)), "Pairs and pairs_g do not match!"

        print('Total number of pairs with same phonon occupation:', len(pairs), 'of', len(basis)*(len(basis)))
        #H_list = init_hamil_pairs(H_hopping, basis, pairs)
        #rows,cols = zip(*pairs)
        #H_hop[rows, cols] = H_list
        
        #H_hop1 = scipy.sparse.lil_matrix((len(basis), len(basis)))
        for i, j in pairs:
            H_hop[i, j] = H_hopping(basis[i], basis[j])

        print('H_hopping', np.sum(abs(H_hop)))
        assert(np.allclose(H_hop.todense(), hp.H_hop.todense())), "Hopping Hamiltonian does not match!"
        exit()

    #---------Interaction Hamiltonian-------------------------------------

    if(0):
        H_int = scipy.sparse.lil_matrix((len(basis), len(basis)))

        # Initialize the Hamiltonian matrices
        #H_int = init_hamil(H_int, H_interaction, basis)
        #H_int1 = init_hamil_chunk(H_interaction, basis, np.arange(len(basis)))

        ch_lst = chunks(np.arange(len(basis)), 5000)
        print('Total number of chunks:', len(ch_lst))

        #Create an array of sparse matrices, each of the size of a chunk
        sparse_lst = []

        for i in range(len(ch_lst)):
            print('Processing chunk:', i)
            chunk = ch_lst[i]
            #sparse_chunk = scipy.sparse.coo_matrix(init_hamil_chunk(H_interaction, basis, chunk))
             
            vals, rows, cols = init_Hint_chunk_pairs(basis, chunk, g)
            sparse_chunk = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(len(chunk), len(basis)))
            sparse_lst.append(sparse_chunk)

            #Count number of non-zero elements in the chunk
            #print('Chunk size:', sparse_chunk.nnz)
            #print('Chunk memory size', sparse_chunk.nnz * 8 / (1024**2), 'MB')  # Each float64 takes 8 bytes

            #print('sparse_lst memory size', sum([m.data.nbytes for m in sparse_lst])/ (1024**2), 'MB')
            print('time taken for chunk:', time.time() - start_time)
    
        H_int = scipy.sparse.vstack(sparse_lst)
        #print('H', np.sum(abs(H_int)))

        assert(np.allclose(H_int.todense(), hp.H_int.todense())), "Interaction Hamiltonian does not match!"

    
    print('Time taken to construct Hamiltonian:', time.time() - start_time)

    #exit()

if __name__ == "__main__":
    #cProfile.run('main()')
    main()

