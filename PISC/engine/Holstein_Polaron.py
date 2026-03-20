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


class HolsteinPolaron:
    def __init__(self, L, M, t0, w0, g):
        self.L = L  # Number of lattice sites
        self.M = M  # Maximum phonon occupation number
        self.t0 = t0  # Hopping parameter
        self.w0 = w0  # Phonon frequency
        self.g = g    # Electron-phonon coupling strength

        self.fname = 'Holstein_Polaron_L_{}_M_{}_t0_{}_w0_{}_g_{}'.format(L, M, np.around(t0,2), np.around(w0,2), np.around(g,2))

        self.basis = self.generate_basis_states()

    def generate_basis_states(self):
        """Generate all basis states for the Holstein polaron model."""
        return basis_states(self.L, self.M)

    def def_hamil(self):
        """
        Define the Hamiltonian matrices for the Holstein polaron model.
        Initializes the phonon, hopping, and interaction Hamiltonian matrices separately.
        The basis size is usually very big, so we use sparse matrices.
        """

        if not hasattr(self, 'basis'):
            print("Basis states not defined. Generating basis states...")
            self.basis = self.generate_basis_states()
        self.H_ph = scipy.sparse.csr_matrix((len(self.basis), len(self.basis)))
        self.H_hop = scipy.sparse.csr_matrix((len(self.basis), len(self.basis)))
        self.H_int = scipy.sparse.csr_matrix((len(self.basis), len(self.basis)))
        self.H_tot = scipy.sparse.csr_matrix((len(self.basis), len(self.basis)))
    
    def set_H_ph(self):
        """
        Initialize the phonon Hamiltonian matrix.
        The phonon Hamiltonian is diagonal, so we only need to fill the diagonal elements.
        """
        H_ph = scipy.sparse.lil_matrix((len(self.basis), len(self.basis)))
        for i in range(len(self.basis)):
            H_ph[i, i] = H_phonon(self.basis[i], self.basis[i], self.w0) # Diagonal elements only

        self.H_ph = H_ph.tocsr()  # Convert to CSR format for efficiency
        
        scipy.sparse.save_npz(f'{self.fname}_H_ph.npz', self.H_ph)

    def set_H_hop(self, exp=False):
        """
        Initialize the hopping Hamiltonian matrix. This corresponds to nearest-neighbor hopping 
        on a lattice with periodic boundary conditions.

        We only consider pairs of basis states where the phonon occupation numbers are the same.
        """

        if exp: # Expensive but straightforward way to get pairs
            print('Using expensive method to get pairs of states with same phonon occupation numbers...')
            pairs= [(i, j) for i in range(len(self.basis)) for j in range(len(self.basis)) if (self.basis[i][1:] == self.basis[j][1:])]
        else: # This works, but needs to be understood!!!
            groups = defaultdict(list)
            for i, b in enumerate(self.basis): 
                groups[tuple(b[1:])].append(i)
            pairs = np.array([pair for g in groups.values() for pair in product(g, repeat=2)])

        # Initialize the hopping Hamiltonian matrix
        H_hop = scipy.sparse.lil_matrix((len(self.basis), len(self.basis)))

        for i, j in pairs:
            H_hop[i, j] = H_hopping(self.basis[i], self.basis[j], self.L, self.t0)
        
        print('Total number of pairs with same phonon occupation:', len(pairs), 'of', len(self.basis)*(len(self.basis)))
    
        self.H_hop = H_hop.tocsr()  # Convert to CSR format for efficiency
        scipy.sparse.save_npz(f'{self.fname}_H_hop.npz', self.H_hop)

    def set_H_int(self, chunk_size=500, exp=False):
        """
        Initialize the interaction Hamiltonian matrix. This corresponds to the interaction 
        between the electron and phonons.

        This matrix has non-trivial matrix elements and needs to be generate in 
        'chunks' of rows to avoid memory issues.
        """
        start_time = time.time()

        if exp:
            print('Using expensive method to initialize interaction Hamiltonian...')
            temp = np.zeros((len(self.basis), len(self.basis)), dtype=np.float64)
            init_Hint(temp, self.basis, self.g)
            self.H_int = scipy.sparse.lil_matrix(temp)
            return
        
        ch_lst = chunks(np.arange(len(self.basis)), chunk_size)
        print('Total number of chunks:', len(ch_lst))

        #Create an array of sparse matrices, each of the size of a chunk
        sparse_lst = []

        if(0): #Inefficient method
            for i in range(len(ch_lst)):
                sparse_lst.append(scipy.sparse.lil_matrix((len(ch_lst[i]), len(self.basis))))

            for i in range(len(ch_lst)):
                print('Processing chunk:', i)
                chunk = ch_lst[i]
                sparse_lst[i] = init_Hint_chunk(self.basis, chunk, self.g)
                print('time taken for chunk:', time.time() - start_time)
    
        if(1):
            for i in range(len(ch_lst)):
                print('Processing chunk:', i)
                chunk = ch_lst[i]
                vals, rows, cols = init_Hint_chunk_pairs(np.array(self.basis), chunk, self.g)
                sparse_chunk = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(len(chunk), len(self.basis))) 
                scipy.sparse.save_npz(f'{self.fname}_H_int_chunk_{i}.npz', sparse_chunk)
                #sparse_lst.append(sparse_chunk)
                print('time taken for chunk:', time.time() - start_time)
        
        self.H_int = (scipy.sparse.vstack(sparse_lst)).tocsr()
        
        #scipy.sparse.save_npz(f'{self.fname}_H_int.npz', self.H_int)

    def set_H(self):
        """
        Initialize the total Hamiltonian matrix by combining the phonon, hopping, and interaction parts.
        """
        self.set_H_ph()
        self.set_H_hop()
        self.set_H_int()
        
        # Combine all parts into the total Hamiltonian
        self.H_tot = self.H_ph + self.H_hop + self.H_int

        scipy.sparse.save_npz(f'{self.fname}_H_tot.npz', self.H_tot)

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


def H_hopping(st_n, st_m, L, t0):
    """
    Electron hopping part of the Hamiltonian for the Holstein polaron model.
    This corresponds to nearest-neighbor hopping on a lattice with 
    periodic boundary conditions.
    """

    ph_n = st_n[1:] # phonon occupation numbers for state i
    ph_m = st_m[1:]

    el_n = st_n[0]+1  # electron site for state i
    el_m = st_m[0]+1  # electron site for state j
    
    if (ph_n != ph_m): # This statement is too expensive!! 
        return 0.0
    elif abs(el_n%L - el_m%L) == 1:
        return -t0
    else:
        return 0.0

@njit
def H_phonon(st_n, st_m, w0):
    """
    Phonon part of the Hamiltonian for the Holstein polaron model.
    This corresponds to the harmonic oscillator Hamiltonian.
    """
    ph = st_n[1:]  # phonon occupation numbers

    #print('st_n:', st_n, 'st_m:', st_m, st_n!=st_m)
    if (st_n != st_m):  # Only diagonal elements contribute
        return 0.0
    else:
        return w0 * sum(ph)  # Return energy proportional to total phonon occupation

@njit
def H_interaction(st_n, st_m, g):
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
def init_Hint(H_mat, basis, g):
    for i in prange(len(basis)):
        for j in prange(len(basis)):
            H_mat[i, j] = H_interaction(basis[i], basis[j], g)

@njit(parallel=True)
def init_Hint_chunk(basis, chunkrange, g):
    """
    Initialize the Hamiltonian matrix in parallel using Numba.
    Each chunk corresponds to a few rows of the Hamiltonian matrix.
    """
    
    chunk = np.zeros((len(chunkrange), len(basis)), dtype=np.float64)

    for i in prange(len(chunkrange)):
        n = chunkrange[i]
        for j in prange(len(basis)):
            m = j
            #print('Processing chunk:', i, j)
            chunk[i, j] = H_interaction(basis[n], basis[m], g)
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
            val = H_interaction(basis[n], basis[m], g)
            #print('Processing chunk:', i, j, val)
            if (val != 0.0):
                #print('nnz:', i, j, val)
                rows.append(i)
                cols.append(j)
                vals.append(val)
    return vals, rows, cols

