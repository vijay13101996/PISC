import numpy as np
from matplotlib import pyplot as plt
from PISC.engine.xxz import XXZ
import cProfile
import time
import scipy
import numpy
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from joblib import Parallel, delayed
import pickle
from functools import partial
from multiprocessing import Pool

"""
# Tasks tomorrow:
# 1. Modify code to include the magnetic field term in the Hamiltonian.
# 2. Check if the eigenvalues obey the modified dispersion relation.
# 3. Understand the relation between dispersion and ground-state energy.
# 4. See if the Hamiltonian can be broken down into parity sectors. If yes, check
#    if the eigenvalues can be obtained from each sector separately.
# 5. If the above works, see related dispersion relations for Delta != 0.

# 6. Understand the Z2 and P symmetry and how to construct the reduced basis set
# 7. Understand how to compute the reduced basis without evaluating the whole Hamiltonian
# 8. Fix the speed issue with computing the H_two_site function.
# 9. Understand and code up the matrix elements for the K=2\pi/L sector
# 10. Check if the parallelization is actually efficient and check the time taken for L=18
"""


def define_lattice(L, J, Delta, g=0.0, J2=0.0, Delta2=0.0):
    """
    Define the lattice for the XXZ model with given parameters.
    L: Length of the chain
    J: Coupling constant
    Delta: Anisotropy parameter
    g: Magnetic field (default 0.0)
    J2: NNN coupling constant (default 0.0)
    Delta2: NNN anisotropy parameter (default 0.0)
    """

    # Initialize the XXZ model with the specified parameters
    xxz = XXZ(L, J, Delta, g, J2, Delta2)
    print(len(xxz.basis_states), "basis states")
    return xxz

def restrict_basis(xxz, k=0.0):
    """
    Restrict the basis states to those states which are 
        1. No net magnetization, i.e. M_z = 0
        2. Invariant under the parity transformation (reflection)
        3. Invariant under the Z2 transformation (spin flip)
    xxz: Instance of the XXZ class

    Returns:
    basis_Mz0: Basis states with M_z = 0
    orbits: Orbits of the basis states
    gr_orbits: Groups of orbits identified by the P and Z2 symmetries
    """
    
    # Restrict basis states to those with M_z = 0
    basis_Mz0 = xxz.find_Sz_0(xxz.basis_states, Mz=0)
    #for state in basis_Mz0:
    #    print(state, xxz.M_z(state), xxz.find_E(state))
    print("Number of basis states with M_z = 0:", len(basis_Mz0), 'Expected:', np.math.comb(xxz.L, xxz.L//2))

    # Find orbits of the basis states - to get k=0 sector
    orbits = xxz.find_orbits(basis_Mz0)
    print("Number of orbits:", len(orbits), sum(len(orbit) for orbit in orbits), "states in total")

    # If k=2*np.pi/L, restrict the orbits to those of length L
    if(k==2*np.pi/xxz.L): # !!! This is potentially dangerous, adjust it later
        print('Limiting to orbits of length', xxz.L)
        orbits = xxz.L_orbits(orbits)

    #gr_orbits = xxz.group_orbits(orbits, xxz.P_operator, xxz.Z2_operator)
    #gr_orbits = xxz.pair_orbits(orbits, xxz.Z2_operator)
    gr_orbits = [[orbit] for orbit in orbits]

    print("Number of groups of orbits:", len(gr_orbits), ",", sum(len(group) for group in gr_orbits), "orbits in total")

    count=0
    for i, group in enumerate(gr_orbits):
        #print("Group", i, ":", group)
        #print("Number of states in group:", len(group))
        #print("States in orbit:", orbit)
        count += len(group)

    print("Total number of states in all orbits:", count)

    return basis_Mz0, orbits, gr_orbits

def compute_H_symm_row(xxz, i, gr_orbits, k=0.0):
    """
    Given an index i and group of orbits(identified by P and Z2 symmetries),
    compute the corresponding row of the desymmetrized Hamiltonian matrix.
    xxz: Instance of the XXZ class
    i: Index of the group of orbits
    gr_orbits: Groups of orbits identified by the P and Z2 symmetries

    Returns:
    H_row: Row of the desymmetrized Hamiltonian matrix corresponding to the group of orbits
    """

    H_row = np.zeros(len(gr_orbits)-i, dtype=complex)
    for j in range(i, len(gr_orbits)):
        H_row[j-i] = xxz.H_symm_adapted(gr_orbits[i], gr_orbits[j], NNN=True, B=True, k=k)
    return  H_row
    
def compute_H_symm(xxz, gr_orbits, parallel=True, k=0.0):
    """
    Compute the symmetric Hamiltonian matrix for the XXZ model using the groups of orbits.
    xxz: Instance of the XXZ class
    gr_orbits: Groups of orbits identified by the P and Z2 symmetries

    Returns:
    H_symm: Symmetric Hamiltonian matrix
    """
    
    H_symm = np.zeros((len(gr_orbits), len(gr_orbits)), dtype=complex)

    print("Computing the symmetric Hamiltonian matrix for k =", k)
    if parallel:
        # Use parallel processing to compute the rows of the symmetric Hamiltonian matrix
        n_jobs = 10
        rows = Parallel(n_jobs=n_jobs)(delayed(compute_H_symm_row)(xxz, i, gr_orbits, k) for i in range(len(gr_orbits)))
   
        for i, row in enumerate(rows):
            H_symm[i, i:] = row
            H_symm[i:, i] = np.conj(row)

        return H_symm

    else:
        
        if(0):# Compute the rows of the symmetric Hamiltonian matrix sequentially
            n_jobs = 1
            func = partial(compute_H_symm_row, xxz, gr_orbits=gr_orbits, k=k)
            #with Pool(processes=n_jobs) as p:
            with ProcessPoolExecutor(max_workers=n_jobs) as p:
                rows = list(p.map(func, range(len(gr_orbits))))
                

            for i, row in enumerate(rows):
                #print("Filling row", i, 'with', np.around(row, decimals=2))
                H_symm[i, i:] = row
                H_symm[i:, i] = np.conj(row)

            return H_symm

        if(1):
            for i in range(len(gr_orbits)):
                #print("Calculating H_symm_row(", i, ")")
                row = compute_H_symm_row(xxz, i, gr_orbits, k)
                H_symm[i, i:] = row
                H_symm[i:, i] = np.conj(row)

            return H_symm

def benchmark_1():
    """
    For the special cases of 
    1. Delta=0, benchmark the resulting XX Hamiltonian
    2. Delta=1, benchmark the resulting Heisenberg Hamiltonian
    """

    L = 6 # Length of the chain
    J = 1.0  # Coupling constant
    Delta = 1.0  # Anisotropy parameter
    g = 0.0  # Magnetic field
   
    print("Benchmarking Heisenberg model with Delta=0")
    xx = define_lattice(L, J, 0.0, g)
    print('g', xx.g, 'J', xx.J, 'Delta', xx.Delta, 'L', xx.L)
    
    H_xx = xx.H(B=True)  # Compute the Hamiltonian matrix with NNN and magnetic field terms
    print("Full Hamiltonian computed for Heisenberg model with Delta=0")

    vals, vecs = np.linalg.eigh(H_xx)

    K_p = (2*np.arange(-L//2, L//2) + 1) * np.pi / L
    K_m = (2*np.arange(-L//2+1, L//2+1)) * np.pi / L 

    print("K_p:", K_p/np.pi, "K_m:", K_m/np.pi)

    E_p = J * np.cos(K_p) + g  # Positive parity
    E_m = J * np.cos(K_m) + g  # Negative parity
    E_p = E_p[E_p <=0]
    E_m = E_m[E_m <=0]
    print("E_p:", J * np.cos(K_p) + g, "E_m:", J * np.cos(K_m) + g)

    """
    The derivation of the ground state energy for the XX model is provided elsewhere
    """
    E_gs = min(np.sum(E_p), np.sum(E_m)) - L * g / 2  # Ground state energy
    print("Ground state energy for Heisenberg model with Delta=0:", vals[0], "Expected:", E_gs)

    if np.isclose(vals[0], E_gs, atol=1e-5):
        print("***Ground state energy matches the expected value***")

    print("Benchmarking XXZ model with Delta=1.0")
    xxx = XXZ(L, J, 1.0, 0.0)
    H_xxx = xxx.H()  # Compute the Hamiltonian matrix with NNN and magnetic field terms
    vals, vecs = np.linalg.eigh(H_xxx)

    """
    The ground-state energies for the Heisenberg model are obtained from:
    van de Braak, H. P., and W. J. Caspers. 
    "Ground‐State Properties of Finite Antiferromagnetic Chains." 
    physica status solidi (b) 35.2 (1969): 933-940
    """
    Egs_exp = np.array([-1.5, -1.0, -0.93426, -0.91277])/2 # for L=2,4,6,8

    print('E per site', vals[0] / L, 'Expected:', Egs_exp[L//2 - 1])

    if np.isclose(vals[0] / L, Egs_exp[L//2 - 1], atol=1e-4):
        print("***Ground state energy per site matches the expected value***")

def benchmark_2():
    """
    Preliminary benchmark for the Hamiltonian of the XXZ model with NN and NNN coupling
    in the Mz=0, spin-reversal invariant, and parity-invariant sector.

    We compute the eigenvalues of the full Hamiltonian matrix and restrict the 
    basis to the Mz=0 sector, and then to the parity and Z2 invariant sector.

    The resulting eigenvalues should be contained in the eigenvalues of the full Hamiltonian matrix. 
    
    Note that, this is still only a preliminary benchmark; if the hamiltonian matrix elements are
    not computed correctly, the eigenvalues will still match, but that does not mean the Hamiltonian is correct.
    """
    
    L = 8  # Length of the chain
    J = 1.0  # Coupling constant
    Delta = 0.2  # Anisotropy parameter
    g = 0.0  # Magnetic field
    J2 = 1.0  # NNN coupling constant
    Delta2 = 0.4  # NNN anisotropy parameter

    k = 2 * np.pi / L  # Wavevector for the k=2\pi/L sector

    print("Benchmarking XXZ model with NN and NNN coupling")
    xxz = define_lattice(L, J, Delta, g, J2, Delta2)
    basis_Mz0, orbits, gr_orbits = restrict_basis(xxz, k=k)  # Restrict the basis states
    print('\n')

    print("Number of basis states with M_z = 0:", len(basis_Mz0),'Expected', np.math.comb(L, L//2))

    H_full = xxz.H(NNN=True)  # Full Hamiltonian with NNN and magnetic field terms
    vals_full, vecs_full = np.linalg.eigh(H_full)
    vals_full = np.around(vals_full, decimals=3)  # Round eigenvalues for comparison

    H_Sz0 = xxz.H(NNN=True, basis_states=basis_Mz0)  # Hamiltonian restricted to M_z = 0 sector
    vals_Sz0, vecs_Sz0 = np.linalg.eigh(H_Sz0)
    vals_Sz0 = np.around(vals_Sz0, decimals=3)  # Round eigenvalues for comparison
    
    H_symm = compute_H_symm(xxz, gr_orbits, parallel=False, k=k) # Hamiltonian restricted to P and Z2 invariant sector
    vals_symm, vecs_symm = np.linalg.eigh(H_symm)
    vals_symm = np.around(vals_symm, decimals=3)  # Round eigenvalues for comparison

    #Check if vals_Sz0 is contained in vals_full
    if np.all(np.isin(vals_Sz0, vals_full)):
        print("***Eigenvalues of the M_z = 0 sector are contained in the full Hamiltonian eigenvalues***")

    #Check if vals_symm is contained in vals_Sz0
    print(vals_symm,'\n', vals_Sz0)
    if np.all(np.isin(vals_symm, vals_Sz0)):
        print("***Eigenvalues of the P and Z2 invariant sector are contained in the M_z = 0 sector eigenvalues***")
    

def main():
    start_time = time.time() 
    
    #benchmark_1()
    benchmark_2()    
    exit(0)

    L = 6  # Length of the chain
    J = 1.0  # Coupling constant
    Delta = 0.5  # Anisotropy parameter
    g = 0.0  # Magnetic field
    J2 = 0.5  # NNN coupling constant
    Delta2 = 0.5  # NNN anisotropy parameter

    k = 2 * np.pi / L  # Wavevector for the k=2\pi/L sector

    xxz = define_lattice(L, J, Delta, g, J2, Delta2)  # Initialize the XXZ model 
    basis_Mz0, orbits, gr_orbits = restrict_basis(xxz, k=k)  # Restrict the basis states

    for ob1 in orbits:
        for ob2 in orbits:
            Hij1 = xxz.H_k2piL(ob1, xxz.T_op(list(ob2)), NNN=True)
            Hij2 = xxz.H_k2piL(xxz.T_op(list(ob2)), ob1, NNN=True)
            print(Hij1, Hij2)
            assert np.isclose(Hij1, np.conj(Hij2)), "H_k2piL is not Hermitian!"
    print('DOne')
    exit(0)

    if(k == 0):
        fname= 'H_symm_L_{}_J_{}_Delta_{}_g_{}_J2_{}_Delta2_{}_k0.pkl'.format(L, J, Delta, g, J2, Delta2)
    elif(k == 2*np.pi/L):
        fname = 'H_symm_L_{}_J_{}_Delta_{}_g_{}_J2_{}_Delta2_{}_k2piL.pkl'.format(L, J, Delta, g, J2, Delta2)
        
    try:
        with open(fname, 'rb') as f:
            H_symm = pickle.load(f)
        print("Loaded H_symm from file:", fname)
    except FileNotFoundError:
        print("File not found:", fname)
        print("Computing the k=0, P and Z2 invariant Hamiltonian matrix")
        H_symm = compute_H_symm(xxz, gr_orbits, parallel=False, k=k)

        #H_symm2 = compute_H_symm(xxz, gr_orbits, parallel=False, k=k)
        #assert np.allclose(H_symm, H_symm2), "Parallel and non-parallel results do not match!"


        #Store H_symm in a pickle file
        with open(fname, 'wb') as f:
            pickle.dump(H_symm, f)

    # Test if H_symm is Hermitian
    if np.allclose(H_symm, H_symm.conj().T, atol=1e-8):
        print("H_symm is Hermitian")
    else:
        print("H_symm is not Hermitian")


    if(1):
        vals_symm, vecs_symm = np.linalg.eigh(H_symm)
        vals_symm = np.around(vals_symm, decimals=3)  # Round


        #Unfold the eigenvalues
        vals_uf = vals_symm[::20]
        irange = np.arange(len(vals_uf))

        # Number of levels with energy less than or equal to each eigenvalue
        nE = np.array([np.sum(vals_symm <= val) for val in vals_uf])

        print("Number of levels with energy less than or equal to each eigenvalue:", nE)


        #Spline fit nE against vals_uf, i.e. nE = f(vals_uf)
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(vals_uf, nE, s=0)
        x_fit = np.linspace(vals_uf.min(), vals_uf.max(), 1000)
        y_fit = spline(x_fit)

        Erange = np.arange(-4,4.01, 0.01)

        nEr = np.array([np.sum(vals_symm <= val) for val in Erange])
        plt.plot(Erange, nEr, label='Number of levels')
        plt.scatter(vals_uf, nE, color='red', label='Data points')
        plt.xlabel('Energy')
        plt.ylabel('Number of levels')
        plt.show()


        if(0):
            plt.plot(vals_uf, nE, 'o', label='Data points')
            plt.plot(x_fit, y_fit, label='Spline fit')
            plt.xlabel('Energy')
            plt.ylabel('Number of levels')
            plt.title('Number of levels vs Energy for k=0, P and Z2 invariant sector')
            plt.legend()
            plt.show()

        if(1):
            bins = np.arange(0,3.01, 0.1)
            diff = np.diff(vals_symm)
            plt.hist(diff, bins=100, density=True)
            plt.xlabel('Energy difference')
            plt.ylabel('Density')
            plt.title('Energy differences for the k=0, P and Z2 invariant sector')
            plt.show()

if __name__ == "__main__":
    start_time = time.time()
    #cProfile.run('main()')
    main()
    end_time = time.time()
    print("Execution time:", end_time - start_time, "seconds")




