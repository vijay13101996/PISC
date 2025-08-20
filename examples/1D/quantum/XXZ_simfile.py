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
"""

def main():

    start_time = time.time()
    L = 18  # Length of the chain
    J = 1.0  # Coupling constant
    Delta = 0.1  # Anisotropy parameter
    g= 0.0 # Magnetic field

    J2 = 1.0 # NNN coupling constant
    Delta2 = 1.0 # NNN anisotropy parameter

    #Hrand = np.zeros((2**L, 2**L), dtype=np.complex128)

    xxz = XXZ(L, J, Delta, g, J2, Delta2) # Initialises the full 2^L basis states
    print(len(xxz.basis_states), "basis states")
    #E0 = xxz.find_E('0000000000')  # Ground state energy
    #print("Ground state energy:", E0)

    #H_full = xxz.H(NNN=True)  # Full Hamiltonian with magnetic field and NNN interactions
    
    #H2 = np.zeros((len(xxz.basis_states), len(xxz.basis_states)), dtype=float)

    if(0):
        for i in range(len(xxz.basis_states)):
            for j in range(len(xxz.basis_states)):
                state_i = xxz.basis_states[i]
                state_j = xxz.basis_states[j]
                #Hij = xxz.H_NNN(state_i, state_j) + xxz.H_NN(state_i, state_j)
                Hij2 = xxz.H_two_site(state_i, state_j, xxz.nnn_pairs, xxz.J2, xxz.Delta2) +\
                        xxz.H_two_site(state_i, state_j, xxz.nn_pairs, xxz.J, xxz.Delta)
                H2[i,j] = Hij2
                print('i,j',i,j)
            
                #assert (Hij == Hij2), f"Mismatch in Hamiltonian elements: H({state_i}, {state_j}) = {Hij}, H_two_site = {Hij2}"

        assert (H_full == H2).all(), "Mismatch in Hamiltonian elements: H_full != H_two_site"
    #exit()
    
    #print('Full Hamiltonian shape:', H_full.shape, 'Expected:', (2**L, 2**L))
    #vals_full, vecs_full = np.linalg.eigh(H_full)

    # Restrict basis states to those with M_z = 0
    basis_Mz0 = xxz.find_Sz_0(xxz.basis_states, Mz=0)
    xxz.basis_states = basis_Mz0
    #for state in xxz.basis_states:
    #    print(state, xxz.M_z(state), xxz.find_E(state))
    print("Number of basis states with M_z = 0:", len(xxz.basis_states), np.math.comb(L, L//2))

    #H_Sz0 = xxz.H(NNN=True,basis_states=basis_Mz0)  # Hamiltonian restricted to M_z = 0 sector
    #vals_Sz0, vecs_Sz0 = np.linalg.eigh(H_Sz0)

    #Find orbits of the basis states - to get k=0 sector
    orbits = xxz.find_orbits(xxz.basis_states)
    print("Number of orbits:", len(orbits), sum(len(orbit) for orbit in orbits), "states in total")

    gr_orbits = xxz.group_orbits(orbits, xxz.P_operator, xxz.Z2_operator)
    #gr_orbits = xxz.pair_orbits(orbits, xxz.Z2_operator)

    print("Number of groups of orbits:", len(gr_orbits), sum(len(group) for group in gr_orbits), "states in total")

    count=0
    for i, group in enumerate(gr_orbits):
        #print("Group", i, ":", group)
        #print("Number of states in group:", len(group))
        #print("States in orbit:", orbit)
        count += len(group)

    print("Total number of states in all orbits:", count)

    #exit(0)

    H_symm= np.zeros((len(gr_orbits), len(gr_orbits)), dtype=float)
    H_symm2 = np.zeros((len(gr_orbits), len(gr_orbits)), dtype=float) 

    start_time = time.time()

    def compute_H_symm_row(i, gr_orbits):
        H_row = np.zeros(len(gr_orbits)-i, dtype=float)
        for j in range(i, len(gr_orbits)):
            H_row[j-i] = xxz.H_symm_adapted(gr_orbits[i], gr_orbits[j])
        return H_row
        if(0):
            for j in range(i, len(gr_orbits)):
               H_symm[i, j] = xxz.H_symm_adapted(gr_orbits[i], gr_orbits[j])
               H_symm[j, i] = H_symm[i, j]
    
    if(0):
        for i in range(len(gr_orbits)):
            print("Calculating H(", i, ")",time.time() - start_time, "seconds elapsed")
            for j in range(i,len(gr_orbits)):
                #print("Calculating H(", i, ",", j, ")")
                H_symm[i,j] = xxz.H_symm_adapted(gr_orbits[i], gr_orbits[j])
                H_symm[j,i] = H_symm[i,j]
                #print("Symmetric matrix element H(", i, ",", j, ") =", H_symm[i,j])


    if(0):
        for i in range(len(gr_orbits)):
            print("Calculating H_symm_row(", i, ")", time.time() - start_time, "seconds elapsed")
            row = compute_H_symm_row(i, gr_orbits)
            H_symm[i, i:] = row
            H_symm[i:, i] = row

    if(1):
        n_jobs = 10
        rows = Parallel(n_jobs=n_jobs)(delayed(compute_H_symm_row)(i, gr_orbits) for i in range(len(gr_orbits)))

        for i, row in enumerate(rows):
            print(f"Calculating H_symm_row({i})", time.time() - start_time, "seconds elapsed")
            H_symm2[i, i:] = row
            H_symm2[i:, i] = row


    assert np.allclose(H_symm, H_symm2), "Mismatch in symmetric Hamiltonian matrix: H_symm != H_symm2"


    #vals, vecs = np.linalg.eigh(H_symm)



    if(0):
        #print eigenvalues found in all 3 of vals, vals_full, vals_Sz0
        vals = np.around(vals, 4)
        vals_full = np.around(vals_full, 4)
        vals_Sz0 = np.around(vals_Sz0, 4)

        inter = set.intersection(set(vals), set(vals_full), set(vals_Sz0))
        print(set(vals)==inter, inter, set(vals))


        plt.scatter(range(len(vals)), vals, marker='o')
        plt.scatter(range(len(vals_full)), vals_full, marker='x', color='red')
        plt.scatter(range(len(vals_Sz0)), vals_Sz0, marker='^', color='green')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues of the Hamiltonian')
        plt.legend(['Symmetric', 'Full', 'M_z=0'])
        plt.grid()
        plt.show()


    if(0):
        diff = np.diff(vals)

        bins = np.arange(0,3.01,0.1)

        plt.hist(diff, bins=bins, density=True)
        plt.xlabel('Energy difference')
        plt.ylabel('Density')
        plt.title('Histogram of Energy Differences')
        plt.grid()
        plt.show()



if __name__ == "__main__":
    start_time = time.time()
    #cProfile.run('main()')
    main()
    end_time = time.time()
    print("Execution time:", end_time - start_time, "seconds")




