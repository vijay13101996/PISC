import numpy as np
from matplotlib import pyplot as plt
import cProfile
import time

"""
Exact diagonalization of the XXZ model with nearest-neighbor interactions.
We consider the following form:
\hat{H} = \sum_{i=1}^L \frac{1}{2} (\hat{S}_i^+\hat{S}_{i+1}^- + \hat{S}_{i}^-\hat{S}_{i+1}^+)
                        + \Delta \hat{S}_{i}^z\hat{S}_{i+1}^z
"""

L = 22  # Length of the chain

Delta = 1.0  # Anisotropy parameter

def basis(L):
    """Generate the basis states for a spin chain of length L."""
    return np.array([np.binary_repr(i, width=L) for i in range(2**L)])

def NN_XX(st2_n, st2_m):
    # Considers the S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ interaction
    if(st2_n == ['1','0'] and st2_m == ['0','1']):
        #print('st2_n:', st2_n, 'st2_m:', st2_m)
        return 1.0/2
    elif(st2_n == ['0','1'] and st2_m == ['1','0']): # Hermitian conjugate
        return 1.0/2
    else:
        return 0.0

def NN_Z(st2_n, st2_m):
    # Considers the S_i^z S_{i+1}^z interaction
    if(st2_n == st2_m):
        return 1.0/4
    else:
        return 0.0

def M_z(state):
    """Calculate the total magnetization in the z-direction."""
    return sum(1.0/2 if s == '1' else -1/2 for s in state)

def trans_inv(state):
    """Check if the state is invariant under translation."""
    """A state is translation invariant if it remains unchanged under cyclic shifts."""
    return all(state[i] == state[(i + 1) % len(state)] for i in range(len(state)))

def orbit(state):
    """Generate the orbit of a state under cyclic shifts.
    This function returns all cyclic permutations of the input state."""
    orbits = [state[i:] + state[:i] for i in range(len(state))]
    #Keep only unique orbits
    unique_orbits = set((orbit) for orbit in orbits)
    return [(orbit) for orbit in unique_orbits]

def find_Sz_0(basis_states, Mz=0.0):
    """Find basis states with total magnetization Sz = 0."""
    return [state for state in basis_states if M_z(state) == 0]

def find_orbits(basis_states):
    """Find unique orbits of basis states."""
    orbit_lst = []
    seen = []
    
    if(0):
        for state in basis_states:
            if state not in seen:
                current_orbit = orbit(state)
                orbit_lst.append(current_orbit)
                seen.extend(current_orbit) 

    if(1): #A lot more efficient
        for state in basis_states:
            current_orbit = orbit(state)
            orbit_lst.append(current_orbit)
            #print('current orbit:', state, current_orbit)
            #print('basis states:', basis_states)
            for elt in current_orbit:
                basis_states.remove(elt)  # Remove the state from the basis states to avoid duplicates

    return orbit_lst

def construct_k_states(orbit_lst,n):
    #!!!! INCOMPLETE
    """
    Construct states with momentum k=2\pi n/L given an orbit list
    These are states with 
        T\psi = e^{ik} \psi
    where k = 2\pi n/L for n = 0, 1, ..., L-1
    This function generates states with a specific momentum k from the orbits."""
    for orbit in orbit_lst:
        r = len(orbit)
        # An orbit of length r permits n = 0, 1, ..., r-1
        if r<n:
            continue

def find_even_parity(orbit_lst):
    """
    Given a list of orbits, select orbits with even parity.
    This means, for each state in an orbit, the state obtained by reversing 
    the order of spins is also in the orbit.
    """
    even_parity_orbits = []
    for orbit in orbit_lst:
        if all(state[::-1] in orbit for state in orbit):
            even_parity_orbits.append(orbit)
    return even_parity_orbits

def find_even_spin_inv(orbit_lst):
    """
    Given a list of orbits, select orbits which are invariant under spin reversal.
    This means, for each state in an orbit, the state obtained by reversing each spin
    (0 -> 1 and 1 -> 0) is also in the orbit.
    """
    even_spin_inv_orbits = []
    for orbit in orbit_lst:
        if all(''.join('1' if s == '0' else '0' for s in state) in orbit for state in orbit):
            even_spin_inv_orbits.append(orbit)
    return even_spin_inv_orbits

def hamiltonian(L, Delta):
    H = np.zeros((2**L, 2**L), dtype=np.float64)
    """Construct the Hamiltonian matrix for the XXZ model."""

    for n in range(N):
        for m in range(N):
            state_n = basis_states[n]
            state_m = basis_states[m]
            
            print('n', n, 'm', m)            
            #print('state n:', state_n, 'state m:', state_m)
            # Interaction at site i
            for i in sites:
                
                #print('\n')
                #print('state n:', state_n, 'state m:', state_m, 'i:', i)
                ind_NN = ([i,(i+1) % L]) # To ensure periodic boundary conditions
                ind_all = np.delete( sites, ind_NN)  # All sites except the nearest neighbor pair

                #print('ind_NN:', ind_NN)
                #print('ind_all:', ind_all)
                # Spin-operators act on 2-site states 
                st2_n = [state_n[k] for k in ind_NN]
                st2_m = [state_m[k] for k in ind_NN]

                #print('st2_n:', st2_n, 'st2_m:', st2_m)
                
                stall_n = [state_n[k] for k in ind_all]
                stall_m = [state_m[k] for k in ind_all]
                #print('stall_n:', stall_n, 'stall_m:', stall_m)
                if (stall_n == stall_m):
                    # Every spin except the pair i,i+1 needs to be the same
                    # If so, add the nearest-neighbor interaction terms
                    H[n, m] += NN_Z(st2_n, st2_m) + Delta * NN_XX(st2_n, st2_m)
                 
    return H

def main(L):
    start_time = time.time()
    N = 2**L  # Number of basis states
    print("Basis states:", N)
    basis_states = basis(L)
    sites = list(range(L))  # Sites in the chain

    Sz0_states = find_Sz_0(basis_states.copy())
    print("Number of states with Sz = 0:", len(Sz0_states))

    orbit_lst = find_orbits(Sz0_states)

    #print("Orbits of states with Sz = 0:",orbit_lst)

    print("Number of orbits found:", len(orbit_lst))

    #exit()
    print("Orbits of each state:")
    #for orbit in orbit_lst:
    #    print(orbit)
    #print(orbit_lst)

    even_P_orbits = find_even_parity(orbit_lst)
    #for orbit in even_P_orbits:
    #    print("Even parity orbit:", orbit)

    spin_inv_orbits = find_even_spin_inv(even_P_orbits)
    #print("Even spin inversion orbits:")
    #for orbit in spin_inv_orbits:
    #    print(orbit)

    print('final basis state length:', len(spin_inv_orbits))

    print("Time taken to find orbits:", time.time() - start_time, '\n')

    if(0):
        #exit(0)

        h = hamiltonian(L, Delta)

        print("Hamiltonian matrix:\n", h.shape)

        eigvals, eigvecs = np.linalg.eigh(h)

        diffs = np.diff(eigvals)
        plt.hist(diffs, bins=20, density=True)
        plt.xlabel('Energy differences')
        plt.ylabel('Density')
        plt.title('Energy Level Spacing Distribution')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    #cProfile.run('main()')
    for L in [22,26]:
        print(f"Running for L = {L}")
        main(L)
    
    

