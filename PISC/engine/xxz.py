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
from numba import njit, prange
import pickle

"""
Exact diagonalization of the XXZ model with nearest-neighbor interactions.
We consider the following form:
\hat{H} = J {\sum_{i=1}^L \frac{1}{2} (\hat{S}_i^+\hat{S}_{i+1}^- + \hat{S}_{i}^-\hat{S}_{i+1}^+)
                        + \Delta \hat{S}_{i}^z\hat{S}_{i+1}^z}

The operators \hat{S}_i^+ and \hat{S}_i^- are the raising and lowering operators for the spin at site i,
and \hat{S}_i^z is the z-component of the spin operator at site i.

Note that these are NOT the Pauli matrices, but rather the spin-1/2 operators.
"""

class XXZ:
    def __init__(self, L, J=1.0, Delta=1.0, g=0.0, J2=0.0, Delta2=None):
        self.L = L  # Length of the chain
        self.J = J  # Coupling constant
        self.g = g  # Magnetic-field
        self.Delta = Delta  # Anisotropy parameter

        self.J2 = J2  # Next-nearest-neighbor coupling constant
        self.Delta2 = Delta2 # Anisotropy parameter for next-nearest-neighbor interactions

        self.sites = np.arange(self.L)  # Sites in the chain
        self.nn_pairs = [(i, (i + 1) % self.L) for i in range(self.L)]  # Nearest-neighbor pairs with periodic boundary conditions
        self.nnn_pairs = [(i, (i + 2) % self.L) for i in range(self.L)]  # Next-nearest-neighbor pairs with periodic boundary conditions

        # By default, we generate the full basis states for the spin chain

        # We also group states based on the orbit they belong to. Each orbit is a 
        # set of states that can be transformed into each other by cyclic shifts.
        # The magnetization of the states in each orbit is the same.
        
        #Store basis and orbits for a given L into a pickle file
        pkl_name = f'xxz_basis_orbits_L{self.L}.pickle'
        try:
            with open(pkl_name, 'rb') as f:
                self.basis_states, self.orbits = pickle.load(f)
                print(f'Loaded basis states and orbits from {pkl_name}')
        except FileNotFoundError:
            print(f'File {pkl_name} not found. Generating basis states and orbits.')
            
            self.basis_states = list(self.generate_basis())
            print('Number of basis states:', len(self.basis_states))
            
            self.orbits = self.find_orbits(self.basis_states.copy())

            with open(pkl_name, 'wb') as f:
                pickle.dump((self.basis_states, self.orbits), f)
                print(f'Saved basis states and orbits to {pkl_name}')


    def generate_basis(self):
        """
        Generate all the basis states for a spin chain of length L.
        The full basis set consists of all binary strings of length L,
        where '1' represents a spin up and '0' represents a spin down.

        The total number of basis states is 2^L, and each state can be represented
        as a binary string of length L. For example, for L=3, the basis states are:
        ['000', '001', '010', '011', '100', '101', '110', '111'].
        """
        return np.array([np.binary_repr(i, width=self.L) for i in range(2**self.L)])

    def order_unique_orbit(self, orbit):
        """
        Given an orbit which has the right ordering, return elements which are unique.
        """
        seen = set() 
        return [x for x in orbit if not (x in seen or seen.add(x))]

    def T_op(self, state, n=1):
        """
        Apply the translation operator T^n to the state.
        The translation operator shifts the state by n sites.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        n = n % self.L
        return state[-n:] + state[:-n]  # Cyclic shift to the right by n positions


    def orbit(self, state):
        """
        Generate the orbit of a state under cyclic shifts.
        This function returns all cyclic permutations of the input state.
        """
        #orbit = [state[i:] + state[:i] for i in range(len(state))]
        orbit = [self.T_op(state,i) for i in range(len(state))]  # Generate cyclic shifts

        #Keep only unique states
        unique_orbit = set((state) for state in orbit)
        return [(state) for state in unique_orbit]

    def M_z(self, state):
        """
        Calculate the total magnetization in the z-direction.
        """        
        return sum(1.0/2 if s == '1' else -1/2 for s in state)
    
    def trans_inv(self, state):
        """
        Check if the state is invariant under translation.
        A state is translation invariant if it remains unchanged under cyclic shifts.
        """
        return all(state[i] == state[(i + 1) % len(state)] for i in range(len(state)))

    def flip_spin(self, state, sites):
        """
        Flip the spins at the specified sites in the state.
        Thesites are given as a list of indices.
        """
        state_list = list(state)
        for site in sites:
            state_list[site] = '1' if state_list[site] == '0' else '0'
        return ''.join(state_list)

    def find_orbits(self, basis_states):
        """
        Find unique orbits of the states specified in  basis states.
        """

        orbit_lst = []
        seen = []
        basis_lst = basis_states.copy()
        if(0):
            for state in basis_st:
                if state not in seen:
                    current_orbit = self.orbit(state)
                    orbit_lst.append(current_orbit)
                    seen.extend(current_orbit) 

        if(1): #A lot more efficient
            while basis_lst:
                state = basis_lst[0]
                current_orbit = self.orbit(state)
                orbit_lst.append(set(current_orbit)) # Append the current orbit to the list
                for elt in current_orbit:
                    basis_lst.remove(elt)  # Remove the state from the basis states to avoid duplicates
                
        return orbit_lst

    def find_Sz_0(self, basis_states, Mz=0.0):
        """
        Filter out basis states with total magnetization Mz.
        """
        return [state for state in basis_states if self.M_z(state) == Mz]

    def P_operator(self, states):
        """
        Parity operator: Generate the state obtained by 
        reversing the order of spins in the input state.
        
        If the input is an orbit, then return the orbit obtained 
        by reversing the order of spins in each state.
        """
        
        if isinstance(states, str):
            return states[::-1]
        elif isinstance(states, list):
            return [s[::-1] for s in states]

    def Z2_operator(self, states):
        """
        Spin flip operator: Generate the state obtained by flipping each spin in the input state.
        This means replacing '0' with '1' and '1' with '0'.

        If the input is an orbit, then return the orbit obtained
        by flipping each spin in each state.

        """
        if isinstance(states, str):
            return ''.join('1' if s == '0' else '0' for s in states)
        elif isinstance(states, list):
            return [''.join('1' if s == '0' else '0' for s in state_i) for state_i in states]

    def L_orbits(self, orbit_lst):
        """
        Given a list of orbits, return the orbits that have the same length as the system size L.
        """
        L_orbits = []
        for orbit in orbit_lst:
            if len(orbit) == self.L:
                L_orbits.append(orbit)
        return L_orbits

    def pair_orbits(self, orbit_lst, O):
        """
        Given a list of orbits, pair the orbits with that generated by the operator O.
        """

        paired_orbits = [] 
        ob_lst_copy = orbit_lst.copy()  # Make a copy of the orbit list to avoid modifying it during iteration

        while ob_lst_copy:
            orbit = list(ob_lst_copy[0])
            Oorbit = set(O(orbit))
            orbit = set(orbit)  # Convert to set for uniqueness 
            #print('Orbit:', orbit, 'Oorbit:', Oorbit)

            unique = set(map(frozenset, [orbit, Oorbit]))
            paired_orbits.append(list(map(set, set(map(frozenset, unique)))))
            for elt in unique:
                if set(elt) in ob_lst_copy:
                    ob_lst_copy.remove(set(elt))


        return paired_orbits

    def group_orbits(self, orbit_lst, O1, O2):
        """

        Given a list of orbits, group the orbits based on the action of two operators O1 and O2.
        We consider the following grouping:

        orbit -> O1(orbit) -> O2(orbit) -> O1(O2(orbit)) -> O2(O1(orbit))

        """

        grouped_orbits = []
        ob_lst_copy = orbit_lst.copy()

        while ob_lst_copy:
            orbit = list(ob_lst_copy[0])
            O1orbit = set(O1(orbit))
            O2orbit = set(O2(orbit))
            O1O2orbit = set(O1(O2(orbit)))
            #O2O1orbit = set(O2(O1(orbit))) # There could be cases where O1 and O2 do not commute, so we need to consider both orders
            orbit = set(orbit)  # Convert to set for uniqueness 

            #print('Orbit:', orbit, '\n O1orbit:', O1orbit, '\n O2orbit:', O2orbit, '\n O1O2orbit:', O1O2orbit, '\n O2O1orbit:', O2O1orbit)

            unique = set(map(frozenset, [orbit, O1orbit, O2orbit, O1O2orbit])) #  , O2O1orbit]))
            grouped_orbits.append(list(map(set, set(map(frozenset, unique)))))
            for elt in unique:
                if set(elt) in ob_lst_copy:
                    ob_lst_copy.remove(set(elt))  # Remove the orbit and its transformed orbits from the copy list
            
        return grouped_orbits

    def find_even_spin_inv(self, orbit_lst):
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

    def h2_XX(self, st2_n, st2_m):
        """
        Considers the two-site interaction term along the X and Y direction
        We consider the following form:
        h2_XX = 1/2 (S_i^+ S_j^- + S_i^- S_j^+)
        """

        if(st2_n == ['1','0'] and st2_m == ['0','1']):
            #print('st2_n:', st2_n, 'st2_m:', st2_m)
            return 1.0/2
        elif(st2_n == ['0','1'] and st2_m == ['1','0']): # Hermitian conjugate
            return 1.0/2
        else:
            return 0.0

    def h2_Z(self, st2_n, st2_m):
        """
        Considers the two-site interaction along the Z direction of the form:
        h2_Z = S_i^z S_j^z
        """
        if (st2_n == st2_m):
            if st2_n[0] == st2_n[1]:
                # Both spins are the same
                return 1.0/4
            else:
                # Spins are different
                return -1.0/4
        else:
            return 0.0

    def H_two_site(self, state_n, state_m, ind_list, J, Delta):
        """
        Compute the interactions between the spins at sites specified by ind_list
        in the XXZ model. This function considers the interaction between two spins
        at the specified indices and returns the corresponding Hamiltonian term.

        Here, the arguments state_n and state_m are assumed to be the full Pauli string
        """
        h2site = 0.0
        if(0):
            for i,j in ind_list: #np.delete is very expensive
                ind_all = np.delete(self.sites, [i,j])  # All sites except the pair i,j

                st2_n = [state_n[k] for k in [i,j]]
                st2_m = [state_m[k] for k in [i,j]]
                st_all_n = [state_n[k] for k in ind_all]
                st_all_m = [state_m[k] for k in ind_all]

                flip_n = self.flip_spin(state_n, [i,j])
                if(flip_n == state_m and st2_n[0]!=st2_n[1]):
                    print('flip_n:', flip_n, 'state_n', state_n, 'state_m:', state_m, i,j)
                
                if (st_all_n == st_all_m):
                    # Every spin except the pair i,j needs to be the same
                    # If so, add the nearest-neighbor interaction terms
                    xxh2 = J*self.h2_XX(st2_n, st2_m)
                    h2site += J*self.h2_XX(st2_n, st2_m) + J*Delta*self.h2_Z(st2_n, st2_m)
                    if(xxh2!=0.0):
                        print('non zero xxh2', flip_n, state_n, state_m,'\n')
            return h2site
        
        if(1):
            for i,j in ind_list:
                st2_n = [state_n[k] for k in [i,j]]
                st2_m = [state_m[k] for k in [i,j]]
                if(state_n == state_m):
                    h2site += J*Delta*self.h2_Z(st2_n, st2_m)

                flip_n = self.flip_spin(state_n, [i,j])
                if (flip_n == state_m and st2_n[0]!=st2_n[1]):
                    # Every spin except the pair i,j needs to be the same
                    # If so, add the nearest-neighbor interaction terms
                    h2site += J*self.h2_XX(st2_n, st2_m) 
                    #print('flip_n:', flip_n, 'state_n', state_n, 'state_m:', state_m, i,j)
                    #print('h2site')
            return h2site


    def H_NN(self, state_n, state_m):
        """
        Compute the nearest-neighbor interaction Hamiltonian for the XXZ model.
        This function considers the interaction between nearest-neighbor spins
        and returns the corresponding Hamiltonian term.

        Here, the arguments state_n and state_m are assumed to be the full Pauli string
        """

        hnn = 0.0

        if(0):
            for i, j in self.nn_pairs:
                if(state_n == state_m):
                    hnn += self.J*self.Delta*self.h2_Z([state_n[i], state_n[j]], [state_m[i], state_m[j]])
                if(state_n == self.flip_spin(state_m, [i, j])):
                    # If the states are related by a spin flip at the nearest-neighbor pair
                    hnn += self.J*self.h2_XX([state_n[i], state_n[j]], [state_m[i], state_m[j]]) 
            return hnn

        

        if(1):
            for i in self.sites:
                #print('\n')
                #print('state n:', state_n, 'state m:', state_m, 'i:', i)
                ind_NN = ([i,(i+1) % self.L]) # To ensure periodic boundary conditions
                ind_all = np.delete( self.sites, ind_NN)  # All sites except the nearest neighbor pair

                #print('ind_NN:', ind_NN)
                #print('ind_all:', ind_all)
                # Spin-operators act on 2-site states 
                st2_n = [state_n[k] for k in ind_NN]
                st2_m = [state_m[k] for k in ind_NN]
                #print('i', i,'st', state_n, state_m, 'st2_n:', st2_n, 'st2_m:', st2_m)

                st_all_n = [state_n[k] for k in ind_all]
                st_all_m = [state_m[k] for k in ind_all]

                if (st_all_n == st_all_m):
                    # Every spin except the pair i,i+1 needs to be the same
                    # If so, add the nearest-neighbor interaction terms
                    ret= self.J*self.h2_XX(st2_n, st2_m) + self.J*self.Delta*self.h2_Z(st2_n, st2_m)
                    hnn += ret

        return hnn

    def H_NNN(self, state_n, state_m):
        """
        Compute the next-nearest-neighbor interaction Hamiltonian for the XXZ model.
        This function considers the interaction between next-nearest-neighbor spins
        and returns the corresponding Hamiltonian term.

        Here, the arguments state_n and state_m are assumed to be the full Pauli string
        """

        hnnn = 0.0
        for i in self.sites:
            ind_NN = ([i,(i+2) % self.L]) # To ensure periodic boundary conditions
            ind_all = np.delete( self.sites, ind_NN)  # All sites except the next-nearest-neighbor pair

            # Spin-operators act on 2-site states 
            st2_n = [state_n[k] for k in ind_NN]
            st2_m = [state_m[k] for k in ind_NN]

            st_all_n = [state_n[k] for k in ind_all]
            st_all_m = [state_m[k] for k in ind_all]

            if (st_all_n == st_all_m):
                # Every spin except the pair i,i+2 needs to be the same
                # If so, add the nearest-neighbor interaction terms
                ret= self.J2*self.h2_XX(st2_n, st2_m) + self.J*self.Delta2*self.h2_Z(st2_n, st2_m)
                hnnn += ret

        return hnnn

    def H_B(self, state_n, state_m):
        """
        Considers interaction with the magnetic field, which is assumed to be in the z-direction.
        """
        if (state_n!= state_m):
            return 0.0
        else:
            #print('state_n:', state_n, 'state_m:', state_m,self.g)
            return np.sum([1.0/2 if s == '1' else -1.0/2 for s in state_n])*self.g

    def H(self, basis_states=None, B=False, NNN=False):
        """
        Construct the Hamiltonian matrix for the XXZ model,
        with the given basis states. Note that this is a full matrix,
        so it can be very large for larger systems.

        A separate sparse version of the Hamiltonian is to be implemented later.
        """
        if basis_states is None:
            print('Using default basis states')
            basis_states = self.basis_states

        N = len(basis_states)
        H = np.zeros((N, N), dtype=np.float64)
        sites = np.arange(self.L)  # Sites in the chain
        if(1):
            for n in range(N):
                print(n, 'th row' )
                for m in range(n, N):
                    state_n = basis_states[n]
                    state_m = basis_states[m]
                                
                    #print('state n:', state_n, 'state m:', state_m)
                
                    ## Nearest-neighbor interaction at site i
                    #H[n, m] += self.H_NN(state_n, state_m) 
                    H[n, m] += self.H_two_site(state_n, state_m, self.nn_pairs, self.J, self.Delta)
                    
                    if B:
                        H[n, m] += self.H_B(state_n, state_m)

                    if NNN:
                        # Implement next-nearest-neighbor interactions if needed
                        #H[n, m] += self.H_NNN(state_n, state_m)
                        H[n, m] += self.H_two_site(state_n, state_m, self.nnn_pairs, self.J2, self.Delta2)

                    H[m, n] = H[n, m]  # Ensure Hermiticity
        else:
            # Using numba for parallel computation
            set_H(H, basis_states, J=self.J, Delta=self.Delta, NNN=NNN)
        return H

    def find_E(self, state):
        """
        Calculate the energy of a given state in the XXZ model.
        """
        return self.H_NN(state, state)

    def H_k0(self, ob1, ob2, B=False, NNN=False):
        """
        Calculate the matrix elements of two states with k=0 (i.e. translation invariant states).
        The inputs are ob1 and ob2, which are orbits of a given state.
        We take the first state in each of the orbits as representative, say s1 and s2.

        The matrix elements are given as:
        <ob1|H|ob2> = \frac{\sqrt{l_i}}{\sqrt{l_j}} \sum_{n=0}^{l_j-1} <s1|H T^n |s2>

        where l_i and l_j are the lengths of the orbits ob1 and ob2, respectively
        and T is the translation operator that shifts the state by one site.
        """

        s1 = list(ob1)[0]  # Take the first state in the orbit as representative
        l_i = len(ob1)
        l_j = len(ob2)

        Hij = 0.0
        for n in range(l_j):
            # Apply the translation operator T^n to s2 
            sj = list(ob2)[n]  # Take the n-th state in the orbit ob2 - this is indeed T^k s2 for some k
            
            #Hij+= self.H_NN(s1, sj)  # Calculate the matrix element <s1|H|T^n s2>
            Hij+= self.H_two_site(s1, sj, self.nn_pairs, self.J, self.Delta)  # Add the two-site interaction term

            if B:
                Hij += self.H_B(s1, sj)

            if NNN:
                #Hij += self.H_NNN(s1, sj)
                Hij += self.H_two_site(s1, sj, self.nnn_pairs, self.J2, self.Delta2)  # Add the next-nearest-neighbor interaction term
        Hij *= np.sqrt(l_i/l_j)  # Normalize by the square root of the lengths of the orbits
        return Hij

    def H_k2piL(self, ob1, ob2, B=False, NNN=False):
        """
        Calculate the matrix elements of two states with k=2pi/L
        The inputs are ob1 and ob2, which are orbits of a given state.
        We take the first state in each of the orbits as representative, say s1 and s2.

        The matrix elements are given as:
        <ob1|H|ob2> = \sum_{n=0}^{L-1} e^{-2\pi n i/ L} <s1|H T^n |s2>

        where l_i and l_j are the lengths of the orbits ob1 and ob2, respectively
        and T is the translation operator that shifts the state by one site.
        """

        s1 = list(ob1)[0]
        s2 = list(ob2)[0]  # Take the first state in the orbit as representative 
        l_i = len(ob1)
        l_j = len(ob2) 

        if l_i != l_j or l_i != self.L:
            raise ValueError("Orbits must have length L for k=2pi/L calculation.")

        Hij = 0.0
        for n in range(self.L):
            Hijn = 0.0
            
            # Apply the translation operator T^n to s2 
            sj = self.T_op(s2, n)

            # Calculate the matrix element <s1|H|T^n s2>
            pref = np.exp(-2j * np.pi * n / self.L)  # Phase factor for k=2pi/L
            
            Hijn += self.H_two_site(s1, sj, self.nn_pairs, self.J, self.Delta)  # Add the two-site interaction term
            if B:
                Hijn += self.H_B(s1, sj)
            if NNN:
                Hijn += self.H_two_site(s1, sj, self.nnn_pairs, self.J2, self.Delta2)
            Hij += Hijn * pref # Multiply by the phase factor
        return Hij

    def H_symm_adapted(self, ob_gp1, ob_gp2, B=False, NNN=False, k=0):
        """
        We consider the matrix elements of the Hamiltonian in the k=0 sector
        or k=2pi/L sector, depending on the value of k, between two 
        symmetry-adapted orbits or orbit groups ob_gp1 and ob_gp2 
        which are defined by operators O such that O^2 = I.

        We assume that the operators are translation invariant, 
        i.e. they commute with the translation operator T.

        Given an orbit ob, there are two possibilities for the 
        symmetry-adapted orbits generated by O:

        1. O(ob) = ob, in which we consider ob as the basis set element
        2. O(ob) != ob, in which case we consider
            \psi = 1/\sqrt{2}*(ob + O(ob)) as the basis set element
 
        Note that, in case we consider more than one such operators, 
        then the orbit group can be longer than 2 orbits. Say, we consider 
        O1 and O2 that commute, we can potentially have an orbit group that
        has 4 elements that define the following basis element:
            \psi = 1/2*(ob + O1(ob) + O2(ob) + O1(O2(ob)))

        We can also consider the case where O1 and O2 do not commute and the 
        case of more than 2 operators, but we will not consider this for now.
        
        The matrix elements for momentum k are calculated for ob_gp1 and ob_gp2, as follows:
        <ob_gp1|H|ob_gp2> = \sum_{ob1 in ob_gp1, ob2 in ob_gp2} H_k(ob1, ob2) / 
                            np.sqrt(len(ob_gp1) * len(ob_gp2))

        where ob_gp1 and ob_gp2 are the orbit groups defined by O1 and O2, respectively.
        """

        Hij = 0.0
        l_i = len(ob_gp1)
        l_j = len(ob_gp2)

        for ob1 in ob_gp1:
            for ob2 in ob_gp2:
                if k == 0:
                    Hij += self.H_k0(ob1, ob2, B=B, NNN=NNN)
                elif k == 2 * np.pi / self.L:
                    Hij += self.H_k2piL(ob1, ob2, B=B, NNN=NNN)

        Hij /= np.sqrt(l_i * l_j)
        return Hij

    def compute_H_symm(self, gr_orbits, parallel=True, k=0):
        """
        Compute the symmetric Hamiltonian matrix for the XXZ model using the groups of orbits.
        xxz: Instance of the XXZ class
        gr_orbits: Groups of orbits identified by the P and Z2 symmetries

        Returns:
        H_symm: Symmetric Hamiltonian matrix
        """
        
        H_symm = np.zeros((len(gr_orbits), len(gr_orbits)), dtype=float)
        
        if parallel:
            # Use parallel processing to compute the rows of the symmetric Hamiltonian matrix
            n_jobs = 20
            rows = Parallel(n_jobs=n_jobs)(delayed(compute_H_symm_row)(xxz, i, gr_orbits, k) for i in range(len(gr_orbits)))

            for i, row in enumerate(rows):
                print(f"Calculating H_symm_row({i})")
                H_symm[i, i:] = row
                H_symm[i:, i] = row
            return H_symm
        
        else:
            for i in range(len(gr_orbits)):
                print("Calculating H_symm_row(", i, ")")
                row = compute_H_symm_row(xxz, i, gr_orbits)
                H_symm[i, i:] = row
                H_symm[i:, i] = row

            return H_symm

@njit
def NN_XX(st2_n, st2_m):
    # Considers the S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ interaction
    if(st2_n == ['1','0'] and st2_m == ['0','1']):
        #print('st2_n:', st2_n, 'st2_m:', st2_m)
        return 1.0/2
    elif(st2_n == ['0','1'] and st2_m == ['1','0']): # Hermitian conjugate
        return 1.0/2
    else:
        return 0.0

@njit
def NN_Z(st2_n, st2_m):
    # Considers the S_i^z S_{i+1}^z interaction
    if(st2_n == st2_m):
        if st2_n[0] == st2_n[1]:
            # Both spins are the same
            return 1.0/4
        else:
            # Spins are different
            return -1.0/4
    else:
        return 0.0

@njit(parallel=True)
def H_NN(state_n, state_m, J, Delta, L):
        sites = np.arange(L)  # Sites in the chain
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

            st_all_n = [state_n[k] for k in ind_all]
            st_all_m = [state_m[k] for k in ind_all]
            #print('st_all_n:', st_all_n, 'st_all_m:', st_all_m)

            if (st_all_n == st_all_m):
                # Every spin except the pair i,i+1 needs to be the same
                # If so, add the nearest-neighbor interaction terms
                ret= J*NN_XX(st2_n, st2_m) + J*Delta*NN_Z(st2_n, st2_m)
                
                ret1 = J*NN_XX(st2_n, st2_m)
                ret2 = J*Delta*NN_Z(st2_n, st2_m)
                #print('ret:', ret1, ret2, J)
                return ret
            else:
                return 0.0


@njit(parallel=True)
def set_H(H, basis_states, J, Delta, NNN):
    """
    Construct the Hamiltonian matrix for the XXZ model,
    with the given basis states. Note that this is a full matrix,
    so it can be very large for larger systems.
    """
    N = len(basis_states)
    sites = np.arange(len(basis_states[0]))
    L = len(sites)  # Length of the chain
    for n in prange(N):
        for m in prange(n, N):
            state_n = basis_states[n]
            state_m = basis_states[m]
            #print('n', n, 'm', m)            
            #print('state n:', state_n, 'state m:', state_m)
        
            ## Nearest-neighbor interaction at site i
            H[n, m] += H_NN(state_n, state_m, J, Delta, L)

            if NNN:
                # Implement next-nearest-neighbor interactions if needed
                # This part is left as an exercise for the user.
                pass
        
            H[m, n] = H[n, m]  # Ensure Hermiticity

