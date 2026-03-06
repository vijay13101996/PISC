import numpy as np
import matplotlib.pyplot as plt
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators 
from quspin.basis import spin_basis_1d # Hilbert space spin basis  
from PISC.engine import Krylov_complexity
from Krylov_XXZ_tools import construct_A, construct_B, construct_Jz, construct_JE, construct_JE_comm
import pickle
import matplotlib

def K_complexity(O,vals,beta,ncoeff=50):
    # Compute the Krylov complexity and moments

    neigs = O.shape[0]
    vals = vals[:neigs]  # Ensure vals is the same length as O

    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    
    return barr

def define_lattice(L, J, Delta, J2, Delta2, basis, suffix):
    try:
        #raise FileNotFoundError
        vals = pickle.load(open('eigenvalues_{}.pkl'.format(suffix), 'rb'))
        vecs = pickle.load(open('eigenvectors_{}.pkl'.format(suffix), 'rb'))
        H_mat = pickle.load(open('H_mat_{}.pkl'.format(suffix), 'rb'))
        print('Eigenvalues and eigenvectors loaded from disk')
        print('shape', vals.shape, vecs.shape, H_mat.shape)
    except FileNotFoundError:
        H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        if 'NNN' in suffix:
            static = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
        else:
            if (J2!=0.0) or (Delta2!=0.0):
                print('Warning: J2 and Delta2 are nonzero but suffix does not contain NNN, ignoring J2 and Delta2')
            static = [["zz",H_zz_NN],["+-",H_xy_NN],["-+",H_xy_NN]]
        H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
        H_mat = H.toarray()
        print('Hamiltonian constructed', H_mat.shape)
        E = H.eigvalsh()
        vals, vecs = H.eigh()
        #vals = np.sort(vals)
        pickle.dump(vals, open('eigenvalues_{}.pkl'.format(suffix), 'wb'))
        pickle.dump(vecs, open('eigenvectors_{}.pkl'.format(suffix), 'wb'))
        pickle.dump(H_mat, open('H_mat_{}.pkl'.format(suffix), 'wb'))
    return vals, vecs, H_mat

def generate_O(Okey, L, J, Delta, J2, Delta2, basis, suffix):
    # Generate the operator O in the computational basis
    try:
        #raise FileNotFoundError
        vals = pickle.load(open('eigenvalues_{}.pkl'.format(suffix), 'rb'))
        vecs = pickle.load(open('eigenvectors_{}.pkl'.format(suffix), 'rb'))
        H_mat = pickle.load(open('H_mat_{}.pkl'.format(suffix), 'rb'))
        print('Eigenvalues and eigenvectors loaded from disk')
        print('shape', vals.shape, vecs.shape, H_mat.shape) 
    except FileNotFoundError:
        if 'NNN' in suffix:
            define_NNN(L, J, Delta, J2, Delta2, basis, suffix)
        else:
            define_NN(L, J, Delta, basis, suffix)
        vals = pickle.load(open('eigenvalues_{}.pkl'.format(suffix), 'rb'))
        vecs = pickle.load(open('eigenvectors_{}.pkl'.format(suffix), 'rb'))
        H_mat = pickle.load(open('H_mat_{}.pkl'.format(suffix), 'rb'))
        print('Eigenvalues and eigenvectors loaded from disk')
        print('shape', vals.shape, vecs.shape, H_mat.shape)
    try:
        O = pickle.load(open('{}_{}.pkl'.format(Okey, suffix), 'rb'))
        print('Operator {} loaded from disk'.format(Okey))
        print('shape', O.shape)
        return O
    except FileNotFoundError:
        if Okey == 'A':
            O = construct_A(vecs, vals, H_mat, basis, L, trunc_perc=0.96 )
        if Okey == 'B':    
            O = construct_B(vecs, vals, H_mat, basis, L, trunc_perc=0.96 )
        elif Okey == 'Jz':
            O = construct_Jz(vecs, vals, H_mat, basis, L, trunc_perc=0.96 )
        elif Okey == 'JE':
            O = construct_JE_comm(vecs, vals, H_mat, basis, L, trunc_perc=0.96 )
        pickle.dump(O, open('{}_{}.pkl'.format(Okey, suffix), 'wb'))
    return O

L = 20  # Length of the chain
J = 1.0  # Coupling constant
Delta = 0.55  # Anisotropy parameter

g = 0.0 # Magnetic field

#Next-nearest neighbor (NNN) coupling parameters
lamda = 1.0
J2 = lamda*J  # NNN coupling constant
Delta2 = 0.5  # NNN anisotropy parameter

k = 1 # x 2pi/L momentum sector

basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=k,zblock=1) # and positive parity sector
print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))


dynamic = []
beta = 1.0
ncoeff = 70
coeff_arr = np.arange(ncoeff)

store = False
fig,ax = plt.subplots(1,2, figsize=(6,3), sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.15, wspace=0.0)

suffix_NN = 'NN_L_{}'.format(L)
suffix_NNN = 'NNN_L_{}_zfull'.format(L)

vals_NN, vecs_NN, H_mat_NN = define_lattice(L, J, Delta, J2, Delta2, basis, suffix_NN)
vals_NNN, vecs_NNN, H_mat_NNN = define_lattice(L, J, Delta, J2, Delta2, basis, suffix_NNN)

Okey = 'A'
O_NN = generate_O(Okey, L, J, Delta, J2, Delta2, basis, suffix_NN)
O_NNN = generate_O(Okey, L, J, Delta, J2, Delta2, basis, suffix_NNN)

neigs = len(O_NN)

for O, vals, key in zip([O_NN, O_NNN], [vals_NN, vals_NNN], ['NN', 'NNN']):

    b_arr = np.zeros(ncoeff)
    b_arr = K_complexity(O,vals,beta=beta,ncoeff=ncoeff)
    ax[0].scatter(coeff_arr, b_arr, label='{} XXZ'.format(key),s=5)

    for i in range(neigs):
        for j in range(neigs):
            if abs(i-j) <=10:
                O[i,j] = 0.0
        #O[i,i] = 0.0

    b_arr = np.zeros(ncoeff)
    b_arr = K_complexity(O,vals,beta=beta,ncoeff=ncoeff)
    ax[1].scatter(coeff_arr, b_arr, label='{} XXZ'.format(key),s=5)

plt.show()

"""
The only conclusion so far is that the NN zz interaction and the NNN spin-flip operators for the
NNN XXZ model transitions from staggered to strictly linear when the diagonal elements of O 
are set to zero, which seems to suggest that the diagonal elements of O are responsible 
for the staggered behavior. 

However, this does not explain why the NN XXZ does not show this transition
when the diagonal elements are set to zero for either of these operators. 
Further investigation is needed to understand the underlying reasons for these observations.
"""
