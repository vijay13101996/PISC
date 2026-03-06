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


L = 20  # Length of the chain
J = 1.0  # Coupling constant
Delta = 0.55  # Anisotropy parameter

g = 0.0  # Magnetic field

#Next-nearest neighbor (NNN) coupling parameters
lamda = 1.0
J2 = lamda*J  # NNN coupling constant
Delta2 = 0.5  # NNN anisotropy parameter

k = 1 # x 2pi/L momentum sector

basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=k,zblock=1) # and positive parity sector
print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))

dynamic = []
beta = 10.0
ncoeff = 70
coeff_arr = np.arange(ncoeff)

store = True
if(1):# NN XXZ model
    try:
        #raise FileNotFoundError
        vals_NN = pickle.load(open('eigenvalues_NN_L_{}.pkl'.format(L), 'rb'))
        vecs_NN = pickle.load(open('eigenvectors_NN_L_{}.pkl'.format(L), 'rb'))
        H_mat_NN = pickle.load(open('H_mat_NN_L_{}.pkl'.format(L), 'rb'))
        print('NN eigenvalues and eigenvectors loaded from disk')
        print('shape', vals_NN.shape, vecs_NN.shape, H_mat_NN.shape)
    except FileNotFoundError:
        H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        static_NN = [["zz",H_zz_NN],["+-",H_xy_NN],["-+",H_xy_NN]]
        H_XXZ_NN = hamiltonian(static_NN,dynamic,basis=basis,dtype=np.complex128)
        H_mat_NN = H_XXZ_NN.toarray()
        print('NN Hamiltonian constructed', H_mat_NN.shape)
        E_NN = H_XXZ_NN.eigvalsh()
        vals_NN, vecs_NN = H_XXZ_NN.eigh()
        #vals_NN = np.sort(vals_NN)
        pickle.dump(vals_NN, open('eigenvalues_NN_L_{}.pkl'.format(L), 'wb'))
        pickle.dump(vecs_NN, open('eigenvectors_NN_L_{}.pkl'.format(L), 'wb'))
        pickle.dump(H_mat_NN, open('H_mat_NN_L_{}.pkl'.format(L), 'wb'))

    O_eigenbasis= construct_B(vecs_NN, vals_NN, H_mat_NN, basis, L, trunc_perc=0.96 )
    if(1):
        plt.imshow(np.abs(O_eigenbasis), aspect='auto')
        plt.colorbar()
        plt.title('Operator in energy eigenbasis (NN XXZ)')
        plt.xlabel('i')
        plt.ylabel('j')
        #plt.show()

        exit()

    b_arr = np.zeros(ncoeff)
    b_arr = K_complexity(O_eigenbasis,vals_NN,beta=beta,ncoeff=ncoeff)
    plt.scatter(coeff_arr, b_arr, label='NN XXZ')

    if store:
        pickle.dump((coeff_arr, b_arr), open('krylov_coeffs_B_NN_beta_{}.pkl'.format(beta), 'wb'))

if(1):# NNN XXZ model
    try:
        #raise FileNotFoundError
        vals_NNN = pickle.load(open('eigenvalues_NNN_L_{}_zfull.pkl'.format(L), 'rb'))
        vecs_NNN = pickle.load(open('eigenvectors_NNN_L_{}_zfull.pkl'.format(L), 'rb'))
        H_mat_NNN = pickle.load(open('H_mat_NNN_L_{}_zfull.pkl'.format(L), 'rb'))
        print('NNN eigenvalues and eigenvectors loaded from disk')
    except FileNotFoundError:
        H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        static_NNN = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
        H_XXZ_NNN = hamiltonian(static_NNN,dynamic,basis=basis,dtype=np.complex128)
        H_mat_NNN = H_XXZ_NNN.toarray()
        print('NNN Hamiltonian constructed', H_mat_NNN.shape)
        E_NNN = H_XXZ_NNN.eigvalsh()
        vals_NNN, vecs_NNN = H_XXZ_NNN.eigh()
        #vals_NNN = np.sort(vals_NNN)
        pickle.dump(vals_NNN, open('eigenvalues_NNN_L_{}_zfull.pkl'.format(L), 'wb'))
        pickle.dump(vecs_NNN, open('eigenvectors_NNN_L_{}_zfull.pkl'.format(L), 'wb'))
        pickle.dump(H_mat_NNN, open('H_mat_NNN_L_{}_zfull.pkl'.format(L), 'wb'))

    O_eigenbasis= construct_B(vecs_NNN, vals_NNN, H_mat_NNN, basis, L)
    
    b_arr = np.zeros(ncoeff)
    b_arr = K_complexity(O_eigenbasis,vals_NNN,beta=beta,ncoeff=ncoeff)
    if store:
        pickle.dump((coeff_arr, b_arr), open('krylov_coeffs_B_NNN_beta_{}.pkl'.format(beta), 'wb'))

    plt.scatter(coeff_arr, b_arr, label='NNN XXZ')
    plt.xlabel('Krylov index')
    plt.ylabel('Krylov coefficient')
    plt.legend()
    #plt.show()

