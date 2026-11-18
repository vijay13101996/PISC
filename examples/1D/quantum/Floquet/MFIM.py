import numpy as np
from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, compute_U_F, compute_U_F_direct
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators 
from quspin.basis import spin_basis_1d # Hilbert space spin basis  

"""
This code is to benchmark Floquet dynamics in spin chains using the QuSpin package. 
The model considered is the transverse field Ising model (TFIM) with a longitudinal field. 

The Hamiltonian is given by:
H(t) = H_IM + H_x(t)
where H_IM is the Ising model Hamiltonian with a longitudinal field, 
and H_x(t) is the transverse field Hamiltonian that is periodically driven.

The parameters are taken from:
Weinberg, Phillip, and Marin Bukov. 
"QuSpin: a Python package for dynamics and exact diagonalisation of quantum many body systems part I: spin chains." 
SciPost Physics 2.1 (2017): 003.

The protocol is thus:
    1. Propagate U_x_op(t) = exp(-i H_x t) for time t = T/4
    2. Propagate U_IM(t) = exp(-i H_IM t) for time t = T/2
    3. Propagate U_x_op(t) = exp(-i H_x t) for time t = T/4
    4. U(T) = U_x_op(T/4) * U_IM(T/2) * U_x_op(T/4) 

So far, only trivial quantities such as the expectation value of the middle spin and the fidelity with the initial state have been computed.
Proper comparison with energy density expectation values to be done later.

"""

L = 14  # Length of the chain
J = 1.0  # Coupling constant
hx = 0.809 # Transverse field strength
hz = 0.0#9045 # Longitudinal field strength

basis = spin_basis_1d(L,pauli=True,kblock=0,pblock=1) # and positive parity sector

full_basis = spin_basis_1d(L,pauli=True) # full basis without symmetry blocks

H_zz_NN = [[J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
H_z = [[hz, i] for i in range(L)]
H_x = [[hx, i] for i in range(L)]

static = [["zz", H_zz_NN], ["x", H_x]]
dynamic = []
H_TFIM = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
H_TFIM = H_TFIM.toarray()

H_IM = hamiltonian([["zz", H_zz_NN], ["z", H_z]], [], basis=basis, dtype=np.complex128)
H_IM = H_IM.toarray()

H_z_op = hamiltonian([["z", H_z]], [], basis=basis, dtype=np.complex128)
H_z_op = H_z_op.toarray()

H_x_op = hamiltonian([["x", H_x]], [], basis=basis, dtype=np.complex128)
H_x_op = H_x_op.toarray()

print('H_IM shape', H_IM.shape, 'H_x_op shape', H_x_op.shape, 'H_z_op shape', H_z_op.shape)


w_F = 4.5
T = 2*np.pi/w_F

if(1):
    try:
        #raise Exception("Force recomputation of U_F, qE, qS")
        U_F = np.load('U_F_TFIM_L_{}_hx_{}_hz_{}_wF_{}.npy'.format(L,hx,hz,w_F))
        qE = np.load('qE_TFIM_L_{}_hx_{}_hz_{}_wF_{}.npy'.format(L,hx,hz,w_F))  
        qS = np.load('qS_TFIM_L_{}_hx_{}_hz_{}_wF_{}.npy'.format(L,hx,hz,w_F))
    except:
        #Compute the unitary matrix U_F in the eigenbasis of H0 and its eigenvalues and eigenvectors
        D_IM, W_IM = np.linalg.eigh(H_IM)
        W_IM_dag = np.conjugate(W_IM.T)
        D_x, W_x = np.linalg.eigh(H_x_op)
        W_x_dag = np.conjugate(W_x.T)

        print('W_IM shape', W_IM.shape, 'D_IM shape', D_IM.shape, 'W_x shape', W_x.shape, 'D_x shape', D_x.shape)
        U_F_IM = W_IM @ np.diag(np.exp(-1j * D_IM * T/2)) @ W_IM_dag
        U_F_x = W_x @ np.diag(np.exp(-1j * D_x * T/4)) @ W_x_dag
    
        U_F = U_F_x @ U_F_IM @ U_F_x
        
        assert np.allclose(U_F @ U_F.conj().T, np.eye(U_F.shape[0])), "U_F is not unitary"

        quasi_vals, quasi_vecs = np.linalg.eig(U_F)
        quasi_vecs /= np.sqrt(np.sum(np.abs(quasi_vecs)**2, axis=0))  # Normalize eigenvectors
        quasienergies = np.angle(quasi_vals)/T
        idx = np.argsort(quasienergies)
        quasienergies = quasienergies[idx]
        quasistates = quasi_vecs[:, idx]

        np.save('U_F_TFIM_L_{}_hx_{}_hz_{}_wF_{}.npy'.format(L,hx,hz,w_F), U_F)
        np.save('qE_TFIM_L_{}_hx_{}_hz_{}_wF_{}.npy'.format(L,hx,hz,w_F), quasienergies)
        np.save('qS_TFIM_L_{}_hx_{}_hz_{}_wF_{}.npy'.format(L,hx,hz,w_F), quasistates)

    s = '1'*L  # Initial state with all spins up
    ind = basis.index(s)
    print('ind', ind, 'basis state', basis[ind], 'state string', s)
    psi_init = np.zeros(H_IM.shape[0], dtype=complex)
    #Initial state is the product state with all spins up in the z-direction
    psi_init[ind] = 1.0 + 0j
 
    for i in range(L):
        Sz = hamiltonian([["z", [[1.0, i]]]], [], basis=basis, dtype=np.complex128, check_symm=False)
        exp_z = np.vdot(psi_init, Sz.toarray() @ psi_init)
        print('Expectation value of Sz at site {}: {}'.format(i, exp_z))

    nT = 100
    psi_t = psi_init.copy()
    
    h_z_mid = [[hz, L//2]]  # Apply the longitudinal field only to the middle spin
    H_mid = hamiltonian([["z", h_z_mid]], [], basis=basis, dtype=np.complex128, check_symm =False)
    H_mid = H_mid.toarray()

    for n in range(nT):
        psi_t = U_F @ psi_t
        if(n%10==0):
            print('nT', n, 'norm', np.sum(np.abs(psi_t)**2))
    
        if(0):
            exp_z = np.vdot(psi_t, H_mid @ psi_t)
            plt.scatter(n, np.real(exp_z), color='blue', s=10)
            plt.xlabel('Number of Floquet periods')
            plt.ylabel(r'$\langle \sigma^z_{L/2} \rangle$')
            plt.title('Expectation value of middle spin for TFIM, L={}, hx={}, hz={}, w_F={}'.format(L,hx,hz,w_F))
            plt.grid()
            plt.pause(0.1)
    
        if(1):
            overlap = np.vdot(psi_init, psi_t)
            plt.scatter(n, np.abs(overlap)**2, color='red', s=10)
            plt.xlabel('Number of Floquet periods')
            plt.ylabel('Fidelity with initial state')
            plt.title('Fidelity with initial state for TFIM, L={}, hx={}, hz={}, w_F={}'.format(L,hx,hz,w_F))
            plt.ylim(0, 1)
            plt.grid()
            plt.pause(0.1)
    plt.show()   
    exit()

    diff = np.diff(qE)
    plt.hist(diff, bins=50, density=True)
    plt.xlabel('Quasienergy spacing')
    plt.ylabel('Probability density')
    plt.title('Quasienergy spacing distribution for TFIM, L={}, hx={}, hz={}, w_F={}'.format(L,hx,hz,w_F))
    plt.grid()
    plt.show()



