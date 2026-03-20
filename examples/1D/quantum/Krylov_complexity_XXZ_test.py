import numpy as np
from matplotlib import pyplot as plt
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators 
from quspin.basis import spin_basis_1d # Hilbert space spin basis 
import math
from PISC.engine import Krylov_complexity
import time

L = 20  # Length of the chain
J = 1.0  # Coupling constant
Delta = 1.0  # Anisotropy parameter
g = 0.0  # Magnetic field
lamda = 0.5
J2 = lamda*J  # NNN coupling constant
Delta2 = 1.0  # NNN anisotropy parameter

k=0

def compute_eigs(L,J,Delta,g,J2,Delta2,k):
    basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=k,zblock=1,pblock=1) # positive parity sector

    print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))
    H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions

    H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions

    static = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
    dynamic = []

    H_XXZ_NNN = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
    H_mat = H_XXZ_NNN.toarray()
    print('Hamiltonian constructed', H_mat.shape)
    E = H_XXZ_NNN.eigvalsh()
    vals, vecs = H_XXZ_NNN.eigh()
    print('Eigenvalues computed')

    return E, vals, vecs, basis, H_XXZ_NNN

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

    H_xx = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions

    static = [["+-",H_xx],["-+",H_xx]]
    dynamic = []
    
    basis = spin_basis_1d(L,pauli=False) # full basis

    H_XX = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
    H_mat = H_XX.toarray()
    print('Hamiltonian constructed', H_mat.shape)

    vals, vecs = H_XX.eigh()
    print('Eigenvalues computed')

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

    print("Benchmarking XXX model with Delta=1.0")
    H_z = [[J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    
    static = [["zz",H_z],["+-",H_xx],["-+",H_xx]]
    dynamic = []
    
    basis = spin_basis_1d(L,pauli=False) # full basis

    H_XXZ = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
    H_mat = H_XXZ.toarray()
    print('Hamiltonian constructed', H_mat.shape)

    vals, vecs = H_XXZ.eigh()
    print('Eigenvalues computed')

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
    
    L = 10  # Length of the chain
    J = 1.0  # Coupling constant
    Delta = 0.2  # Anisotropy parameter
    g = 0.0  # Magnetic field
    J2 = 1.0  # NNN coupling constant
    Delta2 = 0.4  # NNN anisotropy parameter

    k = 1 # x 2*np.pi/L  # Wavevector for the k=2\pi/L sector

    print("Benchmarking XXZ model with NN and NNN coupling")
    H_xx_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_xx_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
    H_z_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_z_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions

    H_XXZ = [["zz",H_z_NN],["zz",H_z_NNN],["+-",H_xx_NN],["-+",H_xx_NN],["+-",H_xx_NNN],["-+",H_xx_NNN]]

    static = H_XXZ
    dynamic = []

    basis_full = spin_basis_1d(L,pauli=False) # full basis
    print("Number of basis states in full basis:", len(basis_full.states),'Expected', 2**L)
    basis_Mz0 = spin_basis_1d(L,pauli=False,Nup=L//2) # Mz=0 sector
    print("Number of basis states with M_z = 0:", len(basis_Mz0.states),'Expected', math.comb(L, L//2))
    basis_symm = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=k,pblock=1,zblock=1) # Mz=0, k=2pi/L, positive parity sector
    print("Number of basis states with M_z = 0, k = 2pi/L, positive parity:", len(basis_symm.states))
    
    H_full = hamiltonian(static,dynamic,basis=basis_full,dtype=np.complex128)
    H_mat_full = H_full.toarray()
    print('Full Hamiltonian constructed', H_mat_full.shape)
    vals_full, vecs_full = H_full.eigh()
    vals_full = np.round(vals_full, decimals=5)  # Round to avoid numerical precision issues
    print('Eigenvalues of full Hamiltonian computed')

    H_Sz0 = hamiltonian(static,dynamic,basis=basis_Mz0,dtype=np.complex128)
    H_mat_Sz0 = H_Sz0.toarray()
    print('Hamiltonian in Mz=0 sector constructed', H_mat_Sz0.shape)
    vals_Sz0, vecs_Sz0 = H_Sz0.eigh()
    vals_Sz0 = np.round(vals_Sz0, decimals=5)  # Round to avoid numerical precision issues
    print('Eigenvalues of Mz=0 Hamiltonian computed')

    H_symm = hamiltonian(static,dynamic,basis=basis_symm,dtype=np.complex128)
    H_mat_symm = H_symm.toarray()
    print('Hamiltonian in Mz=0, k=2pi/L, positive parity sector constructed', H_mat_symm.shape)
    vals_symm, vecs_symm = H_symm.eigh()
    vals_symm = np.round(vals_symm, decimals=5)  # Round to avoid numerical precision issues
    print('Eigenvalues of Mz=0, k=2pi/L, positive parity Hamiltonian computed')

    #print("vals_full:", vals_full)
    #print("vals_Sz0:", vals_Sz0)
    #print("vals_symm:", vals_symm)

    #Check if vals_Sz0 is contained in vals_full
    if np.all(np.isin(vals_Sz0, vals_full)):
        print("***Eigenvalues of the M_z = 0 sector are contained in the full Hamiltonian eigenvalues***")

    #Check if vals_symm is contained in vals_Sz0
    if np.all(np.isin(vals_symm, vals_Sz0)):
        print("***Eigenvalues of the P and Z2 invariant sector are contained in the M_z = 0 sector eigenvalues***")

def benchmark_3():
    """ 
    Benchmark the NNN coupling term in the Hamiltonian using the 
    Majumdar-Ghosh point which is occurs at:
    J2 = J/2 and Delta1 = Delta2 = 1.0

    The ground state is a product of singlets:
    |psi> = (|up,down> - |down,up>) (|up,down> - |down,up>) ...

    And the ground state energy is -0.375 per site.
    """

    J = 1.0
    J2 = 0.5 * J
    Delta = 1.0
    Delta2 = 1.0
    g = 0.0

    for L in [4, 6, 8, 10]: #, 12]:
        print("Benchmarking Majumdar-Ghosh point for L =", L)
        
        H_xx_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_xx_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)]
        H_z_NN = [[J*Delta,i,(i+1)%L] for i in range(L)]
        H_z_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)]

        static = [["zz",H_z_NN],["zz",H_z_NNN],["+-",H_xx_NN],["-+",H_xx_NN],["+-",H_xx_NNN],["-+",H_xx_NNN]]
        dynamic = []

        basis = spin_basis_1d(L,pauli=False) # full basis
        H_XXZ = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
        H_mat = H_XXZ.toarray()
        print('Hamiltonian constructed', H_mat.shape)
        vals_full, vecs_full = H_XXZ.eigh()
        
        print('Ground state energy per site:', vals_full[0]/L, 'Expected:', -0.375)
        if np.isclose(vals_full[0]/L, -0.375, atol=1e-5):
            print("***Ground state energy per site matches the expected value***")
        print('\n')

def benchmark_4():
    """
    Reference:
    Steinigeweg, Robin, Jacek Herbrych, and Peter Prelovšek. 
    "Eigenstate thermalization within isolated spin-chain systems." 
    Physical Review E — Statistical, Nonlinear, and Soft Matter Physics 87.1 (2013): 012118.

    Benchmark the current operator in the eigenbasis of the Hamiltonian
    """

    L=20
    J=1.0
    Delta=0.5
    
    J2=J
    Delta2=0.5

    k=1 # x 2*np.pi/L  # Wavevector for the k=2\pi/L sector

    Mz =-1 # Total magnetization, so Nup = L/2 + Mz

    basis = spin_basis_1d(L,pauli=False,Nup=L//2+Mz,kblock=k) 

    print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))

    try: #Retrieve vals, vecs from file
        data = np.load('XXZ_L%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k))
        vals = data['vals']
        vecs = data['vecs']
        H_mat = data['H_mat']
        print('Loaded eigenvalues and eigenvectors from file')
        U = vecs
        U_inv = np.conjugate(U.T)
        D = np.diag(vals)
        H_reconstructed = U @ D @ U_inv
        assert np.allclose(H_mat, H_reconstructed), "Matrix reconstruction failed!"
    except:
        print('File not found, computing eigenvalues and eigenvectors')
        H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions

        H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions

        static = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN]]
        dynamic = []

        H_XXZ_NNN = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
        H_mat = H_XXZ_NNN.toarray()
        print('Hamiltonian constructed', H_mat.shape)
        
        vals, vecs = H_XXZ_NNN.eigh()
        
        print('Eigenvalues computed')

        #Save the eigenvalues and eigenvectors to a file
        np.savez('XXZ_L%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k), vals=vals, vecs=vecs, H_mat=H_mat)
        exit()

    if(0):
        H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        
        H_kin = hamiltonian([["+-",H_xy_NN],["-+",H_xy_NN]],[],basis=basis,dtype=np.complex128)
        H_kin_mat = H_kin.toarray()
        print('Kinetic Hamiltonian constructed', H_kin_mat.shape)

        H_kin_eigenbasis = U_inv @ H_kin_mat @ U  # Transform H_kin to the eigenbasis
    
        plt.plot(vals, np.diag(H_kin_eigenbasis), '.', label='Diagonal elements')
    
    if(1):
        H_xy_NN_p = [[-0.5j*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NN_m = [[0.5j*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions

        J = hamiltonian([["+-",H_xy_NN_p],["-+",H_xy_NN_m]],[],basis=basis,dtype=np.complex128)
        J_mat = J.toarray()
        print('Current operator constructed', J_mat.shape)

        J_eigenbasis = U_inv @ J_mat @ U  # Transform J to the eigenbasis

        plt.plot(vals, np.diag(J_eigenbasis), '.', label='Diagonal elements')
    plt.show()

def Op_matrix_elts():
    L= 20  # Length of the chain
    J= 1.0  # Coupling constant
    Delta= 0.55  # Anisotropy parameter
    lamda= 0.0 # Scaling factor for the NNN coupling
    J2= lamda*J  # NNN coupling constant
    Delta2= 0.5  # NNN anisotropy parameter

    k= 0 # x 2*np.pi/L  # Wavevector for the k=2\pi/L sector
    Mz = 0 # Total magnetization, so Nup = L/2 + Mz

    basis = spin_basis_1d(L,pauli=False,Nup=L//2+Mz,kblock=k, pblock=1,zblock=1) # and positive parity sector

    print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))

    try: #Retrieve vals, vecs from file
        filename = 'XXZ_L_%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k)
        loc = '/scratch/vgs23/'
        data = np.load(loc+filename)
        vals = data['vals']
        vecs = data['vecs']
        H_mat = data['H_mat']
        print('Loaded eigenvalues and eigenvectors from file')
        U = vecs
        U_inv = np.conjugate(U.T)
        D = np.diag(vals)
        H_reconstructed = U @ D @ U_inv
        assert np.allclose(H_mat, H_reconstructed), "Matrix reconstruction failed!"

    except:
        print('File not found, computing eigenvalues and eigenvectors')
        H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        static = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
        dynamic = []

        H_XXZ_NNN = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
        H_mat = H_XXZ_NNN.toarray()
        print('Hamiltonian constructed', H_mat.shape)
        vals, vecs = H_XXZ_NNN.eigh()
        U = vecs
        U_inv = np.conjugate(U.T)
        print('Eigenvalues computed')
        #Save the eigenvalues and eigenvectors to a file
        #store data in /scratch/vgs23/
        filename = 'XXZ_L_%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k)
        print('Saving eigenvalues and eigenvectors to file:', filename)
        loc = '/scratch/vgs23/'
        np.savez(loc+filename, vals=vals, vecs=vecs, H_mat=H_mat)

    fig,ax = plt.subplots(2,1)

    if(1):
        try: #if error occurs, print message and exit
            filename = 'A_op_L%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k)
            loc = '/scratch/vgs23/'
            A_eigenbasis = np.load(loc+filename)['A_eigenbasis']
            print('Loaded A operator in eigenbasis from file')
            xax = vals/L
            yax = np.diag(A_eigenbasis)
            print('len', len(xax), len(yax))
            ax[0].scatter(xax, 1e2*yax, s=1) 
        except:
            print('File not found, computing A operator in eigenbasis')
            A = [[1.0/L, i, (i+1)%L] for i in range(L)]  # S^z_i S^z_{i+1}
            static_A = [["zz", A]]
            A_op = hamiltonian(static_A, [], basis=basis, dtype=np.complex128)
            A_mat = A_op.toarray()
            A_eigenbasis = U_inv @ A_mat @ U  # Transform A to the eigenbasis
            print('A_eigenbasis shape', A_eigenbasis.shape, len(vals))
            filename = 'A_op_L%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k)
            loc = '/scratch/vgs23/'
            print('Saving A operator in eigenbasis to file:', filename)
            np.savez(loc+filename, A_eigenbasis=A_eigenbasis)

    if(1):
        try:
            filename = 'B_op_L%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k)
            loc = '/scratch/vgs23/'
            B_eigenbasis = np.load(loc+filename)['B_eigenbasis']
            print('Loaded B operator in eigenbasis from file')
            xax = vals/L
            yax = np.diag(B_eigenbasis)
            print('len', len(xax), len(yax))
            ax[1].scatter(xax, 1e2*yax, s=1)

        except:
            print('File not found, computing B operator in eigenbasis')
            B = [[1.0/L, i, (i+2)%L] for i in range(L)]  
            static_B = [["+-", B], ["-+", B]]
            B_op = hamiltonian(static_B, [], basis=basis, dtype=np.complex128)
            B_mat = B_op.toarray()
            B_eigenbasis = U_inv @ B_mat @ U  # Transform B to the eigenbasis
            
            filename = 'B_op_L%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k)
            loc = '/scratch/vgs23/'
            print('Saving B operator in eigenbasis to file:', filename)
            np.savez(loc+filename, B_eigenbasis=B_eigenbasis)

            print('B_eigenbasis shape', B_eigenbasis.shape, len(vals))

        plt.show()

def Krylov_op():
    L=20
    J=1.0
    Delta=0.55
    lamda=0.0 # Scaling factor for the NNN coupling
    J2=lamda*J  # NNN coupling constant
    Delta2=0.5  # NNN anisotropy parameter
    
    k = 0 # x 2*np.pi/L  # Wavevector for the k=2\pi/L sector
    Mz = 0 # Total magnetization, so Nup = L/2 + Mz

    neigs = 2400

    print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))

    filename = 'XXZ_L_%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k)
    loc = '/scratch/vgs23/'
    data = np.load(loc+filename)
    vals = data['vals']
    #vecs = data['vecs']
    #H_mat = data['H_mat']
    vals = vals[:neigs]

    print('Loaded eigenvalues and eigenvectors from file')

    A_filename = 'A_op_L%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k)
    B_filename = 'B_op_L%d_J%.1f_Delta%.1f_J2%.1f_Delta2%.1f_Mz%d_k%d.npz'%(L,J,Delta,J2,Delta2,Mz,k)
    loc = '/scratch/vgs23/'
    
    A_eigenbasis = np.load(loc+A_filename)['A_eigenbasis']
    A_eigenbasis = A_eigenbasis[:neigs,:neigs]
    print('Loaded A operator in eigenbasis from file', A_eigenbasis.shape)


    B_eigenbasis = np.load(loc+B_filename)['B_eigenbasis']
    B_eigenbasis = B_eigenbasis[:neigs,:neigs]
    print('Loaded B operator in eigenbasis from file', B_eigenbasis.shape)
    
    O = B_eigenbasis # Change to A_eigenbasis to compute for A operator

    neigs = O.shape[0]
    
    start_time = time.time()

    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)
    print('Liouville matrix and LO matrix computed')

    ncoeff = 80
    beta = 0.5
    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    
    end_time = time.time()
    print('Lanczos coefficients computed in %.2f seconds'%(end_time - start_time))
    coeffarr = np.arange(ncoeff)
    plt.scatter(coeffarr, barr)
    plt.show()


#Op_matrix_elts()
Krylov_op()


if(0): # Run all benchmarks
    benchmark_1()
    benchmark_2()
    benchmark_3()
    benchmark_4()


