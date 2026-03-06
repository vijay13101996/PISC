import numpy as np
import matplotlib.pyplot as plt
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators 
from quspin.basis import spin_basis_1d # Hilbert space spin basis  
from PISC.engine import Krylov_complexity
import pickle

"""
Function to generate all the strings in the energy current operator
for the nearest neighbor XXZ model. 

The Hamiltonian is given by:
    H_NN = J\sum_{i=1}^L [ S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + Delta S^z_i S^z_{i+1} ]

The local energy current operator is given by:
    dh_i/dt = i [H_NN, h_i] = j_{i-1} - j_i
from which we can read off the local energy current operator j_i as:
    j_i = i[h_i,h_{i+1}]

This function will enumerate all the strings that occur in the local energy current 
operator j_i, and return a list of these strings. Each string will be represented 
as a tuple of the form (site_index, operator_type), where operator_type can be 
'x', 'y', or 'z' corresponding to S^x, S^y, and S^z respectively. 

To process h_i, we define a function to typify each term in h_i as a string of
the form (site_index, operator_type, coeff). For example, the term S^x_i S^x_{i+1} 
would be represented as [(i, 'x'), (i+1, 'x'), 'J'] where 'J' is the coefficient of the term.

"""

def construct_A(vecs, vals, H_mat, basis, L, trunc_perc=0.96):
    """
    Construct the nearest neighbour z-interaction operator A in the eigenbasis of H,
    and truncate it to the top 98% of the eigenvalues to prevent numerical issues
    at the end of the spectrum.
    """

    U = vecs
    U_inv = np.conjugate(U.T)
    D = np.diag(vals)
    H_reconstructed = U @ D @ U_inv
    assert np.allclose(H_mat, H_reconstructed), "Matrix reconstruction failed!"

    A = [[1.0/L, i, (i+1)%L] for i in range(L)]  # S^z_i S^z_{i+1}
    static_A = [["zz", A]]
    A_op = hamiltonian(static_A, [], basis=basis, dtype=np.complex128)
    A_mat = A_op.toarray()
    A_eigenbasis = U_inv @ A_mat @ U  # Transform A to the eigenbasis

    neig = len(vals)
    neig_trunc = int(neig*trunc_perc)
    A_eigenbasis_trunc = A_eigenbasis[:neig_trunc,:neig_trunc]
    return A_eigenbasis_trunc

def construct_B(vecs, vals, H_mat, basis, L, trunc_perc=0.96):
    """
    Construct the next-nearest neighbor spin-flip operator B in the eigenbasis of H,
    and truncate it to the top 98% of the eigenvalues to prevent numerical issues
    at the end of the spectrum.
    """

    U = vecs
    U_inv = np.conjugate(U.T)
    D = np.diag(vals)
    H_reconstructed = U @ D @ U_inv
    assert np.allclose(H_mat, H_reconstructed), "Matrix reconstruction failed!"

    B = [[1.0/L, i, (i+2)%L] for i in range(L)]  # S^+_i S^+_{i+2}
    static_B = [["+-", B], ["-+", B]]
    B_op = hamiltonian(static_B, [], basis=basis, dtype=np.complex128)
    B_mat = B_op.toarray()
    B_eigenbasis = U_inv @ B_mat @ U  # Transform B to the eigenbasis

    neig = len(vals)
    neig_trunc = int(neig*trunc_perc)
    B_eigenbasis_trunc = B_eigenbasis[:neig_trunc,:neig_trunc]
    return B_eigenbasis_trunc

def construct_Jz(vecs, vals, H_mat, basis, L, J, Delta, J2, Delta2, trunc_perc=0.98):
    U = vecs
    U_inv = np.conjugate(U.T)
    D = np.diag(vals)
    H_reconstructed = U @ D @ U_inv
    assert np.allclose(H_mat, H_reconstructed), "Matrix reconstruction failed!"

    Jz_NN = [[1j*J/2, i, (i+1)%L] for i in range(L)]  
    mJz_NN = [[-1j*J/2, i, (i+1)%L] for i in range(L)]  
    Jz_NNN = [[1j*J2, i, (i+2)%L] for i in range(L)]
    mJz_NNN = [[-1j*J2, i, (i+2)%L] for i in range(L)]
    static_Jz = [["+-", Jz_NN], ["-+", mJz_NN], ["+-", Jz_NNN], ["-+", mJz_NNN]]
    Jz_op = hamiltonian(static_Jz, [], basis=basis, dtype=np.complex128)
    Jz_mat = Jz_op.toarray()
    Jz_eigenbasis = U_inv @ Jz_mat @ U  # Transform Jz to the eigenbasis

    neig = len(vals)
    neig_trunc = int(neig*trunc_perc)
    Jz_eigenbasis_trunc = Jz_eigenbasis[:neig_trunc,:neig_trunc]
    return Jz_eigenbasis_trunc

def construct_JE(vecs, vals, H_mat, basis, L, J, Delta, J2, Delta2, trunc_perc=0.98):
    """
    Construct the energy current operator JE in the eigenbasis of H, using the
    exact expression for the local energy current operator in the NN XXZ model.
    Note that this expression is only valid for the NN XXZ model.
    """
    U = vecs
    U_inv = np.conjugate(U.T)
    D = np.diag(vals)

    print('Recon shape', U.shape, D.shape, U_inv.shape)
    H_reconstructed = U @ D @ U_inv
    assert np.allclose(H_mat, H_reconstructed), "Matrix reconstruction failed!"

    pref = 1j*J**2/2
    mzp = [[pref, (i-1)%L, i, (i+1)%L] for i in range(L)]  
    pzm = [[-pref, (i-1)%L,i, (i+1)%L] for i in range(L)]  
    
    pmz = [[pref*Delta,  (i-1)%L, i, (i+1)%L] for i in range(L)]
    zmp = [[-pref*Delta, (i-1)%L, i, (i+1)%L] for i in range(L)]
    zpm = [[pref*Delta,  (i-1)%L, i, (i+1)%L] for i in range(L)]
    mpz = [[-pref*Delta, (i-1)%L, i, (i+1)%L] for i in range(L)]
    static_JE = [["-z+", mzp], ["+z-", pzm], ["+-z", pmz], ["z-+", zmp], ["z+-", zpm], ["-+z", mpz]]
    JE_op = hamiltonian(static_JE, [], basis=basis, dtype=np.complex128, check_symm=False)
    JE_mat = JE_op.toarray()
    #Check if JE is Hermitian
    if not np.allclose(JE_mat, JE_mat.conj().T):
        print("Warning: JE is not Hermitian!")
    print('shape', U.shape, JE_mat.shape, U_inv.shape)
    JE_eigenbasis = U_inv @ JE_mat @ U  # Transform JE to the eigenbasis
    
    #Check if JE commutes with H
    H_mat_temp = np.diag(vals)
    commutator = H_mat_temp @ JE_eigenbasis - JE_eigenbasis @ H_mat_temp
    zero_matrix = np.zeros_like(commutator) + 0j
    if np.allclose(JE_eigenbasis, zero_matrix, atol=1e-10):
        print("JE is zero in the eigenbasis, which is unexpected!")
    if np.allclose(commutator, zero_matrix, atol=1e-10):
        print("JE commutes with H in the truncated basis!")
    else:
        print("Warning: JE does not commute with H in the truncated basis!")
    
    neig = len(vals)
    neig_trunc = int(neig*trunc_perc)
    JE_eigenbasis_trunc = JE_eigenbasis[:neig_trunc,:neig_trunc]
    return JE_eigenbasis#_trunc

def NN_ham_str(i,L,J,Delta):
    # Define the Hamiltonian strings for the nearest neighbor XXZ model
    ham_str = []
    ham_str.append([i, (i+1)%L, 'x', J])
    ham_str.append([i, (i+1)%L, 'y', J])
    ham_str.append([i, (i+1)%L, 'z', J*Delta])
    return ham_str

def NNN_ham_str(i,L,J,Delta,J2,Delta2):
    # Define the Hamiltonian strings for the next nearest neighbor XXZ model
    ham_str = []
    ham_str_NN = NN_ham_str(i,L,J,Delta)
    ham_str.extend(ham_str_NN)
    ham_str.append([i, (i+2)%L, 'x', J2])
    ham_str.append([i, (i+2)%L, 'y', J2])
    ham_str.append([i, (i+2)%L, 'z', J2*Delta2])
    return ham_str

def eps(a,b,c):
    # Define the Levi-Civita symbol
    if ((a+b+c) == 'xyz') or ((a+b+c) == 'yzx') or ((a+b+c) == 'zxy'):
        return 1
    elif ((a+b+c) == 'xzy') or ((a+b+c) == 'yxz') or ((a+b+c) == 'zyx'):
        return -1
    else:
        return 0

def get_bond_center(i, j, L):
    """Calculates the physical center of a bond on a periodic ring."""
    # Find the shortest directed distance from site i to site j
    d = j - i
    if d > L / 2.0: d -= L
    if d < -L / 2.0: d += L
    
    # The center is site i plus half that shortest distance
    c = (i + d / 2.0) % L
    return c

def get_hopping_distance(c1, c2, L):
    """Calculates the shortest path energy traveled between two bond centers."""
    dist = c2 - c1
    if dist > L / 2.0: dist -= L
    if dist < -L / 2.0: dist += L
    return dist

def commutator_str(tup1, tup2, L, static_lst, h_lst):
    """
    Define the commutator of two tuples
    This function will take two tuples, compute their commutator, and return the resulting string
    The commutator is defined as [A, B] = AB - BA
    We will need to apply the commutation relations of the spin operators to compute this
    For example, [S^x_i, S^y_j] = i delta_{ij} S^z_i, and so on for the other combinations
    We will also need to keep track of the coefficients of each term in the resulting string

    We assume that the input strings are of length 4, i.e. (site1, site2, operator_type, coeff)
    Since we expect to compare hi and h_i+1, there can be at most 3 operators per term in the 
    resulting current operator expansion. 

    We first check if all four site indices are different, in which case the commutator is zero. 
    If not, we apply the commutation relations to compute the resulting string.
    
    """
    i,j = tup1[:2]
    k,l = tup2[:2]
    a,b = tup1[2], tup2[2]
    taua,taub = tup1[3], tup2[3]
    
    c1 = get_bond_center(i, j, L)
    c2 = get_bond_center(k, l, L)
    dist = 1.0 #get_hopping_distance(c1, c2, L)
    #dist = (k+l)/2 - (i+j)/2

    if (a==b):
        #print("Commutator is zero since the operator types are the same")
        return
    elif ((i!=k) and (j!=l) and (i!=l) and (j!=k)): # We assume i<j and k<l by default
        #print("Commutator is zero since all site indices are different")
        return
    elif (i==k) and (j==l):
        return
        if(0):
            c = 'xyz'.replace(a,'').replace(b,'')
            coeff = -eps(a,b,c)*taua*taub
            term = [[coeff,i,i,j]]
            h_lst.append(term)
            static_lst.append([a+b+c, term])
    elif (i==k) and (j!=l):
        c = 'xyz'.replace(a,'').replace(b,'')
        coeff = -eps(a,b,c)*taua*taub*dist
        term = [[coeff,i,l,j]]
        h_lst.append(term)
        static_lst.append([c+b+a, term])    
    elif (i!=k) and (j==l):
        c = 'xyz'.replace(a,'').replace(b,'')
        coeff = -eps(a,b,c)*taua*taub*dist
        term = [[coeff,i,k,j]]
        h_lst.append(term)
        static_lst.append([a+b+c, term])
    elif (j==k) and (i!=l):
        c = 'xyz'.replace(a,'').replace(b,'')
        coeff = -eps(a,b,c)*taua*taub*dist
        term = [[coeff,i,j,l]]
        h_lst.append(term)
        static_lst.append([a+c+b, term])
    elif (i==l) and (j!=k):
        c = 'xyz'.replace(a,'').replace(b,'')
        coeff = -eps(a,b,c)*taua*taub*dist
        term = [[coeff,k,i,j]]
        h_lst.append(term)
        static_lst.append([b+c+a, term])

def generate_JE_terms(i,j, L,J,Delta,J2,Delta2):
    # Generate the strings in the local energy current operator for the nearest neighbor XXZ model
    static_lst = []
    h_lst = []
    
    # Get the Hamiltonian strings for h_i and h_{i+1}
    hi_str = NNN_ham_str(i,L,J,Delta,J2,Delta2)
    hj_str = NNN_ham_str(j,L,J,Delta,J2,Delta2)
    #print('hi_str', hi_str)
    #print('hj_str', hj_str)

    # Compute the commutator [h_i, h_{i+1}] to get the local energy current operator j_i
    for term1 in hi_str:
        for term2 in hj_str:
            commutator_str(term1, term2, L, static_lst, h_lst)
    #print('h_lst', h_lst) 
    #print('static_lst', static_lst)
    return static_lst, h_lst

def construct_JE_comm(vecs, vals, H_mat, basis, L, J, Delta, J2, Delta2, trunc_perc=0.98, verify=False):
    """
    Construct the energy current operator JE in the eigenbasis of H, by explicitly computing 
    the commutator [H, h_i] to get j_i, and then summing over i to get JE. 
    This is a more brute-force approach compared to using the exact expression for JE, 
    but it is foolproof and does not rely on any assumptions about the form of JE.
    """
    if verify:
        O_eigenbasis = construct_JE(vecs, vals, H_mat)

    U = vecs
    U_inv = np.conjugate(U.T)
    D = np.diag(vals)

    #print('Recon shape', U.shape, D.shape, U_inv.shape)
    H_reconstructed = U @ D @ U_inv
    assert np.allclose(H_mat, H_reconstructed), "Matrix reconstruction failed!"

    j_i_list = []
    JE_mat=np.zeros_like(H_mat, dtype=np.complex128)
    for i in range(L):
        static_lst, h_lst = generate_JE_terms(i, (i+1)%L, L, J, Delta, J2, Delta2)
        JE_op_i = hamiltonian(static_lst, [], basis=basis, dtype=np.complex128, check_symm=False)
        JE_mat_i = JE_op_i.toarray()
        JE_mat += JE_mat_i
    
        static_lst, h_lst = generate_JE_terms(i, (i+2)%L, L, J, Delta, J2, Delta2)
        JE_op_i = hamiltonian(static_lst, [], basis=basis, dtype=np.complex128, check_symm=False)
        JE_mat_i = JE_op_i.toarray()
        JE_mat += 2*JE_mat_i

    #JE_mat/=2 # Each pair (i,j) is counted twice in the double sum, so we divide by 2 to correct for this.

    #Check if JE is Hermitian
    if not np.allclose(JE_mat, JE_mat.conj().T):
        print("Warning: JE is not Hermitian!")
    #print('shape', U.shape, JE_mat.shape, U_inv.shape)
    JE_eigenbasis = U_inv @ JE_mat @ U  # Transform JE to the eigenbasis

    if(verify):#Check if JE_eigenbasis and O_eigenbasis are the same
        print('JE_eigenbasis shape', np.around(JE_eigenbasis[:5,:5],3))
        print('O_eigenbasis shape', np.around(O_eigenbasis[:5,:5],3))
        if np.allclose(JE_eigenbasis, O_eigenbasis, atol=1e-10):
            print("JE_eigenbasis and O_eigenbasis are the same!")
        else:
            print("Warning: JE_eigenbasis and O_eigenbasis are not the same!")

    neig = len(vals)
    neig_trunc = int(neig*trunc_perc)
    JE_eigenbasis_trunc = JE_eigenbasis[:neig_trunc,:neig_trunc]
    comm = np.diag(vals) @ JE_eigenbasis - JE_eigenbasis @ np.diag(vals)
    zero_matrix = np.zeros_like(comm) + 0j
    if np.allclose(JE_eigenbasis, zero_matrix, atol=1e-10):
        print("JE is zero in the truncated eigenbasis, which is unexpected!")
    if np.allclose(comm, zero_matrix, atol=1e-10):
        print("JE commutes with H in the truncated eigenbasis!")    
    else:
        print("Warning: JE does not commute with H in the truncated eigenbasis!")
    return JE_eigenbasis#_trunc


#---------------------------------------------------------------------------------
if(0):
    L = 18  # Length of the chain
    J = 1.0  # Coupling constant
    Delta = 1#0.55  # Anisotropy parameter

    g = 0.0  # Magnetic field

    #Next-nearest neighbor (NNN) coupling parameters
    lamda = 1.0
    J2 = 0.0#lamda*J  # NNN coupling constant
    Delta2 = 0.#5  # NNN anisotropy parameter

    k=1

    basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=k)#,zblock=1)#,pblock=1) # and positive parity sector
    print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))

    H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
    static_NNN = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
    H_XXZ_NNN = hamiltonian(static_NNN,[],basis=basis,dtype=np.complex128)
    H_mat_NNN = H_XXZ_NNN.toarray()
    print('NNN Hamiltonian constructed', H_mat_NNN.shape)
    E_NNN = H_XXZ_NNN.eigvalsh()
    vals_NNN, vecs_NNN = H_XXZ_NNN.eigh()

    if(1):
        h_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        h_xy_NN = [[J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        h_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        h_xy_NNN = [[J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        static_h_NNN = [["zz",h_zz_NN],["zz",h_zz_NNN],["xx",h_xy_NN],["yy",h_xy_NN],["xx",h_xy_NNN],["yy",h_xy_NNN]]
        h_XXZ_NNN = hamiltonian(static_h_NNN,[],basis=basis,dtype=np.complex128)
        h_mat_NNN = h_XXZ_NNN.toarray()   
        evals_h_NNN, vecs_h_NNN = h_XXZ_NNN.eigh()
        assert np.allclose(vals_NNN, evals_h_NNN, atol=1e-4), "Eigenvalues of H and h do not match!"
        assert np.allclose(vecs_NNN[:500,:500], vecs_h_NNN[:500,:500], atol=1e-4), "Eigenvectors of H and h do not match!"
        #!!!!!! FIGURE OUT WHY THE EIGENVECTORS DO NOT MATCH EXACTLY. 
        #!!!!!! IS IT NUMERICAL PRECISION ISSUES OR IS THERE A DEEPER REASON? (HINT: It works if zblock filter is turned off)
        print("Eigenvalues and eigenvectors of H and h match!")
        exit()

if(0): #Verify that the two constructions of JE give the same result for the NN XXZ model
    L = 18  # Length of the chain
    J = 1.0  # Coupling constant
    Delta = 0.5  # Anisotropy parameter

    g = 0.0  # Magnetic field

    #Next-nearest neighbor (NNN) coupling parameters
    J2 = 0.0  # NNN coupling constant
    Delta2 = 0.0  # NNN anisotropy parameter

    k=1
    basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=1) # and positive parity sector
    print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))

    H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
    static_NNN = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
    H_XXZ_NNN = hamiltonian(static_NNN,[],basis=basis,dtype=np.complex128)
    H_mat_NNN = H_XXZ_NNN.toarray()
    print('NNN Hamiltonian constructed', H_mat_NNN.shape)
    E_NNN = H_XXZ_NNN.eigvalsh()
    vals_NNN, vecs_NNN = H_XXZ_NNN.eigh()
   
    O_eigenbasis = construct_JE(vecs_NNN, vals_NNN, H_mat_NNN, basis, L, J, Delta, J2, Delta2)
    O_eigenbasis_comm = construct_JE_comm(vecs_NNN, vals_NNN, H_mat_NNN, basis, L, J, Delta, J2, Delta2, verify=False)
    #Check if the two constructions of JE are close
    if np.allclose(O_eigenbasis, O_eigenbasis_comm, atol=1e-10):
        print("JE from direct construction and commutator construction are close!")
    else:        
        print("Warning: JE from direct construction and commutator construction differ!")
    
    comm = H_mat_NNN @ O_eigenbasis - O_eigenbasis @ H_mat_NNN
    if np.allclose(comm, np.zeros_like(comm), atol=1e-10):
        print("O_eigenbasis_comm commutes with H_XXZ_NNN!")
    else:
        print("Warning: O_eigenbasis_comm does not commute with H_XXZ_NNN!")
    exit()

if(0): #Compare the current operators in the NNN model and its sublattices
    L = 8# Length of the chain
    J = 0.0  # Coupling constant
    Delta = 0.0  # Anisotropy parameter

    g = 0.0  # Magnetic field

    #Next-nearest neighbor (NNN) coupling parameters
    J2 = 1.0  # NNN coupling constant
    Delta2 = 0.5  # NNN anisotropy parameter

    k=1
    basis = spin_basis_1d(L,pauli=False) # and positive parity sector
    print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))

    H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
    static_NNN = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
    H_XXZ_NNN = hamiltonian(static_NNN,[],basis=basis,dtype=np.complex128)
    H_mat_NNN = H_XXZ_NNN.toarray()
    print('NNN Hamiltonian constructed', H_mat_NNN.shape)
    E_NNN = H_XXZ_NNN.eigvalsh()
    vals_NNN, vecs_NNN = H_XXZ_NNN.eigh()
   
    O_eigenbasis_comm = construct_JE_comm(vecs_NNN, vals_NNN, H_mat_NNN, basis, L, J, Delta, J2, Delta2, verify=False)
    tr2 = np.trace(O_eigenbasis_comm @ O_eigenbasis_comm)
    if np.allclose(O_eigenbasis_comm, np.zeros_like(O_eigenbasis_comm), atol=1e-10):
        print("O_eigenbasis_comm is zero in the eigenbasis, which is unexpected!")

    L = L//2
    basis = spin_basis_1d(L,pauli=False) # and positive parity sector
    H_zz_NN = [[J2*Delta2,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NN = [[0.5*J2,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    static_NN = [["zz",H_zz_NN],["+-",H_xy_NN],["-+",H_xy_NN]]
    H_XXZ_NN = hamiltonian(static_NN,[],basis=basis,dtype=np.complex128)
    H_mat_NN = H_XXZ_NN.toarray()
    print('NN Hamiltonian constructed', H_mat_NN.shape)
    E_NN = H_XXZ_NN.eigvalsh()
    vals_NN, vecs_NN = H_XXZ_NN.eigh()
    O_eigenbasis_comm_NN = construct_JE_comm(vecs_NN, vals_NN, H_mat_NN, basis, L, J2, Delta2, 0,0, verify=False)
    tr2_NN = np.trace(O_eigenbasis_comm_NN @ O_eigenbasis_comm_NN)
    H_diag = np.diag(vals_NN)
    comm = H_diag @ O_eigenbasis_comm_NN - O_eigenbasis_comm_NN @ H_diag
    if np.allclose(comm, np.zeros_like(comm), atol=1e-10):
        print("O_eigenbasis_comm_NN commutes with H_XXZ_NN!")
    else:
        print("Warning: O_eigenbasis_comm_NN does not commute with H_XXZ_NN!")
    
    print('tr2_NN', tr2_NN, 'tr2', tr2, tr2_NN*2*2**(L)*4, tr2_NN*2*2**(L))

    ### It is unclear if the current operator in the NNN model should be
    ### [h_i, h_{i+1}] + [h_i, h_{i+2}] or if the second term should have a prefactor of 2 
    ### Past articles seem to suggest a prefactor of 2, but it is not clear why this should be the case.
    exit()

if(0): #Check if a sublattice JE operator commutes with the sublattice Hamiltonian
    L = 12# Length of the chain
    J = 0.0  # Coupling constant
    Delta = 0.0  # Anisotropy parameter

    g = 0.0  # Magnetic field

    #Next-nearest neighbor (NNN) coupling parameters
    J2 = 1.0  # NNN coupling constant
    Delta2 = 0.5  # NNN anisotropy parameter

    k=1
    basis = spin_basis_1d(L,pauli=False) # and positive parity sector
    print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))

    H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
    H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
    H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
    static_NNN = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
    H_XXZ_NNN = hamiltonian(static_NNN,[],basis=basis,dtype=np.complex128)
    H_mat_NNN = H_XXZ_NNN.toarray()
    print('NNN Hamiltonian constructed', H_mat_NNN.shape)
    E_NNN = H_XXZ_NNN.eigvalsh()
    vals_NNN, vecs_NNN = H_XXZ_NNN.eigh()
    O_eigenbasis_comm = construct_JE_comm(vecs_NNN, vals_NNN, H_mat_NNN, basis, L, J, Delta, J2, Delta2, verify=False) 
    if np.allclose(O_eigenbasis_comm, np.zeros_like(O_eigenbasis_comm), atol=1e-10):
        print("O_eigenbasis_comm is zero in the eigenbasis, which is unexpected!")
    
    #Sublattice A
    H_zz_NN_A = [[J2*Delta2,i,(i+2)%L] for i in range(0,L,2)] # periodic boundary conditions
    H_xy_NN_A = [[0.5*J2,i,(i+2)%L] for i in range(0,L,2)] # periodic boundary conditions
    static_NN_A = [["zz",H_zz_NN_A],["+-",H_xy_NN_A],["-+",H_xy_NN_A]]
    H_XXZ_NN_A = hamiltonian(static_NN_A,[],basis=basis,dtype=np.complex128)
    H_mat_NN_A = H_XXZ_NN_A.toarray()
    print('NN Hamiltonian for sublattice A constructed', H_mat_NN_A.shape)
    E_NN_A = H_XXZ_NN_A.eigvalsh()
    vals_NN_A, vecs_NN_A = H_XXZ_NN_A.eigh()
    O_eigenbasis_comm_NN_A = construct_JE_comm(vecs_NN_A, vals_NN_A, H_mat_NN_A, basis, L, J, Delta, J2, Delta2, verify=False)

    zeromat = np.zeros_like(O_eigenbasis_comm_NN_A)
    if np.allclose(O_eigenbasis_comm_NN_A, zeromat, atol=1e-10):
        print("O_eigenbasis_comm_NN_A is zero in the eigenbasis, which is unexpected!")
    comm_A = np.diag(vals_NN_A) @ O_eigenbasis_comm_NN_A - O_eigenbasis_comm_NN_A @ np.diag(vals_NN_A)
    if np.allclose(comm_A, np.zeros_like(comm_A), atol=1e-10):
        print("O_eigenbasis_comm_NN_A commutes with H_XXZ_NN_A!")
    else:
        print("Warning: O_eigenbasis_comm_NN_A does not commute with H_XXZ_NN_A!")
