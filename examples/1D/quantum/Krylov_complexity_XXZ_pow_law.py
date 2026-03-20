import numpy as np
import matplotlib.pyplot as plt
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators 
from quspin.basis import spin_basis_1d # Hilbert space spin basis  
from PISC.engine import Krylov_complexity
from Krylov_XXZ_tools import construct_A, construct_B, construct_Jz, construct_JE, construct_JE_comm
import pickle
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 10 
yl_fs = 10
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12
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

def define_lattice(L, J, Delta, alpha, basis, suffix):
    ind_list = [(i,j) for i in range(L) for j in range(i+1,L)]
    #print('Pairs of interactions in power-law model:', (ind_list))
    
    try:
        #raise FileNotFoundError
        vals_PL = pickle.load(open('eigenvalues_{}.pkl'.format(suffix), 'rb'))
        vecs_PL = pickle.load(open('eigenvectors_{}.pkl'.format(suffix), 'rb'))
        H_mat_PL = pickle.load(open('H_mat_{}.pkl'.format(suffix), 'rb'))
        print('Eigenvalues and eigenvectors loaded from disk')
        print('shape', vals_PL.shape, vecs_PL.shape, H_mat_PL.shape)
    except FileNotFoundError:
        H_zz_PL =[]
        H_xy_PL = []
        for (i,j) in ind_list:
            dist = min(abs(i-j), L-abs(i-j))  # periodic boundary conditions
            H_zz_PL.append([J*Delta/(dist**alpha), i, j])
            H_xy_PL.append([0.5*J/(dist**alpha), i, j])

        static_PL = [["zz",H_zz_PL],["+-",H_xy_PL],["-+",H_xy_PL]]
        H_XXZ_PL = hamiltonian(static_PL,dynamic,basis=basis,dtype=np.complex128)
        H_mat_PL = H_XXZ_PL.toarray()
        print('PL Hamiltonian constructed', H_mat_PL.shape)
        E_PL = H_XXZ_PL.eigvalsh()
        vals_PL, vecs_PL = H_XXZ_PL.eigh()  

        pickle.dump(vals_PL, open('eigenvalues_{}.pkl'.format(suffix), 'wb'))
        pickle.dump(vecs_PL, open('eigenvectors_{}.pkl'.format(suffix), 'wb'))
        pickle.dump(H_mat_PL, open('H_mat_{}.pkl'.format(suffix), 'wb'))
    return vals_PL, vecs_PL, H_mat_PL

def generate_O(Okey, L, J, Delta, alpha, basis, suffix):
    # Generate the operator O in the computational basis
    try:
        #raise FileNotFoundError
        vals = pickle.load(open('eigenvalues_{}.pkl'.format(suffix), 'rb'))
        vecs = pickle.load(open('eigenvectors_{}.pkl'.format(suffix), 'rb'))
        H_mat = pickle.load(open('H_mat_{}.pkl'.format(suffix), 'rb'))
        print('Eigenvalues and eigenvectors loaded from disk')
        print('shape', vals.shape, vecs.shape, H_mat.shape) 
    except FileNotFoundError:
        vals, vecs, H_mat = define_lattice(L, J, Delta, alpha, basis, suffix)
       
    try:
        #raise FileNotFoundError
        O = pickle.load(open('{}_{}.pkl'.format(Okey, suffix), 'rb'))
        print('Operator {} loaded from disk'.format(Okey))
    except FileNotFoundError:
        if Okey == 'B':
            O = construct_B(vecs, vals, H_mat, basis, L)
        elif Okey == 'A':   
            O = construct_A(vecs, vals, H_mat, basis, L)
        pickle.dump(O, open('{}_{}.pkl'.format(Okey, suffix), 'wb'))
    return O

L = 20  # Length of the chain
J = 1.0  # Coupling constant
Delta = 0.55  # Anisotropy parameter

g = 0.0  # Magnetic field

#Power-law parameter
alpha = 0.5

k = 1 # x 2pi/L momentum sector

basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=k,zblock=1)#,pblock=1) # and positive parity sector
print('L = {}, J = {}, Delta = {}, alpha = {}, k = {}'.format(L,J,Delta,alpha,k))

dynamic = []
beta = 1.0
ncoeff = 100
coeff_arr = np.arange(ncoeff)

store = False
suffix = 'PL_alpha_{}_L_{}_zblock1'.format(alpha,L)
vals, vecs, H_mat = define_lattice(L, J, Delta, alpha, basis, suffix)

Okey = 'B'
B_PL = generate_O(Okey, L, J, Delta, alpha, basis, suffix)

#np.fill_diagonal(B_PL, 0)  # Set diagonal elements to zero for better visualization

if(1):
    fig, ax = plt.subplots(1,2, figsize=(6,3),sharey=True)
    plt.subplots_adjust(wspace=0.0,bottom=0.15)
    #Compute bn at different alpha
    for alpha in [0.5,5.0]:
        suffix = 'PL_alpha_{}_L_{}_zblock1'.format(alpha,L)
        Okey = 'A'
        A_PL = generate_O(Okey, L, J, Delta, alpha, basis, suffix)
        try:
            #raise FileNotFoundError
            barr = pickle.load(open('bn_A_PL_alpha_{}.pkl'.format(alpha), 'rb'))
            print('Lanczos coefficients for A_PL loaded from disk for alpha={}'.format(alpha))
        except FileNotFoundError:
            barr = K_complexity(A_PL,vals,beta,ncoeff)
            pickle.dump(barr, open('bn_A_PL_alpha_{}.pkl'.format(alpha), 'wb'))
        ax[0].plot(coeff_arr, barr, label=r'$\alpha={}$'.format(alpha), marker='o',markersize=2)
        ax[0].annotate(r'$\hat{A}$', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=tp_fs, ha='center')
        ax[0].set_ylim([0, 8])

        Okey = 'B'
        B_PL = generate_O(Okey, L, J, Delta, alpha, basis, suffix)
        try:
            #raise FileNotFoundError
            barr = pickle.load(open('bn_B_PL_alpha_{}.pkl'.format(alpha), 'rb'))
            print('Lanczos coefficients for B_PL loaded from disk for alpha={}'.format(alpha))
        except FileNotFoundError:
            barr = K_complexity(B_PL,vals,beta,ncoeff)
            pickle.dump(barr, open('bn_B_PL_alpha_{}.pkl'.format(alpha), 'wb'))
        ax[1].plot(coeff_arr, barr, marker='o',markersize=2)
        ax[1].annotate(r'$\hat{B}$', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=tp_fs, ha='center')
        ax[1].set_ylim([0, 8])

    ax[0].set_xlabel(r'$n$', fontsize=xl_fs)
    ax[0].set_ylabel(r'$b_n$', fontsize=yl_fs)
    ax[1].set_xlabel(r'$n$', fontsize=xl_fs)
    fig.legend(loc=(0.35,0.02), fontsize=10, ncol=3)
    plt.tight_layout()
    plt.savefig('Fig4_SI_reply.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    exit()

if(0):
    B_PL =B_PL[:500,:500]
    plt.imshow(np.log10(np.abs(B_PL)), aspect='auto', cmap='viridis')
    plt.colorbar(label=r'$\log_{10}(|B_{ij}|)$')
    plt.title(r'$\hat{{B}}$ in computational basis, $\alpha={}$'.format(alpha))
    plt.xlabel(r'$i$')
    plt.ylabel(r'$j$')
    plt.show()

if(0):
    diag = np.diag(B_PL)
    plt.plot(diag, label=r'$\alpha={}$'.format(alpha))
    #plt.yscale('log')
    plt.xlabel(r'Index $i$')
    plt.ylabel(r'$\log_{10}(|B_{ii}|)$')
    plt.title(r'Diagonal elements of $\hat{{B}}$ in computational basis')
    plt.legend()
    plt.show()

if(0):
    plt.imshow(np.log10(np.abs(B_PL)), aspect='auto', cmap='viridis')
    plt.colorbar(label=r'$\log_{10}(|B_{ij}|)$')
    plt.title(r'$\hat{{B}}$ in computational basis, $\alpha={}$'.format(alpha))
    plt.xlabel(r'$i$')
    plt.ylabel(r'$j$')
    plt.show()

if(0):
    for beta in [0.5,1.0,2.0]:
        barr = K_complexity(B_PL,vals,beta,ncoeff)
        plt.plot(coeff_arr, barr, label=r'$\beta={}$'.format(beta), marker='o')
    plt.xlabel(r'Lanczos index $n$')
    plt.ylabel(r'$b_n$')
    plt.title(r'Lanczos coefficients for $\hat{{B}}$ in power-law XXZ model')
    plt.legend()
    plt.show()


