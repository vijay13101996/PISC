import numpy as np
import matplotlib.pyplot as plt
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators 
from quspin.basis import spin_basis_1d # Hilbert space spin basis  
from PISC.engine import Krylov_complexity
from Krylov_XXZ_tools import construct_B, construct_Jz, construct_JE, construct_JE_comm
import pickle
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
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


L = 20  # Length of the chain
J = 1.0  # Coupling constant
Delta = 0.55  # Anisotropy parameter

g = 0.0  # Magnetic field

#Next-nearest neighbor (NNN) coupling parameters
lamda = 1.0
J2 = lamda*J  # NNN coupling constant
Delta2 = 0.5  # NNN anisotropy parameter

#Power-law parameter
alpha = 5.0

k = 1 # x 2pi/L momentum sector

basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=k,zblock=1)#,pblock=1) # and positive parity sector
print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))

dynamic = []
beta = 0.25
ncoeff = 100
coeff_arr = np.arange(ncoeff)

if(0):# NN XXZ model
    try:
        raise FileNotFoundError
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

    B_eigenbasis = construct_B(vecs_NN, vals_NN, H_mat_NN, basis, L)
    b_arr = np.zeros(ncoeff)
    b_arr = K_complexity(B_eigenbasis,vals_NN,beta=beta,ncoeff=ncoeff)
    pickle.dump((coeff_arr, b_arr), open('krylov_coeffs_B_NN_L_{}.pkl'.format(L), 'wb'))
    #plt.scatter(coeff_arr, b_arr, label='NN XXZ')

if(0):# NNN XXZ model
    try:
        raise FileNotFoundError
        vals_NNN = pickle.load(open('eigenvalues_NNN.pkl', 'rb'))
        vecs_NNN = pickle.load(open('eigenvectors_NNN.pkl', 'rb'))
        H_mat_NNN = pickle.load(open('H_mat_NNN.pkl', 'rb'))
        print('NNN eigenvalues and eigenvectors loaded from disk')
    except FileNotFoundError:
        H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions
        static_NNN = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
        H_XXZ_NNN = hamiltonian(static_NNN,dynamic,basis=basis,dtype=np.complex128)
        H_mat_NNN = H_XXZ_NNN.toarray()
        print('NNN Hamiltonian constructed', H_mat_NNN.shape)
        E_NNN = H_XXZ_NNN.eigvalsh()
        vals_NNN, vecs_NNN = H_XXZ_NNN.eigh()
        #vals_NNN = np.sort(vals_NNN)
        pickle.dump(vals_NNN, open('eigenvalues_NNN.pkl', 'wb'))
        pickle.dump(vecs_NNN, open('eigenvectors_NNN.pkl', 'wb'))
        pickle.dump(H_mat_NNN, open('H_mat_NNN.pkl', 'wb'))

    B_eigenbasis = construct_B(vecs_NNN, vals_NNN, H_mat_NNN, basis, L)

    b_arr = np.zeros(ncoeff)
    b_arr = K_complexity(B_eigenbasis,vals_NNN,beta=beta,ncoeff=ncoeff)
    pickle.dump((coeff_arr, b_arr), open('krylov_coeffs_B_NNN.pkl', 'wb'))

if(1):# Power-law XXZ model
    ind_list = [(i,j) for i in range(L) for j in range(i+1,L)]
    #print('Pairs of interactions in power-law model:', (ind_list))

    for alpha in [0.5, 1.0, 5.0]:
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
        #vals_PL = np.sort(vals_PL)
   
        if(0): # Verify power-law interactions by plotting log-log of strength vs distance
            distances = [bond[1] for bond in H_zz_PL] # This assumes dist was stored or re-calculated
            strengths = [bond[0] for bond in H_zz_PL]

            dist_list = []
            strength_list = []
            for bond in H_zz_PL:
                strength = bond[0]
                dist = min(abs(bond[1]-bond[2]), L-abs(bond[1]-bond[2]))  # periodic boundary conditions
                dist_list.append(dist)
                strength_list.append(strength)
            
            plt.plot(np.log(dist_list), np.log(strength_list), 'o-', label='Power-law interactions (alpha=%.1f)'%alpha)
            plt.title(f"Alpha = {alpha} Verification")
            plt.show()
            exit()

        B_eigenbasis = construct_B(vecs_PL, vals_PL, H_mat_PL, basis, L)
        b_arr = np.zeros(ncoeff)
        b_arr = K_complexity(B_eigenbasis,vals_PL,beta=beta,ncoeff=ncoeff)
        pickle.dump((coeff_arr, b_arr), open('krylov_coeffs_power_law_alpha_{}_beta_{}.pkl'.format(alpha,beta), 'wb'))

if(1):
    alpha_list = [0.5, 1.0, 5.0]
    beta_list = [0.25]

    fig,axs = plt.subplots((1,2), len(beta_list), figsize=(6,3))
    plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.25)

    for i, beta in enumerate(beta_list):
        for j, alpha in enumerate(alpha_list):
            coeff_arr, b_arr = pickle.load(open('krylov_coeffs_power_law_alpha_{}_beta_{}.pkl'.format(alpha,beta), 'rb'))
            if(i==0):
                axs[i].scatter(coeff_arr, b_arr, label=r'$\alpha=%.1f$'%alpha, s=5)
            else:
                axs[i].scatter(coeff_arr, b_arr, s=5)
                axs[i].set_yticks([])  # Hide y-axis ticks for the second plot

            axs[i].set_title(r'$\beta={}$'.format(beta), fontsize=tp_fs)
            axs[i].set_xlabel(r'$n$', fontsize=xl_fs)
            if i==0:
                axs[i].set_ylabel(r'$b_n$', fontsize=yl_fs)
   
        axs[i].annotate(r'$\hat{B}$', xy=(0.5, 0.9), xytext=(0.5, 0.9), textcoords='axes fraction', fontsize=ti_fs, ha='center')
        axs[i].set_ylim([-0.5,9.5])
        axs[i].set_xlim([-5, 70])
        
    fig.legend(loc=(0.15,-0.012), ncol=3, fontsize=le_fs)

    #plt.savefig('krylov_coefficients_comparison_3.pdf', bbox_inches='tight')
    #coeff_arr, b_arr = pickle.load(open('krylov_coeffs_NN.pkl', 'rb'))
    #plt.scatter(coeff_arr, b_arr, label='NN XXZ')
    #coeff_arr, b_arr = pickle.load(open('krylov_coeffs_NNN.pkl', 'rb'))
    #plt.scatter(coeff_arr, b_arr, label='NNN XXZ')

    #plt.show()


if(0):
    diff_NN =np.diff(vals_NN)
    diff_NN/= np.mean(diff_NN)
    diff_NNN =np.diff(vals_NNN)
    diff_NNN/= np.mean(diff_NNN)
    diff_PL =np.diff(vals_PL)
    diff_PL/= np.mean(diff_PL)

    #plt.hist(diff_NN, bins=50, alpha=0.5, label='NN XXZ',range=(0,5), density=True)
    #plt.hist(diff_NNN, bins=50, alpha=0.5, label='NNN XXZ',range=(0,5), density=True)
    plt.hist(diff_PL, bins=50, alpha=0.5, label='Power-law XXZ',range=(0,2), density=True)
    plt.xlabel('Energy')
    plt.ylabel('Density of States')
    plt.title('Energy Spectrum Comparison')
    plt.legend()
    plt.show()

