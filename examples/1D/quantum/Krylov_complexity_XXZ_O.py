import numpy as np
import matplotlib.pyplot as plt
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators 
from quspin.basis import spin_basis_1d # Hilbert space spin basis  
from PISC.engine import Krylov_complexity
from Krylov_XXZ_tools import construct_A, construct_B, construct_Jz, construct_JE, construct_JE_comm
import pickle
import matplotlib
import scipy

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

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
        print('File not found, constructing Hamiltonian and diagonalizing...')
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
            define_lattice(L, J, Delta, J2, Delta2, basis, suffix)
        else:
            define_lattice(L, J, Delta, 0, 0, basis, suffix)
        vals = pickle.load(open('eigenvalues_{}.pkl'.format(suffix), 'rb'))
        vecs = pickle.load(open('eigenvectors_{}.pkl'.format(suffix), 'rb'))
        H_mat = pickle.load(open('H_mat_{}.pkl'.format(suffix), 'rb'))
        print('Eigenvalues and eigenvectors loaded from disk')
        print('shape', vals.shape, vecs.shape, H_mat.shape)
    try:
        #raise FileNotFoundError
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
            O = construct_Jz(vecs, vals, H_mat, basis, L, J, Delta, J2, Delta2, trunc_perc=0.96 )
        elif Okey == 'JE':
            O = construct_JE_comm(vecs, vals, H_mat, basis, L, J, Delta, J2, Delta2, trunc_perc=0.96 )
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

basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=k,zblock=1)#, pblock=1) # and positive parity sector
print('L = {}, J = {}, Delta = {}, g = {}, J2 = {}, Delta2 = {}, k = {}'.format(L,J,Delta,g,J2,Delta2,k))


dynamic = []
beta = 1.0
ncoeff = 70
coeff_arr = np.arange(ncoeff)

store = False

suffix_NN = 'RepNN_L_{}'.format(L)
suffix_NNN = 'RepNNN_L_{}'.format(L)

vals_NN, vecs_NN, H_mat_NN = define_lattice(L, J, Delta, J2, Delta2, basis, suffix_NN)
vals_NNN, vecs_NNN, H_mat_NNN = define_lattice(L, J, Delta, J2, Delta2, basis, suffix_NNN)

#Okey = 'JE'
#O_NN = generate_O(Okey, L, J, Delta, 0, 0, basis, suffix_NN)
#O_NNN = generate_O(Okey, L, J, Delta, J2, Delta2, basis, suffix_NNN)

if(0): # Fig. 1 for SI/reply
    def func(key):
        if key == 'NN':
            return vals_NN, suffix_NN
        elif key == 'NNN':
            return vals_NNN, suffix_NNN

    beta = 1.0
    fig, ax = plt.subplots(1,3, figsize=(8,3), sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.15, wspace=0.0, hspace=0.0)    
    for Okey, num in zip(['A','B','JE'], [0,1,2]):
        vals,suffix = func('NN') 
        O = generate_O(Okey, L, J, Delta, 0.0, 0.0, basis, suffix)
        try:
            #raise FileNotFoundError
            b_arr = pickle.load(open('barr_{}_{}.pkl'.format(Okey, suffix), 'rb'))
            print('b_arr for {} loaded from disk'.format(Okey))
        except FileNotFoundError:
            b_arr = K_complexity(O,vals,beta=beta,ncoeff=ncoeff)
            pickle.dump(b_arr, open('barr_{}_{}.pkl'.format(Okey, suffix), 'wb'))
        s=2
        if(num==0):
            ax[num].plot(coeff_arr, b_arr, label='NN XXZ',marker='o', markersize=s)
        else:
            ax[num].plot(coeff_arr, b_arr, marker='o', markersize=s)

        vals,suffix = func('NNN')
        O = generate_O(Okey, L, J, Delta, J2, Delta2, basis, suffix)
        try:
            #raise FileNotFoundError
            b_arr = pickle.load(open('barr_{}_{}.pkl'.format(Okey, suffix), 'rb'))
            print('b_arr for {} loaded from disk'.format(Okey))
        except FileNotFoundError:
            b_arr = K_complexity(O,vals,beta=beta,ncoeff=ncoeff)
            pickle.dump(b_arr, open('barr_{}_{}.pkl'.format(Okey, suffix), 'wb'))
        if(num==0):
            ax[num].plot(coeff_arr, b_arr, label='NNN XXZ',marker='o', markersize=s)
        else:
            ax[num].plot(coeff_arr, b_arr, marker='o', markersize=s)
        keylist = [r'$\hat{A}$', r'$\hat{B}$', r'$\hat{j}_E$']
        ax[num].annotate(keylist[num], xy=(0.45, 0.9), xycoords='axes fraction', ha='center', fontsize=12)
        ax[num].set_ylim([-0.5,10])
    ax[0].set_xlabel(r'$n$', fontsize=10)
    ax[0].set_ylabel(r'$b_n$', fontsize=10)
    fig.legend(loc=(0.39,0.01), ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig('Fig1_SI_reply.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    exit()

if(0): # Fig. 2 for SI/reply
    fig, ax = plt.subplots(2,3, figsize=(9,6), sharex=True, sharey=True,layout='constrained')
    plt.subplots_adjust(bottom=0.15, wspace=0.0, hspace=0.0)
    for Okey, num in zip(['A','B','JE'], [0,1,2]):
        O = generate_O(Okey, L, J, Delta, 0.0, 0.0, basis, suffix_NN)
        im = ax[0,num].imshow(np.log10(np.abs(O[:500,:500])), aspect='auto')
        O = generate_O(Okey, L, J, Delta, J2, Delta2, basis, suffix_NNN)
        im = ax[1,num].imshow(np.log10(np.abs(O[:500,:500])), aspect='auto')
        keylist = [r'$\hat{A}$', r'$\hat{B}$', r'$\hat{j}_E$']
        ax[0,num].set_title(keylist[num], fontsize=12)
    
    ax[0,0].set_ylabel(r'$j$', fontsize=10)
    ax[1,0].set_ylabel(r'$j$', fontsize=10)
    
    ax[1,0].set_xlabel(r'$i$', fontsize=10)
    ax[1,1].set_xlabel(r'$i$', fontsize=10)
    ax[1,2].set_xlabel(r'$i$', fontsize=10)
    plt.tight_layout()
    fig.colorbar(im, ax=ax, shrink=0.95, pad=0.01)
    plt.savefig('Fig2_SI_reply.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    exit()

if(1): # Fig. 3 for SI/reply
    fig, ax = plt.subplots(1,2, figsize=(6,3), sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.15, wspace=0.0)
        
    beta = 1.0
    suffix = suffix_NNN
    for Okey, num in zip(['A','B'], [0,1]):
        O = generate_O(Okey, L, J, Delta, J2, Delta2, basis, suffix)
        try:
            #raise FileNotFoundError
            b_arr = pickle.load(open('barr_{}_{}.pkl'.format(Okey, suffix), 'rb'))
            print('b_arr for {} loaded from disk'.format(Okey))
        except FileNotFoundError:
            b_arr = K_complexity(O,vals_NN,beta=beta,ncoeff=ncoeff)
            pickle.dump(b_arr, open('barr_{}_{}.pkl'.format(Okey, suffix), 'wb'))
        ax[num].plot(coeff_arr, b_arr, marker='o', markersize=1.5, color='0.5')

        Odiag0 = O.copy()   
        np.fill_diagonal(Odiag0, 0.0) # Set diagonal elements of Odiag0 to zero
        try:
            #raise FileNotFoundError
            b_arr = pickle.load(open('barr_{}_{}_diag0.pkl'.format(Okey, suffix), 'rb'))
            print('b_arr for {} diag0 loaded from disk'.format(Okey))
        except FileNotFoundError:
            b_arr = K_complexity(Odiag0,vals_NN,beta=beta,ncoeff=ncoeff)
            pickle.dump(b_arr, open('barr_{}_{}_diag0.pkl'.format(Okey, suffix), 'wb'))
        ax[num].plot(coeff_arr, b_arr, marker='o', markersize=1.5, color='k',lw=1)
        
        keylist = [r'$\hat{A}$', r'$\hat{B}$']
        ax[num].annotate(keylist[num], xy=(0.49, 0.9), xycoords='axes fraction', ha='center', fontsize=12)
        ax[num].set_ylim([-0.5,6])
    ax[0].set_xlabel(r'$n$', fontsize=10)
    ax[0].set_ylabel(r'$b_n$', fontsize=10)
    ax[1].set_xlabel(r'$n$', fontsize=10)
    plt.tight_layout()
    plt.savefig('Fig3_SI_reply.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    exit()

if(0):
    compute_ip = Krylov_complexity.krylov_complexity.compute_ip

    key = 'NN' #N'
    if key == 'NN':
        O = O_NN
        vals = vals_NN
        suffix = suffix_NN
    elif key == 'NNN':
        O = O_NNN
        vals = vals_NNN
        suffix = suffix_NNN
    for Okey in ['JE','B','A']:#, 'B', 'A']:
        O = generate_O(Okey, L, J, Delta, J2, Delta2, basis, suffix)
        vals = vals_NNN[:O.shape[0]] # Ensure vals is the same length as O
        H = np.diag(vals)
        I = np.eye(O.shape[0])
        print('shape', H.shape, O.shape, vals.shape)
        Odiag0 = O.copy() 
        np.fill_diagonal(Odiag0, 0.0) # Set diagonal elements of Odiag0 to zero

        if(0):    
            #np.fill_diagonal(O_NNN, 0.0) # Set diagonal elements of O_NNN to zero
            for beta in [0.00001,1.0]:#4.0,5.0,10.0,20.0]:#,5.0]:#,0.5,1.0,2.0,5.0]:
                b_arr = np.zeros(ncoeff)
                #Remove trace of O
                tr = 0.0
                tr = compute_ip(O,I,beta,vals,0.5,tr,'wgm')
                #tr = np.trace(O)
                print(Okey,'beta',beta,'Trace', tr)

                b_arr = K_complexity(O,vals,beta=beta,ncoeff=ncoeff)
               
                #plot with dots and lines
                plt.plot(coeff_arr, b_arr, label='{} XXZ, operator {},beta={}'.format(key, Okey, beta),marker='o')
            plt.legend()
        if(1):
            plt.imshow(np.log10(np.abs(O[:500,:500])), aspect='auto')
            plt.colorbar()
            plt.title('{} XXZ, operator {}'.format(key,Okey))
            plt.show()
        if(0):
            for beta in [0.1,0.5,1.0,2.0,5.0]:
                ip_O = 0.0j
                ip_O = compute_ip(H,O,beta,vals,0.5,ip_O,'wgm')
                #print('Inner product of {} with H at beta={}:'.format(Okey, beta), ip_O)
                tr = 0.0
                print(Okey,beta,'Trace', compute_ip(O,np.eye(len(O)),beta,vals,0.5,tr,'wgm'))
    plt.show()

    exit()

if(0):
    O_NN = 0.1*np.diag(vals_NN) # Use the diagonal matrix of eigenvalues as O for NN XXZ
    O_NNN = 0.1*np.diag(vals_NNN) # Use the diagonal matrix of eigenvalues as O for NNN XXZ

    O_NN = np.diag(1e-4*np.arange(len(vals_NN),dtype=np.float64)) # Use a simple diagonal matrix as O for NN XXZ

    #print('O_NN', np.around(O_NN[:10,:10], decimals=4))
    O_NN += np.random.normal(0, 1e-2, O_NN.shape) # Add small random noise to O_NN
    O_NNN += np.random.normal(0, 1e-2, O_NNN.shape) # Add small random noise to O_NNN

if(0):
    neigs = len(vals_NN)
    narr = np.arange(neigs)
    plt.plot(narr,vals_NN[:neigs], label='NN XXZ')
    plt.plot(narr,vals_NNN[:neigs], label='NNN XXZ')
    plt.legend()
    plt.title('Eigenvalues of the Hamiltonian')
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.show()
    exit()

if(0):
    plt.imshow(np.log10(np.abs(O_NN)), aspect='auto')
    plt.colorbar()
    plt.title('NN XXZ, operator {}'.format(Okey))
    #plt.show()
    plt.savefig('NN_{}.png'.format(Okey), dpi=300)

    plt.clf()

    plt.imshow(np.log10(np.abs(O_NNN)), aspect='auto')
    plt.colorbar()
    plt.title('NNN XXZ, operator {}'.format(Okey))
    #plt.show()
    plt.savefig('NNN_{}.png'.format(Okey), dpi=300)

    exit()


neigs = len(O_NN)
if(0):
    for O, key in zip([O_NN, O_NNN], ['NN', 'NNN']):
        lenarr = np.arange(len(O))
        plt.plot(lenarr, np.log10(abs(O[:,0])), label='{} XXZ'.format(key),alpha=0.5)
        for i in range(neigs):
            #for j in range(neigs):
                #if abs(i-j) <=100:
                #    O[i,j] = 0.0
            O[i,i] = 0.0
        #O = scipy.ndimage.gaussian_filter(O, sigma=1.0) # Apply Gaussian smoothing to O
        #Ofilt = scipy.ndimage.uniform_filter(O, size=20) # Apply uniform smoothing to O
        #Ofilt = scipy.signal.savgol_filter(O, window_length=51, polyorder=4, axis=0)
        Oma = np.convolve(O[:,0], np.ones(len)/20, mode='same')
        #plt.plot(lenarr, np.log10(abs(Ofilt[:,0])), label='{} XXZ, diag zeroed'.format(key))
        plt.plot(lenarr, np.log10(abs(Oma)), label='{} XXZ, diag zeroed'.format(key))
    plt.show()
    exit()

if(0):
    for O, key in zip([O_NN, O_NNN], ['NN']):#, 'NNN']):
        lenarr = np.arange(len(O))
        plt.plot(lenarr, np.log10(abs(O[:,0])), label='{} XXZ'.format(key),alpha=0.2)
        
        n = 16
        #Plot every n-th value of O[:,0]
        plt.plot(lenarr[::n], np.log10(abs(O[::n,0])), label='{} XXZ, every {}-th value'.format(key,n))
    plt.legend()
    plt.title('Operator {} before smoothing'.format(Okey))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('log10(|O[i,0]|)')
    plt.show()
    exit()

if(0):
    #Plot every n-th value of O
    n = 1
    O_NNN = O_NNN[:500,:500]
    plt.imshow(np.log10(np.abs(O_NNN[::n,::n])), aspect='auto')
    plt.colorbar()
    plt.title('NNN XXZ, operator {}, every {}-th value'.format(Okey,n))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue index')
    plt.show()  
    exit()


fig,ax = plt.subplots(1,2, figsize=(6,3), sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.15, wspace=0.0)

kersize = 12
conv_mat = np.ones((kersize, kersize)) / (kersize**2)  # Normalized kernel for convolution

for O, vals, key in zip([O_NN, O_NNN], [vals_NN, vals_NNN], ['NN', 'NNN']):
    b_arr = np.zeros(ncoeff)
    b_arr = K_complexity(O,vals,beta=beta,ncoeff=ncoeff)
    ax[0].scatter(coeff_arr, b_arr, label='{} XXZ'.format(key),s=5)

    if(0):
        for i in range(neigs):
            #for j in range(neigs):
                #if abs(i-j) <=100:
                #    O[i,j] = 0.0
            O[i,i] = 0.0
        #O = scipy.ndimage.gaussian_filter(O, sigma=1.0) # Apply Gaussian smoothing to O
        #O = scipy.ndimage.uniform_filter(O, size=20) # Apply uniform smoothing to O
        #O = scipy.signal.convolve2d(O, conv_mat, mode='same') # Apply convolution smoothing to O

        b_arr = np.zeros(ncoeff)
        b_arr = K_complexity(O,vals,beta=beta,ncoeff=ncoeff)
        ax[1].scatter(coeff_arr, b_arr, label='{} XXZ'.format(key),s=5)
    
        #ax[1].plot(np.arange(len(O)), np.log10(abs(O[:,0])), label='{} XXZ'.format(key))
#plt.suptitle('Kernel Size = {}'.format(kersize))
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
