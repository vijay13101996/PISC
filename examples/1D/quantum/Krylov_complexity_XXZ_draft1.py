import numpy as np
from matplotlib import pyplot as plt
import time
import scipy
import math
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators 
from quspin.basis import spin_basis_1d # Hilbert space spin basis  
from PISC.engine import Krylov_complexity
#from PISC.dvr.dvr import DVR1D
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det
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

    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    
    return barr


def init_H_XXZ_NNN(L,J,Delta,g,J2,Delta2,k):
    basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=k,zblock=1,pblock=1) # and positive parity sector

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

    return H_mat, vals, vecs, basis

def find_K_XXZ(L,J,Delta,g,J2,Delta2,k,fig,ax,label,beta=0.5,ncoeff=50,neigs=None):
    H_mat, vals, vecs, basis = init_H_XXZ_NNN(L,J,Delta,g,J2,Delta2,k)

    #Store in pickle files
    fname = 'XXZ_NNN_L{}_J2{}'.format(L,J2)
    pickle.dump(H_mat, open(fname + '_Hmat.pkl', 'wb'))
    pickle.dump(vals, open(fname + '_vals.pkl', 'wb'))
    pickle.dump(vecs, open(fname + '_vecs.pkl', 'wb'))

    U = vecs
    U_inv = np.conjugate(U.T)
    D = np.diag(vals)
    H_reconstructed = U @ D @ U_inv
    assert np.allclose(H_mat, H_reconstructed), "Matrix reconstruction failed!"

    pickle.dump(U, open(fname + '_U.pkl', 'wb'))
    
    if(1):
        A = [[1.0/L, i, (i+1)%L] for i in range(L)]  # S^z_i S^z_{i+1}
        static_A = [["zz", A]]
        A_op = hamiltonian(static_A, [], basis=basis, dtype=np.complex128)
        A_mat = A_op.toarray()
        A_eigenbasis = U_inv @ A_mat @ U  # Transform A to the eigenbasis
        
        if(neigs is not None):
            A_eigenbasis = A_eigenbasis[:neigs,:neigs]
            vals = vals[:neigs]
            print('B_eigenbasis shape = ', A_eigenbasis.shape)
        pickle.dump(A_eigenbasis, open(fname + 'A_eigenbasis.pkl', 'wb'))


    if(1):
        B = [[1.0/L, i, (i+2)%L] for i in range(L)]  # S^z_i S^z_{i+2}
        static_B = [["+-", B], ["-+", B]]
        B_op = hamiltonian(static_B, [], basis=basis, dtype=np.complex128)
        B_mat = B_op.toarray()
        B_eigenbasis = U_inv @ B_mat @ U  # Transform B to the eigenbasis
        if(neigs is not None):
            B_eigenbasis = B_eigenbasis[:neigs,:neigs]
            vals = vals[:neigs]
            print('B_eigenbasis shape = ', B_eigenbasis.shape)

        pickle.dump(B_eigenbasis, open(fname + 'B_eigenbasis.pkl', 'wb'))

    Brow = np.log(abs(B_eigenbasis[0,:]))

    #Smooth Brow using a Gaussian filter
    from scipy.ndimage import gaussian_filter1d
    Brow_smooth = gaussian_filter1d(Brow, sigma=10)
    Brow_smooth = Brow_smooth[200:]

    #ax[0].plot(np.log(abs(B_eigenbasis[0,:])), label=label)
    #ax[0].plot(Brow_smooth)

    #Fit log Brow_smooth to a linear function using numpy polyfit
    fit = np.polyfit(np.arange(len(Brow_smooth)), Brow_smooth, 1)
    print('fit = ', fit)
    linear_func = np.poly1d(fit)
    xdata = np.arange(len(Brow_smooth))
    popt = fit
    #ax[0].plot(linear_func(xdata), '--', label='fit slope={:.3f}'.format(popt[0]))


    b_arr = K_complexity(A_eigenbasis,vals,beta=beta,ncoeff=ncoeff)

    #ax[1].scatter(np.arange(len(b_arr[1:])), b_arr[1:], label=label, s=10)

    return np.arange(len(b_arr[1:])), b_arr[1:]


def EEV(J,Delta,g,J2,Delta2,L,k):
    H_mat, vals, vecs, basis = init_H_XXZ_NNN(L,J,Delta,g,J2,Delta2,k)

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

    B = [[1.0/L, i, (i+2)%L] for i in range(L)]  # S^z_i S^z_{i+2}
    static_B = [["zz", B]]
    B_op = hamiltonian(static_B, [], basis=basis, dtype=np.complex128)
    B_mat = B_op.toarray()
    B_eigenbasis = U_inv @ B_mat @ U  # Transform B to the eigenbasis 

    plt.scatter(vals, np.diagonal(B_eigenbasis), s=1, c=abs(np.diagonal(B_eigenbasis)), cmap='viridis')
    plt.show()


def tau_O(J,Delta,g,J2,Delta2,L,k,neigs=None):
    H_mat, vals, vecs, basis = init_H_XXZ_NNN(L,J,Delta,g,J2,Delta2,k)
    
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

    if(neigs is None):
        neigs = H_mat.shape[0]

    if(neigs is not None):
        A_eigenbasis = A_eigenbasis[:neigs,:neigs]
        vals = vals[:neigs]
        print('A_eigenbasis shape = ', A_eigenbasis.shape)

    if(1):
        B = [[1.0/L, i, (i+2)%L] for i in range(L)]  # S^z_i S^z_{i+2}
        static_B = [["zz", B]]
        B_op = hamiltonian(static_B, [], basis=basis, dtype=np.complex128)
        B_mat = B_op.toarray()
        B_eigenbasis = U_inv @ B_mat @ U  # Transform B to the eigenbasis 


    omega_grid = np.linspace(0.0, 10.0, 51)
    tau_O = np.zeros(len(omega_grid))
    for n in range(len(omega_grid)):
        w0 = omega_grid[n]
        tau_O_num = 0.0
        tau_O_den = 0.0
        count = 0
        for i in range(neigs):
            for j in range(neigs):
                E_bar = (vals[i]+vals[j])/2.0
                omega = abs(vals[i]-vals[j])
                if (E_bar/L <= 0.025 and abs(omega-w0)<=0.175):
                    tau_O_num += abs(B_eigenbasis[i,j])**2
                    tau_O_den += abs(B_eigenbasis[i,j])
                    count += 1
        tau_O_den#/=count
        tau_O_num#/=count
        tau_O[n] = tau_O_num#/tau_O_den**2
        print('w0 = {}, tau_O = {}'.format(w0, tau_O[n]))

    plt.plot(omega_grid, tau_O, '-',label='L={}, neigs={}'.format(L,neigs))

    return omega_grid, tau_O
L = 20  # Length of the chain
J = 1.0  # Coupling constant
Delta = 0.55  # Anisotropy parameter
g = 0.0  # Magnetic field
lamda = 1.0
J2 = lamda*J  # NNN coupling constant
Delta2 = 0.5  # NNN anisotropy parameter


k = 1  # Momentum sector

def plot_ax_3b(ax):
        #fig, ax = plt.subplots(1,2, figsize=(10, 5))
        #for L in [20]:
        L=20
        J2 = 1.0
        Delta2 = 0.5
        print('Loading pickle files') 
        beta = 1.0
        fname_pref = 'XXZ_NNN_L{}_Delta{}_J2{}_Delta2{}_k{}_beta_{}'.format(L,Delta,J2,Delta2,k,beta)
        bnarr1 = pickle.load(open('bnarr'+fname_pref+'_neigs4400.pkl', 'rb'))
        print('Loaded bnarr'+fname_pref+'_neigs4400.pkl')
        coeffarr = np.arange(1,len(bnarr1[1:])+1)
        ax.scatter(coeffarr, bnarr1[1:], label=r'$NNN$'.format(L,J2), s=10, color='blue')
        
        fit1 = np.polyfit(coeffarr[1:10], bnarr1[1:10], 1)
        linear_func1 = np.poly1d(fit1)
        ax.plot(coeffarr[0:18], linear_func1(coeffarr[0:18]), '--', lw=2,color='blue')

        J2 = 0.0
        Delta2 = 0.0
        fname_pref = 'XXZ_NNN_L{}_Delta{}_J2{}_Delta2{}_k{}_beta_{}'.format(L,Delta,J2,Delta2,k,beta)
        bnarr2 = pickle.load(open('bnarr'+fname_pref+'_neigs4400.pkl', 'rb'))
        print('Loaded bnarr'+fname_pref+'_neigs4400.pkl')

        bnarr3 = pickle.load(open('bnarr'+fname_pref+'_neigs2000.pkl', 'rb'))
        print('Loaded bnarr'+fname_pref+'_neigs2000.pkl')
        fit3 = np.polyfit(coeffarr[1:15], bnarr3[1:15], 1)
        linear_func3 = np.poly1d(fit3)
        ax.plot(coeffarr[1:44], linear_func3(coeffarr[1:44]), '--', lw=2,color='orange')

        ax.scatter(coeffarr, bnarr2[1:], label=r'$NN$'.format(L), s=10, color='green')
        ax.scatter(coeffarr, bnarr3[1:], label=r'$NN \:(trunc)$'.format(L), s=10, color='orange')

        ax.plot(coeffarr[0:3], np.pi*(coeffarr[0:3]-1)/beta + fit1[0] + fit1[1]*coeffarr[0], '--', color='red', lw=2)

        ax.set_xlabel(r'$n$', fontsize=xl_fs)
        ax.set_ylabel(r'$b_n$', fontsize=yl_fs)
        ax.legend(loc='lower right', ncol=3, columnspacing=0.0, fontsize=9)
        #ax.set_xlim(ax.get_xlim()[0], 60)
        ax.set_ylim(ax.get_ylim()[0]-0.2, ax.get_ylim()[1]*1.2)
        ax.annotate(r'$(b)$', xy=(0.02, 0.9), xycoords='axes fraction', fontsize=xl_fs)

        #### Save into txt files
        np.savetxt('FIG3b_bn_vs_n_XXZ_NNN_J2_{}.txt'.format(J2), np.column_stack((np.arange(len(bnarr1[1:])), bnarr1[1:])), header='n b_n', comments='')
        np.savetxt('FIG3b_bn_vs_n_XXZ_NN.txt', np.column_stack((np.arange(len(bnarr2[1:])), bnarr2[1:])), header='n b_n', comments='')
        np.savetxt('FIG3b_bn_vs_n_XXZ_NN_trunc.txt', np.column_stack((np.arange(len(bnarr3[1:])), bnarr3[1:])), header='n b_n', comments='')

def plot_ax_3b_in(ax):
        k=1
        L=20
        J2=1.0
        Delta2=0.5
        #EEV(J,Delta,g,J2,Delta2,L,k,neigs=2000)
        wgrid = pickle.load(open('Bop_wgrid_L{}_J2{}.pkl'.format(L,J2), 'rb'))
        
        tauO = pickle.load(open('Bop_tauO_L{}_J2{}_act_neigs4400.pkl'.format(L,J2), 'rb'))
        tauO1 = pickle.load(open('Bop_tauO_L{}_J2{}_neigs4400.pkl'.format(L,J2), 'rb'))
        tauO2 = pickle.load(open('Bop_tauO_L{}_J2{}_neigs2000.pkl'.format(L,J2), 'rb'))
        print('Loaded tauO pickle files', 'wgrid_L{}_J2{}.pkl'.format(L,J2), 'tauO_L{}_J2{}_neigs4400.pkl'.format(L,J2), 'tauO_L{}_J2{}_neigs2000.pkl'.format(L,J2))
        
        ax.plot(wgrid, tauO, '-',label=r'$L={},J_2={}$'.format(L,J2), color='blue')
        #Fit wgrid vs tauO1 between w0 =4 and 6
        init = 21
        fit = np.polyfit(wgrid[init:45], np.log(tauO[init:45]), 1)
        print('fit = ', fit, len(wgrid))
        linear_func = np.poly1d(fit)
        ax.plot(wgrid[init:], np.exp(linear_func(wgrid[init:])), '--', color='blue',lw=2)

        #Fit wgrid vs tauO1 between w0 =4 and 6
        init = 19
        J2=0.0
        fit1 = np.polyfit(wgrid[init:30], np.log(tauO1[init:30]), 1)
        print('fit = ', fit1)
        linear_func = np.poly1d(fit1)
        ax.plot(wgrid[init:], np.exp(linear_func(wgrid[init:])), '--', color='green',lw=2)
        ax.plot(wgrid, tauO1, '-',label=r'$L={},J_2={}$'.format(L,J2), color='green')

        fit2 = np.polyfit(wgrid[init:30], np.log(tauO2[init:30]), 1)
        print('fit2 = ', fit2)
        linear_func2 = np.poly1d(fit2)
        ax.plot(wgrid[init:], np.exp(linear_func2(wgrid[init:])), '--', color='orange',lw=2)
        ax.plot(wgrid[:32], tauO2[:32], '-',label=r'$L={},J_2={}$ \: (trunc)'.format(L,J2), color='orange')
        
        ax.set_yscale('log')
        #ax[1].axhline(y=np.pi/2, color='r', linestyle='--', label=r'$\pi/2$')
        ax.set_xlabel(r'$\omega$', fontsize=xl_fs-4)
        ax.set_ylabel(r'$f_{{O}}(0,\omega)$', fontsize=yl_fs-4)
        #ax.legend(fontsize=le_fs-2,loc='upper right')
        #ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1e8)
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.annotate(r'$(b)$', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=tp_fs+2)

        ### Save into txt files
        np.savetxt('FIG3b_tauO_vs_w_XXZ_NNN_J2_{}.txt'.format(J2), np.column_stack((wgrid, tauO)), header='omega tau_O', comments='')
        np.savetxt('FIG3b_tauO_vs_w_XXZ_NN.txt', np.column_stack((wgrid, tauO1)), header='omega tau_O', comments='')
        np.savetxt('FIG3b_tauO_vs_w_XXZ_NN_trunc.txt', np.column_stack((wgrid, tauO2)), header='omega tau_O', comments='')

def main():
    start_time = time.time() 
    

    L = 20  # Length of the chain
    J = 1.0  # Coupling constant
    Delta = 0.55  # Anisotropy parameter
    g = 0.0  # Magnetic field
    lamda = 1.0
    J2 = lamda*J  # NNN coupling constant
    Delta2 = 0.5  # NNN anisotropy parameter


    k = 1  # Momentum sector
    
    #fig, ax = plt.subplots(1,2, figsize=(10, 5))
    #fig.subplots_adjust(wspace=0.3)
    
    def plot_ax_3b(ax):
        #fig, ax = plt.subplots(1,2, figsize=(10, 5))
        #for L in [20]:
        L=20
        nonlocal J2
        nonlocal Delta2
        print('Loading pickle files') 
        beta = 1.0
        fname_pref = 'XXZ_NNN_L{}_Delta{}_J2{}_Delta2{}_k{}_beta_{}'.format(L,Delta,J2,Delta2,k,beta)
        bnarr1 = pickle.load(open('bnarr'+fname_pref+'_neigs4400.pkl', 'rb'))
        print('Loaded bnarr'+fname_pref+'_neigs4400.pkl')
        coeffarr = np.arange(1,len(bnarr1[1:])+1)
        ax.scatter(coeffarr, bnarr1[1:], label=r'$L={},J_2={}$'.format(L,J2), s=10, color='blue')
        
        fit1 = np.polyfit(coeffarr[1:10], bnarr1[1:10], 1)
        linear_func1 = np.poly1d(fit1)
        ax.plot(coeffarr[0:18], linear_func1(coeffarr[0:18]), '--', lw=2,color='blue')

        J2 = 0.0
        Delta2 = 0.0
        fname_pref = 'XXZ_NNN_L{}_Delta{}_J2{}_Delta2{}_k{}_beta_{}'.format(L,Delta,J2,Delta2,k,beta)
        bnarr2 = pickle.load(open('bnarr'+fname_pref+'_neigs4400.pkl', 'rb'))
        print('Loaded bnarr'+fname_pref+'_neigs4400.pkl')

        bnarr3 = pickle.load(open('bnarr'+fname_pref+'_neigs2000.pkl', 'rb'))
        print('Loaded bnarr'+fname_pref+'_neigs2000.pkl')
        fit3 = np.polyfit(coeffarr[1:15], bnarr3[1:15], 1)
        linear_func3 = np.poly1d(fit3)
        ax.plot(coeffarr[1:44], linear_func3(coeffarr[1:44]), '--', lw=2,color='orange')

        ax.scatter(coeffarr, bnarr2[1:], label=r'$L={},J_2=0$'.format(L), s=10, color='green')
        ax.scatter(coeffarr, bnarr3[1:], label=r'$L={},J_2=0 \: (trunc)$'.format(L), s=10, color='orange')

        ax.plot(coeffarr[0:3], np.pi*(coeffarr[0:3]-1)/beta + fit1[0] + fit1[1]*coeffarr[0], '--', color='red', lw=2)

        ax.set_xlabel(r'$n$', fontsize=xl_fs)
        ax.set_ylabel(r'$b_n$', fontsize=yl_fs)
        ax.legend(fontsize=le_fs-2,loc='upper right')
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.15)
        ax.annotate(r'$(a)$', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=tp_fs+2)
    
    def plot_ax_3b_in(ax):
        k=1
        L=20
        J2=1.0
        #EEV(J,Delta,g,J2,Delta2,L,k,neigs=2000)
        wgrid = pickle.load(open('Bop_wgrid_L{}_J2{}.pkl'.format(L,J2), 'rb'))
        
        tauO = pickle.load(open('Bop_tauO_L{}_J2{}_act_neigs4400.pkl'.format(L,J2), 'rb'))
        tauO1 = pickle.load(open('Bop_tauO_L{}_J2{}_neigs4400.pkl'.format(L,J2), 'rb'))
        tauO2 = pickle.load(open('Bop_tauO_L{}_J2{}_neigs2000.pkl'.format(L,J2), 'rb'))
        print('Loaded tauO pickle files', 'wgrid_L{}_J2{}.pkl'.format(L,J2), 'tauO_L{}_J2{}_neigs4400.pkl'.format(L,J2), 'tauO_L{}_J2{}_neigs2000.pkl'.format(L,J2))
        
        ax.plot(wgrid, tauO, '-',label=r'$L={},J_2={}$'.format(L,J2), color='blue')
        #Fit wgrid vs tauO1 between w0 =4 and 6
        init = 21
        fit = np.polyfit(wgrid[init:45], np.log(tauO[init:45]), 1)
        print('fit = ', fit, len(wgrid))
        linear_func = np.poly1d(fit)
        ax.plot(wgrid[init:], np.exp(linear_func(wgrid[init:])), '--', color='blue',lw=2)

        #Fit wgrid vs tauO1 between w0 =4 and 6
        init = 19
        J2=0.0
        fit1 = np.polyfit(wgrid[init:30], np.log(tauO1[init:30]), 1)
        print('fit = ', fit1)
        linear_func = np.poly1d(fit1)
        ax.plot(wgrid[init:], np.exp(linear_func(wgrid[init:])), '--', color='green',lw=2)
        ax.plot(wgrid, tauO1, '-',label=r'$L={},J_2={}$'.format(L,J2), color='green')

        fit2 = np.polyfit(wgrid[init:30], np.log(tauO2[init:30]), 1)
        print('fit2 = ', fit2)
        linear_func2 = np.poly1d(fit2)
        ax.plot(wgrid[init:], np.exp(linear_func2(wgrid[init:])), '--', color='orange',lw=2)
        ax.plot(wgrid[:32], tauO2[:32], '-',label=r'$L={},J_2={}$ \: (trunc)'.format(L,J2), color='orange')
        
        ax.set_yscale('log')
        #ax[1].axhline(y=np.pi/2, color='r', linestyle='--', label=r'$\pi/2$')
        ax.set_xlabel(r'$\omega$', fontsize=xl_fs)
        ax.set_ylabel(r'$f_{{O}}(0,\omega)$', fontsize=yl_fs)
        ax.legend(fontsize=le_fs-2,loc='upper right')
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1e10)
        ax.annotate(r'$(b)$', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=tp_fs+2)

    #plot_ax_3a(ax[0])
    #plot_ax_3b_in(ax[1])
    #fig.savefig('draft1_Krylov_Bop_XXZ_NNN.pdf', dpi=300, bbox_inches='tight')
    #plt.show()

if(0):
    if __name__ == "__main__":
        start_time = time.time()
        #cProfile.run('main()')
        main()
        end_time = time.time()
        print("Execution time:", end_time - start_time, "seconds")




