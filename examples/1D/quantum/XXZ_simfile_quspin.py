import numpy as np
from matplotlib import pyplot as plt
from PISC.engine.xxz import XXZ
import cProfile
import time
import scipy
import numpy
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from joblib import Parallel, delayed
import pickle
from functools import partial
from multiprocessing import Pool
import math
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators 
from quspin.basis import spin_basis_1d # Hilbert space spin basis  


def unfold_spectrum(eigenvalues, num_points=100):
    """
    Unfold the spectrum of eigenvalues using a spline fit.
    eigenvalues: Array of eigenvalues
    num_points: Number of points to use for the spline fit

    Returns:
    unfolded_eigenvalues: Array of unfolded eigenvalues
    """
    # Sort the eigenvalues
    uq_val = (np.sort(eigenvalues))

    #Erange = np.linspace(eigenvalues.min(), eigenvalues.max(), num_points)
    #nE = np.array([np.sum(eigenvalues <= val) for val in Erange])

    Erange = eigenvalues[::20]
    nE = np.array([np.sum(eigenvalues <= val) for val in Erange])

    #Spline fit nE against Erange, i.e. nE = f(Erange)
    print('Fitting spline to the spectrum')
    fit = scipy.interpolate.UnivariateSpline(Erange, nE, s=0.0)
    #fit = scipy.interpolate.PchipInterpolator(Erange, nE)
    unfolded_eigenvalues = fit(eigenvalues)  # Unfold the eigenvalues using the spline fit
    
    print('Unfolded eigenvalues computed', unfolded_eigenvalues[:10])
    print('eigenvalues', eigenvalues[:10])

    return unfolded_eigenvalues

def main():
    start_time = time.time() 
    

    L = 20  # Length of the chain
    J = 1.0  # Coupling constant
    Delta = 0.5  # Anisotropy parameter
    g = 0.0  # Magnetic field
    J2 = 0.5*J  # NNN coupling constant
    Delta2 = 0.5  # NNN anisotropy parameter

    k =  0.0 # X 2 * np.pi / L  # Wavevector for the k=2\pi/L sector

    basis = spin_basis_1d(L,pauli=False,Nup=L//2,kblock=1,pblock=1,zblock=1) # and positive parity sector

    for J2 in [0.0, 1.0*J]:

        H_zz_NN = [[J*Delta,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_zz_NNN = [[J2*Delta2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions

        H_xy_NN = [[0.5*J,i,(i+1)%L] for i in range(L)] # periodic boundary conditions
        H_xy_NNN = [[0.5*J2,i,(i+2)%L] for i in range(L)] # periodic boundary conditions

        static = [["zz",H_zz_NN],["zz",H_zz_NNN],["+-",H_xy_NN],["-+",H_xy_NN],["+-",H_xy_NNN],["-+",H_xy_NNN]]
        dynamic = []

        H_XXZ_NNN = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
        H_mat = H_XXZ_NNN.toarray()
        E = H_XXZ_NNN.eigvalsh()
        vals, vecs = H_XXZ_NNN.eigh()
        print('Eigenvalues computed','basis size', len(vals))
        exit() 
        U = vecs
        U_inv = np.conjugate(U.T)
        D = np.diag(E)
        H_reconstructed = U @ D @ U_inv
        assert np.allclose(H_mat, H_reconstructed), "Matrix reconstruction failed!"

        A = [[1.0/L, i, (i+1)%L] for i in range(L)]  # S^z_i S^z_{i+1}
        static_A = [["zz", A]]
        A_op = hamiltonian(static_A, [], basis=basis, dtype=np.complex128)
        A_mat = A_op.toarray()
        A_eigenbasis = U_inv @ A_mat @ U  # Transform A to the eigenbasis
    
        B = [[1.0/L, i, (i+2)%L] for i in range(L)]  # S^z_i S^z_{i+2}
        static_B = [["+-", B], ["-+", B]]
        B_op = hamiltonian(static_B, [], basis=basis, dtype=np.complex128)
        B_mat = B_op.toarray()
        B_eigenbasis = U_inv @ B_mat @ U  # Transform B to the eigenbasis

        plt.plot(abs(B_eigenbasis)[5,:], label='J2=%.1f'%J2)
    
    plt.yscale('log')
    plt.xlabel('Eigenstate index')
    plt.ylabel('Observable matrix element')
    plt.legend()
    plt.show()
    exit()

    
    if(0):
        A0 = np.log(abs(A_eigenbasis[5,:]))
        nrange = np.arange(len(A0))
        #Fit A0 vs nrange (y vs x) to linear
        coeffs = np.polyfit(nrange, A0, 1)
        fit = np.polyval(coeffs, nrange)
        plt.plot(nrange, A0, label='A in eigenbasis')
        plt.plot(nrange, fit, label='Exponential fit')
        plt.show()
        exit()




    print('Observable matrix in eigenbasis computed')
    plt.plot(abs(B_eigenbasis[5,:]), label='A in site basis')
    #log scale
    plt.yscale('log')
    plt.show()


    #unfolded_E = unfold_spectrum(E)

    

    if(0):
        bins = np.arange(0,3.01, 0.1)
        diff = np.diff(unfolded_E)
        print('Mean level spacing:', np.mean(diff))
        #diff/np.mean(diff)
        
        plt.hist(diff, bins=bins, density=True)
        plt.xlabel('Energy difference')
        plt.ylabel('Density')
        plt.title('Energy differences for the k=0, P and Z2 invariant sector')
        plt.show()



if __name__ == "__main__":
    start_time = time.time()
    #cProfile.run('main()')
    main()
    end_time = time.time()
    print("Execution time:", end_time - start_time, "seconds")




