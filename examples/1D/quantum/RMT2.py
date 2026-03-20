import numpy as np
from PISC.dvr.dvr import DVR1D
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity
import scipy

#Compute Krylov complexity for energy eigenvalues coming from a random-matrix Hamiltonian

sigma = 1
neigs = 500

def random_matrix(neigs):
    """Generate a random NxN matrix with elements drawn from a normal distribution."""
    H = np.zeros((neigs, neigs))

    for i in range(neigs):
        for j in range(i,neigs):
            if i == j:
                H[i, j] = np.random.normal(0, np.sqrt(2)*sigma)
            else:
                H[i, j] = np.random.normal(0, sigma)
            H[j, i] = H[i, j]
    return H

def TCF_O(vals, beta, neigs, O, t_arr):
    
    n_arr = np.arange(neigs)
    m_arr = np.arange(neigs)

    C_arr = np.zeros_like(t_arr) + 0j

    for n in n_arr:
        for m in m_arr:
            C_arr += np.exp(-beta*(vals[n]+vals[m])/2) * np.exp(1j*(vals[n]-vals[m])*t_arr) * np.abs(O[n,m])**2

    Z = np.sum(np.exp(-beta*vals))
    C_arr /= Z

    return C_arr

def Krylov_O(vals, beta, neigs, O, ncoeff):
    L = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,L)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff) 
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    return barr

def rho(E):
    #Density of states
    return 1.0#abs(E)**4

def f_O1(vals,n,m):
    E = (vals[n]+vals[m])/2
    w = (vals[n]-vals[m])
    a = 0.05
    #print('rho', E, rho(E))
    return np.exp(-a*abs(w) - 0.00*abs(w)**2)#*np.random.normal(0,100)/rho(E)**0.5

    #return np.random.normal(0,1)#/rho(E)**0.5

def f_O2(vals,n,m):
    E = (vals[n]+vals[m])/2
    w = (vals[n]-vals[m])
    a = 0.1
    #print('rho', E, rho(E))
    b = 10
    return 1.0 #np.exp(-a*abs(w))#/(1+b*abs(w)))

#f_O2 = np.vectorize(f_O2)

nmat = 1
ncoeff = 100
t_arr = np.linspace(-100, 100, 1000)
C_arr = np.zeros_like(t_arr) + 0j
n_arr = np.arange(ncoeff)
b_arr = np.zeros(ncoeff)

beta = 0.25
diffs = []

for f_O in [f_O2]:
    for i in range(nmat):
        H = random_matrix(neigs)
        vals, vecs = np.linalg.eigh(H)
        vals = np.sort(vals)

        diff = np.diff(vals)
        diffs.extend(diff)

        O = np.zeros((neigs, neigs))
        a = 0.1
        for n in range(neigs):
            for m in range(n,neigs):
                O[n, m] = np.random.normal(0,1)
                #O[n,m] = np.exp(-a*abs(vals[n]-vals[m])**2)
                #O[n,m] = f_O(vals,n,m)*np.random.normal(0,1)
                O[m,n] = O[n,m]

        
        C_arr += TCF_O(vals, beta, neigs, O, t_arr)

        b_arr= Krylov_O(vals, beta, neigs, O, ncoeff)
        OH =  O + 5*np.diag(vals)
        b_arr2 = Krylov_O(vals, beta, neigs, OH, ncoeff)

    #plt.scatter(n_arr[1:], b_arr[1:].real/nmat, label=f'O={f_O.__name__}')


if(0):
    plt.plot(np.arange(len(vals)), vals, label='Eigenvalues')
    plt.show()
    exit()

if(0):
    plt.xlabel('Coefficient index')
    plt.ylabel('Coefficient value')
    plt.title('Krylov Coefficients for Random Matrix Hamiltonians')
    plt.legend()
    plt.show()
    exit()

if(1):
    b_arr /= nmat
    plt.scatter(n_arr[1:], b_arr[1:].real, label='Krylov coefficients')
    plt.scatter(n_arr[1:], b_arr2[1:].real, label='Krylov coefficients with O perturbation')
    #plt.plot(n_arr[:10], np.pi*n_arr[:10]/beta)
    plt.xlabel('Coefficient index')
    plt.ylabel('Coefficient value')
    plt.title('Krylov Coefficients for Random Matrix Hamiltonians')
    plt.legend()
    plt.show()

if(0):
    C_arr /= nmat
    # Compute fourier transform of C_arr
    freqs = np.fft.fftfreq(len(t_arr), d=(t_arr[1]-t_arr[0]))
    C_arr = np.fft.ifftshift(C_arr)
    C_arr_ft = np.fft.fft(C_arr)
    C_arr_ft = np.fft.fftshift(C_arr_ft)
    freqs = np.fft.fftshift(freqs)




    # Plot real part of C_arr
    plt.plot(freqs, np.log(np.real(C_arr_ft)))
    #plt.xlim([0,0.5])
    plt.ylim([0,500])
    plt.xlabel('Frequency')
    plt.ylabel('|C(ω)|')
    plt.title('Fourier Transform of TCF for Random Matrix Hamiltonians')
    plt.show()


    plt.plot(t_arr, C_arr.real, label=f'Matrix {i+1}')
    plt.plot(t_arr, C_arr[0]/np.cosh(np.pi*t_arr/beta)**2, label='1/cosh^2(πt/β)', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('TCF')
    plt.title('Time-Correlation Function for Random Matrix Hamiltonians')
    plt.legend()
    plt.show()

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
    
    #print('Unfolded eigenvalues computed', unfolded_eigenvalues[:10])
    #print('eigenvalues', eigenvalues[:10])

    return unfolded_eigenvalues
if(0):
    meanhist = np.mean(diffs)
    print('Mean energy difference:', meanhist)

    diffs/= meanhist
    print('Mean unfolded energy difference:', np.mean(diffs))
    # Fold the differences to get mean = 1
    
    print('diff len', len(diffs))
    plt.hist(diffs, bins=50, density=True)

#Plot Wigner-Dyson distribution for GOE
    s = np.linspace(0, 4, 100)
    P_s = (np.pi/2) * s * np.exp(- (np.pi/4) * s**2)
    plt.plot(s, P_s, label='Wigner-Dyson GOE', color='red')
    plt.legend()

    plt.xlabel('Energy difference')
    plt.ylabel('Density')
    plt.title('Density of energy differences from random matrix')
    plt.show()
    
    
