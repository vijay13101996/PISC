import numpy as np
from PISC.dvr.dvr import DVR1D
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity

#Compute Krylov complexity for energy eigenvalues coming from a random-matrix Hamiltonian

sigma = 1
neigs = 200

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

def f_O(vals,n,m):
    E = (vals[n]+vals[m])/2
    w = (vals[n]-vals[m])
    a = 0.1
    #print('rho', E, rho(E))
    return np.exp(-a*abs(w)**4)*np.random.normal(0,1)/rho(E)**0.5

    #return np.random.normal(0,1)#/rho(E)**0.5

nmat = 10
ncoeff = 100
t_arr = np.linspace(0, 100, 500)
C_arr = np.zeros_like(t_arr) + 0j
n_arr = np.arange(ncoeff)
b_arr = np.zeros(ncoeff)

beta = 1
diffs = []

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
            #O[n, m] = np.random.normal(0,1)
            #O[n,m] = np.exp(-a*abs(vals[n]-vals[m])**2)
            O[n,m] = f_O(vals,n,m)
            O[m,n] = O[n,m]

    C_arr += TCF_O(vals, beta, neigs, O, t_arr)

    #b_arr+= Krylov_O(vals, beta, neigs, O, ncoeff)

if(0):
    b_arr /= nmat
    plt.scatter(n_arr[1:], b_arr[1:].real, label='Krylov coefficients')
    plt.plot(n_arr[:10], np.pi*n_arr[:10]/beta)
    plt.xlabel('Coefficient index')
    plt.ylabel('Coefficient value')
    plt.title('Krylov Coefficients for Random Matrix Hamiltonians')
    plt.legend()
    plt.show()

if(1):
    C_arr /= nmat
    plt.plot(t_arr, C_arr.real, label=f'Matrix {i+1}')

    plt.xlabel('Time')
    plt.ylabel('TCF')
    plt.title('Time-Correlation Function for Random Matrix Hamiltonians')
    plt.legend()
    plt.show()

if(0):
    plt.hist(diffs, bins=100, density=True)
    plt.xlabel('Energy difference')
    plt.ylabel('Density')
    plt.title('Density of energy differences from random matrix')
    plt.show()
    
    
