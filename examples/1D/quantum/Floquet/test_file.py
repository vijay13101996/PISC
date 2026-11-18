import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity

"""
This module will be used to construct the Floquet Hamiltonian for a simple model system 
and benchmark it with known results. The main objective is to diagonalise the Floquet
Hamiltonian given the time-dependent Hamiltonian and a single-frequency driving term. 
The most important question is to figure how many photon blocks can be used.

We consider the Hamiltonian:
    H(t) = H_0 + V cos(wt)
where H_0 is the time-independent part of the Hamiltonian and V is the driving term.

For simplicity, V is taken to be a constant and H0 is the Harmonic oscillator Hamiltonian. 
"""

m = 1.0
L = 20.0
n_anharm = 4
omega = 1.0
a = 0.0
b = 0.0
ngrid = 200

lb = -L
ub = L
pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

potkey = 'L_{}_MAH_w_{}_a_{}_b_{}_n_{}'.format(L,omega,a,b,n_anharm)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 10
vals,vecs = DVR.Diagonalize(neig_total=neigs)
print('vals',vals)
print('norm', np.linalg.norm(vecs[:,0]), np.linalg.norm(vecs[:,9]))
exit()

x_arr = DVR.grid[:ngrid]
dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

O = pos_mat

w_F = 2.0 #2.0 #0.5
n_ph = 3
g = 1.0
V = np.eye(len(vals))*g

def Floquetiser(vals, w_F, V, n_ph):
    n_F = len(vals)*(2*n_ph+1)
    H_F = np.zeros((n_F,n_F),dtype=complex)

    # Diagonal blocks with photon numbers
    for n in range(-n_ph,n_ph+1):
        idx = (n+n_ph)*len(vals)
        H_F[idx:idx+len(vals),idx:idx+len(vals)] = np.diag(vals + n*w_F)

    # Off-diagonal blocks with driving term
    H_1  = V
    #H_m1 is the conjugate transpose of H_1
    H_m1 = V.conj().T

    for n in range(-n_ph,n_ph):
        idx = (n+n_ph)*len(vals)
        H_F[idx:idx+len(vals),idx+len(vals):idx+2*len(vals)] = H_1
        H_F[idx+len(vals):idx+2*len(vals),idx:idx+len(vals)] = H_m1
    
    print('H_F \n', np.real(H_F))
    
    # Diagonalize the Floquet Hamiltonian
    #Test if H_F is Hermitian
    if np.allclose(H_F, H_F.conj().T):
        print("H_F is Hermitian")
    floquet_vals, floquet_vecs = np.linalg.eig(H_F)

    floquet_vals = np.sort_complex(floquet_vals)

    return floquet_vals, floquet_vecs

if(1):
    w_arr = np.arange(-2, 2.01, 0.1)
    for w_F in w_arr:
        floquet_vals, floquet_vecs = Floquetiser(vals, w_F, V, n_ph)
        print("Floquet eigenvalues for w_F = {}: {}".format(w_F, floquet_vals))
        plt.plot(w_F*np.ones(len(floquet_vals)), floquet_vals, 'o', label='w_F={}'.format(w_F))
        #plt.plot(np.arange(len(floquet_vals)), floquet_vals, 'o', label='w_F={}'.format(w_F))
        #plt.show()
    plt.xlabel('Driving frequency w_F')
    plt.ylabel('Floquet eigenvalues')
    plt.title('Floquet eigenvalues vs driving frequency')
    plt.grid()
    plt.show()

if(1):
    g_arr = np.arange(-1.0, 1.01, 0.1)
    for g in g_arr:
        V = np.eye(len(vals))*g/2
        floquet_vals, floquet_vecs = Floquetiser(vals, w_F, V, n_ph)
        print('g', g, 'floquet_vals', floquet_vals[:5])  # Print first 5 eigenvalues for brevity
        plt.plot(g*np.ones(len(floquet_vals)), floquet_vals, 'o', label='g={}'.format(g))
        #plt.plot(np.arange(len(floquet_vals)), floquet_vals, 'o', label='g={}'.format(g))
        #plt.show()

    plt.xlabel('Driving strength g')
    plt.ylabel('Floquet eigenvalues')
    plt.title('Floquet eigenvalues vs driving strength')
    plt.grid()
    plt.show()


