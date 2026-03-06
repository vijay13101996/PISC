import numpy as np
from PISC.dvr.dvr import DVR1D
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity
import scipy

m = 1.0
omega = 1.0
L = 10.0
ngrid = 1000
lb = -L
ub = L

Len = 1.0/2
neigs = 300
n = 2

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

def potential(x, Len, n):
    if n==infty:
        if np.abs(x)<=Len:
            return 0.0
        else:
            return 1e6
    else:
        return Len*x**n

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return 0.0
    elif(i-j==1):
        return np.sqrt(1/(2*m*omega))*np.sqrt(j+1)
    elif(j-i==1):
        return np.sqrt(1/(2*m*omega))*np.sqrt(i+1)
    else:
        return 0.0

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)

vals_anal = np.zeros(neigs)
for i in range(neigs):
    vals_anal[i] = omega*(i+0.5)

fig, ax = plt.subplots(3,1,figsize=(4,12))

Len = 2.
infty = float('inf')
for n in [2,4,6,infty]:
    pot = lambda x: potential(x, Len, n)
    pot = np.vectorize(pot)

    DVR = DVR1D(ngrid, lb, ub,m, pot)
    vals_num, vecs_num = DVR.Diagonalize(neig_total=neigs)
    
    O_num = np.zeros((neigs,neigs))
    x_arr = DVR.grid[1:ngrid]
    dx = x_arr[1]-x_arr[0]
    
    O_num = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs_num, x_arr, dx, dx, O_num)
    O = O_num
    vals = vals_num

    tarr = np.linspace(0,200,10000)
    beta = 0.5
    C_tcf = TCF_O(vals, beta, neigs, O_num, tarr)
    b_arr = Krylov_O(vals, beta, neigs, O_num, 50)

    freqs = np.fft.fftfreq(len(tarr), d=(tarr[1]-tarr[0]))*2*np.pi
    C_tcf_ft = np.fft.fft(C_tcf)
    #Plot real part of C_tcf
    C_tcf_ft = np.fft.fftshift(C_tcf_ft)
    freqs = np.fft.fftshift(freqs)
    
    ax[0].plot(tarr, C_tcf.real, label='Re C(t) TCF n={}'.format(n))
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('C(t)')
    ax[0].legend()

    ax[1].plot(freqs, (np.abs(C_tcf_ft)), label='TCF FT n={}'.format(n)) 
    ax[1].set_xlabel('Frequency ω')
    ax[1].set_ylabel('Log TCF FT log|C(ω)|')
    ax[1].legend()

    ax[2].plot(b_arr, label='Krylov b_n n={}'.format(n), marker='o', markersize=4)
    ax[2].set_xlabel('n')
    ax[2].set_ylabel('b_n')
    ax[2].legend()

plt.tight_layout()
plt.show()












