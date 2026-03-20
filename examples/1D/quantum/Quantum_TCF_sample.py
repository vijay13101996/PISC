import numpy as np
from PISC.dvr.dvr import DVR1D
#from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials import double_well, morse, quartic, harmonic1D, mildly_anharmonic
from PISC.potentials.triple_well_potential import triple_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D_omp_updated
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from PISC.engine import Krylov_complexity

ngrid = 600

L = 12
lb = -L
ub = L

m=1.0
a=0.0
b=.0
omega=1.0
n_anharm=4


if(1):
    def potential(x):
        if(x<-1 or x>1):
            return 1e12
        else:
            #print('x',x)
            return 0.0#x**4

neigs = 200

T_au = 3.
potkey = 'quartic_a_{}'.format(a)
Tkey = 'T_{}'.format(T_au)	

beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

t_arr = np.linspace(0,20.0,2000)

basis_N = 50
n_eigen = 30

ncoeff = 40

narr = np.arange(ncoeff) 
def TCF(DVR, m, beta, basis_N, n_eigen, corrkey):
    vals,vecs = DVR.Diagonalize(neig_total=neigs) 
    x_arr = DVR.grid[1:DVR.ngrid]
    dx = DVR.dx
    
    k_arr = np.arange(basis_N) +1
    m_arr = np.arange(basis_N) +1

    C_arr = np.zeros_like(t_arr) + 0j
    C_arr = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs, m, x_arr, dx, dx, k_arr, vals, m_arr, t_arr, beta,
                                                                n_eigen, corrkey, 0.5, C_arr)
    return C_arr

def TCF_O(DVR, beta, n_eigen, O):
    vals,vecs = DVR.Diagonalize(neig_total=neigs)
    #Given eigenvectors, eigenvalues compute the symmetric TCF for a given operator O
    
    n_arr = np.arange(n_eigen)
    m_arr = np.arange(n_eigen)

    C_arr = np.zeros_like(t_arr) + 0j

    for n in n_arr:
        for m in m_arr:
            C_arr += np.exp(-beta*(vals[n]+vals[m])/2) * np.exp(1j*(vals[n]-vals[m])*t_arr) * np.abs(O[n,m])**2

    Z = np.sum(np.exp(-beta*vals))
    C_arr /= Z

    return C_arr

def TCF_n(DVR, O, n_eigen, n):
    vals,vecs = DVR.Diagonalize(neig_total=neigs) 
    #Given eigenvectors, eigenvalues compute the symmetric TCF for a given operator O
    
    m_arr = np.arange(n_eigen)

    C_arr = np.zeros_like(t_arr) + 0j

    for m in m_arr:
        C_arr += np.exp(-1j*(vals[n]-vals[m])*t_arr) * np.abs(O[n,m])**2

    return C_arr

def ip(O, beta, n_eigen, vals):
    n_arr = np.arange(n_eigen)
    m_arr = np.arange(n_eigen)

    ip = 0.0
    for n in n_arr:
        for m in m_arr:
            ip += np.exp(-beta*(vals[n]+vals[m])/2) * np.abs(O[n,m])**2

    Z = np.sum(np.exp(-beta*vals))
    ip /= Z

    return ip


def Krylov(DVR, beta):
    vals,vecs = DVR.Diagonalize(neig_total=neigs) 
    x_arr = DVR.grid[1:DVR.ngrid]
    dx = DVR.dx
    
    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
    O = pos_mat

    L = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,L)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff) 
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    return barr

lb= -L
ub= L
pes = mildly_anharmonic(m,a,b,omega,n=n_anharm)
DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
vals,vecs = DVR.Diagonalize(neig_total=neigs) 

x_arr = DVR.grid[1:DVR.ngrid]
dx = DVR.dx
pos_mat = np.zeros((neigs,neigs))
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
#O = pos_mat

O = np.zeros((neigs,neigs))
for n in range(neigs):
    for m in range(neigs):
        #if (abs(n-m) == 1 ):
            #O[n,m] = 1.0
            O[n,m] = np.exp(-0.1*abs(vals[n]-vals[m])**0.9)

Ok = O
for n in range(11):
    Ok = np.matmul(Ok, O)

#print('O shape:', O[:20,:20])

TCF_fort = TCF(DVR, m, beta, basis_N, n_eigen, 'qq1')
TCF_py = TCF_O(DVR, beta, n_eigen, Ok)

TCF_n0 = TCF_n(DVR, O, n_eigen, 0)

plt.plot(t_arr, TCF_n0.real, label='n=0', lw=3)
#plt.plot(t_arr, TCF_n0.imag, label='n=0 imag', lw=3, ls='--')
plt.show()
exit()

fig, ax = plt.subplots(1,1)
#ax.plot(t_arr, (TCF_py.real), label='Python', lw=3)
#ax.plot(t_arr, (TCF_fort.real), label='Fort')
#plt.show()
#exit()

Ok = O
for k in range(51):
    Ok = np.matmul(Ok, O)
    ip_k = ip(Ok, beta, neigs, vals)
    Otemp = Ok/np.sqrt(ip_k)

    if(k % 10 == 0):
        print('k:', k, 'ip:', ip_k)
        TCF_k = TCF_O(DVR, beta, neigs, Otemp)
        ax.plot(t_arr, (TCF_k.real), label='k={}'.format(k+1))


u = np.zeros((neigs,neigs))
for n in range(neigs):
    for m in range(neigs):
            if(abs(n-m)%2==0):
                u[n,m] = 1.0
ip_k = ip(u, beta, neigs, vals)
u = u/np.sqrt(ip_k)

TCF_u = TCF_O(DVR, beta, neigs, u)
ax.plot(t_arr, (TCF_u.real), label='u', lw=3)

ax.legend()
plt.show()




exit()
