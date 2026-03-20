import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import mildly_anharmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det
import argparse
import matplotlib
import math

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12
scat_size = 10

def compute_bn(O,vals,vecs,T_au,ncoeff):

    neigs = len(vals)
    beta = 1.0/T_au
    
    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    bnarr = np.zeros(ncoeff)
    bnarr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, bnarr, beta, vals,0.5, 'wgm')
    
    bnarr = np.array(bnarr)
    coeffarr = np.arange(ncoeff) 

    return coeffarr,bnarr

def compute_moments(O,vals,T_au,nmoments):
    neigs = len(vals)
    beta = 1.0/T_au
    
    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, 'asm', 0.5, moments)
    even_moments = moments[0::2]
    n_arr = np.arange(nmoments+1)
    return n_arr, even_moments

def comput_On_matrix(O,beta,vals,nmat,lamda=0.5,ip='wgm'): 
    neigs = len(vals)
    
    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    On = np.zeros((neigs,neigs))
    barr = np.zeros(nmat+1)

    barr, On = Krylov_complexity.krylov_complexity.compute_on_matrix(O, L, barr, beta, vals, lamda, ip, On, nmat+1) 
    return barr, On   

L=40
ngrid=1000

m=1.0
omega=1.0
T_au = 1.0


lb = -L
ub = L
pes = mildly_anharmonic(m,0,0,w=omega,n=2)

potkey = 'Harmonic_m_{}_w_{}'.format(m,omega)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 400
vals,vecs = DVR.Diagonalize(neig_total=neigs)

print('vals', vals[-1])

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

if(0): #Compare with analytical
    neig = 2

    gauss = (m*omega/np.pi)**0.25 * np.exp(-0.5*m*omega*x_arr**2)
    hermite = scipy.special.hermite(neig)
    norm = (2.0**neig * math.factorial(neig))**0.5
    vecs1_anal = gauss * hermite((m*omega)**0.5 * x_arr) / norm
    
    plt.plot(x_arr,vecs[:,neig])
    plt.plot(x_arr,vecs1_anal)
    plt.show()

# Compute position matrix
pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

# O2 operator
O2 = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        if abs(i-j) == 1:
            O2[i,j] = 1.0
        else:
            O2[i,j] = 0.0

# u operator
u = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        u[i,j] = 1.0

# O2^k operator
O2k = O2.copy()
k=200
for n in range(k-1):
    O2k = np.matmul(O2k,O2)

O2K_approx = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        if abs(i-j)<= k:
            O2K_approx[i,j] = 2**k/np.sqrt(np.pi*k/2) * np.exp(-(i-j)**2/(2*k))
            #print(i,j, abs(O2k[i,j]- O2K_approx[i,j]))
        else:
            O2K_approx[i,j] = 0.0

if(1): # Lanczos coefficients for O2^k - exact and approx
    
    fig, ax = plt.subplots(figsize=(5,5))

    for T_au in [1.0,2.0]:#,4.0]:
        coeffarr,bnarr = compute_bn(O2k,vals,vecs,T_au,50)
        coeffarr2,bnarr2 = compute_bn(O2K_approx,vals,vecs,T_au,50)

        ax.scatter(coeffarr[1:],bnarr[1:],label=r'$T={}$'.format(T_au), s=scat_size)
        ax.plot(coeffarr2[1:],bnarr2[1:],  ls='--')
    
    ax.set_xlabel(r'$n$', fontsize=xl_fs)
    ax.set_ylabel(r'$b_n$', fontsize=yl_fs)
    ax.set_title(r'$V(x) = \frac{1}{2}x^2, \;  \hat{O} = (\hat{u}^{(2)})^{200}$', fontsize=tp_fs)
    ax.legend(fontsize=le_fs)
    
    plt.savefig('draft_O2k_200_lanczos_coeffs.pdf', dpi=300, bbox_inches='tight')

    plt.show()

if(0): # Lanczos coefficients for u - exact and approx

    for T_au in [1.0,2.0,3.0,4.0,5.0]:
        coeffarr,bnarr = compute_bn(u,vals,vecs,T_au,80)
        plt.scatter(coeffarr[1:],bnarr[1:],  label=r'$T={}$'.format(T_au))
        plt.plot(coeffarr[1:20], np.pi*coeffarr[1:20]*T_au, 'k--')
    
    plt.xlabel(r'$n$', fontsize=xl_fs)
    plt.ylabel(r'$b_n$', fontsize=yl_fs)
    plt.legend(fontsize=le_fs)
    plt.show()

