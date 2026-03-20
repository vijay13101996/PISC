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

ngrid=2000
L=20
m=1.0
b=1.0 # V(x) = b*x^n

def plot_bn(L,n_anharm,T_au,ncoeff):

    lb = -L
    ub = L
    pes = mildly_anharmonic(m,0,b,w=0,n=n_anharm)

    #potkey = 'Quartic_m_{}_b_{}'.format(m,b)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    neigs = 300
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    print('vals', vals[-1], 'n_anharm', n_anharm)

    x_arr = DVR.grid[1:ngrid]
    dx = x_arr[1]-x_arr[0]

    # Compute position matrix
    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

    pos_arr = pos_mat[0,1::2]
    w_arr = vals[1::2] - vals[0]
    plt.plot(w_arr,np.log(abs(pos_arr)),label=r'$T={}$'.format(T_au))
    plt.show()
    exit()

    coeffarr,bnarr = compute_bn(pos_mat,vals,vecs,T_au,30)
    plt.scatter(coeffarr,bnarr,label=r'$T={}$'.format(T_au),s=15,marker='o',alpha=0.7)

L_arr = [20,8]
n_anh_arr = [4,8]
for L,n_anh in zip(L_arr,n_anh_arr):
    plot_bn(L,n_anh,1.0,30)

plt.xlabel(r'$n$',fontsize=xl_fs)
plt.ylabel(r'$b_n$',fontsize=yl_fs)
plt.xticks(fontsize=tp_fs)
plt.yticks(fontsize=tp_fs)
plt.show()


