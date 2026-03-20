import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, mildly_anharmonic, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det
import argparse
import matplotlib as mpl
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12

neigs = 100

O = np.zeros((neigs,neigs))

for i in range(neigs):
    for j in range(i,neigs):
        O[i,j] =  1e-13
        O[j,i] = O[i,j]


T_arr = [1,2,3,4,5]
mun_arr = []
mu0_harm_arr = []
mu_all_arr = []
bnarr = []

nmoments = 100
ncoeff = 200

On = np.zeros((neigs,neigs))
nmat = 10 

ip = 'asm'

lamda = 0.0

x = 1.

def lanczos_coeffs(O, L, vals, T_arr, lamda, ip, ncoeff, ax, label=False):
    bnarr = []
    mun_arr = []
    mu0_harm_arr = []
    mu_all_arr = []
    bnarr = []
    
    On = np.zeros((neigs,neigs))

    for T_au in T_arr:
        print('T',T_au)
        Tkey = 'T_{}'.format(T_au)

        beta = 1.0/T_au 

        moments = np.zeros(nmoments+1)
        moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, ip, lamda, moments)
        even_moments = moments[0::2]

        barr = np.zeros(ncoeff)
        barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, lamda, ip)
        bnarr.append(barr)

    mun_arr.append(even_moments)
    mu_all_arr.append(moments)

    mun_arr = np.array(mun_arr)
    bnarr = np.array(bnarr)

    print('bnarr',bnarr.shape,bnarr[0,:12])

    for i in range(len(T_arr)):
        if label:
            ax.scatter(np.arange(1,ncoeff+1),bnarr[i,:],label=r'$T={}$'.format(T_arr[i]),s=10)
        else:
            ax.scatter(np.arange(1,ncoeff+1),bnarr[i,:],s=10)

L = 10
omega = 1.0

neigs_arr = np.arange(1,neigs+1)

harm_vals = 2*neigs_arr # omega*neigs_arr - omega/2
maxval = harm_vals[-1]

#Set max val or pow05_vals to be the same as the max val of harm_vals
c1 = maxval/neigs_arr[-1]**0.5
c2 = maxval/neigs_arr[-1]**1.5
c3 = maxval/neigs_arr[-1]**2

pow05_vals = c1*neigs_arr**0.5
pow15_vals = c2*neigs_arr**1.5
box_vals = c3*neigs_arr**2#np.pi**2*neigs_arr**2/(2*L**2)

print('box_vals',box_vals[-1],maxval)
print('pow15_vals',pow15_vals[-1],maxval)
print('pow05_vals',pow05_vals[-1],maxval)

fig, ax = plt.subplots(2,2,sharex=True,sharey=True)
fig.subplots_adjust(hspace=0.0,wspace=0.0)
ax00 = ax[0,0]
ax01 = ax[0,1]
ax10 = ax[1,0]
ax11 = ax[1,1]

vals = box_vals

for ax, vals in zip([ax00,ax01,ax10,ax11],[pow05_vals,harm_vals,pow15_vals,box_vals]):
    print('vals',vals[-1])

    ax.set_xlim([-1,35])
    ax.set_ylim([-1,vals[-1]*0.63])
    

    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    #label only the first plot
    if ax == ax00:
        lanczos_coeffs(O, L, vals, T_arr, 0.5, 'asm', ncoeff,ax,label=True)
    else:
        lanczos_coeffs(O, L, vals, T_arr, 0.5, 'asm', ncoeff,ax)

    #X-axis label only for bottom plots
    if ax == ax10 or ax == ax11:
        ax.set_xlabel(r'$n$',fontsize=xl_fs)
        ax.tick_params(axis='x',labelsize=tp_fs)

    #Y-axis label only for left plots
    if ax == ax00 or ax == ax10:
        ax.set_ylabel(r'$b_n$',fontsize=yl_fs)
        ax.tick_params(axis='y',labelsize=tp_fs)

    #Subplot labels
    if ax == ax00:
        ax.annotate(r'$(a)$', xy=(0.02, 0.9), xytext=(0.02, 0.9), textcoords='axes fraction', fontsize=xl_fs)
        ax.annotate(r'$E_k \sim k^{1/2}$', xy=(0.4, 0.9), xytext=(0.4, 0.9), textcoords='axes fraction', fontsize=xl_fs)
    if ax == ax01:
        ax.annotate(r'$(b)$', xy=(0.02, 0.9), xytext=(0.02, 0.9), textcoords='axes fraction', fontsize=xl_fs)
        ax.annotate(r'$E_k \sim k$', xy=(0.4, 0.9), xytext=(0.4, 0.9), textcoords='axes fraction', fontsize=xl_fs)
    if ax == ax10:
        ax.annotate(r'$(c)$', xy=(0.02, 0.9), xytext=(0.02, 0.9), textcoords='axes fraction', fontsize=xl_fs)
        ax.annotate(r'$E_k \sim k^{3/2}$', xy=(0.4, 0.9), xytext=(0.4, 0.9), textcoords='axes fraction', fontsize=xl_fs)
    if ax == ax11:
        ax.annotate(r'$(d)$', xy=(0.02, 0.9), xytext=(0.02, 0.9), textcoords='axes fraction', fontsize=xl_fs)
        ax.annotate(r'$E_k \sim k^2$', xy=(0.4, 0.9), xytext=(0.4, 0.9), textcoords='axes fraction', fontsize=xl_fs)

fig.set_size_inches(7,7)	
#fig.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),fontsize=le_fs-2,ncol=5)
fig.legend(fontsize=le_fs-2.,loc=(0.11,0.92),ncol=5)
fig.savefig('/home/vgs23/Images/bn_vs_n_pow.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)

plt.show()
exit()

