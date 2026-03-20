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


neigs = 800
L = 40
omega = 1.0

neigs_arr = np.arange(1,neigs+1)

harm_vals = omega*neigs_arr - omega/2
vals = harm_vals

O = np.zeros((neigs,neigs))

k_diag = 800
for i in range(neigs):
    for j in range(i,neigs):
        #if(abs(i-j)%2==0): 
            if(abs(i-j)<=k_diag):
                O[i,j] =  1.0#np.random.uniform(0,10)
                #O[i,j] = 1.0 #+ 0.2*np.random.normal(0,1)
                O[j,i] = O[i,j]

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

T_arr = [5.]
mun_arr = []
mu0_harm_arr = []
mu_all_arr = []
bnarr = []

nmoments = 100
ncoeff = 50

On = np.zeros((neigs,neigs))
nmat = 10 

eBh = np.diag(np.exp(-0.5*vals/T_arr[0]))
Z = np.sum(np.exp(-vals/T_arr[0]))

ip = 'asm'

lamda = 0.0

x = 1.

fig, ax = plt.subplots(2,3)
fig.subplots_adjust(hspace=0.3, wspace=0.1)

def lanczos_coeffs(O, L, vals, T_au, lamda, ip, ncoeff, nmat, ax):
    bnarr = []
    
    On = np.zeros((neigs,neigs))

    print('T',T_au)
    Tkey = 'T_{}'.format(T_au)

    beta = 1.0/T_au 

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, lamda, ip)
    bnarr.append(barr)

    ax[0].scatter(np.arange(0,ncoeff),barr,label='T = {}'.format(T_au),s=3)
    ax[0].scatter(nmat,barr[nmat],color='red',s=10,zorder=2)


    print('n sat',vals[-1]/(2*np.pi*T_au))

    eBh1 = np.diag(np.exp(-lamda*vals/T_au))
    eBh2 = np.diag(np.exp(-(1-lamda)*vals/T_au))
 
    barr_mat = np.zeros(ncoeff)
    barr_mat, On = Krylov_complexity.krylov_complexity.compute_on_matrix(O, L, barr, beta, vals, lamda, ip, On, nmat+1) 

    if(ip=='asm'):
        On2 = np.matmul(eBh1,np.matmul(On.T,np.matmul(eBh2,On)))/Z
    elif(ip=='dir'):
        On2 = np.matmul(On.T,On)/Z
    
    bval = 0.0
    bval = Krylov_complexity.krylov_complexity.compute_ip(On,On,beta,vals,lamda,bval,ip)
   
    print('lamda, ip, bn, bval, btrace, alpha T',lamda, ip,barr[nmat],bval**0.5,np.trace(On2)**0.5, np.pi*T_au*nmat*0.5/lamda) 
    
    logOn = np.abs(np.log(On))

    ax[1].imshow((np.log(abs(On))),cmap='hot',aspect='auto')

lanczos_coeffs(O, L, vals, T_arr[0], 0.5, 'asm', ncoeff, 2, ax[:,0])
lanczos_coeffs(O, L, vals, T_arr[0], 0.5, 'asm', ncoeff, 14, ax[:,1])
lanczos_coeffs(O, L, vals, T_arr[0], 0.5, 'asm', ncoeff, 40, ax[:,2])

ax[0,0].set_ylabel(r'$b_n$',fontsize=yl_fs)
ax[1,0].set_ylabel(r'$j$',fontsize=yl_fs)
for i in range(3):
    ax[0,i].set_xlabel(r'$n$',fontsize=xl_fs)
    ax[0,i].tick_params(axis='both', which='major', labelsize=tp_fs)

    ax[1,i].set_xlabel(r'$i$',fontsize=xl_fs)
    ax[1,i].set_xticks([0,200,400,600])
    ax[1,i].tick_params(axis='both', which='major', labelsize=tp_fs)

for i in range(1,3):
    ax[0,i].set_yticks([])
    ax[1,i].set_yticks([])
    #ax[1,i].set_xticks([])

fig.set_size_inches(7.5,6)	
fig.savefig('/home/vgs23/Images/bn_vs_n_On.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)
plt.show()
exit()

