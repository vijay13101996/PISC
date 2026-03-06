import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from PISC.potentials import mildly_anharmonic
from scipy.special import gamma
import matplotlib
import pickle

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 11#9.5
ti_fs = 12
ngrid = 2000

L=20
lb=-L
ub=L

m=1.

print('L',L)

a=0.0
b=1.0
omega=0.0
n_anharm = 4

nmoments = 20
ncoeff = 50
T = 1.0
beta = 1.0/T

neigs = 200

def tunneling_factor(p):
    num = gamma( (p-2)/(2*p) )
    denom = gamma( (p-1)/p )
    pref = np.sqrt(2*np.pi)/(p+2)
    ret = pref*(num/denom)*energy_factor(p)
    return ret

def J(p):
    num = gamma(1/p)*gamma(1.5)
    denom = p*gamma(1/p+1.5)
    return num/denom

def energy_factor(p):
    num = np.pi/(2*np.sqrt(2*m))
    denom = J(p)
    pref = (num/denom)**(2*p/(p+2))
    return pref

def compute_pos_mat(anharm, L):
    pes = mildly_anharmonic(m,a,b,omega,anharm)

    potkey = 'MAH_w_{}_a_{}_b_{}_n_{}'.format(omega,a,b,anharm)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    x_arr = DVR.grid[1:ngrid]
    dx = x_arr[1]-x_arr[0]

    if(0):
        plt.plot(x_arr, vecs[:,-1], label='p={}'.format(anharm))
        plt.xlabel('x')
        plt.ylabel('wavefunction')
        plt.title('Wavefunction')
        plt.legend()
        plt.show()

    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

    return vals,pos_mat

def compute_Landau_O(anharm, neigs):
    O = np.zeros((neigs,neigs))

    tf = tunneling_factor(anharm)
    print('anharm',anharm,'tf', tf)

    for i in range(neigs):
        for j in range(i,neigs):
            if (i-j) % 2 == 0 and abs(i-j) > 0:
                O[i,j] = np.exp(-tf*abs(i-j))
                O[j,i] = O[i,j]

    return O

def K_complexity(O,vals,label,ax,fit=False):
    # Compute the Krylov complexity and moments
    # This function is a placeholder for the actual implementation
    
    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')

    #fit barr with a line
    coeffs = np.polyfit(np.arange(ncoeff)[1:], barr[1:], 1)
    poly = np.poly1d(coeffs)

    #print('alpha', coeffs[0], np.pi*T, np.pi*T - tf)

    if(fit):
        ax.plot(np.arange(ncoeff), poly(np.arange(ncoeff)), 'k--')#, label='fit')
    ax.scatter(np.arange(ncoeff)[1:], barr[1:], label=label,s=10)
    print('beta', beta, 'alpha from fit', coeffs[0])
    return np.arange(ncoeff), barr

#K_complexity(pos_mat,r'$V(x) = x^{{}}, \hat{O}=\hat{x}$, (DVR)'.format(n_anharm),fit=False)
#K_complexity(O,r'$V(x)=x^{{}},\hat{O} = \hat{x}$, (Landau)'.format(n_anharm),fit=True)

if(0):
    if(n_anharm==6):
        K_complexity(O,r'$V(x)=x^6,\hat{O} = \hat{x}$ (Landau)',fit=True)
        K_complexity(pos_mat,r'$V(x) = x^6, \hat{O}=\hat{x}$ (DVR)',fit=False)
    elif(n_anharm==4):
        K_complexity(O,r'$V(x)=x^4,\hat{O} = \hat{x}$ (Landau)',fit=True)
        K_complexity(pos_mat,r'$V(x) = x^4, \hat{O}=\hat{x}$ (DVR)',fit=False)

O = compute_Landau_O(4,neigs)
vals, pos_mat = compute_pos_mat(4,20)

fig, ax = plt.subplots(1,1)

def plot_ax_2b(ax):
    narr, barr_L = K_complexity(O,vals,r'$V(x)=x^4,\hat{O} = \hat{x}$ (Landau)',ax,fit=True)
    narr, barr_D = K_complexity(pos_mat,vals,r'$V(x) = x^4, \hat{O}=\hat{x}$ (DVR)',ax,fit=False)

    ax.plot(np.arange(ncoeff), np.pi*T*np.arange(ncoeff), 'r--', label=r'$\alpha = \pi k_B T$')
    ax.set_xlabel(r'$n$', fontsize=xl_fs)
    #ax.set_ylabel(r'$b_n$', fontsize=yl_fs)
    #ax.annotate(r'$V(x)=x^4,\; \hat{O}=\hat{x}$',xy=(0.25,0.9), xycoords='axes fraction', fontsize=le_fs)

    np.savetxt('FIG_2b_Landau_x_bn_T{}.txt'.format(T), np.c_[narr, barr_L], header='n   bn')
    np.savetxt('FIG_2b_DVR_x_bn_T{}.txt'.format(T), np.c_[narr, barr_D], header='n   bn')

#plot_ax_2b(ax)

if(0):

    fig, ax = plt.subplots(1,2,figsize=(8,4))
    fig.subplots_adjust(wspace=0.3)


    O = compute_Landau_O(4,neigs)
    vals, pos_mat = compute_pos_mat(4,20)

    K_complexity(O,vals,r'$V(x)=x^4,\hat{O} = \hat{x}$ (Landau)',ax[0],fit=True)
    K_complexity(pos_mat,vals,r'$V(x) = x^4, \hat{O}=\hat{x}$ (DVR)',ax[0],fit=False)
    ax[0].plot(np.arange(ncoeff), np.pi*T*np.arange(ncoeff), 'r--', label=r'$\alpha = \pi k_B T$')
    ax[0].set_xlabel(r'$n$', fontsize=xl_fs)
    ax[0].set_ylabel(r'$b_n$', fontsize=yl_fs)
    ax[0].annotate(r'$V(x)=x^4,\; \hat{O}=\hat{x}$',xy=(0.25,0.9), xycoords='axes fraction', fontsize=le_fs)


    O = compute_Landau_O(6,neigs)
    vals, pos_mat = compute_pos_mat(6,20)

    K_complexity(O,vals,r'$V(x)=x^6,\hat{O} = \hat{x}$ (Landau)',ax[1],fit=True)
    K_complexity(pos_mat,vals,r'$V(x) = x^6, \hat{O}=\hat{x}$ (DVR)',ax[1],fit=False)
    ax[1].plot(np.arange(ncoeff), np.pi*T*np.arange(ncoeff), 'r--', label=r'$\alpha = \pi k_B T$')
    ax[1].set_xlabel(r'$n$', fontsize=xl_fs)
    ax[1].set_ylabel(r'$b_n$', fontsize=yl_fs)
    ax[1].annotate(r'$V(x)=x^6,\; \hat{O}=\hat{x}$', xy=(0.25,0.9), xycoords='axes fraction', fontsize=le_fs)

    #fig.savefig('draft_Krylov_anharm.pdf', dpi=300, bbox_inches='tight')
    plt.show()

