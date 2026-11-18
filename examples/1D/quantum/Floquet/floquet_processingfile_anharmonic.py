from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct, compute_ipr
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
import numpy as np
import matplotlib
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.animation import FuncAnimation


if(1): # Anharmonic oscillator
    m = 1.0
    L = 10.0
    n_anharm = 4
    omega = 0.0
    a = 0.0
    b = 1.0
    ngrid = 2**10 

    lb = -L
    ub = L
    pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

    potkey = 'L_{}_anharmonic_{}'.format(L,n_anharm)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    neigs = 200 #ngrid-1
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    diff0 = vals[1]-vals[0]
    diff1 = vals[2]-vals[1]

    print('vals',vals[0])

    print('diff',vals[1]-vals[0],vals[2]-vals[1],vals[3]-vals[2])

    x_arr = DVR.grid[:ngrid]
    dx = x_arr[1]-x_arr[0]

    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

    if(0):
        warr = vals - vals[0]
        plt.plot(warr, np.log(abs(pos_mat[0,:])), label='pos_mat')
        plt.title('Anharmonic oscillator, L={}, n_anharm={}'.format(L,n_anharm))
        plt.xlabel('w')
        plt.ylabel('Matrix elements')
        plt.grid()
        plt.legend()
        plt.show()
        exit()

    if(0):
        plt.plot(np.log(abs(pos_mat[0,:])), label='pos_mat')
        plt.title('Anharmonic oscillator, L={}, n_anharm={}'.format(L,n_anharm))
        plt.xlabel('n')
        plt.ylabel('Matrix elements')
        plt.grid()
        plt.legend()
        plt.show()

g = 3.0
w_F = 10.0
w_F_arr = np.arange(0.5, 30.01, 0.25)
w_F_arr = np.around(w_F_arr, decimals=2)

key = 'pos_mat'
#key = 'dir_pl_decay_inv'

diff_1 = [vals[i+1]-vals[i] for i in range(len(vals)-1)]

for w_F in w_F_arr:
    print('key', key, 'w_F', w_F)
    U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)
    #print('U_F shape', U_F.shape, 'qE shape', qE.shape, 'qS shape', qS.shape)

    ipr_max,E_ind,max_ind = compute_ipr(qS, metric='max')
    plt.scatter(w_F, ipr_max, color='r', s=15, zorder=2)

#for val in diff_1:
#    plt.axvline(val, color='k', linestyle='--', alpha=0.5, zorder=1)

plt.title('L={}, n_anharm={}, op_key={}'.format(L,n_anharm,key))
plt.xlabel('w_F')
plt.ylabel('IPR')
plt.grid()
plt.legend()
plt.show()
