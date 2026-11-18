import numpy as np
from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, compute_U_F, compute_U_F_direct
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity

m = 1.0
L = 10.0
n_anharm = 4
omega = 0.0
a = 0.0
b = 1.0
ngrid = 2**9 

lb = -L
ub = L
pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

potkey = 'L_{}_MAH_w_{}_a_{}_b_{}_n_{}'.format(L,omega,a,b,n_anharm)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = ngrid-1
vals,vecs = DVR.Diagonalize(neig_total=neigs)
print('vals',vals[0])

x_arr = DVR.grid[:ngrid]
dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

X = pos_mat
w_F = 1.0
g_opt = 2.497 #1.579
T = 2*np.pi/w_F
n_ph = 3

def V(x,t):
    return pot(x) + x*g_opt*np.sin(w_F*t)

#for g in [2.495, 2.497, 2.499, 2.5]
for g,c in zip([2.495, 2.497, 2.499, 2.5], ['blue', 'green', 'orange', 'red']):
    V = g*X/2j
    print('Driving strength g:', g)
    floquet_vals, floquet_vecs = Floquetiser(vals, w_F, V, n_ph)
    print('Floquet eigenvalues:', floquet_vals[0:10])
    floquet_vals = np.real(floquet_vals)
    floquet_folded = (floquet_vals + w_F/2) % w_F - w_F/2
    floquet_folded = np.sort(floquet_folded)
    print('Floquet eigenvalues folded:', floquet_folded[0:100])
    #plt.plot(g*np.ones_like(floquet_vals), np.real(floquet_vals), 'o', color='blue', markersize=2)
    plt.plot(np.arange(len(floquet_folded)), floquet_folded, 'o', color=c, markersize=2, label='g={}'.format(g))

E0 = vals[0]
E1 = vals[1]
E2 = vals[2]
E3 = vals[3]
for n in range(-n_ph,n_ph+1):
    plt.axhline(E0+n*w_F, color='red', linestyle='--', alpha=0.5)
    plt.axhline(E1+n*w_F, color='green', linestyle='--', alpha=0.5)
    plt.axhline(E2+n*w_F, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(E3+n*w_F, color='orange', linestyle='--', alpha=0.5)

#plt.axhline(E0, color='red', linestyle='--', label='E0')
#plt.axhline(E1, color='green', linestyle='--', label='E1')
#plt.axvline(0.0005714, color='black', linestyle='--', label='S_c')

plt.legend()
#plt.ylim(E0-0.001, E1+0.001)
plt.xlabel('Driving strength S')
plt.ylabel('Floquet eigenvalues')
plt.title('Floquet eigenvalues vs Driving strength S for double-well potential')
plt.grid()
plt.show()


