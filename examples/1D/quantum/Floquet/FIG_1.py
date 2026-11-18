import numpy as np
import matplotlib.pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct, compute_ipr
from PISC.dvr.dvr import DVR1D
import matplotlib
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16
yl_fs = 16
tp_fs = 12

m = 1.0
L = 2.0
ngrid = 2**9 
neigs = ngrid-1

lb = 0
ub = L
grid = np.linspace(lb, ub, ngrid+1)

def pot(x):
    if(x>L or x<0):
        return 1e10
    else:
        return 0.0

potkey = 'L_{}_1D_box_0_MANU'.format(L)

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
vecs_anal = np.zeros((neigs,ngrid-1))
for i in range(neigs):
    vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*grid[1:ngrid]/L)

vals = vals_anal
vecs = vecs_anal

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)

pos_mat = O_anal - L/2*np.diag(np.ones(neigs))


g= 3.0
w_F_arr = np.arange(0.5, 30.01, 0.05)
w_F_arr = np.around(w_F_arr, decimals=2)
key = 'pos_mat'

c = 'r'
ipr_max_arr = []

for w_F in w_F_arr:
    U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)

    ipr_max,E_ind,max_ind = compute_ipr(qS, metric='max')
    ipr_max_arr.append(ipr_max)

#Find all maxima in the ipr_max_arr
ipr_max_arr = np.array(ipr_max_arr) 
maxima_indices = (np.diff(np.sign(np.diff(ipr_max_arr))) < 0).nonzero()[0] + 1
#Keep only the maxima after w_F =3.0
maxima_indices = maxima_indices[w_F_arr[maxima_indices]>3.0]

#Max_IPR > 2.5 corresponds to two_photon resonance, Max_IPR < 2.5 corresponds to one-photon resonance
tp_maxima_indices = maxima_indices[ipr_max_arr[maxima_indices]>2.5]
op_maxima_indices = maxima_indices[ipr_max_arr[maxima_indices]<2.5]

fig, ax = plt.subplots(figsize=(4,2.5))

ax.scatter(w_F_arr[tp_maxima_indices], ipr_max_arr[tp_maxima_indices], color='m',s=15, zorder=2)
ax.scatter(w_F_arr[op_maxima_indices], ipr_max_arr[op_maxima_indices], color='g',s=15, zorder=2)
ax.scatter(w_F_arr, ipr_max_arr, color=c, s=6, zorder=1)

ax.set_xlabel(r'$\omega_F$', fontsize=xl_fs)
ax.set_ylabel(r'$\text{PR}_\text{max}$', fontsize=yl_fs)
ax.set_ylim(0.75,3.5)

plt.savefig('Floquet_FIG_1.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.show()
