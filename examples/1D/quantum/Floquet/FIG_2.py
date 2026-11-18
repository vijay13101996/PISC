from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct, compute_ipr
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.animation import FuncAnimation

xl_fs = 16
yl_fs = 16
tp_fs = 12

m = 1.0
L = 2.0
ngrid = 2**9 
neigs = ngrid-1

lb = 0
ub = L

def pot(x):
    if(x>L or x<0):
        return 1e10
    else:
        return 0.0

potkey = 'L_{}_1D_box_0'.format(L)
DVR = DVR1D(ngrid, lb, ub,m, pot)
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
vecs_anal = np.zeros((neigs,ngrid-1))
for i in range(neigs):
    vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:ngrid]/L)

vals = vals_anal
vecs = vecs_anal

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)

pos_mat = O_anal - L/2*np.diag(np.ones(neigs))


fig, axes= plt.subplots(2,1,figsize=(4,8))
fig.subplots_adjust(hspace=0.33)

#--------------------------------------------------------------------------------------
ax = axes[1]
N = 200
key = 'pos_mat'
n_st = 4

g=3.0
w_F_2p = np.around((vals[6]-vals[4])/2, decimals=3)
w_F_1p = np.around((vals[1]-vals[0]), decimals=3)
w_F =w_F_2p
U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)

psi_init = np.zeros(neigs)
psi_init[n_st] = 1.0

Pnm1 = np.zeros(N)
Pn = np.zeros(N)
Pn1 = np.zeros(N)
Pn2 = np.zeros(N)
Pn3 = np.zeros(N)
Ptot = np.zeros(N)
psi_n = psi_init.copy()

for n in range(N):
    prob_n = np.abs(psi_n)**2
    Pn[n] = prob_n[n_st]
    Pn1[n] = prob_n[n_st+1]
    Pn2[n] = prob_n[n_st+2]
    Pn3[n] = prob_n[n_st+3]
    Pnm1[n] = prob_n[n_st-1]
    Ptot[n] = Pn[n] + Pn1[n] + Pn2[n]
    psi_n = U_F.dot(psi_n)

H_F = np.diag(vals) + g*pos_mat
H_F_3L = np.zeros((3,3),dtype=complex)
U_F_3L = np.eye(3,dtype=complex)
#Integrate over one period to get U_F_3L
dt = 1e-6
t_arr = np.arange(0, 2*np.pi/w_F, dt)
for t in t_arr:
    H_t = np.diag(vals[n_st:n_st+3]) + g*np.sin(w_F*t)*pos_mat[n_st:n_st+3,n_st:n_st+3]
    U_F_3L = scipy.linalg.expm(-1j*H_t*dt).dot(U_F_3L)


Pn_3L = np.zeros(N)
Pn1_3L = np.zeros(N)
Pn2_3L = np.zeros(N)
psi_n_3L = np.zeros(3)
psi_n_3L[0] = 1.0
for n in range(N):
    prob_n_3L = np.abs(psi_n_3L)**2
    Pn_3L[n] = prob_n_3L[0]
    Pn1_3L[n] = prob_n_3L[1]
    Pn2_3L[n] = prob_n_3L[2]
    psi_n_3L = U_F_3L.dot(psi_n_3L)

ax.plot(np.arange(N), Pn, label=r'$k=n$',c='blue')
ax.plot(np.arange(N), Pn1, label=r'$k=n+1$',c='orange')
ax.plot(np.arange(N), Pn2, label=r'$k=n+2$',c='green')
ax.plot(np.arange(N), Ptot,c='red')

ax.plot(np.arange(N), Pn_3L, '--', c='blue')
ax.plot(np.arange(N), Pn1_3L, '--', c='orange')
ax.plot(np.arange(N), Pn2_3L, '--', c='green')

ax.set_xlabel(r'$N$', fontsize=xl_fs-1.5)
ax.set_ylabel(r'$|\psi_N(k)|^2 = |\langle{{k}}|\hat{{U}}_F^N|n \rangle|^2$', fontsize=yl_fs-1.5)
ax.set_xlim(-1,N)
ax.set_ylim(-0.05,1.05)
#Set legend outside the plot on top
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
ax.annotate(r'(b)', xy=(-0.19, 1.), xycoords='axes fraction', fontsize=tp_fs+0.5)

#--------------------------------------------------------------------------------------------------------
ax = axes[0]
#One-photon case
n_st = 0
nph=1
w_F = w_F_1p
N= 50
U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)

psi_init = np.zeros(neigs)
psi_init[n_st] = 1.0

Pnm1 = np.zeros(N)
Pn = np.zeros(N)
Pn1 = np.zeros(N)
Ptot = np.zeros(N)
psi_n = psi_init.copy() 

for n in range(N):
    prob_n = np.abs(psi_n)**2
    Pn[n] = prob_n[n_st]
    Pn1[n] = prob_n[n_st+1]
    Pnm1[n] = prob_n[n_st-1]
    Ptot[n] = Pn[n] + Pn1[n]
    psi_n = U_F.dot(psi_n)

H_F = np.diag(vals) + g*pos_mat
H_F_2L = np.zeros((2,2),dtype=complex)
U_F_2L = np.eye(2,dtype=complex)
#Integrate over one period to get U_F_2L
dt = 1e-6
t_arr = np.arange(0, 2*np.pi/w_F, dt)
for t in t_arr:
    H_t = np.diag(vals[n_st:n_st+2]) + g*np.sin(w_F*t)*pos_mat[n_st:n_st+2,n_st:n_st+2]
    U_F_2L = scipy.linalg.expm(-1j*H_t*dt).dot(U_F_2L)  

psi_n_2L = np.zeros(2)
psi_n_2L[0] = 1.0
Pn_2L = np.zeros(N)
Pn1_2L = np.zeros(N)

for n in range(N):
    prob_n_2L = np.abs(psi_n_2L)**2
    Pn_2L[n] = prob_n_2L[0]
    Pn1_2L[n] = prob_n_2L[1]
    psi_n_2L = U_F_2L.dot(psi_n_2L) 

ax.plot(np.arange(N), Pn, label=r'$k=n$',c='blue')
ax.plot(np.arange(N), Pn1, label=r'$k=n+1$',c='orange')
ax.plot(np.arange(N), Ptot,c='red')

ax.plot(np.arange(N), Pn_2L, '--', c='blue')
ax.plot(np.arange(N), Pn1_2L, '--', c='orange')    
ax.set_xlabel(r'$N$', fontsize=xl_fs-1.5)
ax.set_ylabel(r'$|\psi_N(k)|^2 = |\langle{{k}}|\hat{{U}}_F^N|n \rangle|^2$', fontsize=yl_fs-1.5)
ax.set_xlim(-1,N)
ax.set_ylim(-0.05,1.05)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

ax.annotate(r'(a)', xy=(-0.19, 1.), xycoords='axes fraction', fontsize=tp_fs+0.5)
plt.savefig('Floquet_FIG_2.pdf', bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()
