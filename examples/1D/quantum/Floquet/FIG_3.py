from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct, compute_ipr
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
import numpy as np
import matplotlib
import scipy
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.animation import FuncAnimation

m = 1.0
L = 2.0
ngrid = 2**9 
neigs = ngrid-1

lb = 0
ub = L

xl_fs = 16
yl_fs = 16
tp_fs = 12

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

#--------------------------------------------------------------------------------------

nph = 2
IPR_maxlist_2p = [5.0, 7.5, 10.0, 12.25, 14.75,17.25, 19.75, 22.25, 24.75, 27.25, 29.75]
IPR_maxlist_1p = [3.75,6.25, 8.75, 11.25, 13.75, 16.0, 18.5, 21.0, 23.5, 26.0, 28.5] 

g_arr = np.linspace(0.0, 2.0, 21)
#Extend g_arr to include 2-4
g_arr = np.concatenate((g_arr, np.linspace(2.0, 4.0, 21)))

w_F_2p = np.around((vals[6]-vals[4])/2, decimals=3)
w_F_1p = np.around((vals[1]-vals[0]), decimals=3)
key = 'pos_mat'


def find_qS_pm(qS, n, nph=1):
    """
    Given an array of quasistates qS, find the states with top 2 (or 3) 
    maximum projection on n, n+1 (and n+2) and return their indices and projections.
    """
    proj_n = np.abs(qS[n,:])**2
    proj_n1 = np.abs(qS[n+1,:])**2
    proj_n2 = np.abs(qS[n+2,:])**2

    #Find the indices of the top 2/3 maximum projections
    if(nph==1):
        proj_tot = proj_n + proj_n1
        max_inds = np.argsort(proj_tot)[-2:][::-1]
    elif(nph==2):
        proj_tot = proj_n + proj_n1 + proj_n2
        max_inds = np.argsort(proj_tot)[-3:][::-1]
    return max_inds

fig,axes = plt.subplots(2,1,figsize=(4,8), sharex=True)
#Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.0)

ax = axes[0]
#One-photon case
n=0
nph=1
w_F = w_F_1p

H_2L_diag = np.array([[vals[n], 0],[0, vals[n+1] - w_F]])
H_2L_offdiag = np.array([[0, pos_mat[n,n+1]/2j], [-pos_mat[n+1,n]/2j, 0]])

vals_list_0 = []
vals_list_1 = []

for g in g_arr:
    U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)

    print('n',n, 'g', g, 'nph', nph)
    pm = find_qS_pm(qS, n, nph=nph)

    ipr_max,E_ind,max_ind = compute_ipr(qS, metric='max')

    H_2L = H_2L_diag + g*H_2L_offdiag
    vals_tls, vecs_tls = np.linalg.eigh(H_2L)

    ax.scatter(np.ones(len(qE))*g, qE, s=1, c='0.5')
    if(g==0.0):
        print('refs', qE[pm[0]], qE[pm[1]], vals_tls[0], vals_tls[1])
        ref = qE[pm[0]]-vals_tls[0]
        
    for i in range(len(pm)):
        #Always plot the least energy state with red, highest one with black, and the middle one with green
        #Find which of the pm[i] is the least energy state
        min_ind = np.argmin(qE[pm])
        max_ind = np.argmax(qE[pm])
        if(i==min_ind):
            c = 'r'
        elif(i==max_ind):
            c = 'g'
        ax.scatter(g, qE[pm[i]], color=c, marker='o',s=10, joinstyle='round', alpha=0.5) 

    min_ind = np.argmin(vals_tls)
    max_ind = np.argmax(vals_tls)
    vals_list_0.append(vals_tls[min_ind]+ref)
    vals_list_1.append(vals_tls[max_ind]+ref)

print('vals_list_0', vals_list_0)
ax.plot(g_arr, vals_list_0, color='r', label='TLS 1')
ax.plot(g_arr, vals_list_1, color='g', label='TLS 2')
ax.set_xlim([-0.1, 4.1])
ax.set_ylim([-2.0,-0.3])

ax.set_ylabel(r'$\epsilon_q$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.annotate(r'(a)', xy=(-0.23, 0.93), xycoords='axes fraction', fontsize=tp_fs)
#----------------------------------------------------------------

ax = axes[1]
#Two-photon case
n=4 
nph=2
w_F = w_F_2p

H_3L_diag = np.array([[vals[n], 0, 0],[0, vals[n+1] - w_F, 0],[0, 0, vals[n+2] - 2*w_F]])
H_3L_offdiag = np.array([[0, pos_mat[n,n+1]/2j, 0], [-pos_mat[n,n+1]/2j, 0, pos_mat[n+1,n+2]/2j],[0, -pos_mat[n+1,n+2]/2j, 0]])

vals_list_0 = []
vals_list_1 = []
vals_list_2 = []
ref = 0.0

for g in g_arr:
    U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)

    print('n',n, 'g', g, 'nph', nph)
    pm = find_qS_pm(qS, n, nph=nph)

    ipr_max,E_ind,max_ind = compute_ipr(qS, metric='max')

    if(nph==1):
        H_2L = H_2L_diag + g*H_2L_offdiag
        vals_tls, vecs_tls = np.linalg.eig(H_2L)
    elif(nph==2):
        H_3L = H_3L_diag + g*H_3L_offdiag
        vals_tls, vecs_tls = np.linalg.eig(-H_3L)

    ax.scatter(np.ones(len(qE))*g, qE, s=1, c='0.5')
    if(g==0.0):
        print('refs', qE[pm[0]], qE[pm[1]], vals_tls[0], vals_tls[1])
        ref = qE[pm[0]]-vals_tls[0]
        
    for i in range(len(pm)):
        #Always plot the least energy state with red, highest one with black, and the middle one with green
        #Find which of the pm[i] is the least energy state
        min_ind = np.argmin(qE[pm])
        max_ind = np.argmax(qE[pm])
        if(i==min_ind):
            c = 'r'
        elif(i==max_ind):
            c = 'k'
        else:
            c = 'g'
        ax.scatter(g, qE[pm[i]], color=c, marker='o',s=10, joinstyle='round', alpha=0.5) 
        
    min_ind = np.argmin(vals_tls)
    max_ind = np.argmax(vals_tls)
    intermediate_ind = 3 - min_ind - max_ind
    vals_list_0.append(vals_tls[min_ind]+ref)
    vals_list_1.append(vals_tls[intermediate_ind]+ref)
    vals_list_2.append(vals_tls[max_ind]+ref)

ax.plot(g_arr, vals_list_0, color='r', label='TLS 1')
ax.plot(g_arr, vals_list_1, color='g', label='TLS 2')
ax.plot(g_arr, vals_list_2, color='k', label='TLS 3')
            
ax.set_xlim([-0.1, 4.1])
ax.set_ylim([-2.0,0.75])

ax.set_xlabel(r'$g$', fontsize=xl_fs)
ax.set_ylabel(r'$\epsilon_q$', fontsize=yl_fs)
ax.tick_params(axis='both', which='major', labelsize=tp_fs)

ax.annotate(r'(b)', xy=(-0.23, 0.93), xycoords='axes fraction', fontsize=tp_fs)

plt.savefig('Floquet_FIG_3.pdf', bbox_inches='tight', dpi=300, pad_inches=0.0)

plt.show()

