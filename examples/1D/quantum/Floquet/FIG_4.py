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


#potkey = 'L_{}_anharmonic_{}_MANU'.format(10.0,4)
#potkey = 'L_{}_anharmonic_{}_MANU'.format(4.0,8)

g= 3.0
w_F_arr = np.arange(0.5, 10.01, 0.25)
w_F_arr = np.around(w_F_arr, decimals=2)

fig, ax = plt.subplots(1,2, figsize=(8,3), constrained_layout=True)

c = 'r'
def plot_ipr_max_vs_w_F(ax, potkey, g, w_F_arr, key, c,legend=None): 
    ipr_max_arr = []

    for w_F in w_F_arr:
        U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)

        ipr_max,E_ind,max_ind = compute_ipr(qS, metric='max')
        ipr_max_arr.append(ipr_max)
    if legend is not None:
        ax.scatter(w_F_arr, ipr_max_arr, color=c, s=10, zorder=1,label=legend)
    else:
        ax.scatter(w_F_arr, ipr_max_arr, color=c, s=10, zorder=1)

potkey = 'L_{}_1D_box_MANU'.format(L)
key = 'mom_mat_NN'#
#key = 'linear_growth'
plot_ipr_max_vs_w_F(ax[1], potkey, g, w_F_arr, key, 'm',legend=r'$\hat{{O}} = \hat{{p}}_{{NN}}$')
ax[1].set_ylim([0.5,7.0])
ax[1].set_xlim([0.81,10.2])
ax[1].set_xticks([1,2,3,4,5,6,7,8,9,10])
ax[1].tick_params(axis='both', which='major', labelsize=tp_fs)
ax[1].annotate(r'(b)', xy=(-0.12, 0.93), xycoords='axes fraction', fontsize=tp_fs)

potkey = 'L_{}_1D_box_0_MANU'.format(L)
key = 'pos_mat'
plot_ipr_max_vs_w_F(ax[1], potkey, g, w_F_arr, key, 'c',legend=r'$\hat{{O}} = \hat{{x}}$')

ax[1].legend(fontsize=tp_fs, loc='upper left', frameon=False)

potkey = 'L_{}_anharmonic_{}_MANU'.format(10.0,4)
key = 'pos_mat'
plot_ipr_max_vs_w_F(ax[0], potkey, g, w_F_arr, key, 'b',legend=r'$V(x) = x^4$')

potkey = 'L_{}_anharmonic_{}_MANU'.format(6,6)
key = 'pos_mat'
plot_ipr_max_vs_w_F(ax[0], potkey, g, w_F_arr, key, 'g',legend=r'$V(x) = x^6$')

potkey = 'L_{}_anharmonic_{}_MANU'.format(4.0,8)
key = 'pos_mat'
plot_ipr_max_vs_w_F(ax[0], potkey, g, w_F_arr, key, c,legend=r'$V(x) = x^8$')

ax[0].set_xlim([0.8,10.2])
ax[0].set_xticks([1,2,3,4,5,6,7,8,9,10])
ax[0].tick_params(axis='both', which='major', labelsize=tp_fs)
ax[0].legend(fontsize=tp_fs, loc='upper left', frameon=False)
ax[0].annotate(r'(a)', xy=(-0.17, 0.93), xycoords='axes fraction', fontsize=tp_fs)

ax[0].set_xlabel(r'$\omega_F$', fontsize=xl_fs)
ax[1].set_xlabel(r'$\omega_F$', fontsize=xl_fs)
ax[0].set_ylabel(r'$\text{PR}_\text{max}$', fontsize=yl_fs)
#ax.set_ylim(0.75,3.5)

plt.savefig('Floquet_FIG_4.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.show()
