import numpy as np
from matplotlib import pyplot as plt
from classical_dynamics import hw_Integrator
from scipy import optimize as opt
import matplotlib
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

m = 1.0
g = 3.0

dt = 0.005
L = 2.0
E1 = np.pi**2/(2*m*L**2)  # Ground state energy of a particle in a box
En = np.arange(1, 25)**2 * E1  # Energy levels for the particle in the box 

def plot_box(ax, E_list, w_F,L,l,u):
    ax_inset = ax.inset_axes([0.25, 0.39, 0.5, 0.32])  # [left, bottom, width, height]
    #ax_inset_2 = ax.inset_axes([0.58, 0.41, 0.27, 0.22])  # [left, bottom, width, height]
    for c, E_tot in zip(['blue', 'orange', 'green', 'red'], E_list):
        print(f"Total energy: {E_tot:.4f}, Corresponding momentum: {np.sqrt(2 * m * E_tot):.4f}")
        rng = np.random.default_rng(seed=101)
        x_init_arr = rng.uniform(-L/2+0.05, L/2-0.05,22)  # Sample initial position from a uniform distribution
        p_init_arr = np.sqrt(2 * m * E_tot)*np.ones_like(x_init_arr)  # Sample initial momenta based on energy conservation (V(x) = 0 inside the box)

        nT = 200

        for x_init, p_init in zip(x_init_arr, p_init_arr):
            x_list = []
            p_list = []
            hw_int = hw_Integrator(m, g, w_F, dt, L, x_init, p_init)    
            for n in range(nT):
                hw_int.Tstep_collision()
                #print(f"Step {n+1}: x = {hw_int.x}, p = {hw_int.p}, t = {hw_int.t}")
                x_list.append(hw_int.x)
                p_list.append(hw_int.p)
            
            ax.scatter(x_list, p_list, color=c, alpha=0.9, s=4)
            ax_inset.scatter(x_list, p_list, color=c, alpha=0.9, s=1)
            #ax_inset_2.scatter(x_list, p_list, color=c, alpha=0.9, s=1)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-1.25*np.sqrt(2 * m * E_tot), 1.25*np.sqrt(2 * m * E_tot))

    #plt.figure(figsize=(8, 6))
    ax.axhline(np.sqrt(2 * m * E_tot), color='red', linestyle='--', label='p = sqrt(2mE)')
    ax.axhline(-np.sqrt(2 * m * E_tot), color='red', linestyle='--')
    #ax.set_xlabel(r"$x$")
    #ax.set_ylabel(r"$p$")

    
    ax_inset.set_xlim(-L/2, L/2)
    p = np.sqrt(2 * m * E_tot)
    ax_inset.set_ylim(p-l*p, p+u*p)



#One photon resonance
def one_photon_resonance(n):
    w_F_op = En[n+1]-En[n]
    Is_op = m*L**2*w_F_op/(np.pi**2)
    Emid_op = (En[n+1]+En[n])/2
    print('Emid', (En[n+1]+En[n])/2, 'Is', Is_op, 'E(Is)',np.pi**2*Is_op**2/(2*m*L**2))
    
    E_list_op = [Emid_op,En[n],En[n+1]]  # Energies for the stroboscopic sections
    return w_F_op, Is_op, Emid_op, E_list_op

#Two photon resonance
def two_photon_resonance(n):
    w_F_tp = (En[n+2]-En[n])/2
    Is_tp = m*L**2*w_F_tp/(np.pi**2)
    Emid_tp = En[n+1]
    print('Emid', En[n+1], 'Is', Is_tp, 'E(Is)',np.pi**2*Is_tp**2/(2*m*L**2))

    E_list_tp = [Emid_tp,En[n],En[n+2]]  # Energies for the stroboscopic sections
    return w_F_tp, Is_tp, Emid_tp, E_list_tp

wf_tp1, Is_tp1, Emid_tp1, E_list_tp1 = two_photon_resonance(3)
wf_op1, Is_op1, Emid_op1, E_list_op1 = one_photon_resonance(3)

wf_tp2, Is_tp2, Emid_tp2, E_list_tp2 = two_photon_resonance(10)
wf_op2, Is_op2, Emid_op2, E_list_op2 = one_photon_resonance(10)

fig, ax = plt.subplots(2,2, figsize=(8, 6))

plot_box(ax[0,0], E_list_op1, wf_op1 ,L, 0.35,0.15)
plot_box(ax[1,0], E_list_op2, wf_op2 ,L, 0.15,0.08)
plot_box(ax[0,1], E_list_tp1, wf_tp1 ,L, 0.45,0.1)
plot_box(ax[1,1], E_list_tp2, wf_tp2 ,L,0.2,0.05)

ax[1,0].set_xlabel(r"$x$",fontsize=12)
ax[0,0].set_ylabel(r"$p$",fontsize=12)
ax[1,1].set_xlabel(r"$x$",fontsize=12)
ax[1,0].set_ylabel(r"$p$",fontsize=12)

#Mark labels (a), (b), (c), (d) for the four subplots
ax[0,0].annotate('(a)', xy=(-0.18, 0.87), xycoords='axes fraction', fontsize=12)
ax[1,0].annotate('(b)', xy=(-0.18, 0.87), xycoords='axes fraction', fontsize=12)
ax[0,1].annotate('(c)', xy=(-0.18, 0.87), xycoords='axes fraction', fontsize=12)
ax[1,1].annotate('(d)', xy=(-0.18, 0.87), xycoords='axes fraction', fontsize=12)

ax[0,0].set_title('One-photon resonance', fontsize=12)
ax[0,1].set_title('Two-photon resonance', fontsize=12)

#plt.savefig('Floquet_FIG_5.pdf', bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

