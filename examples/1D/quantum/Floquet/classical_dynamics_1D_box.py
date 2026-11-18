import numpy as np
from matplotlib import pyplot as plt
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from classical_dynamics import hw_Integrator
from scipy import optimize as opt
import matplotlib
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

m = 1.0
g = 3.0

IPR_maxlist_op2 = [5.0, 7.5, 10.0, 12.25, 14.75,17.25, 19.75, 22.25, 24.75, 27.25, 29.75]
IPR_maxlist_op1 = [3.75,6.25, 8.75, 11.25, 13.75, 16.0, 18.5, 21.0, 23.5, 26.0, 28.5] 

w_F = 13.75#19.75#14.75# 19.75
dt = 0.005
L = 2.0
E1 = np.pi**2/(2*m*L**2)  # Ground state energy of a particle in a box
En = np.arange(1, 25)**2 * E1  # Energy levels for the particle in the box 
print('w_F', w_F,'E3-E1', En[2]-En[0],'E4-E2', En[3]-En[1],'E5-E3', En[4]-En[2], 'E6-E4', En[5]-En[3], 'E7-E5', En[6]-En[4])
print('w_F', w_F, 'E2-E1', En[1]-En[0], 'E3-E2', En[2]-En[1], 'E4-E3', En[3]-En[2], 'E5-E4', En[4]-En[3], 'E6-E5', En[5]-En[4])
print('w_F', w_F, 'E8-E6', En[7]-En[5], 'E9-E7', En[8]-En[6], 'E10-E8', En[9]-En[7], 'E11-E9', En[10]-En[8], 'E12-E10', En[11]-En[9])
E_tot = En[3]  # Total energy for the particle in the box
E_list = [En[4],En[5], En[6]]  # List of energies for 2-photon processes
E_list = [En[4],En[5]]

if(1):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax_inset = fig.add_axes([0.18, 0.41, 0.27, 0.22])  # [left, bottom, width, height]
    ax_inset_2 = fig.add_axes([0.58, 0.41, 0.27, 0.22])  # [left, bottom, width, height]
    for c, E_tot in zip(['blue', 'orange', 'green', 'red'], E_list):
        print(f"Total energy: {E_tot:.4f}, Corresponding momentum: {np.sqrt(2 * m * E_tot):.4f}")
        print('E_diff',E_list[1]-E_list[0] , 'E_diff/w_F', (E_list[1]-E_list[0])/w_F)
        rng = np.random.default_rng(seed=100)
        x_init_arr = rng.uniform(-L/2+0.05, L/2-0.05,10)  # Sample initial position from a uniform distribution
        p_init_arr = np.sqrt(2 * m * E_tot)*np.ones_like(x_init_arr)  # Sample initial momenta based on energy conservation (V(x) = 0 inside the box)

        nT = 300
        #x_init = x_init_arr[28]
        #p_init = p_init_arr[28]

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
            ax_inset_2.scatter(x_list, p_list, color=c, alpha=0.9, s=1)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-1.25*np.sqrt(2 * m * E_tot), 1.25*np.sqrt(2 * m * E_tot))
    ax_inset.set_xlim(-L/2, L/2)
    ax_inset.set_ylim(6.5,10.5)
    ax_inset_2.set_xlim(-L/2, L/2)
    ax_inset_2.set_ylim(-11.5,-6)

    #plt.figure(figsize=(8, 6))
    ax.axhline(np.sqrt(2 * m * E_tot), color='red', linestyle='--', label='p = sqrt(2mE)')
    ax.axhline(-np.sqrt(2 * m * E_tot), color='red', linestyle='--')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$p$")

    ax.set_title("Stroboscopic sections at $w_F$ = {:.2f}".format(w_F))
    plt.savefig('phase_space_one_photon_FLOQUET.png', dpi=300, bbox_inches='tight')
    plt.show()

if(0):
    fig, ax = plt.subplots(figsize=(6, 4))
    #Plot inset 
    ax_inset = fig.add_axes([0.18, 0.41, 0.27, 0.22])  # [left, bottom, width, height]
    ax_inset_2 = fig.add_axes([0.58, 0.41, 0.27, 0.22])  # [left, bottom, width, height]
    for c, E_tot in zip(['blue', 'orange', 'green', 'red'], E_list):
        print(f"Total energy: {E_tot:.4f}, Corresponding momentum: {np.sqrt(2 * m * E_tot):.4f}")
        print('E_diff',E_list[1]-E_list[0] , 'E_diff/w_F', (E_list[1]-E_list[0])/w_F)
        rng = np.random.default_rng(seed=100)
        x_init_arr = rng.uniform(-L/2+0.05, L/2-0.05,10)  # Sample initial position from a uniform distribution
        p_init_arr = np.sqrt(2 * m * E_tot)*np.ones_like(x_init_arr)  # Sample initial momenta based on energy conservation (V(x) = 0 inside the box)

        nT = 300
        #x_init = x_init_arr[28]
        #p_init = p_init_arr[28]

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
            ax_inset_2.scatter(x_list, p_list, color=c, alpha=0.9, s=1)

    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-1.25*np.sqrt(2 * m * E_tot), 1.25*np.sqrt(2 * m * E_tot))
    ax_inset.set_xlim(-L/2, L/2)
    ax_inset.set_ylim(6.5,12)
    ax_inset_2.set_xlim(-L/2, L/2)
    ax_inset_2.set_ylim(-12.5,-6)


    #plt.figure(figsize=(8, 6))
    ax.axhline(np.sqrt(2 * m * E_tot), color='red', linestyle='--', label='p = sqrt(2mE)')
    ax.axhline(-np.sqrt(2 * m * E_tot), color='red', linestyle='--')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$p$")



    ax.set_title("Stroboscopic sections at $w_F$ = {:.2f}".format(w_F))


    plt.savefig('phase_space_two_photon_FLOQUET.png', dpi=300, bbox_inches='tight')
    plt.show()

if(0):
    x_all = []
    p_all = []
    for x_init, p_init in zip(x_init_arr, p_init_arr):
        hw_int = hw_Integrator(m, g, w_F, dt, L, x_init, p_init)
        for n in range(nT):
            hw_int.Tstep()
            x_all.append(hw_int.x)
            p_all.append(hw_int.p)

    x_all = np.array(x_all)
    p_all = np.array(p_all)
    print('x_all', x_all.shape, 'p_all', p_all.shape)
    exit()

exit()

