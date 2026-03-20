import numpy as np
from PISC.dvr.dvr import DVR1D
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12
scat_size = 10

#Compute Krylov complexity for energy eigenvalues coming from a random-matrix Hamiltonian

sigma = 1
neigs = 300

def random_matrix(neigs):
    """Generate a random NxN matrix with elements drawn from a normal distribution."""
    H = np.zeros((neigs, neigs))

    for i in range(neigs):
        for j in range(i,neigs):
            if i == j:
                H[i, j] = np.random.normal(0, np.sqrt(2)*sigma)
            else:
                H[i, j] = np.random.normal(0, sigma)
            H[j, i] = H[i, j]
    return H

def TCF_O(vals, beta, neigs, O, t_arr):
    
    n_arr = np.arange(neigs)
    m_arr = np.arange(neigs)

    C_arr = np.zeros_like(t_arr) + 0j

    for n in n_arr:
        for m in m_arr:
            C_arr += np.exp(-beta*(vals[n]+vals[m])/2) * np.exp(1j*(vals[n]-vals[m])*t_arr) * np.abs(O[n,m])**2

    Z = np.sum(np.exp(-beta*vals))
    C_arr /= Z

    return C_arr

def Krylov_O(vals, beta, neigs, O, ncoeff):
    L = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,L)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff) 
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    return barr

def rho(E):
    #Density of states
    return 1.0 

def f_G(vals,n,m,coeff=0.5):
    E = (vals[n]+vals[m])/2
    w = (vals[n]-vals[m]) 
    return np.exp(-coeff*abs(w)**2)*np.random.normal(0,1)/rho(E)**0.5

def f_E(vals,n,m,coeff=1):
    E = (vals[n]+vals[m])/2
    w = (vals[n]-vals[m])
    return np.exp(-coeff*abs(w))*np.random.normal(0,1)/rho(E)**0.5

def f_unif(vals,n,m,coeff=1):
    E = (vals[n]+vals[m])/2
    w = (vals[n]-vals[m])
    return 1.0# coeff*np.random.normal(0,1)/rho(E)**0.5

nmat = 10
ncoeff = 100
n_arr = np.arange(ncoeff)

beta = 1
diffs = []

def compute_avg_bn(nmat, neigs, beta, ncoeff, f_O, coeff):
    b_arr = np.zeros(ncoeff)
    for i in range(nmat):
        H = random_matrix(neigs)
        vals, vecs = np.linalg.eigh(H)
        vals = np.sort(vals)

        diff = np.diff(vals)
        diffs.extend(diff)


        O = np.zeros((neigs, neigs))
        for n in range(neigs):
            for m in range(n,neigs):
                O[n,m] = f_O(vals,n,m)
                O[m,n] = O[n,m]

        b_arr+= Krylov_O(vals, beta, neigs, O, ncoeff)
    
    b_arr /= nmat
    return b_arr


fig, ax = plt.subplots(1,3, figsize=(9,3))

f_list = [f_G, f_E, f_unif]
f_labels = ['Gaussian', 'Exponential', 'Uniform']

ax0 = ax[0]
b_arr_G = compute_avg_bn(nmat, neigs, beta, ncoeff, f_G, coeff=0.5)
alpha_G = np.pi/beta
ax0.scatter(n_arr[1:], b_arr_G[1:].real, label='Gaussian', s=scat_size, color='blue')
ax0.plot(n_arr[:4], alpha_G*n_arr[:4], label=r'$\pi n/\beta$', color='red', linestyle='dashed')
ax0.set_xlabel(r'$n$', fontsize=xl_fs)
ax0.set_ylabel(r'$b_n$', fontsize=yl_fs)
ax0.annotate(r'$\langle m| \hat{O} |n \rangle \sim e^{-(E_m-E_n)^2/2}$', xy=(0.5, 0.85), xycoords='axes fraction', fontsize=tp_fs, ha='center')

print('Gaussian done')

ax1 = ax[1]
E_coeff = 1
alpha_E = np.pi/(beta + 4*E_coeff)
b_arr_E = compute_avg_bn(nmat, neigs, beta, ncoeff, f_E, coeff=E_coeff)
ax1.scatter(n_arr[1:], b_arr_E[1:].real, label='Exponential', s=scat_size, color='magenta')

#fit b_arr_E to alpha*n_arr for n in range(1,10)
n_arr_trunc = n_arr[1:50]
b_arr_trunc = b_arr_E[1:50].real
coeffs = np.polyfit(n_arr_trunc, b_arr_trunc, 1)
#alpha_E = coeffs[0]
print('Exponential fit alpha:', alpha_E, np.pi/(beta + 4*E_coeff))
ax1.plot(n_arr[:70], alpha_E*n_arr[:70], label=r'$\pi n/\beta$', color='black', linestyle='dashed')
ax1.plot(n_arr_trunc[:14], (np.pi/beta)*n_arr_trunc[:14], color='red', linestyle='dashed')
ax1.set_xlabel(r'$n$', fontsize=xl_fs)
ax1.set_ylim([0,ax1.get_ylim()[1]*1.2])
ax1.annotate(r'$\langle m| \hat{O} |n \rangle \sim e^{-|E_m-E_n|}$', xy=(0.5, 0.85), xycoords='axes fraction', fontsize=tp_fs, ha='center')

print('Exponential done')

ax2 = ax[2]
alpha = np.pi/beta
b_arr_U = compute_avg_bn(nmat, neigs, beta, ncoeff, f_unif, coeff=1)
ax2.scatter(n_arr[1:], b_arr_U[1:].real, label='Uniform', s=scat_size, color='green')

#fit b_arr_U to alpha*n_arr for n in range(1,10)
n_arr_trunc = n_arr[1:11]
b_arr_trunc = b_arr_U[1:11].real
coeffs = np.polyfit(n_arr_trunc, b_arr_trunc, 1)
alpha = np.pi/beta #coeffs[0]
print('Uniform fit alpha:', alpha, np.pi/beta)

ax2.plot(n_arr[1:15], alpha*n_arr[1:15], label=r'$\pi n/\beta$', color='red', linestyle='dashed')
ax2.set_xlabel(r'$n$')
ax2.annotate(r'$\langle m| \hat{O} |n \rangle \sim O(1)$', xy=(0.5, 0.85), xycoords='axes fraction', fontsize=tp_fs, ha='center')

if(0):
    for i in range(3):
        f_O = f_list[i]
        b_arr = compute_avg_bn(nmat, neigs, beta, ncoeff, f_O)
        if(0):
            plt.hist(diffs, bins=50, density=True)
            plt.xlabel('Energy level spacing')
            plt.ylabel('Probability density')
            plt.title('Energy Level Spacing Distribution for Random Matrix Hamiltonians')
            plt.show()

        ax.scatter(n_arr[1:], b_arr[1:].real, label=f_labels[i])
        #ax.plot(n_arr[1:20], np.pi*n_arr[1:20]/beta, label=r'$\pi n/\beta$', color='black', linestyle='dashed')

plt.suptitle('Krylov Coefficients for Random Matrix Hamiltonians')
#plt.legend()
#plt.savefig('draft_RMT_lanczos_coeffs.pdf', dpi=300, bbox_inches='tight',pad_inches=0.0)
plt.show()


if(0):
    plt.scatter(n_arr[1:], b_arr[1:].real, label='Krylov coefficients')
    #plt.plot(n_arr[:10], np.pi*n_arr[:10]/beta)
    plt.xlabel('Coefficient index')
    plt.ylabel('Coefficient value')
    plt.title('Krylov Coefficients for Random Matrix Hamiltonians')
    plt.legend()
    plt.show()
    
