import numpy as np
from PISC.dvr.dvr import DVR1D
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity
import matplotlib
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 11 
yl_fs = 11
tp_fs = 7

le_fs = 9#9.5
ti_fs = 8
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

def f_pow(vals,n,m,power=4):
    E = (vals[n]+vals[m])/2
    w = (vals[n]-vals[m])
    return 1/(1+abs(w)**power)

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


fig, ax = plt.subplots(3,2, figsize=(4,6))

fig.subplots_adjust(hspace=0.3, wspace=0.45)

f_list = [f_G, f_E, f_pow]
f_labels = ['Gaussian', 'Exponential', 'Power-law']

ax0 = ax[2,1]
b_arr_G = compute_avg_bn(nmat, neigs, beta, ncoeff, f_G, coeff=0.5)
alpha_G = np.pi/beta
ax0.scatter(n_arr[1:], b_arr_G[1:].real, label='Gaussian', s=scat_size, color='darkblue')
ax0.plot(n_arr[:4], alpha_G*n_arr[:4], label=r'$\pi n/\beta$', color='red', linestyle='dashed')
ax0.set_xlabel(r'$n$', fontsize=xl_fs)
ax0.set_ylabel(r'$b_n$', fontsize=yl_fs)
ax0.annotate(r'$\langle m| \hat{O} |n \rangle \sim e^{-(E_m-E_n)^2/2}$', xy=(0.5, 0.85), xycoords='axes fraction', fontsize=tp_fs, ha='center')
ax0.tick_params(axis='both', which='major', labelsize=le_fs)
print('Gaussian done')
ax0.annotate(r'$(f)$', xy=(-0.4, 0.9), xycoords='axes fraction', fontsize=xl_fs, ha='left')

#### Save gaussian data
np.savetxt('FIG1_Gaussian_beta1.txt', np.column_stack((n_arr, b_arr_G.real)), header='n b_n')

ax1 = ax[1,1]
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
#ax1.set_xlabel(r'$n$', fontsize=xl_fs)
ax1.set_ylabel(r'$b_n$', fontsize=yl_fs)
ax1.set_ylim([0,ax1.get_ylim()[1]*1.2])
ax1.annotate(r'$\langle m| \hat{O} |n \rangle \sim e^{-|E_m-E_n|}$', xy=(0.5, 0.85), xycoords='axes fraction', fontsize=tp_fs, ha='center')
ax1.tick_params(axis='both', which='major', labelsize=le_fs)
ax1.annotate(r'$(d)$', xy=(-0.4, 0.9), xycoords='axes fraction', fontsize=xl_fs, ha='left')
print('Exponential done')

#### Save exponential data
np.savetxt('FIG1_Exponential_beta1.txt', np.column_stack((n_arr, b_arr_E.real)), header='n b_n')

ax2 = ax[0,1]
alpha = np.pi/beta
if(0):
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

b_arr_P = compute_avg_bn(nmat, neigs, beta, ncoeff, f_pow, coeff=2)
ax2.scatter(n_arr[1:], b_arr_P[1:].real, label='Power-law', s=scat_size, color='green')

#### Save power-law data
np.savetxt('FIG1_Powerlaw_beta1.txt', np.column_stack((n_arr, b_arr_P.real)), header='n b_n')

#fit b_arr_P to alpha*n_arr for n in range(1,10)
n_arr_trunc = n_arr[3:10]
b_arr_trunc = b_arr_P[3:10].real
coeffs = np.polyfit(n_arr_trunc, b_arr_trunc, 1)
alpha = coeffs[0]\

print('Power-law fit alpha:', alpha, np.pi/beta)
ax2.plot(n_arr[:20], np.pi*n_arr[:20], label=r'$\pi n/\beta$', color='red', linestyle='dashed')
#ax2.set_xlabel(r'$n$', fontsize=xl_fs)
ax2.set_ylabel(r'$b_n$', fontsize=yl_fs)
ax2.set_ylim([0,ax2.get_ylim()[1]*1.2])
ax2.annotate(r'$\langle m| \hat{O} |n \rangle \sim \frac{1}{1+|E_m-E_n|^2}$', xy=(0.5, 0.85), xycoords='axes fraction', fontsize=tp_fs, ha='center')
ax2.tick_params(axis='both', which='major', labelsize=le_fs)
ax2.annotate(r'$(b)$', xy=(-0.4, 0.9), xycoords='axes fraction', fontsize=xl_fs, ha='left')

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

N = 200
matrix = np.zeros((N, N))


def mat_comp(N, func):
    for i in range(N):
        for j in range(N):
            if i == j:
                matrix[i, j] = 1
            else:
                if func=='pl':
                    matrix[i, j] = 1e3/abs(i-j)**2 #
                elif func=='exp':
                    matrix[i, j] = 1e3*np.exp(-abs(i-j))
                elif func=='gauss':
                    matrix[i, j]  = 1e3*np.exp(-0.5*abs(i-j)**2)
    return matrix

def funct(funckey):
    xarr = np.linspace(1, N, 100)
    yarr = np.zeros(100)
    if funckey=='pl':
        return xarr, 1e3/xarr**2
    elif funckey=='exp':
        return xarr, 1e3*np.exp(-xarr)
    elif funckey=='gauss':
        return xarr, 1e3*np.exp(-1e-4*xarr**2)

def Lanczos(funckey):
    xarr = np.linspace(1, N, 100)
    yarr = np.zeros(100)
    if funckey=='pl':
        return xarr, xarr*np.pi*0.99
    elif funckey=='exp':
        return xarr, xarr*np.pi/2
    elif funckey=='gauss':
        return xarr, 10*xarr**0.5


axes = ax[:,0]

for a,func in zip(axes, ['pl', 'exp', 'gauss']):
    
    matrix = mat_comp(N, func)
    a.imshow((matrix), vmax=1, cmap='YlGn')  # 'magma')
    #a.set_title(r'$O_{ij} \sim$' + func, fontsize=tp_fs)
    a.set_ylabel(r'$i$', fontsize=yl_fs)
    
    #Set y label only for first plot
    if func == 'gauss':
        a.set_xlabel(r'$j$', fontsize=xl_fs)
    a.xaxis.set_major_locator(MultipleLocator(20))  # every 2 units on x-axis
    a.yaxis.set_major_locator(MultipleLocator(20))  # every 0.5 units on y-axis
    a.grid(True, color='k', linewidth=0.5)

    #plot dashed line along diagonal
    a.plot([0, N-1], [0, N-1], color='white', linestyle='--', linewidth=1)

    a.set_xticklabels([])
    a.set_yticklabels([])

    a.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

ax[0,0].annotate(r'$(a)$', xy=(-0.2, 0.9), xycoords='axes fraction', fontsize=xl_fs, ha='left')
ax[1,0].annotate(r'$(c)$', xy=(-0.2, 0.9), xycoords='axes fraction', fontsize=xl_fs, ha='left')
ax[2,0].annotate(r'$(e)$', xy=(-0.2, 0.9), xycoords='axes fraction', fontsize=xl_fs, ha='left')

axes = ax[:,0]

for axe, funckey in zip(axes, ['pl', 'exp', 'gauss']):
    
    # Create an inset axis within the current axis
    ax_inset = axe.inset_axes([0.65, 0.65, 0.3, 0.3])  # [x0, y0, width, height]

    a = ax_inset

    xarr, yarr = funct(funckey)
    a.plot(xarr, np.log(yarr)/np.log(yarr)[0], color='b', linewidth=1)
    a.set_xlabel(r'$|E_i-E_j|$', fontsize=xl_fs-5, labelpad=5)
    a.set_ylabel(r'$\log O_{ij}$', fontsize=yl_fs-5, labelpad=5)
    # Make fonts bolder
    a.xaxis.label.set_fontweight('bold')
    a.yaxis.label.set_fontweight('bold')

    #bring xlabel and ylable closer to axis
    a.xaxis.set_label_coords(0.5,-0.08)
    a.yaxis.set_label_coords(-0.05,0.5)
    
    white_bg = Rectangle(
        (-0.35, -0.3), 1.4, 1.4, transform=a.transAxes, 
                     facecolor='white', edgecolor='black', 
                     linewidth=1, zorder=-1, clip_on=False)
    a.add_patch(white_bg)
    
    # Keep only bottom and left spines (x and y axes)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    #set x limit to 0-300
    a.set_xlim(0, 300)
    a.set_yticks([1])
    a.set_xticks([])
    a.set_xticklabels([])
    a.set_yticklabels([])

    #Arrows on both axes
    a.annotate('', xy=(1, 0), xytext=(0, 0), xycoords='axes fraction', textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='black', lw=1))
    a.annotate('', xy=(0, 1), xytext=(0, 0), xycoords='axes fraction', textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='black', lw=1))


#plt.savefig('FIG1_D2.pdf', dpi=300, bbox_inches='tight',pad_inches=0.05)
plt.show()


if(0):
    plt.scatter(n_arr[1:], b_arr[1:].real, label='Krylov coefficients')
    #plt.plot(n_arr[:10], np.pi*n_arr[:10]/beta)
    plt.xlabel('Coefficient index')
    plt.ylabel('Coefficient value')
    plt.title('Krylov Coefficients for Random Matrix Hamiltonians')
    plt.legend()
    plt.show()
    
