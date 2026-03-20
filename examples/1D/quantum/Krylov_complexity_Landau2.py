import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from PISC.potentials import mildly_anharmonic
from scipy.special import gamma

ngrid = 2000

L=20
lb=-L
ub=L

m=1.

print('L',L)

a=0.0
b=0.0
omega=1.0
n_anharm = 4

def tunneling_factor(p):
    num = gamma( (p-2)/(2*p) )
    denom = gamma( (p-1)/p )
    pref = np.sqrt(2*np.pi)/(p+2)
    ret = pref*(num/denom)*energy_factor(p)
    return ret

def J(p):
    num = gamma(1/p)*gamma(1.5)
    denom = p*gamma(1/p+1.5)
    return num/denom


def energy_factor(p):
    num = np.pi/(2*np.sqrt(2*m))
    denom = J(p)
    pref = (num/denom)**(2*p/(p+2))
    return pref

pes = mildly_anharmonic(m,a,b,omega,n=n_anharm)

potkey = 'MAH_w_{}_a_{}_b_{}_n_{}'.format(omega,a,b,n_anharm)
neigs = 200

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]


if(0):
    plt.plot(x_arr, vecs[:,-1], label='p={}'.format(n_anharm))
    plt.plot(x_arr, pes.potential_func(x_arr), label='potential')
    plt.xlabel('x')
    plt.ylabel('wavefunction')
    plt.title('Wavefunction')
    plt.legend()
    plt.show()

#O = (pos_mat)
O = np.zeros((neigs,neigs))

tf = tunneling_factor(n_anharm)
print('tf', tf)

tf=0.1
for i in range(neigs):
    for j in range(i,neigs):
        if (i-j) % 2 == 0 and abs(i-j) > 0:
            O[i,j] = np.exp(-tf*abs(vals[i]-vals[j])**1.25)
            #O[i,j] = np.exp(-tf*abs(i-j)**2)
            O[j,i] = O[i,j]

quantum = False
if(quantum):
    print('quantum')
    O = pos_mat
else:
    print('semiclassical')

nmoments = 20
ncoeff = 50
T = 3.0
beta = 1.0/T

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

def K_complexity(O,label,fit=False):
    # Compute the Krylov complexity and moments
    # This function is a placeholder for the actual implementation
    
    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, 'asm', 0.5, moments)
    even_moments = moments[0::2]

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')

    #fit barr with a line
    coeffs = np.polyfit(np.arange(ncoeff)[1:], barr[1:], 1)
    poly = np.poly1d(coeffs)

    print('alpha', coeffs[0], np.pi*T, np.pi*T - tf)

    if(fit):
        plt.plot(np.arange(ncoeff), poly(np.arange(ncoeff)), 'k--')#, label='fit')
    plt.scatter(np.arange(ncoeff)[1:], barr[1:], label=label)#, label='p={}'.format(n_anharm))

K_complexity(pos_mat,r'$\hat{O}=\hat{x}$, (DVR)',fit=False)
K_complexity(O,r'$\hat{O} = \hat{x}$, (Landau)',fit=True)

#plt.plot(np.arange(ncoeff), np.pi*T*np.arange(ncoeff), 'r--', label=r'$\alpha = \pi k_B T$')
plt.xlabel(r'$n$')
plt.ylabel(r'$b_n$')
plt.title('Lanczos coefficients')
plt.legend()
plt.show()

exit()
