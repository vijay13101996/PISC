import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from PISC.potentials import mildly_anharmonic
from scipy.special import gamma

ngrid = 1000

L=6
lb=-L
ub=L

m=1.

print('L',L)

a=0.0
b=1.0
omega=0.0
n_anharm = 8

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

print('J', J(n_anharm) )

def energy_factor(p):
    num = np.pi/(2*np.sqrt(2*m))
    denom = J(p)
    pref = (num/denom)**(2*p/(p+2))
    return pref

pes = mildly_anharmonic(m,a,b,omega,n=n_anharm)

potkey = 'MAH_w_{}_a_{}_b_{}_n_{}'.format(omega,a,b,n_anharm)
neigs = 350
narr = np.arange(1,neigs+1)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
vals,vecs = DVR.Diagonalize(neig_total=neigs)
lognarr = np.log(np.arange(1,neigs+1))[50:]
logvals = np.log(vals)[50:]

#fit narr and vals to power law
coeffs = np.polyfit(lognarr, logvals, 1)
poly = np.poly1d(coeffs)
print('Ecoeffs', coeffs[0], 2*n_anharm/(n_anharm+2))
#pref = np.exp(coeffs[1])
pref = energy_factor(n_anharm)
print('pref', np.exp(coeffs[1]), pref)
print('tunneling factor', tunneling_factor(n_anharm))

#plt.plot(np.log(narr), np.log(vals), 'o', label='p={}'.format(n_anharm))
#plt.plot(np.log(narr), poly(np.log(narr)), 'r--', label='fit')
#plt.show()

#print('tunneling factor', tunneling_factor(n_anharm))


x_arr = DVR.grid[1:ngrid]

dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
O = (pos_mat)

k=10
nvals = np.arange(1,neigs+1)[k+1::2]

Omat = abs(pos_mat[k,k+1::2])

logOmat = np.log(Omat[1:11])
narr = nvals[1:11]

plt.plot(narr, logOmat, 'o', label='p={}'.format(n_anharm))

#Fit narr and logOmat to a line
coeffs = np.polyfit(narr, logOmat, 1)
poly = np.poly1d(coeffs)

print('coeffs', abs(coeffs[0]), tunneling_factor(n_anharm))

x = np.linspace(narr[0], narr[-1], 100)
plt.plot(x, poly(x), 'r--', label='fit')



#plt.plot((nvals[:20]), np.log(Omat[:20]), 'o', label='p={}'.format(n_anharm))

plt.show()
