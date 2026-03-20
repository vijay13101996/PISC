import numpy as np
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity_1D as Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
import scipy

"""
In this module, we compute the Lanczos coefficients and the corresponding basis operators
for a given Hamiltonian and initial operator using the Lanczos algorithm.

We then use that to verify that the thermal average computed from the Krylov basis
and the Lanczos coefficients matches the exact wavefunction average.
"""

ngrid = 1000

Len = 10

lb = -Len
ub = Len

a = 0.0
b = 2.0
omega = 0.0
m = 1.0
n_anharm = 4
pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

def pos_mat_anal(neigs, m, omega):
    print('m, omega in anal pos mat:', m, omega)
    pos_mat = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(neigs):
            if(i-j==1):
                pos_mat[i,j] = np.sqrt(1/(2*m*omega))*np.sqrt(j+1)
            elif(j-i==1):
                pos_mat[i,j] = np.sqrt(1/(2*m*omega))*np.sqrt(i+1)
            
    return pos_mat

def C_T(O,tarr, beta, vals): #Compute Wightman TCF at finite temperature
    Z = np.sum(np.exp(-beta*vals))
    neigs = len(vals)
    C_T_arr = np.zeros(len(tarr),dtype=complex)
    for n in range(neigs):
        for m in range(neigs):
            C_T_arr += (1/Z)*np.exp(-beta*(vals[n]+vals[m])/2)*np.exp(1j*(vals[n]-vals[m])*tarr)*np.abs(O[n,m])**2
    return C_T_arr

def ip(O1, O2, vals, beta):
    Z = np.sum(np.exp(-beta*vals))
    neigs = len(vals)
    ip_val = 0.0
    for n in range(neigs):
        for m in range(neigs):
            ip_val += (1/Z)*np.exp(-beta*(vals[n]+vals[m])/2)*np.conj(O1[n,m])*O2[n,m]
    return ip_val

potkey = 'L_{}_MAH_w_{}_a_{}_b_{}_n_{}'.format(Len,omega,a,b,n_anharm)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 200
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

#O = pos_mat_anal(neigs, m, omega)
pos_mat = np.zeros((neigs,neigs))
O = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

ncoeff = 100
beta = 0.5
barr = np.zeros(ncoeff)
barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 0.5, 'wgm')

phi_mat = np.zeros((ncoeff,ncoeff))
for n in range(ncoeff):
    for m in range(ncoeff):
        if(n==m):
            phi_mat[n,m] = 0.0
        elif(n==m+1):
            phi_mat[n,m] = barr[n]
        elif(n==m-1):
            phi_mat[n,m] = -barr[m]

print('Phi matrix shape:', phi_mat[:10,:10])

#Solve for \dot{\phi}(t) = phi_mat \phi, with phi(0) = |0>
tmax = 20.0
ntimes = 201
tarr = np.linspace(0.0, tmax, ntimes)   

def func(y,t, A):
    return A @ y

y0 = np.zeros(ncoeff)
y0[0] = 1.0
phi_t = scipy.integrate.odeint(func, y0, tarr, args=(phi_mat,))

O/=np.sqrt(ip(O,O,vals,beta))

C_T_beta = C_T(O, tarr, beta, vals)
plt.plot(tarr, C_T_beta.real, label=r'Exact $C_{\beta}(t)$')
plt.plot(tarr, phi_t[:,0].real, '--', label=r'Krylov $C_{\beta}(t)$')
plt.xlabel('Time')
plt.ylabel(r'$C_{\beta}(t)$')
plt.legend()
plt.title('Thermal Wightman function comparison')
plt.show()

