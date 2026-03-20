import numpy as np
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity_1D as Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
import scipy
from matplotlib import pyplot as plt
from Krylov_WF_tools import coh_st_coeff, wf_t, comp_Ot, Cn, corr_func, av_O, avg_O, fix_vecs, mom_to_bn

"""
In this module, we compute the Lanczos coefficients and the corresponding basis operators
for a given Hamiltonian and initial operator using the Lanczos algorithm.

We then use that to verify that the wavefunction average computed from the Krylov basis
and the Lanczos coefficients matches the exact wavefunction average.
"""

ngrid = 1000

Len = 40

lb = -Len
ub = Len

a = 0.0
b = 0.0
omega = 2.0
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


potkey = 'L_{}_MAH_w_{}_a_{}_b_{}_n_{}'.format(Len,omega,a,b,n_anharm)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 200
vals,vecs = DVR.Diagonalize(neig_total=neigs)

vecs = fix_vecs(vecs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

O = pos_mat_anal(neigs, m, omega)

#print('O', O[:10,:10])
#pos_mat = np.zeros((neigs,neigs))
#O = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)
#L = L*1j
Ladj = np.conjugate(np.transpose(L))


x0 = 0.0
p0 = 1.0
sigma_x = 1.

coeff_wf, wf = coh_st_coeff(x0, p0, sigma_x, x_arr, vecs, neigs)
#coeff_wf[:] = 0.0
#coeff_wf[:1] = 1.0/np.sqrt(1.0)
print('coeff_wf:', np.sum(np.abs(coeff_wf)**2))

ip = 0.0 + 0j
ip = Krylov_complexity.krylov_complexity.compute_ip_wf(O, O, coeff_wf, ip)
O[:] = O[:] + 0.0j
O[:] = O[:]/np.sqrt(ip)

ncoeff = 50
aarr = np.zeros(ncoeff) + 0j
barr = np.zeros(ncoeff) + 0j
carr = np.zeros(ncoeff) + 0j
beta = 1.0 # Dummy argument
lamda = 0.5 # Dummy argument
ipkey = 'wf'

barr, carr, carr = Krylov_complexity.krylov_complexity.compute_bilanczos_coeffs(O, L, Ladj, \
                    aarr, barr, carr, beta, coeff_wf, vals, lamda, ipkey)

print('barr:', barr[:10])
print('carr:', carr[:10])
print('aarr:', aarr[:10])

barr[:] = 0.0
barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs_wf(O, L, barr, coeff_wf)
print('Lanczos barr:', barr[:10])

barr[:] = 0.0
barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
print('Lanczos barr (no wf):', barr[:10])
