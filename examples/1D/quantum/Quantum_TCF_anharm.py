import numpy as np
from PISC.dvr.dvr import DVR1D
#from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials import double_well, morse, quartic, harmonic1D, mildly_anharmonic
from PISC.potentials.triple_well_potential import triple_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D_omp_updated
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from PISC.engine import Krylov_complexity

ngrid = 600

L = 12
lb = -L
ub = L

m=1.0
a=0.0
b=1.0
omega=0.0
n_anharm=4


if(1):
    def potential(x):
        if(x<-1 or x>1):
            return 1e12
        else:
            #print('x',x)
            return 0.0#x**4

neigs = 200

T_au = 3.
potkey = 'quartic_a_{}'.format(a)
Tkey = 'T_{}'.format(T_au)	

beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

t_arr = np.linspace(0,10.0,100)

basis_N = 50
n_eigen = 30

ncoeff = 40

narr = np.arange(ncoeff) 
def TCF(DVR, m, beta, basis_N, n_eigen, corrkey):
    vals,vecs = DVR.Diagonalize(neig_total=neigs) 
    x_arr = DVR.grid[1:DVR.ngrid]
    dx = DVR.dx
    
    k_arr = np.arange(basis_N) +1
    m_arr = np.arange(basis_N) +1

    C_arr = np.zeros_like(t_arr) + 0j
    C_arr = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs, m, x_arr, dx, dx, k_arr, vals, m_arr, t_arr, beta,
                                                                n_eigen, corrkey, 0.5, C_arr)
    return C_arr

def Krylov(DVR, beta):
    vals,vecs = DVR.Diagonalize(neig_total=neigs) 
    x_arr = DVR.grid[1:DVR.ngrid]
    dx = DVR.dx
    
    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
    O = pos_mat

    L = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,L)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff) 
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    return barr

fig, ax = plt.subplots(2,1)

if(1):
    for n_anharm,L in zip([6,8,10,12,14],[8,6,4,3,2]):
        lb= -L
        ub= L
        pes = mildly_anharmonic(m,a,b,omega,n=n_anharm)

        DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
        C_arr = TCF(DVR, m, beta, basis_N, n_eigen, 'xxC')
        ax[0].plot(t_arr, np.log(C_arr.real), label='n={}'.format(n_anharm))

        barr = Krylov(DVR, beta)
        ax[1].plot(narr, barr.real, label='n={}'.format(n_anharm))
        print('n={}:', n_anharm)

DVR = DVR1D(ngrid,-1,1,m,potential)
C_arr = TCF(DVR, m, beta, basis_N, n_eigen, 'xxC')
ax[0].plot(t_arr, np.log(C_arr.real), label='1DB')

barr = Krylov(DVR, beta)
ax[1].plot(narr, barr.real, label='n={}'.format('1DB'))
ax[1].plot(narr, np.pi*narr/beta,lw=2)

ax[0].set_xlabel('t')
ax[0].set_ylabel('C(t)')

ax[0].legend()
ax[1].legend()
plt.show()

exit()
