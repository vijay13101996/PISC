import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from PISC.engine import OTOC_f_1D_omp_updated
import os 

ngrid = 1000

L=1 #4*np.sqrt(1/(4+np.pi))#10
lb=0
ub=L

lbc=-L
ubc=L

m=0.5

print('L',L)

potkey = '1D_Box_m_{}_L_{}'.format(m,np.around(L,2))

anal = True

neigs = 200

T_au = 0.01
Tkey = 'T_{}'.format(T_au)
beta = 1.0/T_au 
print('T_au',T_au,'beta',beta)

basis_N = 100
n_eigen = 50

nmoments = 60
ncoeff = 50
#----------------------------------------------------------------------

def potential(x):
    if(x<lbc or x>ubc):
        return 1e12
    else:
        #print('x',x)
        return 0.0#x**4

DVR = DVR1D(ngrid, lb, ub,m, potential)
x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

#----------------------------------------------------------------------

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
vecs_anal = np.zeros((neigs,ngrid))
for i in range(neigs):
    vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:]/L)

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)

#----------------------------------------------------------------------
print('Using analytical pos_mat, vals')
vecs = vecs_anal
vals = vals_anal
O = O_anal

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

mun_arr = []
mu0_harm_arr = []
bnarr = []
    
fig, ax = plt.subplots(2,1)

if(0):
    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, moments)
    even_moments = moments[0::2]

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')
    bnarr.append(barr)

    mun_arr.append(even_moments)

    mun_arr = np.array(mun_arr)
    bnarr = np.array(bnarr)
    print('mun_arr',mun_arr.shape)
    print('bnarr',bnarr.shape)


    ax[0].scatter(np.arange(ncoeff),bnarr[:200],label='T={}'.format(T_au),s=3)
    ax[0].set_xlabel(r'$n$')
    ax[0].set_ylabel(r'$b_n$')
#plt.legend()    
#plt.show()

#----------------------------------------------------------------------

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0,100.0,4000)
C_arr = np.zeros_like(t_arr) +0j

path = os.path.dirname(os.path.abspath(__file__))

corrkey = 'qq_TCF'#'OTOC'#'qp_TCF'
enskey = 'Symmetrized'#'mc'#'Kubo'

corrcode = {'OTOC':'xxC','qq_TCF':'qq1','qp_TCF':'qp1'}
enscode = {'Kubo':'kubo','Standard':'stan'}	

if(enskey == 'Symmetrized'):
    C_arr = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,corrcode[corrkey],0.5,C_arr)
else:
    C_arr = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,corrcode[corrkey],enscode[enskey],C_arr) 
fname = 'Quantum_{}_{}_{}_{}_basis_{}_n_eigen_{}'.format(enskey,corrkey,potkey,Tkey,basis_N,n_eigen)	
print('fname',fname)	

ax[1].plot(t_arr,np.real(C_arr),label='T={}'.format(T_au))
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$C(t)$')
plt.legend()
plt.show()
