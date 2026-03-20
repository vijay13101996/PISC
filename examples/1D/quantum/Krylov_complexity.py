import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
import scipy
from matplotlib import pyplot as plt

ngrid = 400

L = 10
lb = -L
ub = L
m = 1.0

if(0):
    w = 2.0
    pes = harmonic1D(m,w)
    potkey = 'harmonic_w_{}'.format(w)

if(1):
    pes = quartic(1.0)
    potkey = 'quartic'

if(0):
    m=0.5
    lamda = 2.0
    g = 0.005
    
    pes = double_well(lamda,g)
    Tc = 0.5*lamda/np.pi
    
    potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

    times = 5.0
    T_au = times*Tc
    Tkey = 'T_{}Tc'.format(times)

if(0):
    pes = mildly_anharmonic(1.0,0.0,0.1)
    potkey = 'mildly_anharmonic'

T_au = 1.
#Tkey = 'T_{}'.format(T_au)

beta = 1.0/T_au 

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 50
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]
#print('x_arr',x_arr[0])

#plt.plot(x_arr,vecs[:,-1])
#plt.show()

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
O = (pos_mat)
#print('pos_mat',np.around(np.sqrt(2)*pos_mat[0:5,0:5],2),vals[0:5])

mom_mat = np.zeros((neigs,neigs))
mom_mat = Krylov_complexity.krylov_complexity.compute_mom_matrix(vecs, vals, x_arr, m, dx, dx, mom_mat)
P = mom_mat

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)
#print('liou_mat',liou_mat[0:5,0:5],vals[0:5])

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

#print('LO', 1j*np.around(np.real(LO)[0:5,0:5],2), '\n\n P', np.around(P[0:5,0:5],2)) 

nmoments = 10
moments = np.zeros(nmoments+1)
moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, moments)
print('moments',np.around(moments[0::2],5))

#ncoeffs = 20
#barr = np.zeros(ncoeffs)
#barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')

#b0 = np.sqrt(1/(2*m*w*np.sinh(0.5*beta*w)))

#print('barr',barr,b0)

#plt.scatter(np.arange(ncoeffs),barr)
#plt.show()

if(0): #Junk 
    eignum = 1
    vec_num = vecs[:,eignum]

    # Analytic solution for the harmonic oscillator, Hermite polynomials for eig_num
    vec_analytic = np.exp(-0.5*m*w*x_arr**2)*scipy.special.eval_hermite(eignum,np.sqrt(m*w)*x_arr)

    norm_analytic = np.sum(vec_analytic**2)*dx
    norm_num = np.sum(vec_num**2)*dx
    vec_num = vec_num/np.sqrt(norm_num)
    vec_analytic = vec_analytic/np.sqrt(norm_analytic)

    print('norm_analytic,norm_num',norm_analytic,norm_num)

    print('vecs',vecs.shape,x_arr.shape,vec_analytic.shape)

    plt.plot(x_arr,vec_num,c='k')
    plt.plot(x_arr,vec_analytic,c='r')
    plt.show()



