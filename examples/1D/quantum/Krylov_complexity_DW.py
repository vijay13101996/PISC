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

m=0.5
lamda = 2.0
g = 0.02

pes = double_well(lamda,g)
Tc = 0.5*lamda/np.pi

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 50
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
O = (pos_mat)

mom_mat = np.zeros((neigs,neigs))
mom_mat = Krylov_complexity.krylov_complexity.compute_mom_matrix(vecs, vals, x_arr, m, dx, dx, mom_mat)
P = mom_mat

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

times_arr = np.arange(0.6,8.05,0.2)#[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
mun_arr = []
bnarr = []
nmoments = 8
ncoeff = 10

for times in times_arr:
    T_au = times*Tc
    Tkey = 'T_{}Tc'.format(times)

    beta = 1.0/T_au 

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, moments)
    even_moments = moments[0::2]

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')
    bnarr.append(barr)

    #print('moments',np.around(moments[0::2],5))
    mun_arr.append(even_moments)

mun_arr = np.array(mun_arr)
bnarr = np.array(bnarr)
print('mun_arr',mun_arr.shape)
print('bnarr',bnarr.shape)

for i in [0,2,4,6,8]:#range(0,ncoeff,2):
    plt.scatter(times_arr,bnarr[:,i]-bnarr[0,i],label='n={}'.format(i))
    #plt.scatter(times_arr,mun_arr[:,i],label='n={}'.format(i))
    plt.legend()
    plt.show()

if(0):
    for i in [1,2,3]:#range(1):#nmoments//2+1):
        # Fit times_arr vs mun_arr[:,i] to a line
        p = np.polyfit(times_arr,mun_arr[:,i],1)
        print('p',p)
        plt.plot(times_arr,p[0]*times_arr+p[1],label='n={}'.format(2*i))
        plt.scatter(times_arr,mun_arr[:,i],label='n={}'.format(2*i))
    #plt.scatter(times_arr,mun_arr)
    plt.legend()
    plt.show()
    

#ncoeffs = 20
#barr = np.zeros(ncoeffs)
#barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')

#b0 = np.sqrt(1/(2*m*w*np.sinh(0.5*beta*w)))

#print('barr',barr,b0)

#plt.scatter(np.arange(ncoeffs),barr)
#plt.show()


