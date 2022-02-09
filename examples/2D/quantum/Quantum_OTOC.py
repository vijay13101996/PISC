import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D
from PISC.engine import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
import time 
start_time = time.time()

L = 10.0
lbx = -L
ubx = L
lby = -L
uby = L
m = 0.5
ngrid = 100
ngridx = ngrid
ngridy = ngrid
omega = 1.0
g0 = 0.1#3e-3#1/100.0
x = np.linspace(lbx,ubx,ngridx+1)
potkey = 'coupled_harmonic'#_w_{}_g_{}'.format(omega,g0)

T_au = 4.0 
beta = 1.0/T_au 

basis_N = 120
n_eigen = 100

print('T in au, potential, basis',T_au, potkey,basis_N )

pes = coupled_harmonic(omega,g0)

fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
path = os.path.dirname(os.path.abspath(__file__))

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
if(1): #Diagonalization
	vals,vecs = DVR.Diagonalize()

	store_arr(vecs,'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
	store_arr(vals,'{}_vals'.format(fname),'{}/Datafiles'.format(path))

vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

#print('vals',vals[:150])

x_arr = DVR.pos_mat()
k_arr = np.arange(basis_N)
m_arr = np.arange(basis_N)

t_arr = np.linspace(0.0,15.0,1000)
OTOC_arr = np.zeros_like(t_arr)

if(1):
	OTOC_arr = OTOC_f_2D.position_matrix.compute_otoc_arr_t(vecs,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 

	path = os.path.dirname(os.path.abspath(__file__))
	fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,basis_N,n_eigen)
	store_1D_plotdata(t_arr,OTOC_arr,fname,'{}/Datafiles'.format(path))
	print('otoc', OTOC_arr[0])

	#plt.plot(t_arr,np.log(OTOC_arr))
	#plt.show()
	
print('time',time.time()-start_time) 
#-------------------------------------------
if(0):
	for i in [5]:#range(5):
		OTOC_arr = np.zeros_like(t_arr)
		OTOC_arr = OTOC_f_2D.position_matrix.compute_c_mc_arr_t(vecs,x_arr,DVR.dx,DVR.dy,k_arr,vals,i-1,m_arr,t_arr,OTOC_arr)
		plt.plot(t_arr,OTOC_arr)
		plt.show()
		#bnm = 0.0
		#bnm = OTOC_f_2D.position_matrix.b_matrix_elts(vecs,x_arr,DVR.dx,DVR.dy,k_arr,vals,i,i,0.0,bnm)
		#print('OTOC at t=0 for n={}'.format(i),bnm)
	
