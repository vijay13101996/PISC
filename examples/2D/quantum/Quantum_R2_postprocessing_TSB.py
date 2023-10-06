import numpy as np
from PISC.potentials import Tanimura_SB
from matplotlib import pyplot as plt
import os 
import time 
import ast
from PISC.utils.misc import find_OTOC_slope
from PISC.utils.readwrite import store_1D_plotdata, store_2D_imagedata_column, read_arr
from PISC.utils.plottools import plot_1D
from PISC.dvr.dvr import DVR2D
from PISC.engine import OTOC_f_2D_omp_updated

start_time = time.time()

path = os.path.dirname(os.path.abspath(__file__))	

m = 1.0
mb= 1.0
delta_anh = 0.1
w_10 = 1.0
wb = w_10
wc = w_10 + delta_anh
alpha = (m*delta_anh)**0.5
D = m*wc**2/(2*alpha**2)

VLL = -0.75*wb
VSL = 0.75*wb
cb = 0.25#0.25*wb

pes = Tanimura_SB(D,alpha,m,mb,wb,VLL,VSL,cb)
	
potkey = 'Tanimura_SB_D_{}_alpha_{}_VLL_{}_VSL_{}_cb_{}'.format(D,alpha,VLL,VSL,cb)

with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey)) as f:	
		for line in f:
			pass
		param_dict = ast.literal_eval(line)

lbx = param_dict['lbx']
ubx = param_dict['ubx']
lby = param_dict['lby']
uby = param_dict['uby']
ngridx = param_dict['ngridx']
ngridy = param_dict['ngridy']
m = param_dict['m']

	
T = 0.125#0.125
beta = 1/T
print('beta', beta)


Tkey = 'T_{}'.format(T)

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)	# Change ngridx!=ngridy

print('dx, dy', DVR.dx, DVR.dy)

vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))

print('vals', vals[:10])

basis_N = 50
n_eigen = 30

corrkey = 'all'	
if(1):
	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1
	
	t0 = 0.0
	ngrid=100
	dt = 50/ngrid
	t1_arr = np.arange(-25.0,25.0,dt)
	t2_arr = np.arange(-25.0,25.0,dt)

	R2_corr_arr = np.zeros((len(t1_arr),len(t2_arr))) + 0j 

	corr='R2'#'xxC'
	
	R2_corr_arr = OTOC_f_2D_omp_updated.otoc_tools.r2_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,\
					vals,m_arr,t2_arr,t1_arr,t0,beta,n_eigen,corrkey,R2_corr_arr.T)
	
	t1,t2 = np.meshgrid(t1_arr, t2_arr)
	fname = 'Quantum_R2_{}_{}_{}_basis_{}_n_eigen_{}_ngrid_{}'.format(corrkey,potkey,Tkey,basis_N,n_eigen,ngrid)	
	store_2D_imagedata_column(np.real(t1),np.real(t2),np.real(R2_corr_arr),fname,'{}/Datafiles'.format(path),extcol = np.imag(R2_corr_arr))


	print('time', time.time() - start_time)
	plt.imshow(np.real(R2_corr_arr).T)
	plt.colorbar()
	#plt.plot(t_arr,CT)
	plt.show()	


