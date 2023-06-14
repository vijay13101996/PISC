import numpy as np
from PISC.potentials import Tanimura_SB
from matplotlib import pyplot as plt
import os 
import time 
import ast
from PISC.utils.misc import find_OTOC_slope, find_minima, find_maxima
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
from PISC.dvr.dvr import DVR2D
from PISC.engine import OTOC_f_2D_omp_updated

path = os.path.dirname(os.path.abspath(__file__))	
start_time = time.time()
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
cb = 0.65#0.25*wb

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

	
pes = Tanimura_SB(D,alpha,m,mb,wb,VLL,VSL,cb)
		
T = 0.25#0.125
beta = 1/T
print('beta', beta)

potkey = 'Tanimura_SB_D_{}_alpha_{}_VLL_{}_VSL_{}_cb_{}'.format(D,alpha,VLL,VSL,cb)
Tkey = 'T_{}'.format(T)

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)	# Change ngridx!=ngridy

vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))

print('vals', vals[:10])

basis_N = 50
n_eigen = 10

	
if(1):
	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1

	t_arr = np.linspace(0.0,20.0,1000)#np.arange(0.0,100,0.01)#
	OTOC_arr = np.zeros_like(t_arr)+0j 

	reg ='Kubo'
	corr='xxC'#'qq1'#'xxC'
	
	if(reg=='Kubo'):
		CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,corr,'kubo',OTOC_arr)
	elif(reg=='Standard'):	
		CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr)
	elif(reg=='Symmetric'):
		CT = OTOC_f_2D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)

	fname = 'Quantum_{}_{}_{}_beta_{}_neigen_{}_basis_{}'.format(reg,corr,potkey,beta,n_eigen,basis_N)
	store_1D_plotdata(t_arr,CT,fname,'{}/Datafiles'.format(path))

	print('time', time.time() - start_time)
	plt.plot(t_arr,CT)
	

if(0):
	corr='qq1'#'xxC'
	for c in [0.0,0.25,0.45,0.55,0.65,0.75]:
		potkey = 'Tanimura_SB_D_{}_alpha_{}_VLL_{}_VSL_{}_cb_{}'.format(D,alpha,VLL,VSL,c)
		ext ='{}/Datafiles//Quantum_Kubo_{}_{}_beta_{}_neigen_{}_basis_{}.txt'.format(path,corr,potkey,beta,n_eigen,basis_N)
		data = np.loadtxt(ext,dtype='complex')
		tarr = data[:,0]
		carr = data[:,1]
		plt.plot(tarr,carr,label='c={}'.format(c))
		
		ind, maxima = find_maxima(carr)
		plt.scatter(tarr[ind], carr[ind])
		maxi = len(tarr[ind])
		print('maximas', maxi)	
	
		ind, minima = find_minima(carr)
		plt.scatter(tarr[ind], carr[ind])
		mini = len(tarr[ind])
		print('minimas', mini)	

		print('total', maxi+mini)

		#fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)	# Change ngridx!=ngridy
		#vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
		#vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))

		#plt.title('c={}'.format(c))
		#plt.imshow(DVR.eigenstate(vecs[:,8])**2,origin='lower')
		#plt.show()	

if(1):
	corr ='xxC'
	for n in [1,2,5,10]:
		potkey = 'Tanimura_SB_D_{}_alpha_{}_VLL_{}_VSL_{}_cb_{}'.format(D,alpha,VLL,VSL,cb)
		ext ='{}/Datafiles//Quantum_Kubo_{}_{}_beta_{}_neigen_{}_basis_{}.txt'.format(path,corr,potkey,beta,n,basis_N)
		data = np.loadtxt(ext,dtype='complex')
		tarr = data[:,0]
		carr = data[:,1]
		plt.plot(tarr,carr,label='n={}'.format(n))
		
		ind, maxima = find_maxima(carr)
		plt.scatter(tarr[ind], carr[ind])
		maxi = len(tarr[ind])
		print('maximas', maxi)	
	
		ind, minima = find_minima(carr)
		plt.scatter(tarr[ind], carr[ind])
		mini = len(tarr[ind])
		print('minimas', mini)	

		print('total', maxi+mini)


plt.legend()
plt.show()
