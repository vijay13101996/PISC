import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.husimi.Husimi import Husimi_2D,Husimi_1D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
from PISC.engine import OTOC_f_1D
from PISC.engine import OTOC_f_2D_omp_updated
from matplotlib import pyplot as plt
import os 
import time 
import ast

def load_data(path,alpha,D,lamda,g,z,k=1.0):
	if(k!=1.0):
		potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}_k_{}'.format(alpha,D,lamda,g,z,k)
	else:
		potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	
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
	pes = quartic_bistable(alpha,D,lamda,g,z,k)

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
	fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)	# Change ngridx!=ngridy

	vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))
	
	#print('vals',vals[:20])

	return vals,vecs,DVR

def plot_wf(path,alpha,D,lamda,g,z,n):
	vals,vecs,DVR = load_data(path,alpha,D,lamda,g,z)
	norm = np.sum(vecs[:,n]**2*DVR.dx*DVR.dy)
	print('E_n',vals[n],norm)
	plt.title(r'$\alpha={}$'.format(alpha))
	plt.imshow(DVR.eigenstate(vecs[:,n])**2,extent=[DVR.lbx,DVR.ubx,DVR.lby,DVR.uby],origin='lower')
	plt.show()
		
def plot_bnm(path,alpha,D,lamda,g,z,n,M,ax,lbltxt,basis_N=30):
	vals,vecs,DVR = load_data(path,alpha,D,lamda,g,z)
	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1

	t_arr = np.linspace(0.0,5.0,1000)
	OTOC_arr = np.zeros_like(t_arr)+0j 
	
	print('E_n',vals[n])
	
	bnm =OTOC_f_2D_omp_updated.otoc_tools.quadop_matrix_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,M+1,t_arr,1,'cm',OTOC_arr)
	
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	fname = 'Quantum_bnm_n_{}_m_{}_{}'.format(n,M,potkey)
	store_1D_plotdata(t_arr,abs(bnm)**2,fname,'{}/Datafiles'.format(path))

	ax.plot(t_arr,(abs(bnm)**2),label=lbltxt)	

def plot_Cn(path,alpha,D,lamda,g,z,n,ax,lbltxt,basis_N=100):
	vals,vecs,DVR = load_data(path,alpha,D,lamda,g,z)
	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1

	t_arr = np.linspace(0.0,5.0,1000)
	OTOC_arr = np.zeros_like(t_arr)+0j 

	C = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)
	ax.plot(t_arr, np.log(abs(C)), label=lbltxt)

def plot_thermal_OTOC(path,alpha,D,lamda,g,z,beta,ax,lbltxt,basis_N=50,n_eigen=20,k=1.0,reg='Kubo',t_arr=None):
	vals,vecs,DVR = load_data(path,alpha,D,lamda,g,z,k)
	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1

	if(t_arr is None):
		t_arr = np.linspace(0.0,5.0,1000)
	OTOC_arr = np.zeros_like(t_arr)+0j 
	
	if(reg=='Kubo'):
		CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','kubo',OTOC_arr)
	elif(reg=='Standard'):	
		CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr)
	elif(reg=='Symmetric'):
		CT = OTOC_f_2D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	fname = 'Quantum_{}_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(reg,potkey,beta,n_eigen,basis_N)
	store_1D_plotdata(t_arr,CT,fname,'{}/Datafiles'.format(path))
	#ax.plot(t_arr,np.log(abs(CT)), label=lbltxt)

def plot_thermal_TCF(path,alpha,D,lamda,g,z,beta,ax,lbltxt,basis_N=50,n_eigen=20,k=1.0,reg='Kubo',tcftype='qq1',t_arr=None):
	vals,vecs,DVR = load_data(path,alpha,D,lamda,g,z,k)
	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1

	if(t_arr is None):
		t_arr = np.linspace(0.0,20.0,1000)
	OTOC_arr = np.zeros_like(t_arr)+0j 
	
	if(reg=='Kubo'):
		CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,tcftype,'kubo',OTOC_arr)
	elif(reg=='Standard'):	
		CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,tcftype,'stan',OTOC_arr)
	elif(reg=='Symmetric'):
		CT = OTOC_f_2D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,tcftype,0.5,OTOC_arr)
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	fname = 'Quantum_{}_{}_TCF_{}_beta_{}_neigen_{}_basis_{}'.format(reg,tcftype[:2],potkey,beta,n_eigen,basis_N)
	print('fname',fname)
	store_1D_plotdata(t_arr,CT,fname,'{}/Datafiles'.format(path))
	ax.plot(t_arr,np.log(abs(CT)), label=lbltxt)

def cplot_stan(path,alpha,D,lamda,g,z,beta,ax,lbltxt,basis_N=50,n_eigen=20):
	vals,vecs,DVR = load_data(path,alpha,D,lamda,g,z)
	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1

	t_arr = np.linspace(0.0,5.0,1000)
	OTOC_arr = np.zeros_like(t_arr)+0j 
	b_arr = np.zeros_like(t_arr) + 0j	

	Z=0.0	
	for i in range(basis_N):
		Z+= np.exp(-beta*vals[i])

	for n in range(n_eigen):	
		for M in range(basis_N):
			bnm =OTOC_f_2D_omp_updated.otoc_tools.quadop_matrix_arr_t\
				(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,M+1,t_arr,1,'cm',b_arr)
			coeff = (1/Z)*np.exp(-beta*vals[n])
			if((bnm>1e-1).any()):
				print('n,m',n,M)
			OTOC_arr+=coeff*abs(bnm)**2
	
	ax.plot(t_arr,np.log(abs(OTOC_arr)), label=lbltxt)

def cplot_kubo(path,alpha,D,lamda,g,z,beta,ax,lbltxt,basis_N=50,n_eigen=20):
	vals,vecs,DVR = load_data(path,alpha,D,lamda,g,z)
	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1

	t_arr = np.linspace(0.0,5.0,1000)
	OTOC_arr = np.zeros_like(t_arr)+0j 
	b_arr = np.zeros_like(t_arr) + 0j

	Z=0.0
	for i in range(basis_N):
		Z+= np.exp(-beta*vals[i])

	for n in range(n_eigen):	
		for M in range(basis_N):
			bnm =OTOC_f_2D_omp_updated.otoc_tools.quadop_matrix_arr_t\
				(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,M+1,t_arr,1,'cm',b_arr)
			if(n!=M):
				coeff = (1/(Z*beta))*((np.exp(-beta*vals[n]) - np.exp(-beta*vals[M]))\
								/(vals[M]-vals[n]))
			else:
				coeff = (1/Z)*np.exp(-beta*vals[M])
			if((bnm>1e-1).any()):
				print('n,m',n,M)
			OTOC_arr+=coeff*abs(bnm)**2

	ax.plot(t_arr,np.log(abs(OTOC_arr)), label=lbltxt)


