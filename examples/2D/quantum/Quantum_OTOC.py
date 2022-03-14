import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D
from PISC.engine import OTOC_f_2D_omp
from PISC.engine import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
import time 

"""
Compilation instructions for f2py:
Use the following command to compile the openmp parallelization 
enabled fortran OTOC code:
f2py -c --f90flags="-fopenmp" -m OTOC_f_2D_omp OTOC_fortran_omp.f90 -lgomp

Don't forget to set export OMP_NUM_THREADS=#thread_count in the .bashrc file!

"""

start_time = time.time()

if(0): #2D double well
	L = 7.0
	lbx = -L#-2*L
	ubx = L#2*L
	lby = -6*L#-L
	uby = 12*L#4*L
	m = 0.5#8.0
	ngrid = 100
	ngridx = ngrid
	ngridy = ngrid

	w = 0.1	
	D = 5.0
	alpha = (0.5*m*w**2/D)**0.5#1.95
	
	lamda = 0.8#4.0
	g = 0.02#4.0

	z = 1.0#1.0#0.0#2.3	

	x = np.linspace(lbx,ubx,ngridx+1)
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

	Tc = lamda*0.5/np.pi
	T_au = Tc#10.0 
	beta = 1.0/T_au 

	basis_N = 100#165
	n_eigen = 10#150

	print('T in au, potential, basis',T_au, potkey,basis_N )

	pes = quartic_bistable(alpha,D,lamda,g,z)

	print('pot',pes.potential_xy(0,0))
	fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	path = os.path.dirname(os.path.abspath(__file__))

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
	if(0): #Diagonalization
		vals,vecs = DVR.Diagonalize()

		store_arr(vecs,'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
		store_arr(vals,'{}_vals'.format(fname),'{}/Datafiles'.format(path))

	vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

	print('vals',vals[:20])
	print('expvals',np.exp(-beta*vals[:70]))
	#print('vecs',vecs[:200,10])	
	n=10
	#plt.contour(DVR.eigenstate(vecs[:,n]))
	#plt.show()

	x_arr = DVR.pos_mat(0)
	print('x_arr',x_arr)
	k_arr = np.arange(basis_N)
	m_arr = np.arange(basis_N)

	t_arr = np.linspace(0.0,10.0,1000)
	OTOC_arr = np.zeros_like(t_arr)

	if(1):
		# Be careful with the Kubo OTOC, when the energy spacing is quite closely packed!
		OTOC_arr = OTOC_f_2D_omp.position_matrix.compute_otoc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
		#OTOC_arr = OTOC_f_2D_omp.position_matrix.compute_kubo_otoc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
		#OTOC_arr = OTOC_f_2D_omp.position_matrix.compute_c_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n,m_arr,t_arr,OTOC_arr)
		plt.plot(t_arr,np.log(OTOC_arr))
		plt.show()
		
		path = os.path.dirname(os.path.abspath(__file__))
		fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,basis_N,n_eigen)
		store_1D_plotdata(t_arr,OTOC_arr,fname,'{}/Datafiles'.format(path))
		print('otoc', OTOC_arr[0])
		
		print('time',time.time()-start_time)
			
if(1):
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
	potkey = 'coupled_harmonic_w_{}_g_{}'.format(omega,g0)

	T_au = 0.5 
	beta = 1.0/T_au 

	basis_N = 40#165
	n_eigen = 30#150

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

	print('vals',vals[:100],vecs[:,10])
	
	#plt.imshow(DVR.eigenstate(vecs[:,0]))
	#plt.show()

	x_arr = DVR.pos_mat()
	k_arr = np.arange(basis_N)
	m_arr = np.arange(basis_N)

	t_arr = np.linspace(0.0,30.0,1000)
	OTOC_arr = np.zeros_like(t_arr)

	if(0):
		#OTOC_arr = OTOC_f_2D.position_matrix.compute_otoc_arr_t(vecs,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
		OTOC_arr = OTOC_f_2D_omp.position_matrix.compute_kubo_otoc_arr_t(vecs,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
		
		path = os.path.dirname(os.path.abspath(__file__))
		fname = 'Kubo_Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,basis_N,n_eigen)
		store_1D_plotdata(t_arr,OTOC_arr,fname,'{}/Datafiles'.format(path))
		print('otoc', OTOC_arr[0])
		
		print('time',time.time()-start_time)
		plt.plot(t_arr,np.log(OTOC_arr))
		plt.show()
		
	 
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
	
