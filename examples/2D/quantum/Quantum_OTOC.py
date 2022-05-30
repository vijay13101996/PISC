import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D
from PISC.engine import OTOC_f_2D_omp, OTOC_f_2D_omp_updated,OTOC_f_2D_omp_test
from matplotlib import pyplot as plt
import os 
import time 

"""
Compilation instructions for f2py:
Use the following command to compile the openmp parallelization 
enabled fortran OTOC code:
f2py3 -c --f90flags="-fopenmp" -m OTOC_f_2D_omp_updated OTOC_fortran_omp_updated.f90 -lgomp

Don't forget to set export OMP_NUM_THREADS=#thread_count in the .bashrc file!

"""

start_time = time.time()

if(1): #2D double well
	L = 10.0
	lbx = -8
	ubx = 8
	lby = -6#
	uby = 20.0#
	m = 0.5
	ngrid = 100
	ngridx = ngrid
	ngridy = ngrid

	w = 0.1	
	D = 10.0
	alpha = 0.255#0.81#0.255#0.41#0.175#1.165#0.255#
	
	lamda = 1.5#4.0
	g = 0.05#lamda**2/32#0.02#4.0

	z = 0.0#2.3	

	Tc = lamda*0.5/np.pi
	T_au = Tc#10.0 
	beta = 1.0/T_au 
	
	print('beta', beta)

	x = np.linspace(lbx,ubx,ngridx+1)
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

	pes = quartic_bistable(alpha,D,lamda,g,z)

	print('pot',pes.potential_xy(0,0))
	fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	path = os.path.dirname(os.path.abspath(__file__))

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
	n_eig_tot = 150
	print('potential',potkey)	
	if(0): #Diagonalization
		param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
		with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:	
			f.write('\n'+str(param_dict))#print(param_dict,file=f)
		
		vals,vecs = DVR.Diagonalize()#_Lanczos(n_eig_tot)

		store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
		store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))

	vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

	basis_N = 100#165
	n_eigen = 30#150

	print('vals',vals[:30])#vals[0],vals[5],vals[16],vals[50],vals[104])
	#print('expvals',np.exp(-beta*vals[:70]))
	#print('vecs',vecs[:200,10])	
	n=12
	M=1
	if(0):
		xg = np.linspace(lbx,ubx,ngridx)
		yg = np.linspace(lby,uby,ngridy)
		xgr,ygr = np.meshgrid(xg,yg)
		plt.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,5,0.25))
	plt.imshow(DVR.eigenstate(vecs[:,n])**2,origin='lower')
	plt.show()
	#plt.contour(DVR.eigenstate(vecs[:,n])**2)
	#plt.show()

	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1

	t_arr = np.linspace(0.0,50.0,1000)
	OTOC_arr = np.zeros_like(t_arr)+0j 
	b_arr = np.zeros_like(OTOC_arr)
	print('time',time.time()-start_time)
			
	if(1):
		# Be careful with the Kubo OTOC, when the energy spacing is quite closely packed!
		#OTOC_arr = OTOC_f_2D_omp.position_matrix.compute_otoc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
		#OTOC_arr = OTOC_f_2D_omp.position_matrix.compute_kubo_otoc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
		#OTOC_arr = OTOC_f_2D_omp_test.position_matrix.compute_c_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n,m_arr,t_arr,OTOC_arr)
		
		#G1 = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xG1',OTOC_arr)
		#G2 = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xG2',OTOC_arr)
		#F = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xxF',OTOC_arr)
		C = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)
		#xp2 = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'qp2',OTOC_arr) 
		#xp1 = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'qp1',OTOC_arr) 
		
		#CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','kubo',OTOC_arr)
		#xp1T = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qp2','stan',OTOC_arr)
			
		if(0):
			for M in range(10,15):
				b_temp = np.zeros_like(OTOC_arr) + 0j
				b_temp = OTOC_f_2D_omp.position_matrix.compute_b_mat_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n,M,t_arr,b_temp)
				b_arr+=abs(b_temp)**2
				#plt.plot(t_arr,np.real(b_temp),label=M)
				#plt.plot(t_arr,np.imag(b_temp),label=M)
				plt.plot(t_arr,abs(b_temp)**2,label=M)
			plt.legend()
			plt.show()	
	
		if(0):	
			ist = 65#119#70#80#100 
			iend = 150#180#110#140#173

			t_trunc = t_arr[ist:iend]
			OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
			slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
			print('slope',slope,2*np.pi/beta)

			a = -OTOC_arr
			x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
			#print('min max',t_arr[124],t_arr[281])

			fig,ax = plt.subplots()
				
			#plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
			#plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
			#plt.plot(t_arr,slope*t_arr+ic,'--',color='k')

		#plt.plot(t_arr, G1)
		#plt.plot(t_arr, G2)
		#plt.plot(t_arr,G1+G2)	
		#plt.plot(t_arr,xp1T,color='g')
		#plt.plot(t_arr,xp1,color='k')
		plt.title('{}'.format(potkey))
		plt.plot(t_arr,np.log(abs(C)))
		#plt.plot(t_arr,G1+G2-2*np.real(F))	
		#plt.plot(t_arr,b_arr,color='k')
		plt.show()
		
		path = os.path.dirname(os.path.abspath(__file__))
		fname = 'Quantum_mc_n_1_OTOC_{}_basis_{}'.format(potkey,basis_N)
		store_1D_plotdata(t_arr,C,fname,'{}/Datafiles'.format(path))

		#fname = 'Quantum_mc_n_1_qqTCF_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,basis_N,n_eigen)
		#store_1D_plotdata(t_arr,xp2,fname,'{}/Datafiles'.format(path))
		#print('otoc', OTOC_arr[0])
		
		print('time',time.time()-start_time)
			
if(0):
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
		