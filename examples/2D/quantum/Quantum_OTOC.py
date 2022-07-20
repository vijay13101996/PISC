import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.potentials.Heller_Davis import heller_davis
from PISC.potentials.harmonic_2D import Harmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D
from PISC.engine import OTOC_f_2D_omp_updated
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
	lbx = -6#-7
	ubx = 6#7
	lby = -2.5#
	uby = 5.0#
	m = 0.5
	ngrid = 100
	ngridx = ngrid
	ngridy = ngrid

	w = 0.1	
	D = 9.375 # 0.569, 
	alpha = 0.52#0.37#0.2#0.52#0.86#2.64#0.363#0.81#0.255#0.41#0.175#1.165#0.255#
	
	lamda = 2.0#4.0
	g = 0.08#lamda**2/32#0.02#4.0

	z = 1.5#2.3	

	Tc = lamda*0.5/np.pi
	times = 1.0
	T_au = times*Tc#10.0 
	beta = 1.0/T_au 
	
	print('beta', beta)

	x = np.linspace(lbx,ubx,ngridx+1)
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

	pes = quartic_bistable(alpha,D,lamda,g,z)

	print('pot',pes.potential_xy(0,0))
	fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	path = os.path.dirname(os.path.abspath(__file__))

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
	n_eig_tot = 200
	print('potential',potkey)	
	if(1): #Diagonalization
		param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
		with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:	
			f.write('\n'+str(param_dict))#print(param_dict,file=f)
		
		vals,vecs = DVR.Diagonalize()#_Lanczos(n_eig_tot)

		store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
		store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))

	vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

	basis_N = 140#165
	n_eigen = 50#150

	#print('vals',vals[:30])#vals[0],vals[5],vals[16],vals[50],vals[104])
	#print('expvals',np.exp(-beta*vals[:70]))
	#print('vecs',vecs[:200,10])	
	n=8#15
	M=1
	print('vals[n]', vals[n])
	if(1):
		xg = np.linspace(lbx,ubx,ngridx)
		yg = np.linspace(lby,uby,ngridy)
		xgr,ygr = np.meshgrid(xg,yg)
		#plt.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,10,0.5))
		#fig, ax = plt.subplots(1,3)
		#for i in range(3):
		#	ax[i].contour(DVR.eigenstate(vecs[:,i+2])**2,levels=np.arange(0,0.1,0.005))
		#plt.show()
	plt.imshow(DVR.eigenstate(vecs[:,n])**2,origin='lower')
	plt.show()
	#plt.contour(DVR.eigenstate(vecs[:,n])**2,levels=np.arange(0,0.1,0.005))
	#plt.show()

	x_arr = DVR.pos_mat(0)
	k_arr = np.arange(basis_N) +1
	m_arr = np.arange(basis_N) +1

	t_arr = np.linspace(0.0,5.0,1000)
	OTOC_arr = np.zeros_like(t_arr)+0j 
	b_arr = np.zeros_like(OTOC_arr)
	print('time',time.time()-start_time)
			
	if(0):
		# Be careful with the Kubo OTOC, when the energy spacing is quite closely packed!
		#OTOC_arr = OTOC_f_2D_omp.position_matrix.compute_otoc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
		#OTOC_arr = OTOC_f_2D_omp.position_matrix.compute_kubo_otoc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
		#OTOC_arr = OTOC_f_2D_omp_test.position_matrix.compute_c_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n,m_arr,t_arr,OTOC_arr)
		
		#G1 = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xG1',OTOC_arr)
		#G2 = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xG2',OTOC_arr)
		#F = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xxF',OTOC_arr)
		#C = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)
		#xp2 = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'qp2',OTOC_arr) 
		#xp1 = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'qp1',OTOC_arr) 
		#c0 = 0.0j
		#c0 = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_elts(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,0.0,'xxC',c0)
		#print('C(0)', c0)	
		#CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','kubo',OTOC_arr)
		#xp1T = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qp2','stan',OTOC_arr)
			
		if(1):
			bnm_arr=np.zeros_like(OTOC_arr)
			OTOC_arr[:] = 0.0
			lda = 0.5
			Z = 0.0
			coefftot = 0.0
			for n in [0]:#range(2):
				Z+= np.exp(-beta*vals[n])		
				for M in range(20,25):
					bnm =OTOC_f_2D_omp_updated.otoc_tools.quadop_matrix_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,M+1,t_arr,1,'cm',OTOC_arr)
					coeff = 0.0
					if(n!=M):
						coeff =(1/beta)*((np.exp(-beta*vals[n]) - np.exp(-beta*vals[M]))/(vals[M]-vals[n]))
					else:
						coeff = np.exp(-beta*(vals[n]))
					coefftot+=coeff
					if(1):#coeff>=1e-4):
						coeffpercent = coeff*100/0.001876
						print('coeff,Z',n,M,coeff,np.exp(-beta*vals[n])*np.exp(-lda*beta*(vals[M]-vals[n]))  )
						plt.plot(t_arr,abs(bnm)**2,label='n,M, % ={},{},{}'.format(n,M,np.around(coeffpercent,2)))
						#print('coeff_sym', np.exp(-beta*vals[n])*np.exp(-lda*beta*(vals[M]-vals[n])))
					bnm_arr+=coeff*abs(bnm)**2
			bnm_arr/=Z
			print('Z',Z,coefftot)
			#plt.plot(t_arr,np.log(bnm_arr),color='m',linewidth=3)
			plt.legend()

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
		#plt.title('{}'.format(potkey))
		#plt.plot(t_arr,np.log(abs(CT)))
		#plt.plot(t_arr,G1+G2-2*np.real(F))	
		#plt.plot(t_arr,b_arr,color='k')
		plt.show()
		
		path = os.path.dirname(os.path.abspath(__file__))
		fname = 'Quantum_OTOC_{}_neigen_{}_basis_{}'.format(potkey,n_eigen,basis_N)
		#store_1D_plotdata(t_arr,CT,fname,'{}/Datafiles'.format(path))

		#fname = 'Quantum_mc_n_1_qqTCF_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,basis_N,n_eigen)
		#store_1D_plotdata(t_arr,xp2,fname,'{}/Datafiles'.format(path))
		#print('otoc', OTOC_arr[0])
		
		print('time',time.time()-start_time)
			
if(0):
	L = 4.0
	lbx = -L
	ubx = L
	lby = -L
	uby = L
	m = 1.0
	hbar = 1.0#0.25
	ngrid = 100
	ngridx = ngrid
	ngridy = ngrid
	omega = 2.0**0.5#1.0
	g0 = 0.1
	x = np.linspace(lbx,ubx,ngridx+1)
	potkey = 'coupled_harmonic_w_{}_g_{}_m_{}_hbar_{}'.format(omega,g0,m,hbar)

	T_au = 0.5 
	beta = 1.0/T_au 

	basis_N = 40#165
	n_eigen = 30#150

	print('T in au, potential, basis',T_au, potkey,basis_N )

	pes = coupled_harmonic(omega,g0)

	fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	path = os.path.dirname(os.path.abspath(__file__))

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy,hbar=hbar)	
	n_eig_tot=200
	if(1): #Diagonalization
		param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot,'hbar':hbar}
		with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:	
			f.write('\n'+str(param_dict))#print(param_dict,file=f)
		
		vals,vecs = DVR.Diagonalize(neig_total=n_eig_tot)

		store_arr(vecs,'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
		store_arr(vals,'{}_vals'.format(fname),'{}/Datafiles'.format(path))

	vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

	print('vals',vals[1700],vecs[:,10])
	
	#plt.imshow(DVR.eigenstate(vecs[:,10]))
	#plt.show()

	x_arr = DVR.pos_mat(0)
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
		

if(0): #Heller Davis
	L = 10.0
	lbx = -7
	ubx = 7
	lby = -9
	uby = 9
	m = 1.0
	ngrid = 100
	ngridx = ngrid
	ngridy = ngrid

	ws= 1.0
	wu= 1.1
	lamda = -0.11
	
	T_au = 1.0 
	beta = 1.0/T_au 
	
	print('beta', beta)

	x = np.linspace(lbx,ubx,ngridx+1)
	potkey = 'Heller_Davis_ws_{}_wu_{}_lamda_{}'.format(ws,wu,lamda)

	pes = heller_davis(ws,wu,lamda)

	print('pot',pes.potential_xy(0,0))
	fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	path = os.path.dirname(os.path.abspath(__file__))

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
	n_eig_tot = 200
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

	basis_N = 140#165
	n_eigen = 50#150

	#print('vals',vals[:30])#vals[0],vals[5],vals[16],vals[50],vals[104])
	#print('expvals',np.exp(-beta*vals[:70]))
	#print('vecs',vecs[:200,10])	
	n=93
	M=1
	print('vals[n]', vals[n])
	if(1):
		xg = np.linspace(lbx,ubx,ngridx)
		yg = np.linspace(lby,uby,ngridy)
		xgr,ygr = np.meshgrid(xg,yg)
		#plt.contour(xgr,ygr,pes.potential_xy(xgr,ygr),levels=np.arange(-20,20,1))
	plt.imshow(DVR.eigenstate(vecs[:,n])**2,origin='lower')
	plt.show()
	#plt.contour(DVR.eigenstate(vecs[:,n])**2)
	#plt.show()

if(0):
	L = 5.0
	lbx = -L
	ubx = L
	lby = -L
	uby = L
	m = 0.5
	ngrid = 100
	ngridx = ngrid
	ngridy = ngrid

	lamda = 2.0
	
	T_au = 1.0 
	beta = 1.0/T_au 
	
	print('beta', beta)

	x = np.linspace(lbx,ubx,ngridx+1)
	potkey = 'Harmonic_2D_lamda_{}'.format(lamda)

	pes = Harmonic(lamda)

	print('pot',pes.potential_xy(0,0))
	fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	path = os.path.dirname(os.path.abspath(__file__))

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
	n_eig_tot = 200
	print('potential',potkey)	
	if(1): #Diagonalization
		param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
		with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:	
			f.write('\n'+str(param_dict))#print(param_dict,file=f)
		
		vals,vecs = DVR.Diagonalize()#_Lanczos(n_eig_tot)

		store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
		store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))

	vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

	print('vals',vals[:10])
