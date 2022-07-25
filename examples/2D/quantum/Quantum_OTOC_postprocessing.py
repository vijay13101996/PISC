import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.husimi.Husimi import Husimi_2D,Husimi_1D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
from PISC.engine import OTOC_f_1D
from PISC.engine import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
import time 

if(0): # Coupled harmonic
	L = 10.0
	lbx = -L
	ubx = L
	lby = -L
	uby = L
	m = 1.0#0.5
	ngrid = 100
	ngridx = ngrid
	ngridy = ngrid
	omega = 2.0**0.5#0.5
	g0 = 0.1#3e-3#1/100.0
	x = np.linspace(lbx,ubx,ngridx+1)
	potkey = 'coupled_harmonic_w_{}_g_{}'.format(omega,g0)
	pes = coupled_harmonic(omega,g0)

	T_au = 1.5
	beta = 1.0/T_au 

	basis_N = 165
	n_eigen = 150

	path = os.path.dirname(os.path.abspath(__file__))
	#fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,basis_N,n_eigen)
	#fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
		
	#vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
	#vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))

	#print('vals',vals[:70])

	if(1):
		xgrid = np.linspace(-L,L,200)
		ygrid = np.linspace(-L,L,200)
		x,y = np.meshgrid(xgrid,ygrid)
		potgrid = pes.potential_xy(x,y)
		hesgrid = 0.25*(omega**2 + 4*g0*omega*(x**2+y**2) - 48*g0**2*x**2*y**2)
		plt.contour(x,y,potgrid,colors='k',levels=np.arange(0,30,0.5))#,levels=vals[:20])#np.arange(0,5,0.5))
		#plt.contour(x,y,hesgrid,colors='m',levels=np.arange(0.0,0.5,0.1))
		#plt.contour(x,y,potgrid,levels=[0.1,vals[0],vals[1],vals[3],vals[4],vals[5],vals[7],vals[100]])
		plt.show()

if(1): # Double well 2D
	L = 10.0#
	lbx = -7.0#-7.0
	ubx = 7.0#7.0
	lby = -1#-5.0
	uby = 5
	m = 0.5#8.0
	ngrid = 100
	ngridx = ngrid
	ngridy = ngrid

	w = 0.1	
	D = 9.375#10.0
	alpha = 1.147#0.148#0.42#55#0.37#0.2#0.52#2.64#0.363#0.81#0.175#0.41#0.255#1.165
	
	lamda = 2.0#4.0
	g = 0.08#0.08#lamda**2/32#4.0

	print('Vb', lamda**4/(64*g))

	z = 1.5#1.25#2.3	

	Tc = lamda*0.5/np.pi
	T_au = Tc#10.0 
	
	pes = quartic_bistable(alpha,D,lamda,g,z)

	path = os.path.dirname(os.path.abspath(__file__))	
	#print('pes', pes.potential_xy(0.0,0.0))
	if(0):
		potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
		fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
		
		vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
		vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))
		print('vals',vals[:20])
	
	xgrid = np.linspace(lbx,ubx,101)#200)
	ygrid = np.linspace(lby,uby,101)#200)
	x,y = np.meshgrid(xgrid,ygrid)

	potgrid = pes.potential_xy(x,y)
	hesgrid = ( -16*alpha**2*g**2*x**2*z**2*(x**2 - lamda**2/(8*g))**2*np.exp(-2*alpha*y*z) 
				+ 4*alpha**2*g*(-2*D*(1 - np.exp(-alpha*y))*np.exp(-alpha*y) + 2*D*np.exp(-2*alpha*y)  
				+ z**2*(g*(8*x**2 - lamda**2/g)**2 - lamda**4/g)*np.exp(-alpha*y*z)/64)*(-2*x**2*(1 - np.exp(-alpha*y*z))  
				+ 3*x**2 - (1 - np.exp(-alpha*y*z))*(8*x**2 - lamda**2/g)/8 - lamda**2/(8*g)) )
	#plt.imshow(potgrid,origin='lower',extent=[lbx,ubx,lby,uby])#,vmax=100)
	#plt.show()
	#hesgrid = 0.25*(omega**4 + 4*g0*omega**2*(x**2+y**2) - 48*g0**2*x**2*y**2)
	#plt.contour(x,y,potgrid,colors='k',levels=vals[:20])#levels=np.arange(0.0,1.11,0.1))##levels=vals[:2])#np.arange(0,5,0.5))
	#DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
	#plt.imshow(DVR.eigenstate(vecs[:,35])**2,origin='lower')	
	plt.title('De Leon-Berne potential')
	plt.contour(x,y,potgrid,colors='k',levels=np.arange(0.0,1.1*D+0.1,0.5))	
	#plt.contour(x,y,potgrid,colors='m',levels=np.arange(-1.0,0.0,0.001))
	#plt.contour(x,y,hesgrid,colors='g',levels=np.arange(-5.0,0.0,0.001))	
	plt.show()


