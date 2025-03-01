import numpy as np
from PISC.dvr.dvr import DVR2D
#from PISC.husimi.Husimi import Husimi_2D,Husimi_1D
from PISC.potentials import coupled_harmonic, quartic_bistable, Harmonic_oblique, Tanimura_SB
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
#from PISC.engine import OTOC_f_1D
#from PISC.engine import OTOC_f_2D
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


if(0): # Oblique harmonic
	L = 10.0#
	lbx = -10.0#
	ubx = 10.0#
	lby = -10.0#
	uby = 10.0
	m = 0.5#8.0
	ngrid = 200
	ngridx = ngrid
	ngridy = ngrid

	omega1 = 1.0
	omega2 = 1.0
	trans = np.array([[1.0,0.7],[0.2,1.0]])
	
	pes	= Harmonic_oblique(trans,m,omega1,omega2)	
	
	xgrid = np.linspace(lbx,ubx,101)#200)
	ygrid = np.linspace(lby,uby,101)#200)
	x,y = np.meshgrid(xgrid,ygrid)

	potgrid = pes.potential_xy(x,y)
	
	plt.contour(x,y,potgrid,colors='k',levels=np.arange(0.0,10,0.5))	
	plt.show()

if(1): #Tanimura_SB
	lbx = -5.0#
	ubx = 10.0#
	lby = -5.0#
	uby = 10.0
	
	m = 1.0
	mb= 1.0
	delta_anh = 0.1
	w_10 = 1.0
	wb = w_10
	wc = w_10 + delta_anh
	alpha = (m*delta_anh)**0.5
	D = m*wc**2/(2*alpha**2)

	VLL = -0.75*wb#0.05*wb
	VSL = 0.75*wb#0.05*wb
	cb = 0.0#0.45*wb#0.75*wb

	pes = Tanimura_SB(D,alpha,m,mb,wb,VLL,VSL,cb)
			
	xgrid = np.linspace(lbx,ubx,101)
	ygrid = np.linspace(lby,uby,101)
	x,y = np.meshgrid(xgrid,ygrid)

	potgrid = pes.potential_xy(x,y)
	
	plt.contour(x,y,potgrid,colors='k',levels=np.arange(0.0,10,0.5))	
	plt.show()


if(0): # Double well 2D
	L = 10.0#
	lbx = -10.0#-7.0
	ubx = 10.0#7.0
	lby = -3#-5.0
	uby = 7
	m = 0.5#8.0
	ngrid = 200
	ngridx = ngrid
	ngridy = ngrid

	w = 0.1	
	
	lamda = 2.0
	g = 0.02

	Vb = lamda**4/(64*g)

	D = 3*Vb
	alpha = 0.382#1.147
		
	print('Vb', lamda**4/(64*g))

	z = 1.0	

	Tc = lamda*0.5/np.pi
	T_au = 10*Tc#10.0 
	
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
	plt.contour(x,y,potgrid,colors='k',levels=np.arange(0.0,1.01*D,0.5))	
	#plt.contour(x,y,potgrid,colors='m',levels=np.arange(-1.0,0.0,0.001))
	#plt.contour(x,y,hesgrid,colors='g',levels=np.arange(-5.0,0.0,0.001))	
	plt.show()


