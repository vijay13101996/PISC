import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
from PISC.engine import OTOC_f_1D
from PISC.engine import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
import time 

L = 10.0
lbx = -L
ubx = L
lby = -L
uby = L
m = 0.5
ngrid = 100
ngridx = ngrid
ngridy = ngrid
omega = 0.5
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
	L = 7.0
	lbx = -L#-2*L
	ubx = L#2*L
	lby = -6*L#-L
	uby = 12*L#4*L
	m = 0.5#8.0
	ngrid = 200
	ngridx = ngrid
	ngridy = ngrid

	w = 0.1	
	D = 5.0#10.0
	alpha = (0.5*m*w**2/D)**0.5#0.5#1.95
	
	lamda = 0.8#4.0
	g = 0.02#4.0

	z = 4.0#2.3	

	Tc = lamda*0.5/np.pi
	T_au = Tc#10.0 
	
	pes = quartic_bistable(alpha,D,lamda,g,z)

	if(0):
		potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
		fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,100)	
		
		vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
		vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))
		print('vals',vals[:20])
	
	xgrid = np.linspace(lbx,ubx,200)
	ygrid = np.linspace(lby,uby,200)
	x,y = np.meshgrid(xgrid,ygrid)
	potgrid = pes.potential_xy(x,y)
	#plt.imshow(potgrid,origin='lower',extent=[lbx,ubx,lby,uby])#,vmax=100)
	#plt.show()
	#hesgrid = 0.25*(omega**4 + 4*g0*omega**2*(x**2+y**2) - 48*g0**2*x**2*y**2)
	#plt.contour(x,y,potgrid,colors='k',levels=vals[:20])#levels=np.arange(0.0,1.11,0.1))##levels=vals[:2])#np.arange(0,5,0.5))
	plt.contour(x,y,potgrid,colors='k',levels=np.arange(0.0,5.1,0.1))	
	plt.contour(x,y,potgrid,colors='m',levels=np.arange(-5.0,0.0,0.1))	
	#plt.contour(x,y,hesgrid,colors='m',levels=[0.0])#np.arange(-0.0001,0,0.00001))
	#plt.contour(x,y,potgrid,levels=[0.1,vals[0],vals[1],vals[3],vals[4],vals[5],vals[7],vals[100]])
	plt.show()

if(0):
	xgrid = np.linspace(-L,L,200)
	ygrid = np.linspace(-L,L,200)
	x,y = np.meshgrid(xgrid,ygrid)
	potgrid = pes.potential_xy(x,y)
	hesgrid = 0.25*(omega**2 + 4*g0*omega*(x**2+y**2) - 48*g0**2*x**2*y**2)
	plt.contour(x,y,potgrid,colors='k',levels=np.arange(0,3,0.05))#,levels=vals[:20])#np.arange(0,5,0.5))
	plt.contour(x,y,hesgrid,colors='m',levels=[0.0])#np.arange(-0.0001,0,0.00001))
	#plt.contour(x,y,potgrid,levels=[0.1,vals[0],vals[1],vals[3],vals[4],vals[5],vals[7],vals[100]])
	plt.show()

#plt.plot(np.arange(len(vals)),vals)
#plt.contour(DVR.eigenstate(vecs[:,20]))
#plt.show()

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,0.0)		
fname1 = '{}/Datafiles/Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(path,potkey,T_au,100,10)
		
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,2.0)		
fname2 = '{}/Datafiles/Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(path,potkey,T_au,100,10)

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,1.0)		
fname3 = '{}/Datafiles/Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(path,potkey,T_au,100,10)

fig,ax = plt.subplots()	
plot_1D(ax,fname1,label='z=0.0',color='m',log=True)
plot_1D(ax,fname2,label='z=2.0',log=True)
plot_1D(ax,fname3,label='z=1.0',color='g',log=True)
plt.legend()
plt.show()

if(0):

	#fname1 = '{}/Datafiles/Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(path,potkey,T_au,120,100)
	fname2 = '{}/Datafiles/Kubo_Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(path,potkey,T_au,100,70)
	fname3 = '{}/Datafiles/Kubo_Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(path,potkey,T_au,90,60)
	fname4 = '/home/vgs23/PISC/examples/2D/classical/Datafiles/correctedClassical_OTOC_{}_T_{}_dt_{}'.format(potkey,T_au,0.005)
	fname5 = '/home/vgs23/PISC/examples/2D/cmd/Datafiles/corrected_CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}'.format(potkey,T_au,1000,4,16,0.005)
	fname6 = '/home/vgs23/PISC/examples/2D/cmd/Datafiles/corrected_CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}'.format(potkey,T_au,1000,8,16,0.005)
			
	fig,ax = plt.subplots()
	#plot_1D(ax,fname1,label='Quantum',color='k',log=True)
	plot_1D(ax,fname2,label='Kubo Quantum',color='m',log=True)
	plot_1D(ax,fname3,label='Kubo Quantum',color='y',log=True)
	plot_1D(ax,fname4,label='Classical',color='g',log=True)
	plot_1D(ax,fname5,label='CMD,B=4',color='r',log=True)
	#plot_1D(ax,fname6,label='CMD,B=8',color='b',log=True)

	plt.legend()
	plt.show()

if(0):
	data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]	

	ind_zero = 124
	ind_max = 281
	ist =  25 #180#360
	iend = 35 #220#380

	t_trunc = t_arr[ist:iend]
	OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
	slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
	print('slope',slope)

	a = -OTOC_arr
	x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
	#print('min max',t_arr[124],t_arr[281])

#plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
#plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
#plt.plot(t_arr,slope*t_arr+ic,'--',color='k')
#plt.show()
if(0):
	fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,80,50)
	data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,np.log(OTOC_arr),label='80,50')

	fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,90,60)
	data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,np.log(OTOC_arr),label='90,60')

	fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,120,100)
	data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,np.log(OTOC_arr),color='k',label='120,100')

if(0):
	#for T_au in [1.0,2.0,3.0,4.0,5.0]:	
		fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,120,100)
		data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]
		plt.plot(t_arr,np.log(OTOC_arr),label='Quantum n_trunc=120,n_eigen=100',color='k')

if(0):
		fname = 'Kubo_Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,165,150)
		data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]
		plt.plot(t_arr,np.log(OTOC_arr),label='Quantum n_trunc=165,n_eigen=150',color='b')

		fname = 'Kubo_Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,150,135)
		data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]
		plt.plot(t_arr,np.log(OTOC_arr),label='Quantum n_trunc=150,n_eigen=135',color='g')

if(0):
		fname = 'Classical_OTOC_{}_T_{}_dt_{}'.format(potkey,T_au,0.005)
		data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/classical/Datafiles/{}.txt'.format(fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]
		plt.plot(t_arr,np.log(OTOC_arr),color='m',label='Classical')

if(0):
		fname = 'corrected_CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}'.format(potkey,T_au,1000,4,16,0.005)
		data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]
		#print('tooc',OTOC_arr)
		plt.plot(t_arr,np.log(OTOC_arr),color='c',label='B=4, NVT')

		fname = 'corrected_CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_15'.format(potkey,T_au,1000,4,16,0.005)
		data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]
		#print('tooc',OTOC_arr)
		plt.plot(t_arr,np.log(OTOC_arr),color='g',label='B=4, NVT,15')
	
		fname = 'corrected_CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_10'.format(potkey,T_au,1000,4,16,0.005)
		data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]
		#print('tooc',OTOC_arr)
		plt.plot(t_arr,np.log(OTOC_arr),color='r',label='B=4, NVT,10')
	
	

		#fname = 'CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_conv'.format(potkey,T_au,1000,4,16,0.005)
		#data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
		#t_arr = data[:,0]
		#OTOC_arr = data[:,1]
		#print('tooc',OTOC_arr)
		#plt.plot(t_arr,np.log(OTOC_arr),label='B=4,NVE')
	
if(0):
	fname = 'testCMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}'.format(potkey,T_au,1000,4,16,0.005)
	data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,np.log(OTOC_arr),label='B=4')
	
	fname = 'testCMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_1'.format(potkey,T_au,1000,4,16,0.005)
	data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,np.log(OTOC_arr),label='B=4, 100 seeds')

	fname = 'testCMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_2'.format(potkey,T_au,1000,4,16,0.005)
	data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,np.log(OTOC_arr),label='B=4, 200 seeds')

	fname = 'testCMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_3'.format(potkey,T_au,1000,4,16,0.005)
	data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,np.log(OTOC_arr),label='B=4, 300 seeds')

if(0):	
	fname = 'CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}'.format(potkey,T_au,1000,8,16,0.005)
	data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,(OTOC_arr),label='B=8')

if(0):
	for i in [20,40,50,60,100]:
		fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,i,i)
		data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]
		plt.plot(t_arr,np.log(OTOC_arr))
		
	plt.show()	


