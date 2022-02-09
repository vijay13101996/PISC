import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
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
omega = 1.0
g0 = 0.1#3e-3#1/100.0
x = np.linspace(lbx,ubx,ngridx+1)
potkey = 'coupled_harmonic'#_w_{}_g_{}'.format(omega,g0)
pes = coupled_harmonic(omega,g0)

T_au = 3.0 
beta = 1.0/T_au 

basis_N = 120
n_eigen = 100

path = os.path.dirname(os.path.abspath(__file__))
fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,basis_N,n_eigen)
fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	
vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))

print('vals',vals[:100])
if(0):
	xgrid = np.linspace(-L,L,200)
	ygrid = np.linspace(-L,L,200)
	x,y = np.meshgrid(xgrid,ygrid)
	potgrid = pes.potential_xy(x,y)
	hesgrid = 0.25*(omega**2 + 4*g0*omega*(x**2+y**2) - 48*g0**2*x**2*y**2)
	plt.contour(x,y,potgrid,colors='k',levels=vals[:20])#np.arange(0,5,0.5))
	plt.contour(x,y,hesgrid,colors='m',levels=[0.0])#np.arange(-0.0001,0,0.00001))
	#plt.contour(x,y,potgrid,levels=[0.1,vals[0],vals[1],vals[3],vals[4],vals[5],vals[7],vals[100]])
	plt.show()

#plt.plot(np.arange(len(vals)),vals)
#plt.contour(DVR.eigenstate(vecs[:,20]))
#plt.show()

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
fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,120,100)
data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
t_arr = data[:,0]
OTOC_arr = data[:,1]
plt.plot(t_arr,np.log(OTOC_arr),label='Quantum')

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

if(1):
	fname = 'Classical_OTOC_{}_T_{}_dt_{}'.format(potkey,T_au,0.005)
	data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/classical/Datafiles/{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,np.log(OTOC_arr))

if(1):
	fname = 'CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}'.format(potkey,T_au,1000,4,16,0.005)
	data = read_1D_plotdata('/home/vgs23/PISC/examples/2D/cmd/Datafiles/{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
	plt.plot(t_arr,np.log(OTOC_arr),label='cmd')

plt.legend()
plt.show()

if(0):
	for i in [20,40,50,60,100]:
		fname = 'Quantum_OTOC_{}_T_{}_basis_{}_n_eigen_{}'.format(potkey,T_au,i,i)
		data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(path,fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]
		plt.plot(t_arr,np.log(OTOC_arr))
		
	plt.show()	


