import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.harmonic_2D import Harmonic
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.harmonic_1D import harmonic
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import pickle
import os

lamda = 1.5#0.8
g = lamda**2/32#1/50.0
times = 0.9
m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.01#05
time_therm = 20.0
time_total = 5.0

path = os.path.dirname(os.path.abspath(__file__))

tarr = np.linspace(0.0,5.0,int(time_total/dt))
OTOCarr = np.zeros_like(tarr) +0j
potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

path = os.path.dirname(os.path.abspath(__file__))
datapath = '{}/Datafiles'.format(path)

if(1):
	keyword1 = 'Classical_OTOC_Selene'
	keyword2 = 'T_{}Tc'.format(times)
	flist = []

	for fname in os.listdir(datapath):
		if keyword1 in fname and keyword2 in fname and potkey in fname:
			#print('fname',fname)
			flist.append(fname)

	count=0

	for f in flist:
		data = read_1D_plotdata('{}/{}'.format(datapath,f))
		#print('data',np.shape(data))
		tarr = data[:,0]
		OTOCarr += data[:,1]
		#plt.plot(data[:,0],np.log(abs(data[:,1])))
		#print('data',data[:,1])
		count+=1

	print('count',count)
	#print('otoc',OTOCarr)
	OTOCarr/=count
	plt.plot(tarr,np.log(abs(OTOCarr)))
	plt.show()
	store_1D_plotdata(tarr,OTOCarr,'Classical_OTOC_{}_T_{}Tc_dt_{}'.format(potkey,times,dt),datapath)

if(0): # Beads
	#fig = plt.figure()
	plt.suptitle(r'Classical OTOC at $T=T_c$')
	data = read_1D_plotdata('{}/Classical_OTOC_{}_T_1Tc_dt_{}.txt'.format(datapath,potkey,dt))
	tarr = data[:,0]
	OTOCarr = data[:,1]
	plt.plot(tarr,np.log(abs(OTOCarr)),linewidth=3)#,color='k')

	ist = 500
	iend = 550
	t_trunc = tarr[ist:iend]
	OTOC_trunc = (np.log(OTOCarr))[ist:iend]
	slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
	print('slope classical',slope)

	a = -OTOCarr
	x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
	#print('min max',t_arr[124],t_arr[281])

	plt.plot(tarr,np.log(OTOCarr), linewidth=2,label='Classical OTOC')
	plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
		
	plt.legend()
	#plt.show()  

if(0): # Quantum  
		tfinal = 10.0
		#f =  open("/home/vgs23/Pickle_files/OTOC_{}_beta_0.2_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',50,50,4.0),'rb+')
		f =  open("{}/OTOC_{}_lambda_{}_1Tc_basis_{}_n_eigen_{}_tfinal_{}.dat".format(datapath,'inv_harmonic',lamda,50,50,tfinal),'rb+')
		t_arr = pickle.load(f,encoding='latin1')
		OTOC_arr = pickle.load(f,encoding='latin1')

		ist = 160
		iend = 180

		t_trunc = t_arr[ist:iend]
		OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
		slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
		print('slope',slope)

		a = -OTOC_arr
		x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
		#print('min max',t_arr[124],t_arr[281])

		plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
		plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
		plt.plot(t_arr,slope*t_arr+ic,'--',color='k')
		f.close()

if(1): # Quantum Eckart
		fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format('eckart',times,30,30)
		data = read_1D_plotdata('./examples/quantum/Datafiles/{}.txt'.format(fname))
		t_arr = data[:,0]
		OTOC_arr = data[:,1]	

		ist =  50 #180#360
		iend = 70 #220#380

		t_trunc = t_arr[ist:iend]
		OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
		slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
		print('slope',slope)

		a = -OTOC_arr
		x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
		#print('min max',t_arr[124],t_arr[281])

		plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
		plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
		plt.plot(t_arr,slope*t_arr+ic,'--',color='k')	

plt.legend()
plt.show()

