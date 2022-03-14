import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import pickle
import os

m = 0.5
omega = 0.5
g0 = 0.1#3e-3#0.0#1/100.0
T_au = 1.5
beta = 1.0/T_au 
print('T in au',T_au)

potkey = 'coupled_harmonic_w_{}_g_{}'.format(omega,g0)
sysname = 'Selene'		

N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 40.0
time_total = 10.0

tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

path = os.path.dirname(os.path.abspath(__file__))
datapath = '{}/Datafiles'.format(path)

if(1):
	keyword1 = 'correctedClassical_OTOC_Selene_{}'.format(potkey)
	keyword2 = 'T_{}'.format(T_au)
	keyword3 = 'dt_{}'.format(dt)
	flist = []
	
	for fname in os.listdir(datapath):
		if keyword1 in fname and keyword2 in fname and keyword3 in fname:
			#print('fname',fname)
			flist.append(fname)

	count=0

	for f in flist:
		data = read_1D_plotdata('{}/{}'.format(datapath,f))
		tarr = data[:,0]
		OTOCarr += data[:,1]
		#plt.plot(data[:,0],np.log(abs(data[:,1])))
		#print('data',data[:,1])
		count+=1

	print('count',count)
	#print('otoc',OTOCarr)
	OTOCarr/=count
	#plt.plot(tarr,np.log(abs(OTOCarr)))
	#plt.show()
	store_1D_plotdata(tarr,OTOCarr,'correctedClassical_OTOC_{}_T_{}_dt_{}'.format(potkey,T_au,dt),datapath)

if(0): # Beads
	#fig = plt.figure()
	plt.suptitle(r'Classical OTOC at $T={}$'.format(T_au))
	data = read_1D_plotdata('{}/Classical_OTOC_{}_T_{}_dt_{}.txt'.format(datapath,potkey,T_au,dt))
	tarr = data[:,0]
	OTOCarr = data[:,1]
	#plt.plot(tarr,np.log(abs(OTOCarr)),linewidth=3)#,color='k')

	ist = 500
	iend = 550
	t_trunc = tarr[ist:iend]
	OTOC_trunc = (np.log(OTOCarr))[ist:iend]
	slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
	print('slope classical',slope)

	a = -OTOCarr
	x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
	#print('min max',t_arr[124],t_arr[281])

	plt.plot(tarr,np.log(OTOCarr), linewidth=2,label='Classical OTOC',color='k')
	#plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
		
	plt.legend()
	plt.show()	


