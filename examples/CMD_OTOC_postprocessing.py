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
import thermalize_PILE_L
from thermalize_PILE_L import thermalize_rp
import pickle

dim = 1
lamda = 0.8
g = 1/50.0
Tc = lamda*(0.5/np.pi)#5.0
T = Tc
print('T',T)
m = 0.5
N = 1000

nbeads = 8#32 
rng = np.random.RandomState(1)
qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = None
dt = 0.005
beta = 1/T

#for i in range(1,6):
rngSeed = 1 
rp = RingPolymer(qcart=qcart,m=m) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(rngSeed) 

rp.bind(ens,motion,rng)

potkey = 'inv_harmonic_lambda_{}'.format(lamda)	
pes = double_well(lamda,g)#harmonic(2.0*np.pi)# Harmonic(2*np.pi)#harmonic(2.0)#Harmonic(2*np.pi)#
pes.bind(ens,rp)

time_therm = 100.0

dt = 0.005
gamma = 32

dt = dt/gamma


if(0):	
	tfinal = 10.0
	#f =  open("/home/vgs23/Pickle_files/OTOC_{}_beta_0.2_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',50,50,4.0),'rb+')
	f =  open("/home/vgs23/Pickle_files/OTOC_{}_lambda_{}_1Tc_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',lamda,50,50,tfinal),'rb+')
	t_arr = pickle.load(f,encoding='latin1')
	OTOC_arr = pickle.load(f,encoding='latin1')
	plt.plot(t_arr,np.log(OTOC_arr), linewidth=4,label='Quantum OTOC')
	f.close()
	#plt.show()


if(0):	
	for nbeads in [1,4,8]:#,8,16]:#[1,4,8,32]:#,16,32]:# in range(1,6):
		fname = '/home/vgs23/Pickle_files/Test_CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_thermtime_{}_seed_{}.txt'.format(potkey,T,N,nbeads,gamma,dt,time_therm,rngSeed)
		data = read_1D_plotdata(fname)
		tarr = data[:,0]
		Mqqarr = data[:,1]
		#plt.plot(tarr,np.log(abs(Mqqarr)**2))
		#plt.plot(tarr,Mqqarr)
		plt.plot(tarr,np.log(Mqqarr),label=nbeads)
	plt.legend()
	plt.show()	


