import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.harmonic_2D import Harmonic
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.eckart import eckart
from PISC.potentials.harmonic_1D import harmonic
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
from thermalize import thermalize
import pickle
import h5py

def main(filename,pathname,sysname,potkey,nrun,lamda,g,times,m,N,dt_therm,dt,rngSeed,time_therm,time_total):
	dim = 1
	Tc = lamda*(0.5/np.pi)#5.0
	T = times*Tc
	print('T',T)
	
	nbeads = 1
	rng = np.random.RandomState(1)
	qcart = rng.normal(size=(N,dim,nbeads))
	q = np.random.normal(size=(N,dim,nbeads))
	M = np.random.normal(size=(N,dim,nbeads))

	pcart = None
	dt = dt_therm
	beta = 1/T
	
	rngSeed = rngSeed
	rp = RingPolymer(qcart=qcart,m=m) 
	ens = Ensemble(beta=beta,ndim=dim)
	motion = Motion(dt = dt,symporder=2) 
	rng = np.random.default_rng(rngSeed) 

	rp.bind(ens,motion,rng)

	#pes = eckart(1.0,0.4,0.001)
	#pes.bind(ens,rp)
	pes = double_well(lamda,g)
	pes.bind(ens,rp)

	time_therm = time_therm
	thermalize(pathname,ens,rp,pes,time_therm,dt,potkey,rngSeed)
	
	tarr=[]
	qarr=[]
	potarr=[]
	Mqqarr = []
	Mqqarrfull = []
	Earr = []
	dt = dt
	
	qcart = read_arr('Thermalized_q_N_{}_beta_{}_{}_seed_{}'.format(rp.nsys,ens.beta,potkey,rngSeed),"{}/Datafiles/".format(pathname))
	pcart = read_arr('Thermalized_p_N_{}_beta_{}_{}_seed_{}'.format(rp.nsys,ens.beta,potkey,rngSeed),"{}/Datafiles/".format(pathname))

	rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,mode='rp')
	motion = Motion(dt = dt,symporder=4) 
	rp.bind(ens,motion,rng)
	pes.bind(ens,rp)	

	therm = PILE_L(tau0=0.1,pile_lambda=1000.0) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_IV()
	propa.bind(ens, motion, rp, pes, rng, therm)
	
	sim = RP_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)

	time_total = time_total
	nsteps = int(time_total/dt)	

	start_time = time.time()
		
	for i in range(nsteps):
		sim.step(mode="nve",var='monodromy',pc=False)
		Mqq = np.mean(abs(rp.Mqq[:,0,0,0,0]**2)) #rp.Mqq[0,0,0,0,0]#
		tarr.append(sim.t)
		#qarr.append(rp.q[:,0,0].copy())
		#Mqqarrfull.append(rp.Mqq[:,0,0,0,0].copy())
		#potarr.append(pes.ddpot[0,0,0,0,0])
		#Mqqarr.append(propa.rp.Mqq[0,0,0,0,0])
		Mqqarr.append(Mqq)
		#Earr.append(np.sum(pes.pot)+np.sum(rp.pot)+rp.kin)

	if(0):
		with h5py.File(filename, 'a') as f:
			group = f['Run#{}'.format(nrun)]
			group.attrs['lambda'] = lamda
			group.attrs['g'] = g
			group.attrs['T'] = T
			group.attrs['xTc'] = times
			group.attrs['m'] = m
			group.attrs['N'] = N
			group.attrs['nbeads'] = nbeads
			group.attrs['dt_therm'] = dt	
			group.attrs['therm_time'] = time_therm
			group.attrs['total_OTOC_time'] = time_total
			group.attrs['gamma'] = gamma
			group.attrs['seed'] = rngSeed
			group.create_dataset('tarr',data=tarr)
			group.create_dataset('Mqqarr',data=Mqqarr)

	fname = 'Classical_OTOC_{}_{}_T_{}Tc_N_{}_dt_{}_seed_{}'.format(sysname,potkey,times,N,dt,rngSeed)
	store_1D_plotdata(tarr,Mqqarr,fname,'{}/Datafiles'.format(pathname))
		
	plt.plot(tarr,np.log(Mqqarr))
	plt.show()	
