import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II
from PISC.engine.beads import RingPolymer
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time

def thermalize_rp(pathname,m,dim,N,nbeads,ens,pes,rng,time_therm,dt_therm,potkey,rngSeed,tau0=1.0,pile_lambda=100.0):	
	qcart = rng.normal(size=(N,dim,nbeads))
	qcart[:N//2,:,0]-=1.0 #These two lines need to be checked.
	qcart[N//2:,:,0]+=1.0 
	rp = RingPolymer(qcart=qcart,m=m) 
		
	motion = Motion(dt = dt_therm,symporder=2)
	rp.bind(ens,motion,rng)
 
	therm = PILE_L(tau0=tau0,pile_lambda=pile_lambda) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_II()
	propa.bind(ens, motion, rp, pes, rng, therm)

	sim = RP_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)
	start_time = time.time()

	nthermsteps = int(time_therm/motion.dt)
	pmats = np.array([True for i in range(rp.nbeads)])
		
	#tarr = []
	#kinarr = []
	for i in range(nthermsteps):
		sim.step(mode="nvt",var='pq',RSP=True,pc=True)
		#tarr.append(sim.t)
		#kinarr.append((rp.pcart**2).sum())#kin.sum())

	#plt.plot(tarr,kinarr)
	#plt.show()

	#qar = rp.q[:,0,0]
	#E = pes.potential(rp.qcart) + rp.pcart**2/(2*m) + 0.5*rp.dynm3*rp.dynfreq2*rp.q**2
	#E = np.sum(E,axis=2)
	#E = E[:,0]/nbeads
	#plt.hist(E,bins=50)
	#plt.show()
	#print('E',E.shape)

	print('kin',rp.kin.sum(),0.5*rp.ndim*rp.nsys*rp.nbeads**2/ens.beta)	

	store_arr(rp.qcart,'Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"{}/Datafiles".format(pathname))
	store_arr(rp.pcart,'Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"{}/Datafiles".format(pathname)) 
	
