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

def thermalize(pathname,ens,rp,pes,time_therm,dt,potkey,rngSeed):
	motion = Motion(dt = dt,symporder=2)
	rng = np.random.default_rng(rngSeed)
	rp.bind(ens,motion,rng)
 
	therm = PILE_L(tau0=0.1,pile_lambda=100.0) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_II()
	propa.bind(ens, motion, rp, pes, rng, therm)

	sim = RP_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)
	start_time = time.time()

	nthermsteps = int(time_therm/motion.dt)
	
	for i in range(nthermsteps):
		sim.step(mode="nvt",var='pq',RSP=False,pc=True)
		#tarr.append(i*dt)
		#kinarr.append((rp.pcart**2).sum())#kin.sum())

	#plt.plot(tarr,kinarr)
	#plt.show()

	print('kin',rp.kin.sum(),0.5*rp.ndim*rp.nsys/ens.beta)	
	
	store_arr(rp.qcart,'Thermalized_q_N_{}_beta_{}_{}_seed_{}'.format(rp.nsys,ens.beta,potkey,rngSeed),'{}/Datafiles/'.format(pathname))
	store_arr(rp.pcart,'Thermalized_p_N_{}_beta_{}_{}_seed_{}'.format(rp.nsys,ens.beta,potkey,rngSeed),'{}/Datafiles/'.format(pathname)) 
	