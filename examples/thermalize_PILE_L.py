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

def thermalize_rp(ens,rp,pes,time_therm,dt,potkey,rngSeed):
	motion = Motion(dt = dt,symporder=2)
	rng = np.random.default_rng(rngSeed)
	rp.bind(ens,motion,rng)
 
	therm = PILE_L(tau0=0.1,pile_lambda=100.0) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_II()
	propa.bind(ens, motion, rp, pes, rng, therm)

	sim = Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)
	start_time = time.time()

	nthermsteps = int(time_therm/motion.dt)
	pmats = np.array([True for i in range(rp.nbeads)])
	
	
	#tarr = []
	#kinarr = []
	for i in range(nthermsteps):
		sim.step(mode="nvt",var='pq',pmats=pmats)
		#tarr.append(i*dt)
		#kinarr.append((rp.pcart**2).sum())#kin.sum())

	#plt.plot(tarr,kinarr)
	#plt.show()

	print('kin',rp.kin.sum(),rp.pcart[0],rp.qcart[0])	


	store_arr(rp.qcart,'Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed))
	store_arr(rp.pcart,'Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed)) 
	
