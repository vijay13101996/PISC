import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L,Constrain_theta 
from PISC.engine.simulation import RP_Simulation, Matsubara_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr
import time

def constrained_theta_thermalize(pathname,theta,ens,rp,pes,time_therm,time_relax,dt,potkey,rngSeed):
	motion = Motion(dt = dt,symporder=2)
	rng = np.random.default_rng(rngSeed)  
	therm = PILE_L(tau0=100.0,pile_lambda=1.0)
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_II()
	propa.bind(ens, motion, rp, pes, rng, therm)

	sim = RP_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)

	# Sampling q
	nthermsteps = int(time_therm/dt)
	sim.step(ndt=nthermsteps,mode="nvt_theta",var='pq',pc=True)

	rp.p[...,rp.nmats:] = 0.0
	rp.q[...,rp.nmats:] = 0.0
		
	# Constrain theta
	theta_ens = Constrain_theta()
	theta_ens.bind(rp,motion,rng,ens)
	theta_ens.theta_constrained_randomize()
	
	rp = RingPolymer(q=rp.q,p=rp.p,m=rp.m,nmats=rp.nmats,mode='rp/mats')
	sim.bind(ens,motion,rng,rp,pes,propa,therm)

	# Relax the distribution
	nrelaxsteps = int(time_relax/dt)
	sim.step(ndt = nrelaxsteps,mode="nve",var='pq')
	#print('theta',rp.theta,theta)
	
	store_arr(rp.qcart,'Theta_{}_thermalized_rp_qcart_N_{}_nbeads_{}_nmats_{}_beta_{}_{}_seed_{}'.format(theta,rp.nsys,rp.nbeads,rp.nmats,ens.beta,potkey,rngSeed),"{}/Datafiles/".format(pathname))
	store_arr(rp.pcart,'Theta_{}_thermalized_rp_pcart_N_{}_nbeads_{}_nmats_{}_beta_{}_{}_seed_{}'.format(theta,rp.nsys,rp.nbeads,rp.nmats,ens.beta,potkey,rngSeed),"{}/Datafiles/".format(pathname)) 
	
