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

def thermalize_rp(pathname,m,dim,N,nbeads,ens,pes,rng,time_therm,dt_therm,potkey,rngSeed,qlist=None,tau0=1.0,pile_lambda=100.0):	
	if(qlist is None):
		qcart = np.zeros((N,dim,nbeads))
		pcart = np.zeros((N,dim,nbeads))
		qcart[:N//2,0,:]-=2.5 #These two lines are specific to the 2D double well. 
		qcart[N//2:,0,:]+=2.5
	else:
		if(dim==1):
			expbeta = np.exp(-ens.beta*pes.potential(qlist))[:,0]
		else:
			expbeta = np.exp(-ens.beta*pes.potential_xy(qlist[:,0],qlist[:,1]))
		probgrid = expbeta/np.sum(expbeta) 
		index_arr = rng.choice(len(qlist),N, p=probgrid)  # Choose N points at random from the qlist
		qcart = np.zeros((N,dim,nbeads))
		pcart = np.zeros((N,dim,nbeads))		
		for i in range(nbeads):
			qcart[:,:,i] = qlist[index_arr]  # Initialize ring polymers with collapsed configuration at these points
			pcart[:,:,i] = rng.normal(0.0,(m/ens.beta)**0.5,(N,dim))

	rp = RingPolymer(qcart=qcart,pcart=pcart,m=m) 
		
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

	#pot = pes.potential_xy(rp.qcart[:,0,0],rp.qcart[:,1,0])
	#kin = np.sum(rp.pcart**2/(2*m),axis=1)[:,0]
	#tot = pot+kin
	#print('pot', np.around(pot[pot>10],2),pot[pot>10].shape,pot.shape)
		
	print('End of thermalization. Seed: {} Classical Kinetic energy {:5.3f} Target value {:5.3f} '.format(rngSeed,rp.kin.sum()/rp.nsys,0.5*rp.ndim*rp.nbeads**2/ens.beta))	
	
	store_arr(rp.qcart,'Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"{}/Datafiles".format(pathname))
	store_arr(rp.pcart,'Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"{}/Datafiles".format(pathname)) 
	
