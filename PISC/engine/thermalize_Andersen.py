import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II
from PISC.engine.beads import RingPolymer
from PISC.engine.motion import Motion
from PISC.engine.thermostat import Andersen
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time

def thermalize_rp_andersen(pathname,m,dim,N,nbeads,ens,pes,rng,time_therm,dt_therm,potkey,rngSeed):	
	qcart = rng.normal(size=(N,dim,nbeads))
	qcart[:N//2,:,0]-=1.0 #These two lines need to be checked.
	qcart[N//2:,:,0]+=1.0 
	rp = RingPolymer(qcart=qcart,m=m) 
		
	motion = Motion(dt = dt_therm,symporder=2)
	rp.bind(ens,motion,rng)
 
	therm = Andersen(N)
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_II()
	propa.bind(ens, motion, rp, pes, rng, therm)

	sim = RP_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)
	start_time = time.time()

	nthermsteps = int(time_therm/motion.dt)
	
	#tarr = []
	#kinarr = []
	#print('KI',(rp.pcart**2).sum()/sim.rp.nbeads)

	for i in range(nthermsteps):
		sim.step(mode="nve",var='pq',RSP=True)
		rp.pcart=sim.therm.generate_p()
		rp.p = rp.nmtrans.cart2mats(rp.pcart)


	print('kin',rp.kin.sum()/rp.nsys,0.5*rp.ndim*rp.nbeads**2/ens.beta)	

	store_arr(rp.qcart,'Andersen_thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"{}/Datafiles".format(pathname))
	store_arr(rp.pcart,'Andersen_thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"{}/Datafiles".format(pathname)) 
	
