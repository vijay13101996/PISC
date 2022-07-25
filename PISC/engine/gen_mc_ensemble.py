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

def generate_rp(pathname,m,dim,N,nbeads,ens,pes,rng,time_relax,dt_relax,potkey,rngSeed,E,qlist):
	index_arr = rng.choice(len(qlist),N)  # Choose N points at random from the qlist
	qcart = np.zeros((N,dim,nbeads)) 
	pcart = np.zeros((N,dim,nbeads))
	for i in range(nbeads):	
		qcart[:,:,i] =qlist[index_arr]# Initialize ring polymers with collapsed configuration at these points	

	pot = pes.potential(qcart)
	if(dim>1):
		pot= pot[:,np.newaxis,:]	
	V = np.sum(pot,axis=2) ##May need to be changed for 2D
	V = V[:,0]
	#print('V',V,E*nbeads)

	pcoeff = rng.dirichlet(np.ones(dim*nbeads),size=1)
	pcoeff=pcoeff[0]
	count=0
	# Initialize the centroid momenta along each component to the appropriate value so that the total energy is E	
	for d in range(dim):
		for b in range(nbeads):
			pcart[:,d,b] = (np.sqrt(2*m*pcoeff[count]*(E*nbeads-V))) 
			count+=1	
	
	rp = RingPolymer(qcart=qcart,pcart=pcart,m=m) 	
	motion = Motion(dt = dt_relax,symporder=2)
	rp.bind(ens,motion,rng)
 
	therm = PILE_L(tau0=1.0,pile_lambda=1000.0) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_II()
	propa.bind(ens, motion, rp, pes, rng, therm)

	sim = RP_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)
	start_time = time.time()

	nthermsteps = int(time_relax/motion.dt)

	print('rp = (E*nbeads) = %.3f; E = %.3f; nbeads = %i'% (E*nbeads,np.sum(rp.pcart[0]**2) +np.sum(pot[0]),nbeads))
	#print('V', pes.pot[0],rp.qcart[0])
	#print('E tot', np.sum(rp.pcart**2/(2*m),axis=2) + np.sum(pes.potential(rp.qcart),axis=2 ) )
	#plt.axis([-10, 10, 0, 5])		
	
	# Run NVE steps until time_therm	
	for i in range(nthermsteps):
		sim.step(mode="nve",var='pq',RSP=True)
		#if(i%50==0):
			#print('t, cent E',sim.t,np.sum(rp.p[0,:,0]**2/nbeads) + pes.potential(rp.q[:,:,0]/nbeads**0.5)[0],E )
			#print('rp',rp.qcart[0],rp.p[0])
			#print('Energy', np.sum(pes.potential(rp.qcart[0]) +  rp.pcart[0]**2))
			#plt.scatter(rp.qcart[0,0],np.ones(nbeads)*E)
			#plt.pause(0.2)
		#tarr.append(i*dt)
		#kinarr.append((rp.pcart**2).sum())#kin.sum())

	#plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:])
	#plt.show()
	store_arr(rp.qcart,'Microcanonical_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"{}/Datafiles".format(pathname))
	store_arr(rp.pcart,'Microcanonical_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"{}/Datafiles".format(pathname)) 