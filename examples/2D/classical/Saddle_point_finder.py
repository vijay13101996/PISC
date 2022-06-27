import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.instanton import instantonize
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time

### This code finds the minima and the saddle point in a potential (2D double well in this case) and 
### returns the 'transition path' connecting those two. This code can be modified slightly to do the 
### same for other potentials.

def separatrix_path():
	### Potential parameters
	m=0.5

	D = 10.0
	alpha = 0.363

	lamda = 2.0
	g = 0.08

	z = 1.5	

	pes = quartic_bistable(alpha,D,lamda,g,z)

	### Simulation parameters
	T_au = 0.5*lamda*0.5/np.pi
	beta = 1.0/T_au 

	rngSeed = 1
	N = 1
	dim = 2
	nbeads = 1
	T = T_au	
	dt = 0.005

	### Plot extent and axis
	L = 7.0
	lbx = -L
	ubx = L
	lby = -5
	uby = 10
	ngrid = 200
	ngridx = ngrid
	ngridy = ngrid

	xgrid = np.linspace(lbx,ubx,200)
	ygrid = np.linspace(lby,uby,200)
	x,y = np.meshgrid(xgrid,ygrid)

	rng = np.random.RandomState(1)
	ens = Ensemble(beta=beta,ndim=dim)
	motion = Motion(dt = dt,symporder=2) 
	rng = np.random.default_rng(rngSeed) 

	pes = quartic_bistable(alpha,D,lamda,g,z)
	therm = PILE_L(tau0=0.1,pile_lambda=100.0) 

	propa = Symplectic_order_II()
		
	sim = RP_Simulation()

	q = np.zeros((N,dim,nbeads))
	x0 = 0.5
	y0 = 0.5
	q[...,0,0] = x0
	q[...,1,0] = y0
	p = rng.normal(size=q.shape)
	p[...,0] = 0.0	
	rp = RingPolymer(q=q,p=p,m=m)
	sim.bind(ens,motion,rng,rp,pes,propa,therm)	

	#potgrid = pes.potential_xy(x,y)
	#plt.contour(x,y,potgrid,colors='k',levels=np.arange(0,D,D/20))
	#print('pot',pes.potential_xy(0,0))

	inst = instantonize(stepsize=1e-3,tol=1e-2)
	inst.bind(pes,rp)

	### Gradient descent to get to a minima
	step = inst.grad_desc_step()#eigvec_follow_step(eig_dir=0)		
	count = 0
	while(step!="terminate"):
		inst.slow_step_update(step)
		step = 	inst.grad_desc_step()#inst.eigvec_follow_step(eig_dir=0)
		#if(count%3000==0):
		#	plt.scatter(rp.q[0,0,0],rp.q[0,1,0],color='m')
		#	plt.pause(0.01)
		#count+=1

	minima = rp.q.copy()
	#print('Minima', rp.q)

	### Saddle point finding using Eigenvector following
	inst.tol=1e-3
	inst.stepsize=1e-3
	eig_dir = 1

	vals, vecs = np.linalg.eigh(inst.hess)
	#print('vals,vecs',vals,vecs)
	if(vals[eig_dir] > 0.5*vals[eig_dir-1]): ## Comment this part out when the e.v. along saddle point coordinate is "soft"
		scale = 0.25
		#print('before scaling', rp.q,inst.q) 
		qscale = (inst.q.copy()).reshape((-1,dim,nbeads))	
		qscale[:,eig_dir]/=scale
		inst.reinit_inst_scaled(qscale,scale,eig_dir)
		vals, vecs = np.linalg.eigh(inst.hess)
		#print('vals,vecs after scaling',vals,vecs)
		eig_dir=0
		#print('scaled',rp.q,inst.q)
		
	step = inst.eigvec_follow_step(eig_dir=eig_dir)
	rp.mats2cart()
	pes.update()
	#plt.scatter(rp.q[0,0,0],rp.q[0,1,0])	

	print('step',step)
	qarr = []
	vecsarr = []

	count=0
	while(step!="terminate"):
		inst.slow_step_update_soft(step,scale,eig_dir)  ## Change here when the e.v. along saddle point coordinate is "soft"
		step = 	inst.eigvec_follow_step(eig_dir=eig_dir)
		if(count%10==0):
			vals,vecs = np.linalg.eigh(pes.ddpot[0,:,:,0,0])
			qarr.append(rp.q)
			vecsarr.append(vecs.copy())
			#print('rp',rp.q)
			#plt.scatter(rp.q[0,0,0],rp.q[0,1,0],color='c')
			#plt.pause(0.01)
		##	print('count',count)
		count+=1
	
	qarr = np.array(qarr)
	vecsarr = np.array(vecsarr)
	qarr = qarr[:,0,:,0]
	
	return qarr,vecsarr
	
