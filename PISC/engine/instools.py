import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.instanton import inst
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
from functools import partial
from matplotlib import pyplot as plt
from scipy import interpolate
from PISC.utils import nmtrans

def find_minima(m,pes,qinit,ax,plt,plot=False,dim=2):
	ens = Ensemble(ndim=dim)
	motion = Motion(symporder=2) 
	rng = np.random.default_rng(0) 
	therm = PILE_L() 
	propa = Symplectic_order_II()
		
	sim = RP_Simulation()

	q = np.zeros((1,dim,1))
	q[...,:,0] = qinit
	p = rng.normal(size=q.shape)
	p[...,0] = 0.0	
	rp = RingPolymer(q=q,p=p,m=m)
	sim.bind(ens,motion,rng,rp,pes,propa,therm)	

	insta = inst(stepsize=1e-3,tol=1e-2)
	insta.bind(pes,rp)

	### Gradient descent to get to a minima
	step = insta.grad_desc_step()	
	count = 0
	while(step!="terminate"):
		insta.slow_step_update(step)
		step = 	insta.grad_desc_step()
		if(count%300==0 and plot is True):
			ax.scatter(rp.q[0,0,0],rp.q[0,1,0],color='m')
			plt.pause(0.01)
		count+=1

	minima = rp.q[0,:,0]
	return minima

def find_saddle(m,pes,qinit,ax,plt,plot=False,dim=2,scale=1.0):
	ens = Ensemble(ndim=dim)
	motion = Motion(symporder=2) 
	rng = np.random.default_rng(0) 
	therm = PILE_L() 
	propa = Symplectic_order_II()
		
	sim = RP_Simulation()

	q = np.zeros((1,dim,1))
	q[...,:,0] = qinit
	p = rng.normal(size=q.shape)
	p[...,0] = 0.0	
	rp = RingPolymer(q=q,p=p,m=m)
	sim.bind(ens,motion,rng,rp,pes,propa,therm)	

	insta = inst(stepsize=1e-3,tol=1e-4)
	insta.bind(pes,rp)

	vals,vecs = insta.diag_hess()
	if(vals[0]>vals[1]/4):
		#scale = 8
		qscale = (insta.q.copy()).reshape((-1,dim,1))	
		qscale[:,0]/=scale
		insta.reinit_inst_scaled(qscale,scale)
		upd_step = partial(insta.slow_step_update_soft,scale,0)
	else:
		upd_step = insta.slow_step_update	
	
	### Streambed walking to the barrier top
	step = insta.eigvec_follow_step()	
	count = 0
	
	while(step!="terminate"):
		upd_step(step)  # Change here when the e.v. along saddle point coordinate is "soft"
		step = 	insta.eigvec_follow_step()
		if(count%1000==0 and plot is True):
			#vals,vecs = np.linalg.eigh(pes.ddpot)
			#qarr.append(rp.q)
			#vecsarr.append(vecs)
			#print('rp',rp.q)
			ax.scatter(rp.q[0,0,0],rp.q[0,1,0],color='c')
			plt.pause(0.01)
			#print('count',count)
		count+=1

	sp = rp.q[0,:,0]	
	vals,vecs = insta.diag_hess()
	print('rp',np.around(pes.ddpot[0,:,:,0,0],3))
	return sp, vals, vecs

def find_extrema(m,pes,qinit,ax,plt,plot=False,dim=2,stepsize=1e-3,tol=1e-2):
	ens = Ensemble(ndim=dim)
	motion = Motion(symporder=2) 
	rng = np.random.default_rng(0) 
	therm = PILE_L() 
	propa = Symplectic_order_II()
		
	sim = RP_Simulation()

	pinit = rng.normal(size=qinit.shape)
	rp = RingPolymer(qcart=qinit,pcart=pinit,m=m)
	sim.bind(ens,motion,rng,rp,pes,propa,therm)	

	insta = inst(stepsize=stepsize,tol=tol)
	insta.bind(pes,rp,cart=True)
	
	#print('q cart after doubling', rp.qcart)
	#print('dpot cart after doubling', rp.dpot_cart)
	#print('pot after doubling', rp.pot)
	
	ax.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:],color='k')
			
	### Gradient descent to get to a minima
	step = insta.grad_desc_step()	
	count = 0
	while(step!="terminate"):
		insta.slow_step_update(step,cart=False)
		step = 	insta.grad_desc_step()
		if(count%1000==0 and plot is True):
			print('grad norm', np.sum(insta.grad**2)**0.5)	
			ax.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:])
			plt.pause(0.01)
		count+=1

	minima = rp.qcart
	return minima

def inst_init(sp,delta,eigvec,nbeads):
	insta = inst()
	rp_init = insta.saddle_init(np.array(sp),delta,eigvec,nbeads)
	return rp_init		

def inst_double(instanton,dim=2,per_dev=2,plot=False): #Needs to be redefined for higher dimensions
	if(0):
		nbg = len(instanton[0,0])
		nbeads = 2*nbg
		
		x = instanton[0,0,:]
		xmin = np.min(x)
		xmax = np.max(x)
		y = instanton[0,1,:]	
		f = interpolate.interp1d(x, y)

		xarr = np.linspace(xmin,xmax,nbeads)
		yarr = f(xarr)
		
		rp_init = np.zeros((1,dim,nbeads))
		rp_init[0,0] = xarr
		rp_init[0,1] = yarr

	if(1):
		#print('before', instanton)
		nbg = len(instanton[0,0])
		nbeads = 2*nbg
		print('nb after doubling', nbeads)
		rp_init = np.zeros((1,dim,nbeads))
		rp_init[:,:,::2] = instanton
		for i in range(nbg):
			rp_init[:,:,2*i-1] = (instanton[:,:,i] + instanton[:,:,i-1])/2
			if(plot is True):
				plt.scatter(instanton[:,0,i], instanton[:,1,i],color='r')
				plt.scatter(rp_init[:,0,2*i+-1], rp_init[:,1,2*i-1],color='g')
				plt.scatter(instanton[:,0,i-1], instanton[:,1,i-1],color='b')
				plt.pause(1)
		#print('after',rp_init)
			
	if(0):
		nbg = len(instanton[0,0])
		nbeads = 2*nbg
		print('nbeads after doubling', nbeads)
		rp_init = np.zeros((1,dim,nbeads))
		rp_init[:,:,::2] = instanton
		rp_init[:,:,1::2] = (1+per_dev/100)*instanton	

	if(0):
		nbg = len(instanton[0,0])
		nbeads = 2*nbg
		print('nb after doubling', nbeads)
		rp_init = np.zeros((1,dim,nbeads))
		rp_init[:,:,:nbg] = instanton
		rp_init[:,:,nbg:] = (1+per_dev/100)*instanton	
		
	return rp_init

def inst_double_nm(instanton,dim=2,per_dev=2): #Needs to be redefined for higher dimensions
	nbg = len(instanton[0,0])
	nbeads = 2*nbg
	#print('nbeads after doubling', nbeads)
	FFT = nmtrans.FFT(dim, nbg)
	instanton = FFT.cart2mats(instanton)	
	
	rp_init = np.zeros((1,dim,nbeads))
	rp_init[:,:,::2] = (1+per_dev/100)*instanton	
	
	FFT = nmtrans.FFT(dim, nbeads)
	rp_init = FFT.mats2cart(rp_init)	
	
	return rp_init

def find_instanton(m,pes,qinit,beta,nbeads,ax,plt,plot=False,dim=2,scale=1.0,stepsize=1e-4,tol=1e-2):
	start_time = time.time()
	ens = Ensemble(beta=beta,ndim=dim)
	motion = Motion(symporder=2) 
	rng = np.random.default_rng(0) 
	therm = PILE_L() 
	propa = Symplectic_order_II()
		
	sim = RP_Simulation()

	pinit = rng.normal(size=qinit.shape)
	rp = RingPolymer(qcart=qinit,pcart=pinit,m=m)
	sim.bind(ens,motion,rng,rp,pes,propa,therm)	

	insta = inst(stepsize=stepsize,tol=tol)	
	insta.bind(pes,rp,cart=True)

	vals, vecs = insta.diag_hess()
		
	if(vals[0]>vals[1]/4):
		print('scaling')
		#scale = 8
		qscale = (insta.q.copy()).reshape((-1,dim,1))	
		qscale[:,0]/=scale
		insta.reinit_inst_scaled(qscale,scale)
		upd_step = partial(insta.slow_step_update_soft,scale,0)
	else:
		upd_step = insta.slow_step_update
	
	### Streambed walking to the barrier top
	step = insta.eigvec_follow_step_inst()	
	count = 0	
		
	while(step!="terminate"):
		upd_step(step,cart=True)  # Change here when the e.v. along saddle point coordinate is "soft"
		step = 	insta.eigvec_follow_step_inst()
		if(count%10000==0 and plot is True):
			#vals,vecs = np.linalg.eigh(pes.ddpot)
			#qarr.append(rp.q)
			#vecsarr.append(vecs)
			#grad = rp.dpot_cart+pes.dpot_cart
			#print('grad norm', np.sum(insta.grad**2)**0.5)
			ax.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:])
			plt.pause(0.01)
			#print('count',count)
		count+=1

	instant = rp.qcart
	print('time', time.time()-start_time)
	#print('qcart after evfollowing', rp.qcart)
	#print('dpot cart after evfollowing', rp.dpot_cart)	
	#print('pot after evfollowing', rp.pot)
	return instant
