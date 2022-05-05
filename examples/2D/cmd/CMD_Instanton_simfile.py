import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.potentials.Adams_function import adams_function
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.instanton import instantonize
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time

### Potential parameters
m=0.5#8.0

w = 0.2#0.1
D = 5.0#10.0
alpha = (0.5*m*w**2/D)**0.5#1.0#1.95

lamda = 0.8 #4.0
g = 0.02#4.0

z = 0.0#2.3	

pes = quartic_bistable(alpha,D,lamda,g,z)

### Simulation parameters
T_au = 0.95*lamda*0.5/np.pi
beta = 1.0/T_au 
print('T in au',T_au)

N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 20.0
time_total = 10.0#15.0
time_relax = 10.0

rngSeed = 1
N = 1
dim = 2
nbeads = 1
T = 1.0	

### Plot extent and axis
L = 7.0
lbx = -L
ubx = L
lby = -6*L
uby = 12*L
ngrid = 200
ngridx = ngrid
ngridy = ngrid

xgrid = np.linspace(lbx,ubx,200)
ygrid = np.linspace(lby,uby,200)
x,y = np.meshgrid(xgrid,ygrid)

if(0): ### Setting up the ring polymer
	rng = np.random.RandomState(1)
	qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
	q = np.random.normal(size=(N,dim,nbeads))
	M = np.random.normal(size=(N,dim,nbeads))

	pcart = None
	beta = 1/T

	rp = RingPolymer(qcart=qcart,m=m,nmats=nbeads) 
	ens = Ensemble(beta=beta,ndim=dim)
	motion = Motion(dt = dt,symporder=2) 
	rng = np.random.default_rng(rngSeed) 

	pes = quartic_bistable(alpha,D,lamda,g,z)#adams_function()#

	print('T',T)
	nsteps_therm = int(time_therm/dt)
	nsteps_relax = int(time_relax/dt)

	therm = PILE_L(tau0=0.1,pile_lambda=100.0) 

	propa = Symplectic_order_II()
		
	sim = RP_Simulation()

	start_time = time.time()
	q = np.zeros((N,dim,nbeads))
	x0 = 0.5
	y0 = 0.5
	q[...,0,0] = x0
	q[...,1,0] = y0
	p = rng.normal(size=q.shape)
	p[...,0] = 0.0	
	rp = RingPolymer(q=q,p=p,m=m,nmats=1,sgamma=1)
	sim.bind(ens,motion,rng,rp,pes,propa,therm)	


potgrid = pes.potential_xy(x,y)
plt.contour(x,y,potgrid,colors='k',levels=np.arange(0,1.5*D,D/20))
print('pot',pes.potential_xy(0,0))
#plt.show()

if(0): #Thermalize
	for i in range(nsteps_therm):
		sim.step(mode="nvt",var='pq')
		#if(i%100==0):
			#plt.scatter(rp.q[0,0,0],rp.q[0,1,0])
			#plt.pause(0.01)


if(0): # Code for finding the saddle in the PE # Code for finding the saddle in the PESS	
	inst = instantonize(stepsize=1e-5,tol=1e-2)
	inst.bind(pes,rp)

	### Gradient descent to get to a minima
	step = inst.grad_desc_step()#eigvec_follow_step(eig_dir=0)		
	if(1):
		count = 0
		while(step!="terminate"):
			inst.slow_step_update(step)
			step = 	inst.grad_desc_step()#inst.eigvec_follow_step(eig_dir=0)
			#if(count%100==0):
			#	plt.scatter(rp.q[0,0,0],rp.q[0,1,0],color='m')
			#	plt.pause(0.01)
			count+=1

	### Saddle point finding using Eigenvector following
	inst.tol=1e-12
	inst.stepsize=1e-5
	step = inst.eigvec_follow_step(eig_dir=0)
	rp.q[0,0,0]+= 5e-2
	rp.q[0,1,0]+= 5e-2 
	rp.mats2cart()
	pes.update()
	plt.scatter(rp.q[0,0,0],rp.q[0,1,0])	

	if(1):
		count=0
		while(step!="terminate"):
			inst.slow_step_update(step)
			step = 	inst.eigvec_follow_step(eig_dir=0)
			#if(count%3000==0):
			#	plt.scatter(rp.q[0,0,0],rp.q[0,1,0],color='c')
			#	plt.pause(0.01)
			##	print('count',count)
			count+=1
		print('count',count)
		plt.scatter(rp.q[0,0,0],rp.q[0,1,0],color='k')
	#plt.show()
	vals,vecs = inst.diag_hess()
	eigvec = -vecs[0]
	print('eigvec',eigvec)
	print('eigval',vals)

eigvec = np.array([1.0,0.0])
nbeads = 1
dim = 2
inst = instantonize(stepsize=1e-4, tol=1e-8)
qcart = np.zeros((1,dim,nbeads))
rp = RingPolymer(qcart=qcart,m=m,nmats=nbeads)
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(rngSeed) 

rp.bind(ens, motion, rng)
pes.bind(ens, rp)
inst.bind(pes,rp)

rp_init = inst.saddle_init(np.array([0.0,0.0]),0.0,eigvec)
rp = RingPolymer(qcart=rp_init,m=m,nmats=nbeads)
rp.bind(ens, motion, rng)
pes.bind(ens, rp)
pes.update()

inst.bind(pes,rp)

plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:],color='r')	
vals,vecs = inst.diag_hess()
eigdir =1 

print('hess',inst.hess,pes.ddpot)
print('eigval',vals[:2])
step = inst.grad_desc_step()
if(1): # Gradient descent helps move away from the classical saddle to a lower energy RP configuration below crossover temperature.
	count=0
	while(step!="terminate" and abs(rp.q[0,0,0])<1e-1):
		inst.slow_step_update(step)
		step = 	inst.grad_desc_step()
		eigval = np.linalg.eigvalsh(inst.hess)#[eigdir]
		if(count%1000==0):
			print('eigval',eigval[:2])
		#if(count%1000==0):
		#	plt.scatter(count,(pes.pot+rp.pot).sum(),color='r')
		#	plt.scatter(count,abs(inst.grad).max(),color='k')
		#	plt.pause(0.05)
			#print('grad',abs(inst.grad).max())
			#print('pot',(pes.pot+rp.pot).sum())
		if(0):#count%10000==0):
			plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:])
			plt.pause(0.05)
			#print('count',count)
		if(0):#count%5000==0):
			print(count,'grad',np.linalg.eigvalsh(inst.hess)[eigdir])
			print('step', np.around(np.linalg.norm(step),5))
			
		count+=1
	print('count',count)
	print('gradient',np.around(inst.grad,4))
	plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:],color='m')

#plt.show()
step = inst.eigvec_follow_step(eig_dir=eigdir)
eigval = np.linalg.eigvalsh(inst.hess)[eigdir]
inst.stepsize=1e-3
inst.tol=5e-6

gradmax = abs(inst.grad).max()

if(0):
	count=0
	while(step!="terminate"):# and eigval>-0.1):
		inst.slow_step_update(step)
		step = 	inst.eigvec_follow_step(eig_dir=eigdir)
		eigval = np.linalg.eigvalsh(inst.hess)#[eigdir]
		print('eigval',eigval[:2])
		if(0):#count%10000==0):
			plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:])
			plt.pause(0.05)
			#print('count',count)
		if(0):#count%5000==0):
			print(count,'grad',abs(inst.grad).max())
			#eigval = np.linalg.eigvalsh(inst.hess)#[eigdir]	
			#print(count,'vals',eigval)
			#print('step', count,np.around(np.linalg.norm(step),5))
		count+=1
	print('count',count)
	print('gradient',np.around(inst.grad,6))
	eigval = np.linalg.eigvalsh(inst.hess)[eigdir]	
	print('eigval',eigval)
	plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:],color='k')

plt.show()

