import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.harmonic_2D import Harmonic
from PISC.potentials.Quartic import Quartic
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.harmonic_1D import harmonic
from PISC.potentials.Matsubara_M3_Quartic import Matsubara_Quartic 
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation, Matsubara_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata
import time

###Things to check:
#1. Time evolution of p,q without thermostat of a harmonic oscillator
#	for 2nd and 4th order integrators.
#2. Point 1 for Morse oscillator and comparison with standard 
#	integrators.
#3. Repeat 1 and 2 for the M matrix elements for sufficiently 
#	long time to check the stability and accuracy of the integrator.
#4. Do the same with a thermostat, for one MF Matsubara mode (Centroid)
#	and compare with results from 3

dim = 1
lamda = 0.8
g = 1/50.0
T = lamda*(0.5/np.pi)#5.0
m = 0.5
N = 1

nbeads = 16#32 
rng = np.random.RandomState(1)
qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = rng.normal(size=qcart.shape)#None
p = rng.normal(size=pcart.shape)
dt = 0.001
beta = 1/T

rngSeed = 6
rp = RingPolymer(qcart=qcart,m=m) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(rngSeed) 

rp.bind(ens,motion,rng)

pes = Quartic()#double_well(lamda,g)# harmonic(2.0*np.pi)#Harmonic(2*np.pi)#harmonic(2.0)#Harmonic(2*np.pi)#
pes.bind(ens,rp)
therm = PILE_L(tau0=100.0,pile_lambda=1.0) 
therm.bind(rp,motion,rng,ens)

### Scipy odeint integrator
propa = Symplectic_order_II()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)
start_time = time.time()
if(0):
	time_therm = 50.0
	nthermsteps = int(time_therm/dt)
	pmats = np.array([True for i in range(rp.nbeads)])
	for i in range(nthermsteps):
		sim.step(mode="nvt",var='pq',pmats=pmats)

	print('kin',rp.kin.sum()/N)

if(0):
	propa = Runge_Kutta_order_VIII()
	propa.bind(ens, motion, rp, pes, rng, therm)

	tarr = np.linspace(0,4,10000)
	sol = propa.integrate(tarr)
	Mqqcent = np.array(propa.centroid_Mqq(sol))
	Mqqcent = np.mean(abs(Mqqcent**2),axis=1)
		
	qcent = np.array(propa.ret_q(sol))
	print('q',qcent.shape)
	#qcent = qcent[:,30,0]

	#plt.plot(tarr,np.log(abs(Mqqcent[:,0,0]**2)))
	plt.plot(tarr,np.log(Mqqcent[:,0,0]),color='r')
	#for i in range(N):
	#	if (qcent[0,i,0]*qcent[9999,i,0] < 0.0):
	#		print('i',i)
			#plt.plot(tarr,Mqqcent[:,i,0])
	#plt.show()

tarr=[]
qarr=[]
potarr=[]
Mqqarr = []
Earr = []
dt = 0.01
gamma = 16

dt = dt/gamma

if(0): # CMD 	
	sim.t =0.0	
	rp = RingPolymer(qcart=rp.qcart,m=m,scaling=None,mode='rp',sgamma=gamma)
	motion = Motion(dt = dt,symporder=4) 
	rp.bind(ens,motion,rng)
	propa = Symplectic_order_IV()
	propa.bind(ens, motion, rp, pes, rng, therm)
	
	sim = RP_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)

	time_total = 10.0
	nsteps = int(time_total/dt)
	#pmats = np.array([True for i in range(rp.nbeads)])
	#pmats[:rp.nmats] = False	
		
	for i in range(nsteps):
		sim.step(mode="nvt",var='monodromy',pc=False)
		Mqq = np.mean(abs(rp.Mqq[:,0,0,0,0]**2)) #rp.Mqq[0,0,0,0,0]#
		tarr.append(i*dt)
		#qarr.append(propa.rp.p[0,0,0])
		#potarr.append(pes.ddpot[0,0,0,0,0])
		#Mqqarr.append(propa.rp.Mqq[0,0,0,0,0])
		Mqqarr.append(Mqq)
		#Earr.append(np.sum(pes.pot)+np.sum(rp.pot)+rp.kin)

	Mqqarr = np.array(Mqqarr)
	print('time',time.time()-start_time)


if(1): # Matsubara (N -> infty)
	sim.t =0.0	
	rp = RingPolymer(q=q,p=p,m=m,scaling=None,nmats=3,mode='mats')
	motion = Motion(dt = dt,symporder=2) 
	rp.bind(ens,motion,rng)

	pes = Matsubara_Quartic()#double_well(lamda,g)# harmonic(2.0*np.pi)#Harmonic(2*np.pi)#harmonic(2.0)#Harmonic(2*np.pi)#
	pes.bind(ens,rp)
	therm = PILE_L(tau0=100.0,pile_lambda=1.0) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_II()
	propa.bind(ens, motion, rp, pes, rng, therm)
	
	sim = Matsubara_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)
	#print('q p', rp.q[...,:3], rp.p[...,:3])
		
	time_total = 10.0
	nsteps = int(time_total/dt)
	#pmats = np.array([True for i in range(rp.nbeads)])
	#pmats[:rp.nmats] = False	
		
	for i in range(nsteps):
		sim.step(mode="nve",var='pq',pc=True)
		#print('q',rp.q[...,0])
		print('theta', rp.theta)
		#Mqq = np.mean(abs(rp.Mqq[:,0,0,0,0]**2)) #rp.Mqq[0,0,0,0,0]#
		tarr.append(i*dt)
		#qarr.append(propa.rp.p[0,0,0])
		#potarr.append(pes.ddpot[0,0,0,0,0])
		#Mqqarr.append(propa.rp.Mqq[0,0,0,0,0])
		#Mqqarr.append(Mqq)
		#Earr.append(np.sum(pes.pot)+np.sum(rp.pot)+rp.kin)

	Mqqarr = np.array(Mqqarr)
	print('time',time.time()-start_time)
	
	
