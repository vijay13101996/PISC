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
from matplotlib import pyplot as plt
### Things to check:
#1. Evolution at the end of one full time step for 2nd and 4th order
#   symplectic integrator - Done (for much longer times)
#2. RSP matrix integration vs normal integration - 
#3. Time evolution of p,q and M without a thermostat for the harmonic oscillator 
#	for one time step - Done (for much longer times)
#4. Point 3 for a morse oscillator - Done again!

dim = 2
T = 2.0
m = 0.5
N=1

nbeads = 5 
rng = np.random.RandomState(1)
qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = None
dt = 0.001
beta = 1/T

rp = RingPolymer(qcart=qcart,m=m,mode='MFmats',nmats=3) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(1) 

rp.bind(ens,motion,rng)

pes = Harmonic(2*np.pi)#double_well(1.0,1/50.0)#harmonic(2.0)#Harmonic(2*np.pi)#
pes.bind(ens,rp)
therm = PILE_L(tau0=100.0,pile_lambda=1.0) 

### Scipy odeint integrator
propa = Runge_Kutta_order_VIII()
propa.bind(ens, motion, rp, pes, rng, therm)

tarr = np.linspace(0,4,1000)
sol = propa.integrate(tarr)
Mqqcent = np.array(propa.centroid_Mqq(sol))

#plt.plot(tarr,np.log(abs(Mqqcent[:,0,0,0]**2)),color='r')
#plt.show()
if(0): ### Symplectic 2nd order
	rp = RingPolymer(qcart=qcart,m=m,mode='MFmats',nmats=3) 
	ens = Ensemble(beta=beta,ndim=dim)
	motion = Motion(dt = dt,symporder=2) 
	rng = np.random.default_rng(1) 

	rp.bind(ens,motion,rng)

	pes.bind(ens,rp)
	therm = PILE_L(tau0=100.0,pile_lambda=1.0) 

	propa =Symplectic_order_II()

	propa.bind(ens, motion, rp, pes, rng, therm)

	tarr=[]
	qarr=[]
	potarr=[]
	Mqqarr = []
	Earr = []
	time = 4.0
	nsteps = int(time/dt)
	
	for i in range(nsteps):
		propa.Monodromy_step()
		tarr.append(i*dt)
		qarr.append(propa.rp.q[0,0,0])
		potarr.append(pes.ddpot[0,0,0,0,0])
		Mqqarr.append(propa.rp.Mqq[0,0,0,0,0])
		Earr.append(np.sum(pes.pot)+np.sum(rp.pot)+rp.kin)
	
	#plt.plot(tarr,qarr)
	#plt.plot(tarr,potarr)
	#plt.plot(tarr,np.log(abs(np.array(Mqqarr)**2)),color='b')
	plt.plot(tarr,Earr,color='b')
	
if(1): ### Symplectic 4th order
	rp = RingPolymer(qcart=qcart,m=m,mode='MFmats',nmats=3) 
	ens = Ensemble(beta=beta,ndim=dim)
	motion = Motion(dt = dt,symporder=4) 
	rng = np.random.default_rng(1) 

	rp.bind(ens,motion,rng)

	pes.bind(ens,rp)
	therm = PILE_L(tau0=100.0,pile_lambda=1.0) 

	propa =Symplectic_order_IV()

	propa.bind(ens, motion, rp, pes, rng, therm)

	tarr=[]
	qarr=[]
	potarr=[]
	Mqqarr = []

	time = 4.0
	nsteps = int(time/dt)
	for i in range(nsteps):
		propa.Monodromy_step()
		tarr.append(i*dt)
		qarr.append(propa.rp.q[0,0,0])
		potarr.append(pes.ddpot[0,0,0,0,0])
		Mqqarr.append(propa.rp.Mqq[0,0,0,0,0])
		print('Energy',np.sum(rp.pot)+np.sum(pes.pot)+rp.kin)
	
	#plt.plot(tarr,qarr)
	#plt.plot(tarr,potarr)
	#plt.plot(tarr,np.log(abs(np.array(Mqqarr)**2)),color='g')
	plt.plot(tarr,Earr,color='r')
plt.show()
