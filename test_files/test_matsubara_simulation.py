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

dim = 1
lamda = 0.8
g = 1/50.0
T = lamda*(0.5/np.pi)#5.0
m = 0.5
N = 1

nbeads = 3#32 
nmats = 3
rng = np.random.RandomState(1)
#qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
q = rng.normal(size=(N,dim,nbeads))
#M = np.random.normal(size=(N,dim,nbeads))

#pcart = rng.normal(size=qcart.shape)#None
p = rng.normal(size=q.shape)
dt = 0.001
beta = 1/T

print('q,p',q,p)

rngSeed = 6
rp = RingPolymer(q=q,p=p,m=m,nmats=3,mode='mats') 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(rngSeed) 

rp.bind(ens,motion,rng)

pes = Matsubara_Quartic()#double_well(lamda,g)# harmonic(2.0*np.pi)#Harmonic(2*np.pi)#harmonic(2.0)#Harmonic(2*np.pi)#
pes.bind(ens,rp)
therm = PILE_L(tau0=100.0,pile_lambda=1.0) 
therm.bind(rp,motion,rng,ens)

propa = Symplectic_order_II()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = Matsubara_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)
start_time = time.time()
if(1):
	time_therm = 50.0
	nthermsteps = int(time_therm/dt)
	for i in range(nthermsteps):
		sim.step(mode="nvt",var='pq',pc=True)

	print('kin',rp.kin.sum()/N)

sim.t =0.0	
rp = RingPolymer(q=q,p=p,m=m,scaling=None,nmats=3,mode='mats')
motion = Motion(dt = dt,symporder=2) 
rp.bind(ens,motion,rng)

sim = Matsubara_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

time_total = 1.0
nsteps = int(time_total/dt)

tarr = []

for i in range(nsteps):
	sim.step(mode="nve",var='pq',pc=True)
	#print('q',rp.q[...,0])
	print('ddpot',pes.ddpot)
	#print('theta', rp.theta)
	#Mqq = np.mean(abs(rp.Mqq[:,0,0,0,0]**2)) #rp.Mqq[0,0,0,0,0]#
	tarr.append(i*dt)
	#qarr.append(propa.rp.p[0,0,0])
	#potarr.append(pes.ddpot[0,0,0,0,0])
	#Mqqarr.append(propa.rp.Mqq[0,0,0,0,0])
	#Mqqarr.append(Mqq)
	#Earr.append(np.sum(pes.pot)+np.sum(rp.pot)+rp.kin)

#Mqqarr = np.array(Mqqarr)
print('time',time.time()-start_time)


