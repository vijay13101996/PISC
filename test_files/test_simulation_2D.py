import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.harmonic_1D import harmonic
from PISC.potentials.Matsubara_M3_Quartic import Matsubara_Quartic
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation, Matsubara_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata
import time     
                
dim = 2         
omega = 1
g0 = 0.0#1/10.0
T = 1.0 #lamda*(0.5/np.pi)#5.0
m = 0.5         
N = 10           

nbeads = 4#32 
rng = np.random.RandomState(1)
qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = rng.normal(size=qcart.shape)#None
p = rng.normal(size=pcart.shape)
dt = 0.001
beta = 1/T

gamma=16
rngSeed = 6
rp = RingPolymer(qcart=qcart,m=m)
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2)
rng = np.random.default_rng(rngSeed)

rp.bind(ens,motion,rng)

pes = coupled_harmonic(omega,g0)
pes.bind(ens,rp)
pes.update()

print('dpotential', pes.dpotential(qcart).shape)
print('ddpotential',pes.compute_hessian()[0,0,1])

therm = PILE_L(tau0=100.0,pile_lambda=1.0)
therm.bind(rp,motion,rng,ens)

propa = Symplectic_order_II()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

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

tarr=[]
Mqqarr=[]
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


