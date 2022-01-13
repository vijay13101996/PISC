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
from PISC.engine.thermostat import PILE_L,Constrain_theta 
from PISC.engine.simulation import RP_Simulation, Matsubara_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata
import time

dim = 1
lamda = 0.8
g = 1/50.0
T = lamda*(0.5/np.pi)#5.0
m = 0.5
N = 10

nbeads = 16#32
nmats = 3 
rng = np.random.RandomState(2)
q = np.ones((N,dim,nbeads))
q[...,:nmats] = rng.normal(size=(N,dim,nmats)) #np.random.normal(size=(N,dim,nbeads))

p = np.ones_like(q)
p[...,:nmats] = rng.normal(size=(N,dim,nmats))
dt = 0.001
beta = 1/T

rngSeed = 1
rp = RingPolymer(q=q,p=p,m=m,nmats=nmats,mode='rp/mats')
ens = Ensemble(beta=beta,ndim=dim,theta=2.0)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(rngSeed) 

rp.bind(ens,motion,rng)

pes = double_well(lamda,g)
pes.bind(ens,rp)
therm = PILE_L(tau0=100.0,pile_lambda=1.0) 
therm.bind(rp,motion,rng,ens)

propa = Symplectic_order_II()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

# Sampling q
time_therm = 1.0
nthermsteps = int(time_therm/dt)
sim.step(ndt=nthermsteps,mode="nvt_theta",var='pq',pc=True)

rp.p[...,nmats:] = 0.0
rp.q[...,nmats:] = 0.0
	
# Constrain theta
theta_ens = Constrain_theta()
theta_ens.bind(rp,motion,rng,ens)
theta_ens.theta_constrained_randomize()

rp = RingPolymer(q=rp.q,p=rp.p,m=m,nmats=nmats,mode='rp/mats')
sim.bind(ens,motion,rng,rp,pes,propa,therm)

# Relax the distribution
time_relax = 20.0
nrelaxsteps = int(time_therm/dt)
sim.step(ndt = nrelaxsteps,mode="nve",var='pq')
#print('theta',rp.theta)
