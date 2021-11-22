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
from PISC.engine.simulation import Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import thermalize_PILE_L
from thermalize_PILE_L import thermalize_rp
import pickle

dim = 1
lamda = 0.8#1.5
g = 1/50.0
Tc = lamda*(0.5/np.pi)#5.0
T = 1*Tc
print('T',T)
m = 0.5
N = 1000

nbeads = 32#32 
rng = np.random.RandomState(1)
qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = None
dt = 0.005
beta = 1/T

rngSeed = 1
rp = RingPolymer(qcart=qcart,m=m,nmats=nbeads) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(rngSeed) 

rp.bind(ens,motion,rng)

potkey = 'inv_harmonic_lambda_{}'.format(lamda)	
pes = double_well(lamda,g)#harmonic(2.0*np.pi)# Harmonic(2*np.pi)#harmonic(2.0)#Harmonic(2*np.pi)#
pes.bind(ens,rp)

rp.p = rp.q*0.0
rp.pcart = rp.qcart*0.0

time_therm = 20.0
time_relax = 5.0
nsteps_therm = int(time_therm/dt)
nsteps_relax = int(time_relax/dt)

pmats = np.array([True for i in range(rp.nbeads)])
pmats[0] = False

qgrid = np.linspace(-10.0,10.0,101)
fgrid = np.zeros_like(qgrid)

nsample = 5

therm = PILE_L(tau0=0.1,pile_lambda=10.0) 
therm.bind(rp,motion,rng,ens)

propa = Symplectic_order_II()
propa.bind(ens, motion, rp, pes, rng, therm)
		
sim = Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

start_time = time.time()
for k in range(len(qgrid)):
	rp.q[:] = qgrid[k]
	rp.qcart[:] = qgrid[k]
	#print('q before',rp.q[0,0,0],pes.dpot[0,0,0])
	sim.step(ndt=nsteps_therm,mode="nvt",var='pq',pmats=pmats)
	#print('q after',rp.q[0,0,0],pes.dpot[0,0,0])
	for i in range(nsample):
		sim.step(ndt=nsteps_relax,mode="nvt",var='pq',pmats=pmats)
		fgrid[k]+=np.mean(pes.dpot[...,0])	
	print('k',k)	
	print('time',time.time()-start_time)

fgrid/=nsample

fname = 'CMD_PMF_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_thermtime_{}_relaxtime_{}_nsample_{}_seed_{}'.format(potkey,T,N,nbeads,gamma,dt,time_therm,time_relax,nsample,rngSeed)
store_1D_plotdata(qgrid,fgrid,fname)
			
plt.plot(qgrid,fgrid)
plt.plot(qgrid,pes.ret_force(qgrid))
plt.show()
			
