import numpy as np
import sys
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Henon_Heiles import henon_heiles
from PISC.engine.PI_sim_core import SimUniverse
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV
from PISC.engine.beads import RingPolymer
from PISC.utils.nmtrans import FFT
import os

#Henon Heiles potential parameters
mass = 1.0
dim = 2

lamda = 1.0
g = 0.0

pes = henon_heiles(lamda, g)

#Only relevant for ring polymer and canonical simulations
Tc = 0.5*lamda/np.pi
times = 0.7
T = times*Tc
beta = 1/T

potkey = 'henon_heiles_2D_lamda_{}_g_{}'.format(lamda, g)	
Tkey = 'T_{}Tc'.format(times)

#Initialize p and q
E = 0.125
q[0, 0] = 0.0
q[0, 1] = 0.558

path = os.path.dirname(os.path.abspath(__file__))

### ------------------------------------------------------------------------------
def norm_dp_dq(dp, dq, norm = 0.001):
    div=(1/norm) * np.linalg.norm(np.concatenate((dp, dq)))
    return dp/div, dq/div

def run_var(sim, dt, time_run, tau, norm):
	tarr = []
	mLCE = []
	alpha = []
	qx_cent = []
	qy_cent = []
	nsteps = int(tau/dt)
	N = int(time_run/tau)
	for k in range(N):
		#Calculate initial w
		dp_cent = sim.rp.dp[...,0]
		dq_cent = sim.rp.dq[...,0]
		prev = np.linalg.norm(np.concatenate((dp_cent,dq_cent))) #Norm of the deviation vector from a given orbit at time t=0
		#Propagation
		for i in range(nsteps):
			sim.step(mode="nve",var='variation')
		#Calculate centroid
		cent_xy = sim.rp.q[...,0]/nbeads**0.5
		qx_cent.append(cent_xy[0,0])
		qy_cent.append(cent_xy[0,1])
		#Calculate w after nsteps
		dp_cent = sim.rp.dp[...,0]
		dq_cent = sim.rp.dq[...,0]
		current = np.linalg.norm(np.concatenate((dp_cent,dq_cent))) #Norm of the deviation vector from a given orbit at time t
		alpha.append(current/prev)
		#Update
		sim.rp.dp,sim.rp.dq=norm_dp_dq(sim.rp.dp,sim.rp.dq,norm=norm)
		sim.rp.mats2cart()
		tarr.append(sim.t)
		mLCE.append((1/sim.t)*np.sum(np.log(alpha)))
	return tarr,mLCE,qx_cent,qy_cent

#-------------------------------------------------------------------------------------------
#CONVERGENCE PARAMETERS
norm = 0.001 #CONVERGENCE PARAMETER#todo:vary
dt = 0.005 #CONVERGENCE PARAMETER#todo:vary
time_run = 40 #CONVERGENCE PARAMETER#todo:vary
tau = 0.05 #CONVERGENCE PARAMETER#todo:vary
nbeads = 1 #Classical dynamics

qcart = []
pcart = np.zeros_like(qcart)
	
rng = np.random.default_rng(rngSeed)
	
dq=np.zeros_like(qcart)
dp=np.zeros_like(pcart)
dq[:,0,0]=-1e-4
dp[:,0,0]=1e-4

dp[:,1,0]=1e-4*rng.uniform(-1,1)
dq[:,1,0]=1e-4*rng.uniform(-1,1)

#Set up the simulation
sim=RP_Simulation()
ens = Ensemble(beta=1/T,ndim=dim)
motion = Motion(dt=dt,symporder=2)#4
propa = Symplectic_order_II()#IV
rp=RingPolymer(pcart=pcart,qcart=qcart,dp=dp,dq=dq,m=m)
rp.bind(ens,motion,rng)
therm = PILE_L(tau0=1.0,pile_lambda=100.0)#only important due to initalization 
sim.bind(ens,motion,rng,rp,pes,propa,therm)

tarr,mLCE,qx_cent,qy_cent=run_var(sim,dt,time_run,tau,norm)
lmax = max(mLCE)

#plt.scatter(T_arr,2*np.pi*T_arr*Tc)
#plt.show()
