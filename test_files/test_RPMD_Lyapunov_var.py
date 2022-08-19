import numpy as np
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
from PISC.engine.Poincare_section import Poincare_SOS
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
import os


### Potential parameters
m=1
dim=2
g=0
lamda=1
pes = henon_heiles(lamda,g)

#Only relevant for ring polymer and canonical simulations
Tc = 0.5*lamda/np.pi
times = 1#0.9
T = times*Tc
beta = 1/T

path = os.path.dirname(os.path.abspath(__file__))

### ------------------------------------------------------------------------------
def norm_dp_dq(dp,dq,norm=0.001):
    div=(1/norm)*np.linalg.norm(np.concatenate((dp,dq)))
    return dp/div,dq/div

def run_var(sim,dt,time_run,tau,norm):
	tarr=[]
	alpha=[]
	mLCE=[]
	qx_cent=[]
	qy_cent=[]
	nsteps = int(tau/dt)
	N=int(time_run/tau)
	for k in range(N):
		#calculate initial w
		dp_cent=sim.rp.dp[...,0]
		dq_cent=sim.rp.dq[...,0]
		prev=np.linalg.norm(np.concatenate((dp_cent,dq_cent)))
		#propagate
		for i in range(nsteps):
			sim.step(mode="nve",var='variation')
		#calculate centroid
		cent_xy=sim.rp.q[...,0]/nbeads**0.5
		qx_cent.append(cent_xy[0,0])
		qy_cent.append(cent_xy[0,1])
		#calculate w after nsteps
		dp_cent=sim.rp.dp[...,0]
		dq_cent=sim.rp.dq[...,0]
		current=np.linalg.norm(np.concatenate((dp_cent,dq_cent)))
		alpha.append(current/prev)
		#update
		sim.rp.dp,sim.rp.dq=norm_dp_dq(sim.rp.dp,sim.rp.dq,norm=norm)
		sim.rp.mats2cart()
		tarr.append(sim.t)
		mLCE.append((1/sim.t)*np.sum(np.log(alpha)))
	return tarr,mLCE,qx_cent,qy_cent

#-------------------------------------------------------------------------------------------
#Display where the Instanton is
#fig,ax = plt.subplots()
xg = np.linspace(-5,5,int(1e2)+1)
yg = np.linspace(-3,7,int(1e2)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0.0,D,D/20))

#CONVERGENCE PARAMETERS
norm=0.01#small enough s.t. deviation is actually small
dt=0.001#small enough s.t. simulation correct
time_run=600#long enough to get to plateau
tau=0.05#a bit ore than one order of magnitude bigger than dt
nbeads=1#32

if(False):#get instanton
	try:
		q = read_arr('Instanton_beta_{}_nbeads_{}_z_{}'.format(beta,nbeads,z),'{}/Datafiles'.format(path)) #instanton
	except:
		instanton = find_instanton_DW(nbeads,m,pes,beta,ax=None,plt=None,plot=False,path=path) 
		q = read_arr('Instanton_beta_{}_nbeads_{}_z_{}'.format(beta,nbeads,z),'{}/Datafiles'.format(path)) #instanton
	#q = np.zeros((1,dim,nbeads))
	q[:,0,:]+=1e-4

	#Trial initialization of instanton 
	#q=np.zeros((1,dim,nbeads))
	#q[...,0] = 0.0
	p=np.zeros_like(q)#or initialize differently
	p[:,0,:]=-1e-4
q=np.zeros((1,2,1))
p=np.zeros((1,2,1))
#set like in paper
#regular orbit, R1:#expect mLCE=0 #got 0.005 but decreases monotonically. One order of magnitude smaller than C1
if(True):
	q[0,0,0]=0
	p[0,0,0]=0.2334
	q[0,1,0]=0.558
	p[0,1,0]=0
#chaotic orbit, C1:#expect mLCE=0.045 for long times (got 0.055) but is supposed to decrease monotonically
if(False):
	q[0,0,0]=0
	p[0,0,0]=0.4208
	q[0,1,0]=-0.25
	p[0,1,0]=0
#initialize small deviation
rng = np.random.default_rng(10)
dq=np.zeros_like(q)
dq=rng.standard_normal(np.shape(q))
dq[:,0,0]=1e-1
dp=np.zeros_like(p)
dp=rng.standard_normal(np.shape(q))
dp[:,0,0]=1e-1
dp,dq=norm_dp_dq(dp,dq,norm)
#print('dq,dp',dp,dq)

#--------------------------------------------------------------------------------------------

#set up the simulation
sim=RP_Simulation()
rngSeed=range(1)
rng = np.random.default_rng(rngSeed)
ens = Ensemble(beta=1/T,ndim=dim)
motion = Motion(dt=dt,symporder=2)#4
propa = Symplectic_order_II()#IV
rp=RingPolymer(pcart=p,qcart=q,dp=dp,dq=dq,m=m)
rp.bind(ens,motion,rng)
therm = PILE_L(tau0=1.0,pile_lambda=100.0)#only important due to initalization 
sim.bind(ens,motion,rng,rp,pes,propa,therm)

tarr,mLCE,qx_cent,qy_cent=run_var(sim,dt,time_run,tau,norm)
fig,ax=plt.subplots(3,sharex=True)
ax[0].plot(tarr,mLCE)
ax[1].plot(tarr,qx_cent)
ax[2].plot(tarr,qy_cent)
plt.show()

#Initialize trajectory at saddle point with almost zero velocity
#Set dq, dp to a certain predefined norm (dp and dq are together w so consider that when normalizing)
#Propagate trajectories in the 'variation' mode for time 'T'
#'Renormalize' the perturbations, reinitialize the rp class and rerun trajectories for time T
#Do this M times and find Lyapunov exponent
