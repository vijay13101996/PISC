import numpy as np
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.engine.PI_sim_core import SimUniverse
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from Saddle_point_finder import separatrix_path, find_minima
import time
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV
from PISC.engine.beads import RingPolymer
import os
from plt_util import prepare_fig, prepare_fig_ax

def norm_dp_dq(dp,dq,norm):
    div=(1/norm)*np.linalg.norm(np.concatenate((dp,dq)))
    return dp/div,dq/div
def run_var(sim,dt,time_run,tau,norm):
    tarr=[]
    mLCE=[]
    qx=[]
    #qy=[]
    alpha = []
    nsteps = int(tau/dt)
    N=int(time_run/tau)
    for k in range(N):
        prev=np.linalg.norm(np.concatenate((sim.rp.dp[0],sim.rp.dq[0])))
        for i in range(nsteps):
            sim.step(mode="nve",var='variation')#pc=False?
        tarr.append(sim.t)
        current=np.linalg.norm(np.concatenate((sim.rp.dp[0],sim.rp.dq[0])))
        alpha.append(current/prev)
        prev=current
        sim.rp.dp,sim.rp.dq=norm_dp_dq(sim.rp.dp,sim.rp.dq,norm)
        mLCE.append((1/sim.t)*np.sum(np.log(alpha)))
        qx.append(sim.rp.qcart[0,0,0])
        #qy.append(sim.rp.q[0,1,0])
    return tarr, mLCE,qx
#--------------------------------------------------------------------------------------------

### Potential parameters
m=0.5#0.5
dim=2
nbeads=1

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
D = 3*Vb
alpha = 0.38

z = 1.0

pes = quartic_bistable(alpha,D,lamda,g,z)
#Only for RP and Canonic simulations important
Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc
### ------------------------------------------------------------------------------
#CONVERGENCE PARAMETERS
norm=0.001
dt=0.001
time_run=100
tau=0.05
#--------------------------------------------------------------------------------------------

#initialization of p and q
q=0.001*np.ones((1,dim,nbeads))
p=-0.001*np.ones((1,dim,nbeads))
if(False):#manual init
    q[0,0,0]=0.01#x
    p[0,0,0]=-0.01#px
    q[0,1,0]=0#y
    p[0,1,0]=0#py
if(False):#random init
    rng = np.random.default_rng(10)
    q=0.01*rng.standard_normal(np.shape(q))
    p=0.01*rng.standard_normal(np.shape(q))
if(True):#get to 2 (maximal lyapunov)
    q=np.zeros((1,dim,nbeads))
    p=np.zeros((1,dim,nbeads))
    #q[0,0,0]=0
    #p[0,0,0]=0

#initialize small deviation
rng = np.random.default_rng(10)
dq=np.zeros_like(q)
dq=rng.standard_normal(np.shape(q))#todo:vary
dq[:,0,0]=1e-1
dp=np.zeros_like(p)
dp=rng.standard_normal(np.shape(q))
dp[:,0,0]=1e-1
dp,dq=norm_dp_dq(dp,dq,norm)
### ------------------------------------------------------------------------------

#set up the simulation
sim=RP_Simulation()
rngSeed=range(1)
rng = np.random.default_rng(rngSeed)
ens = Ensemble(beta=1/T,ndim=dim)
motion = Motion(dt=dt,symporder=2)#4
propa = Symplectic_order_II()#IV
rp=RingPolymer(pcart=p,qcart=q,dpcart=dp,dqcart=dq,m=m)
rp.bind(ens,motion,rng)
therm = PILE_L(tau0=1.0,pile_lambda=100.0)#only important due to initalization 
sim.bind(ens,motion,rng,rp,pes,propa,therm)

tarr,mLCE,qx=run_var(sim,dt,time_run,tau,norm)
fig,ax=prepare_fig_ax(tex=True)#plt.subplots(2,sharex=True)
ax.plot(tarr,mLCE)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$X(t)$')
#ax[1].plot(tarr,qx)
plt.show()
#Initialize trajectory at saddle point with almost zero velocity
#Set dq, dp to a certain predefined norm (dp and dq are together w so consider that when normalizing)
#Propagate trajectories in the 'variation' mode for time 'T'
#'Renormalize' the perturbations, reinitialize the rp class and rerun trajectories for time T
#Do this M times and find Lyapunov exponent
