import numpy as np
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
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
from DW_Instanton import find_instanton_DW


### Potential parameters
m=0.5#0.5
dim=2
nbeads=1

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
D = 3*Vb
alpha = 0.37
print('Vb',Vb, 'D', D)

z = 1.0

pes = quartic_bistable(alpha,D,lamda,g,z)

#Only relevant for ring polymer and canonical simulations
Tc = 0.5*lamda/np.pi
times = 0.9
T = times*Tc
beta = 1/T

#Initialize p and q
E = 1.001*Vb 

#random initialization of p and q
q=0.003*np.ones((1,dim,nbeads))#todo:vary
p=0.001*np.ones((1,dim,nbeads))#todo:vary

### ------------------------------------------------------------------------------
#normalization of deviation
norm=0.001#CONVERGENCE PARAMETER#todo:vary
dt=0.001#CONVERGENCE PARAMETER#todo:vary
time_run=10#CONVERGENCE PARAMETER#todo:vary
tau=0.01#CONVERGENCE PARAMETER#todo:vary
def norm_dp_dq(dp,dq,norm=0.001):
    div=(1/norm)*np.linalg.norm(dp+dq)
    return dp/div,dq/div

def run_var(sim,dt,time_run,tau):
    tarr=[]
    varp=[]
    varq=[]
    mLCE=[]
    alpha = []
    nsteps = int(tau/dt)
    N=int(time_run/tau)
    for k in range(N):
        for i in range(nsteps):
            sim.step(mode="nve",var='variation')#pc=False?
        tarr.append(sim.t)
        varp.append(sim.rp.dp)
        varq.append(sim.rp.dq)
        dp_new,dq_new=norm_dp_dq(sim.rp.dp,sim.rp.dq)
        alpha.append((1/norm)*np.linalg.norm(sim.rp.dp[0]+sim.rp.dq[0]))
        sim.rp.dp=dp_new
        sim.rp.dq=dq_new
        #print(alpha)
        mLCE.append((1/sim.t)*np.sum(np.log(alpha)))
    return tarr, mLCE

#-------------------------------------------------------------------------------------------
fig,ax = plt.subplots()
xg = np.linspace(-5,5,int(1e2)+1)
yg = np.linspace(-3,7,int(1e2)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0.0,D,D/20))
nbeads = 32

#Gives a slightly unconverged instanton configuration
instanton = find_instanton_DW(nbeads,m,pes,beta,ax,plt,plot=True) 

#Gives instanton configuration upto an accuracy of 1e-4
inst_opt = find_instanton_DW(32,m,pes,beta,ax,plt,plot=True,step=1e-6,tol=1e-4,nb_start=32) 

#--------------------------------------------------------------------------------------------

#initialize small deviation
dq=0.001*np.ones_like(q)#todo:vary
dp=0.5*np.ones_like(p)#todo:vary
dp,dq=norm_dp_dq(dp,dq,norm)
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

tarr,mLCE=run_var(sim,dt,time_run,tau)
plt.plot(tarr,mLCE)
plt.show()
#Initialize trajectory at saddle point with almost zero velocity
#Set dq, dp to a certain predefined norm (dp and dq are together w so consider that when normalizing)
#Propagate trajectories in the 'variation' mode for time 'T'
#'Renormalize' the perturbations, reinitialize the rp class and rerun trajectories for time T
#Do this M times and find Lyapunov exponent
