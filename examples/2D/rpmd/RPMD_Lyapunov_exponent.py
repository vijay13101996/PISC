import numpy as np
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
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
from plt_util import prepare_fig, prepare_fig_ax

### Potential parameters
m=0.5#0.5
dim=2

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
times = 0.9#0.9
T = times*Tc
beta = 1/T

#Initialize p and q
E = 1.001*Vb 

path = os.path.dirname(os.path.abspath(__file__))

### ------------------------------------------------------------------------------
def norm_dp_dq(dp,dq,norm=0.001):
    div=(1/norm)*np.linalg.norm(np.concatenate((dp,dq)))
    return dp/div,dq/div

def run_var(sim,dt,time_run,tau,norm):
	tarr=[]
	mLCE=[]
	alpha = []
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
norm=0.01#CONVERGENCE PARAMETER#todo:vary
dt=0.001#CONVERGENCE PARAMETER#todo:vary
time_run=10#CONVERGENCE PARAMETER#todo:vary
tau=0.001#CONVERGENCE PARAMETER#todo:vary
nbeads=32#32

#Gives a slightly unconverged instanton configuration
#instanton = find_instanton_DW(nbeads,m,pes,beta,ax=None,plt=None,plot=False,path=path) 

#Gives instanton configuration upto an accuracy of 1e-5
tol=1e-5
#inst_opt = find_instanton_DW(32,m,pes,beta,ax=None,plt=None,plot=False,step=1e-7,tol=tol,nb_start=32,qinit=instanton,path=path)
#print('inst_opt',inst_opt)
try:
	q = read_arr('Instanton_beta_{}_nbeads_{}_z_{}_tol_{}'.format(beta,nbeads,z,tol),'{}/Datafiles'.format(path)) #instanton
except:
	instanton_bad = find_instanton_DW(nbeads,m,pes,beta,ax=None,plt=None,plot=False,path=path)    
	instanton = find_instanton_DW(nbeads,m,pes,beta,ax=None,plt=None,plot=False,path=path,step=1e-7,qinit=instanton_bad)
	q = read_arr('Instanton_beta_{}_nbeads_{}_z_{}_tol_{}'.format(beta,nbeads,z,tol),'{}/Datafiles'.format(path)) #instanton
#q = np.zeros((1,dim,nbeads))
q[:,0,:]+=1e-4

#Trial initialization of instanton 
#q=np.zeros((1,dim,nbeads))
#q[...,0] = 0.0
p=np.zeros_like(q)#or initialize differently
p[:,0,:]=-1e-4

#initialize small deviation
rng = np.random.default_rng(10)
dq=np.zeros_like(q)
dq=rng.standard_normal(np.shape(q))#todo:vary
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
fig2,ax2=prepare_fig_ax(tex=True)
ax2.plot(tarr,mLCE)
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$X(t)$')
file_dpi=600
fig2.savefig('plots/convergence_mLCE_%i_beads.pdf'%nbeads,format='pdf',bbox_inches='tight', dpi=file_dpi)
fig2.savefig('plots/convergence_mLCE_%i_beads.png'%nbeads,format='png',bbox_inches='tight', dpi=file_dpi)
plt.show()

#Initialize trajectory at saddle point with almost zero velocity
#Set dq, dp to a certain predefined norm (dp and dq are together w so consider that when normalizing)
#Propagate trajectories in the 'variation' mode for time 'T'
#'Renormalize' the perturbations, reinitialize the rp class and rerun trajectories for time T
#Do this M times and find Lyapunov exponent
