import numpy as np
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials import quartic_bistable
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

#Start execution time

seconds = time.time()

### Potential parameters
m = 0.5
dim = 2
nbeads = 64

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
D = 3*Vb
alpha = 0.382
print('Vb',Vb, 'D', D)

z = 1.0

pes = quartic_bistable(alpha,D,lamda,g,z)

#Only relevant for ring polymer and canonical simulations
T = 0.5*lamda/np.pi

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)    

#Initialization of p and q
q=np.zeros((1, dim, nbeads))#todo:vary
p=np.zeros((1, dim, nbeads))#todo:vary
#Regular orbit R1
q[0, 0, 0] = 0
q[0, 1, 0] = 0.0
p[0, 0, 0] = 0.0
p[0, 1, 0] = 0

#Chaotic orbit C1
#q[0, 0, 0] = 0
#q[0, 1, 0] = 0.4208
#p[0, 0, 0] = -0.25
#p[0, 1, 0] = 0

#Chaotic orbit C2
#q[0, 0, 0] = 0
#q[0, 1, 0] = 0.335036
#p[0, 0, 0] = 0.11879
#p[0, 1, 0] = -0.385631

### ------------------------------------------------------------------------------
#Normalization of deviation
norm = 0.00001#CONVERGENCE PARAMETER#todo:vary
dt = 0.01#CONVERGENCE PARAMETER#todo:vary
time_run = 300#CONVERGENCE PARAMETER#todo:vary
tau = 0.5#CONVERGENCE PARAMETER#todo:vary
def norm_dp_dq(dp,dq,norm=0.00001):
    div=(1/norm)*np.linalg.norm(np.concatenate((dq, dp)))
    return dp/div,dq/div
def run_var(sim, dt, time_run, tau):
    tarr=[]
    varp=[]
    varq=[]
    mLCE=[]
    alpha = []
    nsteps = int(tau/dt)
    N=int(time_run/tau)
    for k in range(N):
        for i in range(nsteps):
            sim.step(mode="nve", var='variation')
        tarr.append(sim.t)
        varp.append(sim.rp.dp)
        varq.append(sim.rp.dq)
        dp_new,dq_new=norm_dp_dq(sim.rp.dp, sim.rp.dq)
        alpha.append((1/norm)*np.linalg.norm(np.concatenate((sim.rp.dp[0], sim.rp.dq[0]))))
        sim.rp.dp = dp_new
        sim.rp.dq = dq_new
        #print(alpha)
        mLCE.append((1/sim.t)*np.sum(np.log(alpha)))
    return tarr, mLCE

    

#Initialize small deviation
dq = 0.00001*np.ones_like(q)#todo:vary
dp = 0.00001*np.ones_like(p)#todo:vary
dp, dq = norm_dp_dq(dp, dq, norm)
#Set up the simulation
sim = RP_Simulation()
rngSeed = range(1)
rng = np.random.default_rng(rngSeed)
ens = Ensemble(beta = 1/T, ndim = dim)
motion = Motion(dt = dt, symporder = 2)#4
propa = Symplectic_order_II()#IV
rp = RingPolymer(pcart = p, qcart = q, dpcart = dp, dqcart = dq, m = m)
rp.bind(ens, motion, rng)
therm = PILE_L(tau0 = 1.0, pile_lambda = 100.0)#only important due to initalization 
sim.bind(ens, motion, rng, rp, pes, propa, therm)

tarr,mLCE = run_var(sim, dt, time_run, tau)
print("Seconds since epoch = ", seconds)
print('mlce',mLCE[-1])
plt.plot(np.log10(tarr[1:]),np.log10(mLCE[1:]))
plt.xlim(1, 4)
plt.xlabel("log$_{10}$t")
plt.ylabel("log$_{10}X_{1}$")
plt.show()
#Initialize trajectory at saddle point with almost zero velocity
#Set dq, dp to a certain predefined norm (dp and dq are together w so consider that when normalizing)
#Propagate trajectories in the 'variation' mode for time 'T'
#'Renormalize' the perturbations, reinitialize the rp class and rerun trajectories for time T
#Do this M times and find Lyapunov exponent
