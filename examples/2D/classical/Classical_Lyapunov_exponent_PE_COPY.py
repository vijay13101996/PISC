import numpy as np
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
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
dim = 2 # (x, y) 
nbeads = 1 #Classical simulation
dt = 0.01
omega = 1.0
g0 = 0.1
potkey = 'Pullen_Edmonds_omega_{}_g0_{}'.format(omega, g0)

pes = coupled_harmonic(omega, g0)
T = 1 #Not relevant in case of a classical simulation
Tkey = 'T_{}'.format(T) 

E = 4.0 #Energy at which the Poincare section shows chaotic trajectories
N = 10 #Number of trajectories
pathname = '/home/ss3120/PISC/examples/2D/classical/'#os.path.dirname(os.path.abspath(__file__))

xg = np.linspace(-8, 8, int(1e2)+1)
yg = np.linspace(-8, 8, int(1e2)+1)

xgrid, ygrid = np.meshgrid(xg, yg)
potgrid = pes.potential_xy(xgrid, ygrid) 

PSOS = Poincare_SOS('Classical', pathname, potkey, Tkey)
PSOS.set_sysparams(pes, T, m, 2)
PSOS.set_simparams(N, dt, dt, nbeads = nbeads, rngSeed = 1)	
PSOS.set_runtime(50.0, 500.0)
qlist = PSOS.find_initcondn(xgrid, ygrid, potgrid, E)
PSOS.bind(qcartg = qlist, E = E, sym_init = False)


Y, PY, X, PX = PSOS.PSOS_Y(x0 = 0.0)
X = qlist[:, 0]
Y = qlist[:, 1]

#Normalization of deviation
norm = 0.00001
time_run = 1000
tau = 0.1

def norm_dp_dq(dp, dq, norm = 0.00001):
    div = (1/norm) * np.linalg.norm(np.concatenate((dq, dp), axis = 1), axis = 1)
    div = div[:, None, :] 
    #print('div', div.shape, dq.shape)
    return dp/div, dq/div

def run_var(sim, dt, time_run, tau):
    tarr = []
    varp = []
    varq = []
    mLCE = []
    alpha = []
    nsteps = int(tau/dt)
    time = int(time_run/tau)
    for k in range(time):
        for i in range(nsteps):
            sim.step(mode = "nve", var = 'variation')
        tarr.append(sim.t)
        varp.append(sim.rp.dp)
        varq.append(sim.rp.dq)
        dp_new,dq_new = norm_dp_dq(sim.rp.dp, sim.rp.dq)
        alpha.append((1/norm) * np.linalg.norm(np.concatenate((sim.rp.dp, sim.rp.dq), axis = 1), axis = 1)) 
        #print('alpha', np.array(alpha).shape)
        sim.rp.dp = dp_new
        sim.rp.dq = dq_new
        mLCE.append((1/sim.t) * np.sum(np.log(alpha), axis = 0))
        #print(np.array(mLCE).shape)
    return tarr, mLCE


#Initialization of p and q
NT = len(np.array(X))
#print('NT = ', NT)
q = np.zeros((NT, dim, nbeads))
p = np.zeros((NT, dim, nbeads))

#Initialize small deviation
dq = 0.00001 * np.ones_like(q)
dp = np.zeros_like(p)
dp, dq = norm_dp_dq(dp, dq, norm)

mLyap = []
q[:, 0, 0] = X
q[:, 1, 0] = Y
#p[:, 0, 0] = PX
#p[:, 1, 0] = PY
#for xi, yi, pxi, pyi in zip(X, Y, PX, PY):
    #q[0, 0, 0] = xi
    #q[0, 1, 0] = yi
    #p[0, 0, 0] = pxi
    #p[0, 1, 0] = pyi

#Regular orbit R1
#q[0, 0, 0] = 0
#q[0, 1, 0] = 0.558
#p[0, 0, 0] = 0.2334
#p[0, 1, 0] = 0

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


#Set up the simulation
if(1):
    sim = RP_Simulation()
    rngSeed = range(1)
    rng = np.random.default_rng(rngSeed)
    ens = Ensemble(beta = 1/T, ndim = dim)
    motion = Motion(dt = dt, symporder = 2)#4
    propa = Symplectic_order_II()#IV
    rp = RingPolymer(pcart = p, qcart = q, dpcart = dp, dqcart = dq, m = m)
    rp.bind(ens, motion, rng)
    therm = PILE_L(tau0 = 1.0, pile_lambda = 100.0) #Only important due to initalization 
    sim.bind(ens, motion, rng, rp, pes, propa, therm)

sim.step(ndt = 10000, mode = 'nve', var = 'pq')

print(PSOS.rp.qcart.shape)
print(PSOS.rp.pcart.shape)

sim = RP_Simulation()
rngSeed = range(1)
rng = np.random.default_rng(rngSeed)
ens = Ensemble(beta = 1/T, ndim = dim)
motion = Motion(dt = dt, symporder = 2)
propa = Symplectic_order_II()
rp = RingPolymer(pcart = PSOS.rp.pcart, qcart = PSOS.rp.qcart, dpcart = dp, dqcart = dq, m = m)
rp.bind(ens, motion, rng)
therm = PILE_L(tau0 = 1.0, pile_lambda = 100.0) #Only important due to initalization 
sim.bind(ens, motion, rng, rp, pes, propa, therm)
tarr, mLCE = run_var(sim, dt, time_run, tau)


#print("Seconds since epoch = ", seconds)
#print('mlce',mLCE[-1])
mLyap = np.array(mLCE)[:, 15]
#plt.plot(np.log10(tarr), np.log10(mLyap))
plt.plot(tarr, mLyap)
#plt.xlim(0, 3)
plt.xlabel("time")
plt.ylabel("mLCE")
#plt.xlabel("log$_{10}$t")
#plt.ylabel("log$_{10}X_{1}$")

#fig, ax = plt.subplots()
#ax.hist(mLyap, bins = 30)
#ax.set_xlabel('mLCE')
#ax.set_ylabel('Population')
plt.show()

#Initialize trajectory at saddle point with almost zero velocity
#Set dq, dp to a certain predefined norm (dp and dq are together w so consider that when normalizing)
#Propagate trajectories in the 'variation' mode for time 'T'
#'Renormalize' the perturbations, reinitialize the rp class and rerun trajectories for time T
#Do this M times and find Lyapunov exponent
