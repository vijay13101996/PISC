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

#Normalization of deviation
norm = 0.00001
time_run = 10000
tau = 0.1

def norm_dp_dq(dp, dq, norm = 0.00001):
    div = (1/norm) * np.linalg.norm(np.concatenate((dq, dp), axis = 1), axis = 1)
    #print(dp.shape)
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

E = 5.0 #Energy at which the Poincare section shows chaotic trajectories
N = 1 #Number of trajectories
pathname = '/home/ss3120/PISC/examples/2D/classical/'#os.path.dirname(os.path.abspath(__file__))

xg = np.linspace(-1, 1, int(1e2)+1)

yg = np.linspace(-1, 1, int(1e2)+1)

xgrid,ygrid = np.meshgrid(xg, yg)
potgrid = pes.potential_xy(xgrid, ygrid) 

PSOS = Poincare_SOS('Classical', pathname, potkey, Tkey)
PSOS.set_sysparams(pes, T, m, 2)
PSOS.set_simparams(N, dt, dt, nbeads = nbeads, rngSeed = 1)	
PSOS.set_runtime(50.0, 500.0)
qlist = PSOS.find_initcondn(xgrid, ygrid, potgrid, E)
PSOS.bind(qcartg = qlist, E = E, sym_init = False)

q_ps = PSOS.rp.qcart
p_ps = PSOS.rp.pcart

print('E', pes.potential(q_ps)+np.sum(p_ps**2/(2*m),axis=1))
mLyap = []
#Set up the simulation

if(1): # Relaxation
    sim = RP_Simulation()
    rngSeed = range(1)
    rng = np.random.default_rng(rngSeed)
    ens = Ensemble(beta = 1/T, ndim = dim)
    motion = Motion(dt = dt, symporder = 2)#4
    propa = Symplectic_order_II()#IV
    rp = RingPolymer(pcart = p_ps, qcart = q_ps, m = m)
    rp.bind(ens, motion, rng)
    therm = PILE_L(tau0 = 1.0, pile_lambda = 100.0) #Only important due to initalization 
    sim.bind(ens, motion, rng, rp, pes, propa, therm)

    nsteps = int(100/dt)
    sim.step(ndt = nsteps, mode = 'nve', var = 'pq')

#Initialize small deviation
dq = 0.00001 * np.ones_like(rp.qcart)
dp = np.zeros_like(dq)
dp, dq = norm_dp_dq(dp, dq, norm)

motion = Motion(dt = dt, symporder = 4)
propa = Symplectic_order_IV()
rp = RingPolymer(pcart = sim.rp.pcart, qcart = sim.rp.qcart, dpcart = dp, dqcart = dq, m = m)
rp.bind(ens, motion, rng)
therm = PILE_L(tau0 = 1.0, pile_lambda = 100.0) #Only important due to initalization 
sim.bind(ens, motion, rng, rp, pes, propa, therm)
tarr, mLCE = run_var(sim, dt, time_run, tau)

#print('Dimensions of mLCE are ', np.array(mLCE).shape)
#quit()

#If the Lyapunov exponent is lower than 0.001, then set it to 0
#if (max(mLCE) < 1e-3):
    #mLyap.append(0)
#else:
    #mLyap.append(np.max(mLCE))
#print("Seconds since epoch = ", seconds)
#print('mlce',mLCE[-1])
mLyap = np.array(mLCE)[:, -1]

h = pes.ddpotential(sim.rp.qcart)
print(h.shape)
h = np.transpose(h, (0, 3, 1, 2))

eigrid = np.sort(np.linalg.eigvals(h), axis = 2)
print(eigrid[:, :, 0])
omega1 = np.sqrt(np.absolute(np.min(eigrid[:, :, 0])))/m
Tc = 2 * np.pi/omega1
print(Tc)
quit()



plt.plot(tarr, mLyap)
#plt.xlim(0, 3)
plt.xlabel("time")
plt.ylabel("mLCE")
plt.title('Classical E = 5 g = 0.1')
#plt.xlabel("log$_{10}$t")
#plt.ylabel("log$_{10}X_{1}$")
plt.show()

fig, ax = plt.subplots()
ax.hist(mLyap, bins = 30)#, range = (0.4, 0.7))
ax.set_xlabel('mLCE')
ax.set_ylabel('Population')
plt.show()

#Initialize trajectory at saddle point with almost zero velocity
#Set dq, dp to a certain predefined norm (dp and dq are together w so consider that when normalizing)
#Propagate trajectories in the 'variation' mode for time 'T'
#'Renormalize' the perturbations, reinitialize the rp class and rerun trajectories for time T
#Do this M times and find Lyapunov exponent
