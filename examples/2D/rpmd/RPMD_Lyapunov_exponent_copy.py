import numpy as np
import sys
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
from PISC.utils.nmtrans import FFT
import os

### Potential parameters
m = 0.5
dim = 2
nbeads = 64

lamda = 2.0
g = 0.1
omega = 1.0
NT = 1 #Number of trajectories
pes = coupled_harmonic(omega, g)

#Only relevant for ring polymer and canonical simulations
Tc = 0.5 * lamda/np.pi

potkey = 'coupled_harmonic_2D_omega_{}_g_{}'.format(omega, g)    

path = os.path.dirname(os.path.abspath(__file__))

### ------------------------------------------------------------------------------
def norm_dp_dq(dp, dq, norm=0.00001):
    div = (1/norm) * np.linalg.norm(np.concatenate((dp, dq), axis = 1), axis = 1)
    div = div[:, None, :]
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
                prev = np.linalg.norm(np.concatenate((dp_cent, dq_cent), axis = 1), axis = 1)
                #Propagate
                for i in range(nsteps):
                       sim.step(mode = "nve", var = 'variation')
                #Calculate centroid
                cent_xy = sim.rp.q[...,0]/nbeads**0.5
                qx_cent.append(cent_xy[0, 0])
                qy_cent.append(cent_xy[0, 1])
                #Calculate w after nsteps
                dp_cent = sim.rp.dp[...,0]
                dq_cent = sim.rp.dq[...,0]
                current = np.linalg.norm(np.concatenate((dp_cent, dq_cent), axis = 1), axis = 1)
                alpha.append(current/prev)
                #Update
                sim.rp.dp,sim.rp.dq = norm_dp_dq(sim.rp.dp, sim.rp.dq, norm = norm)
                sim.rp.mats2cart()
                tarr.append(sim.t)
                mLCE.append((1/sim.t) * np.sum(np.log(alpha), axis = 0))
        return tarr, mLCE, qx_cent, qy_cent

#-------------------------------------------------------------------------------------------
#CONVERGENCE PARAMETERS
norm = 0.00001#CONVERGENCE PARAMETER#todo:vary
dt = 0.01#CONVERGENCE PARAMETER#todo:vary
time_run = 500#CONVERGENCE PARAMETER#todo:vary
tau = 0.1#CONVERGENCE PARAMETER#todo:vary

#Initialize small deviation for the centroid mode - Sandra
dq_cent = np.zeros((1, 2, 64))#todo:vary
dq_cent[:, 0, 0] = 1e-5
dp_cent = np.zeros_like(dq_cent)#todo:vary
dp_cent, dq_cent = norm_dp_dq(dp_cent, dq_cent, norm)

lamda_arr  = []
Hess_arr = []
T_arr = [1.2]#np.linspace(0.7, 0.95, 7)

sim = RP_Simulation()
motion = Motion(dt = dt, symporder = 2)#4
propa = Symplectic_order_II()#IV
therm = PILE_L(tau0 = 1.0, pile_lambda = 100.0)#only important due to initalization 
rng = np.random.default_rng(0)

for times in T_arr:
    Tkey = 'T_{}Tc'.format(times)
    T = times * Tc

    qcart = read_arr('Instanton_{}_{}_nbeads_{}'.format(potkey, Tkey, 32),'{}/Datafiles'.format(path)) #Instanton
    #qcart = np.zeros_like(dq_cent)
    pcart = np.zeros_like(qcart)
   
    fft = FFT(1, nbeads)
    q_nm = fft.cart2mats(qcart)/nbeads**0.5 
         
    #------------------------------------------------------------------------------
    #Set up the simulation
    ens = Ensemble(beta = 1/T, ndim = dim)
    rp = RingPolymer(pcart = pcart, qcart = qcart, dp = dp_cent, dq = dq_cent, m = m)
    rp.bind(ens, motion, rng)
    sim.bind(ens, motion, rng, rp, pes, propa, therm)

    #Hess = (rp.ddpot + pes.ddpot).swapaxes(2, 3).reshape(len(rp.ddpot)*2*32, len(rp.ddpot)*2*32)
    #vals, vecs = np.linalg.eigh(Hess) #Find the Hessian eigenvalues and eigenvectors
    #Hess_arr.append(-vals[0]) #Append the first Hessian eigenvalue
    #w1 = 2 * np.pi * times * Tc
    #print('Hessval', -1.0*vals[:3],4-w1**2 ,'\n Hess', Hess[:3,:3])     

    q0 = q_nm[:, :, 0]
    rg = np.mean(np.sum((qcart - q0[:, :, None])**2, axis = 1), axis = 1)
    #print('potential', (rp.pot+pes.pot.sum())/nbeads,rg)
    tarr, mLCE, qx_cent, qy_cent = run_var(sim, dt, time_run, tau, norm)

    lmax = max(mLCE)
    print('Temp', times, 'mLCE', lmax) 
    #fname = 'Lambda_max_{}_{}_{}'.format(potkey,Tkey,nbeads)
    #f = open('{}/Datafiles/{}.txt'.format(path,fname),'a')
    #f.write(str(rngSeed) + "  " + str(lmax) + '\n')
    #f.close()

    lamda_arr.append(lmax)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(np.log10(tarr), np.log10(mLCE))
    ax[1].plot(tarr, qx_cent)
    ax[2].plot(tarr, qy_cent)
    plt.show()

    #store_1D_plotdata(T_arr,lamda_arr,'RPMD_Lyapunov_exponent_{}_nbeads_{}_ext_2'.format(potkey,nbeads),'{}/Datafiles'.format(path))

if(0):
    plt.scatter(T_arr, lamda_arr, marker='x', color='r')
    plt.scatter(T_arr, Hess_arr, color='g')
    T_ext = np.arange(1.0, 10.1, 0.5)
    lamda_ext = 2.0*np.ones_like(T_ext)
    plt.scatter(T_arr, lamda_arr, marker='x', color='b')
    plt.show()

#T_arr = np.arange(0.0,10.0,0.01)
#plt.scatter(T_arr,2*np.pi*T_arr*Tc)
