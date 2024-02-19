import numpy as np
import PISC
from PISC.engine.integrators import Symplectic
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.double_well_potential import double_well
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

dim = 1
m=0.5

l1 = 2.0
g1 = 0.05

pes = double_well(l1,g1)

Vb = l1**4/(64*g1)
Tc = 0.5*l1/np.pi

N = 10000
dt_therm = 0.05
dt = 0.02
time_therm = 50.0
time_relax = 400.0

T_therm = 10*Tc
beta = 1/T_therm

#-------------------------------------------------------------------------------------------

# Set up the simulation parameters
nsteps_therm = int(time_therm/dt)
nsteps_relax = int(time_relax/dt)

# Set up the initial conditions
qcart = np.zeros((N, dim, 1))
pcart = np.zeros((N, dim, 1))
# These two lines are specific to the 2D double well. They are used to 
# initialize the ring polymers in the two wells of the double well potential.
qcart[: N // 2, 0, :] -= 2.5  
qcart[N // 2 :, 0, :] += 2.5


# Set up the simulation
rng = np.random.default_rng(0)
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt_therm,symporder=2) 
therm = PILE_L(tau0=1.,pile_lambda=100.0) 
propa = Symplectic()
sim = RP_Simulation()

#-------------------------------------------------------------------------------------------
#Thermalize the system at the initial temperature
rp = RingPolymer(qcart=qcart,pcart=pcart,m=m)
sim.bind(ens,motion,rng,rp,pes,propa,therm,pes_fort=False,propa_fort=False,transf_fort=False)
sim.step(ndt=nsteps_therm, mode="nvt", var="pq",pc=True)

q_equil_1 = sim.rp.qcart.copy()
p_equil_1 = sim.rp.pcart.copy()

#-------------------------------------------------------------------------------------------
# Thermalize the system with the quenched potential
l2 = 1.0
g2 = g1*l2**2/l1**2
Tc2 = 0.5*l2/np.pi

print('Tc1 = ',Tc,' Tc2 = ',Tc2,' T_therm = ',T_therm)


pes = double_well(l2,g2)
rp = RingPolymer(qcart=qcart,pcart=pcart,m=m)
sim.pes = pes
sim.rebind()
sim.step(ndt=nsteps_therm, mode="nvt", var="pq",pc=True)

q_equil_2 = sim.rp.qcart.copy()
p_equil_2 = sim.rp.pcart.copy()

bins = np.linspace(-8,8,101)

hist1, bins1 = np.histogram(q_equil_1[:,0,0], bins=bins, density=True)
hist2, bins2 = np.histogram(q_equil_2[:,0,0], bins=bins, density=True)
#plt.axhline(hist1.max(),c='r')
#plt.axhline(hist2.max(),c='k')
#plt.plot(bins1[:-1], hist1, label="l = 2.0, g = 0.08",color='r')
#plt.plot(bins2[:-1], hist2, label="l = 1.0, g = 0.02",color='k')
#plt.show()
#exit()

#-------------------------------------------------------------------------------------------
# Set up the simulation for the quenched dynamics
sim.t = 0.0
motion = Motion(dt = dt,symporder=2)
rp = RingPolymer(qcart=q_equil_1,pcart=p_equil_1,m=m)
sim.rp = rp
sim.motion = motion
sim.rebind()

# Relax the system with the new potential
for i in range(nsteps_relax):
    sim.step(mode="nve", var="pq")
    hist, bins = np.histogram(sim.rp.qcart[:,0,0], bins=bins, density=True)
    maxP = np.max(hist)

    vardist1 = np.sum((hist - hist1)**2)
    vardist2 = np.sum((hist - hist2)**2)
    if i%100 == 0:
        plt.scatter(sim.t, vardist1, c='r',s=10)
        plt.scatter(sim.t, vardist2, c='k',s=10)
        plt.pause(0.05)


    if 0:#i%10 == 0:
        plt.plot(bins[:-1], hist, c='0.5')
        plt.pause(0.05)

    if 0:#i%100 == 0:
        #plt.hist(sim.rp.qcart[:,0,0],bins=40)
        plt.scatter(sim.t, np.mean(sim.rp.qcart[:,0,0]**4), c='r')
        plt.pause(0.05)

plt.axhline(hist.max(),c='b')
#plt.plot(bins[:-1], hist, c='b')
plt.show()
