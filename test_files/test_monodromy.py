import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.harmonic_oblique_2D import Harmonic_oblique
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.misc import hess_compress

dim = 2
T = 2.0
m = 1.0
N=1

nbeads = 4 
rng = np.random.default_rng(1)
qcart = np.ones((N,dim,nbeads))#rng.standard_normal(size=(N,dim,nbeads))
pcart = np.zeros_like(qcart)
q = rng.standard_normal(size=(N,dim,nbeads))

pcart = None
dt = 0.01
beta = 1/T

rp = RingPolymer(qcart=qcart,pcart=pcart,m=m) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=4) 
therm = PILE_L()
rp.bind(ens,motion,rng)

trans = np.array([[1.0,1.0],[0.5,1.0]])
T11,T12,T21,T22 = trans[0,0],trans[0,1],trans[1,0],trans[1,1]

trans_inv = np.linalg.inv(trans)
omega1 = 1.0
omega2 = 2.0
pes = Harmonic_oblique(trans,m,omega1,omega2)
pes.bind(ens,rp)

propa = Symplectic_order_IV()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

time_total = 10.0
nsteps = int(time_total/dt)

tarr = []
Mqqarr = []

"""
Testing procedure:
1. Find and store the unitary matrix Umat of eigenvectors of the 
   Hessian and its inverse Umat_inv.
2. Store the diagonalized Hessian separately. The frequencies 
   change along the rows of stability matrix elements
3. Compute the 'normal mode frequencies' and store it in an array
4. Multiply the required stability matrix with Umat_inv
5. Compute the analytical expression of the stability matrix elements
   at time t and compare with the one from the symplectic propagator.

"""

ddpot = pes.ddpot+rp.ddpot
ddpot = hess_compress(ddpot,rp)
print('ddpot', ddpot)

vals,vecs = np.linalg.eigh(ddpot[0])

Umat = vecs 
Umat_inv = np.linalg.inv(Umat)


omega_arr = vals**0.5

Mqq = hess_compress(rp.Mqq,rp)
Mpq = hess_compress(rp.Mpq,rp)

i,j = [2,2]	
v0 = np.matmul(Umat_inv, Mqq[0])[i,j]
vd0 = np.matmul(Umat_inv, Mpq[0])[i,j]

for n in range(nsteps):
	sim.step(mode="nve",var='monodromy')
	Mqq = hess_compress(rp.Mqq,rp)
	Mpq = hess_compress(rp.Mpq,rp)
	UinvMqq =  np.matmul(Umat_inv,Mqq[0])[i,j]
	UinvMpq = np.matmul(Umat_inv,Mpq[0])[i,j]
	print('Mqq t',UinvMqq)
	
	cwt = np.cos(omega_arr[i]*sim.t)
	swt = np.sin(omega_arr[i]*sim.t)
	print('Mqq analytical', v0*cwt + vd0*swt)
	print('\n')	

		
