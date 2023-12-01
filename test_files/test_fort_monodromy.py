import numpy as np
#import PISC
from PISC.engine.integrators import Symplectic
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.harmonic_oblique_2D import Harmonic_oblique
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.utils.misc import hess_compress

"""
Test file for verifying the accuracy of the monodromy matrix propagation
using the symplectic propagator. 

* The test is performed for a 2D harmonic oscillator with an oblique
  (non-diagonal) Hessian - this causes 'mixing' between the x and y modes 
  which causes the monodromy matrices to be non-diagonal.
* However, an analytical result still exists since there is a basis along
  which the Hessian is diagonal. The analytical result is used to verify
  the numerical result.

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

# Standard book-keeping
dim = 2
T = 2.0
m = 1.0
N = 1

nbeads = 8
rng = np.random.default_rng(1)
qcart = rng.standard_normal(size=(N,dim,nbeads))
pcart = np.zeros_like(qcart)

dt = 0.001 # A small time step is required for better comparison with analytical result
beta = 1/T # Temperature only to initialize the ensemble

rp = RingPolymer(qcart=qcart,m=m) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=4) 
therm = PILE_L()
rp.bind(ens,motion,rng)

trans = np.array([[1.0,1.0],[0.5,1.0]]) # Oblique transformation matrix
T11,T12,T21,T22 = trans[0,0],trans[0,1],trans[1,0],trans[1,1]

trans_inv = np.linalg.inv(trans)

omega1 = 1.0 # Frequency along x
omega2 = 2.0 # Frequency along y
pes = Harmonic_oblique(trans,m,omega1,omega2)
pes.bind(ens,rp)

propa = Symplectic(fort=True) # Toggle Fortran to enable/disable the use of Fortran propagator
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

time_total = 10.0 
nsteps = int(time_total/dt)

ddpot = pes.ddpot+rp.ddpot
ddpot = hess_compress(ddpot,rp) # Obtain the Hessian in the normal mode basis

vals,vecs = np.linalg.eigh(ddpot[0]) # Obtain the eigenvalues and eigenvectors of the Hessian

Umat = vecs # Unitary matrix of eigenvectors
Umat_inv = np.linalg.inv(Umat) # Inverse of the unitary matrix

omega_arr = vals**0.5 # Array of normal mode frequencies

Mqq = hess_compress(rp.Mqq,rp) # Mqq in the normal mode basis
Mpq = hess_compress(rp.Mpq,rp) # Mpq in the normal mode basis

i,j = [3,2] # Choose the element of the stability matrix to be tested
v0 = np.matmul(Umat_inv, Mqq[0])[i,j] # Initial value of the element
vd0 = np.matmul(Umat_inv, Mpq[0])[i,j] # Initial value of the derivative of the element

for n in range(nsteps):
    sim.step(mode="nve",var='monodromy')
    Mqq = hess_compress(rp.Mqq,rp)
    Mpq = hess_compress(rp.Mpq,rp)
    UinvMqq =  np.matmul(Umat_inv,Mqq[0])[i,j]
    UinvMpq = np.matmul(Umat_inv,Mpq[0])[i,j]
    Mqq_num = np.around(UinvMqq,4) # Numerical value of the element
    
    cwt = np.cos(omega_arr[i]*sim.t)
    swt = np.sin(omega_arr[i]*sim.t)
    Mqq_anal = np.around(v0*cwt + vd0*swt,4) # Analytical value of the element

    #print(Mqq_num,Mqq_anal)
    assert(np.allclose(Mqq_num,Mqq_anal,atol=1e-3)) # Compare the numerical and analytical values 
    # Note: A smaller tolerance necessitates a smaller time step

        
