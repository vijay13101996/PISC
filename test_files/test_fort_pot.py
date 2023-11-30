import numpy as np
from PISC.potentials.harmonic_2D_f import harmonic_2d
from PISC.potentials.quartic_bistable_f import quartic_bistable as quartic_bistable_f
from PISC.potentials.harmonic_2D import Harmonic
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials import quartic_bistable
import time

m = 0.5
omega = 1.0
param_list = [m,omega]# 
#param_list = [0.382,9.375,2.0,0.08,1.0] 

N = 1000
dim = 2
nbeads = 32
beta=1.0
dt = 0.1

rng = np.random.default_rng(0)
qcart = rng.random((N,dim,nbeads))
vqf = np.zeros((N,nbeads))
dvqf = np.zeros_like(qcart)
ddvqf = np.zeros((N,dim,dim,nbeads,nbeads))

rp = RingPolymer(qcart=qcart,m=m,mode='rp') 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rp.bind(ens,motion,rng,fort=True)

pes = quartic_bistable(0.382,9.375,2.0,0.08,1.0)
#pes = Harmonic(1.0)
pes.bind(ens,rp,fort=True)

pes.potential_f(qcart.T,vqf.T)
vqp = pes.potential(qcart)

pes.dpotential_f(qcart.T,dvqf.T)
dvqp = pes.dpotential(qcart)

pes.ddpotential_f(qcart.T,ddvqf.T)
ddvqp = pes.ddpotmat*pes.ddpotential(qcart)[...,np.newaxis]

nsteps = 10
start = time.time()
for i in range(nsteps):
    #ddvqp = pes.ddpotential(qcart)
    pes.update(update_hess=True,fortran=False)

#ddvqp = pes.ddpotmat*ddvqp[...,np.newaxis]
print('python time: ',time.time()-start)


start = time.time()
for i in range(nsteps):
    #pes.ddpotential_f(qcart.T,ddvqf.T)
    pes.update(update_hess=True,fortran=True)
print('fortran time: ',time.time()-start)

assert np.allclose(vqf,vqp)
assert np.allclose(dvqf,dvqp)
assert np.allclose(ddvqf,ddvqp)

