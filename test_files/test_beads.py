import numpy as np
import PISC
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV
from PISC.potentials.harmonic_oblique_2D import Harmonic_oblique
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation

"""
 Things to check:
1. Initialization of p,q and M matrix
2. Normal mode transformations
3. RSP Matrix
4. Normal mode force and Hessian
5. Also check whether the bind function does everything required
6. Check rp.mats2cart, rp.cart2mats
7. Compare normal mode force with bead mode force
8. Multiply RSP matrix with rp.q and rp.p and compare it with normal 
   integration.
"""

hbar = 1.0
dim = 2
T = 3.0
m = 1.0
N=2

nbeads = 4 
qcart = np.random.normal(size=(N,dim,nbeads))#
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = None
dt = 0.1
beta = 1/T

# Testing the ring polymer definition with qcart
rp = RingPolymer(qcart=qcart,m=m) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2)
therm = PILE_L()
rng = np.random.default_rng(1) 

rp.bind(ens,motion,rng)

if(0):
	trans = np.array([[1.0,1.0],[0.5,1.0]])
	T11,T12,T21,T22 = trans[0,0],trans[0,1],trans[1,0],trans[1,1]

	trans_inv = np.linalg.inv(trans)
	omega1 = 1.0
	omega2 = 2.0
	pes = Harmonic_oblique(trans,m,omega1,omega2)

if(1):
	lamda = 2.0
	g = 0.08
	Vb = lamda**4/(64*g)
	D = 3*Vb
	alpha = 0.37
	
	z = 1.0
	pes = quartic_bistable(alpha,D,lamda,g,z)

pes.bind(ens,rp)

propa = Symplectic_order_IV()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

# Checking initialization of p's and q's.
print('p, q',rp.p,rp.q) 
 
# Checking correct definitions of Monodromy matrices.
print('Mpp,Mqq', rp.Mpp[0,0,0],rp.Mpp[0,0,1]) 
print('Mqp,Mpq', rp.Mqp[0,0,0])

# Checking correct definition of reference system propagator
print('RSP Matrix', rp.RSP_coeffs) 

print('T matrix', rp.nm_matrix)
#Checking correct initialization of the Hessian matrix
print('ddpot', rp.ddpot)
print('ddpot_cart',rp.ddpot_cart) 

#Check ring polymer frequencies
print('freqs2', rp.get_rp_freqs()**2)
print('omegan',rp.omegan)

# Checking the force from rp springs and pes
# (Can be toggled b/w rp.dpot and pes.dpot)
dpot = rp.dpot
dpot_cart = rp.dpot_cart

# Verifying whether the normal mode transformation functions and matrices are working alright.
# (Sometimes the arrays don't agree upto a tolerance of machine precision. This is because 
# there are quite a few zeros in these arrays, which mess with the comparison. But this is not a
# cause of major concern)
print('q',rp.q,rp.nmtrans.cart2mats(rp.qcart),np.einsum('ij,...j',rp.nm_matrix.T,rp.qcart))

transf_dpot =  rp.nmtrans.cart2mats(dpot_cart)
transf_dpotcart = rp.nmtrans.mats2cart(dpot)
print('dpot', np.allclose(dpot,transf_dpot,rtol=1e-9,atol=1e-12))
print('dpot_cart', np.allclose(dpot_cart, transf_dpotcart,rtol=1e-10,atol=1e-14)) 

transf_qcart = rp.nmtrans.mats2cart(rp.nmtrans.cart2mats(rp.qcart))
transf_qcart_mat = np.einsum('ij,...j',rp.nm_matrix.T,np.einsum('ij,...j',rp.nm_matrix,rp.qcart))
print('cart2mats2cart',np.allclose(transf_qcart,qcart,rtol=1e-10,atol=1e-14))
print('cart2mats2cart matrix', np.allclose(transf_qcart_mat, rp.qcart,rtol=1e-10,atol=1e-14))

transf_qcart = rp.nmtrans.cart2mats(rp.qcart)
transf_qcart_mat = np.einsum('ij,...i', rp.nm_matrix, rp.qcart)
print('cart2mats', np.allclose(transf_qcart, transf_qcart_mat,rtol=1e-10, atol=1e-14 ))

transf_q = rp.nmtrans.mats2cart(rp.q)
transf_q_mat = np.einsum('ij,...i',rp.nm_matrix.T,rp.q)
print('mats2cart', np.allclose(transf_q,transf_q_mat,rtol=1e-10,atol=1e-14 ))

# Verifying whether the Hessian matrix transforms as expected.
# (Can be toggled b/w rp.ddpot and pes.ddpot)
ddpot =  rp.ddpot
ddpot_cart =  rp.ddpot_cart

print('printing ddpot', ddpot.shape,ddpot_cart[0])

"""
To check:
1. pes.ddpot_cart should be diagonal along the bead coordinates; i.e
   pes.ddpot_cart[i,j,k,:,:] should be diagonal for all i,j,k
2. The ring polymer spring potential is always separable wrt the 
   dimensions. So rp.ddpot[:,j,k] and rp.ddpot[:,j,k] ought to be zero
   if j!=k. And, rp.ddpot[:,i,i] should be diagonal and rp.ddpot_cart[:,i,i]
   should be 'sparse' with just diagonal and the adjacent sub-diagonal elements. 
3. rp.ddpot and rp.ddpot_cart and initialized separately while rp is 'binded'.
   So, if the transformation matrix T is converting rp.ddpot to rp.ddpot_cart and
   vice versa, then the transformations are alright. 
"""

transf_ddpot = np.einsum('ij,...jk',rp.nm_matrix.T, np.einsum('...ij,jk',ddpot_cart,rp.nm_matrix))
transf_ddpot_cart = np.einsum('ij,...jk',rp.nm_matrix, np.einsum('...ij,jk',ddpot,rp.nm_matrix.T))
print('ddpot',np.allclose(transf_ddpot, ddpot, rtol=1e-8, atol=1e-12 ))
print('ddpot_cart', np.allclose(transf_ddpot_cart, ddpot_cart, rtol=1e-8, atol=1e-14)) 

transf_ddpot = rp.nmtrans.cart2mats_hessian(ddpot_cart)
transf_ddpot_cart = rp.nmtrans.mats2cart_hessian(ddpot)
print('ddpot',np.allclose(transf_ddpot, ddpot, rtol = 1e-8,atol=1e-10))
print('ddpot_cart',np.allclose(transf_ddpot_cart, ddpot_cart, rtol = 1e-8, atol=1e-10))
 
