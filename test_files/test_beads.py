import numpy as np
import PISC
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion

### Things to check:
#1. Initialization of p,q and M matrix
#2. Normal mode transformations
#3. RSP Matrix
#4. Normal mode force and Hessian
#5. Also check whether the bind function does everything required
#6. Check rp.mats2cart, rp.cart2mats
#7. Compare normal mode force with bead mode force
#8. Multiply RSP matrix with rp.q and rp.p and compare it with normal 
#   integration -  This is better to be done when checking integrators.

dim = 2
T = 2.0
m = 1.0
N=2

nbeads = 4 
qcart = np.random.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = None
dt = 0.1
beta = 1/T

# Testing the ring polymer definition with qcart
rp = RingPolymer(qcart=qcart,m=m,mode='MFmats',nmats=3) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(1) 

rp.bind(ens,motion,rng)

# Checking initialization of p's and q's.
print('p, q',rp.p,rp.q) 
 
# Checking correct definitions of Monodromy matrices.
print('Mpp,Mqq', rp.Mpp[0,0,0],rp.Mpp[0,0,1]) 
print('Mqp,Mpq', rp.Mqp[0,0,0])

# Checking correct definition of reference system propagator
print('RSP Matrix', rp.RSP_coeffs) 

#Checking correct initialization of the Hessian matrix
print('ddpot', rp.ddpot[0,1,1],rp.ddpot_cart) 

dpot = rp.dpot
dpot_cart = rp.dpot_cart

# Verifying whether the normal mode transformation functions and matrices are working alright.
print('dpot', dpot[1], rp.nmtrans.cart2mats(dpot_cart)[1])
print('dpot_cart', dpot_cart[1], rp.nmtrans.mats2cart(dpot)[1]) 

print('cart2mats2cart',rp.nmtrans.mats2cart(rp.nmtrans.cart2mats(rp.qcart)),rp.qcart)
print('cart2mats2cart matrix', np.einsum('ij,...j',rp.nm_matrix.T,np.einsum('ij,...j',rp.nm_matrix,rp.qcart))[0], rp.qcart[0])

print('cart2mats', np.einsum('ij,...i', rp.nm_matrix, rp.qcart),rp.nmtrans.cart2mats(rp.qcart)  )
print('mats2cart', np.einsum('ij,...i',rp.nm_matrix.T,rp.q),rp.nmtrans.mats2cart(rp.q))

print('T matrix', rp.nm_matrix)

# Verifying whether the Hessian matrix transforms as expected.
#print('ddpot', np.einsum('ij,...jk',rp.nm_matrix.T, np.einsum('...ij,jk',rp.ddpot_cart,rp.nm_matrix))[0,0,0],rp.ddpot[0,0,0])

#print('ddpot', np.einsum('ij,...jk',rp.nm_matrix, np.einsum('...ij,jk',rp.ddpot,rp.nm_matrix.T))[0,0,0],rp.ddpot_cart[0,0,0]) 

print('ddpot',rp.nmtrans.cart2mats_hessian(rp.ddpot_cart)[0],rp.ddpot[0])
print('ddpot_cart',rp.nmtrans.mats2cart_hessian(rp.ddpot)[0],rp.ddpot_cart[0])
 
