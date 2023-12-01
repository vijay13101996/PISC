import numpy as np
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.harmonic_oblique_2D import Harmonic_oblique
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.utils.misc import hess_compress, hess_expand, hess_mul
from PISC.utils.misc_f import misc
from PISC.utils.nmtrans_f import nmtrans
import time

"""
Test file for checking whether the transformation functions implemented in fortran
are working correctly. The functions tested are:
* hess_compress
* hess_expand
* hess_mul
"""

N = 1000
dim = 2
nbeads = 32
m=0.5
beta = 1.0
dt = 0.1

rng = np.random.default_rng(1)

qcart = rng.standard_normal(size=(N,dim,nbeads))
pcart = rng.standard_normal(size=(N,dim,nbeads))

ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=4) 
rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,mode='rp') 
rp.bind(ens,motion,rng)


if(0): #Test hess_compress
    """
    hess_compress changes the shape of the hessian from (N,dim,nbeads,dim,nbeads) 
    to (N,dim*nbeads,dim*nbeads) - this is done in place, so it takes practically
    no time to run. The fortran version is slower since it seems to be copying the
    array instead of changing the shape in place.
    """

    hess = rng.random((N,dim,nbeads,dim,nbeads))
    hessf = np.zeros((N,dim*nbeads,dim*nbeads))
    
    hessp = hess_compress(hess,rp)
    misc.hess_compress(hess.T,hessf.T)
   
    nsteps = 1
    start = time.time()
    for i in range(nsteps):
        hessp = hess_compress(hess,rp)
    print('python time: ',time.time()-start)

    start = time.time()
    for i in range(nsteps):
        misc.hess_compress(hess.T,hessf.T)
    print('fortran time: ',time.time()-start)

    assert np.allclose(hessp,hessf)
    print('hess_compress : OK')

if(0): #Test hess_expand
    """
    hess_expand changes the shape of the hessian from (N,dim*nbeads,dim*nbeads) to 
    (N,dim,nbeads,dim,nbeads) - this is also done in place, so it takes practically no
    time to run. The fortran version is slower since it seems to be copying the array.
    """

    hess = rng.random((N,dim*nbeads,dim*nbeads))
    hessp = hess_expand(hess,rp)
    hessf = misc.hess_expand(hess,dim,nbeads)
    assert np.allclose(hessp,hessf)
    print('hess_expand : OK')

if(0): #Test hess_mul
    """
    hess_mul multiplies the hessian by the matrix Mqq and Mpq, which are of shape 
    (N,dim,nbeads,dim,nbeads).They are converted to (N,dim*nbeads,dim*nbeads) by
    hess_compress, and then multiplied by the hessian (which is also converted to
    (N,dim*nbeads,dim*nbeads) by hess_compress). The result is then converted back
    to (N,dim,nbeads,dim,nbeads) by hess_expand. The result is stored in Mpqp and Mqqp, 
    respectively for the python version and Mpqf and Mqqf for the fortran version.

    NOTE: The fortran function hess_mul_v calls hess_mul and hess_expand internally, so
    it is slower than the python version.
    """

    hess = rng.random((N,dim,nbeads,dim,nbeads))
    Mqq = rng.random((N,dim,nbeads,dim,nbeads))
    Mpq = rng.random((N,dim,nbeads,dim,nbeads))
    
    Mqqp = Mqq.copy()
    Mpqp = Mpq.copy()
    Mqqf = Mqq.copy().T
    Mpqf = Mpq.copy().T

    hess_mul(hess,Mqqp,Mpqp,rp,dt)
    misc.hess_mul_v(hess.T,Mqqf,Mpqf,dt)

    assert np.allclose(Mpqp,Mpqf.T)
    print('hess_mul : OK')

if(0): #Test hess_mul speed
    """
    Since it is not advantageous to use the fortran version hess_mul_v, we will use hess_compress 
    and hess_expand to convert the hessian to the appropriate shape, and then use fortran only to do
    the matrix multiplication. This is faster than the python version, but not by much. hess_mul function
    is a customised version of np.einsum('ijk,kl->ijl',hess,Mqq).
    """
    hess = rng.random((N,dim,nbeads,dim,nbeads))
    Mqqp = rng.random((N,dim,nbeads,dim,nbeads))
    Mpqp = rng.random((N,dim,nbeads,dim,nbeads))
    
    Mqqf = Mqqp.copy()
    Mpqf = Mpqp.copy()
   
    nsteps = 1
    start = time.time()
    for i in range(nsteps):
        #hess_compress(hess,rp)
        hess_mul(hess,Mqqp,Mpqp,rp,dt)
    
    print('python time: ',time.time()-start)

    start = time.time()
    for i in range(nsteps):
        hess_mul(hess,Mqqf,Mpqf,rp,dt,fort=True)
    print('fortran time: ',time.time()-start)

    #print('Mpqp',Mpqp)
    #print('Mpqf',Mpqf)

    assert np.allclose(Mpqp,Mpqf)
    assert np.allclose(Mqqp,Mqqf)

if(0): #Test Cartesian to Matsubara transformation (and back) for positions and momenta
    """
    Test to check whether the fortran version of the transformation functions are working correctly
    on the positions and momenta. The fortran version is slower than the python version, but not by
    much. So, it is better to use scipy.fft for the transformation.
    """


    # The below line is to compare scipy's fft with direct matrix multiplication
    #qmat_p = np.einsum('ijk,kl->ijl',qcart,rp.nmtrans.nm_matrix)

    qcart_f = np.zeros_like(qcart).T
    qmat_f = np.zeros_like(qcart).T

    for i in range(1):
        print('Cartesian to Matsubara transformation')
        start = time.time()
        nmtrans.cart2mats(qcart.T,qmat_f,rp.nmtrans.nm_matrix.T)
        print('fortran',time.time()-start)
        start = time.time()
        qmat_p = rp.nmtrans.cart2mats(qcart)
        print('python',time.time()-start, '\n')

        print('Matsubara to Cartesian transformation')
        start = time.time()
        nmtrans.mats2cart(qmat_f,qcart_f,rp.nmtrans.nm_matrix.T)
        print('fortran',time.time()-start)
        start = time.time()
        qcart_p = rp.nmtrans.mats2cart(qmat_p)
        print('python',time.time()-start)

        assert np.allclose(qmat_f.T,qmat_p)
        assert np.allclose(qcart_f.T,qcart_p)
        print('Cartesian to Matsubara transformation : OK', '\n')

if(1): #Test Cartesian to Matsubara transformation (and back) for hessian
   
    """
    Test to check whether the fortran version of the transformation functions are working correctly
    on the hessian. The fortran version is between 10-20 times faster than python.
    """

    hess_p = rng.random((N,dim,nbeads,dim,nbeads))
    hess_f = hess_p.copy().T
    
    hesst_p = np.zeros_like(hess_p)
    hesst_f = np.zeros_like(hess_f)

    hesstt_p = np.zeros_like(hess_p)
    hesstt_f = np.zeros_like(hess_f)

    start = time.time()
    nmtrans.cart2mats_hessian(hess_f,hesst_f,rp.nmtrans.nm_matrix.T)
    print('fortran',time.time()-start)
    start = time.time()
    hesst_p = rp.nmtrans.cart2mats_hessian(hess_p)
    print('python',time.time()-start)

    start = time.time()
    nmtrans.mats2cart_hessian(hesst_f,hesstt_f,rp.nmtrans.nm_matrix.T)
    print('fortran',time.time()-start)
    start = time.time()
    hesstt_p = rp.nmtrans.mats2cart_hessian(hesst_p)
    print('python',time.time()-start)

    assert np.allclose(hesstt_f,hess_f)
    assert np.allclose(hesstt_p,hess_p)
    assert np.allclose(hesst_f.T,hesst_p)
    assert np.allclose(hesstt_f.T,hesstt_p)

    print('Cartesian to Matsubara transformation (hessian) : OK')

