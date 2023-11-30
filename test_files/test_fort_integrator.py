import numpy as np
from PISC.engine.integrator_f import integrator
from PISC.engine.integrators import Symplectic
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.harmonic_2D import Harmonic
from PISC.potentials import quartic_bistable
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation, Matsubara_Simulation
import time

"""
Test file for testing the fortran integrators 

* The A,B,b,O and Monodromy steps are tested separately
* The NVE and NVT pq_step and Monodromy steps are tested separately

"""

N = 1000 # Number of particles
dim = 2 # Number of dimensions
nbeads = 32 # Number of beads
m=0.5 # Mass of the particles
beta = 1.0 # Inverse temperature

# Standard variable/class declaration for testing
rng = np.random.default_rng(1)
dt = 0.01

pcart = rng.random((N, dim,nbeads))
qcart = rng.random((N, dim,nbeads))

rp = RingPolymer(qcart=qcart,m=m,mode='rp') 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=4) 
rp.bind(ens,motion,rng)

pes = quartic_bistable(0.382,9.375,2.0,0.08,1.0)
#pes = Harmonic(2*np.pi)
pes.bind(ens,rp)

therm = PILE_L(tau0=100.0,pile_lambda=1.0) 

if(0): #Check A,B,b and O steps separately
    propa = Symplectic()
    propa.bind(ens, motion, rp, pes, rng, therm)

    sim = RP_Simulation()
    sim.bind(ens,motion,rng,rp,pes,propa,therm)

    substep = 0
    # A step of the integrator
    rpq = rp.qcart.copy() # Copies are necessary because the integrator modifies the arrays
    rpp = rp.pcart.copy()
    propa.A(substep)
    integrator.a_f(rpq.T, rpp.T,  motion.qdt[substep], rp.dynm3.T)
    assert(np.allclose(rp.qcart, rpq))
    print('A step of the integrator: OK')

    # B step of the integrator
    rpq = rp.qcart.copy()
    rpp = rp.pcart.copy()
    dpot = pes.dpot_cart.copy()
    propa.B(substep)
    propa.rp.mats2cart()
    integrator.b_f(rpp.T, dpot.T,  motion.pdt[substep], True, rp.nbeads)
    assert(np.allclose(rp.pcart, rpp))
    print('B step of the integrator: OK')

    # b step of the integrator
    rpq = rp.qcart.copy()
    rpp = rp.pcart.copy()
    dpot = rp.dpot_cart.copy()
    propa.b(substep)
    propa.rp.mats2cart()
    integrator.b_f(rpp.T, dpot.T,  motion.pdt[substep], True, rp.nbeads)
    assert(np.allclose(rp.pcart, rpp))
    print('b step of the integrator: OK')

    # O step of the integrator
    rpq = rp.q.copy()
    rpp = rp.p.copy()
    rp = RingPolymer(q=rp.q,p=rp.p,m=m,mode='rp')
    rng = np.random.default_rng(1)
    sim.bind(ens,motion,rng,rp,pes,propa,therm,fort=False)
    therm.thalfstep(pc=True,fort=False)

    rppf = rp.p.copy()
    rp = RingPolymer(q=rpq,p=rpp,m=m,mode='rp')
    rng = np.random.default_rng(1)
    sim.bind(ens,motion,rng,rp,pes,propa,therm,fort=False)
    therm.thalfstep(pc=True,fort=False)
    assert(np.allclose(rppf,rp.p))
    print('O step of the integrator: OK')

    # M step of the integrator
    rpq = rp.q.copy()
    rpp = rp.p.copy()
    rp = RingPolymer(q=rp.q,p=rp.p,m=m,mode='rp')
    sim.bind(ens,motion,rng,rp,pes,propa,therm,fort=False)
    propa.M1(substep)
    Mqq_p,Mqp_p,Mpq_p,Mpp_p = rp.Mqq,rp.Mqp,rp.Mpq,rp.Mpp
    
    rp = RingPolymer(q=rpq,p=rpp,m=m,mode='rp')
    rng = np.random.default_rng(1)
    sim.bind(ens,motion,rng,rp,pes,propa,therm,fort=True)
    propa.M1(substep)
    Mqq_f,Mqp_f,Mpq_f,Mpp_f = rp.Mqq,rp.Mqp,rp.Mpq,rp.Mpp

    assert(np.allclose(Mqq_p,Mqq_f))
    assert(np.allclose(Mqp_p,Mqp_f))
    assert(np.allclose(Mpq_p,Mpq_f))
    assert(np.allclose(Mpp_p,Mpp_f))
    print('M step of the integrator: OK')

if(1): #Check one NVT/NVE pq_step
    qcartp = qcart.copy() # Copies are necessary because the integrator modifies the arrays
    qcartf = qcart.copy()
    pcartp = pcart.copy()
    pcartf = pcart.copy()

    motion = Motion(dt=dt, symporder=2)
    
    def run_test(qcart,pcart,fort):
        rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,mode='rp') 
        propa = Symplectic(fort)
                
        rng = np.random.default_rng(1)
        sim = RP_Simulation()
        sim.bind(ens,motion,rng,rp,pes,propa,therm,fort=fort)
     
        start_time = time.time()
        for i in range(25):
            """ Choose any of the following steps to test """
            #propa.O(pmats=True)
            #propa.b(0) 
            
            #pes.update(update_hess=True,fortran=fort)
            #sim.NVE_pqstep()       
            #sim.NVE_pqstep_RSP()
            sim.NVT_pqstep(pc=True)
            sim.NVT_pqstep_RSP(pc=True)
            #sim.NVE_Monodromystep()

            #sim.step(mode='nvt',pc=True)
        print('fortran:',fort,'time', time.time()-start_time)
        return rp.qcart, rp.q, rp.pcart, rp.p, rp.Mqq, rp.Mqp, rp.Mpq, rp.Mpp

    qcartp,qp,pcartp,pp,Mqq_p,Mqp_p,Mpq_p,Mpp_p = run_test(qcartp,pcartp,False)
    qcartf,qf,pcartf,pf,Mqq_f,Mqp_f,Mpq_f,Mpp_f = run_test(qcartf,pcartf,True)

    #print('pcartp',pcartp)
    #print('pcartf',pcartf)
    #print('pf',pf)
    #print('pp',pp)
    
    assert(np.allclose(qcartp,qcartf))
    assert(np.allclose(qp,qf))
    assert(np.allclose(pcartp,pcartf))
    assert(np.allclose(pp,pf))
    assert(np.allclose(Mqq_p,Mqq_f))
    assert(np.allclose(Mqp_p,Mqp_f))
    assert(np.allclose(Mpq_p,Mpq_f))
    assert(np.allclose(Mpp_p,Mpp_f))
    print('NVE/NVT pq_step: OK')

