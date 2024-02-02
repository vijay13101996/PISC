import numpy as np
import PISC
from PISC.engine.integrators import Symplectic
from PISC.engine.beads import RingPolymer
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import (
    store_1D_plotdata,
    read_1D_plotdata,
    store_arr,
    read_arr,
)
import time

"""
This function is used to thermalize a ring polymer ensemble
at a constant value of centroid position (and momentum). This
is very similar to the code that generates centroid PMF (and in
fact, it can be used to generate centroid PMF as well). 
"""

def thermalize_rp_const_qp(
    pathname,
    m,
    dim,
    N,
    nbeads,
    ens,
    pes,
    rng,
    time_therm,
    dt_therm,
    potkey,
    rngSeed,
    tau0=1.0,
    pile_lambda=100.0,
    folder_name="Datafiles",
    store_thermalization=True,
    pes_fort=False,
    propa_fort=False,
    transf_fort=False,
    qp=False,
    Q0 = 0.0,
    P0 = 0.0
):
     
    q = np.zeros((N, dim, nbeads))
    p = np.zeros((N, dim, nbeads))
    for i in range(nbeads):
        p[:, :, i] = rng.normal(0.0, (m / ens.beta) ** 0.5, (N, dim))
    
    if qp:
        p[:, :, 0] = P0 # Set the momentum of the centroid to P0
    
    q[:, :, 0] = Q0 

    # Initialize ring polymers with collapsed configuration at these points
    rp = RingPolymer(q=q, p=p, m=m)

    motion = Motion(dt=dt_therm, symporder=2)
    therm = PILE_L(tau0=tau0, pile_lambda=pile_lambda)
    propa = Symplectic()
    sim = RP_Simulation()
    sim.bind(ens, motion, rng, rp, pes, propa, therm, 
             transf_fort=transf_fort, pes_fort=pes_fort, propa_fort=False)
    # propa_fort is set to False because centmove=False is not implemented in FORTRAN for the A step
    start_time = time.time()

    nthermsteps = int(time_therm / motion.dt)
    
    """ The commented code below is for debugging purposes. """

    if not qp:
        # Thermalise ring polymer momentum if only the position is constrained
        sim.step(ndt=nthermsteps, mode="nvt", var="pq", RSP=True, pc=True)
        # Reset the position to the constraint value
        sim.rp.q[:, 0, 0] = Q0/nbeads**0.5 
        # The above line needs to be updated when the constraint position is not zero (or 1D)
        sim.propa.force_update()
    
        # Thermalise ring polymer position at the constrained value
        sim.propa.centmove = False
        sim.propa.rebind()
        sim.step(ndt=nthermsteps, mode="nvt", var="pq", RSP=False, pc=True)


    # Check if the thermalization is successful
    if qp:
        assert(np.allclose(sim.rp.p[:,:,0], P0))
        assert(np.allclose(sim.rp.q[:,:,0], Q0/nbeads**0.5))
        print('All momenta are equal to P0 and all positions are equal to Q0')
    else:
        assert(np.allclose(sim.rp.q[:,0,0], Q0/nbeads**0.5))
        print('All positions are equal to Q0') 
    
    print(
        "End of thermalization. Seed: {} Classical Kinetic energy {:5.3f} Target value {:5.3f} ".format(
            rngSeed, rp.kin.sum() / rp.nsys, 0.5 * rp.ndim * rp.nbeads**2 / ens.beta
        )
    )

    if store_thermalization:
        qpkey = "qp" if qp else "q"
        store_arr(
            rp.qcart,
            "Const_{}_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                qpkey, rp.nsys, rp.nbeads, ens.beta, potkey, rngSeed
            ),
            "{}/{}".format(pathname, folder_name),
        )
        store_arr(
            rp.pcart,
            "Const_{}_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                qpkey, rp.nsys, rp.nbeads, ens.beta, potkey, rngSeed
            ),
            "{}/{}".format(pathname, folder_name),
        )

    return rp.qcart, rp.pcart

