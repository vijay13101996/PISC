import numpy as np
import PISC
from PISC.engine.integrators import Symplectic
from PISC.engine.beads import RingPolymer
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.thermalize_PILE_L import thermalize_rp
from PISC.potentials import PotAdder, harmonic1D
from matplotlib import pyplot as plt
from PISC.utils.readwrite import (
    store_1D_plotdata,
    read_1D_plotdata,
    store_arr,
    read_arr,
)
import time

"""
    This code generates an ensemble of initial conditions
    on the stable manifold of a given phase-space saddle point
    and runs OTOC on this ensemble. For a saddle point at the 
    origin, the stable and unstable manifold are given as:
        A_- = p - m*omega*q
        A_+ = p + m*omega*q

    This convention is based on the paper:
    "Does Scrambling Equal Chaos?"
    Tianrui Xu, Thomas Scaffidi, and Xiangyu Cao
    Phys. Rev. Lett. 124, 140602 
    
    The numerical procedure to generate this ensemble is as follows:
    1. Use the standard PILE_L thermostat to sample positions and 
        momenta from the canonical distribution on the PES given by:
        V_s(q) = V(q) + 0.5*m*omega^2*q^2
    2. Reassign the momenta as p = m*omega*q
    3. Store the initial conditions for the ensemble in a file structure 
        similar to the one used for storing canonical distribution data.
    4. Run OTOC (or other dynamical variables) on this ensemble of initial 
        conditions to generate the data for the stable manifold.
"""

def generate_stable_manifold_rp(
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
    Am=0.0,
    lamda=1.0,
    qlist=None,
    tau0=1.0,
    pile_lambda=100.0,
    folder_name="Datafiles",
    store_thermalization=True,
    pes_fort=False,
    propa_fort=False,
    transf_fort=False
):

    """
    Args:
    lamda: The mass-scaled (imaginary) frequency at the saddle point
    Am: The position along the stable manifold where the initial conditions are generated
        (Not implemented yet for Am != 0.0)
    All other arguments are as in thermalize_PILE_L
    """

    # Define the modified potential energy surface
    pes_inv = harmonic1D(m,lamda) #The saddle point is at the origin - needs to be modified when required
    pes_mod = PotAdder(pes,pes_inv)


    if(0): #Plot the modified PES for debugging
        qgrid = np.linspace(-5.0,5.0,1000)
        q = np.zeros((1000,1,1))
        q[:,0,0] = qgrid
        potgrid = pes_mod.potential(q)
        plt.plot(qgrid,potgrid)
        plt.show()
        exit()

    # Thermalize the system on the modified PES
    qcart, pcart = thermalize_rp(pathname,m,dim,N,nbeads,ens,pes_mod,rng,time_therm,
                                 dt_therm,potkey,rngSeed,qlist,tau0,pile_lambda,
                                 folder_name,store_thermalization=False,pes_fort=pes_fort,
                                 propa_fort=propa_fort,transf_fort=transf_fort)
    
    #Reassign momenta
    pcart = m*lamda*qcart

    if store_thermalization:
        store_arr(
            qcart,
            "Stable_manifold_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                N, nbeads, ens.beta, potkey, rngSeed
            ),
            "{}/{}".format(pathname, folder_name),
        )
        store_arr(
            pcart,
            "Stable_manifold_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                N, nbeads, ens.beta, potkey, rngSeed
            ),
            "{}/{}".format(pathname, folder_name),
        )

    return qcart, pcart
