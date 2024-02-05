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
import time
import pickle

def main(filename,pathname,sysname,lamda,g,times,m,N,nbeads,dt,rngSeed,time_therm,time_relax,qgrid,nsample):
    dim = 1
    Tc = lamda*(0.5/np.pi)
    T = times*Tc    
    beta = 1/T
   
    # Set up the simulation
    rng = np.random.default_rng(rngSeed)
    ens = Ensemble(beta=beta,ndim=dim)
    motion = Motion(dt = dt,symporder=2) 
    pes = double_well(lamda,g)
    therm = PILE_L(tau0=1.0,pile_lambda=1.0) 
    propa = Symplectic()
    sim = RP_Simulation()

    # Set up the simulation parameters
    nsteps_therm = int(time_therm/dt)
    nsteps_relax = int(time_relax/dt)

    # Force grid and Hessian grid
    fgrid = np.zeros_like(qgrid)
    hessgrid = np.zeros_like(qgrid)
 
    start_time = time.time()
    for k in range(len(qgrid)):
        
        # Initialize the positions in normal mode coordinates and
        # fix the centroid mode to the value of qgrid[k]
        q = np.zeros((N,dim,nbeads))
        q[...,0] = qgrid[k]

        # Initialize the momenta in normal mode coordinates
        # and freeze the centroid mode with p[...,0] = 0.0
        p = rng.normal(size=q.shape)
        p[...,0] = 0.0
        
        # Initialize the ring polymer
        rp = RingPolymer(q=q,p=p,m=m,nmats=1,sgamma=1)

        # Bind the simulation object with the various components
        sim.bind(ens,motion,rng,rp,pes,propa,therm,pes_fort=False,propa_fort=True,transf_fort=True)  
        
        # Thermalize the system ensuring that the centroid mode is frozen    
        sim.propa.centmove = False
        sim.propa.update_hess = True
        sim.propa.rebind()
        sim.step(ndt=nsteps_therm, mode="nvt", var="pq", RSP=False, pc=False)
    
        if(rngSeed==0):
            print('k, q[k]',k, rp.q[0,0,0])
        
        # Compute the force and Hessian at the centroid mode as an average of 'nsample' samples
        for i in range(nsample):
            sim.step(ndt=nsteps_relax,mode="nvt",var='pq',RSP=False,pc=False)
            fgrid[k]+=np.mean(pes.dpot[...,0])  # Centroid force
            #print('force',pes.dpot[...,0], np.mean(pes.dpot[...,0]))
            #print('rp.q, rp.p',rp.q, rp.p)
            #print('hessian', np.mean(pes.ddpot[:,0,0,0,0]))
            hessgrid[k]+=np.mean(pes.ddpot[:,0,0,0,0]) #- (np.mean(pes.dpot[...,0])**2) # Centroid Hessian
        #print('time',time.time()-start_time)

    fgrid/=nsample
    hessgrid/=nsample

    #print('grid',qgrid)

    fname = 'CMD_PMF_ORIGIN_{}_inv_harmonic_T_{}Tc_N_{}_nbeads_{}_dt_{}_thermtime_{}_relaxtime_{}_nsample_{}_seed_{}'.format(sysname,times,N,nbeads,dt,time_therm,time_relax,nsample,rngSeed)
    store_1D_plotdata(qgrid,fgrid,fname,"{}/Datafiles".format(pathname))

    #print('grid 2',qgrid)
    fname = 'CMD_Hess_ORIGIN_{}_inv_harmonic_T_{}Tc_N_{}_nbeads_{}_dt_{}_thermtime_{}_relaxtime_{}_nsample_{}_seed_{}'.format(sysname,times,N,nbeads,dt,time_therm,time_relax,nsample,rngSeed)
    store_1D_plotdata(qgrid,hessgrid,fname,"{}/Datafiles".format(pathname))
