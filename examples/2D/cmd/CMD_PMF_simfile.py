import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import scipy
from scipy import interpolate
import CMD_PMF
from PISC.utils.mptools import chunks, batching        
from functools import partial
import os
from PISC.potentials import quartic_bistable


### FIGURE OUT WHAT IS WRONG WITH INSTANTON FINDER CODE!

def main(nbeads=4,times = 1.0):
    sysname = 'Selene'
    path = os.path.dirname(os.path.abspath(__file__))

    m = 0.5
    
    lamda = 2.0
    g = 0.08
    Vb = lamda**4/(64*g)

    alpha = 0.382
    D = 3*Vb

    z = 1.0
     
    pes = quartic_bistable(alpha,D,lamda,g,z)
    potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
    
    Tc = lamda*(0.5/np.pi)
    T = times*Tc    
    Tkey = 'T_{}Tc'.format(times)
    
    N = 1000
    dt = 0.01
    time_therm = 50.0
    time_relax = 10.0
    nsample = 5

    print('nbeads',nbeads)
    print('Temperature',times,'Tc')

    def begin_simulation(nbeads,rngSeed):
       
        #ygrid = np.linspace(0.0,2.0,21)
        #xgrid = np.zeros_like(ygrid)
        #qgrid = list(zip(xgrid,ygrid))
       
        if(1):
            instpath = '/home/vgs23/PISC/examples/2D/rpmd/Datafiles/'
            fname = 'Instanton2_{}_T_{}Tc_nbeads_{}.dat'.format(potkey,times,nbeads)
            f = open(instpath+fname,'rb')

            arr = pickle.load(f)
            centroid = np.mean(arr,axis=2)
            print('centroid', centroid, centroid.shape)
            qgrid = centroid
        
        qgrid = np.array([[0.0,0.0]])
        CMD_PMF.main('{}/CMD_PMF_{}_{}.txt'.format(path,sysname,potkey),path,sysname,pes,potkey,T,Tkey,m,N,nbeads,dt,rngSeed,time_therm,time_relax,qgrid,nsample)

    # 12 cores for 32 beads, 10 cores for 16 beads, 4 cores for 8 beads and 2 cores for 4 beads.

    func = partial(begin_simulation, nbeads)
    seeds = range(0,12)
    seed_split = chunks(seeds,12)

    start_time = time.time()
    batching(func,seed_split,max_time=1e6)
    print('Total time taken: ',time.time()-start_time)


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='Run Simulations to get CMD PMF and Hessian in two dimensions')
    parser.add_argument('-nbeads',type=int,help='Number of beads',default=8)
    parser.add_argument('-times',type=float,help='Temperature in units of Tc',default=1.0)
    args = parser.parse_args()
    main(args.nbeads,args.times)




