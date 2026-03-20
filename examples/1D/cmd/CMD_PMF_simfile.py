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

"""
To do:
    1. Benchmark the PMF with the classical potential.
    2. Evaluate the time taken for the 8,16,32 beads simulations.
    3. Setup simulations on papageno.
    4. Read through Stuart's emails once again.
    5. Start work for the 2D case right away. 
"""


def main(nbeads=16,times = 1.0):
    potkey = 'inv_harmonic'
    sysname = 'Papageno'
    path = os.path.dirname(os.path.abspath(__file__))

    lamda = 2.0
    g = 0.08
    m = 0.5

    N = 1000
    dt = 0.05
    time_therm = 50.0
    time_relax = 10.0
    nsample = 5

    print('nbeads',nbeads)
    print('Temperature',times,'Tc')


    def begin_simulation(nbeads,rngSeed):
        qgrid = [0.0]#np.linspace(-12.0,12.0,101) 
        CMD_PMF.main('{}/CMD_PMF_origin_{}_{}.txt'.format(path,sysname,potkey),path,sysname,lamda,g,times,m,N,nbeads,dt,rngSeed,time_therm,time_relax,qgrid,nsample)

    # 12 cores for 32 beads, 10 cores for 16 beads, 4 cores for 8 beads and 2 cores for 4 beads.

    func = partial(begin_simulation, nbeads)
    seeds = range(20)
    seed_split = chunks(seeds,20)
    
    start_time = time.time()

    batching(func,seed_split,max_time=1e6)

    print('time', time.time()-start_time)

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='Run Simulations to get CMD PMF and Hessian in one dimension')
    parser.add_argument('-nbeads',type=int,help='Number of beads',default=16)
    parser.add_argument('-times',type=float,help='Temperature in units of Tc',default=1.0)
    args = parser.parse_args()
    main(args.nbeads,args.times)




