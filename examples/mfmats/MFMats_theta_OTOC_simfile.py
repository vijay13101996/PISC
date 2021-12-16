import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import scipy
from scipy import interpolate
import h5py
import MFMats_theta_OTOC
from functools import partial
from PISC.utils.mptools import chunks, batching

def get_var_value(filename="./examples/mfmats/MFMats_theta_OTOC_simulation_count.dat"):
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val

potkey = 'inv_harmonic'
sysname = 'Selene'		
	
lamda = 0.8
g = 1/50.0
times = 1
m = 0.5
N = 10
dt_therm = 0.01
dt = 0.005
time_therm = 20.0
time_total = 10.0

def begin_simulation(nbeads,theta,gamma,rngSeed):
	counter = get_var_value()
	print("This simulation has been run {} times.".format(counter))

	with h5py.File('./examples/mfmats/MFMats_theta_OTOC_{}_{}.hdf5'.format(sysname,potkey), 'a') as f:
		try:
			group = f.create_group('Run#{}'.format(counter))		
		except:
			pass
				
	MFMats_theta_OTOC.main('./examples/mfmats/MFMats_theta_OTOC_{}_{}.hdf5'.format(sysname,potkey),sysname,\
	potkey,nrun,lamda,g,times,m,N,nbeads,nmats,dt_therm,dt,time_therm,gamma,time_total,theta,rngSeed)

# 12 cores for 32 beads, 8 cores for 16 beads, 6 cores for 8 beads and 2 cores for 4 beads.
gamma = 16
nbeads = 16

theta = 0.0
func = partial(begin_simulation, nbeads,theta,gamma)
seeds = range(0,2)
seed_split = chunks(seeds,12)
batching(func,seed_split,max_time=1e6)
