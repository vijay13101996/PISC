import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import scipy
from scipy import interpolate
import h5py
import CMD_PMF

def get_var_value(filename="./examples/CMD_OTOC_simulation_count.dat"):
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val

potkey = 'inv_harmonic'
			
lamda = 0.8
g = 1/50.0
times = 1
m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 20.0
time_total = 10.0

#gamma = 16
#nbeads = 1
#rngSeed = 4

def begin_simulation(nbeads,gamma,rngSeed):
	counter = get_var_value()
	print("This simulation has been run {} times.".format(counter))

	with h5py.File('./examples/CMD_OTOC_{}.hdf5'.format(potkey), 'a') as f:
		try:
			group = f.create_group('Run#{}'.format(counter))		
		except:
			pass
				
	CMD_OTOC.main('./examples/CMD_OTOC_{}.hdf5'.format(potkey),counter,lamda,g,times,m,N,nbeads,dt_therm,dt,rngSeed,time_therm,gamma,time_total)

# 12 cores for 32 beads, 8 cores for 16 beads, 6 cores for 8 beads and 2 cores for 4 beads.
func = partial(begin_simulation, nbeads,gamma)
seeds = np.range(0,24)
seed_split = chunks(seeds,12)
batching(func,seed_split,max_time=1e7)
