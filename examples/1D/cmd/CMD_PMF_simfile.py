import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import scipy
from scipy import interpolate
import h5py
import CMD_PMF
from PISC.utils.mptools import chunks, batching        
from functools import partial
import os

def get_var_value(path):
	filename = "{}/CMD_PMF_simulation_count.dat".format(path)
	with open(filename, "a+") as f:
		f.seek(0)
		val = int(f.read() or 0) + 1
		f.seek(0)
		f.truncate()
		f.write(str(val))
		return val

potkey = 'inv_harmonic'
sysname = 'Selene'
path = os.path.dirname(os.path.abspath(__file__))

lamda = 0.8
g = 1/50.0
times = 1
m = 0.5
N = 1000
dt = 0.01
time_therm = 20.0
time_relax = 5.0
nsample = 5

def begin_simulation(nbeads,rngSeed):
	counter = get_var_value(path)
	print("This Simulation has been run {} times.".format(counter))

	qgrid = np.linspace(0.0,8.0,101)
	with h5py.File('{}/CMD_PMF_{}_{}.hdf5'.format(path,sysname,potkey), 'a') as f:
		try:
			group = f.create_group('Run#{}'.format(counter))		
		except:
			pass
				
	CMD_PMF.main('{}/CMD_PMF_{}_{}.hdf5'.format(path,sysname,potkey),path,sysname,counter,lamda,g,times,m,N,nbeads,dt,rngSeed,time_therm,time_relax,qgrid,nsample)

# 12 cores for 32 beads, 10 cores for 16 beads, 4 cores for 8 beads and 2 cores for 4 beads.

nbeads = 32
func = partial(begin_simulation, nbeads)
seeds = range(0,100)
seed_split = chunks(seeds,12)
batching(func,seed_split,max_time=1e6)
