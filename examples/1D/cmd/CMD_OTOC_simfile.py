import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import scipy
from scipy import interpolate
import h5py
import CMD_OTOC
from functools import partial
from PISC.utils.mptools import chunks, batching
import os 

def get_var_value(path):
	filename="{}/CMD_OTOC_simulation_count.dat".format(path)
	with open(filename, "a+") as f:
		f.seek(0)
		val = int(f.read() or 0) + 1
		f.seek(0)
		f.truncate()
		f.write(str(val))
		return val


lamda = 1.5#0.8
g = lamda**2/32#1/50.0
times = 1
m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 20.0
time_total = 10.0

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
sysname = 'Selene'		
	
path = os.path.dirname(os.path.abspath(__file__))

def begin_simulation(nbeads,gamma,rngSeed):
	counter = get_var_value(path)
	#print("This simulation has been run {} times.".format(counter))

	if(0):
		with h5py.File('{}/CMD_OTOC_{}_{}.hdf5'.format(path,sysname,potkey), 'a') as f:
			try:
				group = f.create_group('Run#{}'.format(counter))		
			except:
				pass
				
	CMD_OTOC.main('{}/CMD_OTOC_{}_{}.hdf5'.format(path,sysname,potkey),path,sysname,potkey,counter,lamda,g,times,m,N,nbeads,dt_therm,dt,rngSeed,time_therm,gamma,time_total)

gamma = 16
nbeads = 8

func = partial(begin_simulation, nbeads,gamma)
seeds = range(0,100)
seed_split = chunks(seeds,4)
batching(func,seed_split,max_time=1e6)
