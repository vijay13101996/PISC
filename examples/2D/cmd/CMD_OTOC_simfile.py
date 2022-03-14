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

m = 0.5
omega = 0.5
g0 = 0.1
T_au = 1.5
beta = 1.0/T_au 
print('T in au',T_au)

potkey = 'coupled_harmonic_w_{}_g_{}'.format(omega,g0)
sysname = 'Selene'		

N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 40.0
time_total = 10.0#15.0

path = os.path.dirname(os.path.abspath(__file__))

def begin_simulation(rngSeed):
	counter = 1#get_var_value(path)
	#print("This simulation has been run {} times.".format(counter))
	
	if(0):
		with h5py.File('{}/CMD_OTOC_{}_{}.hdf5'.format(path,sysname,potkey), 'a') as f:
			try:
				group = f.create_group('Run#{}'.format(counter))		
			except:
				pass
	
	CMD_OTOC.main('{}/CMD_OTOC_{}_{}.hdf5'.format(path,sysname,potkey),path,sysname,potkey,counter,omega,g0,T_au,m,N,nbeads,dt_therm,dt,rngSeed,time_therm,gamma,time_total)

nbeads=4
gamma=16

start_time = time.time()

func = partial(begin_simulation)
seeds = range(0,100)
seed_split = chunks(seeds,5)
batching(func,seed_split,max_time=1e6)

print('time', time.time()-start_time)
