import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import scipy
from scipy import interpolate
import h5py
import CMD_PMF

lamda = 0.8
g = 1/50.0
times = 2
m = 0.5
N = 10
nbeads = 1
dt = 0.01
rngSeed = 2
time_therm = 20.0
time_relax = 5.0
nsample = 5
	
qgrid = np.linspace(0.0,8.0,11)
potkey = 'inv_harmonic'
with h5py.File('./examples/CMD_PMF_{}.hdf5'.format(potkey), 'r') as f:	
	for run in f.keys():
		for k in f[run].attrs.keys():
			print('k',f[run].attrs[k])
		
		
	if(0):
		group = f.create_group('Run#{}'.format(counter))
		group.attrs['lambda'] = lamda
		group.attrs['g'] = g
		group.attrs['T'] = T
		group.attrs['xTc'] = times
		group.attrs['m'] = m
		group.attrs['N'] = N
		group.attrs['nbeads'] = nbeads
		group.attrs['dt'] = dt	
		group.attrs['therm_time'] = time_therm
		group.attrs['relax_time'] = time_relax
		group.attrs['nsample'] = nsample
		group.attrs['seed'] = rngSeed
		group.attrs['seed'] = rngSeed

	
