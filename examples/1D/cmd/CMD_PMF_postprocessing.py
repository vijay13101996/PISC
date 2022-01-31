import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import scipy
from scipy import interpolate
import h5py
import CMD_PMF
import os
from PISC.utils.readwrite import read_1D_plotdata

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

path = os.path.dirname(os.path.abspath(__file__))
datapath = '{}/Datafiles'.format(path)

if(0):
	keyword1 = 'CMD_PMF'
	keyword2 = 'nbeads_{}'.format(nbeads)

	flist = []

	for fname in os.listdir(datapath):
		if keyword1 in fname and keyword2 in fname:
			flist.append(fname)

	for f in flist:
		data = read_1D_plotdata('{}/{}'.format(datapath,f))
		qgrid = data[:,0]
		fgrid = data[:,1]
		plt.plot(qgrid,fgrid)
	
	fgrid/=count
    plt.plot(qgrid,fgrid)
    plt.show()
    store_1D_plotdata(qgrid,fgrid,'CMD_PMF_T_1Tc_nbeads_{}'.format(nbeads),datapath)

if(1):
    fig = plt.figure()
    plt.suptitle(r'CMD PMF at $T=T_c$')
    for i in [4,8,16,32]:
        #print('i', i)
        data = read_1D_plotdata('{}/CMD_PMF_T_1Tc_nbeads_{}.txt'.format(datapath,i))
        qgrid = data[:,0]
        fgrid = data[:,1]
        plt.plot(qgrid,fgrid,label=r'$N_b$ = {}'.format(i))

    plt.plot(qgrid,pes.dpotential(qgrid),label=r'Classical')
    plt.legend()
    plt.show() 	

if(0):
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

	
