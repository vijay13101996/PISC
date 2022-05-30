import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from PISC.engine import RPMD_core
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.Quartic import quartic
import time
import os 

dim=1
lamda = 1.5#0.8
g = 0.035#1/50.0

pes = double_well(lamda,g)

Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 40.0
time_total = 20.0

potkey = 'testinv_harmonic_lambda_{}_g_{}'.format(lamda,g)
sysname = 'Selene'		
Tkey = '{}Tc'.format(times)
corrkey = 'qq'
	
path = os.path.dirname(os.path.abspath(__file__))

def begin_simulation(nbeads,rngSeed):
	RPMD_core.main(path,sysname,potkey,pes,Tkey,T,m,dim,\
											N,nbeads,dt_therm,dt,rngSeed,time_therm,time_total,corrkey)
	
nbeads = 4

start_time=time.time()

func = partial(begin_simulation, nbeads)
seeds = range(100)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
