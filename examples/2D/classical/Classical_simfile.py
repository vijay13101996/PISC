import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from PISC.engine import Classical_core
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.Quartic_bistable import quartic_bistable
import time
import os 

dim=2

alpha = 0.363
D = 10.0 

lamda = 2.0
g = 0.08

z = 0.0#1.5
 
pes = quartic_bistable(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 40.0
time_total = 10.0

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
sysname = 'Selene'		
Tkey = '{}Tc'.format(times)
corrkey = 'OTOC'
	
path = os.path.dirname(os.path.abspath(__file__))

def begin_simulation(rngSeed):
	Classical_core.main(path,sysname,potkey,pes,Tkey,T,m,dim,\
											N,dt_therm,dt,rngSeed,time_therm,time_total,corrkey)
	
start_time=time.time()
func = partial(begin_simulation)
seeds = range(1000)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/Classical_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
