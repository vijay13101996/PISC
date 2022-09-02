import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 

dim=2

alpha = 0.38
D = 9.375 

lamda = 2.0
g = 0.08

z = 1.5
 
pes = quartic_bistable(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 0.95
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 100.0
time_total = 8.0
nbeads = 16

method = 'RPMD'
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
sysname = 'Selene'		
Tkey = 'T_{}Tc'.format(times)
corrkey = 'OTOC'
enskey = 'thermal'
	
path = os.path.dirname(os.path.abspath(__file__))

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams(tau0=1.0,pile_lambda=100.0)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(1000)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
