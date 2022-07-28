import numpy as np
import time
import pickle
from PISC.engine import CMD_core
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
import os 

dim=2

alpha = 0.255
D = 10.0 

lamda = 1.5
g = 0.035

z = 0.5
 
pes = quartic_bistable(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 0.7
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

def begin_simulation(nbeads,gamma,rngSeed):
	CMD_core.main(path,sysname,potkey,pes,Tkey,T,m,dim,\
											N,nbeads,dt_therm,dt,rngSeed,time_therm,gamma,time_total,corrkey)
	
gamma = 1
nbeads = 4

start_time=time.time()

func = partial(begin_simulation, nbeads,gamma)
seeds = range(0,10)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'gamma':gamma,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/CMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)

print('time', time.time()-start_time)
