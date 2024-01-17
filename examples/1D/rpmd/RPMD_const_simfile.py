import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials import double_well,quartic,morse, asym_double_well
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 

dim=1
m=0.5

#Double well potential
lamda = 2.0
g = 0.08

print('Vb', lamda**4/(64*g))

pes = double_well(lamda,g)

Tc = 0.5*lamda/np.pi
times = 0.95
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.05
dt = 0.002
time_therm = 50.0
time_total = 5.0

nbeads = 32

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
Tkey = 'T_{}Tc'.format(times)

method = 'RPMD'
sysname = 'Papageno'		
corrkey = 'OTOC'
enskey = 'const_q'

pes_fort=False
propa_fort=True
transf_fort=True

path = os.path.dirname(os.path.abspath(__file__))

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt,pes_fort=pes_fort,propa_fort=propa_fort,transf_fort=transf_fort)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams(tau0=1.0,pile_lambda=100.0)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(100)
seed_split = chunks(seeds,20)

param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
