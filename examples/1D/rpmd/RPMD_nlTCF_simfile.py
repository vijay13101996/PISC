import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
from PISC.potentials import quartic, mildly_anharmonic
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 

dim=1
m = 1.0

if(0): #Quartic
	a = 1.0
	pes = quartic(a)
	potkey = 'quartic_a_{}'.format(a) 

if(1): #Mildly anharmonic
	omega = 1.0
	a = 1.0/10
	b = 1.0/100
	pes = mildly_anharmonic(m,a,b)
	potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,b)

Tc = 1.0#0.5*lamda/np.pi
times = 1.0
T = 1.0/8#times*Tc

N = 1000
dt_therm = 0.05
dt = 0.01
time_therm = 50.0
time_total = 30.0

nbeads = 1

op_list = ['p','p','q']

method = 'RPMD'

sysname = 'Selene'		
Tkey = 'T_{}'.format(T)#'{}Tc'.format(times)
corrkey = 'R2'
enskey = 'thermal'
#ext_kwlist = ['qqq']
	
path = os.path.dirname(os.path.abspath(__file__))
#'/scratch/vgs23/PISC/examples/1D/rpmd'#

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)#,ext_kwlist)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt,op_list)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams(tau0=1.0,pile_lambda=100.0)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(10)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
