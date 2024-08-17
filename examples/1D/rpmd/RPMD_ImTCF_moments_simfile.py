import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials import quartic, double_well, harmonic1D, mildly_anharmonic
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 

dim=1

if(1):
    m=1.0
    omega = 1.0
    
    pes = harmonic1D(m,omega)
    potkey = 'harmonic_omega_{}'.format(omega)
    
    T = 1.
    Tkey = 'T_{}'.format(T)

if(0):
    m = 1.0
    a = 1.
    
    pes = quartic(a)
    potkey = 'quartic_a_{}'.format(a)
    
    T = 0.25
    Tkey = 'T_{}'.format(T)

if(0):
    m=0.5
    lamda = 2.0
    g = 0.1
    
    pes = double_well(lamda,g)
    Tc = 0.5*lamda/np.pi
    
    potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

    times = 1.
    T = times*Tc
    Tkey = 'T_{}Tc'.format(times)

if(0):
    m=1.0
    a=0.0
    b=0.0

    pes = mildly_anharmonic(m,a,b)
    potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,b)

    T = 1
    Tkey =  'T_{}'.format(T)

N = 1000
dt_therm = 0.05
dt = 0.01
time_therm = 50.0
time_total = 20.0

nbeads = 15

method = 'RPMD'
sysname = 'Selene'		
corrkey = 'ImTCF_moments'
enskey = 'thermal'
	
path = os.path.dirname(os.path.abspath(__file__))

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams(tau0=1.0,pile_lambda=100.0)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed,nmoments=10)
seeds = range(100)
seed_split = chunks(seeds,20)

param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)

