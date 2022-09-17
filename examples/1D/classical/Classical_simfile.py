import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
from functools import partial
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 

dim=1
lamda = 2.0
g = 0.08

pes = double_well(lamda,g)

Tc = 0.5*lamda/np.pi
for times in (0.7,0.8,0.9,0.95):
	T = times*Tc

	m = 0.5
	N = 1000
	dt_therm = 0.05
	dt = 0.002
	time_therm = 40.0
	time_total = 6.0

	path = os.path.dirname(os.path.abspath(__file__))

	method = 'Classical'
	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
	sysname = 'Selene'		
	Tkey = 'T_{}Tc'.format(times)#'{}Tc'.format(times)
	corrkey = 'OTOC'
	enskey = 'thermal'
		
	path = os.path.dirname(os.path.abspath(__file__))

	Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
	Sim_class.set_sysparams(pes,T,m,dim)
	Sim_class.set_simparams(N,dt_therm,dt)
	Sim_class.set_methodparams()
	Sim_class.set_ensparams(tau0=1.0)
	Sim_class.set_runtime(time_therm,time_total)

	start_time=time.time()
	func = partial(Sim_class.run_seed)
	seeds = range(100)
	seed_split = chunks(seeds,10)

	param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
		'therm_time':time_therm,'time_total':time_total,'dt':dt,'dt_therm':dt_therm}
			
	with open('{}/Datafiles/Classical_input_log_{}.txt'.format(path,potkey),'a') as f:	
		f.write('\n'+str(param_dict))

	batching(func,seed_split,max_time=1e6)
	print('time', time.time()-start_time)
