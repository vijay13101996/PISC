import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from PISC.engine import RPMD_core
from PISC.engine.PI_sim_core import SimUniverse
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
import time
import os 

dim=1
lamda = 1.5#0.8
g = 0.13#0.075#0.035#1/50.0

Vb=lamda**4/(64*g)

print('Vb', Vb)

pes = double_well(lamda,g)

Tc = 0.5*lamda/np.pi
times = 0.7
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 40.0
time_total = 5.0
nbeads = 8

method = 'RPMD'
potkey = 'tst2_harmonic_lambda_{}_g_{}'.format(lamda,g)
sysname = 'Selene'		
Tkey = 'T_{}Tc'.format(times)#'{}Tc'.format(times)
corrkey = 'OTOC'
enskey = 'mc'
	
path = os.path.dirname(os.path.abspath(__file__))

E = Vb
qgrid = np.linspace(-10,10,int(1e5)+1)
potgrid = pes.potential(qgrid)

qlist = qgrid[np.where(potgrid<E)]
qlist = qlist[:,np.newaxis]

if(1):
	Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
	Sim_class.set_sysparams(pes,T,m,dim)
	Sim_class.set_simparams(N,dt_therm,dt)
	Sim_class.set_methodparams(nbeads=nbeads)
	Sim_class.set_ensparams(E=E,qlist=qlist)
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

if(0):

	def begin_simulation(rngSeed):
		Classical_core.main(path,sysname,potkey,pes,Tkey,T,m,dim,\
												N,dt_therm,dt,rngSeed,time_therm,time_total,corrkey)
		
	start_time=time.time()
	func = partial(begin_simulation)
	seeds = range(10)
	seed_split = chunks(seeds,10)

	param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
		'therm_time':time_therm,'time_total':time_total,'dt':dt,'dt_therm':dt_therm}
			
	with open('{}/Datafiles/Classical_input_log_{}.txt'.format(path,potkey),'a') as f:	
		f.write('\n'+str(param_dict))

	batching(func,seed_split,max_time=1e6)
	print('time', time.time()-start_time)
