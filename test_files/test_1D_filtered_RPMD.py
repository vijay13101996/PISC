import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
from PISC.engine.PI_sim_core import SimUniverse
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.Quartic_bistable import quartic_bistable
import time
import os 

dim=1
z=0.5
alpha=0.383

lamda = 2.0
g = 0.08

Vb=lamda**4/(64*g)

print('Vb', Vb)

pes = double_well(lamda,g)

Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.05
dt = 0.002
time_therm = 10.0#100
time_total = 5.0
nbeads = 8

method = 'RPMD'
potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
sysname = 'Selene'		
Tkey = 'T_{}Tc'.format(times)#'{}Tc'.format(times)
corrkey = 'OTOC'
enskey = 'mc_filtered'
#nskey= 'mc'
	
path = os.path.dirname(os.path.abspath(__file__))

#--------------------------------------------------------------------
E = 1.1*Vb
rg_lower=0.2
rg_upper=4
qgrid = np.linspace(-10.0,10.0,int(1e5)+1)
potgrid = pes.potential(qgrid)

qlist = qgrid[np.where(potgrid<E)]
qlist = qlist[:,np.newaxis]
#--------------------------------------------------------------------

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams(E=E,qlist=qlist,filter_lower=rg_lower,filter_upper=rg_upper)
Sim_class.set_runtime(time_therm,time_total)


start_time=time.time()
seeds = (0,)
Sim_class.run_seed(seeds)


param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))


print('time', time.time()-start_time)


