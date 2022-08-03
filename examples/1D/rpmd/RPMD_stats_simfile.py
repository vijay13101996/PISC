import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from PISC.engine.PI_sim_core import SimUniverse
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.harmonic_1D import harmonic
import time
import os 

dim=1
m=0.5
#------------------------------------------------------
lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
pes = double_well(lamda,g)

Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc
Tkey = 'T_{}Tc'.format(times)

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

#------------------------------------------------------
N = 1000
dt_therm = 0.05
time_therm = 100.0
nbeads = 8
method = 'RPMD'

#------------------------------------------------------
sysname = 'Selene'
corrkey = ''
enskey = 'thermal'

#-----------------------------------------------------
qgrid = np.linspace(-10.0,10.0,int(1e5)+1)
potgrid = pes.potential(qgrid) 

E=1.5*Vb
qlist = qgrid[np.where(potgrid<E)]
qlist = qlist[:,np.newaxis]

#-----------------------------------------------------

path = os.path.dirname(os.path.abspath(__file__))

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams(tau0=1.0, pile_lambda=1e2,E=E,qlist=qlist)
Sim_class.set_runtime(time_therm)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(1000)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
