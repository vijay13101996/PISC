import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from PISC.engine.PI_sim_core import SimUniverse
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
import time
import os 

dim=1
lamda = 2.0
g = 0.02

Vb = lamda**4/(64*g)

pes = double_well(lamda,g)

Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.05
dt = 0.002
time_therm = 100.0
time_total = 5.0

method = 'Classical'
potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
sysname = 'Selene'		
Tkey = 'T_{}Tc'.format(times)#'{}Tc'.format(times)
corrkey = 'OTOC'#'singcomm'
enskey = 'mc'
	
path = os.path.dirname(os.path.abspath(__file__))

#--------------------------------------------------------------------
E = 4.09#1.5*Vb
qgrid = np.linspace(-10,10,int(1e5)+1)
potgrid = pes.potential(qgrid)

qlist = qgrid[np.where(potgrid<E)]
qlist = qlist[:,np.newaxis]
Ekey = 'E_{}'.format(E)
kwlist=[Ekey]
#--------------------------------------------------------------------

def main():
	Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey,kwlist)
	Sim_class.set_sysparams(pes,T,m,dim)
	Sim_class.set_simparams(N,dt_therm,dt)
	Sim_class.set_methodparams()
	Sim_class.set_ensparams(E=E,qlist=qlist)
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

if __name__ == "__main__":
   main()
