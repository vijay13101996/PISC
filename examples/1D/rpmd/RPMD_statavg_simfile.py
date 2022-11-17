import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.Quartic import quartic
from PISC.engine.PI_sim_core import SimUniverse
import os

def gauss_q(q,p,sigma=2.0,mu=0.0):
	return np.sum(np.exp(-(q-mu)**2/(2*sigma**2))/(sigma*(np.sqrt(2*np.pi))), axis=1) 

def q(q,p):
	return q[:,0,:]

def p2(q,p):
	return p[:,0,:]**2/2

dim=1

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
minima = lamda/np.sqrt(8*g)
print('Vb, minima', Vb,minima)

pes = double_well(lamda,g)

Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.05
dt = 0.002
time_therm = 50.0
time_total = 5.0

nbeads = 32
sigma = 0.21
mu = minima

method = 'RPMD'
potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
sysname = 'Selene'		
Tkey = 'T_{}Tc'.format(times)#'{}Tc'.format(times)
corrkey = 'stat_avg'
enskey = 'thermal'

sigmakey = 'sigma_{}'.format(sigma)
mukey = 'mu_{}'.format(mukey)
kwlist=[sigmakey,mukey]
path = os.path.dirname(os.path.abspath(__file__))

def main():
	Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey,kwlist)
	Sim_class.set_sysparams(pes,T,m,dim)
	Sim_class.set_simparams(N,dt_therm,dt)
	Sim_class.set_methodparams(nbeads=nbeads)
	Sim_class.set_ensparams(tau0=1.0,pile_lambda=100.0)
	Sim_class.set_runtime(time_therm,time_total)

	start_time=time.time()
	func = partial(Sim_class.run_seed,op=partial(gauss_q,sigma=sigma,mu=mu))
	seeds = range(1000)
	seed_split = chunks(seeds,10)

	param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
		'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
			
	with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
		f.write('\n'+str(param_dict))

	batching(func,seed_split,max_time=1e6)
	print('time', time.time()-start_time)

if __name__ == "__main__":
   main()
