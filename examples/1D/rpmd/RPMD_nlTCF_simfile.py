import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
from PISC.potentials import quartic, mildly_anharmonic, morse
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 

dim=1
m = 1.0

if(0): #Quartic
	a = 1.0
	pes = quartic(a)
	potkey = 'quartic_a_{}'.format(a) 

if(0): #Mildly anharmonic
	omega = 1.0
	a = 0.4#-0.605#0.5#1.0/10
	b = a**2#0.427#a**2#1.0/100
	pes = mildly_anharmonic(m,a,b)
	
	T = 1.0
	potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,b)
	Tkey = 'T_{}'.format(np.around(T,3))

if(0): #Double well
	lamda = 2.0
	g = 0.08
	Vb=lamda**4/(64*g)
	print('Vb', Vb)
	pes = double_well(lamda,g)

	Tc = 0.5*lamda/np.pi
	times = 0.95
	T = times*Tc

	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
	Tkey = 'T_{}Tc'.format(times)
	
	E = T
	qgrid = np.linspace(-6,6,int(1e2)+1)
	potgrid = pes.potential(qgrid)

	ind = np.where(potgrid<E)
	
	qlist = potgrid[ind]
	qlist = qlist[:,None]

if(1): #Morse
	#D = 0.0234
	#alpha = 0.00857
	
	delta_anh = 0.05
	w_10 = 1.0
	wb = w_10
	wc = w_10 + delta_anh
	alpha = (m*delta_anh)**0.5
	D = m*wc**2/(2*alpha**2)


	pes = morse(D,alpha)
	T = 1.0#TinK*K2au
	beta = 1/T

	potkey = 'Morse_D_{}_alpha_{}'.format(D,alpha)
	Tkey = 'T_{}'.format(T)
	
	E = T
	qgrid = np.linspace(-6,6,int(1e3)+1)
	potgrid = pes.potential(qgrid)

	ind = np.where(potgrid<E)
	
	qlist = potgrid[ind]
	qlist = qlist[:,None]


N = 10
dt_therm = 0.05
dt = 0.5
time_therm = 1.0
time_total = 1.0

nbeads = 1


method = 'RPMD'

sysname = 'Selene'		
op_list = ['p','p','q'] #t2,t1,t0
corrkey = 'R2'
enskey = 'thermal'
#ext_kwlist = ['qqq']
	
#path = os.path.dirname(os.path.abspath(__file__))
path = '/scratch/vgs23/PISC/examples/1D/rpmd'

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)#,ext_kwlist)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt,op_list)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams(tau0=1.0,pile_lambda=1000.0,qlist=qlist)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(1)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
