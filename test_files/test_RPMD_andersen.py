#==========================================
# Title:  Test file for Andersen thermostat
# Author: Lars Meuser
# Date:   10 August 2022
#==========================================


import os 
import sys
import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.readwrite import read_arr
from PISC.utils.mptools import chunks, batching
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.potentials.harmonic_2D import Harmonic
from PISC.engine.PI_sim_core import SimUniverse
import time

#Testing
def thermalized_qp(Sim_class,seed_range,T,dim):
	from PISC.engine.ensemble import Ensemble
	rng = np.random.default_rng(seed_range)
	ens = Ensemble(beta=1/T,ndim=dim)
	qcart,pcart = Sim_class.gen_ensemble(ens,rng,seed_range)
	return qcart,pcart

#convergence w.r.t. N, dt_therm, t_therm
def kin_energy_beads(p_arr,m):#p_arr has shape: #N,dim,nbeads
	K_x_beads=(0.5/m)*p_arr[:,0,:]*p_arr[:,0,:]
	#K_x=np.sum(K_x_beads,axis=1)
	K_y_beads=(0.5/m)*p_arr[:,1,:]*p_arr[:,1,:]
	#K_y=np.sum(K_y_beads,axis=1)
	K_tot_beads=K_x_beads+K_y_beads
	#K_tot=K_x+K_y
	return K_tot_beads

def pot_energy_beads(q_arr,pes):
	V_beads=np.zeros((len(q_arr[:,0,0]),len(q_arr[0,0,:])))
	for i in range(len(q_arr[:,0,0])):
		for j in range(len(q_arr[0,0,:])):
			V_beads[i,j]=pes.potential_xy(q_arr[i,0,j],q_arr[i,1,j])
	return V_beads

def spring_energy_beads(q_arr,m,beta_N,hbar=1):
	diff_q_sq=np.zeros((len(q_arr[:,0,0]),len(q_arr[0,0,:])))
	const=0.5*m*(1/(beta_N**2 * hbar**2))
	for i in range(len(q_arr[0,0,:])):
		diff_q_sq[:,i]= (q_arr[:,0,i]-q_arr[:,0,i-1])**2+(q_arr[:,1,i]-q_arr[:,1,i-1])**2
	return const*diff_q_sq

def tot_energy_beads(p_arr,q_arr,m,pes,beta_N,hbar=1):
	T_beads=kin_energy_beads(p_arr,m)
	V_beads=pot_energy_beads(q_arr,pes)
	V_springs=spring_energy_beads(q_arr,m,beta_N,hbar=hbar)
	E_beads=V_beads+T_beads+V_springs
	#E_tot=np.sum(E_beads,axis=1)
	return E_beads 


dim=2

alpha = 0.382
D = 9.375 

lamda = 2.0
g = 0.08

z = 1.5

omega=1
pes= Harmonic(omega)
#pes = quartic_bistable(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc

#------------------------CONVERGENCE PARAMETERS
N =1000#needs to be converged, PILE_L: 4000 # for calculation of Mqq: avg over N time Mqq # should go with sqrtN peak in distribution
dt_therm = 0.05#Convergence Parameter, PILE_L: 0.05  
dt = 0.005
time_therm = 40.0#Convergence Parameter, PILE_L: 2.0
time_total = 0.30#3.0
nbeads = 8 #don't change for now
m = 0.5
#------------------------CONVERGENCE PARAMETERS
method = 'RPMD'
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
sysname = 'Selene'		
Tkey = 'T_{}Tc'.format(times)
corrkey = 'OTOC'
enskey = 'thermal_Andersen'
#enskey = 'thermal'
print(enskey)

path = os.path.dirname(os.path.abspath(__file__))

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams()
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()

func = partial(Sim_class.run_seed)
seeds = range(20)
seed_split = chunks(seeds,10)


param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}


with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

#comment if already used (and saved) that seed
batching(func,seed_split,max_time=1e6)

print('time', time.time()-start_time)
q_arr=np.zeros((len(seeds),N,dim,nbeads))
p_arr=np.zeros_like(q_arr)
for seed in seeds:
	if(enskey == 'thermal'):
		fname='Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(N,nbeads,1/T,potkey,seed)
	elif(enskey == 'thermal_Andersen'):
		fname='Andersen_thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(N,nbeads,1/T,potkey,seed)
	elif(enskey == 'mc'):
		fname='Microcanonical_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(N,nbeads,1/T,potkey,seed)
	elif(enskey == 'thermal_Andersen_filtered'):
		gyr_upper=1#change/initialize somewhere
		gyr_lower=0.1
		fname='Filtered_MC_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}_rg_min{}_rg_max{}'.format(N,nbeads,1/T,potkey,seeds,gyr_lower,gyr_upper)

	q_arr[seed,:,:,:]=read_arr(fname,fpath='{}/Datafiles'.format(path))

	if(enskey == 'thermal'):
		fname='Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(N,nbeads,1/T,potkey,seed)
	elif(enskey == 'thermal_Andersen'):
		fname='Andersen_thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(N,nbeads,1/T,potkey,seed)
	elif(enskey == 'mc'):
		fname='Microcanonical_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(N,nbeads,1/T,potkey,seed)
	elif(enskey == 'thermal_Andersen_filtered'):
		gyr_upper=1#change/initialize somewhere
		gyr_lower=0.1
		fname='Filtered_MC_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}_rg_min{}_rg_max{}'.format(N,nbeads,1/T,potkey,seeds,gyr_lower,gyr_upper)

	p_arr[seed,:,:,:]=read_arr(fname,fpath='{}/Datafiles'.format(path))
beta_N=1/(T*nbeads)
E_tot=np.zeros((len(seeds),N))
E_kin=np.zeros_like(E_tot)
E_pot=np.zeros_like(E_tot)
for seed in seeds:
	E_kin[seed,:]=np.sum(kin_energy_beads(p_arr[seed,:,:,:],m),axis=1)
	E_tot[seed,:]=np.sum(tot_energy_beads(p_arr[seed,:,:,:],q_arr[seed,:,:,:],m,pes,beta_N,hbar=1),axis=1)
E_pot=E_tot-E_kin
print(np.mean(E_kin,axis=1))
print(np.mean(E_pot,axis=1))
if(False):
	key = [method,enskey,corrkey,sysname,potkey,Tkey,'N_{}'.format(N),'dt_{}'.format(dt)]
	fext = '_'.join(key)
	methext = '_nbeads_{}_'.format(nbeads)
	seedext = 'seed_{}'.format(8)
	fname = ''.join([fext,methext,seedext])	
