#=====================================================
# Title:  Test file for filtered mc 2D RPMD simulation
# Author: Lars Meuser
# Date:   10 August 2022
#=====================================================

import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import sys
from PISC.engine.PI_sim_core import SimUniverse
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.Quartic_bistable import quartic_bistable
import time
import os 

dim=2
m=0.5
D = 9.375
z=1
alpha=0.37

lamda = 2.0
g = 0.08

Vb=lamda**4/(64*g)

print('Vb', Vb)

pes = quartic_bistable(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc

m = 0.5
N = 2000
dt_therm = 0.05
dt = 0.002
time_therm = 100.0#100
time_total = 0.1#5
nbeads = 8

method = 'RPMD'
potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
sysname = 'Selene'		
Tkey = 'T_{}Tc'.format(times)#'{}Tc'.format(times)
corrkey = 'OTOC'
enskey = 'mc_filtered'
#enskey= 'mc'

print(enskey, 'Therm_time: ', time_therm)
path = os.path.dirname(os.path.abspath(__file__))

#--------------------------------------------------------------------
E = 1.01*Vb
rg_lower=0.0
rg_upper=0.1

xmin=2.4999
ymin=-1.0502
xg = np.linspace(-5,5,int(5*1e2)+1)
yg = np.linspace(ymin-1,ymin+8.5,int(5*1e2)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

ind = np.where(potgrid<E)
xind,yind = ind #ind are two arrays with e.g. 1740 elements, which fullfil energy condition  
qlist = []#list of q values
for x,y in zip(xind,yind):
	qlist.append([xgrid[x,y],ygrid[x,y]])
qlist = np.array(qlist)
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


