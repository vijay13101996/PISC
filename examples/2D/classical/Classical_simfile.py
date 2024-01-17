import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 

dim=2

lamda = 2.0
g = 0.08

Vb = lamda**4/(64*g)

alpha = 0.382
D = 3*Vb

z = 1.0
 
pes = quartic_bistable(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 3.0#0.6
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.05
dt = 0.002#0.002
time_therm = 50.0
time_total = 5.0#5.0

method = 'Classical'
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
sysname = 'Papageno'		
Tkey = 'T_{}Tc'.format(times)
corrkey = 'OTOC'
enskey = 'const_q'#'thermal'
	
path = os.path.dirname(os.path.abspath(__file__))
### -----------------------------------------------------------------------------
E = T
xg = np.linspace(-6,6,int(1e2)+1)
yg = np.linspace(-5,10,int(1e2)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

print('pot',potgrid.shape,T)
ind = np.where(potgrid<E)
xind,yind = ind

#fig,ax = plt.subplots(1)
#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/30))
	
qlist= []
for x,y in zip(xind,yind):
	#x = i[0]
	#y = i[1]
	#expbeta = np.exp(-pes.potential_xy(xgrid[x,y],ygrid[x,y])/T)/2.0
	#ax.scatter( xgrid[x,y],ygrid[x,y],alpha=expbeta)#xgrid[x][y] , ygrid[x][y] )
	qlist.append([xgrid[x,y],ygrid[x,y]])

#plt.show()
qlist = np.array(qlist)	
### ------------------------------------------------------------------------------
Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt)
Sim_class.set_methodparams(nbeads=1)
Sim_class.set_ensparams(tau0=1e-2)#,qlist=qlist)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(100)
seed_split = chunks(seeds,20)

param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/Classical_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
