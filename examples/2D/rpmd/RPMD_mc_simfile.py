import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
from PISC.engine.PI_sim_core import SimUniverse
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.Quartic_bistable import quartic_bistable
import time
import os 

dim=2

lamda = 2.0
g = 0.08

Vb = lamda**4/(64*g)
 
alpha = 0.382
D = 3*Vb#9.375 

z = 1.0#1.0

pes = quartic_bistable(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 10.0
T = times*Tc

m = 0.5
N = 100
dt_therm = 0.05
dt = 0.002
time_therm = 100.0
time_total = 5.0
nbeads = 1

method = 'RPMD'
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
sysname = 'Selene'		
Tkey = 'T_{}Tc'.format(times)
corrkey = 'OTOC'
enskey = 'mc'
	
path = os.path.dirname(os.path.abspath(__file__))

### -----------------------------------------------------------------------------
E = 1.001*Vb
xg = np.linspace(-6,6,int(1e2)+1)
yg = np.linspace(-5,10,int(1e2)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

print('pot',potgrid.shape)
ind = np.where(potgrid<E)
xind,yind = ind

fig,ax = plt.subplots(1)
#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/30))

qlist= []
for x,y in zip(xind,yind):
	#x = i[0]
	#y = i[1]
	#ax.scatter( xgrid[x,y],ygrid[x,y])#xgrid[x][y] , ygrid[x][y] )
	qlist.append([xgrid[x,y],ygrid[x,y]])
#plt.show()
qlist = np.array(qlist)
	
### ------------------------------------------------------------------------------
Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams(E=E,qlist=qlist)#,filt_func=cross_filt)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(100)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)

def cross_filt(rp,rp0):
	q0 = rp0.q[:,0,0]
	qt = rp.q[:,0,0]
	ind = np.where(q0*qt < 0.0)
	return ind	
