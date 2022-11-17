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

m = 0.5
dim=2

alpha = 0.382
D = 9.375 

lamda = 2.0
g = 0.08
wm = (2*D*alpha**2/m)**0.5

z = 1.0

Vb = lamda**4/(64*g)
 
pes = quartic_bistable(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 10.0
T = times*Tc

N = 1000
dt_therm = 0.05
dt = 0.002
time_therm = 100.0
time_total = 2.5

method = 'Classical'
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
sysname = 'Selene'		
Tkey = 'T_{}Tc'.format(times)
corrkey = 'OTOC'
enskey = 'mc'
	
path = os.path.dirname(os.path.abspath(__file__))

### -----------------------------------------------------------------------------
E = 2*Vb 
xg = np.linspace(-6,6,int(1e2)+1)
yg = np.linspace(-5,10.0,int(1e2)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

extkey = ['E_2Vb']

print('pot',potgrid.shape)
ind = np.where(potgrid<E)
xind,yind = ind

#--------------------------------------------------------------------------
qlist= []
plist= []
for x,y in zip(xind,yind):
	#x = i[0]
	#y = i[1]
	#ax.scatter( xgrid[x,y],ygrid[x,y])#xgrid[x][y] , ygrid[x][y] )
	qlist.append([xgrid[x,y],ygrid[x,y]])
	V = pes.potential_xy(xgrid[x,y],ygrid[x,y])
	p = np.sqrt(2*m*(E-V))
	plist.append([p,0.0])

#plt.show()
qlist = np.array(qlist)
plist = np.array(plist)	

#fig,ax = plt.subplots(1)
#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/30))

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey,extkey)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt)
Sim_class.set_methodparams(nbeads=1)
Sim_class.set_ensparams(E=E,qlist=qlist)#,filt_func=cross_filt)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(100)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/Classical_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)


### ------------------------------------------------------------------------------
def cross_filt(rp,rp0): # Function to filter those which crosses the 'barrier'
	q0 = rp0.q[:,0,0]
	qt = rp.q[:,0,0]
	#print('rp',q0[0],qt[0])
	ind = np.where(q0*qt < 1e-4)
	return ind

### ------------------------------------------------------------------------------

