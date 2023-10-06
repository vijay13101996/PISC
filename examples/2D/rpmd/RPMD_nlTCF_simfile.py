import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials.double_well_potential import double_well
from PISC.potentials import quartic, mildly_anharmonic, Tanimura_SB
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 

dim=2
# Tanimura's system-bath potential
m = 1.0
mb= 1.0
delta_anh = 0.1
w_10 = 1.0
wb = w_10
wc = w_10 + delta_anh
alpha = (m*delta_anh)**0.5
D = m*wc**2/(2*alpha**2)

#D = 0.0234 
#alpha = 0.00857
#m = 1.0
#mb = 1.0
#wb = 0.0072901 #(= w_10)
VLL = -0.75*wb#0.05*wb
VSL = 0.75*wb#0.05*wb
cb = 0.35*wb#0.65*wb#0.75*wb

pes = Tanimura_SB(D,alpha,m,mb,wb,VLL,VSL,cb)
			
TinK = 300
K2au = 3.1667e-6
T = 0.125#TinK*K2au
beta = 1/T

potkey = 'Tanimura_SB_D_{}_alpha_{}_VLL_{}_VSL_{}_cb_{}'.format(D,alpha,VLL,VSL,cb)
Tkey = 'T_{}'.format(T)

N = 1000
dt_therm = 0.05#10.0
dt = 0.002#2.0
time_therm = 50.0#60000.0
time_total = 20.0#20000.0

nbeads = 1

op_list = ['p','p','q']

method = 'RPMD'

sysname = 'Selene'		
corrkey = 'R2'
enskey = 'thermal'

### -----------------------------------------------------------------------------
E = T
xg = np.linspace(-3,3,int(1e2)+1)
yg = np.linspace(-3,3,int(1e2)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

print('pot',potgrid.shape)
ind = np.where(potgrid<E)
xind,yind = ind

#fig,ax = plt.subplots(1)
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
path = '/scratch/vgs23/PISC/examples/2D/rpmd'

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)#,ext_kwlist)
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt,op_list)
Sim_class.set_methodparams(nbeads=nbeads)
Sim_class.set_ensparams(tau0=1.0,pile_lambda=1000.0,qlist=qlist)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(10)
seed_split = chunks(seeds,10)

param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
