import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials import quartic_bistable,DW_harm
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 

dim=2
m=0.5

#Double well potential
lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

z = 1.0

if(0):
    alpha = 0.382
    D = 3*Vb
     
    pes = quartic_bistable(alpha,D,lamda,g,z)
    potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

if(1):
    w = 2.0

    pes = DW_harm(m,w,lamda,g,z)
    potkey = 'DW_harm_2D_m_{}_w_{}_lamda_{}_g_{}_z_{}'.format(m,w,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 3.0
T = times*Tc
Tkey = 'T_{}Tc'.format(times)

N = 1000
dt_therm = 0.05
dt = 0.005
time_therm = 50.0
time_total = 5.0

method = 'Classical'
sysname = 'Papageno'		
corrkey = 'fd_OTOC'

filt=False
if(filt):
    kw = 'filt'
else:
    kw = 'nofilt'

const = False
if(const):
    enskey = 'const_q'
    filt = False
    kw = 'const_q'
else:
    enskey = 'thermal'
	
pes_fort=False
propa_fort=True
transf_fort=True

### -----------------------------------------------------------------------------
E = T
xg = np.linspace(-6,6,int(1e2)+1)
yg = np.linspace(-3,3,int(1e2)+1)
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
path = os.path.dirname(os.path.abspath(__file__))

Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey,ext_kwlist=[kw])
Sim_class.set_sysparams(pes,T,m,dim)
Sim_class.set_simparams(N,dt_therm,dt,pes_fort=pes_fort,propa_fort=propa_fort,transf_fort=transf_fort,filt=filt)
Sim_class.set_methodparams(nbeads=1)
Sim_class.set_ensparams(tau0=1e-2,qlist=qlist)
Sim_class.set_runtime(time_therm,time_total)

start_time=time.time()
func = partial(Sim_class.run_seed)
seeds = range(1000)
seed_split = chunks(seeds,20)

param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
	'therm_time':time_therm,'time_total':time_total,'dt':dt,'dt_therm':dt_therm}
		
with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))

batching(func,seed_split,max_time=1e6)
print('time', time.time()-start_time)
