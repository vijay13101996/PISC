import numpy as np
import PISC
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import os

m=0.5
N=50#20
dt=0.005

w = 0.5
D = 10.0
alpha = 0.363

lamda = 2.0
g = 0.08

Vb = lamda**4/(64*g)

z = 1.25
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc
Tkey = 'T_{}Tc'.format(times) 

pes = quartic_bistable(alpha,D,lamda,g,z)

pathname = os.path.dirname(os.path.abspath(__file__))

E = Vb
xg = np.linspace(-8,8,int(1e2)+1)
yg = np.linspace(-5,10,int(1e2)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

print('pot',potgrid.shape)
ind = np.where(potgrid<E)
xind,yind = ind

qlist = []

fig,ax = plt.subplots(1)

for x,y in zip(xind,yind):
	#x = i[0]
	#y = i[1]
	#ax.scatter( xgrid[x,y],ygrid[x,y])#xgrid[x][y] , ygrid[x][y] )
	qlist.append([xgrid[x,y],ygrid[x,y]])

ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/30))
plt.show()

qlist = np.array(qlist)
print('qlist',qlist.shape)

nbeads = 5
PSOS = Poincare_SOS('Classical',pathname,potkey,Tkey)
PSOS.set_sysparams(pes,T,m,2)
PSOS.set_simparams(N,dt,dt,nbeads=nbeads)	
PSOS.set_runtime(20.0,500.0)
PSOS.bind(qcartg=qlist,E=E)

if(0):
	PSOS.run_traj(0,ax)
	plt.show()
	
if(0):
	X,PX,Y = PSOS.PSOS_X(y0=0.0)#-0.267)
	plt.title(r'PSOS, $N_b={}$'.format(nbeads))
	plt.scatter(X,PX,s=2)
	plt.show()
	#fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
	#store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))
			
	
