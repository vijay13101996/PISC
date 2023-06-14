import numpy as np
import PISC
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials import Tanimura_SB
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from Saddle_point_finder import separatrix_path, find_minima
import time
import os
import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False
### Potential parameters
N=60#30
dt=0.005

m=1.0
mb= 1.0
delta_anh = 0.1
w_10 = 1.0
wb = w_10
wc = w_10 + delta_anh
alpha = (m*delta_anh)**0.5
D = m*wc**2/(2*alpha**2)

VLL = -0.75*wb
VSL = 0.75*wb
cb = 0.65#0.45*wb

pes = Tanimura_SB(D,alpha,m,mb,wb,VLL,VSL,cb)
		
T = 0.125
beta = 1/T
print('beta', beta)

potkey = 'Tanimura_SB_D_{}_alpha_{}_VLL_{}_VSL_{}_cb_{}'.format(D,alpha,VLL,VSL,cb)
Tkey = 'T_{}'.format(T)

pathname = os.path.dirname(os.path.abspath(__file__))

E = 0.25

xg = np.linspace(-5,10,int(1e2)+1)
yg = np.linspace(-5,10,int(1e2)+1)

xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid) ###

qlist = []

### 'nbeads' can be set to >1 for ring-polymer simulations.
nbeads = 1#8
PSOS = Poincare_SOS('Classical',pathname,potkey,Tkey)
PSOS.set_sysparams(pes,T,m,2)
PSOS.set_simparams(N,dt,dt,nbeads=nbeads,rngSeed=2)	
PSOS.set_runtime(50.0,500.0)
if(1):
	#xg = np.linspace(xmin-0.1,xmin+0.1,int(1e2)+1)
	#yg = np.linspace(ymin-0.1,ymin+0.1,int(1e3)+1)

	#xg = np.linspace(0,2*xmin,int(1e2)+1)
	#yg = np.linspace(-2*abs(ymin),4*abs(ymin),int(1e3)+1)

	#xg = np.linspace(-1.0,5,int(1e2)+1)
	#yg = np.linspace(-4,4,int(1e2)+1)

	xgrid,ygrid = np.meshgrid(xg,yg)
	potgrid = pes.potential_xy(xgrid,ygrid)

	qlist = PSOS.find_initcondn(xgrid,ygrid,potgrid,E)
	PSOS.bind(qcartg=qlist,E=E,sym_init=False)#,specific_traj=[7])#pcartg=plist)#E=E)

	if(0): ## Plot the trajectories that make up the Poincare section
		xg = np.linspace(-3,4,int(1e2)+1)
		yg = np.linspace(-3,4,int(1e2)+1)
		xgrid,ygrid = np.meshgrid(xg,yg)
		potgrid = pes.potential_xy(xgrid,ygrid)

		fig,ax=plt.subplots()
		ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/5))
		PSOS.run_traj(0,ax) #(1,2,3,4,8,13 for z=1.25), (2,3) 
		plt.show()
	
	if(1): ## Collect the data from the Poincare section and plot. 
		X,PX,Y = PSOS.PSOS_X(y0=0.0)#ymin)
		plt.scatter(X,PX,s=1)
		
		plt.title(r'$c={}$, E={}'.format(cb,E))#E=$V_b$+$3\omega_m/2$'.format(alpha) )#$N_b={}$'.format(nbeads))
		plt.xlabel(r'x')
		plt.ylabel(r'$p_x$')
	
	plt.show()
	#fname = 'Classical_Poincare_Section_x_px_{}_E_{}'.format(potkey,E)
	#store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))
	
