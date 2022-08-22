import numpy as np
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials import quartic_bistable
from PISC.potentials.eckart import eckart
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import os
import matplotlib
from Saddle_point_finder import separatrix_path, find_minima
import scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline

### Potential parameters
m=0.5#0.5
N=1#20
dt=0.005

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
D = 3*Vb
alpha = 0.37
print('Vb',Vb, 'D', D)
Tc = 0.5*lamda/np.pi
T_au = 1.0*Tc
beta = 1/T_au


z = 1.0
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

pes = quartic_bistable(alpha,D,lamda,g,z)

pathname = os.path.dirname(os.path.abspath(__file__))

nbeads = 32
q = np.zeros((1,2,nbeads))
p = np.zeros_like(q)
p[:,0,0] = 0.05*4
p[:,0,1:3] = 0.5*4

ens = Ensemble(beta,ndim=2)
motion = Motion(dt = dt,symporder=4) 
rng = np.random.default_rng(1) 

rp = RingPolymer(q=q,p=p,m=m,mode='rp')
rp.bind(ens,motion,rng)
pes.bind(ens,rp)	

print('rp',rp.qcart)

therm = PILE_L() 
therm.bind(rp,motion,rng,ens)

propa = Symplectic_order_IV()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

time_total = 20
time_total = time_total
nsteps = int(time_total/dt)	

print('steps',nsteps)

qarr=[]
Mqqarr=[]
tarr=[]
pathname = os.path.dirname(os.path.abspath(__file__))

print('Vb', Vb)
#fig,ax = plt.subplots(2, sharex=True,gridspec_kw={'hspace': 0})
xg = np.linspace(-5,5,int(1e2)+1)
yg = np.linspace(-2.5,5.5,int(1e2)+1)

xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid) ###

qlist = []

fig,ax = plt.subplots(1)
#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/20))

for i in range(nsteps):
		sim.step(mode="nve",var='monodromy',pc=False)	
		if(i%25==0 and i>1320):	
			xb = rp.qcart[0,0,:]
			yb = rp.qcart[0,1,:]
			x,y = rp.q[0,:,0]/nbeads**0.5
			ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/10),linewidths=0.1)	
			axb = ax.scatter(xb,yb,s=15,color='r',alpha=0.25)
			ax.scatter(x,y,s=7,color='k')	
			plt.pause(0.01)
			axb.remove()#clear()	
		if(i%20==0):
			Mqq = rp.Mqq[0,0,0,0,0]	
			Mqqarr.append(Mqq**2)	
			x,y = rp.q[0,:,0]/nbeads**0.5	
			qarr.append(x.copy())
			tarr.append(sim.t)
		

#ax[0].plot(tarr,qarr)
#ax[1].plot(tarr,np.log(Mqqarr))
plt.show()
	
