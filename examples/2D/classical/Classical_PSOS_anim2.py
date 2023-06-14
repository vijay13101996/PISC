import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from Saddle_point_finder import separatrix_path, find_minima
import time
import os
import matplotlib
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects

matplotlib.rcParams['axes.unicode_minus'] = False

### Potential parameters
m=1.0#0.5#0.5
dt=0.02#05

omega = 1.0
g0 = 0.0#1#1
	
potkey = 'coupled_harmonic_w_{}_g_{}'.format(omega,g0)

T_au = 0.5 
beta = 1.0/T_au 

basis_N = 40#
n_eigen = 30#

print('T in au, potential, basis',T_au, potkey,basis_N )

pes = coupled_harmonic(omega,g0)

pathname = os.path.dirname(os.path.abspath(__file__))

E = 2.0 

xg = np.linspace(-6,6,int(1e2)+1)
yg = np.linspace(-6,6,int(1e2)+1)

xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid) 

fig, ax = plt.subplots(1)
ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,8.01,1.0),linewidths=1.5)
ax.contour(xgrid,ygrid,potgrid,levels=np.arange(E,E+0.01,1.0),linewidths=2.5,colors='k')
ax.set_xlabel(r'$x$',fontsize=12)
ax.set_ylabel(r'$y$',fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([xg[0],xg[-1]])
ax.set_ylim([yg[0],yg[-1]])

### Temperature is only relevant for the ring-polymer Poincare section
T = 3.0
Tkey = 'T_{}'.format(T) 

pathname = os.path.dirname(os.path.abspath(__file__))

PSOS = Poincare_SOS('Classical',pathname,potkey,Tkey)
PSOS.set_sysparams(pes,T,m,2)
PSOS.set_simparams(1000,dt,dt,nbeads=1,rngSeed=1)	
PSOS.set_runtime(0.0,0.0)

xg = np.linspace(-1,1,int(1e2)+1)
yg = np.linspace(-1,1,int(1e2)+1)

xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

qlist = PSOS.find_initcondn(xgrid,ygrid,potgrid,E)
PSOS.bind(qcartg=qlist,E=E,sym_init=False)

qcart = PSOS.rp.qcart
pcart = PSOS.rp.pcart

rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,mode='rp') 
ens = Ensemble(beta=1.0,ndim=2)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(0) 

rp.bind(ens,motion,rng)

print('rp.pcart', rp.pcart[:,0,0])
print('E',E,np.sum(rp.pcart[:,:,0]**2/(2*m),axis=1) + pes.potential_xy(rp.qcart[:,0,0],rp.qcart[:,1,0])) 


pes.bind(ens,rp)
therm = PILE_L(tau0=100.0,pile_lambda=1.0) 

propa =Symplectic_order_II()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

time_run = 100.0
nsteps = int(time_run/dt)

x = rp.qcart[:,0,:]
y = rp.qcart[:,1,:]
		
#line, = ax.plot(xarr,yarr,color='g',lw=1.25)
scatter1 = ax.scatter(x,y,s=12,facecolor='r', edgecolor='r',alpha=0.75) 
#scatter2 = ax.scatter([],[],s=1,facecolor='k',edgecolor='k',zorder=2)
#scatter3 = ax.scatter([],[],s=4,facecolor='r', edgecolor='r',zorder=3) 

timeax = ax.annotate(r'$t=0.0$', xy=(3.75, 3.1), xytext=(0.78, 0.9),xycoords = 'axes fraction',fontsize=11)

ndt=10
offset=40
def animate(i):
	if(i>offset):
		sim.step(ndt=ndt,mode="nve",var='pq')
		x = rp.qcart[:,0,0]
		y = rp.qcart[:,1,0]
		scatter1.set_offsets(np.array([x,y]).T)	
		t = np.around(sim.t,1)
		print('t',t,i)
		timeax.set_text(r'$t={}$'.format(t))	
		
			
fig.set_size_inches(4,4)
print('nframes', nsteps//ndt+offset)

path = '/home/vgs23/Images'
anim = animation.FuncAnimation(fig, animate,interval=1, frames=nsteps//ndt + offset,repeat=False,save_count=10)
#anim.save('{}/Chaotic_ensemble_z_0.gif'.format(path),dpi=150,fps=30,writer='imagemagick')#bitrate=-1, codec="libx264")
plt.show()


