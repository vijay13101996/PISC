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
m=0.5#0.5
dt=0.01#05

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
D = 3*Vb
alpha = 0.382
print('Vb',Vb, 'D', D)

z = 1.0
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

pes = quartic_bistable(alpha,D,lamda,g,z)

pathname = os.path.dirname(os.path.abspath(__file__))

E = 1.05*Vb 

xg = np.linspace(-4,4,int(1e2)+1)
yg = np.linspace(-2.0,2.5,int(1e2)+1)#-1.5,3.5,int(1e2)+1)

xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid) ###

qlist = []

fig,ax = plt.subplots(2,1)
ax[0].contour(xgrid,ygrid,potgrid,levels=np.arange(0,0.5*D,D/10))
ax[0].axhline(y=0.0,xmin=0.0, xmax = 1.0,linestyle='--',color='gray')
ax[0].set_xlabel(r'$x$',fontsize=12)
ax[0].set_ylabel(r'$y$',fontsize=12)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlim([xg[0],xg[-1]])
ax[0].set_ylim([yg[0],yg[-1]])

ax[1].set_xlabel(r'$x$',fontsize=12)
ax[1].set_ylabel(r'$p_x$',fontsize=12)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xlim([xg[0],xg[-1]])
ax[1].set_ylim([-3,3])
#plt.show()

qcart = np.zeros((1,2,1))
pcart = np.zeros_like(qcart)
qcart[0,:,0] = [-1.0,-0.1]
V = pes.potential_xy(qcart[0,0,0],qcart[0,1,0])
xfrac = 0.75
pcart[0,0] = -(2*m*(E-V))**0.5*xfrac**0.5
pcart[0,1] = (2*m*(E-V))**0.5*(1-xfrac)**0.5

rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,mode='rp') 
ens = Ensemble(beta=1.0,ndim=2)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(1) 

rp.bind(ens,motion,rng)

print('E',E,np.sum(rp.pcart**2/(2*m)) + pes.potential_xy(rp.qcart[0,0,0],rp.qcart[0,1,0])) 


pes.bind(ens,rp)
therm = PILE_L(tau0=100.0,pile_lambda=1.0) 

propa =Symplectic_order_II()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

time_run = 600.0
nsteps = int(time_run/dt)


prev = rp.qcart[0,1,0]
curr = rp.qcart[0,1,0]
	
xarr = []
yarr = []

PS_xarr=[]
PS_pxarr=[]

matplotlib.rcParams['axes.unicode_minus'] = False

fig.set_size_inches(3.5,5.5)#2.4, 1.6)	

start_time = time.time()
if(0):
	x = rp.qcart[0,0,:]
	px = rp.pcart[0,0,:]
	y = rp.qcart[0,1,:]
	py = rp.pcart[0,1,:]
		
	line, = ax[0].plot(xarr,yarr,color='g',lw=1.25)
	scatter1 = ax[0].scatter(x,y,s=4,facecolor='r', edgecolor='r') 
	scatter2 = ax[1].scatter([],[],s=1,facecolor='k',edgecolor='k',zorder=2)
	scatter3 = ax[1].scatter([],[],s=4,facecolor='r', edgecolor='r',zorder=3) 
	
	timeax = ax[0].annotate(r'$t=0.0$', xy=(3.75, 3.1), xytext=(0.75, 0.9),xycoords = 'axes fraction',fontsize=11)

	ndt=10
	def animate(i):
		global prev 
		global curr 
		sim.step(ndt=ndt,mode="nve",var='pq')
		x = rp.qcart[0,0,0]
		px = rp.pcart[0,0,0]
		y = rp.qcart[0,1,0]
		py = rp.pcart[0,1,0]
		curr = y

		if(1):	
			if(i%1==0):
				xarr.append(x)
				yarr.append(y)
				line.set_data(xarr,yarr)	
				scatter1.set_offsets(np.array([x,y]).T)	
			if(prev*curr < 0.0 and py>0.0):
				#print('here')
				PS_xarr.append(x)
				PS_pxarr.append(px)
				arr = np.array([PS_xarr,PS_pxarr]).T
				scatter2.set_offsets(arr)
				scatter3.set_offsets(np.array([x,px]).T)
			prev = curr
				
		t = np.around(sim.t,1)
		print('t',t,i)
		timeax.set_text(r'$t={}$'.format(t))	
		
			
	
	path = '/home/vgs23/Images'
	anim = animation.FuncAnimation(fig, animate,interval=1, frames=nsteps//ndt,repeat=False,save_count=10)
	anim.save('{}/Chaotic_trajectory_z_1.gif'.format(path),dpi=150,fps=50,writer='imagemagick')#bitrate=-1, codec="libx264")
	#plt.show()


if(1):
	for i in range(nsteps):
		sim.step(mode="nve",var='pq')	
		x = rp.qcart[0,0,:]
		px = rp.pcart[0,0,:]
		y = rp.qcart[0,1,:]
		py = rp.pcart[0,1,:]
		curr = y
		#print('curr, prev', curr,prev,py)
		if(i%1==0):
			#print('i',i)
			#ax.scatter(self.sim.t,y)
			xarr.append(x)
			yarr.append(y)
			#ax[0].scatter(x,y,s=7,color='r')
		if(prev*curr < 0.0 and py>0.0):
			#print('here')
			PS_xarr.append(x)
			PS_pxarr.append(px)
			#ax[1].scatter(x,px,s=7)
		#plt.pause(0.05)
		prev = curr
			
	ax[0].plot(xarr,yarr,color='g',lw=1.25)
	ax[1].scatter(PS_xarr,PS_pxarr,s=1,color='k')
	
	fig.savefig('/home/vgs23/Images/anim_snapshot_z1.png'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	
	plt.show()
	print('time', time.time() - start_time)


	

