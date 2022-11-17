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
from scipy.interpolate import interp1d
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
import matplotlib

### Potential parameters
m=0.5#0.5
N=1#20
dt=0.005

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
D = 3*Vb
alpha = 0.382#7
z = 1.0
print('Vb',Vb, 'D', D)

Tc = 0.5*lamda/np.pi
times=0.95
T_au = times*Tc
beta = 1/T_au
Tkey = 'T_{}Tc'.format(times)

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

pes = quartic_bistable(alpha,D,lamda,g,z)

pathname = os.path.dirname(os.path.abspath(__file__))

nbeads = 32
E = 1.01*Vb
q = np.zeros((1,2,nbeads))
p = np.zeros_like(q)
p[:,0,0] = 0.1*4
p[:,0,1:3] = 0.5*4

ens = Ensemble(beta,ndim=2)
motion = Motion(dt = dt,symporder=4) 
rng = np.random.default_rng(2) 

rp = RingPolymer(q=q,p=p,m=m,mode='rp')
rp.bind(ens,motion,rng)
pes.bind(ens,rp)	

pot = pes.potential(rp.qcart)
print('rp E', E*nbeads,np.sum(rp.pcart[0]**2/(2*m)) +np.sum(pot[0]))
	
#print('rp',rp.qcart)

therm = PILE_L() 
therm.bind(rp,motion,rng,ens)

propa = Symplectic_order_IV()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

time_total = 28
time_total = time_total
nsteps = int(time_total/dt)	

print('steps',nsteps)

qarr=[]
Mqqarr=[]
tarr=[]
pathname = os.path.dirname(os.path.abspath(__file__))

print('Vb', Vb)
#fig,ax = plt.subplots(2, sharex=True,gridspec_kw={'hspace': 0})
xg = np.linspace(-4.5,4.5,int(1e2)+1)
yg = np.linspace(-2,3.5,int(1e2)+1)

xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid) ###

qlist = []

fig,ax = plt.subplots(1)
#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/20))

path = '/home/vgs23/PISC/examples/2D/rpmd/'
instanton = read_arr('Instanton_{}_{}_nbeads_{}'.format(potkey,Tkey,nbeads),'{}/Datafiles'.format(path)) #instanton

	
xl_fs = 14
yl_fs = 14
tp_fs = 9
ti_fs = 13

mss = 7
msl = 8


ax.contour(xgrid,ygrid,potgrid,levels=np.arange(-0.3,0.4*D,D/10),colors='k',linewidths=1.0)	
#ax.scatter(instanton[0,0],instanton[0,1],color='g', s=10)	
#ax.plot(instanton[0,0],instanton[0,1],color='navy',alpha=0.5,lw=4,marker='o')#,path_effects=[path_effects.SimpleLineShadow(),path_effects.Normal()])	
xi = instanton[0,0]
yi = instanton[0,1]
f = interp1d(xi,yi)

xnew = np.linspace(xi.min(),xi.max(),1000)
ynew = f(xnew)

ax.plot(xnew,ynew,color='pink',alpha=0.65,lw=1,marker='o',ms=4,zorder=1)
#ax.fill_between(xnew, 0.6*ynew, 1.4*ynew, facecolor='navy', alpha=0.5)
ax.scatter(0.0,0.0, color='k',s=msl,zorder=2)


init_offset = 3000
ndt = 10
sim.step(ndt=init_offset,mode="nve",var='pq',pc=False)

nframes = (nsteps-init_offset)//ndt
xarr = []
yarr = []

xb = rp.qcart[0,0,:]
yb = rp.qcart[0,1,:]
x,y = rp.q[0,:,0]/nbeads**0.5

xarr.append(x)
yarr.append(y)

ax.set_xlabel(r'$x$',fontsize=xl_fs)
ax.set_ylabel(r'$y$',fontsize=yl_fs)

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.serif':'Times New Roman'})
matplotlib.rcParams['axes.unicode_minus'] = False

line, = plt.plot(xarr,yarr,color='g',lw=1.5)
scatter1 = plt.scatter(x,y,s=mss,facecolor='k', edgecolor='k') 
scatter2 = plt.scatter(xb,yb,s=msl,facecolor='navy',edgecolor='navy',alpha=0.75,zorder=3)
timeax = plt.annotate('t=0.0', xy=(3.75, 3.1), xytext=(0.41, 0.8),xycoords = 'axes fraction',fontsize=ti_fs)

ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-1.8,2.0])
ax.set_xlim([-4,4])
ax.tick_params(axis='both', which='major', labelsize=tp_fs)

fig.set_size_inches(3,2)#2.4, 1.6)	
		
def animate(i):
	xb = rp.qcart[0,0,:]
	yb = rp.qcart[0,1,:]
	
	cent = rp.q[0,:,0]/nbeads**0.5
	x,y = cent

	xarr.append(x)
	yarr.append(y)

	line.set_data(xarr,yarr)
	scatter1.set_offsets(cent)
	scatter2.set_offsets(rp.qcart[0].T)
	t = np.around(sim.t-init_offset*dt ,1)
	print('t',t,i)
	timeax.set_text('t={}'.format(t))	
	
	sim.step(ndt=ndt,mode="nve",var='monodromy',pc=False)	

	#plt.gca().relim()
	#plt.gca().autoscale_view()
	#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/10),linewidths=0.1)	
	#axb = ax.scatter(xb,yb,s=15,color='r',alpha=0.25)	
	#ax.scatter(x,y,s=7,color='k')	
	#ax.scatter(instanton[0,0],instanton[0,1],color='g', s=3)	
	
	#axb.remove()

# Frames 9,62,159 are the best
for i in range(159):
	animate(i)

#fig.savefig('/home/vgs23/Images/Animation_snapshot_3_thesis.pdf', dpi=200,bbox_inches='tight',pad_inches=0.0)


#anim = animation.FuncAnimation(fig, animate, frames=range(nframes),repeat=False)



#anim.save('Chaotic_Instanton.mp4'.format(path),dpi=400,fps=15,bitrate=-1, codec="libx264")
plt.show()

if(0):
	for i in range(nsteps):
			sim.step(mode="nve",var='monodromy',pc=False)	
			if(i%25==0 and i>3000):	
				xb = rp.qcart[0,0,:]
				yb = rp.qcart[0,1,:]
				x,y = rp.q[0,:,0]/nbeads**0.5
				ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/10),linewidths=0.1)	
				axb = ax.scatter(xb,yb,s=15,color='r',alpha=0.25)
				ax.scatter(x,y,s=7,color='k')	
				ax.scatter(instanton[0,0],instanton[0,1],color='g', s=3)#,label='nbeads = {}'.format(nbeads))	
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
	
