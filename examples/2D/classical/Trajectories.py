import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV_multidim
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.harmonic_2D import Harmonic
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from thermalize import thermalize
import pickle
import os
import time

m = 0.5
omega = 1.0#0.5
g0 = 0.1#1/100.0
T_au = 5.0#0.5
beta = 1.0/T_au 
print('T in au',T_au)

potkey = 'coupled_harmonic'#_w_{}_g_{}'.format(omega,g0)
sysname = 'Selene'		

N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 40.0
time_total = 60.0

pathname = os.path.dirname(os.path.abspath(__file__))

dim = 2
T = T_au 
print('T',T)

nbeads = 1
rng = np.random.RandomState(1)
qcart = rng.normal(size=(N,dim,nbeads))
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = None
beta = 1/T

rngSeed = 1
rp = RingPolymer(qcart=qcart,m=m) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt_therm,symporder=2) 
rng = np.random.default_rng(rngSeed) 

rp.bind(ens,motion,rng)

pes = coupled_harmonic(omega,g0)
pes.bind(ens,rp)

time_therm = time_therm
thermalize(pathname,ens,rp,pes,time_therm,dt_therm,potkey,rngSeed)

tarr=[]
qarr=[]
parr=[]

qcart = read_arr('Thermalized_q_N_{}_beta_{}_{}_seed_{}'.format(rp.nsys,ens.beta,potkey,rngSeed),"{}/Datafiles/".format(pathname))
pcart = read_arr('Thermalized_p_N_{}_beta_{}_{}_seed_{}'.format(rp.nsys,ens.beta,potkey,rngSeed),"{}/Datafiles/".format(pathname))

rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,mode='rp')
motion = Motion(dt = dt,symporder=4) 
rp.bind(ens,motion,rng)
pes.bind(ens,rp)

therm = PILE_L(tau0=0.1,pile_lambda=1000.0) 
therm.bind(rp,motion,rng,ens)

propa = Symplectic_order_IV_multidim()	
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

time_total = time_total
nsteps = int(time_total/dt)	

start_time = time.time()

L=6
fig,ax = plt.subplots(1,2)
xgrid = np.linspace(-L,L,200)
ygrid = np.linspace(-L,L,200)
x,y = np.meshgrid(xgrid,ygrid)
potgrid = pes.potential_xy(x,y)
hesgrid = 0.25*(omega**2 + 4*g0*omega*(x**2+y**2) - 48*g0**2*x**2*y**2)
ax[0].contour(x,y,potgrid,colors='k',levels=np.arange(0,5,0.5))#,levels=vals[:20])#np.arange(0,5,0.5))
ax[0].contour(x,y,hesgrid,levels=np.arange(-5,0.01,0.5))#np.arange(-0.0001,0,0.00001))
ax[0].contour(x,y,hesgrid,colors='m',levels=[0.0])
#ax[0].contour(x,y,hesgrid,levels=np.arange(0.01,3.0,0.5))#np.arange(-0.0001,0,0.00001))
#ax[0].scatter(rp.q[:,0,0],rp.q[:,1,0])
#plt.show()
#np.arange(-0.0001,0,0.00001))
#plt.contour(x,y,potgrid,levels=[0.1,vals[0],vals[1],vals[3],vals[4],vals[5],vals[7],vals[100]])
#plt.show()
n = 60
print('E',np.sum(rp.p[n]**2))
rp.q[n,0,0] = 1.0
rp.q[n,1,0] = 1.0
rp.p[n,0,0] = -2.0
rp.p[n,1,0] = 2.0

if(1):
	ax[0].axis([-L,L,-L,L])	
	ax[1].axis([0,10,-10,20])
	for i in range(nsteps):
		sim.step(mode="nve",var='monodromy',pc=False)	
		tarr.append(sim.t)
		qarr.append(rp.q[:,:,0].copy())
		parr.append(rp.p[:,:,0].copy())	
		if(i%10==0):
			ax[0].scatter(rp.q[n,0,0],rp.q[n,1,0])
			ax[1].scatter(sim.t,np.log(rp.Mqq[n,0,0,0,0]**2),color='k',s=2)
			plt.pause(0.05)
			print('t', sim.t)
	qarr=np.array(qarr)
	parr=np.array(parr)
	tarr=np.array(tarr)
	print('qarr',qarr.shape)	

	plt.show()


