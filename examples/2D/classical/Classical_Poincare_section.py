import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.potentials.Adams_function import adams_function
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.instanton import instantonize
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import os

m=0.5
w = 0.2#0.1
D = 5.0#10.0
alpha = (0.5*m*w**2/D)**0.5#1.0#1.95

lamda = 0.8 #4.0
g = 0.02#4.0

z = 2.0#2.3	
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

# Thermalization
N = 1000
dim = 2
nbeads = 1
T = 0.5*lamda/np.pi#2.0	

dt = 0.01
time_therm = 50.0

rng = np.random.RandomState(1)
qcart = rng.normal(size=(N,dim,nbeads))
q = np.random.normal(size=(N,dim,nbeads))

pcart = np.zeros((N,dim,nbeads))
beta = 1/T

rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,nmats=nbeads) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(0) 

pes = quartic_bistable(alpha,D,lamda,g,z)#coupled_harmonic(1.0,0.1)#

nsteps_therm = int(time_therm/dt)

therm = PILE_L(tau0=0.1,pile_lambda=100.0) 
propa = Symplectic_order_II()
sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)	

sim.step(ndt = nsteps_therm,mode="nvt",var='pq')

ind = np.where(pes.pot[:,0]<T)
qcart_arr = rp.qcart[ind]
print('qcart_arr',qcart_arr.shape)

### Poincare surface of section
X = []
Y =[]
PX = []
PY =[]

# PSOS
N = 1
nbeads = 1

time_total = 500.0
qcart = np.zeros((1,dim,nbeads))#rng.normal(size=(N,dim,nbeads))
pcart = np.zeros_like(qcart)
nsteps = int(time_total/dt)

def PSOS_Y(qcartg,pcartg):
	rp = RingPolymer(qcart=qcartg,pcart=pcartg,m=m,nmats=nbeads)
	sim.bind(ens,motion,rng,rp,pes,propa,therm)	
	
	prev = rp.q[0,0,0]
	curr = rp.q[0,0,0]
	count=0

	print('E,kin,pot',rp.kin+pes.pot,rp.kin,pes.pot)
	for i in range(nsteps):
		sim.step(mode="nve",var='pq')	
		x = rp.q[0,0,0]
		px = rp.p[0,0,0]
		y = rp.q[0,1,0]
		py = rp.p[0,1,0]
		curr = x
		if(count%1==0):
			#print(prev,curr)
			#plt.scatter(x,y)
			#plt.pause(0.05)
			if( prev*curr<0.0 and px>0.0 ):
				#X.append(x)
				#PX.append(px)
			
				Y.append(y)
				PY.append(py)

		prev = curr
		count+=1
	
	plt.show()

def PSOS_X(qcartg,pcartg):
	rp = RingPolymer(qcart=qcartg,pcart=pcartg,m=m,nmats=nbeads)
	sim.bind(ens,motion,rng,rp,pes,propa,therm)	

	prev = rp.q[0,1,0]
	curr = rp.q[0,1,0]
	count=0

	print('E,kin,pot',rp.kin+pes.pot,rp.kin,pes.pot)	
	for i in range(nsteps):
		sim.step(mode="nve",var='pq')	
		x = rp.q[0,0,0]
		px = rp.p[0,0,0]
		y = rp.q[0,1,0]
		py = rp.p[0,1,0]
		curr = y
		if(count%1==0):
			#print(prev,curr)
			if( prev*curr<0.0 and py<0.0 ):
				X.append(x)
				PX.append(px)
			
				#Y.append(y)
				#PY.append(py)

		prev = curr
		count+=1

print('T',T)
for i in range(len(qcart_arr)):
	qcart[0] = qcart_arr[i]#rng.uniform(-1.5,1.5,size=2)#np.array([0.0,1.0])
	Ekin = 2*(T -pes.potential_xy(qcart[0,0],qcart[0,1]))*m

	xkincomp = rng.uniform(0,1)
	ykincomp = 1-xkincomp

	pcart[0,0,0] = (xkincomp*Ekin)**0.5
	pcart[0,1,0] = (ykincomp*Ekin)**0.5
	
	PSOS_X(qcart,pcart)

	pcart[0,0,0] = -(xkincomp*Ekin)**0.5
	pcart[0,1,0] = -(ykincomp*Ekin)**0.5	

	#if(i%2==0):
	#	plt.scatter(X,PX,s=4)
		#plt.scatter(Y,PY,s=2)
	#	plt.show()
#plt.scatter(X,PX)

pathname = os.path.dirname(os.path.abspath(__file__))
fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))
		
	
	
