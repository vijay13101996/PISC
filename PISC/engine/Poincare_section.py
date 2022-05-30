import numpy as np
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.integrators import Symplectic_order_II

class Poincare_SOS(object):
	def __init__(self,pes,m,dim=2,traj_time = 500.0,dt = 0.01,beta=1.0):
		self.pes = pes
		self.m = m
		self.dim = dim
		self.traj_time = traj_time
		self.X = []
		self.Y = []
		self.PX = []
		self.PY = []
		qcart = np.zeros((1,dim,1))
		pcart = np.zeros((1,dim,1))
		self.rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,nmats=1)	
		self.ens = Ensemble(beta=beta,ndim=dim)
		self.motion = Motion(dt = dt,symporder=2) 
		self.rng = np.random.default_rng(0) 
		self.therm = PILE_L(tau0=0.1,pile_lambda=100.0) 
		self.propa = Symplectic_order_II()
		self.sim = RP_Simulation()
		
	def bind(self,qcartg,pcartg):
		self.rp = RingPolymer(qcart=qcartg,pcart=pcartg,m=self.m,nmats=1)
		self.sim.bind(self.ens,self.motion,self.rng,self.rp,self.pes,self.propa,self.therm)

	def PSOS_Y(self,qcartg,pcartg,x0):
		self.bind(qcartg,pcartg)
		prev = self.rp.q[0,0,0] - x0
		curr = self.rp.q[0,0,0] - x0
		count=0

		nsteps = int(self.traj_time/self.motion.dt)
		print('E,kin,pot',self.rp.kin+self.pes.pot,self.rp.kin,self.pes.pot)
		Y_list = []
		PY_list = []
		for i in range(nsteps):
			self.sim.step(mode="nve",var='pq')	
			x = self.rp.q[0,0,0]
			px = self.rp.p[0,0,0]
			y = self.rp.q[0,1,0]
			py = self.rp.p[0,1,0]
			curr = x-x0
			if(count%1==0):
				if( prev*curr<0.0 and px>0.0 ):
					Y_list.append(y)
					PY_list.append(py)
					X_list.append(x)
			prev = curr
			count+=1

		self.Y.extend(Y_list)
		self.PY.extend(PY_list)

		return Y_list,PY_list,X_list
	
	def PSOS_X(self,qcartg,pcartg,y0):
		self.bind(qcartg,pcartg)	
		prev = self.rp.q[0,1,0] - y0
		curr = self.rp.q[0,1,0] - y0
		count=0

		nsteps = int(self.traj_time/self.motion.dt)
		print('E,kin,pot',self.rp.kin+self.pes.pot,self.rp.kin,self.pes.pot)
		X_list = []
		PX_list = []
		Y_list = []
		for i in range(nsteps):
			self.sim.step(mode="nve",var='pq')	
			x = self.rp.q[0,0,0]
			px = self.rp.p[0,0,0]
			y = self.rp.q[0,1,0]
			py = self.rp.p[0,1,0]
			curr = y-y0
			if(count%1==0):
				if( prev*curr<0.0 and py<0.0 ):
					X_list.append(x)
					PX_list.append(px)
					Y_list.append(y)
			prev = curr
			count+=1

		self.X.extend(X_list)
		self.PX.extend(PX_list)

		return X_list,PX_list,Y_list		
	

 
