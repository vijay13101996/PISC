import numpy as np
from PISC.potentials.base import PES

class heller_davis(PES):
		def __init__(self,ws,wu,lamda):
			super(heller_davis).__init__()
			self.ws = ws
			self.wu = wu

			self.lamda = lamda

		def bind(self,ens,rp):
			super(heller_davis,self).bind(ens,rp)
 
		def potential(self,q):
			x = q[:,0] 
			y = q[:,1]
			return self.potential_xy(x,y) 
		
		def dpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			Vx= self.lamda*y**2 + 1.0*self.ws**2*x
			Vy= 2*self.lamda*x*y + 1.0*self.wu**2*y
			return np.transpose([Vx,Vy],axes=[1,0,2])
	
		def ddpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			Vxx= 1.0*self.ws**2*np.ones_like(x)
			Vxy= 2*self.lamda*y
			Vyy= 2*self.lamda*x + 1.0*self.wu**2*np.ones_like(y)
			return np.transpose([[Vxx, Vxy],[Vxy, Vyy]],axes=[2,0,1,3]) 
			
		def potential_xy(self,x,y):
			vy = 0.5*self.wu**2*y**2
			vx = 0.5*self.ws**2*x**2
			vxy = self.lamda*y**2*x
			return vx + vy + vxy 
