import numpy as np
from PISC.potentials.base import PES
from numpy import exp

class quartic_bistable(PES):
		def __init__(self,alpha,D,lamda,g,z,invert_pes=1.0):
			super(quartic_bistable).__init__()
			self.D = D
			self.alpha = alpha

			self.g = g
			self.lamda = lamda
			
			self.z = z	
	
		def bind(self,ens,rp):
			super(quartic_bistable,self).bind(ens,rp)
 
		def potential(self,q):
			x = q[:,0] 
			y = q[:,1]
			return self.potential_xy(x,y) 
		
		def dpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			Vx = 4*self.g*x*(-1 + np.exp(-self.alpha*y*self.z))*(x**2 - self.lamda**2/(8*self.g)) + 4*self.g*x*(x**2 - self.lamda**2/(8*self.g))
			Vy = 2*self.D*self.alpha*(1 - np.exp(-self.alpha*y))*np.exp(-self.alpha*y) \
					- self.alpha*self.z*(self.g*(x**2 - self.lamda**2/(8*self.g))**2 - self.lamda**4/(64*self.g))*np.exp(-self.alpha*y*self.z)		 
			
			return np.transpose([Vx,Vy],axes=[1,0,2])
	
		def ddpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			Vxx = 4*self.g*(-2*x**2*(1 - np.exp(-self.alpha*y*self.z)) + 3*x**2 - (1 - np.exp(-self.alpha*y*self.z))*(8*x**2 - self.lamda**2/self.g)/8 - self.lamda**2/(8*self.g))
			Vxy = -4*self.alpha*self.g*x*self.z*(x**2 - self.lamda**2/(8*self.g))*np.exp(-self.alpha*y*self.z)
			Vyy = self.alpha**2*(-2*self.D*(1 - np.exp(-self.alpha*y))*np.exp(-self.alpha*y) + 2*self.D*np.exp(-2*self.alpha*y) \
						+ self.z**2*(self.g*(8*x**2 - self.lamda**2/self.g)**2 - self.lamda**4/self.g)*np.exp(-self.alpha*y*self.z)/64)
			
			return np.transpose([[Vxx, Vxy],[Vxy, Vyy]],axes=[2,0,1,3]) 
			
		def potential_xy(self,x,y):
			eay = np.exp(-self.alpha*y)
			quartx = (x**2 - self.lamda**2/(8*self.g))
			vy = self.D*(1-eay)**2
			vx = self.g*quartx**2
			vxy = (vx-self.lamda**4/(64*self.g))*(np.exp(-self.z*self.alpha*y) - 1) # 
			return vx + vy + vxy 
