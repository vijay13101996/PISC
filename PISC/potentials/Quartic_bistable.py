import numpy as np
from PISC.potentials.base import PES
from numpy import exp
class quartic_bistable(PES):
		def __init__(self,alpha,D,lamda,g,z,k=1.0):
			super(quartic_bistable).__init__()
			self.D = D
			self.alpha = alpha

			self.g = g
			self.lamda = lamda
			
			self.k = k
			self.z = z
			self.eta=1.0	
	
		def bind(self,ens,rp):
			super(quartic_bistable,self).bind(ens,rp)
 
		def potential(self,q):
			x = q[:,0] 
			y = q[:,1]
			return self.potential_xy(x,y) 
		
		def dpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			#Vx = 4*self.g*x*(-1 + np.exp(-self.alpha*y*self.z))*(x**2 - self.lamda**2/(8*self.g)) + 4*self.g*x*(x**2 - self.lamda**2/(8*self.g))
			#Vy = 2*self.D*self.alpha*(1 - np.exp(-self.alpha*y))*np.exp(-self.alpha*y) \
			#		- self.alpha*self.z*(self.g*(x**2 - self.lamda**2/(8*self.g))**2 - self.lamda**4/(64*self.g))*np.exp(-self.alpha*y*self.z)		 
			
			#Vx = 4*self.g*x*(-1 + exp(-self.alpha*self.z*y))*(x**2 - self.lamda**2/(8*self.g)) + 4*self.g*x*(x**2 - self.lamda**2/(8*self.g))
			#Vy = 2*self.D*self.alpha*(1 - exp(-self.alpha*y))*exp(-self.alpha*y) - \
			#		self.alpha*self.z*(self.g*(x**2 - self.lamda**2/(8*self.g))**2 - self.lamda**4/(256*self.g))*exp(-self.alpha*self.z*y)	
			
			#Vx = 4*self.g*x*(-1 + exp(-self.alpha*self.z*y))*(x**2 - self.lamda**2/(8*self.g)) + 4*self.g*x*(x**2 - self.lamda**2/(8*self.g))
			#Vy = 2*self.D*self.alpha*(1 - exp(-self.alpha*y))*exp(-self.alpha*y) - self.alpha*self.z*(self.g*(x**2 - self.lamda**2/(8*self.g))**2 \
			#		- self.k*self.lamda**4/(64*self.g))*exp(-self.alpha*self.z*y)
			
			Vx = 4*self.eta*self.g*x*(-1 + exp(-self.alpha*self.z*y))*(x**2 - self.lamda**2/(8*self.g)) + 4*self.g*x*(x**2 - self.lamda**2/(8*self.g))
			Vy = 2*self.D*self.alpha*(1 - exp(-self.alpha*y))*exp(-self.alpha*y) - self.alpha*self.eta*self.z*(self.g*(x**2 -\
				 self.lamda**2/(8*self.g))**2 - self.k*self.lamda**4/(64*self.g))*exp(-self.alpha*self.z*y)
			return np.transpose([Vx,Vy],axes=[1,0,2])
	
		def ddpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			#Vxx = 4*self.g*(-2*x**2*(1 - np.exp(-self.alpha*y*self.z)) + 3*x**2 - (1 - np.exp(-self.alpha*y*self.z))*(8*x**2 - self.lamda**2/self.g)/8 - self.lamda**2/(8*self.g))
			#Vxy = -4*self.alpha*self.g*x*self.z*(x**2 - self.lamda**2/(8*self.g))*np.exp(-self.alpha*y*self.z)
			#Vyy = self.alpha**2*(-2*self.D*(1 - np.exp(-self.alpha*y))*np.exp(-self.alpha*y) + 2*self.D*np.exp(-2*self.alpha*y) \
			#			+ self.z**2*(self.g*(8*x**2 - self.lamda**2/self.g)**2 - self.lamda**4/self.g)*np.exp(-self.alpha*y*self.z)/64)
			
			#Vxx =  4*self.g*(-2*x**2*(1 - exp(-self.alpha*self.z*y)) + 3*x**2 - (1 - exp(-self.alpha*self.z*y))*(8*x**2 - self.lamda**2/self.g)/8 - self.lamda**2/(8*self.g))
			#Vxy = -4*self.alpha*self.g*self.z*x*(x**2 - self.lamda**2/(8*self.g))*exp(-self.alpha*self.z*y)
			#Vyy = self.alpha**2*(-2*self.D*(1 - exp(-self.alpha*y))*exp(-self.alpha*y) \
			#	+ 2*self.D*exp(-2*self.alpha*y) + self.z**2*(4*self.g*(8*x**2 - self.lamda**2/self.g)**2 - self.lamda**4/self.g)*exp(-self.alpha*self.z*y)/256)
			
			#Vxx =  4*self.g*(-2*x**2*(1 - exp(-self.alpha*self.z*y)) + 3*x**2 - (1 - exp(-self.alpha*self.z*y))*(8*x**2 - self.lamda**2/self.g)/8 - self.lamda**2/(8*self.g))
			#Vxy = -4*self.alpha*self.g*self.z*x*(x**2 - self.lamda**2/(8*self.g))*exp(-self.alpha*self.z*y)
			#Vyy = self.alpha**2*(-2*self.D*(1 - exp(-self.alpha*y))*exp(-self.alpha*y) + 2*self.D*exp(-2*self.alpha*y)\
			#		 + self.z**2*(self.g*(8*x**2 - self.lamda**2/self.g)**2 - self.k*self.lamda**4/self.g)*exp(-self.alpha*self.z*y)/64)

			Vxx =  4*self.g*(-2*self.eta*x**2*(1 - exp(-self.alpha*self.z*y)) - self.eta*(1 - exp(-self.alpha*self.z*y))*(8*x**2 - self.lamda**2/self.g)/8 + 3*x**2 - self.lamda**2/(8*self.g))
			Vxy = -4*self.alpha*self.eta*self.g*self.z*x*(x**2 - self.lamda**2/(8*self.g))*exp(-self.alpha*self.z*y)
			Vyy = self.alpha**2*(-2*self.D*(1 - exp(-self.alpha*y))*exp(-self.alpha*y) +\
				 2*self.D*exp(-2*self.alpha*y) + self.eta*self.z**2*(self.g*(8*x**2 - self.lamda**2/self.g)**2 - self.k*self.lamda**4/self.g)*exp(-self.alpha*self.z*y)/64)

			return np.transpose([[Vxx, Vxy],[Vxy, Vyy]],axes=[2,0,1,3]) 
			
		def potential_xy(self,x,y):
			eay = np.exp(-self.alpha*y)
			quartx = (x**2 - self.lamda**2/(8*self.g))
			vy = self.D*(1-eay)**2#self.g*quarty**2#self.D*self.alpha**2*y**2#
			vx = self.g*quartx**2#self.D*(1-eax)**2#
			#vxy = (vx-self.lamda**4/(64*self.g))*(np.exp(-self.z*self.alpha*y) - 1) # 
			#vxy = (vx-self.lamda**4/(256*self.g))*(exp(-self.z*self.alpha*y) - 1)	
			Vb = self.lamda**4/(64*self.g)
			vxy = self.eta*(vx-self.k*Vb)*(exp(-self.z*self.alpha*y) - 1)	
			return vx + vy + vxy 
