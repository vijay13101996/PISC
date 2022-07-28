import numpy as np
from PISC.potentials.base import PES

class adams_function(PES):
		def __init__(self):
			super(adams_function).__init__()	
				
		def bind(self,ens,rp):
			super(adams_function,self).bind(ens,rp)
 
		def potential(self,q):
			x = q[:,0] 
			y = q[:,1]
			return self.potential_xy(x,y) 
		
		def dpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			Vx = 17*x**2*y*np.exp(-x**2/4 - y**2/4)/2 - 2*x**2 + 4*x*(4 - x) + y*(6 - 17*np.exp(-x**2/4 - y**2/4))
			Vy = 17*x*y**2*np.exp(-x**2/4 - y**2/4)/2 + x*(6 - 17*np.exp(-x**2/4 - y**2/4)) + y**2 + 2*y*(y + 4)
			return np.transpose([Vx,Vy],axes=[1,0,2])
	
		def ddpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			Vxx = -17*x**3*y*np.exp(-x**2/4 - y**2/4)/4 + 51*x*y*np.exp(-x**2/4 - y**2/4)/2 - 12*x + 16
			Vxy = -17*x**2*y**2*np.exp(-x**2/4 - y**2/4)/4 + 17*x**2*np.exp(-x**2/4 - y**2/4)/2 + 17*y**2*np.exp(-x**2/4 - y**2/4)/2 - 17*np.exp(-x**2/4 - y**2/4) + 6
			Vyy = -17*x*y**3*np.exp(-x**2/4 - y**2/4)/4 + 51*x*y*np.exp(-x**2/4 - y**2/4)/2 + 6*y + 8
			return np.transpose([[Vxx, Vxy],[Vxy, Vyy]],axes=[2,0,1,3]) 
			
		def potential_xy(self,x,y):
			return 2*x**2*(4 - x) + x*y*(6 - 17*np.exp(-x**2/4 - y**2/4)) + y**2*(y + 4)

 
