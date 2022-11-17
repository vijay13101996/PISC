import numpy as np
from PISC.potentials.base import PES
from numpy import exp

class four_well(PES):
	def __init__(self,lamdax,gx,lamday,gy,z,invert_pes=1.0):
		super(four_well).__init__()
		self.gx = gx
		self.lamdax = lamdax
		self.gy = gy
		self.lamday = lamday
		
		self.z = z

		self.invert_pes = invert_pes

	def bind(self,ens,rp):
		super(four_well,self).bind(ens,rp)

	def potential(self,q):
		x = q[:,0] 
		y = q[:,1]
		return self.invert_pes*self.potential_xy(x,y) 
		
	def dpotential(self,q):
		x = q[:,0] 
		y = q[:,1]
		Vx = 4*self.gx*x*(x**2 - self.lamdax**2/(8*self.gx)) + 2*self.z*x*y**2
		Vy = 4*self.gy*y*(y**2 - self.lamday**2/(8*self.gy)) + 2*self.z*x**2*y
		return np.transpose([Vx,Vy],axes=[1,0,2])
	
	def ddpotential(self,q):
		x = q[:,0] 
		y = q[:,1]
		Vxx =  2*(4*self.gx*x**2 + self.gx*(8*x**2 - self.lamdax**2/self.gx)/4 + self.z*y**2)
		Vxy = 4*self.z*x*y
		Vyy = 2*(4*self.gy*y**2 + self.gy*(8*y**2 - self.lamday**2/self.gy)/4 + self.z*x**2)
		#print('Vxx,Vyy, Vxy', x,y)
		return np.transpose([[Vxx, Vxy],[Vxy, Vyy]],axes=[2,0,1,3]) 
			
	def potential_xy(self,x,y):
		quartx = (x**2 - self.lamdax**2/(8*self.gx))
		quarty = (y**2 - self.lamday**2/(8*self.gy))	
		vx = self.gx*quartx**2
		vy = self.gy*quarty**2
		vxy = self.z*x**2*y**2
		
		V = vx + vy + vxy
		return V
							
