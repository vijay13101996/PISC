import numpy as np
from PISC.potentials.base import PES

class Veff_classical_1D_LH(PES):
		"""
		The local harmonic ansatz for the effective classical potential
		"""

		def __init__(self,pes,beta,m,grad=None,hess=None):
				super(Veff_classical_1D_LH).__init__()
				self.beta = beta
				self.m=m
				self.pes = pes
				self.grad = grad
				self.hess = hess

		def bind(self,ens,rp):
				super(Veff_classical_1D_LH,self).bind(ens,rp)

		def potential(self,q):
				#This provides an option to pass a customised hessian field if required
				if(self.hess is None):
						hess = self.pes.ddpotential(q)/self.m
				else:
						hess = self.hess(q)/self.m

				if(self.grad is None):
						grad = self.pes.dpotential(q)/self.m
				else:
						grad = self.grad(q)/self.m
	   
				wa=0.0
				ka = 2*grad
				Vc = self.pes.potential(q)
				if (hess >0):
						wa = np.sqrt(hess)
						xi = 0.5*self.beta*wa
						Vnc = self.m*ka**2*(np.tanh(xi)/xi -1)/(8*hess) #+ 0.5*np.log(np.sinh(2*xi)/(2*xi))/self.beta
						#- 0.5*np.log(np.tanh(xi)/xi)/self.beta + np.log(np.sinh(xi)/xi)/self.beta			  
						return Vc+Vnc
				elif (hess <0):
						wa = np.sqrt(-hess)
						xi = 0.5*self.beta*wa
						#print('q,sinxi/xi',q, np.sin(2*xi)/(2*xi),np.sin(2*xi),2*xi)
						# The potential is defined in terms of tan instead of tanh when the hessian is negative.
						Vnc = self.m*ka**2*(np.tan(xi)/xi-1)/(8*hess) + 0.5*np.log(np.sin(2*xi)/(2*xi))/self.beta
						#if(Vnc<-0.25):
						#	 print('q,Vnc, Veff', np.tan(xi)/xi, np.around(q,2),np.around(Vnc,3),Vc+Vnc)
						#- 0.5*np.log(np.tan(xi)/xi)/self.beta + np.log(np.sin(xi)/xi)/self.beta
						return Vc + Vnc
				elif (hess==0):
						return Vc

		def dpotential(self,q):
				return None

		def ddpotential(self,q):
				return None

class Veff_classical_1D_GH(PES):
	"""
	The global harmonic ansatz for the effective classical potential
	"""
	def __init__(self,pes,beta,m,hess=None):
		super(Veff_classical_1D_GH).__init__()	
		self.beta = beta 
		self.m=m
		self.pes = pes
		self.hess = hess
		
	def bind(self,ens,rp):
		super(Veff_classical_1D_GH,self).bind(ens,rp)
		
	def potential(self,q):
		#This provides an option to pass a customised hessian field if required
		if(self.hess is None):
			hess = self.pes.ddpotential(q)/self.m
		else:
			print('here')
			hess = self.hess(q)/self.m
		wa = 0.0
		Vc = self.pes.potential(q)
		if (hess >0):
			wa = np.sqrt(hess)
			ka = 2*grad
			xi = 0.5*self.beta*wa
			Vnc = m*ka**2*(np.tanh(xi)/xi -1)/(8*wa**2) + 0.5*self.m*hess*q**2*(np.tanh(xi)/xi - 1) - 0.5*np.log(2*self.m*xi*np.tanh(xi)/(np.pi*self.beta))/self.beta
			ret = 0.5*self.m*hess*q**2*(np.tanh(xi)/xi - 1) - 0.5*np.log(2*self.m*xi*np.tanh(xi)/(np.pi*self.beta))/self.beta
		elif (hess <0): 
			wa = np.sqrt(-hess)
			xi = 0.5*self.beta*wa
			# Potential is defined in terms of tan when the hessian is negative.		
			ret = self.pes.potential(q) + 0.5*self.m*hess*q**2*(np.tan(xi)/xi - 1) - 0.5*np.log(2*self.m*xi*np.tan(xi)/(np.pi*self.beta))/self.beta
		elif (hess==0):
			ret = Vc 

		return ret

	def dpotential(self,q):
		return None

	def ddpotential(self,q):
		return None

def convolution(pot,qgrid,q,beta,m):
	integral = 0.0
	dq = qgrid[1]-qgrid[0]
	for i in range(len(qgrid)):
		integral += np.sqrt(6*m/(np.pi*beta))*pot(qgrid[i])*np.exp(-6*m*(qgrid[i]-q )**2/beta)*dq
	return integral
	
class Veff_classical_1D_FH(PES):
	"""
	Feynman-Hibbs effective potential defined as a convolution.
	"""
	def __init__(self,pes,beta,m,qgrid):
		super(Veff_classical_1D_FH).__init__()	
		self.beta = beta 
		self.m=m
		self.pes = pes
		self.qgrid = qgrid
		
	def bind(self,ens,rp):
		super(Veff_classical_1D_FH,self).bind(ens,rp)
		
	def potential(self,q):
		# Eq. 11.23 in Feynman-Hibbs book
		veffq = convolution(self.pes.potential, self.qgrid, q, self.beta, self.m)
		return veffq	
	
	def dpotential(self,q):
		return None

	def ddpotential(self,q):
		return None

class Veff_classical_1D_FK(PES):
	"""
	Feynman-Kleinert approximate potential, computed without the
	variational minimisation step and truncated at the quadratic term.
	"""
	
	def __init__(self,pes,beta,m,hess=None):
		super(Veff_classical_1D_FK).__init__()	
		self.beta = beta 
		self.m=m
		self.pes=pes
		self.hess=hess
		
	def bind(self,ens,rp):
		super(Veff_classical_1D_FK,self).bind(ens,rp)
		
	def potential(self,q):
		#This provides an option to pass a customised hessian field if required
		if(self.hess is None):
			hess = self.pes.ddpotential(q)/self.m
		else:
			hess = self.hess(q)/self.m
		wa = 0.0
		if (hess >0):
			wa = np.sqrt(hess)
			xi = 0.5*self.beta*wa
			# Eq. 3.36 in text
			return self.pes.potential(q) + np.log(np.sinh(xi)/xi)/self.beta
		elif (hess <0):		
			wa = np.sqrt(-hess)
			xi = 0.5*self.beta*wa
			# Potential modified for negative hessian
			return self.pes.potential(q) + np.log(np.sin(xi)/xi)/self.beta
		elif (hess==0):
			return self.pes.potential(q)
	
	def dpotential(self,q):
		return None

	def ddpotential(self,q):
		return None


