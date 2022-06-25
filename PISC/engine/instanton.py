import numpy as np
import scipy

class instantonize(object):
	def __init__(self,tol=1e-4,stepsize=0.01):
		self.tol = tol
		self.stepsize = stepsize

	def bind(self,pes,rp):
		self.rp = rp
		self.pes = pes

		self.ndim = self.rp.ndim
		self.nbeads = self.rp.nbeads
		
		self.reinit_inst()

	def reinit_inst(self):
		# Ordering is [x1,x2,...xN,y1,y2,...yN]
		self.q = (self.rp.q).flatten() 
		self.grad = (self.rp.dpot+self.pes.dpot).flatten()
		self.hess = self.flatten_hess(self.rp.ddpot+self.pes.ddpot)
	
	def reinit_inst_scaled(self,qscaled,n,eig_dir):
		self.q = qscaled.flatten()	
		dpot = self.rp.dpot+self.pes.dpot
		dpot[:,eig_dir]/=n
		self.grad = dpot.flatten()
		ddpot = self.rp.ddpot+self.pes.ddpot
		ddpot[:,eig_dir]/=n
		ddpot[:,:,eig_dir]/=n
		self.hess = self.flatten_hess(ddpot)
	
	def flatten_hess(self,hess):
		# Rearranges the Hessian matrix to be consistent with the ordering [x1,x2,...xN,y1,y2,...yN]
		arr = hess.swapaxes(2,3).reshape(len(hess)*self.ndim*self.nbeads,len(hess)*self.ndim*self.nbeads)
		return arr		

	def Powell_update_hessian(self,step):
		grad_curr = self.grad
		#Increment current position
		#Find potential, force at new position
		#Compute yk and gamma_k
		#Powell-update the hessian
	
	def saddle_init(self,q,delta,z):
		rpqcart = np.zeros_like(self.rp.qcart)
		for i in range(self.rp.nbeads):
			rpqcart[...,i] = q + delta*np.cos(np.pi*i/(self.rp.nbeads-1))*z	 #initialize beads as in eqn. 56
		return rpqcart	
			
	def eigvec_follow_step(self,eig_dir=0):
		# J. Phys. Chem. A 2005, 109, 9578-9583
		# Check for the sign of the eigenvectors; the code breaks when the eigenvector points in a wrong direction	
		vals,vecs = np.linalg.eigh(self.hess)
		F = np.matmul(vecs.T,self.grad)
		if((abs(F)<self.tol).all()):#abs(F[eig_dir])<self.tol):# and abs(F[0])<self.tol):
			print('F',F)
			return "terminate"
		lamda = np.zeros(len(self.q)) - 0.1
		if(vals[eig_dir]>=-2.0): # Here, the 'cutoff' for using Lagrange multiplier may need to be altered depending on PES.
			lamda[eig_dir] = vals[eig_dir] + abs(F[eig_dir]/self.stepsize)
			#lamda[eig_dir+1] = vals[eig_dir+1] + abs(F[eig_dir+1]/self.stepsize)
			h = np.matmul(vecs, F/(lamda-vals))	
		else:
			h = -self.stepsize*F	#The directionality ought to be changed 'by hand' for the time being. 	
		hnorm = np.sum(h**2)**0.5
		if(hnorm>self.stepsize):
			h/=(hnorm/self.stepsize)
		return h

	def eigvec_follow_step_soft(self,eig_dir=0):
		vals,vecs = np.linalg.eigh(self.hess)
		F = np.matmul(vecs.T,self.grad)
		if(vals[eig_dir] >= 0.5*vals[eig_dir-1]):
			h = -self.stepsize*vecs[eig_dir]
			print('here 1')
			return h			
		if((abs(F)<self.tol).all()):#abs(F[eig_dir])<self.tol):# and abs(F[0])<self.tol):
			print('F',F)
			return "terminate"
		lamda = np.zeros(len(self.q))
		if(vals[eig_dir]>=-2.0): # Here, the 'cutoff' for using Lagrange multiplier may need to be altered depending on PES.
			if(abs(F[eig_dir])>1e-2):
				print('here 2')
				lamda[eig_dir] = vals[eig_dir] + abs(F[eig_dir]/self.stepsize)
			else:
				print('here 3')
				lamda[eig_dir] = 0.0
			#lamda[eig_dir+1] = vals[eig_dir+1] + abs(F[eig_dir+1]/self.stepsize)	
			h = np.matmul(vecs, F/(lamda-vals))	
		else:
			print('here 4')
			h = -self.stepsize*F	#The directionality ought to be changed 'by hand' for the time being. 	
		hnorm = np.sum(h**2)**0.5
		if(hnorm>self.stepsize):
			h/=(hnorm/self.stepsize)
		return h

	def grad_desc_step(self):
		#print('rp', self.rp.ens.beta)
		gradnorm = np.sum(self.grad**2)**0.5
		h = -self.stepsize*self.grad/gradnorm
		if(gradnorm<self.tol):
			return "terminate"
		hnorm = np.sum(h**2)**0.5
		if(hnorm>self.stepsize):
			h/=(hnorm/self.stepsize)	
		return h

	def diag_hess(self):
		vals,vecs = np.linalg.eigh(self.hess)
		return vals,vecs
		
	def slow_step_update(self,step):
		self.q+=step
		self.rp.q = self.q.reshape((-1,self.ndim,self.nbeads))
		self.rp.mats2cart()
		self.pes.update()
		self.grad = (self.rp.dpot+self.pes.dpot).flatten()
		self.hess = self.flatten_hess(self.rp.ddpot+self.pes.ddpot)

	def slow_step_update_soft(self,step,n,eig_dir):
		self.q+=step
		self.rp.q = (self.q.copy()).reshape((-1,self.ndim,self.nbeads))
		self.rp.q[:,eig_dir+1]*=n
		self.rp.mats2cart()
		self.pes.update()
		dpot = self.rp.dpot+self.pes.dpot
		dpot[:,eig_dir]*=n
		self.grad = dpot.flatten()
		ddpot = self.rp.ddpot+self.pes.ddpot
		ddpot[:,eig_dir]*=n
		ddpot[:,:,eig_dir]*=n
		self.hess = self.flatten_hess(ddpot)

	def Newton_Raphson_step(self,eps=0.1):
		gradnorm = np.sum(self.grad**2)**0.5
		if(gradnorm<self.tol):
			return "terminate"		
		hess_aux = self.hess + eps*np.eye(self.rp.nbeads*self.rp.ndim)
		h = np.matmul(scipy.linalg.inv(hess_aux),-(self.grad))
		hnorm = np.sum(h**2)**0.5
		if(hnorm>self.stepsize):
			h/=(hnorm/self.stepsize)
		return h	
		
