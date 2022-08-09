import numpy as np
import scipy

class inst(object):
	def __init__(self,tol=1e-4,stepsize=0.01):
		self.tol = tol
		self.stepsize = stepsize

	def bind(self,pes,rp,cart=False):
		self.rp = rp
		self.pes = pes

		self.ndim = self.rp.ndim
		self.nbeads = self.rp.nbeads
		
		self.reinit_inst(cart)

	def reinit_inst(self,cart=False):
		# Ordering is [x1,x2,...xN,y1,y2,...yN]
		if(cart is False):
			self.q = (self.rp.q).flatten() 
			self.grad = (self.rp.dpot+self.pes.dpot).flatten()
			self.hess = self.flatten_hess(self.rp.ddpot+self.pes.ddpot)	
		else:
			self.q = (self.rp.qcart).flatten() 
			self.grad = (self.rp.dpot_cart+self.pes.dpot_cart).flatten()
			self.hess = self.flatten_hess(self.rp.ddpot_cart+self.pes.ddpot_cart)

	def reinit_inst_scaled(self,qscaled,n,eig_dir=0,cart=False):
		if(cart is False):
			self.q = qscaled.flatten()	
			dpot = self.rp.dpot+self.pes.dpot
			dpot[:,eig_dir]/=n
			self.grad = dpot.flatten()
			ddpot = self.rp.ddpot+self.pes.ddpot
			ddpot[:,eig_dir]/=n
			ddpot[:,:,eig_dir]/=n
			self.hess = self.flatten_hess(ddpot)
		else:	
			self.q = qscaled.flatten()	
			dpot = self.rp.dpot_cart+self.pes.dpot_cart
			dpot[:,eig_dir]/=n
			self.grad = dpot.flatten()
			ddpot = self.rp.ddpot_cart+self.pes.ddpot_cart
			ddpot[:,eig_dir]/=n
			ddpot[:,:,eig_dir]/=n
			self.hess = self.flatten_hess(ddpot)
	
	def flatten_hess(self,hess):
		# Rearranges the Hessian matrix to be consistent with the ordering [x1,x2,...xN,y1,y2,...yN]
		arr = hess.swapaxes(2,3).reshape(len(hess)*self.ndim*self.nbeads,len(hess)*self.ndim*self.nbeads)
		return arr		
	
	def diag_hess(self):
		vals,vecs = np.linalg.eigh(self.hess)
		return vals,vecs
		
	def saddle_init(self,q,delta,z,nbeads):
		rpqcart = np.zeros((1,len(q),nbeads))
		for i in range(nbeads):
			rpqcart[...,i] = q + delta*np.cos(np.pi*i/(nbeads-1))*z	 #initialize beads as in eqn. 6
		return rpqcart	
	
	def eigvec_follow_step(self):
		vals,vecs = np.linalg.eigh(self.hess)
		#print('vals',np.around(vals,4))
		F = np.matmul(vecs.T,self.grad)
		if((abs(F)<self.tol).all()):
			print('F,q,vals,vecs',F,self.rp.qcart,vals,vecs)
			return "terminate"
		lamda = 0.0
		alpha = 1.0
		if(vals[0]>0.0 and vals[1]/2 > vals[0]):
			alpha=1.0
			lamda=(vals[0]+vals[1]/2)/2
		elif(vals[0]>0.0 and vals[1]/2 <= vals[0]):
			alpha=(vals[1]-vals[0])/vals[1]
			lamda=(vals[0]+(vals[0]+vals[1])/2)/2
		elif(vals[0]<0.0):
			alpha = 1.0
			lamda = (vals[1]+vals[0])/4
		step = alpha*F/(lamda-vals)
		h = np.matmul(vecs, step)
		hnorm = np.sum(h**2)**0.5
		if(hnorm>self.stepsize):
			h/=(hnorm/self.stepsize)
		return h

	def eigvec_follow_step_inst(self):
		vals,vecs = np.linalg.eigh(self.hess)
		# vals[n] is the eigen value corresponding to vecs[:,n]
		F = np.matmul(vecs.T,self.grad)
		if((abs(self.grad)<self.tol).all()):## Check this once again later!!!
			print('F',np.linalg.norm(F))
			print('vals',vals[:2])
			return "terminate"
		lamda = 0.0
		alpha = 1.0
		if(vals[0]>0.0): 
			if(vals[1]/2 > vals[0]):
				alpha=1.0
				lamda=(vals[0]+vals[1]/2)/2
			else:
				alpha=(vals[1]-vals[0])/vals[1]
				lamda=(vals[0]+(vals[0]+vals[1])/2)/2
		elif(vals[1]<0.0):
			if (vals[1] >= vals[0]/2):
				alpha = 1.0
				lamda = (vals[0] + 2*vals[1])/4		
			else:
				alpha = (vals[0] - vals[1])/vals[1]
				lamda = (vals[0] + 3 * vals[1])/4	
		else:
			alpha = 1.0
			lamda = (vals[1]+vals[0])/4
		step = alpha*F/(lamda-vals)
		h = np.matmul(vecs, step)
		hnorm = np.sum(h**2)**0.5
		if(hnorm>self.stepsize):
			h/=(hnorm/self.stepsize)
		return h

	def grad_desc_step(self):
		vals,vecs = np.linalg.eigh(self.hess)
		F = np.matmul(vecs.T,self.grad)
		if((abs(F)<self.tol).all()):
			print('F,vals, terminating',F,vals)
			return "terminate"
		lamda = 0.0
		alpha = 1.0
		step = alpha*F/(lamda-vals)	
		if((np.dot(step,step)>self.stepsize**2 or vals[0]<0.0)):
			lamda = vals[0] - abs(F[0]/step.size)
			alpha = 1.0	
		step = alpha*F/(lamda-vals)
		h = np.matmul(vecs, step)
		hnorm = np.sum(h**2)**0.5
		if(hnorm>self.stepsize):
			h/=(hnorm/self.stepsize)
		return h
	
	def slow_step_update(self,step,cart=False):
		if(cart is False):
			self.q+=step
			self.rp.q = self.q.reshape((-1,self.ndim,self.nbeads))
			self.rp.mats2cart()
			self.pes.update()
			self.grad = (self.rp.dpot+self.pes.dpot).flatten()
			self.hess = self.flatten_hess(self.rp.ddpot+self.pes.ddpot)
		else:
			self.q+=step
			self.rp.qcart = self.q.reshape((-1,self.ndim,self.nbeads))
			self.rp.cart2mats()
			self.pes.update()
			self.grad = (self.rp.dpot_cart+self.pes.dpot_cart).flatten()
			self.hess = self.flatten_hess(self.rp.ddpot_cart+self.pes.ddpot_cart)
		
	def slow_step_update_soft(self,n,eig_dir,step):
		self.q+=step
		self.rp.q = (self.q.copy()).reshape((-1,self.ndim,self.nbeads))
		self.rp.q[:,eig_dir]*=n
		self.rp.mats2cart()
		self.pes.update()
		dpot = self.rp.dpot+self.pes.dpot
		dpot[:,eig_dir]*=n
		self.grad = dpot.flatten()
		ddpot = self.rp.ddpot+self.pes.ddpot
		ddpot[:,eig_dir]*=n
		ddpot[:,:,eig_dir]*=n
		self.hess = self.flatten_hess(ddpot)

	def powell_update(self,step,cart=False):
		grad_curr = self.grad
		if(cart is False): #Not written yet
			self.q+=step
		else:
			self.q+=step
			self.rp.qcart = self.q.reshape((-1,self.ndim,self.nbeads))
			self.rp.cart2mats()
			self.pes.update(update_hess=False)
			self.grad = (self.rp.dpot_cart+self.pes.dpot_cart).flatten()
			pk = step
			qk = self.grad - grad_curr
			Qp = np.matmul(self.hess,pk)
			Qp = qk - Qp
			self.hess += np.matmul(Qp,Qp.T)/np.dot(pk,Qp)
		
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
	
	def saddle_init(self,q,delta,z,nbeads):
		rpqcart = np.zeros((1,len(q),nbeads))
		for i in range(nbeads):
			rpqcart[...,i] = q + delta*np.cos(np.pi*i/(nbeads-1))*z	 #initialize beads as in eqn. 56
		return rpqcart	
			
	def eigvec_follow_step(self,eig_dir=0):
		# J. Phys. Chem. A 2005, 109, 9578-9583
		# Check for the sign of the eigenvectors; the code breaks when the eigenvector points in a wrong direction	
		vals,vecs = np.linalg.eigh(self.hess)
		F = np.matmul(vecs.T,self.grad)
		if(abs(F[eig_dir])<self.tol):
			print('F',F)
			return "terminate"
		lamda = np.zeros(len(self.q))
		ind = np.where(vals<0.0)
		lamda[ind] = -2*vals[ind]	
		if(vals[eig_dir]>=0.0): # Here, the 'cutoff' for using Lagrange multiplier may need to be altered depending on PES.
			lamda[eig_dir] = (vals[eig_dir]) + abs(F[eig_dir]/self.stepsize)
		else:
			lamda[eig_dir] = 0.0
		h = np.matmul(vecs, F/(lamda-vals))
		#h = -self.stepsize*F	#The directionality ought to be changed 'by hand' for the time being.	
		hnorm = np.sum(h**2)**0.5
		if(hnorm>self.stepsize):
			h/=(hnorm/self.stepsize)
		return h

	def eigvec_follow_step_inst(self,eig_dir=1):
		vals,vecs = np.linalg.eigh(self.hess)
		F = np.matmul(vecs.T,self.grad)
		if(abs(F[eig_dir])<self.tol):
			print('F',F)
			return "terminate"
		lamda = np.zeros(len(self.q))
		ind = np.where(vals<0.0)
		lamda[ind] = -2*vals[ind]	
		lamda[0]=0.0
		if(vals[eig_dir]>=0.0): # Here, the 'cutoff' for using Lagrange multiplier may need to be altered depending on PES.
			lamda[eig_dir] = (vals[eig_dir]) + abs(F[eig_dir]/self.stepsize)
		else:
			lamda[eig_dir] = 0.0
		h = np.matmul(vecs, F/(lamda-vals))
		#h = -self.stepsize*F	#The directionality ought to be changed 'by hand' for the time being.	
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
		if((abs(F)<self.tol).all()):
			print('F',F)
			return "terminate"
		lamda = np.zeros(len(self.q))
		if(vals[eig_dir]>=-2.0): # Here, the 'cutoff' for using Lagrange multiplier may need to be altered depending on PES.
			if(abs(F[eig_dir])>1e-2):
				lamda[eig_dir] = vals[eig_dir] + abs(F[eig_dir]/self.stepsize)
			else:
				lamda[eig_dir] = 0.0
			h = np.matmul(vecs, F/(lamda-vals))	
		else:
			h = -self.stepsize*F	#The directionality ought to be changed 'by hand' for the time being.	
		hnorm = np.sum(h**2)**0.5
		if(hnorm>self.stepsize):
			h/=(hnorm/self.stepsize)
		return h

	def grad_desc_step(self):
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
	
	
