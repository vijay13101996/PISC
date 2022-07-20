from __future__ import division, print_function, absolute_import
import numpy as np
from PISC.husimi import husimi

class Husimi_1D(object):
	def __init__(self,qgrid,sigma,hbar=1.0):
		self.qgrid = qgrid
		self.sigma = sigma
		self.hbar = 1.0
		self.dq = self.qgrid[1]-self.qgrid[0]

	def coherent_state(self,q0,p0):
		return (1/(np.pi*self.sigma**2))**0.25*np.exp(-(self.qgrid-q0)**2/(2*self.sigma**2) + 1j*p0*(self.qgrid-q0)/self.hbar)
		
	def coherent_projection(self,q0,p0,wf):
		if(len(wf)!=len(self.qgrid)):
			raise ValueError
		amp = 0.0
		coh_state = self.coherent_state(q0,p0)
		for i in range(len(self.qgrid)-1):
			amp+= np.conj(coh_state[i])*wf[i]*self.dq
		return amp		

	def Husimi_distribution(self,qbasis,pbasis,wf):
		dist = np.zeros((len(qbasis),len(pbasis)))
		for i in range(len(qbasis)):
			for j in range(len(pbasis)):
				dist[i][j] = abs(self.coherent_projection(qbasis[i],pbasis[j],wf))**2
		return dist

class Husimi_2D(object):
	def __init__(self,xgrid,sigmax,ygrid,sigmay,hbar=1.0):
		self.xgrid = xgrid
		self.sigmax = sigmax
		self.ygrid = ygrid
		self.sigmay = sigmay
		self.hbar = hbar
		x,y = np.meshgrid(self.xgrid,self.ygrid)
		self.x = x
		self.y = y
		self.dx = self.xgrid[1]-self.xgrid[0]
		self.dy = self.ygrid[1]-self.ygrid[0]

	def coherent_state(self,x0,px0,y0,py0):
		cohx = (1/(np.pi*self.hbar*self.sigmax**2))**0.25*np.exp(-(self.x-x0)**2/(2*self.hbar*self.sigmax**2) + 1j*px0*(self.x-x0)/self.hbar)
		cohy = (1/(np.pi*self.hbar*self.sigmay**2))**0.25*np.exp(-(self.y-y0)**2/(2*self.hbar*self.sigmay**2) + 1j*py0*(self.y-y0)/self.hbar)
		return cohx*cohy
	
	def coherent_state_fort(self,x0,px0,y0,py0):
		coh = np.zeros_like(self.x) + 0j
		coh = husimi.husimi_section.coherent_state(self.x,self.y, x0,px0,y0,py0,self.sigmax,self.sigmay,self.hbar,coh)
		return coh
		
	def coherent_projection(self,x0,px0,y0,py0,wf):
		amp = 0.0j
		coh_state = self.coherent_state(x0,px0,y0,py0)
		for i in range(len(self.xgrid)):
			for j in range(len(self.ygrid)):
				amp+= np.conj(coh_state[i,j])*wf[i,j]*self.dx*self.dy		
		return amp	

	def coherent_projection_fort(self,x0,px0,y0,py0,wf):	
		amp = 0.0j
		amp = husimi.husimi_section.coherent_projection(self.x,self.y, x0,px0,y0,py0,self.sigmax,self.sigmay,self.dx,self.dy,self.hbar,wf,amp)
		return amp

	def Husimi_section_x(self,xbasis,pxbasis,y0,py0_sign,wf,E_wf,potfunc,m):
		dist = np.zeros((len(xbasis),len(pxbasis)))
		for i in range(len(xbasis)):
			for j in range(len(pxbasis)):
				pot = potfunc(xbasis[i],y0)
				py0_sq = (2*m*(E_wf - pot - pxbasis[j]**2/(2*m)))
				if(py0_sq > 0.0):
					py0 = py0_sq**0.5 
					dist[i][j] = abs(self.coherent_projection(xbasis[i],pxbasis[j],y0,py0*py0_sign,wf))**2
		return dist
	
	def Husimi_section_x_fort(self,xbasis,pxbasis,y0,wf,E_wf,potfunc,m):
		potgrid = potfunc(xbasis,y0)
		dist = np.zeros((len(xbasis),len(pxbasis)))
		dist = husimi.husimi_section.husimi_section_x(self.x,self.y,xbasis,pxbasis,\
					y0,potgrid,wf,E_wf,m,self.hbar,self.sigmax,self.sigmay, self.dx, self.dy,dist)
		return dist

	def Husimi_section_y(self,ybasis,pybasis,xbasis,wf,E_wf,potfunc,m):
		dist = np.zeros((len(ybasis),len(pybasis)))
		for i in range(len(ybasis)):
			for j in range(len(pybasis)):
				for x0 in xbasis:
					pot = potfunc(x0,ybasis[i])
					px0_sq = (2*m*(E_wf - pot - pybasis[j]**2/(2*m)))
					if(px0_sq > 0.0):
						px0 = px0_sq**0.5
						dist[i][j] += abs(self.coherent_projection(x0,px0,ybasis[i],pybasis[j],wf))**2
						dist[i][j] += abs(self.coherent_projection(x0,-px0,ybasis[i],pybasis[j],wf))**2	
		return dist

	def Husimi_section_y_fort(self,ybasis,pybasis,x0,wf,E_wf,potfunc,m):
		potgrid = potfunc(x0,ybasis)
		dist = np.zeros((len(ybasis),len(pybasis)))
		dist = husimi.husimi_section.husimi_section_y(self.x,self.y,ybasis,pybasis,\
					x0,potgrid,wf,E_wf,m,self.hbar,self.sigmax,self.sigmay, self.dx, self.dy,dist)
		return dist

	def Husimi_rep_y_fort(self,ybasis,pybasis,wf,E_wf,m):
		rep = np.zeros((len(ybasis),len(pybasis)))
		rep = husimi.husimi_section.husimi_rep_y(self.x,self.y,ybasis,pybasis,\
					wf,E_wf,m,self.hbar,self.sigmax,self.sigmay,self.dx,self.dy,rep)
		return rep
	
	def Husimi_rep_x_fort(self,xbasis,pxbasis,wf,E_wf,m):
		rep = np.zeros((len(xbasis),len(pxbasis)))
		rep = husimi.husimi_section.husimi_rep_x(self.x,self.y,xbasis,pxbasis,\
					wf,E_wf,m,self.hbar,self.sigmax,self.sigmay,self.dx,self.dy,rep)
		return rep

	def Husimi_rep_fort(self,xbasis,pxbasis,ybasis,pybasis,wf,E_wf,m):
		rep = np.zeros((len(xbasis),len(pxbasis),len(ybasis),len(pybasis))) + 0j
		rep = husimi.husimi_section.husimi_rep(self.x,self.y,xbasis,pxbasis,ybasis,pybasis,\
					wf,E_wf,m,self.hbar,self.sigmax,self.sigmay,self.dx,self.dy,rep)
		return abs(rep)**2

	def tot_density(self,dist,qbasis,pbasis):
		norm = 0.0
		A = 0.0
		dq = qbasis[1]-qbasis[0]
		dp = pbasis[1]-pbasis[0]
		for i in range(len(qbasis)):
			for j in range(len(pbasis)):
				norm+=dist[i,j]*dq*dp
				if(dist[i,j]>0.0):
					A+=dq*dp
		Smax = np.log(A)
		print('Smax',Smax)
		return norm, Smax

	def Renyi_entropy(self,xbasis, pxbasis,ybasis,pybasis,dist):
		S = 0.0	
		S = husimi.husimi_section.renyi_entropy(xbasis,pxbasis,ybasis,pybasis,dist,S)
		return -S
		
	def Renyi_entropy_1D(self,qbasis, pbasis,dist,order):
		S = 0.0
		norm,Smax = self.tot_density(dist,qbasis,pbasis)	
		dist/=norm
		S = husimi.husimi_section.renyi_entropy_1d(qbasis,pbasis,dist,order,S)
		return -S#/Smax

		

		
