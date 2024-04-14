from __future__ import division, print_function, absolute_import
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
from scipy import sparse
from PISC.utils import misc
import pickle
import time

class DVR1D(object):
	def __init__(self,ngrid,lb,ub,m,potential,hbar=1.0):
		self.grid = np.linspace(lb,ub,ngrid)
		self.dx = self.grid[1]-self.grid[0]
		self.ngrid = ngrid
		self.lb = lb
		self.ub = ub
		self.m = m
		self.hbar = hbar
		self.potential = potential

		self.vals = np.zeros(ngrid)
		self.vecs = np.zeros((ngrid,ngrid))

	def Kin_matrix_elt(self,i,j,lb=None,ub=None,ngrid=None):
		if lb is None:	
			lb = self.lb
		if ub is None:	
			ub = self.ub
		if ngrid is None:
			ngrid = self.ngrid	
		const = (self.hbar**2/(2*self.m))
		prefactor = (np.pi**2/2)*( (-1)**(i-j)/(ub-lb)**2 )
		if(i!=j):
			sin_term1 = 1/(np.sin(np.pi*(i-j)/(2*ngrid)))**2
			sin_term2 = 1/(np.sin(np.pi*(i+j)/(2*ngrid)))**2
			return const*prefactor*(sin_term1 - sin_term2)
		else:
			if(i!=0 and i!=ngrid):
				const_term = (2*ngrid**2 +1)/3
				sin_term3 = 1/(np.sin(np.pi*i/ngrid))**2
				return const*prefactor*(const_term - sin_term3)
			else:
				return 0.0

	def Kin_matrix(self):
		Kin_mat = np.zeros((self.ngrid,self.ngrid))
		for i in range(0,self.ngrid):
			for j in range(0,self.ngrid):
					Kin_mat[i][j] = self.Kin_matrix_elt(i+1,j+1)	
		return Kin_mat

	def Pot_matrix(self):
		Pot_mat = np.zeros((self.ngrid,self.ngrid))
		for i in range(0,self.ngrid):
			for j in range(0,self.ngrid):
				if(i==j):
					Pot_mat[i][j] = self.potential(self.grid[i])
		return Pot_mat  

	def Diagonalize(self,neig_total=150):
		T = self.Kin_matrix()
		V = self.Pot_matrix()

		H = T+V
		vals, vecs = eigsh(H,k=neig_total,which='SM') # np.linalg.eigh(H)
		
		norm = 1/(np.sum(vecs[:,0]**2*self.dx))**0.5
		vecs*=norm

		self.vecs = vecs
		self.vals = vals
	
		return vals,vecs
	
class DVR2D(DVR1D):
	def __init__(self,ngridx,ngridy,lbx,ubx,lby,uby,m,potential,hbar=1.0):
		self.xgrid = np.linspace(lbx,ubx,ngridx+1)
		self.ygrid = np.linspace(lby,uby,ngridy+1)
		self.dx = self.xgrid[1]-self.xgrid[0]
		self.dy = self.ygrid[1]-self.ygrid[0]
		self.ngridx = ngridx
		self.ngridy = ngridy
		self.lbx = lbx
		self.ubx = ubx
		self.lby = lby
		self.uby = uby
		self.m = m
		self.hbar = hbar
		self.potential = potential

		self.vals = None#np.zeros((self.ngridx+1)*(self.ngridy+1))
		self.vecs = None#np.zeros(((self.ngridx+1)*(self.ngridy+1),(self.ngridx+1)*(self.ngridy+1)))

	def Kin_matrix_2D_elt(self,ix,jx,iy,jy):
		if(iy==jy and ix!=jx):
			return self.Kin_matrix_elt(ix,jx,self.lbx,self.ubx,self.ngridx)
		elif(ix==jx and iy!=jy):
			return self.Kin_matrix_elt(iy,jy,self.lby,self.uby,self.ngridy)
		elif(ix==jx and iy==jy):
			return self.Kin_matrix_elt(ix,jx,self.lbx,self.ubx,self.ngridx) + self.Kin_matrix_elt(iy,jy,self.lby,self.uby,self.ngridy)
		else:
			return 0.0

	def Kin_matrix(self): 
		tuple_arr = self.tuple_index()
		length = len(tuple_arr)
		Kin_mat = np.zeros((length,length))
		for i in range(length):
			for j in range(length):
				ix = tuple_arr[i][0]
				iy = tuple_arr[i][1]
				jx = tuple_arr[j][0]
				jy = tuple_arr[j][1]
				Kin_mat[i][j] = self.Kin_matrix_2D_elt(ix,jx,iy,jy)
		np.set_printoptions(threshold=np. inf)
		return Kin_mat

	def Kin_matrix_mod(self):
		tuple_arr = self.tuple_index()
		length = len(tuple_arr)
		row_mat = []
		col_mat = []
		data_mat = []
		for i in range(length):
			for j in range(length):
				ix = tuple_arr[i][0]
				iy = tuple_arr[i][1]
				jx = tuple_arr[j][0]
				jy = tuple_arr[j][1]
				kin = self.Kin_matrix_2D_elt(ix,jx,iy,jy)
				if(kin!=0.0):
					row_mat.append(i)
					col_mat.append(j)
					data_mat.append(kin)
		row_mat = np.array(row_mat)
		col_mat = np.array(col_mat)
		data_mat = np.array(data_mat)
		Kin_mat = sparse.csr_matrix((data_mat, (row_mat,col_mat)))
		return Kin_mat

	def Pot_matrix(self): 
		tuple_arr = self.tuple_index()
		length = len(tuple_arr)
		Pot_mat =  np.zeros((length,length))
		for i in range(length):
			for j in range(length):
				ix = tuple_arr[i][0]
				iy = tuple_arr[i][1]
				jx = tuple_arr[j][0]
				jy = tuple_arr[j][1]
				if(ix==jx and iy==jy):
					Pot_mat[i][j] = self.potential(self.xgrid[ix],self.ygrid[iy])	
		return Pot_mat	 

	def Pot_matrix_mod(self):
		tuple_arr = self.tuple_index()
		length = len(tuple_arr)
		data_mat = []
		for i in range(length):
			i_ind = tuple_arr[i][0]
			j_ind = tuple_arr[i][1]
			data_mat.append(self.potential(self.xgrid[i_ind],self.ygrid[j_ind]))	
		Pot_mat = sparse.csr_matrix((data_mat, (range(length),range(length))))
		return Pot_mat	 

	def Diagonalize(self,neig_total=150):
		start_time = time.time()
		T = self.Kin_matrix_mod()
		V = self.Pot_matrix_mod()	
		
		H = T+V
		print('time',time.time()-start_time)	

		vals, vecs = eigsh(H,k=neig_total,which='SM') #np.linalg.eigh(H)	
		norm = 1/(np.sum(vecs[:,0]**2*self.dx*self.dy))**0.5
		vecs*=norm

		self.vecs = vecs
		self.vals = vals
	
		return vals,vecs

	def tuple_index(self):
		temp = []
		count = 0
		for i in range(self.ngridx+1):
			for j in range(self.ngridy+1):
				temp.append((i,j))
				count+=1
		return temp
	
	def pos_mat(self,ind):
		temp = self.tuple_index()
		x_arr = np.zeros(len(temp))
		for p in range(len(temp)):
			i = temp[p][ind]  ##It works alright, except for the fact the xgrid/ygrid needs to be toggled. 
			x_arr[p] = self.xgrid[i]
		return x_arr

	# The function below swaps x and y axis for some reason. However, the ordering of the tuple_index is x-major as usual.

	def eigenstate(self,vector):
		temp = self.tuple_index()
		wf = np.zeros((self.ngridx+1,self.ngridy+1))	
		for p in range(len(temp)):
			i = temp[p][0] 
			j = temp[p][1]
			wf[i,j] = vector[p]
		return wf.T 
