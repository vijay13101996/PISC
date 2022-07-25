import numpy as np
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.integrators import Symplectic_order_II
from PISC.engine.gen_mc_ensemble import generate_rp
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt

class Poincare_SOS(object):
	def __init__(self,method,pathname,potkey,Tkey):
		self.method = method
		self.pathname = pathname
		self.potkey = potkey
		self.Tkey = Tkey
		self.X = []
		self.Y = []
		self.PX = []
		self.PY = []
					
	def set_sysparams(self,pes,T,m,dim):
		self.pes = pes
		self.T = T
		self.beta=1/T
		self.m = m
		self.dim = dim 
		
	def set_simparams(self,N,dt_ens,dt,nbeads=1,rngSeed=0):	
		self.N = N
		self.dt_ens = dt_ens
		self.dt = dt
		self.nbeads = nbeads
		self.rngSeed = rngSeed

	def set_runtime(self,time_ens,time_run):
		self.time_ens = time_ens
		self.time_run = time_run
	
	def bind(self,qcartg,pcartg=None,E=None,specific_traj=False,sym_init=False):
		self.ens = Ensemble(beta=self.beta,ndim=self.dim)
		self.motion = Motion(dt = self.dt,symporder=2) 
		self.rng = np.random.default_rng(self.rngSeed) 
		self.therm = PILE_L(tau0=0.1,pile_lambda=100.0) 
		self.propa = Symplectic_order_II()
		self.sim = RP_Simulation()
		self.E = E

		# If E is specified, the 'gen_mc_ensemble' function initializes a mc ensemble, 'ergodizes' the phase space and
		# returns the initial conditions pcartg, qcartg to use for plotting the Poincare section. 	
		if(E is not None):	
			#####look at generate rp (in gen_mc_ens)!!!!!!!###################### print( wie viele beads(und das es kein rp ist))
			generate_rp(self.pathname,self.m,self.dim,self.N,self.nbeads,self.ens,self.pes,self.rng,self.time_ens,self.dt,self.potkey,self.rngSeed,E,qcartg)
			qcartg = read_arr('Microcanonical_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,self.beta,self.potkey,self.rngSeed),"{}/Datafiles".format(self.pathname))
			pcartg = read_arr('Microcanonical_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,self.beta,self.potkey,self.rngSeed),"{}/Datafiles".format(self.pathname)) 
			
		# Specific trajectories could be chosen by specifying the 'ind' and uncommenting the lines below. 
		if(specific_traj!=False):
			ind= specific_traj
			#ind = [0,1]# how do I make them colourful???
			qcartg = qcartg[ind]
			pcartg = pcartg[ind]
		if(sym_init):
			qc = np.repeat(qcartg,2,axis=0)
			pc = np.repeat(pcartg,2,axis=0)
			qc[::2,0] = -qc[::2,0]
			pc[::2,0] = -pc[::2,0]
			qcartg = qc
			pcartg = pc
		self.rp = RingPolymer(qcart=qcartg,pcart=pcartg,m=self.m)	
		self.sim.bind(self.ens,self.motion,self.rng,self.rp,self.pes,self.propa,self.therm)

	def run_traj(self,ind,ax,nsteps=False,colour=False,sample=1e1):	
		if(nsteps==False):
			nsteps = int(self.time_run/self.motion.dt)
		print('Num steps in trajectory (plotted): ', nsteps)
		for i in range(nsteps):
			self.sim.step(mode="nve",var='pq')	
			x = self.rp.qcart[ind,0,:]
			px = self.rp.pcart[ind,0,:]
			y = self.rp.qcart[ind,1,:]
			py = self.rp.pcart[ind,1,:]
			if(i%sample==0):
				if(colour==False):
					ax.scatter(x,y,s=5)
				else:
					ax.scatter(x,y,s=5,c=colour)#s=marker size
				plt.pause(0.005)	
			
	def PSOS_Y(self,x0):
		prev = self.rp.q[:,0,0] - x0
		curr = self.rp.q[:,0,0] - x0
		count=0

		nsteps = int(self.time_run/self.motion.dt)
		print('E,kin,pot',self.rp.kin+self.pes.pot,self.rp.kin,self.pes.pot)
		Y_list = []
		PY_list = []###dont I also have to create X_list
		X_list = []
		for i in range(nsteps):
			self.sim.step(mode="nve",var='pq')	
			x = self.rp.q[0,0,0]/self.rp.nbeads**0.5#normal mode: 0th entry gives that
			px = self.rp.p[0,0,0]
			y = self.rp.q[0,1,0]/self.rp.nbeads**0.5
			py = self.rp.p[0,1,0]
			curr = x-x0
			if( prev*curr<0.0 and px>0.0 ):
				Y_list.append(y)
				PY_list.append(py)
				X_list.append(x)
			prev = curr
			count+=1

		self.Y.extend(Y_list)
		self.PY.extend(PY_list)

		return Y_list,PY_list,X_list
	
	def PSOS_X_gyr(self,y0,gyr_min,gyr_max,hist=False,Rg_of_t=False):
		prev = self.rp.q[:,1,0] - y0
		curr = self.rp.q[:,1,0] - y0
		count=0
		
		nsteps = int(self.time_run/self.motion.dt)
		therm_steps=int(30/self.motion.dt)
		#print('E,kin,pot',np.sum(self.rp.pcart**2/(2*self.m),axis=1)+self.pes.pot,self.rp.kin,self.pes.pot)
		X_list = []
		PX_list = []
		Y_list = []
		gyr_list_np=np.zeros((nsteps,2*self.N))
		for i in range(nsteps):
			self.sim.step(mode="nve",var='pq')	
			x = self.rp.q[:,0,0]/self.rp.nbeads**0.5
			px = self.rp.p[:,0,0]/self.rp.nbeads**0.5
			y = self.rp.q[:,1,0]/self.rp.nbeads**0.5
			py = self.rp.p[:,1,0]/self.rp.nbeads**0.5
			#cent_E = np.sum(self.rp.p[:,:,0]**2/self.rp.nbeads,axis=1) + self.pes.potential(self.rp.q[:,:,0]/self.rp.nbeads**0.5)
			
			gyr_x=0
			gyr_y=0
			for j in range(self.rp.nbeads): 
				gyr_x+=(x-self.rp.qcart[:,0,j])**2
				gyr_y+=(y-self.rp.qcart[:,1,j])**2
			gyr_x/=self.rp.nbeads
			gyr_y/=self.rp.nbeads#stays small anyways
			gyr_tot=gyr_x+gyr_y
			gyr_list_np[i,:]=gyr_tot[:]

			curr = y-y0
			ind = np.where( (prev*curr<0.0) & (py<0.0)& (gyr_min<np.max(gyr_list_np,axis=0))&(np.max(gyr_list_np,axis=0)<gyr_max))
			nsteps = int(self.time_run/self.motion.dt)
			if(10*i>=nsteps and i>therm_steps):#1/10th discarded
				X_list.extend(x[ind])#.append() adds a single element to the end of the list while .extend() can add multiple individual elements to the end of the list.
				PX_list.extend(px[ind])
				Y_list.extend(y[ind])
			prev = curr
			count+=1
		
		gyr_list_max= np.zeros(2*self.N)
		for s in range(2*self.N):
			gyr_list_max[s]=np.max(gyr_list_np[:,s])
		
		#Rg as a function of time
		print(self.N)
		if(Rg_of_t==True):
			for k in range(int(self.N/6)):
				fig, axs =plt.subplots(3,2, sharey='all')
				l=0
				for ax in axs.flat: 
					ax.plot(gyr_list_np[:,l+k*6])
					l+=1
			plt.show(block=False)
			plt.pause(1)
		###histogram
		if(hist==True):#hist iwth max and mean Rg
			#hist_fig, hist_ax= plt.subplots(2,sharex='all')
			hist_fig, hist_ax= plt.subplots(2)
			hist_fig.suptitle(r'max(R$_g)$')
			hist_ax[0].hist(gyr_list_max,range=(0,4),bins=80)
			#hist_ax[1].hist(gyr_list_mean,bins=50)
			#hist_ax[1].set_title(r'mean(R$_g)$')
			#hist_fig, hist_ax= plt.subplots(1,sharex='all')
			hist_ax[1].hist(gyr_list_max,range=(0,1),bins=50)
			#hist_ax[1].hist(gyr_list_mean,range=(0,1),bins=50)
			#hist_ax[1].set_title(r'mean(R$_g)$')		
			plt.show(block=False)
			plt.pause(1)
		if(hist==True):#Hist with all Rg
			hist_fig, hist_ax= plt.subplots(2)
			hist_fig.suptitle(r'R$_g$')
			hist_ax[0].hist(gyr_list_np.flatten(),range=(0,1),bins=100)
			hist_ax[1].hist(gyr_list_np.flatten(),range=(0,0.5),bins=1000)	
			plt.show(block=False)
			plt.pause(1)
		print('shape(X):',np.array(X_list).shape,'(at x: sign of y changes and py<0)')
		self.X.extend(X_list)
		self.PX.extend(PX_list)
		return X_list,PX_list,Y_list, gyr_list_np
		
	def PSOS_X(self,y0):
		prev = self.rp.q[:,1,0] - y0
		curr = self.rp.q[:,1,0] - y0
		count=0
		
		nsteps = int(self.time_run/self.motion.dt)
		#print('E,kin,pot',np.sum(self.rp.pcart**2/(2*self.m),axis=1)+self.pes.pot,self.rp.kin,self.pes.pot)
		X_list = []
		PX_list = []
		Y_list = []
		for i in range(nsteps):
			self.sim.step(mode="nve",var='pq')	
			x = self.rp.q[:,0,0]/self.rp.nbeads**0.5
			px = self.rp.p[:,0,0]/self.rp.nbeads**0.5
			y = self.rp.q[:,1,0]/self.rp.nbeads**0.5
			py = self.rp.p[:,1,0]/self.rp.nbeads**0.5
			#cent_E = np.sum(self.rp.p[:,:,0]**2/self.rp.nbeads,axis=1) + self.pes.potential(self.rp.q[:,:,0]/self.rp.nbeads**0.5)
			#print(len(x))
			
			curr = y-y0
			
			ind = np.where( (prev*curr<0.0) & (py<0.0))

			nsteps = int(self.time_run/self.motion.dt)
			X_list.extend(x[ind])#.append() adds a single element to the end of the list while .extend() can add multiple individual elements to the end of the list.
			PX_list.extend(px[ind])
			Y_list.extend(y[ind])
			prev = curr
			count+=1
		
		print('shape(X):',np.array(X_list).shape,'(at x: sign of y changes and py<0)')
		self.X.extend(X_list)
		self.PX.extend(PX_list)
		
		return X_list,PX_list,Y_list
		

	def store_data(self,coord): 
		key = [self.method,'Poincare_section',self.potkey,self.Tkey,'{}'.format(self.N)]
		fext = '_'.join(key)	
		fname = ''.join([fext])

		if(coord=='x'):	
			store_1D_plotdata(self.X,self.PX,fname,'{}/Datafiles'.format(self.pathname))	

		if(coord=='y'):	
			store_1D_plotdata(self.Y,self.PY,fname,'{}/Datafiles'.format(self.pathname))
