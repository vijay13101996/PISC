import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.utils.readwrite import store_1D_plotdata, store_2D_imagedata, read_arr
from PISC.utils.tcf_fft import gen_tcf, gen_2pt_tcf
from PISC.engine.thermalize_PILE_L import thermalize_rp
from PISC.engine.gen_mc_ensemble import generate_rp

class SimUniverse(object):
	def __init__(self,method,pathname,sysname,potkey,corrkey,enskey,Tkey,ext_kwlist=None):
		self.method = method
		self.pathname = pathname
		self.sysname = sysname
		self.potkey = potkey
		self.corrkey = corrkey
		self.enskey = enskey
		self.Tkey = Tkey
		self.ext_kwlist = ext_kwlist
	
	def set_sysparams(self,pes,T,mass,dim):
		''' Set system paramenters 
		    pes = potential energy surface
		    T = temperature
		    mass = particle mass
		    dim = dimensionality of the classical system '''

		self.pes = pes
		self.T = T
		self.m = mass
		self.dim = dim 
		
	def set_simparams(self,N,dt_ens=1e-2,dt=5e-3,extparam=None):	
		self.N = N
		self.dt_ens = dt_ens
		self.dt = dt
		if extparam is not None:
			self.extparam = extparam

	def set_methodparams(self,nbeads=1,gamma=1):
		if(self.method=='Classical'):
			self.nbeads = 1
		else:
			self.nbeads = nbeads
		if(self.method=='CMD'):
			self.gamma = gamma

	def set_runtime(self,time_ens=100.0,time_run=5.0):
		self.time_ens = time_ens
		self.time_run = time_run
	
	def set_ensparams(self,tau0 = 1.0, pile_lambda=100.0, E=None, qlist= None, plist = None, filt_func = None):
		self.tau0 = tau0
		print('tau0, pi',tau0) 
		self.pile_lambda = pile_lambda
		self.E = E
		self.qlist = qlist
		self.plist = plist
		self.filt_func = filt_func
	
	def gen_ensemble(self,ens,rng,rngSeed):
		if(self.enskey== 'thermal'):
			thermalize_rp(self.pathname,self.m,self.dim,self.N,self.nbeads,ens,self.pes,rng,self.time_ens,self.dt_ens,self.potkey,rngSeed,self.qlist,self.tau0,self.pile_lambda)	
			qcart = read_arr('Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,ens.beta,self.potkey,rngSeed),"{}/Datafiles".format(self.pathname))
			pcart = read_arr('Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,ens.beta,self.potkey,rngSeed),"{}/Datafiles".format(self.pathname))
			return qcart,pcart	
		elif(self.enskey=='mc'):
			generate_rp(self.pathname,self.m,self.dim,self.N,self.nbeads,ens,self.pes,rng,self.time_ens,self.dt_ens,self.potkey,rngSeed,self.E,self.qlist,self.plist,self.filt_func)
			qcart = read_arr('Microcanonical_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,ens.beta,self.potkey,rngSeed),"{}/Datafiles".format(self.pathname))
			pcart = read_arr('Microcanonical_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,ens.beta,self.potkey,rngSeed),"{}/Datafiles".format(self.pathname)) 
			return qcart,pcart	

	def run_OTOC(self,sim,single=False):
		tarr = []
		Mqqarr = []
		if(self.method == 'CMD'):
			stride = self.gamma	
			dt = self.dt/self.gamma
			nsteps = int(2*self.time_run/dt)
			for i in range(nsteps):
				sim.step(mode="nvt",var='monodromy',pc=False)
				if(i%stride == 0):
					Mqq = np.mean(abs(sim.rp.Mqq[:,0,0,0,0]**2)) 
					tarr.append(sim.t)
					Mqqarr.append(Mqq)
		else:
			dt = self.dt
			nsteps = int(self.time_run/dt)
			q = sim.rp.q.copy()
			sigma = 0.5
			ind = np.where(abs(q) < sigma)
			wgt = np.exp(-q/(2*sigma**2))/(2*np.pi*sigma)**0.5
			for i in range(nsteps):
				sim.step(mode="nve",var='monodromy')
				if(single):
					Mqq = np.mean(sim.rp.Mqp[:,0,0,0,0]) # Change notations when required!
				else:
					Mqq = np.mean(abs(sim.rp.Mqq[:,0,0,0,0]**2))
				tarr.append(sim.t)
				Mqqarr.append(Mqq)

		return tarr, Mqqarr

	def run_R2(self,sim,A='p',B='p',C='q'):
		""" Run simulation to compute second order (sym and asym) response functions """
		# IMPORTANT: Be careful when you use it for 2D! There are parts used in this code,
		# which are 1D-specific.
	
		tarr, qarr, parr, Marr = [], [], [], []
		dt = self.dt
		nsteps = int(self.time_run/dt) + 1
		sqrtnbeads = sim.rp.nbeads**0.5
		#comm_dict = {'qq':[-1.0,'Mqp'],'qp':[1.0,'Mqq'], 'pq':[-1.0,'Mpp'], 'pp':[1.0,'Mpq']}

		def record_var():
			Mqq = sim.rp.Mqq[:,0,0,0,0].copy()
			#Mqp = sim.rp.Mqp[:,0,0,0,0].copy()
			#Mpq = sim.rp.Mpq[:,0,0,0,0].copy()
			Mpp = sim.rp.Mpp[:,0,0,0,0].copy()
			q = sim.rp.q[:,0,0].copy()		
			p = sim.rp.p[:,0,0].copy()/sim.rp.m
		
			Mval = Mqq/sim.rp.m#-Mpp#comm_dict[B+A][0]*(vars()[comm_dict[B+A][1]])	
			tarr.append(sim.t)
			Marr.append(Mval)
			qarr.append(q/sqrtnbeads)  #Needed to define centroids with correct scaling
			parr.append(p/sqrtnbeads)
					
		pcart = sim.rp.pcart.copy()
		qcart = sim.rp.qcart.copy() 
 
		#Forward propagation
		for i in range(nsteps):
			record_var()
			sim.step(mode="nve",var='monodromy')
		
		#Reinitialising position and momenta for backward propagation
		sim.rp = RingPolymer(qcart=qcart,pcart=pcart,m=sim.rp.m,mode='rp') #Only RPMD here!
		sim.motion = Motion(dt=-self.dt,symporder=sim.motion.order)
		sim.bind(sim.ens,sim.motion,sim.rng,sim.rp,sim.pes,sim.propa,sim.therm)
		sim.t = 0.0	

		#Backward propagation
		for i in range(nsteps-1):
			sim.step(mode="nve",var='monodromy')
			record_var()

		print('Propagation completed')
		
		op_dict = {'I':np.ones_like(np.array(qarr)),'q':qarr,'p':parr}
		Aarr = np.array(op_dict[A].copy())
		Barr = np.array(op_dict[B].copy())
		Carr = np.array(op_dict[C].copy())

		Marr = np.array(Marr.copy()) 
		Marr = Marr[:,:,None] # NEEDS TO CHANGE FOR 2D
		tarr = np.array(tarr)
	
		tarr1,Csym = gen_2pt_tcf(dt,tarr,Carr,Barr,Aarr)  # In the order t2,t1 and t0.
		tarr2,Casym = gen_2pt_tcf(dt,tarr,Carr,Marr)
		if(np.alltrue(tarr1 == tarr2)):
			tarr = tarr1

		return tarr, Csym, Casym

	def run_R3(self,sim):
		tarr, qarr, parr, Marr = [], [], [], []
		dt = self.dt
		nsteps = int(self.time_run/dt) + 1
		sqrtnbeads = sim.rp.nbeads**0.5

		def record_var():
			Mqq = sim.rp.Mqq[:,0,0,0,0].copy()
			#Mqp = sim.rp.Mqp[:,0,0,0,0].copy()
			#Mpq = sim.rp.Mpq[:,0,0,0,0].copy()
			Mpp = sim.rp.Mpp[:,0,0,0,0].copy()
			q = sim.rp.q[:,:,0].copy()		
			p = sim.rp.p[:,:,0].copy()
			
			tarr.append(sim.t)
			Marr.append(Mval)
			qarr.append(q/sqrtnbeads)  #Needed to define centroids with correct scaling
			parr.append(p/sqrtnbeads)
		

	def run_TCF(self,sim):
		tarr = []
		qarr = []
		parr = []
		if(self.method == 'CMD'):
			stride = self.gamma	
			dt = self.dt/self.gamma
			nsteps = int(self.time_run/dt)
			for i in range(nsteps):
				sim.step(mode="nvt",var='pq',pc=False)
				if(i%stride == 0):
					q = sim.rp.q[:,:,0].copy()
					p = sim.rp.p[:,:,0].copy()
					tarr.append(sim.t)
					qarr.append(q)
					parr.append(p)	
		else:
			dt = self.dt
			nsteps = int(2*self.time_run/dt) + 1	
			for i in range(nsteps):
				q = sim.rp.q[:,:,0].copy()
				p = sim.rp.p[:,:,0].copy()
				tarr.append(sim.t)
				qarr.append(q)
				parr.append(p)
				sim.step(mode="nve",var='pq')
				

		qarr = np.array(qarr)
		parr = np.array(parr)	
		#NOTE: The correlation function is between vector quantities unless explicitly specified.
		#This needs to be rewritten at some point
		if(self.corrkey=='qq_TCF'):
			tarr,tcf = gen_tcf(qarr,qarr,tarr,corraxis=0)
		elif(self.corrkey=='qq2_TCF'):
			tarr,tcf = gen_tcf(qarr**2,qarr**2,tarr)
		elif(self.corrkey=='pp_TCF'):
			tarr,tcf = gen_tcf(parr,parr,tarr,corraxis=0)
		elif(self.corrkey=='pp2_TCF'):
			tarr,tcf = gen_tcf(parr**2,parr**2,tarr)	
		elif(self.corrkey=='qp_TCF'):
			tarr,tcf = gen_tcf(qarr,parr,tarr,corraxis=0)
		elif(self.corrkey=='pq_TCF'):		
			tarr,tcf = gen_tcf(parr,qarr,tarr,corraxis=0)
		return tarr,tcf	
	
	def run_seed(self,rngSeed,op=None):
		print('Seed {} : T {}, nbeads {}'.format(rngSeed,self.T,self.nbeads))	
		rng = np.random.default_rng(rngSeed)
		ens = Ensemble(beta=1/self.T,ndim=self.dim)
		qcart,pcart = self.gen_ensemble(ens,rng,rngSeed)

		if(self.method=='Classical' or self.method=='RPMD'):
			rp = RingPolymer(qcart=qcart,pcart=pcart,m=self.m,mode='rp')
		elif(self.method=='CMD'):
			rp = RingPolymer(qcart=qcart,pcart=pcart,m=self.m,mode='rp',nmats=1,sgamma=self.gamma)
	
		therm = PILE_L(tau0=self.tau0,pile_lambda=self.pile_lambda) 
		if(self.corrkey=='OTOC'):
			motion = Motion(dt = self.dt,symporder=4)	
			propa = Symplectic_order_IV()
		else:
			motion = Motion(dt=self.dt,symporder=2)
			propa = Symplectic_order_II()
			
		sim = RP_Simulation()
		sim.bind(ens,motion,rng,rp,self.pes,propa,therm)
		
		if(self.corrkey =='OTOC'):
			tarr, Carr = self.run_OTOC(sim)
		elif('TCF' in self.corrkey):
			tarr, Carr = self.run_TCF(sim)
		elif(self.corrkey =='stat_avg'):
			# The assumption here is that 'op' is scalar-valued function (i.e. returns a scalar for every bead)
			avg = np.mean(np.mean(op(sim.rp.qcart,sim.rp.pcart),axis=1))
			self.store_scalar(avg,rngSeed)
			return
		elif(self.corrkey == 'singcomm'):
			tarr, Carr = self.run_OTOC(sim,single=True)
		elif(self.corrkey == 'R2' and self.extparam is not None):
			tarr, Csym, Casym = self.run_R2(sim,self.extparam[0],self.extparam[1],self.extparam[2])
			self.store_time_series_2D(tarr,Csym,rngSeed,'sym')
			self.store_time_series_2D(tarr,Casym,rngSeed,'asym')	
			return
		else:
			return	
	
		self.store_time_series(tarr,Carr,rngSeed)

	def assign_fname(self,rngSeed,suffix=None):
		key = [self.method,self.enskey,self.corrkey,self.sysname,self.potkey,self.Tkey,'N_{}'.format(self.N),'dt_{}'.format(self.dt)]
		fext = '_'.join(key)
		if(self.method=='Classical'):
			methext	= '_'
		elif(self.method=='RPMD'):
			methext = '_nbeads_{}_'.format(self.nbeads)
		elif(self.method=='CMD'):
			methext = '_nbeads_{}_gamma_{}_'.format(self.nbeads,self.gamma)
		
		if(self.corrkey!='stat_avg'):
			seedext = 'seed_{}'.format(rngSeed)
		else:
			seedext = ''

		if(self.ext_kwlist is None and suffix is None):
			fname = ''.join([fext,methext,seedext])	
		elif(self.ext_kwlist is None):
			fname = ''.join([fext,methext,suffix+'_',seedext])	
		else:	
			namelst = [fext,methext,suffix+'_']
			namelst.append('_'.join(self.ext_kwlist) + '_')
			namelst.append(seedext)
			fname = ''.join(namelst)

		return fname

	def store_time_series(self,tarr,Carr,rngSeed): 
		fname = self.assign_fname(rngSeed)	
		store_1D_plotdata(tarr,Carr,fname,'{}/Datafiles'.format(self.pathname))	

	def store_time_series_2D(self,tarr,Carr,rngSeed,suffix=None): 
		fname = self.assign_fname(rngSeed,suffix)	
		store_2D_imagedata(tarr,tarr,Carr,fname,'{}/Datafiles'.format(self.pathname))	

	def store_scalar(self,scalar,rngSeed):
		# Scalar values are stored in the same filename
		fname = self.assign_fname(rngSeed)
		f = open('{}/Datafiles/{}.txt'.format(self.pathname,fname),'a')
		f.write(str(rngSeed) + "  " + str(scalar) + '\n')
		f.close()	
