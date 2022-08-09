import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.utils.readwrite import store_1D_plotdata, read_arr
from PISC.utils.tcf_fft import gen_tcf
from PISC.engine.thermalize_PILE_L import thermalize_rp
from PISC.engine.gen_mc_ensemble import generate_rp

class SimUniverse(object):
	def __init__(self,method,pathname,sysname,potkey,corrkey,enskey,Tkey):
		self.method = method
		self.pathname = pathname
		self.sysname = sysname
		self.potkey = potkey
		self.corrkey = corrkey
		self.enskey = enskey
		self.Tkey = Tkey
	
	def set_sysparams(self,pes,T,m,dim):
		self.pes = pes
		self.T = T
		self.m = m
		self.dim = dim 
		
	def set_simparams(self,N,dt_ens=1e-2,dt=5e-3):	
		self.N = N
		self.dt_ens = dt_ens
		self.dt = dt

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
		self.pile_lambda = pile_lambda
		self.E = E
		self.qlist = qlist
		self.plist = plist
		self.filt_func = filt_func
	
	def gen_ensemble(self,ens,rng,rngSeed):
		if(self.enskey== 'thermal'):
			thermalize_rp(self.pathname,self.m,self.dim,self.N,self.nbeads,ens,self.pes,rng,self.time_ens,self.dt_ens,self.potkey,rngSeed,self.qlist)	
			qcart = read_arr('Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,ens.beta,self.potkey,rngSeed),"{}/Datafiles".format(self.pathname))
			pcart = read_arr('Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,ens.beta,self.potkey,rngSeed),"{}/Datafiles".format(self.pathname))
			return qcart,pcart	
		elif(self.enskey=='mc'):
			generate_rp(self.pathname,self.m,self.dim,self.N,self.nbeads,ens,self.pes,rng,self.time_ens,self.dt_ens,self.potkey,rngSeed,self.E,self.qlist,self.plist,self.filt_func)
			qcart = read_arr('Microcanonical_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,ens.beta,self.potkey,rngSeed),"{}/Datafiles".format(self.pathname))
			pcart = read_arr('Microcanonical_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,ens.beta,self.potkey,rngSeed),"{}/Datafiles".format(self.pathname)) 
			return qcart,pcart	

	def run_OTOC(self,sim):
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
			for i in range(nsteps):
				sim.step(mode="nve",var='monodromy',pc=False)	
				Mqq = np.mean(abs(sim.rp.Mqq[:,0,0,0,0]**2)) 
				tarr.append(sim.t)
				Mqqarr.append(Mqq)
		return tarr, Mqqarr

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
					p = sim.rp.q[:,:,0].copy()
					tarr.append(sim.t)
					qarr.append(q)
					parr.append(p)	
		else:
			dt = self.dt
			nsteps = int(2*self.time_run/dt)	
			for i in range(nsteps):
				sim.step(mode="nve",var='pq',pc=False)
				q = sim.rp.q[:,:,0]
				p = sim.rp.p[:,:,0]
				tarr.append(sim.t)
				qarr.append(q)
				parr.append(p)
	
		if(self.corrkey=='qq_TCF'):
			tarr,tcf = gen_tcf(qarr,qarr,tarr)
		elif(self.corrkey=='pp_TCF'):
			tarr,tcf = gen_tcf(parr,parr,tarr)
		elif(self.corrkey=='qp_TCF'):
			tarr,tcf = gen_tcf(qarr,parr,tarr)
		elif(self.corrkey=='pq_TCF'):		
			tarr,tcf = gen_tcf(parr,qarr,tarr)
		
		return tarr,tcf
		
	def run_seed(self,rngSeed):
		print('T, nbeads',self.T,self.nbeads)	
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
		else:
			return	
	
		self.store_data(tarr,Carr,rngSeed)

	def store_data(self,tarr,Carr,rngSeed,ext_kwlist=None): 
		key = [self.method,self.enskey,self.corrkey,self.sysname,self.potkey,self.Tkey,'N_{}'.format(self.N),'dt_{}'.format(self.dt)]
		fext = '_'.join(key)
		if(self.method=='Classical'):
			methext	= '_'
		elif(self.method=='RPMD'):
			methext = '_nbeads_{}_'.format(self.nbeads)
		elif(self.method=='CMD'):
			methext = '_nbeads_{}_gamma_{}_'.format(self.nbeads,self.gamma)
		seedext = 'seed_{}'.format(rngSeed)

		if(ext_kwlist is None):
			fname = ''.join([fext,methext,seedext])	
		
		store_1D_plotdata(tarr,Carr,fname,'{}/Datafiles'.format(self.pathname))		
