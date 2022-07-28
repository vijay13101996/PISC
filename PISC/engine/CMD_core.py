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

def main(pathname,sysname,potkey,pes,Tkey,T,m,dim,N,nbeads,dt_therm,dt,rngSeed,time_therm,gamma,time_total,corrkey,tau0=0.1,pile_lambda=1000.0):
	print('T, nbeads',T,nbeads)
	beta = 1/T
	
	rng = np.random.default_rng(rngSeed)
	ens = Ensemble(beta=beta,ndim=dim)
		
	thermalize_rp(pathname,m,dim,N,nbeads,ens,pes,rng,time_therm,dt_therm,potkey,rngSeed)
	
	dtg = dt/gamma
		
	qcart = read_arr('Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(N,nbeads,beta,potkey,rngSeed),"{}/Datafiles".format(pathname))
	pcart = read_arr('Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(N,nbeads,beta,potkey,rngSeed),"{}/Datafiles".format(pathname))
	
	rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,mode='rp',nmats=1,sgamma=gamma)
	motion = Motion(dt = dtg,symporder=4) 
	rp.bind(ens,motion,rng)
	pes.bind(ens,rp)

	therm = PILE_L(tau0=tau0,pile_lambda=pile_lambda) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_IV()
	propa.bind(ens, motion, rp, pes, rng, therm)
	
	sim = RP_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)

	stride = gamma

	tarr=[]
	qarr=[]
	Mqqarr=[]
	parr=[]
	
	if(corrkey=='OTOC'):
		time_total = time_total
		nsteps = int(time_total/dtg)
		for i in range(nsteps):
			sim.step(mode="nvt",var='monodromy',pc=False)
			if(i%stride == 0):
				Mqq = np.mean(abs(rp.Mqq[:,0,0,0,0]**2)) 
				tarr.append(sim.t)
				Mqqarr.append(Mqq)
	else:
		time_total = 2*time_total
		nsteps = int(time_total/dtg)	
		for i in range(nsteps):
			sim.step(mode="nvt",var='pq',pc=False)
			if(i%stride == 0):
				q = rp.q[:,:,0].copy()
				p = rp.q[:,:,0].copy()
				tarr.append(sim.t)
				qarr.append(q)
				parr.append(p)
	
	if(corrkey=='qq'):
		tarr,tcf = gen_tcf(qarr,qarr,tarr)
	elif(corrkey=='pp'):
		tarr,tcf = gen_tcf(parr,parr,tarr)
	elif(corrkey=='qp'):
		tarr,tcf = gen_tcf(qarr,parr,tarr)
	elif(corrkey=='pq'):		
		tarr,tcf = gen_tcf(parr,qarr,tarr)

	if(corrkey=='OTOC'):
		fname = 'CMD_OTOC_{}_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_seed_{}'.format(sysname,potkey,Tkey,N,nbeads,gamma,dt,rngSeed)
		store_1D_plotdata(tarr,Mqqarr,fname,'{}/Datafiles'.format(pathname))
	else:
		fname = 'CMD_{}_TCF_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_seed_{}'.format(sysname,corrkey,potkey,Tkey,N,nbeads,gamma,dt,rngSeed)
		store_1D_plotdata(tarr,tcf,fname,'{}/Datafiles'.format(pathname))
		
	


