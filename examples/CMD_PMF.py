import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.harmonic_2D import Harmonic
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.harmonic_1D import harmonic
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import thermalize_PILE_L
from thermalize_PILE_L import thermalize_rp
import pickle
import scipy
from scipy import interpolate
import h5py

def main(filename,nrun,lamda,g,times,m,N,nbeads,dt,rngSeed,time_therm,time_relax,qgrid,nsample):
	dim = 1
	Tc = lamda*(0.5/np.pi)#5.0
	T = times*Tc
	
	rng = np.random.RandomState(1)
	qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
	q = np.random.normal(size=(N,dim,nbeads))
	M = np.random.normal(size=(N,dim,nbeads))

	pcart = None
	beta = 1/T

	rp = RingPolymer(qcart=qcart,m=m,nmats=nbeads) 
	ens = Ensemble(beta=beta,ndim=dim)
	motion = Motion(dt = dt,symporder=2) 
	rng = np.random.default_rng(rngSeed) 

	pes = double_well(lamda,g)
	
	print('T',T)
	nsteps_therm = int(time_therm/dt)
	nsteps_relax = int(time_relax/dt)

	pmats = np.array([True for i in range(rp.nbeads)])
	pmats[0] = False

	fgrid = np.zeros_like(qgrid)

	therm = PILE_L(tau0=0.1,pile_lambda=100.0) 
	
	propa = Symplectic_order_II()
		
	sim = Simulation()
	
	start_time = time.time()
	for k in range(len(qgrid)):
		q = np.zeros((N,dim,nbeads))
		q[...,0] = qgrid[k]
		p = rng.normal(size=q.shape)
		p[...,0] = 0.0	
		rp = RingPolymer(q=q,p=p,m=m,nmats=1,sgamma=1)
		sim.bind(ens,motion,rng,rp,pes,propa,therm)	
		sim.step(ndt=nsteps_therm,mode="nvt",var='pq',RSP=True,pmats=pmats)
		print('k, q[k]',k, rp.q[0,0,0])
		for i in range(nsample):
			sim.step(ndt=nsteps_relax,mode="nvt",var='pq',RSP=True,pmats=pmats)
			fgrid[k]+=np.mean(pes.dpot[...,0])	
		print('time',time.time()-start_time)

	fgrid/=nsample

	with h5py.File(filename, 'a') as f:
		group = f['Run#{}'.format(nrun)]
		group.attrs['lambda'] = lamda
		group.attrs['g'] = g
		group.attrs['T'] = T
		group.attrs['xTc'] = times
		group.attrs['m'] = m
		group.attrs['N'] = N
		group.attrs['nbeads'] = nbeads
		group.attrs['dt'] = dt	
		group.attrs['therm_time'] = time_therm
		group.attrs['relax_time'] = time_relax
		group.attrs['nsample'] = nsample
		group.attrs['seed'] = rngSeed
		group.attrs['pile_lamda'] = 100.0
		qg = group.create_dataset('qgrid',data=qgrid)
		qg.attrs['extent'] = qgrid[len(qgrid)-1]
		qg.attrs['ngrid'] = len(qgrid)	
		group.create_dataset('fgrid',data=fgrid)

	fname = 'CMD_PMF_inv_harmonic_run_{}_T_{}_N_{}_nbeads_{}_dt_{}_thermtime_{}_relaxtime_{}_nsample_{}_seed_{}'.format(nrun,T,N,nbeads,dt,time_therm,time_relax,nsample,rngSeed)
	store_1D_plotdata(qgrid,fgrid,fname,"./examples/Datafiles")
			
###-----------------Stray code-----------------------------------------------------

#This code needs to be parallelized so that it can be run on clusters. 

def integrate(qgrid,fgrid):
	potgrid = np.zeros_like(fgrid)
	for i in range(1,len(qgrid)):
		potgrid[i] = potgrid[i-1] + fgrid[i]*(qgrid[1]-qgrid[0])
	return potgrid


if(0):
	for nbeads in [1,4,8,16,32]:
		fname = '/home/vgs23/Pickle_files/CMD_PMF_{}_T_{}_N_{}_nbeads_{}_dt_{}_thermtime_{}_relaxtime_{}_nsample_{}_seed_{}.txt'.format(potkey,T,N,nbeads,dt,time_therm,time_relax,nsample,rngSeed)
		data = read_1D_plotdata(fname)
					
		qgrid = np.real(data[:,0])
		fgrid = np.real(data[:,1])

		#qgrid = qgrid[::3]
		#fgrid = fgrid[::3]

		f=interpolate.interp1d(qgrid,fgrid,kind='cubic')		

		pot = integrate(qgrid,fgrid)
		plt.plot(qgrid,fgrid,label=nbeads)
		#plt.plot(qgrid,f(qgrid),label=nbeads)
		#plt.plot(qgrid[15:85],pot-pot[len(pot)//2],label=nbeads)
	
	plt.legend()
	#plt.plot(qgrid,pes.ret_force(qgrid))
	#plt.plot(qgrid,pes.ret_pot(qgrid)-pes.ret_pot(0.0))
	plt.show()
	
if(0):		
	tarr = []
	qarr = []
	if(0):
		for i in range(nsteps_therm):
			sim.step(ndt=1,mode="nve",RSP=True,var='pq')	
			#tarr.append(i*dt):
			#qarr.append(rp.q[0,0,3])
			print('qp rsp', rp.q[0,0,1],rp.p[0,0,1])
		
		#plt.plot(tarr,qarr)
		#plt.show()
