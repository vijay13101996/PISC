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
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import thermalize_PILE_L
from thermalize_PILE_L import thermalize_rp
import pickle
import h5py

def main(filename,pathname,sysname,potkey,nrun,lamda,g,times,m,N,nbeads,dt_therm,dt,rngSeed,time_therm,gamma,time_total):
	dim = 1
	Tc = lamda*(0.5/np.pi)#5.0
	T = times*Tc
	print('T, nbeads',T,nbeads)
	
	nbeads = nbeads
	rng = np.random.RandomState(1)
	qcart = rng.normal(size=(N,dim,nbeads))
	q = np.random.normal(size=(N,dim,nbeads))
	M = np.random.normal(size=(N,dim,nbeads))

	pcart = None
	dt = dt_therm
	beta = 1/T
	
	rngSeed = rngSeed
	rp = RingPolymer(qcart=qcart,m=m) 
	ens = Ensemble(beta=beta,ndim=dim)
	motion = Motion(dt = dt,symporder=2) 
	rng = np.random.default_rng(rngSeed) 

	rp.bind(ens,motion,rng)

	pes = double_well(lamda,g)
	pes.bind(ens,rp)

	time_therm = time_therm
	thermalize_rp(ens,rp,pes,time_therm,dt,potkey,rngSeed)
	
	tarr=[]
	qarr=[]
	potarr=[]
	Mqqarr = []
	Mqqarrfull = []
	Earr = []
	dt = dt
	gamma = gamma

	dt = dt/gamma
		
	qcart = read_arr('Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"./examples/cmd/Datafiles")
	pcart = read_arr('Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed),"./examples/cmd/Datafiles")
	
	rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,scaling='MFmats',mode='MFmats',nmats=1,sgamma=gamma)
	motion = Motion(dt = dt,symporder=4) 
	rp.bind(ens,motion,rng)
	pes.bind(ens,rp)

	therm = PILE_L(tau0=0.1,pile_lambda=1000.0) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_IV()
	propa.bind(ens, motion, rp, pes, rng, therm)
	
	sim = RP_Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)

	time_total = time_total
	nsteps = int(time_total/dt)	

	start_time = time.time()
	stride = gamma	
	for i in range(nsteps):
		sim.step(mode="nvt",var='monodromy',pc=False)
		if(i%stride == 0):
			Mqq = np.mean(abs(rp.Mqq[:,0,0,0,0]**2)) #rp.Mqq[0,0,0,0,0]#
			tarr.append(sim.t)
			#qarr.append(rp.q[:,0,0].copy())
			#Mqqarrfull.append(rp.Mqq[:,0,0,0,0].copy())
			#potarr.append(pes.ddpot[0,0,0,0,0])
			#Mqqarr.append(propa.rp.Mqq[0,0,0,0,0])
			Mqqarr.append(Mqq)
			#Earr.append(np.sum(pes.pot)+np.sum(rp.pot)+rp.kin)

	if(0):
		with h5py.File(filename, 'a') as f:
			group = f['Run#{}'.format(nrun)]
			group.attrs['lambda'] = lamda
			group.attrs['g'] = g
			group.attrs['T'] = T
			group.attrs['xTc'] = times
			group.attrs['m'] = m
			group.attrs['N'] = N
			group.attrs['nbeads'] = nbeads
			group.attrs['dt_therm'] = dt	
			group.attrs['therm_time'] = time_therm
			group.attrs['total_OTOC_time'] = time_total
			group.attrs['gamma'] = gamma
			group.attrs['seed'] = rngSeed
			group.create_dataset('tarr',data=tarr)
			group.create_dataset('Mqqarr',data=Mqqarr)

	fname = 'CMD_OTOC_{}_inv_harmonic_run_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_seed_{}'.format(sysname,nrun,T,N,nbeads,gamma,dt,rngSeed)
	store_1D_plotdata(tarr,Mqqarr,fname,'{}/Datafiles'.format(pathname))
	
if(0):
	if __name__ == '__main__':
		import argparse

		parser = argparse.ArgumentParser(description='Adiabatic CMD OTOC simulation for the double well potential')
		parser.add_argument('--seed',type=int, required=True)
		parser.add_argument('--beads',type=int, required=True)
		parser.add_argument('--gamma',type=int, required=True)
		args = parser.parse_args()
		main(seed=args.seed, beads=args.beads, gamma=args.gamma)

if(0):	
	f =  open("/home/vgs23/Pickle_files/OTOC_{}_beta_0.2_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',50,50,4.0),'rb+')
	t_arr = pickle.load(f,encoding='latin1')
	OTOC_arr = pickle.load(f,encoding='latin1')
	plt.plot(t_arr,np.log(OTOC_arr), linewidth=1,label='Quantum OTOC')
	f.close()
	plt.show()

	qarr = np.array(qarr)
	Mqqarrfull = np.array(Mqqarrfull)
	count=0
	for i in range(N):
		if(qarr[0,i]*qarr[len(qarr)-1,i]<0.0):
			count+=1
			plt.plot(tarr,qarr[:,i],color='g')
			#plt.plot(tarr,(Mqqarrfull[:,i]),color='r')
			#plt.show()
	print('Count',count)
	
	#print('qarr',qarr.shape)
	#Mqqarr = np.array(Mqqarr)
	print('time',time.time()-start_time)	
			
if(0):
	propa = Runge_Kutta_order_VIII()
	propa.bind(ens, motion, rp, pes, rng, therm)

	tarr = np.linspace(0,4,10000)
	sol = propa.integrate(tarr)
	Mqqcent = np.array(propa.centroid_Mqq(sol))
	Mqqcent = np.mean(abs(Mqqcent**2),axis=1)
		
	qcent = np.array(propa.ret_q(sol))
	print('q',qcent.shape)
	#qcent = qcent[:,30,0]

	#plt.plot(tarr,np.log(abs(Mqqcent[:,0,0]**2)))
	plt.plot(tarr,np.log(Mqqcent[:,0,0]),color='r')
	#for i in range(N):
	#	if (qcent[0,i,0]*qcent[9999,i,0] < 0.0):
	#		print('i',i)
			#plt.plot(tarr,Mqqcent[:,i,0])
	#plt.show()

if(0):
	fname = '/home/vgs23/Pickle_files/CMD_OTOC_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_thermtime_{}_seed_{}.txt'.format(T,1000,32,16,0.000625,100.0,6)
	data=read_1D_plotdata(fname)
	x=data[:,0]
	y=data[:,1]
	plt.plot(x,np.log(y),color='r')

	fname = '/home/vgs23/Pickle_files/CMD_OTOC_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_thermtime_{}_seed_{}.txt'.format(T,10000,16,16,0.003125,100.0,6)
	data=read_1D_plotdata(fname)
	x=data[:,0]
	y=data[:,1]
	plt.plot(x,np.log(y),color='b')

	plt.show()

if(0):
	for i in range(1,6):
		fname = '/home/vgs23/Pickle_files/CMD_OTOC_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_thermtime_{}_seed_{}.txt'.format(T,N,nbeads,gamma,dt,50.0,i)
		data=read_1D_plotdata(fname)
		x=data[:,0]
		y=data[:,1]
		plt.plot(x,np.log(y))

	plt.show()	
