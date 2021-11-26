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

def main(seed,beads,gamma):
	dim = 1
	lamda = 0.8#1.5
	g = 1/50.0
	Tc = lamda*(0.5/np.pi)#5.0
	T = 1*Tc
	print('T',T)
	print('nbeads', beads)
	m = 0.5
	N = 1000

	nbeads = beads#32 
	rng = np.random.RandomState(1)
	qcart = rng.normal(size=(N,dim,nbeads))
	q = np.random.normal(size=(N,dim,nbeads))
	M = np.random.normal(size=(N,dim,nbeads))

	pcart = None
	dt = 0.01
	beta = 1/T
	
	rngSeed = seed
	rp = RingPolymer(qcart=qcart,m=m) 
	ens = Ensemble(beta=beta,ndim=dim)
	motion = Motion(dt = dt,symporder=2) 
	rng = np.random.default_rng(rngSeed) 

	rp.bind(ens,motion,rng)

	potkey = 'inv_harmonic_lambda_{}'.format(lamda)	
	pes = double_well(lamda,g)#harmonic(2.0*np.pi)# Harmonic(2*np.pi)#harmonic(2.0)#Harmonic(2*np.pi)#
	pes.bind(ens,rp)

	time_therm = 20.0
	thermalize_rp(ens,rp,pes,time_therm,dt,potkey,rngSeed)
	
	tarr=[]
	qarr=[]
	potarr=[]
	Mqqarr = []
	Mqqarrfull = []
	Earr = []
	dt = 0.001
	gamma = gamma

	dt = dt/gamma
		
	qcart = read_arr('Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed))
	pcart = read_arr('Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed))
	
	rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,scaling='MFmats',mode='MFmats',nmats=1,sgamma=gamma)
	#rp = RingPolymer(qcart=rp.qcart,m=m,mode='MFmats',nmats=1)		
	motion = Motion(dt = dt,symporder=4) 
	rp.bind(ens,motion,rng)
	pes.bind(ens,rp)

	print('kin',rp.kin.sum(), 0.5*rp.ndim*rp.nsys*rp.nbeads**2/beta,rp.pcart[0],rp.qcart[0] )

	therm = PILE_L(tau0=0.1,pile_lambda=1000.0) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_IV()
	propa.bind(ens, motion, rp, pes, rng, therm)
	
	sim = Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)

	time_total = 10.0
	nsteps = int(time_total/dt)
	pmats = np.array([True for i in range(rp.nbeads)])
	pmats[:rp.nmats] = False

	start_time = time.time()
	stride = gamma	
	for i in range(nsteps):
		sim.step(mode="nvt",var='monodromy',pmats=pmats)
		if(i%stride == 0):
			Mqq = np.mean(abs(rp.Mqq[:,0,0,0,0]**2)) #rp.Mqq[0,0,0,0,0]#
			tarr.append(sim.t)
			#qarr.append(rp.q[:,0,0].copy())
			#Mqqarrfull.append(rp.Mqq[:,0,0,0,0].copy())
			#potarr.append(pes.ddpot[0,0,0,0,0])
			#Mqqarr.append(propa.rp.Mqq[0,0,0,0,0])
			Mqqarr.append(Mqq)
			#Earr.append(np.sum(pes.pot)+np.sum(rp.pot)+rp.kin)

	if(1):
		fname = 'CMD_OTOC_{}_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_thermtime_{}_seed_{}'.format(potkey,T,N,nbeads,gamma,dt,time_therm,rngSeed)
		store_1D_plotdata(tarr,Mqqarr,fname)

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
