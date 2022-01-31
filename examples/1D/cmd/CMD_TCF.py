import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.harmonic_2D import Harmonic
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.harmonic_1D import harmonic
from PISC.potentials.Quartic import Quartic
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import thermalize_PILE_L
from thermalize_PILE_L import thermalize_rp
import pickle

dim = 1
lamda = 2.0#1.5
g = 1/50.0
Tc = lamda*(0.5/np.pi)#5.0
T = 1.0#Tc
print('T',T)
m = 1.0
N = 1000

nbeads = 4#32 
rng = np.random.RandomState(1)
qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = None
dt = 0.001
beta = 1/T

#for i in range(1,6):
rngSeed = 5
rp = RingPolymer(qcart=qcart,m=m) 
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(rngSeed) 

rp.bind(ens,motion,rng)

potkey = 'quartic'	
pes = Quartic()#double_well(lamda,g)#harmonic(2.0*np.pi)# Harmonic(2*np.pi)#harmonic(2.0)#Harmonic(2*np.pi)#
pes.bind(ens,rp)

time_therm = 20.0
thermalize_rp(ens,rp,pes,time_therm,dt,potkey,rngSeed)

tarr=[]
qarr=[]
potarr=[]
Mqqarr = []
Mqqarrfull = []
Earr = []
dt = 0.01
gamma = 16

dt = dt/gamma

if(1):		
	qcart = read_arr('Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed))
	pcart = read_arr('Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(rp.nsys,rp.nbeads,ens.beta,potkey,rngSeed))
	
	rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,scaling='MFmats',mode='MFmats',nmats=1,sgamma=gamma)
	#rp = RingPolymer(qcart=rp.qcart,m=m,mode='MFmats',nmats=1)		
	motion = Motion(dt = dt,symporder=2) 
	rp.bind(ens,motion,rng)
	pes.bind(ens,rp)

	print('kin',rp.kin.sum(), 0.5*rp.ndim*rp.nsys*rp.nbeads**2/beta,rp.pcart[0],rp.qcart[0] )

	therm = PILE_L(tau0=0.1,pile_lambda=10.0) 
	therm.bind(rp,motion,rng,ens)

	propa = Symplectic_order_II()
	propa.bind(ens, motion, rp, pes, rng, therm)
	
	sim = Simulation()
	sim.bind(ens,motion,rng,rp,pes,propa,therm)

	time_total = 40.0
	nsteps = int(time_total/dt)
	pmats = np.array([True for i in range(rp.nbeads)])
	pmats[:rp.nmats] = False

	start_time = time.time()
	stride = 8	
	for i in range(nsteps):
		sim.step(mode="nvt",var='pq',pmats=pmats)
		if(i%stride==0):
			tarr.append(i*dt)
			qarr.append(rp.q[:,0,0].copy())

	tarr = np.array(tarr)
	qarr = np.array(qarr)
	print('qarr',qarr.shape)	
	if(1):
		q_tilde = np.fft.rfft(qarr,axis=0)
		tcf_cr = np.fft.irfft(np.conj(q_tilde)*q_tilde,axis=0)
			
		tcf_cr = tcf_cr[:len(tcf_cr)//2,:]  #Truncating the padded part
		tcf= np.mean(tcf_cr,axis=1)    #Summing over the particles
		   
		tcf/=(2*len(tcf)*rp.nbeads)
		
		plt.plot(tarr[:len(tcf)],tcf)
		plt.show()
	print('time',time.time()-start_time)	
		
	if(0):
		fname = 'Test_CMD_OTOC_T_{}_N_{}_nbeads_{}_gamma_{}_dt_{}_thermtime_{}_seed_{}'.format(T,N,nbeads,gamma,dt,time_therm,rngSeed)
		store_1D_plotdata(tarr,Mqqarr,fname)
		#plt.plot(tarr,np.log(abs(Mqqarr)**2))
		#plt.plot(tarr,Mqqarr)
		#plt.plot(tarr,np.log(Mqqarr))
		#plt.show()	


