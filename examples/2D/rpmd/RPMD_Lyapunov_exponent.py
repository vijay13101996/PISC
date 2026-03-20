import numpy as np
import sys
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.engine.PI_sim_core import SimUniverse
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.integrators import Symplectic
from PISC.engine.beads import RingPolymer
from PISC.utils.nmtrans import FFT
import os

### Potential parameters
m=0.5#0.5
dim=2

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
D = 3*Vb
alpha = 0.382
print('Vb',Vb, 'D', D)

z = 1.0

pes = quartic_bistable(alpha,D,lamda,g,z)

#Only relevant for ring polymer and canonical simulations
Tc = 0.5*lamda/np.pi
times = 0.7
T = times*Tc
beta = 1/T

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)    
Tkey = 'T_{}Tc'.format(times)

print('Bound', 2*np.pi/beta)

#Initialize p and q
E = 1.001*Vb 

path = os.path.dirname(os.path.abspath(__file__))

### ------------------------------------------------------------------------------
def norm_dp_dq(dp,dq,norm=0.001):
    div=(1/norm)*np.linalg.norm(np.concatenate((dp,dq)))
    return dp/div,dq/div

def run_var(sim,dt,time_run,tau,norm):
    tarr=[]
    mLCE=[]
    alpha = []
    qx_cent=[]
    qy_cent=[]
    nsteps = int(tau/dt)
    N=int(time_run/tau)
    for k in range(N):
        #calculate initial w
        dp_cent=sim.rp.dp[...,0]
        dq_cent=sim.rp.dq[...,0]
        prev=np.linalg.norm(np.concatenate((dp_cent,dq_cent)))
        #propagate
        for i in range(nsteps):
            sim.step(mode="nve",var='variation')
        #calculate centroid
        cent_xy=sim.rp.q[...,0]/nbeads**0.5
        qx_cent.append(cent_xy[0,0])
        qy_cent.append(cent_xy[0,1])
        #calculate w after nsteps
        dp_cent=sim.rp.dp[...,0]
        dq_cent=sim.rp.dq[...,0]
        current=np.linalg.norm(np.concatenate((dp_cent,dq_cent)))
        alpha.append(current/prev)
        #update
        sim.rp.dp,sim.rp.dq=norm_dp_dq(sim.rp.dp,sim.rp.dq,norm=norm)
        sim.rp.mats2cart()
        tarr.append(sim.t)
        mLCE.append((1/sim.t)*np.sum(np.log(alpha)))
    return tarr,mLCE,qx_cent,qy_cent

#-------------------------------------------------------------------------------------------
#CONVERGENCE PARAMETERS
norm=0.001#CONVERGENCE PARAMETER#todo:vary
dt=0.005#CONVERGENCE PARAMETER#todo:vary
time_run=40#CONVERGENCE PARAMETER#todo:vary
tau=0.01#CONVERGENCE PARAMETER#todo:vary
nbeads=32#32#32

#dq=np.zeros_like(q)
#dp=np.zeros_like(p)
#dq[:,0,0]=1e-2
#dp[:,0,0]=1e-2

#initialize small deviation
#rng = np.random.default_rng(rngSeed)
#dq=rng.standard_normal(np.shape(q))
#dp=rng.standard_normal(np.shape(q))

#dp,dq=norm_dp_dq(dp,dq,norm)
#print('dq,dp',dp,dq)


rngSeed = 10

lamda_arr  = []
Hess_arr = []
T_arr = np.around(np.arange(0.7,0.951,0.05),2)
#T_arr = np.append(T_arr,[0.97,0.99])

#plt.scatter(qcart[:,0],qcart[:,1])
#plt.scatter(data[:,0],data[:,1])
#plt.show()

#print('Pot Yair', data[:,2].sum()*0.0367493)

if(0):
    for times in [0.9]:#T_arr:
        Tkey='T_{}Tc'.format(times)
        T=times*Tc
        qcart = read_arr('Instanton_{}_{}_nbeads_{}'.format(potkey,Tkey,nbeads),'{}/Datafiles'.format(path)) #instanton 
        qcartarr =  read_arr('Instanton_{}_{}_nbeads_{}'.format(potkey,Tkey,nbeads),'{}/Datafiles'.format(path)) #instanton

        sim=RP_Simulation()
        rng = np.random.default_rng(rngSeed)
        ens = Ensemble(beta=1/T,ndim=dim)
        motion = Motion(dt=dt,symporder=2)#4
        propa = Symplectic()
        rp=RingPolymer(qcart=qcart,m=m)
        rp.bind(ens,motion,rng)
        therm = PILE_L(tau0=1.0,pile_lambda=100.0)#only important due to initalization 
        sim.bind(ens,motion,rng,rp,pes,propa,therm)

        print(pes.potential_xy(qcart[0,0,0],qcart[0,1,0]))
        #print('PES', pes.potential(qcart)[0,0])
        pot = pes.potential(qcart)[0]
        print('pot', pot.sum())
        #store_1D_plotdata(qcart[0,0,:],qcart[0,1,:],'Instanton_{}_{}_nbeads_{}'.format(potkey,Tkey,nbeads) ,'{}/Datafiles'.format(path),ebar=pot)

    print('Done')

for times in T_arr:
    Tkey='T_{}Tc'.format(times)
    T=times*Tc

    qcart = read_arr('Instanton_{}_{}_nbeads_{}'.format(potkey,Tkey,nbeads),'{}/Datafiles'.format(path)) #instanton
    
    fft = FFT(1,nbeads)
    q_nm = fft.cart2mats(qcart)/nbeads**0.5

    #xc_dev = q_nm[0,0,0]
    xc_dev = qcart.sum(axis=2)[0,0]

    #print('xc', xc_dev,q_nm[0,0,0])
        
    pcart=np.zeros_like(qcart)
    pcart[:,0,:] = -xc_dev
    
    rng = np.random.default_rng(200)#rngSeed)
    
    dq=np.zeros_like(qcart)
    dp=np.zeros_like(pcart)
    dq[:,0,0]=-1e-4
    dp[:,0,0]=1e-4

    dp[:,1,0]=1e-4*rng.uniform(-1,1)
    dq[:,1,0]=1e-4*rng.uniform(-1,1)

    #------------------------------------------------------------------------------
    #Set up the simulation
    sim=RP_Simulation()
    ens = Ensemble(beta=1/T,ndim=dim)
    motion = Motion(dt=dt,symporder=2)#4
    propa = Symplectic()#IV
    rp=RingPolymer(pcart=pcart,qcart=qcart,dp=dp,dq=dq,m=m)
    rp.bind(ens,motion,rng)
    therm = PILE_L(tau0=1.0,pile_lambda=100.0)#only important due to initalization 
    sim.bind(ens,motion,rng,rp,pes,propa,therm)

    Hess = (rp.ddpot+pes.ddpot).reshape(len(rp.ddpot)*2*32,len(rp.ddpot)*2*32)
    vals,vecs = np.linalg.eigh(Hess)
    vals = np.sort(vals)
    Hess_arr.append(-vals[0])
    lamda = np.sqrt(-vals[0]/m)
    lamda_arr.append(lamda)
    w1 = 2*np.pi*times*Tc
    #print('Hessval',-1.0*vals[:3],4-w1**2 ,'\n Hess',Hess[:3,:3])  
    
    print('T={}Tc'.format(times), 'lamda', lamda)

    q0 = q_nm[:,:,0]
    rg = np.mean(np.sum((qcart-q0[:,:,None])**2,axis=1),axis=1)
    #print('potential', (rp.pot+pes.pot.sum())/nbeads,rg)
    tarr,mLCE,qx_cent,qy_cent=run_var(sim,dt,time_run,tau,norm)
    fig,ax=plt.subplots(3,sharex=True)

    lmax = max(mLCE)
    print('Temp',times, 'mLCE', lmax) 
    #fname = 'Lambda_max_{}_{}_{}'.format(potkey,Tkey,nbeads)
    #f = open('{}/Datafiles/{}.txt'.format(path,fname),'a')
    #f.write(str(rngSeed) + "  " + str(lmax) + '\n')
    #f.close()

    lamda_arr.append(lmax)

    ax[0].plot(tarr,mLCE)
    ax[1].plot(tarr,qx_cent)
    ax[2].plot(tarr,qy_cent)
    plt.show()

#store_1D_plotdata(T_arr,lamda_arr,'RPMD_Lyapunov_exponent_{}_nbeads_{}_ext_2'.format(potkey,nbeads),'{}/Datafiles'.format(path))

plt.scatter(T_arr,lamda_arr,marker='x',color='r')
#plt.scatter(T_arr,Hess_arr,color='g')
T_ext = np.arange(1.0,10.1,0.5)
lamda_ext = 2.0*np.ones_like(T_ext)
#plt.scatter(T_arr,lamda_arr,marker='x',color='b')

T_arr = np.arange(0.0,10.0,0.01)
#plt.scatter(T_arr,2*np.pi*T_arr*Tc)
plt.show()
