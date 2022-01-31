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
import os


dim = 1
lamda = 0.8
g = 1/50.0
Tc = lamda*(0.5/np.pi)#5.0
T = Tc
print('T',T)
m = 0.5
N = 1000

nbeads = 24
rng = np.random.RandomState(1)
qcart = rng.normal(size=(N,dim,nbeads))#np.ones((N,dim,nbeads))#np.random.normal(size=(N,dim,nbeads))#np.zeros((N,dim,nbeads))#
q = np.random.normal(size=(N,dim,nbeads))
M = np.random.normal(size=(N,dim,nbeads))

pcart = None
dt = 0.005
beta = 1/T

rngSeed = 1
rp = RingPolymer(qcart=qcart,m=m)
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2)
rng = np.random.default_rng(rngSeed)

rp.bind(ens,motion,rng)

potkey = 'inv_harmonic_lambda_{}'.format(lamda)
pes = double_well(lamda,g)#harmonic(2.0*np.pi)# Harmonic(2*np.pi)#harmonic(2.0)#Harmonic(2*np.pi)#
pes.bind(ens,rp)

time_therm = 100.0

dt = 0.005
gamma = 16

#dt = dt/gamma

tarr = np.linspace(0.0,10.0,1000)
OTOCarr = np.zeros_like(tarr) +0j
potkey = 'inv_harmonic'

path = os.path.dirname(os.path.abspath(__file__))
datapath = '{}/Datafiles'.format(path)

if(1):
    keyword1 = 'CMD_OTOC_Selene'
    keyword2 = 'nbeads_{}_'.format(nbeads)
    keyword3 = 'gamma_{}'.format(gamma)
    flist = []

    for fname in os.listdir(datapath):
        if keyword1 in fname and keyword2 in fname and keyword3 in fname:
            #print('fname',fname)
            flist.append(fname)

    count=0

    for f in flist:
        data = read_1D_plotdata('{}/{}'.format(datapath,f))
        tarr = data[:,0]
        OTOCarr += data[:,1]
        #plt.plot(data[:,0],np.log(abs(data[:,1])))
        #print('data',data[:,1])
        count+=1

    print('count',count)
    #print('otoc',OTOCarr)
    OTOCarr/=count
    #plt.plot(tarr,np.log(abs(OTOCarr)))
    #plt.show()
    store_1D_plotdata(tarr,OTOCarr,'CMD_OTOC_T_1Tc_nbeads_{}_gamma_{}_dt_{}'.format(nbeads,gamma,dt),datapath)

if(1): # Beads
    #fig = plt.figure()
    plt.suptitle(r'CMD OTOC at $T=T_c$, $\gamma={}$'.format(gamma))
    for i in [4,8,12,16,24,32]:#[4,8,16,32]:
        #print('i', i)
        data = read_1D_plotdata('{}/CMD_OTOC_T_1Tc_nbeads_{}_gamma_{}_dt_{}.txt'.format(datapath,i,gamma,dt))
        tarr = data[:,0]
        OTOCarr = data[:,1]
        plt.plot(tarr,np.log(abs(OTOCarr)),label=r'$N_b$ = {}'.format(i),linewidth=3)#,color='k')

    plt.legend()
    #plt.show()  

if(0): # gamma
    #fig = plt.figure()
    plt.suptitle(r'CMD OTOC at $T=T_c$, $N_b={}$'.format(nbeads))
    for i in [16,32]:
        data = read_1D_plotdata('{}/CMD_OTOC_T_1Tc_nbeads_{}_gamma_{}_dt_{}.txt'.format(datapath,nbeads,i,dt))
        tarr = data[:,0]
        OTOCarr = data[:,1]
        plt.plot(tarr,np.log(abs(OTOCarr)),label=r'$\gamma$ = {}'.format(i),linewidth=3)

    plt.legend()


if(0): #slope
    plt.suptitle(r'Classical, CMD and Quantum OTOCs at $T=T_c$')
    plt.title('$\gamma=32$ for CMD')

    for nb,ind_zero,c in zip([1,4,8,16,32],[169,214,227,200,185],['r','g','b','c','m']):
        data = read_1D_plotdata('{}/CMD_OTOC_T_1Tc_nbeads_{}_gamma_{}_dt_{}.txt'.format(datapath,nb,16,dt))
        tarr = data[:,0]
        OTOCarr = data[:,1]

        #print('zero',np.where(abs(np.log(OTOCarr)-0.0)<1e-2),np.log(OTOCarr)[185],tarr[185])
        #ind_zero = 185  #169 for 1, 214 for 4, 227 for 8, 200 for 16, 185 for 32
        ist = ind_zero+10
        iend = ist+100

        t_trunc = tarr[ist:iend]
        OTOC_trunc = (np.log(OTOCarr))[ist:iend]
        slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
        print('slope',slope)

        #a = -OTOCarr
        #x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])

        plt.plot(tarr,np.log(abs(OTOCarr)),label=r'$N_b = {}$'.format(nb),linewidth=2,color=c)
        plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color=c)
        #plt.plot(tarr,slope*tarr+ic,'--',color=c)


if(1): # Beads
    #fig = plt.figure()
    plt.suptitle(r'CMD OTOC at $T=T_c$, $\gamma={}$'.format(gamma))
    for i in [4,8,12,16,24,32]:#[4,8,16,32]:
        #print('i', i)
        data = read_1D_plotdata('{}/CMD_OTOC_T_1Tc_nbeads_{}_gamma_{}_dt_{}.txt'.format(datapath,i,gamma,dt))
        tarr = data[:,0]
        OTOCarr = data[:,1]
        plt.plot(tarr,np.log(abs(OTOCarr)),label=r'$N_b$ = {}'.format(i),linewidth=3)#,color='k')

    plt.legend()
    #plt.show()  

if(0): # gamma
    #fig = plt.figure()
    plt.suptitle(r'CMD OTOC at $T=T_c$, $N_b={}$'.format(nbeads))
    for i in [16,32]:
        data = read_1D_plotdata('{}/CMD_OTOC_T_1Tc_nbeads_{}_gamma_{}_dt_{}.txt'.format(datapath,nbeads,i,dt))
        tarr = data[:,0]
        OTOCarr = data[:,1]
        plt.plot(tarr,np.log(abs(OTOCarr)),label=r'$\gamma$ = {}'.format(i),linewidth=3)

    plt.legend()


if(0): #slope
    plt.suptitle(r'Classical, CMD and Quantum OTOCs at $T=T_c$')
    plt.title('$\gamma=32$ for CMD')

    for nb,ind_zero,c in zip([1,4,8,16,32],[169,214,227,200,185],['r','g','b','c','m']):
        data = read_1D_plotdata('{}/CMD_OTOC_T_1Tc_nbeads_{}_gamma_{}_dt_{}.txt'.format(datapath,nb,16,dt))
        tarr = data[:,0]
        OTOCarr = data[:,1]

        #print('zero',np.where(abs(np.log(OTOCarr)-0.0)<1e-2),np.log(OTOCarr)[185],tarr[185])
        #ind_zero = 185  #169 for 1, 214 for 4, 227 for 8, 200 for 16, 185 for 32
        ist = ind_zero+10
        iend = ist+100

        t_trunc = tarr[ist:iend]
        OTOC_trunc = (np.log(OTOCarr))[ist:iend]
        slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
        print('slope',slope)

        #a = -OTOCarr
        #x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])

        plt.plot(tarr,np.log(abs(OTOCarr)),label=r'$N_b = {}$'.format(nb),linewidth=2,color=c)
        plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color=c)
        #plt.plot(tarr,slope*tarr+ic,'--',color=c)

if(1): # Quantum  
        tfinal = 10.0
        #f =  open("/home/vgs23/Pickle_files/OTOC_{}_beta_0.2_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',50,50,4.0),'rb+')
        f =  open("{}/OTOC_{}_lambda_{}_1Tc_basis_{}_n_eigen_{}_tfinal_{}.dat".format(datapath,'inv_harmonic',lamda,50,50,tfinal),'rb+')
        t_arr = pickle.load(f,encoding='latin1')
        OTOC_arr = pickle.load(f,encoding='latin1')

        ind_zero = 124
        ind_max = 281
        ist = 160
        iend = 180

        t_trunc = t_arr[ist:iend]
        OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
        slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
        print('slope',slope)

        a = -OTOC_arr
        x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
        #print('min max',t_arr[124],t_arr[281])

        plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
        plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
        plt.plot(t_arr,slope*t_arr+ic,'--',color='k')
        f.close()
plt.legend()
plt.show()

