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
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import pickle
import os

potkey = 'inv_harmonic'
sysname = 'Selene'		
	
lamda = 0.8
g = 1/50.0
times = 1
m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 20.0
time_total = 10.0
Tc = lamda*(0.5/np.pi)#5.0
T = Tc

path = os.path.dirname(os.path.abspath(__file__))

datapath = '{}/Datafiles'.format(path)

nmats = 3
gamma = 16
nbeads = 16

theta = 0.0

tarr = np.linspace(0.0,10.0,1000)
OTOCarr = np.zeros_like(tarr) +0j

if(1):
    keyword1 = 'MFMats_OTOC_Selene'
    keyword2 = 'nmats_{}_'.format(nmats)
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
    store_1D_plotdata(tarr,OTOCarr,'MFMats_OTOC_theta_{}_inv_harmonic_T_{}_N_{}_nmats_{}_nbeads_{}_gamma_{}_dt_{}'.format(theta,T,N,nmats,nbeads,gamma,dt),datapath)

if(1): # Beads
    #fig = plt.figure()
    plt.suptitle(r'Matsubara OTOC at $T=T_c$, $\gamma={}$'.format(gamma))
    for nmats in [3,5]:#[4,8,16,32]:
        #print('i', i)
        data = read_1D_plotdata('{}/MFMats_OTOC_theta_{}_inv_harmonic_T_{}_N_{}_nmats_{}_nbeads_{}_gamma_{}_dt_{}.txt'.format(datapath,theta,T,N,nmats,nbeads,gamma,dt))
        tarr = data[:,0]
        OTOCarr = data[:,1]
        plt.plot(tarr,np.log(abs(OTOCarr)),label=r'$N_b$ = {}'.format(nmats),linewidth=3)#,color='k')

    plt.legend()
    #plt.show()  

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

