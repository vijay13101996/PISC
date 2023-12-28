import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import scipy
from scipy import interpolate
import CMD_PMF
import os
from PISC.utils.readwrite import read_1D_plotdata

lamda = 0.8
g = 1/50.0
times = 0.9
m = 0.5
N = 1000
nbeads = 32
dt = 0.01
rngSeed = 2
time_therm = 50.0
time_relax = 10.0
nsample = 5
    
qgrid = np.linspace(0.0,8.0,11)
potkey = 'inv_harmonic'

path = os.path.dirname(os.path.abspath(__file__))
datapath = '{}/Datafiles'.format(path)

if(1):
    keyword1 = 'CMD_Hess'
    keyword2 = 'nbeads_{}'.format(nbeads)

    flist = []

    #for fname in os.listdir(datapath):
    #    if keyword1 in fname and keyword2 in fname:
    #        flist.append(fname)

    for times in [0.6,0.7,0.8]:#,0.9,1.0]:#,5.0]:
        f = 'CMD_Hess_Selene_inv_harmonic_T_{}Tc_N_{}_nbeads_{}_dt_{}_thermtime_{}_relaxtime_{}_nsample_{}_seed_{}.txt'.format(times,N,nbeads,dt,time_therm,time_relax,nsample,rngSeed)
        data = read_1D_plotdata('{}/{}'.format(datapath,f))
        qgrid = np.real(data[0])
        fgrid = np.real(data[1])

        hess = np.sqrt(-fgrid/m)
        print('T,q,hess', times,'Tc',qgrid,hess)
        #plt.plot(qgrid,fgrid)
  
    exit()
    fgrid/=count
    plt.plot(qgrid,fgrid)
    plt.show()
    store_1D_plotdata(qgrid,fgrid,'CMD_PMF_T_1Tc_nbeads_{}'.format(nbeads),datapath)


if(0):
    fig = plt.figure()
    plt.suptitle(r'CMD PMF at $T=T_c$')
    for i in [4,8,16,32]:
        #print('i', i)
        data = read_1D_plotdata('{}/CMD_PMF_T_1Tc_nbeads_{}.txt'.format(datapath,i))
        qgrid = data[:,0]
        fgrid = data[:,1]
        plt.plot(qgrid,fgrid,label=r'$N_b$ = {}'.format(i))

    plt.plot(qgrid,pes.dpotential(qgrid),label=r'Classical')
    plt.legend()
    plt.show()  

    
