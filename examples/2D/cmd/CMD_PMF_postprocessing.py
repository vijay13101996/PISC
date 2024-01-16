import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import scipy
from scipy import interpolate
import CMD_PMF
import os
from PISC.utils.readwrite import read_1D_plotdata


m = 0.5
    
lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

alpha = 0.382
D = 3*Vb

z = 1.0
 
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

nbeads=32

times = 0.9
Tc = lamda*(0.5/np.pi)
T = times*Tc    
Tkey = 'T_{}Tc'.format(times)

N = 1000
dt = 0.01
time_therm = 50.0
time_relax = 10.0
nsample = 5

path = os.path.dirname(os.path.abspath(__file__))
datapath = '{}/Datafiles'.format(path)

if(1):
    for times in [0.7,0.71,0.725,0.75,0.8,0.85,0.9,0.95]:
        keywords = ['CMD_Hess','nbeads_{}'.format(nbeads), 'T_{}Tc'.format(times), potkey]
        flist = []

        for fname in os.listdir(datapath):
            if all(key in fname for key in keywords):
                flist.append(fname)
                #print(fname)

        data = read_1D_plotdata('{}/{}'.format(datapath,flist[0]))

        for f in flist[1:]:
            data+= read_1D_plotdata('{}/{}'.format(datapath,f))
        
        data/=len(flist)
        lamda= np.sqrt(-data[2]/m)

        print('times, hess, lamda', times, data[2],lamda)
        #plt.scatter(data[:,1],data[:,1])
        plt.scatter(times,lamda)
    plt.show()

    exit()


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

    
