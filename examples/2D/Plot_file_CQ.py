import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.potentials import coupled_quartic
from PISC.utils.nmtrans import FFT
import scipy
import random

dim=2
m=1.0

g1 = 10
g2 = 0.1
pes = coupled_quartic(g1,g2)

potkey = 'CQ_g1_{}_g2_{}'.format(g1,g2)

beta = 5.0
T = 1.0/beta

Tkey = 'beta_{}'.format(beta)

N = 1000
dt_therm = 0.05
dt = 0.002
time_therm = 50.0
time_total = 5.0

path = os.path.dirname(os.path.abspath(__file__))

method = 'RPMD'
syskey = 'Papageno'      
corrkey = 'fd_OTOC'#'qq_TCF
enskey = 'thermal'

tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))   
#path = '/scratch/vgs23/PISC/examples/2D'
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)


fig,ax = plt.subplots()
#Plotting
colors = ['g','b','r','c','m','y','k','orange']


if(1):
    beta_arr = [1.0]#,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
    T_arr = np.array(beta_arr)

    dt = 0.005

    def linear(x,a,b):
        return a*x+b

    def plotter_beta(beads,ax,beta_arr,dt=0.002):
        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(beta_arr))] 
        slope_arr = []
        for c,beta in zip(colors,beta_arr):
            Tkey = 'beta_{}'.format(beta)
            fname= 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,beads,dt)
            ext = rpext+fname
            plot_1D(ax,ext, label=r'$\beta={}$'.format(beta),color=c, log=True,linewidth=1)
            slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.,3.)
            #ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color=c)
            slope_arr.append(slope)
        return slope_arr

    slope_1 = plotter_beta(1,ax,beta_arr)
    slope_16 = plotter_beta(16,ax,beta_arr)
    #slope_16 = plotter_beta(16,ax,beta_arr,dt=0.005)
    #slope_32 = plotter_beta(32,ax,beta_arr)
   
    plt.show()
    exit()

    ax.plot(beta_arr,slope_1,color='r',ls='-')
    ax.plot(beta_arr,slope_16,color='g',ls='-')
    #plt.plot(beta_arr,slope_32,color='b',ls='-')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


if(0):
    for c,beads in zip(colors,[32,64]):
        fname= 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,beads,dt) 
        ext = rpext+fname 
        plot_1D(ax,ext, label=r'$N_b={},beta={}$'.format(beads,beta),color=c, log=True,linewidth=1)
        slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.3,3.3)
        #ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')



if(0):
    
    beta_arr = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
    T_arr = np.array(beta_arr)
    
    nbeads = 1
    slope_arr = []
    for c,beta in zip(colors,beta_arr):
        Tkey = 'beta_{}'.format(beta)
        fname= 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
        ext = rpext+fname
        #plot_1D(ax,ext, label=r'$\beta={}$'.format(beta),color=c, log=True,linewidth=1)
        slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.5,2.5)
        #ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
        slope_arr.append(slope)
    
    slope_arr = np.log(slope_arr)

    plt.plot(T_arr,slope_arr,color='k')
    plt.scatter(T_arr,slope_arr,color='k')
    popt, pcov = scipy.optimize.curve_fit(lambda x,a,b,n: a*x+b, T_arr, slope_arr)
    #plt.plot(T_arr,popt[0]*T_arr+popt[1],color='k',ls='--')
    print('a, nbeads',popt[0],nbeads)

    nbeads = 8
    slope_arr = []
    for c,beta in zip(colors,beta_arr):
        Tkey = 'beta_{}'.format(beta)
        fname= 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
        ext = rpext+fname
        #plot_1D(ax,ext, label=r'$\beta={}$'.format(beta),color=c, log=True,linewidth=1)
        slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.5,3.5)
        #ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
        slope_arr.append(slope)
    
    slope_arr = np.log(slope_arr)

    plt.plot(T_arr,slope_arr,color='r')
    plt.scatter(T_arr,slope_arr,color='r')
    popt, pcov = scipy.optimize.curve_fit(lambda x,a,b,n: a*x+b, T_arr, slope_arr)
    #plt.plot(T_arr,popt[0]*T_arr+popt[1],color='r',ls='--')
    print('a, nbeads',popt[0],nbeads)

    nbeads = 16
    slope_arr = []
    for c,beta in zip(colors,beta_arr):
        Tkey = 'beta_{}'.format(beta)
        fname= 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
        ext = rpext+fname
        #plot_1D(ax,ext, label=r'$\beta={}$'.format(beta),color=c, log=True,linewidth=1)
        slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.5,3.5)
        #ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
        slope_arr.append(slope)
    
    slope_arr = np.log(slope_arr)

    plt.plot(T_arr,slope_arr,color='g')
    plt.scatter(T_arr,slope_arr,color='g')
    popt, pcov = scipy.optimize.curve_fit(lambda x,a,b,n: a*x+b, T_arr, slope_arr)
    #plt.plot(T_arr,popt[0]*T_arr + popt[1],color='g',ls='--')
    print('a, nbeads',popt[0],nbeads)


#plt.legend()
#plt.show()



plt.show()
