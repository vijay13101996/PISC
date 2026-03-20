import numpy as np
from matplotlib import pyplot as plt
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,estimate_OTOC_slope,seed_collector,seed_finder
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from PISC.utils.readwrite import store_1D_plotdata, read_arr
import os
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False


fig, ax = plt.subplots()
m=0.5
lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

alpha = 0.382
D = 3*Vb

w = 2.0

dt = 0.005

path = '/scratch/vgs23/PISC/examples/2D/'
path = os.path.dirname(os.path.abspath(__file__))   
rpext = '{}/rpmd/Datafiles/'.format(path)
nbeads = 16

def plot_and_slope(z,enskey,pot,times,c='k',ls='-',nbeads=1,label=None,lb=2.4,ub=3.2):
    Tkey = 'T_{}Tc'.format(times)
    if pot=='dw_qb':# Double well
        potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
    elif pot=='dw_harm':# Double well with harmonic coupling
        #potkey = 'DW_harm_2D_T_{}Tc_m_{}_w_{}_lamda_{}_g_{}_z_{}'.format(times,m,w,lamda,g,z)
        #potkey = 'DW_harm_2D_m_{}_w_{}_lamda_{}_g_{}_z_{}'.format(m,w,lamda,g,z)
        potkey = 'DW_Morse_harm_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
    
    ext = rpext + 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,'fd_OTOC',potkey,Tkey,nbeads,dt)
    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,lb,ub)
    data = np.loadtxt(ext+'.txt',dtype=complex)
    if(label is not None):
        ax.plot(data[:,0],np.log(data[:,1]),label=label,lw=1,ls=ls,color=c)
    else:
        ax.plot(data[:,0],np.log(data[:,1]),lw=1,ls=ls)

    ax.plot(t_trunc, slope*t_trunc+ic,color='k',lw=1.5)
    return slope

def statavg(z,enskey,pot,times,hesstype ='centroid',c='k',ls='-',nbeads=1,label=None):
    N=1
    Tkey = 'T_{}Tc'.format(times)
    if hesstype=='centroid':
        hesskey = 'centroid_Hessian'
    elif hesstype=='full':
        hesskey = 'Hessian'
    if pot=='dw_qb':# Double well
        potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
    elif pot=='dw_harm':# Double well with harmonic coupling
        potkey = 'DW_Morse_harm_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

    ext = rpext + 'RPMD_{}_stat_avg_Papageno_{}_{}_N_{}_dt_{}_nbeads_{}_{}'.format(enskey,potkey,Tkey,N,dt,nbeads,hesskey)
    data = np.loadtxt(ext+'.txt')
    lamda2 = data[:,1]/m
    lamdaavg = lamda2 #np.sqrt(lamda2)
    ax.hist(lamdaavg,bins=100,range=(-4,4),density=True,label=label,alpha=0.65,color=c)
    avg = np.mean(lamdaavg)
    return avg


colors = ['r','b','g','m','c','y','k','brown','tan','orange']
z_arr = [2.0]#,2.0] #0.5,1.0,1.5,2.0]


if(0):
    T_arr = [0.8,0.85,0.95,1.0,1.2,1.4,1.8,2.2,2.6,3.0]
    slopearr16 = []
    for times,c in zip(T_arr,colors):
        if(times>2.0):
            lb = 2.4
            ub = 3.2
        else:
            lb = 3.5
            ub = 4.3
        slopet = plot_and_slope(2.0,'thermal','dw_harm',times,label='z=2.0, thermal',c=c,ls='--',nbeads=16,lb=lb,ub=ub)
        slopearr16.append(slopet)
        #slopet32 = plot_and_slope(2.0,'thermal','dw_harm',times,label='z=2.0, thermal',c=c,ls='-',nbeads=32,lb=3.5,ub=4.3)
        print('z=2.0, T={}, slopet={}'.format(times,np.around(slopet,2)))

    plt.show()
    fname = 'DW_Morse_harm_lamda_vs_T_nbeads_16'
    store_1D_plotdata(T_arr, slopearr16, fname, '/home/vgs23/PISC/manu_mod')

    plt.plot(T_arr,slopearr16,'o-',label='z=2.0, nbeads=16',c='k')

if(1):    
    Cavgarr = []
    Ravgarr = []
    Clavgarr = []
    for z,c in zip(z_arr,colors):#,1.5,2.0,2.5,3.0]:
        if z==0.0:
            ls = '-'
        else:
            ls = '--'
        #slopec = plot_and_slope(z,'const_qp','dw_harm',3.0,label='z={}, const_q'.format(z),c=c,ls=ls,lb=2.4,ub=3.)
        #slopet = plot_and_slope(z,'thermal','dw_harm',3.0,label='z={}, thermal'.format(z),c=c,ls=ls,lb=2.4,ub=3.)
        #print('z={}, slopec={}'.format(z,np.around(slopec,2)))
        #print('z={}, slopec={}, slopet={}'.format(z,np.around(slopec,2),np.around(slopet,2)))
        
        #slopec = plot_and_slope(z,'const_q','dw_qb',0.95,c='k',ls='-',nbeads=16,lb=3.05,ub=4.1)
        
        #slopec = plot_and_slope(z,'const_q','dw_harm',0.95,c=c,ls='--',nbeads=16,lb=3.2,ub=4.2,label='z={}, const_q'.format(z))
        #slopet = plot_and_slope(z,'thermal','dw_harm',0.95,c=c,ls='-',nbeads=32,lb=3.2,ub=4.2,label='z={}, thermal'.format(z))
        #print('z={}, slopec={}, slopet={}'.format(z,np.around(slopec,2),np.around(slopet,2)))

        avg = statavg(z,'const_q','dw_harm',0.95,hesstype='centroid',label=w,nbeads=32,c='0.1')
        Cavgarr.append(avg)
        #print('Cavgarr',Cavgarr)
        avg = statavg(z,'const_q','dw_harm',0.95,hesstype='full',label=w,nbeads=32,c='0.6')
        Ravgarr.append(avg)
        
        #avg = statavg(z,'const_q','dw_harm',3.0,hesstype='centroid',nbeads=1,c='0.1')
        #Clavgarr.append(avg)

#ax.axvline(Clavgarr[0],color='k',ls='--',label='Centroid',c='0.4')
ax.axvline(Cavgarr[0],color='k',ls='--',label='Centroid',c='0.4')
ax.axvline(Ravgarr[0],color='k',ls='-',label='Full',c='0.8')
#ax.plot([0.0, 0.5, 1.0, 1.5, 2.0],Cavgarr,'o',label='Centroid')
#ax.plot([0.0, 0.5, 1.0, 1.5, 2.0],Ravgarr,'o',label='Full')
#fig.set_size_inches(3,4)
#plt.legend()#loc='upper left',ncol=1,fontsize=7.25)

plt.show()

#------------------------------------------------------------------
#'z={}, $\lambda$={}, {}'.format(z,np.around(slope,2),enskey)


#slopet32z15 = plot_and_slope(1.5,'thermal','dw_qb',0.95,c='k',ls='-',nbeads=32)
#for z in [0.0,1.5]:#,1.5,2.0]:#,0.2,0.5,1.0]:#,2.0]:#0.5,1.0,1.5,2.0]:#

