import numpy as np
from matplotlib import pyplot as plt
from PISC.utils.plottools import plot_1D
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

path = os.path.dirname(os.path.abspath(__file__))
rpext = '{}/rpmd/Datafiles/'.format(path)

enskey = 'thermal'
corrkey = 'Im_qq_TCF'

dt = 0.01

if(1):
    lamda = 2.0
    g = 0.08
    Tc = 0.5*lamda/np.pi
    potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
    Tkey = 'T_{}Tc'.format(0.95)
    title = 'Double Well Potential'
    figname = 'DW.png'

if(0):
    m=1.0
    omega = 1.0
    potkey = 'harmonic_omega_{}'.format(omega)
    Tkey = 'T_{}'.format(1.0)
    title = 'Harmonic Oscillator'
    figname = 'Harmonic.png'

if(1):
    m=1.0
    a = 1.0
    potkey = 'quartic_a_{}'.format(a)
    Tkey = 'T_{}'.format(0.125)
    title = 'Quartic Oscillator'
    figname = 'Quartic.png'


fig, ax = plt.subplots()

#for nb in [32,64]:  #8,16,32,64,128]:
nb=32
times=1.0
a=0.0#2
b=0.0#4
#for times in [0.125,0.25,0.5,1.0,1.5,2.0,3.0,4.0]:

#for times in [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]:
for times in [1.0,2.0,3.0,4.0]:
#for times in [0.95,2.0,3.0,4.0]:#[0.95,1.5,2.0,3.0,4.0]:
#for a,b in zip([0.0,0.1,0.2,0.3,0.4],[0.0,0.01,0.04,0.09,0.16]):
    #potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,b)
    T=times#*Tc
    Tkey = 'T_{}'.format(times)
    data = read_1D_plotdata('{}RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}.txt'.format(rpext,enskey,corrkey,potkey,Tkey,nb,dt))
    time = data[:,0]*T
    ImTCF = data[:,1]*nb
    time = np.concatenate((time,[1.0]))
    ImTCF = np.concatenate((ImTCF,[ImTCF[0]]))
    #print('avg', np.mean(ImTCF),time[-1])
    print('C0',ImTCF[0])
    #ax.plot(time,ImTCF,label='a={},b={}'.format(a,b))
    ax.plot(time,ImTCF,label=r'$T={}T_c$'.format(times))
             #label='nbeads={}'.format(nb))

ax.set_xlabel(r'$\tau \slash \beta$',fontsize=16)
ax.set_ylabel(r'$K_{xx}(\tau \slash \beta)$',fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
#ax.set_ylim([0.0,5])

fig.suptitle(title,fontsize=16)

fig.legend(loc=[0.21,0.75],fontsize=14,ncol=2)
fig.set_size_inches(6,6)

fig.savefig(figname,dpi=400,bbox_inches='tight')
            #,bbox_inches='tight')

plt.show()
