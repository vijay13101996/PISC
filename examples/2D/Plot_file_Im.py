import numpy as np
from matplotlib import pyplot as plt
from PISC.utils.plottools import plot_1D
import os
from PISC.potentials import coupled_harmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

path = os.path.dirname(os.path.abspath(__file__))
rpext = '{}/rpmd/Datafiles/'.format(path)

enskey = 'thermal'
corrkey = 'Im_qq_TCF'

dim=2
m=0.5

omega=1.0
dt = 0.01
title = 'CH'#potkey

nb = 16

fig, ax = plt.subplots()

for g,ls in zip([0.0,0.05],['-','--']):  
    pes = coupled_harmonic(omega,g)
    potkey = 'CH_omega_{}_g_{}'.format(omega,g)
    for times,c in zip([1.0,2.0,3.0,4.0],['r','g','b','k']):
        T=times
        Tkey = 'T_{}'.format(times)
        data = read_1D_plotdata('{}RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}.txt'.format(rpext,enskey,corrkey,potkey,Tkey,nb,dt))
        time = data[:,0]*T
        ImTCF = data[:,1]#*nb
        time = np.concatenate((time,[1.0]))
        ImTCF = np.concatenate((ImTCF,[ImTCF[0]]))
        print('avg', np.mean(ImTCF),time[-1])
        ax.plot(time,ImTCF,label=r'$T={}$'.format(times),linestyle=ls,color=c)

ax.set_xlabel(r'$\tau \slash \beta$',fontsize=16)
ax.set_ylabel(r'$K_{xx}(\tau \slash \beta)$',fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
#ax.set_ylim([0.0,5])

fig.suptitle(title,fontsize=16)

fig.legend(loc=[0.21,0.75],fontsize=14,ncol=2)
fig.set_size_inches(6,6)

#fig.savefig(figname,dpi=400,bbox_inches='tight')
            #,bbox_inches='tight')

plt.show()
