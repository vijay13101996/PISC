import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import morse, Veff_classical_1D_LH, Veff_classical_1D_GH, Veff_classical_1D_FH, Veff_classical_1D_FK,double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec

ngrid = 401

lb = -5
ub = 20.0
dx = (ub-lb)/(ngrid-1)

if(1): # Anharmonicity parametrisation
    m = 1
    we = 1
    xe = 0.05

    alpha = np.sqrt(2*m*we*xe)
    D = we/(4*xe)

pes = morse(D,alpha)#,0.0)

print('w', 2*D*alpha**2/(m))

times = 8
beta = times/we
print('beta',beta, beta*we)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

print('vals num',vals[:20])
print('vals anh', we*(np.arange(20)+0.5)-we*xe*(np.arange(20)+0.5)**2)


qgrid = np.linspace(lb,ub,ngrid)
Pq = np.zeros_like(qgrid)
potgrid = pes.potential(qgrid)
hessgrid = pes.ddpotential(qgrid)
potgridn = potgrid[hessgrid<0]
qgridn = qgrid[hessgrid<0]

lamdac = np.sqrt(2*abs(min(hessgrid))/m)
maxhessn = np.argmin(hessgrid)
Tca = 1.415*lamdac/(2*np.pi)
betac = 1/Tca
print('lamdac',lamdac,Tca,betac)

fig,axis = plt.subplots(4,2)
ax1,ax2,ax3,ax4 = axis[:,0]
ax5,ax6,ax7,ax8 = axis[:,1]

axlist = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
betalist = [0.1,0.2,0.5,1.0,2.0,3.0,4.0,5.0]

slist = [10,10,10,10,4,4,4,4]

fig.subplots_adjust(top=0.95,wspace=0.0, hspace=0.0)

for beta,ax,s in zip(betalist,axlist,slist):
    #Quantum
    Pq[:] = 0.0
    Zq = np.sum(np.exp(-beta*vals))
    for n in range(len(vals)):
            Pq += np.exp(-beta*vals[n])*vecs[:,n]**2/Zq

    Qind = find_maxima(Pq)

    #Classical
    Zcl = np.sum(dx*np.exp(-beta*pes.potential(qgrid)))
    Pcl = np.exp(-beta*pes.potential(qgrid))/Zcl

    Cind = find_maxima(Pcl)

    #Local harmonic approximation
    pes_eff_LH = Veff_classical_1D_LH(pes,beta,m) 
    Veff_LH = np.zeros_like(qgrid) 
    for i in range(len(qgrid)): 
            Veff_LH[i] = pes_eff_LH.potential(qgrid[i])

    qgrid_LH = qgrid[~np.isnan(Veff_LH)]
    Veff_LH = Veff_LH[~np.isnan(Veff_LH)]
    Zeff_LH = np.sum(dx*np.exp(-beta*Veff_LH))
    Peff_LH = np.exp(-beta*Veff_LH)/Zeff_LH

    LHind = find_maxima(Peff_LH)

    #Feynman Hibbs approximation
    pes_eff_FH = Veff_classical_1D_FH(pes,beta,m,qgrid) 
    Veff_FH = np.zeros_like(qgrid) 
    for i in range(len(qgrid)): 
            Veff_FH[i] = pes_eff_FH.potential(qgrid[i])
    Zeff_FH = np.sum(dx*np.exp(-beta*Veff_FH))
    Peff_FH = np.exp(-beta*Veff_FH)/Zeff_FH

    #Feynman Kleinert approximation (upto quadratic)
    pes_eff_FK = Veff_classical_1D_FK(pes,beta,m) 
    Veff_FK = np.zeros_like(qgrid) 
    for i in range(len(qgrid)): 
            Veff_FK[i] = pes_eff_FK.potential(qgrid[i])
    Zeff_FK = np.sum(dx*np.exp(-beta*Veff_FK))
    Peff_FK = np.exp(-beta*Veff_FK)/Zeff_FK

    FKind = find_maxima(Peff_FK)

    #Check normalisation
    print('norm', dx*Pq.sum(), dx*Pcl.sum(), dx*Peff_LH.sum())

    ax.set_ylim([0,2*D])
    ax.set_xlim([-3,10])#[lb,ub])

    for i in range(8):
        ax.axhline(y=vals[i],color="0.7",ls='--')

    ax.plot(qgrid,potgrid)
    #ax.plot(qgrid,hessgrid)

    ax.plot(qgridn,potgridn,color='m')
    ax.scatter(qgrid[maxhessn],potgrid[maxhessn],marker='s',color='m',zorder=4,s=5)

    #plt.plot(qgrid,Pq_analytic)
    ax.plot(qgrid_LH,s*Peff_LH,color='k',lw=1.5,label=r'$V_{eff} \; LH$')
    #plt.plot(qgrid,Peff_GH,color='m',label='Veff GH')
    #plt.plot(qgrid,Peff_FH,color='c',linestyle = '--',label='Veff FH')
    ax.plot(qgrid,s*Peff_FK,color='b',linestyle='-',label=r'$V_{eff} \; FK$',alpha=0.9)
    ax.plot(qgrid,s*Pq,color='r',label=r'$Quantum$')
    ax.plot(qgrid,s*Pcl,color='g',label=r'$Classical$',alpha=0.9)

    ax.scatter(qgrid[LHind],s*Peff_LH[LHind],color='k',zorder=5,s=5)
    ax.scatter(qgrid[Qind], s*Pq[Qind],color='r',zorder=6,s=5)	
    #plt.scatter(qgrid[FKind], s*Peff_FK[FKind],color='b')
    #plt.scatter(qgrid[Cind], s*Pcl[Cind],color='g')
    
    ax.set_xticks([])
    ax.set_yticks([])

    ax.annotate(r'$\beta={}$'.format(beta),xy=(0.7,0.7),xytext=(0.55,0.8),xycoords='axes fraction',fontsize=14)

ax4.set_xticks([0,4,8])
ax8.set_xticks([0,4,8])

for ax in [ax1,ax2,ax3,ax4]:
    ax.set_yticks([1,3,5,7,9])

fig.suptitle(r'$\chi_e={}$'.format(xe),fontsize=15)
#plt.title(r'$T={}, \beta E_0={} $'.format(T_au,np.around(0.5*beta,2)))

handles, labels = ax1.get_legend_handles_labels()


fig.legend(handles,labels,loc=[0.2,0.0],ncol=2)
fig.set_size_inches(4,8)
#fig.savefig('Morse_harm_allbeta.png',dpi=400,bbox_inches='tight',pad_inches=0.0)
plt.show()



