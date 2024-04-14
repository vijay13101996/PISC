import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import harmonic1D, Veff_classical_1D_LH
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import matplotlib
import os 
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
ngrid = 801

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 14 
yl_fs = 14
tp_fs = 12
 
le_fs = 9
ti_fs = 12

renorm = 'harm'
exponentiate = True

betalist = [0.1,1.0,10.0,100.0]
if(renorm=='harm'):
    imfile = 'harmonic_harm.pdf'
    llim_list = [-5,-3,-2.5,-2]
    ulim_list = [5,3,2.5,2]
else:
    imfile = 'harmonic_PF.pdf'
    llim_list = [-5,-3,-1.5,-0.5]
    ulim_list = [5,3,1.5,0.5]

#Harmonic
lb = -10
ub = 10.0
dx = (ub-lb)/(ngrid-1)
m = 2.0
omega = 2.0

tol=1

pes = harmonic1D(m,omega)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

qgrid = np.linspace(lb,ub,ngrid)
Pq = np.zeros_like(qgrid)
potgrid = pes.potential(qgrid)

fig, ax = plt.subplots(2,2,gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

ax1,ax2,ax3,ax4 = ax.flatten()

axlist = [ax1,ax2,ax3,ax4]

for beta,ax,llim,ulim in zip(betalist,axlist,llim_list,ulim_list):
    #Quantum
    Pq = np.zeros_like(qgrid)
    for i in range(len(vals)):
        Pq+= np.exp(-beta*vals[i])*vecs[:,i]**2
    Pq/=np.sum(Pq*dx)
    bVq = -np.log(Pq)
    if(exponentiate):
        bVq = np.exp(-bVq)
        lq = r'$P_{q}(q)$' 
    else:
        lq = r'$\beta V_{\rm q}(q)$'
    ax.plot(qgrid,bVq,color='r',label=lq,alpha=0.9)

    #----------------------------------------------------------------------------------------

    #Classical
    bVcl = beta*potgrid 
    if(renorm=='harm'):
        bVcl+= 0.5*np.log(m/2/np.pi/beta)/beta #added to offset the kinetic energy term in the Partition function
    if(exponentiate):
        bVcl = np.exp(-bVcl)
        bVcl/=np.sum(bVcl*dx)
        lcl = r'$P_{cl}(q)$' #r'$\exp(-\beta V_{\rm cl}(q))$'
    else:
        lcl = r'$\beta V_{\rm cl}(q)$'
    ax.plot(qgrid,bVcl,color='g',label=lcl,alpha=0.9)

    #----------------------------------------------------------------------------------------

    #Effective potential
    pes_eff = Veff_classical_1D_LH(pes,beta,m,tol=tol,renorm=renorm)
    bVeff = np.zeros_like(qgrid)
    for i in range(len(qgrid)): 
        bVeff[i] = beta*pes_eff.potential(qgrid[i])
    if(exponentiate):
        bVeff = np.exp(-bVeff)
        norm = np.sum(bVeff*dx)
        print('norm',norm,np.pi/(m*omega))
        bVeff/=norm
        leff = r'$P_{eff}(q)$' #'$\exp(-\beta V_{eff}(q))$'
    else:
        leff = r'$\beta V_{eff}(q)$'

    ax.plot(qgrid,bVeff,color='k',label=leff)
    #ax.plot(qgrid,bVeff-bVcl,color='r',label=r'$\beta V_{eff}(q) - \beta V_{\rm cl}(q)$',alpha=0.9)

    #----------------------------------------------------------------------------------------

    #Beta-independent part
    hess = pes.ddpotential(qgrid)/m
    hess[hess<tol] = tol
    wa = np.sqrt(hess)
    ka = pes.dpotential(qgrid)/m
    xi = beta*wa/2
    
    if(renorm=='harm'):# Harmonic renormalisation
        bVeff_betainf = np.tanh(xi)*(m*ka**2)/wa**3 - 0.5*np.log(np.tanh(xi)) + 0.5*np.log(m*wa/np.pi)  
    elif(renorm=='PF'):# Partition function renormalisation
        bVeff_betainf = np.tanh(xi)*(m*ka**2)/wa**3 - 0.5*np.log(np.tanh(xi)/xi) + np.log(np.sinh(xi)/xi) 

    if(exponentiate):
        bVeff_betainf = np.exp(-bVeff_betainf)
        lbinf = r'$\exp(-\beta V_{asymp}(q))$'
    else:
        lbinf = r'$\beta V_{asymp}(q)$'

    #ax.plot(qgrid,bVeff_betainf,color='y',label=lbinf,alpha=0.9)

    #----------------------------------------------------------------------------------------

    #Beta dependent part
    bVeff_betadep = -beta*m*ka**2/(2*hess) + beta*potgrid
    if(exponentiate):
        bVeff_betadep = np.exp(-bVeff_betadep)
        lbdep = r'$\exp(-\beta V_{\beta}(q))$'
    else:
        lbdep = r'$\beta V_{\beta}(q)$'
    #ax.plot(qgrid,bVeff_betadep,color='b',label=lbdep,alpha=0.9)
    
    #if(exponentiate):
    #    ax.plot(qgrid,bVeff_betainf*bVeff_betadep/norm,color='0.5',alpha=0.9,ls=':',lw=3) #label=r'$\exp(-\beta V_{\beta}(q) - \beta V_{asymp}(q))$',
    #else:
    #    ax.plot(qgrid,bVeff_betainf + bVeff_betadep,color='0.5',label=r'$\beta V_{\beta}(q) + \beta V_{asymp}(q)$',alpha=0.9,ls=':',lw=3)

    # Analytical expression for ground state of harmonic oscillator
    #ax.plot(qgrid,np.sqrt(m/np.pi)*np.exp(-m*qgrid**2),color='r',label=r'$\sqrt{m\omega}/\pi$',alpha=0.9,ls='--',lw=1)

    if(ax==ax3 or ax==ax4):
        ax.set_xlabel(r'$q$',fontsize=xl_fs)
    else:
        ax.set_xlabel('')
    if(ax==ax1 or ax==ax3):
        if(exponentiate):
            if(renorm=='harm'):
                ax.set_ylabel(r'$P^{harm}(q)$',fontsize=yl_fs)
            else:
                ax.set_ylabel(r'$P^{PF}(q)$',fontsize=yl_fs)
        else:
            ax.set_ylabel(r'$\beta V(q)$',fontsize=yl_fs)
        ax.yaxis.set_label_coords(-0.26,0.5)
    ax.set_xlim(llim,ulim)

    ax.set_ylim(0,1.25*ax.get_ylim()[1])
    #ax.set_ylim(0,5*D)
    ax.annotate(r'$\beta={}$'.format(beta),xy=(0.7,0.7),xytext=(0.28,0.85),xycoords='axes fraction',fontsize=xl_fs)

    ax.tick_params(labelsize=tp_fs)
    #ax.yaxis.set_major_formatter(ticker.LogLocator())#(labelOnlyBase=False,base=10))

handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles,labels,loc=[0.21,0.87],ncol=2)
fig.set_size_inches(6,6)#4,4)
#fig.savefig(imfile,dpi=400,bbox_inches='tight')#,pad_inches=0.0)

plt.show()
