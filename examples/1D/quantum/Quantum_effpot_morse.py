import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import mildly_anharmonic, Veff_classical_1D_LH, morse, anharmonic, polynomial
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import matplotlib
import os 
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import scipy
from scipy.optimize import curve_fit
from Distribution_plotter import plot_dist
from Truncated_pes import truncate_pes

ngrid = 801

#----------------------------------------------
#Morse
lb = -5
ub = 20.0
dx = (ub-lb)/(ngrid-1)
qgrid = np.linspace(lb,ub,ngrid)

m = 1
we = 1
xe = 0.05

alpha = np.sqrt(2*m*we*xe)
D = we/(4*xe)

pes = morse(D,alpha) 
trunc = False
if(trunc):
    pes = truncate_pes(pes,ngrid,-2,2)
    llim_list = [-5,-3,-2.5,-2]
    ulim_list = [8,5,4.5,3]
else:
    llim_list = [-5,-3,-2.5,-2]
    ulim_list = [8,5,4.5,3]

#----------------------------------------------
tol = 1e-2

betalist = [0.1,1.0,10.0,100.0]

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

#----------------------------------------------
fig, ax = plt.subplots(2,2,gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,renorm='NCF',tol=tol)

fig.set_size_inches(4,4)
imfile = 'morse_h_anh.pdf'
#fig.savefig(imfile,dpi=400,bbox_inches='tight')#,pad_inches=0.0)
plt.show()

exit()


#----------------------------------------------
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
    if(renorm=='harm'or renorm=='NCF'):
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
        #print('norm',norm)
        bVeff/=norm
        leff = r'$P_{eff}(q)$' #'$\exp(-\beta V_{eff}(q))$'
    else:
        leff = r'$\beta V_{eff}(q)$'

    ax.plot(qgrid,bVeff,color='k',label=leff)

    #----------------------------------------------------------------------------------------
    #Beta-independent part
    hess = pes.ddpotential(qgrid)/m
    hess[hess<tol] = tol
    wa = np.sqrt(hess)
    ka = pes.dpotential(qgrid)/m
    xi = beta*wa/2
    
    if(renorm=='harm' or renorm=='NCF'):# Harmonic renormalisation
        bVeff_betainf = np.tanh(xi)*(m*ka**2)/wa**3 - 0.5*np.log(np.tanh(xi)) - 0.5*np.log(m*wa/np.pi)  
    elif(renorm=='PF'):# Partition function renormalisation
        bVeff_betainf = np.tanh(xi)*(m*ka**2)/wa**3 - 0.5*np.log(np.tanh(xi)/xi) + np.log(np.sinh(xi)/xi) 

    if(exponentiate):
        bVeff_betainf = np.exp(-bVeff_betainf)
        lbinf = r'$\exp(-\beta V_{asymp}(q))$'
    else:
        lbinf = r'$\beta V_{asymp}(q)$'

    #----------------------------------------------------------------------------------------
    #Beta dependent part
    bVeff_betadep = -beta*m*ka**2/(2*hess) + beta*potgrid
    if(exponentiate):
        bVeff_betadep = np.exp(-bVeff_betadep)
        lbdep = r'$\exp(-\beta V_{\beta}(q))$'
    else:
        lbdep = r'$\beta V_{\beta}(q)$'
    
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
    ax.annotate(r'$\beta={}$'.format(beta),xy=(0.7,0.7),xytext=(0.28,0.85),xycoords='axes fraction',fontsize=xl_fs)

    ax.tick_params(labelsize=tp_fs)
    #ax.yaxis.set_major_formatter(ticker.LogLocator())#(labelOnlyBase=False,base=10))

handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles,labels,loc=[0.15,0.9],ncol=3)
fig.set_size_inches(4,4)
fig.savefig(imfile,dpi=400,bbox_inches='tight')#,pad_inches=0.0)

plt.show()
