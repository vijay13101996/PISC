import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import Veff_classical_1D_LH
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from Truncated_pes import truncate_pes
from numpy import nan
import os 

# Set formatting for the plots
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 14 
yl_fs = 14
tp_fs = 12
 
le_fs = 9
ti_fs = 12

#def plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,renorm='NCF',tol=1e-8,TinKlist=None):
def plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,
              exponentiate=True,tol=1e-8,TinKlist=None,ylcoord=-0.26,renorm='NCF',fur_renorm='VGS'):
    """
    Plot the distribution of the quantum, classical and effective potential for a given set of betas
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
    llim_list : list of lower limits for the x-axis
    ulim_list : list of upper limits for the x-axis
    betalist : list of betas
    qgrid : grid for the coordinate
    vals : Eigenvalues
    vecs : Eigenvectors
    pes : Potential Energy Surface object
    m : mass of the particle
    exponentiate : If True, exponentiate the potential
    renorm : If 'harm' or 'NCF', renormalize the potential
    tol : Tolerance for the effective potential (frequency cutoff)
    TinKlist : List of temperatures in K (optional)
    """
    

    ax1,ax2,ax3,ax4 = ax.flatten()
    axlist = [ax1,ax2,ax3,ax4]
    
    dx = qgrid[1]-qgrid[0]
    potgrid = pes.potential(qgrid)
    
    for beta,ax,llim,ulim in zip(betalist,axlist,llim_list,ulim_list):
        #Quantum
        Pq = np.zeros_like(qgrid)
        for i in range(len(vals)):
            Pq+= np.exp(-beta*vals[i])*vecs[:,i]**2
        if(np.sum(Pq*dx)<1e-4): #if the sum is too small, then use the ground state distribution
            Pq = vecs[:,0]**2
        else:
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
        #renorm = 'NCF'
        if(renorm=='harm' or renorm=='NCF'):
            bVcl+= 0.5*np.log(m/2/np.pi/beta)/beta #added to offset the kinetic energy term in the Partition function
        if(exponentiate):
            bVcl = np.exp(-bVcl)
            bVcl/=np.sum(bVcl*dx)
            lcl = r'$P_{cl}(q)$' 
        else:
            lcl = r'$\beta V_{\rm cl}(q)$'
        ax.plot(qgrid,bVcl,color='g',label=lcl,alpha=0.9,zorder=3)

        #----------------------------------------------------------------------------------------
        #Effective potential with NCF renormalization
        pes_eff = Veff_classical_1D_LH(pes,beta,m,tol=tol,renorm=renorm,fur_renorm=fur_renorm)
        bVeff = np.zeros_like(qgrid)
        for i in range(len(qgrid)): 
            bVeff[i] = beta*pes_eff.potential(qgrid[i])
        if(exponentiate):
            bVeff = np.exp(-bVeff)
            norm = np.sum(bVeff*dx)
            bVeff/=norm
            leff = r'$P_{{LH}}(q) \;\; ({})$'.format(renorm)
        else:
            leff = r'$\beta V_{\rm LH}(q)$'

        ax.plot(qgrid,bVeff,color='k',label=leff)

        #----------------------------------------------------------------------------------------
        ylim = np.max([np.max(bVcl),np.max(bVeff)])
        print('max',np.max(bVeff),np.max(bVcl))
        
        #Effective potential with PF renormalization
        pes_eff = Veff_classical_1D_LH(pes,beta,m,tol=tol,renorm='PF')
        bVeff = np.zeros_like(qgrid)
        for i in range(len(qgrid)): 
            bVeff[i] = beta*pes_eff.potential(qgrid[i])
        if(exponentiate):
            bVeff = np.exp(-bVeff)
            norm = np.sum(bVeff*dx)
            bVeff/=norm
            leff = r'$P_{eff}(q)$'
        else:
            leff = r'$\beta V_{eff}(q)$'

        bVeff[bVeff>1.01*ylim] = ylim
        #ax.plot(qgrid,bVeff,color='k',ls=':',lw=3)

        #----------------------------------------------------------------------------------------
        #ax.set_xlim(llim,ulim)
        ax.set_ylim(0,1.25*ylim)


        if(TinKlist is not None): #If temperature is given in K, then annotate the temperature
            TinK = TinKlist[np.where(betalist==beta)][0]
            ax.annotate(r'$T={}K$'.format(TinK),xy=(0.7,0.7),xytext=(0.17,0.85),xycoords='axes fraction',fontsize=xl_fs)
            ax.tick_params(labelsize=tp_fs)
        else:
            ax.annotate(r'$\beta={}$'.format(beta),xy=(0.7,0.7),xytext=(0.28,0.85),xycoords='axes fraction',fontsize=xl_fs)

        if(ax==ax3 or ax==ax4):
            ax.set_xlabel(r'$q$',fontsize=xl_fs)
        else:
            ax.set_xlabel('')
        
        if(ax==ax1 or ax==ax3):
            if(exponentiate):
                ax.set_ylabel(r'$P(q)$',fontsize=yl_fs)
            else:
                ax.set_ylabel(r'$\beta V(q)$',fontsize=yl_fs)
            ax.yaxis.set_label_coords(ylcoord,0.5)
        
        #ax.yaxis.set_major_formatter(ticker.LogLocator())#(labelOnlyBase=False,base=10))

    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(handles,labels,loc=[0.1,0.9],ncol=3)


