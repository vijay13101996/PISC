import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import mildly_anharmonic, Veff_classical_1D_LH, morse, anharmonic, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import matplotlib
import os 
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from Distribution_plotter2 import plot_dist
from Free_energy import compute_free_energy, compute_free_energy_FK_iteration, compute_free_energy_LH_iteration

ngrid = 401

#Quartic/Harmonic/Mildly anharmonic
lb = -10.
ub = 10.0
dx = (ub-lb)/(ngrid-1)
qgrid = np.linspace(lb,ub,ngrid)

m = 1
w = 1. #quadratic term (frequency)
a = 0.0 #cubic term
b = 0.25 #quartic term
g = 1.0 #anharmonicity parameter

tol = 1e-4

renorm = 'PF'
fur_renorm = 'VGS'
magic = True#False

imfile = 'NO_MAGIC_FK_p_quart_{}_{}.png'.format(renorm,fur_renorm)
llim_list = [-5,-3,-2.5,-2]
ulim_list = [5,3,2.5,2]

beta = 1

betaarr = np.arange(0.25,5.01,0.25)#0.1)
FEarr = []

fig,ax = plt.subplots(2,2,sharey=True)

plt.subplots_adjust(hspace=0.3)
MAH = True

for g,ax in zip([0.0,2.0,4.0,40.0],ax.flatten()):
    if(MAH):
        pes = mildly_anharmonic(m,a,b*g,w)#,sign=1)
        coeff = [0.0,0.0,0.5*m*w**2,0.0,g/4]
    else:
        lamda = 1.0
        pes = double_well(lamda,g/4,m=1)
        coeff = [m**2*lamda**4/(4*g),0.0,-0.5*m*lamda**2,0.0,g/4]

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    vals,vecs = DVR.Diagonalize() 

    for beta in betaarr:
        Z_FK = compute_free_energy(beta,qgrid,vals,vecs,pes,m,'FK',tol=tol)
        FE_FK = -np.log(Z_FK)/beta
        
        Z_q = compute_free_energy(beta,qgrid,vals,vecs,pes,m,'quantum',tol=tol)
        FE_q = -np.log(Z_q)/beta
       
        Z_cl = compute_free_energy(beta,qgrid,vals,vecs,pes,m,'classical',tol=tol)
        FE_cl = -np.log(Z_cl)/beta
        
        Z_NCF = compute_free_energy(beta,qgrid,vals,vecs,pes,m,'LH',tol=tol, renorm=renorm, magic=magic)
        FE_NCF = -np.log(Z_NCF)/beta

        Z_FH = compute_free_energy(beta,qgrid,vals,vecs,pes,m,'FH',tol=tol)
        FE_FH = -np.log(Z_FH)/beta

        #Z_FK_iter = compute_free_energy_FK_iteration(beta,qgrid,coeff,m)
        #FE_FK_iter = -np.log(Z_FK_iter)/beta

        V = pes.potential_func
        ddV = pes.ddpotential
        Z_FH_iter = compute_free_energy_LH_iteration(beta,qgrid,V,ddV,m)
        FE_FH_iter = -np.log(Z_FH_iter)/beta

        # Label only one point
        if(beta == betaarr[0] and g==0.0):
            ax.scatter(beta,FE_FK,color='b',s=4,label='FK')
            ax.scatter(beta,FE_q,color='k',s=4,label='quantum')
            ax.scatter(beta,FE_NCF,color='r',s=4,label='LH')
            ax.scatter(beta,FE_cl,color='g',s=4,label='classical')
            ax.scatter(beta,FE_FH,color='m',s=4,label='FH')
            #ax.scatter(beta,FE_FK_iter,color='0.5',s=4,label='FK iteration')
            ax.scatter(beta,FE_FH_iter,color='0.5',s=4,label='FH iteration')
        else:
            #ax.scatter(beta,FE_FK,color='b',s=4)
            ax.scatter(beta,FE_q,color='k',s=4)
            ax.scatter(beta,FE_NCF,color='r',s=4)
            ax.scatter(beta,FE_cl,color='g',s=4)
            #ax.scatter(beta,FE_FH,color='m',s=4)
            #ax.scatter(beta,FE_FK_iter,color='0.5',s=4)
            ax.scatter(beta,FE_FH_iter,color='0.5',s=4)

        ax.set_ylim([-0.5,2])
        ax.set_xlim([0,5.5])
        ax.set_title('g = {}'.format(g),loc='center',y=0.8)
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$F(\beta)$')
        ax.set_xticks([0,1,2,3,4,5])

if(magic):
    fig.suptitle('Free energy of the MAH ({} renormalisation with Magic)'.format(renorm))
else:
    fig.suptitle('Free energy of the MAH ({} renormalisation, no Magic)'.format(renorm))
fig.legend(loc='upper center',bbox_to_anchor=(0.5,0.95),ncol=4)
fig.set_size_inches(6,6)
plt.show()
exit()
if(magic):
    fig.savefig('FE_MAH_{}_MAGIC.png'.format(renorm),dpi=400,bbox_inches='tight')#,pad_inches=0.0)
else:
    fig.savefig('FE_MAH_{}_NOMAGIC.png'.format(renorm),dpi=400,bbox_inches='tight')

plt.show()
exit()

for method in ['quantum','classical','NCF','FK']:
    Z = compute_free_energy(beta,qgrid,vals,vecs,pes,m,method,tol=tol)
    #bF = np.sum(np.exp(-bV)*dx)
    Zq = 1/(2*np.sinh(beta*w/2))
    Zcl = 1/(beta*w)
    FE = -np.log(Z)/beta
    print('Z',method,Z,Zq,Zcl,FE)
    #print('Z',method,bF/np.sqrt(2*np.pi*beta/m),bF,Zanal)
    #F = -np.log(bF)/beta
    #Fanal = w/2 + np.log(1-np.exp(-beta*w))/beta
    #print(method,F,Fanal)
    #plt.plot(qgrid,bV,label=method)

#plt.legend()
#plt.show()


exit()
#--------------------------------------------

betalist = [0.1,1.0,10.0,100.0]


#--------------------------------------------
fig, ax = plt.subplots(2,2,gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

#plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,renorm='NCF',tol=tol,TinKlist=Tlist)

plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,tol=tol,renorm=renorm,fur_renorm=fur_renorm)

fig.set_size_inches(4,4)
#fig.savefig(imfile,dpi=400,bbox_inches='tight')#,pad_inches=0.0)
plt.show()


