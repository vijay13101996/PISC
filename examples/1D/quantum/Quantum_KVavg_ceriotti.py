import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import anharmonic, Veff_classical_1D_LH, Veff_classical_1D_GH, Veff_classical_1D_FH, Veff_classical_1D_FK,double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib

# Set formatting for the plots
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 14 
yl_fs = 14
tp_fs = 12
 
le_fs = 9
ti_fs = 12

ngrid = 201

lbarr=np.array([-20,-15,-10,-10,-10,-10,-8,-7,-5,-4])
ubarr=np.array([300,200,150,100,80,60,40,30,20,16])

# Anharmonicity parametrisation
m = 1836#741.1
cm1toau = 219474.63
K2au = 0.000003167
k= 1/1.8897 # 1 Angstrom in a.u.
TinK =100
T = TinK*0.000003167
beta = 1/T
print('beta',beta)


omega_arr = list(np.logspace(1,3,10))
neig_arr = [120,100,100,100,10,10,10,10,10,10]
print('omega_arr',omega_arr)

fig, ax = plt.subplots(2,1)
fig.subplots_adjust(hspace=0.0)


def func(lbarr,ubarr,ma):
    for c,ngrid in zip(['k','r','g'],[801]):
        for wcm in omega_arr:
            lb = lbarr[omega_arr.index(wcm)]
            ub = ubarr[omega_arr.index(wcm)]
            neig = neig_arr[omega_arr.index(wcm)]
                    
            qgrid = np.linspace(lb,ub,ngrid)
            potgrid = np.zeros_like(qgrid)
            print('qgrid',qgrid[-1],qgrid[0])
            Pq = np.zeros_like(qgrid)
            dx = qgrid[1]-qgrid[0]    

            omega = wcm/cm1toau # 1 cm^-1 in a.u.
            pes = anharmonic(m,omega,k)
            DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
            vals,vecs = DVR.Diagonalize(neig_total=neig)
            
            potgrid[:] = pes.potential(qgrid)
            dVdx = np.gradient(potgrid,dx)
            
            #Quantum
            Pq[:] = 0.0
            Zq = np.sum(np.exp(-beta*vals))
            for n in range(len(vals)):
                Pq += np.exp(-beta*vals[n])*vecs[:,n]**2/Zq
            Eavg_q = np.sum(np.exp(-beta*vals)*vals)/Zq
            Vavg_q = np.sum(Pq*potgrid)*dx
            Kavg_q = 0.5*np.sum(Pq*dVdx*qgrid)*dx
            #Kavg_q = Eavg_q - Vavg_q

            tol = 1e-10
            #Classical
            Zcl = np.sum(dx*np.exp(-beta*pes.potential(qgrid)))
            Pcl = np.exp(-beta*pes.potential(qgrid))/Zcl
            Vavg_cl = np.sum(Pcl*potgrid)*dx
           
            print('dVdx',dVdx.shape)
            Kavg_cl = 0.5*np.sum(Pcl*dVdx*qgrid)*dx

            #Local harmonic approximation
            tol = 1e-8
            pes_eff_LH_NCF = Veff_classical_1D_LH(pes,beta,m,tol=tol,renorm='NCF')
            #pes_eff_LH_PF = Veff_classical_1D_LH(pes,beta,m,tol=tol,renorm='PF')
            Veff_LH_NCF = np.zeros_like(qgrid)
            #Veff_LH_PF = np.zeros_like(qgrid)

            for i in range(len(qgrid)):
                Veff_LH_NCF[i] = pes_eff_LH_NCF.potential(qgrid[i])
                #Veff_LH_PF[i] = pes_eff_LH_PF.potential(qgrid[i])
    
            Zeff_LH_NCF = np.sum(dx*np.exp(-beta*Veff_LH_NCF))
            Peff_LH_NCF = np.exp(-beta*Veff_LH_NCF)/Zeff_LH_NCF
            Vavg_LH_NCF = np.sum(Peff_LH_NCF*potgrid)*dx
            Kavg_LH_NCF = 0.5*np.sum(Peff_LH_NCF*dVdx*qgrid)*dx

            #Zeff_LH_PF = np.sum(dx*np.exp(-beta*Veff_LH_PF))
            #Peff_LH_PF = np.exp(-beta*Veff_LH_PF)/Zeff_LH_PF
            #Vavg_LH_PF = np.sum(Peff_LH_PF*potgrid)*dx
            #Kavg_LH_PF = 0.5*np.sum(Peff_LH_PF*dVdx*qgrid)*dx

            T = 1/beta*K2au
            print('wcm',wcm, np.sum(Pq*dx),np.sum(Pcl*dx),np.sum(Peff_LH_NCF*dx))#,np.sum(Peff_LH_PF*dx))
                
            if(wcm==omega_arr[0]):
                ax[0].scatter(wcm,Vavg_q*beta,marker=ma,color='r',label=r'$\beta \langle V \rangle$ (Quantum)')
                ax[0].scatter(wcm,Vavg_cl*beta,marker=ma,color='g',label=r'$\beta \langle V \rangle$ (Classical)')
                ax[0].scatter(wcm,Vavg_LH_NCF*beta,marker=ma,color='k',label=r'$\beta \langle V \rangle$ (LHA)')
                #ax[0].scatter(wcm,Vavg_LH_PF*beta,marker='x',color='k')
            
                ax[1].scatter(wcm,Kavg_q*beta,marker=ma,color='r',label=r'$\beta \langle K \rangle$ (Quantum)')
                ax[1].scatter(wcm,Kavg_cl*beta,marker=ma,color='g',label=r'$\beta \langle K \rangle$ (Classical)')
                ax[1].scatter(wcm,Kavg_LH_NCF*beta,marker=ma,color='k',label=r'$\beta \langle K \rangle$ (LHA)')
                #ax[1].scatter(wcm,Kavg_LH_PF*beta,marker=ma,color='k')
            else: 
                ax[0].scatter(wcm,Vavg_q*beta,marker=ma,color='r')
                ax[0].scatter(wcm,Vavg_cl*beta,marker=ma,color='g')
                ax[0].scatter(wcm,Vavg_LH_NCF*beta,marker=ma,color='k')
                #ax[0].scatter(wcm,Vavg_LH_PF*beta,marker='x',color='k')

                ax[1].scatter(wcm,Kavg_q*beta,marker=ma,color='r')
                ax[1].scatter(wcm,Kavg_cl*beta,marker=ma,color='g')
                ax[1].scatter(wcm,Kavg_LH_NCF*beta,marker=ma,color='k')
                #ax[1].scatter(wcm,Kavg_LH_PF*beta,marker='x',color='k')


func(lbarr,ubarr,'o')

ax[1].set_xlabel(r'$\omega$ (cm$^{-1}$)',fontsize=xl_fs)
ax[0].set_ylabel(r'$\beta \langle V \rangle$' ,fontsize=yl_fs)
ax[1].set_ylabel(r'$\beta \langle K \rangle$' ,fontsize=yl_fs)

ax[1].set_yticks([0.5,1,2,3,4])

ax[0].tick_params(axis='both',labelsize=ti_fs)
ax[1].tick_params(axis='both',labelsize=ti_fs)

ax[0].legend()
ax[1].legend()

ax[0].set_xscale('log')
ax[1].set_xscale('log')

ax[0].set_yscale('log')
ax[1].set_yscale('log')

fig.set_size_inches(4,8)
fig.savefig('/home/vgs23/Images/Quantum_KVavg_ceriotti.pdf',dpi=400,bbox_inches='tight',pad_inches=0.0)

plt.show()
