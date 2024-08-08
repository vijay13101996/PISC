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
lbarr1 = np.array([-20,-20,-20,-20,-20,-20,-20,-20,-20,-20])
#lbarr=np.array([-25,-20,-15,-12,-10,-10,-10,-10,-10,-10])
ubarr1=np.array([300,200,150,100,60,40,25,18,12,8])
ubarr2=np.array([300,200,150,100,80,60,40,30,20,16])

ubarr3=np.ones(10)*200
lb = -5#-5#-4#-20
ub = 15#15#8#300
dx = (ub-lb)/(ngrid-1)

# Anharmonicity parametrisation
m = 1836#741.1
cm1toau = 219474.63
K2au = 0.000003167
k= 1/1.8897 # 1 Angstrom in a.u.
TinK =100
T = TinK*0.000003167
beta = 1/T
print('beta',beta)

renorm = 'NCF'

omega_arr = list(np.logspace(1,3,10))
neig_arr = [120,100,100,100,10,10,10,10,10,10]
print('omega_arr',omega_arr)

fig, ax = plt.subplots()

def func(lbarr,ubarr,ma):
    for c,ngrid in zip(['k','r','g'],[801]):
        for wcm in omega_arr:
            lb = lbarr[omega_arr.index(wcm)]
            ub = ubarr[omega_arr.index(wcm)]
            neig = neig_arr[omega_arr.index(wcm)]
            #print('wcm,lb,ub,neig',wcm,lb,ub,neig)
                    
            qgrid = np.linspace(lb,ub,ngrid)
            potgrid = np.zeros_like(qgrid)
            print('qgrid',qgrid[-1],qgrid[0])
            Pq = np.zeros_like(qgrid)
            dx = qgrid[1]-qgrid[0]    

            omega = wcm/cm1toau # 1 cm^-1 in a.u.
            pes = anharmonic(m,omega,k)
            DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
            vals,vecs = DVR.Diagonalize(neig_total=neig)
            #print('ebeta',wcm,np.exp(-beta*(vals[-1] - vals[0])))
            
            print('vals',2*vals[:2]*cm1toau)
            potgrid[:] = pes.potential(qgrid)
            
            if(0):
                plt.plot(qgrid,potgrid)
                plt.plot(qgrid,vecs[:,-1]**2/np.sum(vecs[:,-1]**2))
                plt.ylim(0.0,50/beta)
                plt.show() 
            
            if(1):
                #Quantum
                Pq[:] = 0.0
                Zq = np.sum(np.exp(-beta*vals))
                for n in range(len(vals)):
                    Pq += np.exp(-beta*vals[n])*vecs[:,n]**2/Zq
                Eavg_q = np.sum(np.exp(-beta*vals)*vals)/Zq
                Vavg_q = np.sum(Pq*potgrid)*dx
                Kavg_q = Eavg_q - Vavg_q

                tol = 1e-10
                #Classical
                Zcl = np.sum(dx*np.exp(-beta*pes.potential(qgrid)))
                Pcl = np.exp(-beta*pes.potential(qgrid))/Zcl
                #Pcl[Pcl<tol] = tol
                Vavg_cl = np.sum(Pcl*potgrid)*dx
                

                #Local harmonic approximation
                tol = 1e-12
                pes_eff_LH = Veff_classical_1D_LH(pes,beta,m,tol=tol,renorm=renorm)
                Veff_LH = np.zeros_like(qgrid)
                for i in range(len(qgrid)):
                    Veff_LH[i] = pes_eff_LH.potential(qgrid[i])
                Zeff_LH = np.sum(dx*np.exp(-beta*Veff_LH))
                Peff_LH = np.exp(-beta*Veff_LH)/Zeff_LH
                #Peff_LH[Peff_LH<tol] = tol
                Vavg_LH = np.sum(Peff_LH*potgrid)*dx

                if(0):
                    #Feynman-Kleinert approximation
                    pes_eff_FK = Veff_classical_1D_FK(pes,beta,m)
                    Veff_FK = np.zeros_like(qgrid)
                    for i in range(len(qgrid)):
                        Veff_FK[i] = pes_eff_FK.potential(qgrid[i])
                    Zeff_FK = np.sum(dx*np.exp(-beta*Veff_FK))
                    Peff_FK = np.exp(-beta*Veff_FK)/Zeff_FK
                    #Peff_FK[Peff_FK<tol] = tol
                    Vavg_FK = np.sum(Peff_FK*potgrid)*dx

                T = 1/beta*K2au
                #print('wcm',wcm,Vavg_q,Vavg_cl,Vavg_LH)
                print('wcm',wcm, np.sum(Pq*dx),np.sum(Pcl*dx),np.sum(Peff_LH*dx))
                if(1):
                    if(wcm==omega_arr[0]):
                        ax.scatter(wcm,Vavg_q*beta,marker=ma,color='r',label=r'$\beta \langle V \rangle$ (Quantum)')
                        ax.scatter(wcm,Vavg_cl*beta,marker=ma,color='g',label=r'$\beta \langle V \rangle$ (Classical)')
                        ax.scatter(wcm,Vavg_LH*beta,marker=ma,color='k',label=r'$\beta \langle V \rangle$ (LHA)')
                        #ax.scatter(wcm,Eavg_q*beta,marker=ma,color='b',label=r'$\beta \langle E \rangle$ (Quantum)')
                        #ax.scatter(wcm,Vavg_FK*beta,marker=ma,color='y',label=r'$\beta \langle V \rangle$ (FKA)')
                    else: 
                        ax.scatter(wcm,Vavg_q*beta,marker=ma,color='r')
                        ax.scatter(wcm,Vavg_cl*beta,marker=ma,color='g')
                        ax.scatter(wcm,Vavg_LH*beta,marker=ma,color='k')
                        #ax.scatter(wcm,Eavg_q*beta,marker=ma,color='b')
                        #ax.scatter(wcm,Vavg_FK*beta,marker=ma,color='y')
                if(0):
                    ax[omega_arr.index(wcm)].plot(qgrid,Pq,color='r')
                    ax[omega_arr.index(wcm)].plot(qgrid,Pcl,color='g')
                    ax[omega_arr.index(wcm)].plot(qgrid,Peff_LH,color='k')


func(lbarr,ubarr2,'o')
#func(lbarr,ubarr1,'x')


ax.set_xlabel(r'$\omega$ (cm$^{-1}$)',fontsize=xl_fs)
ax.set_ylabel(r'$\beta \langle V \rangle$' ,fontsize=yl_fs)

ax.set_yticks([0.5,1,2,3,4])

ax.tick_params(axis='both',labelsize=ti_fs)
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')

fig.set_size_inches(4,4)
fig.savefig('/home/vgs23/Images/Quantum_avg_ceriotti.pdf',dpi=400,bbox_inches='tight',pad_inches=0.0)

plt.show()
