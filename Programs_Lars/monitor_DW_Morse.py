import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
import numpy as np
from PISC.dvr.dvr import DVR2D, DVR1D
from mylib.twoD import DVR2D_mod
#from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_2D_omp_updated
from mylib.twoD import PY_OTOC2D
from matplotlib import pyplot as plt
import os 
import time 
from mylib.testing import Check_DVR 
from Programs_Lars import Functions_DW_Morse_coupling as DMCC

##########Potential#########
#quartic bistable
lamda = 2.0   
#D=10.0  
D=9.375#3Vb
g = 0.08

##########Parameters########## (fixed)
#Grid
Lx=7.0
lbx = -Lx
ubx = Lx
lby = -4#Morse, will explode quickly #-2 not enough for alpha 0.35  
uby = 10#Morse, will explode quickly
m = 0.5
ngrid = 100
ngridx = ngrid
ngridy = ngrid
xg = np.linspace(lbx,ubx,ngridx)#ngridx+1 if "prettier"
yg = np.linspace(lby,uby,ngridy)
xgr,ygr = np.meshgrid(xg,yg)
T_c = 0.5*lamda/np.pi
#keep:
n_eig_tot=150#for DVR, n_eig_tot=150 (the saved one's, very few are 250)
N_trunc=100#how long m_arr is and therefore N_trunc 

##############################################################
###----------Start Options/Parameters looped over----------###
##############################################################

### MICROCANONIC OTOC: C_n
calc_Cn=False
C_n_range=np.arange(2,8,dtype=int)


#z_range=(0,0.25,0.5,1,1.5)
z_range= (0,1.5)
z_range=(1.5,)
#z_range=(0,)

#alpha_range=(0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75)
alpha_range=(0.25,0.35,0.45,0.55,0.65,0.75)

plot_eigenstates=False
plot_C_n_for_eigenstates=False
plot_C_T=True

#even more available at the end
where_exp_thermal=False
plot_DW2_M0=False
plot_DW3_M0=False
plot_DW2_M1=False

##############################################################
###----------End Options/Parameters looped over----------###
##############################################################
###Thermal OTOC to estimate where exponential(exp from 1 to 2), different for different temperatur
if(where_exp_thermal==True):
    alpha_range=(0.2,0.4,0.6)
    for alpha in alpha_range:
        for z in (0,1.5):
            fig, ax =plt.subplots(3)
            #########Parameters
            n_eigen=30
            N_trunc_thermal=50
            T=1#T=0.01 exp:[0.6,2.2], T=0.1 exp:[0.6,2.2], T=1 exp:[0.6,2.2] (exp for sure at 1.2 to 1.8)
            t_arr,OTOC=DMCC.get_thermal_otoc(alpha,z,D,lamda,g,ngridx,ngridy,lbx,ubx,lby,uby,m,ngrid,n_eig_tot,T=0.1,n_eigen=n_eigen,N_trunc=N_trunc_thermal)
            fig.suptitle(r'Thermal OTOC, log(OTOC), d(log(OTOC))/dt for N_trunc=%i, m=%i, T = %.2f,$\alpha$=%.2f' %(N_trunc_thermal,n_eigen,T,alpha))
            ax[0].plot(t_arr,OTOC,label='z=%.2f'%z )
            ax[1].plot(t_arr,np.log(OTOC),label='z=%.2f'%z )
            log_OTOC = np.log(OTOC)
            grad= np.gradient(log_OTOC,t_arr)
            ax[2].plot(t_arr,grad,label='z=%.2f'%z )
            plt.show(block=False)
#Plot Eigenstates
if(plot_eigenstates==True):
    contour_eigenstates=C_n_range
    for alpha in alpha_range:
        for z in z_range:
            pes = quartic_bistable(alpha,D,lamda,g,z)
            DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
            if(alpha>=0.9):
                vals,vecs=DMCC.get_EV_and_EVECS(alpha,D,lamda,g,z,ngrid,n_eig_tot,lbx,ubx,-2,6,m,ngridx,ngridy,pes,DVR)
            else:
                vals,vecs=DMCC.get_EV_and_EVECS(alpha,D,lamda,g,z,ngrid,n_eig_tot,lbx,ubx,lby,uby,m,ngridx,ngridy,pes,DVR)
            DMCC.contour_plots_eigenstates(pes,DVR,lbx,lby,ubx,uby,ngridx,ngridy,z,vecs,alpha,l_range=contour_eigenstates,remove_upper_part=True)
            #usually comment out
            #DMCC.plot_pot_E_Ecoupled_and_3D(pes,ngridx,ngridy,lbx,ubx,lby,uby,m,xg,yg,vals,z,plt_V_and_E=True,vecs=vecs)
    #plt.show(block=True)
    if(True):#plot one uncoupled 
        z=0
        alpha=0.5
        pes = quartic_bistable(alpha,D,lamda,g,z)
        DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
        vals,vecs=DMCC.get_EV_and_EVECS(alpha,D,lamda,g,0,ngrid,n_eig_tot,lbx,ubx,lby,uby,m,ngridx,ngridy,pes,DVR)
        DMCC.contour_plots_eigenstates(pes,DVR,lbx,lby,ubx,uby,ngridx,ngridy,z,vecs,alpha,l_range=contour_eigenstates,remove_upper_part=True)
    #plt.show(block=True)
    plt.show(block=False)
###plot the corresponding microcanonic OTOCS (corresponding to the previous eigenstates)
if(plot_C_n_for_eigenstates==True):
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    DMCC.plot_C_n_for_contour_eigenstates(alpha_range, z_range, C_n_range,t_arr,  C_n_loop_alpha,N_trunc,log=True,deriv=False)
    plt.show(block=True)
        

#plot C_T
if(plot_C_T==True):
    fig, ax=plt.subplots(2)
    n_eigen=50
    N_trunc=70
    T=T_c*0.8
    print(T_c)
    T=5
    z=1.5
    fig.suptitle(r'C$_T$, N_trunc=%i, N_eigen=%i, T=%i, z=%.1f' % (N_trunc,n_eigen,T,z),fontsize=12)
    for alpha in alpha_range:
        print()
        print('alpha = ',alpha)
        t_arr, OTOC_arr= DMCC.get_thermal_otoc(alpha,z,D,lamda,g,ngridx,ngridy,lbx,ubx,lby,uby,m,ngrid,n_eig_tot,T=T,n_eigen=n_eigen,N_trunc=N_trunc)
        OTOC_arr=np.real(OTOC_arr)
        ax[0].plot(t_arr,OTOC_arr,'--',label=r'$\alpha$= %.2f'%alpha)
        ax[1].plot(t_arr,np.log(OTOC_arr),'--',label=r'$\alpha$= %.2f'%alpha)
    t_arr, OTOC_arr= DMCC.get_thermal_otoc(0.5,0,D,lamda,g,ngridx,ngridy,lbx,ubx,lby,uby,m,ngrid,n_eig_tot,T=T,n_eigen=n_eigen,N_trunc=N_trunc)
    OTOC_arr=np.real(OTOC_arr)
    ax[0].plot(t_arr,OTOC_arr,'-',label='z=0')
    ax[1].plot(t_arr,np.log(OTOC_arr),'-',label='z=0')
    ax[0].legend()
    ax[1].legend()

###DEPRICATED, should still work though and might be interesting
###microcanonic, plotting varying different parameters
if(calc_Cn==True):
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    DMCC.plot_C_n_var_n_fix_alpha_z(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha)
    DMCC.plot_C_n_var_z_fix_alpha_n(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha)
    DMCC.plot_C_n_var_alpha_fix_z_n(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha)

###DEPRICATED, should still work though and might be interesting
### specific plotting
log=False
Specific_plotting=False
plot_deriv=False
if(Specific_plotting==True):
    fig,ax =plt.subplots(int(len(z_range)/2),2,sharex='all',sharey='all')
    fig.suptitle(r'b$_{20}$ for different $\alpha$ and z',fontsize=15)
    for alpha in alpha_range:
        if(alpha>=0.34):
            t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),z_range,((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
            z_cntr=0
            for axs in ax.flat:
                if(log==True):
                    axs.plot(t_arr,np.abs(b_nm_loop_alpha[0][z_cntr][0]),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1) 
                else:
                    axs.plot(t_arr,np.abs(b_nm_loop_alpha[0][z_cntr][0]),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)       
                axs.set_title(r'z=%.2f' % z_range[z_cntr])
                z_cntr +=1
                axs.legend(fontsize=6)       
        if(alpha==0.3):########WORKS ONLY FOR 0,0.25,0.5,1
            z_cntr=0
            t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0,),((4,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
            for axs in ax.flat:
                if(z_cntr==0):
                    if(log==True):
                        axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[0][0][0])),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                    else:
                        axs.plot(t_arr,np.abs(b_nm_loop_alpha[0][0][0]),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==1):
                    t_arr,b_nm_loop_alpha1=DMCC.compute_b_nm((alpha,),(0.25,),((3,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    if(log==True):
                        axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[0][0][0])),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                    else:
                        axs.plot(t_arr,np.abs(b_nm_loop_alpha[0][0][0]),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==2):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0.5,),((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    if(log==True):
                        axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[0][0][0])),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                    else:
                        axs.plot(t_arr,np.abs(b_nm_loop_alpha[0][0][0]),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==3):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(1,),((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    if(log==True):
                        axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[0][0][0])),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                    else:
                        axs.plot(t_arr,np.abs(b_nm_loop_alpha[0][0][0]),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                z_cntr +=1
        if(alpha==0.252):
            z_cntr=0
            t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0,0.25,0.5),((4,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
            for axs in ax.flat:
                if(z_cntr<3):
                    if(log==True):
                        axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[0][z_cntr][0])),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                    else:
                        axs.plot(t_arr,np.abs(b_nm_loop_alpha[0][0][0]),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                else:
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(1,),((6,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    if(log==True):
                        axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[0][0][0])),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                    else:
                        axs.plot(t_arr,np.abs(b_nm_loop_alpha[0][0][0]),'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1) 
                z_cntr +=1
        
    plt.show()
if(plot_deriv==True):
    ###Derivative
    fig,ax =plt.subplots(int(len(z_range)/2),2,sharex='all',sharey='all')
    fig.suptitle(r'derivative of log(b$_{20}$) for different $\alpha$ and z',fontsize=15)
    offset=10
    for alpha in alpha_range:
        if(alpha>=0.34):
            t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),z_range,((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
            z_cntr=0
            for axs in ax.flat:
                tmp = np.log(np.abs(b_nm_loop_alpha[0][z_cntr][0]))
                grad=np.gradient(tmp,t_arr)
                axs.plot(t_arr[offset:],grad[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1) 
                axs.set_title(r'z=%.2f' % z_range[z_cntr])
                z_cntr +=1
                axs.legend(fontsize=6)
        if(alpha==0.3):
            z_cntr=0
            t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0,),((4,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
            for axs in ax.flat:
                if(z_cntr==0):
                    tmp=np.log(np.abs(b_nm_loop_alpha[0][0][0]))
                    grad= np.gradient(tmp,t_arr)###DEPRICATED should still work though
                    axs.plot(t_arr[offset:],grad[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==1):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0.25,),((3,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    tmp=np.log(np.abs(b_nm_loop_alpha[0][0][0]))
                    grad= np.gradient(tmp,t_arr)
                    axs.plot(t_arr[offset:],grad[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==2):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0.5,),((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    tmp=np.log(np.abs(b_nm_loop_alpha[0][0][0]))
                    grad= np.gradient(tmp,t_arr)
                    axs.plot(t_arr[offset:],grad[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==3):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(1,),((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    tmp=np.log(np.abs(b_nm_loop_alpha[0][0][0]))
                    grad= np.gradient(tmp,t_arr)
                    axs.plot(t_arr[offset:],grad[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                z_cntr +=1
    ###Normal to compare
    fig,ax =plt.subplots(int(len(z_range)/2),2,sharex='all',sharey='all')
    fig.suptitle(r'b$_{20}$ for different $\alpha$ and z',fontsize=15)
    for alpha in alpha_range:
        if(alpha>=0.34):
            t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),z_range,((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
            z_cntr=0
            for axs in ax.flat:
                tmp2=np.abs(b_nm_loop_alpha[0][z_cntr][0])
                axs.plot(t_arr[offset:],tmp2[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1) 
                axs.set_title(r'z=%.2f' % z_range[z_cntr])
                z_cntr +=1
                axs.legend(fontsize=6)
        if(alpha==0.3):
            z_cntr=0
            t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0,),((4,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
            for axs in ax.flat:
                if(z_cntr==0):
                    tmp=np.abs(b_nm_loop_alpha[0][0][0])
                    axs.plot(t_arr[offset:],tmp[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==1):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0.25,),((3,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    tmp=np.abs(b_nm_loop_alpha[0][0][0])
                    axs.plot(t_arr[offset:],tmp[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==2):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0.5,),((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    tmp=np.abs(b_nm_loop_alpha[0][0][0])
                    axs.plot(t_arr[offset:],tmp[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==3):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(1,),((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    tmp=np.abs(b_nm_loop_alpha[0][0][0])
                    axs.plot(t_arr[offset:],tmp[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                    z_cntr +=1
    ### log to compare
    fig,ax =plt.subplots(int(len(z_range)/2),2,sharex='all',sharey='all')
    fig.suptitle(r'log(b$_{20}$) for different $\alpha$ and z',fontsize=15)
    for alpha in alpha_range:
        if(alpha>=0.34):
            t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),z_range,((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
            z_cntr=0
            for axs in ax.flat:
                tmp2=np.log(np.abs(b_nm_loop_alpha[0][z_cntr][0]))
                axs.plot(t_arr[offset:],tmp2[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1) 
                axs.set_title(r'z=%.2f' % z_range[z_cntr])
                z_cntr +=1
                axs.legend(fontsize=6)
        if(alpha==0.3):
            z_cntr=0
            t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0,),((4,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
            for axs in ax.flat:
                if(z_cntr==0):
                    tmp=np.log(np.abs(b_nm_loop_alpha[0][0][0]))
                    axs.plot(t_arr[offset:],tmp[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==1):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0.25,),((3,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    tmp=np.log(np.abs(b_nm_loop_alpha[0][0][0]))
                    axs.plot(t_arr[offset:],tmp[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==2):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(0.5,),((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    tmp=np.log(np.abs(b_nm_loop_alpha[0][0][0]))
                    axs.plot(t_arr[offset:],tmp[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                elif(z_cntr==3):
                    t_arr,b_nm_loop_alpha=DMCC.compute_b_nm((alpha,),(1,),((2,0),),D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
                    tmp=np.log(np.abs(b_nm_loop_alpha[0][0][0]))
                    axs.plot(t_arr[offset:],tmp[offset:],'--',label=r'$\alpha$=%.3f' % alpha,linewidth=1)
                    z_cntr +=1
    plt.show()
###################
####Specific Cn####
###################
###for DW 2, Morse 0 (counted from 0)
if(plot_DW2_M0==True):
    ###plot some specific C_n for different alphas
    z_range=(1.5,)
    fig,ax = plt.subplots(3)#C_n, log(C_n), dlog(C_n)/dt
    fig.suptitle('DW 2, Morse 0')

    #smaller values
    C_n_range=(4,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
    #alpha_range=(0.2,0.25,0.3)
    alpha_range=(0.25,)
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    for alpha in alpha_range:
        ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
        ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
        tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
        alpha_counter+=1

    #intermediate values
    z_range=(1.5,)
    C_n_range=(2,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
    #alpha_range=(0.35,0.4,0.45,0.55,0.65,0.75)
    alpha_range=(0.35,0.45,0.55,0.65,0.75)
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    fig.suptitle('DW 2, Morse 0')
    for alpha in alpha_range:
        ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
        ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
        tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr,np.real(grad),'--',label=r'$\alpha$ %.3f'%alpha)
        alpha_counter+=1

    #uncoupled
    alpha_range=(0.5,)#arbitray, just have to choose correct n
    C_n_range=(2,)
    z_range=(0,)
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    for alpha in alpha_range:
        ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),label='z=0')
        ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),label='z=0')
        tmp=np.real(np.log(C_n_loop_alpha[alpha_counter][0][0]))
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr,grad,label='z=0')
        alpha_counter+=1
    for axs in ax.flat:
        axs.legend()
    plt.draw()
    plt.pause(0.001)

###for DW 2, Morse 1 (counted from 0)
if(plot_DW2_M1==True):
###plot some specific C_n for different alphas
    fig,ax = plt.subplots(3)#C_n, log(C_n), dlog(C_n)/dt
    fig.suptitle('DW 2, Morse 1')
    z_range=(1.5,)
    if(True):###lower values
        C_n_range=(8,)#for the alpha's below this C_n is the one of interest
        #alpha_range=(0.3,0.31,0.32,0.33,0.34,0.35)

        alpha_range=(0.25,0.3,0.35)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
            tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
            alpha_counter+=1

    if(True):#intermediate values
        C_n_range=(7,)#for the alpha's below this C_n is the one of interest
        alpha_range=(0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75)
        #alpha_range=(0.4,0.5,0.6,0.7)
        #alpha_range=(0.45,0.55,0.65,0.75)
        #alpha_range = np.linspace(0.5,0.6,11)
        #alpha_range=np.linspace(0.5,0.55,6)
        #alpha_range=np.linspace(0.5,0.6,6)
        #alpha_range=np.concatenate(alpha_range,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
            tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
            alpha_counter+=1
    if(False):#higher values: levels change between 0.75 and 0.9 again
        alpha_range=(0.9,1.147)
        C_n_range=(9,)
        z_range=(1.5,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,-2,6,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
            tmp=np.real(np.log(C_n_loop_alpha[alpha_counter][0][0]))
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,grad,'--',label='alpha %.3f'%alpha)
            alpha_counter+=1
    if(True):
        alpha_range=(0.5,)#arbitray, just have to choose correct n
        C_n_range=(7,)
        z_range=(0,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),label='z=0')
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),label='z=0')
            tmp=np.real(np.log(C_n_loop_alpha[alpha_counter][0][0]))
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,grad,label='z=0')
            alpha_counter+=1
    for axs in ax.flat:
        axs.legend()
    plt.draw()
    plt.pause(0.001)
###for DW 3, Morse 0 (counted from 0) we do not expect exponential
if(plot_DW3_M0==True):
    ###plot some specific C_n for different alphas
    z_range=(1.5,)
    C_n_range=(3,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
    alpha_range=(0.9,1.0,1.1)
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,-2,6,m,n_eig_tot,N_trunc)
    fig,ax = plt.subplots(3)#C_n, log(C_n), dlog(C_n)/dt
    fig.suptitle('DW 3, Morse 0, expect no exp')
    for alpha in alpha_range:
        ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
        ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
        tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
        alpha_counter+=1
    
    z_range=(1.5,)
    C_n_range=(3,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
    alpha_range=(0.5,0.6,0.7,0.8)
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    for alpha in alpha_range:
        ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
        ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
        tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
        alpha_counter+=1
    if(True):
        z_range=(1.5,)
        C_n_range=(7,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
        alpha_range=(0.2,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
            tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
            alpha_counter+=1
    if(True):
        alpha_range=(0.5,)#arbitray, just have to choose correct n
        C_n_range=(4,)
        z_range=(0,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),label='z=0')
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),label='z=0')
            tmp=np.real(np.log(C_n_loop_alpha[alpha_counter][0][0]))
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,grad,label='z=0')
            alpha_counter+=1
    for axs in ax.flat:
        axs.legend()
    plt.draw()
    plt.pause(0.001)
###for DW 4, Morse 0 (counted from 0)
if(False):
    ###plot some specific C_n for different alphas

    z_range=(1.5,)
    C_n_range=(4,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
    alpha_range=(0.9,1.0,1.1)
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,-2,6,m,n_eig_tot,N_trunc)
    fig,ax = plt.subplots(3)#C_n, log(C_n), dlog(C_n)/dt
    fig.suptitle('DW 4, Morse 0')
    for alpha in alpha_range:
        ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
        ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
        tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
        alpha_counter+=1
    if(True):
        z_range=(1.5,)
        C_n_range=(12,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
        alpha_range=(0.2,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
            tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
            alpha_counter+=1
    if(True):
        z_range=(1.5,)
        C_n_range=(6,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
        alpha_range=(0.5,0.6,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
            tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
            alpha_counter+=1
    if(True):
        alpha_range=(0.5,)#arbitray, just have to choose correct n
        C_n_range=(6,)
        z_range=(0,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),label='z=0')
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),label='z=0')
            tmp=np.real(np.log(C_n_loop_alpha[alpha_counter][0][0]))
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,grad,label='z=0')
            alpha_counter+=1
    for axs in ax.flat:
        axs.legend()
    plt.draw()
    plt.pause(0.001)
if(False):#####DW GS(0) Morse 1 ...maybe no for GS at 1 
    ###plot some specific C_n for different alphas
    z_range=(1.5,)
    C_n_range=(2,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
    alpha_range=(0.2,0.3)
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    fig,ax = plt.subplots(3)#C_n, log(C_n), dlog(C_n)/dt
    fig.suptitle('DW sym, Morse 1')
    for alpha in alpha_range:
        ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
        ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
        tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
        alpha_counter+=1
    if(True):
        z_range=(1.5,)
        C_n_range=(4,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
        alpha_range=(0.4,0.5,0.6,0.7) #bei 0.4 nicht sicher
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
            tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
            alpha_counter+=1

        alpha_range=(0.5,)#arbitray, just have to choose correct n
        C_n_range=(3,)
        z_range=(0,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),label='z=0')
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),label='z=0')
            tmp=np.real(np.log(C_n_loop_alpha[alpha_counter][0][0]))
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,grad,label='z=0')
            alpha_counter+=1
    for axs in ax.flat:
        axs.legend()
    plt.draw()
    plt.pause(0.001)
if(False):#####DW GS(1) Morse 1 ...maybe no for GS at 1 
    ###plot some specific C_n for different alphas
    z_range=(1.5,)
    C_n_range=(3,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
    alpha_range=(0.2,0.3,0.4)#bei 0.4 not sure
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    fig,ax = plt.subplots(3)#C_n, log(C_n), dlog(C_n)/dt
    fig.suptitle('DW anti, Morse 1, not sure for 0.4 also not for sym')
    for alpha in alpha_range:
        ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
        ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
        tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
        alpha_counter+=1
    if(True):
        z_range=(1.5,)
        C_n_range=(5,)#CAREFUL ONLY ONE HERE!!!! otw loop createn
        alpha_range=(0.5,0.6,0.7)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),'--',label='alpha %.3f'%alpha)
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),'--',label='alpha %.3f'%alpha)
            tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,np.real(grad),'--',label='alpha %.3f'%alpha)
            alpha_counter+=1

        alpha_range=(0.5,)#arbitray, just have to choose correct n
        C_n_range=(5,)
        z_range=(0,)
        alpha_counter=0
        t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
        for alpha in alpha_range:
            ax[0].plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][0][0]),label='z=0')
            ax[1].plot(t_arr,np.real(np.log(C_n_loop_alpha[alpha_counter][0][0])),label='z=0')
            tmp=np.real(np.log(C_n_loop_alpha[alpha_counter][0][0]))
            grad=np.gradient(tmp,t_arr)
            ax[2].plot(t_arr,grad,label='z=0')
            alpha_counter+=1
    for axs in ax.flat:
        axs.legend()
    plt.draw()
    plt.pause(0.001)
plt.show(block=True)