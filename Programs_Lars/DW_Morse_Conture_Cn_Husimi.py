from cProfile import label
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

#######################################
###----------FUNCTIONS END----------###
#######################################

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
#keep:
n_eig_tot=150#for DVR, n_eig_tot=150 (the saved one's, very few are 250)
N_trunc=100#how long m_arr is and therefore N_trunc 

##############################################################
###----------Start Options/Parameters looped over----------###
##############################################################

### MICROCANONIC OTOC: C_n
calc_Cn=True
C_n_range=np.arange(4,10,dtype=int)
print(C_n_range)
#C_n_range =(2,)

#z_range=(0,0.25,0.5,1,1.5)
#z_range= (0,1.5)
z_range=(1.5,)
#alpha_range=(0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37)
alpha_range=(0.34,0.35,0.352,0.355,0.357,0.36)
#alpha_range=(0.363,)#For D=10 maybe interesting

contour_eigenstates=C_n_range

if(False):###Thermal OTOC to estimate where exponential(exp from 1 to 2), different for different temperatures
    for alpha in alpha_range:
        for z in (0,1.5):
            fig, ax =plt.subplots(3)
            #########Parameters
            n_eigen=30
            N_trunc_thermal=50
            t_arr,OTOC=DMCC.get_thermal_otoc(alpha,z,D,lamda,g,ngridx,ngridy,lbx,ubx,lby,uby,m,ngrid,n_eig_tot,T=1,n_eigen=n_eigen,N_trunc=N_trunc_thermal)
            fig.suptitle('OTOC, log(OTOC), d(log(OTOC))/dt for N_trunc=%i and %i -states, T = 1' %(N_trunc_thermal,n_eigen))
            ax[0].plot(t_arr,OTOC,label='z=%.2f'%z )
            ax[1].plot(t_arr,np.log(OTOC),label='z=%.2f'%z )
            log_OTOC = np.log(OTOC)
            grad= np.gradient(log_OTOC,t_arr)
            ax[2].plot(t_arr[15:],grad[15:],label='z=%.2f'%z )
            plt.show(block=False)
#Plot the eigenstates 
alpha_range=(0.36,0.38,0.4,0.42,0.44,0.46,0.5)
for alpha in alpha_range:
    for z in z_range:
        pes = quartic_bistable(alpha,D,lamda,g,z)
        DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
        vals,vecs=DMCC.get_EV_and_EVECS(alpha,D,lamda,g,z,ngrid,n_eig_tot,lbx,ubx,lby,uby,m,ngridx,ngridy,pes,DVR)
        DMCC.contour_plots_eigenstates(pes,DVR,lbx,lby,ubx,uby,ngridx,ngridy,z,vecs,alpha,l_range=contour_eigenstates,remove_upper_part=True)
plt.show(block=False)

#plot the corresponding microcanonic OTOCS
t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
DMCC.plot_C_n_for_contour_eigenstates(alpha_range, z_range, C_n_range,t_arr,  C_n_loop_alpha,N_trunc,log=False,deriv=False)
plt.show()

###plot some specific C_n for different alphas
if(False):
    fig,ax = plt.subplots(3)#C_n, log(C_n), dlog(C_n)/dt
    
    C_n_range=(7,)#for the alpha's below this C_n is the one of interest
    alpha_range=(0.3,0.31,0.32,0.33,0.34,0.35)
    alpha_range=(0.3,0.35)
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    for alpha in alpha_range:
        ax[0].plot(t_arr,C_n_loop_alpha[alpha_counter][0][0],label='alpha %.3f'%alpha)
        ax[1].plot(t_arr,np.log(C_n_loop_alpha[alpha_counter][0][0]),label='alpha %.3f'%alpha)
        tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr[15:],grad[15:],label='alpha %.3f'%alpha)
        alpha_counter+=1
    
    C_n_range=(8,)#for the alpha's below this C_n is the one of interest
    alpha_range=(0.4,0.46,0.5)
    alpha_counter=0
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    for alpha in alpha_range:
        ax[0].plot(t_arr,C_n_loop_alpha[alpha_counter][0][0],label='alpha %.3f'%alpha)
        ax[1].plot(t_arr,np.log(C_n_loop_alpha[alpha_counter][0][0]),label='alpha %.3f'%alpha)
        tmp=np.log(C_n_loop_alpha[alpha_counter][0][0])
        grad=np.gradient(tmp,t_arr)
        ax[2].plot(t_arr[15:],grad[15:],label='alpha %.3f'%alpha)
        alpha_counter+=1
    for axs in ax.flat:
        axs.legend()
    plt.show(block=True)

###microcanonic, plotting varying different parameters
if(calc_Cn==True):
    t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
    DMCC.plot_C_n_var_n_fix_alpha_z(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha)
    DMCC.plot_C_n_var_z_fix_alpha_n(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha)
    DMCC.plot_C_n_var_alpha_fix_z_n(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha)

### specific plotting
log=False
Specific_plotting=False
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
plot_deriv=False
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
                    grad= np.gradient(tmp,t_arr)
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
