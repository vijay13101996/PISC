import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
import numpy as np
from PISC.dvr.dvr import DVR2D, DVR1D
from mylib.twoD import DVR2D_mod
#from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_2D_omp_updated
from Programs_Lars.mylib.twoD import PY_OTOC2D
from matplotlib import pyplot as plt
import os 
import time 
from mylib.testing import Check_DVR 

def plot_pot_E_Ecoupled_and_3D(pes,ngridx,ngridy,lbx,ubx,lby,uby,m,xg,yg,vals,z):
#plot x and y potentials, eigenvalues and eigenvalues of the coupled problem, and 3D potential
    from PISC.dvr.dvr import DVR1D
    pot_DW= lambda a: pes.potential_xy(a,0)
    DVR_DW = DVR1D(2*ngridx,lbx,ubx,m,pot_DW)
    pot_Morse= lambda a: pes.potential_xy(0,a)-pes.potential_xy(0,0)
    DVR_Morse = DVR1D(2*ngridy,lby,2*uby,m,pot_Morse)
    vals_M, vecs_M= DVR_Morse.Diagonalize()
    vals_DW, vecs_DW= DVR_DW.Diagonalize()
    #Check_DVR.plot_V_and_E(xg,yg,vals=vals,vecs=vecs,vals_x=vals_DW,vals_y= vals_M ,pot=pes.potential_xy,z=z,threeDplot=True)
    Check_DVR.plot_DW_MO_DW_inter(xg,yg,pot=pes.potential_xy,vals=vals,vals_x=vals_DW,vals_y= vals_M, z=z)

def plot_pot_and_Eigenvalues(pes,DVR,lbx,lby,ubx,uby,ngridx,ngridy,z,vecs,lrange=(2,)):
#some inside/intuition about potential and Eigenvalues
    for l in lrange:
        xg = np.linspace(lbx,ubx,ngridx)#ngridx+1 if "prettier"
        yg = np.linspace(lby,uby,ngridy)
        xgr,ygr = np.meshgrid(xg,yg)
        contour_eigenstate=l
        plt.title('Pot_Contour, Eigenstate: %i, z = %.2f '% (contour_eigenstate,z))
        plt.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,5,0.25))
        plt.imshow(DVR.eigenstate(vecs[:,contour_eigenstate])**2,origin='lower')#cannot be fit to actual axis #plt.imshow(np.reshape(vecs[:,12],(ngridx+1,ngridy+1),'F')**2,origin='lower')#same
        #plt.contour(xg,yg,pes.potential_xy(xgr,ygr),levels=np.arange(0,5,0.25))#in case I need the values
        plt.show()
def contour_plots_eigenstates(pes,DVR,lbx,lby,ubx,uby,ngridx,ngridy,z,vecs,alpha,l_range=(2,),remove_upper_part=True):
#some inside/intuition about potential and Eigenvalues
    fig, ax = plt.subplots(int(len(l_range)/2),2)
    l_cntr=0
    for axs in ax.flat: 
        xg = np.linspace(lbx,ubx,ngridx)#ngridx+1 if "prettier"
        yg = np.linspace(lby,uby,ngridy)
        xgr,ygr = np.meshgrid(xg,yg)
        contour_eigenstate=l_range[l_cntr]
        fig.suptitle(r'Eigenstates for z = %.2f, $\alpha$= %.3f'% (z,alpha))
        #axs.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,7.5,1.5))
        axs.imshow(DVR.eigenstate(vecs[:,contour_eigenstate])**2,origin='lower')#cannot be fit to actual axis #plt.imshow(np.reshape(vecs[:,12],(ngridx+1,ngridy+1),'F')**2,origin='lower')#same
        #axs.set_title(r'$\psi_%i$'%(l_cntr))
        axs.set_title(r'$n=%i$'%(l_cntr))
        axs.axis('off')
        if(remove_upper_part==True):
            axs.set_ylim([0,55])
        l_cntr +=1
    plt.draw()
    #plt.show(block=False)
def plot_C_n_for_contour_eigenstates(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha,N_trunc,log=True,deriv=True):
    for alpha_counter in range(len(alpha_range)):
        for z_counter in range(len(z_range)):
            print('alpha, z = %.2f,%.2f' %(alpha_range[alpha_counter],z_range[z_counter]))
            if(log==True):
                fig_log, ax_log = plt.subplots(int(len(C_n_range)/2),2,sharex='all',sharey='all')
            if(deriv==True):
                fig_deriv, ax_deriv = plt.subplots(int(len(C_n_range)/2),2,sharex='all',sharey='all')
            fig, ax= plt.subplots(int(len(C_n_range)/2),2,sharex='all',sharey='all')
            n_counter=0
            for axs in ax.flat:
                axs.plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][z_counter][n_counter]),'--',label='n = %i' % C_n_range[n_counter],linewidth=1)
                axs.set_title(r'$n=%i$'%(n_counter))
                n_counter+=1
            fig.suptitle(r'C_n for $\alpha$ = %.3f, z = %.2f, N_trunc = %i' %(alpha_range[alpha_counter],z_range[z_counter],N_trunc))
            if(log==True):
                n_counter=0
                for axs in ax_log.flat:
                    tmp=np.log(np.abs(C_n_loop_alpha[alpha_counter][z_counter][n_counter]))
                    axs.plot(t_arr[15:],tmp[15:],'--',label='n = %i' % C_n_range[n_counter],linewidth=1)
                    axs.set_title(r'$n=%i$'%(n_counter))
                    n_counter+=1
                fig_log.suptitle(r'log(C_n) for $\alpha$ = %.3f, z = %.2f, N_trunc = %i' %(alpha_range[alpha_counter],z_range[z_counter],N_trunc))
            if(deriv==True):
                n_counter=0
                for axs in ax_deriv.flat:
                    tmp=np.log(np.abs(C_n_loop_alpha[alpha_counter][z_counter][n_counter]))
                    grad=np.gradient(tmp,t_arr)
                    axs.plot(t_arr[15:],grad[15:],'--',label='n = %i' % C_n_range[n_counter],linewidth=1)
                    axs.set_title(r'$n=%i$'%(n_counter))
                    n_counter+=1
                fig_deriv.suptitle(r'dlog(C_n)/dt for $\alpha$ = %.3f, z = %.2f, N_trunc = %i' %(alpha_range[alpha_counter],z_range[z_counter],N_trunc))
            plt.show(block=False)
def get_thermal_otoc(alpha,z,D,lamda,g,ngridx,ngridy,lbx,ubx,lby,uby,m,ngrid,n_eig_tot,T=1,n_eigen=50,N_trunc=70):
    beta=1/T
    #n_eigen=50#up to which n C_n should be calculated
    #N_trunc=70#how long m_arr is and therefore N_trunc 
    print('Calculating thermal OTOC with n_eig = %i N_trunc = %i' % (n_eigen, N_trunc))
    s_time=time.time()
    ##########Arrays
    pes = quartic_bistable(alpha,D,lamda,g,z)
    DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
    x_arr = DVR.pos_mat(0)#defines grid points on x
    k_arr = np.arange(N_trunc) +1 #what
    m_arr = np.arange(N_trunc) +1 #what
    t_arr = np.linspace(0.0,15.0,150) # time interval
    #t_arr = np.linspace(0.0,5.0,80) # time interval
    OTOC_arr = np.zeros_like(t_arr,dtype='complex') #Store OTOC 
    vals,vecs=get_EV_and_EVECS(alpha,D,lamda,g,z,ngrid,n_eig_tot,lbx,ubx,lby,uby,m,ngridx,ngridy,pes,DVR)
    OTOC_arr = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr) 
    print('Time OTOC calc: %.2f s' %(time.time()-s_time))
    return t_arr, OTOC_arr

def get_EV_and_EVECS(alpha,D,lamda,g,z,ngrid,n_eig_tot,lbx,ubx,lby,uby,m,ngridx,ngridy,pes,DVR):
    import os 
    from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
    #from PISC.dvr.dvr import DVR2D
    ##########Save so only has to be done once
    DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
    potkey = 'DW_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
    fname = 'Eigen_basis_{}_ngrid_{}_n_eig_tot_{}'.format(potkey,ngrid,n_eig_tot)
    path = os.path.dirname(os.path.abspath(__file__))

    #####Check if data is available
    diag= True
    try:
        with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'r') as f:
            param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot} 
            if str(param_dict) in f.read():
                print('Retrieving data...')
                diag=False
            else:    
                print('Diagonalization of (%i x %i) matrix in:' % (ngridx*ngridy,ngridx*ngridy))
    except:
        diag=True
        print('Diagonalization of (%i x %i) matrix in:' % (ngridx*ngridy,ngridx*ngridy))
    
    #####Diagonalize if not available
    if(diag):
        param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
        vals, vecs= DVR.Diagonalize(neig_total=n_eig_tot)
        with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a+') as f:	
            f.write('\n'+str(param_dict))
        store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
        store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
        print('Data stored!')

    #####read out arrays s.t.has to be diag only once
    vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
    vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    print('... data retrieved!')
    return vals,vecs

def compute_C_n(alpha_range,z_range,C_n_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc):
    from PISC.dvr.dvr import DVR2D
    from PISC.engine import OTOC_f_2D_omp_updated
    C_n_loop_alpha = []
    for alpha in alpha_range:
        C_n_loop_z = []
        for z in z_range:
            print()
            print('alpha = %.3f, z = %.2f' % (alpha,z))
            C_n_loop_n = []

            pes = quartic_bistable(alpha,D,lamda,g,z)
            DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
            
            #Diagonalize or Retrieve Eigenvalues
            vals,vecs=get_EV_and_EVECS(alpha,D,lamda,g,z,ngrid,n_eig_tot,lbx,ubx,lby,uby,m,ngridx,ngridy,pes,DVR)
            
            ###Give some intuituion about potential, but not necessary
            #plot_pot_E_Ecoupled_and_3D(pes,ngridx,ngridy,lbx,ubx,lby,uby,m,xg,yg,vals,z)
            #plot_pot_and_Eigenvalues(pes,DVR,lbx,lby,ubx,uby,ngridx,ngridy,z,vecs,lrange=(2,))
            
            ###Arrays
            x_arr = DVR.pos_mat(0)#defines grid points on x
            k_arr = np.arange(N_trunc) +1 #what
            m_arr = np.arange(N_trunc) +1 #what
            t_arr = np.linspace(0.0,5.0,150) # time intervaln=2
            #t_arr = np.linspace(0.0,5.0,80) # time interval
            OTOC_arr = np.zeros_like(t_arr,dtype='complex') #Store OTOC 

            ###Mircocanonic OTOC
            for n in C_n_range:#should be the n=2 for uncoupled 
                C_n = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)
                C_n_loop_n.append(C_n)
        
            C_n_loop_z.append(C_n_loop_n)
        C_n_loop_alpha.append(C_n_loop_z)
    return t_arr,C_n_loop_alpha

def plot_C_n_var_n_fix_alpha_z(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha):
    for alpha_counter in range(len(alpha_range)):
        for z_counter in range(len(z_range)):
            fig, ax= plt.subplots(1)
            for n_counter in range(len(C_n_range)):
                ax.plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][z_counter][n_counter]),'--',label='n = %i' % C_n_range[n_counter],linewidth=1)
            ax.set_title(r'C_n for $\alpha$ = %.3f, z = %.2f' %(alpha_range[alpha_counter],z_range[z_counter]))
            ax.legend()
            plt.show()
def plot_C_n_var_z_fix_alpha_n(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha):
    for alpha_counter in range(len(alpha_range)):
        for n_counter in range(len(C_n_range)):
            fig, ax= plt.subplots(1)
            for z_counter in range(len(z_range)):
                ax.plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][z_counter][n_counter]),'--',label='z = %.2f' % z_range[z_counter],linewidth=1)
            ax.set_title(r'C_n for $\alpha$ = %.3f, n = %i' %(alpha_range[alpha_counter],C_n_range[n_counter]))
            ax.legend()
            plt.show()

def plot_C_n_var_alpha_fix_z_n(alpha_range, z_range, C_n_range,t_arr, C_n_loop_alpha):
    for z_counter in range(len(z_range)):
        for n_counter in range(len(C_n_range)):
            fig, ax= plt.subplots(1)
            for alpha_counter in range(len(alpha_range)):
                ax.plot(t_arr,np.abs(C_n_loop_alpha[alpha_counter][z_counter][n_counter]),'--',label=r'$\alpha$ = %.3f' % alpha_range[alpha_counter],linewidth=1)
            ax.set_title(r'C_n for n = %i, z = %.2f' %(C_n_range[n_counter],z_range[z_counter]))
            ax.legend()
            plt.show()

def compute_b_nm(alpha_range,z_range,b_nm_range,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc):
    from PISC.dvr.dvr import DVR2D
    from PISC.engine import OTOC_f_2D_omp_updated
    b_nm_loop_alpha = []
    for alpha in alpha_range:
        b_nm_loop_z = []
        for z in z_range:
            print()
            print('alpha = %.3f, z = %.2f' % (alpha,z))

            pes = quartic_bistable(alpha,D,lamda,g,z)
            DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
            
            #Diagonalize or Retrieve Eigenvalues
            vals,vecs=get_EV_and_EVECS(alpha,D,lamda,g,z,ngrid,n_eig_tot,lbx,ubx,lby,uby,m,ngridx,ngridy,pes,DVR)
            
            ###Give some intuituion about potential, but not necessary
            #plot_pot_E_Ecoupled_and_3D(pes,ngridx,ngridy,lbx,ubx,lby,uby,m,xg,yg,vals,z)
            #plot_pot_and_Eigenvalues(pes,DVR,lbx,lby,ubx,uby,ngridx,ngridy,z,vecs,lrange=(2,))
            
            ###Arrays
            x_arr = DVR.pos_mat(0)#defines grid points on x
            k_arr = np.arange(N_trunc) +1 #what
            m_arr = np.arange(N_trunc) +1 #what
            t_arr = np.linspace(0.0,5.0,150) # time intervaln=2
            #t_arr = np.linspace(0.0,5.0,80) # time interval
            OTOC_arr = np.zeros_like(t_arr,dtype='complex') #Store OTOC 
            b_arr = np.zeros_like(OTOC_arr)#store B
            pos_mat = np.zeros_like(k_arr)
            X=np.zeros((N_trunc,N_trunc))

            for i in range(N_trunc):
                for j in range(N_trunc):
                    X[i,j] = OTOC_f_2D_omp_updated.otoc_tools.pos_matrix_elts(vecs,x_arr,DVR.dx,DVR.dy,i+1,j+1,pos_mat)
            b_nm_loop_nm = []
            for nm in b_nm_range:
                b_nm = PY_OTOC2D.b_nm(t=t_arr,n=nm[0],m=nm[1],X=X,w=vals)
                b_nm_loop_nm.append(b_nm)
            b_nm_loop_z.append(b_nm_loop_nm)
        b_nm_loop_alpha.append(b_nm_loop_z)
    return t_arr,b_nm_loop_alpha

def plot_b_nm_var_nm_fix_alpha_z(alpha_range, z_range, b_nm_range,t_arr, b_nm_loop_alpha,plot_alot=False,log=False):
    if((len(alpha_range)-1) %2): 
        for z_counter in range(len(z_range)):
            fig, ax= plt.subplots(int(len(alpha_range)/2),2,sharex='all',sharey='all')
            for nm_counter in range(len(b_nm_range)):
                    fig.suptitle(r'b_nm for z = %.2f' %(z_range[z_counter]))
                    if(b_nm_range[nm_counter]==(2,0)):
                        alpha_cntr=0
                        for axs in ax.flat:
                            if(log==True):
                                axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[alpha_cntr][z_counter][nm_counter])),'-',label='(n,m) = (%i,%i)' % (b_nm_range[nm_counter]),linewidth=1)
                            else:
                                axs.plot(t_arr,np.abs(b_nm_loop_alpha[alpha_cntr][z_counter][nm_counter]),'-',label='(n,m) = (%i,%i)' % (b_nm_range[nm_counter]),linewidth=1)
                            axs.set_title(r'$\alpha$ = %.3f' %(alpha_range[alpha_cntr]),fontsize=8)
                            axs.legend(fontsize=6)
                            alpha_cntr +=1
                    else:
                        alpha_cntr=0
                        for axs in ax.flat:
                            if(log==True):
                                axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[alpha_cntr][z_counter][nm_counter])),'--',label='(n,m) = (%i,%i)' % (b_nm_range[nm_counter]),linewidth=1)
                            else:
                                axs.plot(t_arr,np.abs(b_nm_loop_alpha[alpha_cntr][z_counter][nm_counter]),'--',label='(n,m) = (%i,%i)' % (b_nm_range[nm_counter]),linewidth=1)
                            axs.set_title(r'$\alpha$ = %.3f' %(alpha_range[alpha_cntr]),fontsize=8)
                            axs.legend(fontsize=6)
                            alpha_cntr +=1

        plt.show()
    else:
        if(plot_alot==True):
            for alpha_counter in range(len(alpha_range)):
                for z_counter in range(len(z_range)):
                    fig, ax= plt.subplots(1)
                    for nm_counter in range(len(b_nm_range)):
                            if(log==True):
                                ax.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[alpha_counter][z_counter][nm_counter])),'--',label='(n,m) = (%i,%i)' % (b_nm_range[nm_counter]),linewidth=1)
                            else:
                                ax.plot(t_arr,np.abs(b_nm_loop_alpha[alpha_counter][z_counter][nm_counter]),'--',label='(n,m) = (%i,%i)' % (b_nm_range[nm_counter]),linewidth=1)
                    ax.set_title(r'b_nm for $\alpha$ = %.3f, z = %.2f' %(alpha_range[alpha_counter],z_range[z_counter]))
                    ax.legend()
            plt.show()
def plot_b_nm_var_z_fix_alpha_nm(alpha_range, z_range, b_nm_range,t_arr, b_nm_loop_alpha,plot_alot=False,log=False):
    if(plot_alot==True):
        for alpha_counter in range(len(alpha_range)):
            for nm_counter in range(len(b_nm_range)):
                fig, ax= plt.subplots(1)
                for z_counter in range(len(z_range)):
                    if(log==True):
                        ax.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[alpha_counter][z_counter][nm_counter])),'--',label='z = %.2f' % z_range[z_counter],linewidth=1)
                    else:
                        ax.plot(t_arr,np.abs(b_nm_loop_alpha[alpha_counter][z_counter][nm_counter]),'--',label='z = %.2f' % z_range[z_counter],linewidth=1)
                ax.set_title(r'b_nm for $\alpha$ = %.3f, (n,m) = (%i,%i)' %(alpha_range[alpha_counter],b_nm_range[nm_counter][0],b_nm_range[nm_counter][1]))
                ax.legend()
        plt.show()
    if(len(alpha_range)-1 %2):
        for nm_counter in range(len(b_nm_range)):
            fig, ax= plt.subplots(int(len(alpha_range)/2),2,sharex='all',sharey='all')
            alpha_cntr=0
            for axs in ax.flat:
                for z_counter in range(len(z_range)):
                    if(log==True):
                        axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[alpha_cntr][z_counter][nm_counter])),'--',label='z = %.2f' % z_range[z_counter],linewidth=1)
                    else:
                        axs.plot(t_arr,np.abs(b_nm_loop_alpha[alpha_cntr][z_counter][nm_counter]),'--',label='z = %.2f' % z_range[z_counter],linewidth=1)
                    axs.set_title(r'$\alpha$ = %.3f' %(alpha_range[alpha_cntr]),fontsize=8)
                axs.legend(fontsize=6)
                alpha_cntr+=1
            fig.suptitle(r'b_nm for (n,m) = (%i,%i)' %(b_nm_range[nm_counter][0],b_nm_range[nm_counter][1]))
        plt.show()
def plot_b_nm_var_alpha_fix_z_nm(alpha_range, z_range, b_nm_range,t_arr, b_nm_loop_alpha,log=False):
    if(len(z_range)-1%2):
        for nm_counter in range(len(b_nm_range)):
            fig, ax= plt.subplots(int(len(z_range)/2),2,sharex='all',sharey='all')
            z_cntr=0
            for axs in ax.flat:
                for alpha_counter in range(len(alpha_range)):
                    if(log==True):
                        axs.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[alpha_counter][z_cntr][nm_counter])),'--',label=r'$\alpha$=%.2f' % (alpha_range[alpha_counter]),linewidth=1)
                    else:
                        axs.plot(t_arr,np.abs(b_nm_loop_alpha[alpha_counter][z_cntr][nm_counter]),'--',label=r'$\alpha$=%.2f' % (alpha_range[alpha_counter]),linewidth=1)
                    axs.set_title(' z = %.2f' %z_range[z_cntr])
                z_cntr +=1
            fig.suptitle(r'b_nm for (n,m) = (%i,%i)' %(b_nm_range[nm_counter][0],b_nm_range[nm_counter][1]))
            for axs in ax.flat:
                axs.legend(fontsize=6)
        plt.show()
    else:
        for z_counter in range(len(z_range)):
            for nm_counter in range(len(b_nm_range)):
                fig, ax= plt.subplots(1)
                for alpha_counter in range(len(alpha_range)):
                    if(log==True):
                        ax.plot(t_arr,np.log(np.abs(b_nm_loop_alpha[alpha_counter][z_counter][nm_counter])),'--',label=r'$\alpha$ = %.3f' % alpha_range[alpha_counter],linewidth=1)
                    else:
                        ax.plot(t_arr,np.abs(b_nm_loop_alpha[alpha_counter][z_counter][nm_counter]),'--',label=r'$\alpha$ = %.3f' % alpha_range[alpha_counter],linewidth=1)
                ax.set_title(r'b_nm for (n,m) = (%i,%i), z = %.2f' %(b_nm_range[nm_counter][0],b_nm_range[nm_counter][1],z_range[z_counter]))
                ax.legend()
            plt.show()

def Compare_DW_and_Morse_energies(Terminal_output,lbx=-5,ubx=5,lby=-2,uby=10,m=0.5,ngridx=100,ngridy=100,lamda = 2.0,D=10.0, g = 0.08,plotting=True,alpha_range = (0.252,0.344,0.363,0.525)):
    ###Parameters
    #0.153,0.157,0.193,0.252,0.344,0.363,0.525,0.837,1.1,1.665,2.514###DVR nicht genug converged fuer 3 stellen
    #alpha_range = (0.153,0.157,0.193,0.220,0.252,0.344,0.363,0.525,0.837,1.1,1.665,2.514)
    #alpha_range = (0.344,0.363,0.525)
    
    #get DW energies, in later loop Morse
    pes = quartic_bistable(alpha_range[0],D,lamda,g,z=0)
    pot_DW= lambda a: pes.potential_xy(a,0)
    DVR_DW = DVR1D(4*ngridx,lbx,2*ubx,m,pot_DW)
    vals_DW, vecs_DW= DVR_DW.Diagonalize()
    
    if(plotting==True):  
        xg = np.linspace(lbx,ubx,2*ngridx)#ngridx+1 if "prettier"
        yg = np.linspace(lby,uby,2*ngridy)
        colors= ['b','g','r','c','m','y','k']
    
    if(Terminal_output==True):
        print('Lowest energy levels of DW:')
        print(vals_DW[0:4])

    cntr=0
        
    for alpha in alpha_range:
        if(plotting==True):
            fig= plt.figure()
            gs = fig.add_gridspec(1,2,hspace=0,wspace=0)
            axs = gs.subplots(sharey='row')
            axs[0].plot(xg,pot_DW(xg))
            axs[0].set_title('Pot x')
            for n in range(6):
                axs[0].plot(xg,vals_DW[n]*np.ones_like(xg),'--',label='n = %i' %n )
            color = colors[cntr % len(colors)]
            cntr+=1

        pes = quartic_bistable(alpha,D,lamda,g,z=0)
        pot_Morse= lambda a: pes.potential_xy(0,a)-pes.potential_xy(0,0)
        DVR_Morse = DVR1D(2*ngridy,lby,4*uby,m,pot_Morse)
        vals_M, vecs_M= DVR_Morse.Diagonalize()

        if(plotting==True):
            axs[1].plot(yg, pot_Morse(yg), color)#,label = r'$\alpha$ = %.3f'%alpha)
            for n in range(4):#amount of ev
                axs[1].plot(xg,vals_M[n]*np.ones_like(yg), ('--')+color )
            axs[1].set_title('Pot y')
            axs[0].set_ylim([0,9])
            axs[1].set_ylim([0,9])
            #axs[1].legend(loc='upper center',ncol=3,fancybox=True,shadow=True,fontsize=7)
            axs[0].legend()
            fig.suptitle( r'$\alpha$ = %.2f' %(alpha))

        if(Terminal_output==True):
            print()
            print('alpha= %.3f'%(alpha))
            print(vals_M[0:3])
            print('ratio Dw2 to Morse0 = %.3f, (1/x = %.3f)' %((vals_DW[2]/vals_M[0]),(vals_M[0]/vals_DW[2])))
            print('ratio Dw2 to Morse1 = %.3f, (1/x = %.3f)' %((vals_DW[2]/vals_M[1]),(vals_M[1]/vals_DW[2])))
        plt.show(block=False)


#######################################
###----------FUNCTIONS END----------###
#######################################