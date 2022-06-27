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
from matplotlib import pyplot as plt
import os 
import time 
from mylib.testing import Check_DVR 

##########Parameters##########
Lx=5.0
lbx = -Lx
ubx = Lx
lby = -2#Morse, will explode quickly
uby = 10#Morse, will explode quickly
m = 0.5
ngrid = 100
ngridx = ngrid
ngridy = ngrid

##########potential#########

###quartic bistable
#parameters
lamda = 2.0   
D=10.0  
g = 0.08

##########Start- Educated guessing:
alpha= 1
if(False):###Educated guessing:
    ###Parameter
    z=0
    #alpha_range= np.linspace(0.1,1,10)
    #for alpha in (0.175,0.41,0.255,1.165, 0.363,0.5):#,0.9,1.5):
    #alpha = 0.#,1#0.363 #1.165#0.81#0.175#0.41#0.255#1.165 
    #0.153,0.157,0.193,0.252,0.344,0.363,0.525,0.837,1.1,1.665,2.514###DVR nicht genug converged fuer 3 stellen
    alpha_range = (0.153,0.157,0.193,0.252,0.344,0.363,0.525,0.837,1.1,1.665,2.514)
    print(len(alpha_range))
    pes = quartic_bistable(alpha,D,lamda,g,z)
    pot_DW= lambda a: pes.potential_xy(a,0)
    DVR_DW = DVR1D(4*ngridx,lbx,2*ubx,m,pot_DW)
    vals_DW, vecs_DW= DVR_DW.Diagonalize()
    fig= plt.figure()

    gs = fig.add_gridspec(1,2,hspace=0,wspace=0)
    axs = gs.subplots(sharey='row')
    xg = np.linspace(lbx,ubx,2*ngridx)#ngridx+1 if "prettier"
    yg = np.linspace(lby,uby,2*ngridy)
    axs[0].plot(xg,pot_DW(xg))
    axs[0].set_title('Pot x')
    #plot the eigenvalues
    for n in range(6):
        axs[0].plot(xg,vals_DW[n]*np.ones_like(xg),'--',label='n = %i' %n )
    #plot morse EV for different alphas
    colors= ['b','g','r','c','m','y','k']
    print('Lowest energy levels of DW:')
    print(vals_DW[0:4])

    cntr=0
    for alpha in alpha_range:
        color = colors[cntr % len(colors)]
        cntr+=1
        pes = quartic_bistable(alpha,D,lamda,g,z)
        pot_Morse= lambda a: pes.potential_xy(0,a)-pes.potential_xy(0,0)
        DVR_Morse = DVR1D(2*ngridy,lby,4*uby,m,pot_Morse)
        vals_M, vecs_M= DVR_Morse.Diagonalize()

        axs[1].plot(yg, pot_Morse(yg), color,label = r'$\alpha$ = %.3f'%alpha)
        for n in range(2):#amount of ev
            axs[1].plot(xg,vals_M[n]*np.ones_like(yg), ('--'+color),label=r'n = %i, $\alpha$ = %.2f' %(n,alpha) )
        #axs[1].plot(xg,vals_M[0]*np.ones_like(yg),('--'+color),label=r'n = %i, $\alpha$ = %.2f' %(n,alpha) )
        print()
        print('alpha= %.3f'%(alpha))
        print(vals_M[0:2])
        print('ratio Dw2 to Morse0 = %.3f, (1/x = %.3f)' %((vals_DW[2]/vals_M[0]),(vals_M[0]/vals_DW[2])))
        print('ratio Dw2 to Morse1 = %.3f, (1/x = %.3f)' %((vals_DW[2]/vals_M[1]),(vals_M[1]/vals_DW[2])))
      
    axs[1].set_title('Pot y')
    axs[0].set_ylim([0,9])
    axs[1].set_ylim([0,9])
    axs[1].legend(loc='upper center',ncol=3,fancybox=True,shadow=True,fontsize=7)
    axs[0].legend()

    #plt.show()


##########Parameter to vary: alpha
###loop over different parameters (only alpha is changed), alpha only changes Morse
###exponent in Morse D*(1-e**(-alpha y)) the bigger alpha, the smaller time until plateau
###because of that energy spacing will also change


alpha_range=(0.153,0.157,0.193,0.252,0.344,0.363,0.525,0.837,1.1,1.665,2.514)###found the corresponding energy levels on the barrier top, but for slight coupling almost impossible to identify which one is which
#alpha_range=(0.363,0.525,0.837,1.1,1.665,2.514)###note how close 0.363 and 0.525 are
cntr=0###counter for outer loop (over alpha)

###loop over alpha
for alpha in alpha_range:
    ###loop over different coupling strengths z 
    for z in (0,0.5):#,1,1.5):#,1.25,1.5): #(1.25)
        if(cntr!=0 and z==0):####skip if not needed
            continue
        ###########define potential, grid and DVR
        pes = quartic_bistable(alpha,D,lamda,g,z)
        xg = np.linspace(lbx,ubx,ngridx)#ngridx+1 if "prettier"
        yg = np.linspace(lby,uby,ngridy)
        xgr,ygr = np.meshgrid(xg,yg)

        DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
        n_eig_tot = 150


        ##########Save so only has to be done once
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
            with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a+') as f:	
                f.write('\n'+str(param_dict))
            vals, vecs= DVR.Diagonalize(neig_total=n_eig_tot)
            store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
            store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
            print('Data stored!')

        #####read out arrays s.t.has to be diag only once
        vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
        vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
        print('... data retrieved!')

        ##########plot x and y potentials, eigenvalues and eigenvalues of the coupled problem, and 3D potential
        if(False):
            pot_DW= lambda a: pes.potential_xy(a,0)
            DVR_DW = DVR1D(2*ngridx,lbx,ubx,m,pot_DW)
            pot_Morse= lambda a: pes.potential_xy(0,a)-pes.potential_xy(0,0)
            DVR_Morse = DVR1D(2*ngridy,lby,2*uby,m,pot_Morse)
            vals_M, vecs_M= DVR_Morse.Diagonalize()
            vals_DW, vecs_DW= DVR_DW.Diagonalize()
            #Check_DVR.plot_V_and_E(xg,yg,vals=vals,vecs=vecs,vals_x=vals_DW,vals_y= vals_M ,pot=pes.potential_xy,z=z,threeDplot=True)
            Check_DVR.plot_DW_MO_DW_inter(xg,yg,pot=pes.potential_xy,vals=vals,vals_x=vals_DW,vals_y= vals_M, z=z)
        if(False): #some inside/intuition about potential and Eigenvalues
        #for l in (2,3,5,8):
            l=2
            xg = np.linspace(lbx,ubx,ngridx)#ngridx+1 if "prettier"
            yg = np.linspace(lby,uby,ngridy)
            xgr,ygr = np.meshgrid(xg,yg)
            contour_eigenstate=l
            plt.title('Pot_Contour, Eigenstate: %i, z = %.2f '% (contour_eigenstate,z))
            #plt.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,5,0.25))
            plt.imshow(DVR.eigenstate(vecs[:,contour_eigenstate])**2,origin='lower')#cannot be fit to actual axis #plt.imshow(np.reshape(vecs[:,12],(ngridx+1,ngridy+1),'F')**2,origin='lower')#same
            #plt.contour(xg,yg,pes.potential_xy(xgr,ygr),levels=np.arange(0,5,0.25))#in case I need the values
            plt.show()

        ##########Parameter(s), maybe some for b_nm
        N_trunc=100#how long m_arr is and therefore N_trunc 

        ##########Arrays
        x_arr = DVR.pos_mat(0)#defines grid points on x
        k_arr = np.arange(N_trunc) +1 #what
        m_arr = np.arange(N_trunc) +1 #what
        t_arr = np.linspace(0.0,5.0,150) # time intervaln=2
        #t_arr = np.linspace(0.0,5.0,80) # time interval
        OTOC_arr = np.zeros_like(t_arr,dtype='complex') #Store OTOC 
        b_arr = np.zeros_like(OTOC_arr)#store B


        ##########OTOC calc

        time_mc=time.time()
        k=0
        
        ###Mircocanonic OTOC
        
        if(False): #if want to plot different C_n's (other than 2 )
            for n in range(5):#should be the n=2 for uncoupled 
                C_n = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)
                plt.plot(t_arr,np.real(C_n),'--',label='n = %i' % (n),linewidth=1.5) #is real anyway, just for the type of the variable
                plt.title('C_n for z= %.2f, alpha = %.3f' % (z,alpha))
                plt.legend()
                if(n<20):#plot the numbers of C_n somewhere visible
                    l=np.random.randint(int((len(t_arr)/2)))
                    l+=int(len(t_arr)/2) # so thaat in second part of plot
                    #plt.text(t_arr[-1],np.real(C_n[-1]),f'n = {n+1}',fontsize=8 )
                    plt.text(t_arr[l],np.real(C_n[l]),f'n = {n}',fontsize=6 )
                k +=1
            print('Time for 1 McOTOC (avg over %i): %.2f s ' % (k,(time.time()-time_mc)/k))
        
        plt.title(r'C_2 for different $\alpha$')
        n=2

        #####was to find the corresponding barrier top state, but does not make sense, bc mixing after coupling
        if(alpha<0.36):
            n=8
        if(alpha<0.252):
            n=15
        if(alpha<0.194):
            n=17
        if(alpha<0.16):
            n=19
        if(alpha<0.155):
            n=24
        
        #######
        C_n = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)
        if(z==0):# and cntr ==0 ):
            plt.plot(t_arr,np.real(C_n),'-' ,linewidth=2)#,label=r'n = %i, $\alpha$ = %.2f' % (n, alpha)) #is real anyway
        if(z!=0):
            plt.plot(t_arr,np.real(C_n),'--',label=r'z = %.2f, $\alpha$ = %.2f' % (z, alpha),linewidth=1.5) #is real anyway
            plt.text(t_arr[10+8*cntr], np.real(C_n[10+8*cntr]),'%.2f' % alpha )
    plt.legend()
    #plt.show()
    cntr+=1
plt.show()