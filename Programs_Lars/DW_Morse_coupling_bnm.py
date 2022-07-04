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

z_range=(0,0.5,1,1.5)#,1.25,1.5): #(1.25))
#z_range=(0,0.5,)
alpha_range=(0.363,0.525,0.837,1.1,1.665,2.514)###note how close 0.363 and 0.525 are
#alpha_range=(0.837,)
cntr=0###counter for outer loop (over alpha)

###loop over alpha
for alpha in alpha_range:
    ###loop over different coupling strengths z 
    for z in z_range:
        #if(cntr!=0 and z==0):####skip if not needed
        #    continue

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
        pos_mat = np.zeros_like(k_arr)

        ##########OTOC calc

        time_mc=time.time()
        k=ngridx
        
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
        
        n_for_b=2
        m_for_b=0
        plt.title(r'B$_{%i%i}$ for different $\alpha$'%(n_for_b,m_for_b))
        X=np.zeros((N_trunc,N_trunc))
        for i in range(N_trunc):
            for j in range(N_trunc):
                X[i,j] = OTOC_f_2D_omp_updated.otoc_tools.pos_matrix_elts(vecs,x_arr,DVR.dx,DVR.dy,i+1,j+1,pos_mat)
        b_nm = PY_OTOC2D.b_nm(t=t_arr,n=n_for_b,m=m_for_b,X=X,w=vals)
        if(z==0):# and cntr ==0 ):
            plt.plot(t_arr,np.abs(b_nm)**2,'-' ,linewidth=2,label=r'z = %.2f, $\alpha$ = %.2f' % (z, alpha))#,label=r'n = %i, $\alpha$ = %.2f' % (n, alpha)) #is real anyway
        if(z!=0):
            plt.plot(t_arr,np.abs(b_nm)**2,'--',label=r'z = %.2f, $\alpha$ = %.2f' % (z, alpha),linewidth=1.5) 
            plt.text(t_arr[10+4*cntr], np.abs(b_nm[10+4*cntr])**2,'%.2f' % alpha )
        cntr+=1
    plt.legend(bbox_to_anchor=(0.65,1),ncol=3,fontsize=8)
    plt.show()
    
plt.show()
