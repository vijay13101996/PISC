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
Lx=7.0
lbx = -Lx
ubx = Lx
lby = -1#Morse, will explode quickly
uby = 15#Morse, will explode quickly
m = 0.5
ngrid = 100
ngridx = ngrid
ngridy = ngrid

##########potential##########
###CHO_Lars
if(False):
    omega=1
    g=0.1
    pes= lambda a,b : DVR2D_mod.pot_2D_CHO(a,b,g=g, omega=omega,m=m)
    DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes)

###quartic bistable
if(True):#parameters
    lamda = 2.0 #already desired parameter -> either 2.0 (or 1.5), but 2.0 better b.c.. in other paper 
    if(lamda==2):
        D= 10.0
    else:  
        D=10.0#5.0 ##hight of Morse ### much higher than barrier hight (ALWAYS!),
    ######exponent in Morse D*(1-e**(-alpha y)) the bigger alpha, the smaller time until plateau
    #sqeezes corrod system... energy spacing: 
    alpha = 1 #1.165#0.81#0.175#0.41#0.255#1.165 


    g = lamda**2/32##### v(x)=g*(x**2 -lambda**2/(8g)**2)   #lamda**2/32#4.0###g=0.01  many states in double well , larger g, 0.035 

    z = 0.2#1.25#2.3	###coupling, ###2.3 ...breaks at around that point

pes = quartic_bistable(alpha,D,lamda,g,z)
xg = np.linspace(lbx,ubx,ngridx)#ngridx+1 if "prettier"
yg = np.linspace(lby,uby,ngridy)
xgr,ygr = np.meshgrid(xg,yg)

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
n_eig_tot = 250##leave it like that s.t. don't do too much work 


##########Save so only has to be done once
#potkey = 'CHO_omega_{}_g0_{}_m_{}'.format(omega, g,m)
potkey = 'DW_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
fname = 'Eigen_basis_{}_ngrid_{}_n_eig_tot_{}'.format(potkey,ngrid,n_eig_tot)
path = os.path.dirname(os.path.abspath(__file__))

#####Check if data is available
diag= True
try:
    with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'r') as f:
        param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot} 
        if(f.read()=='\n'+str(param_dict)):
            print('Retrieving data...')
            diag=False
        else:
            print('Diagonalization of (%i x %i) matrix in:' % (ngridx*ngridy,ngridx*ngridy))
except:
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

##########plot x and y potentials, eigenvalues and eigenvalues of the coupled problem, and 3D potential
if(True):
    pot_DW= lambda a: pes.potential_xy(a,0)
    DVR_DW = DVR1D(2*ngridx,lbx,ubx,m,pot_DW)
    pot_Morse= lambda a: pes.potential_xy(0,a)-pes.potential_xy(0,0)#lamda**4/(64*g),a)
    DVR_Morse = DVR1D(2*ngridy,lby,2*uby,m,pot_Morse)#get t a little more exact
    vals_M, vecs_M= DVR_Morse.Diagonalize()
    vals_DW, vecs_DW= DVR_DW.Diagonalize()
    Check_DVR.plot_V_and_E(xg,yg,vals=vals,vecs=vecs,vals_x=vals_DW,vals_y= vals_M ,pot=pes.potential_xy,z=z)

if(True): #some inside/intuition about potential and Eigenvalues
    xg = np.linspace(lbx,ubx,ngridx)#ngridx+1 if "prettier"
    yg = np.linspace(lby,uby,ngridy)
    xgr,ygr = np.meshgrid(xg,yg)

    contour_eigenstate=12
    plt.title('Pot_Contour, Eigenstate: %i, z = %.2f '% (contour_eigenstate,z))
    plt.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,5,0.25))
    plt.imshow(DVR.eigenstate(vecs[:,contour_eigenstate])**2,origin='lower')#cannot be fit to actual axis #plt.imshow(np.reshape(vecs[:,12],(ngridx+1,ngridy+1),'F')**2,origin='lower')#same
    #plt.contour(xg,yg,pes.potential_xy(xgr,ygr),levels=np.arange(0,5,0.25))#in case I need the values
    
    plt.show()


##########Parameters
T=1
beta=1/T
n_eigen=30#up to which n C_n should be calculated
N_trunc=70#how long m_arr is and therefore N_trunc 

##########Arrays
x_arr = DVR.pos_mat(0)#defines grid points on x
k_arr = np.arange(N_trunc) +1 #what
m_arr = np.arange(N_trunc) +1 #what
t_arr = np.linspace(0.0,15.0,150) # time interval
#t_arr = np.linspace(0.0,5.0,80) # time interval
OTOC_arr = np.zeros_like(t_arr,dtype='complex') #Store OTOC 
b_arr = np.zeros_like(OTOC_arr)#store B


##########OTOC calc

time_mc=time.time()
k=0
#for n in (0,5,10,20):####MC OTOC
for n in range(15):###Mircocanonic OTOC
    n=n
    C_n = OTOC_f_2D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)
    #plt.plot(t_arr,np.real(C_n),label='n = %i' % (n+1)) #is real anyway
    plt.plot(t_arr,np.real(C_n),'--')#,label='n = %i' % (n+1)) #is real anyway
    if(n<20):
        l=np.random.randint((len(t_arr)))
        #plt.text(t_arr[-1],np.real(C_n[-1]),f'n = {n+1}',fontsize=8 )
        plt.text(t_arr[l],np.real(C_n[l]),f'n = {n+1}',fontsize=10 )
    k +=1
print('Time for 1 McOTOC (avg over %i): %.2f s ' % (k,(time.time()-time_mc)/k))

if(False):###Thermal OTOC
    start_time=time.time()
    OTOC_arr = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr) 
    plt.title('z = %.3f ' % z)
    plt.plot(t_arr,np.real(OTOC_arr),'k-',label='T = %.1f' % T,linewidth=2.5)###remember to also comment out line above
    print('Time OTOC calc (calc of %i ) : %.2f s  (t/n = %.3f)' %(n_eigen,(time.time()-start_time),(time.time()-start_time)/n_eigen))
    plt.legend()
#plt.ylim([0,100])#to compare with paper

plt.show()