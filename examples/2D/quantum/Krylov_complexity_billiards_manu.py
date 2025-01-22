import numpy as np
from PISC.engine import Krylov_complexity_2D
from PISC.dvr.dvr import DVR2D
from PISC.potentials import coupled_harmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
import time
import os
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12



ngrid = 100 #Number of grid points
ngridx = ngrid #Number of grid points along x
ngridy = ngrid #Number of grid points along y

neigs = 200 #Number of eigenstates to be calculated

R = np.sqrt(1/(4+np.pi))
a = R#0.0

if(1): #Stadium billiards
    def potential_xy(x,y):
        if( (x+a)**2 + y**2 < R**2 or (x-a)**2 + y**2 < R**2):
            return 0.0
        elif(x>-a and x<a and y>-R and y<R):
            return 0.0 
        else:
            return 1e6
        
    potential_xy = np.vectorize(potential_xy)   
    potkey = 'MANU_stadium_billiards_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))

if(0): #Rectangular box
    def potential_xy(x,y):
        if (x>(a+R) or x<(-a-R) or y>R or y<-R):
            return 1e6
        else:
            return 0.0

    potential_xy = np.vectorize(potential_xy)

    potkey = 'MANU_rectangular_box_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))

#System parameters
m = 0.5
L = (a+R)*1.5
lbx = -L
ubx = L
lby = -2*R
uby = 2*R
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid
dx = (ubx-lbx)/ngridx
dy = (uby-lby)/ngridy

start_time = time.time()
print('R',R)

if(0): #Plot the potential
    xg = np.linspace(lbx,ubx,ngridx)
    yg = np.linspace(lby,uby,ngridy)
    xgr,ygr = np.meshgrid(xg,yg)
    plt.contour(xgr,ygr,potential_xy(xgr,ygr),levels=np.arange(-1,30,1.0))
    #plt.imshow(potential_xy(xgr,ygr),origin='lower')
    plt.show()    
    #exit()

x = np.linspace(lbx,ubx,ngridx+1)
#print('Vs',pes.potential_xy(0,0))
fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
path = os.path.dirname(os.path.abspath(__file__))

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,potential_xy)
print('potential',potkey)   

#Diagonalization
param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':neigs}
with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:    
    f.write('\n'+str(param_dict))

if(0): #Diagonalize the Hamiltonian
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    store_arr(vecs[:,:neigs],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    store_arr(vals[:neigs],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
    print('Time taken:',time.time()-start_time)
    
    #plt.plot(vals[:neigs])
    #plt.show()

    #exit()

if(1): #Read eigenvalues and eigenvectors and test whether the Wavefunctions look correct
    vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
    vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    
    n=99
    print('vals[n]', vals[-1])
    
    #plt.imshow(DVR.eigenstate(vecs[:,-1])**2,origin='lower')
    #plt.show()

x_arr = DVR.pos_mat(0)
#-------------------------------------------------------
pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
O = pos_mat

print('O',np.around(O[:5,:5],3),'vals',vals[-1])
#exit()

mom_mat = np.zeros((neigs,neigs))
mom_mat = Krylov_complexity_2D.krylov_complexity.compute_mom_matrix(vecs, vals, x_arr, m, dx, dy, mom_mat)
P = mom_mat

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity_2D.krylov_complexity.compute_liouville_matrix(vals,liou_mat)
#L = L.T

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity_2D.krylov_complexity.compute_hadamard_product(L,O,LO)

T_arr = [10,20,40,100]#[10.0,20.0,40.,100.0]#np.arange(1.0,5.05,0.5)#[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
mun_arr = []
mu0_harm_arr = []
bnarr = []
nmoments = 40
ncoeff = 40

if(0):
    for T_au in T_arr:    
        Tkey = 'T_{}'.format(T_au)

        beta = 1.0/T_au 

        moments = np.zeros(nmoments+1)
        moments = Krylov_complexity_2D.krylov_complexity.compute_moments(O, vals, beta, 'wgm', 0.5, moments)
        even_moments = moments[0::2]

        barr = np.zeros(ncoeff)
        barr = Krylov_complexity_2D.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5,'wgm')
        bnarr.append(barr)

        mun_arr.append(even_moments)


    mun_arr = np.array(mun_arr)
    bnarr = np.array(bnarr)
    print('mun_arr',mun_arr.shape)
    print('bnarr',bnarr.shape)


    store_arr(T_arr,'T_arr_{}_neigs_{}'.format(potkey,neigs))
    store_arr(mun_arr,'mun_arr_{}_neigs_{}'.format(potkey,neigs))
    store_arr(bnarr,'bnarr_{}_neigs_{}'.format(potkey,neigs))

    exit()

if(1):
    #Plot the potentials
    fig, ax = plt.subplots(1,1)
    xg = np.linspace(-(a+R),(a+R),501)
    yg = np.linspace(-R,R,501)

    xgr,ygr = np.meshgrid(xg,yg)
    ax.contour(xgr,ygr,potential_xy(xgr,ygr),levels=np.arange(-1,30,1.0),colors='g' )
  
    #No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlabel(r'$x$',fontsize=xl_fs)
    ax.set_ylabel(r'$y$',fontsize=yl_fs)

    fig.set_size_inches(5,3.5)
    fig.savefig('/home/vgs23/Images/potential_billiards.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)
    plt.show()


if(0):
    potkey_bil = 'MANU_stadium_billiards_a_{:.2f}_R_{:.2f}'.format(a,R)
    potkey_box = 'MANU_rectangular_box_a_{:.2f}_R_{:.2f}'.format(a,R)

    fig, ax = plt.subplots(1,2,sharey=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax0 = ax[0]
    ax1 = ax[1]

    for ax, neigs in zip(ax,[100,200]):
        T_arr = read_arr('T_arr_{}_neigs_{}'.format(potkey,neigs))
        bn_arr_bil = read_arr('bnarr_{}_neigs_{}'.format(potkey_bil,neigs))
        bn_arr_box = read_arr('bnarr_{}_neigs_{}'.format(potkey_box,neigs))

        for i in [0,1,2,3]:
            if(ax==ax0):
                ax.scatter(np.arange(ncoeff),bn_arr_bil[i,:],label=r'$T={}$'.format(T_arr[i]),s=10)
                ax.plot(np.arange(ncoeff),bn_arr_box[i,:],ls='--')
            else:
                ax.scatter(np.arange(ncoeff),bn_arr_bil[i,:],s=10)
                ax.plot(np.arange(ncoeff),bn_arr_box[i,:],ls='--')

        ax.set_xlabel(r'$n$',fontsize=xl_fs)
        if(ax==ax0):
            ax.set_ylabel(r'$b_n$',fontsize=yl_fs)
            ax.annotate(r'(a)',(0.02,0.9),xycoords='axes fraction',fontsize=xl_fs)
        else:
            ax.annotate(r'(b)',(0.02,0.9),xycoords='axes fraction',fontsize=xl_fs)
        ax.tick_params(axis='both',labelsize=ti_fs)

    fig.set_size_inches(7,3.5)	
    fig.legend(fontsize=le_fs-2,loc=(0.16,0.91),ncol=4)
    fig.savefig('/home/vgs23/Images/bn_vs_n_billiards.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)
    plt.show()

    exit()


