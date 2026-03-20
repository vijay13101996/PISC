import numpy as np
from PISC.engine import Krylov_complexity_2D
from PISC.dvr.dvr import DVR2D
from PISC.potentials import coupled_harmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
import time
import os

ngrid = 100 #Number of grid points
ngridx = ngrid #Number of grid points along x
ngridy = ngrid #Number of grid points along y

neigs = 500 #Number of eigenstates to be calculated

a = 1.5
b = 1.5

def potential_xy(x,y):
    if (x>a or x<0 or y>b or y<0):
        return 1e9
    else:
        return 0.0

potential_xy = np.vectorize(potential_xy)

potkey = 'lowT_rectangular_box_a_{}_b_{}'.format(np.around(a,2),np.around(b,2))

#System parameters
m = 0.5
lbx = -.5*a
ubx = 1.5*a
lby = -.5*b
uby = 1.5*b
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid
dx = (ubx-lbx)/ngridx
dy = (uby-lby)/ngridy

start_time = time.time()

if(0): #Plot the potential
    xg = np.linspace(lbx,ubx,ngridx)
    yg = np.linspace(lby,uby,ngridy)
    xgr,ygr = np.meshgrid(xg,yg)
    plt.contour(xgr,ygr,potential_xy(xgr,ygr),levels=np.arange(-1,30,1.0))
    #plt.imshow(potential_xy(xgr,ygr),origin='lower')
    plt.show()    
    exit()

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
y_arr = DVR.pos_mat(1)

#-------------------------------------------------------
pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
X = pos_mat

pos_mat = np.zeros((neigs,neigs))
pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, y_arr, dx, dy, pos_mat)
Y = pos_mat

prod = 0
if prod:
    O = np.matmul(X,Y)
    print('O','XY')
else:
    O = X
    print('O','X')


print('O',np.around(O[:5,:5],3),'vals',vals[-1])
#exit()

mom_mat = np.zeros((neigs,neigs))
mom_mat = Krylov_complexity_2D.krylov_complexity.compute_mom_matrix(vecs, vals, x_arr, m, dx, dy, mom_mat)
P = mom_mat

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity_2D.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity_2D.krylov_complexity.compute_hadamard_product(L,O,LO)

T_arr = [1.0,2.0,4.0]#,100.0]#[10.0,20.0,40.,100.0]#np.arange(1.0,5.05,0.5)#[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
mun_arr = []
mu0_harm_arr = []
bnarr = []
nmoments = 40
ncoeff = 50

for T_au in T_arr:    
    Tkey = 'T_{}'.format(T_au)

    beta = 1.0/T_au 

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity_2D.krylov_complexity.compute_moments(O, vals, beta, moments)
    even_moments = moments[0::2]

    mu0, mu2, mu4 = even_moments[0], even_moments[1], even_moments[2]
    
    print('2 mu0 mu2,  mu2, mu2_sq',2*mu0*mu2, mu2, mu2**2)

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity_2D.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')
    bnarr.append(barr)

    #print('T', T_au, 'barr',barr[1])

    #print('moments',np.around(moments[0],5))
    mun_arr.append(even_moments)

    #mu0_harm_arr.append(1.0/(2*m*omega*np.sinh(0.5*beta*omega)))
    #print('mu0_harm',mu0_harm_arr[-1])

mun_arr = np.array(mun_arr)
bnarr = np.array(bnarr)
print('mun_arr',mun_arr.shape)
print('bnarr',bnarr.shape)


store_arr(T_arr,'T_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(mun_arr,'mun_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(bnarr,'bnarr_{}_neigs_{}'.format(potkey,neigs))

exit()

if(1):
    for i in [0,1,2,3]:#,2,4,6,8]:
        plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={}'.format(T_arr[i]))
        #plt.scatter((np.arange(1,nmoments//2+1)),(mun_arr[i,1:]),label='T={}'.format(np.around(T_arr[i],2)))

    #plt.ylim([0,700])
    plt.title(potkey)
    plt.legend()    
    plt.show()
    exit()


slope_arr = []
mom_list = [0,2,4,6,8,10,12,14]#,16,18,20]
for i in mom_list:#[0,1,2,3,4,5,6,7,8,9,10]:#,3,4]:#range(0,ncoeff,2):
    #plt.scatter(T_arr,bnarr[:,i],label='n={}'.format(i))
    
    #plt.scatter((T_arr),(mun_arr[:,i]),label='n={}'.format(2*i))
    plt.scatter(np.log(T_arr),np.log(mun_arr[:,i]),label='n={}'.format(2*i))

    lT_arr = np.log(T_arr)
    lmun_arr = np.log(mun_arr[:,i])

    p = np.polyfit(lT_arr,lmun_arr,1)

    slope_arr.append(p[0])
    
    plt.plot(lT_arr,p[0]*lT_arr+p[1])


    # Fit mun_arr[:,i] to a T_arr**(i/2)
    #p = np.polyfit(T_arr,mun_arr[:,i],i/2)
    #print('p',p)
    #plt.plot(T_arr,p[0]*T_arr**(i/2),label='n={}'.format(2*i))

#plt.scatter(T_arr,np.array(mu0_harm_arr),label='n=0, harm',color='black')
plt.legend()
plt.show()

plt.scatter([0,1,2,3,4,5,6,7,8,9,10],np.array(slope_arr)**2)
plt.show()

if(0):
    for i in [1,2,3]:#range(1):#nmoments//2+1):
        # Fit times_arr vs mun_arr[:,i] to a line
        p = np.polyfit(times_arr,mun_arr[:,i],1)
        print('p',p)
        plt.plot(times_arr,p[0]*times_arr+p[1],label='n={}'.format(2*i))
        plt.scatter(times_arr,mun_arr[:,i],label='n={}'.format(2*i))
    #plt.scatter(times_arr,mun_arr)
    plt.legend()
    plt.show()
    

#ncoeffs = 20
#barr = np.zeros(ncoeffs)
#barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')

#b0 = np.sqrt(1/(2*m*w*np.sinh(0.5*beta*w)))

#print('barr',barr,b0)

#plt.scatter(np.arange(ncoeffs),barr)
#plt.show()


