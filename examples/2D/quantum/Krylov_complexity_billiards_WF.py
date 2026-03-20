import numpy as np
from PISC.engine import Krylov_complexity_2D 
from PISC.dvr.dvr import DVR2D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
import time
import os

ngrid = 100 #Number of grid points
ngridx = ngrid #Number of grid points along x
ngridy = ngrid #Number of grid points along y

neigs = 500 #Number of eigenstates to be calculated

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
    potkey = 'stadium_billiards_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))

if(1): #Rectangular box
    def potential_xy(x,y):
        if (x>(a+R) or x<(-a-R) or y>R or y<-R):
            return 1e6
        else:
            return 0.0

    potential_xy = np.vectorize(potential_xy)

    potkey = 'rectangular_box_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))

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
#-------------------------------------------------------
pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
O = pos_mat

if(0):
    #Compute level-spacing statistics
    vals = np.sort(vals)
    #Remove degenerate states
    vals_u = np.unique(vals)
    diffs = np.diff(vals_u)

    plt.hist(diffs, bins=50, density=True)
    plt.title('Level spacing distribution')
    plt.xlabel('Level spacing')
    plt.ylabel('Density')
    plt.show()


    #plt.plot(vals[:neigs])
    #plt.show()

    plt.imshow(O)
    plt.colorbar()
    plt.title('Position matrix O')
    plt.show()
    exit()

    print('O',np.around(O[:5,:5],3),'vals',vals[-1])
    #exit()

if(1): #Use Eigenstates
    n_wf = 50
    wf = vecs[:,n_wf]  # ground state wavefunction
    
    coeff_wf = np.zeros(neigs)
    coeff_wf[n_wf] = 1.0  # initial state is the ground state

mom_mat = np.zeros((neigs,neigs))
mom_mat = Krylov_complexity_2D.krylov_complexity.compute_mom_matrix(vecs, vals, x_arr, m, dx, dy, mom_mat)
P = mom_mat

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity_2D.krylov_complexity.compute_liouville_matrix(vals,liou_mat)
#L = L.T

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity_2D.krylov_complexity.compute_hadamard_product(L,O,LO)

nmoments = 40
ncoeff = 100

barr = np.zeros(ncoeff)
barr = Krylov_complexity_2D.krylov_complexity.compute_lanczos_coeffs_wf(O, L, barr, coeff_wf)
plt.scatter(np.arange(ncoeff),barr)
plt.xlabel('n')
plt.ylabel('b_n')
plt.title('Lanczos Coefficients b_n')
plt.show()
