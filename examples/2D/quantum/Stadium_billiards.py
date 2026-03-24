import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR2D
import os
from PISC.utils.readwrite import store_arr, read_arr

m = 0.5
r = np.sqrt(1/(4+np.pi))
a = r

potkey = 'stadium_billiard_m_{}_a_{}_r_{}'.format(m,np.round(a,3),np.round(r,3))

def potential(x,y):
    if( (x+a)**2 + y**2 < r**2 or (x-a)**2 + y**2 < r**2):
        return 0.0
    elif(x>-a and x<a and y>-r and y<r):
        return 0.0 
    else:
        return 1e6

if(0): # Create a grid for the potential    
    x = np.linspace(-a-r, (a + r), 500)
    y = np.linspace(-r, r, 500)
    X, Y = np.meshgrid(x, y)
    V = np.vectorize(potential)(X,Y)
    plt.imshow(V, extent=(-a-r, a + r, -r, r), origin='lower')
    
    plt.colorbar(label='Potential')
    plt.title('Potential Landscape of the Stadium Billiard')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    exit()

# Create a DVR grid and solve for eigenvalues
#System parameters
L = (a+r)
offset = 0.0
lbx = -L*(1+offset)
ubx = L*(1+offset)
lby = -r*(1+offset)
uby = r*(1+offset)
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid
dx = (ubx-lbx)/ngridx
dy = (uby-lby)/ngridy

x = np.linspace(lbx,ubx,ngridx+1)
#print('Vs',pes.potential_xy(0,0))
fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
path = os.path.dirname(os.path.abspath(__file__))

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,potential)
print('potential',potkey)    

n_eig_tot = 200
if(1): #Diagonalization
    param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
    #with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:    
    #    f.write('\n'+str(param_dict))#print(param_dict,file=f)
    
    vals,vecs = DVR.Diagonalize(n_eig_tot)

    store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
    plt.imshow(DVR.eigenstate(vecs[:,0])**2,origin='lower')
    plt.show()

vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))


