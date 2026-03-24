import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR2D, DVR1D
import os
from PISC.utils.readwrite import store_arr, read_arr

m = 0.5
Lx = 1.0
Ly = 2.0

def potential(x,y):
    if(0<=x<=Lx and 0<=y<=Ly):
        return 0.0
    else:
        return 1e6

def potential_1D(x):
    if(0<=x<=Lx):
        return 0.0
    else:
        return 1e6

if(0): # Create a grid for the potential
    x = np.linspace(-0.1, 1.1, 500)
    y = np.linspace(-0.1, 1.1, 500)
    X, Y = np.meshgrid(x, y)
    V = np.vectorize(potential)(X,Y)
    plt.imshow(V, extent=(-0.1, 1.1, -0.1, 1.1), origin='lower')
    
    plt.colorbar(label='Potential')
    plt.title('Potential Landscape of the Square Billiard')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    exit()

# Create a DVR grid and solve for eigenvalues
#System parameters
offset = 0.00
lbx = -offset
ubx = Lx*(1+offset)
lby = -offset
uby = Ly*(1+offset)
hbar = 1.0
ngridx = 100
ngridy = 100
dx = (ubx-lbx)/ngridx
dy = (uby-lby)/ngridy   

if(0):
    dvr1d = DVR1D(ngridx,lbx,ubx,m,potential_1D)
    vals_1d, vecs_1d = dvr1d.Diagonalize()
    vals_1d_anal = np.arange(1,ngridx+1)**2*np.pi**2*hbar**2/(2*m*Lx**2)
    print('1D box eigenvalues (numerical):\n',vals_1d[:20])
    print('1D box eigenvalues (analytical):\n',vals_1d_anal[:20])
    exit()

x = np.linspace(lbx,ubx,ngridx-1)
fname = 'Eigen_basis_2D_box_m_{}_Lx_{}_Ly_{}_ngrid_{}'.format(m,np.round(Lx,3),np.round(Ly,3),ngridx)  
path = os.path.dirname(os.path.abspath(__file__))

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,potential)
print('potential: 2D box')

n_eig_tot = 200
if(1): #Diagonalization
    param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
    #with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:    
    #    f.write('\n'+str(param_dict))#print(param_dict,file=f)

    vals,vecs = DVR.Diagonalize()#_Lanczos(n_eig_tot)
    
    store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))

vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

vals_x = np.arange(1,n_eig_tot+1)**2*np.pi**2*hbar**2/(2*m*Lx**2)
vals_y = np.arange(1,n_eig_tot+1)**2*np.pi**2*hbar**2/(2*m*Ly**2)

vals_anal = [vals_x[i]+vals_y[j] for i in range(n_eig_tot) for j in range(n_eig_tot)]
vals_anal = np.sort(vals_anal)
vals_anal = vals_anal[:n_eig_tot]

print('Numerical eigenvalues:\n',vals[:20])
print('Analytical eigenvalues:\n',vals_anal[:20])

diff = np.abs(vals - vals_anal)
print('% Error in eigenvalues:\n',diff/vals_anal*100)


