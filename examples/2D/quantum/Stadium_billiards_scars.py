import numpy as np
from matplotlib import pyplot as plt
from PISC.plane_wave.stadium_billiards import StadiumBilliard
from PISC.dvr.dvr import DVR2D
import os
from PISC.utils.readwrite import store_arr, read_arr


m = 0.5
r = np.sqrt(1/(4+np.pi))
a = r

potkey = 'quart_stadium_billiard_m_{}_a_{}_r_{}'.format(m,np.round(a,3),np.round(r,3))

def potential(x, y):
    # Potential for a quarter stadium billiard: 0 inside the billiard, infinity outside
    if (0 <= x <= a and 0 <= y <= r):
        return 0 # Inside the rectangular part
    elif((x - a)**2 + (y)**2 <= r**2 and x-a>0 and y>0):
        return 0 # Inside the semicircular part
    else:
        return 1e6  # A large number to represent infinity

x = np.linspace(0, (a + r), 500)
y = np.linspace(0, r, 500)
X, Y = np.meshgrid(x, y)
V = np.vectorize(potential)(X, Y)

if(0): # Create a grid for the potential    
    x = np.linspace(-a-r, (a + r), 500)
    y = np.linspace(-r, r, 500)
    X, Y = np.meshgrid(x, y)
    Vxy = np.vectorize(potential_xy)(X,Y)
    V = np.vectorize(potential)(X, Y)
    plt.imshow(V, extent=(-a-r, a + r, -r, r), origin='lower')
    #plt.imshow(Vxy, extent=(-a-r, a + r, -r, r), origin='lower')
    
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
lbx = -offset
ubx = L*(1+offset)
lby = -offset
uby = r*(1+offset)
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid
dx = (ubx-lbx)/ngridx
dy = (uby-lby)/ngridy

x = np.linspace(lbx,ubx,ngridx+1)
fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
path = os.path.dirname(os.path.abspath(__file__))

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,potential)
print('potential',potkey)    

n_eig_tot = 200
if(0): #Diagonalization
    param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
    #with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:    
    #    f.write('\n'+str(param_dict))#print(param_dict,file=f)
    
    vals,vecs = DVR.Diagonalize()#_Lanczos(n_eig_tot)

    store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
    plt.imshow(DVR.eigenstate(vecs[:,0])**2,origin='lower')
    plt.show()

vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
vals_k = vals*2*m/hbar**2

# Read eigenvalues of the whole stadium billiard for comparison
potkey_full = 'stadium_billiard_m_{}_a_{}_r_{}'.format(m,np.round(a,3),np.round(r,3))
fname_fuLl = 'Eigen_basis_{}_ngrid_{}'.format(potkey_full,ngrid)  
vals_full = read_arr('{}_vals'.format(fname_fuLl),'{}/Datafiles'.format(path))  
vals_k_full = vals_full*2*m/hbar**2

if(0): #Check thermal fluctuations vs eigenvalue spacing
    T = 10.0
    for T in [10,20,50,100]:
        beta = 1/T
        rho_T = np.exp(-beta*vals_full)
        diff = np.diff(vals_full)
        therm_fluc = T
        print('T',T,therm_fluc,diff[:5])
        plt.plot(vals_full, np.log(rho_T), label='Thermal distribution of eigenvalues')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Thermal weight')
    plt.title('Thermal distribution of eigenvalues for the stadium billiard')
    plt.legend()
    plt.show()
    exit()

stadium = StadiumBilliard(a, r)

#bdtype = 'DD'  # Assuming we are using Dirichlet boundary conditions
for bdtype in ['DD','DN','ND','NN']:
    k_values = np.linspace(8,10,1000)
    eigenstates, eigenvecs = stadium.solve_eigenstates(k_values, bdtype=bdtype, N=7)
    plt.plot(k_values**2, eigenstates)

    min_indices = (np.diff(np.sign(np.diff(eigenstates))) > 0).nonzero()[0] + 1  # Indices of local minima
    plt.plot(k_values[min_indices]**2, np.array(eigenstates)[min_indices], 'ro', label='Local minima of eigenstates')

for ev in vals_k_full[:50]:  # Plot the first 50 eigenvalues of the full stadium billiard
    plt.axvline(ev, color='r', linestyle='--', label=f'Expected eigenvalue: {ev:.2f}')

plt.xlim(min(k_values)**2, max(k_values)**2)
plt.xlabel('k^2')
plt.ylabel('Minimum singular value')
plt.title('Plane Wave Expansion Min Singular Value vs k^2')
plt.show()
