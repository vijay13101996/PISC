import numpy as np
import matplotlib.pyplot as plt
from PISC.plane_wave.stadium_billiards import StadiumBilliard
from PISC.dvr.dvr import DVR2D
import os
from PISC.utils.readwrite import store_arr, read_arr

if(0): #Test code for plane-wave expansion and eigenstate search with square/rectangular billiard
    a = 1.0  # X Length of straight sections
    r = 2.0  # Y Length of straight sections 
    num_points = 20  # Number of boundary points
    stadium = StadiumBilliard(a, r, num_points)
    k_values = np.linspace(13,16,1000)
    eigenstates = stadium.solve_eigenstates(k_values, bdtype='DD')

    boxvals = np.array([np.pi**2*(n**2/a**2 + m**2/r**2) for n in range(1,10) for m in range(1,10)])
    boxvals = np.sort(boxvals)
    evs = boxvals[:50]

    for ev in evs:
        plt.axvline(ev, color='r', linestyle='--', label=f'Expected eigenvalue: {ev:.2f}')

    plt.plot(k_values**2, eigenstates)
    plt.xlim(min(k_values)**2, max(k_values)**2)
    plt.xlabel('Wave number k')
    plt.ylabel('Minimum singular value')
    plt.title('Eigenstate Search for Square/Rectangular Billiard (DD boundary conditions)')
    plt.show()

if(1): #Benchmarking code with stadium billiards eigenvalues obtained by DVR
    a = 1.0  # Length of straight sections
    r = 1.0  # Radius of semicircular sections

    potkey = 'quart_stadium_billiard_a_{}_r_{}'.format(a,r)
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
        plt.imshow(V, extent=(0, a + r, 0, r), origin='lower')
        plt.colorbar(label='Potential')
        plt.title('Potential Landscape of the Stadium Billiard')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    # Create a DVR grid and solve for eigenvalues
    #System parameters
    m = 1.0
    L = (a+r)*1
    offset = 0.001
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
    #print('Vs',pes.potential_xy(0,0))
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

    num_points = 20  # Number of boundary points
    bdtype = 'DD'  # Assuming we are using Dirichlet boundary conditions
    stadium = StadiumBilliard(a, r, num_points)

    k_set1 = np.linspace(0,5,1000)
    k_set2 = np.linspace(5,8,1000)
    k_set3 = np.linspace(8,12,1000)
    N1 = 3
    N2 = None
    N3 = None
    allowed_k = []
    for k_values, N in zip([k_set1, k_set2, k_set3], [N1, N2, N3]):
        eigenstates, eigenvecs = stadium.solve_eigenstates(k_values, bdtype=bdtype, N=N)
    
        #Find minima in eigenstates and plot dots at those locations
        min_indices = (np.diff(np.sign(np.diff(eigenstates))) > 0).nonzero()[0] + 1  # Indices of local minima
        allowed_k.extend(k_values[min_indices])
    
    allowed_k = np.array(allowed_k)
    print('Allowed k values at local minima:', allowed_k**2, 'Expected eigenvalues from DVR:', vals_k[:15])
    print('% Error', 100*np.abs(allowed_k**2 - vals_k[:len(allowed_k)])/vals_k[:len(allowed_k)])
    

    if(0): #Plot the corresponding wavefunctions at the local minima 
        points = stadium.boundary_points(num_points)
        
        for idx in min_indices:
            k_min = k_values[idx]
            c = eigenvecs[idx]
            Psi = np.zeros_like(X)
            N = len(c) 
            for j in range(N):
                if bdtype == 'DD':
                    wave_j = stadium.DD_wf(X, Y, j, k_min, N)
                elif bdtype == 'DN':
                    wave_j = stadium.DN_wf(X, Y, j, k_min, N)
                elif bdtype == 'ND':
                    wave_j = stadium.ND_wf(X, Y, j, k_min, N)
                elif bdtype == 'NN':
                    wave_j = stadium.NN_wf(X, Y, j, k_min, N)
                    
                Psi += c[j] * wave_j
            if(1): #Full Stadium WF
                psi_full = np.zeros((2*Psi.shape[0], 2*Psi.shape[1]))  
                psi_full[Psi.shape[0]:, Psi.shape[1]:] = Psi
                psi_full[Psi.shape[0]:, :Psi.shape[1]] = Psi[:, ::-1]
                psi_full[:Psi.shape[0], Psi.shape[1]:] = Psi[::-1, :]
                psi_full[:Psi.shape[0], :Psi.shape[1]] = Psi[::-1, ::-1]
                plt.imshow(np.abs(psi_full)**2, extent=(-a-r, a + r, -r, r), origin='lower')

                x_points = points[:,0]
                y_points = points[:,1]
                plt.plot(x_points, y_points, color='k', marker='o', linestyle='None')
                plt.plot(-x_points, y_points, color='k', marker='o', linestyle='None')
                plt.plot(x_points, -y_points, color='k', marker='o', linestyle='None')
                plt.plot(-x_points, -y_points, color='k', marker='o', linestyle='None')
                
                #Draw a line through x=0 and y=0 to show the symmetry axes
                plt.axhline(0, color='k', linestyle='--')
                plt.axvline(0, color='k', linestyle='--')
                plt.colorbar(label='|Psi|^2')
                plt.title(f'Wavefunction at local minimum k={k_min:.2f}')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()


    plt.plot(k_values[min_indices]**2, np.array(eigenstates)[min_indices], 'ro', label='Local minima of eigenstates')

    for ev in vals_k[:50]:
        plt.axvline(ev, color='r', linestyle='--', label=f'Expected eigenvalue: {ev:.2f}')

    plt.plot(k_values**2, eigenstates)
    plt.xlim(min(k_values)**2, max(k_values)**2)
    plt.xlabel('Wave number k')
    plt.ylabel('Minimum singular value')
    plt.title('Eigenstate Search for Stadium Billiard (DD boundary conditions)')
    plt.show()

