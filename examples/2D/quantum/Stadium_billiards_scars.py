import numpy as np
from matplotlib import pyplot as plt
from PISC.plane_wave.stadium_billiards import StadiumBilliard
from PISC.dvr.dvr import DVR2D
import os
from PISC.utils.readwrite import store_arr, read_arr
from PISC.engine import Krylov_complexity_2D
import pickle
from PISC.engine.integrators_hw import Integrator_HW

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

def potential_full(x,y):
    if( (x+a)**2 + y**2 < r**2 or (x-a)**2 + y**2 < r**2):
        return 0.0
    elif(x>-a and x<a and y>-r and y<r):
        return 0.0 
    else:
        return 1e6

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

if(1): # Compute the boundary of the quarter stadium billiard
    flat_x = np.linspace(0, a, 500)
    curv_x = np.linspace(a, (a + r), 500)
    curv_y = np.sqrt(r**2 - (curv_x - a)**2)

    bdy_x = np.concatenate((flat_x, curv_x))
    bdy_y = np.concatenate((r*np.ones_like(flat_x), curv_y))
    
# Create a DVR grid and solve for eigenvalues
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
    exit()

vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
vals_k = vals*2*m/hbar**2

# Read eigenvalues of the whole stadium billiard for comparison
potkey_full = 'stadium_billiard_m_{}_a_{}_r_{}'.format(m,np.round(a,3),np.round(r,3))
fname_fuLl = 'Eigen_basis_{}_ngrid_{}'.format(potkey_full,ngrid)  
vals_full = read_arr('{}_vals'.format(fname_fuLl),'{}/Datafiles'.format(path))  
vecs_full = read_arr('{}_vecs'.format(fname_fuLl),'{}/Datafiles'.format(path))
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

k_set1 = np.linspace(0.1, 8, 1000)
N1 = 4
k_set2 = np.linspace(8, 12, 1000)
N2 = 7
k_set3 = np.linspace(12, 17, 1000)
N3 = 10
k_set4 = np.linspace(17, 23, 1000)
N4 = 12
k_set5 = np.linspace(23, 26.5, 1000)
N5 = 14
k_set6 = np.linspace(26.5, 30, 1000)
N6 = 15
k_set7 = np.linspace(30, 32, 1000)
N7 = 16
k_set8 = np.linspace(32, 34, 1000)
N8 = 18
k_set9 = np.linspace(34, 36, 1000)
N9 = 19
k_set10 = np.linspace(36, 37.5, 1000)
N10 = 21
k_set11 = np.linspace(37.5, 39, 1000)
N11 = 22
k_set12 = np.linspace(39, 41.5, 1000)
N12 = 23
k_set13 = np.linspace(41.5, 43, 1000)
N13 = 25
k_set14 = np.linspace(43, 45, 1000)
N14 = 27

k_sets = [k_set1, k_set2, k_set3, k_set4, k_set5, k_set6, k_set7, k_set8, k_set9, k_set10]
N_values = [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]

if(0): # Manually Scan for allowed k values and the required number of plane waves
    k_values = np.linspace(43, 45, 1000)
    for bdtype in ['DD','DN','ND','NN']:
        eigenstates, eigenvecs = stadium.solve_eigenstates(k_values, bdtype=bdtype, N=27)
        min_indices = (np.diff(np.sign(np.diff(eigenstates))) > 0).nonzero()[0] + 1  # Indices of local minima
        
        if(1):
            plt.plot(k_values**2, eigenstates)
            plt.plot(k_values[min_indices]**2, np.array(eigenstates)[min_indices], 'ro', label='Local minima of eigenstates')
    
    for ev in vals_k_full[:150]:  # Plot the first 50 eigenvalues of the full stadium billiard
        plt.axvline(ev, color='r', linestyle='--', label=f'Expected eigenvalue: {ev:.2f}')

    plt.xlim(min(k_values)**2, max(k_values)**2)
    plt.xlabel('k^2')
    plt.ylabel('Minimum singular value')
    plt.title('Plane Wave Expansion Min Singular Value vs k^2')
    plt.show()
    exit()

try:
    #raise FileNotFoundError
    print('Trying to load allowed k values from file...')
    sorted_states = pickle.load(open('sorted_states.pkl', 'rb'))
except FileNotFoundError:
    print('Allowed k values file not found. Computing allowed k values...')
    allowed_k = []
    allowed_vectors = []
    
    # We store the allowed k values, corresponding coefficients from the plane wave expansion, and the respective boundary
    # condition type for each local minimum so that we can reconstruct the wavefunction as required.
    allowed_states = [] 

    for k_values, N in zip(k_sets, N_values):
        for bdtype in ['DD','DN','ND','NN']:
            eigenstates, eigenvecs = stadium.solve_eigenstates(k_values, bdtype=bdtype, N=N)
            min_indices = (np.diff(np.sign(np.diff(eigenstates))) > 0).nonzero()[0] + 1  # Indices of local minima
            for idx in min_indices:
                allowed_states.append( (k_values[idx], (eigenvecs[idx], bdtype)) )
            
    k_vals = [np.array(allowed_states[i][0]) for i in range(len(allowed_states))]
    sort_ind = np.argsort(np.array(k_vals)**2)
    sorted_states = []
    for i in range(len(sort_ind)):
        sorted_states.append(allowed_states[sort_ind[i]])
    
    pickle.dump(sorted_states, open('sorted_states.pkl', 'wb'))

allowed_k = np.array([state[0] for state in sorted_states])

print('Allowed k values from plane wave expansion:', (allowed_k[:20])**2)
print('Eigenvalues from DVR:', vals_k_full[:20])


if(0): #Compare the wavefunction from DVR and plane wave expansion for the nth state
    n = 53
    fig,ax = plt.subplots(1,2,figsize=(12,6))

    for n in [67]:#range(60,70):#100):
        DVR_wf = DVR.eigenstate(vecs_full[:,n])
        PW_wf = stadium.compute_wf(sorted_states[n][1][0], sorted_states[n][0], sorted_states[n][1][1])
        PW_wf_full = stadium.full_stadium_wf(PW_wf)

        ax[0].cla()
        ax[1].cla()

        ax[0].imshow(np.abs(DVR_wf)**2, origin='lower', extent=(-a-r, a + r, -r, r))
        ax[0].set_title(f'DVR Wavefunction (n={n})')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        #fig.colorbar(im0, ax=ax[0], label='|Psi|^2')

        ax[1].imshow(np.abs(PW_wf_full)**2, origin='lower', extent=(-a-r, a + r, -r, r))
        #ax[1].imshow(V_full, extent=(-a-r, a + r, -r, r), origin='lower', alpha=0.3, cmap='gray')
        ax[1].set_title(f'Plane Wave Expansion Wavefunction (n={n})')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        #fig.colorbar(im1, ax=ax[1], label='|Psi|^2')

        for axe in ax:
            axe.plot(bdy_x, bdy_y, color='r',linewidth=2)  # Plot the boundary of the quarter stadium
            axe.plot(-bdy_x, bdy_y, color='r',linewidth=2)  # Plot the boundary of the quarter stadium reflected across x=0
            axe.plot(bdy_x, -bdy_y, color='r',linewidth=2)  # Plot the boundary of the quarter stadium reflected across y=0
            axe.plot(-bdy_x, -bdy_y, color='r',linewidth=2)  # Plot the boundary of the quarter stadium reflected across both axes

        plt.draw()
        plt.pause(1.0)  # Pause to visually inspect each state before moving to the next
    #plt.ioff()
    plt.tight_layout()
    plt.show()
    exit()

if(0): # Plot the distribution of the wavefunction amplitude for the nth state to check for scarring
    fig,ax = plt.subplots(1,2,figsize=(12,6))
    for n in range(60,70):
        PW_wf = stadium.compute_wf(sorted_states[n][1][0], sorted_states[n][0], sorted_states[n][1][1])
        PW_wf_full = stadium.full_stadium_wf(PW_wf)
        ax[0].cla()
        log_amplitude = (np.abs(PW_wf_full.flatten())**2)
        ax[0].hist(log_amplitude, bins=50, density=True, alpha=0.7, color='blue')
        ax[0].set_yscale('log')
        #ax[0].set_xscale('log')
        ax[0].set_title(f'Amplitude Distribution for State n={n}')
        ax[0].set_xlabel('|Psi|^2')
        ax[0].set_ylabel('Probability Density')
        
        ax[1].cla()
        ax[1].imshow(np.abs(PW_wf_full)**2, origin='lower', extent=(-a-r, a + r, -r, r))
        ax[1].set_title(f'Wavefunction Amplitude for State n={n}')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')

        plt.draw()
        plt.pause(3.5)
    plt.show()

if(0):# Compute IPR for each state:
    narr = np.arange(50,100,dtype=int)
    IPR_values = []
    for n in narr:
        PW_wf = stadium.compute_wf(sorted_states[n][1][0], sorted_states[n][0], sorted_states[n][1][1])
        PW_wf_full = stadium.full_stadium_wf(PW_wf)
        IPR = np.sum(np.abs(PW_wf_full)**4) / (np.sum(np.abs(PW_wf_full)**2)**2)
        IPR_values.append(IPR)

    plt.plot(narr, IPR_values, marker='o')
    plt.xlabel('State index n')
    plt.ylabel('Inverse Participation Ratio (IPR)')
    plt.title('IPR of the first 50 states from plane wave expansion')
    plt.grid()
    plt.show()  

if(1):
    x = np.linspace(-a-r, a + r, 1000)
    y = np.linspace(-r, r, 2000)
    X, Y = np.meshgrid(x, y)

    sigma = 0.05
    x0 = 0.0
    y0 = 0.0
    gauss_pref = np.zeros_like(X)
    for q0 in np.linspace(-3*a/4,3*a/4,5):
        x0 = q0
        y0 = q0
        gauss_pref+= np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        

    #theta = np.pi/4
    #k0 = 10.0
    
    klist= []
    overlaps = []
    for n in range(100):
        k0 = sorted_states[n][0]
        PW_wf = stadium.compute_wf(sorted_states[n][1][0], sorted_states[n][0], sorted_states[n][1][1])
        PW_wf_full = stadium.full_stadium_wf(PW_wf)
        theta = np.pi/4
        kx = k0 * np.cos(theta)
        ky = k0 * np.sin(theta)
        plane_wave = np.cos(kx*X)*np.cos(ky*Y)
        psi0 = gauss_pref * plane_wave
        overlap = abs(np.sum(psi0 * PW_wf_full))**2
        klist.append(k0)
        overlaps.append(overlap)

    plt.plot(np.arange(100), np.log(overlaps), marker='o')
    plt.xlabel('State index n')
    plt.ylabel('Overlap with Gaussian wavepacket')
    plt.title('Overlap of eigenstates with a Gaussian wavepacket centered at the origin')
    plt.grid()
    plt.show()

    if(0):
        plt.imshow(psi0, origin='lower', extent=(-a-r, a + r, -r, r))
        plt.colorbar(label='Initial Wavefunction Amplitude')
        plt.title('Initial Wavefunction (Gaussian wavepacket)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

if(0): #Draw classical periodic orbits
    def f(q):
        x, y = q
        if abs(x) <= a:
            return abs(y) - r
        else:
            return (x - np.sign(x)*a)**2 + y**2 - r**2

    def grad_f(q):
        x,y = q
        if abs(x) <= a:
            return np.array([0.0, np.sign(y)])
        else:
            dx = 2*(x - np.sign(x)*a)
            dy = 2*y
            return np.array([dx, dy])


    integrator = Integrator_HW(f)
    integrator.bind(f, grad_f)
    #r0 = np.array([0.999*a,0.999*r])
    #v0 = np.array([1.0, 1.0])
    
    r0 = np.array([0.0, 0.0])
    v0 = np.array([np.cos(np.pi/6), np.sin(np.pi/6)])
    

    print('f', f(r0), 'grad_f', grad_f(r0))
    t_final = 10.0
    r_final, v_final, traj = integrator.propagate(r0, v0, t_final, record_traj=True)
    coll_times = integrator.collision_times
    print("Collision times:", coll_times)
    print("Final position:", r_final)
    
    mul = 4
    fig, ax = plt.subplots(figsize=(mul*(2*a+2*r), mul*(2*r)))
    # Plot the stadium boundary
    x = np.linspace(-a-r, a+r, 100)
    y_top = np.where(abs(x) <= a, r, np.sqrt(r**2 - (x - np.sign(x)*a)**2))
    y_bottom = np.where(abs(x) <= a, -r, -np.sqrt(r**2 - (x - np.sign(x)*a)**2))
    ax.plot(x, y_top, 'r-', label='Boundary')
    ax.plot(x, y_bottom, 'r-')
    

    x_traj = [r[0] for t, r, v in traj]
    x_traj.append(r_final[0])
    y_traj = [r[1] for t, r, v in traj]
    y_traj.append(r_final[1])   

    dx = np.diff(x_traj)
    dy = np.diff(y_traj)
    ax.quiver(x_traj[:-1], y_traj[:-1], dx, dy, angles='xy', scale_units='xy', scale=1, color='b', label='Trajectory')
    ax.scatter(x_traj, y_traj)

    if(0): # Plot normal vectors at collision points
        for t, r, v in traj:
            if abs(f(r)) < integrator.tol:
                n = grad_f(r)
                n = n / np.linalg.norm(n)
                ax.quiver(r[0], r[1], n[0], n[1], angles='xy', scale_units='xy', scale=0.5, color='g', label='Normal Vector')


    plt.show()

"""
Tasks tomorrow:
    1. Store the eigenvectors as well and preserve the order when eigenvalues are sorted.
    2. Add a class function to stadium_billiards.py to compute the wavefunction from the plane wave expansion coefficients
    3. Add another class function to compute the full wavefunction from the quarter stadium wavefunction using symmetry.
    4. Plot the wavefunction corresponding to the local minima in the plane wave expansion and compare with the DVR wavefunctions.
    5. Write code to compute IPR and see if it distinguishes scarred vs non-scarred states.
    6. Also, find the Lanczos sequence on the scarred and non-scarred states to see if they differ in their complexity growth.
"""

