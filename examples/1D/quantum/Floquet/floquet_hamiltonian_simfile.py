import numpy as np
from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from multiprocessing import Pool
import time
"""
Create one unified file for all Floquet simulations.
"""

if(1): # 1D Box
    m = 1.0
    L = 2.0
    ngrid = 2**9 
    neigs = ngrid-1

    lb = 0
    ub = L

    def pot(x):
        if(x>L or x<0):
            return 1e10
        else:
            return 0.0

    potkey = 'L_{}_1D_box_0'.format(L)
    DVR = DVR1D(ngrid, lb, ub,m, pot)
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    x_arr = DVR.grid[1:ngrid]
    dx = x_arr[1]-x_arr[0]
    
    def pos_mat_anal(i,j,neigs):
        if(i==j):
            return L/2
        else:
            return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

    vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
    vecs_anal = np.zeros((neigs,ngrid-1))
    for i in range(neigs):
        vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:ngrid]/L)

    vals = vals_anal
    vecs = vecs_anal

    O_anal = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(neigs):
            O_anal[i,j] = pos_mat_anal(i,j,neigs)
    
    pos_mat = O_anal - np.diag(L/2*np.ones(neigs))

if(0): # Hexic oscillator
    m = 1.0
    L = 6.0
    ngrid = 2**9

    lb = -L
    ub = L
    def pot(x):
        return x**6

    potkey = 'L_{}_hexic'.format(L)

    DVR = DVR1D(ngrid,lb,ub,m,pot)
    neigs = 200 # :ngrid-1
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    if(0):
        plt.plot(DVR.grid[1:ngrid], vecs[:,-1])
        plt.title('Hexic oscillator, L={}'.format(L))
        plt.xlabel('x')
        plt.ylabel('Wavefunction')
        plt.grid()
        plt.show()
        exit()

    x_arr = DVR.grid[:ngrid]
    dx = x_arr[1]-x_arr[0]

    pos_mat = np.zeros((neigs,neigs))
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

if(0): # Quartic oscillator
    m = 1.0
    L = 10.0
    n_anharm = 4
    omega = 0.0
    a = 0.0
    b = 1.0
    ngrid = 2**9 

    lb = -L
    ub = L
    pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

    potkey = 'L_{}_quartic'.format(L)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    neigs = 200 # :ngrid-1
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    if(0):
        plt.plot(DVR.grid[1:ngrid], vecs[:,-1])
        plt.title('Quartic oscillator, L={}, n_anharm={}'.format(L,n_anharm))
        plt.xlabel('x')
        plt.ylabel('Wavefunction')
        plt.grid()
        plt.show()
        exit()

    diff0 = vals[1]-vals[0]
    diff1 = vals[2]-vals[1]

    print('vals',vals[0])

    print('diff',vals[1]-vals[0],vals[2]-vals[1],vals[3]-vals[2])

    x_arr = DVR.grid[:ngrid]
    dx = x_arr[1]-x_arr[0]

    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

    def tunneling_factor(p):
        num = gamma( (p-2)/(2*p) )
        denom = gamma( (p-1)/p )
        pref = np.sqrt(2*np.pi)/(p+2)
        ret = pref*(num/denom)*energy_factor(p)
        return ret
    
    O_landau = np.zeros((neigs,neigs))
    tf = tunneling_factor(n_anharm)
    print('tf', tf)
    for i in range(neigs):
        for j in range(1,neigs):
            if (i-j) % 2 == 0 and abs(i-j) > 0:
                O_landau[i,j] = np.exp(-tf*abs(i-j))
                O_landau[j,i] = O_landau[i,j]

    

if(0): # Harmonic oscillator
    m = 1.0
    omega = 1.0
    ngrid = 2**9
    L = 20.0
    
    lb = -L
    ub = L
    pes = mildly_anharmonic(m,0.0,0.0,w=omega,n=4)  

    potkey = 'L_{}_harmonic'.format(L)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    neigs = 200 # ngrid-1
    vals,vecs = DVR.Diagonalize(neig_total=neigs)
    
    vals_anal = (np.arange(0,neigs) + 0.5)*omega

    x_arr = DVR.grid[:ngrid]
    dx = x_arr[1]-x_arr[0]

    pos_mat = np.zeros((neigs,neigs))
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

    pos_mat_anal = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(i,neigs):
            if abs(i-j) == 1:
                pos_mat_anal[i,j] = np.sqrt(j/2.0)
            else:
                pos_mat_anal[i,j] = 0.0
            pos_mat_anal[j,i] = pos_mat_anal[i,j]
    
    print('Position matrix (numerical):\n', np.around(pos_mat[100:110,100:110],3))
    print('Position matrix (analytical):\n', np.around(pos_mat_anal[100:110,100:110],3))

    print('vals (numerical):', np.round(vals[180:190],3))
    print('vals (analytical):', np.round(vals_anal[180:190],3))
    #assert np.allclose(abs(pos_mat), abs(pos_mat_anal), atol=1e-1), "Position matrix does not match analytical result"
    #print('Position matrix matches analytical result')

    pos_mat = pos_mat_anal
    
g = 3.0
w_F = 5.0
V = g*pos_mat
n_ph = 5

w_F_arr = np.arange(4.0, 6.0, 0.1)

loc = '/scratch2/vgs23/PISC/examples/1D/quantum/Floquet/Datafiles/'
for w_F in w_F_arr:
    try:
        fl_vals = np.load(loc+'floquet_vals_{}_g_{}_w_F_{}_n_ph_{}_{}.npy'.format(potkey, g, w_F, n_ph, 'pos_mat'))
        #fl_vecs = np.load(loc+'floquet_vecs_{}_g_{}_w_F_{}_n_ph_{}_{}.npy'.format(potkey, g, w_F, n_ph, 'pos_mat'))
        
        fl_vals = np.array(fl_vals)[:100]
        print('Loaded Floquet quasi-energies and vectors from file.')
    except:
        print('Computing Floquet quasi-energies and vectors for w_F =', w_F)
        start = time.time()
        floquet_vals, floquet_vecs = Floquetiser(vals, w_F, V, n_ph)
        print('Floquetiser time:', time.time()-start)
        fl_vals = np.array(floquet_vals)
        #fl_vecs = np.array(floquet_vecs)
        np.save(loc+'floquet_vals_{}_g_{}_w_F_{}_n_ph_{}_{}.npy'.format(potkey, g, w_F, n_ph, 'pos_mat'), fl_vals)
        #np.save(loc+'floquet_vecs_{}_g_{}_w_F_{}_n_ph_{}_{}.npy'.format(potkey, g, w_F, n_ph, 'pos_mat'), fl_vecs)

    plt.plot(w_F*np.ones(len(fl_vals)), fl_vals, 'o', label='w_F = {}'.format(w_F))

E0 = vals[0]
E1 = vals[1]
E2 = vals[2]
for E in [E0, E1, E2]:
    plt.axhline(y=E, color='r', linestyle='--', label='E = {}'.format(E))

plt.title('Floquet quasi-energies')
plt.xlabel('w_F')
plt.ylabel('Quasi-energy')
plt.grid()
plt.show()






