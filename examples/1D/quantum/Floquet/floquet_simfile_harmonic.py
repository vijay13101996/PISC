import numpy as np
from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from multiprocessing import Pool

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
    

print('vals',np.around(vals[:5],3), np.diff(vals[:5]))

key_list = ['gaussian_decay', 'exponential_decay', 'no_decay', 'pos_mat']
key_list = ['pos_mat']
O = pos_mat
if(0):
    asymp = 2*L/np.pi**2
    Of = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(1,neigs):
            if(abs(i-j)%2==1):
                #Define Of such that Of[n,n+1] = asymp 
                #Of[i,j] = asymp*np.exp(-abs(i-j))/np.exp(-1) # exponential_decay_dir
                Of[i,j] = asymp/(abs(j-i)**2*(i+1)**0.5)
                Of[j,i] = Of[i,j]

if(1):
    Of = np.zeros((neigs,neigs))
    for i in range(neigs-1):
        Of[i,i+1] = i
        Of[i+1,i] = i

if(0):
    plt.imshow(abs(Of[:50,:50]), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Position matrix')
    plt.show()
    exit()

if(0):
    #Plot O[n,n+1] vs n
    lenO = 50
    Ofnn = np.array([Of[n,n+1] for n in range(lenO)])
    Onn = np.array([O[n,n+1] for n in range(lenO)])
    plt.plot(np.arange(lenO), abs(Onn))
    plt.plot(np.arange(lenO), abs(Ofnn))
    plt.title('O[n,n+1] vs n')
    plt.xlabel('n')
    plt.ylabel('O[n,n+1]')
    plt.show()
    exit()

O = Of
key_list = ['linear_growth']


w_F_arr = np.arange(0.5, 30.01, 0.25)
w_F_arr = np.around(w_F_arr, decimals=2)

g = 3.0

if(1):
    with Pool(processes=10) as pool:
        for key in key_list:
            pool.starmap(load_U_F, [(potkey, g, w_F, vals, O, key, True) for w_F in w_F_arr])

    exit()

with Pool(processes=10) as pool:
    for key in key_list:
        #pool.starmap(load_U_F, [(potkey, g, w_F, vals, pos_mat, key, True) for w_F in w_F_arr])
        pool.starmap(load_U_F, [(potkey, g, w_F, vals, pos_mat, key, True) for g in g_arr])
exit()

for key in key_list:
    load_U_F(potkey, g, w_F, vals, pos_mat, key, force_recompute=True)




