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

    potkey = 'L_{}_1D_box_MANU'.format(L)
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

    mom_mat = np.zeros((neigs,neigs)) + 0j
    for i in range(neigs):
        for j in range(neigs):
            mom_mat[i,j] = 1j*m*(vals[i]-vals[j])*pos_mat[i,j]

    #Check hermiticity of pos_mat and mom_mat
    #print('pos_mat hermitian?', np.allclose(pos_mat, pos_mat.T.conj()))
    #print('mom_mat hermitian?', np.allclose(mom_mat, mom_mat.T.conj()))

    #Verify if mom_mat is indeed the matrix representation of the momentum operator in the energy eigenbasis
    psq = mom_mat @ mom_mat
    psq= np.around(psq, decimals=2)
    print('psq', np.diag(psq)[:5], 2*vals[:5])
    print('mom_mat', mom_mat.shape, pos_mat.shape)
     
    if(0):#Plot post_mat[n,n+1] vs n
        pnn1 = np.array([mom_mat[n,n+1] for n in range(neigs-1)])
        plt.plot(np.arange(neigs-1), abs(pnn1))
        plt.title('Position matrix elements |<n|x|n+1>| vs n')
        plt.xlabel('n')
        plt.ylabel('|<n|x|n+1>|')
        plt.grid()
        plt.show()

print('vals',np.around(vals[:5],3), np.diff(vals[:5]))

key_list = ['gaussian_decay', 'exponential_decay', 'no_decay', 'pos_mat']
key_list = ['pos_mat']
O = pos_mat
key_list = ['0.1mom_mat']
O = mom_mat
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
        Of[i+1,i] = Of[i,i+1]

if(1): #Retain only n,n+1 and n+1,n elements of O
    Of2 = np.zeros((neigs,neigs)) +0j
    for i in range(neigs-1):
        Of2[i,i+1] = O[i,i+1]
        Of2[i+1,i] = O[i,i+1]

if(0):
    plt.imshow(abs(Of2[:50,:50]), cmap='hot', interpolation='nearest')
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
    plt.plot(np.arange(lenO), abs(Ofnn),color='r')
    plt.title('O[n,n+1] vs n')
    plt.xlabel('n')
    plt.ylabel('O[n,n+1]')
    plt.show()
    exit()

O = Of2
#key_list = ['linear_growth']
key_list = ['mom_mat_NN']

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




