import numpy as np
from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from multiprocessing import Pool

m = 1.0
L = 6
n_anharm = 6 #32
omega = 0.0
a = 0.0
b = 1.0
ngrid = 2**10 

lb = -L
ub = L
pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

potkey = 'L_{}_anharmonic_{}_MANU'.format(L,n_anharm)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 400 # :ngrid-1
vals,vecs = DVR.Diagonalize(neig_total=neigs)

if(0):
    plt.plot(DVR.grid[1:ngrid], DVR.potential(DVR.grid[1:ngrid]))
    plt.plot(DVR.grid[1:ngrid], DVR.grid[1:ngrid]**n_anharm)
    plt.show()

    plt.plot(DVR.grid[1:ngrid], vecs[:,-1])
    plt.title('Anharmonic oscillator, L={}, n_anharm={}'.format(L,n_anharm))
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

if(0):
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

  

print('vals',np.around(vals[:5],3), np.diff(vals[:5]))

key_list = ['gaussian_decay', 'exponential_decay', 'no_decay', 'pos_mat']
key_list = ['pos_mat']
O = pos_mat

if(1):
    asymp = O[0,1]
    Of = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(1,neigs):
            if(abs(i-j)%2==1):
                #Define Of such that Of[n,n+1] = asymp 
                #Of[i,j] = asymp*np.exp(-abs(i-j))/np.exp(-1) # exponential_decay_dir
                #Of[i,j] = asymp/abs(j-i)**2
                Of[i,j] = asymp/(i+1)**(2.0/3)
                Of[j,i] = Of[i,j]

if(0):
    plt.imshow(abs(O[:50,:50]), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Position matrix')
    plt.show()
    exit()

if(0):
    #Plot O[n,n+1] vs n
    lenO = 199
    Ofnn = np.array([Of[n,n+1] for n in range(lenO)])
    Onn = np.array([O[n,n+1] for n in range(lenO)])
    plt.plot(np.arange(lenO), abs(Onn))
    plt.loglog(np.arange(lenO), abs(Ofnn))
    plt.title('O[n,n+1] vs n')
    plt.xlabel('n')
    plt.ylabel('O[n,n+1]')
    plt.show()
    exit()

w_F_arr = np.arange(0.5, 30.01, 0.25)
w_F_arr = np.around(w_F_arr, decimals=2)

#O = Of
#key_list = ['dir_pl_decay_inv']

g = 3.0
if(0): #Compare different methods of computing U_F
    U_F_expm, qE_expm, qS_expm = load_U_F(potkey, g, w_F, vals, pos_mat, 'pos_mat', True, method='expm')
    U_F,qE, qS = load_U_F(potkey, g, w_F, vals, pos_mat, 'pos_mat', True, method='split')
    U_F_X, qE_X, qS_X = load_U_F(potkey, g, w_F, vals, pos_mat, 'pos_mat', True, method='split_X')

    print('U_F', U_F[:5,:5])
    print('U_F_X', U_F_X[:5,:5])
    assert np.allclose(U_F, U_F_X, atol=1e-5), "U_F and U_F_X are not close enough"
    assert np.allclose(U_F, U_F_expm, atol=1e-2), "U_F and U_F_expm are not close enough"
    exit()

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




