import numpy as np
from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from multiprocessing import Pool
from scipy.special import gamma

"""
Create one unified file for all quartic oscillator Floquet simulations
"""

# Quartic oscillator
m = 1.0
L = 4.0
n_anharm = 8
omega = 0.0
a = 0.0
b = 1.0
ngrid = 2**9 

lb = -L
ub = L
pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

potkey = 'L_{}_octic'.format(L)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 200 # :ngrid-1
vals,vecs = DVR.Diagonalize(neig_total=neigs)
#potkey = 'L_{}_octic'.format(L, neigs)

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

def J(p):
    num = gamma(1/p)*gamma(1.5)
    denom = p*gamma(1/p+1.5)
    return num/denom

def energy_factor(p):
    num = np.pi/(2*np.sqrt(2*m))
    denom = J(p)
    pref = (num/denom)**(2*p/(p+2))
    return pref

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
        if (i-j) % 2 == 1 and abs(i-j) > 0:
            O_landau[i,j] = np.exp(-tf*abs(i-j))
            O_landau[j,i] = O_landau[i,j]
if(0):
    print('pos_mat', pos_mat[0,:])
    print('O_landau', np.log(abs(O_landau[0,:])))

    log_landau = np.log(abs(O_landau[0,1::2]))
    #Replace -inf with a large negative number for plotting
    log_landau[log_landau == -np.inf] = -100
    log_pos_mat = np.log(abs(pos_mat[0,1::2]))
    plt.plot(np.arange(neigs)[1::2], log_pos_mat, label='pos_mat')
    plt.plot(np.arange(neigs)[1::2], log_landau, label='Landau')  
    plt.title('Quartic oscillator, L={}, n_anharm={}'.format(L,n_anharm))
    plt.xlabel('State index')
    plt.ylabel('Log of matrix element')
    plt.legend()
    plt.grid()
    plt.show()
    exit()
    
print('vals',np.around(vals[:5],3), np.diff(vals[:5]))

key_list = ['landau']#'gaussian_decay', 'exponential_decay', 'no_decay', 'pos_mat']
#w_F_arr = np.arange(0.5,5.01,0.1)
#w_F_arr = np.linspace(1.0, 5.0, 21)
#w_F_arr = np.linspace(0.2, 3.0, 29)
#w_F_arr = np.linspace(20.0, 30.0, 21)
#w_F_arr = np.linspace(5.0, 10.0, 21)
w_F_arr = np.arange(0.5, 30.01, 0.25)

g = 3.0
w_F = 10.0


if(0): #Compare different methods of computing U_F
    U_F_expm, qE_expm, qS_expm = load_U_F(potkey, g, w_F, vals, pos_mat, 'pos_mat', True, method='expm')
    U_F,qE, qS = load_U_F(potkey, g, w_F, vals, pos_mat, 'pos_mat', True, method='split')
    U_F_X, qE_X, qS_X = load_U_F(potkey, g, w_F, vals, pos_mat, 'pos_mat', True, method='split_X')

    print('U_F', U_F[:5,:5])
    print('U_F_X', U_F_X[:5,:5])
    assert np.allclose(U_F, U_F_X, atol=1e-5), "U_F and U_F_X are not close enough"
    assert np.allclose(U_F, U_F_expm, atol=1e-2), "U_F and U_F_expm are not close enough"
    exit()

with Pool(processes=10) as pool:
    pool.starmap(load_U_F, [(potkey, g, w_F, vals, pos_mat, 'pos_mat', True) for w_F in w_F_arr])

with Pool(processes=10) as pool:
    pool.starmap(load_U_F, [(potkey, g, w_F, vals, O_landau, 'landau', True) for w_F in w_F_arr])

exit()
with Pool(processes=10) as pool:
    for key in key_list:
        pool.starmap(load_U_F, [(potkey, g, w_F, vals, O_landau, key, True) for w_F in w_F_arr])
        pool.starmap(load_U_F, [(potkey, g, w_F, vals, pos_mat, key, True) for w_F in w_F_arr])
        #pool.starmap(load_U_F, [(potkey, g, w_F, vals, pos_mat, key, True) for g in g_arr])
exit()




