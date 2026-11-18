import numpy as np
from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, compute_U_F, compute_U_F_direct
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from multiprocessing import Pool

m = 1.0
L = 20.0
n_anharm = 4
omega = 1.0
a = 0.0
b = 0.0
ngrid = 2**9 

lb = -L
ub = L
pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

potkey = 'L_{}_MAH_w_{}_a_{}_b_{}_n_{}'.format(L,omega,a,b,n_anharm)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = ngrid-1
vals,vecs = DVR.Diagonalize(neig_total=neigs)

diff0 = vals[1]-vals[0]
diff1 = vals[2]-vals[1]

print('vals',vals[0])

print('diff',vals[1]-vals[0],vals[2]-vals[1],vals[3]-vals[2])

x_arr = DVR.grid[:ngrid]
dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

#Create fictional operator with matrix elements that are random, exponentially decaying and gaussian decaying
X = np.zeros((neigs,neigs))
key = 'gaussian_decay'
key = 'exponential_decay'
#key = 'no_decay'
key = 'pos_mat'
for i in range(neigs):
    for j in range(i, neigs):
            if(key=='gaussian_decay'):
                X[i,j] = np.exp(-np.abs(i-j)**2/2) 
            elif(key=='exponential_decay'):
                X[i,j] = np.exp(-np.abs(i-j)*2)
            elif(key=='no_decay'):
                X[i,j] = 1.0 # np.random.normal(0,1)
            X[j,i] = X[i,j]


if key == 'pos_mat':
    X = pos_mat

w_F = vals[0] + 2.0 

g_opt = 3.0
T = 2*np.pi/w_F

print('w_F', w_F, 'T', T, 'g_opt', g_opt)


def compute_U_F(g_opt, w_F, vals, X, key):
    try:
        #raise Exception("Force recomputation of U_F, qE, qS")
        U_F = np.load('U_F_harmonic_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key))
        qE = np.load('qE_harmonic_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key))
        qS = np.load('qS_harmonic_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key))
    except:
        #Compute the unitary matrix U_F in the eigenbasis of H0 and its eigenvalues and eigenvectors
        U_F, qE, qS = compute_U_F_direct(vals, X, g_opt, w_F, n_steps=2000)#,method='expm')
        np.save('U_F_harmonic_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key), U_F)
        np.save('qE_harmonic_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key), qE)
        np.save('qS_harmonic_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key), qS)
    return U_F, qE, qS

key_list = ['gaussian_decay', 'exponential_decay', 'no_decay', 'pos_mat']
with Pool(processes=4) as pool:
    pool.starmap(compute_U_F, [(g_opt, w_F, vals, X, key) for key in key_list])

for key in key_list:
    U_F, qE, qS = compute_U_F(g_opt, w_F, vals, X, key)
    print('U_F shape', U_F.shape, 'qE shape', qE.shape, 'qS shape', qS.shape)
    count = 0
    for i in range(500):
        if(max(np.abs(qS[:,i])**2) < 0.7):
            count += 1
            plt.scatter(np.arange(len(qS[:,i])), np.abs(qS[:,i])**2,s=4)
            plt.plot(np.arange(len(qS[:,i])), np.abs(qS[:,i])**2, label='FS {}'.format(i), linewidth=0.5)
        print('# Mixed states for key={} g_opt={} w_F={}: {}'.format(key, g_opt, w_F, count),count)
    plt.xlabel('H0 eigenstate index')
    plt.ylabel('Projection onto H0 eigenstates')
    plt.title('Projection of Floquet states, wF={}, g={}, key={}'.format(np.around(w_F,2), g_opt, key))
    plt.legend()
    plt.grid()
    plt.show()


if(0): # Tools to understand the Floquet eigenstates and their projections onto the basis of H0
    U_F, qE, qS = compute_U_F(g_opt, w_F, vals, X, key)    
    if(0): #Understand the projection of the floquet eigenstates onto the basis of H0
        for i in [110,214,451]:#range(230,133):
            plt.scatter(np.arange(len(qS[:,i])), np.abs(qS[:,i])**2, label='Floquet state {}'.format(i),s=4)
            plt.plot(np.arange(len(qS[:,i])), np.abs(qS[:,i])**2, label='Floquet state {}'.format(i), linewidth=0.5)
            plt.xlabel('H0 eigenstate index')
            plt.ylabel('Projection onto H0 eigenstates')
            plt.title('Projection of Floquet eigenstates onto H0 eigenstates for wF={}, g={}'.format(w_F, g_opt))
            plt.legend()
            plt.grid()
            plt.show()

    print('qS shape',qS.shape,x_arr.shape)
    N = ngrid  
    x = DVR.grid[:ngrid-1]
    dx = x[1]-x[0]

    #Initial wavefunction localised inside the left well of the double-well potential
    wf_init = np.exp(-(x)**2/1) 
    wf_init /= np.sqrt(np.sum(np.abs(wf_init)**2)*(x[1]-x[0]))

    #Decompose the initial wavefunction in the basis of the Hamiltonian eigenstates
    wf_init_proj = np.zeros_like(vals, dtype=complex) 
    for i in range(len(vals)):
        wf_init_proj[i] = np.sum(np.conj(vecs[:,i])*wf_init*dx)
    print('wf_init_proj', wf_init_proj[:10])
    print('sum', np.sum(np.abs(wf_init_proj)**2))

    beta = 0.1
    rho_init = np.zeros((len(vals), len(vals)), dtype=complex)
    for i in range(len(vals)):
        for j in range(len(vals)):
            #rho_init[i,j] = wf_init_proj[i]*np.conj(wf_init_proj[j])
            if(i==j):
                rho_init[i,j] = np.exp(-beta*vals[i])
    rho_init /= np.trace(rho_init)
    
    #Project the initial wavefunction onto the Floquet eigenstates
    wf_init_proj_qS = np.zeros_like(qE, dtype=complex)
    for i in range(len(qE)):
        wf_init_proj_qS[i] = np.sum(np.conj(qS[:,i])*wf_init_proj)

    rho_init_qS = qS.conj().T @ rho_init @ qS
    print('rho_init_qS', np.trace(rho_init_qS), np.trace(rho_init))
    
    if(1): #Plot the projection of the initial wavefunction onto the Floquet states
        plt.plot(np.arange(len(wf_init_proj_qS)), np.abs(wf_init_proj_qS)**2)
        plt.xlabel('Floquet state index')
        plt.ylabel('Projection onto Floquet states')
        plt.title('Projection of initial wavefunction onto Floquet states for wF={}, g={}'.format(w_F, g_opt))
        plt.grid()
        plt.show()

    #Compute fidelity after n periods of the driving
    narr = np.arange(0,1024)
    fidelity_arr = np.zeros_like(narr, dtype=float)
    for n in narr:
        proj_sum = np.sum(np.abs(wf_init_proj_qS)**2*np.exp(1j*qE*n*T))
        fidelity_arr[n] = np.abs(proj_sum)**2
    plt.plot(narr, fidelity_arr)
    plt.xlabel('Time steps')
    plt.ylabel('Fidelity')
    plt.title('Fidelity of initial wavefunction wF={}, g={}'.format(w_F, g_opt))
    plt.grid()
    plt.show()

    #Compute density matrix fidelity after n periods of the driving
    narr = np.arange(0,1024)
    fidelity_arr_rho = np.zeros_like(narr, dtype=float)

    denom = np.trace(rho_init_qS @ rho_init_qS)
    for n in narr:
        rho_n_qS = np.diag(np.exp(1j*qE*n*T)) @ rho_init_qS @ np.diag(np.exp(-1j*qE*n*T))
        
        fidelity_arr_rho[n] = np.abs(np.trace(np.conjugate(rho_n_qS).T @ rho_init_qS))/denom
        fidelity_arr_rho[n] = 0.5*np.trace(abs(rho_n_qS - rho_init_qS))
        #print('diag', rho_n_qS.diagonal()[:10], rho_init_qS.diagonal()[:10])
        print('n', n, 'fidelity_arr_rho[n]', fidelity_arr_rho[n])
        
        if(n % 100 == 0):
            U_F_n = np.linalg.matrix_power(U_F, n)
            plt.imshow(np.abs(U_F_n), cmap='hot', interpolation='nearest')
            #plt.colorbar()
            plt.title('Density matrix at n={} for wF={}, g={}'.format(n, w_F, g_opt))
            plt.pause(0.1)

    plt.plot(narr, fidelity_arr_rho,color='r',ls='--',label='Density matrix fidelity')
    plt.xlabel('Time steps')
    plt.ylabel('Density matrix fidelity')
    plt.title('Density matrix fidelity of initial wavefunction wF={}, g={}'.format(w_F, g_opt))
    plt.grid()
    plt.show()

if(0):
    #for g_opt in [2.495, 2.497, 2.499, 2.5]:
    for g_opt in [1.5,2.0, 2.25, 2.5, 3.0, 4.0]:
        qE = np.load('qE_quartic_g_{}.npy'.format(g_opt))
        print('g_opt', g_opt, 'qE', qE[:10])
        plt.scatter(np.arange(len(qE)), qE, label='g={}'.format(g_opt),s=5)


    plt.xlabel('Floquet state index')
    plt.ylabel('Quasienergy')
    plt.title('Quasienergies for different g values')
    plt.legend()
    #plt.grid()
    plt.show()

