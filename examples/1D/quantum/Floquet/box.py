import numpy as np
from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, compute_U_F, compute_U_F_direct
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from multiprocessing import Pool

m = 1.0
L = 1.0
ngrid = 2**9 
neigs = ngrid-1

lb = 0
ub = L

def pot(x):
    if(x>L or x<0):
        return 1e10
    else:
        return 0.0

potkey = 'L_{}_1D_box'.format(L)
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

print('vals',np.around(vals[:5],3))
#print('O_anal',O_anal[:5,:5])
pos_mat = O_anal

X = np.zeros((neigs,neigs))
key = 'gaussian_decay'
key = 'exponential_decay'
key = 'no_decay'
#key = 'pos_mat'

def operator_matrix(key):#Create fictional operator with matrix elements that are random, exponentially decaying and gaussian decaying
    if key == 'pos_mat':
        X = pos_mat
        return X
    else:
        X = np.zeros((neigs,neigs))
        for i in range(neigs):
            for j in range(i, neigs):
                if(key=='gaussian_decay'):
                    X[i,j] = np.exp(-np.abs(i-j)**2/2) 
                elif(key=='exponential_decay'):
                    X[i,j] = np.exp(-np.abs(i-j)*5)
                elif(key=='no_decay'):
                    X[i,j] = 1.0#  np.random.normal(0,1)
                X[j,i] = X[i,j]
    return X
w_F = vals[1] + 2.0 

g_opt= 3.0
T = 2*np.pi/w_F

print('w_F', w_F, 'T', T, 'g_opt', g_opt)

def compute_U_F(g_opt, w_F, vals, X, key):
    try:
        #raise Exception("Force recomputation of U_F, qE, qS")
        U_F = np.load('U_F_1D_box_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key))
        qE = np.load('qE_1D_box_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key))
        qS = np.load('qS_1D_box_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key))
    except:
        #Compute the unitary matrix U_F in the eigenbasis of H0 and its eigenvalues and eigenvectors
        print('Computing U_F, qE, qS for g_opt={}, w_F={}, key={}'.format(g_opt,w_F,key))
        X = operator_matrix(key)
        U_F, qE, qS = compute_U_F_direct(vals, X, g_opt, w_F, n_steps=2000)#,method='expm')
    
        np.save('U_F_1D_box_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key), U_F)
        np.save('qE_1D_box_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key), qE)
        np.save('qS_1D_box_g_{}_wF_{}_{}.npy'.format(g_opt,w_F,key), qS)
    return U_F, qE, qS

#Do it in parallel for different values of g_opt and w_F
g_opt_list = np.arange(0.0, 5.01, 0.5)
#with Pool(processes=10) as pool:
#    pool.starmap(compute_U_F, [(g_opt, w_F, vals, X, key) for g_opt in g_opt_list])

#Do it in parallel for different values of key
key_list = ['gaussian_decay', 'exponential_decay', 'no_decay', 'pos_mat']
with Pool(processes=4) as pool:
    pool.starmap(compute_U_F, [(g_opt, w_F, vals, X, key) for key in key_list])
#exit()

for key in key_list:
    U_F, qE, qS = compute_U_F(g_opt, w_F, vals, X, key)
    #print('U_F shape', U_F.shape, 'qE shape', qE.shape, 'qS shape', qS.shape)
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

exit()

if(0):
    for g_opt in [3.0]: # g_opt_list:
        U_F, qE, qS = compute_U_F(g_opt, w_F, vals, X, key)
        print('g_opt', g_opt, 'qE', qE[:10])
       
        if(0): #Find Floquet state with maximum overlap with the ground state of H0
            ground_state_index = 60
            overlap = np.abs(qS[ground_state_index,:])**2
            max_overlap_index = np.argmax(overlap)
            plt.plot(np.arange(len(vals)), abs(qS[:,max_overlap_index])**2, label='Floquet state {}'.format(max_overlap_index), linewidth=2)
            print('g_opt', g_opt, 'ground state index', ground_state_index, 'max overlap index', max_overlap_index, 'max overlap', np.max(overlap))

        if(1): #Plot the projection of the Floquet eigenstates onto the basis of H0
            for n in range(40) :#len(qE)):
                plt.plot(np.arange(len(qS[:,n])), np.abs(qS[:,n])**2, label='Floquet state {}'.format(n), linewidth=0.5)    
            
    #plt.xlim([0, 200])
    plt.xlabel('H0 eigenstate index')
    plt.ylabel('Projection onto H0 eigenstates')
    plt.legend()
    #plt.grid()

    plt.show()

if(1): # A lot of tools - to be classified
    U_F,qE, qS = compute_U_F(g_opt, w_F, vals, X, key)
    if(0): #Understand the projection of the floquet eigenstates onto the basis of H0
        for i in range(40): #range(500): #[110,214,451]:#range(230,133):
            #plt.scatter(np.arange(len(qS[:,i])), np.abs(qS[:,i])**2, label='Floquet state {}'.format(i),s=4)
            plt.plot(np.arange(len(qS[:,i])), np.abs(qS[:,i])**2, label='Floquet state {}'.format(i), linewidth=1)
        plt.xlabel('H0 eigenstate index')
        plt.ylabel('Projection onto H0 eigenstates')
        plt.title('Projection of Floquet eigenstates onto H0 eigenstates for wF={}, g={}'.format(w_F, g_opt))
        plt.legend()
        plt.grid()
        plt.show()

    if(1): #Find the Floquet state with maximum overlap with the system eigenstates
        ind_list = []
        for n in range(len(vals)):
            ind = np.argmax(np.abs(qS[n,:])**2)
            ind_list.append(ind)
            print('n', n, 'ind', ind, 'max overlap', np.max(np.abs(qS[n,:])**2))


    print('qS shape',qS.shape,x_arr.shape)
    N = ngrid  
    x = DVR.grid[:ngrid-1]
    dx = x[1]-x[0]

    #Initial wavefunction localised inside the left well of the double-well potential
    wf_init = np.exp(-(x-L/2)**2/1) 
    wf_init /= np.sqrt(np.sum(np.abs(wf_init)**2)*(x[1]-x[0]))

    #Decompose the initial wavefunction in the basis of the Hamiltonian eigenstates
    wf_init_proj = np.zeros_like(vals, dtype=complex) 
    for i in range(len(vals)):
        wf_init_proj[i] = np.sum(np.conj(vecs[:,i])*wf_init*dx)
    print('wf_init_proj', wf_init_proj[:10])
    print('sum', np.sum(np.abs(wf_init_proj)**2))

    beta = 0.01
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

    if(0): #Plot the diagonal elements of the initial density matrix in the H0 and Floquet basis
        plt.plot(np.arange(len(vals)), np.diag(rho_init), label='Initial density matrix diagonal in H0 basis')
        plt.plot(np.arange(len(qE)), np.diag(rho_init_qS), label='Initial density matrix diagonal in Floquet basis')
        for n in range(20):
            plt.axvline(ind_list[n], color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('State index')
        plt.ylabel('Density matrix diagonal elements')
        plt.title('Initial density matrix diagonal elements for wF={}, g={}'.format(w_F, g_opt))
        plt.legend()
        plt.grid()
        plt.show()

    if(0): #Plot the projection of the initial wavefunction onto the Floquet states
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
    fidelity_arr_rho = np.zeros_like(narr, dtype=complex)

    denom = np.trace(rho_init_qS @ rho_init_qS)
    for n in narr:
        rho_n_qS = np.diag(np.exp(1j*qE*n*T)) @ rho_init_qS @ np.diag(np.exp(-1j*qE*n*T))
        
        fidelity_arr_rho[n] = (np.trace(np.conjugate(rho_n_qS).T @ rho_init_qS))/denom
        print('n', n, 'fidelity_arr_rho[n]', fidelity_arr_rho[n])

        #fidelity_arr_rho[n] = 0.5*np.trace(abs(rho_n_qS - rho_init_qS))
        #print('diag', rho_n_qS.diagonal()[:10], rho_init_qS.diagonal()[:10])
        
        #U_n = np.linalg.matrix_power(U_F, n)
        #rho_n_qS_direct = U_n @ rho_init @ np.conjugate(U_n).T

        #fid_direct = (np.trace(np.conjugate(rho_n_qS_direct).T @ rho_init))/denom
        #print('n', n, 'fidelity_arr_rho[n]', fidelity_arr_rho[n], 'fid_direct', fid_direct)
        #print('rho', np.around(np.diagonal(rho_n_qS),3))


    plt.plot(narr, fidelity_arr_rho,color='r',ls='--',label='Density matrix fidelity')
    plt.xlabel('Time steps')
    plt.ylabel('Density matrix fidelity')
    plt.title('Density matrix fidelity of initial wavefunction wF={}, g={}'.format(w_F, g_opt))
    #plt.grid()
    plt.show()

