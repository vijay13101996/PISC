import numpy as np
import matplotlib.pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from floquet_tools import Floquetiser, fidelity, propagate_wf_split_op, compute_U_F, compute_U_F_direct
import math

m = 1.0
L = 20.0
omega = 1.0
ngrid = 2**8 #101

lb = -L
ub = L

def pot(x):
    return 0.5*m*omega**2*x**2

DVR = DVR1D(ngrid,lb,ub,m,pot)
neigs = ngrid-1#100
vals,vecs = DVR.Diagonalize(neig_total=neigs)
print('vals',vals.shape, 'vecs', vecs.shape)

if(0):
    """
    Compare the projection of the initial wavefunction onto the eigenstates 
    with the analytical result for a displaced Gaussian.
    """
    N = ngrid  
    x = DVR.grid[:ngrid-1]
    dx = x[1]-x[0]
    x_0 = 4
    wf_init = (1/(np.pi)**0.25)*np.exp(-(x-x_0)**2/2)
    plt.plot(x, np.abs(wf_init)**2, label='Initial wavefunction')
    plt.plot(x, np.abs(vecs[:,0])**2, label='Ground state wavefunction')
    plt.legend()
    plt.show()

    alpha = x_0/np.sqrt(2)
    wf_init_proj_anal = np.zeros(neigs, dtype=complex)
    sqrtfactn = 1.0
    wf_init_proj_anal[0] = np.exp(-alpha**2/2)
    for n in range(1,neigs):
        sqrtfactn*= np.sqrt(n)
        if(n<10):
            print('fact',n, sqrtfactn**2)
        wf_init_proj_anal[n] = np.exp(-alpha**2/2)*alpha**n/sqrtfactn

    wf_init_proj = np.zeros(neigs, dtype=complex)
    for n in range(neigs):
        wf_init_proj[n] = np.sum(vecs[:,n]*wf_init)*dx

    print('wf_init_proj_anal',wf_init_proj_anal[:10], 'norm', np.sum(np.abs(wf_init_proj_anal)**2))
    print('wf_init_proj',wf_init_proj[:10], 'norm', np.sum(np.abs(wf_init_proj)**2))
    exit()


x_arr = DVR.grid[:ngrid]
dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

#X = pos_mat

#Create fictional operator with matrix elements that are random, exponentially decaying and gaussian decaying
X = np.zeros((neigs,neigs))
key = 'gaussian_decay'
key = 'exponential_decay'
#key = 'no_decay'
for i in range(neigs):
    for j in range(i, neigs):
        if(key=='gaussian_decay'):
            X[i,j] = np.exp(-np.abs(i-j)**2/2) 
        elif(key=='exponential_decay'):
            X[i,j] = np.exp(-np.abs(i-j)*0.05)
        elif(key=='no_decay'):
            X[i,j] = np.random.normal(0,1)
        X[j,i] = X[i,j]

w_F = 0.5 # 
T = 2*np.pi/w_F 
n_ph = 5
S = 1.0
S_opt = 0.1

def V(x,t):
    return pot(x) - x*S_opt*np.sin(w_F*t)

if(1): #Comparing expm and split operator method for computing U_F
    U_F_expm, qE_expm, qS_expm = compute_U_F_direct(vals, X, S_opt, w_F, n_steps=500, method='expm')
    U_F_split, qE_split, qS_split = compute_U_F_direct(vals, X, S_opt, w_F, n_steps=500, method='split')
   
    print('U_F_expm', np.around(U_F_expm,3)[0:5,0:5])
    print('U_F_split', np.around(U_F_split,3)[0:5,0:5])
    assert np.allclose(U_F_expm, U_F_split, atol=1e-2), "U_F matrices from expm and split operator do not match!"
    assert np.allclose(qE_expm, qE_split, atol=1e-3), "Quasienergies from expm and split operator do not match!"
    assert np.allclose(qS_expm, qS_split, atol=1e-2), "Floquet eigenstates from expm and split operator do not match!"

    print('All methods produce consistent results for U_F, quasienergies, and Floquet eigenstates.')

if(1):
    """
    Compute the Floquet unitary matrix U_F, quasienergies qE and Floquet eigenstates qS.
    """
    try:
        raise Exception("Force recomputation of U_F, qE, qS")
        U_F = np.load('U_F_harmonic_S_{}_wF_{}_{}.npy'.format(S_opt,w_F, key))
        #print('Loaded U_F from file')
        qE = np.load('qE_harmonic_S_{}_wF_{}_{}.npy'.format(S_opt,w_F, key))
        qS = np.load('qS_harmonic_S_{}_wF_{}_{}.npy'.format(S_opt,w_F, key))
    except:
        #Compute the unitary matrix U_F in the eigenbasis of H0 and its eigenvalues and eigenvectors
        U_F, qE, qS = compute_U_F_direct(vals, X, S_opt, w_F, n_steps=1000)
        np.save('U_F_harmonic_S_{}_wF_{}_{}.npy'.format(S_opt,w_F,key), U_F)
        np.save('qE_harmonic_S_{}_wF_{}_{}.npy'.format(S_opt,w_F,key), qE)
        np.save('qS_harmonic_S_{}_wF_{}_{}.npy'.format(S_opt,w_F,key), qS)
        
        print('U_F', np.around(U_F,2)[0:5,0:5])
        print('qE', np.around(qE,2)[0:10])
        print('qS', np.around(qS,2)[:,0:5])

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

    #Project the initial wavefunction onto the Floquet eigenstates
    wf_init_proj_qS = np.zeros_like(qE, dtype=complex)
    for i in range(len(qE)):
        wf_init_proj_qS[i] = np.sum(np.conj(qS[:,i])*wf_init_proj)

    if(0): #Plot the projection of the initial wavefunction onto the Floquet states
        plt.plot(np.arange(len(wf_init_proj_qS)), np.abs(wf_init_proj_qS)**2)
        plt.xlabel('Floquet state index')
        plt.ylabel('Projection onto Floquet states')
        plt.title('Projection of initial wavefunction onto Floquet states for wF={}, S={}'.format(w_F, S_opt))
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
    plt.title('Fidelity of initial wavefunction wF={}, S={}'.format(w_F, S_opt))
    plt.grid()
    plt.show()




if(0):
    """
    Floquet quasienergies and eigenstates computed using the Floquet Hamiltonian H_F.
    """
    S_arr = np.linspace(0.0, 5e-3, 20)
    for S in [S_opt]:# S_arr:
        V = S*X/2j
        print('Driving strength S:', S)
        floquet_vals, floquet_vecs = Floquetiser(vals, w_F, V, n_ph)
        plt.plot(S*np.ones_like(floquet_vals), np.real(floquet_vals), 'o', color='blue', markersize=2)

    floquet_vals = np.real(floquet_vals)
    floquet_folded = (floquet_vals + w_F/2) % w_F - w_F/2
    floquet_folded = np.sort(floquet_folded)
    print('Floquet eigenvalues:', floquet_folded[0:10])
    E0 = vals[0]
    E1 = vals[1]

    for n in range(-n_ph,n_ph+1):
        plt.axhline(E0+n*w_F, color='red', linestyle='--', alpha=0.5)
        plt.axhline(E1+n*w_F, color='green', linestyle='--', alpha=0.5)

    #plt.axhline(E0, color='red', linestyle='--', label='E0')
    #plt.axhline(E1, color='green', linestyle='--', label='E1')

    plt.legend()
    plt.ylim(E0-0.001, E1+0.001)
    plt.xlabel('Driving strength S')
    plt.ylabel('Floquet eigenvalues')
    plt.title('Floquet eigenvalues vs Driving strength S for double-well potential')
    plt.grid()
    plt.show()


