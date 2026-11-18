import numpy as np
import matplotlib.pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from floquet_tools import Floquetiser, fidelity, compute_U_F, compute_U_F_direct
import matplotlib


plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False


"""
This module will be used to construct the Floquet Hamiltonian and Unitary for the double-well potential 
of the form:
    V(x) = -x^2/4 + x^4/(64D) + xS sin(wt)

Thus, 
    H0 = p^2/(2m) - x^2/4 + x^4/(64D)
    H_1 = xS/2i e^{iwt}
    H_{-1} = -xS/2i e^{-iwt}

Benchmarked with figure 2a of:
Grossmann, Frank, Thomas Dittrich, Peter Jung, and Peter Hänggi. 
"Coherent destruction of tunneling." Physical Review Letters 67, no. 4 (1991): 516.

"""

m = 1.0
L = 20.0
omega = 1.0
a = 0.0
b = 0.0
ngrid = 2**9 #101

lb = -L
ub = L

D = 2.0

def pot(x):
    return -x**2/4 + x**4/(64*D)

DVR = DVR1D(ngrid,lb,ub,m,pot)
neigs = ngrid-1#100
vals,vecs = DVR.Diagonalize(neig_total=neigs)
#print('vals',vals)

#Tunneling splitting
E_d = vals[1]-vals[0]
print('Tunneling splitting E_d',E_d)

x_arr = DVR.grid[:ngrid]
dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

X = pos_mat

print('x_arr',x_arr.shape, 'X',X.shape)

w_F = 0.01 # 
T = 2*np.pi/w_F
n_ph = 3
S = 1.0
#S_opt = 3.19e-3# 0.0005714
S_opt = 2e-3#3.171e-3

def V(x,t):
    return pot(x) + x*S_opt*np.sin(w_F*t)

if(1):
    """
    TWO ISSUES HERE:
    1. There is an incompatibility between size of x_arr and X which causes the wavefunction norm to exceed 1.
    2. The initial wavefunction is to be projected on to the floquet eigenstates to compute fidelity. Fix that too.

    UPDATE: 
    1. Issues resolved. The initial wavefunction should be defined with DVR.grid[:ngrid-1] for getting the correct norm.
    2. The projection onto Floquet eigenstates is trivial
    
    NOTE: When propagating the wavefunction using split-operator method, the initial wavefunction should be defined on
    x = np.linspace(-L, L, N, endpoint=False) where N = ngrid-1. 
    This is because the DVR grid has ngrid points, but the wavefunction is defined on ngrid-1 points due to the boundary conditions.

    In general, the wavefunction from these two grids are slightly shifted from one another, so one needs to be careful when
    switching between these two approaches.

    """
    try:
        #raise Exception("Force recomputation of U_F, qE, qS")
        U_F = np.load('U_F_double_well_wF_{}_S_{}.npy'.format(w_F, S_opt))
        qE = np.load('qE_double_well_wF_{}_S_{}.npy'.format(w_F, S_opt))
        qS = np.load('qS_double_well_wF_{}_S_{}.npy'.format(w_F, S_opt))
    except:
        #Compute the unitary matrix U_F in the eigenbasis of H0 and its eigenvalues and eigenvectors
        U_F, qE, qS = compute_U_F_direct(vals, X, S_opt, w_F, n_steps=1000)
        np.save('U_F_double_well_wF_{}_S_{}.npy'.format(w_F, S_opt), U_F)
        np.save('qE_double_well_wF_{}_S_{}.npy'.format(w_F, S_opt), qE)
        np.save('qS_double_well_wF_{}_S_{}.npy'.format(w_F, S_opt), qS)

    print('qS shape',qS.shape,x_arr.shape)
    N = ngrid  
    x = DVR.grid[:ngrid-1]
    dx = x[1]-x[0]

    #Initial wavefunction localised inside the left well of the double-well potential
    wf_init = np.exp(-(x+4)**2/2) 
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

    if(1): #Plot projection of Floquet states onto the basis of H0
        plt.plot(np.arange(len(qE)), np.abs(qS[:,176])**2, label='Floquet state 0')
        plt.plot(np.arange(len(qE)), np.abs(qS[:,175])**2, label='Floquet state 1')
        plt.scatter(np.arange(len(qE)), np.abs(qS[:,0])**2, color='red', s=10)
        plt.scatter(np.arange(len(qE)), np.abs(qS[:,1])**2, color='green', s=10)
        plt.xticks(np.arange(0, len(qE), 1))
        #plt.xlim(-1,200)
        plt.xlabel('Eigenstate index')
        plt.ylabel('Projection')
        plt.title('Projection of Floquet states onto $H_0$ eigenstates')
        plt.legend()
        plt.savefig('Floquet_projection_double_well_FLOQUET.png', dpi=300, bbox_inches='tight')
        plt.show()
        exit()

    if(0): #Plot projection of Floquet states onto the basis of H0
        plt.plot(np.arange(len(qE)), np.abs(wf_init_proj)**2, label='Projection onto H0 eigenstates')
        plt.scatter(np.arange(len(qE)), np.abs(wf_init_proj)**2, color='red', s=10)
        plt.xticks(np.arange(0, len(qE), 1))
        plt.xlim(-1,20)
        plt.xlabel('Eigenstate index')
        plt.ylabel('Projection')
        plt.title('Projection of initial wavefunction onto $H_0$ eigenstates')
        plt.savefig('H0_projection_double_well_FLOQUET.png', dpi=300, bbox_inches='tight')
        plt.show()

    if(1):
        plt.plot(np.arange(len(wf_init_proj_qS)), np.abs(wf_init_proj_qS)**2)
        plt.xlabel('Floquet state index')
        plt.ylabel('Projection onto Floquet states')
        plt.title('Projection of initial wavefunction onto Floquet states for wF={}, S={}'.format(w_F, S_opt))
        plt.grid()
        plt.show()
    
    exit()
    #Compute fidelity after n periods of the driving
    narr = np.arange(0,1024)
    fidelity_arr = np.zeros_like(narr, dtype=float)
    for n in narr:
        proj_sum = np.sum(np.abs(wf_init_proj_qS)**2*np.exp(1j*qE*n*T))
        fidelity_arr[n] = np.abs(proj_sum)**2
    plt.plot(narr, fidelity_arr)
    plt.xlabel(r'Time steps $n$')
    plt.ylabel(r'Fidelity $|\langle \psi(nT)|\psi(0) \rangle|^2$')
    plt.title(r'Fidelity of initial wavefunction localised in the left well')
    plt.grid()
    
    plt.savefig('Fidelity_double_well_FLOQUET.png', dpi=300, bbox_inches='tight')
    plt.show()

exit()

#--------------------------------------------------------------------------------------------------------
if(0): #Compute unitary matrix
    """
    START TOMORROW BY UNDERSTANDING HOW TO BENCHMARK THE QUASIENERGIES OBTAINED 
    FROM THE UNITARY MATRIX U_F. 
    """

    N = ngrid  
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1]-x[0]
    U_F, qE, qS = compute_U_F(x, V, w_F)
    
    #Verify orthogonality of the eigenvectors
    print('qS shape',qS.shape)
    print('qS orthogonality check', np.sum(np.conj(qS[:,3])*qS[:,3]*dx))

    wf_init = np.exp(-(x+4)**2/2) 
    wf_init /= np.sqrt(np.sum(np.abs(wf_init)**2)*(x[1]-x[0])) 


    #Project wf_init onto the U_F eigenbasis
    wf_init_proj = np.zeros_like(qE, dtype=complex)
    for i in range(len(qE)):
        wf_init_proj[i] = np.sum(np.conj(qS[:,i])*wf_init*dx)
    print('sum', np.sum(np.abs(wf_init_proj)**2))
    print('qE', qE.shape, 'wf_init_proj', wf_init_proj.shape)
    #Compute the fidelity of the initial wavefunction with respect to the U_F eigenstates
    narr = np.arange(0,1024)
    fidelity_arr = np.zeros_like(narr, dtype=float)
    for n in narr:
        proj_sum = np.sum(np.abs(wf_init_proj)**2*np.exp(1j*qE*n*T))
        fidelity_arr[n] = np.abs(proj_sum)**2 
    plt.plot(narr, fidelity_arr)
    plt.xlabel('Time steps')
    plt.ylabel('Fidelity')
    plt.title('Fidelity of initial wavefunction with respect to U_F eigenstates')
    plt.grid()
    plt.show()
    exit()


if(0):
    N = ngrid   
    x = np.linspace(-L, L, N, endpoint=False)
    #Interpolate the ground state wavefunction to the new grid
    #wf_init = np.interp(x, x_arr[:ngrid-1], vecs[:,0])
    #wf_init = np.exp(-x**2/2)
    wf_init = np.exp(-(x+4)**2/2)
    
    wf_init /= np.sqrt(np.sum(np.abs(wf_init)**2)*(x[1]-x[0]))
    if(0):
        plt.plot(x, np.abs(wf_init)**2)
        plt.plot(x_arr[:ngrid-1], np.abs(vecs[:,0])**2, 'o', markersize=2)
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.title('Initial wavefunction probability density')
        plt.grid()
        plt.show()
    print('Initial wavefunction norm:', np.sum(np.abs(wf_init)**2)*(x[1]-x[0]), np.sum(np.abs(vecs[:,0])**2)*dx)

    T = 2*np.pi/w_F
    dt = T/1000
    print('T/dt',int(T/dt))
    tarr = np.arange(0, 500*T, dt)
    fidelity_arr = fidelity(x,wf_init, m, V, tarr)
    
    #Plot only at T, 2T, 3T, 4T, 5T...
    tarr_trunc = tarr[::int(T/dt)]
    narr = np.arange(len(tarr_trunc))
    plt.plot(narr, fidelity_arr[::int(T/dt)])
    
    plt.xlabel('Time')
    plt.ylabel('Fidelity')
    plt.title('Fidelity vs Time for S_opt = {}'.format(S_opt))
    plt.grid()
    plt.show()
    exit()

S_arr = np.linspace(0.0, 5e-3, 50)

for S in S_arr:
    V = S*X/2j
    print('Driving strength S:', S)
    try:
        floquet_vals = np.load('floquet_vals_S_{}.npy'.format(np.around(S,6)))
        #floquet_vecs = np.load('floquet_vecs_S_{}.npy'.format(np.around(S,6)))
    except:
        floquet_vals, floquet_vecs = Floquetiser(vals, w_F, V, n_ph)
        np.save('floquet_vals_S_{}.npy'.format(np.around(S,6)), floquet_vals)
    print('Floquet eigenvalues:', floquet_vals[0:10])
    floquet_vals = np.real(floquet_vals)
    floquet_folded = (floquet_vals + w_F/2) % w_F - w_F/2
    floquet_folded = np.sort(floquet_folded)
    #print('Floquet eigenvalues folded:', floquet_folded[0:100])
    plt.plot(S*np.ones_like(floquet_vals), np.real(floquet_vals), 'o', color='blue', markersize=2)

E0 = vals[0]
E1 = vals[1]

for n in range(1):#-n_ph,n_ph+1):
    plt.axhline(E0+n*w_F, color='red', linestyle='--', alpha=0.5, label=r'$E_0$')
    plt.axhline(E1+n*w_F, color='green', linestyle='--', alpha=0.5, label=r'$E_1$')

#plt.axhline(E0, color='red', linestyle='--', label='E0')
#plt.axhline(E1, color='green', linestyle='--', label='E1')
plt.axvline(S_opt, color='black', linestyle='--', label=r'$S_c$')

plt.legend()
plt.ylim(E0-0.001, E1+0.001)
plt.xlabel('Driving strength S')
plt.ylabel('Floquet eigenvalues')
plt.title('Quasienergies vs Driving strength S for double-well potential')
plt.grid()
plt.savefig('Floquet_eigenvalues_vs_S_double_well_FLOQUET.png', dpi=300, bbox_inches='tight')
plt.show()


