import numpy as np
from scipy.linalg import expm
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from floquet_tools import Floquetiser, fidelity, propagate_wf_split_op, compute_U_F
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

"""
The Quasienergies agree when the drive strength is 0. It needs to be understood as to
what the right comparison between the quasienergies from the U_F and the H_F is. 

"""


m = 1.0
L = 20.0
omega = 1.0
ngrid = 200 

lb = -L
ub = L

def pot(x):
    return 0.5*m*omega**2*x**2

DVR = DVR1D(ngrid,lb,ub,m,pot)
neigs = ngrid-1#100
vals,vecs = DVR.Diagonalize(neig_total=neigs)

#overwrite vals with analytical values
vals = np.array([0.5 + n for n in range(neigs)])

x_arr = DVR.grid[:ngrid]
dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

pos_mat_analytical = np.zeros((neigs,neigs))
for n in range(neigs):
    for m in range(neigs):
        if (n + 1 == m):
            pos_mat_analytical[n, m] = np.sqrt((n + 1) / 2)
        elif (n == m + 1):
            pos_mat_analytical[n, m] = np.sqrt(n / 2)

#print('Position matrix X shape:', pos_mat[:5,:5],pos_mat_analytical[:5,:5])
X = pos_mat_analytical
print('Position matrix X shape:', X.shape)
#----------------------------------------------------------------------------------------------

w_F = 0.93  
S_opt = 0.1

def V(x,t):
    return pot(x) - x*S_opt*np.sin(w_F*t)

def Vfree(x,t):
    return 0.0

#----------------------------------------------------------------------------------------------
N = ngrid-1
T = 2*np.pi/w_F
n_steps = 1000
dt = T/n_steps
tarr = np.arange(0, 2*np.pi/w_F + dt, dt)

x = np.linspace(-L, L, N, endpoint=False)
dx = x[1]-x[0]
N = len(x)
k = 2 * np.pi * np.fft.fftfreq(N, d=(x[1]-x[0])) 
U_F = np.zeros((N, N), dtype=complex)

print('vals', vals[0:100])

def H_eigen(t):
    return np.diag(vals) - X*S_opt*np.sin(w_F*t)

def f(t,c):
    H = H_eigen(t)
    return -1j*H @ c

"""
We construct the Floquet Unitary by direct matrix exponentiation of the time-dependent Hamiltonian over one period T.
The time evolution operator is:
    U(T) = T exp(-i ∫_0^T H(t) dt)
where T is the time-ordering operator. We can approximate this by discretizing the time interval into small steps 
and using the matrix exponential of the Hamiltonian at each time step.
"""
U_F = np.eye(N, dtype=complex)

for i in range(n_steps):
    t = i * dt
    H = H_eigen(t)


    if(i % 10 == 0):
        #print('Step {}/{}: t={:.3f}, U_F shape: {}'.format(i, n_steps, t, U_F.shape))
        #print('U_F', np.around(U_F[0:5,0:5], decimals=2))

        if(0):
            eigs = np.linalg.eigvals(U_F)[:10]
            eigs_H = np.exp(-1j*vals*(t))[:10]
            print('Step {}/{}: t={:.3f}'.format(i, n_steps, t))
            print('eigs from U_F:', np.around(eigs, decimals=3))
            print('eigs from H0:', np.around(eigs_H, decimals=3))
            print('Difference:', np.around(eigs - eigs_H, decimals=3))
            print('\n')
        if(0):      
            eigs = np.linalg.eigvals(U_F)
            #plt.clf()
            eig0 = eigs[0]
            plt.plot(t, np.real(eig0), 'bo', label='Re(eig0)')
            plt.plot(t, np.imag(eig0), 'ro', label='Im(eig0)')
            plt.title('Floquet Unitary Eigenvalue Evolution')
            plt.xlabel('Time t')
            plt.ylabel('Eigenvalue')
            plt.xlim(0, T)
            plt.ylim(-1.1, 1.1)
            plt.pause(0.1)
    
    U_F = expm(-1j * H * dt) @ U_F

#plt.show()

quasienergies = np.angle(np.linalg.eigvals(U_F)) / T
#quasienergies = np.sort(quasienergies)
print('Quasienergies from U_F:', np.around(quasienergies[0:100],4))

n_ph = 5
V = S_opt*X/2j
print('Driving strength S:', S_opt)
floquet_vals, floquet_vecs = Floquetiser(vals, w_F, V, n_ph)
floquet_vals = np.real(floquet_vals)
floquet_folded = floquet_vals
floquet_folded =   (floquet_vals + w_F/2) % w_F - w_F/2
print('Floquet eigenvalues:', floquet_folded[0:100])



#Plot the quasienergies and Floquet eigenvalues for comparison - represent them as energy levels (dash)
plt.plot(np.zeros_like(quasienergies), quasienergies, 'o', color='blue', alpha=0.5, label='Quasienergies from U_F')
plt.plot(0.0*np.ones_like(floquet_folded), floquet_folded, 'o', color='red', alpha=0.5, label='Floquet eigenvalues from Floquetiser', markersize=2)
plt.xlim(-0.4,0.8)
plt.legend()
plt.title('Comparison of Quasienergies and Floquet Eigenvalues')
plt.xlabel('Energy')
plt.ylabel('Density')
plt.grid()
plt.show()

exit()










if(0):
    """
    We construct the Floquet Unitary by propagating the eigenstates of H0 over one period T.
    The time evolution equation is:
        H(t) C(t) = i dC/dt
    where C(t) is the vector of coefficients in the eigenbasis of H0, starting 
    from the initial condition C(0) = [0,0,...,1,...,0] (i.e., the i-th eigenstate of H0).
    """
    for i in range(N):
        wf_init = np.zeros(N, dtype=complex) #Expansion in eigenbasis of H0
        wf_init[i] = 1.0

        solution = solve_ivp(f, (0,T), wf_init, method='RK45')
        #plt.plot(tarr, solution.y[0,:], label='i={}'.format(i))
        #plt.plot(tarr, solution.y[-1,:], label='i={}'.format(i))
        #plt.show()
        U_F[:,i] = solution.y[:,-1]

print('U_F shape:', U_F[0:5,0:5])
print('U_F is unitary?', np.allclose(U_F.conj().T @ U_F, np.eye(N), atol=1e-10))

Floquet_eigenvalues, Floquet_eigenvectors = np.linalg.eig(U_F)
Floquet_eigenvalues = np.real(Floquet_eigenvalues)
print('Floquet eigenvalues before folding:', Floquet_eigenvalues[0:10])



exit()
#----------------------------------------------------------------------------------------------

H1 = S_opt*X/2j
print('Driving strength S:', S_opt)
floquet_vals, floquet_vecs = Floquetiser(vals, w_F, H1, n_ph)

floquet_vals = np.real(floquet_vals)
print('Floquet eigenvalues before folding:', floquet_vals[0:10])

floquet_folded = (floquet_vals + w_F/2) % w_F - w_F/2
floquet_folded = np.sort(floquet_folded)
print('Floquet eigenvalues after folding:', floquet_folded[0:10])

N = ngrid  
x = np.linspace(-L, L, N, endpoint=False)
dx = x[1]-x[0]
U_F, qE, qS = compute_U_F(x, V, w_F)

print('qs orthogonal?', np.around(qS.conj().T @ qS*dx, decimals=2))

print('Floquet eigenvalues from U_F:', qE[0:10])

#---------------------------------------------------------------------------------------------

