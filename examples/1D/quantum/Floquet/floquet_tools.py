import numpy as np
import scipy.linalg
import scipy
from matplotlib import pyplot as plt
#from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.linalg import expm
import time

def compute_U_F_direct(vals, X, S_opt, w_F, n_steps=1000,method='split'):
    """
    Compute the Floquet operator U_F in the eigenbasis of H0
    for a given time-dependent potential Vt and driving frequency w_F.
    
    Parameters:
    vals : ndarray
        Eigenvalues of the unperturbed Hamiltonian H0.
    X : ndarray
        Position operator matrix in the eigenbasis of H0.
    S_opt : float
        Driving strength.
    w_F : float
        Driving frequency.
    n_steps : int
        Number of time steps for propagation.

    Returns:
    U_F : ndarray
        Floquet operator matrix.
    """
    
    T = 2*np.pi/w_F
    dt = T/n_steps
    N = len(vals)

    if(method=='expm'): #Slower method using matrix exponentiation
        def H_eigen(t):
            return np.diag(vals) - X*S_opt*np.sin(w_F*t)

        U_F = np.eye(N, dtype=complex)
        for i in range(n_steps+1):
            t = i * dt
            H = H_eigen(t)
            U_F = expm(-1j * H * dt) @ U_F
            if(i%100==0):
                print(f"Step {i}/{n_steps}, t={t:.4f}")

    if(method=='split'): #Faster method using split operator in the eigenbasis of H0
        expD = np.diag(np.exp(-1j*vals*dt/2)) # Half-step evolution under H0
        expD_diag = np.exp(-1j*vals*dt/2)
        
        D, W = np.linalg.eigh(X)   #Diagonalize X to get its eigenvalues and eigenvectors
        Wd = W.conj().T
        U_F = np.eye(N, dtype=complex)
        
        tarr = np.arange(0, T+0.1*dt, dt)

        for i in range(n_steps+1):
            t = i * dt
            # Half-step evolution under H0
            U_F = expD @ U_F

            # Full-step evolution under the driving term -S_opt*X*sin(w_F*t)
            U_F = W @ np.diag(np.exp(1j*S_opt*D*np.sin(w_F*t)*dt)) @ Wd @ U_F

            # Half-step evolution under H0
            U_F = expD @ U_F

            if(i%100==0):
                print(f"Step {i}/{n_steps}, t={t:.4f}")

    if(method=='split_X'): #Split operator in the eigenbasis of X
        """
        This method uses the split operator technique in the eigenbasis of the position operator X.
        NOTE: This code is AI written and needs to be tested thoroughly. It agrees with the previous method,
        but that is not sufficient to conclude that it is bug-free.
        """

        D, W = np.linalg.eigh(X)   #Diagonalize X to get its eigenvalues and eigenvectors
        Wd = W.conj().T

        H0 = np.diag(vals)
        H0_in_X = Wd @ H0 @ W  #Transform H0 into the eigenbasis of X
        expH0_half = expm(-1j * H0_in_X * dt/2)  #Half-step evolution under H0 in the eigenbasis of X
        expH0_full = expm(-1j * H0_in_X * dt)  #Full-step evolution under H0 in the eigenbasis of X

        tarr = np.arange(0, T+0.1*dt, dt)
        Diagonal = np.exp(1j*S_opt*D[:,None]*np.sin(w_F*tarr)*dt).T

        U_F = np.eye(N, dtype=complex)
        U_F = expH0_half @ U_F  #Initial half-step evolution under H0
        for i in range(n_steps):
            U_F = Diagonal[i][:,None] * U_F  #Full-step evolution under the driving term in the eigenbasis of X 
            if(i<=n_steps-1):
                U_F = expH0_full @ U_F  #Full-step evolution under H0 in the eigenbasis of X
            if(i%100==0):
                print(f"Step {i}/{n_steps}, t={i*dt:.4f}")
        U_F = expH0_half @ U_F  #Final half-step evolution under H0
        U_F = W @ U_F @ Wd  #Transform back to the eigenbasis of H0

    #Check if U_F is unitary
    if np.allclose(U_F.conj().T @ U_F, np.eye(N)):
        print("U_F is unitary")

    quasi_vals, quasi_vecs = np.linalg.eig(U_F)
    quasi_vecs /= np.sqrt(np.sum(np.abs(quasi_vecs)**2, axis=0))  # Normalize eigenvectors
    quasienergies = np.angle(quasi_vals)/T
    idx = np.argsort(quasienergies)
    quasienergies = quasienergies[idx]
    quasistates = quasi_vecs[:, idx]
    return U_F, quasienergies, quasistates

def load_U_F(potkey, g, w_F, vals, X=None, key='pos_mat', force_recompute=False,method='split_X'):
    path = '/scratch2/vgs23/PISC/examples/1D/quantum/Floquet/Datafiles/'
    try:
        if force_recompute:
            raise Exception("Force recomputation of U_F, qE, qS")
        U_F = np.load('{}/U_F_{}_g_{}_wF_{}_{}.npy'.format(path,potkey,g,w_F,key))
        qE = np.load('{}/qE_{}_g_{}_wF_{}_{}.npy'.format(path,potkey,g,w_F,key))
        qS = np.load('{}/qS_{}_g_{}_wF_{}_{}.npy'.format(path,potkey,g,w_F,key))
    except:
        #Compute the unitary matrix U_F in the eigenbasis of H0 and its eigenvalues and eigenvectors
        print('Computing U_F, qE, qS for key={} g_opt={} w_F={}'.format(key, g, w_F), potkey)
        start_time = time.time()
        if X is None:
            X = operator_matrix(key, vals, pos_mat)
        U_F, qE, qS = compute_U_F_direct(vals, X, g, w_F, n_steps=2000,method=method)
        print('Time taken to compute U_F, qE, qS: {:.2f} seconds'.format(time.time() - start_time))
        np.save('{}/U_F_{}_g_{}_wF_{}_{}.npy'.format(path,potkey,g,w_F,key), U_F)
        np.save('{}/qE_{}_g_{}_wF_{}_{}.npy'.format(path,potkey,g,w_F,key), qE)
        np.save('{}/qS_{}_g_{}_wF_{}_{}.npy'.format(path,potkey,g,w_F,key), qS)
    return U_F, qE, qS

def operator_matrix(key,vals,pos_mat):
    #Create fictional operator with matrix elements that are random, exponentially decaying and gaussian decaying
    neigs = len(vals)
    if key == 'pos_mat':
        X = pos_mat
        return X
    else:
        X = np.zeros((neigs,neigs))
        a_gauss = 1.0
        a_exp = 1.0
        for i in range(neigs):
            for j in range(i, neigs):
                if(key=='gaussian_decay'):
                    X[i,j] = np.exp(-a_gauss*np.abs(vals[i]-vals[j])**2) 
                elif(key=='exponential_decay'):
                    X[i,j] = np.exp(-a_exp*np.abs(vals[i]-vals[j]))
                elif(key=='no_decay'):
                    X[i,j] = 1.0#  np.random.normal(0,1)
                elif(key=='no_decay_2'):
                    X[i,j] = 2.0
                elif(key=='no_decay_half'):
                    X[i,j] = 0.5
                elif(key=='no_decay_quart'):
                    X[i,j] = 0.25
                X[j,i] = X[i,j]
    return X

def compute_ipr(qS, metric='max'):
    """
    Compute the maximum inverse participation ratio (IPR) of the Floquet states.
    
    Parameters:
    qS : ndarray
        Floquet states (eigenvectors of the Floquet operator).

    Returns:
    max_ipr : float
        Maximum IPR value among all Floquet states.
    """
    
    n_qS = qS.shape[1]
    count = 0
    inv_ipr = np.zeros(n_qS)
    for i in range(n_qS):
        if(max(np.abs(qS[:,i])**2) < 0.6):
           count += 1
        inv_ipr[i] = 1.0/np.sum(np.abs(qS[:,i])**4, axis=0)
    
    if(metric=='max'):
        argmax_ipr = np.argmax(inv_ipr)
        ipr_max = np.max(inv_ipr)
        #Record the top 2 projection of the state with max IPR
        E_ind = np.argsort(np.abs(qS[:,argmax_ipr])**2)[-4:][::-1]
        print('E_ind', E_ind)
    elif(metric=='mean'):
        ipr_max = np.mean(inv_ipr)
    
    if(0):
        plt.plot(np.abs(qs[:,argmax_ipr])**2, label='floquet state with max ipr={:.2f}'.format(ipr_max))
        plt.scatter(e_ind, abs(qs[e_ind,argmax_ipr])**2, color='b', label='top 2 projections')
        plt.annotate('e_ind={}'.format(e_ind), xy = (0.6,0.6), xycoords='axes fraction', fontsize=10)
        plt.title('floquet state with maximum ipr')
        plt.xlabel('state index')
        plt.ylabel('amplitude')
        plt.ylim(0,1)
        #plt.legend()
        plt.show()
    
    return ipr_max, E_ind, argmax_ipr

def Floquetiser(vals, w_F, V, n_ph):
    n_F = len(vals)*(2*n_ph+1)
    H_F = np.zeros((n_F,n_F),dtype=complex)

    # Diagonal blocks with photon numbers
    for n in range(-n_ph,n_ph+1):
        idx = (n+n_ph)*len(vals)
        H_F[idx:idx+len(vals),idx:idx+len(vals)] = np.diag(vals + n*w_F)

    # Off-diagonal blocks with driving term
    H_1  = V
    #H_m1 is the conjugate transpose of H_1
    H_m1 = V.conj().T

    for n in range(-n_ph,n_ph):
        idx = (n+n_ph)*len(vals)
        H_F[idx:idx+len(vals),idx+len(vals):idx+2*len(vals)] = H_1
        H_F[idx+len(vals):idx+2*len(vals),idx:idx+len(vals)] = H_m1
    
    #Set H_F as sparse matrix
    #H_F = scipy.sparse.csr_matrix(H_F)
    #floquet_vals, floquet_vecs = scipy.sparse.linalg.eigsh(H_F, k=100, which='SA')  # Compute the lowest 20 eigenvalues and eigenvectors

    #print('H_F \n', np.real(H_F))
    
    # Diagonalize the Floquet Hamiltonian
    #Test if H_F is Hermitian
    #if np.allclose(H_F, H_F.conj().T):
    #    print("H_F is Hermitian")
    
    floquet_vals, floquet_vecs = np.linalg.eigh(H_F)

    #floquet_vals = np.sort(floquet_vals)

    return floquet_vals, floquet_vecs

def test_propa():
    sigma0 = 1.0
    x = np.linspace(-10, 10, 1024)
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=(x[1]-x[0]))
    dx = x[1]-x[0]
    wf0 = np.exp(-x**2/(4*sigma0**2)) / (2*np.pi*sigma0**2)**0.25
    wf0 = wf0.astype(complex)

    Vt_zero = lambda x, t: 0*x
    wf_t = wf0.copy()
    t_total = 5.0
    n_steps = 1000
    dt = t_total/n_steps
    for i in range(n_steps):
        wf_t = propagate_wf_split_op(x, k, wf_t, 1.0, Vt_zero, i*dt, dt)

    # Compare width
    sigma_numerical = np.sqrt(np.sum(x**2 * np.abs(wf_t)**2)*dx)
    sigma_exact = np.sqrt(sigma0**2 + (t_total/(2*sigma0))**2)
    print(f"Numerical σ: {sigma_numerical:.6f}, Exact σ: {sigma_exact:.6f}")

def propagate_wf_split_op(x, k, wf, m, Vt, t, dt):
    """
    Propagate wavefunction using the split operator method.
    We assume that the system is one dimensional 
    Parameters:
    wf_init : ndarray
        Initial wavefunction.
    H : ndarray
        Hamiltonian matrix.
    dt : float
        Time step for propagation.

    Returns:
    wf_arr : ndarray
        Array of wavefunctions at each time point.
    """
    hbar = 1.0

    if(1):
        V = (np.exp(-1j*Vt(x,t)*dt/2)) 
        K = (np.exp(-1j*(1/(2*m*hbar))*k**2*dt))
        
        wf = V*wf
        wf = fft(wf)
        wf = K*wf
        wf = ifft(wf)
        V = (np.exp(-1j*Vt(x,t+dt)*dt/2))
        wf = V*wf
    
    if(0):
        #k = 2*np.pi*np.linspace(-N/2, N/2, N, endpoint=False)/L
        V = (np.exp(-1j*Vt(x,t)*dt/2)) 
        K = (np.exp(-1j*(1/(2*m*hbar))*k**2*dt))
    
        wf = V*wf  
        wf = np.fft.fftshift(np.fft.fft(wf))
        wf = K*wf
        wf = np.fft.ifft(np.fft.ifftshift(wf))
        V = (np.exp(-1j*Vt(x,t+dt)*dt/2))
        wf = V*wf

    return wf


def compute_U_F(x, Vt, w_F):
    """
    Compute the Floquet operator U_F in the position basis
    for a given time-dependent potential Vt and driving frequency w_F.
    
    Parameters:
    x : ndarray
        Spatial grid points.
    Vt : function
        Time-dependent potential function.
    w_F : float
        Driving frequency.

    Returns:
    U_F : ndarray
        Floquet operator matrix.
    """
    
    T = 2*np.pi/w_F
    n_steps = 1000
    dt = T/n_steps
    #tarr = np.arange(0, 2*np.pi/w_F +dt, dt)
    N = len(x)
    k = 2 * np.pi * np.fft.fftfreq(N, d=(x[1]-x[0])) 
    #U_F = np.eye(N, dtype=complex) #Equivalent to starting with a delta function at each point at t=0
    U_F = np.zeros((N, N), dtype=complex)
    
    for i in range(N):                          # outer: basis vector
        wf = np.zeros(N, dtype=complex)
        wf[i] = 1.0                              # delta at grid point i
        for step in range(n_steps):             # inner: propagate over the period
            t = step*dt
            wf = propagate_wf_split_op(x, k, wf, 1.0, Vt, t, dt)
        U_F[:, i] = wf                           # store the full-period result

    #Check if U_F is unitary
    if np.allclose(U_F.conj().T @ U_F, np.eye(N)):
        print("U_F is unitary")
   
    dx = x[1]-x[0]
    # Note that norm is 1 in the spatial grid, so we need to fix the normalization of U_F
    vals, vecs = np.linalg.eig(U_F)
    vecs /= np.sqrt(np.sum(np.abs(vecs)**2, axis=0)*dx)  # Normalize eigenvectors

    quasienergies = np.angle(vals)/T
    idx = np.argsort(quasienergies)
    quasienergies = quasienergies[idx]
    quasistates = vecs[:, idx]
    #print('Quasienergies:', quasienergies[:10])

    return U_F, quasienergies, quasistates


def fidelity(x, wf_init, m, Vt, tarr):
    """
    Compute the fidelity of a wavefunction under time evolution.
    Parameters:
    x : ndarray
        Spatial grid points.
    wf_init : ndarray
        Initial wavefunction.
    Vt : function
        Time-dependent potential function.
    tarr : ndarray
        Array of time points.

    Returns:
    fidelity_arr : ndarray
        Array of fidelity values at each time point.
    """
    dx = x[1]-x[0]
    dt = tarr[1]-tarr[0]
    N = len(x)
    L = x[-1]-x[0]
    #k = 2*np.pi*np.linspace(-N/2, N/2, N, endpoint=False)/L
    k = 2 * np.pi * np.fft.fftfreq(N, d=(x[1]-x[0])) 
    
    fidelity_arr = np.zeros(len(tarr))
    wf_t = wf_init.copy() 
    
    fid_0 = np.abs(np.sum(np.conj(wf_init)*wf_t)*dx)**2
    print('Initial fidelity:', fid_0)
    for i, t in enumerate(tarr):
        fidelity_arr[i] = np.abs(np.sum(np.conj(wf_init)*wf_t)*dx)**2
        if(i%1000==0):
            print('t={:.2f}, fidelity={:.4f}'.format(t, fidelity_arr[i]))

        #if i % 100 == 0: 
            #plt.plot(x, np.abs(wf_t)**2, label='t={:.2f}'.format(t))
        #V = Vt(x, t)
        wf_t = propagate_wf_split_op(x, k, wf_t, m, Vt, t, dt)
        #print('norm', np.sum(np.abs(wf_t)**2)*dx)
    return fidelity_arr

