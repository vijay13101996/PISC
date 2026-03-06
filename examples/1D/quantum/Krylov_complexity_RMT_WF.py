import numpy as np
import matplotlib.pyplot as plt
from PISC.engine import Krylov_complexity_1D as Krylov_complexity
from Krylov_WF_tools import coh_st_coeff, wf_t, comp_Ot, Cn, corr_func, av_O, fix_vecs, av_O_therm
from scipy.optimize import bisect
from PISC.engine import Krylov_complexity

sigma = 1

rngSeed = 102
rng = np.random.default_rng(rngSeed)


def random_matrix(neigs):
    """Generate a random NxN matrix with elements drawn from a normal distribution."""
    H = np.zeros((neigs, neigs))

    for i in range(neigs):
        for j in range(i,neigs):
            if i == j:
                H[i, j] = rng.normal(0, np.sqrt(2)*sigma)
            else:
                H[i, j] = rng.normal(0, sigma)
            H[j, i] = H[i, j]
    return H

def TCF_O(vals, beta, neigs, O, t_arr):
    
    n_arr = np.arange(neigs)
    m_arr = np.arange(neigs)

    C_arr = np.zeros_like(t_arr) + 0j

    for n in n_arr:
        for m in m_arr:
            C_arr += np.exp(-beta*(vals[n]+vals[m])/2) * np.exp(1j*(vals[n]-vals[m])*t_arr) * np.abs(O[n,m])**2

    Z = np.sum(np.exp(-beta*vals))
    C_arr /= Z

    return C_arr

def TCF_O_wf(vals, vecs, neigs, O, t_arr, coeff_wf):
    
    l_arr = np.arange(neigs)
    k_arr = np.arange(neigs)
    m_arr = np.arange(neigs)

    C_arr = np.zeros_like(t_arr) + 0j

    for l in l_arr:
        c_l = coeff_wf[l]
        for k in k_arr: 
            Olk = O[l,k]
            Elk = vals[l]-vals[k]
            for m in m_arr:
                Okm = O[k,m]
                c_m = coeff_wf[m]
                Ekm = vals[k]-vals[m]
                C_arr += np.conj(c_l)*c_m * Olk * Okm * (np.exp(1j*Elk*t_arr) + np.exp(1j*Ekm*t_arr))
    
    print('Normalization of coeff_wf:', np.sum(np.abs(coeff_wf)**2))

    return C_arr

def O_E(E):
    return E #e-1*E**2

def f_O(vals,n,m):
    E = (vals[n]+vals[m])/2
    w = (vals[n]-vals[m])
    if(n==m):
        return O_E(E)
    else:
        return rng.normal(0,1)

neigs = 200

H = random_matrix(neigs)
vals, vecs = np.linalg.eigh(H)
vals = np.sort(vals)

vals -= vals[0]  # Shift ground state energy to zero

#Scale values to have mean level spacing of 1
mean_level_spacing = np.mean(np.diff(vals))
#vals /= mean_level_spacing

L = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,L)

O = np.zeros((neigs, neigs))
for n in range(neigs):
    for m in range(n,neigs):
        O[n,m] = f_O(vals,n,m)
        O[m,n] = O[n,m]

#Check if O is Hermitian
if not np.allclose(O, O.conj().T):
    print("Error: Operator O is not Hermitian!")


#Find temperature such that <E> = vals[nc]
def func_to_solve(beta, nc):
    Z = np.sum(np.exp(-beta*vals))
    E_avg = np.sum(vals * np.exp(-beta*vals)) / Z    
    return E_avg - vals[nc]

def find_beta_opt(nc):
    beta_opt = bisect(func_to_solve, 0.01, 5.0, args=(nc,), xtol=1e-6)
    return beta_opt

if(0):

    nc_arr = np.arange(5,51,5)
    beta_arr = []
    for nc in nc_arr:
        beta_opt = find_beta_opt(nc)
        print('Optimal beta for <E> = vals[{}] = {} is beta = {}'.format(nc, vals[nc], beta_opt))
        beta_arr.append(beta_opt)

    plt.plot(nc_arr, beta_arr, marker='o')
    plt.xlabel('nc (state index)')
    plt.ylabel('Optimal beta for <E> = vals[nc]')
    plt.title('Optimal Inverse Temperature vs State Index')
    plt.grid()
    plt.show()

    exit()

if(0): # Plot O(t) and see if it equilibrates to the ETH prediction
    nc = 50
    beta = find_beta_opt(nc)
    print('Using beta = {} for nc = {}'.format(beta, nc))
    t_arr = np.linspace(0, 100, 501)
    
    coeff_wf = np.exp(-0.1*(np.arange(neigs)-nc)**2)
    coeff_wf /= np.sum(coeff_wf**2)**0.5
    #print('Normalization of coeff_wf:', np.sum(np.abs(coeff_wf)**2), coeff_wf)
    
    Ot_arr_therm = np.zeros(len(t_arr), dtype=complex)
    Ot_arr_wf = np.zeros(len(t_arr), dtype=complex)
    for i in range(len(t_arr)):
        t = t_arr[i]
        Ot = comp_Ot(O, L, t)
        #Check if Ot is Hermitian
        if not np.allclose(Ot, Ot.conj().T):
            print("Error: Operator O(t) is not Hermitian at t={}".format(t))

        Ot_arr_therm[i] = av_O_therm(Ot, vals, beta)
    
    Ot_arr_wf = av_O(O, t_arr, coeff_wf, vals)
    Oavg_therm = av_O_therm(O, vals, beta)
    Oavg_wf = av_O(O, t_arr, coeff_wf, vals)[0]
    print('Long-time average O(t) thermal = {}, wavefunction = {}'.format(Oavg_therm, Oavg_wf)) 
    
    plt.plot(t_arr, Ot_arr_therm.real, label='O(t) thermal avg')
    plt.plot(t_arr, Ot_arr_wf.real, '--', label='O(t) wavefunction avg nc={}'.format(nc))
    

    plt.xlabel('Time')
    plt.ylabel('O(t)')
    plt.title('Time Evolution of Operator O(t)')
    plt.legend()
    plt.show()
    exit()


if(0): #Compare the TCFs obtained from thermal average and wavefunction average
    nc = 30
    beta = find_beta_opt(nc)
    print('Using beta = {} for nc = {}'.format(beta, nc))
    t_arr = np.linspace(0, 100, 500)
    C_arr = TCF_O(vals, beta, neigs, O, t_arr)
    
    Cavg = np.mean(C_arr[-50:].real) # Average over last 10 time points
    plt.plot(t_arr, C_arr, label='TCF_O real')
    plt.hlines(Cavg, t_arr[0], t_arr[-1], colors='r', linestyles='dashed', label='Long-time avg={:.4f}'.format(Cavg))

    coeff_wf = np.exp(-10*(np.arange(neigs)-nc)**2)
    coeff_wf /= np.sum(coeff_wf**2)**0.5

    C_arr_wf = TCF_O_wf(vals, vecs, neigs, O, t_arr, coeff_wf)
    Cavg_wf = np.mean(C_arr_wf[-50:].real)
    plt.plot(t_arr, C_arr_wf.real, '--', label='TCF_O_wf real nc={}'.format(nc))
    plt.hlines(Cavg_wf, t_arr[0], t_arr[-1], colors='g', linestyles='dashed', label='Long-time avg wf={:.4f}'.format(Cavg_wf))

    plt.xlabel('Time')
    plt.ylabel('TCF_O')
    plt.title('Time-Correlation Function for Random Matrix Hamiltonian')
    plt.legend()
    plt.show()

    exit()

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

for nc in [40]: #4,10,20,30,40]:

    beta = find_beta_opt(nc)
    print('Using beta = {} for nc = {}'.format(beta, nc))

    ncoeff = 100
    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')

    plt.scatter(np.arange(1,ncoeff), barr[1:].real, label='Krylov coefficients for O')

    coeff_wf = np.exp(-1*(np.arange(neigs)-nc)**2)
    coeff_wf /= np.sum(coeff_wf**2)**0.5

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs_wf(O, L, barr, coeff_wf)

    plt.scatter(np.arange(1,ncoeff), barr[1:].real, label='Krylov coefficients for O with wf nc={}'.format(nc),marker='x')
    
    vals_half = vals[-1]//2
    print('vals[neigs/2]/2 = ', vals_half)
    plt.hlines(vals_half, 1, ncoeff-1, colors='r', linestyles='dashed', label='vals[neigs/2]/2={}'.format(vals_half))

plt.xlabel('Coefficient index')
plt.ylabel('Coefficient value')
plt.title('Krylov Coefficients for Random Matrix Hamiltonian')
plt.legend()
plt.show()

