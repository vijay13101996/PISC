import numpy as np
from PISC.engine import Krylov_complexity_1D as Krylov_complexity
import math
from scipy.special import factorial as sp_factorial
from matplotlib import pyplot as plt
import scipy

def fix_vecs(vecs, tol=1e-3):
    # Ensure all vectors have a consistent sign
    neigs = vecs.shape[1]
    for n in range(neigs):
        # Find the maximum index after which values are below the tolerance
        ind_list = np.where(np.abs(vecs[:, n]) > tol)[0]
        #ind_min = ind_list[0]
        ind_max = ind_list[-1]
        max_index = ind_max
        if vecs[max_index, n] < 0:
            vecs[:, n] = -vecs[:, n]
    return vecs

def comp_Ot(O, L, t):
    # Compute the time-evolved operator Ot
    return O*np.exp(1j*L*t)

def Cn(t, vals, O, n):
    # Compute the correlation function Cn(t)
    neigs = len(vals)
    Cn_t = 0.0
    for m in range(neigs):
            Cn_t += abs(O[n,m])**2 * np.cos((vals[n]-vals[m])*t)#np.exp(1j*(vals[n]-vals[m])*t)
    return Cn_t

def av_O(O, tarr, coeff_wf, vals):
    avg = np.zeros(len(tarr), dtype=complex)
    neigs = len(vals)
    for n in range(neigs):
        for m in range(neigs):
            avg += np.conj(coeff_wf[n]) * O[n,m] * coeff_wf[m] * np.exp(1j*(vals[n]-vals[m])*tarr)
    return avg

def avg_O(O, coeff_wf):
    avg = 0.0
    neigs = O.shape[0]
    for n in range(neigs):
        for m in range(neigs):
            avg += np.conj(coeff_wf[n]) * O[n,m] * coeff_wf[m]
    return avg

def av_O_therm(O, vals, beta):
    Z = np.sum(np.exp(-beta*vals))
    neigs = len(vals)
    avg = 0.0
    for n in range(neigs):
        avg += (np.exp(-beta*vals[n]) / Z) * O[n,n]
    return avg

def find_coeff_wf(wf, vecs, dx):
    neigs = vecs.shape[1]
    coeff_wf = np.zeros(neigs, dtype=complex)
    for n in range(neigs):
        coeff_wf[n] = np.sum(np.conj(vecs[:,n]) * wf)*dx

    return coeff_wf

def find_coeff_wf_2D(wf, vecs, dx, dy, DVR):
    neigs = vecs.shape[1]
    coeff_wf = np.zeros(neigs, dtype=complex)
    for n in range(neigs):
        eigstate_n = DVR.eigenstate(vecs[:,n])
        coeff_wf[n] = np.sum(np.conj(eigstate_n) * wf)*dx*dy

    return coeff_wf

def coh_st_coeff(x0, p0, sigma_x, x_arr, vecs, neigs):
    dx = x_arr[1]-x_arr[0]
    
    # Define the coherent state wavefunction
    wf = (1.0/(np.pi*sigma_x**2))**0.25 * np.exp(-(x_arr - x0)**2/(2.0*sigma_x**2) + 1j*p0*x_arr)
    
    # Normalize the wavefunction
    norm = np.sqrt(np.sum(np.abs(wf)**2)*dx)
    print('Norm before normalization:', norm)
    wf = wf / norm  

    alpha = (1/np.sqrt(2)) *(x0/sigma_x + 1j*p0*sigma_x)

    # Expand the wavefunction in the energy eigenbasis
    coeff_wf = np.zeros(neigs, dtype=complex)
    for n in range(neigs):
        coeff_wf[n] = np.sum(np.conj(vecs[:,n]) * wf)*dx
        
        #coeff_anal = (alpha**n) / np.sqrt(sp_factorial(n)) * np.exp(-0.5 * np.abs(alpha)**2)
        #print(f'n={n}, coeff_numerical={coeff_wf[n]:.10f}, coeff_analytical={coeff_anal:.10f}')
        #coeff_wf[n] = coeff_anal
        
        #Verify if coeff_wf and coeff_anal are close
        #if not np.isclose(coeff_wf[n], coeff_anal, atol=1e-5):
        #    print(f'Warning: Coefficient mismatch at n={n}: numerical={coeff_wf[n]:.10f}, analytical={coeff_anal:.10f}')

    return coeff_wf, wf

def coh_st_coeff_2D(x0, y0, p0x, p0y, sigma_x, sigma_y, x_arr, y_arr, vecs, neigs, DVR):
    dx = x_arr[1]-x_arr[0]
    dy = y_arr[1]-y_arr[0]
    
    # Define the coherent state wavefunction in 2D
    X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
    wf = (1.0/(np.pi*sigma_x*sigma_y))**0.5 * np.exp(-(X - x0)**2/(2.0*sigma_x**2) - (Y - y0)**2/(2.0*sigma_y**2) + 1j*(p0x*X + p0y*Y))
    
    # Normalize the wavefunction
    norm = np.sqrt(np.sum(np.abs(wf)**2)*dx*dy)
    print('Norm before normalization:', norm)
    wf = wf / norm  

    # Expand the wavefunction in the energy eigenbasis
    coeff_wf = np.zeros(neigs, dtype=complex)
    for n in range(neigs):
        eigstate_n = DVR.eigenstate(vecs[:,n])
        coeff_wf[n] = np.sum(np.conj(eigstate_n) * wf)*dx*dy

    return coeff_wf, wf

def corr_func(tarr, vals, O, L, coeff_wf):
    Ct_arr = np.zeros(len(tarr), dtype=complex)

    for i in range(len(tarr)):
        t = tarr[i]
        Ot = comp_Ot(O, L, t)
        ip = 0.0 + 0j 
        ip = Krylov_complexity.krylov_complexity.compute_ip_wf(Ot, O, coeff_wf, ip)
        #print('Inner product at time {} is {}'.format(t, ip))
        Ct_arr[i] = ip

    return Ct_arr

def wf_t(wf, vecs, vals, tarr,N=10):
    wf_tarr = np.zeros((len(wf), len(tarr)), dtype=complex)
    coeff_wf = find_coeff_wf(wf, vecs, dx)
    
    narr = np.arange(1, N+1)
    avg_xn = np.zeros((N, len(tarr)), dtype=complex)
    for i in range(len(tarr)):
        t = tarr[i]
        wf_t = np.zeros(len(wf), dtype=complex)
        for n in range(neigs):
            wf_t += coeff_wf[n] * vecs[:,n] * np.exp(-1j*vals[n]*t)
        wf_tarr[:,i] = wf_t
        for k in range(N):
            avg_xn[k,i] = np.sum(np.conj(wf_tarr[:,i]) * (x_arr**(k+1)) * wf_tarr[:,i]) * dx
    return wf_tarr, avg_xn


def mom_to_bn(even_moments):
    ncoeff = len(even_moments) 
    bnarr = np.zeros(ncoeff)
    Marr = np.zeros((ncoeff,ncoeff)) # index l is the row, index j is the column
    Marr[:,0] = even_moments # The M matrix is to be filled from left to the diagonal, first column is the moments

    bnarr[0] = 1.0 # We assume that the operator is normalized

    for l in range(1,ncoeff): # b0 and M[0,0] are already set, and the first row is filled until the diagonal.
        for j in range(1,l+1): # First column is already filled, so we start from j=1
            if (j==1): # Fill the first column
                Marr[l,j] = Marr[l,j-1]/bnarr[j-1]**2 # M[:,-1] is set to zero by default
            else:
                Marr[l,j] = Marr[l,j-1]/bnarr[j-1]**2 - Marr[l-1,j-2]/bnarr[j-2]**2
        
        bnarr[l] = np.sqrt(Marr[l,l]) # The diagonal element is the next b

    return bnarr


def compute_Wn(O, coeffs, vals, bn):
    Wn = np.zeros(len(bn)) + 0j
    neigs = len(vals)

    O_m1 = np.zeros((neigs, neigs), dtype=complex)
    O_n = O.copy()
    O_p1 = np.zeros((neigs, neigs), dtype=complex)

    for n in range(len(bn)-1):
        b_n = bn[n]
        b_p1 = bn[n+1] 

        for j in range(neigs):
            for k in range(neigs):
                temp =  np.conj(coeffs[j])*coeffs[k]*O_n[j, k]     
                #if(np.abs(temp)>1e-12):
                #    print('j={}, k={}, term={}'.format(j, k, temp))
                Wn[n] += temp
                Ej = vals[j]
                Ek = vals[k]
                O_p1[j, k] = ((Ej-Ek)*O_n[j, k] - b_n*O_m1[j, k]) / b_p1

        print('Wn[{}] = {}'.format(n, Wn[n]))
        O_m1 = O_n.copy()
        O_n = O_p1.copy()
        O_p1 = np.zeros((neigs, neigs), dtype=complex)
        
        for j in range(neigs):
            for k in range(neigs):
                Wn[-1] += np.conj(coeffs[j])*coeffs[k]*O_n[j, k]

    
    print('Wn[{}] = {}'.format(len(bn)-1, Wn[-1]))
    return Wn

def get_phi_t(barr, tarr):
    L_krylov = np.zeros((len(barr), len(barr)), dtype=complex)
    for n in range(len(barr)):
        if(n>0):
            L_krylov[n, n-1] = barr[n]
        if(n<len(barr)-1):
            L_krylov[n, n+1] = -barr[n+1]

    phi_t = np.zeros((len(tarr), len(barr)), dtype=complex)
    phi_t[0,0] = 1.0 + 0j   
    for i in range(1, len(tarr)):
        dt = tarr[i] - tarr[i-1]
        U_dt = scipy.linalg.expm(L_krylov * dt)
        phi_t[i,:] = U_dt @ phi_t[i-1,:]
    
    return phi_t

def verify_Ct(O, vals, barr, beta, neigs, tarr=None, ip='wgm'):
    if tarr is None:
        tarr = np.arange(0.0, 10.01, 0.01)
    Ct_arr = np.zeros(len(tarr), dtype=complex)

    if(ip=='wgm'):
        print('Using Wightman inner product for verification')
        Z = 0.0 
        for i in range(neigs):
            for j in range(neigs):
                Ct_arr += np.exp(-0.5*beta*(vals[i]+vals[j])) * abs(O[i,j])**2 * np.exp(1j*(vals[i]-vals[j])*tarr)
            Z += np.exp(-beta*vals[i]) 
        
        Ct_arr = Ct_arr / Z

    elif(ip=='dir'):
        print('Using Hilbert-Schmidt inner product for verification')
        for i in range(neigs):
            for j in range(neigs):
                Ct_arr += abs(O[i,j])**2 * np.exp(1j*(vals[i]-vals[j])*tarr)

    phi_t = get_phi_t(barr, tarr)

    # Compare phi_t[0] with Ct_arr
    plt.plot(tarr, Ct_arr.real, label='Ct_arr real')
    #plt.plot(tarr, Ct_arr.imag, label='Ct_arr imag')
    plt.plot(tarr, phi_t[:,0].real, '--', label='phi_t[0] real')
    #plt.plot(tarr, phi_t[:,0].imag, '--', label='phi_t[0] imag')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('C(t)')
    plt.show()


