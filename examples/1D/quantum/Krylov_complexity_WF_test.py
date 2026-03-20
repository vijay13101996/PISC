import numpy as np
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity_1D as Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
import scipy
from matplotlib import pyplot as plt
from Krylov_WF_tools import coh_st_coeff, wf_t, comp_Ot, Cn, corr_func, av_O, avg_O, fix_vecs, mom_to_bn

"""
In this module, we compute the Lanczos coefficients and the corresponding basis operators
for a given Hamiltonian and initial operator using the Lanczos algorithm.

We then use that to verify that the wavefunction average computed from the Krylov basis
and the Lanczos coefficients matches the exact wavefunction average.
"""

ngrid = 1000

Len = 40

lb = -Len
ub = Len

a = 0.0
b = 0.0
omega = 2.0
m = 1.0
n_anharm = 4
pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

def pos_mat_anal(neigs, m, omega):
    print('m, omega in anal pos mat:', m, omega)
    pos_mat = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(neigs):
            if(i-j==1):
                pos_mat[i,j] = np.sqrt(1/(2*m*omega))*np.sqrt(j+1)
            elif(j-i==1):
                pos_mat[i,j] = np.sqrt(1/(2*m*omega))*np.sqrt(i+1)
            
    return pos_mat

def mom_pref(coeff_wf, sign=1):
    pref = 0.0 + 0j
    for l in range(len(coeff_wf)):
        pref += np.abs(coeff_wf[l])**2 *(2*l +1)
        if l>=2:
            pref += np.conj(coeff_wf[l])*coeff_wf[l-2]*np.sqrt(l*(l-1))*sign
        if l<=len(coeff_wf)-3:
            pref += np.conj(coeff_wf[l])*coeff_wf[l+2]*np.sqrt((l+1)*(l+2))*sign
    #print('mass', m, 'omega', omega)
    return pref/(2*m*omega)

def Cx_t(coeff_wf, O, tarr):
    Ct_arr = np.zeros(len(tarr), dtype=complex)
    pref = 1/(2*m*omega)
    for l in range(len(coeff_wf)):
        Ct_arr += pref * np.abs(coeff_wf[l])**2 * (2*l +1) * np.cos(omega*tarr) 
        if l>=2:
            Ct_arr += pref * np.conj(coeff_wf[l])*coeff_wf[l-2]*np.sqrt(l*(l-1)) * np.exp(1j*omega*tarr)
        if l<=len(coeff_wf)-3:
            Ct_arr += pref * np.conj(coeff_wf[l])*coeff_wf[l+2]*np.sqrt((l+1)*(l+2)) * np.exp(-1j*omega*tarr)
    return Ct_arr

def Cx_t_full(coeff_wf, vals, O, tarr):
    Ct_arr = np.zeros(len(tarr), dtype=complex)
    for l in range(len(O)):
        for m in range(len(O)):
            for k in range(len(O)):
                Olk = O[l,k]
                Okm = O[k,m]
                Elk = vals[l]-vals[k]
                Ekm = vals[k]-vals[m]
                Ct_arr += np.conj(coeff_wf[l])*coeff_wf[m]* Olk * Okm *\
                        (np.exp(1j*Elk*tarr) + np.exp(1j*Ekm*tarr))/2
    return Ct_arr

def compute_ip_WF(O1, O2, coeff_wf):
    ip = 0.0 + 0j
    for l in range(len(coeff_wf)):
        for m in range(len(coeff_wf)):
            for k in range(len(coeff_wf)):
                ip += np.conj(coeff_wf[l])*coeff_wf[m]* np.conj(O1[k,l])*O2[k,m]
    
    return ip

potkey = 'L_{}_MAH_w_{}_a_{}_b_{}_n_{}'.format(Len,omega,a,b,n_anharm)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 200
vals,vecs = DVR.Diagonalize(neig_total=neigs)

vecs = fix_vecs(vecs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

O = pos_mat_anal(neigs, m, omega)

#print('O', O[:10,:10])
#pos_mat = np.zeros((neigs,neigs))
#O = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)
#L = L*1j

x0 = 0.0
p0 = 1.0
sigma_x = 1.

coeff_wf, wf = coh_st_coeff(x0, p0, sigma_x, x_arr, vecs, neigs)


if(0): #Test if the correlation function works correctly
    #print('coeff_wf:', coeff_wf, np.sum(np.abs(coeff_wf)**2))
    tmax = 20.0
    ntimes = 401
    tarr = np.linspace(0.0, tmax, ntimes)

    Ct_arr_FORT = np.zeros(len(tarr), dtype=complex)
    Ct_arr_PY = Cx_t(coeff_wf, O, tarr)
    #Ct_arr_PY_full = Cx_t_full(coeff_wf, vals, O, tarr)

    for i in range(len(tarr)):
        t = tarr[i]
    #if(1):
        #t = 1.
        Ot = comp_Ot(O, L, t)
        ip = 0.0
        ip = Krylov_complexity.krylov_complexity.compute_ip_wf(Ot, O, coeff_wf, ip)
        print('t={}, ip={}'.format(t, ip))
        
        ip_py = Cx_t(coeff_wf, O, np.array([t]))[0]
        #print('t={}, ip_py={}'.format(t, ip_py))

        #ip_full = Cx_t_full(coeff_wf, vals, O, np.array([t]))[0]
        #print('t={}, ip_full={}'.format(t, ip_full))
        Ct_arr_FORT[i] = ip
    #exit()


    plt.plot(tarr, Ct_arr_FORT.real, label='C(t) FORT real')
    plt.plot(tarr, Ct_arr_FORT.imag, label='C(t) FORT imag')    
    plt.plot(tarr, Ct_arr_PY.real, '--', label='C(t) PY real', lw=3)
    #plt.plot(tarr, Ct_arr_PY.imag, '--', label='C(t) PY imag', lw=3)
    plt.xlabel('Time')
    plt.ylabel('C(t)')
    plt.title('Correlation Function C(t) vs Time')
    plt.legend()
    plt.show()
    exit()

#coeff_wf[:] = 0.0
#coeff_wf[:1] = 1.0/np.sqrt(1.0)
print('coeff_wf:', np.sum(np.abs(coeff_wf)**2))



ip = 0.0 + 0j
ip = Krylov_complexity.krylov_complexity.compute_ip_wf(O, O, coeff_wf, ip)
O[:] = O[:] + 0.0j
O[:] = O[:]/np.sqrt(ip)

if(0):
    An = L*L*O
    #Check if An is Hermitian
    #assert np.allclose(An, An.conj().T), "An is not Hermitian"

    LO_LO_ip = compute_ip_WF(O, L*L*L*O, coeff_wf)
    print('<LO|LO> = ', LO_LO_ip)
    O_LLO_ip = compute_ip_WF(L*L*L*O, O, coeff_wf)
    print('<O|LLO> = ', O_LLO_ip)
    

    print('OO', compute_ip_WF(O, O, coeff_wf))
    for p0 in [0.0, 0.5, 1.0, 1.5, 2.0]:
        coeff_wf, wf = coh_st_coeff(x0, p0, sigma_x, x_arr, vecs, neigs)
        AnAn = compute_ip_WF(An, An, coeff_wf)
        print('p0:', p0, ' AnAn = ', AnAn)
        
        AnAn_anal = mom_pref(coeff_wf, sign=1)*omega**2
        print('p0:', p0, ' AnAn_anal = ', AnAn_anal)
        #print('coeff',coeff_wf[:5])
    exit()

print('Initial inner product <O|O> = ', Krylov_complexity.krylov_complexity.compute_ip_wf(O, O, coeff_wf, 0.0))


#L = L*L
ncoeff = 50
barr = np.zeros(ncoeff)
barr[:] = 0.0
barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs_wf(O, L, barr, coeff_wf)

prefactor = mom_pref(coeff_wf)
print('Prefactor for moments from wf:', prefactor)

temp = 0.0
for l in range(neigs):
    for m in range(neigs):
        for k in range(neigs):
            temp += 2*O[l,k]*O[k,m]* np.conj(coeff_wf[l])*coeff_wf[m]/2
print('Temp check for moment prefactor:', temp)

moments = np.zeros(10)
muarr = Krylov_complexity.krylov_complexity.compute_moments_wf(O, vals, coeff_wf, moments)
even_moments = np.real(muarr[::2])
#print('Computed moments from wf:', np.around(muarr,5))

if(0):
    An = L*O
    b2 = compute_ip_WF(An, An, coeff_wf)**0.5
    
    O1O2 = compute_ip_WF(O, An, coeff_wf)
    print('O1O2:', O1O2)
    O2O1 = compute_ip_WF(An, An, coeff_wf)
    print('O2O1:', O2O1)

    print('Computed b_1 from An = L O:', b2, ' vs Lanczos b_1:', barr[1])
    print('mu', even_moments[:4])
    exit()


barr_from_moments = mom_to_bn(even_moments)
print('Computed Lanczos b_n from moments:', np.around(barr_from_moments,5)[:5], '\n from Lanczos alg:', np.around(barr,5)[:50])

exit()

plt.plot(barr, label='Lanczos from Krylov alg')
#plt.plot(barr_from_moments, '--', label='Lanczos from Moments')
plt.xlabel('n')
plt.ylabel('b_n')
plt.title('Lanczos Coefficients Comparison')
plt.legend()
plt.show()

exit()

if(0):
    avg_On = np.zeros(ncoeff, dtype=complex)

    for n in range(ncoeff):
        On_mat_n = np.zeros((neigs,neigs), dtype=complex)
        barr_n, On_mat_n = Krylov_complexity.krylov_complexity.compute_on_matrix_wf(O, L, barr, coeff_wf, On_mat_n, n)
        print('Computed O_{} matrix.'.format(n), On_mat_n.shape)

        #Compute <\psi | On | \psi >
        av_On = avg_O(On_mat_n, coeff_wf) 
        print('<psi| O_{} |psi> = {}'.format(n, av_On))
        avg_On[n] = av_On

phi_mat = np.zeros((ncoeff,ncoeff))
for n in range(ncoeff):
    for m in range(ncoeff):
        if(n==m):
            phi_mat[n,m] = 0.0
        elif(n==m+1):
            phi_mat[n,m] = barr[n]
        elif(n==m-1):
            phi_mat[n,m] = -barr[m]

print('Phi matrix shape:', phi_mat[:10,:10])

#Solve for \dot{\phi}(t) = phi_mat \phi, with phi(0) = |0>
tmax = 10.0
ntimes = 201
tarr = np.linspace(0.0, tmax, ntimes)   

def func(y,t, A):
    return A @ y

y0 = np.zeros(ncoeff)
y0[0] = 1.0
phi_t = scipy.integrate.odeint(func, y0, tarr, args=(phi_mat,))

for i in range(len(tarr)):
    norm_phi = np.sum( np.abs(phi_t[i,:])**2 )
    #print('Norm of phi_t at time {} is {}'.format(tarr[i], norm_phi))
    if( abs(norm_phi-1.0) > 1e-6 ):
        print('Warning: Norm of phi_t at time {} is {}'.format(tarr[i], norm_phi))

C_t = corr_func(tarr, vals, O, L, coeff_wf)
plt.plot(tarr, C_t.real, label='C(t) from exact diag')
plt.plot(tarr, C_t.imag, label='C(t) imag from exact diag')

plt.plot(tarr, phi_t[:,0].real, '--', label='C(t) from Krylov', lw=3)
plt.xlabel('Time')
plt.ylabel('C(t)')
plt.title('Correlation Function C(t) vs Time')
plt.legend()
plt.show()

K_comp = np.zeros(ntimes)
for it in range(ntimes):
    K_comp[it] = np.sum( np.arange(ncoeff) * np.abs(phi_t[it,:])**2 )
plt.plot(tarr, K_comp, label='Krylov Complexity from Lanczos Coeffs')
plt.yscale('log')

plt.xlabel('Time')
plt.ylabel('Krylov Complexity')
plt.title('Krylov Complexity vs Time')
plt.legend()
plt.show()
