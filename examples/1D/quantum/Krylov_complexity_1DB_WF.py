import numpy as np
from PISC.engine import Krylov_complexity_1D as Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
import scipy
from matplotlib import pyplot as plt
from Krylov_WF_tools import coh_st_coeff, wf_t, comp_Ot, Cn, corr_func, av_O, fix_vecs

ngrid = 1000

Len = 8.0

R = np.sqrt(1/(4+np.pi))
a = R#0.0

lb = 0# -(a+R)#0
ub = Len # a+R#Len

m = 1.0 #0.5


def potential(x):
    if(x<lb or x>ub):
        return 1e12
    else:
        return 0.0

neigs = 200
potential = np.vectorize(potential)

DVR = DVR1D(ngrid, lb, ub, m, potential)
vals,vecs = DVR.Diagonalize(neig_total=neigs)

print('vecs', vecs.shape)

vecs = fix_vecs(vecs, tol=1e-3)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

def pos_mat_anal(neigs, m, L):
    pos_mat_anal = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(neigs):
            if(i==j):
                pos_mat_anal[i,j] = L/2
            else:
                pos_mat_anal[i,j] = L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2
    return pos_mat_anal


vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*Len**2)
vecs_anal = np.zeros((ngrid-1,neigs))
for i in range(neigs):
    vecs_anal[:,i] = np.sqrt(2/Len)*np.sin((i+1)*np.pi*x_arr/Len)

O = pos_mat_anal(neigs, m, Len)
vecs = vecs_anal
vals = vals_anal

potkey = 'L_{}_m_{}_1DB'.format(Len,m)


if(1): #Use Eigenstates
    n_wf = 10
    wf = vecs[:,n_wf]  # ground state wavefunction
    
    coeff_wf = np.zeros(neigs)
    coeff_wf[n_wf] = 1.0  # initial state is the ground state

if(0): #Use Coherent State
    x0 = 0.0    
    p0 = 1.0
    sigma_x = 0.25
    

    if(0):
        # Reconstruct the wavefunction from the eigenbasis expansion
        wf_reconstructed = np.zeros_like(wf, dtype=complex)
        for n in range(neigs):
            wf_reconstructed += coeff_wf[n] * vecs[:,n]

        #plt.plot(x_arr, np.abs(wf)**2, label='Coherent State |ψ(x)|²')
        #plt.plot(x_arr, np.abs(wf_reconstructed)**2, '--', label='Reconstructed |ψ_reconstructed(x)|²')
        plt.plot(x_arr, wf.real, label='Coherent State Re(ψ(x))')
        plt.plot(x_arr, wf_reconstructed.real, '--', label='Reconstructed Re(ψ_reconstructed(x))')
        plt.plot(x_arr, wf.imag, label='Coherent State Im(ψ(x))')
        plt.plot(x_arr, wf_reconstructed.imag, '--', label='Reconstructed Im(ψ_reconstructed(x))')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.title('Coherent State and its Reconstruction from Eigenstates')
        plt.show()
        exit()

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

if(0):  # TEST!!
    tarr = np.arange(0.0,10.01,0.1)
    Ot = np.zeros((neigs,neigs), dtype=complex)

    Ct_arr = np.zeros(len(tarr), dtype=complex)
    Ct_arr_exact = np.zeros(len(tarr), dtype=complex)

    for i in range(len(tarr)):
        t = tarr[i]
        Ot = comp_Ot(O, L, t)
        ip = 0.0
        ip = Krylov_complexity.krylov_complexity.compute_ip_wf(Ot, O, coeff_wf, ip)
        print('t={}, ip={}'.format(t, ip))
        Ct_arr_exact[i] = ip

        Cn_t = Cn(t, vals, O, n_wf)
        print('t={}, Cn_t={}'.format(t, Cn_t))
        Ct_arr[i] = Cn_t

    plt.plot(tarr, Ct_arr.real, label='Cn_t (sum over states)')
    plt.plot(tarr, Ct_arr_exact.real, '--', label='Ct_arr_exact (Krylov)')

    plt.plot(tarr, Ct_arr.imag, label='Cn_t imag (sum over states)')
    plt.plot(tarr, Ct_arr_exact.imag, '--', label='Ct_arr_exact imag (Krylov)')
    plt.legend()

    plt.xlabel('Time')
    plt.ylabel('Correlation Function')
    plt.title('Comparison of Correlation Functions')
    plt.show()

    exit()


LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

ncoeff = 100
barr = np.zeros(ncoeff)

x0 = Len/2
p0 = 0.0
sigma_x = 1.0 #0.25

coeff_wf, wf = coh_st_coeff(x0, 0.5, sigma_x, x_arr, vecs, neigs)

tarr = np.arange(0.0,10.01,0.1)

if(0):
    wf_tarr, avg_x2 = wf_t(wf, vecs, vals, tarr, N=10)

    for i in range(5):
        plt.plot(tarr, avg_x2[2*i+1].real, label='⟨x^{}⟩'.format(i+1))
    plt.xlabel('Time')
    plt.ylabel('Expectation Values')
    plt.title('Time Evolution of Position Moments')
    plt.legend()
    plt.show()

    exit()

if(0):
    plt.xlabel('x')
    plt.ylabel('|ψ(x,t)|²')
    plt.xlim([-10,10])
    for i in range(0, len(tarr), 20):
        plt.plot(x_arr, np.abs(wf_tarr[:,i])**2, label='t={}'.format(np.around(tarr[i],2)))
        #plt.scatter(avg_x2[i], 0.0, color='red', marker='x', s=100, label='⟨x²⟩ at t={}'.format(np.around(tarr[i],2)))
        plt.pause(0.25)
        plt.clf()
    plt.title('Time Evolution of Coherent State Wavefunction')
    plt.legend()
    plt.show()

    exit()

p0_arr = np.arange(0.0,5.01,1.0)
sigma_arr = np.arange(1.0,5.01,1.0)
x0_arr = np.arange(Len/2-2,Len/2+2.01,1.0)

slope_arr = []

change_var = 'x0' #'sigma_x'  # 'p0' or 'sigma_x' or 'x0'

if(change_var=='p0'):
    var_arr = p0_arr
elif(change_var=='sigma_x'):
    var_arr = sigma_arr
elif(change_var=='x0'):
    var_arr = x0_arr

for var in var_arr:
    if(change_var=='p0'):
        p0 = var
    elif(change_var=='sigma_x'):
        sigma_x = var
    elif(change_var=='x0'):
        x0 = var

    print('Computing for sigma_x={}, p0={}, x0={}'.format(sigma_x, p0, x0))

    coeff_wf, wf = coh_st_coeff(x0, p0, sigma_x, x_arr, vecs, neigs)
    #plt.plot(coeff_wf[1:].real)

    barr[:] = 0.0
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs_wf(O, L, barr, coeff_wf)
    
    if(change_var=='p0'):
        plt.scatter(np.arange(ncoeff),barr,label='$p_0$={}'.format(p0))
    elif(change_var=='sigma_x'):
        plt.scatter(np.arange(ncoeff),barr,label='$\sigma_x$={}'.format(sigma_x))
    elif(change_var=='x0'):
        plt.scatter(np.arange(ncoeff),barr,label='$x_0$={}'.format(x0))

    #FIT Lanczos coeffs to a line
    p = np.polyfit(np.arange(ncoeff), barr, 1)
    #plt.plot(np.arange(ncoeff), p[0]*np.arange(ncoeff)+p[1], '--')
    print('slope for sigma_x, p0, x0, L={},{},{},{}: {}'.format(sigma_x, p0, x0, Len, p[0]))
    slope_arr.append(p[0])

    #av_x = av_O(O, tarr, coeff_wf, vals)
    #plt.plot(tarr, av_x.real, label='$\sigma_x$={}'.format(sigma_x))
    
    #Ct_arr = corr_func(tarr, vals, O, L, coeff_wf)
    #plt.plot(tarr, Ct_arr.real, label='$p_0$={}'.format(p0))
    #plt.plot(tarr, Ct_arr.real, label='$\sigma_x$={}'.format(sigma_x))

plt.xlabel(r'$n$')
plt.ylabel('$b_n$')
plt.suptitle('Lanczos Coefficients for 1D Box Potential')
plt.legend()

if(change_var=='p0'):
    plt.title('Varying $p_0$, $\sigma_x$={}, $x_0$={}'.format(sigma_x, x0))
    plt.savefig('Lanczos_1DB_vary_p0_sigma_{}_x0_{}.pdf'.format(sigma_x, x0),dpi=300)
elif(change_var=='x0'):
    plt.title('Varying $x_0$, $\sigma_x$={}, $p_0$={}'.format(sigma_x, p0))
    plt.savefig('Lanczos_1DB_vary_x0_sigma_{}_p0_{}.pdf'.format(sigma_x, p0),dpi=300)
elif(change_var=='sigma_x'):
    plt.title('Varying $\sigma_x$, $p_0$={}, $x_0$={}'.format(p0, x0))
    plt.savefig('Lanczos_1DB_vary_sigma_x_p0_{}_x0_{}.pdf'.format(p0, x0),dpi=300)

plt.show()

if(change_var=='p0'):
    plt.plot(p0_arr, slope_arr, 'o-')
    plt.xlabel('$p_0$ of Coherent State')
    plt.title('Slope vs Initial Momentum of Coherent State')
elif(change_var=='x0'):
    plt.plot(x0_arr, slope_arr, 'o-')
    plt.xlabel('$x_0$ of Coherent State')
    plt.title('Slope vs Initial Position of Coherent State')
elif(change_var=='sigma_x'):
    plt.plot(sigma_arr, slope_arr, 'o-')
    plt.xlabel('$\sigma_x$ of Coherent State')
    plt.title('Slope vs Width of Coherent State')


plt.ylabel('Slope of Lanczos Coefficients')
plt.grid()
plt.show()


