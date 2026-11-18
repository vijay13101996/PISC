import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from floquet_tools import Floquetiser, fidelity, propagate_wf_split_op, compute_U_F

"""
Test code for split operator propagation 

Status: Working correctly, at least for a time-independent harmonic oscillator.

Expression for the time-evolved wavefunction obtained from:
Eric. J. Heller, J. Chem. Phys. 62, 1544–1555 (1975)
"""

m = 1.0
L = 20.0
omega = 1.0
ngrid = 2**9 #101

lb = -L
ub = L

def pot(x):
    return 0.5*m*omega**2*x**2

DVR = DVR1D(ngrid,lb,ub,m,pot)
neigs = ngrid-1#100
vals,vecs = DVR.Diagonalize(neig_total=neigs)

def V(x,t):
    return pot(x)

#----------------------------------------------------------------------------------

N = ngrid   
x = np.linspace(-L, L, N, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(N, d=(x[1]-x[0])) 
dx = x[1]-x[0]

x_0 = 0.0
p_0 = 1.0
wf_init = (m*omega/(np.pi))**0.25*np.exp(-m*omega*(x-x_0)**2/2 + 1j*p_0*(x-x_0))
wf_init /= np.sqrt(np.sum(np.abs(wf_init)**2)*dx)

tarr = np.linspace(0,100,10000)
dt = tarr[1]-tarr[0]
xc_t = x_0*np.cos(omega*tarr) + p_0/(m*omega)*np.sin(omega*tarr)
pc_t = p_0*np.cos(omega*tarr) - m*omega*x_0*np.sin(omega*tarr)
gamma_t =(0.25*(p_0**2 / (m * omega) - m * omega * x_0**2) * np.sin(2 * omega * tarr)
+ 0.5 * x_0 * p_0 * (np.cos(2 * omega * tarr) - 1.0) - 0.5 * omega * tarr)

print('xc_t',xc_t[0],'pc_t',pc_t[0],'gamma_t',gamma_t[0])

wf_t = wf_init.copy()
for i, t in enumerate(tarr):
    wf_analytic = (m*omega/(np.pi))**0.25*np.exp(-m*omega*(x-xc_t[i])**2/2 + 1j*pc_t[i]*(x-xc_t[i]) + 1j*gamma_t[i])
    wf_analytic /= np.sqrt(np.sum(np.abs(wf_analytic)**2)*dx)
    
    fidelity_val = np.abs(np.sum(np.conj(wf_analytic)*wf_t)*dx)**2
    print('t={:.2f}, fidelity={:.4f}'.format(t, fidelity_val))
    if i % 100 == 0: 
        plt.plot(x, np.real(wf_t), label='t={:.2f}'.format(t))
        plt.plot(x, np.real(wf_analytic), 'r--', label='Analytic t={:.2f}'.format(t))
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.title('Wavefunction probability density at t={:.2f}'.format(t))
        plt.grid()
        plt.legend()
        plt.show()
    wf_t = propagate_wf_split_op(x, k, wf_t, m, V, t, dt)
