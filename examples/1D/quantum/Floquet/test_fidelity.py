import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
from floquet_tools import Floquetiser, fidelity, propagate_wf_split_op, compute_U_F

"""
Test code for fidelity calculation using split operator propagation.
Status: Working correctly, at least for a time-dependent harmonic oscillator with a sinusoidal driving term.

Reference for the analytic expression of the fidelity obtained using Claude. 

"""


m = 1.0
L = 20.0
omega = 1.0
ngrid = 2**9 

lb = -L
ub = L

def pot(x):
    return 0.5*m*omega**2*x**2

DVR = DVR1D(ngrid,lb,ub,m,pot)
neigs = ngrid-1#100
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[:ngrid]
dx = x_arr[1]-x_arr[0]

#----------------------------------------------------------------------------------

w_F = 0.2 
n_ph = 5
S = 1.0
S_opt = 0.5

def V(x,t):
    return pot(x) - x*S_opt*np.sin(w_F*t)

#----------------------------------------------------------------------------------

N = ngrid   
x = np.linspace(-L, L, N, endpoint=False)

#Interpolate the ground state wavefunction to the new grid
wf_init = np.interp(x, x_arr[:ngrid-1], vecs[:,0])

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
tarr = np.arange(0, 3*T, T/2000)
fidelity_arr = fidelity(x,wf_init, m, V, tarr)

x_c = S_opt/(m*(omega**2 - w_F**2))*(np.sin(w_F*tarr) - w_F/omega*np.sin(omega*tarr))
p_c = S_opt*w_F/(omega**2 - w_F**2)*(np.cos(w_F*tarr) - np.cos(omega*tarr))
fidelity_analytic = np.exp(-0.5*m*omega*x_c**2 - 0.5*p_c**2/(m*omega))

plt.plot(tarr, fidelity_arr)
plt.plot(tarr, fidelity_analytic, 'r--', label='Analytic')
plt.xlabel('Time')
plt.ylabel('Fidelity')
plt.title('Fidelity vs Time for S_opt = {}'.format(S_opt))
plt.grid()
plt.show()
