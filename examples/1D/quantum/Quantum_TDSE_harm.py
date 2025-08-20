import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy.fft import fftfreq, fft, fft2, ifft, ifft2, fftshift, ifftshift
from PISC.dvr.dvr import DVR1D
from PISC.potentials import mildly_anharmonic



# Constants
hbar = 1.0
m = 0.5
N = 2**8
L = 24.0
lb = -L/2
ub = L/2
dx = L/N
x = np.linspace(-L/2, L/2, N, endpoint=False)
k = 2*np.pi*np.linspace(-N/2, N/2, N, endpoint=False)/L
dt = 0.01
nsteps = 1000

#Potential
m=1.0
a=0.0
b=1.0
omega=0.0
n_anharm=4
ngrid=N

pes = mildly_anharmonic(m,a,b,omega,n=n_anharm)
potkey = 'MAH_w_{}_a_{}_b_{}_n_{}'.format(omega,a,b,n_anharm)
DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 150
vals,vecs = DVR.Diagonalize(neig_total=neigs)
V_x = pes.potential_func(x)


if(1):# Initial wavefunction
    a0 = 1.0
    a1 = 1.
    beta = 0.5
    an_arr = np.exp(-beta*vals[0:10])
    
    psi0 = np.sum(an_arr[0:10]*vecs[:,0:10],axis=1)

    #psi0 = a0*vecs[:,0] + a1*vecs[:,1]
    # Make psi0 of size N
    psi0 = np.concatenate((psi0, np.zeros(N-len(psi0))))
else:
    psi0 = np.exp(-0.5*x**2)

# Normalize psi0
psi0 = psi0/np.sqrt(np.sum(dx*abs(psi0)**2))
print('norm:',np.sum(dx*abs(psi0)**2))

# Decompose psi0 in the DVR basis
vecs_aux = np.zeros((N,neigs),dtype=complex)
vecs_aux[0:N-1,:] = vecs

def find_coeffs(vecs_aux,psi):
    return np.dot(vecs_aux.T, psi)/np.dot(psi.conj(),psi)

coeffs = find_coeffs(vecs_aux,psi0)
print('coeffs:',np.shape(coeffs), np.sum(abs(coeffs)**2))


#plt.plot(x, np.abs(psi0)**2)
#plt.show()

#exit()

# Time evolution operator
K = np.exp(-1j*(1/(2*m*hbar))*k**2*dt)
V = np.exp(-0.5j*V_x*dt/hbar)

KIm = np.exp(-(1/(2*m*hbar))*k**2*dt)
VIm = np.exp(-0.5*V_x*dt/hbar)

#Vd = np.diag(V)
#Kd = np.diag(K)

#Vct = np.exp(0.5j*V_x*dt/hbar)
#Kct = np.exp(1j*(1/(2*m*hbar))*k**2*dt)

#Vdct = np.diag(Vct)
#Kdct = np.diag(Kct)

# Time evolution
psi_x = psi0


def p_redef(t,V_x,Vd,Vdct):
    V_x = potentialt(x,t)
    V[:] = np.exp(-0.5j*V_x*dt/hbar)
    Vd[:] = np.diag(V)
    Vct[:] = np.exp(0.5j*V_x*dt/hbar)
    Vdct[:] = np.diag(Vct)


fig, ax = plt.subplots(2,1)
ax[0].set_xlim([-L/4,L/4])

for i in range(nsteps):
    t = i*dt
    psi_x = psi_x*V#

    psi_x = fftshift(fft(psi_x))
    psi_x = psi_x*K#
    psi_x = ifft(ifftshift(psi_x))
   
    psi_x = psi_x*V#

    coeffs_t = find_coeffs(vecs_aux,psi_x)

    if(i % 100 == 0):
        ax[0].plot(x, np.abs(psi_x)**2)
        ax[1].plot(abs(coeffs_t)[:20]**2,'o')
        plt.pause(0.01)

plt.show()
