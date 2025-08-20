import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy.fft import fftfreq, fft, fft2, ifft, ifft2, fftshift, ifftshift
from PISC.dvr.dvr import DVR1D

# Constants
hbar = 1.0
m = 0.5
N = 2**8
L = 20.0
dx = L/N
x = np.linspace(-L/2, L/2, N, endpoint=False)
k = 2*np.pi*np.linspace(-N/2, N/2, N, endpoint=False)/L

#Potential
omega = 2.0
g = 0.05
Tc = 0.5*omega/np.pi
def potential(x):
    #if(x < 5 or x > -5):
    #    return 0.0
    #else:
    #    return 1e8
    return x**4   #-0.5*m*omega**2*x**2 + g*x**4 + m*omega**4/(32*g)
    

potential = np.vectorize(potential)
V_x = potential(x)

#Hamiltonian definition
DVR = DVR1D(N+1,-L/2,L/2,m,potential)
vals,vecs = DVR.Diagonalize(neig_total=N-10) 

# Initial wavefunction
sigma = 1.0
x0 = 0.0
p0 = 1.0

#Coherent state
psi_x0 = (m*omega/(np.pi*hbar))**0.25*np.exp(-m*omega*(x-x0)**2/(2*hbar) + 1j*p0*(x-x0)/hbar)
psi_x0 /= np.sqrt(simps(np.abs(psi_x0)**2, x))

# Time evolution
dt = 0.01
time_total = 100#0.0*dt
nsteps = int(time_total/dt)

# Time evolution operator
K = np.exp(-1j*(1/(2*m*hbar))*k**2*dt)
V = np.exp(-0.5j*V_x*dt/hbar)

# Time evolution
psi_x = psi_x0

fig, ax = plt.subplots(2,1)

ax[0].plot(x, np.abs(psi_x)**2, color='r')

for i in range(nsteps):
    t = i*dt
    psi_x = psi_x*V

    psi_x = fftshift(fft(psi_x))
    psi_x = psi_x*K
    psi_x = ifft(ifftshift(psi_x))
   
    psi_x = psi_x*V

    xp = p0*np.sin(omega*i*dt)/(m*omega) + x0*np.cos(omega*i*dt)
    argmaxpsi = np.argmax(abs(psi_x)**2)
    #print(x[argmaxpsi], xp)

    #print('t',t*dt)
    if i % 10 == 0:
        # Plot  
        ax[0].plot(x, np.abs(psi_x)**2,color='b')
        #ax.scatter(p0*t*dt/m, max(psi_x)**2, color='r')
        #plt.scatter(t*dt,x[argmaxpsi])
        plt.pause(0.01)
        ax[0].clear()
    if 0: #i % 20 == 0:
        #trace_dist1 = np.sum(np.diag(rho_x)*x**2*dx)
        trace_dist1 = np.linalg.norm(rho_x@rho1_x0,'nuc')/norm  #abs(rho_x-rho1_x0),'nuc')
        trace_dist = np.trace(abs(rho_x-rho_x0))

        ax[1].scatter(i*dt,trace_dist1,color='k')
        #ax[1].scatter(t*dt,trace_dist,color='b')
        #plt.scatter(t*dt,trace_dist,color='b')
        plt.pause(0.01)
    if 0:#i % 10 == 0:
        # Plot overlap with psi_x0
        overlap = np.abs(simps(np.conj(psi_x0)*psi_x, x))**2
        ax[1].scatter(i*dt, overlap, color='g')
    if i % 10 == 0:
        #Project psi_x onto the basis of vecs
        psi_x_proj = np.dot(vecs.T, psi_x)
        ax[1].plot(np.arange(len(vals)), psi_x_proj)
        #ax[1].set_ylim([-1, 1])
        #plt.pause(0.01)
        #ax[1].clear()
plt.show()
