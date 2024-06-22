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
    return -0.5*m*omega**2*x**2 + g*x**4 + m*omega**4/(32*g)
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
psi_x0 = vecs[:,3]
print('len',len(psi_x0),len(x))
psi_x0 /= np.sqrt(simps(np.abs(psi_x0)**2, x))

#Density matrix
T = 0.5*Tc
beta = 1.0/T
rho_x0 = np.outer(psi_x0, np.conj(psi_x0))
rho_x0[:] = 0

for i in range(50):
    rho_x0 += np.outer(vecs[:,i], np.conj(vecs[:,i]))*np.exp(-beta*vals[i])

norm = np.sum(np.diag(rho_x0)*dx)
rho_x0 /= norm

#print('rho',np.sum(np.diag(rho_x0)*dx),norm)
#exit()

#Redefine potential
omega1 = 1.
g1 = g*omega1**2/omega**2
def potential1(x):
    return -0.5*m*omega1**2*x**2 + g1*x**4 + m*omega1**4/(32*g1)
V_x = potential1(x)

print('Tc',0.5*omega/np.pi,'Tc1',0.5*omega1/np.pi,'T',T)

DVR = DVR1D(N+1,-L/2,L/2,m,potential1)
vals1,vecs1 = DVR.Diagonalize(neig_total=N-10)

rho1_x0 = np.zeros((N,N),dtype=complex)
for i in range(30):
    rho1_x0 += np.outer(vecs1[:,i], np.conj(vecs1[:,i]))*np.exp(-beta*vals1[i])

norm = np.sum(np.diag(rho1_x0)*dx)
rho1_x0 /= norm

# Time evolution
dt = 0.01
time_total = 100#0.0*dt
nsteps = int(time_total/dt)

def potentialt(x,t):
    omegat = (omega1-omega)*t/time_total + omega
    gt = (g1-g)*t/time_total + g
    return -0.5*m*omegat**2*x**2 + gt*x**4 + m*omegat**4/(32*gt) 

# Time evolution operator
K = np.exp(-1j*(1/(2*m*hbar))*k**2*dt)
V = np.exp(-0.5j*V_x*dt/hbar)

Vd = np.diag(V)
Kd = np.diag(K)

Vct = np.exp(0.5j*V_x*dt/hbar)
Kct = np.exp(1j*(1/(2*m*hbar))*k**2*dt)

Vdct = np.diag(Vct)
Kdct = np.diag(Kct)

# Time evolution
psi_x = psi_x0
rho_x = rho_x0

fig, ax = plt.subplots(2,1)

def p_redef(t,V_x,Vd,Vdct):
    V_x = potentialt(x,t)
    V[:] = np.exp(-0.5j*V_x*dt/hbar)
    Vd[:] = np.diag(V)
    Vct[:] = np.exp(0.5j*V_x*dt/hbar)
    Vdct[:] = np.diag(Vct)

norm = np.trace(rho_x@rho_x)
print('norm',norm)

for i in range(nsteps):
    t = i*dt
    #p_redef(t,V_x,Vd,Vdct)
    psi_x = psi_x*V
    rho_x = np.matmul(np.matmul(Vd,rho_x),Vdct)

    psi_x = fftshift(fft(psi_x))
    psi_x = psi_x*K
    psi_x = ifft(ifftshift(psi_x))
   
    rho_x = fftshift(fft2(rho_x))
    rho_x = np.matmul(np.matmul(Kd,rho_x),Kdct)
    rho_x = ifft2(ifftshift(rho_x))

    #p_redef(t+dt/2,V_x,Vd,Vdct)
    psi_x = psi_x*V
    rho_x = np.matmul(np.matmul(Vd,rho_x),Vdct)

    xp = p0*np.sin(omega*i*dt)/(m*omega) + x0*np.cos(omega*i*dt)
    argmaxpsi = np.argmax(abs(psi_x)**2)
    #print(x[argmaxpsi], xp)

    #print('t',t*dt)
    if i % 20 == 0:
        # Plot        
        ax[0].plot(x, np.diag(rho1_x0),color='k')
        ax[0].plot(x, np.diag(rho_x0),color='b')
        ax[0].annotate('t = '+str(i*dt), xy=(0.5, 0.5), xycoords='axes fraction')
        ax[0].set_ylim(0,0.5)
        #ax[0].plot(x, np.abs(psi_x)**2,color='b')
        ax[0].plot(x, np.diag(rho_x),color='r')
        #plt.scatter(p0*t*dt/m, max(psi_x)**2, color='r')
        #plt.scatter(t*dt,x[argmaxpsi])
        plt.pause(0.01)
        #ax[0].clear()
    if i % 20 == 0:
        #trace_dist1 = np.sum(np.diag(rho_x)*x**2*dx)
        trace_dist1 = np.linalg.norm(rho_x@rho1_x0,'nuc')/norm  #abs(rho_x-rho1_x0),'nuc')
        trace_dist = np.trace(abs(rho_x-rho_x0))

        ax[1].scatter(i*dt,trace_dist1,color='k')
        #ax[1].scatter(t*dt,trace_dist,color='b')
        #plt.scatter(t*dt,trace_dist,color='b')
        plt.pause(0.01)

ax[0].plot(x, np.diag(rho_x),color='k',lw=3)
plt.show()
