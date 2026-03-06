import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve, least_squares
from PISC.dvr.dvr import DVR1D
import matplotlib
from FK_variational import opt_FK, V_smear

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

hbar = 1.0  # Planck's constant (reduced)
L = 10.0

ngrid = 1000
#Testing the FK potential
xarr = np.linspace(0.0, L, ngrid)
dx = xarr[1] - xarr[0]

m = 1741.1
cm1toau = 219474.63
we = 3737.76/cm1toau
xe = 84.881/(cm1toau*we)
req = 1.832

alpha = np.sqrt(2*m*we*xe)
D = we/(4*xe)
w = np.sqrt(2*D*alpha**2/m)
x0 = req


#Colors
Qc = 'r'
Cc = 'g'
LHc = 'k'
FKc = 'c'

#Fontsize
xl_fs = 12
yl_fs = 12
title_fs = 14
le_fs = 10
ti_fs = 12

tol = 1e-4
#Define Morse potential and its derivatives
def V(x, D, alpha, x0):
    xd=x-x0
    eax = np.exp(-alpha*(xd))
    return D*(1-eax)**2 

def dV_dx(x, D, alpha, x0):
    xd=x-x0
    eax=np.exp(-alpha*(xd))
    return 2*D*alpha*(1 - eax)*np.exp(-alpha*xd)

def d2V_dx2(x, D, alpha, x0):
    xd=x-x0
    return alpha**2*(-2*D*(1 - np.exp(-alpha*xd))*np.exp(-alpha*xd) + 2*D*np.exp(-2*alpha*xd))

d2V = lambda x: d2V_dx2(x, D, alpha, x0)
d0V = lambda x: V(x, D, alpha, x0)

# Feynman-Kleinert effective classical potential
def V_FK(x, d2V, beta):
    omega_init = max(1e-4,d2V(x)/m)**0.5
    omega, a2 = opt_FK(omega_init,beta, m, x, d2V, a=req, b=req+1000)
    Vsmear = V_smear(x, a2, d0V)

    eta = beta*hbar*omega/2
    Vtemp = np.log( np.sinh(eta)/eta ) / beta - 0.5*m*omega**2*a2

    return Vsmear + Vtemp

def V_FH(x, d0V, beta):
    return V_smear(x, beta*hbar**2/(12*m), d0V)

# Local-harmonic effective classical potential for the quartic potential
def V_LH(x, D, alpha, x0, beta):
    k = dV_dx(x, D, alpha, x0)/m
    w2 = max(d2V_dx2(x, D, alpha, x0)/m, w*tol)  # Prevent division by zero or negative sqrt

    eta = beta*hbar*np.sqrt(w2)/2

    Vc = V(x, D, alpha, x0)

    pref = m*k**2/(2*w2)
    Vnc1 = pref*( np.tanh(eta)/eta - 1 )
    Vnc2 = 0.5/beta * np.log( np.sinh(2*eta)/(2*eta) )
    Vnc = Vnc1 + Vnc2

    return Vc + Vnc

def V_nc1(x, D, alpha, x0, beta):
    k = dV_dx(x, D, alpha, x0)/m
    w2 = max(d2V_dx2(x, D, alpha, x0)/m, w*tol)  # Prevent division by zero or negative sqrt

    eta = beta*hbar*np.sqrt(w2)/2

    if eta <= 1e-6:
        eta = 1e-6
    
    pref = m*k**2/(2*w2)
    Vnc1 = pref*( np.tanh(eta)/eta - 1 )
    return Vnc1

def V_nc2(x, D, alpha, x0, beta):
    w2 = max(d2V_dx2(x, D, alpha, x0)/m, w*tol)  # Prevent division by zero or negative sqrt

    eta = beta*hbar*np.sqrt(w2)/2

    if eta <= 1e-6:
        eta = 1e-6
    
    Vnc2 = 0.5/beta * np.log( np.sinh(2*eta)/(2*eta) )
    return Vnc2


def Z(V, beta, xarr):
    dx = xarr[1] - xarr[0]
    return np.sum( np.exp(-beta*V) ) * dx * np.sqrt(m/(2*np.pi*beta*hbar**2))

def F(Z):
    return -1/beta * np.log(Z)

def P(V, beta, xarr):
    dx = xarr[1] - xarr[0]
    P = np.exp(-beta*V)
    #P /= np.sum(P)*dx
    print('Integral P dx=', np.sum(P)*dx)
    return P

def Fq(x, D, alpha, x0, beta):
    ngrid = 500
    lb = -L
    ub = L
    Vq = lambda x: V(x, D, alpha, x0)
    DVR = DVR1D(ngrid,lb,ub,m,Vq)
    neigs = 100
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    Zq = np.sum( np.exp(-beta*vals) )
    Fq = -1/beta * np.log(Zq)
    return Fq

def Pq(x, D, alpha, x0, beta):
    lb = 0
    ub = L
    Vq = lambda x: V(x, D, alpha, x0)
    DVR = DVR1D(ngrid,lb,ub,m,Vq)
    neigs = 100
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    Zq = np.sum( np.exp(-beta*vals) )
    Pq = np.sum( vecs**2 * np.exp(-beta*vals), axis=1 ) / Zq
    xarr = DVR.grid[1:ngrid] 
    
    if Zq<1e-12:
        print("Warning: Partition function is very small, results may be inaccurate.")
        Pq = abs(vecs[:,0])**2
    return xarr,Pq

def V_LH_magic(x, D, alpha, x0, beta):
    w2 = d2V_dx2(x, D, alpha, x0)/m
    
    if(w2<1e-12): # Prevent division by zero or negative values
        w2 = 1e-12

    eta = beta*hbar*np.sqrt(w2)/2
    Xi = np.tanh(eta)/eta

    Vc = V(x, D, alpha, x0)
    Vlh = Vc*Xi - 0.5/beta * np.log(Xi)
    return Vlh

def P_LH_magic(x, D, alpha, x0, beta):
    pref = 1.0 #np.sqrt( m/(2*np.pi*beta*hbar**2) )
    Vc = V(x, D, alpha, x0)
    w2 = d2V_dx2(x, D, alpha, x0)/m
    
    w2[w2<1e-12] = 1e-12  # Prevent division by zero or negative values

    eta = beta*hbar*np.sqrt(w2)/2
    Xi = np.tanh(eta)/eta

    P = pref * np.exp(-beta*Vc*Xi) * np.sqrt(Xi)
    dx = x[1] - x[0]
    #P /= np.sum(P)*dx
    return P


T = 100.0 
K2au = 315775.13
beta = 1/T*K2au

Vcl = V(xarr, D, alpha, x0)
Pcl = P(Vcl, beta, xarr)

VLH = np.zeros_like(xarr)
VLH_nc1 = np.zeros_like(xarr)
VLH_nc2 = np.zeros_like(xarr)
VLH_magic = np.zeros_like(xarr)

for i,x in enumerate(xarr):
    VLH_nc1[i] = V_nc1(x, D, alpha, x0, beta)
    VLH_nc2[i] = V_nc2(x, D, alpha, x0, beta)
    VLH[i] = V_LH(x, D, alpha, x0, beta)
    VLH_magic[i] = V_LH_magic(x, D, alpha, x0, beta)

PLH = P(VLH, beta, xarr)
print('LH integral P dx=', np.sum(PLH)*dx)

PLH_nc1 = P(VLH_nc1, beta, xarr)
PLH_nc2 = P(VLH_nc2, beta, xarr)
PLH_magic = P(VLH_magic, beta, xarr)
PLH_magic_dir = P_LH_magic(xarr, D, alpha, x0, beta)




xqarr, Pq_arr = Pq(xarr, D, alpha, x0, beta)

fig, ax  = plt.subplots(2,1, figsize=(5,10))
ax0 = ax[1]
ax1 = ax[0]

if(1):
    
    ax0.plot(xarr, Pcl, color=Cc, linestyle='-', label=r'$P_{cl}$', linewidth=1.5)    
    ax0.plot(xarr, PLH, color=LHc, linestyle='--', label=r'$\tilde{P}_{LH}$', linewidth=1.5)
    ax0.plot(xarr, PLH_nc1, color='orange', linestyle=':', label=r'$\tilde{P}_{LH,nc1}$', linewidth=2)
    ax0.plot(xarr, PLH_nc2, color='c', linestyle=':', label=r'$\tilde{P}_{LH,nc2}$', linewidth=2.)
    ax0.plot(xarr, PLH_nc1*Pcl, color='brown', linestyle=':', label=r'$\tilde{P}_{LH,nc1} \times P_{cl}$', linewidth=2)
    #ax0.plot(xarr, Pcl*PLH_nc1*PLH_nc2, color='m', linestyle=':', label=r'$P_{cl}*P_{LH\_nc1}*P_{LH\_nc2}$', linewidth=2.5)
    #ax0.plot(xqarr, Pq_arr, color=Qc, linestyle='-', label=r'$P_{QM}$', linewidth=1.5)
    ax0.plot(xarr, PLH_magic, color=LHc, linestyle='-', label=r'$P_{LH}$', linewidth=1.5)
    #ax0.plot(xarr, PLH_magic_dir, color='red', linestyle=':', label=r'$P_{LH,magic\_dir}$', linewidth=1.5)

    ax0.set_xlabel(r'$x$', fontsize=xl_fs)
    ax0.set_ylabel(r'$\rho(x) = e^{-\beta V_{eff}(x)}$', fontsize=yl_fs)
    ax0.set_title('Density matrix Comparison', fontsize=title_fs)
    ax0.legend(fontsize=le_fs)
    ax0.grid()
    ax0.set_ylim([0.0,2.0])
    ax0.set_xlim([0.0,5.0])
    #plt.show()

if(1):
    ax1.plot(xarr, Vcl, color=Cc, linestyle='-', label=r'$V_{cl}$', linewidth=1.5)    
    ax1.plot(xarr, VLH, color=LHc, linestyle='--', label=r'$\tilde{V}_{LH}$', linewidth=1.5)
    ax1.plot(xarr, VLH_nc1, color='orange', linestyle=':', label=r'$\tilde{V}_{LH, nc1}$', linewidth=2)
    ax1.plot(xarr, VLH_nc2, color='c', linestyle=':', label=r'$\tilde{V}_{LH, nc2}$', linewidth=2.) 
    ax1.plot(xarr, VLH_nc1 + Vcl, color='brown', linestyle=':', label=r'$\tilde{V}_{LH, nc1}+V_{cl}$', linewidth=2)
    #ax1.plot(xarr, Vcl + VLH_nc1 + VLH_nc2, color='m', linestyle=':', label=r'$V_{cl}+V_{LH\_nc1}+V_{LH\_nc2}$', linewidth=2.5)
    ax1.plot(xarr, VLH_magic, color=LHc, linestyle='-', label=r'$V_{LH}$', linewidth=1.5)

    ax1.set_xlabel(r'$x$', fontsize=xl_fs)
    ax1.set_ylabel(r'$V_{eff}(x)$', fontsize=yl_fs)
    ax1.set_title('Effective Potential Comparison', fontsize=title_fs)
    ax1.legend(fontsize=le_fs)
    ax1.grid()
    ax1.set_ylim([-1.3,1.3])
    ax1.set_xlim([0.0,5.0])

plt.suptitle(r'$V(x) = D(1-e^{{-\alpha(x-x_0)}})^2$, T={} K'.format(T), fontsize=ti_fs+2)
plt.tight_layout()
plt.savefig('compare_Veff_morse.pdf', dpi=300)
plt.show()
  
