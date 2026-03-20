import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve, least_squares
from PISC.dvr.dvr import DVR1D
import matplotlib
from FK_variational import opt_FK 

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 1.5

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
LHc = 'darkblue'
FKc = 'c' 
FHc = 'darkmagenta'

#Fontsize
xl_fs = 12
yl_fs = 12
title_fs = 14
le_fs = 10
ti_fs = 12

tol = 1e-4
#Define Morse potential and its derivatives
def V0(x, D, alpha, x0):
    xd=x-x0
    eax = np.exp(-alpha*(xd))
    return D*(1-eax)**2 

def V1(x, D, alpha, x0):
    xd=x-x0
    eax=np.exp(-alpha*(xd))
    return 2*D*alpha*(1 - eax)*np.exp(-alpha*xd)

def V2(x, D, alpha, x0):
    xd=x-x0
    return alpha**2*(-2*D*(1 - np.exp(-alpha*xd))*np.exp(-alpha*xd) + 2*D*np.exp(-2*alpha*xd))

V = lambda x: V0(x, D, alpha, x0)
dV_dx = lambda x: V1(x, D, alpha, x0)
d2V_dx2 = lambda x: V2(x, D, alpha, x0)

# Local-harmonic effective classical potential for the quartic potential
def V_LH_tilde(x, beta):
    k = dV_dx(x)/m
    w2 = max(d2V_dx2(x)/m, w*tol)  # Prevent division by zero or negative sqrt

    eta = beta*hbar*np.sqrt(w2)/2

    Vc = V(x)

    pref = m*k**2/(2*w2)
    Vnc1 = pref*( np.tanh(eta)/eta - 1 )
    Vnc2 = 0.5/beta * np.log( np.sinh(2*eta)/(2*eta) )
    Vnc = Vnc1 + Vnc2

    return Vc + Vnc

def V_LH_magic(x, beta):
    w2 = max(d2V_dx2(x)/m, w*tol)  # Prevent division by zero or negative sqrt
    eta = beta*hbar*np.sqrt(w2)/2
    if eta <= 1e-6:
        eta = 1e-6
    
    Vc = V(x)

    Xi = np.tanh(eta)/eta
    return Vc*Xi - 0.5/beta * np.log(Xi)

def V_LH_renorm(x, beta):
    k = dV_dx(x)/m
    w2 = max(d2V_dx2(x)/m, w*tol)  # Prevent division by zero or negative sqrt

    eta = beta*hbar*np.sqrt(w2)/2

    Vc = V(x)

    pref = m*k**2/(2*w2)
    Vnc1 = pref*( np.tanh(eta)/eta - 1 )
    Vnc2 = -0.5/beta * np.log( np.tanh(eta)/eta )
    Vnc = Vnc1 + Vnc2

    return Vc + Vnc

def V_q(beta):
    ngrid = 500
    lb = 0
    ub = L
    DVR = DVR1D(ngrid,lb,ub,m,V)
    neigs = 100
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    Zq = np.sum( np.exp(-beta*vals) )
    rhoq = np.sum( vecs**2 * np.exp(-beta*vals), axis=1 ) #/ Zq
    xarr = DVR.grid[1:ngrid] 
    Veff = -1/beta*np.log(rhoq) + 0.5*np.log(m/(2*np.pi*beta*hbar**2))/beta
    return xarr,Veff

# Classical probability distribution
def P_cl(V, beta, xarr):
    dx = xarr[1] - xarr[0]
    Vc = V(xarr)
    P = np.exp(-beta*Vc)
    P /= np.sum(P)*dx
    print('Integral P dx=', np.sum(P)*dx)
    return P

def P_LH_tilde(x, beta):
    pref = np.sqrt( m/(2*np.pi*beta*hbar**2) )
    Vc = V(x)
    Veff = np.zeros_like(x)
    for i,xi in enumerate(x):
         Veff[i] = V_LH_tilde(xi, beta)

    P = pref * np.exp(-beta*Veff)
    dx = x[1] - x[0]
    P /= np.sum(P)*dx
    return P

def P_q(x, beta):
    ngrid = 500
    lb = 0
    ub = L
    DVR = DVR1D(ngrid,lb,ub,m,V)
    neigs = 100
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    Zq = np.sum( np.exp(-beta*vals) )
    Pq = np.sum( vecs**2 * np.exp(-beta*vals), axis=1 ) / Zq
    xarr = DVR.grid[1:ngrid] 
    return xarr,Pq

def P_LH_magic(x, beta):
    pref = np.sqrt( m/(2*np.pi*beta*hbar**2) )
    Vc = V(x)
    w2 = np.zeros_like(x)
    for i,xi in enumerate(x):
        w2[i] = max(d2V_dx2(xi)/m, w*tol)  # Prevent division by zero or negative sqrt

    eta = beta*hbar*np.sqrt(w2)/2
    Xi = np.tanh(eta)/eta

    P = pref * np.exp(-beta*Vc*Xi) * np.sqrt(Xi)
    dx = x[1] - x[0]
    P /= np.sum(P)*dx
    return P

def P_LH_renorm(x, beta):
    pref = np.sqrt( m/(2*np.pi*beta*hbar**2) )
    Vc = V(x)

    k = dV_dx(x)/m
    w2 = np.zeros_like(x)
    for i,xi in enumerate(x):
        w2[i] = max(d2V_dx2(xi)/m, w*tol)  # Prevent division by zero or negative sqrt
    
    eta = beta*hbar*np.sqrt(w2)/2
    Xi = np.tanh(eta)/eta
    sinh_term = np.sinh(2*eta)/(2*eta)

    Vnc1_pref = m*k**2/(2*w2)

    ZQM = 1/(2*np.sinh(eta))
    Zcl = 1/(2*eta)
    ZNCF = ZQM / Zcl
 
    P = pref * np.sqrt(1/sinh_term) * np.exp(-beta*(Vc - Vnc1_pref) - beta*Vnc1_pref*Xi)

    P /= ZNCF 

    dx = x[1] - x[0]
    P /= np.sum(P)*dx
    print('Integral P dx=', np.sum(P)*dx)
    return P

fig,ax = plt.subplots(3,2, figsize=(7,9))
plt.subplots_adjust(wspace=0.3, hspace=0.13)

K2au = 315775.13
Tarr = [5000, 1000, 300]
beta_arr = [1/T*K2au for T in Tarr]

axe1 = ax[0,:]
axe2 = ax[1,:]
axe3 = ax[2,:]

for ax, beta in zip([axe1, axe2, axe3], beta_arr):
    print('Beta = ', beta)
    Vcl = np.zeros_like(xarr)
    VLH_tilde = np.zeros_like(xarr)
    VLH_renorm = np.zeros_like(xarr)
    VLH_magic = np.zeros_like(xarr)
    VQM = np.zeros_like(xarr)

    Pcl = np.zeros_like(xarr)
    PLH_tilde = np.zeros_like(xarr)
    PLH_renorm = np.zeros_like(xarr)
    PLH_magic = np.zeros_like(xarr)
    PQM = np.zeros_like(xarr)

    Pcl1 = np.zeros_like(xarr)
    PLH_tilde1 = np.zeros_like(xarr)
    PLH_renorm1 = np.zeros_like(xarr)
    PLH_magic1 = np.zeros_like(xarr)
    PQM1 = np.zeros_like(xarr)

    Vcl = V(xarr)
    Pcl = P_cl(V, beta, xarr)
    Pcl1 = np.exp(-beta*Vcl)

    VQM = V_q(beta)[1]
    xq_arr, PQM = P_q(xarr, beta)
    PQM1 = np.exp(-beta*VQM)

    for i,x in enumerate(xarr):
        VLH_tilde[i] = V_LH_tilde(x, beta)
        VLH_renorm[i] = V_LH_renorm(x, beta)
        VLH_magic[i] = V_LH_magic(x, beta)

        PLH_tilde1[i] = np.exp(-beta*VLH_tilde[i])
        PLH_renorm1[i] = np.exp(-beta*VLH_renorm[i])
        PLH_magic1[i] = np.exp(-beta*VLH_magic[i])
        

    Pcl = P_cl(V, beta, xarr)
    PLH_tilde = P_LH_tilde(xarr, beta)
    PLH_renorm = P_LH_renorm(xarr, beta)
    PLH_magic = P_LH_magic(xarr, beta)
    PQM = P_q(xarr, beta)[1]


    ax[0].plot(xarr, Vcl, label=r'$V_{cl}(x)$', color=Cc)
    ax[0].plot(xarr, VLH_tilde, label=r'$\tilde{V}_{LH}(x)$', color=LHc, linestyle='--')
    ax[0].plot(xarr, VLH_renorm, label=r'$V_{LH}^{(r)}(x)$', color=LHc, linestyle=':',lw=2.)
    ax[0].plot(xarr, VLH_magic, label=r'$V_{LH}(x)$', color=LHc, linestyle='-')
    ax[0].plot(xq_arr, VQM, label=r'$V_{QM}(x)$', color=Qc, linestyle='-')

    if(beta==beta_arr[2]):
        ax[0].set_xlabel(r'$x$', fontsize=xl_fs)
    ax[0].set_ylabel(r'$V(x)$', fontsize=yl_fs)
    ax[0].set_xlim([1,4])
    ax[0].set_ylim([-0.015,0.15])
   
    T = Tarr[ beta_arr.index(beta) ]
    ax[0].annotate(r'$T\;=\;\;{}$K'.format(T), xy=(0.32,0.85), xycoords='axes fraction', fontsize=ti_fs)
    if(beta==beta_arr[0]):
        ax[0].legend(fontsize=le_fs-1.5, loc=(0.02,1.02), ncol=3, columnspacing=0.5)

    ax[1].plot(xarr, Pcl, label=r'$P_{cl}(x)$', color=Cc)
    ax[1].plot(xarr, PLH_tilde, label=r'$\tilde{P}_{LH}(x)$', color=LHc, linestyle='--')
    ax[1].plot(xarr, PLH_renorm, label=r'$P_{LH}^{(r)}(x)$', color=LHc, linestyle=':',lw=2.)
    ax[1].plot(xarr, PLH_magic, label=r'$P_{LH}(x)$', color=LHc, linestyle='-')
    ax[1].plot(xq_arr, PQM, label=r'$P_{QM}(x)$', color=Qc, linestyle='-')

    if(beta==beta_arr[2]):
        ax[1].set_xlabel(r'$x$', fontsize=xl_fs)
    ax[1].set_ylabel(r'$P(x)$', fontsize=yl_fs)
    ax[1].set_xlim([1,4])
    ax[1].set_ylim([0,1.2*max(PQM)])
    ax[1].annotate(r'$T\;=\;\;{}$K'.format(T), xy=(0.33,0.85), xycoords='axes fraction', fontsize=ti_fs)
    
    if(beta==beta_arr[0]):
        ax[1].legend(fontsize=le_fs-1.5, loc=(0.02,1.02), ncol=3, columnspacing=0.5)


    if(0):
        ax[1].plot(xarr, Pcl1/np.sum(Pcl1*dx), label=r'$P_{cl}(x)$ (uncorrected)', color=Cc, linestyle='--', alpha=0.5)
        #ax[1].plot(xarr, PLH_tilde1/np.sum(PLH_tilde1*dx), label=r'$\tilde{P}_{LH}(x)$ (uncorrected)', color=LHc, linestyle='--', alpha=0.5)
        ax[1].plot(xarr, PLH_renorm1/np.sum(PLH_renorm1*dx), label=r'$P_{LH}^{(r)}(x)$ (uncorrected)', color=LHc, linestyle=':', alpha=0.5)
        #ax[1].plot(xarr, PLH_magic1/np.sum(PLH_magic1*dx), label=r'$P_{LH}(x)$ (uncorrected)', color=LHc, linestyle='-', alpha=0.5)
        #ax[1].plot(xq_arr, PQM1/np.sum(PQM1*dx), label=r'$P_{QM}(x)$ (uncorrected)', color=Qc, linestyle='-', alpha=0.5)

fig.suptitle(r'$V(x) = D(1-e^{-\alpha(x-x_0)})^2$')

plt.savefig('compare_morse_Veff.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.show()
