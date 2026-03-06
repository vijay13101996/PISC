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
m = 1.0     # Mass of the particle
L = 10.0

#Testing the FK potential
xarr = np.linspace(-L, L, 1000)
dx = xarr[1] - xarr[0]
w = 0.0
g = 1.0
b = g/4#1./4

#Colors
Qc = 'r'
Cc = 'g'
LHc = 'k'
FKc = 'c' 
FHc = 'darkmagenta'


#Fontsize
xl_fs = 12
yl_fs = 12
title_fs = 14
le_fs = 10
ti_fs = 12

#Define quartic potential and its derivatives
def V(x, w, b):    
    return 0.5*m*w**2*x**2 + b*x**4

def dV_dx(x, w, b):
    return m*w**2*x + 4*b*x**3

def d2V_dx2(x, w, b):
    return m*w**2 + 12*b*x**2

#Define the Feynman-Hibbs effective potential for the quartic potential
def V_FH(x, w, b, beta):
    quad = 0.5*m*w**2*(x**2 + beta*hbar**2/(12*m))
    quart = b*(x**4 + beta*hbar**2*x**2/(2*m) + beta**2*hbar**4/(48*m**2))
    return quad + quart

#Define Feynman-Kleinert effective potential for the quartic potential
def V_FK(x, w, b, a2, omega, beta):
    quad = 0.5*m*w**2*(x**2 + a2)
    quart = b*(x**4 + 6*a2*x**2 + 3*a2**2)
    Vsmear = quad + quart

    eta = beta*hbar*omega/2
    Vtemp = np.log( np.sinh(eta)/eta ) / beta - 0.5*m*omega**2*a2

    return Vsmear + Vtemp

def equations(p, x, w, b, beta):
    a2, omega = p

    eta = beta*hbar*omega/2
    
    eq1 = ( eta/np.tanh(eta) - 1 )/(beta*m*omega**2) - a2
    eq2 = m*w**2 + 12*b*(x**2 + a2) - m*omega**2
    
    #print('b', b, 'a2', a2, 'omega', omega, 'eq1', eq1, 'eq2', eq2)

    return (eq1, eq2) 
   
#Variationally optimized Feynman-Kleinert effective potential for the quartic potential
def V_FK_var(x, w, b, beta):
    a2_init = beta*hbar**2/(12*m)
    omega_init = 1.0 #w

    #p = fsolve(equations, (a2_init, omega_init), args=(x, w, b, beta))
    p = least_squares(equations, (a2_init, omega_init), args=(x, w, b, beta), bounds=(0,np.inf)).x  
    a2, omega = p

    #print('a2', a2, 'omega', omega, 'x', x, 'b', b)
    return V_FK(x, w, b, a2, omega, beta)


# Local-harmonic effective classical potential for the quartic potential
def V_LH(x, w, b, beta):
    k = dV_dx(x, w, b)/m
    w2 = d2V_dx2(x, w, b)/m

    eta = beta*hbar*np.sqrt(w2)/2

    Vc = V(x, w, b)

    pref = m*k**2/(2*w2)
    Vnc1 = pref*( np.tanh(eta)/eta - 1 )
    Vnc2 = 0.5/beta * np.log( np.sinh(2*eta)/(2*eta) )
    Vnc = Vnc1 + Vnc2

    return Vc + Vnc

def V_nc1(x, w, b, beta):
    k = dV_dx(x, w, b)/m
    w2 = d2V_dx2(x, w, b)/m

    eta = beta*hbar*np.sqrt(w2)/2

    if eta <= 1e-6:
        eta = 1e-6
    
    pref = m*k**2/(2*w2)
    Vnc1 = pref*( np.tanh(eta)/eta - 1 )
    return Vnc1

def V_nc2(x, w, b, beta):
    w2 = d2V_dx2(x, w, b)/m

    eta = beta*hbar*np.sqrt(w2)/2

    if eta <= 1e-6:
        eta = 1e-6
    
    Vnc2 = 0.5/beta * np.log( np.sinh(2*eta)/(2*eta) )
    return Vnc2


def V_LH_magic(x, w, b, beta):
    w2 = d2V_dx2(x, w, b)/m
    eta = beta*hbar*np.sqrt(w2)/2
    if eta <= 1e-6:
        eta = 1e-6
    
    Vc = V(x, w, b)

    Xi = np.tanh(eta)/eta
    return Vc*Xi - 0.5/beta * np.log(Xi)

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

def Fq(x, w, b, beta):
    ngrid = 500
    lb = -L
    ub = L
    Vq = lambda x: V(x, w, b)
    DVR = DVR1D(ngrid,lb,ub,m,Vq)
    neigs = 100
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    Zq = np.sum( np.exp(-beta*vals) )
    Fq = -1/beta * np.log(Zq)
    return Fq

def Pq(x, w, b, beta):
    ngrid = 500
    lb = -L
    ub = L
    Vq = lambda x: V(x, w, b)
    DVR = DVR1D(ngrid,lb,ub,m,Vq)
    neigs = 100
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    Zq = np.sum( np.exp(-beta*vals) )
    Pq = np.sum( vecs**2 * np.exp(-beta*vals), axis=1 ) / Zq
    xarr = DVR.grid[1:ngrid] 
    return xarr,Pq

def V_LH_magic(x, w, b, beta):
    w2 = d2V_dx2(x, w, b)/m
    eta = beta*hbar*np.sqrt(w2)/2
    if eta <= 1e-6:
        eta = 1e-6
    
    Vc = V(x, w, b)

    Xi = np.tanh(eta)/eta
    return Vc*Xi - 0.5/beta * np.log(Xi)

def P_LH_magic(x, w, b, beta):
    pref = np.sqrt( m/(2*np.pi*beta*hbar**2) )
    Vc = V(x, w, b)
    w2 = d2V_dx2(x, w, b)/m
    eta = beta*hbar*np.sqrt(w2)/2
    Xi = np.tanh(eta)/eta

    P = pref * np.exp(-beta*Vc*Xi) * np.sqrt(Xi)
    dx = x[1] - x[0]
    P /= np.sum(P)*dx
    return P

beta = 40.0

Vcl = V(xarr, w, b)
Pcl = P(Vcl, beta, xarr)

VLH=V_LH(xarr, w, b, beta)
PLH=P(VLH, beta, xarr)

VLH_nc1 = np.zeros_like(xarr)
VLH_nc2 = np.zeros_like(xarr)
VLH_magic = np.zeros_like(xarr)
for i,x in enumerate(xarr):
    VLH_nc1[i] = V_nc1(x, w, b, beta)
    VLH_nc2[i] = V_nc2(x, w, b, beta)
    VLH_magic[i] = V_LH_magic(x, w, b, beta)

PLH_nc1 = P(VLH_nc1, beta, xarr)
PLH_nc2 = P(VLH_nc2, beta, xarr)

PLH_magic = P(VLH_magic, beta, xarr)
PLH_magic_dir = P_LH_magic(xarr, w, b, beta)


xqarr, Pq_arr = Pq(xarr, w, b, beta)



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
    ax0.set_xlim([-3.0,3.0])
    #plt.show()

if(1):


    ax1.plot(xarr, Vcl, color=Cc, linestyle='-', label=r'$V_{cl}$', linewidth=1.5)    
    ax1.plot(xarr, VLH, color=LHc, linestyle='--', label=r'$\tilde{V}_{LH}$', linewidth=1.5)
    ax1.plot(xarr, VLH_nc1, color='orange', linestyle=':', label=r'$\tilde{V}_{LH, nc1}$', linewidth=2)
    ax1.plot(xarr, VLH_nc2, color='c', linestyle=':', label=r'$\tilde{V}_{LH, nc2}$', linewidth=2.) 
    ax1.plot(xarr, VLH_nc1 + Vcl, color='brown', linestyle=':', label=r'$\tilde{V}_{LH, nc1}+V_{cl}$', linewidth=2)
    #ax1.plot(xarr, Vcl + VLH_nc1 + VLH_nc2, color='m', linestyle=':', label=r'$V_{cl}+V_{LH\_nc1}+V_{LH\_nc2}$', linewidth=2.5)
    ax1.plot(xarr, VLH_magic, color=LHc, linestyle='-', label=r'$V_{LH}$', linewidth=1.5)

    if(0):
        ax1.plot(xarr, Vcl, color=Cc, linestyle='-', label=r'$V_{cl}$', linewidth=1.5)    
        ax1.plot(xarr, VLH, color=LHc, linestyle='--', label=r'$V_{LH}$', linewidth=1.5)
        ax1.plot(xarr, VLH_nc1, color='orange', linestyle=':', label=r'$V_{LH, nc1}$', linewidth=2)
        ax1.plot(xarr, VLH_nc2, color='c', linestyle=':', label=r'$V_{LH, nc2}$', linewidth=2.) 
        ax1.plot(xarr, VLH_nc1 + Vcl, color='brown', linestyle=':', label=r'$V_{LH,nc1}+V_{cl}$', linewidth=2)
        #ax1.plot(xarr, Vcl + VLH_nc1 + VLH_nc2, color='m', linestyle=':', label=r'$V_{cl}+V_{LH\_nc1}+V_{LH\_nc2}$', linewidth=2.5)
    


    ax1.set_xlabel(r'$x$', fontsize=xl_fs)
    ax1.set_ylabel(r'$V_{eff}(x)$', fontsize=yl_fs)
    ax1.set_title('Effective Potential Comparison', fontsize=title_fs)
    ax1.legend(fontsize=le_fs)
    ax1.grid()
    ax1.set_ylim([-1.1,1.1])
    ax1.set_xlim([-3.0,3.0])

plt.suptitle(r'$V(x) = x^4/4$, $\beta=%.1f$' % (beta), fontsize=ti_fs)
plt.tight_layout()

plt.savefig('compare_Veff_quartic.pdf', dpi=300)
plt.show()

