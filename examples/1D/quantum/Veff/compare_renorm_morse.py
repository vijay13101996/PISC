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
    P /= np.sum(P)*dx
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
    P /= np.sum(P)*dx
    return P


def P_LH_renorm(x, D, alpha, x0, beta, magic=False, renorm = None):
    pref = np.sqrt( m/(2*np.pi*beta*hbar**2) )
    Vc = V(x, D, alpha, x0)

    k = dV_dx(x, D, alpha, x0)/m
    w2 = d2V_dx2(x, D, alpha, x0)/m
    
    w2[w2<1e-12] = 1e-12  # Prevent division by zero or negative values

    eta = beta*hbar*np.sqrt(w2)/2
    Xi = np.tanh(eta)/eta
    sinh_term = np.sinh(2*eta)/(2*eta)


    Vnc1_pref = m*k**2/(2*w2)

    ZQM = 1/(2*np.sinh(eta))
    Zcl = 1/(2*eta)
    ZNCF = ZQM / Zcl
 
    if magic:
        print("Using Magic LH Renormalization")
        Vnc1_pref = Vc

    P = pref * np.sqrt(1/sinh_term) * np.exp(-beta*(Vc - Vnc1_pref) - beta*Vnc1_pref*Xi)

    if renorm == 'NCF':
        print("Using NCF Renormalization", ZNCF.shape, P.shape)
        P /= ZNCF
    elif renorm == 'PF':
        P /= ZQM
    elif renorm == None:
        P /= 1.0

    dx = x[1] - x[0]
    P /= np.sum(P)*dx
    print('Integral P dx=', np.sum(P)*dx)
    return P

T = 300 
K2au = 315775.13
beta = 1/T*K2au


Vcl = V(xarr, D, alpha, x0)
Pcl = P(Vcl, beta, xarr)

#VLH = V_LH(xarr, D, alpha, x0, beta)
VLH = np.zeros_like(xarr)
for i,x in enumerate(xarr):
    VLH[i] = V_LH(x, D, alpha, x0, beta)

PLH = P(VLH, beta, xarr)

PLH_magic = P_LH_magic(xarr, D, alpha, x0, beta)
print('Integral PLH_magic dx=', np.sum(PLH_magic)*(xarr[1]-xarr[0]))


PLH_renorm_None = P_LH_renorm(xarr, D, alpha, x0, beta, magic=False, renorm=None)
PLH_renorm_NCF = P_LH_renorm(xarr, D, alpha, x0, beta, magic=False, renorm='NCF')
PLH_renorm_PF = P_LH_renorm(xarr, D, alpha, x0, beta, magic=False, renorm='PF')
PLH_renorm_magic_None = P_LH_renorm(xarr, D, alpha, x0, beta, magic=True, renorm=None)
PLH_renorm_magic_NCF = P_LH_renorm(xarr, D, alpha, x0, beta, magic=True, renorm='NCF')
PLH_renorm_magic_PF = P_LH_renorm(xarr, D, alpha, x0, beta, magic=True, renorm='PF')

np.isclose( PLH_magic, PLH_renorm_magic_NCF ).all(), "LH Magic and LH Renorm Magic NCF should be the same!"

xarr_q, Pq = Pq(xarr, D, alpha, x0, beta)

plt.plot(xarr, Pcl, label=r'$P_{cl}$', color=Cc, linestyle='-')
plt.plot(xarr, PLH, label=r'$\tilde{P}_{LH}$', color=LHc, linestyle='--')


if(1):
    plt.plot(xarr, PLH_renorm_NCF, label=r'$P^{(r)}_{LH}$', color=LHc, linestyle=':')
    #plt.plot(xarr, PLH_renorm_PF, label='LH Renorm PF', linestyle='--')
    #plt.plot(xarr, PLH_renorm_magic_None, label='LH Magic Renorm None', color='orange', linestyle=':')
    #plt.plot(xarr, PLH_renorm_magic_NCF, label='LH Magic Renorm NCF', color='purple', linestyle=':',lw=3)
    #plt.plot(xarr, PLH_renorm_magic_PF, label='LH Magic Renorm PF', linestyle=':')
plt.plot(xarr, PLH_magic, label=r'$P_{LH}$', color=LHc, linestyle='-')
plt.plot(xarr_q, Pq, label=r'$P_{QM}$', color=Qc, linestyle='-')


plt.xlabel('x', fontsize=xl_fs)
plt.ylabel('Probability Density P(x)', fontsize=yl_fs)
plt.title(r'Probability Density Comparison at $T={}K$'.format(T), fontsize=title_fs)
plt.legend(fontsize=le_fs)
plt.xlim([1, 3])
plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0])
plt.ylim([0, 10.0])

plt.grid()
plt.tight_layout()

plt.savefig('compare_renorm_morse.pdf'.format(beta), dpi=300)
plt.show()













