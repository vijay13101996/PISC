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
w = 1.0
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
    P /= np.sum(P)*dx
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

def P_LH_renorm(x, w, b, beta, magic=False, renorm = None):
    pref = np.sqrt( m/(2*np.pi*beta*hbar**2) )
    Vc = V(x, w, b)

    k = dV_dx(x, w, b)/m
    w2 = d2V_dx2(x, w, b)/m
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

beta = 10.0

for beta in [0.1, 1.0, 10.0]:

Vcl = V(xarr, w, b)
Pcl = P(Vcl, beta, xarr)

VLH = V_LH(xarr, w, b, beta)
PLH = P(VLH, beta, xarr)

PLH_magic = P_LH_magic(xarr, w, b, beta)
print('Integral PLH_magic dx=', np.sum(PLH_magic)*(xarr[1]-xarr[0]))

#PLH_renorm_None = P_LH_renorm(xarr, w, b, beta, magic=False, renorm=None)
PLH_renorm_NCF = P_LH_renorm(xarr, w, b, beta, magic=False, renorm='NCF')
PLH_renorm_PF = P_LH_renorm(xarr, w, b, beta, magic=False, renorm='PF')
#PLH_renorm_magic_None = P_LH_renorm(xarr, w, b, beta, magic=True, renorm=None)
PLH_renorm_magic_NCF = P_LH_renorm(xarr, w, b, beta, magic=True, renorm='NCF')
#PLH_renorm_magic_PF = P_LH_renorm(xarr, w, b, beta, magic=True, renorm='PF')


assert np.isclose( PLH_magic, PLH_renorm_magic_NCF ).all(), "LH Magic and LH Renorm Magic NCF should be the same!"

xarr_q, Pq = Pq(xarr, w, b, beta)

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
plt.title(r'Probability Density Comparison at $\beta={}$'.format(beta), fontsize=title_fs)
plt.legend(fontsize=le_fs)
plt.xlim([-5, 5])
plt.grid()
plt.tight_layout()

#plt.savefig('compare_renorm_quartic.pdf'.format(beta), dpi=300)
plt.show()





