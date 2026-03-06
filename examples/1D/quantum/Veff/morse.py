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
LHc = 'darkslateblue'
FKc = 'c'

#Fontsize
xl_fs = 12
yl_fs = 12
title_fs = 14
le_fs = 10
ti_fs = 12

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
    w2 = max(d2V_dx2(x, D, alpha, x0)/m, w*0.01)  # Prevent division by zero or negative sqrt

    eta = beta*hbar*np.sqrt(w2)/2

    Vc = V(x, D, alpha, x0)

    pref = m*k**2/(2*w2)
    Vnc1 = pref*( np.tanh(eta)/eta - 1 )
    Vnc2 = 0.5/beta * np.log( np.sinh(2*eta)/(2*eta) )
    Vnc = Vnc1 + Vnc2

    return Vc + Vnc

def V_LH_magic(x, D, alpha, x0, beta):
    w2 = d2V_dx2(x, D, alpha, x0)/m
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

def P_LH_magic(x, D, alpha, x0, beta):
    pref = np.sqrt( m/(2*np.pi*beta*hbar**2) )
    Vc = V(x, D, alpha, x0)
    w2 = d2V_dx2(x, D, alpha, x0)/m
    
    w2[w2<1e-12] = 1e-12  # Prevent division by zero or negative values

    eta = beta*hbar*np.sqrt(w2)/2
    Xi = np.tanh(eta)/eta

    P = pref * np.exp(-beta*Vc*Xi) * np.sqrt(Xi)
    dx = x[1] - x[0]
    P /= np.sum(P)*dx
    return P
    
# Plot Position distributions for potentials with negative curvature
if(1):
    fig, ax = plt.subplots(2,2, figsize=(4,4.5))

    fig.subplots_adjust(hspace=0.1, wspace=0.2)

    ax1 = ax[0,0]
    ax2 = ax[0,1]
    ax3 = ax[1,0]
    ax4 = ax[1,1]

    axarr = [ax1, ax2, ax3, ax4]

    Tlist = np.flip([1, 100, 300, 5000])
    K2au = 315775.13
    beta_arr = [1/T*K2au for T in Tlist]
    
    for ax, beta in zip(axarr, beta_arr):
        #Manually set axis sizes
        ax.set_box_aspect(1)

        Vcl = V(xarr, D, alpha, x0)
        Pcl = P(Vcl, beta, xarr)

        #ax.plot(xarr, Vcl, color='black')
        
        #VFK = np.zeros_like(xarr)
        #for i, x in enumerate(xarr):
        #    VFK[i] = V_FK(x, d2V, beta)
        #PFK = P(VFK, beta, xarr)

        #VFH = np.zeros_like(xarr)
        #for i, x in enumerate(xarr):
        #    VFH[i] = V_FH(x, d0V, beta)
        #PFH = P(VFH, beta, xarr)

        VLH = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            VLH[i] = V_LH(x, D, alpha, x0, beta)
        PLH = P(VLH, beta, xarr)
        print('LH Integral for T={}, P dx='.format(Tlist[np.where(np.array(beta_arr)==beta)[0][0]]), np.sum(PLH)*dx)

        PLH_magic = P_LH_magic(xarr, D, alpha, x0, beta)

        xqarr, Pq_arr = Pq(xarr, D, alpha, x0, beta)
        if (ax == ax2):
            ax.plot(xarr, Pcl, label=r'$P_{cl}$', color=Cc)
            #ax.plot(xarr, PFH, label=r'$FH$', color=FKc, linestyle='--')
            #ax.plot(xarr, PFK, label=r'$FK$', color=FKc)
            #ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, label=r'$P_{LH}$', color=LHc)
            ax.plot(xqarr, Pq_arr, label=r'$P_{QM}$', color=Qc)
        else:
            ax.plot(xarr, Pcl, color=Cc)
            #ax.plot(xarr, PFH, color=FKc, linestyle='--')
            #ax.plot(xarr, PFK, color=FKc)
            #ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, color=LHc)
            ax.plot(xqarr, Pq_arr, color=Qc)

        ax.set_xlim([0.5, 2.8])
        ax.margins(y=0.5)
        
        ax.set_ylim([0, 1.5*max(Pq_arr)])
        ax.annotate(r'$T={}$ K'.format(Tlist[np.where(np.array(beta_arr)==beta)[0][0]]), xy=(0.025, 0.88), xycoords='axes fraction', fontsize=ti_fs-1.5)
        ax.tick_params(axis='both', which='major', labelsize=ti_fs-1.5)

    ax1.set_ylabel(r'$P(x)$', fontsize=yl_fs)
    ax3.set_ylabel(r'$P(x)$', fontsize=yl_fs)
    ax3.set_xlabel(r'$x$', fontsize=xl_fs)
    ax4.set_xlabel(r'$x$', fontsize=xl_fs)

    ax3.xaxis.set_label_coords(0.48,-0.125)
    ax4.xaxis.set_label_coords(0.48,-0.125)

    #plt.xlabel('x')
    #plt.ylabel('P(x)')
    #fig.suptitle('Position Distribution at Beta={}'.format(beta))
    fig.legend(ncol=3, fontsize=le_fs+1., loc=(0.17,0.002))
    
    #fig.tight_layout()

    plt.savefig('Veff_morseTRPMD.pdf', dpi=300, bbox_inches='tight',pad_inches=0.02)
    plt.show()

