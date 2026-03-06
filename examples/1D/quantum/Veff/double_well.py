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

ngrid=1000
#Testing the FK potential
xarr = np.linspace(-L, L, ngrid)
dx = xarr[1] - xarr[0]

m = 1.0
w = 1.0
g = 0.1
b = g/4

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

#Define Double-well potential and its derivatives
def V(x, w, b):    
    return -0.5*m*w**2*x**2 + b*x**4 + m*w**4/(16*b)

def dV_dx(x, w, b):
    return -m*w**2*x + 4*b*x**3

def d2V_dx2(x, w, b):
    return -m*w**2 + 12*b*x**2

d0V = lambda x: V(x, w, b)
d2V = lambda x: d2V_dx2(x, w, b)

# Feynman-Kleinert effective classical potential
def V_FK(x, d2V, beta):
    omega_init = max(1e-4,d2V(x)/m)**0.5
    omega, a2 = opt_FK(omega_init,beta, m, x, d2V)
    Vsmear = V_smear(x, a2, d0V)

    eta = beta*hbar*omega/2
    Vtemp = np.log( np.sinh(eta)/eta ) / beta - 0.5*m*omega**2*a2

    return Vsmear + Vtemp

# Feynman-Hibbs effective classical potential 
def V_FH(x, d0V, beta):
    return V_smear(x, beta*hbar**2/(12*m), d0V)

# Local-harmonic effective classical potential for the quartic potential
def V_LH(x, w, b, beta):
    k = dV_dx(x, w, b)/m
    w2 = max(d2V_dx2(x, w, b)/m, 1e-12)  # Prevent division by zero or negative sqrt

    eta = beta*hbar*np.sqrt(w2)/2

    Vc = V(x, w, b)

    pref = m*k**2/(2*w2)
    Vnc1 = pref*( np.tanh(eta)/eta - 1 )
    Vnc2 = 0.5/beta * np.log( np.sinh(2*eta)/(2*eta) )
    Vnc = Vnc1 + Vnc2

    return Vc + Vnc
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
    
    w2[w2<1e-12] = 1e-12  # Prevent division by zero or negative sqrt

    eta = beta*hbar*np.sqrt(w2)/2
    Xi = np.tanh(eta)/eta

    P = pref * np.exp(-beta*Vc*Xi) * np.sqrt(Xi)
    dx = x[1] - x[0]
    P /= np.sum(P)*dx
    return P


def plot_PDF(ax, w, b, legend=False):

    axV = ax[0]
    xgrid = np.linspace(-5,5,1000)
    axV.plot(xgrid, V(xgrid, w, b), color='black')
    axV.tick_params(axis='both', which='major', labelsize=ti_fs-3)
    axV.set_ylim([-0.05,10.0])

    axarr = ax[1:]
    beta_arr = [0.1, 1.0, 10.0]

    for ax, beta in zip(axarr, beta_arr):
        Vcl = V(xarr, w, b)
        Pcl = P(Vcl, beta, xarr)

        VLH = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            VLH[i] = V_LH(x, w, b, beta)
        PLH = P(VLH, beta, xarr)
        PLH_magic = P_LH_magic(xarr, w, b, beta)


        xqarr, Pq_arr = Pq(xarr, w, b, beta)
        if(beta==0.1 and legend==True): 
            print('Plotting legend')
            ax.plot(xarr, Pcl, label=r'$P_{cl}$', color=Cc)
            #ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, label=r'$P_{LH}$', color=LHc)
            ax.plot(xqarr, Pq_arr, label=r'$P_{QM}$', color=Qc)
        else:
            ax.plot(xarr, Pcl, color=Cc)
            #ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, color=LHc)
            ax.plot(xqarr, Pq_arr, color=Qc)
            
        #Find point where Pq_arr is less than tol
        tol = 1e-4
        idx = np.where(Pq_arr > tol)[0]
        if len(idx) > 0:
            xlim = abs( xqarr[idx[0]])
            ax.set_xlim([-xlim, xlim])
    
        ax.set_ylim([0, 1.8*max(Pq_arr)])
        #ax.set_xticks([])

        ax.tick_params(axis='both', which='major', labelsize=ti_fs-3)

        ax.annotate(r'$\beta={}$'.format(beta), xy=(0.3, 1.05), xycoords='axes fraction', fontsize=ti_fs-1.5)
if(1):
    fig, ax = plt.subplots(4,2, figsize=(4,8))
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    plot_PDF(ax[:,0], w, 0.1/4, legend=True)
    plot_PDF(ax[:,1], w, 0.5/4, legend=False)

    for axe in ax[3,:]:
        axe.set_xlabel(r'$x$', fontsize=xl_fs)
        #axe.tick_params(axis='both', which='major', labelsize=ti_fs-3)

    for axe, w, g in zip(ax[0,:], [1.0, 1.0, 1.0, 0.0], [0.1, 0.5]):
        axe.annotate(r'$\omega={}$'.format(w), xy=(0.28, 0.83), xycoords='axes fraction', fontsize=ti_fs-1.5)
        axe.annotate(r'$g={}$'.format(g), xy=(0.28, 0.63), xycoords='axes fraction', fontsize=ti_fs-1.5)
    
    for axe, st in zip(ax[0,:], ['(a)', '(b)']):
        axe.set_title(st, fontsize=title_fs-1.5)

    for axe in ax[1:,0]:
        axe.set_ylabel(r'$P(x)$', fontsize=yl_fs)
        #set location of ylabel
        axe.yaxis.set_label_coords(-0.28,0.5)

    ax[0,0].set_ylabel(r'$V(x)$', fontsize=yl_fs)
    ax[0,0].yaxis.set_label_coords(-0.28,0.5)
    fig.legend(ncol=3, fontsize=le_fs+1., loc=(0.183,0.005))

    plt.savefig('DW_PDFs.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()


# Plot Position distributions for potentials with negative curvature
if(0):
    fig, ax = plt.subplots(2,2, figsize=(5,5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    ax1 = ax[0,0]
    ax2 = ax[0,1]
    ax3 = ax[1,0]
    ax4 = ax[1,1]

    axarr = [ax1, ax2, ax3, ax4]
    beta_arr = [0.1, 1.0, 5.0, 20.0]
    
    for ax, beta in zip(axarr, beta_arr):
        Vcl = V(xarr, w, b)
        Pcl = P(Vcl, beta, xarr)

        VFK = np.zeros_like(xarr)
        #for i, x in enumerate(xarr):
        #    VFK[i] = V_FK(x, d2V, beta)
        PFK = P(VFK, beta, xarr)

        VFH = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            VFH[i] = V_FH(x, d0V, beta)
        PFH = P(VFH, beta, xarr)

        VLH = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            VLH[i] = V_LH(x, w, b, beta)
        PLH = P(VLH, beta, xarr)
    
        PLH_magic = P_LH_magic(xarr, w, b, beta)

        xqarr, Pq_arr = Pq(xarr, w, b, beta)

        #for Pd in Pcl, PFK, PFH, PLH, PLH_magic:
        #    Pd[Pd>1.5*Pq_arr.max()] = np.nan  # Avoid plotting extremely large values

        if (ax == ax2):
            ax.plot(xarr, Pcl, label=r'$P_{cl}$', color=Cc)
            #ax.plot(xarr, PFH, label=r'$FH$', color=FKc, linestyle='--')
            #ax.plot(xarr, PFK, label=r'$FK$', color=FKc)
            ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, label=r'$P_{LH}$', color=LHc)
            ax.plot(xqarr, Pq_arr, label=r'$P_{QM}$', color=Qc)
        else:
            ax.plot(xarr, Pcl, color=Cc)
            #ax.plot(xarr, PFH, color=FKc, linestyle='--')
            #ax.plot(xarr, PFK, color=FKc)
            ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, color=LHc)
            ax.plot(xqarr, Pq_arr, color=Qc)
            #ax.set_xlim([-3,3])
        ax.annotate(r'$\beta={}$'.format(beta), xy=(0.3, 1.025), xycoords='axes fraction', fontsize=ti_fs)
        ax.tick_params(axis='both', which='major', labelsize=ti_fs-2)
        
        #ax.set_xlim([-5, 5])
        ax.set_ylim([0, 1.8*max(Pq_arr)])


    ax1.set_ylabel(r'$P(x)$', fontsize=yl_fs)
    ax3.set_ylabel(r'$P(x)$', fontsize=yl_fs)
    ax3.set_xlabel(r'$x$', fontsize=xl_fs)
    ax4.set_xlabel(r'$x$', fontsize=xl_fs)

    #plt.xlabel('x')
    #plt.ylabel('P(x)')
    #fig.suptitle('Position Distribution at Beta={}'.format(beta))
    fig.legend(ncol=3, fontsize=le_fs, loc=(0.25,0.94))

    #plt.savefig('Veff_DW_g_{}.pdf'.format(g), dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

