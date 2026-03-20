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
LHc = 'darkslateblue'
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

if(1): #Test if integrating the Gaussian integral and using the equations above give the same result
    beta = 1.0
    x = 1.0
    a2_init = beta*hbar**2/(12*m)
    omega_init = 1.0 #w
    p = least_squares(equations, (a2_init, omega_init), args=(x, w, b, beta), bounds=(0,np.inf)).x
    a2, omega = p

    d2V = lambda x: d2V_dx2(x, w, b)
    omega_opt, a2_opt = opt_FK(omega_init, beta, m, x, d2V)
    print('a2 from LS:', a2, 'a2 from integral:', a2_opt)
    print('omega from LS:', omega, 'omega from integral:', omega_opt)
    

def plot_PDF(ax, w, b, legend=False):
    axV = ax[0]
    xgrid = np.linspace(-2,2,1000)
    axV.plot(xgrid, V(xgrid, w, b), color='black')
    if b!=0:
        axV.plot(xgrid, V(xgrid, w, 0.0), color='gray', linestyle='--')
    #axV.set_ylabel(r'$V(x)$', fontsize=yl_fs)
    axV.tick_params(axis='both', which='major', labelsize=ti_fs-3)
    axV.set_ylim([0, 4.0])

    axarr = ax[1:]
    beta_arr = [0.1, 1.0, 10.0]

    for ax, beta in zip(axarr, beta_arr):
        Vcl = V(xarr, w, b)
        Pcl = P(Vcl, beta, xarr)

        VFH = V_FH(xarr, w, b, beta)
        PFH = P(VFH, beta, xarr)

        VFK = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            VFK[i] = V_FK_var(x, w, b, beta)
        PFK=P(VFK, beta, xarr)

        VLH=V_LH(xarr, w, b, beta)
        PLH=P(VLH, beta, xarr)

        VLH_magic = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            VLH_magic[i] = V_LH_magic(x, w, b, beta)
        #PLH_magic=P(VLH_magic, beta, xarr)
        PLH_magic = P_LH_magic(xarr, w, b, beta)


        xqarr, Pq_arr = Pq(xarr, w, b, beta)
        if(beta==0.1 and legend==True): 
            print('Plotting legend')
            ax.plot(xarr, Pcl, label=r'$P_{cl}$', color=Cc)
            ax.plot(xarr, PFH, label=r'$P_{FH}$', color=FHc)
            ax.plot(xarr, PFK, label=r'$P_{FK}$', color=FKc)
            #ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, label=r'$P_{LH}$', color=LHc)
            ax.plot(xqarr, Pq_arr, label=r'$P_{QM}$', color=Qc)
        else:
            ax.plot(xarr, Pcl, color=Cc)
            ax.plot(xarr, PFH, color=FHc)
            ax.plot(xarr, PFK, color=FKc)
            #ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, color=LHc)
            ax.plot(xqarr, Pq_arr, color=Qc)
            
            
        #Find point where Pq_arr is less than tol
        tol = 5e-3
        idx = np.where(Pq_arr > tol)[0]
        if len(idx) > 0:
            xlim = abs( xqarr[idx[0]])
            ax.set_xlim([-xlim, xlim])


        ax.set_ylim([0, 1.2*ax.get_ylim()[1]])
        ax.tick_params(axis='both', which='major', labelsize=ti_fs-3)
        ax.annotate(r'$\beta={}$'.format(beta), xy=(0.275, 0.86), xycoords='axes fraction', fontsize=ti_fs-1.5)

if(1):
    fig, ax = plt.subplots(4,4, figsize=(8,8))
    fig.subplots_adjust(hspace=0.23, wspace=0.33)

    plot_PDF(ax[:,0], w=1.0, b=0.0, legend=True)
    plot_PDF(ax[:,1], w=1.0, b=0.1/4)
    plot_PDF(ax[:,2], w=1.0, b=1.0/4)
    plot_PDF(ax[:,3], w=0.0, b=1.0/4)

    for axe in ax[3,:]:
        axe.set_xlabel(r'$x$', fontsize=xl_fs)

    for axe, w, g in zip(ax[0,:], [1.0, 1.0, 1.0, 0.0], [0.0, 0.1, 1.0, 1.0]):
        axe.annotate(r'$\omega={}$'.format(w), xy=(0.28, 0.83), xycoords='axes fraction', fontsize=ti_fs-1.5)
        axe.annotate(r'$g={}$'.format(g), xy=(0.28, 0.63), xycoords='axes fraction', fontsize=ti_fs-1.5)

    for axe, st in zip(ax[0,:], ['(a)', '(b)', '(c)', '(d)']):
        axe.set_title(st, fontsize=title_fs-1.5)

    for axe in ax[1:,0]:
        axe.set_ylabel(r'$P(x)$', fontsize=yl_fs)
        #set location of ylabel
        axe.yaxis.set_label_coords(-0.35,0.5)

    ax[0,0].set_ylabel(r'$V(x)$', fontsize=yl_fs)
    ax[0,0].yaxis.set_label_coords(-0.35,0.5)
    fig.legend(ncol=6, fontsize=le_fs+1., loc=(0.2,0.005))

    plt.savefig('Quartic_PDFs.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()
    


# Plot Position distributions for different effective potentials
if(0):    
    fig, ax = plt.subplots(2,2, figsize=(5,5))

    ax1 = ax[0,0]
    ax2 = ax[0,1]
    ax3 = ax[1,0]
    ax4 = ax[1,1]

    axarr = [ax1, ax2, ax3, ax4]
    beta_arr = [0.1, 1.0, 5.0, 20.0]

    for ax, beta in zip(axarr, beta_arr):

        Vcl = V(xarr, w, b)
        Pcl = P(Vcl, beta, xarr)

        VFH = V_FH(xarr, w, b, beta)
        PFH = P(VFH, beta, xarr)

        VFK = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            VFK[i] = V_FK_var(x, w, b, beta)
        PFK=P(VFK, beta, xarr)

        VLH=V_LH(xarr, w, b, beta)
        PLH=P(VLH, beta, xarr)

        VLH_magic = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            VLH_magic[i] = V_LH_magic(x, w, b, beta)
        #PLH_magic=P(VLH_magic, beta, xarr)
        PLH_magic = P_LH_magic(xarr, w, b, beta)


        xqarr, Pq_arr = Pq(xarr, w, b, beta)
        if(beta==0.1): 
            ax.plot(xarr, Pcl, label=r'$P_{cl}$', color=Cc)
            ax.plot(xarr, PFH, label=r'$P_{FH}$', color=FKc, linestyle='--')
            ax.plot(xarr, PFK, label=r'$P_{FK}$', color=FKc)
            ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, label=r'$P_{LH}$', color=LHc)
            ax.plot(xqarr, Pq_arr, label=r'$P_{QM}$', color=Qc)
        else:
            ax.plot(xarr, Pcl, color=Cc)
            ax.plot(xarr, PFH, color=FKc, linestyle='--')
            ax.plot(xarr, PFK, color=FKc)
            ax.plot(xarr, PLH, linestyle='--', color=LHc)
            ax.plot(xarr, PLH_magic, color=LHc)
            ax.plot(xqarr, Pq_arr, color=Qc)
            ax.set_xlim([-3,3])
        ax.tick_params(axis='both', which='major', labelsize=ti_fs-2)
        ax.annotate(r'$\beta={}$'.format(beta), xy=(0.025, 0.88), xycoords='axes fraction', fontsize=ti_fs)

    ax1.set_ylabel(r'$P(x)$', fontsize=yl_fs)
    ax3.set_ylabel(r'$P(x)$', fontsize=yl_fs)
    ax3.set_xlabel(r'$x$', fontsize=xl_fs)
    ax4.set_xlabel(r'$x$', fontsize=xl_fs)
    
    #fig.suptitle('Position Distribution at Beta={}'.format(beta))
    fig.legend(ncol=5, fontsize=le_fs, loc=(0.1,0.93))

    #plt.savefig('Veff_w_{}_g_{}.pdf'.format(w,g), dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

# Benchmarking the free energies for different effective potentials
if(0): 
    # Reference: Feynman, R. P., and H. Kleinert. 
    # "Effective classical partition functions." Physical Review A 34.6 (1986): 5080.

    beta_arr = np.arange(0.1,5.01,0.2)
    Fcl_arr = []
    FFH_arr = []
    FFK_arr = []
    Fq_arr = []
    FLH_arr = []
    FLH_magic_arr = []

    for beta in beta_arr:
        Vcl=V(xarr, w, b)
        Zcl=Z(Vcl, beta, xarr)
        Fcl=F(Zcl)
        Fcl_arr.append(Fcl)
        
        VFH=V_FH(xarr, w, b, beta)
        ZFH=Z(VFH, beta, xarr)
        FFH=F(ZFH)
        FFH_arr.append(FFH)

        VFK = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            VFK[i] = V_FK_var(x, w, b, beta)
        
        #VFK = V_FK(xarr, w, b, beta, 0.5, w)  # Using initial guess for a2 and omega for speed

        ZFK=Z(VFK, beta, xarr)
        FFK=F(ZFK)
        FFK_arr.append(FFK)
        
        VLH=V_LH(xarr, w, b, beta)
        ZLH=Z(VLH, beta, xarr)
        FLH=F(ZLH)
        FLH_arr.append(FLH)

        VLH_magic=V_LH_magic(xarr, w, b, beta)
        ZLH_magic=Z(VLH_magic, beta, xarr)
        FLH_magic=F(ZLH_magic)
        FLH_magic_arr.append(FLH_magic)

        Fq_val = Fq(xarr, w, b, beta)
        Fq_arr.append(Fq_val)

    # a2 becomes negative. Find out why!!!

    plt.plot(beta_arr, Fcl_arr, label='Classical Free Energy', marker='o')
    plt.plot(beta_arr, FFH_arr, label='Feynman-Hibbs Free Energy', marker='s')
    plt.plot(beta_arr, FFK_arr, label='Feynman-Kleinert Free Energy', marker='^')
    plt.plot(beta_arr, Fq_arr, label='Quantum Free Energy', marker='x')
    plt.ylim([-0.5,2.0])

    plt.xlabel('Beta')
    plt.ylabel('Free Energy')
    plt.title('Free Energy vs Beta for Quartic Potential')
    plt.legend()
    plt.grid()
    plt.show()
