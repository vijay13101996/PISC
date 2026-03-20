import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import Veff_classical_1D_LH, Veff_classical_1D_FK, Veff_classical_1D_FH
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from Truncated_pes import truncate_pes
from numpy import nan
import os 
from scipy.optimize import fsolve
from math import exp

def compute_free_energy(beta,qgrid,vals,vecs,pes,m,method,tol=1e-4,renorm='NCF',fur_renorm='VGS',magic=True): 
    
    dx = qgrid[1]-qgrid[0]
    potgrid = pes.potential_func(qgrid)
    
    if (method=='quantum'):
        Pq = np.zeros_like(qgrid)
        for i in range(len(vals)):
            Pq+= np.exp(-beta*vals[i])*vecs[:,i]**2
        #if(np.sum(Pq*dx)<1e-4): #if the sum is too small, then use the ground state distribution
        #    Pq = vecs[:,0]**2
        #else:
        #    Pq/=np.sum(Pq*dx)
        #bVq = -np.log(Pq) # Free energy function 
        Zq = np.sum(Pq*dx) # Free energy
        return Zq

    elif (method=='classical'):
        bVcl = beta*potgrid
        #bVcl+= 0.5*np.log(m/2/np.pi/beta)/beta
        Pcl = np.exp(-bVcl)
        #Pcl/=np.sum(Pcl*dx)
        #bVcl = -np.log(Pcl)
        Zcl = np.sum(Pcl*dx)/np.sqrt(2*np.pi*beta/m)
        return Zcl

    elif (method=='LH'):
        pes_eff = Veff_classical_1D_LH(pes,beta,m,tol=tol,renorm=renorm,fur_renorm=fur_renorm,magic=magic)
        bVeff = np.zeros_like(qgrid)
        for i in range(len(qgrid)): 
            bVeff[i] = beta*pes_eff.potential(qgrid[i])
        Peff = np.exp(-bVeff)
        norm = np.sum(Peff*dx)
        #Peff/=norm
        #bVeff = -np.log(Peff)  
        Zeff_LH = np.sum(Peff*dx)/np.sqrt(2*np.pi*beta/m)
        return Zeff_LH

    elif (method=='FK'):
        pes_eff = Veff_classical_1D_FK(pes,beta,m)#,tol=tol)
        bVeff = np.zeros_like(qgrid)
        for i in range(len(qgrid)): 
            bVeff[i] = beta*pes_eff.potential(qgrid[i])
        Peff = np.exp(-bVeff)
        norm = np.sum(Peff*dx)
        #Peff/=norm
        Zeff_FK = np.sum(Peff*dx)/np.sqrt(2*np.pi*beta/m)   
        return Zeff_FK

    elif (method=='FH'):
        pes_eff = Veff_classical_1D_FH(pes,beta,m,qgrid)
        bVeff = np.zeros_like(qgrid)
        for i in range(len(qgrid)): 
            bVeff[i] = beta*pes_eff.potential(qgrid[i])
        Peff = np.exp(-bVeff)
        norm = np.sum(Peff*dx)
        #Peff/=norm
        Zeff_FH = np.sum(Peff*dx)/np.sqrt(2*np.pi*beta/m)   
        return Zeff_FH

def compute_free_energy_FK_iteration(beta, qgrid, coeff, m):
    ngrid = len(qgrid)
    dx = qgrid[1]-qgrid[0]
    def equations(varis,x0,beta,coeff):
        a2, Omega = varis
        k0,k1,k2,k3,k4 = coeff
        xi = 0.5*beta*Omega
        eq1 = a2 - (xi*np.cosh(xi)/np.sinh(xi) - 1)/(beta*Omega**2)
        #eq2 = Omega**2 - 1 - 3*g*a2 - 3*g*x0**2
        eq2 = m*Omega**2/2 - (k0 + 3*k3*x0 + 6*k4*(x0**2 + a2))
        return [eq1, eq2]
    
    W1arr = np.zeros(ngrid)
    for i in range(len(qgrid)):
        x0 = qgrid[i]
        var = fsolve(equations, [0.1, 0.1], args=(x0,beta,coeff))

        a2 = var[0]
        Omega = var[1]
        k0,k1,k2,k3,k4 = coeff

        #print('a2 = ', a2, 'Omega = ', Omega)

        xi = 0.5*beta*Omega
        Va2 = k0 + k1*x0 + k2*(x0**2+a2) + k3*(x0**3 + 3*a2*x0) + k4*(x0**4 + 6*a2*x0**2 + 3*a2**2)

        W1arr[i] = np.log(np.sinh(xi)/xi)/beta - 0.5*Omega**2*a2 + Va2


    Zeff_FK_opt = np.sum(np.exp(-beta*W1arr)*dx)/np.sqrt(2*np.pi*beta/m)
    return Zeff_FK_opt


def compute_free_energy_LH_iteration(beta, qgrid, Vc, ddVc, m):
    ngrid = len(qgrid)
    dx = qgrid[1]-qgrid[0]
    V = Vc(qgrid)
    ddV = ddVc(qgrid)
    
    def func(x, Vx):
        num = (1-2*x/np.tanh(2*x))*x
        denom = 2*(x/np.cosh(x)**2 - np.tanh(x))
        return denom/num - 1/(beta*Vx)

    #Find optimal xi_a at each point
    xiarr = np.zeros(ngrid)
    for i in range(ngrid):
        w = np.sqrt(ddV[i]/m)
        xi = beta*w/2
        sol = fsolve(func, xi, args=(V[i]))
        print('sol', sol)
        xiarr[i] = sol[0]
        #xiarr[i] = xi
        print('xi', np.tanh(xiarr[i])/xiarr[i], np.tanh(xi)/xi)

    #Compute the effective potential
    Veff = np.zeros(ngrid)
    for i in range(ngrid):
        xi = xiarr[i]
        Veff[i] = V[i]*(np.tanh(xi)/xi) + np.log(np.sinh(2*xi)/(2*xi))/(2*beta)

    Zeff_LH_opt = np.sum(np.exp(-beta*Veff)*dx)/np.sqrt(2*np.pi*beta/m)
    return Zeff_LH_opt


