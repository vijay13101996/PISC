import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq as scipy_brentq

hbar = 1.0  # Planck's constant (reduced)

def integrand_FK(u, Omega, beta, m, x, d2V):
    eta = beta*Omega/2
    a2 = 1/(m*beta*Omega**2)*(eta/np.tanh(eta)-1)

    integrand = np.exp(-u**2/(2*a2))*d2V(x+u)/np.sqrt(2*np.pi*a2)
    return integrand

def opt_FK(Omega, beta, m, x, d2V, a=1e-6, b=1e2):
    integrand = lambda u: integrand_FK(u, Omega, beta, m, x, d2V)
    result, _ = quad(integrand, -np.inf, np.inf)
    
    #Solve for Omega2 = m^{-1} * result using brentq method
    opt = lambda Omega2: Omega2 - result/m
    Omega2_opt = scipy_brentq(opt, a, b)
    Omega_opt = np.sqrt(Omega2_opt)
    eta = beta*Omega_opt/2
    a2_opt = 1/(m*beta*Omega_opt**2)*(eta/np.tanh(eta)-1)
    return Omega_opt, a2_opt

def V_smear(x, a2, d0V):
    integrand = lambda u: np.exp(-u**2/(2*a2)) * d0V(x + u) / np.sqrt(2*np.pi*a2)
    result, _ = quad(integrand, -np.inf, np.inf)
    return result


    





