import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from math import exp
from Free_energy import compute_free_energy

ngrid = 1000
#Quartic/Harmonic/Mildly anharmonic
lb = -10.
ub = 10.0
dx = (ub-lb)/(ngrid-1)
qgrid = np.linspace(lb,ub,ngrid)

m = 1
w = 1. #quadratic term (frequency)
a = 0.0 #cubic term
b = 0.25 #quartic term
g = 2.0 #anharmonicity parameter

beta = 1.0

def equations(vecs,x0,beta,g):
    a2, Omega = vecs
    xi = 0.5*beta*Omega
    eq1 = a2 - (xi*np.cosh(xi)/np.sinh(xi) - 1)/(beta*Omega**2)
    eq2 = Omega**2 - 1 - 3*g*a2 - 3*g*x0**2
    return [eq1, eq2]


for g in [0.0,2.0,4.0,40.0]:
    beta_arr = np.linspace(0.5, 5.0, 20)
    FE_arr = []
    for beta in beta_arr:
        W1arr = np.zeros(ngrid)
        for i in range(len(qgrid)):
            x0 = qgrid[i]
            var = fsolve(equations, [0.1, 0.1], args=(x0,beta,g))

            a2 = var[0]
            Omega = var[1]

            print('a2 = ', a2, 'Omega = ', Omega)

            xi = 0.5*beta*Omega

            Va2 = 0.5*a2 + 0.75*g*a2**2 + 0.5*(1+3*g*a2)*x0**2 + 0.25*g*x0**4

            W1arr[i] = np.log(np.sinh(xi)/xi)/beta - 0.5*Omega**2*a2 + Va2

        Z = np.sum(np.exp(-beta*W1arr)*dx)
        FE = -np.log(Z)/beta
        FE_arr.append(FE)

    plt.plot(beta_arr, FE_arr, label='g = '+str(g))

plt.legend()
plt.show()


