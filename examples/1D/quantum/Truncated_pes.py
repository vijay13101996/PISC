import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from scipy.optimize import curve_fit
from PISC.potentials import polynomial

def trunc_quartic(x,c2,c3,c4):
    return c2*x**2 + c3*x**3 + c4*x**4 

def truncate_pes(pes,ngrid,lbt,ubt,q0=0.0):
    qgridtrunc = np.linspace(lbt,ubt,ngrid) 
    pottrunc = pes.potential(qgridtrunc)
    quartic_fit = curve_fit(trunc_quartic,qgridtrunc-q0,pottrunc)
    c2,c3,c4 = quartic_fit[0]
    coeff = [c4,c3,c2,0,0]

    pes_trunc = polynomial(coeff,q0)

    if(0):
        plt.plot(qgridtrunc,pottrunc)
        plt.plot(qgridtrunc,trunc_quartic(qgridtrunc-q0,c2,c3,c4))
        plt.plot(qgridtrunc,pes_trunc.potential(qgridtrunc))
        #plt.plot(qgrid,trunc_quartic(qgrid,c2,c3,c4))
        #plt.plot(qgrid,pes.dpotential(qgrid))
        plt.show()
        exit()

    return pes_trunc
