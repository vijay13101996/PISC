import numpy as np  
from PISC.potentials import morse, Veff_classical_1D_LH, Veff_classical_1D_GH, Veff_classical_1D_FH, Veff_classical_1D_FK
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from numpy.polynomial import polynomial as P

#Morse potential
D = 0.18748
alpha = 1.1605
req = 0.0#1.8324
pes = morse(D,alpha,req)

qgrid = np.linspace(-1,4,1001)
V = pes.potential(qgrid)
dV = pes.dpotential(qgrid)
ddV = pes.ddpotential(qgrid)

def fit_func(x,a,b,c):#,d,e):#,f,g):
    coeff = [a,b,c]#,d,e]#,f,g]
    return coeff[0]*x**2 + coeff[1]*x**3 + coeff[2]*x**4 #+ coeff[3]*x**5 + coeff[4]*x**6 #+ coeff[5]*x**7 + coeff[6]*x**8


popt,pcov = curve_fit(fit_func,qgrid,V)

print(popt)

plt.plot(qgrid,V)
#plt.plot(qgrid,dV)
#plt.plot(qgrid,ddV)
plt.plot(qgrid,fit_func(qgrid,*popt))
plt.ylim([-0.1,2])
plt.show()

