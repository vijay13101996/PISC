import numpy as np
from PISC.dvr.dvr import DVR1D
#from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials import double_well, morse, quartic, asym_double_well, harmonic1D
from PISC.potentials.triple_well_potential import triple_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D_omp_updated
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from scipy.optimize import curve_fit
import argparse

ngrid = 600

L = 10.0
lb = -L
ub = L
m = 0.5

lamda = 2.0
g = 0.08

Tc = 0.5*np.pi/lamda

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

path = os.path.dirname(os.path.abspath(__file__))
times_arr = [0.5,1.0,1.5,2.0,2.5]
#[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
b1_arr = []


for times in times_arr:
    Tkey = 'T_{}Tc'.format(times)
    T = times*Tc

    data = read_1D_plotdata('{}/Datafiles/Wightman_Im_qqTCF_{}_{}.txt'.format(path,potkey,Tkey))
    #data = read_1D_plotdata('{}/Datafiles/Im_qqTCF_{}_{}_{}.txt'.format(path,potkey,Tkey))
    
    t_arr = data[:,0]
    C_arr = data[:,1]

    #Plot first derivative
    der = T*np.gradient(C_arr,t_arr[1]-t_arr[0])
    derder = np.gradient(der,t_arr[1]-t_arr[0])

    #Fit derivative to linear function
    popt,pcov = curve_fit(lambda x,l: l*x,t_arr,der) 

    b1_arr.append(popt[0])

    #plt.scatter(times,abs(der[0]))
    #plt.plot(t_arr,der,label='T = {}Tc, der'.format(times))
    plt.plot(t_arr,C_arr,label='T = {}Tc'.format(times))

plt.legend()
plt.show()

if(0):
    plt.scatter(times_arr,b1_arr)
    plt.plot(times_arr,b1_arr)
    plt.plot(times_arr,2*np.pi*np.array(times_arr)*Tc)
    plt.show()
