import numpy as np
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import os

path = os.path.dirname(os.path.abspath(__file__))

m = 0.5
L = 5.0
lbx = -L
ubx = L
lby = -L
uby = L
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid
omega = 1.0

g1= 0.25

for g2 in [0.0,0.1,0.2,0.3]:
    #potkey = 'coupled_quartic_g1_{}_g2_{}'.format(g1,g2)
    potkey = 'coupled_harmonic_w_{}_g_{}'.format(1.0,g2)
    
    fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
    vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
    vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

    if(g2==0.0):
        plt.plot(np.arange(0,vals.shape[0]),vals,color='black')
    else:
        plt.plot(np.arange(0,vals.shape[0]),vals)

plt.show()



