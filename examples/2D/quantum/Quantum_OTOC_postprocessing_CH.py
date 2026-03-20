import numpy as np
import matplotlib.pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

omega = 1.0
ngrid = 100

beta = 0.1

#fig, ax = plt.subplots(3,2)

garr = [0.0,0.02,0.04,0.06,0.08,0.1]#,0.12,0.14,0.16,0.18,0.2]

for g in garr:
    potkey = 'coupled_harmonic_w_{}_g_{}'.format(omega,g)

    fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
    path = os.path.dirname(os.path.abspath(__file__))

    vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
    vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

    vals=vals[:6]

    Eavg = np.mean(np.exp(-beta*vals)*vals)/np.mean(np.exp(-beta*vals))
    print('Eavg = ',Eavg)

    plt.scatter(g*np.ones(len(vals)),vals,label='g={}'.format(g))

    if(0):
        #Plot histogram of the energy level differences 
        diff = np.diff(vals)
        ax.flat[garr.index(g)].hist(diff,bins=40,alpha=0.5,label='g={}'.format(g))
        ax.flat[garr.index(g)].legend()

        #plt.xlabel('Energy level difference')
        #plt.ylabel('Frequency')

plt.legend()
plt.show()


