import numpy as np
from PISC.dvr.dvr import DVR2D
from matplotlib import pyplot as plt
from PISC.potentials import Morse_harm_2D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
import time 
import argparse

ngrid = 100 #Number of grid points
ngridx = ngrid #Number of grid points along x
ngridy = ngrid #Number of grid points along y

n_eig_tot = 100 #Number of eigenstates to be calculated

#System parameters
m = 1.0
L = 10.0
lbx = -L
ubx = L
lby = -L
uby = L
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid

omega = 1.0
xe = 0.05

alpha = np.sqrt(2*m*omega*xe)
D = omega/(4*xe)

z = 0.1


potkey = 'Morse_harmonic_m_{}_omega_{}_xe_{}_z_{}'.format(m,omega,xe,z)
pes = Morse_harm_2D(m,omega,D,alpha,z)

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
if(0):
    xg = np.linspace(lbx,ubx,ngridx)
    yg = np.linspace(lby,uby,ngridy)
    xgr,ygr = np.meshgrid(xg,yg)
    plt.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,10,1.0))
    plt.show()    
    exit()

fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
path = os.path.dirname(os.path.abspath(__file__))

if(1): #Test whether the Wavefunctions look correct
    vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
    vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    #n=8
    #print('vals[n]', vals[n])
    #plt.imshow(DVR.eigenstate(vecs[:,n])**2,origin='lower')

#Compute the position distribution functions
beta = 1.0
Z = 0.0
for n in range(len(vals)):
    Z += np.exp(-beta*vals[n])


pdf = np.zeros((ngridx+1,ngridy+1))
for n in range(len(vals)):
    pdf += (np.exp(-beta*vals[n])/Z)*DVR.eigenstate(vecs[:,n])**2

plt.imshow(pdf,origin='lower')
plt.show()





