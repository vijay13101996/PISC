import numpy as np
from PISC.dvr.dvr import DVR2D
from matplotlib import pyplot as plt
from PISC.potentials import coupled_harmonic_oblique
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
hbar = 1.0

L = 10.0
lbx = -L
ubx = L
lby = -L
uby = L

ngrid = 100
ngridx = ngrid
ngridy = ngrid

dx = (ubx-lbx)/ngridx
dy = (uby-lby)/ngridy

xg = np.linspace(lbx,ubx,ngridx+1)
yg = np.linspace(lby,uby,ngridy+1)
xgr,ygr = np.meshgrid(xg,yg)

w1 = 1.0
w2 = 2.0
g = 0.0

potkey = 'coupled_harmonic_w1_{}_w2_{}_g_{}'.format(w1,w2,g)
pes = coupled_harmonic_oblique(m,w1,w2,g)

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
if(0):
    plt.contour(xgr,ygr,pes.potential_xy(xgr,ygr),levels=np.arange(0,10,1.0))
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

#---------------------------------------------------------------

#Compute the position distribution functions
beta = 10.0
Z = 0.0
for n in range(len(vals)):
    Z += np.exp(-beta*vals[n])

pdf = np.zeros((ngridx+1,ngridy+1))
for n in range(len(vals)):
    pdf += (np.exp(-beta*vals[n])/Z)*DVR.eigenstate(vecs[:,n])**2

print('norm',np.sum(pdf*dx*dy))

#plt.imshow(pdf,origin='lower')
#plt.show()

#-----------------------------------------------------------------

#Compute the classical position distribution function analytically
pdf_analytic_cl = np.exp(-beta*pes.potential_xy(xgr,ygr))
pdf_analytic_cl /= np.sum(pdf_analytic_cl*dx*dy)

print('norm c',np.sum(pdf_analytic_cl*dx*dy))

#-----------------------------------------------------------------

#Compute the quantum mechanical position distribution function analytically
pdf_analytic_qm = np.zeros((ngridx+1,ngridy+1))

xi_x = (m*w1/hbar)*np.tanh(beta*hbar*w1/2)
xi_y = (m*w2/hbar)*np.tanh(beta*hbar*w2/2)

prefx = np.sqrt(xi_x/np.pi)
prefy = np.sqrt(xi_y/np.pi)

for i in range(ngridx+1):
    for j in range(ngridy+1):
        pdf_analytic_qm[i,j] = prefx*prefy*np.exp(-xi_x*(xg[i]**2) - xi_y*(yg[j]**2))

print('norm q',np.sum(pdf_analytic_qm*dx*dy))

kl_div_q = np.sum(pdf*np.log(pdf/pdf_analytic_qm)*dx*dy)
kl_div_c = np.sum(pdf*np.log(pdf/pdf_analytic_cl)*dx*dy)

print('kl_div_q',kl_div_q)
print('kl_div_c',kl_div_c)

diff_qm = np.abs(pdf_analytic_qm - pdf)
diff_cl = np.abs(pdf_analytic_cl - pdf)

plt.contour(xgr,ygr,diff_qm,origin='lower',levels=np.arange(0,0.5,0.01))
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.show()

print('diff_qm',np.sum(diff_qm*dx*dy))
print('diff_cl',np.sum(diff_cl*dx*dy))




