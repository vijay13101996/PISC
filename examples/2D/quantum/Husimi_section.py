import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.husimi.Husimi import Husimi_2D,Husimi_1D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
from PISC.engine import OTOC_f_1D
from PISC.engine import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
import time 


L = 10.0#
lbx = -7.0#
ubx = 7.0#
lby = -5.0#
uby = 10.0
m = 0.5#8.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid

w = 0.1	
D = 10.0
alpha = 0.363

lamda = 2.0
g = 0.08

z = 1.5

Tc = lamda*0.5/np.pi
T_au = Tc#10.0 

pes = quartic_bistable(alpha,D,lamda,g,z)

path = os.path.dirname(os.path.abspath(__file__))	

if(1):
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	
	vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))
	print('vals',vals[:20])

xgrid = np.linspace(lbx,ubx,101)#200)
ygrid = np.linspace(lby,uby,101)#200)
x,y = np.meshgrid(xgrid,ygrid)

sigma = 1.0#0.5
xgrid = np.linspace(lbx,ubx,ngridx+1)
#xgrid = xgrid[1:ngridx]
ygrid = np.linspace(lby,uby,ngridy+1)
#ygrid = ygrid[1:ngridy]	
husimi = Husimi_2D(xgrid,sigma,ygrid,sigma)

n = 15
DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)		
wf = DVR.eigenstate(vecs[:,n])

plt.imshow(wf**2,origin='lower')
plt.show()

E_wf = vals[n]
print('E_wf', E_wf)

xbasis = np.linspace(-7,7,30)
pxbasis = np.linspace(-5,5,30)
ybasis = np.linspace(-3,5,30)

start_time = time.time()
dist = husimi.Husimi_section_x(xbasis,pxbasis,ybasis,wf,E_wf,pes.potential_xy,m)
#dist = husimi.Husimi_section_x_old(xbasis,pxbasis,0.0,1.0,wf,E_wf,pes.potential_xy,m)

print('time', time.time()-start_time)

plt.imshow(dist,origin='lower')
plt.show()

