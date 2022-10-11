import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.husimi.Husimi import Husimi_2D,Husimi_1D
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.potentials.harmonic_2D import Harmonic
from PISC.potentials.Heller_Davis import heller_davis
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
from PISC.utils.colour_tools import lighten_color
from PISC.engine import OTOC_f_1D
from PISC.engine import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
import time 
import ast

path = os.path.dirname(os.path.abspath(__file__))	

if(1):#
	hbar = 1.0
	m=0.5
	
	w = 0.1	
	D = 9.375#10.0
	alpha = 0.229#382#35#

	lamda = 2.0
	g = 0.08

	z = 1.0

	Tc = lamda*0.5/np.pi
	T_au = Tc#10.0 
	
	pes = quartic_bistable(alpha,D,lamda,g,z)
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	
	with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey)) as f:	
		for line in f:
			pass
		param_dict = ast.literal_eval(line)

	lbx = param_dict['lbx']
	ubx = param_dict['ubx']
	lby = param_dict['lby']
	uby = param_dict['uby']
	ngridx = param_dict['ngridx']
	ngridy = param_dict['ngridy']

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
	fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)	# Change ngridx!=ngridy

	vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))
	#print('vals',vals[:20])
	
if(0):
	L = 10.0
	lbx = -L
	ubx = L
	lby = -L
	uby = L
	m = 1.0
	hbar = 1.0
	ngrid = 100
	ngridx = ngrid
	ngridy = ngrid
	omega = 2.0**0.5#1.0
	g0 = 0.1
	x = np.linspace(lbx,ubx,ngridx+1)
	potkey = 'coupled_harmonic_w_{}_g_{}_m_{}_hbar_{}'.format(omega,g0,m,hbar)

	T_au = 0.5 
	beta = 1.0/T_au 

	basis_N = 40#165
	n_eigen = 30#150

	print('T in au, potential, basis',T_au, potkey,basis_N )

	pes = coupled_harmonic(omega,g0)

	fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	path = os.path.dirname(os.path.abspath(__file__))

	vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

if(0):
	ws= 1.0
	wu= 1.1
	lamda = -0.11
	lbx = -7.0
	ubx = 7.0
	lby = -9.0
	uby = 9.0
	m = 1.0

	pes = heller_davis(ws,wu,lamda)#Harmonic(lamda)#

	potkey = 'Heller_Davis_ws_{}_wu_{}_lamda_{}'.format(ws,wu,lamda)
	fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	
	vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))
	print('vals',vals[:20])

if(0):
	potkey = 'Harmonic_2D_lamda_{}'.format(lamda)
	fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
	
	vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
	vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))
	print('vals',vals[:20])


xgrid = np.linspace(lbx-1,ubx+1,201)#200)
ygrid = np.linspace(lby,uby,201)#200)
x,y = np.meshgrid(xgrid,ygrid)

### For z=0.0, consider n=3,8,9. For z=1.0, consider n=3,6,7
### Also, do n=0, 2 for both z=0.0,1.0

n=8#7
wf = DVR.eigenstate(vecs[:,n])
E_wf = vals[n]
print('E_wf', E_wf)

plt.imshow(abs(wf)**2,origin='lower')
plt.show()

sigma = 1.0
xgrid = np.linspace(lbx,ubx,ngridx+1)
#xgrid = xgrid[1:ngridx]
ygrid = np.linspace(lby,uby,ngridy+1)
#ygrid = ygrid[1:ngridy]	
husimi = Husimi_2D(xgrid,sigma,ygrid,sigma,hbar=hbar)
	
nx = ngridx//2
ny = ngridy//2

"""
Code below is to check how 'diffuse' the coherent states are.
"""
#coh00 = husimi.coherent_state(xgrid[nx],0.0,ygrid[ny],0.0)
#coh01 = husimi.coherent_state(xgrid[nx],0.0,ygrid[ny+40],0.0)
#coh10 = husimi.coherent_state(xgrid[nx+40],0.0,ygrid[ny],0.0)
#coh11 = husimi.coherent_state(xgrid[nx+40],0.0,ygrid[ny+40],0.0)

#coh = coh00 + coh01 + coh10 + coh11

#plt.imshow(abs(coh),origin='lower')
#plt.imshow(abs(coh00),origin='lower')
#plt.show()

nbasis = 201
xbasis = np.linspace(-4.5,4.5,nbasis)
pxbasis = np.linspace(-3.5,3.5,nbasis)
ybasis = np.linspace(-3,3,nbasis)
pybasis = np.linspace(-3,3,nbasis)

start_time = time.time()

dist = husimi.Husimi_section_x_fort(xbasis,pxbasis,0.0,wf,E_wf,pes.potential_xy,m)
#dist = husimi.Husimi_section_y_fort(ybasis,pybasis,0.0,wf,E_wf,pes.potential_xy,m)

plt.imshow(dist.T)
plt.show()
param_dict = {'lbx':xbasis[0],'ubx':xbasis[len(xbasis)-1],'lbpx':pxbasis[0],'ubpx':pxbasis[len(pxbasis)-1]}
with open('{}/Datafiles/Husimi_input_log_{}.txt'.format(path,potkey),'a') as f:	
	f.write('\n'+str(param_dict))#print(param_dict,file=f)
		
#store_arr(dist.T, 'Husimi_section_x_{}_nbasis_{}_E_{}'.format(potkey,nbasis,E_wf), '{}/Datafiles'.format(path))
store_arr(dist.T, 'Supp_Husimi_section_x_{}_nbasis_{}_n_{}'.format(potkey,nbasis,n), '{}/Datafiles'.format(path))

S = husimi.Renyi_entropy_1D(xbasis,pxbasis,dist,1)
print('S', S)

"""
The code below is compare the FORTRAN results with quantum
"""

#dist[:] = 0.0
#dist = husimi.Husimi_section_x(xbasis,pxbasis,ybasis,wf,E_wf,pes.potential_xy,m)
#dist = husimi.Husimi_section_y(ybasis,pybasis,xbasis,wf,E_wf,pes.potential_xy,m)
#print('dist old', dist)


