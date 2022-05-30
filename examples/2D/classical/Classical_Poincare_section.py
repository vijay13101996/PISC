import numpy as np
import PISC
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import os

m=0.5
w = 0.5
D = 10.0#10.0
alpha = 0.81#0.175#0.255#1.165#0.255

lamda = 1.5
g = 0.035

z = 1.0
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

pes = quartic_bistable(alpha,D,lamda,g,z)
poincare = Poincare_SOS(pes,m)

qcart = np.zeros((1,2,1))
pcart = np.zeros_like(qcart)

pathname = os.path.dirname(os.path.abspath(__file__))

L = lamda/np.sqrt(8*g)
xcart_lin = np.linspace(-L,L,11)
ycart_lin = np.linspace(-2.0,2.0,21)

E = 4.8#lamda**4/(64*g)	

#for i in range(len(xcart_lin)):
for j in range(len(ycart_lin)):
		qcart[0,:,0] = [-3.0,ycart_lin[j]]
		Pot = pes.potential_xy(qcart[0,0],qcart[0,1])
		if(E<Pot):
			continue
		Ekin = 2*(E - Pot)*m

		xkincomp = poincare.rng.uniform(0,1)
		ykincomp = 1-xkincomp

		pcart[0,0,0] = (xkincomp*Ekin)**0.5
		pcart[0,1,0] = (ykincomp*Ekin)**0.5
	
		#xgrid = np.linspace(-5,5,200)
		#ygrid = np.linspace(-5,5,200)
		#xmesh,ymesh = np.meshgrid(xgrid,ygrid)
		#potgrid = pes.potential_xy(xmesh,ymesh)	
		#plt.contour(xmesh,ymesh,potgrid,colors='k',levels=np.arange(0.0,5.1,0.5))	

		x,px,y= poincare.PSOS_X(qcart,pcart,0.0)
		x = np.array(x[::10])
		y = np.array(y[::10])
		#plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
		#plt.scatter(x[0],y[0],s=30,color='r')		
		#plt.show()
		
		pcart[0,0,0] = -(xkincomp*Ekin)**0.5
		pcart[0,1,0] = -(ykincomp*Ekin)**0.5	

		if(j%2==0):
			plt.scatter(poincare.X,poincare.PX,s=4,color='r')
			plt.show()
	
fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))
		
	
