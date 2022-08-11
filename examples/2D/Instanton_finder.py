import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.potentials.Four_well import four_well 
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.instanton import inst
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine.instools import find_minima, find_saddle, find_instanton, find_extrema, inst_init, inst_double
import time
from mpl_toolkits.mplot3d import Axes3D
from PISC.utils.nmtrans import FFT

### Potential parameters
m=0.5

if(0):
	D = 9.375
	alpha = 0.37

	lamda = 2.0
	g = 0.08

	z = 1.5

	pes = quartic_bistable(alpha,D,lamda,g,z)

	# Simulation parameters
	T_au = 0.95*lamda*0.5/np.pi

	#---------------------------------------------------------------------
	# Plot extent and axis
	L = 7.0
	lbx = -L
	ubx = L
	lby = -5
	uby = 10
	ngrid = 200
	ngridx = ngrid
	ngridy = ngrid

if(1):
	lamdax = 2.0
	gx = 0.08
	lamday = 2.0
	gy = 0.08
	z=1.0
	Vbx = lamdax**4/(64*gx)
	Vby = lamday**4/(64*gy)
	Vb = Vbx+Vby
	
	pes = four_well(lamdax,gx,lamday,gy,z)

	lamda = 1.65
	T_au = 2.0*lamda*0.5/np.pi

	#---------------------------------------------------------------------
	# Plot extent and axis
	L = 5.0
	lbx = -L
	ubx = L
	lby = -L
	uby = L
	ngrid = 200
	ngridx = ngrid
	ngridy = ngrid


beta = 1.0/T_au 

N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 20.0
time_total = 10.0#15.0
time_relax = 10.0

rngSeed = 1
N = 1
dim = 2
nbeads = 32
T = T_au	

xgrid = np.linspace(lbx,ubx,200)
ygrid = np.linspace(lby,uby,200)
x,y = np.meshgrid(xgrid,ygrid)

potgrid = pes.potential_xy(x,y)
fig,ax = plt.subplots()

ax.contour(x,y,potgrid,levels=np.arange(0.0,Vb,Vb/20))#D,D/20))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x,y,-potgrid)
plt.show()
# ---------------------------------------------------------------------
if(0):
	qinit = [0.5,0.6]
	minima = find_minima(m,pes,qinit,ax,plt,plot=True,dim=2)
	ax.scatter(minima[0],minima[1],color='k')
	print('minima', minima)

	qinit = minima+[-0.05,0.05]
	sp,vals,vecs = find_saddle(m,pes,qinit,ax,plt,plot=True,dim=2,scale=0.25)
	print('sp',sp)

if(1):
	sp=np.array([0.0,2.5])#0.0])
	eigvec = np.array([1.0,0.0])#vecs[0]
	
	nb = 4
	qinit = inst_init(sp,0.1,eigvec,nb)	
	while nb<=nbeads:
		instanton = find_instanton(m,pes,qinit,beta,nb,ax,plt,plot=False,dim=2,scale=1.0)
		#fft = FFT(1,nb)
		#q = fft.cart2mats(instanton)[0,:,0]/nb**0.5
		
		omegan = nb/beta
		potspr = np.sum(np.sum(0.5*m*omegan**2*(instanton-np.roll(instanton,1,axis=-1))**2,axis=2),axis=1)	
		potsys = np.sum(pes.potential(instanton),axis=1)
		print('Centroid energy', (potsys)/nb)	
		ax.scatter(instanton[0,0],instanton[0,1],label='nbeads = {}'.format(nb))
		print('Instanton config. with nbeads=', nb, 'computed')	
		qinit=inst_double(instanton)
		nb*=2
	
	### Increasing tolerance for the last 'doubling'. This can also probably be replaced with a gradient descent step	
	#instanton = find_extrema(m,pes,instanton,ax,plt,plot=True,dim=2,stepsize=1e-6,tol=1e-4)
	#instanton = find_instanton(m,pes,qinit,beta,nb,ax,plt,plot=False,dim=2,scale=1.0,stepsize=1e-6,tol=1e-4)	
	#ax.scatter(instanton[0,0],instanton[0,1],label='nbeads = {}'.format(nb))	
	plt.legend()
	plt.show()	
