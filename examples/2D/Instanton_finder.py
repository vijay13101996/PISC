import numpy as np
import PISC
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.potentials.Four_well import four_well 
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine.instools import find_minima, find_saddle, find_instanton, find_extrema, inst_init, inst_double
import time
from mpl_toolkits.mplot3d import Axes3D
from PISC.utils.nmtrans import FFT

### Potential parameters
m=0.5

path = os.path.dirname(os.path.abspath(__file__))

if(1):
	lamda = 2.0
	g = 0.08
	Vb = lamda**4/(64*g)
	
	D = 3*Vb
	alpha = 0.382

	z = 1.0

	pes = quartic_bistable(alpha,D,lamda,g,z)

	# Simulation parameters
	Tc = lamda*0.5/np.pi
	times = 0.5#0.95
	T_au = times*Tc
	
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	Tkey = 'T_{}Tc'.format(times)

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

if(0): #Four well potential
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

dim = 2
nbeads = 32
	
xgrid = np.linspace(lbx,ubx,200)
ygrid = np.linspace(lby,uby,200)
x,y = np.meshgrid(xgrid,ygrid)

potgrid = pes.potential_xy(x,y)
fig,ax = plt.subplots()

ax.contour(x,y,potgrid,levels=np.arange(0.0,D,D/20))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x,y,-potgrid)
#plt.show()
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
	sp=np.array([0.0,0.0])
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
		#print('Centroid energy', (potsys)/nb)	
		ax.scatter(instanton[0,0],instanton[0,1],label='nbeads = {}'.format(nb))
		print('Instanton config. with nbeads=', nb, 'computed')	
		qinit=inst_double(instanton)
		nb*=2

	store_arr(qinit,'Instanton_{}_{}_nbeads_{}'.format(potkey, Tkey,nbeads),'{}/rpmd/Datafiles/'.format(path)) 	 
	
	### Increasing tolerance for the last 'doubling'. This can also probably be replaced with a gradient descent step	
	#instanton = find_extrema(m,pes,instanton,ax,plt,plot=True,dim=2,stepsize=1e-5,tol=1e-3)
	#instanton = find_instanton(m,pes,qinit,beta,nb,ax,plt,plot=False,dim=2,scale=1.0,stepsize=1e-5,tol=1e-3)	
	#ax.scatter(instanton[0,0],instanton[0,1],label='nbeads = {}'.format(nb))	
	plt.legend()
	plt.show()	
