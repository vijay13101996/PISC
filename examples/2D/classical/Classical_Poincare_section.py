import numpy as np
import PISC
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from Saddle_point_finder import separatrix_path, find_minima
import time
import os


### Potential parameters
m=0.5#0.5
N=10#20
dt=0.002

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
D = 3*Vb#9.375#10.0
alpha = 0.25

print('Vb',Vb, 'D', D)

z = 1.5
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

### Temperature is only relevant for the ring-polymer Poincare section
Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc
Tkey = 'T_{}Tc'.format(times) 

pes = quartic_bistable(alpha,D,lamda,g,z)

pathname = os.path.dirname(os.path.abspath(__file__))

print('w', 2*np.pi/(2*D*alpha**2/m)**0.5)

w_db = np.sqrt(lamda/m)
w_m = (2*D*alpha**2/m)**0.5
E = Vb #+ 0.5*w_m# 0.5*w_db + 0.5*w_m#10.83#

minima = find_minima(m,D,alpha,lamda,g,z)
xmin,ymin = minima
print('xmin, ymin,alpha*ymin', xmin, ymin,alpha*ymin)
print('pot at min', pes.potential_xy(xmin,ymin), pes.potential_xy(0.0,0.0))

xg = np.linspace(-8,8,int(1e2)+1)
yg = np.linspace(-5,20,int(1e2)+1)

xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

#print('potmin', np.min(potgrid))

qlist = []

fig,ax = plt.subplots(1)
#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/30))
#plt.show()


### 'nbeads' can be set to >1 for ring-polymer simulations.
nbeads = 1
PSOS = Poincare_SOS('Classical',pathname,potkey,Tkey)
PSOS.set_sysparams(pes,T,m,2)
PSOS.set_simparams(N,dt,dt,nbeads=nbeads,rngSeed=1)	
PSOS.set_runtime(50.0,500.0)
if(1):
	xg = np.linspace(xmin-1.75,xmin+1.75,int(1e2)+1)
	yg = np.linspace(ymin-1.75,ymin+1.75,int(1e2)+1)

	xgrid,ygrid = np.meshgrid(xg,yg)
	potgrid = pes.potential_xy(xgrid,ygrid)

	qlist = PSOS.find_initcondn(xgrid,ygrid,potgrid,E)
	PSOS.bind(qcartg=qlist,E=E,sym_init=True)#pcartg=plist)#E=E)

	if(0): ## Plot the trajectories that make up the Poincare section
		xg = np.linspace(-8,8,int(1e2)+1)
		yg = np.linspace(-5,10,int(1e2)+1)
		xgrid,ygrid = np.meshgrid(xg,yg)
		potgrid = pes.potential_xy(xgrid,ygrid)

		ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/5))
		PSOS.run_traj(0,ax) #(1,2,3,4,8,13 for z=1.25), (2,3) 
		plt.show()
	
	if(1): ## Collect the data from the Poincare section and plot. 
		X,PX,Y = PSOS.PSOS_X(y0=0.0)#ymin)
		plt.scatter(X,PX,s=1)
		
	plt.title(r'$\alpha={}$, E=$V_b$+$3\omega_m/2$'.format(alpha) )#$N_b={}$'.format(nbeads))
	plt.xlabel(r'x')
	plt.ylabel(r'$p_x$')
	
	plt.show()
	#fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
	#store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))
				
if(0): ## Collect the data from the Poincare section and plot. 
	Y,PY,X = PSOS.PSOS_Y(x0=xmin)
	plt.scatter(Y,PY,s=1)	
	#PSOS.set_simparams(N,dt,dt,nbeads=nbeads,rngSeed=1)
	#PSOS.set_runtime(50.0,500.0)
	#PSOS.bind(qcartg=qlist,E=E)#pcartg=plist)#E=E)
	#Y,PY,X = PSOS.PSOS_Y(x0=0.0)
	plt.title(r'PSOS, $N_b={}$'.format(nbeads))
	#plt.scatter(Y,PY,s=2)
	plt.show()
	#fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
	#store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))


if(0): ## Initial conditions chosen along the 'transition path' from minima to saddle point
	qlist,vecslist = separatrix_path(m,D,alpha,lamda,g,z)
	plist = []
	print('qlist', qlist[0])
	for i in range(len(qlist)):
		x,y = qlist[i]
		V = pes.potential_xy(x,y)

		K = E-V
		p = (2*m*K)**0.5
		
		eigdir = 1	
		px = vecslist[i,eigdir,0]*p
		py = vecslist[i,eigdir,1]*p
	
		#print('E',V+(px**2+py**2)/(2*m), vecslist[i,eigdir])	
		plist.append([px,py])			

	plist = np.array(plist)
	rng = np.random.default_rng(0)
	ind = rng.choice(len(qlist),N)  # Choose N points at random from the qlist,plist
	ind = [236]
	print('ind',ind)
	qlist = qlist[ind,:,np.newaxis]
	plist = plist[ind,:,np.newaxis]

	if(0): # Trajectory initialized on barrier-top
		qlist = np.array([[1e-2,0.0]])
		plist = np.array([[0.0,0.0]])
		qlist = qlist[:,:,np.newaxis]
		plist = plist[:,:,np.newaxis]
		print('p',qlist.shape,plist.shape)	

	### Choice of initial conditions
	
if(0): ## Initial conditions are chosen by scanning through the PES. 
	ind = np.where(potgrid<E)
	xind,yind = ind

	for x,y in zip(xind,yind):
		#x = i[0]
		#y = i[1]
		#ax.scatter( xgrid[x,y],ygrid[x,y])#xgrid[x][y] , ygrid[x][y] )
		qlist.append([xgrid[x,y],ygrid[x,y]])
	#plt.show()
	qlist = np.array(qlist)
	#ind = [599]
	#qlist = qlist[ind,:]
	print('qlist',qlist.shape)


