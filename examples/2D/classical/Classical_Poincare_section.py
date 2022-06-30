import numpy as np
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
import PISC
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from Saddle_point_finder import separatrix_path
import time
import os
import contextlib
import io




########## Parameters

###Potential Parameters
m=0.5
D = 10.0
lamda = 2.0
g = 0.08
###Parameters to be changed
alpha = 0.363
z = 0.5

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
#Vb = lamda**4/(64*g)#Specific energy parameter which is not used at the moment # interpretation..?  

###Simulation Parameters
N=20#20 #Number of Trajectories
dt=0.005 #0.005

### Temperature (is only relevant for the ring-polymer Poincare section)
Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc
Tkey = 'T_{}Tc'.format(times) 

##########Potential, Grid and Energy level
pes = quartic_bistable(alpha,D,lamda,g,z)
pathname = os.path.dirname(os.path.abspath(__file__))

E = 4.202#8.613#Vb ####energy on the top of the DW
xg = np.linspace(-4,4,int(1e2)+1)
yg = np.linspace(-5,10,int(1e2)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

print()
print('pot grid: ',potgrid.shape)
qlist = []#list of q values

fig,ax = plt.subplots(1)
#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/30))#not very useful since too tighly spaced
#plt.show()#comment out #if not commented out this will give just the contour of the potential, but we want the trajectory with the contour in the back!
	
### Choice of initial conditions
if(False): ## Initial conditions chosen along the 'transition path' from minima to saddle point
	qlist,vecslist = separatrix_path(m,D,alpha,lamda, g,z)
	plist = []

	for i in range(len(qlist)):
		x,y = qlist[i]#q is 2 dim coord
		V = pes.potential_xy(x,y)

		K = E-V#kintetic energy
		p = (2*m*K)**0.5
		
		eigdir = 1	#direction that EV points towards 
		px = vecslist[i,eigdir,0]*p
		py = vecslist[i,eigdir,1]*p
	
		#print('E',V+(px**2+py**2)/(2*m), vecslist[i,eigdir])	
		plist.append([px,py])			

	plist = np.array(plist)
	rng = np.random.default_rng(0)#rng RaNdom Generator
	ind = rng.choice(len(qlist),N)  # Choose N points at random from the qlist,plist
	ind = [236]#Idea was to use one traj and find out what it does
	print('index monitored: ',ind)
	qlist = qlist[ind,:,np.newaxis]####np.newaxis changes e.g. (4,) to (4,1) or (1,4) depending on where np.newaxis is put 
	plist = plist[ind,:,np.newaxis]

	if(0): # Trajectory initialized on barrier-top
		qlist = np.array([[1e-2,0.0]])
		plist = np.array([[0.0,0.0]])
		qlist = qlist[:,:,np.newaxis]
		plist = plist[:,:,np.newaxis]
		print('p',qlist.shape,plist.shape)	
		
if(1): ## Initial conditions are chosen by scanning through the PES. 
	ind = np.where(potgrid<E)
	xind,yind = ind #ind are two arrays with e.g. 1740 elements, which fullfil energy condition  
	for x,y in zip(xind,yind):
		qlist.append([xgrid[x,y],ygrid[x,y]])
	#plt.show()
	qlist = np.array(qlist)
	print('qlist shape: ',qlist.shape)

### 'nbeads' can be set to >1 for ring-polymer simulations.
nbeads = 1
##################what is surface of section? 
PSOS = Poincare_SOS('Classical',pathname,potkey,Tkey) 
PSOS.set_sysparams(pes,T,m,2)
PSOS.set_simparams(N,dt,dt,nbeads=nbeads)#dt_ens = time step the first 20 sec (for thermalization), dt = time step for normal evolution
PSOS.set_runtime(20.0,500.0)
PSOS.bind(qcartg=qlist,E=E,specific_traj=False)#pcartg=plist)#E=E)###################some important stuff going on here

if(True): ## Plot the trajectories that make up the Poincare section 
	ind=0 #specify the index of the trajectory of interest
	
	xg = np.linspace(-8,8,int(1e2)+1)
	yg = np.linspace(-5,10,int(1e2)+1)
	xgrid,ygrid = np.meshgrid(xg,yg)
	potgrid = pes.potential_xy(xgrid,ygrid)
	ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/5))
	PSOS.run_traj(ind=ind,ax=ax,nsteps=2000)#2000 at least  #takes a long time if nsteps are not specified
	plt.show()

if(True): ## Plot poincare of specific trajectory. 
	specific_traj=[0]#[0,1]
	
	start_time=time.time()
	print()
	with contextlib.redirect_stdout(io.StringIO()):#don't want the info
		PSOS.bind(qcartg=qlist,E=E,specific_traj=specific_traj)
	print('Poincare section generated in ...')
	X,PX,Y = PSOS.PSOS_X(y0=0)
	print('.... %.2f s ' %(time.time()-start_time))
	plt.title(r'PSOS, $N_b={}$'.format(nbeads))
	plt.scatter(X,PX,s=1)
	plt.show()
	#fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
	#store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))

if(True): ## Collect the data from the Poincare section and plot. 
	start_time=time.time()
	print()
	with contextlib.redirect_stdout(io.StringIO()):#don't want the info
		PSOS.bind(qcartg=qlist,E=E,specific_traj=False)
	print('Poincare section generated in ...')
	X,PX,Y = PSOS.PSOS_X(y0=0)
	print('.... %.2f s ' %(time.time()-start_time))
	plt.title(r'PSOS, $N_b={}$'.format(nbeads))
	plt.scatter(X,PX,s=1)
	plt.show()
	#fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
	#store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))

	
