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

from matplotlib.pyplot import draw

########## Parameters

###Potential Parameters
m=0.5
D = 10.0
lamda = 2.0
g = 0.08

##########-----------------------------##########
##########Parameters to be changed##########
alpha = 0.363
z = 1.5


indx=[0,1]#[0,1,2,3]Individual trajectories of interest (are just some of the N=20 random traj)
sample=10#10 to be precise
Nsteps=2000 #2000 #(or more)
runtime_to=400.0 #500
#E = 3.25#4.202#is specified below, Energy on barrier top# DON'T change(for now) #8.613#Vb ####energy on the top of the DW
print()
print('alpha={}, z={}'.format(alpha,z))

Full_Poincare=True
Specific_Poincare=True
##########-----------------------------##########

###Simulation Parameters
N=20#20 #Number of Trajectories
dt=0.005 #0.005
nbeads = 1 # 'nbeads' can be set to >1 for ring-polymer simulations.

### Temperature (is only relevant for the ring-polymer Poincare section)
Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc
Tkey = 'T_{}Tc'.format(times) 

###Potential, Grid
pes = quartic_bistable(alpha,D,lamda,g,z)
pathname = os.path.dirname(os.path.abspath(__file__))
xg = np.linspace(-4,4,int(1e3)+1)
yg = np.linspace(-5,10,int(1e3)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

###########ENERGIES
#print(pes.potential_xy(0,0))#(alpha=0.363,z=0.5,E=3.150)
if(True):###1D
	from PISC.dvr.dvr import DVR1D
	pot_DW= lambda a: pes.potential_xy(a,0)
	DVR_DW = DVR1D(int(1e3)+1,-4,4,m,pot_DW)
	pot_Morse= lambda a: pes.potential_xy(0,a)-pes.potential_xy(0,0)#lamda**4/(64*g),a)
	DVR_Morse = DVR1D(int(1e3)+1,-5,10,m,pot_Morse)#get t a little more exact
	vals_M, vecs_M= DVR_Morse.Diagonalize()
	vals_DW, vecs_DW= DVR_DW.Diagonalize()
	#print(vals_M[0]+vals_DW[2]) #(alpha=0.363,z=(0.25,0.5),E=4.368)  
	E=vals_M[0]+vals_DW[2]
if(False):###2D
	from PISC.dvr.dvr import DVR2D
	#from mylib.twoD import DVR2D_mod
	DVR = DVR2D(int(1e2)+1,int(1e2)+1,-4,4,-5,10,m,pes.potential_xy)
	n_eig_tot = 20##leave it like that s.t. don't do too much work 
	vals, vecs= DVR.Diagonalize(neig_total=n_eig_tot)
	print(vals[0:10])##(0.363,0.5,4.368)

########## Choice of initial conditions

## Initial conditions are chosen by scanning through the PES. 
ind = np.where(potgrid<E)
xind,yind = ind #ind are two arrays with e.g. 1740 elements, which fullfil energy condition  
qlist = []#list of q values
for x,y in zip(xind,yind):
	qlist.append([xgrid[x,y],ygrid[x,y]])
#plt.show()
qlist = np.array(qlist)


print('pot grid: ',potgrid.shape)
print('qlist shape: ',qlist.shape)

##########Define and initialize Poincare SOS
##################what is surface of section? 
PSOS = Poincare_SOS('Classical',pathname,potkey,Tkey) 
PSOS.set_sysparams(pes,T,m,2)
PSOS.set_simparams(N,dt,dt,nbeads=nbeads)#dt_ens = time step the first 20 sec (for thermalization), dt = time step for normal evolution
PSOS.set_runtime(20.0,runtime_to)#500
PSOS.bind(qcartg=qlist,E=E,specific_traj=False)#pcartg=plist)#E=E)###################some important stuff going on here

##########Ploting 
colours=['r','g','b','c','m','y', 'k']
fig,ax = plt.subplots(1)
#ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/30))#not very useful since too tighly spaced

#Full_Poincare=True #defined above
#indx=[0,1]#Individual trajectories of interest#defined above

##########Full Poincare section

if(Full_Poincare): ## Collect the data from the full Poincare section and plot. 
	start_time=time.time()
	print()
	with contextlib.redirect_stdout(io.StringIO()):#don't want the info
		PSOS.bind(qcartg=qlist,E=E,specific_traj=False)
	print('Full Poincare section generated in ...')
	with contextlib.redirect_stdout(io.StringIO()):#don't want the info
		X,PX,Y = PSOS.PSOS_X(y0=0)
	print('.... %.2f s ' %(time.time()-start_time))
	plt.title(r'PSOS, $N_b={}$, z={}, $\alpha={}$'.format(nbeads,z,alpha))
	plt.scatter(X,PX,s=1)
	#plt.show()
	plt.draw()
	#fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
	#store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))

if(Specific_Poincare==True):
	fig,ax = plt.subplots(2)
	###########INTERACTIVE PART: Plot only Traj in ind
	### Plot poincare of some specific trajectories.
	for index in indx:  
		specific_traj=[index]#[0,1]
		start_time=time.time()
		print()
		with contextlib.redirect_stdout(io.StringIO()):#don't want the info
			PSOS.bind(qcartg=qlist,E=E,specific_traj=specific_traj)
		print('Poincare section generated in ...')
		with contextlib.redirect_stdout(io.StringIO()):#don't want the info
			X,PX,Y = PSOS.PSOS_X(y0=0)
		print('.... %.2f s ' %(time.time()-start_time))
		ax[0].set_title(r'PSOS, $N_b={}$, ind={}, $\alpha={}$ '.format(nbeads,indx,alpha))
		ax[1].scatter(X,PX,s=1,c=colours[index])
		#plt.show()

	#needs to be done just once since over indx (list) and not index (integers)
	with contextlib.redirect_stdout(io.StringIO()):#don't want the info
		PSOS.bind(qcartg=qlist,E=E,specific_traj=indx)

	### Interactively plot some of the trajectories that make up the Poincare section
	for index in indx:  
		xg = np.linspace(-8,8,int(1e2)+1)
		yg = np.linspace(-5,10,int(1e2)+1)
		xgrid,ygrid = np.meshgrid(xg,yg)
		potgrid = pes.potential_xy(xgrid,ygrid)
		ax[0].contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/5))
		with contextlib.redirect_stdout(io.StringIO()):#don't want the info
			PSOS.run_traj(ind=index,ax=ax[0],nsteps=Nsteps,colour=colours[index],sample=sample)#2000 at least  #takes a long time if nsteps are not specified
	plt.draw()
plt.show()


	
