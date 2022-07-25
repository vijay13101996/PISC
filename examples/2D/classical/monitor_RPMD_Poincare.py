from optparse import Option
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
#plt.ion()

def Energy_Barrier_Top(pes,m,option='oneD'):
	if(option=='oneD'):
		from PISC.dvr.dvr import DVR1D
		pot_DW= lambda a: pes.potential_xy(a,0)
		DVR_DW = DVR1D(int(3*1e2)+1,-4,4,m,pot_DW)
		pot_Morse= lambda a: pes.potential_xy(0,a)-pes.potential_xy(0,0)#lamda**4/(64*g),a)
		DVR_Morse = DVR1D(int(3*1e2)+1,-5,10,m,pot_Morse)#get t a little more exact
		vals_M, vecs_M= DVR_Morse.Diagonalize()
		vals_DW, vecs_DW= DVR_DW.Diagonalize()
		#print(vals_M[0]+vals_DW[2]) #(alpha=0.363,z=(0.25,0.5),E=4.368)  
		print('M0: ',vals_M[0])
		print('DW2: ',vals_DW[2])
		E=np.round(vals_M[0]+vals_DW[2],3)
	elif(option=='twoD'):###2D
		from PISC.dvr.dvr import DVR2D
		#from mylib.twoD import DVR2D_mod
		DVR = DVR2D(int(1e2)+1,int(1e2)+1,-4,4,-2,10,m,pes.potential_xy)
		n_eig_tot = 20##leave it like that s.t. don't do too much work 
		vals, vecs= DVR.Diagonalize(neig_total=n_eig_tot)
		print(vals[0:10])##(0.363,0.5,4.368)
		E=vals[2]#only if 0.363
	elif(option=='cl_barrier_top'):
		E=pes.potential_xy(0,0)#(alpha=0.363,z=0.5,E=3.150)#Classical but we want quantum one
	else:
		print('Energy set manually!')
		E=option
	return E


#######################################################
#############---Start_Important_Options---#############
#######################################################

###Potential
m=0.5
D = 9.375
lamda = 2.0
g = 0.08
###saves
save_all_4=False #recommended: True
save_plotdata=False#recommended: True
save_individual=False #not so important 
save_everything=False#False saves everything irrespective of what set before

###IMPORTANT parameters for full poincare
N=100#20 #Number of Trajectories###actually 2*20 bc negative initialization
dt=0.005 #0.005
runtime_from=10#50 # modified the code s.t. not necessary anylonger
runtime_to=130 #500
plot_all_4=True#recommended: True
plot_each_full=True#recommended: False

###RP parameters
nbeads = 8
Filter_gyr=False
hist_data=False
hist_plt =True
Rg_of_t=False
gyr_lower=0.2
gyr_upper=0.8

### Temperature, might change Energy options at some point
Tc = 0.5*lamda/np.pi
times_range=(0.9,)
T=times_range[0]*Tc
T= np.round(T,3)
Tkey = 'T_{}Tc'.format(times_range[0]) 

###Energy
#careful with 2D for too high alpha
option='oneD'#'oneD','twoD','cl_barrier_top',or just number of energy
#option='cl_barrier_top'
#option=3.85# will set E to value which is specified
option=3.85

##########Parameters to loop over##########
alpha_range = (0.153,0.157,0.193,0.220,0.252,0.344,0.363,0.525,0.837,1.1,1.665,2.514)
alpha_range=(0.236,)#0.363,)#,0.363,0.5)
z_range=(0.5,1,1.25,1.5)
z_range=(1.5,)
#######################################################
#############---End_Important_Options---#############
#######################################################

###safty 
if(nbeads==1):
	Filter_gyr = False
	hist_data=False
	hist_plt =False
	Rg_of_t=False

#loops over all specified radius of gyration, alpha and z
for times in times_range:
	T=times*Tc
	for alpha in alpha_range:
		X_z=[]
		PX_z=[]
		Rg_z=[]
		for z in z_range:
			###Potential, Grid
			pes = quartic_bistable(alpha,D,lamda,g,z)
			pathname = os.path.dirname(os.path.abspath(__file__))
			#xg = np.linspace(-4,4,int(5*1e2)+1)
			#yg = np.linspace(-5,10,int(5*1e2)+1)
			xmin=2.4999
			ymin=-1.0502
			xg = np.linspace(xmin-2.5,xmin+2.5,int(5*1e2)+1)
			yg = np.linspace(ymin-2.5,ymin+8.5,int(5*1e2)+1)
			xgrid,ygrid = np.meshgrid(xg,yg)
			potgrid = pes.potential_xy(xgrid,ygrid)
			#potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
			if(Filter_gyr==False):
				potkey = 'alpha_{}_D_{}_lamda_{}_g_{}_z_{}_beads_{}_T_by_Tc={}'.format(alpha,D,lamda,g,z,nbeads,times)	
			else:
				potkey = 'alpha_{}_D_{}_lamda_{}_g_{}_z_{}_beads_{}_T_by_Tc={}_{}<Rg<{}'.format(alpha,D,lamda,g,z,nbeads,times,gyr_lower,gyr_upper)

			###ENERGIES
			E=Energy_Barrier_Top(pes=pes,m=m,option=option)
			
			### Choice of initial conditions
			ind = np.where(potgrid<E)
			xind,yind = ind #ind are two arrays with e.g. 1740 elements, which fullfil energy condition  
			qlist = []#list of q values
			for x,y in zip(xind,yind):
				qlist.append([xgrid[x,y],ygrid[x,y]])
			qlist = np.array(qlist)

			#print('pot grid: ',potgrid.shape)
			#print('qlist shape: ',qlist.shape)
			
			if(Filter_gyr==False):
				print('alpha={}, z={}, beads={}, E={}, no filter'.format(alpha,z,nbeads,E))
			else:
				print('alpha={}, z={}, beads={}, E={}, {}<Rg<{}, T/T_c={}'.format(alpha,z,nbeads,E,gyr_lower,gyr_upper,times))
			###Define and initialize Poincare SOS
			PSOS = Poincare_SOS('Classical',pathname,potkey,Tkey) 
			PSOS.set_sysparams(pes,T,m,2)
			PSOS.set_simparams(N,dt,dt,nbeads=nbeads)#dt_ens = time step the first 20 sec (for thermalization), dt = time step for normal evolution
			PSOS.set_runtime(runtime_from,runtime_to)#500
			with contextlib.redirect_stdout(io.StringIO()):#don't want the info
				PSOS.bind(qcartg=qlist,E=E,specific_traj=False,sym_init=True)#pcartg=plist)#E=E)###################some important stuff going on here

			###Ploting 
			colours=['r','g','b','c','m','y', 'k']

			##########Full Poincare section
			start_time=time.time()
			with contextlib.redirect_stdout(io.StringIO()):#don't want the info
				PSOS.bind(qcartg=qlist,E=E,specific_traj=False,sym_init=True)
			print('Full Poincare section generated in ...')
			#with contextlib.redirect_stdout(io.StringIO()):#don't want the info
			if(Filter_gyr==False):
				X,PX,Y = PSOS.PSOS_X(y0=0)
			else:
				X,PX,Y,Rg_list = PSOS.PSOS_X_gyr(y0=0,gyr_min=gyr_lower,gyr_max=gyr_upper,hist=hist_plt,Rg_of_t=Rg_of_t)
			print('.... %.2f s ' %(time.time()-start_time))

			if(plot_each_full==True):
				fig,ax = plt.subplots(1)
				if(Filter_gyr==False):
					plt.title(r'PSOS, $N_b={}$, z={}, $\alpha={}$, E={}, T/Tc={}'.format(nbeads,z,alpha,E,times))
				else:
					plt.title(r'PSOS, $N_b={}$, z={}, $\alpha={}$, E={}, T/Tc={}, {}<$R_g$<{} '.format(nbeads,z,alpha,E,times,gyr_lower,gyr_upper))	
				plt.scatter(X,PX,s=1)
				#fig.canvas.draw()
				plt.show(block=False)
				plt.pause(0.001)
			X_z.append(X)
			PX_z.append(PX)
			if(hist_data==True):
				Rg_z.append(Rg_list)
				
			fname = 'PSOS_{}_T_{}_E_{}_runtime_{}_beads_{}_T_by_Tc={}'.format(potkey,T,E,runtime_to,nbeads,times)
			if(save_everything==True or save_plotdata==True):
				store_1D_plotdata(X,PX,fname,'{}/Datafiles/Poincare_RPMD'.format(pathname))
			if(plot_each_full==True and (save_everything==True or save_individual==True)):
				fig.savefig(('Plots/Poincare_RPMD/Fig_'+fname+'.svg'))
				print('Data and Fig saved!')
			#4 plots for different z values (and one alpha value)
			if(len(z_range)==4 and plot_all_4==True):
				figz=plt.figure()
				gs = figz.add_gridspec(2,2,hspace=0,wspace=0)
				if(Filter_gyr==False):
					figz.suptitle(r'PSOS: $N_b={}$, $\alpha={}, E={}$, T={}Tc,N_traj={} )'.format(nbeads,alpha,E,times,N))
				else:
					figz.suptitle(r'PSOS: $N_b={}$, $\alpha={}, E={}$, T={}Tc, {}<$r_g$<{},N_traj={}'.format(nbeads,alpha,E,times,gyr_lower,gyr_upper,N))
				axz=gs.subplots(sharex='col', sharey='row')
				axz[0,0].scatter(X_z[0],PX_z[0],s=0.35,label='z=%.2f'% z_range[0])
				#axz[0,0].text(-0.25,-2,'z=%.2f'% z_range[0],fontsize=9)
				axz[0,1].scatter(X_z[1],PX_z[1],s=0.35,label='z=%.2f'% z_range[1])
				#axz[0,1].text(-0.25,-2,'z=%.2f'% z_range[1],fontsize=9)
				axz[1,0].scatter(X_z[2],PX_z[2],s=0.35,label='z=%.2f'% z_range[2])
				#axz[1,0].text(-0.25,-2,'z=%.2f'% z_range[2],fontsize=9)
				axz[1,1].scatter(X_z[3],PX_z[3],s=0.35,label='z=%.2f'% z_range[3])
				#axz[1,1].text(-0.25,-2,'z=%.2f'% z_range[3],fontsize=9)
				for ax in axz.flat:
					ax.label_outer()
					ax.legend(loc=8,fontsize=9,frameon=False)
					ax.set_xlim([-4.3,4.3])
				plt.show(block=False)
				#figz.canvas.draw()
				plt.pause(1)
				fname = 'All_z_PSOS_{}_T_{}_E_{}_runtime_{}_beads_{}_T_by_Tc={}_N_traj={}'.format(potkey,T,E,runtime_to,nbeads,times,N)
				if(save_all_4==True or save_everything==True):
					figz.savefig(('Plots/Poincare_RPMD_all_z/Fig_'+fname+'.svg'))
					print('Fig saved!')
			
plt.show(block=True)


		
