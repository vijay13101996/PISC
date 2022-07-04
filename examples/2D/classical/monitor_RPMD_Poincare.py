from cProfile import label
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


##########Parameters to be changed##########
#alpha = 0.363
#z = 1.5
alpha_range = (0.153,0.157,0.193,0.220,0.252,0.344,0.363,0.525,0.837,1.1,1.665,2.514)
alpha_range=(0.363,)#0.363,)#,0.363,0.5)
z_range=(0.5,1,1.25,1.5)
gyr_range=(0.05,0.02,0.01,0.005)
#z_range=(0.5,)
for gyr in gyr_range:
	for alpha in alpha_range:
		X_z=[]
		PX_z=[]
		for z in z_range:
			###Animation Parameters
			indx=[0,1]#[0,1,2,3]Individual trajectories of interest (are just some of the N=20 random traj)
			sample=15#10# since I  to be precise#how often traj will be depicted
			Nsteps=400 #2000 #(or more)#steps for animation#3000
			
			DVR_energy=True#Barrier top of 2 councoupled, otw. specify E
			#E = 3.25#4.202#is specified below, Energy on barrier top# DON'T change(for now) #8.613#Vb ####energy on the top of the DW

			Full_Poincare=True
			Specific_Poincare=False#True
			
			###Convergence Parameters
			N=2*30#20 #Number of Trajectories#times 2 since initialized also at negative N
			dt=0.005 #0.005
			runtime_to=50.0 #500

			##########RP parameters
			nbeads = 8 # 'nbeads' can be set to >1 for ring-polymer simulations.
			Full_RP=False
			if(Full_RP==True or nbeads==1):
				gyr=10#radius of gyration for ring polymer
				greater=False#greater=True: higher radius of gyration than gyr  
			else:
				gyr=gyr#radius of gyration for ring polymer
				greater=False#greater=True: higher radius of gyration than gyr  
			#set gyr=10 and smaller for no such criteria (b.c. 0 and True will not do if nbeads=1 )

			### Temperature (is only relevant for the ring-polymer Poincare section)
			Tc = 0.5*lamda/np.pi
			times = 0.9
			T = times*Tc
			T= np.round(T,3)
			Tkey = 'T_{}Tc'.format(times) 

			###Potential, Grid
			pes = quartic_bistable(alpha,D,lamda,g,z)
			pathname = os.path.dirname(os.path.abspath(__file__))
			xg = np.linspace(-4,4,int(5*1e2)+1)
			yg = np.linspace(-5,10,int(5*1e2)+1)
			xgrid,ygrid = np.meshgrid(xg,yg)
			potgrid = pes.potential_xy(xgrid,ygrid)
			#potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
			potkey = 'alpha_{}_D_{}_lamda_{}_g_{}_z_{}_beads_{}_T_by_Tc={}'.format(alpha,D,lamda,g,z,nbeads,times)

			###########ENERGIES
			#print(pes.potential_xy(0,0))#(alpha=0.363,z=0.5,E=3.150)#Classical but we want quantum one
			if(DVR_energy==True):###1D
				from PISC.dvr.dvr import DVR1D
				pot_DW= lambda a: pes.potential_xy(a,0)
				DVR_DW = DVR1D(int(0.5*1e3)+1,-4,4,m,pot_DW)
				pot_Morse= lambda a: pes.potential_xy(0,a)-pes.potential_xy(0,0)#lamda**4/(64*g),a)
				DVR_Morse = DVR1D(int(0.5*1e3)+1,-5,10,m,pot_Morse)#get t a little more exact
				vals_M, vecs_M= DVR_Morse.Diagonalize()
				vals_DW, vecs_DW= DVR_DW.Diagonalize()
				#print(vals_M[0]+vals_DW[2]) #(alpha=0.363,z=(0.25,0.5),E=4.368)  
				E=np.round(vals_M[0]+vals_DW[2],3)
			if(False):###2D
				from PISC.dvr.dvr import DVR2D
				#from mylib.twoD import DVR2D_mod
				DVR = DVR2D(int(1e2)+1,int(1e2)+1,-4,4,-5,10,m,pes.potential_xy)
				n_eig_tot = 20##leave it like that s.t. don't do too much work 
				vals, vecs= DVR.Diagonalize(neig_total=n_eig_tot)
				print(vals[0:10])##(0.363,0.5,4.368)
					
			print()
			print('alpha={}, z={}, beads={}, E={}'.format(alpha,z,nbeads,E))
			########## Choice of initial conditions

			## Initial conditions are chosen by scanning through the PES. 
			ind = np.where(potgrid<E)
			xind,yind = ind #ind are two arrays with e.g. 1740 elements, which fullfil energy condition  
			qlist = []#list of q values
			for x,y in zip(xind,yind):
				qlist.append([xgrid[x,y],ygrid[x,y]])
			qlist = np.array(qlist)

			#print('pot grid: ',potgrid.shape)
			#print('qlist shape: ',qlist.shape)

			##########Define and initialize Poincare SOS
			##################what is surface of section? 
			PSOS = Poincare_SOS('Classical',pathname,potkey,Tkey) 
			PSOS.set_sysparams(pes,T,m,2)
			PSOS.set_simparams(N,dt,dt,nbeads=nbeads)#dt_ens = time step the first 20 sec (for thermalization), dt = time step for normal evolution
			PSOS.set_runtime(20.0,runtime_to)#500
			with contextlib.redirect_stdout(io.StringIO()):#don't want the info
				PSOS.bind(qcartg=qlist,E=E,specific_traj=False)#pcartg=plist)#E=E)###################some important stuff going on here

			##########Ploting 
			colours=['r','g','b','c','m','y', 'k']
			fig,ax = plt.subplots(1)

			##########Full Poincare section
			if(Full_Poincare): ## Collect the data from the full Poincare section and plot. 
				start_time=time.time()
				with contextlib.redirect_stdout(io.StringIO()):#don't want the info
					PSOS.bind(qcartg=qlist,E=E,specific_traj=False)
				print('Full Poincare section generated in ...')
				#with contextlib.redirect_stdout(io.StringIO()):#don't want the info
				
				if(Full_RP==True or nbeads==1):
					X,PX,Y = PSOS.PSOS_X(y0=0)
				else:
					X,PX,Y = PSOS.PSOS_X(y0=0,gyr=gyr,greater=greater)
				print('.... %.2f s ' %(time.time()-start_time))
			
				if(Full_RP==True or nbeads==1):
					plt.title(r'PSOS, $N_b={}$, z={}, $\alpha={}$, E={}, T/Tc={}'.format(nbeads,z,alpha,E,times))
				else:
					plt.title(r'PSOS, $N_b={}$, z={}, $\alpha={}$, E={}, T/Tc={}, '.format(nbeads,z,alpha,E,times,gyr))
				plt.scatter(X,PX,s=1)
				X_z.append(X)
				PX_z.append(PX)
				#plt.show()
				#fig.canvas.draw()
				fname = 'PSOS_{}_T_{}_E_{}_rtime_{}_beads_{}_T_by_Tc={}'.format(potkey,T,E,runtime_to,nbeads,times)
				#store_1D_plotdata(X,PX,fname,'{}/Datafiles/Poincare_RPMD'.format(pathname))
				#fig.savefig(('Plots/Poincare_RPMD/Fig_'+fname+'.svg'))
				#print('Data and Fig saved!')


			if(Specific_Poincare==True):
				fig,ax = plt.subplots(2)
				###########INTERACTIVE PART: Plot only Traj in ind
				### Plot poincare of some specific trajectories.
				for index in indx:  
					specific_traj=[index]#[0,1]
					start_time=time.time()
					with contextlib.redirect_stdout(io.StringIO()):#don't want the info
						PSOS.bind(qcartg=qlist,E=E,specific_traj=specific_traj)
					print('Poincare section generated in ...')
					with contextlib.redirect_stdout(io.StringIO()):#don't want the info
						X,PX,Y = PSOS.PSOS_X(y0=0)
					print('.... %.2f s ' %(time.time()-start_time))
					ax[0].set_title(r'PSOS, $N_b={}$, ind={}, z={}, $\alpha={}$, E={}, T/T$_c$={} '.format(nbeads,indx,z,alpha,E,times))
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
				fig.canvas.draw()
				fname = 'Interactive_PSOS_{}_T_{}_E_{}_rtime_{}_beads_{}_T_by_Tc={}'.format(potkey,T,E,runtime_to,nbeads,times)
				#fig.savefig(('Plots/Poincare_Interactive/Fig_'+fname+'.svg'))
				#print('Fig saved!')
		if(len(z_range)==4):
			figz=plt.figure()
			gs = figz.add_gridspec(2,2,hspace=0,wspace=0)
			if(nbeads==1 or Full_RP==True):
				figz.suptitle(r'PSOS: $N_b={}$, $\alpha={}, E={}$, T={}Tc)'.format(nbeads,alpha,E,times))
			elif(greater==True):
				figz.suptitle(r'PSOS: $N_b={}$, $\alpha={}, E={}$, T={}Tc, $r_g$>{})'.format(nbeads,alpha,E,times,gyr))
			else:
				figz.suptitle(r'PSOS: $N_b={}$, $\alpha={}, E={}$, T={}Tc, $r_g$<{})'.format(nbeads,alpha,E,times,gyr))
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
			figz.canvas.draw()
			fname = 'All_z_PSOS_{}_T_{}_E_{}_rtime_{}_beads_{}_T_by_Tc={}'.format(potkey,T,E,runtime_to,nbeads,times)
			#figz.savefig(('Plots/Poincare_RPMD_all_z/Fig_'+fname+'.svg'))
			#print('Fig saved!')
	plt.show()


		
