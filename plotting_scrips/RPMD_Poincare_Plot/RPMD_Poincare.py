import numpy as np
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
import PISC
import plt_util
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from Saddle_point_finder import separatrix_path, find_minima
import time
import os

###convergence Parameters
N=24
dt=0.005#0.005 #0.002
t_relax=800#200 
max_runtime=1000.0#sometimes 1000

### Potential parameters
m=0.5#0.5
lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

D = 3*Vb
alpha = 0.38
print('Vb',Vb, 'D', D)
z = 1.0
potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

Tc = 0.5*lamda/np.pi
times = 0.95
T = times*Tc
Tkey = 'T_{}Tc'.format(times) 

pes = quartic_bistable(alpha,D,lamda,g,z)

pathname = os.path.dirname(os.path.abspath(__file__))

w_db = np.sqrt(lamda/m)
w_m = (2*D*alpha**2/m)**0.5
E = 1.01*Vb #+ w_m/2

xg = np.linspace(-5,5,int(1e3)+1)
yg = np.linspace(-2,10,int(1e3)+1)

xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

qlist = []
###Pot
if(0):#get minima, but expensive (at least it was with old scipy)
	minima = find_minima(m,D,alpha,lamda,g,z)
	xmin,ymin = minima
	print('xmin, ymin,alpha*ymin', xmin, ymin,alpha*ymin)
	print('pot at min', pes.potential_xy(xmin,ymin), pes.potential_xy(0.0,0.0))
else:
	xmin=0
	ymin=0
if(0):#get a specific plot of the Potential
	fig2,ax = plt.subplots(1)
	ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/30))
	plt.show()
###Plot
if(1):#prepare 1D fig
#        fig = plt_util.prepare_fig(tex=True)
        fig1, ax1 = plt_util.prepare_fig_ax(dim=1,tex=True)
        fig2, ax2 =plt_util.prepare_fig_ax(dim=1,tex=True)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$p_x$')
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$p_x$')
        ax1.set_xlim([-4.1,4.1])
        ax1.set_ylim([-1.8,1.8])
        ax2.set_xlim([-4.1,4.1])
        ax2.set_ylim([-1.8,1.8])
        
if(1):#prepare 2D fig
        fig,axs = plt_util.prepare_fig_ax(dim=2,height=8,width=20,tex=True,sharex=True)#w=20 h=10
        axs[0].set_xlabel(r'$x$')
        axs[0].set_ylabel(r'$p_x$')
        axs[0].set_xlim([-4,4])
        axs[0].set_ylim([-1.9,1.9])
        axs[1].set_xlim([-4,4])
        axs[1].set_ylim([-1.9,1.9])
        axs[1].set_xlabel(r'$x$')
        axs[0].label_outer()
        axs[1].label_outer()
        #axs[1].get_yaxis().set_ticks([])
        #fig.supxaxis.labelpad = 20
        #fig.supxlabel(r'$x$')
        #fig.supylabel(r'$p_x$')

### 'nbeads' can be set to >1 for ring-polymer simulations.
nbeads = 8
PSOS = Poincare_SOS('Classical',pathname,potkey,Tkey)
PSOS.set_sysparams(pes,T,m,2)
PSOS.set_simparams(N,dt,dt,nbeads=nbeads,rngSeed=0)	
PSOS.set_runtime(200,max_runtime)#50.0,max_runtime)

#Some way of initalization
xg = np.linspace(xmin-2.5,xmin+2.5,int(1e2)+1)
yg = np.linspace(ymin-2.5,ymin+2.5,int(1e3)+1)

#xg = np.linspace(0,2*xmin,int(1e2)+1)
#yg = np.linspace(-2*abs(ymin),4*abs(ymin),int(1e3)+1)

#xg = np.linspace(0.0,1e-8,int(1e2)+1)
#yg = np.linspace(-1e-8,1e-8,int(1e2)+1)

#xg = np.linspace(0.0,2.5,int(1e2)+1)
#yg = np.linspace(-0.2,0.2,int(1e3)+1)
xgrid,ygrid = np.meshgrid(xg,yg)
potgrid = pes.potential_xy(xgrid,ygrid)

qlist = PSOS.find_initcondn(xgrid,ygrid,potgrid,E)

PSOS.bind(qcartg=qlist,E=E,sym_init=True)#pcartg=plist)#E=E)
	
	
max_rg=0.35
fname = 'PSOS_un_and_filtered_x_px_{}_T_{}_N_{}_long_{}_nbeads_{}'.format(potkey,T,N,max_runtime,nbeads)
fgyr_name = 'gyr_{}'.format(fname)
filter_name= 'filt_{}'.format(fname)
unfilter_name= 'unfilt_{}'.format(fname)
if(1): ## Calculate the data of the Poincare section and save
	X_full,PX_full,Y_full = PSOS.PSOS_X(y0=0.0)
	X,PX,Y, gyr_list_np = PSOS.PSOS_X_gyr(y0=0.0,gyr_min=0.0,gyr_max=max_rg,time_relax=t_relax)
	####saving Data
	store_1D_plotdata(X_full,PX_full,fname=unfilter_name,fpath='{}/Datafiles'.format(pathname))
	store_1D_plotdata(X,PX,fname=filter_name,fpath='{}/Datafiles'.format(pathname))
	store_arr(gyr_list_np,fgyr_name,'{}/Datafiles'.format(pathname))
if(1):## read the data from the Poincare section and plot
        data=read_1D_plotdata(fname=(unfilter_name+'.txt'),fpath='{}/Datafiles'.format(pathname))
        X_full=data[:,0]
        PX_full=data[:,1]
        data=read_1D_plotdata(fname=(filter_name+'.txt'),fpath='{}/Datafiles'.format(pathname))
        X=data[:,0]
        PX=data[:,1]
        gyr_list_np=read_arr((fgyr_name),'{}/Datafiles'.format(pathname))

        axs[0].scatter(X_full,PX_full,s=0.4)
        axs[1].scatter(X,PX,s=0.4)
        filename='RPMD_Poincare_filtered_unfiltered_N_{}_z_{}_D_3Vb_T_{}Tc_beads_{}_rg_{}.pdf'.format(N,z,times,nbeads,max_rg)
        filename_png='RPMD_Poincare_filtered_unfiltered_N_{}_z_{}_D_3Vb_T_{}Tc_beads_{}_rg_{}.png'.format(N,z,times,nbeads,max_rg)

        file_dpi=600
        print(axs[0].get_xlim())
        print(axs[0].get_ylim())
        fig.savefig('plots/'+filename,format='pdf',bbox_inches='tight', dpi=file_dpi)
        fig.savefig('plots/'+filename_png,format='png',bbox_inches='tight', dpi=file_dpi)
        #plt.show()
if(1):## read the data from the Poincare section and plot differently
        data=read_1D_plotdata(fname=(unfilter_name+'.txt'),fpath='{}/Datafiles'.format(pathname))
        X_full=data[:,0]
        PX_full=data[:,1]
        data=read_1D_plotdata(fname=(filter_name+'.txt'),fpath='{}/Datafiles'.format(pathname))
        X=data[:,0]
        PX=data[:,1]
        print(PX.shape)
        print(PX_full.shape)
        gyr_list_np=read_arr((fgyr_name),'{}/Datafiles'.format(pathname))
        ax1.scatter(X_full,PX_full,s=0.4)
        filename='RPMD_Poincare_filtered_unfiltered_N_{}_z_{}_D_3Vb_T_{}Tc_beads_{}_rg_{}.pdf'.format(N,z,times,nbeads,max_rg)
        filename_png='RPMD_Poincare_filtered_unfiltered_N_{}_z_{}_D_3Vb_T_{}Tc_beads_{}_rg_{}.png'.format(N,z,times,nbeads,max_rg)
        file_dpi=600
        fig1.savefig('plots/'+'sep1'+filename,format='pdf',bbox_inches='tight', dpi=file_dpi)
        fig1.savefig('plots/'+'sep1'+filename_png,format='png',bbox_inches='tight', dpi=file_dpi)
        ax2.scatter(X,PX,s=0.4)
        fig2.savefig('plots/'+'sep2'+filename,format='pdf',bbox_inches='tight', dpi=file_dpi)
        fig2.savefig('plots/'+'sep2'+filename_png,format='png',bbox_inches='tight', dpi=file_dpi)
        plt.show()

if(1):#create histogram
	gyr_list_max= np.zeros_like(gyr_list_np[0,:])
	for k in range(len(gyr_list_max)):
		gyr_list_max[k]=np.max(gyr_list_np[:,k])
	plt.hist(gyr_list_max,bins=20,edgecolor='black', linewidth=1.2)

	filename='Max_hist_beads_{}_T_{}Tc_z_{}.pdf'.format(nbeads,times,z)
	filename_png='Max_hist_beads_{}_T_{}Tc_z_{}.png'.format(nbeads,times,z)
	plt.xlabel(r'$r_{\small \textup{G}}$')
	plt.ylabel(r'Trajectories')
	plt.savefig('plots/'+filename,format='pdf',bbox_inches='tight', dpi=file_dpi)
	plt.savefig('plots/'+filename_png,format='png',bbox_inches='tight', dpi=file_dpi)
	plt.show()
	

#fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
#store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))
if(0): ## Plot the trajectories that make up the Poincare section
	xg = np.linspace(-8,8,int(1e2)+1)
	yg = np.linspace(-5,10,int(1e2)+1)
	xgrid,ygrid = np.meshgrid(xg,yg)
	potgrid = pes.potential_xy(xgrid,ygrid)

	ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/5))
	PSOS.run_traj(0,ax) #(1,2,3,4,8,13 for z=1.25), (2,3) 
	plt.show()

sys.exit()#maybe interesting stuff inside though
if(0):#normal PSOS 
	X,PX,Y = PSOS.PSOS_X(y0=0.0)#ymin)
	plt.scatter(X,PX,s=0.4)
	plt.show()			
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


