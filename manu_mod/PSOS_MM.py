import numpy as np
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
import matplotlib.gridspec as gridspec
import matplotlib

rpext = '/home/vgs23/PISC/examples/2D/rpmd/Datafiles/'
Cext ='/home/vgs23/PISC/examples/2D/classical/Datafiles/'
#plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.serif':'Times New Roman'})
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

m=0.5

lamda = 2.0
g = 0.08

Vb =lamda**4/(64*g)
Tc = lamda*0.5/np.pi

D = 3*Vb
alpha = 0.382

print('Vb', Vb)
z = 1.0	

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

xlim = [-4,4]
ylim = [-2.25,2.25]
yticks = np.arange(-2,2.01)

xl_fs = 14 
yl_fs = 14
tp_fs = 14

le_fs = 8.5#9.5
ti_fs = 12

ms = 0.05
if(0):
	fig,ax = plt.subplots()

	Tkey = 'T_{}Tc'.format(10.0)
	fname =  'RP_PSOS_x_px_{}_{}_N_50_unfiltered_nbeads_1_seed_{}.txt'.format(potkey,Tkey,0)
	ext = rpext+fname
	data = read_1D_plotdata(ext)
	X = data[:,0]
	PX = data[:,1]
	ax.scatter(X,PX,s=0.25,color='slateblue',rasterized=True)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	
	ax.set_title(r'Classical', fontsize=ti_fs,x=0.5,y=0.83)
	ax.tick_params(axis='both', which='major', labelsize=18)
	ax.set_ylabel(r'$p_x$',fontsize=yl_fs)
	ax.set_yticks(yticks)
	#ax.set_xlabel(r'$x$',fontsize=22)
	
	fig.set_size_inches(9,3 )
	fig.savefig('/home/vgs23/Images/PSOS_Classical_D3.pdf'.format(g), dpi=400, bbox_inches='tight')
	
	plt.show()

if(1):
	fig,ax = plt.subplots(3,1)
	plt.subplots_adjust(wspace=0, hspace=0)

	Tkey = 'T_{}Tc'.format(10.0)
	fname =  'RP_PSOS_x_px_{}_{}_N_50_unfiltered_nbeads_1_seed_{}.txt'.format(potkey,Tkey,0)
	ext = rpext+fname
	data = read_1D_plotdata(ext)
	X = data[:,0]
	PX = data[:,1]
	ax[0].scatter(X,PX,s=ms,color='slateblue',rasterized=True)
	ax[0].set_xlim(xlim)
	ax[0].set_ylim(ylim)
	
	ax[0].set_title(r'$Classical$', fontsize=ti_fs,x=0.5,y=0.73)
	ax[0].tick_params(axis='both', which='major', labelsize=tp_fs)
	ax[0].set_ylabel(r'$p_x$',fontsize=yl_fs)
	ax[0].set_yticks(yticks)
	ax[0].set_xticks([])	

	Tkey = 'T_{}Tc'.format(3.0)
	fname =  'RP_PSOS_x_px_{}_{}_N_50_unfiltered_nbeads_8_seed_{}.txt'.format(potkey,Tkey,0)
	ext = rpext+fname
	data = read_1D_plotdata(ext)
	X = data[:,0]
	PX = data[:,1]
	ax[1].scatter(X,PX,s=ms,color='tomato',rasterized=True)
	ax[1].set_xlim(xlim)
	ax[1].set_ylim(ylim)

	Tkey = 'T_{}Tc'.format(0.95)
	fname =  'RP_PSOS_x_px_{}_{}_N_50_unfiltered_nbeads_8_seed_{}.txt'.format(potkey,Tkey,0)
	ext = rpext+fname
	data = read_1D_plotdata(ext)
	X = data[:,0]
	PX = data[:,1]
	ax[2].scatter(X,PX,s=ms,color='olivedrab',rasterized=True)
	ax[2].set_xlim(xlim)
	ax[2].set_ylim(ylim)

	ax[1].set_ylabel(r'$p_x$',fontsize=yl_fs)
	ax[2].set_xlabel(r'$x$',fontsize=xl_fs)
	ax[2].set_ylabel(r'$p_x$',fontsize=yl_fs)

	ax[2].set_title(r'$T=0.95T_c$', fontsize=ti_fs,x=0.5,y=0.73)
	ax[1].set_title(r'$T=3T_c$', fontsize=ti_fs,x=0.5,y=0.73)

	ax[1].set_xticks([])
	ax[1].set_yticks(np.arange(-2,2.01))
	ax[2].set_yticks(np.arange(-2,2.01))
	ax[1].tick_params(axis='both', which='major', labelsize=tp_fs)
	ax[2].tick_params(axis='both', which='major', labelsize=tp_fs)

	fig.set_size_inches(3,6)
	fig.savefig('/home/vgs23/Images/PSOS_RP_D3.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	#plt.show()

if(0):
	fig,ax = plt.subplots(1,3)
	plt.subplots_adjust(wspace=0.1, hspace=0.0)

	Tkey = 'T_{}Tc'.format(10.0)
	fname =  'RP_PSOS_x_px_{}_{}_N_50_unfiltered_nbeads_1_seed_{}.txt'.format(potkey,Tkey,0)
	ext = rpext+fname
	data = read_1D_plotdata(ext)
	X = data[:,0]
	PX = data[:,1]
	ax[0].scatter(X,PX,s=ms,color='slateblue',rasterized=True)
	ax[0].set_xlim(xlim)
	ax[0].set_ylim(ylim)
	
	ax[0].set_title(r'$Classical$', fontsize=ti_fs,x=0.5,y=0.83)
	ax[0].tick_params(axis='both', which='major', labelsize=tp_fs)
	ax[0].set_ylabel(r'$p_x$',fontsize=yl_fs)
	ax[0].set_yticks(yticks)
	ax[0].set_xticks([])	

	Tkey = 'T_{}Tc'.format(3.0)
	fname =  'RP_PSOS_x_px_{}_{}_N_50_unfiltered_nbeads_8_seed_{}.txt'.format(potkey,Tkey,0)
	ext = rpext+fname
	data = read_1D_plotdata(ext)
	X = data[:,0]
	PX = data[:,1]
	ax[1].scatter(X,PX,s=ms,color='tomato',rasterized=True)
	ax[1].set_xlim(xlim)
	ax[1].set_ylim(ylim)

	Tkey = 'T_{}Tc'.format(0.95)
	fname =  'RP_PSOS_x_px_{}_{}_N_50_unfiltered_nbeads_8_seed_{}.txt'.format(potkey,Tkey,0)
	ext = rpext+fname
	data = read_1D_plotdata(ext)
	X = data[:,0]
	PX = data[:,1]
	ax[2].scatter(X,PX,s=ms,color='olivedrab',rasterized=True)
	ax[2].set_xlim(xlim)
	ax[2].set_ylim(ylim)

	ax[0].set_ylabel(r'$p_x$',fontsize=yl_fs)
	for i in range(3):
		ax[i].set_xlabel(r'$x$',fontsize=xl_fs)	
		ax[i].set_xticks([-2.5,0,2.5])
		ax[i].tick_params(axis='both', which='major', labelsize=tp_fs)
	
	ax[2].set_title(r'$T=0.95T_c$', fontsize=ti_fs,x=0.5,y=0.83)
	ax[1].set_title(r'$T=3T_c$', fontsize=ti_fs,x=0.5,y=0.83)

	ax[1].set_yticks([])#np.arange(-2,2.01))
	ax[2].set_yticks([])#np.arange(-2,2.01))
	#ax[2].tick_params(axis='both', which='major', labelsize=tp_fs)

	fig.set_size_inches(8,2)
	fig.savefig('/home/vgs23/Images/PSOS_RP_D3.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	#plt.show()

if(0):
	fig, ax = plt.subplots()
	dt = 0.002
	nbeads = 32
	lw = 2.

	#---------------------------------------------------------------------------------------

	range_arr = [[1.7,2.2],[1.0,1.6],[1,1.6],[1.,1.6],[1.,1.6],[1,1.6],[1,1.6]]
	beads_arr = [32,32,16,16,16,8,16]
	dt_arr = [0.002,0.002,0.005,0.005,0.005,0.005]
	ls_arr = [':','--','-.','--','-','-.']
	color_arr = ['olivedrab','olivedrab','olivedrab','tomato','tomato','tomato']

	for times,sty,rang,nbeads,dt,c in zip([0.6,0.8,0.95,2.0,3.0,5.0],ls_arr,range_arr,beads_arr,dt_arr,color_arr):
		Tkey = 'T_{}Tc'.format(times)
		print(Tkey)
		ext ='RPMD_mc_OTOC_{}_{}_nbeads_{}_dt_{}_E_Vb'.format(potkey,Tkey,nbeads,dt)
		extrp = rpext + ext	
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(extrp,rang[0],rang[1])#1.7,2.2)
		plot_1D(ax,extrp, label=r'$T={}T_c$'.format(times),color=c,style=sty, log=True,linewidth=lw)      
		#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

	print('Classical')
	dt = 0.002
	ext = 'Classical_mc_OTOC_{}_T_20.0Tc_dt_{}_E_Vb'.format(potkey,dt)
	extclass = Cext + ext	
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(extclass,1.,2.)
	plot_1D(ax,extclass, label=r'$Classical$'.format(times),color='slateblue', log=True,linewidth=lw)      
	#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')


	ax.set_xlim([0.0,2.5])
	ax.set_ylim([-1.5,4.5])
	ax.set_xticks(np.arange(0,3.0,0.5))
	ax.legend(fontsize=le_fs)
	ax.tick_params(axis='both', which='major', labelsize=tp_fs)
	ax.set_xlabel(r'$t$',fontsize=xl_fs)
	ax.set_ylabel(r'$ln \; C_{mc}(t)$',fontsize=yl_fs)

	fig.set_size_inches(3,4)#3, 4)
	fig.savefig('/home/vgs23/Images/MC_OTOC_D3.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	
	#fig.set_size_inches(5,5)#3, 4)	
	#fig.savefig('/home/vgs23/Images/MC_OTOC_thesis.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	
	#plt.show()



