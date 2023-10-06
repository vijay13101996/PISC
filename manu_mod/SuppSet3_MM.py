import numpy as np
import os
import ast
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Quartic_bistable import quartic_bistable
#from PISC.utils.colour_tools import lighten_color
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import matplotlib
import itertools

path = '/home/vgs23/PISC/examples/2D'
qext = '{}/quantum'.format(path)
Cext = '{}/classical'.format(path)

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

tp_fs = 13
xl_fs = 16
yl_fs = 16

le_fs = 10.
ti_fs = 11
if(1): #Gather system data to plot
	hbar = 1.0
	m=0.5
	
	D = 9.375
	alpha = 0.382

	lamda = 2.0
	g = 0.08

	z = 1.0

	pes = quartic_bistable(alpha,D,lamda,g,z)
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	
	with open('{}/Datafiles/Input_log_{}.txt'.format(qext,potkey)) as f:	
		for line in f:
			pass
		param_dict = ast.literal_eval(line)

	lbx = param_dict['lbx']
	ubx = param_dict['ubx']
	lby = param_dict['lby']
	uby = param_dict['uby']
	ngridx = param_dict['ngridx']
	ngridy = param_dict['ngridy']

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
	fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)	# Change ngridx!=ngridy

	vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(qext))
	vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(qext))

fig, ax = plt.subplots(3,4)

alph_arr = [0.229, 0.382, 0.53, 1.147]
seed_arr = [[1,2,3],[4,5,6,7],[1,2,3,4],[1,2,3]]
narr = [8,7,7,8]
int_arr = [5.0,5.0,5.0,10.0]

for i in range(len(alph_arr)):
	alpha = alph_arr[i]
	n =narr[i]
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	
	with open('{}/Datafiles/Input_log_{}.txt'.format(qext,potkey)) as f:	
		for line in f:
			pass
		param_dict = ast.literal_eval(line)

	lbx = param_dict['lbx']
	ubx = param_dict['ubx']
	lby = param_dict['lby']
	uby = param_dict['uby']
	ngridx = param_dict['ngridx']
	ngridy = param_dict['ngridy']

	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
	fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)	# Change ngridx!=ngridy

	vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(qext))
	vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(qext))
	
	xgrid = np.linspace(lbx-0.1,ubx+0.1,201)
	ygrid = np.linspace(lby-0.1,uby+0.1,201)

	wf = DVR.eigenstate(vecs[:,n])
	wf2 = abs(wf)**2
	ax[0,i].imshow(wf2,extent=[xgrid[0],xgrid[len(xgrid)-1],ygrid[0],ygrid[len(ygrid)-1]],origin='lower',cmap=plt.get_cmap('magma'),vmin=0.0)	
	#ax[0,0].set_xlabel(r'$x$',fontsize=xl_fs)	
	ax[0,0].set_ylabel(r'$y$',fontsize=yl_fs)
	#ax[0,i].scatter(0.0,0.0,color='k',marker='X')
	ax[0,i].tick_params(axis='both', which='major', labelsize=tp_fs)
	ax[0,i].set_xticks([])#np.arange(-4,4.01,2))
	ax[0,0].set_yticks(np.arange(-2,5.01,2))
	ax[0,i].set_title(r'$n={}, \alpha={}$'.format(n+1,alpha), fontsize=ti_fs, x=0.5,y=0.75,color='w')

	ax[0,i].patch.set_facecolor('black')
	if(i!=0):
		ax[0,i].set_yticks([])
	ax[0,i].set_xlim([-5.1,5.1])
	ax[0,i].set_ylim([-2.5,5.5])

	for seed in seed_arr[i]:
		E = np.around(vals[n],2)
		fname = 'Classical_Poincare_Section_x_px_{}_E_{}_seed_{}'.format(potkey,E,seed)	
		data =read_1D_plotdata('{}/Datafiles/{}.txt'.format(Cext,fname))
		ax[1,i].scatter(data[:,0],data[:,1],s=0.15,color='slateblue',rasterized=True)
	
	ax[1,i].set_xticks([])
	if(i==0):	
		ax[1,i].set_yticks(np.arange(-3.,3.01,1.0))
	else:
		ax[1,i].set_yticks([])
	ax[1,i].set_xlim([-4.5,4.5])
	ax[1,i].set_ylim([-3.5,3.5])
	ax[1,i].tick_params(axis='both', which='major', labelsize=tp_fs)
	ax[1,0].set_ylabel(r'$p_x$',fontsize=yl_fs)	
			
	nbasis=201

	with open('{}/Datafiles/Husimi_input_log_{}.txt'.format(qext,potkey)) as f:	
		for line in f:
			pass
		param_dict = ast.literal_eval(line)

	lbx = param_dict['lbx']
	ubx = param_dict['ubx']
	lbpx = param_dict['lbpx']
	ubpx = param_dict['ubpx']

	dist = read_arr('Supp_Husimi_section_x_{}_nbasis_{}_n_{}'.format(potkey,nbasis,n), '{}/Datafiles'.format(qext))

	dist_gfilt = np.zeros(dist.shape+(3,))
	dist_gfilt[:,:,1] = int_arr[i]*dist

	ax[2,i].imshow(dist_gfilt,extent=[lbx,ubx,lbpx,ubpx],origin='lower')
	if(i!=0):
		ax[2,i].set_yticks([])
	ax[2,i].set_xlabel(r'$x$',fontsize=xl_fs)	
	ax[2,0].set_ylabel(r'$p_x$',fontsize=yl_fs)	
	#ax[1,i].scatter(0.0,0.0,color='k')
	ax[2,i].tick_params(axis='both', which='major', labelsize=tp_fs)
	ax[2,0].set_yticks(np.arange(-3,3.01))
	ax[2,i].set_xticks(np.arange(-4,4.01,2))
	ax[2,i].set_xlim([-4.5,4.5])
	ax[2,i].set_ylim([-3.5,3.5])
			
plt.subplots_adjust(wspace=0.06,hspace=0.0)
fig.set_size_inches(8, 6)		
#fig.savefig('/home/vgs23/Images/S5_D2.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)
fig.savefig('/home/vgs23/Images/S5_thesis.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)

plt.show()

