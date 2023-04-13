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
#path = os.path.dirname(os.path.abspath(__file__))	
qext = '{}/quantum'.format(path)
Cext = '{}/classical'.format(path)


plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

tp_fs = 13
xl_fs = 16
yl_fs = 16

le_fs = 9.5
ti_fs = 12
if(1): #Gather system data to plot
	hbar = 1.0
	m=0.5
	
	w = 0.1	
	D = 9.375#10.0
	alpha = 0.382#35#

	lamda = 2.0
	g = 0.08

	z = 0.0

	Tc = lamda*0.5/np.pi
	T_au = Tc#10.0 
	
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

	print('lb and ub', lbx,ubx,lby,uby)
	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
	fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)	# Change ngridx!=ngridy

	vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(qext))
	vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(qext))

xgrid = np.linspace(lbx-0.1,ubx+0.1,201)
ygrid = np.linspace(lby-0.1,uby+0.1,201)
x,y = np.meshgrid(xgrid,ygrid)

nbasis=201
with open('{}/Datafiles/Husimi_input_log_{}.txt'.format(qext,potkey)) as f:	
		for line in f:
			pass
		param_dict = ast.literal_eval(line)

lbx = param_dict['lbx']
ubx = param_dict['ubx']
lbpx = param_dict['lbpx']
ubpx = param_dict['ubpx']

neig_total = 200
DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy,hbar=hbar)		

narr = [3,8,9] #[3,6,7]

if(1):
	fig, ax = plt.subplots(2,len(narr))

	for i in range(len(narr)):
		n = narr[i]
		wf = DVR.eigenstate(vecs[:,n])
		wf2 = abs(wf)**2
		ax[0,i].imshow(wf2,extent=[xgrid[0],xgrid[len(xgrid)-1],ygrid[0],ygrid[len(ygrid)-1]],origin='lower',cmap=plt.get_cmap('magma'))	
		#ax[0,i].set_xlabel(r'$x$',fontsize=xl_fs)	
		ax[0,0].set_ylabel(r'$y$',fontsize=yl_fs)
		#ax[0,i].scatter(0.0,0.0,color='k',marker='X')
		ax[0,i].tick_params(axis='both', which='major', labelsize=tp_fs)
		ax[0,i].set_xticks(np.arange(-4,4.01,2))
		ax[0,0].set_yticks(np.arange(-2,5.01,2))
		ax[0,i].set_title(r'$n={}$'.format(n+1), fontsize=ti_fs, x=0.5,y=0.75,color='w')


		if(i!=0):
			ax[0,i].set_yticks([])
		ax[0,i].set_xlim([-5.1,5.1])
		ax[0,i].set_ylim([-2.5,5.5])

		dist = read_arr('Husimi_section_x_{}_nbasis_{}_n_{}'.format(potkey,nbasis,n), '{}/Datafiles'.format(qext))

		dist_gfilt = np.zeros(dist.shape+(3,))
		dist_gfilt[:,:,1] = 4.5*dist

		ax[1,i].imshow(dist_gfilt,extent=[lbx,ubx,lbpx,ubpx],origin='lower')
		if(i!=0):
			ax[1,i].set_yticks([])
		ax[1,i].set_xlabel(r'$x$',fontsize=xl_fs)	
		ax[1,0].set_ylabel(r'$p_x$',fontsize=yl_fs)
		#ax[1,i].scatter(0.0,0.0,color='k')
		ax[1,i].tick_params(axis='both', which='major', labelsize=tp_fs)
		ax[1,0].set_yticks(np.arange(-3,3.01))
		ax[1,i].set_xticks(np.arange(-4,4.01,2))
		ax[1,i].set_xlim([-4.1,4.1]) ## 
		ax[1,i].set_ylim([-3.1,3.1])
			
	fig.subplots_adjust(hspace = 0.25,wspace=0.1)
	fig.set_size_inches(6, 4)	
	fig.savefig('/home/vgs23/Images/S3_a_D1.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)
	plt.show()

if(1):
	fig,ax = plt.subplots()

	nmarr = list(itertools.combinations_with_replacement(narr,2))

	for n,M in nmarr:
		print(n,M)
		fname = 'Quantum_bnm_n_{}_m_{}_{}'.format(n,M,potkey)	
		data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(qext,fname))
		tarr = data[:,0]
		bnmarr=data[:,1]
		if(not((bnmarr<1e-3).all())):
			if(n==3 and M==7): #n==0 and M==2
				ax.plot(tarr,bnmarr,label=r'${}\leftrightarrow{}$'.format(n+1,M+1),lw=3)
			else:
				ax.plot(tarr,bnmarr,label=r'${}\leftrightarrow{}$'.format(n+1,M+1))
	plt.legend(fontsize=le_fs,ncol=1)	
	ax.tick_params(axis='both', which='major', labelsize=tp_fs)
	ax.set_xlabel(r'$t$',fontsize=xl_fs)	
	ax.set_ylabel(r'$|b_{nm}(t)|^2$',fontsize=yl_fs)
	ax.set_xticks(np.arange(0.0,5.01))	
	
	fig.set_size_inches(3.5, 4)	
	fig.savefig('/home/vgs23/Images/S3_b_D1.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)
	
	plt.show()

