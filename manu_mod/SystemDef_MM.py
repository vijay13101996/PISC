import numpy as np
import os
import ast
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Quartic_bistable import quartic_bistable
#from PISC.utils.colour_tools import lighten_color
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import matplotlib

path = '/home/vgs23/PISC/examples/2D'#os.path.dirname(os.path.abspath(__file__))	
qext = '{}/quantum'.format(path)
Cext = '{}/classical'.format(path)

#plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.serif':'Times New Roman'})
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

if(1):#
	hbar = 1.0
	m=0.5
	
	w = 0.1	
	D = 9.375#10.0
	alpha = 0.382#35#

	lamda = 2.0
	g = 0.08

	z = 1.0

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
	
n = 7

tp_fs = 10
xl_fs = 14
yl_fs = 14

if(0):  #PES plot
	xg = np.linspace(-5,5,int(1e3)+1)
	yg = np.linspace(-3,6,int(1e3)+1)

	xgrid,ygrid = np.meshgrid(xg,yg)
	potgrid = pes.potential_xy(xgrid,ygrid)

	fig,ax = plt.subplots(1)
	ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/15))

	ax.set_xlabel(r'$x$',fontsize=xl_fs)	
	ax.set_ylabel(r'$y$',fontsize=yl_fs)
	ax.scatter(0.0,0.0,color='k',marker='X')
	ax.tick_params(axis='both', which='major', labelsize=tp_fs)
	ax.set_xticks(np.arange(-5,5.01))
	ax.set_yticks(np.arange(-2,5.01))
	ax.set_xlim([-5.1,5.1])
	ax.set_ylim([-2.5,5.5])
	fig.set_size_inches(3, 2)
	fig.savefig('/home/vgs23/Images/PES_D3.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)

	#plt.show()

xgrid = np.linspace(lbx-0.1,ubx+0.1,201)
ygrid = np.linspace(lby-0.1,uby+0.1,201)
x,y = np.meshgrid(xgrid,ygrid)

neig_total = 200
DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy,hbar=hbar)		

fig, ax = plt.subplots()

wf = DVR.eigenstate(vecs[:,n])
wf2 = abs(wf)**2

E_wf = vals[n]
print('E_wf', E_wf)

if(0): #Wavefunction plot
	print('xgrid',ygrid[0])	
	ax.imshow(wf2,extent=[xgrid[0],xgrid[len(xgrid)-1],ygrid[0],ygrid[len(ygrid)-1]],origin='lower',cmap=plt.get_cmap('magma'))
	#plt.contour(x,y,pes.potential_xy(x,y),levels=np.arange(0.0,1.01*D,D/10), colors='white',linewidths=0.5)
	#plt.show()
	ax.set_xlabel(r'$x$',fontsize=xl_fs)	
	ax.set_ylabel(r'$y$',fontsize=yl_fs)
	ax.scatter(0.0,0.0,color='k',marker='X')
	ax.tick_params(axis='both', which='major', labelsize=tp_fs)
	ax.set_xticks(np.arange(-5,5.01))
	ax.set_yticks(np.arange(-2,5.01))
	ax.set_xlim([-5.1,5.1])
	ax.set_ylim([-2.5,5.5])
	

	fig.set_size_inches(3, 2)
	plt.axis('image')
	plt.axis('tight')
	fig.savefig('/home/vgs23/Images/Wavefunction_D3.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)

	#plt.show()

if(1): #Husimi section
	nbasis = 201
	dist = read_arr('Husimi_section_x_{}_nbasis_{}_E_{}'.format(potkey,nbasis,E_wf), '{}/quantum/Datafiles'.format(path))

	dist_gfilt = np.zeros(dist.shape+(3,))
	dist_gfilt[:,:,1] = 7*dist

	with open('{}/quantum/Datafiles/Husimi_input_log_{}.txt'.format(path,potkey)) as f:	
		for line in f:
			pass
		param_dict = ast.literal_eval(line)

	lbx = param_dict['lbx']
	ubx = param_dict['ubx']
	lbpx = param_dict['lbpx']
	ubpx = param_dict['ubpx']

	#plt.title(r'$\alpha={}$'.format(alpha))
	ax.imshow(dist_gfilt,extent=[lbx,ubx,lbpx,ubpx],origin='lower')

	ax.set_xlabel(r'$x$',fontsize=xl_fs)	
	ax.set_ylabel(r'$p_x$',fontsize=yl_fs)
	ax.scatter(0.0,0.0,color='k')
	ax.tick_params(axis='both', which='major', labelsize=tp_fs)
	ax.set_yticks(np.arange(-3,3.01))
	ax.set_xticks(np.arange(-4,4.01))
	ax.set_xlim([-4.1,4.1])
	ax.set_ylim([-3.1,3.1])
	
	plt.axis('image')
	plt.axis('tight')
	
	fig.set_size_inches(3, 2)
	fig.savefig('/home/vgs23/Images/Husimi_section_D3.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)

	#plt.show()

if(0): # Poincare section
	E = 6.52

	fname = 'Classical_Poincare_Section_x_px_{}_E_{}_seed_1'.format(potkey,E)	
	data1 =read_1D_plotdata('{}/Datafiles/{}.txt'.format(Cext,fname))
		
	fname = 'Classical_Poincare_Section_x_px_{}_E_{}_seed_2'.format(potkey,E)	
	data2 =read_1D_plotdata('{}/Datafiles/{}.txt'.format(Cext,fname))

	fname = 'Classical_Poincare_Section_x_px_{}_E_{}_seed_3'.format(potkey,E)	
	data3 =read_1D_plotdata('{}/Datafiles/{}.txt'.format(Cext,fname))


	xl_fs=14
	yl_fs=14
	tp_fs=10

	ax.scatter(data1[:,0],data1[:,1],s=0.5,color='olivedrab',rasterized=True)
	ax.scatter(data2[:,0],data2[:,1],s=0.65,color='coral',rasterized=True)
	ax.scatter(data3[:,0],data3[:,1],s=0.65,color='mediumpurple',rasterized=True)
			
	#plt.title(r'$\alpha={}$, E=$V_b$+$3\omega_m/2$'.format(alpha) )#$N_b={}$'.format(nbeads))
	ax.set_xlabel(r'$x$',fontsize=xl_fs)	
	ax.set_ylabel(r'$p_x$',fontsize=yl_fs)
	ax.scatter(0.0,0.0,color='k')
	ax.tick_params(axis='both', which='major', labelsize=tp_fs)
	ax.set_yticks(np.arange(-3,3.01))
	ax.set_xticks(np.arange(-4,4.01))
	ax.set_xlim([-4.1,4.1])
	ax.set_ylim([-3.1,3.1])

	fig.set_size_inches(3, 2)
	fig.savefig('/home/vgs23/Images/Poincare_section_D3.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)

	#fig.set_size_inches(6, 4)
	#fig.savefig('/home/vgs23/Images/Poincare_section_thesis.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)

	#plt.show()


